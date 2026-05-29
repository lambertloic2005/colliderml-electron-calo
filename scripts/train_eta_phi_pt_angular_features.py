"""
Train eta, phi AND pT (as log pT) with the angular-feature inputs.

Derived from scripts/train_eta_phi_angular_features.py. The only physics target
added is log(pT); the input feature set is unchanged.

Model output layout (output_dim = 4):

    [eta, phi_cos, phi_sin, log_pt]   (all in z-scored space except the
                                       phi pair, which lives on the unit circle)

Targets are gathered from the parquet in the order [eta, phi, log_pt]; the loss
turns the normalized phi scalar into (cos, sin) targets internally, exactly as
the eta/phi angular-features script did.

Prereqs (same as the eta/phi/pT edits):
  - parquet has a `truth_log_pt` column
  - target_stats.json contains stats for `truth_log_pt`
"""

import json
from pathlib import Path

import torch
from torch import nn
import wandb

from colliderml_electron.dataset import make_loader, TARGET_COLS
from colliderml_electron.model import ConcatCaloRegressor, ConvCaloRegressor


ETA_INDEX = TARGET_COLS.index("truth_eta")
PHI_INDEX = TARGET_COLS.index("truth_phi")
LOGPT_INDEX = TARGET_COLS.index("truth_log_pt")


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def compute_grad_norm(model: nn.Module) -> float:
    norms = [
        p.grad.detach().norm(2)
        for p in model.parameters()
        if p.grad is not None
    ]

    if len(norms) == 0:
        return 0.0

    return torch.norm(torch.stack(norms), 2).item()


def wrapped_angle_delta(pred_phi: torch.Tensor, true_phi: torch.Tensor) -> torch.Tensor:
    """
    Compute pred_phi - true_phi while respecting angular wraparound.

    This matters because phi = +pi and phi = -pi are physically close.
    """
    delta = pred_phi - true_phi
    return torch.atan2(torch.sin(delta), torch.cos(delta))


class KinematicLoss(nn.Module):
    """
    Loss for eta, phi (cos/sin) and log pT.

    The model outputs four values:

        [normalized_eta, phi_cos, phi_sin, normalized_log_pt]

    - eta:    ordinary MSE in normalized space
    - phi:    the target phi is denormalized to radians, mapped to (cos, sin)
              on the unit circle, and the model's (phi_cos, phi_sin) are pulled
              toward it.  ((cos-c)^2 + (sin-s)^2 = 2(1 - cosΔ).)
    - log_pt: ordinary MSE in normalized space.  The residual in un-normalized
              ln(pT) is reported as pt_rel_rmse, i.e. ~ sigma(pT)/pT.
    """

    def __init__(
        self,
        target_stats_path: str | Path,
        eta_weight: float = 1.0,
        phi_weight: float = 1.0,
        logpt_weight: float = 1.0,
    ):
        super().__init__()

        stats = json.loads(Path(target_stats_path).read_text())

        self.register_buffer(
            "phi_mean",
            torch.tensor(stats["truth_phi"]["mean"], dtype=torch.float32),
        )
        self.register_buffer(
            "phi_std",
            torch.tensor(stats["truth_phi"]["std"], dtype=torch.float32),
        )
        self.register_buffer(
            "logpt_mean",
            torch.tensor(stats["truth_log_pt"]["mean"], dtype=torch.float32),
        )
        self.register_buffer(
            "logpt_std",
            torch.tensor(stats["truth_log_pt"]["std"], dtype=torch.float32),
        )

        self.eta_weight = eta_weight
        self.phi_weight = phi_weight
        self.logpt_weight = logpt_weight

    def forward(self, pred, target):
        # pred:   [eta, phi_cos, phi_sin, log_pt]
        # target: [eta_norm, phi_norm, logpt_norm]  (gathered in this order)
        pred_eta_norm = pred[:, 0]
        phi_cos, phi_sin = pred[:, 1], pred[:, 2]
        pred_logpt_norm = pred[:, 3]

        target_eta_norm = target[:, 0]
        target_phi_norm = target[:, 1]
        target_logpt_norm = target[:, 2]

        eta_loss = torch.mean((pred_eta_norm - target_eta_norm) ** 2)

        target_phi = target_phi_norm * self.phi_std + self.phi_mean      # radians
        cos_t, sin_t = torch.cos(target_phi), torch.sin(target_phi)
        phi_loss = ((phi_cos - cos_t) ** 2 + (phi_sin - sin_t) ** 2).mean()

        logpt_loss = torch.mean((pred_logpt_norm - target_logpt_norm) ** 2)

        w = self.eta_weight + self.phi_weight + self.logpt_weight
        total_loss = (
            self.eta_weight * eta_loss
            + self.phi_weight * phi_loss
            + self.logpt_weight * logpt_loss
        ) / w

        # diagnostics
        pred_phi = torch.atan2(phi_sin, phi_cos)
        delta_phi = wrapped_angle_delta(pred_phi, target_phi)
        # residual in un-normalized ln(pT) ~ dpT/pT -> fractional pT resolution
        d_lnpt = (pred_logpt_norm - target_logpt_norm) * self.logpt_std

        logs = {
            "loss_total": total_loss.detach(),
            "loss_eta": eta_loss.detach(),
            "loss_phi": phi_loss.detach(),
            "loss_logpt": logpt_loss.detach(),
            "phi_mae_rad": delta_phi.abs().mean().detach(),
            "phi_rmse_rad": torch.sqrt(torch.mean(delta_phi ** 2)).detach(),
            "pt_rel_rmse": torch.sqrt(torch.mean(d_lnpt ** 2)).detach(),
        }
        return total_loss, logs


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    loss_fn: KinematicLoss,
    device: torch.device,
) -> dict[str, float]:
    model.eval()

    totals = {
        "loss_total": 0.0,
        "loss_eta": 0.0,
        "loss_phi": 0.0,
        "loss_logpt": 0.0,
        "phi_mae_rad": 0.0,
        "phi_rmse_rad": 0.0,
        "pt_rel_rmse": 0.0,
    }

    n_batches = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        pred = model(
            batch["x_sampled"],
            batch["x_high_level"],
            batch["mask"],
        )

        target = batch["target"][:, [ETA_INDEX, PHI_INDEX, LOGPT_INDEX]]

        loss, logs = loss_fn(pred, target)

        for key in totals:
            totals[key] += logs[key].item()

        n_batches += 1

    return {
        key: value / max(n_batches, 1)
        for key, value in totals.items()
    }


def main():
    config = {
        "architecture": "concat_transformer_eta_phi_pt_angular_features",
        "high_level_dim": 12,
        "use_angular_features": True,
        "dataset": "colliderml_release1_zee_prompt_electrons",
        "parquet_path": "data/electrons/electrons.parquet",
        "target_stats_path": "data/electrons/target_stats.json",

        "target_cols": ["truth_eta", "truth_phi", "truth_log_pt"],

        "max_cells": 128,
        "model_dim": 128,
        "n_heads": 4,
        "n_layers": 3,
        "dim_feedforward": 256,
        "dropout": 0.1,
        "output_dim": 4,

        "batch_size": 4,
        "n_epochs": 30,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,

        "eta_weight": 1.0,
        "phi_weight": 1.0,
        "logpt_weight": 1.0,

        "log_freq_batches": 10,
        "watch_gradients": False,

        "model_type": "conv",     # "concat" reproduces your current baseline
        "conv_dim": 128,
        "kernel_size": 5,

        "feature_set": "xyz_loge_eta_phi_theta",
    }

    with wandb.init(
        project="colliderml-electron-calo",
        name="eta-phi-pt",
        job_type="training",
        config=config,
    ) as run:
        cfg = run.config

        device = get_device()
        print(f"Using device: {device}")

        parquet_path = Path(cfg["parquet_path"])
        stats_path = Path(cfg["target_stats_path"])

        if not parquet_path.exists():
            raise FileNotFoundError(
                f"Could not find {parquet_path}. Build the electron table first."
            )

        if not stats_path.exists():
            raise FileNotFoundError(
                f"Could not find {stats_path}. Run the split/stat step first."
            )

        train_loader = make_loader(
            parquet_path=parquet_path,
            split="train",
            target_stats_path=stats_path,
            batch_size=cfg["batch_size"],
            shuffle=True,
            use_angular_features=cfg["use_angular_features"],
        )

        val_loader = make_loader(
            parquet_path=parquet_path,
            split="val",
            target_stats_path=stats_path,
            batch_size=cfg["batch_size"],
            shuffle=False,
            use_angular_features=cfg["use_angular_features"],
        )

        common = dict(
            max_cells=cfg["max_cells"], model_dim=cfg["model_dim"],
            n_heads=cfg["n_heads"], n_layers=cfg["n_layers"],
            dim_feedforward=cfg["dim_feedforward"], dropout=cfg["dropout"],
            output_dim=cfg["output_dim"], high_level_dim=cfg["high_level_dim"],
        )
        if cfg.get("model_type", "concat") == "conv":
            model = ConvCaloRegressor(**common, conv_dim=cfg["conv_dim"],
                                      kernel_size=cfg["kernel_size"]).to(device)
        else:
            model = ConcatCaloRegressor(**common).to(device)

        if cfg["watch_gradients"]:
            run.watch(model, log="gradients", log_freq=100)

        loss_fn = KinematicLoss(
            target_stats_path=stats_path,
            eta_weight=cfg["eta_weight"],
            phi_weight=cfg["phi_weight"],
            logpt_weight=cfg["logpt_weight"],
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg["learning_rate"],
            weight_decay=cfg["weight_decay"],
        )

        global_step = 0
        best_val_loss = float("inf")
        best_val_phi_loss = float("inf")
        best_val_pt_rel_rmse = float("inf")

        for epoch in range(1, cfg["n_epochs"] + 1):
            model.train()

            total_train_loss = 0.0
            n_batches = 0

            for batch_idx, batch in enumerate(train_loader):
                batch = move_batch_to_device(batch, device)

                pred = model(
                    batch["x_sampled"],
                    batch["x_high_level"],
                    batch["mask"],
                )

                target = batch["target"][:, [ETA_INDEX, PHI_INDEX, LOGPT_INDEX]]

                loss, logs = loss_fn(pred, target)

                optimizer.zero_grad()
                loss.backward()

                grad_norm = compute_grad_norm(model)

                optimizer.step()

                total_train_loss += loss.item()
                n_batches += 1

                if batch_idx % cfg["log_freq_batches"] == 0:
                    run.log(
                        {
                            "train/loss_total_batch": loss.item(),
                            "train/loss_eta_batch": logs["loss_eta"].item(),
                            "train/loss_phi_batch": logs["loss_phi"].item(),
                            "train/loss_logpt_batch": logs["loss_logpt"].item(),
                            "train/phi_mae_rad_batch": logs["phi_mae_rad"].item(),
                            "train/phi_rmse_rad_batch": logs["phi_rmse_rad"].item(),
                            "train/pt_rel_rmse_batch": logs["pt_rel_rmse"].item(),
                            "grad_norm": grad_norm,
                            "learning_rate": optimizer.param_groups[0]["lr"],
                            "epoch": epoch,
                            "batch_idx": batch_idx,
                        },
                        step=global_step,
                    )

                global_step += 1

            train_loss = total_train_loss / max(n_batches, 1)

            val_logs = evaluate(
                model=model,
                loader=val_loader,
                loss_fn=loss_fn,
                device=device,
            )

            val_loss = val_logs["loss_total"]
            best_val_loss = min(best_val_loss, val_loss)
            best_val_phi_loss = min(best_val_phi_loss, val_logs["loss_phi"])
            best_val_pt_rel_rmse = min(best_val_pt_rel_rmse, val_logs["pt_rel_rmse"])

            run.log(
                {
                    "train/loss_total_epoch": train_loss,

                    "val/loss_total": val_logs["loss_total"],
                    "val/loss_eta": val_logs["loss_eta"],
                    "val/loss_phi": val_logs["loss_phi"],
                    "val/loss_logpt": val_logs["loss_logpt"],
                    "val/phi_mae_rad": val_logs["phi_mae_rad"],
                    "val/phi_rmse_rad": val_logs["phi_rmse_rad"],
                    "val/pt_rel_rmse": val_logs["pt_rel_rmse"],

                    "best_val_loss": best_val_loss,
                    "best_val_phi_loss": best_val_phi_loss,
                    "best_val_pt_rel_rmse": best_val_pt_rel_rmse,
                    "epoch": epoch,
                },
                step=global_step,
            )

            print(
                f"epoch {epoch:03d} | "
                f"train loss {train_loss:.6f} | "
                f"val loss {val_logs['loss_total']:.6f} | "
                f"val eta {val_logs['loss_eta']:.6f} | "
                f"val phi {val_logs['loss_phi']:.6f} | "
                f"phi RMSE {val_logs['phi_rmse_rad']:.6f} rad | "
                f"pT res {val_logs['pt_rel_rmse']:.4f}"
            )

        Path("checkpoints").mkdir(exist_ok=True)
        checkpoint_path = Path(f"checkpoints/eta_phi_pt_{cfg['model_type']}.pt")

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": dict(cfg),
                "target_cols": ["truth_eta", "truth_phi", "truth_log_pt"],
                "best_val_loss": best_val_loss,
                "best_val_phi_loss": best_val_phi_loss,
                "best_val_pt_rel_rmse": best_val_pt_rel_rmse,
            },
            checkpoint_path,
        )

        artifact = wandb.Artifact(
            name="eta_phi_pt_baseline",
            type="model",
            metadata={
                "best_val_loss": best_val_loss,
                "best_val_phi_loss": best_val_phi_loss,
                "best_val_pt_rel_rmse": best_val_pt_rel_rmse,
                "architecture": cfg["architecture"],
            },
        )

        artifact.add_file(str(checkpoint_path))
        run.log_artifact(artifact)

        run.summary["best_val_loss"] = best_val_loss
        run.summary["best_val_phi_loss"] = best_val_phi_loss
        run.summary["best_val_pt_rel_rmse"] = best_val_pt_rel_rmse
        run.summary["checkpoint_path"] = str(checkpoint_path)

        print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()