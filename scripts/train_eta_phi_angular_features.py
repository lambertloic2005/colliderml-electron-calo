import json
from pathlib import Path

import torch
from torch import nn
import wandb

from colliderml_electron.dataset import make_loader, TARGET_COLS
from colliderml_electron.model import ConcatCaloRegressor


ETA_INDEX = TARGET_COLS.index("truth_eta")
PHI_INDEX = TARGET_COLS.index("truth_phi")


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


class EtaPhiLoss(nn.Module):
    """
    Loss for training only eta and phi.

    The model outputs two normalized values:

        [normalized_eta, normalized_phi]

    Eta uses ordinary MSE in normalized space.

    Phi is handled more carefully:
        1. denormalize predicted phi back to radians
        2. denormalize true phi back to radians
        3. compute wrapped angular difference
        4. divide by phi std to put it back on normalized scale
        5. compute MSE
    """

    def __init__(
        self,
        target_stats_path: str | Path,
        eta_weight: float = 1.0,
        phi_weight: float = 1.0,
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

        self.eta_weight = eta_weight
        self.phi_weight = phi_weight

    def forward(self, pred, target_eta_phi_norm):
        pred_eta_norm = pred[:, 0]
        phi_cos, phi_sin = pred[:, 1], pred[:, 2]

        target_eta_norm = target_eta_phi_norm[:, 0]
        target_phi_norm = target_eta_phi_norm[:, 1]

        eta_loss = torch.mean((pred_eta_norm - target_eta_norm) ** 2)

        target_phi = target_phi_norm * self.phi_std + self.phi_mean      # radians
        cos_t, sin_t = torch.cos(target_phi), torch.sin(target_phi)
        phi_loss = ((phi_cos - cos_t) ** 2 + (phi_sin - sin_t) ** 2).mean()  # = 2(1-cosΔ) on the circle

        total_loss = (
            self.eta_weight * eta_loss + self.phi_weight * phi_loss
        ) / (self.eta_weight + self.phi_weight)

        # decoded phi for diagnostics only
        pred_phi = torch.atan2(phi_sin, phi_cos)
        delta_phi = wrapped_angle_delta(pred_phi, target_phi)

        logs = {
            "loss_total": total_loss.detach(),
            "loss_eta": eta_loss.detach(),
            "loss_phi": phi_loss.detach(),
            "phi_mae_rad": delta_phi.abs().mean().detach(),
            "phi_rmse_rad": torch.sqrt(torch.mean(delta_phi ** 2)).detach(),
        }
        return total_loss, logs


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    loss_fn: EtaPhiLoss,
    device: torch.device,
) -> dict[str, float]:
    model.eval()

    totals = {
        "loss_total": 0.0,
        "loss_eta": 0.0,
        "loss_phi": 0.0,
        "phi_mae_rad": 0.0,
        "phi_rmse_rad": 0.0,
    }

    n_batches = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        pred = model(
            batch["x_sampled"],
            batch["x_high_level"],
            batch["mask"],
        )

        target_eta_phi = batch["target"][:, [ETA_INDEX, PHI_INDEX]]

        loss, logs = loss_fn(pred, target_eta_phi)

        for key in totals:
            totals[key] += logs[key].item()

        n_batches += 1

    return {
        key: value / max(n_batches, 1)
        for key, value in totals.items()
    }


def main():
    config = {
        "architecture": "concat_transformer_eta_phi_angular_features",
        "high_level_dim": 10,
        "use_angular_features": True,
        "dataset": "colliderml_release1_zee_prompt_electrons",
        "parquet_path": "data/electrons/electrons.parquet",
        "target_stats_path": "data/electrons/target_stats.json",

        "target_cols": ["truth_eta", "truth_phi"],

        "max_cells": 128,
        "model_dim": 128,
        "n_heads": 4,
        "n_layers": 3,
        "dim_feedforward": 256,
        "dropout": 0.1,
        "output_dim": 3,

        "batch_size": 4,
        "n_epochs": 30,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,

        "eta_weight": 1.0,
        "phi_weight": 1.0,

        "log_freq_batches": 10,
        "watch_gradients": False,
    }

    with wandb.init(
        project="colliderml-electron-calo",
        name="eta-phi-only",
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

        model = ConcatCaloRegressor(
            max_cells=cfg["max_cells"],
            model_dim=cfg["model_dim"],
            n_heads=cfg["n_heads"],
            n_layers=cfg["n_layers"],
            dim_feedforward=cfg["dim_feedforward"],
            dropout=cfg["dropout"],
            output_dim=cfg["output_dim"],
            high_level_dim=cfg["high_level_dim"],
        ).to(device)

        if cfg["watch_gradients"]:
            run.watch(model, log="gradients", log_freq=100)

        loss_fn = EtaPhiLoss(
            target_stats_path=stats_path,
            eta_weight=cfg["eta_weight"],
            phi_weight=cfg["phi_weight"],
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg["learning_rate"],
            weight_decay=cfg["weight_decay"],
        )

        global_step = 0
        best_val_loss = float("inf")
        best_val_phi_loss = float("inf")

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

                target_eta_phi = batch["target"][:, [ETA_INDEX, PHI_INDEX]]

                loss, logs = loss_fn(pred, target_eta_phi)

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
                            "train/phi_mae_rad_batch": logs["phi_mae_rad"].item(),
                            "train/phi_rmse_rad_batch": logs["phi_rmse_rad"].item(),
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

            run.log(
                {
                    "train/loss_total_epoch": train_loss,

                    "val/loss_total": val_logs["loss_total"],
                    "val/loss_eta": val_logs["loss_eta"],
                    "val/loss_phi": val_logs["loss_phi"],
                    "val/phi_mae_rad": val_logs["phi_mae_rad"],
                    "val/phi_rmse_rad": val_logs["phi_rmse_rad"],

                    "best_val_loss": best_val_loss,
                    "best_val_phi_loss": best_val_phi_loss,
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
                f"phi RMSE {val_logs['phi_rmse_rad']:.6f} rad"
            )

        Path("checkpoints").mkdir(exist_ok=True)
        checkpoint_path = Path("checkpoints/eta_phi_baseline.pt")

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": dict(cfg),
                "target_cols": ["truth_eta", "truth_phi"],
                "best_val_loss": best_val_loss,
                "best_val_phi_loss": best_val_phi_loss,
            },
            checkpoint_path,
        )

        artifact = wandb.Artifact(
            name="eta_phi_baseline",
            type="model",
            metadata={
                "best_val_loss": best_val_loss,
                "best_val_phi_loss": best_val_phi_loss,
                "architecture": cfg["architecture"],
            },
        )

        artifact.add_file(str(checkpoint_path))
        run.log_artifact(artifact)

        run.summary["best_val_loss"] = best_val_loss
        run.summary["best_val_phi_loss"] = best_val_phi_loss
        run.summary["checkpoint_path"] = str(checkpoint_path)

        print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()