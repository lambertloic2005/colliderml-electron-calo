# =============================================================================
# JOB 1 -- RETREAT CONTROL -- branch: retreat-control  (created off pointing-upgrade)
# Paste as: scripts/train_eta_phi_pt_z0_charge.py   (keep this exact filename/path)
# Content: high_level_dim = 41, NO min_pt (trains on all pT), min_epochs = 100,
#          durable best-checkpoint save.
# Pairs with: the retreat-control dataset.py (K=6, no phi_slope).
# =============================================================================
import json
import os

from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
import wandb

from colliderml_electron.dataset import make_loader, TARGET_COLS
from colliderml_electron.model import ConcatCaloRegressor, ConvCaloRegressor

#HELLO WORLD
ETA_INDEX = TARGET_COLS.index("truth_eta")
PHI_INDEX = TARGET_COLS.index("truth_phi")
LOGPT_INDEX = TARGET_COLS.index("truth_log_pt")
Z0_INDEX = TARGET_COLS.index("truth_z0")


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
        z0_weight: float = 1.0,
        charge_weight: float = 1.0,
    ):
        super().__init__()

        stats = json.loads(Path(target_stats_path).read_text())

        self.register_buffer(
            "eta_mean",
            torch.tensor(stats["truth_eta"]["mean"], dtype=torch.float32),
        )
        self.register_buffer(
            "eta_std",
            torch.tensor(stats["truth_eta"]["std"], dtype=torch.float32),
        )
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
        self.register_buffer(
            "z0_mean",
            torch.tensor(stats["truth_z0"]["mean"], dtype=torch.float32),
        )
        self.register_buffer(
            "z0_std",
            torch.tensor(stats["truth_z0"]["std"], dtype=torch.float32),
        )

        self.eta_weight = eta_weight
        self.phi_weight = phi_weight
        self.logpt_weight = logpt_weight
        self.z0_weight = z0_weight

        # learned per-task log-sigma for the 4 REGRESSION tasks (eta, phi, logpt, z0).
        # Charge is a classification (BCE) task and does NOT share the Gaussian-noise
        # assumption the homoscedastic scheme is derived for, so it is weighted manually.
        self.log_sigma = nn.Parameter(torch.zeros(4))
        self.charge_weight = charge_weight

    def forward(self, pred, target, phi_centroid, eta_centroid, log_sum_et,
                z0_anchor, truth_charge):
        # pred: [delta_eta, delta_phi, delta_logpt, delta_z0, charge_logit]
        # target: [eta_norm, phi_norm, logpt_norm, z0_norm] (z-scored, this order)
        # truth_charge: (B,) in {-1, +1}; used ONLY as the classification label now.
        pred_deta    = pred[:, 0]
        pred_dphi    = pred[:, 1]      # single signed bend; its sign IS the charge handle
        pred_dlogpt  = pred[:, 2]
        pred_dz0     = pred[:, 3]
        charge_logit = pred[:, 4]      # logit > 0 => predict positron (q = +1)

        target_eta   = target[:, 0] * self.eta_std + self.eta_mean
        target_phi   = target[:, 1] * self.phi_std + self.phi_mean
        target_logpt = target[:, 2] * self.logpt_std + self.logpt_mean

        deta_target    = target_eta - eta_centroid
        dphi_target    = wrapped_angle_delta(target_phi, phi_centroid)  # signed; sign = bend dir
        dlogpt_target  = target_logpt - log_sum_et
        z0_target_norm = target[:, 3]                                   # already z-scored

        eta_loss = F.huber_loss(pred_deta, deta_target, delta=0.1)
        phi_err  = wrapped_angle_delta(pred_dphi, dphi_target)
        phi_loss = F.huber_loss(phi_err, torch.zeros_like(phi_err), delta=0.05)
        logpt_loss = F.huber_loss(pred_dlogpt, dlogpt_target, delta=0.2)
        z0_loss = F.huber_loss(pred_dz0, z0_target_norm, delta=1.0)

        # charge: binary classification. label 1 = positron (q=+1), 0 = electron (q=-1)
        charge_label = (truth_charge > 0).float()
        charge_loss = F.binary_cross_entropy_with_logits(charge_logit, charge_label)

        # Homoscedastic weighting over the 4 REGRESSION tasks only.
        reg_losses = torch.stack([eta_loss, phi_loss, logpt_loss, z0_loss])
        precision = torch.exp(-2.0 * self.log_sigma)
        total_loss = (precision * reg_losses + self.log_sigma).sum()
        # Charge added with a fixed weight so its gradient cannot be suppressed.
        total_loss = total_loss + self.charge_weight * charge_loss

        # diagnostics: phi is now decoded WITHOUT any truth charge
        pred_phi = phi_centroid + pred_dphi
        delta_phi = wrapped_angle_delta(pred_phi, target_phi)
        d_lnpt = pred_dlogpt - dlogpt_target
        dz0_err = (pred_dz0 - z0_target_norm) * self.z0_std            # back to mm
        charge_acc = ((charge_logit > 0).float() == charge_label).float().mean()

        logs = {
            "loss_total": total_loss.detach(),
            "loss_eta": eta_loss.detach(),
            "loss_phi": phi_loss.detach(),
            "loss_logpt": logpt_loss.detach(),
            "loss_z0": z0_loss.detach(),
            "loss_charge": charge_loss.detach(),
            "charge_acc": charge_acc.detach(),
            "phi_mae_rad": delta_phi.abs().mean().detach(),
            "phi_rmse_rad": torch.sqrt(torch.mean(delta_phi ** 2)).detach(),
            "pt_rel_rmse": torch.sqrt(torch.mean(d_lnpt ** 2)).detach(),
            "z0_rmse_mm": torch.sqrt(torch.mean(dz0_err ** 2)).detach(),
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
        "loss_z0": 0.0,
        "phi_mae_rad": 0.0,
        "phi_rmse_rad": 0.0,
        "pt_rel_rmse": 0.0,
        "z0_rmse_mm": 0.0,
        "loss_charge": 0.0,
        "charge_acc": 0.0,
    }

    n_batches = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        pred = model(
            batch["x_sampled"],
            batch["x_high_level"],
            batch["mask"],
        )

        target = batch["target"][:, [ETA_INDEX, PHI_INDEX, LOGPT_INDEX, Z0_INDEX]]

        loss, logs = loss_fn(
            pred, target,
            batch["phi_centroid"], batch["eta_centroid"], batch["log_sum_et"],
            batch["z0_anchor"], batch["truth_charge"],
        )

        for key in totals:
            totals[key] += logs[key].item()

        n_batches += 1

    return {
        key: value / max(n_batches, 1)
        for key, value in totals.items()
    }


def main():
    REGION = os.environ.get("REGION", "full")
    SEED = int(os.environ.get("SEED", "0"))
    import random
    import numpy as np
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    _REGION_ETA = {
        "full":   dict(min_abs_eta=None, max_abs_eta=3),
        "barrel": dict(min_abs_eta=None, max_abs_eta=1.7),
        "endcap": dict(min_abs_eta=1.3,  max_abs_eta=3),
    }
    if REGION not in _REGION_ETA:
        raise ValueError(f"REGION must be 'full', 'barrel' or 'endcap', got {REGION!r}")

    config = {
        "architecture": "concat_transformer_eta_phi_pt_z0_charge",
        "high_level_dim": 41,
        "region": REGION,
        "max_abs_eta": _REGION_ETA[REGION]["max_abs_eta"],
        "min_abs_eta": _REGION_ETA[REGION]["min_abs_eta"],
        "use_angular_features": True,
        "use_cluster_features": True,
        "dataset": "colliderml_release1_zee_prompt_electrons",
        "parquet_path": "data/electrons/electrons.parquet",
        "target_stats_path": "data/electrons/target_stats.json",

        "target_cols": ["truth_eta", "truth_phi", "truth_log_pt", "truth_z0"],

        "max_cells": 128,
        "model_dim": 256,
        "n_heads": 8,
        "n_layers": 6,
        "dim_feedforward": 1024,
        "dropout": 0.1,
        "output_dim": 5,

        "batch_size": 256,
        "n_epochs": 120,
        "min_epochs": 120,
        "learning_rate": 5e-4,
        "weight_decay": 1e-4,
        "warmup_epochs": 5,

        "eta_weight": 1.0,
        "phi_weight": 1.0,
        "logpt_weight": 1.0,
        "z0_weight": 1.0,
        "charge_weight": 1.0,

        "log_freq_batches": 10,
        "watch_gradients": False,

        "model_type": "attnpool",  # "concat" or "conv" or "attnpool"
        "conv_dim": 256,
        "kernel_size": 5,
        "n_queries": 4,

        "feature_set": "xyz_loge_eta_phi_theta",

        "seed": SEED,

        "num_workers": 8,
        "grad_clip": 5.0,
        "use_amp": True,
    }

    with wandb.init(
        project="colliderml-electron-calo",
        name="eta-phi-pt-z0",
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
            use_cluster_features=cfg["use_cluster_features"],
            max_abs_eta=cfg.get("max_abs_eta"),
            min_abs_eta=cfg.get("min_abs_eta"),
            num_workers=cfg.get("num_workers", 0)
        )

        val_loader = make_loader(
            parquet_path=parquet_path,
            split="val",
            target_stats_path=stats_path,
            batch_size=cfg["batch_size"],
            shuffle=False,
            use_angular_features=cfg["use_angular_features"],
            use_cluster_features=cfg["use_cluster_features"],
            max_abs_eta=cfg.get("max_abs_eta"),
            min_abs_eta=cfg.get("min_abs_eta"),
            num_workers=cfg.get("num_workers", 0)
        )

        common = dict(
            max_cells=cfg["max_cells"], model_dim=cfg["model_dim"],
            n_heads=cfg["n_heads"], n_layers=cfg["n_layers"],
            dim_feedforward=cfg["dim_feedforward"], dropout=cfg["dropout"],
            output_dim=cfg["output_dim"], high_level_dim=cfg["high_level_dim"],
        )
        model_type = cfg.get("model_type", "concat")
        if model_type == "conv":
            model = ConvCaloRegressor(**common, conv_dim=cfg["conv_dim"],
                                      kernel_size=cfg["kernel_size"]).to(device)
        elif model_type == "attnpool":
            from colliderml_electron.model import AttnPoolCaloRegressor
            model = AttnPoolCaloRegressor(**common, n_queries=cfg.get("n_queries", 4)).to(device)
        else:
            model = ConcatCaloRegressor(**common).to(device)

        if cfg["watch_gradients"]:
            run.watch(model, log="gradients", log_freq=100)

        loss_fn = KinematicLoss(
            target_stats_path=stats_path,
            eta_weight=cfg["eta_weight"],
            phi_weight=cfg["phi_weight"],
            logpt_weight=cfg["logpt_weight"],
            z0_weight=cfg.get("z0_weight", 1.0),
            charge_weight=cfg.get("charge_weight", 1.0),
        ).to(device)

        optimizer = torch.optim.AdamW(
            [
                {"params": model.parameters()},
                {"params": loss_fn.parameters(), "weight_decay": 0.0},
            ],
            lr=cfg["learning_rate"],
            weight_decay=cfg["weight_decay"],
        )

        warmup = cfg.get("warmup_epochs", 0)
        n_epochs = cfg["n_epochs"]

        def lr_lambda(epoch):  # epoch is 0-indexed by LambdaLR
            if warmup > 0 and epoch < warmup:
                return (epoch + 1) / warmup
            import math
            progress = (epoch - warmup) / max(1, n_epochs - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        global_step = 0
        best_val_loss = float("inf")
        best_val_phi_loss = float("inf")
        best_val_pt_rel_rmse = float("inf")

        best_state = None
        epochs_no_improve = 0

        for epoch in range(1, cfg["n_epochs"] + 1):
            model.train()

            total_train_loss = 0.0
            n_batches = 0

            for batch_idx, batch in enumerate(train_loader):
                batch = move_batch_to_device(batch, device)

                amp_on = bool(cfg.get("use_amp", False)) and device.type == "cuda"
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp_on):
                    pred = model(
                        batch["x_sampled"],
                        batch["x_high_level"],
                        batch["mask"],
                    )

                target = batch["target"][:, [ETA_INDEX, PHI_INDEX, LOGPT_INDEX, Z0_INDEX]]

                loss, logs = loss_fn(
                    pred.float(), target,
                    batch["phi_centroid"], batch["eta_centroid"], batch["log_sum_et"],
                    batch["z0_anchor"], batch["truth_charge"],
                )

                optimizer.zero_grad()
                loss.backward()

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.get("grad_clip", 5.0)
                ).item()

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
            # Charge converges far later than the regression heads and contributes
            # negligibly to loss_total, so selecting/stopping on loss_total alone
            # saves a pre-charge-climb checkpoint. Select on a combined score that
            # rewards charge accuracy explicitly. charge_acc in [0.5,1.0] -> the
            # (0.5 - acc) term is <= 0 and shrinks the score as charge improves;
            # the 2.0 scale makes a 0.90 charge head worth ~0.8 of loss_total units,
            # comparable to the regression spread, without letting charge dominate.
            charge_acc = val_logs["charge_acc"]
            selection_score = val_loss - 2.0 * (charge_acc - 0.5)
            if selection_score < best_val_loss:
                best_val_loss = selection_score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
                Path("checkpoints").mkdir(exist_ok=True)
                _tmp = Path("checkpoints/ruche_eta_phi_pt_z0_charge.pt.tmp")
                torch.save(
                    {
                        "model_state_dict": best_state,
                        "config": dict(cfg),
                        "target_cols": ["truth_eta", "truth_phi", "truth_log_pt", "truth_z0"],
                        "best_val_loss": best_val_loss,
                        "best_val_charge_acc": charge_acc,
                        "epoch": epoch,
                        "partial": True,
                    },
                    _tmp,
                )
                _tmp.replace(Path("checkpoints/ruche_eta_phi_pt_z0_charge.pt"))
            else:
                epochs_no_improve += 1
            # Early stopping is gated on a minimum epoch count: the charge head
            # converges far more slowly than the regression heads that dominate
            # total val loss, and patience-10 alone cut a previous run at epoch
            # 49 with charge accuracy still climbing.
            if epoch >= cfg.get("min_epochs", 60) and epochs_no_improve >= 10:
                print(f"early stop at epoch {epoch}")
                break
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
                    "val/charge_acc": val_logs["charge_acc"],

                    "best_val_loss": best_val_loss,
                    "best_val_phi_loss": best_val_phi_loss,
                    "best_val_pt_rel_rmse": best_val_pt_rel_rmse,
                    "epoch": epoch,
                },
                step=global_step,
            )

            scheduler.step()

            print(
                f"epoch {epoch:03d} | "
                f"train loss {train_loss:.6f} | "
                f"val loss {val_logs['loss_total']:.6f} | "
                f"val eta {val_logs['loss_eta']:.6f} | "
                f"val phi {val_logs['loss_phi']:.6f} | "
                f"phi RMSE {val_logs['phi_rmse_rad']:.6f} rad | "
                f"pT res {val_logs['pt_rel_rmse']:.4f} | "
                f"charge acc {val_logs['charge_acc']:.4f}"
            )

        Path("checkpoints").mkdir(exist_ok=True)
        checkpoint_path = Path("checkpoints/ruche_eta_phi_pt_z0_charge.pt")

        if best_state is not None:
            model.load_state_dict(best_state)

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": dict(cfg),
                "target_cols": ["truth_eta", "truth_phi", "truth_log_pt", "truth_z0"],
                "best_val_loss": best_val_loss,
                "best_val_phi_loss": best_val_phi_loss,
                "best_val_pt_rel_rmse": best_val_pt_rel_rmse,
            },
            checkpoint_path,
        )

        artifact = wandb.Artifact(
            name="eta_phi_pt_z0",
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