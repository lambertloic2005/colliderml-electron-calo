from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
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


def wrapped_angle_delta(pred_phi: np.ndarray, true_phi: np.ndarray) -> np.ndarray:
    """
    Compute pred_phi - true_phi while respecting angle wraparound.

    This keeps the difference in [-pi, pi].
    """
    delta = pred_phi - true_phi
    return np.arctan2(np.sin(delta), np.cos(delta))


def denormalize_eta_phi(values_norm: np.ndarray, stats: dict) -> np.ndarray:
    """
    Convert normalized eta/phi back to physical values.

    Input shape:
        (N, 2)

    Column 0 = eta
    Column 1 = phi
    """

    values = values_norm.copy()

    eta_mean = stats["truth_eta"]["mean"]
    eta_std = stats["truth_eta"]["std"]

    phi_mean = stats["truth_phi"]["mean"]
    phi_std = stats["truth_phi"]["std"]

    values[:, 0] = values[:, 0] * eta_std + eta_mean
    values[:, 1] = values[:, 1] * phi_std + phi_mean

    return values


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()

    all_preds = []
    all_targets = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        pred = model(
            batch["x_sampled"],
            batch["x_high_level"],
            batch["mask"],
        )

        target_eta_phi = batch["target"][:, [ETA_INDEX, PHI_INDEX]]

        all_preds.append(pred.cpu().numpy())
        all_targets.append(target_eta_phi.cpu().numpy())

    pred_norm = np.concatenate(all_preds, axis=0)
    target_norm = np.concatenate(all_targets, axis=0)

    return pred_norm, target_norm


def plot_expected_vs_predicted(
    true_values: np.ndarray,
    pred_values: np.ndarray,
    name: str,
    output_dir: Path,
) -> Path:
    output_path = output_dir / f"expected_vs_predicted_{name}.png"

    plt.figure(figsize=(6, 6))
    plt.scatter(true_values, pred_values, s=8, alpha=0.5)

    low = min(true_values.min(), pred_values.min())
    high = max(true_values.max(), pred_values.max())

    plt.plot([low, high], [low, high], linestyle="--", label="perfect prediction")

    plt.xlabel(f"True {name}")
    plt.ylabel(f"Predicted {name}")
    plt.title(f"Expected vs predicted: {name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path


def plot_residuals(
    residuals: np.ndarray,
    name: str,
    output_dir: Path,
) -> Path:
    output_path = output_dir / f"residuals_{name}.png"

    plt.figure(figsize=(7, 5))
    plt.hist(residuals, bins=50)
    plt.xlabel(f"Prediction - truth for {name}")
    plt.ylabel("Count")
    plt.title(f"Residuals: {name}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path


def main():
    device = get_device()
    print(f"Using device: {device}")

    checkpoint_path = Path("checkpoints/eta_phi_baseline.pt")
    parquet_path = Path("data/electrons/electrons.parquet")
    stats_path = Path("data/electrons/target_stats.json")

    output_dir = Path("results/eta_phi_baseline")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Could not find {checkpoint_path}")

    if not parquet_path.exists():
        raise FileNotFoundError(f"Could not find {parquet_path}")

    if not stats_path.exists():
        raise FileNotFoundError(f"Could not find {stats_path}")

    stats = json.loads(stats_path.read_text())

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]

    test_loader = make_loader(
        parquet_path=parquet_path,
        split="test",
        target_stats_path=stats_path,
        batch_size=config["batch_size"],
        shuffle=False,
    )

    model = ConcatCaloRegressor(
        max_cells=config["max_cells"],
        model_dim=config["model_dim"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        dim_feedforward=config["dim_feedforward"],
        dropout=config["dropout"],
        output_dim=2,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    pred_norm, target_norm = collect_predictions(model, test_loader, device)

    pred = denormalize_eta_phi(pred_norm, stats)
    true = denormalize_eta_phi(target_norm, stats)

    pred_eta = pred[:, 0]
    pred_phi = pred[:, 1]

    true_eta = true[:, 0]
    true_phi = true[:, 1]

    eta_residual = pred_eta - true_eta
    phi_residual = wrapped_angle_delta(pred_phi, true_phi)

    metrics = {
        "test/eta_mae": float(np.mean(np.abs(eta_residual))),
        "test/eta_rmse": float(np.sqrt(np.mean(eta_residual**2))),
        "test/eta_bias": float(np.mean(eta_residual)),

        "test/phi_mae_rad": float(np.mean(np.abs(phi_residual))),
        "test/phi_rmse_rad": float(np.sqrt(np.mean(phi_residual**2))),
        "test/phi_bias_rad": float(np.mean(phi_residual)),
    }

    print("\nTest metrics:")
    print(f"eta MAE:       {metrics['test/eta_mae']:.6f}")
    print(f"eta RMSE:      {metrics['test/eta_rmse']:.6f}")
    print(f"eta bias:      {metrics['test/eta_bias']:.6f}")
    print(f"phi MAE rad:   {metrics['test/phi_mae_rad']:.6f}")
    print(f"phi RMSE rad:  {metrics['test/phi_rmse_rad']:.6f}")
    print(f"phi bias rad:  {metrics['test/phi_bias_rad']:.6f}")

    plot_paths = []

    plot_paths.append(
        plot_expected_vs_predicted(
            true_values=true_eta,
            pred_values=pred_eta,
            name="eta",
            output_dir=output_dir,
        )
    )

    plot_paths.append(
        plot_expected_vs_predicted(
            true_values=true_phi,
            pred_values=pred_phi,
            name="phi",
            output_dir=output_dir,
        )
    )

    plot_paths.append(
        plot_residuals(
            residuals=eta_residual,
            name="eta",
            output_dir=output_dir,
        )
    )

    plot_paths.append(
        plot_residuals(
            residuals=phi_residual,
            name="phi",
            output_dir=output_dir,
        )
    )

    metrics_path = output_dir / "test_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print(f"\nSaved plots to: {output_dir}")
    print(f"Saved metrics to: {metrics_path}")

    with wandb.init(
        project="colliderml-electron-calo",
        name="eta-phi-test",
        job_type="evaluation",
        config=dict(config),
    ) as run:
        run.log(metrics)

        for path in plot_paths:
            run.log({path.stem: wandb.Image(str(path))})

        run.save(str(metrics_path))


if __name__ == "__main__":
    main()