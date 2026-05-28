import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb

from colliderml_electron.dataset import make_loader, TARGET_COLS
from colliderml_electron.model import ConcatCaloRegressor, ConvCaloRegressor
from colliderml_electron.resolution import gaussian_resolution, plot_residual_fit

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


def wrap_phi(phi: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(phi), np.cos(phi))


def angular_residual(pred_phi: np.ndarray, true_phi: np.ndarray) -> np.ndarray:
    return wrap_phi(pred_phi - true_phi)


def denormalize_eta_phi(values_norm: np.ndarray, stats: dict) -> np.ndarray:
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

    preds = []
    targets = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        pred = model(
            batch["x_sampled"],
            batch["x_high_level"],
            batch["mask"],
        )

        target_eta_phi = batch["target"][:, [ETA_INDEX, PHI_INDEX]]

        preds.append(pred.cpu().numpy())
        targets.append(target_eta_phi.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)

    return preds, targets


def plot_expected_vs_predicted(true_values, pred_values, name, output_dir):
    path = output_dir / f"expected_vs_predicted_{name}.png"

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
    plt.savefig(path, dpi=150)
    plt.close()

    return path


def plot_residuals(residuals, name, output_dir, unit="", wrap=False):
    path = output_dir / f"residuals_{name}.png"

    fig, ax = plt.subplots(figsize=(7, 5))

    plot_residual_fit(
        residuals=residuals,
        name=name,
        unit=unit,
        wrap=wrap,
        bins=60,
        ax=ax,
    )

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

    return path


def main():
    device = get_device()
    print(f"Using device: {device}")

    checkpoint_path = Path("checkpoints/eta_phi_conv_theta.pt")
    parquet_path = Path("data/electrons/electrons.parquet")
    stats_path = Path("data/electrons/target_stats.json")
    output_dir = Path("results/eta_phi_conv_theta")


    output_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Could not find {checkpoint_path}")

    if not parquet_path.exists():
        raise FileNotFoundError(f"Could not find {parquet_path}")

    if not stats_path.exists():
        raise FileNotFoundError(f"Could not find {stats_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    stats = json.loads(stats_path.read_text())

    test_loader = make_loader(
        parquet_path=parquet_path,
        split="test",
        target_stats_path=stats_path,
        batch_size=config["batch_size"],
        shuffle=False,
        use_angular_features=True,
    )

    common = dict(
        max_cells=config["max_cells"], model_dim=config["model_dim"],
        n_heads=config["n_heads"], n_layers=config["n_layers"],
        dim_feedforward=config["dim_feedforward"], dropout=config["dropout"],
        output_dim=config["output_dim"], high_level_dim=config["high_level_dim"],
    )
    if config.get("model_type", "concat") == "conv":
        model = ConvCaloRegressor(**common, conv_dim=config["conv_dim"],
                                  kernel_size=config["kernel_size"])
    else:
        model = ConcatCaloRegressor(**common)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    pred_norm, target_norm = collect_predictions(model, test_loader, device)

    eta_mean, eta_std = stats["truth_eta"]["mean"], stats["truth_eta"]["std"]
    pred_eta = pred_norm[:, 0] * eta_std + eta_mean
    pred_phi = np.arctan2(pred_norm[:, 2], pred_norm[:, 1])   # decode cos/sin -> radians

    true = denormalize_eta_phi(target_norm, stats)            # targets unchanged: 2 cols
    true_eta = true[:, 0]
    true_phi = wrap_phi(true[:, 1])

    eta_residual = pred_eta - true_eta
    phi_residual = angular_residual(pred_phi, true_phi)

    metrics = {
        "test/eta_mae": float(np.mean(np.abs(eta_residual))),
        "test/eta_rmse": float(np.sqrt(np.mean(eta_residual**2))),
        "test/eta_bias": float(np.mean(eta_residual)),
        "test/phi_mae_rad": float(np.mean(np.abs(phi_residual))),
        "test/phi_rmse_rad": float(np.sqrt(np.mean(phi_residual**2))),
        "test/phi_bias_rad": float(np.mean(phi_residual)),
    }

    eta_fit = gaussian_resolution(eta_residual, wrap=False)
    phi_fit = gaussian_resolution(phi_residual, wrap=True)
    metrics.update({
        "test/eta_sigma":        eta_fit.sigma,
        "test/eta_bias_fit":     eta_fit.mu,
        "test/eta_tail_frac":    eta_fit.tail_fraction,
        "test/phi_sigma_rad":    phi_fit.sigma,
        "test/phi_bias_fit_rad": phi_fit.mu,
        "test/phi_tail_frac":    phi_fit.tail_fraction,
    })
    print(f"eta  sigma={eta_fit.sigma:.6f}  tail={eta_fit.tail_fraction:.2%}")
    print(f"phi  sigma={phi_fit.sigma:.6f} rad  tail={phi_fit.tail_fraction:.2%}")

    print("\nTest metrics:")
    print(f"eta MAE:       {metrics['test/eta_mae']:.6f}")
    print(f"eta RMSE:      {metrics['test/eta_rmse']:.6f}")
    print(f"eta bias:      {metrics['test/eta_bias']:.6f}")
    print(f"phi MAE rad:   {metrics['test/phi_mae_rad']:.6f}")
    print(f"phi RMSE rad:  {metrics['test/phi_rmse_rad']:.6f}")
    print(f"phi bias rad:  {metrics['test/phi_bias_rad']:.6f}")

    plot_paths = [
    plot_expected_vs_predicted(true_eta, pred_eta, "eta", output_dir),
    plot_expected_vs_predicted(true_phi, pred_phi, "phi", output_dir),
    plot_residuals(
        eta_residual,
        "eta",
        output_dir,
        unit="",
        wrap=False,
    ),
    plot_residuals(
        phi_residual,
        "phi",
        output_dir,
        unit="rad",
        wrap=True,
    ),
]

    metrics_path = output_dir / "test_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print(f"\nSaved plots to: {output_dir}")
    print(f"Saved metrics to: {metrics_path}")

    with wandb.init(
        project="colliderml-electron-calo",
        name="eta-phi-angular-features-test",
        job_type="evaluation",
        config=dict(config),
    ) as run:
        run.log(metrics)

        for path in plot_paths:
            run.log({path.stem: wandb.Image(str(path))})

        run.save(str(metrics_path))


if __name__ == "__main__":
    main()