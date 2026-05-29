"""
Test / evaluate the eta + phi + pT model (angular-feature inputs).

Derived from scripts/test_eta_phi_angular_features.py, extended for the third
physics target log(pT).

Model output layout (output_dim = 4):

    [eta, phi_cos, phi_sin, log_pt]

Decoding:
    eta    -> denormalize with target_stats
    phi    -> atan2(phi_sin, phi_cos)            (radians)
    pT     -> exp(denormalize(log_pt))           (GeV)

pT is reported as a *fractional* resolution: the residual
(pred_pT - true_pT) / true_pT is fit with the same 3-sigma-truncated Gaussian
used for eta/phi, so its sigma is directly comparable to the "1.5%"-style
track-parameter resolutions in the thesis.

Prereqs (same as training):
  - parquet has a `truth_log_pt` column
  - target_stats.json contains stats for `truth_log_pt`
  - a checkpoint saved by train_eta_phi_pt_angular_features.py
"""

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


def wrap_phi(phi: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(phi), np.cos(phi))


def angular_residual(pred_phi: np.ndarray, true_phi: np.ndarray) -> np.ndarray:
    return wrap_phi(pred_phi - true_phi)


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

        # gathered in head-output order: [eta, phi, log_pt]
        target = batch["target"][:, [ETA_INDEX, PHI_INDEX, LOGPT_INDEX]]

        preds.append(pred.cpu().numpy())
        targets.append(target.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)

    return preds, targets


def plot_expected_vs_predicted(true_values, pred_values, name, output_dir, unit=""):
    path = output_dir / f"expected_vs_predicted_{name}.png"

    plt.figure(figsize=(6, 6))
    plt.scatter(true_values, pred_values, s=8, alpha=0.5)

    low = min(true_values.min(), pred_values.min())
    high = max(true_values.max(), pred_values.max())

    plt.plot([low, high], [low, high], linestyle="--", label="perfect prediction")

    suffix = f" [{unit}]" if unit else ""
    plt.xlabel(f"True {name}{suffix}")
    plt.ylabel(f"Predicted {name}{suffix}")
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

    # If you trained the "concat" variant, point these at eta_phi_pt_concat.* instead.
    checkpoint_path = Path("checkpoints/eta_phi_pt_conv.pt")
    parquet_path = Path("data/electrons/electrons.parquet")
    stats_path = Path("data/electrons/target_stats.json")
    output_dir = Path("results/eta_phi_pt_conv")

    output_dir.mkdir(parents=True, exist_ok=True)

    for p in (checkpoint_path, parquet_path, stats_path):
        if not p.exists():
            raise FileNotFoundError(f"Could not find {p}")

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
    phi_mean, phi_std = stats["truth_phi"]["mean"], stats["truth_phi"]["std"]
    logpt_mean, logpt_std = stats["truth_log_pt"]["mean"], stats["truth_log_pt"]["std"]

    # ---- decode predictions ----
    pred_eta = pred_norm[:, 0] * eta_std + eta_mean
    pred_phi = np.arctan2(pred_norm[:, 2], pred_norm[:, 1])      # cos/sin -> radians
    pred_logpt = pred_norm[:, 3] * logpt_std + logpt_mean
    pred_pt = np.exp(pred_logpt)                                  # GeV

    # ---- decode truth ----
    true_eta = target_norm[:, 0] * eta_std + eta_mean
    true_phi = wrap_phi(target_norm[:, 1] * phi_std + phi_mean)
    true_logpt = target_norm[:, 2] * logpt_std + logpt_mean
    true_pt = np.exp(true_logpt)                                  # GeV

    # ---- residuals ----
    eta_residual = pred_eta - true_eta
    phi_residual = angular_residual(pred_phi, true_phi)
    pt_rel_residual = (pred_pt - true_pt) / true_pt               # fractional (thesis-style)

    metrics = {
        "test/eta_mae": float(np.mean(np.abs(eta_residual))),
        "test/eta_rmse": float(np.sqrt(np.mean(eta_residual**2))),
        "test/eta_bias": float(np.mean(eta_residual)),
        "test/phi_mae_rad": float(np.mean(np.abs(phi_residual))),
        "test/phi_rmse_rad": float(np.sqrt(np.mean(phi_residual**2))),
        "test/phi_bias_rad": float(np.mean(phi_residual)),
        "test/pt_rel_mae": float(np.mean(np.abs(pt_rel_residual))),
        "test/pt_rel_rmse": float(np.sqrt(np.mean(pt_rel_residual**2))),
        "test/pt_rel_bias": float(np.mean(pt_rel_residual)),
        "test/pt_abs_rmse_gev": float(np.sqrt(np.mean((pred_pt - true_pt) ** 2))),
    }

    eta_fit = gaussian_resolution(eta_residual, wrap=False)
    phi_fit = gaussian_resolution(phi_residual, wrap=True)
    pt_fit = gaussian_resolution(pt_rel_residual, wrap=False)
    metrics.update({
        "test/eta_sigma":        eta_fit.sigma,
        "test/eta_bias_fit":     eta_fit.mu,
        "test/eta_tail_frac":    eta_fit.tail_fraction,
        "test/phi_sigma_rad":    phi_fit.sigma,
        "test/phi_bias_fit_rad": phi_fit.mu,
        "test/phi_tail_frac":    phi_fit.tail_fraction,
        "test/pt_sigma_rel":     pt_fit.sigma,        # fractional pT resolution
        "test/pt_bias_fit_rel":  pt_fit.mu,
        "test/pt_tail_frac":     pt_fit.tail_fraction,
    })

    print(f"eta  sigma={eta_fit.sigma:.6f}       tail={eta_fit.tail_fraction:.2%}")
    print(f"phi  sigma={phi_fit.sigma:.6f} rad   tail={phi_fit.tail_fraction:.2%}")
    print(f"pT   sigma={pt_fit.sigma:.4%} (frac)  tail={pt_fit.tail_fraction:.2%}")

    print("\nTest metrics:")
    print(f"eta MAE:        {metrics['test/eta_mae']:.6f}")
    print(f"eta RMSE:       {metrics['test/eta_rmse']:.6f}")
    print(f"eta bias:       {metrics['test/eta_bias']:.6f}")
    print(f"phi MAE rad:    {metrics['test/phi_mae_rad']:.6f}")
    print(f"phi RMSE rad:   {metrics['test/phi_rmse_rad']:.6f}")
    print(f"phi bias rad:   {metrics['test/phi_bias_rad']:.6f}")
    print(f"pT rel MAE:     {metrics['test/pt_rel_mae']:.4%}")
    print(f"pT rel RMSE:    {metrics['test/pt_rel_rmse']:.4%}")
    print(f"pT rel bias:    {metrics['test/pt_rel_bias']:.4%}")
    print(f"pT abs RMSE:    {metrics['test/pt_abs_rmse_gev']:.4f} GeV")

    plot_paths = [
        plot_expected_vs_predicted(true_eta, pred_eta, "eta", output_dir),
        plot_expected_vs_predicted(true_phi, pred_phi, "phi", output_dir, unit="rad"),
        plot_expected_vs_predicted(true_pt, pred_pt, "pt", output_dir, unit="GeV"),
        plot_residuals(eta_residual, "eta", output_dir, unit="", wrap=False),
        plot_residuals(phi_residual, "phi", output_dir, unit="rad", wrap=True),
        plot_residuals(pt_rel_residual, "pt_rel", output_dir, unit="", wrap=False),
    ]

    metrics_path = output_dir / "test_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print(f"\nSaved plots to: {output_dir}")
    print(f"Saved metrics to: {metrics_path}")

    with wandb.init(
        project="colliderml-electron-calo",
        name="eta-phi-pt-angular-features-test",
        job_type="evaluation",
        config=dict(config),
    ) as run:
        run.log(metrics)

        for path in plot_paths:
            run.log({path.stem: wandb.Image(str(path))})

        run.save(str(metrics_path))


if __name__ == "__main__":
    main()