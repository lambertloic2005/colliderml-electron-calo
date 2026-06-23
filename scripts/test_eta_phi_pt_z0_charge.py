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
from sklearn.metrics import roc_auc_score
import torch
import wandb
from sklearn.metrics import roc_auc_score

from colliderml_electron.dataset import make_loader, TARGET_COLS
from colliderml_electron.model import ConcatCaloRegressor, ConvCaloRegressor
from colliderml_electron.resolution import gaussian_resolution, plot_residual_fit

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


def wrap_phi(phi: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(phi), np.cos(phi))


def angular_residual(pred_phi: np.ndarray, true_phi: np.ndarray) -> np.ndarray:
    return wrap_phi(pred_phi - true_phi)


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()

    preds, targets = [], []
    phi_cents, eta_cents, log_sum_ets, z0_anchors, charges = [], [], [], [], []

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        pred = model(batch["x_sampled"], batch["x_high_level"], batch["mask"])
        target = batch["target"][:, [ETA_INDEX, PHI_INDEX, LOGPT_INDEX, Z0_INDEX]]

        preds.append(pred.cpu().numpy())
        targets.append(target.cpu().numpy())
        phi_cents.append(batch["phi_centroid"].cpu().numpy())
        eta_cents.append(batch["eta_centroid"].cpu().numpy())
        log_sum_ets.append(batch["log_sum_et"].cpu().numpy())
        z0_anchors.append(batch["z0_anchor"].cpu().numpy())
        charges.append(batch["truth_charge"].cpu().numpy())

    return (
        np.concatenate(preds), np.concatenate(targets),
        np.concatenate(phi_cents), np.concatenate(eta_cents),
        np.concatenate(log_sum_ets), np.concatenate(z0_anchors),
        np.concatenate(charges),
    )


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


def plot_residuals(residuals, name, output_dir, unit="", wrap=False, view_sigma=5.0):
    path = output_dir / f"residuals_{name}.png"

    fig, ax = plt.subplots(figsize=(7, 5))

    plot_residual_fit(
        residuals=residuals,
        name=name,
        unit=unit,
        wrap=wrap,
        bins=60,
        ax=ax,
        view_sigma=view_sigma,
    )

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

    return path


def main():
    device = get_device()
    print(f"Using device: {device}")

    # If you trained the "concat" variant, point these at eta_phi_pt_concat.* instead.
    checkpoint_path = Path("checkpoints/ruche/ruche_Jun23_singlePhi_z0Slice.pt")
    parquet_path = Path("data/electrons/eta_phi_pt_z0_charge/zee_pu200_z0_charge.parquet")
    stats_path = Path("data/electrons/eta_phi_pt_z0_charge/target_stats.json")
    output_dir = Path("results/ruche/Jun23_singlePhi_z0Slice")

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
        use_cluster_features=config.get("use_cluster_features", False),
        max_abs_eta=config.get("max_abs_eta"),
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

    (pred_norm, target_norm, phi_centroid,
     eta_centroid, log_sum_et, z0_anchor, charge) = collect_predictions(
        model, test_loader, device)

    eta_mean, eta_std = stats["truth_eta"]["mean"], stats["truth_eta"]["std"]
    phi_mean, phi_std = stats["truth_phi"]["mean"], stats["truth_phi"]["std"]
    logpt_mean, logpt_std = stats["truth_log_pt"]["mean"], stats["truth_log_pt"]["std"]
    z0_mean, z0_std = stats["truth_z0"]["mean"], stats["truth_z0"]["std"]

    # ---- decode predictions ----
    pred_eta = eta_centroid + pred_norm[:, 0]                    # anchor + predicted Δη
    # two phi hypotheses; convention q=-1 -> electron uses head index 1
    pred_phi = wrap_phi(phi_centroid + pred_norm[:, 1])         # charge-free, no truth used
    pred_logpt = log_sum_et + pred_norm[:, 2]                   # anchor + predicted Δln pT
    pred_pt = np.exp(pred_logpt)                                # GeV
    pred_z0 = pred_norm[:, 3] * z0_std + z0_mean          

    # ---- decode truth ----
    true_eta = target_norm[:, 0] * eta_std + eta_mean
    true_phi = wrap_phi(target_norm[:, 1] * phi_std + phi_mean)
    true_logpt = target_norm[:, 2] * logpt_std + logpt_mean
    true_pt = np.exp(true_logpt)

    charge_logit = pred_norm[:, 4]
    pred_charge = np.where(charge_logit > 0, +1, -1)           # predicted charge
    p_pos = 1.0 / (1.0 + np.exp(-charge_logit))                # P(positron) = sigmoid(logit)
    confidence = np.abs(charge_logit)                          # |logit| = how sure
    charge_acc = float(np.mean(pred_charge == charge))
    print(f"\ncharge accuracy (calo-only): {charge_acc:.4f}  n={len(charge)}")

    # per-class accuracy (Z->ee is ~balanced; confirm no sign bias)
    y = (charge > 0).astype(int)                               # 1 = positron truth
    yhat = (pred_charge > 0).astype(int)
    tp = int(np.sum((yhat == 1) & (y == 1))); tn = int(np.sum((yhat == 0) & (y == 0)))
    fp = int(np.sum((yhat == 1) & (y == 0))); fn = int(np.sum((yhat == 0) & (y == 1)))
    print(f"confusion: TP={tp} TN={tn} FP={fp} FN={fn}")
    if (tn + fp) > 0: print(f"electron acc = {tn / (tn + fp):.4f}")
    if (tp + fn) > 0: print(f"positron acc = {tp / (tp + fn):.4f}")
    
    true_z0 = target_norm[:, 3] * z0_std + z0_mean
    # Anchor-only baselines: what you'd get with the model head outputting zero.
    # The trained model must beat these, otherwise the head is learning nothing.
    print(f"anchor-only eta  std: {np.std(eta_centroid - true_eta):.5f}")
    print(f"anchor-only phi  std: {np.std(angular_residual(phi_centroid, true_phi)):.5f} rad")
    print(f"anchor-only lnpt std: {np.std(log_sum_et - true_logpt):.5f}")
    print(f"anchor-only z0   std: {np.std(z0_anchor - true_z0):.2f} mm")
    print(f"beamspot prior   std: {np.std(true_z0):.2f} mm (z0 RMS with no model at all)")                              

    # ---- residuals ----
    eta_residual = pred_eta - true_eta
    phi_residual = angular_residual(pred_phi, true_phi)
    pt_rel_residual = (pred_pt - true_pt) / true_pt               # fractional (thesis-style)

    # clean, bounded fractional-pT resolution (d ln pT ~ dpT/pT); not blown up
    # by low-pT electrons the way the linear relative residual is.
    logpt_residual = pred_logpt - true_logpt
    z0_residual = pred_z0 - true_z0
    PT_FLOOR_GEV = 5.0
    hi = true_pt >= PT_FLOOR_GEV
    pt_rel_hi = pt_rel_residual[hi]

    # --- low-pt diagnostic for the relative-pT error ---
    order = np.argsort(-np.abs(pt_rel_residual))
    print("\nWorst 10 pt_rel offenders (true_pt, pred_pt, rel_err):")
    for j in order[:10]:
        print(f"  true={true_pt[j]:8.3f} GeV  pred={pred_pt[j]:8.3f} GeV  "
              f"rel={pt_rel_residual[j]:+.2f}")
    print("\npt_rel error by true-pT bin:")
    for lo, hi_edge in [(0, 2), (2, 5), (5, 10), (10, 30), (30, 1e9)]:
        m = (true_pt >= lo) & (true_pt < hi_edge)
        if m.any():
            print(f"  [{lo:>3},{hi_edge:>5}) GeV: n={int(m.sum()):5d}  "
                  f"rel_rmse={np.sqrt(np.mean(pt_rel_residual[m]**2)):.3f}  "
                  f"rel_mae={np.mean(np.abs(pt_rel_residual[m])):.3f}")

    import polars as pl
    charge_df = (
        pl.read_parquet(parquet_path, columns=["split", "truth_charge", "truth_eta"])
          .filter(pl.col("split") == "test")
    )
    if config.get("max_abs_eta") is not None:
        charge_df = charge_df.filter(pl.col("truth_eta").abs() <= config["max_abs_eta"])
    charge = charge_df["truth_charge"].to_numpy()
    assert len(charge) == len(phi_residual), "charge/residual length mismatch"
    for q in (-1, +1):
        sel = charge == q
        print(f"phi residual (q={q:+d}): mean={phi_residual[sel].mean():+.5f} rad  "
              f"std={phi_residual[sel].std():.5f} rad  n={int(sel.sum())}")
    plt.figure(figsize=(7, 5))
    qbins = np.linspace(-0.1, 0.1, 60)
    plt.hist(phi_residual[charge == -1], bins=qbins, alpha=0.6, label="electron (q=-1)")
    plt.hist(phi_residual[charge == +1], bins=qbins, alpha=0.6, label="positron (q=+1)")
    plt.xlabel("Prediction - truth for phi [rad]"); plt.ylabel("Count"); plt.legend()
    plt.title("Phi residual split by truth charge")
    plt.tight_layout()
    plt.savefig(output_dir / "residuals_phi_by_charge.png", dpi=150)
    plt.close()

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
        "test/logpt_rmse": float(np.sqrt(np.mean(logpt_residual**2))),
        "test/logpt_bias": float(np.mean(logpt_residual)),
        f"test/pt_rel_mae_above_{int(PT_FLOOR_GEV)}gev": float(np.mean(np.abs(pt_rel_hi))) if hi.any() else float("nan"),
        f"test/pt_rel_rmse_above_{int(PT_FLOOR_GEV)}gev": float(np.sqrt(np.mean(pt_rel_hi**2))) if hi.any() else float("nan"),
        "test/z0_rmse_mm": float(np.sqrt(np.mean(z0_residual**2))),
        "test/z0_mae_mm": float(np.mean(np.abs(z0_residual))),
        "test/z0_bias_mm": float(np.mean(z0_residual)),
        "test/z0_anchor_rmse_mm": float(np.sqrt(np.mean((z0_anchor - true_z0)**2))),
        "test/z0_prior_rmse_mm": float(np.sqrt(np.mean((z0_mean - true_z0) ** 2))),
        "test/charge_acc": float(np.mean(pred_charge == charge)),
    "test/charge_auc": float(roc_auc_score((charge > 0).astype(int), p_pos)),
    }

    print("\nz0 resolution by |eta| region:")
    for lo, hi_e, label in [(0.0, 1.2, "barrel"), (1.2, 2.5, "endcap"), (2.5, 99, "fwd")]:
        m = (np.abs(true_eta) >= lo) & (np.abs(true_eta) < hi_e)
        if m.any():
            print(f"  |eta| [{lo},{hi_e}) {label:7s}: n={int(m.sum()):5d}  "
                  f"rmse={np.sqrt(np.mean(z0_residual[m]**2)):6.1f} mm  "
                  f"prior={np.std(true_z0[m]):6.1f} mm")
            
    eta_fit = gaussian_resolution(eta_residual, wrap=False)
    phi_fit = gaussian_resolution(phi_residual, wrap=True)
    pt_fit = gaussian_resolution(pt_rel_residual, wrap=False)
    logpt_fit = gaussian_resolution(logpt_residual, wrap=False)
    z0_fit = gaussian_resolution(z0_residual, wrap=False)

    # Calo-only charge ID, binned by true pT. The bend handle (Δφ ~ q·B·L/pT)
    # shrinks with pT, so accuracy should fall from ~good at low pT toward 0.5
    # (chance) at high pT. This curve IS the complementarity result.
    print("\ncharge accuracy by true-pT bin:")
    for lo, hi_edge in [(0, 2), (2, 5), (5, 10), (10, 30), (30, 1e9)]:
        m = (true_pt >= lo) & (true_pt < hi_edge)
        if m.any():
            acc = float(np.mean(pred_charge[m] == charge[m]))
            print(f"  [{lo:>4.0f}, {hi_edge:<5.0f}) GeV : acc={acc:.4f}  n={int(m.sum())}")
    print(f"phi (charge-free, no truth used): sigma~{np.std(phi_residual):.5f} rad  [headline]")
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
        "test/logpt_sigma":      logpt_fit.sigma,     # clean fractional pT resolution
        "test/logpt_tail_frac":  logpt_fit.tail_fraction,
        "test/z0_sigma_mm":      z0_fit.sigma,
        "test/z0_bias_fit_mm":   z0_fit.mu,
        "test/z0_tail_frac":     z0_fit.tail_fraction,
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
        plot_expected_vs_predicted(true_z0, pred_z0, "z0", output_dir, unit="mm"),
        plot_residuals(eta_residual, "eta", output_dir, unit="", wrap=False),
        plot_residuals(phi_residual, "phi", output_dir, unit="rad", wrap=True),
        plot_residuals(pt_rel_residual, "pt_rel", output_dir, unit="", wrap=False),
        plot_residuals(z0_residual, "z0", output_dir, unit="mm", wrap=False),   # NEW
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
