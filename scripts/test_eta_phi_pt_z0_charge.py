# =============================================================================
# JOB 2 -- COMBO -- branch: combo-floor
# Paste as: scripts/test_eta_phi_pt_z0_charge.py   (keep this exact filename/path)
# Same evaluation code as the retreat-control copy; only the default
# CHECKPOINT/OUTPUT_DIR strings differ (env overrides supersede both anyway).
# =============================================================================
"""
Test / evaluate the eta + phi + pT + z0 + charge model.

Model output layout (output_dim = 5):

    [eta, phi, log_pt, z0, charge_logit]

Decoding:
    eta    -> denormalize with target_stats
    phi    -> denormalize (single signed head; radians)
    pT     -> exp(denormalize(log_pt))           (GeV)
    z0     -> denormalize with target_stats      (mm)
    charge -> sigmoid(charge_logit) > 0.5 => positron

Resolutions are 3-sigma-truncated Gaussian fits; pT is reported as a
fractional resolution (pred - true)/true.

Env overrides (all optional; defaults below):
    CHECKPOINT, OUTPUT_DIR, PARQUET, STATS_PATH   -- paths
    MIN_PT_EVAL, MAX_ABS_ETA_EVAL, MIN_ABS_ETA_EVAL -- eval-time acceptance
STATS_PATH must always match the stats the checkpoint was TRAINED with.
"""

import inspect
import json
import os
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
    # Both overridable from the shell so baseline vs candidate scoring never
    # requires editing this file:
    #   CHECKPOINT=... OUTPUT_DIR=... MIN_PT_EVAL=10 python scripts/test_...
    checkpoint_path = Path(os.environ.get(
        "CHECKPOINT", "checkpoints/ruche/ruche_Jul08_pointing_upgrade_full.pt"))
    parquet_path = Path(os.environ.get(
        "PARQUET", "data/electrons/eta_phi_pt_z0_charge/zee_pu200_z0_charge.parquet"))
    # CRITICAL after any dataset rebuild: a checkpoint must be decoded with the
    # SAME target stats it was trained with. Point STATS_PATH at the versioned
    # stats file matching the checkpoint being scored.
    stats_path = Path(os.environ.get(
        "STATS_PATH", "data/electrons/eta_phi_pt_z0_charge/target_stats.json"))
    output_dir = Path(os.environ.get(
        "OUTPUT_DIR", "results/ruche/Jul08_pointing_upgrade_full"))

    output_dir.mkdir(parents=True, exist_ok=True)

    for p in (checkpoint_path, parquet_path, stats_path):
        if not p.exists():
            raise FileNotFoundError(f"Could not find {p}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]

    # --- A/B overrides: force eval-time acceptance cuts regardless of what
    # the checkpoint was trained with, so two models can be scored on the
    # IDENTICAL test population. Usage:
    #   MIN_PT_EVAL=10 MAX_ABS_ETA_EVAL=1.7 python scripts/test_...
    # These flow through config, so the loader AND the charge_df reload
    # apply the same cuts.
    for _env, _key in (("MIN_PT_EVAL", "min_pt"),
                       ("MAX_ABS_ETA_EVAL", "max_abs_eta"),
                       ("MIN_ABS_ETA_EVAL", "min_abs_eta")):
        _v = os.environ.get(_env)
        if _v is not None:
            config[_key] = float(_v)
            print(f"[override] eval {_key} forced to {config[_key]}")

    stats = json.loads(stats_path.read_text())

    # Older branches (e.g. final-change) have no min_pt in make_loader; there
    # the pT floor is applied as a post-inference event mask instead (below),
    # which selects exactly the same events for the metrics.
    loader_kwargs = dict(
        parquet_path=parquet_path,
        split="test",
        target_stats_path=stats_path,
        batch_size=config["batch_size"],
        shuffle=False,
        use_angular_features=True,
        use_cluster_features=config.get("use_cluster_features", False),
        max_abs_eta=config.get("max_abs_eta"),
        min_abs_eta=config.get("min_abs_eta"),
    )
    loader_filters_pt = "min_pt" in inspect.signature(make_loader).parameters
    if loader_filters_pt:
        loader_kwargs["min_pt"] = config.get("min_pt")
    test_loader = make_loader(**loader_kwargs)

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

    # --- post-inference pT floor (only when the loader could not filter):
    # mask the raw arrays BEFORE decoding so every downstream quantity
    # (residuals, charge, plots, metrics) sees the identical event set. ---
    post_mask = None
    if (config.get("min_pt") is not None) and not loader_filters_pt:
        _true_pt_full = np.exp(target_norm[:, 2] * logpt_std + logpt_mean)
        post_mask = _true_pt_full >= float(config["min_pt"])
        print(f"[post-mask] pT >= {config['min_pt']} GeV: "
              f"{len(post_mask)} -> {int(post_mask.sum())} events")
        pred_norm = pred_norm[post_mask]
        target_norm = target_norm[post_mask]
        phi_centroid = phi_centroid[post_mask]
        eta_centroid = eta_centroid[post_mask]
        log_sum_et = log_sum_et[post_mask]
        z0_anchor = z0_anchor[post_mask]
        charge = charge[post_mask]

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

        # ---- z0 diagnostic excluding central model guesses ----
    # This keeps only events where the MODEL PREDICTION is outside [-10, +10] mm.
    z0_outside_center = (pred_z0 < -10.0) | (pred_z0 > 10.0)

    true_z0_out = true_z0[z0_outside_center]
    pred_z0_out = pred_z0[z0_outside_center]
    z0_residual_out = z0_residual[z0_outside_center]
    z0_anchor_out = z0_anchor[z0_outside_center]

    np.savez(
        output_dir / "preds.npz",
        truth_pt=true_pt, truth_eta=true_eta, truth_phi=true_phi,
        pred_eta=pred_eta, pred_phi=pred_phi, pred_pt=pred_pt,
        truth_z0=true_z0, pred_z0=pred_z0, z0_anchor=z0_anchor,
        charge=charge, charge_logit=charge_logit,
        # optional anchor-baseline overlay (these names already exist in your decode):
        eta_anchor=eta_centroid, phi_anchor=phi_centroid, pt_anchor=np.exp(log_sum_et),
    )

    print("\nz0 filtered sample: predicted z0 outside [-10,+10] mm")
    print(
        f" kept {int(z0_outside_center.sum())}/{len(z0_outside_center)} events "
        f"({100.0 * z0_outside_center.mean():.1f}%)"
    )

    if z0_outside_center.any():
        print(f" filtered z0 RMSE: {np.sqrt(np.mean(z0_residual_out**2)):.2f} mm")
        print(f" filtered z0 MAE: {np.mean(np.abs(z0_residual_out)):.2f} mm")
        print(f" filtered z0 bias: {np.mean(z0_residual_out):+.2f} mm")
        print(
            f" filtered anchor RMSE: "
            f"{np.sqrt(np.mean((z0_anchor_out - true_z0_out)**2)):.2f} mm"
        )
    else:
        print(" no events passed the filtered z0 cut")
    
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
        pl.read_parquet(parquet_path, columns=["split", "truth_charge", "truth_eta", "truth_log_pt"])
          .filter(pl.col("split") == "test")
    )
    if config.get("max_abs_eta") is not None:
        charge_df = charge_df.filter(pl.col("truth_eta").abs() <= config["max_abs_eta"])
    if config.get("min_abs_eta") is not None:
        charge_df = charge_df.filter(pl.col("truth_eta").abs() >= config["min_abs_eta"])
    if loader_filters_pt and config.get("min_pt") is not None:
        charge_df = charge_df.filter(pl.col("truth_log_pt") >= float(np.log(config["min_pt"])))
    charge = charge_df["truth_charge"].to_numpy()
    if post_mask is not None:
        charge = charge[post_mask]        # same event mask as the residual arrays
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

    # ============== charge classifier performance plots ==============
    y_true = (charge > 0).astype(int)          # 1 = positron
    p_pos = 1.0 / (1.0 + np.exp(-charge_logit)) # ensure defined here too

    # (1) logit distribution separated by truth charge -- the raw separation
    plt.figure(figsize=(7, 5))
    lo, hi = np.percentile(charge_logit, [1, 99])
    lbins = np.linspace(lo, hi, 60)
    plt.hist(charge_logit[charge == -1], bins=lbins, alpha=0.6, label="electron (q=-1)")
    plt.hist(charge_logit[charge == +1], bins=lbins, alpha=0.6, label="positron (q=+1)")
    plt.axvline(0.0, color="k", ls="--", lw=1, label="decision boundary")
    plt.xlabel("charge logit  (>0 => predict positron)"); plt.ylabel("Count"); plt.legend()
    plt.title("Charge logit separated by truth charge")
    plt.tight_layout(); plt.savefig(output_dir / "charge_logit_by_truth.png", dpi=150); plt.close()

    # (2) ROC curve -- threshold-free separation, area = charge_auc
    thr = np.sort(np.unique(charge_logit))
    P = max(int(np.sum(y_true == 1)), 1); N = max(int(np.sum(y_true == 0)), 1)
    tpr = np.array([np.sum((charge_logit >= t) & (y_true == 1)) / P for t in thr])
    fpr = np.array([np.sum((charge_logit >= t) & (y_true == 0)) / N for t in thr])
    auc = float(np.trapz(np.sort(tpr), np.sort(fpr)))  # matches roc_auc_score up to rounding
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, lw=2, label=f"ROC (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], ls="--", color="grey", label="chance")
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("Charge ROC (positron = positive)"); plt.legend()
    plt.tight_layout(); plt.savefig(output_dir / "charge_roc.png", dpi=150); plt.close()

    # (3) accuracy vs pT, with per-event mean confidence on the SAME [0.5,1] axis.
    # Both curves now in probability units, so any gap = genuine miscalibration,
    # not an axis artifact. confidence = P(chosen class) = max(p_pos, 1-p_pos).
    pc_all = np.maximum(p_pos, 1.0 - p_pos)
    edges = np.array([0, 2, 5, 10, 20, 50, 100, 1e9])
    centers, accs, confs = [], [], []
    for k in range(len(edges) - 1):
        m = (true_pt >= edges[k]) & (true_pt < edges[k + 1])
        if m.sum() >= 10:
            centers.append(np.median(true_pt[m]))
            accs.append(float(np.mean(pred_charge[m] == charge[m])))
            confs.append(float(np.mean(pc_all[m])))            # mean per-event confidence
    plt.figure(figsize=(7, 5))
    plt.semilogx(centers, accs, "o-", color="C0", label="accuracy")
    plt.semilogx(centers, confs, "s--", color="C1", alpha=0.8, label="mean confidence")
    plt.axhline(0.5, color="grey", ls="--", lw=1, label="chance")
    plt.xlabel("true pT [GeV]"); plt.ylabel("probability")
    plt.ylim(0.45, 1.0)
    plt.title("Calo-only charge ID vs pT")
    plt.legend(loc="upper right")
    plt.tight_layout(); plt.savefig(output_dir / "charge_acc_vs_pt.png", dpi=150); plt.close()

    # (4) reliability / calibration -- does p=0.8 mean right 80% of the time?
    pc = np.maximum(p_pos, 1.0 - p_pos)        # confidence in the chosen class
    cbins = np.linspace(0.5, 1.0, 6)
    bx, by = [], []
    for k in range(len(cbins) - 1):
        m = (pc >= cbins[k]) & (pc < cbins[k + 1])
        if m.sum() >= 10:
            bx.append(0.5 * (cbins[k] + cbins[k + 1]))
            by.append(float(np.mean(pred_charge[m] == charge[m])))
    plt.figure(figsize=(6, 6))
    plt.plot(bx, by, "o-", label="observed")
    plt.plot([0.5, 1.0], [0.5, 1.0], ls="--", color="grey", label="perfect calibration")
    plt.xlabel("predicted confidence"); plt.ylabel("empirical accuracy")
    plt.title("Charge calibration"); plt.legend()
    plt.tight_layout(); plt.savefig(output_dir / "charge_calibration.png", dpi=150); plt.close()

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
    if z0_outside_center.any():
        z0_out_fit = gaussian_resolution(z0_residual_out, wrap=False)

        metrics.update({
            "test/z0_outside_pm10_n": int(z0_outside_center.sum()),
            "test/z0_outside_pm10_frac": float(z0_outside_center.mean()),

            "test/z0_outside_pm10_rmse_mm": float(np.sqrt(np.mean(z0_residual_out**2))),
            "test/z0_outside_pm10_mae_mm": float(np.mean(np.abs(z0_residual_out))),
            "test/z0_outside_pm10_bias_mm": float(np.mean(z0_residual_out)),

            "test/z0_outside_pm10_sigma_mm": z0_out_fit.sigma,
            "test/z0_outside_pm10_bias_fit_mm": z0_out_fit.mu,
            "test/z0_outside_pm10_tail_frac": z0_out_fit.tail_fraction,

            "test/z0_outside_pm10_anchor_rmse_mm": float(
                np.sqrt(np.mean((z0_anchor_out - true_z0_out) ** 2))
            ),
        })
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

    # ================= per-region (barrel / endcap) breakdown =================
    # Set from scripts/diagnose_detector_regions.py; default matches the old
    # by-|eta| print block. The diagnostic is truth-eta-labelled; predicted eta
    # (sigma ~ 0.019) would route just as well at inference.
    BARREL_ETA_MAX = 1.5
    ENDCAP_ETA_MAX = 3.0
    REGIONS = [
        ("barrel", 0.0,            BARREL_ETA_MAX),
        ("endcap", BARREL_ETA_MAX, ENDCAP_ETA_MAX),
        ("fwd",    ENDCAP_ETA_MAX, 99.0),
    ]

    def _auc(scores, pos):  # threshold-free, no sklearn dependency
        pos = np.asarray(pos, dtype=bool)
        n_pos, n_neg = int(pos.sum()), int((~pos).sum())
        if n_pos == 0 or n_neg == 0:
            return float("nan")
        order = np.argsort(scores, kind="mergesort")
        ranks = np.empty(len(scores), dtype=np.float64)
        ranks[order] = np.arange(1, len(scores) + 1)
        return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

    abs_eta = np.abs(true_eta)
    print("\nPer-region resolution (truth-eta split):")
    for label, lo, hi in REGIONS:
        m = (abs_eta >= lo) & (abs_eta < hi)
        n = int(m.sum())
        metrics[f"test/region/{label}/n"] = n
        if n < 50:                      # guard against empty / tiny bins
            print(f"  {label:7s}: n={n} (skipped, too few)")
            continue

        eta_f = gaussian_resolution(eta_residual[m], wrap=False)
        phi_f = gaussian_resolution(phi_residual[m], wrap=True)
        pt_f  = gaussian_resolution(pt_rel_residual[m], wrap=False)
        z0_f  = gaussian_resolution(z0_residual[m], wrap=False)

        metrics[f"test/region/{label}/eta_sigma"]     = eta_f.sigma
        metrics[f"test/region/{label}/phi_sigma_rad"] = phi_f.sigma
        metrics[f"test/region/{label}/pt_sigma_rel"]  = pt_f.sigma
        metrics[f"test/region/{label}/z0_sigma_mm"]   = z0_f.sigma
        metrics[f"test/region/{label}/z0_rmse_mm"]    = float(np.sqrt(np.mean(z0_residual[m] ** 2)))
        metrics[f"test/region/{label}/z0_prior_mm"]   = float(np.std(true_z0[m]))

        # charge ID per region -- only if your current script defines `charge`
        # (truth, in {-1,+1}) and the charge logit. Rename `charge_logit` to your
        # variable. Delete this `if` block if you don't have a charge head here.
        if "charge_logit" in dir() and charge is not None:
            metrics[f"test/region/{label}/charge_auc"] = float(_auc(charge_logit[m], charge[m] > 0))
            metrics[f"test/region/{label}/charge_acc"] = float(np.mean((charge_logit[m] > 0) == (charge[m] > 0)))

        print(f"  {label:7s}: n={n:5d}  z0_sigma={z0_f.sigma:6.1f} mm "
              f"(prior {np.std(true_z0[m]):5.1f})  phi_sigma={phi_f.sigma:.4f} rad  "
              f"eta_sigma={eta_f.sigma:.4f}  pt_sigma={pt_f.sigma:.3%}")

        # per-region z0 plots (z0 is where region geometry matters most)
        plot_expected_vs_predicted(true_z0[m], pred_z0[m], f"z0_{label}", output_dir, unit="mm")
        plot_residuals(z0_residual[m], f"z0_{label}", output_dir, unit="mm", wrap=False)
    # ==========================================================================
    
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
    if z0_outside_center.any():
        plot_paths.extend([
            plot_expected_vs_predicted(
                true_z0_out,
                pred_z0_out,
                "z0_pred_outside_pm10",
                output_dir,
                unit="mm",
            ),
            plot_residuals(
                z0_residual_out,
                "z0_pred_outside_pm10",
                output_dir,
                unit="mm",
                wrap=False,
            ),
        ])
    metrics["test/min_pt_eval"] = config.get("min_pt")
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