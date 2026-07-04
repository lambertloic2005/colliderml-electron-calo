"""
Permutation feature importance for the eta/phi/pT/z0/charge calorimeter-only
model (ConcatCaloRegressor / ConvCaloRegressor, final-change branch).

Answers the question: "how much does the test-set loss get worse if I destroy
the information in feature X, leaving every other feature, every other cell,
and the trained model itself untouched?"

METHOD
------
Standard permutation importance (Breiman 2001), adapted to a per-cell sequence
model instead of flat tabular rows:

  1. Run the trained model once over the test set with inputs untouched
     -> baseline losses.
  2. For each input feature -- each of the 3 `x_sampled` coordinates, and each
     of the `high_level_dim` `x_high_level` columns -- independently:
       a. Within each batch, shuffle that one feature's values across all REAL
          (non-padding) cells of the batch. Every cell keeps its own values
          for every other feature, and keeps its own event/position; only the
          permuted feature's value has been swapped for another (random) real
          cell's value of the same feature. This exactly destroys the
          statistical association between that feature and the targets while
          leaving the feature's marginal distribution, the padding mask, and
          every other feature untouched.
       b. Re-run the model, recompute losses -> permuted loss.
       c. importance = permuted_loss - baseline_loss. Bigger = more important.
          A small negative value is expected sampling noise for a genuinely
          uninformative feature, not evidence the feature actively hurts.
  3. Repeat step 2 `--n-repeats` times with independent random shuffles and
     report the mean and std of the importance, since any single shuffle is a
     noisy estimate (Fisher, Rudin & Dominici 2019, "model class reliance").

IMPORTANT CAVEAT ON THE "LOSS" BEING REPORTED
----------------------------------------------
KinematicLoss (scripts/train_eta_phi_pt_z0_charge.py) combines the four
regression Huber losses using a LEARNED homoscedastic weighting
(`loss_fn.log_sigma`, an nn.Parameter fit jointly with the model, one value
per regression task). That parameter lives on the loss module, not the model,
and the training script's checkpoint only saves `model.state_dict()` -- the
fitted log_sigma values are not recoverable from a checkpoint alone.

So this script does NOT reconstruct that exact composite training loss.
Instead it reports:
  (a) each of the five per-target losses separately, in the same functional
      form used at train time (Huber for eta/phi/logpt/z0, BCE for charge),
      so you can see which physics target a given feature actually supports;
  (b) an explicit UNWEIGHTED sum of those five as `total_loss_unweighted`,
      clearly labelled as such -- a convenient single ranking number, but NOT
      the quantity that was actually minimized during training;
  (c) the same physical-unit resolutions the test script reports (eta RMSE,
      phi RMSE in rad, fractional pT RMSE, z0 RMSE in mm, charge accuracy),
      which are unit-comparable and arguably the most interpretable numbers
      for "how much does this feature matter physically".

If you want the exact trained homoscedastic-weighted total reproduced, the
fix is on the training side: save `loss_fn.state_dict()` into the checkpoint
next to `model.state_dict()`, then load it here instead of recomputing an
unweighted sum.

USAGE
-----
    python scripts/permutation_importance.py \
        --checkpoint checkpoints/ruche/ruche_Jun23_ConstChargeWeight.pt \
        --n-repeats 3

Diagnostic outputs (JSON + bar plot) are written under `results/permutation_importance/`.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from colliderml_electron.dataset import make_loader, TARGET_COLS
from colliderml_electron.model import ConcatCaloRegressor, ConvCaloRegressor

ETA_INDEX = TARGET_COLS.index("truth_eta")
PHI_INDEX = TARGET_COLS.index("truth_phi")
LOGPT_INDEX = TARGET_COLS.index("truth_log_pt")
Z0_INDEX = TARGET_COLS.index("truth_z0")

# Detector subsystem codes, in the order dataset.py one-hot-encodes them.
DETECTOR_CODES = [9, 10, 11, 12, 13, 14]


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


def wrapped_angle_delta(pred_phi: torch.Tensor, true_phi: torch.Tensor) -> torch.Tensor:
    delta = pred_phi - true_phi
    return torch.atan2(torch.sin(delta), torch.cos(delta))


def x_sampled_feature_names() -> list[str]:
    # ElectronDataset.__getitem__ rotates the event so the energy-weighted phi
    # centroid sits at phi=0, THEN stacks (x_rot, y_rot, z) as x_sampled.
    return ["cell_x_rot", "cell_y_rot", "cell_z"]


def x_high_level_feature_names(high_level_dim: int, use_cluster_features: bool) -> list[str]:
    """
    Reproduce the exact column order ElectronDataset.__getitem__ builds
    (use_angular_features=True always here -- the test/eval scripts hardcode
    it). Falls back to generic names if high_level_dim doesn't match either
    known layout, so a checkpoint from an older/different branch never gets
    silently mislabeled.
    """
    base = ["log_e", "cell_eta", "sin_dphi_to_centroid", "cos_dphi_to_centroid",
            "theta", "cos_theta"]
    det = [f"det_code_{code}" for code in DETECTOR_CODES]
    names = base + det  # 12 columns

    if use_cluster_features:
        cluster_scalars = [
            "log_sum_e", "log_sum_et", "log_n_cells",
            "std_phi", "skew_phi", "std_eta", "skew_eta",
            "z0_anchor_over_1000", "pointing_slope_dzdr",
            "r_spread_over_1000", "pointing_fit_rms_over_100",
        ]
        prof_r = [f"radial_slice_r_{k}" for k in range(6)]
        prof_z = [f"radial_slice_z_{k}" for k in range(6)]
        prof_f = [f"radial_slice_efrac_{k}" for k in range(6)]
        names = names + cluster_scalars + prof_r + prof_z + prof_f  # +29 = 41

    if len(names) != high_level_dim:
        names = [f"high_level_feature_{i}" for i in range(high_level_dim)]

    return names


def permute_feature_inplace(tensor: torch.Tensor, feat_idx: int, mask: torch.Tensor,
                             generator: torch.Generator) -> None:
    """
    Shuffle tensor[..., feat_idx] across all REAL (mask==False) cells of the
    batch, in place. Padding cells (mask==True) are left untouched.

    tensor: (B, L, D) -- modified in place.
    mask:   (B, L)     True = padding, False = real cell.
    """
    real = ~mask
    flat_real = real.reshape(-1)
    col = tensor[..., feat_idx].reshape(-1)

    real_idx = flat_real.nonzero(as_tuple=True)[0]
    if real_idx.numel() <= 1:
        return  # nothing to shuffle

    idx_cpu = real_idx.cpu()
    perm_order = torch.randperm(idx_cpu.numel(), generator=generator)
    perm_cpu = idx_cpu[perm_order]
    perm = perm_cpu.to(real_idx.device)

    col_permuted = col.clone()
    col_permuted[real_idx] = col[perm]
    tensor[..., feat_idx] = col_permuted.reshape(tensor.shape[:-1])


@torch.no_grad()
def compute_losses(model, loader, device, target_stats: dict,
                    permute: tuple[str, int] | None = None,
                    generator: torch.Generator | None = None) -> dict:
    """
    Run the full loader once and return dataset-mean losses/metrics.

    permute: optional (tensor_name, feature_index), tensor_name in
             {"x_sampled", "x_high_level"}. If given, that one feature column
             is shuffled across the real cells of EACH BATCH independently
             (see permute_feature_inplace) before the forward pass. None runs
             the unmodified (baseline) inputs.
    """
    model.eval()

    eta_mean, eta_std = target_stats["eta_mean"], target_stats["eta_std"]
    phi_mean, phi_std = target_stats["phi_mean"], target_stats["phi_std"]
    logpt_mean, logpt_std = target_stats["logpt_mean"], target_stats["logpt_std"]
    z0_mean, z0_std = target_stats["z0_mean"], target_stats["z0_std"]

    sums = {
        "loss_eta": 0.0, "loss_phi": 0.0, "loss_logpt": 0.0,
        "loss_z0": 0.0, "loss_charge": 0.0,
        "sq_eta": 0.0, "sq_phi": 0.0, "sq_pt_rel": 0.0, "sq_z0_mm": 0.0,
        "charge_correct": 0.0,
    }
    n_events = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        x_sampled = batch["x_sampled"]
        x_high_level = batch["x_high_level"]
        mask = batch["mask"]  # True = padding

        if permute is not None:
            tensor_name, feat_idx = permute
            x_sampled = x_sampled.clone()
            x_high_level = x_high_level.clone()
            tensor = x_sampled if tensor_name == "x_sampled" else x_high_level
            permute_feature_inplace(tensor, feat_idx, mask, generator)

        pred = model(x_sampled, x_high_level, mask)

        target = batch["target"][:, [ETA_INDEX, PHI_INDEX, LOGPT_INDEX, Z0_INDEX]]
        phi_centroid = batch["phi_centroid"]
        eta_centroid = batch["eta_centroid"]
        log_sum_et = batch["log_sum_et"]
        truth_charge = batch["truth_charge"]

        pred_deta, pred_dphi = pred[:, 0], pred[:, 1]
        pred_dlogpt, pred_dz0 = pred[:, 2], pred[:, 3]
        charge_logit = pred[:, 4]

        target_eta = target[:, 0] * eta_std + eta_mean
        target_phi = target[:, 1] * phi_std + phi_mean
        target_logpt = target[:, 2] * logpt_std + logpt_mean
        z0_target_norm = target[:, 3]

        deta_target = target_eta - eta_centroid
        dphi_target = wrapped_angle_delta(target_phi, phi_centroid)
        dlogpt_target = target_logpt - log_sum_et

        # --- same functional form as KinematicLoss.forward, unweighted ---
        eta_loss = F.huber_loss(pred_deta, deta_target, delta=0.1, reduction="sum")
        phi_err = wrapped_angle_delta(pred_dphi, dphi_target)
        phi_loss = F.huber_loss(phi_err, torch.zeros_like(phi_err), delta=0.05, reduction="sum")
        logpt_loss = F.huber_loss(pred_dlogpt, dlogpt_target, delta=0.2, reduction="sum")
        z0_loss = F.huber_loss(pred_dz0, z0_target_norm, delta=1.0, reduction="sum")

        charge_label = (truth_charge > 0).float()
        charge_loss = F.binary_cross_entropy_with_logits(
            charge_logit, charge_label, reduction="sum"
        )

        B = pred.shape[0]
        sums["loss_eta"] += eta_loss.item()
        sums["loss_phi"] += phi_loss.item()
        sums["loss_logpt"] += logpt_loss.item()
        sums["loss_z0"] += z0_loss.item()
        sums["loss_charge"] += charge_loss.item()

        # --- physical-unit diagnostics (same decoding as the test script) ---
        pred_eta = eta_centroid + pred_deta
        pred_phi = phi_centroid + pred_dphi
        pred_pt = torch.exp(log_sum_et + pred_dlogpt)
        true_pt = torch.exp(target_logpt)
        pred_z0 = pred_dz0 * z0_std + z0_mean
        true_z0 = z0_target_norm * z0_std + z0_mean

        sums["sq_eta"] += ((pred_eta - target_eta) ** 2).sum().item()
        sums["sq_phi"] += (wrapped_angle_delta(pred_phi, target_phi) ** 2).sum().item()
        sums["sq_pt_rel"] += (((pred_pt - true_pt) / true_pt) ** 2).sum().item()
        sums["sq_z0_mm"] += ((pred_z0 - true_z0) ** 2).sum().item()
        sums["charge_correct"] += ((charge_logit > 0).float() == charge_label).sum().item()

        n_events += B

    out = {k: v / n_events for k, v in sums.items()}
    out["rmse_eta"] = out.pop("sq_eta") ** 0.5
    out["rmse_phi_rad"] = out.pop("sq_phi") ** 0.5
    out["rmse_pt_rel"] = out.pop("sq_pt_rel") ** 0.5
    out["rmse_z0_mm"] = out.pop("sq_z0_mm") ** 0.5
    out["charge_acc"] = out.pop("charge_correct")
    out["total_loss_unweighted"] = (
        out["loss_eta"] + out["loss_phi"] + out["loss_logpt"]
        + out["loss_z0"] + out["loss_charge"]
    )
    out["n_events"] = n_events
    return out


def plot_importance(records: list[dict], metric: str, output_dir: Path, top_k: int | None = None):
    ranked = sorted(records, key=lambda r: r[f"delta_{metric}_mean"], reverse=True)
    if top_k is not None:
        ranked = ranked[:top_k]

    names = [r["feature"] for r in ranked][::-1]
    means = [r[f"delta_{metric}_mean"] for r in ranked][::-1]
    stds = [r[f"delta_{metric}_std"] for r in ranked][::-1]

    fig_h = max(4, 0.3 * len(names))
    plt.figure(figsize=(8, fig_h))
    plt.barh(names, means, xerr=stds, color="C0", ecolor="black", capsize=3)
    plt.axvline(0.0, color="grey", lw=1)
    plt.xlabel(f"increase in {metric} when feature is permuted")
    plt.title(f"Permutation importance ({metric})")
    plt.tight_layout()
    path = output_dir / f"permutation_importance_{metric}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path,
                     default=Path("checkpoints/ruche/ruche_Jun23_ConstChargeWeight.pt"))
    ap.add_argument("--parquet", type=Path,
                     default=Path("data/electrons/eta_phi_pt_z0_charge/zee_pu200_z0_charge.parquet"))
    ap.add_argument("--stats", type=Path,
                     default=Path("data/electrons/eta_phi_pt_z0_charge/target_stats.json"))
    ap.add_argument("--output-dir", type=Path, default=Path("results/permutation_importance"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--n-repeats", type=int, default=3,
                     help="independent random shuffles per feature, averaged for a mean+std importance")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--group", choices=["all", "x_sampled", "x_high_level"], default="all",
                     help="restrict which tensor's columns get permuted (useful for a quick pass)")
    ap.add_argument("--max-batches", type=int, default=None,
                     help="cap the number of test batches for a fast dev run")
    ap.add_argument("--top-k", type=int, default=None, help="only plot the top-K features by importance")
    ap.add_argument("--wandb", action="store_true", help="log results to wandb")
    args = ap.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    for p in (args.checkpoint, args.parquet, args.stats):
        if not p.exists():
            raise FileNotFoundError(f"Could not find {p}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint["config"]
    stats = json.loads(args.stats.read_text())

    loader = make_loader(
        parquet_path=args.parquet,
        split=args.split,
        target_stats_path=args.stats,
        batch_size=config["batch_size"],
        shuffle=False,
        use_angular_features=True,
        use_cluster_features=config.get("use_cluster_features", False),
        max_abs_eta=config.get("max_abs_eta"),
        min_abs_eta=config.get("min_abs_eta"),
    )
    if args.max_batches is not None:
        # keep it simple: wrap in a list once, up front, rather than re-slicing every call.
        loader = list(loader)[: args.max_batches]

    common = dict(
        max_cells=config["max_cells"], model_dim=config["model_dim"],
        n_heads=config["n_heads"], n_layers=config["n_layers"],
        dim_feedforward=config["dim_feedforward"], dropout=config["dropout"],
        output_dim=config["output_dim"], high_level_dim=config["high_level_dim"],
    )
    if config.get("model_type", "concat") == "conv":
        model = ConvCaloRegressor(**common, conv_dim=config["conv_dim"], kernel_size=config["kernel_size"])
    else:
        model = ConcatCaloRegressor(**common)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    target_stats = {
        "eta_mean": torch.tensor(stats["truth_eta"]["mean"], device=device),
        "eta_std": torch.tensor(stats["truth_eta"]["std"], device=device),
        "phi_mean": torch.tensor(stats["truth_phi"]["mean"], device=device),
        "phi_std": torch.tensor(stats["truth_phi"]["std"], device=device),
        "logpt_mean": torch.tensor(stats["truth_log_pt"]["mean"], device=device),
        "logpt_std": torch.tensor(stats["truth_log_pt"]["std"], device=device),
        "z0_mean": torch.tensor(stats["truth_z0"]["mean"], device=device),
        "z0_std": torch.tensor(stats["truth_z0"]["std"], device=device),
    }

    print("Computing baseline (unpermuted) losses over the full split...")
    baseline = compute_losses(model, loader, device, target_stats, permute=None)
    print(f"baseline: {json.dumps({k: round(v, 6) for k, v in baseline.items()}, indent=2)}")

    feature_specs: list[tuple[str, int, str]] = []  # (tensor_name, index, human name)
    if args.group in ("all", "x_sampled"):
        for i, name in enumerate(x_sampled_feature_names()):
            feature_specs.append(("x_sampled", i, name))
    if args.group in ("all", "x_high_level"):
        hl_names = x_high_level_feature_names(
            config["high_level_dim"], config.get("use_cluster_features", False)
        )
        for i, name in enumerate(hl_names):
            feature_specs.append(("x_high_level", i, name))

    metrics_to_track = [
        "loss_eta", "loss_phi", "loss_logpt", "loss_z0", "loss_charge",
        "total_loss_unweighted",
        "rmse_eta", "rmse_phi_rad", "rmse_pt_rel", "rmse_z0_mm", "charge_acc",
    ]

    records = []
    for tensor_name, feat_idx, feat_name in feature_specs:
        generator = torch.Generator().manual_seed(args.seed + hash((tensor_name, feat_idx)) % 100000)
        run_metrics = {m: [] for m in metrics_to_track}
        for rep in range(args.n_repeats):
            result = compute_losses(
                model, loader, device, target_stats,
                permute=(tensor_name, feat_idx), generator=generator,
            )
            for m in metrics_to_track:
                run_metrics[m].append(result[m] - baseline[m])

        record = {"feature": feat_name, "tensor": tensor_name, "index": feat_idx}
        for m in metrics_to_track:
            vals = np.array(run_metrics[m])
            record[f"delta_{m}_mean"] = float(vals.mean())
            record[f"delta_{m}_std"] = float(vals.std())
        records.append(record)

        print(
            f"  {tensor_name:12s}[{feat_idx:2d}] {feat_name:28s}  "
            f"d(total_loss_unweighted)={record['delta_total_loss_unweighted_mean']:+.5f} "
            f"+/- {record['delta_total_loss_unweighted_std']:.5f}"
        )

    results_path = args.output_dir / "permutation_importance.json"
    results_path.write_text(json.dumps({"baseline": baseline, "features": records}, indent=2))
    print(f"\nSaved raw results to: {results_path}")

    plot_paths = []
    for metric in ("total_loss_unweighted", "rmse_eta", "rmse_phi_rad", "rmse_pt_rel", "rmse_z0_mm"):
        plot_paths.append(plot_importance(records, metric, args.output_dir, top_k=args.top_k))
    print(f"Saved plots to: {args.output_dir}")

    print("\nTop features by increase in total_loss_unweighted:")
    for r in sorted(records, key=lambda r: r["delta_total_loss_unweighted_mean"], reverse=True)[:15]:
        print(f"  {r['feature']:28s} +{r['delta_total_loss_unweighted_mean']:.5f}")

    if args.wandb:
        import wandb
        with wandb.init(
            project="colliderml-electron-calo",
            name="permutation-importance",
            job_type="evaluation",
            config=dict(config),
        ) as run:
            run.log({"baseline_" + k: v for k, v in baseline.items()})
            table = wandb.Table(
                columns=["feature", "tensor", "index"] + [f"delta_{m}_mean" for m in metrics_to_track],
                data=[
                    [r["feature"], r["tensor"], r["index"]] + [r[f"delta_{m}_mean"] for m in metrics_to_track]
                    for r in records
                ],
            )
            run.log({"permutation_importance": table})
            for path in plot_paths:
                run.log({path.stem: wandb.Image(str(path))})
            run.save(str(results_path))


if __name__ == "__main__":
    main()