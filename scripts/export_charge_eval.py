"""Export per-electron charge predictions from a trained checkpoint to an .npz
that scripts/plot_charge_results.py can plot.

This mirrors the loading in scripts/test_eta_phi_pt_z0_charge.py (checkpoint ->
config -> make_loader -> build model -> load_state_dict), runs a forward pass
over the test split, and saves:
    truth_charge : (N,) in {-1,+1}
    truth_pt     : (N,) GeV          (decoded from the truth_log_pt target)
    charge_score : (N,) P(positron)  (sigmoid of the charge logit)

IMPORTANT — the one model-specific line:
  This branch's committed model has NO charge head (output is
  [eta, phi_cos, phi_sin, log_pt]).  Your latest checkpoint adds a charge logit.
  You must tell this script where that logit is:
    --charge-index K   column K of the model output is the charge logit
                       (default -1 = last column)
  If your model.forward returns a tuple/dict instead of a single tensor, edit
  the marked block below.

Sanity check: the script prints the model-output shape on the first batch and a
final AUC.  If AUC ~ your known 0.816, the index and sign are right.  If AUC ~
0.184, the sign is flipped -> add --flip.  If AUC ~ 0.5, wrong column.

Run:
    python scripts/export_charge_eval.py \
        --checkpoint checkpoints/ruche/<your_latest>.pt \
        --parquet   data/electrons/eta_phi_pt_z0_charge/zee_pu200_z0_charge.parquet \
        --stats     data/electrons/eta_phi_pt_z0_charge/target_stats.json \
        --out charge_eval.npz --charge-index -1

then:
    python scripts/plot_charge_results.py charge_eval.npz --out-dir results/charge
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from colliderml_electron.dataset import make_loader, TARGET_COLS
from colliderml_electron.model import ConcatCaloRegressor, ConvCaloRegressor

LOGPT_INDEX = TARGET_COLS.index("truth_log_pt")


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def auc_quick(scores, labels):
    """Rank-based AUC (Mann-Whitney). labels in {0,1}, 1 = positron."""
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    P = labels.sum()
    N = len(labels) - P
    if P == 0 or N == 0:
        return float("nan")
    return float((ranks[labels == 1].sum() - P * (P + 1) / 2) / (P * N))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--out", default="charge_eval.npz")
    ap.add_argument("--charge-index", type=int, default=-1,
                    help="column of the model output holding the charge logit")
    ap.add_argument("--flip", action="store_true",
                    help="flip the logit sign (use if AUC comes out < 0.5)")
    args = ap.parse_args()

    device = get_device()
    print(f"device: {device}")

    for p in (args.checkpoint, args.parquet, args.stats):
        if not Path(p).exists():
            raise FileNotFoundError(p)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint["config"]
    stats = json.loads(Path(args.stats).read_text())
    logpt_mean = stats["truth_log_pt"]["mean"]
    logpt_std = stats["truth_log_pt"]["std"]

    test_loader = make_loader(
        parquet_path=args.parquet,
        split="test",
        target_stats_path=args.stats,
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
    model.to(device).eval()

    charge_logits, true_pts, truth_charges = [], [], []
    printed = False
    with torch.no_grad():
        for batch in test_loader:
            xs = batch["x_sampled"].to(device)
            xh = batch["x_high_level"].to(device)
            mask = batch["mask"].to(device)

            out = model(xs, xh, mask)

            # ---- model-specific charge extraction (EDIT HERE if needed) ----
            if isinstance(out, (tuple, list)):
                # e.g. model returns (regression, charge_logit)
                charge_logit = out[1].squeeze(-1)
                pred = out[0]
            elif isinstance(out, dict):
                charge_logit = out["charge"].squeeze(-1)
                pred = out.get("regression", None)
            else:
                pred = out
                charge_logit = out[:, args.charge_index]
            # ----------------------------------------------------------------

            if not printed:
                shp = tuple(out.shape) if torch.is_tensor(out) else type(out)
                print(f"first-batch model output: {shp}; "
                      f"charge logit from index {args.charge_index}")
                printed = True

            target = batch["target"]
            true_logpt = target[:, LOGPT_INDEX].cpu().numpy() * logpt_std + logpt_mean
            true_pts.append(np.exp(true_logpt))
            truth_charges.append(batch["truth_charge"].cpu().numpy())
            charge_logits.append(charge_logit.cpu().numpy())

    charge_logit = np.concatenate(charge_logits).astype(np.float64)
    if args.flip:
        charge_logit = -charge_logit
    truth_charge = np.concatenate(truth_charges).astype(np.float64)
    true_pt = np.concatenate(true_pts).astype(np.float64)

    p_pos = 1.0 / (1.0 + np.exp(-charge_logit))   # P(positron)
    labels = (truth_charge > 0).astype(int)
    auc = auc_quick(p_pos, labels)
    print(f"N={len(p_pos)}  positrons={int(labels.sum())}  AUC={auc:.4f}")
    if auc < 0.5:
        print("  AUC < 0.5 -> sign convention is flipped; re-run with --flip")

    np.savez(args.out, truth_charge=truth_charge, truth_pt=true_pt,
             charge_score=p_pos)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()