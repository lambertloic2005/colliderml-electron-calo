"""
Check whether the learned homoscedastic uncertainty weights correspond to the
measured per-task resolution.

Theory: the training loss per regression task is
    total_i = exp(-2 s_i) * L_i + s_i,   s_i = log_sigma[i], L_i = mean Huber.
At the stationary point in s_i:  sigma_i^2 = exp(2 s_i*) = 2 * E[L_i].
So sigma is pinned to the mean Huber loss, and equals the residual RMS only
where residuals sit inside the Huber delta (quadratic regime). Working units:
    eta: raw | phi: rad | logpt: ln(pT) | z0: z-scored (mm / z0_std).
log_sigma is one scalar per task, so it averages over the whole population and
cannot represent the barrel/endcap split -- the per-region table shows what a
single sigma is averaging over.

Usage:
    env PREDS=<run>/preds.npz STATS_PATH=<...>/target_stats.json \
        python scripts/check_uncertainty_vs_resolution.py
Optional: LOG_SIGMA="s1,s2,s3,s4" or SIGMA="v1,v2,v3,v4" (order eta,phi,logpt,z0)
          ETA_BOUNDARY=1.5
"""

import json
import os
from pathlib import Path

import numpy as np

# Huber deltas: MUST mirror scripts/train_eta_phi_pt_z0_charge.py exactly.
DELTAS = {"eta": 0.10, "phi": 0.05, "logpt": 0.20, "z0": 1.00}
TASKS = ["eta", "phi", "logpt", "z0"]


def huber(e: np.ndarray, delta: float) -> np.ndarray:
    a = np.abs(e)
    return np.where(a <= delta, 0.5 * e * e, delta * (a - 0.5 * delta))


def wrap(dphi: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(dphi), np.cos(dphi))


def robust_sigma(e: np.ndarray) -> float:
    q16, q84 = np.percentile(e, [16.0, 84.0])
    return 0.5 * float(q84 - q16)


def summarize(residuals, label, learned):
    print(f"\n=== {label} (N = {len(next(iter(residuals.values())))}) ===")
    header = (f"{'task':<6} {'sigma*=sqrt(2<Huber>)':>22} {'RMS':>12} "
              f"{'robust(16-84)':>14} {'frac |e|<delta':>15}")
    if learned is not None:
        header += f" {'learned sigma':>14} {'learned/sigma*':>15}"
    print(header)
    for t in TASKS:
        e = residuals[t]
        d = DELTAS[t]
        sig_star = float(np.sqrt(2.0 * huber(e, d).mean()))
        rms = float(np.sqrt(np.mean(e * e)))
        rob = robust_sigma(e)
        frac = float((np.abs(e) < d).mean())
        line = f"{t:<6} {sig_star:>22.5f} {rms:>12.5f} {rob:>14.5f} {frac:>15.3f}"
        if learned is not None:
            ls = learned[t]
            line += f" {ls:>14.5f} {ls / sig_star:>15.3f}"
        print(line)


def main():
    preds_path = Path(os.environ["PREDS"])
    stats_path = Path(os.environ.get(
        "STATS_PATH", "data/electrons/eta_phi_pt_z0_charge/target_stats.json"))
    eta_boundary = float(os.environ.get("ETA_BOUNDARY", "1.5"))

    d = np.load(preds_path)
    stats = json.loads(stats_path.read_text())
    z0_std = float(stats["truth_z0"]["std"])

    # Residuals in the SAME units the loss terms are computed in. Note that
    # (pred_deta - deta_target) == (pred_eta - truth_eta): anchors cancel.
    residuals = {
        "eta": d["pred_eta"] - d["truth_eta"],
        "phi": wrap(d["pred_phi"] - d["truth_phi"]),
        "logpt": np.log(d["pred_pt"]) - np.log(d["truth_pt"]),
        "z0": (d["pred_z0"] - d["truth_z0"]) / z0_std,
    }

    learned = None
    if os.environ.get("LOG_SIGMA"):
        vals = [float(v) for v in os.environ["LOG_SIGMA"].split(",")]
        learned = {t: float(np.exp(v)) for t, v in zip(TASKS, vals)}
    elif os.environ.get("SIGMA"):
        vals = [float(v) for v in os.environ["SIGMA"].split(",")]
        learned = dict(zip(TASKS, vals))

    print(f"preds: {preds_path}")
    print(f"stats: {stats_path}  (z0_std = {z0_std:.3f} mm; "
          f"z0 residuals below are in z-scored units)")
    print(f"Huber deltas: {DELTAS}")

    summarize(residuals, "full population", learned)

    abs_eta = np.abs(d["truth_eta"])
    barrel = abs_eta < eta_boundary
    endcap = (abs_eta >= eta_boundary) & (abs_eta < 3.0)
    summarize({t: e[barrel] for t, e in residuals.items()},
              f"barrel |eta| < {eta_boundary}", learned)
    summarize({t: e[endcap] for t, e in residuals.items()},
              f"endcap {eta_boundary} <= |eta| < 3.0", learned)

    print(
        "\nReading the table:\n"
        "  * If training converged, learned sigma sits on sigma* for the\n"
        "    TRAINING population; test-set sigma* is the iid estimate of that.\n"
        "  * sigma* ~= RMS only where 'frac |e|<delta' is close to 1. A gap\n"
        "    between sigma* and RMS is the Huber tail, not a training failure.\n"
        "  * A single homoscedastic sigma cannot represent the barrel/endcap\n"
        "    difference; the per-region rows show what it averages over.\n"
    )


if __name__ == "__main__":
    main()