"""STEP 0 (run before any retrain): does the azimuthal trajectory curvature
dphi/dr actually separate charge -- and does the top-K cap keep it?

Motivation: the depth-integrated phi-SKEW was shown to carry ~no charge info
(charge_vs_cap: all-cells accuracy ~chance), yet the trained model reaches
~0.7-0.78 charge accuracy. The leading hypothesis is that the charge lives in
a Delta-phi-versus-radius CURVATURE: the solenoid bends e+/e- oppositely, so the
azimuthal position drifts with shower depth/radius, and the SIGN of dphi/dr
tracks the charge. That is a first-moment-per-slice slope, orthogonal to the
depth-integrated third moment (skew).

This script computes, per electron, the energy-weighted least-squares slope of
dphi vs r -- the exact azimuthal analogue of the existing z0 pointing slope in
dataset.py (dz/dr) -- and tests sign(phi_slope) as a model-free charge
classifier. It reports:
  (a) accuracy vs cell budget: all-cells vs top-K-by-energy, per pT band
      -> if all-cells >> chance, the CURVATURE hypothesis is confirmed.
      -> if top-K ~ all-cells, the cap does NOT starve this signal (so a cap
         tweak would not help charge; the signal is already in kept cells).
  (b) mean per-radial-slice <dphi> split by truth charge -> the curves should
      fan apart with opposite slope for e- vs e+ if the mechanism is real.
  (c) phi_slope distribution by charge (separability at a glance).

Model-free: sign(phi_slope) is a floor on the information, not the trained head.
If even this separates charge, the feature is worth adding to the model.

Run:
  python scripts/diagnose_charge_curvature.py \
      --parquet .../zee_pu200_supervised_dbscan_TEST.parquet \
      --split test --max-cells 128 --slices 6 --out-dir results/diagnostics
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

TOT_COL = "cell_e_calibrated"


def phi_slope_and_profile(cell_x, cell_y, cell_phi, e, K, idx=None):
    """E-weighted LS slope of dphi vs r (rad/mm) and per-slice <dphi>, mirroring
    the z0 pointing fit in dataset.py. `idx` optionally restricts to selected cells
    (the slope/profile are then computed on that subset, e.g. the top-K by energy)."""
    w = np.clip(e.astype(np.float64), 1e-9, None)
    # centroid from ALL cells (as dataset.py does) -- the frame is defined globally
    cphi = np.arctan2(np.sum(w * np.sin(cell_phi)), np.sum(w * np.cos(cell_phi)))
    dphi = np.arctan2(np.sin(cell_phi - cphi), np.cos(cell_phi - cphi))
    r = np.hypot(cell_x, cell_y)
    if idx is not None:
        w, dphi, r = w[idx], dphi[idx], r[idx]
    wsum = w.sum()
    if wsum <= 0 or r.size < 3:
        return np.nan, np.full(K, np.nan)
    r_bar = np.sum(w * r) / wsum
    var_r = np.sum(w * (r - r_bar) ** 2) / wsum
    dphi_bar = np.sum(w * dphi) / wsum
    cov = np.sum(w * (r - r_bar) * (dphi - dphi_bar)) / wsum
    slope = cov / var_r if var_r > 1e-9 else np.nan
    # per-slice <dphi>
    prof = np.full(K, np.nan)
    lo, hi = r.min(), r.max()
    edges = np.linspace(lo, hi + 1e-6, K + 1)
    for k in range(K):
        m = (r >= edges[k]) & (r < edges[k + 1])
        if m.any():
            prof[k] = np.sum(w[m] * dphi[m]) / w[m].sum()
    return float(slope), prof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-cells", type=int, default=128)
    ap.add_argument("--max-abs-eta", type=float, default=3.0)
    ap.add_argument("--slices", type=int, default=6)
    ap.add_argument("--max-events", type=int, default=None)
    ap.add_argument("--out-dir", default="results/diagnostics")
    args = ap.parse_args()

    K = args.slices
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    need = ["truth_charge", "cell_phi", "cell_x", "cell_y"]
    have = pl.read_parquet(args.parquet, n_rows=1).columns
    miss = [c for c in need if c not in have]
    if miss:
        raise SystemExit(f"[fatal] parquet missing {miss}.")

    cols = ["split", "truth_pt", "truth_eta", "truth_charge",
            "cell_phi", "cell_x", "cell_y", TOT_COL]
    df = pl.read_parquet(args.parquet, columns=cols).filter(
        (pl.col("split") == args.split)
        & (pl.col("truth_eta").abs() <= args.max_abs_eta))
    if args.max_events and df.height > args.max_events:
        df = df.sample(n=args.max_events, seed=0)

    pt, charge = [], []
    slope_all, slope_cap = [], []
    prof_all = []
    for row in df.iter_rows(named=True):
        e = np.clip(np.asarray(row[TOT_COL], np.float64), 0, None)
        px = np.asarray(row["cell_x"], np.float64)
        py = np.asarray(row["cell_y"], np.float64)
        ph = np.asarray(row["cell_phi"], np.float64)
        if e.size < 3 or e.sum() <= 0:
            continue
        s_all, p_all = phi_slope_and_profile(px, py, ph, e, K)
        top = np.argsort(-e, kind="mergesort")[:min(args.max_cells, e.size)]
        s_cap, _ = phi_slope_and_profile(px, py, ph, e, K, idx=top)
        pt.append(float(row["truth_pt"])); charge.append(int(row["truth_charge"]))
        slope_all.append(s_all); slope_cap.append(s_cap); prof_all.append(p_all)

    pt = np.asarray(pt); charge = np.asarray(charge)
    slope_all = np.asarray(slope_all); slope_cap = np.asarray(slope_cap)
    prof_all = np.asarray(prof_all)
    n = pt.size

    g = np.isfinite(slope_all)
    corr = np.sign(np.mean(np.sign(slope_all[g]) * charge[g])) or 1.0
    print(f"phi_slope->charge sign: corr={corr:+.0f}  "
          f"(all-cells acc {np.mean((corr*np.sign(slope_all[g]))==charge[g]):.3f})")

    def acc(slope, sel):
        s = np.asarray(slope)[sel]; c = charge[sel]; m = np.isfinite(s)
        return float(np.mean((corr * np.sign(s[m])) == c[m])) if m.sum() else np.nan

    hi = pt >= np.percentile(pt, 90); lo = pt <= np.percentile(pt, 50)
    mid = (~hi) & (~lo)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"STEP 0: does d$\\phi$/dr curvature separate charge?  "
                 f"model-free, ColliderML Z$\\to$ee pu200, {args.split} (n={n})",
                 fontsize=12)

    # (a) accuracy: all-cells vs top-K cap, per pT band
    ax = axes[0]
    bands = [("low pT", lo), ("mid pT", mid), ("high pT", hi)]
    x = np.arange(len(bands))
    ax.bar(x - 0.2, [acc(slope_all, s) for _, s in bands], 0.4, label="all cells")
    ax.bar(x + 0.2, [acc(slope_cap, s) for _, s in bands], 0.4,
           label=f"top-{args.max_cells} cap")
    ax.axhline(0.5, color="grey", ls="--", lw=1, label="chance")
    ax.set_xticks(x); ax.set_xticklabels([b for b, _ in bands])
    ax.set_ylabel("charge accuracy (sign of d$\\phi$/dr)")
    ax.set_ylim(0.4, 1.0); ax.legend(fontsize=8)
    ax.set_title("(a) Curvature accuracy: all vs capped", fontsize=10)

    # (b) mean per-slice <dphi> by charge
    ax = axes[1]
    sl = np.arange(1, K + 1)
    for q, col, lab in [(-1, "C0", "electron (q=-1)"), (+1, "C3", "positron (q=+1)")]:
        sub = prof_all[charge == q]
        mean = np.nanmean(sub, axis=0)
        ax.plot(sl, mean, "o-", color=col, label=lab)
    ax.axhline(0, color="grey", ls=":", lw=1)
    ax.set_xlabel("radial slice (inner -> outer)")
    ax.set_ylabel("mean $\\langle\\Delta\\phi\\rangle$ [rad]")
    ax.set_title("(b) Azimuthal drift with depth, by charge", fontsize=10)
    ax.legend(fontsize=8)

    # (c) slope distribution by charge
    ax = axes[2]
    s = slope_all[g] * 1000.0; c = charge[g]     # rad/m for readability
    rng = np.nanpercentile(s, [1, 99])
    bins = np.linspace(rng[0], rng[1], 60)
    ax.hist(s[c < 0], bins=bins, alpha=0.6, color="C0", label="electron (q=-1)")
    ax.hist(s[c > 0], bins=bins, alpha=0.6, color="C3", label="positron (q=+1)")
    ax.axvline(0, color="grey", ls="--", lw=1)
    ax.set_xlabel("d$\\phi$/dr [rad/m]"); ax.set_ylabel("electrons / bin")
    ax.set_title("(c) Is d$\\phi$/dr charge-separating?", fontsize=10)
    ax.legend(fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fp = out / "charge_curvature_step0.png"
    fig.savefig(fp, dpi=200); fig.savefig(fp.with_suffix(".pdf"))

    print(f"\n=== d(phi)/dr curvature charge test, {n} electrons ===")
    for lab, sel in bands:
        print(f"  {lab:<8} all-cells {acc(slope_all,sel):.3f} | "
              f"top-{args.max_cells} {acc(slope_cap,sel):.3f}")
    print("\nread: all-cells >> 0.5  => curvature carries charge (hypothesis confirmed).")
    print("      top-K ~ all-cells   => cap keeps the signal (no cap tweak needed).")
    print("      top-K << all-cells  => cap starves it (then a selection change helps).")
    print(f"saved {fp}")


if __name__ == "__main__":
    main()