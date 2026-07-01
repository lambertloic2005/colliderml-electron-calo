"""Overlay: does resolution degrade WHERE the top-K cap drops electron energy?

Puts two independent measurements on one shared true-pT axis:
  * LEFT axis  -- core resolution vs pT for eta / phi / pT, and charge accuracy
                  vs pT (from the model's prediction export).
  * RIGHT axis -- f_sig(pT), the median fraction of the ELECTRON'S OWN calibrated
                  energy retained by the top-K-by-energy cap (from the parquet,
                  same quantity as panel (a) of diagnose_cell_cap_truth.py).

If a resolution curve worsens exactly where f_sig falls, the cap's signal loss is
plausibly costing real performance for that observable. If resolution stays flat
where f_sig drops, the discarded energy was information-poor for that observable
and raising the cap would not help it.

IMPORTANT read caveats:
  * Correlation on a shared axis is suggestive, NOT causal. Only a retrain at a
    larger cap proves causation. This plot tells you WHERE to spend that GPU.
  * CHARGE specifically: the magnetic bend angle ~ 1/pT, so the charge signal
    weakens with pT for a physics reason independent of the cap. A charge-vs-pT
    roll-off is EXPECTED and does not implicate the cap on its own. Use
    diagnose_charge_vs_cap.py (cell budget at FIXED pT) to isolate the cap.

Inputs:
  --preds        preds.npz from test_eta_phi_pt_z0_charge.py
                 (keys: truth_pt, truth_eta, truth_phi, pred_eta, pred_phi, pred_pt)
  --charge-npz   OPTIONAL charge export from export_charge_eval.py
                 (keys: truth_charge, truth_pt, charge_score = P(positron))
  --parquet      the truth-matched test parquet (for f_sig; needs cell_e_from_e_cal)

Run:
  python scripts/plot_resolution_vs_signalloss.py \
      --preds results/test/preds.npz \
      --charge-npz results/test/charge_eval.npz \
      --parquet data/electrons/testRuche/zee_pu200_supervised_dbscan_TEST.parquet \
      --max-cells 128 --out-dir results/resolution
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

try:
    from colliderml_electron.resolution import gaussian_resolution, wrap_angle
except Exception:                                    # standalone fallback
    def wrap_angle(x):
        x = np.asarray(x, float)
        return np.arctan2(np.sin(x), np.cos(x))

    class _F:
        def __init__(s, sigma, n_core): s.sigma = sigma; s.n_core = n_core

    def gaussian_resolution(r, n_sigma=3.0, max_iter=100, wrap=False):
        r = np.asarray(r, float); r = r[np.isfinite(r)]
        if wrap: r = wrap_angle(r)
        core = r
        for _ in range(max_iter):
            mu, sd = core.mean(), core.std()
            if sd == 0: break
            keep = np.abs(core - mu) <= n_sigma * sd
            if keep.all(): break
            core = core[keep]
        return _F(float(core.std()) if core.size else np.nan, int(core.size))


SIG_COL, TOT_COL = "cell_e_from_e_cal", "cell_e_calibrated"


def res_vs_pt(pt, resid, edges, wrap, min_count):
    """Core sigma per pT bin with a min-count guard; returns (centers, sigma, err)."""
    c, s, e = [], [], []
    for i in range(len(edges) - 1):
        hi = edges[i + 1]
        m = (pt >= edges[i]) & (pt <= hi if i == len(edges) - 2 else pt < hi)
        if int(m.sum()) < min_count:
            continue
        fit = gaussian_resolution(resid[m], wrap=wrap)
        if not np.isfinite(fit.sigma) or fit.sigma <= 0:
            continue
        nc = int(getattr(fit, "n_core", m.sum()))
        c.append(np.sqrt(edges[i] * hi)); s.append(fit.sigma)
        e.append(fit.sigma / np.sqrt(2 * max(nc, 1)))
    return np.asarray(c), np.asarray(s), np.asarray(e)


def acc_vs_pt(pt, correct, edges, min_count):
    """Binomial accuracy per pT bin with Wilson-ish error; returns (centers, acc, err)."""
    c, a, e = [], [], []
    for i in range(len(edges) - 1):
        hi = edges[i + 1]
        m = (pt >= edges[i]) & (pt <= hi if i == len(edges) - 2 else pt < hi)
        n = int(m.sum())
        if n < min_count:
            continue
        p = float(correct[m].mean())
        c.append(np.sqrt(edges[i] * hi)); a.append(p)
        e.append(np.sqrt(max(p * (1 - p), 1e-9) / n))
    return np.asarray(c), np.asarray(a), np.asarray(e)


def fsig_profile(parquet, split, max_abs_eta, K, edges, min_count):
    """Median fraction of the electron's own energy kept by top-K, per pT bin."""
    have = pl.read_parquet(parquet, n_rows=1).columns
    if SIG_COL not in have:
        return None
    df = pl.read_parquet(parquet, columns=["split", "truth_pt", "truth_eta",
                                           TOT_COL, SIG_COL]).filter(
        (pl.col("split") == split) & (pl.col("truth_eta").abs() <= max_abs_eta))
    pts, fs = [], []
    for row in df.iter_rows(named=True):
        e_tot = np.clip(np.asarray(row[TOT_COL], np.float64), 0, None)
        e_sig = np.clip(np.asarray(row[SIG_COL], np.float64), 0, None)
        tot = e_sig.sum()
        if tot <= 0:
            continue
        order = np.argsort(-e_tot, kind="mergesort")
        kept = np.cumsum(e_sig[order])[min(K, e_tot.size) - 1]
        pts.append(float(row["truth_pt"])); fs.append(float(kept / tot))
    pts = np.asarray(pts); fs = np.asarray(fs)
    c, med = [], []
    for i in range(len(edges) - 1):
        m = (pts >= edges[i]) & (pts < edges[i + 1])
        if m.sum() < min_count:
            continue
        c.append(np.sqrt(edges[i] * edges[i + 1])); med.append(np.median(fs[m]))
    return np.asarray(c), np.asarray(med)


def add_fsig(ax, fsig):
    """Overlay f_sig(pT) on a right twin axis (same on every panel)."""
    if fsig is None:
        return
    c, med = fsig
    axr = ax.twinx()
    axr.plot(c, med, "--", color="0.4", lw=1.3, marker="s", ms=3)
    axr.set_ylabel("electron energy kept by cap", color="0.4", fontsize=8)
    axr.tick_params(axis="y", labelcolor="0.4", labelsize=8)
    axr.set_ylim(0.5, 1.02)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--charge-npz", default=None)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-cells", type=int, default=128)
    ap.add_argument("--max-abs-eta", type=float, default=3.0)
    ap.add_argument("--n-bins", type=int, default=10)
    ap.add_argument("--min-count", type=int, default=50)
    ap.add_argument("--out-dir", default="results/resolution")
    args = ap.parse_args()

    d = np.load(args.preds)
    pt = d["truth_pt"].astype(np.float64)
    res_eta = d["pred_eta"] - d["truth_eta"]
    res_phi = wrap_angle(d["pred_phi"] - d["truth_phi"])
    res_ptrel = (d["pred_pt"] - d["truth_pt"]) / d["truth_pt"]

    edges = np.logspace(np.log10(max(pt.min(), 1e-3)), np.log10(pt.max()),
                        args.n_bins + 1)
    fsig = fsig_profile(args.parquet, args.split, args.max_abs_eta,
                        args.max_cells, edges, args.min_count)

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        f"Resolution vs pT with the top-{args.max_cells} cap's electron-energy "
        f"loss overlaid (grey dashed, right axis)", fontsize=12)

    specs = [
        (axes[0, 0], res_eta, False, r"core $\sigma_\eta$", r"$\eta$ resolution"),
        (axes[0, 1], res_phi, True, r"core $\sigma_\phi$ [rad]", r"$\phi$ resolution"),
        (axes[1, 0], res_ptrel, False, r"core $\sigma_{p_T}/p_T$", r"$p_T$ resolution"),
    ]
    for ax, resid, wrap, ylab, title in specs:
        c, s, e = res_vs_pt(pt, resid, edges, wrap, args.min_count)
        ax.errorbar(c, s, yerr=e, fmt="o-", color="C0", ms=4, capsize=2, lw=1.4)
        ax.set_xscale("log"); ax.set_xlabel("true $p_T$ [GeV]")
        ax.set_ylabel(ylab, color="C0"); ax.set_title(title, fontsize=10)
        ax.set_ylim(bottom=0)
        add_fsig(ax, fsig)

    # charge accuracy panel
    ax = axes[1, 1]
    if args.charge_npz:
        cd = np.load(args.charge_npz)
        cpt = cd["truth_pt"].astype(np.float64)
        pred_pos = cd["charge_score"] > 0.5          # P(positron) > 0.5 -> +1
        truth_pos = cd["truth_charge"] > 0
        correct = (pred_pos == truth_pos).astype(float)
        c, a, e = acc_vs_pt(cpt, correct, edges, args.min_count)
        ax.errorbar(c, a, yerr=e, fmt="o-", color="C3", ms=4, capsize=2, lw=1.4)
        ax.axhline(0.5, color="grey", ls=":", lw=1, label="chance")
        ax.set_ylim(0.45, 1.02); ax.legend(fontsize=8, loc="lower left")
        ax.set_ylabel("charge accuracy", color="C3")
    else:
        ax.text(0.5, 0.5, "no --charge-npz supplied\n(run export_charge_eval.py)",
                ha="center", va="center", transform=ax.transAxes, fontsize=9)
    ax.set_xscale("log"); ax.set_xlabel("true $p_T$ [GeV]")
    ax.set_title("charge accuracy  (NB: bend ~1/pT confounds cap effect)", fontsize=10)
    add_fsig(ax, fsig)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fp = out / f"resolution_vs_signalloss_K{args.max_cells}.png"
    fig.savefig(fp, dpi=200); fig.savefig(fp.with_suffix(".pdf"))
    print(f"saved {fp}")
    if fsig is not None:
        print("f_sig(pT) overlay: median electron-energy kept by cap, per pT bin:")
        for c, m in zip(*fsig):
            print(f"  pT~{c:7.1f} GeV : {m:.3f}")
    else:
        print(f"[warn] {SIG_COL} not in parquet; f_sig overlay skipped "
              "(point --parquet at the truth-matched build).")


if __name__ == "__main__":
    main()