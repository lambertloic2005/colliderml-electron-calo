"""Does the top-`max_cells` cap discard the ELECTRON's energy, or just PILEUP?

This is the definitive follow-up to diagnose_cell_cap.py. That script normalised
retained energy to the cluster total (Sigma of cell_e_calibrated), which mixes the
electron's own shower with any PU200 pileup that lands in the same cells -- so it
could not tell "the cap starves real signal" from "the cap usefully drops pileup".

The parquet already stores the truth split per cell (built in pipeline.py):
    cell_e_calibrated   = calibrated TOTAL cell energy (electron + in-cell pileup)
    cell_e_from_e_cal   = calibrated energy from the ELECTRON family only (MC truth)
The model ranks/caps on cell_e_calibrated (the TOTAL), so a pileup-hot cell can
outrank a genuine soft electron-tail cell. We reproduce that exact ranking and
then ask what happens to the electron's OWN energy.

Fork resolver (panel a):  f_sig = Sigma e_from_e(top-K) / Sigma e_from_e(all)
    f_sig -> 1 at high pT  => the cap keeps ~all the electron's real energy;
                              the "lost" cluster energy in the earlier plot was
                              pileup. The cap is beneficial denoising. Do NOT
                              raise it; if anything the win is upstream cluster
                              purity.
    f_sig -> 0.6 at high pT => the cap throws away real electron energy. Genuine
                              signal loss; a larger cap (or per-cell handling
                              that is not pure energy-ranking) is warranted.

Also reported:
  * cluster purity  Sigma e_from_e / Sigma e_total  (all cells)         -> contamination level
  * kept-cell purity Sigma e_from_e / Sigma e_total (top-K only)        -> does energy-ranking
                                                                           clean up or not?
  * truth-normalised transverse energy (follow-up #1):
        R_sig = Sigma E_T^{from_e}(all) / truth_pt   (containment/calibration check, ~1 expected)
        R_tot = Sigma E_T^{total}(all)  / truth_pt   (>1 => the model's pre-cap
                                                       Sigma E_T feature is pileup-inflated)

Model-free apart from reproducing the cap. Run:
    python scripts/diagnose_cell_cap_truth.py \
        --parquet data/electrons/testRuche/zee_pu200_supervised_dbscan_TEST.parquet \
        --split test --max-cells 128 --out-dir results/diagnostics
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

SIG_COL = "cell_e_from_e_cal"   # calibrated electron-only per-cell energy (truth)
TOT_COL = "cell_e_calibrated"   # calibrated total per-cell energy (what the model ranks on)


def frac_topk(cumsum_in_energy_order: np.ndarray, k: int) -> float:
    """Fraction of a per-cell quantity retained by the top-k cells (energy order)."""
    total = cumsum_in_energy_order[-1]
    if total <= 0:
        return float("nan")
    return float(cumsum_in_energy_order[min(k, cumsum_in_energy_order.size) - 1] / total)


def profile(x, y, edges, min_count=50):
    """Median and 16-84% band of y in bins of x (NaNs dropped per bin)."""
    c, med, lo, hi, n = [], [], [], [], []
    for i in range(len(edges) - 1):
        m = (x >= edges[i]) & (x < edges[i + 1])
        yi = y[m]
        yi = yi[np.isfinite(yi)]
        if yi.size < min_count:
            continue
        c.append(np.sqrt(edges[i] * edges[i + 1]) if edges[i] > 0
                 else 0.5 * (edges[i] + edges[i + 1]))
        med.append(np.median(yi)); lo.append(np.percentile(yi, 16))
        hi.append(np.percentile(yi, 84)); n.append(int(yi.size))
    return (np.asarray(c), np.asarray(med), np.asarray(lo),
            np.asarray(hi), np.asarray(n))


def band(ax, x, edges, color, label, min_count=50):
    c, med, lo, hi, _ = profile(x[0], x[1], edges, min_count)
    ax.plot(c, med, "o-", color=color, label=label)
    ax.fill_between(c, lo, hi, alpha=0.22, color=color)
    return c, med


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-cells", type=int, default=128)
    ap.add_argument("--max-abs-eta", type=float, default=3.0)
    ap.add_argument("--rank-max", type=int, default=512,
                    help="max cell rank to accumulate for the cumulative-recovery panel")
    ap.add_argument("--max-events", type=int, default=None)
    ap.add_argument("--out-dir", default="results/diagnostics")
    args = ap.parse_args()

    K = args.max_cells
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    have = pl.read_parquet(args.parquet, n_rows=1).columns
    if SIG_COL not in have:
        raise SystemExit(
            f"[fatal] column '{SIG_COL}' not in {args.parquet}.\n"
            f"        This parquet was built truth-free (cluster_pipeline.py) and cannot\n"
            f"        separate signal from pileup. Point --parquet at the truth-matched\n"
            f"        build (pipeline.py / build_electron_dataset.py), which writes\n"
            f"        {SIG_COL}.")

    cols = ["split", "truth_pt", "truth_eta",
            "cell_x", "cell_y", "cell_z", TOT_COL, SIG_COL]
    df = pl.read_parquet(args.parquet, columns=cols).filter(
        (pl.col("split") == args.split)
        & (pl.col("truth_eta").abs() <= args.max_abs_eta))
    if args.max_events and df.height > args.max_events:
        df = df.sample(n=args.max_events, seed=0)

    pt, abseta = [], []
    f_sig_E, f_sig_ET = [], []          # electron energy kept by top-K
    pur_all, pur_top = [], []           # cluster purity vs kept-cell purity
    R_sig, R_tot = [], []               # truth-normalised E_T (containment / pileup)
    # cumulative electron-energy fraction vs cell rank, aligned by rank
    cum_lo = np.zeros(args.rank_max); n_lo = np.zeros(args.rank_max)
    cum_hi = np.zeros(args.rank_max); n_hi = np.zeros(args.rank_max)

    pts_tmp = df["truth_pt"].to_numpy()
    hi_thr = float(np.percentile(pts_tmp, 90))
    lo_thr = float(np.percentile(pts_tmp, 50))

    for row in df.iter_rows(named=True):
        e_tot = np.clip(np.asarray(row[TOT_COL], np.float64), 0, None)
        e_sig = np.clip(np.asarray(row[SIG_COL], np.float64), 0, None)
        x = np.asarray(row["cell_x"], np.float64)
        y = np.asarray(row["cell_y"], np.float64)
        z = np.asarray(row["cell_z"], np.float64)
        r3 = np.sqrt(x * x + y * y + z * z)
        sin_t = np.hypot(x, y) / np.clip(r3, 1e-9, None)

        order = np.argsort(-e_tot, kind="mergesort")   # EXACTLY the model's ranking
        e_tot_o, e_sig_o, sin_o = e_tot[order], e_sig[order], sin_t[order]
        cs_tot = np.cumsum(e_tot_o)
        cs_sig = np.cumsum(e_sig_o)
        cs_et_tot = np.cumsum(e_tot_o * sin_o)
        cs_et_sig = np.cumsum(e_sig_o * sin_o)

        sig_tot = cs_sig[-1]; tot_tot = cs_tot[-1]
        ptv = float(row["truth_pt"])
        pt.append(ptv); abseta.append(abs(float(row["truth_eta"])))

        f_sig_E.append(frac_topk(cs_sig, K))
        f_sig_ET.append(frac_topk(cs_et_sig, K))
        pur_all.append(sig_tot / tot_tot if tot_tot > 0 else np.nan)
        ktop = min(K, e_tot.size)
        pur_top.append(cs_sig[ktop - 1] / cs_tot[ktop - 1] if cs_tot[ktop - 1] > 0 else np.nan)
        R_sig.append(cs_et_sig[-1] / max(ptv, 1e-9))
        R_tot.append(cs_et_tot[-1] / max(ptv, 1e-9))

        # aligned cumulative electron-energy recovery vs rank (per pT band)
        if sig_tot > 0:
            frac_curve = cs_sig / sig_tot
            r = min(frac_curve.size, args.rank_max)
            if ptv >= hi_thr:
                cum_hi[:r] += frac_curve[:r]; n_hi[:r] += 1
            elif ptv <= lo_thr:
                cum_lo[:r] += frac_curve[:r]; n_lo[:r] += 1

    pt = np.asarray(pt); abseta = np.asarray(abseta)
    f_sig_E = np.asarray(f_sig_E); f_sig_ET = np.asarray(f_sig_ET)
    pur_all = np.asarray(pur_all); pur_top = np.asarray(pur_top)
    R_sig = np.asarray(R_sig); R_tot = np.asarray(R_tot)
    n_tot = pt.size

    pt_edges = np.logspace(np.log10(max(pt.min(), 1e-3)), np.log10(pt.max()), 13)
    eta_edges = np.linspace(0, args.max_abs_eta, 13)

    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))
    fig.suptitle(
        f"Does the top-{K} cap drop the ELECTRON or PILEUP?  ColliderML Z$\\to$ee "
        f"pu200, {args.split}, $|\\eta|\\leq{args.max_abs_eta:g}$  (n={n_tot})",
        fontsize=12)

    # (a) THE fork resolver: electron's own energy kept by the cap ----------
    ax = axes[0, 0]
    band(ax, (pt, f_sig_E), pt_edges, "C0", "median")
    ax.axhline(1.0, color="grey", ls=":", lw=1)
    ax.set_xscale("log"); ax.set_ylim(top=1.02)
    ax.set_xlabel("true $p_T$ [GeV]")
    ax.set_ylabel(f"$\\Sigma E^{{e}}$(top {K}) / $\\Sigma E^{{e}}$(all)")
    ax.set_title("(a) Electron's OWN energy kept by the cap", fontsize=10)
    ax.legend(fontsize=8)

    # (b) cluster purity vs pT ---------------------------------------------
    ax = axes[0, 1]
    band(ax, (pt, pur_all), pt_edges, "C3", "cluster (all cells)")
    band(ax, (pt, pur_top), pt_edges, "C2", f"kept cells (top {K})")
    ax.set_xscale("log"); ax.set_ylim(0, 1.02)
    ax.set_xlabel("true $p_T$ [GeV]")
    ax.set_ylabel("purity  $\\Sigma E^{e}/\\Sigma E^{tot}$")
    ax.set_title("(b) Purity: whole cluster vs cells the cap keeps", fontsize=10)
    ax.legend(fontsize=8)

    # (c) truth-normalised E_T: containment (signal) vs pileup inflation ----
    ax = axes[0, 2]
    band(ax, (pt, R_sig), pt_edges, "C0", "$\\Sigma E_T^{e}$ / $p_T^{truth}$")
    band(ax, (pt, R_tot), pt_edges, "C1", "$\\Sigma E_T^{tot}$ / $p_T^{truth}$")
    ax.axhline(1.0, color="grey", ls=":", lw=1, label="perfect containment")
    ax.set_xscale("log")
    ax.set_xlabel("true $p_T$ [GeV]")
    ax.set_ylabel("$\\Sigma E_T$ / $p_T^{truth}$")
    ax.set_title("(c) Truth-normalised $E_T$ (signal vs total)", fontsize=10)
    ax.legend(fontsize=8)

    # (d) fork resolver vs |eta| -------------------------------------------
    ax = axes[1, 0]
    band(ax, (abseta, f_sig_E), eta_edges, "C4", "median")
    ax.axvline(1.5, color="grey", ls="--", lw=1, label="barrel/endcap (1.5)")
    ax.axhline(1.0, color="grey", ls=":", lw=1)
    ax.set_xlabel("$|\\eta^{truth}|$")
    ax.set_ylabel(f"$\\Sigma E^{{e}}$(top {K}) / $\\Sigma E^{{e}}$(all)")
    ax.set_title("(d) Electron energy kept by the cap vs |eta|", fontsize=10)
    ax.legend(fontsize=8)

    # (e) cumulative electron-energy recovery vs cell rank ------------------
    ax = axes[1, 1]
    ranks = np.arange(1, args.rank_max + 1)
    with np.errstate(invalid="ignore"):
        mean_hi = np.where(n_hi > 0, cum_hi / np.maximum(n_hi, 1), np.nan)
        mean_lo = np.where(n_lo > 0, cum_lo / np.maximum(n_lo, 1), np.nan)
    ax.plot(ranks, mean_lo, "-", color="C1", label="low pT (bottom 50%)")
    ax.plot(ranks, mean_hi, "-", color="C0", label="high pT (top 10%)")
    ax.axvline(K, color="r", ls=":", lw=1.5, label=f"cap ({K})")
    ax.axhline(1.0, color="grey", ls=":", lw=1)
    ax.set_xlabel("cell rank (by total energy)")
    ax.set_ylabel("cumulative electron-energy fraction")
    ax.set_title("(e) How fast the electron's energy accumulates", fontsize=10)
    ax.legend(fontsize=8)

    # (f) distribution of f_sig at high pT ---------------------------------
    ax = axes[1, 2]
    hi = pt >= hi_thr; lo = pt <= lo_thr
    ax.hist(f_sig_E[lo][np.isfinite(f_sig_E[lo])], bins=np.linspace(0, 1.01, 41),
            alpha=0.6, color="C1", label=f"low pT (n={int(lo.sum())})")
    ax.hist(f_sig_E[hi][np.isfinite(f_sig_E[hi])], bins=np.linspace(0, 1.01, 41),
            alpha=0.7, color="C0", label=f"high pT (n={int(hi.sum())})")
    ax.set_yscale("log")
    ax.set_xlabel(f"$\\Sigma E^{{e}}$(top {K}) / $\\Sigma E^{{e}}$(all)")
    ax.set_ylabel("electrons / bin")
    ax.set_title("(f) Per-electron signal recovery", fontsize=10)
    ax.legend(fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig_path = out / f"cell_cap_truth_K{K}.png"
    fig.savefig(fig_path, dpi=200); fig.savefig(fig_path.with_suffix(".pdf"))

    # ---- printed summary --------------------------------------------------
    def stats(sel, name):
        fe = f_sig_E[sel][np.isfinite(f_sig_E[sel])]
        pa = pur_all[sel][np.isfinite(pur_all[sel])]
        pt_ = pur_top[sel][np.isfinite(pur_top[sel])]
        rt = R_tot[sel][np.isfinite(R_tot[sel])]
        print(f"  {name:<20} n={int(sel.sum()):>6}  "
              f"e-energy kept(top{K}) med {np.median(fe):6.2%} "
              f"(1st-pct {np.percentile(fe,1):6.2%}) | "
              f"purity all {np.median(pa):5.2%} top {np.median(pt_):5.2%} | "
              f"SumET_tot/pT med {np.median(rt):4.2f}")

    print(f"\n=== truth-resolved top-{K} cap, {n_tot} electrons "
          f"(|eta|<={args.max_abs_eta:g}, {args.split}) ===")
    print("f_sig = fraction of the ELECTRON'S OWN calibrated energy kept by the cap")
    stats(np.ones(n_tot, bool), "all")
    stats(pt >= hi_thr, f"high pT (>={hi_thr:.0f})")
    stats(pt <= lo_thr, "low pT (bottom 50%)")
    stats(abseta > 1.5, "endcap |eta|>1.5")
    stats(abseta <= 1.5, "barrel |eta|<=1.5")
    print("\nread: f_sig -> ~1  => cap drops mostly PILEUP (beneficial); "
          "f_sig well below 1 => cap drops real electron energy (signal loss).")
    print(f"saved {fig_path} and {fig_path.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()