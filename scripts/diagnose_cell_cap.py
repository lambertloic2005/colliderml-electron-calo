"""Does the top-`max_cells`-by-energy cap discard shower energy for HIGH-pT electrons?

Model-free -- reads the test parquet directly and reproduces EXACTLY the cell
selection the model performs in `ConcatCaloRegressor._select_top_cells`:
keep the `MAX_CELLS` cells with the largest calibrated energy (the model ranks
on x_high_level[..., 0] = log(cell_e_calibrated), which is a monotonic function
of the energy, so top-K-by-log-E == top-K-by-E). Everything past the cap is
dropped before the transformer ever sees it.

The worry: a higher-energy electron showers into MORE cells, so a fixed cap may
retain a SMALLER fraction of its shower -- throwing information away precisely
for the electrons where we would hope to reconstruct pT / direction best.

Two caveats this script is designed to expose, not hide:
  * The model still receives Sigma E and Sigma E_T as *cluster-level scalars*
    computed over ALL cells BEFORE the cap (see dataset.py). So the total-energy
    information survives truncation even when per-cell detail does not. That
    means the cap most endangers the PER-CELL spatial resolution (eta/phi), and
    pT is partially shielded by the global-energy feature. We therefore report
    the retained fraction of BOTH Sigma E and Sigma E_T so the two effects can be
    read separately.
  * "Fraction of cells kept" and "fraction of ENERGY kept" are very different:
    a shower is steeply peaked, so keeping 128 of 400 cells can still retain
    >99% of the energy. The energy fraction is the physically relevant one.

Outputs a 2x3 supervisor-ready figure + a printed summary with the numbers you
would quote on a slide, plus a K-sweep that answers "would raising the cap to
256 actually help, and for which electrons?".

Run:
    python scripts/diagnose_cell_cap.py \
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


def retained_fraction(e_desc_cumsum: np.ndarray, k: int) -> float:
    """Fraction of a positive quantity retained by keeping its k largest cells.

    `e_desc_cumsum` is the cumulative sum of the per-cell quantity sorted in
    DESCENDING energy order (so index k-1 is the top-k partial sum).
    """
    total = e_desc_cumsum[-1]
    if total <= 0:
        return 1.0
    kept = e_desc_cumsum[min(k, e_desc_cumsum.size) - 1]
    return float(kept / total)


def profile(x, y, edges):
    """Median and 16-84% band of y in bins of x. Returns (centers, med, lo, hi, n)."""
    centers, med, lo, hi, n = [], [], [], [], []
    for i in range(len(edges) - 1):
        m = (x >= edges[i]) & (x < edges[i + 1])
        if m.sum() == 0:
            continue
        yi = y[m]
        centers.append(np.sqrt(edges[i] * edges[i + 1])
                       if edges[i] > 0 else 0.5 * (edges[i] + edges[i + 1]))
        med.append(np.median(yi))
        lo.append(np.percentile(yi, 16))
        hi.append(np.percentile(yi, 84))
        n.append(int(m.sum()))
    return (np.asarray(centers), np.asarray(med),
            np.asarray(lo), np.asarray(hi), np.asarray(n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-cells", type=int, default=128,
                    help="the cap currently used in the training config")
    ap.add_argument("--max-abs-eta", type=float, default=3.0)
    ap.add_argument("--k-sweep", type=int, nargs="+",
                    default=[32, 64, 96, 128, 192, 256, 384, 512])
    ap.add_argument("--max-events", type=int, default=None,
                    help="subsample this many electrons for a quick look")
    ap.add_argument("--out-dir", default="results/diagnostics")
    args = ap.parse_args()

    K = args.max_cells
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cols = ["split", "truth_pt", "truth_eta", "n_cells",
            "cell_x", "cell_y", "cell_z", "cell_e_calibrated"]
    df = pl.read_parquet(args.parquet, columns=cols).filter(
        (pl.col("split") == args.split)
        & (pl.col("truth_eta").abs() <= args.max_abs_eta)
    )
    if args.max_events is not None and df.height > args.max_events:
        df = df.sample(n=args.max_events, seed=0)

    pt, abseta, ncells = [], [], []
    f_e_K, f_et_K = [], []                       # retained frac at the current cap
    f_e_sweep = {k: [] for k in args.k_sweep}    # retained E frac per sweep K
    f_et_sweep = {k: [] for k in args.k_sweep}   # retained E_T frac per sweep K

    for row in df.iter_rows(named=True):
        e = np.asarray(row["cell_e_calibrated"], dtype=np.float64)
        e = np.clip(e, 0.0, None)
        x = np.asarray(row["cell_x"], dtype=np.float64)
        y = np.asarray(row["cell_y"], dtype=np.float64)
        z = np.asarray(row["cell_z"], dtype=np.float64)
        r3 = np.sqrt(x * x + y * y + z * z)
        sin_theta = np.hypot(x, y) / np.clip(r3, 1e-9, None)
        et = e * sin_theta                       # per-cell transverse energy

        # rank by ENERGY, descending -- exactly what the model's cap does
        order = np.argsort(-e, kind="mergesort")
        e_cumsum = np.cumsum(e[order])
        et_cumsum = np.cumsum(et[order])         # E_T retained by top-K-BY-E cells

        pt.append(float(row["truth_pt"]))
        abseta.append(abs(float(row["truth_eta"])))
        ncells.append(int(row["n_cells"]))

        f_e_K.append(retained_fraction(e_cumsum, K))
        f_et_K.append(retained_fraction(et_cumsum, K))
        for k in args.k_sweep:
            f_e_sweep[k].append(retained_fraction(e_cumsum, k))
            f_et_sweep[k].append(retained_fraction(et_cumsum, k))

    pt = np.asarray(pt); abseta = np.asarray(abseta); ncells = np.asarray(ncells)
    f_e_K = np.asarray(f_e_K); f_et_K = np.asarray(f_et_K)
    n_tot = pt.size
    binds = ncells > K                            # cap actually removes cells

    # shared log-pT bins (guard against tiny bins the same way as resolution plots)
    MIN_BIN = 50
    pt_edges = np.logspace(np.log10(max(pt.min(), 1e-3)), np.log10(pt.max()), 13)
    eta_edges = np.linspace(0, args.max_abs_eta, 13)

    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))
    fig.suptitle(
        f"Top-{K}-by-energy cell cap: what does it discard? "
        f"ColliderML Z$\\to$ee pu200, {args.split} split, "
        f"$|\\eta^{{truth}}|\\leq{args.max_abs_eta:g}$  (n={n_tot})",
        fontsize=12,
    )

    # (a) how often does the cap even bind, vs pT ---------------------------
    ax = axes[0, 0]
    c, frac_bind = [], []
    for i in range(len(pt_edges) - 1):
        m = (pt >= pt_edges[i]) & (pt < pt_edges[i + 1])
        if m.sum() < MIN_BIN:
            continue
        c.append(np.sqrt(pt_edges[i] * pt_edges[i + 1]))
        frac_bind.append(binds[m].mean())
    ax.plot(c, frac_bind, "o-", color="C3")
    ax.set_xscale("log"); ax.set_ylim(0, 1.02)
    ax.set_xlabel("true $p_T$ [GeV]")
    ax.set_ylabel(f"fraction with $n_{{cells}} > {K}$")
    ax.set_title("(a) How often the cap removes cells", fontsize=10)

    # (b) retained Sigma-E fraction vs pT -----------------------------------
    ax = axes[0, 1]
    cc, med, lo, hi, nn = profile(pt[binds], f_e_K[binds], pt_edges)
    keep = nn >= MIN_BIN
    ax.plot(cc[keep], med[keep], "o-", color="C0", label="median")
    ax.fill_between(cc[keep], lo[keep], hi[keep], alpha=0.25, color="C0",
                    label="16-84%")
    ax.set_xscale("log"); ax.set_xlabel("true $p_T$ [GeV]")
    ax.set_ylabel(f"$\\Sigma E$ retained by top {K}")
    ax.set_title("(b) Energy kept vs pT (capped electrons only)", fontsize=10)
    ax.legend(fontsize=8)

    # (c) retained Sigma-E_T fraction vs pT (pT-relevant) -------------------
    ax = axes[0, 2]
    cc, med, lo, hi, nn = profile(pt[binds], f_et_K[binds], pt_edges)
    keep = nn >= MIN_BIN
    ax.plot(cc[keep], med[keep], "o-", color="C2", label="median")
    ax.fill_between(cc[keep], lo[keep], hi[keep], alpha=0.25, color="C2",
                    label="16-84%")
    ax.set_xscale("log"); ax.set_xlabel("true $p_T$ [GeV]")
    ax.set_ylabel(f"$\\Sigma E_T$ retained by top {K}")
    ax.set_title("(c) Transverse energy kept vs pT", fontsize=10)
    ax.legend(fontsize=8)

    # (d) retained Sigma-E fraction vs |eta| (barrel/endcap) ----------------
    ax = axes[1, 0]
    cc, med, lo, hi, nn = profile(abseta[binds], f_e_K[binds], eta_edges)
    keep = nn >= MIN_BIN
    ax.plot(cc[keep], med[keep], "o-", color="C4", label="median")
    ax.fill_between(cc[keep], lo[keep], hi[keep], alpha=0.25, color="C4",
                    label="16-84%")
    ax.axvline(1.5, color="grey", ls="--", lw=1, label="barrel/endcap (1.5)")
    ax.set_xlabel("$|\\eta^{truth}|$")
    ax.set_ylabel(f"$\\Sigma E$ retained by top {K}")
    ax.set_title("(d) Energy kept vs |eta| (capped electrons only)", fontsize=10)
    ax.legend(fontsize=8)

    # (e) K-sweep: would a bigger cap help, and for whom? -------------------
    ax = axes[1, 1]
    hi_pt = pt >= np.percentile(pt, 90)          # top-decile pT band
    lo_pt = pt <= np.percentile(pt, 50)
    ks = np.asarray(args.k_sweep)
    for band, sel, style in [("high pT (top 10%)", hi_pt, "o-"),
                             ("low pT (bottom 50%)", lo_pt, "s--")]:
        med_e = [np.median(np.asarray(f_e_sweep[k])[sel]) for k in ks]
        ax.plot(ks, med_e, style, label=f"$\\Sigma E$, {band}")
    ax.axvline(K, color="r", ls=":", lw=1.5, label=f"current cap ({K})")
    ax.set_xlabel("cap size $K$ (cells kept)")
    ax.set_ylabel("median $\\Sigma E$ retained")
    ax.set_title("(e) Retained energy vs cap size", fontsize=10)
    ax.legend(fontsize=8)

    # (f) n_cells vs pT, with the cap line ----------------------------------
    ax = axes[1, 2]
    hb = ax.hexbin(pt, ncells, gridsize=40, xscale="log", yscale="log",
                   mincnt=1, cmap="viridis")
    ax.axhline(K, color="r", ls="--", lw=1.5, label=f"cap = {K}")
    ax.set_xlabel("true $p_T$ [GeV]")
    ax.set_ylabel("$n_{cells}$ in cluster")
    ax.set_title("(f) Cluster size vs pT", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    fig.colorbar(hb, ax=ax, label="electrons")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig_path = out / f"cell_cap_report_K{K}.png"
    fig.savefig(fig_path, dpi=200)
    fig.savefig(fig_path.with_suffix(".pdf"))

    # ---- printed summary (slide numbers) ----------------------------------
    hi_pt_thr = float(np.percentile(pt, 90))
    def band_stats(sel, name):
        fe = f_e_K[sel]; fet = f_et_K[sel]
        print(f"  {name:<22} n={sel.sum():>6}  "
              f"cap binds {binds[sel].mean():5.1%}  "
              f"median SumE kept {np.median(fe):6.3%}  "
              f"1st-pct SumE kept {np.percentile(fe,1):6.3%}  "
              f"median SumE_T kept {np.median(fet):6.3%}")

    print(f"\n=== top-{K} cell cap, {n_tot} electrons "
          f"(|eta|<={args.max_abs_eta:g}, {args.split}) ===")
    print(f"cap binds (n_cells > {K}) for {binds.mean():.1%} of all electrons; "
          f"median n_cells = {np.median(ncells):.0f}, "
          f"95th pct = {np.percentile(ncells,95):.0f}, max = {ncells.max()}")
    band_stats(np.ones(n_tot, bool), "all")
    band_stats(pt >= hi_pt_thr, f"high pT (>={hi_pt_thr:.0f} GeV)")
    band_stats(pt <= np.percentile(pt, 50), "low pT (bottom 50%)")
    band_stats((abseta > 1.5), "endcap (|eta|>1.5)")
    band_stats((abseta <= 1.5), "barrel (|eta|<=1.5)")

    # K-sweep table for the high-pT band
    print(f"\nK-sweep, median SumE retained (high-pT top-decile, "
          f"thr={hi_pt_thr:.0f} GeV):")
    for k in args.k_sweep:
        print(f"  K={k:>4}: {np.median(np.asarray(f_e_sweep[k])[pt>=hi_pt_thr]):.3%}")

    print(f"\nsaved {fig_path} and {fig_path.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()