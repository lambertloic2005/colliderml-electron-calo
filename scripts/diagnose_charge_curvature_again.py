"""PLAN B: sharpen the charge-curvature observable before deciding to retrain.

The Step-0 dphi/dr slope was weak (~0.58-0.64) and its per-slice curves were
jagged. Two physics-motivated reasons it may be UNDER-summarizing a real signal:

  1. WRONG DEPTH COORDINATE BY REGION. A shower's "depth" runs along r in the
     BARREL (develops radially outward at ~fixed z) but along |z| in the ENDCAP
     (develops in z at ~fixed r). A single dphi/dr fit divides by var(r), which
     is ~0 for endcap electrons -> noise. We therefore also fit dphi/d|z|, and an
     ADAPTIVE slope that uses whichever of {r,|z|} has the larger (weighted)
     spread for that electron, and we split every result by barrel/endcap.

  2. A LINEAR SLOPE MAY BE A POOR SUMMARY of a curved, jagged profile. We test
     (a) an inner-vs-outer <dphi> endpoint difference (robust to mid-slice
     scatter) and (b) a Fisher linear discriminant fit on the FULL 6-slice
     <dphi> profile (train/test split) -- the ceiling of what a linear readout
     of the profile can do, which is what the model actually receives.

All model-free. Decision: if dphi/d|z| (endcap) or the adaptive slope or the
profile-LDA clears the Step-0 dphi/dr baseline by a clear margin, put the
BETTER observable in the model. If nothing beats ~0.6, the charge signal here is
genuinely weak and the model's ~0.78 comes from elsewhere.

Run:
  python scripts/diagnose_charge_curvature_pathB.py \
      --parquet .../zee_pu200_supervised_dbscan_TEST.parquet \
      --split test --slices 6 --out-dir results/diagnostics
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


def wmean(v, w):
    return float(np.sum(w * v) / np.sum(w))


def wvar(v, w):
    m = wmean(v, w)
    return float(np.sum(w * (v - m) ** 2) / np.sum(w))


def ls_slope(depth, dphi, w):
    """E-weighted least-squares slope d(dphi)/d(depth). NaN if depth has no spread."""
    vr = wvar(depth, w)
    if vr <= 1e-9:
        return np.nan
    db = wmean(depth, w); pb = wmean(dphi, w)
    cov = float(np.sum(w * (depth - db) * (dphi - pb)) / np.sum(w))
    return cov / vr


def slice_profile(depth, dphi, w, K):
    """Per-slice E-weighted <dphi> along `depth`; empty slices -> 0 (as dataset.py)."""
    prof = np.zeros(K, dtype=np.float64)
    lo, hi = depth.min(), depth.max()
    if hi - lo < 1e-9:
        return prof
    edges = np.linspace(lo, hi + 1e-6, K + 1)
    for k in range(K):
        m = (depth >= edges[k]) & (depth < edges[k + 1])
        if m.any():
            prof[k] = np.sum(w[m] * dphi[m]) / w[m].sum()
    return prof


def inner_outer(prof):
    """Outer-third minus inner-third mean <dphi> (endpoint contrast)."""
    K = prof.size; t = max(K // 3, 1)
    return float(prof[-t:].mean() - prof[:t].mean())


def fisher_lda(X, y, ridge=1e-6):
    """Fit Fisher LDA direction on (X,y in {-1,+1}); return w, b."""
    Xp, Xn = X[y > 0], X[y < 0]
    mp, mn = Xp.mean(0), Xn.mean(0)
    Sw = np.cov(Xp.T) * (len(Xp) - 1) + np.cov(Xn.T) * (len(Xn) - 1)
    Sw += ridge * np.eye(X.shape[1])
    w = np.linalg.solve(Sw, mp - mn)
    b = -0.5 * (mp + mn) @ w
    return w, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-abs-eta", type=float, default=3.0)
    ap.add_argument("--barrel-endcap", type=float, default=1.5)
    ap.add_argument("--slices", type=int, default=6)
    ap.add_argument("--max-events", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="results/diagnostics")
    args = ap.parse_args()

    K = args.slices
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    need = ["truth_charge", "cell_phi", "cell_x", "cell_y", "cell_z"]
    have = pl.read_parquet(args.parquet, n_rows=1).columns
    miss = [c for c in need if c not in have]
    if miss:
        raise SystemExit(f"[fatal] parquet missing {miss}.")

    cols = ["split", "truth_pt", "truth_eta", "truth_charge",
            "cell_phi", "cell_x", "cell_y", "cell_z", TOT_COL]
    df = pl.read_parquet(args.parquet, columns=cols).filter(
        (pl.col("split") == args.split)
        & (pl.col("truth_eta").abs() <= args.max_abs_eta))
    if args.max_events and df.height > args.max_events:
        df = df.sample(n=args.max_events, seed=args.seed)

    pt, ae, charge = [], [], []
    s_r, s_z, s_ad = [], [], []
    io_ad = []
    prof_ad = []
    for row in df.iter_rows(named=True):
        e = np.clip(np.asarray(row[TOT_COL], np.float64), 0, None)
        px = np.asarray(row["cell_x"], np.float64)
        py = np.asarray(row["cell_y"], np.float64)
        pz = np.asarray(row["cell_z"], np.float64)
        ph = np.asarray(row["cell_phi"], np.float64)
        w = np.clip(e, 1e-9, None)
        if e.size < 3 or e.sum() <= 0:
            continue
        cphi = np.arctan2(np.sum(w * np.sin(ph)), np.sum(w * np.cos(ph)))
        dphi = np.arctan2(np.sin(ph - cphi), np.cos(ph - cphi))
        r = np.hypot(px, py); absz = np.abs(pz)

        sr = ls_slope(r, dphi, w)
        sz = ls_slope(absz, dphi, w)
        # adaptive: use the coordinate with the larger weighted spread
        depth = r if wvar(r, w) >= wvar(absz, w) else absz
        sad = ls_slope(depth, dphi, w)
        prof = slice_profile(depth, dphi, w, K)

        pt.append(float(row["truth_pt"])); ae.append(abs(float(row["truth_eta"])))
        charge.append(int(row["truth_charge"]))
        s_r.append(sr); s_z.append(sz); s_ad.append(sad)
        io_ad.append(inner_outer(prof)); prof_ad.append(prof)

    pt = np.asarray(pt); ae = np.asarray(ae); charge = np.asarray(charge)
    s_r = np.asarray(s_r); s_z = np.asarray(s_z); s_ad = np.asarray(s_ad)
    io_ad = np.asarray(io_ad); prof_ad = np.asarray(prof_ad)
    n = pt.size
    barrel = ae <= args.barrel_endcap; endcap = ae > args.barrel_endcap

    def sign_acc(score, sel):
        s = np.asarray(score)[sel]; c = charge[sel]; m = np.isfinite(s)
        if m.sum() == 0:
            return np.nan
        corr = np.sign(np.mean(np.sign(s[m]) * c[m])) or 1.0
        return float(np.mean((corr * np.sign(s[m])) == c[m]))

    # profile-LDA with a train/test split (avoid in-sample optimism)
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(n); half = n // 2
    tr, te = idx[:half], idx[half:]
    Xtr, ytr = np.nan_to_num(prof_ad[tr]), charge[tr]
    w_lda, b_lda = fisher_lda(Xtr, ytr)
    score_lda = np.nan_to_num(prof_ad) @ w_lda + b_lda
    # calibrate sign on train, evaluate on test subsets
    corr_lda = np.sign(np.mean(np.sign(score_lda[tr]) * charge[tr])) or 1.0

    def lda_acc(sel_mask):
        sel = np.intersect1d(te, np.nonzero(sel_mask)[0])
        if sel.size == 0:
            return np.nan
        return float(np.mean((corr_lda * np.sign(score_lda[sel])) == charge[sel]))

    allm = np.ones(n, bool)
    methods = [
        ("dphi/dr", s_r), ("dphi/d|z|", s_z), ("adaptive slope", s_ad),
        ("inner-outer", io_ad),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))
    fig.suptitle(f"PLAN B: sharpening the charge-curvature observable  "
                 f"(model-free, {args.split}, n={n})", fontsize=12)

    # (a) overall accuracy by method (+ LDA)
    ax = axes[0, 0]
    names = [m[0] for m in methods] + ["profile-LDA"]
    vals = [sign_acc(s, allm) for _, s in methods] + [lda_acc(allm)]
    ax.bar(names, vals, color=["C0", "C1", "C2", "C3", "C4"])
    ax.axhline(0.5, color="grey", ls="--", lw=1)
    ax.set_ylim(0.45, max(0.75, np.nanmax(vals) + 0.05))
    ax.set_ylabel("charge accuracy"); ax.set_title("(a) Overall, by method", fontsize=10)
    ax.tick_params(axis="x", labelrotation=20)

    # (b) barrel vs endcap for the three slope variants  -- the geometry story
    ax = axes[0, 1]
    variants = [("dphi/dr", s_r), ("dphi/d|z|", s_z), ("adaptive", s_ad)]
    x = np.arange(len(variants))
    ax.bar(x - 0.2, [sign_acc(s, barrel) for _, s in variants], 0.4, label="barrel", color="C0")
    ax.bar(x + 0.2, [sign_acc(s, endcap) for _, s in variants], 0.4, label="endcap", color="C3")
    ax.axhline(0.5, color="grey", ls="--", lw=1)
    ax.set_xticks(x); ax.set_xticklabels([v for v, _ in variants])
    ax.set_ylim(0.45, 0.85); ax.set_ylabel("charge accuracy")
    ax.set_title("(b) Depth coordinate x region", fontsize=10); ax.legend(fontsize=8)

    # (c) accuracy vs pT for the best of {adaptive, LDA}
    ax = axes[0, 2]
    edges = np.logspace(np.log10(max(pt.min(), 1e-3)), np.log10(pt.max()), 8)
    c_ad, c_ld = [], []; cc = []
    for i in range(len(edges) - 1):
        m = (pt >= edges[i]) & (pt < edges[i + 1])
        if m.sum() < 50:
            continue
        cc.append(np.sqrt(edges[i] * edges[i + 1]))
        c_ad.append(sign_acc(s_ad, m)); c_ld.append(lda_acc(m))
    ax.plot(cc, c_ad, "o-", color="C2", label="adaptive slope")
    ax.plot(cc, c_ld, "s--", color="C4", label="profile-LDA")
    ax.axhline(0.5, color="grey", ls="--", lw=1)
    ax.set_xscale("log"); ax.set_ylim(0.45, 0.85)
    ax.set_xlabel("true $p_T$ [GeV]"); ax.set_ylabel("charge accuracy")
    ax.set_title("(c) Best observables vs pT", fontsize=10); ax.legend(fontsize=8)

    # (d,e) per-slice <dphi> by charge, barrel then endcap (adaptive depth)
    sl = np.arange(1, K + 1)
    for ax, mask, title in [(axes[1, 0], barrel, "(d) Profile by charge — BARREL"),
                            (axes[1, 1], endcap, "(e) Profile by charge — ENDCAP")]:
        for q, col, lab in [(-1, "C0", "e- (q=-1)"), (+1, "C3", "e+ (q=+1)")]:
            sub = prof_ad[mask & (charge == q)]
            if sub.size:
                ax.plot(sl, np.nanmean(sub, axis=0), "o-", color=col, label=lab)
        ax.axhline(0, color="grey", ls=":", lw=1)
        ax.set_xlabel("radial/depth slice (inner->outer)")
        ax.set_ylabel("mean $\\langle\\Delta\\phi\\rangle$ [rad]")
        ax.set_title(title, fontsize=10); ax.legend(fontsize=8)

    # (f) profile-LDA score distribution by charge (test set)
    ax = axes[1, 2]
    stest = score_lda[te]; ctest = charge[te]
    rng2 = np.nanpercentile(stest, [1, 99])
    bins = np.linspace(rng2[0], rng2[1], 50)
    ax.hist(stest[ctest < 0], bins=bins, alpha=0.6, color="C0", label="e- (q=-1)")
    ax.hist(stest[ctest > 0], bins=bins, alpha=0.6, color="C3", label="e+ (q=+1)")
    ax.axvline(0, color="grey", ls="--", lw=1)
    ax.set_xlabel("LDA score on <dphi> profile"); ax.set_ylabel("electrons / bin")
    ax.set_title("(f) Profile-LDA separation (test)", fontsize=10); ax.legend(fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fp = out / "charge_curvature_pathB.png"
    fig.savefig(fp, dpi=200); fig.savefig(fp.with_suffix(".pdf"))

    # ---- printed summary ----
    print(f"\n=== PLAN B charge-curvature, {n} electrons "
          f"(barrel {int(barrel.sum())}, endcap {int(endcap.sum())}) ===")
    print("method            overall   barrel   endcap")
    for name, s in methods:
        print(f"  {name:<15} {sign_acc(s,allm):.3f}    {sign_acc(s,barrel):.3f}    {sign_acc(s,endcap):.3f}")
    print(f"  {'profile-LDA':<15} {lda_acc(allm):.3f}    {lda_acc(barrel):.3f}    {lda_acc(endcap):.3f}")
    print("\nbaseline to beat: Step-0 dphi/dr ~ 0.58-0.64.")
    print("if dphi/d|z| (endcap) or adaptive or profile-LDA clears that clearly,")
    print("put THAT observable in the model instead of the bare dphi/dr slope.")
    print(f"saved {fp}")


if __name__ == "__main__":
    main()