#!/usr/bin/env python3
"""
make_summary_figs.py
====================

Build the two comparison figures for the truth-free (unsupervised) clustering
study. Neither exists in the per-run outputs: the test script plots one model on
one dataset, whereas the question here is the PAIRED difference between two
reconstructions of the same events, split by detector region.

fig1_resolution_by_region.png
    2 rows (barrel |eta| < 1.5, endcap 1.5-3.0) x 3 columns
    (phi residual, relative pT residual, charge ROC), supervised vs truth-free
    overlaid on the electrons reconstructed by BOTH pipelines.

fig2_efficiency.png
    Cluster-matching efficiency vs |truth eta| and vs truth pT, with binomial
    errors. This is the acceptance loss, which the paired figure cannot show --
    electrons the truth-free pipeline never reconstructs are absent from it.

Usage
-----
    python scripts/make_summary_figs.py \
        --preds-a A/preds.npz --preds-c C/preds.npz \
        --supervised sup.parquet --unsup unsup.parquet \
        --outdir $HOME/results/figs
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

A_LABEL = "supervised (truth-seeded) clusters"
C_LABEL = "truth-free DBSCAN clusters"
A_COL, C_COL = "#1f77b4", "#d62728"
BANDS = [("barrel  |eta| < 1.5", 0.0, 1.5), ("endcap  1.5 < |eta| < 3.0", 1.5, 3.0)]


def wrap(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2 * np.pi) - np.pi


def load(path: Path) -> dict:
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def keys(d: dict) -> list:
    return list(zip(np.round(d["truth_eta"], 6),
                    np.round(d["truth_phi"], 6),
                    np.round(d["truth_pt"], 4)))


def match(a: dict, c: dict) -> tuple[np.ndarray, np.ndarray]:
    idx = {}
    for i, k in enumerate(keys(a)):
        idx.setdefault(k, i)
    ia, ic = [], []
    for j, k in enumerate(keys(c)):
        i = idx.get(k)
        if i is not None:
            ia.append(i)
            ic.append(j)
    return np.asarray(ia, int), np.asarray(ic, int)


def robust_sigma(x: np.ndarray) -> float:
    q16, q84 = np.percentile(x, [15.865, 84.135])
    return float((q84 - q16) / 2)


def roc(score: np.ndarray, pos: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    order = np.argsort(-score)
    p = pos[order].astype(float)
    tp = np.cumsum(p)
    fp = np.cumsum(1 - p)
    npos, nneg = max(p.sum(), 1), max((1 - p).sum(), 1)
    tpr, fpr = tp / npos, fp / nneg
    auc = float(np.trapezoid(tpr, fpr)) if hasattr(np, "trapezoid") else float(np.trapz(tpr, fpr))
    return fpr, tpr, auc


def fig_resolution(a: dict, c: dict, ia, ic, out: Path) -> None:
    aeta = np.abs(a["truth_eta"][ia])
    res = {
        "A": dict(
            phi=wrap(a["pred_phi"][ia] - a["truth_phi"][ia]),
            pt=(a["pred_pt"][ia] - a["truth_pt"][ia]) / np.clip(a["truth_pt"][ia], 1e-9, None),
            logit=a.get("charge_logit", np.zeros(len(a["truth_pt"])))[ia],
            q=a["charge"][ia] > 0),
        "C": dict(
            phi=wrap(c["pred_phi"][ic] - c["truth_phi"][ic]),
            pt=(c["pred_pt"][ic] - c["truth_pt"][ic]) / np.clip(c["truth_pt"][ic], 1e-9, None),
            logit=c.get("charge_logit", np.zeros(len(c["truth_pt"])))[ic],
            q=c["charge"][ic] > 0),
    }

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    for r, (label, lo, hi) in enumerate(BANDS):
        sel = (aeta >= lo) & (aeta < hi)
        n = int(sel.sum())

        ax = axes[r, 0]
        bins = np.linspace(-0.05, 0.05, 90)
        for tag, col, lab in (("A", A_COL, A_LABEL), ("C", C_COL, C_LABEL)):
            v = res[tag]["phi"][sel]
            s = robust_sigma(v)
            ax.hist(v, bins=bins, histtype="step", lw=1.6, color=col, density=True,
                    label=f"{lab}\n  sigma = {s:.4f} rad")
        ax.set_xlabel("phi residual [rad]")
        ax.set_ylabel("density")
        ax.set_title(f"{label}   (n = {n})", loc="left")
        ax.legend(fontsize=7.5, frameon=False)

        ax = axes[r, 1]
        bins = np.linspace(-0.4, 0.4, 90)
        for tag, col, lab in (("A", A_COL, A_LABEL), ("C", C_COL, C_LABEL)):
            v = res[tag]["pt"][sel]
            s = robust_sigma(v)
            ax.hist(np.clip(v, bins[0], bins[-1]), bins=bins, histtype="step",
                    lw=1.6, color=col, density=True,
                    label=f"{lab}\n  sigma = {100*s:.2f} %")
        ax.set_xlabel("(pT_pred - pT_true) / pT_true")
        ax.set_title("relative pT residual", loc="left")
        ax.legend(fontsize=7.5, frameon=False)

        ax = axes[r, 2]
        for tag, col, lab in (("A", A_COL, A_LABEL), ("C", C_COL, C_LABEL)):
            fpr, tpr, auc = roc(res[tag]["logit"][sel], res[tag]["q"][sel])
            ax.plot(fpr, tpr, color=col, lw=1.6, label=f"{lab}\n  AUC = {auc:.4f}")
        ax.plot([0, 1], [0, 1], color="grey", lw=0.8, ls=":")
        ax.set_xlabel("false positive rate")
        ax.set_ylabel("true positive rate")
        ax.set_title("charge sign discrimination", loc="left")
        ax.legend(fontsize=7.5, frameon=False, loc="lower right")

    fig.suptitle("Electron reconstruction from calorimeter cells: supervised vs "
                 "truth-free clustering\npaired on electrons reconstructed by "
                 "both pipelines (zee_pu200, OpenDataDetector)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")


def fig_efficiency(sup_pq: str, uns_pq: str, out: Path, min_pt: float) -> None:
    K = ["event_id", "kx", "ky", "kz"]

    def load_pq(f):
        d = (pl.scan_parquet(f).filter(pl.col("split") == "test")
             .select(["event_id", "truth_px", "truth_py", "truth_pz",
                      "truth_eta", "truth_log_pt"]).collect())
        return d.with_columns([pl.col("truth_px").round(6).alias("kx"),
                               pl.col("truth_py").round(6).alias("ky"),
                               pl.col("truth_pz").round(6).alias("kz"),
                               pl.col("truth_log_pt").exp().alias("pt")])

    s, u = load_pq(sup_pq), load_pq(uns_pq)
    cut = lambda d: d.filter((pl.col("pt") >= min_pt) & (pl.col("truth_eta").abs() <= 3))
    s, u = cut(s), cut(u)
    j = (s.join(u.select(K).with_columns(pl.lit(True).alias("found")), on=K, how="left")
           .with_columns(pl.col("found").fill_null(False)))

    aeta = np.abs(j["truth_eta"].to_numpy())
    pt = j["pt"].to_numpy()
    found = j["found"].to_numpy().astype(bool)
    print(f"overall efficiency (pT >= {min_pt}): {found.mean():.4f}  n = {len(found)}")

    def binned(v, edges):
        x, y, ex, ey = [], [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (v >= lo) & (v < hi)
            if m.sum() < 20:
                continue
            e = found[m].mean()
            x.append(0.5 * (lo + hi))
            ex.append(0.5 * (hi - lo))
            y.append(e)
            ey.append(np.sqrt(max(e * (1 - e), 1e-12) / m.sum()))
        return map(np.asarray, (x, y, ex, ey))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    x, y, ex, ey = binned(aeta, np.linspace(0, 3.0, 13))
    axes[0].errorbar(x, y, xerr=ex, yerr=ey, fmt="o", color=C_COL, ms=4, lw=1.2)
    axes[0].axvline(1.5, color="grey", ls="--", lw=0.9)
    axes[0].text(1.55, 0.06, "barrel / endcap", fontsize=8, color="grey")
    axes[0].set_xlabel("|truth eta|")
    axes[0].set_ylabel("cluster-matching efficiency")
    axes[0].set_ylim(0, 1.05)

    x, y, ex, ey = binned(pt, np.array([10, 15, 20, 25, 30, 40, 60, 100, 200]))
    axes[1].errorbar(x, y, xerr=ex, yerr=ey, fmt="o", color=C_COL, ms=4, lw=1.2)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("truth pT [GeV]")
    axes[1].set_ylabel("cluster-matching efficiency")
    axes[1].set_ylim(0, 1.05)

    fig.suptitle(f"Truth-free DBSCAN acceptance, denominator = truth-seeded "
                 f"electrons (pT >= {min_pt:g} GeV, |eta| <= 3)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds-a", required=True)
    ap.add_argument("--preds-c", required=True)
    ap.add_argument("--supervised", required=True)
    ap.add_argument("--unsup", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--min-pt", type=float, default=10.0)
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    a, c = load(Path(args.preds_a)), load(Path(args.preds_c))
    ia, ic = match(a, c)
    print(f"A n = {len(a['truth_pt'])}   C n = {len(c['truth_pt'])}   "
          f"matched = {len(ia)}")
    fig_resolution(a, c, ia, ic, out / "fig1_resolution_by_region.png")
    fig_efficiency(args.supervised, args.unsup,
                   out / "fig2_efficiency.png", args.min_pt)


if __name__ == "__main__":
    main()
