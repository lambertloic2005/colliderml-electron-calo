"""Does the top-K energy cap starve the CHARGE signal -- and would a tail-aware
selection recover it?  Model-free, no retrain needed.

Charge physics in this pipeline: each event is rotated so the energy-weighted
azimuthal centroid sits at phi=0, which zeros the FIRST moment of dphi. The
leading charge-shape signal that survives is therefore the SKEW (third moment)
of dphi -- the bremsstrahlung-tail asymmetry, whose SIGN tracks the charge. That
tail lives in LOW-energy cells offset to one phi side, i.e. exactly the cells a
top-K-by-ENERGY cap discards first. So energy ranking is adversarial to charge.

This script quantifies that with a deliberately simple, model-free classifier:
predict charge = sign_cal * sign(skew_phi), where sign_cal is fixed once on the
full-cluster skew so that it agrees with truth on average. We then recompute the
skew on three cell selections and report charge accuracy vs cell budget K:

  1. energy-ranked  : top-K cells by total energy   (what the model does now)
  2. tail-aware     : (1-f)*K by energy  +  f*K by |dphi| among the remainder
                      (explicitly keeps the phi-tail that energy ranking drops)
  3. all cells      : the information ceiling of the skew classifier

Accuracy is reported in pT bands, so the 1/pT physics falloff of the bend is
held roughly fixed WITHIN a band and the remaining variation with K isolates the
cap. If energy-ranked accuracy at the current cap is well below all-cells, the
cap is starving charge; if tail-aware at the same budget recovers most of the
gap, the right tweak is the SELECTION RULE, not a bigger K.

Caveat: sign(skew) is a proxy for the information content, not the trained head.
It sets a model-free bound: if even this loses accuracy under the cap, the cap is
suppressing recoverable charge information. It cannot prove the trained model
would recover it -- but it tells you whether the information is there to recover.

Run:
  python scripts/diagnose_charge_vs_cap.py \
      --parquet data/electrons/testRuche/zee_pu200_supervised_dbscan_TEST.parquet \
      --split test --max-cells 128 --tail-frac 0.5 --out-dir results/diagnostics
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

TOT_COL = "cell_e_calibrated"          # what the model ranks on (total energy)


def weighted_skew(dphi, w):
    """Energy-weighted skew of dphi about its weighted mean. NaN if degenerate."""
    wsum = w.sum()
    if wsum <= 0 or dphi.size < 3:
        return np.nan
    mu = np.sum(w * dphi) / wsum
    d = dphi - mu
    var = np.sum(w * d * d) / wsum
    if var <= 1e-12:
        return np.nan
    return float(np.sum(w * d ** 3) / wsum / var ** 1.5)


def select_energy(e, k):
    """Indices of the top-k cells by energy."""
    k = min(k, e.size)
    return np.argsort(-e, kind="mergesort")[:k]


def select_tail_aware(e, absdphi, k, tail_frac):
    """(1-tail_frac)*k hottest cells + tail_frac*k most phi-extreme of the rest."""
    k = min(k, e.size)
    n_e = int(round((1.0 - tail_frac) * k))
    n_t = k - n_e
    order_e = np.argsort(-e, kind="mergesort")
    core = order_e[:n_e]
    rest = order_e[n_e:]
    if n_t > 0 and rest.size:
        tail = rest[np.argsort(-absdphi[rest], kind="mergesort")[:n_t]]
        return np.concatenate([core, tail])
    return core


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-cells", type=int, default=128)
    ap.add_argument("--max-abs-eta", type=float, default=3.0)
    ap.add_argument("--tail-frac", type=float, default=0.5,
                    help="fraction of the K budget spent on phi-tail cells")
    ap.add_argument("--k-sweep", type=int, nargs="+",
                    default=[16, 32, 64, 96, 128, 192, 256, 384, 512])
    ap.add_argument("--max-events", type=int, default=None)
    ap.add_argument("--out-dir", default="results/diagnostics")
    args = ap.parse_args()

    K = args.max_cells
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    need = ["truth_charge", "cell_phi"]
    have = pl.read_parquet(args.parquet, n_rows=1).columns
    miss = [c for c in need if c not in have]
    if miss:
        raise SystemExit(f"[fatal] parquet missing {miss}; need truth_charge and "
                         "per-cell cell_phi to measure the charge skew.")

    cols = ["split", "truth_pt", "truth_eta", "truth_charge", "cell_phi", TOT_COL]
    df = pl.read_parquet(args.parquet, columns=cols).filter(
        (pl.col("split") == args.split)
        & (pl.col("truth_eta").abs() <= args.max_abs_eta))
    if args.max_events and df.height > args.max_events:
        df = df.sample(n=args.max_events, seed=0)

    ks = list(args.k_sweep)
    pt, charge = [], []
    skew_all = []
    skew_e = {k: [] for k in ks}          # energy-ranked skew at budget k
    skew_t = {k: [] for k in ks}          # tail-aware skew at budget k

    for row in df.iter_rows(named=True):
        e = np.clip(np.asarray(row[TOT_COL], np.float64), 0, None)
        phi = np.asarray(row["cell_phi"], np.float64)
        if e.size < 3 or e.sum() <= 0:
            continue
        # energy-weighted circular centroid, then dphi about it
        cphi = np.arctan2(np.sum(e * np.sin(phi)), np.sum(e * np.cos(phi)))
        dphi = np.arctan2(np.sin(phi - cphi), np.cos(phi - cphi))
        absd = np.abs(dphi)

        pt.append(float(row["truth_pt"])); charge.append(int(row["truth_charge"]))
        skew_all.append(weighted_skew(dphi, e))
        for k in ks:
            ie = select_energy(e, k)
            skew_e[k].append(weighted_skew(dphi[ie], e[ie]))
            it = select_tail_aware(e, absd, k, args.tail_frac)
            skew_t[k].append(weighted_skew(dphi[it], e[it]))

    pt = np.asarray(pt); charge = np.asarray(charge)
    skew_all = np.asarray(skew_all)
    n = pt.size

    # calibrate sign(skew)->charge once, on the full-cluster skew
    good = np.isfinite(skew_all)
    corr = np.sign(np.mean(np.sign(skew_all[good]) * charge[good]))
    if corr == 0:
        corr = 1.0
    print(f"skew->charge sign convention: corr={corr:+.0f} "
          f"(all-cluster accuracy {np.mean((corr*np.sign(skew_all[good]))==charge[good]):.3f})")

    def acc(skew_arr, sel):
        s = np.asarray(skew_arr)[sel]
        c = charge[sel]
        m = np.isfinite(s)
        if m.sum() == 0:
            return np.nan
        return float(np.mean((corr * np.sign(s[m])) == c[m]))

    hi = pt >= np.percentile(pt, 90)
    lo = pt <= np.percentile(pt, 50)
    bands = [("high pT (top 10%)", hi), ("low pT (bottom 50%)", lo)]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"Does the top-{K} energy cap starve the charge (skew) signal? "
        f"model-free, ColliderML Z$\\to$ee pu200, {args.split} (n={n})",
        fontsize=12)

    # (a) accuracy vs K: energy-ranked, per band, + all-cells ceiling
    ax = axes[0]
    for (label, sel), col in zip(bands, ["C0", "C1"]):
        ae = [acc(skew_e[k], sel) for k in ks]
        ax.plot(ks, ae, "o-", color=col, label=f"energy-ranked, {label}")
        ceil = acc(skew_all, sel)
        ax.axhline(ceil, color=col, ls=":", lw=1)
    ax.axvline(K, color="r", ls=":", lw=1.5, label=f"current cap ({K})")
    ax.axhline(0.5, color="grey", ls="--", lw=1)
    ax.set_xlabel("cell budget K"); ax.set_ylabel("charge accuracy (sign of skew)")
    ax.set_title("(a) Energy-ranked cap vs all-cells ceiling (dotted)", fontsize=10)
    ax.legend(fontsize=8)

    # (b) energy-ranked vs tail-aware at each K (high-pT band, the hard case)
    ax = axes[1]
    sel = hi
    ae = [acc(skew_e[k], sel) for k in ks]
    at = [acc(skew_t[k], sel) for k in ks]
    ax.plot(ks, ae, "o-", color="C0", label="energy-ranked")
    ax.plot(ks, at, "s--", color="C2",
            label=f"tail-aware (frac={args.tail_frac:g})")
    ax.axhline(acc(skew_all, sel), color="grey", ls=":", lw=1, label="all-cells ceiling")
    ax.axvline(K, color="r", ls=":", lw=1.5, label=f"cap ({K})")
    ax.axhline(0.5, color="grey", ls="--", lw=1)
    ax.set_xlabel("cell budget K"); ax.set_ylabel("charge accuracy")
    ax.set_title("(b) Selection RULE at fixed budget (high pT)", fontsize=10)
    ax.legend(fontsize=8)

    # (c) skew separability by truth charge (sanity: is there any signal at all?)
    ax = axes[2]
    s = skew_all[good]; c = charge[good]
    rng = np.percentile(s, [1, 99])
    bins = np.linspace(rng[0], rng[1], 60)
    ax.hist(s[c < 0], bins=bins, alpha=0.6, color="C0", label="electron (q=-1)")
    ax.hist(s[c > 0], bins=bins, alpha=0.6, color="C3", label="positron (q=+1)")
    ax.axvline(0, color="grey", ls="--", lw=1)
    ax.set_xlabel("skew of $\\Delta\\phi$ (all cells)"); ax.set_ylabel("electrons / bin")
    ax.set_title("(c) Is the skew charge-separating at all?", fontsize=10)
    ax.legend(fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fp = out / f"charge_vs_cap_K{K}.png"
    fig.savefig(fp, dpi=200); fig.savefig(fp.with_suffix(".pdf"))

    # ---- printed summary ----
    print(f"\n=== charge-vs-cap (skew classifier), {n} electrons ===")
    for label, sel in bands:
        ceil = acc(skew_all, sel)
        e_at = acc(skew_e[K], sel)
        t_at = acc(skew_t[K], sel)
        print(f"  {label:<22} all-cells {ceil:.3f} | "
              f"energy-ranked@{K} {e_at:.3f} | tail-aware@{K} {t_at:.3f} | "
              f"gap recovered {'' if not np.isfinite(ceil-e_at) or ceil==e_at else f'{(t_at-e_at)/max(ceil-e_at,1e-6):.0%}'}")
    print("\nread: energy-ranked@cap << all-cells  => the cap starves charge.")
    print("      tail-aware@cap ~ all-cells        => fix the SELECTION RULE, not K.")
    print(f"saved {fp}")


if __name__ == "__main__":
    main()