"""Charge plots for slide 17 (no error bars version).

Produces two figures from per-electron charge predictions:
  (1) ROC curve (positron = positive) with AUC
  (2) Charge ID vs pT: per-bin accuracy and mean confidence, with a chance line
      and per-bin counts. No error bars.

Notes on the two review comments this still addresses:
  - y-axis is NOT labelled "probability" (a classifier score is not a calibrated
    probability). Accuracy is a genuine fraction; the second curve is explicitly
    "mean confidence" = mean of max(p, 1-p), the model's probability for its
    chosen class. The vs-pT panel is framed as a calibration check.
  - bins are equal-statistics (quantile) by default, and per-bin counts are
    shown, so sparse bins remain visible even without error bars.

Input: a file with three per-electron arrays
  truth_charge : in {-1,+1}  (positron = +1)   [or {0,1}, positron = 1]
  truth_pt     : GeV
  charge_score : P(positron) in [0,1]          OR  charge_logit (raw logit)

Accepted formats: .npz (array keys), .parquet, .csv.

Run:
    python scripts/plot_charge_results.py charge_eval.npz --out-dir results/charge
    python scripts/plot_charge_results.py charge_eval.npz --bins log
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------- io -----------------------------
def load_arrays(path, charge_col, pt_col, score_col, logit_col):
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".npz":
        d = np.load(path)
        get = lambda k: np.asarray(d[k]) if k in d.files else None
    elif ext in (".parquet", ".csv"):
        import polars as pl
        d = pl.read_parquet(path) if ext == ".parquet" else pl.read_csv(path)
        get = lambda k: d[k].to_numpy() if k in d.columns else None
    else:
        raise ValueError(f"unsupported extension: {ext}")

    q = get(charge_col)
    pt = get(pt_col)
    score = get(score_col)
    logit = get(logit_col)
    if q is None or pt is None:
        raise KeyError(f"need '{charge_col}' and '{pt_col}' in {path}")
    if score is None and logit is None:
        raise KeyError(f"need one of '{score_col}' or '{logit_col}' in {path}")
    if score is None:
        score = 1.0 / (1.0 + np.exp(-np.asarray(logit, dtype=np.float64)))  # sigmoid
    return (np.asarray(q, dtype=np.float64),
            np.asarray(pt, dtype=np.float64),
            np.asarray(score, dtype=np.float64))


# ----------------------------- stats -----------------------------
def roc_curve_auc(scores, labels):
    """labels in {0,1}, 1 = positive (positron). Returns fpr, tpr, auc."""
    order = np.argsort(-scores, kind="mergesort")
    y = labels[order].astype(np.float64)
    P = y.sum()
    N = y.size - P
    if P == 0 or N == 0:
        return np.array([0, 1]), np.array([0, 1]), float("nan")
    tpr = np.concatenate([[0.0], np.cumsum(y) / P])
    fpr = np.concatenate([[0.0], np.cumsum(1.0 - y) / N])
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    auc = float(_trapz(tpr, fpr))
    return fpr, tpr, auc


def quantile_edges(pt, n_bins):
    edges = np.quantile(pt, np.linspace(0, 1, n_bins + 1))
    return np.unique(edges)


def log_edges(pt, n_bins):
    lo = max(pt.min(), 1e-3)
    return np.logspace(np.log10(lo), np.log10(pt.max()), n_bins + 1)


# ----------------------------- plots -----------------------------
def plot_roc(scores, labels, out_path):
    fpr, tpr, auc = roc_curve_auc(scores, labels)
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.plot(fpr, tpr, lw=2, label=f"ROC (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="chance")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("Charge ROC (positron = positive)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return auc


def plot_vs_pt(pt, scores, labels, out_path, n_bins=8, bin_mode="quantile",
               show_counts=True):
    pred = (scores >= 0.5).astype(int)             # 1 = predict positron
    correct = (pred == labels).astype(int)
    confidence = np.maximum(scores, 1.0 - scores)  # model prob for its chosen class

    edges = quantile_edges(pt, n_bins) if bin_mode == "quantile" else log_edges(pt, n_bins)
    centers, acc, conf, counts = [], [], [], []
    acc_err = []
    for i in range(len(edges) - 1):
        hi_edge = edges[i + 1]
        m = (pt >= edges[i]) & (pt <= hi_edge if i == len(edges) - 2 else pt < hi_edge)
        n = int(m.sum())
        if n == 0:
            continue
        centers.append(np.sqrt(edges[i] * edges[i + 1]))   # geometric center
        acc.append(float(correct[m].mean()))
        a = acc[-1]
        acc_err.append(1.96 * np.sqrt(a * (1 - a) / n))   # 95% Wald
        conf.append(float(confidence[m].mean()))
        counts.append(n)

    centers = np.asarray(centers)

    fig, ax = plt.subplots(figsize=(6.4, 5))
    ax.errorbar(centers, acc, yerr=acc_err, fmt="o-", capsize=3, lw=1.5,
                label="accuracy")
    ax.plot(centers, conf, "s--", color="tab:orange", lw=1.5, label="mean confidence")
    ax.axhline(0.5, ls="--", color="grey", lw=1, label="chance")

    if show_counts:
        ax2 = ax.twinx()
        width = np.diff(np.concatenate([[centers[0] * 0.8], centers])) * 0.6
        ax2.bar(centers, counts, width=width, alpha=0.12, color="grey", zorder=0)
        ax2.set_ylabel("electrons per bin", color="grey")
        ax2.tick_params(axis="y", labelcolor="grey")
        ax2.set_ylim(0, max(counts) * 4)
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)

    ax.set_xscale("log")
    ax.set_xlabel("true pT [GeV]")
    ax.set_ylabel("accuracy / mean confidence")    # NOT "probability"
    ax.set_ylim(0.45, 1.0)
    ax.set_title("Calo-only charge identification vs pT")
    ax.legend(loc="upper right", framealpha=0.9)

    fig.text(0.01, -0.02,
             "confidence = mean of max(p, 1-p), the model's probability for its "
             "chosen class; gap above accuracy = overconfidence (miscalibration).",
             fontsize=7.5, ha="left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"{'pT center':>10} {'n':>7} {'acc':>7} {'conf':>7}")
    for c, n, a, cf in zip(centers, counts, acc, conf):
        print(f"{c:10.2f} {n:7d} {a:7.3f} {cf:7.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help=".npz/.parquet/.csv with charge predictions")
    ap.add_argument("--charge-col", default="truth_charge")
    ap.add_argument("--pt-col", default="truth_pt")
    ap.add_argument("--score-col", default="charge_score")
    ap.add_argument("--logit-col", default="charge_logit")
    ap.add_argument("--out-dir", default="results/charge")
    ap.add_argument("--n-bins", type=int, default=8)
    ap.add_argument("--bins", choices=["quantile", "log"], default="quantile")
    ap.add_argument("--no-counts", action="store_true",
                    help="also hide the grey per-bin count bars")
    args = ap.parse_args()

    q, pt, score = load_arrays(args.input, args.charge_col, args.pt_col,
                               args.score_col, args.logit_col)
    labels = (q > 0).astype(int)   # positron (+1) = positive class
    print(f"{labels.size} electrons | positrons={int(labels.sum())} "
          f"electrons={int((1 - labels).sum())}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    auc = plot_roc(score, labels, out / "charge_roc.png")
    print(f"AUC = {auc:.4f}  ->  {out/'charge_roc.png'}")
    plot_vs_pt(pt, score, labels, out / "charge_id_vs_pt.png",
               n_bins=args.n_bins, bin_mode=args.bins,
               show_counts=not args.no_counts)
    print(f"wrote {out/'charge_id_vs_pt.png'}")


if __name__ == "__main__":
    main()