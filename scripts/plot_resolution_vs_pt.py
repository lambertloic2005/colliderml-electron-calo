"""Resolution vs pT for eta, phi, and pT  (David's requested plots).

For each quantity it bins electrons in true pT and reports the CORE resolution
(3-sigma-truncated Gaussian sigma) per bin -- the SAME definition used for the
single-number resolutions on the results slides (via gaussian_resolution).

If anchor columns are supplied, it also overlays the anchor-only resolution
(energy-weighted centroid for eta/phi, sum E_T for pT). That overlay answers
"does the network actually improve on the simple barycenter/Sum-ET baseline?"

The pT panel is plotted as relative resolution sigma(pT)/pT, which for electrons
is essentially sigma_E/E -- i.e. David's "sigma E vs pT".

Input file (.npz/.parquet/.csv) per-electron columns:
  required: truth_pt, truth_eta, truth_phi, pred_eta, pred_phi, pred_pt
  optional (baseline overlay): eta_anchor, phi_anchor, pt_anchor

Run:
    python scripts/plot_resolution_vs_pt.py preds.npz --out-dir results/resolution
    python scripts/plot_resolution_vs_pt.py preds.npz --bins log --n-bins 10
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from colliderml_electron.resolution import gaussian_resolution
    _HAVE_REPO_RES = True
except Exception:                      # standalone fallback (e.g. local testing)
    _HAVE_REPO_RES = False

    def _wrap(r):
        return (r + np.pi) % (2 * np.pi) - np.pi

    class _Fit:
        def __init__(self, sigma, n_core, n_total):
            self.sigma = sigma
            self.n_core = n_core
            self.n_total = n_total

    def gaussian_resolution(residuals, n_sigma=3.0, max_iter=100, wrap=False):
        r = np.asarray(residuals, float)
        r = r[np.isfinite(r)]
        n_total = r.size
        if wrap:
            r = _wrap(r)
        core = r
        for _ in range(max_iter):
            mu, sd = core.mean(), core.std()
            if sd == 0:
                break
            keep = np.abs(core - mu) <= n_sigma * sd
            if keep.all():
                break
            core = core[keep]
        sigma = float(core.std()) if core.size else float("nan")
        return _Fit(sigma, int(core.size), int(n_total))


def load(path, cols):
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".npz":
        d = np.load(path)
        return {c: (np.asarray(d[c]) if c in d.files else None) for c in cols}
    import polars as pl
    d = pl.read_parquet(path) if ext == ".parquet" else pl.read_csv(path)
    return {c: (d[c].to_numpy() if c in d.columns else None) for c in cols}


def quantile_edges(pt, n_bins):
    return np.unique(np.quantile(pt, np.linspace(0, 1, n_bins + 1)))


def log_edges(pt, n_bins):
    return np.logspace(np.log10(max(pt.min(), 1e-3)), np.log10(pt.max()), n_bins + 1)


def sigma_in_bins(pt, residual, edges, wrap, min_count=50):
    """Core resolution per pT bin, with a minimum-count guard.

    Returns (idx, centers, sigma, sigma_err, counts) over SURVIVING bins only.
    `idx` is the index into `edges` of each surviving bin, so the caller can
    tell which bins were dropped and avoid drawing a connecting line across the
    gap (the sparse-/empty-bin interpolation artifact David flagged).

    A bin is kept only if it holds at least `min_count` electrons AND the core
    fit returns a finite, positive sigma. sigma_err is the standard error on a
    Gaussian sigma, sigma / sqrt(2 * n_core).
    """
    idx, centers, sig, err, counts = [], [], [], [], []
    for i in range(len(edges) - 1):
        hi = edges[i + 1]
        m = (pt >= edges[i]) & (pt <= hi if i == len(edges) - 2 else pt < hi)
        n = int(m.sum())
        if n < min_count:                 # not enough electrons for a stable fit
            continue
        fit = gaussian_resolution(residual[m], wrap=wrap)
        if not np.isfinite(fit.sigma) or fit.sigma <= 0:
            continue
        n_core = int(getattr(fit, "n_core", n))
        idx.append(i)
        centers.append(np.sqrt(edges[i] * hi))
        sig.append(fit.sigma)
        err.append(fit.sigma / np.sqrt(2 * max(n_core, 1)))
        counts.append(n)
    return (np.asarray(idx, dtype=int), np.asarray(centers),
            np.asarray(sig), np.asarray(err), counts)


def _contiguous_runs(idx):
    """Yield slices [a:b] over positions where edge-indices `idx` are consecutive."""
    if len(idx) == 0:
        return
    start = 0
    for k in range(1, len(idx)):
        if idx[k] != idx[k - 1] + 1:
            yield slice(start, k)
            start = k
    yield slice(start, len(idx))


def plot_resolution_series(ax, idx, centers, sig, err, *, color=None,
                           marker="o", linestyle="-", label=None):
    """Plot sigma vs pT with error bars, connecting ONLY adjacent bins.

    Every surviving bin gets an error-bar marker; the connecting line is broken
    wherever a bin was dropped by the min-count guard, so the eye never
    interpolates a resolution across a gap that holds no (trustworthy) data.
    """
    ax.errorbar(centers, sig, yerr=err, fmt=marker, color=color,
                ms=4, capsize=2, lw=0, elinewidth=1, label=label)
    for sl in _contiguous_runs(idx):
        if sl.stop - sl.start >= 2:
            ax.plot(centers[sl], sig[sl], linestyle=linestyle, color=color, lw=1.5)


def panel(ax, pt, res_model, res_base, edges, wrap, ylabel, title, min_count=50):
    idx, c, s, e, n = sigma_in_bins(pt, res_model, edges, wrap, min_count)
    plot_resolution_series(ax, idx, c, s, e, color="C0", marker="o",
                           linestyle="-", label="model")
    if res_base is not None:
        bidx, cb, sb, eb, _ = sigma_in_bins(pt, res_base, edges, wrap, min_count)
        plot_resolution_series(ax, bidx, cb, sb, eb, color="grey", marker="s",
                               linestyle="--", label="anchor baseline")
    ax.set_xscale("log")
    ax.set_xlabel("true pT [GeV]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(bottom=0)
    ax.legend()
    return c, s, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--out-dir", default="results/resolution")
    ap.add_argument("--n-bins", type=int, default=8)
    ap.add_argument("--bins", choices=["quantile", "log"], default="quantile")
    ap.add_argument("--min-count", type=int, default=50,
                    help="drop pT bins holding fewer than this many electrons; "
                         "the connecting line is broken at the resulting gaps so "
                         "no resolution is interpolated across sparse bins")
    ap.add_argument("--split-eta", type=float, default=None,
                    help="if set, plot barrel (|eta|<S) vs endcap (|eta|>=S) "
                         "instead of the anchor-baseline overlay")
    args = ap.parse_args()

    cols = ["truth_pt", "truth_eta", "truth_phi", "pred_eta", "pred_phi", "pred_pt",
            "eta_anchor", "phi_anchor", "pt_anchor"]
    d = load(args.input, cols)
    for req in ["truth_pt", "truth_eta", "truth_phi", "pred_eta", "pred_phi", "pred_pt"]:
        if d[req] is None:
            raise KeyError(f"missing required column '{req}' in {args.input}")

    pt = d["truth_pt"].astype(np.float64)
    res_eta = d["pred_eta"] - d["truth_eta"]
    res_phi = d["pred_phi"] - d["truth_phi"]                       # wrapped in fit
    res_ptrel = (d["pred_pt"] - d["truth_pt"]) / d["truth_pt"]

    has_base = all(d[k] is not None for k in ("eta_anchor", "phi_anchor", "pt_anchor"))
    base_eta = (d["eta_anchor"] - d["truth_eta"]) if has_base else None
    base_phi = (d["phi_anchor"] - d["truth_phi"]) if has_base else None
    base_ptrel = ((d["pt_anchor"] - d["truth_pt"]) / d["truth_pt"]) if has_base else None
    print(f"{pt.size} electrons | baseline overlay: {'yes' if has_base else 'no'}")

    edges = quantile_edges(pt, args.n_bins) if args.bins == "quantile" else log_edges(pt, args.n_bins)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    panels = [
        (res_eta, base_eta, False, r"core $\sigma_\eta$", r"$\eta$ resolution vs pT"),
        (res_phi, base_phi, True, r"core $\sigma_\phi$ [rad]", r"$\phi$ resolution vs pT"),
        (res_ptrel, base_ptrel, False, r"core $\sigma_{p_T}/p_T$",
         r"$p_T$ resolution vs pT"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))

    if args.split_eta is None:
        # original behaviour: model vs anchor baseline, all eta combined
        for ax, (res_m, res_b, wrap, ylab, title) in zip(axes, panels):
            panel(ax, pt, res_m, res_b, edges, wrap, ylab, title, args.min_count)
        fname = "resolution_vs_pt.png"
    else:
        # barrel vs endcap (model only), shared pT bins for comparability
        barrel = np.abs(d["truth_eta"]) < args.split_eta
        regions = [(f"barrel |eta|<{args.split_eta:g}", barrel),
                   (f"endcap |eta|>={args.split_eta:g}", ~barrel)]
        for ax, (res_m, _b, wrap, ylab, title) in zip(axes, panels):
            for ci, (label, sel) in enumerate(regions):
                idx, c, s, e, n = sigma_in_bins(pt[sel], res_m[sel], edges, wrap,
                                                args.min_count)
                if c.size:
                    plot_resolution_series(ax, idx, c, s, e, color=f"C{ci}",
                                           marker="o", linestyle="-",
                                           label=f"{label} (n={sum(n)})")
            ax.set_xscale("log")
            ax.set_xlabel("true pT [GeV]")
            ax.set_ylabel(ylab)
            ax.set_title(title)
            ax.set_ylim(bottom=0)
            ax.legend(fontsize=8)
        suptitle = "Core resolution vs pT - barrel vs endcap"
        fname = "resolution_vs_pt_barrel_endcap.png"

    fig.tight_layout()
    fig.savefig(out / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out/fname}")


if __name__ == "__main__":
    main()