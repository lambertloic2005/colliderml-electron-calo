#!/usr/bin/env python3
"""Resolution estimation for regression residuals (scalar and angular).

Workflow per parameter:
    fit = gaussian_resolution(pred - truth, wrap=<True for phi/theta>)
    fit.sigma          -> resolution
    fit.mu             -> bias
    fit.tail_fraction  -> mis-reconstruction rate (non-Gaussian outliers)

The 3-sigma-truncated mean/std IS the maximum-likelihood Gaussian fit on the
core, so it matches a ROOT / curve_fit Gaussian over the same truncated range
while avoiding binning bias. Report sigma as the resolution and the tail
fraction separately, the way the thesis does ("outliers remain small").

For phi (or any angle) you MUST wrap the residual first: a single prediction
that lands across the +/-pi seam otherwise contributes ~2*pi and wrecks both the
bias and the first truncation iteration.
"""

from dataclasses import dataclass
import numpy as np


def wrap_angle(x):
    """Fold values into (-pi, pi]. Apply to angle residuals before measuring."""
    x = np.asarray(x, dtype=float)
    return np.arctan2(np.sin(x), np.cos(x))


@dataclass
class FitResult:
    mu: float          # fitted mean -> bias
    sigma: float       # fitted std  -> resolution
    n_total: int
    n_core: int
    core: np.ndarray   # inlier residuals after truncation (for plotting)

    @property
    def tail_fraction(self) -> float:
        return 1.0 - self.n_core / self.n_total if self.n_total else float("nan")

    def __repr__(self) -> str:
        return (f"FitResult(bias={self.mu:.4g}, resolution={self.sigma:.4g}, "
                f"tail={self.tail_fraction:.2%}, n={self.n_core}/{self.n_total})")


def gaussian_resolution(residuals, n_sigma: float = 3.0,
                        max_iter: int = 100, wrap: bool = False) -> FitResult:
    """Iteratively drop points beyond n_sigma, then report the core Gaussian.

    Parameters
    ----------
    residuals : array-like   (prediction - truth)
    n_sigma   : truncation half-width in std units (thesis uses 3)
    wrap      : set True for phi/theta residuals (folds into (-pi, pi] first)
    """
    r = np.asarray(residuals, dtype=float)
    r = r[np.isfinite(r)]
    if wrap:
        r = wrap_angle(r)
    n_total = r.size
    core = r
    for _ in range(max_iter):
        mu, sd = core.mean(), core.std()
        if sd == 0:
            break
        keep = np.abs(core - mu) <= n_sigma * sd
        if keep.all():
            break
        core = core[keep]
    return FitResult(mu=float(core.mean()), sigma=float(core.std()),
                     n_total=int(n_total), n_core=int(core.size), core=core)


def plot_residual_fit(residuals, name: str = "phi", unit: str = "rad",
                      wrap: bool = False, bins: int = 60, ax=None,
                      n_sigma: float = 3.0):
    """Histogram of residuals with the fitted-Gaussian overlay annotated.

    Returns (FitResult, matplotlib axis).
    """
    import matplotlib.pyplot as plt

    fit = gaussian_resolution(residuals, n_sigma=n_sigma, wrap=wrap)
    r = wrap_angle(residuals) if wrap else np.asarray(residuals, dtype=float)
    r = r[np.isfinite(r)]

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    _, edges, _ = ax.hist(r, bins=bins, alpha=0.8)
    width = edges[1] - edges[0]
    xs = np.linspace(edges[0], edges[-1], 400)
    # Gaussian scaled to the core counts (n_core * bin_width)
    g = (fit.n_core * width / (fit.sigma * np.sqrt(2 * np.pi))
         * np.exp(-0.5 * ((xs - fit.mu) / fit.sigma) ** 2))
    ax.plot(xs, g, "--", lw=2,
            label=(f"Gaussian fit\nbias = {fit.mu:.3g} {unit}\n"
                   f"$\\sigma$ = {fit.sigma:.3g} {unit}\n"
                   f"tail = {fit.tail_fraction:.1%}"))
    ax.set_title(f"Residuals: {name}")
    ax.set_xlabel(f"Prediction - truth for {name} [{unit}]")
    ax.set_ylabel("Count")
    ax.legend(loc="upper right", fontsize=9)
    return fit, ax


if __name__ == "__main__":
    # quick self-test
    rng = np.random.default_rng(0)
    core = rng.normal(0.05, 0.30, 95_000)
    outliers = rng.uniform(-3, 3, 5_000)
    print("eta-like:", gaussian_resolution(np.concatenate([core, outliers])))

    phi_t = (rng.random(20_000) * 2 - 1) * np.pi
    phi_p = phi_t + rng.normal(0, 0.4, phi_t.size)
    print("phi (wrapped):", gaussian_resolution(phi_p - phi_t, wrap=True))
    print("phi (no wrap, WRONG):", gaussian_resolution(phi_p - phi_t, wrap=False))