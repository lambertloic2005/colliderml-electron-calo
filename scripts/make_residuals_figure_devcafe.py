"""
Residual-prediction concept figure (slide 11) — single simple plot.
Shower cells in the (eta, phi) plane, the energy-weighted centroid (the anchor),
the truth point, and the small residual Delta the model predicts.

To make it data-true, replace the synthetic eta/phi/E with one real electron.
Run:  python residuals_figure.py   ->   residuals_figure.pdf / .png
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

rng = np.random.default_rng(7)

# synthetic shower cells (replace with real per-cell arrays)
n = 40
eta = rng.normal(0.35, 0.045, n)
phi = rng.normal(1.20, 0.045, n)
E = rng.lognormal(0.0, 0.8, n)

# energy-weighted centroid == the anchor
eta_c = np.sum(E * eta) / np.sum(E)
phi_c = np.sum(E * phi) / np.sum(E)

# truth electron position, slightly offset from the centroid
eta_t, phi_t = eta_c + 0.06, phi_c + 0.05

plt.rcParams.update({"font.size": 13, "font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(7, 5.5))

ax.scatter(eta, phi, s=30 + 380 * (E / E.max()),
           c="#bfdbfe", edgecolors="#60a5fa", linewidths=1.2, zorder=2)

ax.scatter([eta_c], [phi_c], marker="D", s=130, c="#0f172a", zorder=4)
ax.annotate("anchor", (eta_c, phi_c), textcoords="offset points",
            xytext=(0, -22), ha="center", fontsize=13, color="#0f172a")

ax.scatter([eta_t], [phi_t], marker="*", s=340, c="#f59e0b",
           edgecolors="#b45309", linewidths=1.0, zorder=4)
ax.annotate("truth", (eta_t, phi_t), textcoords="offset points",
            xytext=(12, 8), fontsize=13, color="#b45309")

ax.add_patch(FancyArrowPatch((eta_c, phi_c), (eta_t, phi_t),
             arrowstyle="-|>", mutation_scale=20, lw=2.5,
             color="#b45309", zorder=5))


ax.set_xlabel(r"$\eta$")
ax.set_ylabel(r"$\phi$")
ax.margins(0.28)

fig.tight_layout()
outdir = os.path.dirname(os.path.abspath(__file__))
fig.savefig(os.path.join(outdir, "residuals_figure.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(outdir, "residuals_figure.png"), dpi=200,
            bbox_inches="tight")
print("wrote residuals_figure.pdf and residuals_figure.png")