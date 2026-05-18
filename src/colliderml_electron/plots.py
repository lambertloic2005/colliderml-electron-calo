"""Exploratory plots for ColliderML calorimeter data."""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt


def detector_geometry(
    frames,
    n_events: int | None = None,
    max_cells_per_event: int = 5000,
    seed: int = 0,
):
    """Scatter r=sqrt(x^2+y^2) vs z for calo hits, colored by detector code.

    Subsamples up to `max_cells_per_event` hits per event so matplotlib
    stays responsive at pu200 (hundreds of thousands of cells per event).
    Returns the matplotlib Figure.
    """
    rng = np.random.default_rng(seed)
    n_events = n_events or frames["calo_hits"].shape[0]

    z_all, r_all, det_all = [], [], []
    for i in range(n_events):
        c = frames["calo_hits"].row(i, named=True)
        x = np.asarray(c["x"]); y = np.asarray(c["y"]); z = np.asarray(c["z"])
        det = np.asarray(c["detector"])
        if len(x) > max_cells_per_event:
            idx = rng.choice(len(x), size=max_cells_per_event, replace=False)
            x, y, z, det = x[idx], y[idx], z[idx], det[idx]
        z_all.append(z); r_all.append(np.hypot(x, y)); det_all.append(det)

    z = np.concatenate(z_all)
    r = np.concatenate(r_all)
    det = np.concatenate(det_all)

    fig, ax = plt.subplots(figsize=(11, 6))
    codes = np.unique(det)
    cmap = plt.colormaps.get_cmap("tab10")
    for k, code in enumerate(codes):
        mask = det == code
        ax.scatter(z[mask], r[mask], s=0.4, alpha=0.35,
                   color=cmap(k % 10),
                   label=f"detector={code} (n={mask.sum()})")
    ax.set_xlabel("z [mm]")
    ax.set_ylabel(r"r = $\sqrt{x^2+y^2}$ [mm]")
    ax.set_title(f"Calo cell geometry — {n_events} events "
                 f"(<= {max_cells_per_event} cells/event subsampled)")
    ax.legend(markerscale=12, fontsize=9, loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig
