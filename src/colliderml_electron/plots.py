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


def cell_energy_spectrum(
    frames,
    n_events: int | None = None,
    bins: int = 80,
):
    """Histogram of all cell energies (log10) across n_events.

    Useful for spotting the threshold cut-off and the dynamic range.
    Returns the matplotlib Figure.
    """
    n_events = n_events or frames["calo_hits"].shape[0]
    e_all = []
    for i in range(n_events):
        c = frames["calo_hits"].row(i, named=True)
        e_all.append(np.asarray(c["total_energy"]))
    e = np.concatenate(e_all)
    e = e[e > 0]
    log_e = np.log10(e)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(log_e, bins=bins, color="steelblue",
            edgecolor="white", linewidth=0.3)
    ax.set_xlabel(r"$\log_{10}$(cell energy)  [uncalibrated]")
    ax.set_ylabel("cells")
    ax.set_yscale("log")
    ax.set_title(f"Cell-energy spectrum — {n_events} events, {len(e):,} cells")

    pct = np.percentile(log_e, [1, 50, 99])
    ymax = ax.get_ylim()[1]
    for p, lbl in zip(pct, ["1%", "50%", "99%"]):
        ax.axvline(p, color="crimson", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.text(p, ymax * 0.5, f"  {lbl}", rotation=90,
                va="top", fontsize=8, color="crimson")

    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig


def cells_per_electron(
    frames,
    n_events: int | None = None,
):
    """Two-panel diagnostic: histogram of cells per prompt electron,
    and a scatter of cells vs the electron's truth energy.
    Returns the matplotlib Figure.
    """
    from .io import get_event, prompt_electrons, cells_for_electron
    n_events = n_events or frames["calo_hits"].shape[0]

    n_cells, energies = [], []
    for i in range(n_events):
        p_row, c_row = get_event(frames, i)
        for e in prompt_electrons(p_row):
            cells = cells_for_electron(c_row, e["particle_id"])
            n_cells.append(len(cells["x"]))
            energies.append(e["energy"])
    n_cells = np.asarray(n_cells)
    energies = np.asarray(energies)

    fig, (ax_h, ax_s) = plt.subplots(1, 2, figsize=(12, 5))

    ax_h.hist(n_cells, bins=30, color="seagreen",
              edgecolor="white", linewidth=0.3)
    ax_h.set_xlabel("cells per electron")
    ax_h.set_ylabel("electrons")
    ax_h.set_title(f"Cells per prompt electron — {len(n_cells)} electrons")
    ax_h.grid(True, alpha=0.2)

    ax_s.scatter(energies, n_cells, alpha=0.6, s=20, color="seagreen")
    ax_s.set_xlabel("electron truth energy [GeV]")
    ax_s.set_ylabel("cells per electron")
    ax_s.set_title("Cells vs. electron energy")
    ax_s.grid(True, alpha=0.2)

    fig.tight_layout()
    return fig
