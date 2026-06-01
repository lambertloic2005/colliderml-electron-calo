"""Truth-free clustering of calorimeter cells with DBSCAN.

The functions here take ONLY observable quantities (cell positions and
energies) and group nearby cells into clusters, flagging sparse cells as
outliers (DBSCAN label -1). No MC truth is used at any point.

Truth is only consulted afterwards, in `match_clusters_to_electrons`, to
attach regression labels to the clusters that correspond to real electrons.
That keeps selection (which cells belong together) fully decoupled from the
labels, which is the whole point of this dataset variant.

Clustering is done in (eta, phi) space, embedded as [eta, cos phi, sin phi]
so that:
  * the phi wraparound at +/-pi is handled automatically, and
  * sklearn can use a KD-tree (fast, low memory) instead of a precomputed
    N x N distance matrix -- important because a full pu200 event has many
    thousands of cells.
For small angular separations the Euclidean distance in this 3D embedding
approximates the usual dR = sqrt(deta^2 + dphi^2), so `eps` keeps its
intuitive "cone size in (eta, phi)" meaning.
"""
from __future__ import annotations

import numpy as np

from .coords import xyz_to_eta_phi
from .calibration import calibrate


def _phi_embedding(eta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """[eta, cos phi, sin phi] -- wrap-safe feature space for DBSCAN."""
    return np.column_stack(
        [np.asarray(eta, float), np.cos(phi), np.sin(phi)]
    ).astype(np.float64)


def cluster_event_cells(
    cells: dict[str, np.ndarray],
    eps: float = 0.05,
    min_samples: int = 4,
    e_thresh_gev: float = 0.1,
    energy_weighted: bool = False,
) -> np.ndarray:
    """Assign a DBSCAN cluster label to every cell of an event.

    Parameters
    ----------
    cells
        Output of `io.all_event_cells` (keys: x, y, z, detector, e_total).
    eps
        Neighbourhood radius in (eta, phi) units. Smaller -> more, tighter
        clusters; larger -> fewer, looser clusters that may merge pileup.
    min_samples
        Core-point threshold. With `energy_weighted=False` this is a plain
        cell count; with `energy_weighted=True` it is a *sum of calibrated
        energies* (GeV) in the neighbourhood (energy-density coring).
    e_thresh_gev
        Cells below this calibrated energy are treated as outliers up front
        (never clustered, labelled -1). This is the main pileup suppressor
        and is fully observable. Set to 0.0 to disable.
    energy_weighted
        If True, pass calibrated cell energy as DBSCAN sample_weight so dense
        energetic regions are favoured as cluster cores.

    Returns
    -------
    labels : (n_cells,) int array
        Cluster id per cell; -1 means outlier/noise (removed downstream).
        Cells below `e_thresh_gev` are also labelled -1.
    """
    from sklearn.cluster import DBSCAN

    n = len(cells["x"])
    labels = np.full(n, -1, dtype=int)
    if n == 0:
        return labels

    e_cal = calibrate(cells["e_total"], cells["x"], cells["y"], cells["z"])
    keep = e_cal > e_thresh_gev
    idx = np.where(keep)[0]
    if idx.size == 0:
        return labels

    eta, phi = xyz_to_eta_phi(cells["x"][idx], cells["y"][idx], cells["z"][idx])
    X = _phi_embedding(eta, phi)

    sample_weight = e_cal[idx] if energy_weighted else None
    sub = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(
        X, sample_weight=sample_weight
    )
    labels[idx] = sub
    return labels


def cluster_centroid(
    eta: np.ndarray, phi: np.ndarray, e: np.ndarray
) -> tuple[float, float]:
    """Energy-weighted (eta, phi) centroid; phi via circular mean (wrap-safe)."""
    w = np.clip(np.asarray(e, float), 1e-9, None)
    eta_c = float(np.average(np.asarray(eta, float), weights=w))
    phi_c = float(
        np.arctan2(np.average(np.sin(phi), weights=w),
                   np.average(np.cos(phi), weights=w))
    )
    return eta_c, phi_c


def iter_clusters(labels: np.ndarray):
    """Yield (cluster_id, member_index_array) for every non-noise cluster."""
    for cid in np.unique(labels):
        if cid == -1:
            continue
        yield int(cid), np.where(labels == cid)[0]


def match_clusters_to_electrons(
    cluster_centroids: list[tuple[float, float]],
    electron_etaphi: list[tuple[float, float]],
    dR_max: float = 0.1,
) -> dict[int, tuple[int, float]]:
    """Match each truth electron to its nearest cluster (truth used ONLY here).

    Returns {cluster_index: (electron_index, dR)}. Each electron claims its
    closest cluster within `dR_max`; if two electrons want the same cluster
    the closer one wins. Clusters with no electron stay unmatched (background).
    """
    assign: dict[int, tuple[int, float]] = {}
    for j, (te, tp) in enumerate(electron_etaphi):
        best, best_dR = -1, np.inf
        for i, (ce, cp) in enumerate(cluster_centroids):
            dphi = (cp - tp + np.pi) % (2 * np.pi) - np.pi
            dR = float(np.hypot(ce - te, dphi))
            if dR < best_dR:
                best, best_dR = i, dR
        if best >= 0 and best_dR <= dR_max:
            if best not in assign or assign[best][1] > best_dR:
                assign[best] = (j, best_dR)
    return assign