"""Build a per-cluster training table from truth-free DBSCAN clustering.

For each event:
  1. load ALL calo cells (observable only)              -> io.all_event_cells
  2. group them into clusters, drop outliers            -> cluster_event_cells
  3. build one record per cluster (observable features)
  4. attach truth labels by matching clusters to prompt electrons (truth used
     ONLY for labelling, never for selecting cells)     -> match_clusters_to_electrons

The output parquet is schema-compatible with `dataset.ElectronDataset`, so the
existing training/eval scripts work on it unchanged. By default only clusters
matched to a truth electron are written (they carry regression targets);
`keep_unmatched=True` also writes background clusters with null targets and
is_electron=0 for future classification work.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import polars as pl

from .io import load_frames, get_event, prompt_electrons, all_event_cells  # noqa: F401
from .coords import xyz_to_eta_phi, momentum_to_eta_phi
from .calibration import calibrate
from .cluster import (
    cluster_event_cells,
    cluster_centroid,
    iter_clusters,
    match_clusters_to_electrons,
)
from .pipeline import truth_kinematics, _shard_index


# Null kinematics for unmatched (background) clusters.
_NULL_TRUTH = {
    "truth_energy": None, "truth_px": None, "truth_py": None, "truth_pz": None,
    "truth_p": None, "truth_pt": None, "truth_log_pt": None,
    "truth_eta": None, "truth_phi": None, "truth_charge": None,
    "truth_z0": None,
}


def build_cluster_row(
    event_id: int,
    cluster_id: int,
    cells: dict[str, np.ndarray],
    members: np.ndarray,
    electron: dict | None,
    match_dR: float | None,
) -> dict:
    """One record for a single DBSCAN cluster (observable features + labels)."""
    x = cells["x"][members]
    y = cells["y"][members]
    z = cells["z"][members]
    e_total = cells["e_total"][members]
    det = cells["detector"][members]

    eta_c, phi_c = xyz_to_eta_phi(x, y, z)
    e_total_cal = calibrate(e_total, x, y, z)

    clu_eta, clu_phi = cluster_centroid(eta_c, phi_c, e_total_cal)

    if electron is not None:
        labels = truth_kinematics(electron)
        is_electron = 1
    else:
        labels = dict(_NULL_TRUTH)
        is_electron = 0

    return {
        "event_id": int(event_id),
        "cluster_id": int(cluster_id),
        "is_electron": is_electron,
        "match_dR": float(match_dR) if match_dR is not None else None,

        # labels (truth) -- null for background clusters
        **labels,

        # cluster-level observables (handy for diagnostics / cuts)
        "cluster_eta": float(clu_eta),
        "cluster_phi": float(clu_phi),
        "cluster_energy": float(e_total_cal.sum()),

        # per-cell observable features (names match ElectronDataset)
        "n_cells": int(members.size),
        "cell_x": x.astype(np.float32).tolist(),
        "cell_y": y.astype(np.float32).tolist(),
        "cell_z": z.astype(np.float32).tolist(),
        "cell_eta": eta_c.astype(np.float32).tolist(),
        "cell_phi": phi_c.astype(np.float32).tolist(),
        "cell_e_total": e_total.astype(np.float32).tolist(),
        "cell_e_calibrated": e_total_cal.astype(np.float32).tolist(),
        "cell_detector": det.astype(np.int32).tolist(),
    }


def build_cluster_table(
    channel: str = "zee",
    pileup: str = "pu200",
    max_events: int | None = None,
    eps: float = 0.05,
    min_samples: int = 4,
    e_thresh_gev: float = 0.1,
    energy_weighted: bool = False,
    dR_match: float = 0.1,
    keep_unmatched: bool = False,
    out_path: str | Path = "data/clusters/clusters.parquet",
    shard_min: int | None = None,
    shard_max: int | None = None,
    task_id: int = 0,
    n_tasks: int = 1,
) -> pl.DataFrame:
    """Truth-free DBSCAN cluster table over a shard range, with striding.

    Shard discovery honours COLLIDERML_DATA_DIR (same rule as the patched
    pipeline.py); rows for THIS worker's shard slice are held in memory and
    written once at the end -- bounded because each worker in a chunked build
    only sees CHUNK_PAIRS / n_tasks shards.
    """
    env = os.environ.get("COLLIDERML_DATA_DIR")
    base_root = Path(env).expanduser() if env else Path.home() / ".cache" / "colliderml"
    base = str(base_root / "CERN__ColliderML-Release-1")
    p_pat = f"{base}/{channel}_{pileup}_particles/data/{channel}_{pileup}_particles/train-*.parquet"
    c_pat = f"{base}/{channel}_{pileup}_calo_hits/data/{channel}_{pileup}_calo_hits/train-*.parquet"

    p_by_idx = {_shard_index(p): p for p in glob.glob(p_pat)}
    c_by_idx = {_shard_index(p): p for p in glob.glob(c_pat)}
    common = sorted(set(p_by_idx) & set(c_by_idx))
    if shard_min is not None:
        common = [i for i in common if i >= shard_min]
    if shard_max is not None:
        common = [i for i in common if i <= shard_max]
    # Empty BEFORE striding = the data genuinely isn't on disk -> hard failure.
    if not common:
        raise RuntimeError(
            f"no matched shards in range [{shard_min},{shard_max}] under {base} "
            f"({len(p_by_idx)} particles / {len(c_by_idx)} calo_hits shards found)"
        )
    common = common[task_id::n_tasks]
    # Empty AFTER striding = this worker just drew no shards. Benign, and what
    # pipeline.py does. Raising here would abort a whole chunk over load balance.
    if not common:
        print(f"task {task_id}/{n_tasks}: no shards assigned in range "
              f"[{shard_min},{shard_max}]; exiting without writing.")
        return pl.DataFrame()
    print(f"task {task_id}/{n_tasks}: {len(common)} shard pairs "
          f"(range [{shard_min},{shard_max}]) under {base}")

    rows: list[dict] = []
    events_done = 0
    n_clusters_total = n_matched_total = 0
    n_electrons_total = n_electrons_matched = 0
    n_align_skipped = 0

    for shard_pos, idx in enumerate(common):
        if max_events is not None and events_done >= max_events:
            break

        p_df = pl.read_parquet(p_by_idx[idx])
        c_df = pl.read_parquet(c_by_idx[idx])

        # --- alignment guard: join calo rows to particle rows BY event_id, never
        # by position. Closes the known integrity gap inherited from pipeline.py.
        c_index = {int(e): j for j, e in enumerate(c_df["event_id"].to_list())}

        n = p_df.height
        if max_events is not None and events_done + n > max_events:
            n = max_events - events_done

        for i in range(n):
            p_row = p_df.row(i, named=True)
            event_id = int(p_row["event_id"])
            j = c_index.get(event_id)
            if j is None:
                n_align_skipped += 1
                continue
            c_row = c_df.row(j, named=True)

            cells = all_event_cells(c_row)
            labels = cluster_event_cells(
                cells, eps=eps, min_samples=min_samples,
                e_thresh_gev=e_thresh_gev, energy_weighted=energy_weighted,
            )
            clusters = list(iter_clusters(labels))
            n_clusters_total += len(clusters)

            electrons, seen = [], set()
            for e in prompt_electrons(p_row):
                key = (round(float(e["px"]), 6), round(float(e["py"]), 6),
                       round(float(e["pz"]), 6), int(e["pdg_id"]))
                if key in seen:
                    continue
                seen.add(key)
                electrons.append(e)
            n_electrons_total += len(electrons)

            centroids = [
                cluster_centroid(
                    *xyz_to_eta_phi(cells["x"][m], cells["y"][m], cells["z"][m]),
                    calibrate(cells["e_total"][m], cells["x"][m], cells["y"][m], cells["z"][m]),
                )
                for _, m in clusters
            ]
            e_etaphi = [
                tuple(float(v) for v in momentum_to_eta_phi(e["px"], e["py"], e["pz"]))
                for e in electrons
            ]
            assign = match_clusters_to_electrons(centroids, e_etaphi, dR_max=dR_match)
            n_matched_total += len(assign)
            n_electrons_matched += len({el for el, _ in assign.values()})

            for ci, (cid, members) in enumerate(clusters):
                if ci in assign:
                    el_idx, dR = assign[ci]
                    rows.append(build_cluster_row(
                        event_id, cid, cells, members, electrons[el_idx], dR))
                elif keep_unmatched:
                    rows.append(build_cluster_row(
                        event_id, cid, cells, members, None, None))
            events_done += 1

        del p_df, c_df
        print(f"  shard {shard_pos + 1}/{len(common)} (idx {idx}): "
              f"{events_done} events, {n_clusters_total} clusters, "
              f"{n_matched_total} matched, {n_align_skipped} align-skipped")

    df = pl.DataFrame(rows)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if df.height == 0:
        print(f"\nNo clusters produced for this task; not writing {out_path}.")
        return df
    df.write_parquet(out_path)

    eff = (n_electrons_matched / n_electrons_total) if n_electrons_total else 0.0
    print(
        f"\nWrote {df.height} cluster records from {events_done} events to {out_path}\n"
        f"  clusters found      : {n_clusters_total}\n"
        f"  clusters matched    : {n_matched_total}\n"
        f"  truth electrons     : {n_electrons_total}\n"
        f"  electrons recovered : {n_electrons_matched} ({eff:.1%} matching efficiency)\n"
        f"  events skipped (event_id not in calo shard): {n_align_skipped}"
    )
    return df


if __name__ == "__main__":
    build_cluster_table(max_events=50)