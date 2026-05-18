"""Loaders for ColliderML calorimeter data, focused on prompt electrons."""
from __future__ import annotations
import numpy as np
import polars as pl
from colliderml.core import load_tables, collect_tables


def load_frames(
    channel: str = "zee",
    pileup: str = "pu200",
    max_events: int = 50,
) -> dict[str, pl.DataFrame]:
    """Load `particles` and `calo_hits` for a channel via the colliderml library.

    Returns dict with keys 'particles' and 'calo_hits'. Each is a Polars
    DataFrame with one row per event and list columns for per-object fields.
    """
    cfg = {
        "dataset_id": "CERN/ColliderML-Release-1",
        "channels": channel,
        "pileup": pileup,
        "objects": ["particles", "calo_hits"],
        "split": "train",
        "lazy": False,
        "max_events": max_events,
    }
    return collect_tables(load_tables(cfg))


def get_event(frames: dict[str, pl.DataFrame], idx: int) -> tuple[dict, dict]:
    """Return (particles_row, calo_row) for the idx-th event, aligned by event_id."""
    particles_row = frames["particles"].row(idx, named=True)
    calo_row = frames["calo_hits"].row(idx, named=True)
    if particles_row["event_id"] != calo_row["event_id"]:
        raise ValueError(
            f"event_id mismatch at idx {idx}: "
            f"particles={particles_row['event_id']} calo={calo_row['event_id']}"
        )
    return particles_row, calo_row


def prompt_electrons(particles_row: dict) -> list[dict]:
    """Prompt electrons: pdg_id == +/-11, primary == True, vertex_primary == 1."""
    pdg = np.asarray(particles_row["pdg_id"])
    primary = np.asarray(particles_row["primary"])
    vp = np.asarray(particles_row["vertex_primary"])
    pids = np.asarray(particles_row["particle_id"])
    px = np.asarray(particles_row["px"])
    py = np.asarray(particles_row["py"])
    pz = np.asarray(particles_row["pz"])
    energy = np.asarray(particles_row["energy"])
    mask = (np.abs(pdg) == 11) & primary & (vp == 1)
    return [
        {
            "particle_id": int(pids[i]),
            "pdg_id": int(pdg[i]),
            "px": float(px[i]),
            "py": float(py[i]),
            "pz": float(pz[i]),
            "energy": float(energy[i]),
        }
        for i in np.where(mask)[0]
    ]


def cells_for_electron(calo_row: dict, electron_pid: int) -> dict[str, np.ndarray]:
    """Per-cell arrays for cells the given electron contributed to.

    Keys: x, y, z (mm), detector (int), e_total (cell total energy),
    e_from_e (this electron's share of the cell), t_from_e (energy-weighted
    time of its contributions).
    """
    x = np.asarray(calo_row["x"], dtype=np.float32)
    y = np.asarray(calo_row["y"], dtype=np.float32)
    z = np.asarray(calo_row["z"], dtype=np.float32)
    det = np.asarray(calo_row["detector"], dtype=np.int32)
    e_tot = np.asarray(calo_row["total_energy"], dtype=np.float32)

    keep: list[int] = []
    e_from_e: list[float] = []
    t_from_e: list[float] = []
    for i, (pids, es, ts) in enumerate(zip(
        calo_row["contrib_particle_ids"],
        calo_row["contrib_energies"],
        calo_row["contrib_times"],
    )):
        pids = np.asarray(pids)
        hit = np.where(pids == electron_pid)[0]
        if hit.size:
            es_arr = np.asarray(es)[hit]
            ts_arr = np.asarray(ts)[hit]
            esum = float(es_arr.sum())
            keep.append(i)
            e_from_e.append(esum)
            t_from_e.append(
                float((ts_arr * es_arr).sum() / esum) if esum > 0 else float("nan")
            )

    k = np.asarray(keep, dtype=np.int64)
    return {
        "x": x[k],
        "y": y[k],
        "z": z[k],
        "detector": det[k],
        "e_total": e_tot[k],
        "e_from_e": np.asarray(e_from_e, dtype=np.float32),
        "t_from_e": np.asarray(t_from_e, dtype=np.float32),
    }


if __name__ == "__main__":
    print("Loading 5 events of zee_pu200 (first run downloads data)...")
    frames = load_frames(channel="zee", pileup="pu200", max_events=5)
    print(f"  particles: {frames['particles'].shape}")
    print(f"  calo_hits: {frames['calo_hits'].shape}")

    p_row, c_row = get_event(frames, 0)
    print(f"\nEvent 0: event_id={p_row['event_id']}, n_calo_cells={len(c_row['x'])}")

    electrons = prompt_electrons(p_row)
    print(f"  prompt electrons: {len(electrons)}")
    for k, e in enumerate(electrons):
        pt = np.hypot(e["px"], e["py"])
        print(f"    [{k}] pid={e['particle_id']} pdg={e['pdg_id']} "
              f"E={e['energy']:.2f} GeV  pT={pt:.2f} GeV")

    if electrons:
        cells = cells_for_electron(c_row, electrons[0]["particle_id"])
        n = len(cells["x"])
        print(f"\n  Electron 0: {n} cells, "
              f"sum(e_from_e)={cells['e_from_e'].sum():.3f}, "
              f"sum(e_total)={cells['e_total'].sum():.3f}")
