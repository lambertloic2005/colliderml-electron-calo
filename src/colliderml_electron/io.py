"""Loaders for ColliderML calorimeter data, focused on prompt electrons."""
from __future__ import annotations
from itertools import chain
import numpy as np
import polars as pl
from colliderml.core import load_tables, collect_tables


def load_frames(
    channel: str = "zee",
    pileup: str = "pu200",
    max_events: int = 50,
) -> dict[str, pl.DataFrame]:
    """Load `particles` and `calo_hits` for a channel via the colliderml library."""
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


def descendant_pids(particles_row: dict, root_pid: int) -> set[int]:
    """{root_pid} union all particles descended from it via parent_id."""
    pids = np.asarray(particles_row["particle_id"], dtype=np.int64)
    parents = np.asarray(particles_row["parent_id"], dtype=np.int64)
    children: dict[int, list[int]] = {}
    for pid, parent in zip(pids.tolist(), parents.tolist()):
        children.setdefault(int(parent), []).append(int(pid))
    family: set[int] = set()
    stack = [int(root_pid)]
    while stack:
        cur = stack.pop()
        if cur in family:
            continue
        family.add(cur)
        stack.extend(children.get(cur, []))
    return family


def cells_for_particle_set(calo_row: dict, pid_set: set[int]) -> dict[str, np.ndarray]:
    """Vectorised per-cell selection: cells contributed to by any pid in pid_set.

    Returns keys: x, y, z (mm), detector, e_total, e_from_e (sum of pid_set
    contributions in the cell), t_from_e (energy-weighted contribution time).
    """
    x = np.asarray(calo_row["x"], dtype=np.float32)
    y = np.asarray(calo_row["y"], dtype=np.float32)
    z = np.asarray(calo_row["z"], dtype=np.float32)
    det = np.asarray(calo_row["detector"], dtype=np.int32)
    e_tot = np.asarray(calo_row["total_energy"], dtype=np.float32)
    n_cells = len(x)

    empty_f = np.empty(0, dtype=np.float32)
    empty_i = np.empty(0, dtype=np.int32)
    empty_out = {"x": empty_f, "y": empty_f, "z": empty_f, "detector": empty_i,
                 "e_total": empty_f, "e_from_e": empty_f, "t_from_e": empty_f}
    if n_cells == 0 or not pid_set:
        return empty_out

    pid_arr = np.fromiter(pid_set, dtype=np.int64, count=len(pid_set))
    pids_lol = calo_row["contrib_particle_ids"]
    es_lol = calo_row["contrib_energies"]
    ts_lol = calo_row["contrib_times"]

    # Flatten variable-length nested lists into one long array per field.
    n_per_cell = np.fromiter((len(p) for p in pids_lol),
                             dtype=np.int64, count=n_cells)
    total = int(n_per_cell.sum())
    if total == 0:
        return empty_out

    cell_idx = np.repeat(np.arange(n_cells, dtype=np.int64), n_per_cell)
    all_pids = np.fromiter(chain.from_iterable(pids_lol), dtype=np.int64, count=total)
    all_es = np.fromiter(chain.from_iterable(es_lol), dtype=np.float32, count=total)
    all_ts = np.fromiter(chain.from_iterable(ts_lol), dtype=np.float32, count=total)

    hit = np.isin(all_pids, pid_arr)
    if not hit.any():
        return empty_out
    hit_cells = cell_idx[hit]
    hit_es = all_es[hit]
    hit_ts = all_ts[hit]

    e_per_cell = np.bincount(hit_cells, weights=hit_es, minlength=n_cells)
    te_per_cell = np.bincount(hit_cells, weights=hit_es * hit_ts, minlength=n_cells)
    keep = np.where(e_per_cell > 0)[0]

    return {
        "x": x[keep], "y": y[keep], "z": z[keep],
        "detector": det[keep],
        "e_total": e_tot[keep],
        "e_from_e": e_per_cell[keep].astype(np.float32),
        "t_from_e": (te_per_cell[keep] / np.maximum(e_per_cell[keep], 1e-30)).astype(np.float32),
    }


def cells_for_electron(calo_row: dict, electron_pid: int) -> dict[str, np.ndarray]:
    """Direct electron contributions only (no descendants)."""
    return cells_for_particle_set(calo_row, {int(electron_pid)})


def cells_for_electron_full(
    particles_row: dict, calo_row: dict, electron_pid: int
) -> dict[str, np.ndarray]:
    """Electron + all descendants via parent_id — the full EM shower."""
    family = descendant_pids(particles_row, electron_pid)
    return cells_for_particle_set(calo_row, family)

def energy_containment_mask(cells, electron, containment = 0.98):
    from colliderml_electron.coords import (
        xyz_to_eta_phi,
        momentum_to_eta_phi,
        delta_eta_phi,
    )
    from colliderml_electron.calibration import calibrate

    eta_e, phi_e = momentum_to_eta_phi(
        electron["px"], electron["py"], electron["pz"]
    )
    eta_c, phi_c = xyz_to_eta_phi(
        cells["x"], cells["y"], cells["z"]
    )
    
    deta, dphi = delta_eta_phi(
        eta_c, phi_c, float(eta_e), float(phi_e)
    )
    dR = np.sqrt(deta**2 + dphi**2)

    e_cal = calibrate(cells["e_from_e"], cells["x"], cells["y"], cells["z"])

    order = np.argsort(dR)
    cumulative = np.cumsum(e_cal[order])
    total = cumulative[-1]

    cutoff_index = np.searchsorted(cumulative, containment*total)
    R_cut = dR[order[cutoff_index]]

    return dR <= R_cut

def dR_max_mask(cells, electron, dR_max: float = 0.1):
    from colliderml_electron.coords import (
        xyz_to_eta_phi,
        momentum_to_eta_phi,
        delta_eta_phi,
    )

    eta_e, phi_e = momentum_to_eta_phi(
        electron["px"], electron["py"], electron["pz"]
    )
    eta_c, phi_c = xyz_to_eta_phi(
        cells["x"], cells["y"], cells["z"]
    )
    
    deta, dphi = delta_eta_phi(
        eta_c, phi_c, float(eta_e), float(phi_e)
    )
    dR = np.sqrt(deta**2 + dphi**2)

    return dR <= dR_max


if __name__ == "__main__":
    print("Loading 5 events of zee_pu200...")
    frames = load_frames(channel="zee", pileup="pu200", max_events=5)
    p_row, c_row = get_event(frames, 0)
    electrons = prompt_electrons(p_row)
    if electrons:
        e = electrons[0]
        cells_d = cells_for_electron(c_row, e["particle_id"])
        cells_f = cells_for_electron_full(p_row, c_row, e["particle_id"])
        print(f"electron pid={e['particle_id']}  E={e['energy']:.2f} GeV")
        print(f"  direct      : {len(cells_d['x']):5d} cells, "
              f"sum(e_from_e)={cells_d['e_from_e'].sum():.4f}")
        print(f"  + descendants: {len(cells_f['x']):5d} cells, "
              f"sum(e_from_e)={cells_f['e_from_e'].sum():.4f}")
        

