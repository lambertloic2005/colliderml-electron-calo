"Build per-electron training data"

from pathlib import Path
import numpy as np
import polars as pl
import glob
import os

from .io import (
    load_frames,
    get_event,
    prompt_electrons,
    descendant_pids,
    cells_for_particle_set,
    dR_max_mask,
    dbscan_keep_mask,
)
from .coords import (
    xyz_to_eta_phi,
    momentum_to_eta_phi,
    delta_eta_phi
)

from .calibration import calibrate


def truth_kinematics(electron: dict) -> dict:
    px, py, pz = electron["px"], electron["py"], electron["pz"]
    p = float(np.sqrt(px**2 + py**2 + pz**2))
    pt = float(np.sqrt(px**2 + py**2))
    eta, phi = momentum_to_eta_phi(px, py, pz)
    charge = -int(np.sign(electron["pdg_id"])) # pdg_id = +11 -> electron, =-11 -> positron
    return {
        "truth_energy": float(electron["energy"]),
        "truth_px": float(px),
        "truth_py": float(py),
        "truth_pz": float(pz),
        "truth_p": p,
        "truth_pt": pt,
        "truth_log_pt": float(np.log(pt)),
        "truth_eta": float(eta),
        "truth_phi": float(phi),
        "truth_charge": charge,
    }

def build_electron_row(
    particles_row: dict,
    calo_row: dict,
    electron: dict,
    dR_max: float = 0.1,
    mask_kind: str = "dbscan",         # "cone" or "dbscan"
    eps: float = 0.08,
    min_samples: int = 2,
) -> dict | None:
    # # --- early-rejection cuts on the electron's own kinematics ---
    # px, py, pz = electron["px"], electron["py"], electron["pz"]
    # p  = np.sqrt(px*px + py*py + pz*pz)
    # pt = np.sqrt(px*px + py*py)
    # eta = np.arctanh(pz / p) if p > 0 else 999.0
    # if abs(eta) > eta_max or pt < pt_min:
    #     return None
    
    # Find the full shower family by tracking particle id
    family = descendant_pids(particles_row, electron["particle_id"])
    cells = cells_for_particle_set(calo_row, family)
    if len(cells["x"]) == 0:
        return None
    
    if mask_kind == "dbscan":
        keep = dbscan_keep_mask(cells, electron, eps=eps, min_samples=min_samples)
    else:
        keep = dR_max_mask(cells, electron, dR_max)
    if not keep.any():
        return None
    
    x = cells["x"][keep]
    y = cells["y"][keep]
    z = cells["z"][keep]
    e_total = cells["e_total"][keep]
    e_from_e = cells["e_from_e"][keep]
    t = cells["t_from_e"][keep]
    det = cells["detector"][keep]

    # observable derived quantities
    eta_c, phi_c = xyz_to_eta_phi(x, y, z)
    e_total_cal = calibrate(e_total, x, y, z)

    # truth derived quantities
    eta_e, phi_e = momentum_to_eta_phi(
        electron["px"],
        electron["py"],
        electron["pz"],
    )
    deta_truth, dphi_truth = delta_eta_phi(
        eta_c,
        phi_c,
        float(eta_e),
        float(phi_e),
    )
    dR_truth = np.sqrt(deta_truth**2 + dphi_truth**2)
    e_from_e_cal = calibrate(e_from_e, x, y, z)
    
    return {
        "event_id": int(particles_row["event_id"]),
        "particle_id": int(electron["particle_id"]),

        # labels
        **truth_kinematics(electron),

        # observable per cell features
        "n_cells": int(keep.sum()),
        "cell_x": x.astype(np.float32).tolist(),
        "cell_y": y.astype(np.float32).tolist(),
        "cell_z": z.astype(np.float32).tolist(),
        "cell_eta": eta_c.astype(np.float32).tolist(),
        "cell_phi": phi_c.astype(np.float32).tolist(),
        "cell_e_total": e_total.astype(np.float32).tolist(),
        "cell_e_calibrated": e_total_cal.astype(np.float32).tolist(),
        "cell_detector": det.astype(np.int32).tolist(),

        # truth derived per cell features
        "cell_t_from_e": t.astype(np.float32).tolist(),
        "cell_e_from_e": e_from_e.astype(np.float32).tolist(),
        "cell_e_from_e_cal": e_from_e_cal.astype(np.float32).tolist(),
        "cell_dR_truth": dR_truth.astype(np.float32).tolist(),
    }

def _shard_index(path: str) -> int:
    """Extract integer index from a 'train-NNNNN-of-NNNNN.parquet' filename."""
    name = os.path.basename(path)
    return int(name.split("-")[1])


def build_electron_table(
    channel: str = "zee",
    pileup: str = "pu200",
    max_events: int | None = None,
    dR_max: float = 0.1,
    mask_kind: str = "dbscan",        # add
    eps: float = 0.08,              # add
    min_samples: int = 2,           # add
    out_path: str | Path = "data/electrons/electrons.parquet",
) -> pl.DataFrame:
    home = os.path.expanduser("~")
    base = f"{home}/.cache/colliderml/CERN__ColliderML-Release-1"
    p_pat = f"{base}/{channel}_{pileup}_particles/data/{channel}_{pileup}_particles/train-*.parquet"
    c_pat = f"{base}/{channel}_{pileup}_calo_hits/data/{channel}_{pileup}_calo_hits/train-*.parquet"

    p_by_idx = {_shard_index(p): p for p in glob.glob(p_pat)}
    c_by_idx = {_shard_index(p): p for p in glob.glob(c_pat)}
    common = sorted(set(p_by_idx) & set(c_by_idx))

    if not common:
        raise RuntimeError(f"no matched shards. Found {len(p_by_idx)} particles, {len(c_by_idx)} calo_hits.")

    print(
        f"Found {len(common)} matched shards "
        f"(of {len(p_by_idx)} particles / {len(c_by_idx)} calo_hits total)"
    )

    rows: list[dict] = []
    skipped = 0
    duplicates = 0
    events_done = 0

    for shard_pos, idx in enumerate(common):
        if max_events is not None and events_done >= max_events:
            break

        p_df = pl.read_parquet(p_by_idx[idx])
        c_df = pl.read_parquet(c_by_idx[idx])

        n = p_df.height
        if max_events is not None and events_done + n > max_events:
            n = max_events - events_done

        for i in range(n):
            p_row = p_df.row(i, named=True)
            c_row = c_df.row(i, named=True)
            seen: set[tuple] = set()
            for e in prompt_electrons(p_row):
                key = (
                    round(float(e["px"]), 6),
                    round(float(e["py"]), 6),
                    round(float(e["pz"]), 6),
                    int(e["pdg_id"]),
                )
                if key in seen:
                    duplicates += 1
                    continue
                seen.add(key)
                row = build_electron_row(
                    particles_row=p_row,
                    calo_row=c_row,
                    electron=e,
                    dR_max=dR_max,
                    mask_kind=mask_kind,        # add
                    eps=eps,                    # add
                    min_samples=min_samples,    # add
                )
                if row is None:
                    skipped += 1
                    continue
                rows.append(row)
            events_done += 1

        del p_df, c_df
        print(
            f"  shard {shard_pos + 1}/{len(common)} (idx {idx}): "
            f"{events_done} events processed, {len(rows)} electrons kept"
        )

    df = pl.DataFrame(rows)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path)
    print(
        f"\nWrote {len(df)} electrons from {events_done} events "
        f"({skipped} skipped by cuts, {duplicates} duplicates collapsed) "
        f"to {out_path}"
    )
    return df

if __name__ == "__main__":
    build_electron_table()

    

