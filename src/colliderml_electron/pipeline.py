"Build per-electron training data"

from pathlib import Path
import numpy as np
import polars as pl

from .io import (
    load_frames,
    get_event,
    prompt_electrons,
    descendant_pids,
    cells_for_particle_set,
    dR_max_mask,
)
from .coords import (
    xyz_to_eta_phi,
    momentum_to_eta_phi,
    delta_eta_phi
)

from .calibration import calibrate


def truth_kinematics(electron: dict) -> dict:
    px, py, pz = electron["px"], electron["py"], electron["pz"]
    p = float(np.sqrt(px**2 + py**2 + px**2))
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
        "truth_eta": float(eta),
        "truth_phi": float(phi),
        "truth_charge": charge,
    }

def build_electron_row(
    particles_row: dict,
    calo_row: dict,
    electron: dict,
    dR_max: float = 0.1,
) -> dict | None:
    

