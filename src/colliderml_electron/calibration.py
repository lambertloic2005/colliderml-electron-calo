"""Calibration factors from Elitez et al. (ColliderML release paper, Table 4).

Multipliers convert raw cell deposits (the sampling-fraction energy stored in
calo_hits.total_energy) to calibrated GeV.
"""
from __future__ import annotations
import numpy as np

ECAL_BARREL = 37.5
ECAL_ENDCAP = 38.7
HCAL_BARREL = 45.0
HCAL_ENDCAP = 46.9

# Approximate geometric thresholds. The tracker outer edge is z=+/-3030 mm,
# so anything past it is endcap. ECAL/HCAL split is rougher; we use
# r=1700 mm in the barrel and |z|=3500 mm in the endcap as defaults.
BARREL_ENDCAP_Z = 3030.0
ECAL_HCAL_R_BARREL = 1700.0
ECAL_HCAL_Z_ENDCAP = 3500.0


def approximate_factor(x, y, z):
    """Per-cell calibration factor from cell position. Vectorised."""
    x = np.asarray(x); y = np.asarray(y); z = np.asarray(z)
    r = np.hypot(x, y)
    abs_z = np.abs(z)
    is_endcap = abs_z > BARREL_ENDCAP_Z
    is_hcal = np.where(is_endcap, abs_z > ECAL_HCAL_Z_ENDCAP, r > ECAL_HCAL_R_BARREL)
    return np.where(
        is_endcap,
        np.where(is_hcal, HCAL_ENDCAP, ECAL_ENDCAP),
        np.where(is_hcal, HCAL_BARREL, ECAL_BARREL),
    )


def calibrate(e_raw, x, y, z):
    """Apply approximate per-cell calibration to raw cell energies."""
    return np.asarray(e_raw) * approximate_factor(x, y, z)
