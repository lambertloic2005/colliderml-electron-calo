"""
Analytic time-of-flight (ToF) ceiling for z0 — no model involved.

For each electron, the cells satisfy  t_i ~= t0 + d_i / c , where
d_i = sqrt(r_i^2 + (z_i - z_v)^2) is the path from the production vertex
(0, 0, z_v) to the cell, t0 is the collision time, and c = 299.792458 mm/ns.

We recover z_v two ways and compare to truth_z0:
  * FAIR  : t0 profiled out (uses only relative cell-time differences ->
            calorimeter-only, no collision-time reference needed).
  * IDEAL : t0 fixed to 0 (assumes a perfect collision-time reference;
            only meaningful if the simulation put the hard-scatter vertex
            at t0 ~= 0 -- the fitted-t0 print below tells you).

Read it like this:
  - FAIR  >> PRIOR  -> timing carries little calorimeter-only vertex info
                       (expected for compact showers; the power is locked
                        behind t0).
  - IDEAL << FAIR   -> the information exists but needs a collision-time
                       reference the calorimeter alone cannot supply.
  - regional split  -> timing should help most at high |eta| (forward),
                       opposite to pointing.

Run from repo root:  python scripts/diag_tof_z0.py
"""

import numpy as np
import polars as pl
from pathlib import Path

C_MM_PER_NS = 299.792458
PARQUET = Path("data/electrons/eta_phi_pt_z0_charge/zee_pu200_z0_charge.parquet")
EWEIGHT = "cell_e_from_e_cal"   # fall back to "cell_e_calibrated" if absent
ZV_GRID = np.linspace(-400.0, 400.0, 401)   # mm, 2 mm spacing (z0 scale ~50 mm)


def fit_one(r, z, t, w):
    """Return (z_v_fair, z_v_ideal, t0_fair) via vectorized 1-D grid search."""
    dz = z[:, None] - ZV_GRID[None, :]                 # (N, G)
    d = np.sqrt(r[:, None] ** 2 + dz ** 2)             # (N, G)
    resid = t[:, None] - d / C_MM_PER_NS               # (N, G)  = t_i - d_i/c
    W = w / w.sum()
    mu = (W[:, None] * resid).sum(0)                   # (G,) weighted mean over cells
    obj_fair = (W[:, None] * (resid - mu[None, :]) ** 2).sum(0)   # t0 profiled out
    obj_ideal = (W[:, None] * resid ** 2).sum(0)                 # t0 fixed = 0
    i_fair, i_ideal = int(obj_fair.argmin()), int(obj_ideal.argmin())
    return ZV_GRID[i_fair], ZV_GRID[i_ideal], float(mu[i_fair])


def main():
    cols = ["split", "cell_x", "cell_y", "cell_z", "cell_t_from_e",
            EWEIGHT, "truth_z0", "truth_eta"]
    df = pl.read_parquet(PARQUET, columns=cols).filter(pl.col("split") == "test")

    z_fair, z_ideal, t0_fair, z_true, eta_true = [], [], [], [], []
    for row in df.iter_rows(named=True):
        x = np.asarray(row["cell_x"], float)
        y = np.asarray(row["cell_y"], float)
        z = np.asarray(row["cell_z"], float)
        t = np.asarray(row["cell_t_from_e"], float)
        w = np.asarray(row[EWEIGHT], float)
        if len(z) < 3 or not np.isfinite(w).all() or w.sum() <= 0:
            continue
        r = np.hypot(x, y)
        zf, zi, t0 = fit_one(r, z, t, w)
        z_fair.append(zf); z_ideal.append(zi); t0_fair.append(t0)
        z_true.append(float(row["truth_z0"])); eta_true.append(float(row["truth_eta"]))

    z_fair = np.array(z_fair); z_ideal = np.array(z_ideal)
    t0_fair = np.array(t0_fair); z_true = np.array(z_true); eta_true = np.array(eta_true)
    rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))

    print(f"\nN electrons = {len(z_true)}")
    print(f"PRIOR  (predict mean z0)        : {np.std(z_true):7.1f} mm")
    print(f"FAIR   (ToF, t0 profiled out)   : {rmse(z_fair,  z_true):7.1f} mm")
    print(f"IDEAL  (ToF, t0 fixed = 0)      : {rmse(z_ideal, z_true):7.1f} mm")
    print(f"fitted t0 (fair): mean={t0_fair.mean():.3f} ns  std={t0_fair.std():.3f} ns "
          f"-> if mean~0 & std small, the sim's collision time is ~fixed, so IDEAL "
          f"leans on truth t0")
    print("\nby |eta| region (rmse vs truth, mm):")
    for lo, hi, lab in [(0.0, 1.2, "barrel"), (1.2, 2.5, "endcap"), (2.5, 9, "fwd")]:
        m = (np.abs(eta_true) >= lo) & (np.abs(eta_true) < hi)
        if m.any():
            print(f"  |eta| [{lo},{hi}) {lab:7s} n={int(m.sum()):5d}  "
                  f"prior={np.std(z_true[m]):6.1f}  "
                  f"fair={rmse(z_fair[m], z_true[m]):6.1f}  "
                  f"ideal={rmse(z_ideal[m], z_true[m]):6.1f}")


if __name__ == "__main__":
    main()