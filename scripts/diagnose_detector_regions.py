"""Characterize each detector subsystem code geometrically (barrel vs endcap)
and locate the |eta| at which the ECAL hands off from barrel to endcap.

Run once on a sample of the training parquet to set BARREL_ETA_MAX principledly.
"""
from pathlib import Path
import numpy as np
import polars as pl

PARQUET = Path("data/electrons/eta_phi_pt_z0_charge/zee_pu200_z0_charge.parquet")
N_EVENTS = 4000  # a sample is plenty for geometry

df = (
    pl.read_parquet(PARQUET, columns=["cell_x", "cell_y", "cell_z", "cell_detector"])
      .head(N_EVENTS)
)

# explode the per-cell list columns into a flat (cell-level) table
flat = df.explode(["cell_x", "cell_y", "cell_z", "cell_detector"])
x = flat["cell_x"].to_numpy(); y = flat["cell_y"].to_numpy()
z = flat["cell_z"].to_numpy(); det = flat["cell_detector"].to_numpy()

r = np.hypot(x, y)
theta = np.arctan2(r, z)
eta = -np.log(np.tan(np.clip(theta, 1e-6, np.pi - 1e-6) / 2.0))

print(f"{'code':>4} {'n':>8} {'med_r':>8} {'iqr_r':>8} "
      f"{'med|z|':>8} {'iqr|z|':>8}  shape   |eta| range")
for code in sorted(np.unique(det)):
    m = det == code
    rr, zz = r[m], np.abs(z[m])
    iqr = lambda a: float(np.percentile(a, 75) - np.percentile(a, 25))
    # barrel: r tightly constrained, z spread out. endcap: the reverse.
    shape = "barrel" if iqr(rr) < iqr(zz) else "endcap"
    ee = np.abs(eta[m])
    print(f"{code:>4} {m.sum():>8d} {np.median(rr):>8.0f} {iqr(rr):>8.0f} "
          f"{np.median(zz):>8.0f} {iqr(zz):>8.0f}  {shape:>6}  "
          f"[{ee.min():.2f}, {ee.max():.2f}]")

# If you can identify the ECAL barrel code (innermost barrel-shaped subsystem)
# and the ECAL endcap code, the transition is where their |eta| coverage meets:
print("\nSet BARREL_ETA_MAX to the |eta| where the ECAL barrel coverage ends "
      "and the ECAL endcap begins (read it off the two rows above).")