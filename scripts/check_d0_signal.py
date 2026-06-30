# scripts/check_d0_signal.py  — does truth d0 have any recoverable spread?
import numpy as np, polars as pl
from colliderml_electron.io import prompt_electrons  # reuses your particle reader

# If vx/vy are in the raw particle parquet, read a sample and look at the spread.
# Quick version if you only have momentum + a vertex parquet with vx,vy,vz,phi:
df = pl.read_parquet("data/electrons/eta_phi_pt_z0_charge/zee_pu200_z0_charge.parquet",
                     columns=["truth_phi", "truth_z0"])  # add truth_vx, truth_vy if present
print("z0 spread (mm):", float(df["truth_z0"].std()))   # expect ~56
# If you can build d0:  d0 = vy*cos(phi) - vx*sin(phi)
# print("d0 spread (mm):", float(d0.std()))
# Decision rule: if std(d0) is well below ~5 mm (the cell size), the calorimeter
# cannot beat a "predict 0" prior -> there is nothing to learn. Stop here.