"""Sanity-check a per-electron parquet before scaling up."""
import sys
import numpy as np
import polars as pl


def sanity_check(parquet_path: str) -> None:
    print(f"\n=== {parquet_path} ===\n")
    df = pl.read_parquet(parquet_path)
    n = df.height
    print(f"Total electrons:         {n}")
    if n == 0:
        print("EMPTY FILE — every electron was skipped. Pipeline broken.")
        return

    # --- Event-level ---
    n_events = df["event_id"].n_unique()
    print(f"Unique events:           {n_events}")
    print(f"Electrons per event:     {n / n_events:.2f}  (expect ~2 for zee)")

    # --- Truth kinematics ---
    print("\n--- Truth kinematics ---")
    for col in ["truth_energy", "truth_pt", "truth_eta", "truth_phi"]:
        a = df[col].to_numpy()
        print(f"  {col:14s} min={a.min():+9.3f}  max={a.max():+9.3f}  mean={a.mean():+9.3f}")
    pos = int((df["truth_charge"].to_numpy() == +1).sum())
    neg = int((df["truth_charge"].to_numpy() == -1).sum())
    print(f"  charge balance:  +1 → {pos},  -1 → {neg}  (expect roughly equal)")

    # --- Cell counts ---
    n_cells = df["n_cells"].to_numpy()
    print("\n--- Cells per electron ---")
    print(f"  min={n_cells.min()}  max={n_cells.max()}  "
          f"mean={n_cells.mean():.1f}  median={int(np.median(n_cells))}")

    # --- ΔR cut verification ---
    print("\n--- ΔR cut verification ---")
    all_dR = np.concatenate(df["cell_dR_truth"].to_list())
    print(f"  cell ΔR: min={all_dR.min():.4f}  max={all_dR.max():.4f}")
    if all_dR.max() > 0.1 + 1e-6:
        print("  WARNING: cells with ΔR > 0.1 leaked through — bug in dR_max_mask")
    else:
        print("  OK: every cell has ΔR ≤ 0.1")

    # --- Energy containment (sanity of cell-electron matching + calibration) ---
    print("\n--- Energy containment ---")
    cont = []
    for i in range(n):
        r = df.row(i, named=True)
        e_e = np.asarray(r["cell_e_from_e_cal"])
        if r["truth_energy"] > 0:
            cont.append(e_e.sum() / r["truth_energy"])
    cont = np.array(cont)
    print(f"  sum(cell_e_from_e_cal) / truth_energy")
    print(f"    median={np.median(cont):.3f}  mean={cont.mean():.3f}")
    print(f"    10th/90th pct = [{np.percentile(cont, 10):.3f}, {np.percentile(cont, 90):.3f}]")

    # --- Observable cell features ---
    print("\n--- Observable cell features ---")
    e_t = np.concatenate(df["cell_e_total"].to_list())
    e_c = np.concatenate(df["cell_e_calibrated"].to_list())
    print(f"  cell_e_total       min={e_t.min():.4g}  max={e_t.max():.4g}")
    print(f"  cell_e_calibrated  min={e_c.min():.4g}  max={e_c.max():.4g}")
    if (e_t <= 0).any() or (e_c <= 0).any():
        print("  WARNING: non-positive cell energies present")

    # --- Detector codes present ---
    print("\n--- Detector subsystem codes ---")
    codes = np.unique(np.concatenate(df["cell_detector"].to_list()))
    print(f"  codes: {codes.tolist()}")
    print(f"  → paste this list into DETECTOR_CODES in dataset.py")

    print("\n=== END REPORT ===\n")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/electrons/smoke.parquet"
    sanity_check(path)