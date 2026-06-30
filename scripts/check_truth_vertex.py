"""Empirical test: is truth upstream of the magnetic deflection (i.e. at the
production vertex) or at the calorimeter face?

Method (pure geometry, NO model involved):
  - For each electron, compute the energy-weighted phi centroid of its shower
    (this is the phi anchor).
  - delta_phi = wrap(centroid_phi - truth_phi), split by truth charge.
      * truth upstream of the bend  -> the two charges separate (opposite signs)
      * truth at the calo face      -> both centered on zero
  - Control: the same split in eta must NOT separate by charge (the solenoid
    bends in phi, not eta). If eta separates, the test is broken.

Run:
    python scripts/check_truth_is_vertex.py /path/to/electrons_dbscan.parquet
    python scripts/check_truth_is_vertex.py electrons.parquet --max 20000 --out truth_check.png
"""
import argparse
import numpy as np
import polars as pl
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def wrap_pi(x: np.ndarray) -> np.ndarray:
    """Wrap angle differences into (-pi, pi]."""
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def energy_weighted_circular_mean_phi(phi: np.ndarray, w: np.ndarray) -> float:
    """Circular (wrap-safe) energy-weighted mean of phi."""
    s = np.sum(w * np.sin(phi))
    c = np.sum(w * np.cos(phi))
    return float(np.arctan2(s, c))


def energy_weighted_mean(vals: np.ndarray, w: np.ndarray) -> float:
    wsum = float(np.sum(w))
    return float(np.sum(w * vals) / wsum) if wsum > 0 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet", help="per-electron parquet (e.g. electrons_dbscan.parquet)")
    ap.add_argument("--max", type=int, default=20000, help="max electrons to use")
    ap.add_argument("--energy-col", default="cell_e_calibrated",
                    help="per-cell energy column for weighting")
    ap.add_argument("--out", default="truth_location_check.png")
    args = ap.parse_args()

    needed = ["cell_phi", "cell_eta", args.energy_col,
              "truth_phi", "truth_eta", "truth_charge"]
    df = pl.read_parquet(args.parquet, columns=needed)
    if args.max and df.height > args.max:
        df = df.head(args.max)
    print(f"Loaded {df.height} electrons from {args.parquet}")

    dphi_q_minus, dphi_q_plus = [], []   # q = -1 (electron), q = +1 (positron)
    deta_q_minus, deta_q_plus = [], []

    for row in df.iter_rows(named=True):
        phi = np.asarray(row["cell_phi"], dtype=np.float64)
        eta = np.asarray(row["cell_eta"], dtype=np.float64)
        w = np.asarray(row[args.energy_col], dtype=np.float64)
        if phi.size == 0 or np.sum(w) <= 0:
            continue

        phi_c = energy_weighted_circular_mean_phi(phi, w)
        eta_c = energy_weighted_mean(eta, w)

        dphi = wrap_pi(np.array([phi_c - float(row["truth_phi"])]))[0]
        deta = eta_c - float(row["truth_eta"])
        q = int(round(float(row["truth_charge"])))

        if q < 0:
            dphi_q_minus.append(dphi); deta_q_minus.append(deta)
        else:
            dphi_q_plus.append(dphi); deta_q_plus.append(deta)

    dphi_q_minus = np.asarray(dphi_q_minus)
    dphi_q_plus = np.asarray(dphi_q_plus)
    deta_q_minus = np.asarray(deta_q_minus)
    deta_q_plus = np.asarray(deta_q_plus)

    def med(a):
        return float(np.median(a)) if a.size else float("nan")

    print("\n=== phi:  centroid_phi - truth_phi  (the test) ===")
    print(f"  electron (q=-1): median = {med(dphi_q_minus):+.5f} rad   n={dphi_q_minus.size}")
    print(f"  positron (q=+1): median = {med(dphi_q_plus):+.5f} rad   n={dphi_q_plus.size}")
    sep = med(dphi_q_minus) - med(dphi_q_plus)
    print(f"  charge separation (e- median minus e+ median) = {sep:+.5f} rad")

    print("\n=== eta:  centroid_eta - truth_eta  (control: should NOT separate) ===")
    print(f"  electron (q=-1): median = {med(deta_q_minus):+.5f}")
    print(f"  positron (q=+1): median = {med(deta_q_plus):+.5f}")

    print("\n--- verdict ---")
    if abs(sep) > 0.003:   # ~few mrad; well above numerical noise
        if abs(med(deta_q_minus) - med(deta_q_plus)) < abs(sep) / 3:
            print("  phi separates by charge, eta does NOT -> truth is UPSTREAM of the")
            print("  deflection (production vertex). Charge/phi story is valid.")
        else:
            print("  phi AND eta separate by charge -> test is confounded; investigate.")
    else:
        print("  phi does NOT separate by charge -> truth appears to be at/after the")
        print("  deflection (e.g. calo face), OR the field effect is negligible here.")

    # ---- figure ----
    fig, (axp, axe) = plt.subplots(1, 2, figsize=(11, 4))
    rng = (-0.1, 0.1)
    axp.hist(dphi_q_minus, bins=80, range=rng, alpha=0.6, label="electron (q=-1)")
    axp.hist(dphi_q_plus, bins=80, range=rng, alpha=0.6, label="positron (q=+1)")
    axp.axvline(0, color="k", lw=0.8, ls="--")
    axp.set_xlabel("centroid phi - truth phi [rad]")
    axp.set_ylabel("electrons")
    axp.set_title("phi: split by charge (expect separation)")
    axp.legend()

    rng_e = (-0.1, 0.1)
    axe.hist(deta_q_minus, bins=80, range=rng_e, alpha=0.6, label="electron (q=-1)")
    axe.hist(deta_q_plus, bins=80, range=rng_e, alpha=0.6, label="positron (q=+1)")
    axe.axvline(0, color="k", lw=0.8, ls="--")
    axe.set_xlabel("centroid eta - truth eta")
    axe.set_title("eta: control (expect NO separation)")
    axe.legend()

    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()