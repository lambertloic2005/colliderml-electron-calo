"""Supervisor-ready version of the energy-starved cluster diagnostics.

Same analysis as diagnose_energy_starved.py, but with interpretable axes:
raw event counts on a log scale (every bar = real number of electrons),
population sizes in the legends, and annotated cut lines.

Model-free -- reads the test parquet directly.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

PARQUET = Path("data/electrons/testRuche/zee_pu200_supervised_dbscan_TEST.parquet")
OUTPUT_DIR = Path("results/diagnostics")
MAX_ABS_ETA = 3.5      # current training acceptance
FIDUCIAL_ETA = 3.2     # proposed tightened fiducial cut
RATIO_CUT = 0.5        # sum E_T / true pT below this = "energy-starved"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = (
        pl.read_parquet(PARQUET, columns=[
            "split", "truth_pt", "truth_eta", "n_cells",
            "cell_x", "cell_y", "cell_z", "cell_e_calibrated",
        ])
        .filter((pl.col("split") == "test")
                & (pl.col("truth_eta").abs() <= MAX_ABS_ETA))
    )

    ratios, pts, etas, ncells = [], [], [], []
    for row in df.iter_rows(named=True):
        x = np.asarray(row["cell_x"], dtype=np.float64)
        y = np.asarray(row["cell_y"], dtype=np.float64)
        z = np.asarray(row["cell_z"], dtype=np.float64)
        e = np.asarray(row["cell_e_calibrated"], dtype=np.float64)
        r3 = np.sqrt(x * x + y * y + z * z)
        sin_theta = np.hypot(x, y) / np.clip(r3, 1e-9, None)
        ratios.append(float((e * sin_theta).sum()) / max(row["truth_pt"], 1e-9))
        pts.append(row["truth_pt"])
        etas.append(abs(row["truth_eta"]))
        ncells.append(row["n_cells"])

    ratios = np.array(ratios); pts = np.array(pts)
    etas = np.array(etas); ncells = np.array(ncells)
    starved = ratios < RATIO_CUT
    n_tot, n_starved = len(ratios), int(starved.sum())
    frac = n_starved / n_tot

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.5))
    fig.suptitle(
        "Cluster energy completeness, ColliderML Z$\\to$ee pu200 "
        "(truth-supervised DBSCAN selection), test split, "
        f"$|\\eta^{{truth}}| \\leq {MAX_ABS_ETA}$",
        fontsize=11,
    )

    # ---- (a) completeness vs true pT --------------------------------------
    ax = axes[0, 0]
    ax.scatter(pts[~starved], ratios[~starved], s=6, alpha=0.35,
               label=f"healthy (n={n_tot - n_starved})")
    ax.scatter(pts[starved], ratios[starved], s=8, alpha=0.6, color="tab:orange",
               label=f"starved (n={n_starved}, {frac:.1%})")
    ax.axhline(RATIO_CUT, color="r", ls="--", lw=1,
               label=f"starved threshold ({RATIO_CUT})")
    ax.axhline(1.0, color="grey", ls=":", lw=1)
    ax.set_xlabel("true $p_T$ [GeV]")
    ax.set_ylabel("$\\Sigma E_T^{cells}\\,/\\,p_T^{truth}$")
    ax.set_ylim(0, 2)
    ax.set_title("(a) Energy completeness vs true $p_T$", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")

    # ---- (b) |eta|, raw counts, log y -------------------------------------
    ax = axes[0, 1]
    bins_eta = np.linspace(0, MAX_ABS_ETA, 36)
    ax.hist(etas[~starved], bins=bins_eta, alpha=0.6,
            label=f"healthy (n={n_tot - n_starved})")
    ax.hist(etas[starved], bins=bins_eta, alpha=0.7, color="tab:orange",
            label=f"starved (n={n_starved})")
    ax.axvline(FIDUCIAL_ETA, color="r", ls="--", lw=1,
               label=f"proposed fiducial cut ({FIDUCIAL_ETA})")
    ax.set_yscale("log")
    ax.set_xlabel("$|\\eta^{truth}|$")
    ax.set_ylabel("electrons / bin")
    ax.set_title("(b) Starved clusters vs pseudorapidity", fontsize=10)
    ax.legend(fontsize=8)

    # ---- (c) cluster size, raw counts, log-log ----------------------------
    ax = axes[1, 0]
    bins_n = np.logspace(0, np.log10(max(ncells.max(), 10)), 40)
    ax.hist(ncells[~starved], bins=bins_n, alpha=0.6,
            label=f"healthy (n={n_tot - n_starved})")
    ax.hist(ncells[starved], bins=bins_n, alpha=0.7, color="tab:orange",
            label=f"starved (n={n_starved})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$n_{cells}$ in selected cluster")
    ax.set_ylabel("electrons / bin")
    ax.set_title("(c) Selected-cluster size", fontsize=10)
    ax.legend(fontsize=8)

    # ---- (d) completeness distribution, log y -----------------------------
    ax = axes[1, 1]
    ax.hist(ratios, bins=np.linspace(0, 2, 80), color="tab:blue", alpha=0.8)
    ax.axvline(RATIO_CUT, color="r", ls="--", lw=1,
               label=f"starved threshold ({RATIO_CUT})")
    ax.axvline(1.0, color="grey", ls=":", lw=1, label="complete cluster")
    ax.set_yscale("log")
    ax.set_xlabel("$\\Sigma E_T^{cells}\\,/\\,p_T^{truth}$")
    ax.set_ylabel("electrons / bin")
    ax.set_title("(d) Energy-completeness distribution", fontsize=10)
    ax.legend(fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = OUTPUT_DIR / "energy_starved_report.png"
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".pdf"))

    print(f"electrons: {n_tot}   starved: {n_starved} ({frac:.2%})")
    edge = etas > FIDUCIAL_ETA
    print(f"starved with |eta| > {FIDUCIAL_ETA} (edge population): "
          f"{int((starved & edge).sum())} "
          f"({(starved & edge).sum() / max(n_starved, 1):.1%} of starved)")
    print(f"starved with |eta| <= {FIDUCIAL_ETA} (flat population): "
          f"{int((starved & ~edge).sum())}")
    print(f"median n_cells: starved {np.median(ncells[starved]):.0f}, "
          f"healthy {np.median(ncells[~starved]):.0f}")
    print(f"saved {out} and {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()