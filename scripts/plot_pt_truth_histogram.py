"""
Histogram of truth pT (transverse momentum), read directly from the parquet.

Model-free -- no checkpoint, no dataset.py, no decoding. Reads the raw
`truth_pt` column written by pipeline.py's truth_kinematics() at generation
time, so this is the ground truth as simulated, not a round-trip through the
log(pT) training target.

Produces two panels:
  (1) full range, log-scale y  -- shows the falling spectrum and the tail
  (2) zoomed to [0, 50] GeV    -- resolves the low-pT region, which is where
      charge ID has most of its power (bend handle ~ q*B*L/pT)

The current tester (scripts/test_eta_phi_pt_z0_charge.py) applies a
PT_MIN_GEV = 10.0 floor to the whole evaluation; that cut is drawn on both
panels so you can see exactly what fraction of events -- and which part of
the spectrum -- it discards, split out by train/test/val.

Run from repo root:
    python scripts/plot_pt_truth_histogram.py
    python scripts/plot_pt_truth_histogram.py --parquet path/to/other.parquet
    python scripts/plot_pt_truth_histogram.py --pt-cut-gev 5
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

PARQUET = Path("data/electrons/eta_phi_pt_z0_charge/zee_pu200_z0_charge.parquet")
OUTPUT_DIR = Path("results/diagnostics")
PT_MIN_GEV_DEFAULT = 10.0  # matches PT_MIN_GEV in test_eta_phi_pt_z0_charge.py


def load_truth_pt(parquet_path: Path) -> pl.DataFrame:
    schema_cols = pl.read_parquet_schema(parquet_path)

    if "truth_pt" in schema_cols:
        df = pl.read_parquet(parquet_path, columns=["split", "truth_pt"])
    elif "truth_log_pt" in schema_cols:
        # Fallback only -- prefer the raw column when it exists (see module
        # docstring). exp() of the training target introduces no meaningful
        # error here, but it's a derived quantity, not the source truth.
        print("WARNING: 'truth_pt' column not found; deriving from "
              "exp(truth_log_pt) instead.")
        df = pl.read_parquet(parquet_path, columns=["split", "truth_log_pt"])
        df = df.with_columns(pl.col("truth_log_pt").exp().alias("truth_pt"))
    else:
        raise KeyError(
            "Neither 'truth_pt' nor 'truth_log_pt' found in "
            f"{parquet_path}. Columns present: {schema_cols}"
        )

    return df


def print_summary(df: pl.DataFrame) -> None:
    print(f"\nN electrons total: {df.height}")
    for split_name in sorted(df["split"].unique().to_list()):
        pt = df.filter(pl.col("split") == split_name)["truth_pt"].to_numpy()
        pct = np.percentile(pt, [1, 5, 50, 95, 99])
        print(
            f"  split={split_name:6s} n={len(pt):7d}  "
            f"min={pt.min():7.2f}  max={pt.max():8.2f}  "
            f"mean={pt.mean():7.2f}  median={np.median(pt):7.2f}  "
            f"p1={pct[0]:6.2f}  p5={pct[1]:6.2f}  p95={pct[3]:7.2f}  p99={pct[4]:8.2f}  GeV"
        )


def plot_histograms(df: pl.DataFrame, pt_cut_gev: float, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = sorted(df["split"].unique().to_list())

    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(13, 5))

    pt_all = df["truth_pt"].to_numpy()
    hi_edge = float(np.percentile(pt_all, 99.5))  # trim extreme outlier tail for bin range

    full_bins = np.linspace(0, hi_edge, 80)
    zoom_bins = np.linspace(0, 50, 80)

    for split_name in splits:
        pt = df.filter(pl.col("split") == split_name)["truth_pt"].to_numpy()
        ax_full.hist(pt, bins=full_bins, histtype="step", lw=1.6,
                     label=f"{split_name} (n={len(pt)})")
        ax_zoom.hist(pt, bins=zoom_bins, histtype="step", lw=1.6,
                     label=f"{split_name} (n={len(pt)})")

    for ax, title in [(ax_full, f"Full range (99.5th pct = {hi_edge:.0f} GeV)"),
                       (ax_zoom, "Zoomed: [0, 50] GeV")]:
        ax.axvline(pt_cut_gev, color="k", ls="--", lw=1.2,
                   label=f"pT floor = {pt_cut_gev:.0f} GeV")
        ax.set_xlabel("truth pT [GeV]")
        ax.set_ylabel("Count")
        ax.set_yscale("log")
        ax.set_title(title)
        ax.legend(fontsize=8)

    fig.suptitle("Truth pT distribution (CERN ColliderML Release 1, Z\u2192ee)")
    fig.tight_layout()

    path = output_dir / "truth_pt_histogram.png"
    fig.savefig(path, dpi=200)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", type=Path, default=PARQUET)
    ap.add_argument("--pt-cut-gev", type=float, default=PT_MIN_GEV_DEFAULT,
                     help="cut line to draw on the plot (does not filter data)")
    args = ap.parse_args()

    if not args.parquet.exists():
        raise FileNotFoundError(f"Could not find {args.parquet}")

    df = load_truth_pt(args.parquet)
    print_summary(df)

    for split_name in sorted(df["split"].unique().to_list()):
        pt = df.filter(pl.col("split") == split_name)["truth_pt"].to_numpy()
        below = float(np.mean(pt < args.pt_cut_gev))
        print(f"  split={split_name:6s}: {below:.1%} of events have "
              f"truth pT < {args.pt_cut_gev:.0f} GeV")

    path = plot_histograms(df, args.pt_cut_gev, OUTPUT_DIR)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()