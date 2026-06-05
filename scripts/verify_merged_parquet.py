from pathlib import Path
import glob
import json
import math
import sys

import polars as pl


# Change these if your paths are different
MERGED = Path("/data/atlas/lambert/processed/zee_pu200_supervised_dbscan.parquet")
PARTS_GLOB = "/data/atlas/lambert/processed/parts/part_*.parquet"
STATS = Path("/data/atlas/lambert/processed/target_stats.json")


REQUIRED_COLS = [
    # identifiers
    "event_id",
    "particle_id",

    # truth labels / targets
    "truth_energy",
    "truth_px",
    "truth_py",
    "truth_pz",
    "truth_p",
    "truth_pt",
    "truth_log_pt",
    "truth_eta",
    "truth_phi",
    "truth_charge",

    # cell-level shower inputs
    "n_cells",
    "cell_x",
    "cell_y",
    "cell_z",
    "cell_eta",
    "cell_phi",
    "cell_e_total",
    "cell_e_calibrated",
    "cell_detector",

    # truth-derived diagnostic cell quantities
    "cell_t_from_e",
    "cell_e_from_e",
    "cell_e_from_e_cal",
    "cell_dR_truth",

    # added during merge/stats
    "split",
]

TARGET_COLS = [
    "truth_energy",
    "truth_px",
    "truth_py",
    "truth_pz",
    "truth_eta",
    "truth_phi",
    "truth_log_pt",
]

CELL_LIST_COLS = [
    "cell_x",
    "cell_y",
    "cell_z",
    "cell_eta",
    "cell_phi",
    "cell_e_total",
    "cell_e_calibrated",
    "cell_detector",
    "cell_t_from_e",
    "cell_e_from_e",
    "cell_e_from_e_cal",
    "cell_dR_truth",
]

EXPECTED_DETECTOR_CODES = {9, 10, 11, 12, 13, 14}


failures = []


def ok(msg):
    print(f"✅ {msg}")


def warn(msg):
    print(f"⚠️  {msg}")


def fail(msg):
    failures.append(msg)
    print(f"❌ {msg}")


def row_count(path):
    return pl.scan_parquet(path).select(pl.count().alias("n")).collect().item()


print("\n=== Checking file existence ===")

if not MERGED.exists():
    fail(f"Merged parquet does not exist: {MERGED}")
    sys.exit(1)

ok(f"Found merged parquet: {MERGED}")

parts = sorted(glob.glob(PARTS_GLOB))
if not parts:
    fail(f"No part files matched: {PARTS_GLOB}")
else:
    ok(f"Found {len(parts)} part parquet files")

if len(parts) != 18:
    warn(f"Expected 18 part files, found {len(parts)}")


print("\n=== Checking schema / required columns ===")

try:
    schema = pl.read_parquet_schema(MERGED)
except Exception:
    schema = pl.scan_parquet(MERGED).schema

cols = list(schema.keys())

missing = [c for c in REQUIRED_COLS if c not in cols]
if missing:
    fail(f"Missing required columns: {missing}")
else:
    ok("All required columns are present")

print("\nColumns in merged parquet:")
for c in cols:
    print(f"  {c}: {schema[c]}")


print("\n=== Checking row counts: parts vs merged ===")

if parts:
    part_counts = []
    for p in parts:
        n = row_count(p)
        part_counts.append(n)
        print(f"{Path(p).name}: {n} rows")

    merged_n = row_count(MERGED)
    summed_parts = sum(part_counts)

    print(f"\nSum of part rows: {summed_parts}")
    print(f"Merged rows:       {merged_n}")

    if summed_parts == merged_n:
        ok("Merged parquet row count equals sum of all part files")
    else:
        fail("Merged parquet row count does NOT equal sum of part files")


print("\n=== Checking nulls in scalar columns ===")

scalar_cols = [
    "event_id",
    "particle_id",
    "truth_energy",
    "truth_px",
    "truth_py",
    "truth_pz",
    "truth_p",
    "truth_pt",
    "truth_log_pt",
    "truth_eta",
    "truth_phi",
    "truth_charge",
    "n_cells",
    "split",
]

available_scalar_cols = [c for c in scalar_cols if c in cols]

if available_scalar_cols:
    nulls = (
        pl.scan_parquet(MERGED)
        .select([pl.col(c).is_null().sum().alias(c) for c in available_scalar_cols])
        .collect()
        .to_dicts()[0]
    )

    bad_nulls = {c: n for c, n in nulls.items() if n != 0}
    if bad_nulls:
        fail(f"Null values found in scalar columns: {bad_nulls}")
    else:
        ok("No nulls in scalar columns")


print("\n=== Checking cell-list lengths ===")

available_cell_cols = [c for c in CELL_LIST_COLS if c in cols]

if "n_cells" in cols and available_cell_cols:
    length_mismatches = (
        pl.scan_parquet(MERGED)
        .select([
            (pl.col(c).list.len() != pl.col("n_cells")).sum().alias(c)
            for c in available_cell_cols
        ])
        .collect()
        .to_dicts()[0]
    )

    bad_lengths = {c: n for c, n in length_mismatches.items() if n != 0}
    if bad_lengths:
        fail(f"Some cell-list lengths do not match n_cells: {bad_lengths}")
    else:
        ok("Every cell-list column has length equal to n_cells")

    empty_rows = (
        pl.scan_parquet(MERGED)
        .select((pl.col("n_cells") <= 0).sum().alias("bad"))
        .collect()
        .item()
    )

    if empty_rows:
        fail(f"Found {empty_rows} rows with n_cells <= 0")
    else:
        ok("Every electron row has at least one cell")


print("\n=== Checking train/val/test split ===")

if "split" in cols and "event_id" in cols:
    split_counts = (
        pl.scan_parquet(MERGED)
        .group_by("split")
        .agg([
            pl.count().alias("n_electrons"),
            pl.col("event_id").n_unique().alias("n_events"),
        ])
        .collect()
        .sort("split")
    )

    print(split_counts)

    split_values = set(split_counts["split"].to_list())
    expected_splits = {"train", "val", "test"}

    if split_values == expected_splits:
        ok("Found train, val, and test splits")
    else:
        fail(f"Unexpected split values: {split_values}")

    event_split_leakage = (
        pl.read_parquet(MERGED, columns=["event_id", "split"])
        .group_by("event_id")
        .agg(pl.col("split").n_unique().alias("n_splits"))
        .filter(pl.col("n_splits") > 1)
        .height
    )

    if event_split_leakage == 0:
        ok("No event appears in more than one split")
    else:
        fail(f"{event_split_leakage} events appear in more than one split")


print("\n=== Checking basic physics consistency ===")

needed = {"truth_px", "truth_py", "truth_pt", "truth_log_pt", "truth_phi", "truth_energy"}
if needed.issubset(cols):
    physics = (
        pl.scan_parquet(MERGED)
        .select([
            (
                pl.col("truth_pt")
                - (pl.col("truth_px") ** 2 + pl.col("truth_py") ** 2).sqrt()
            )
            .abs()
            .max()
            .alias("max_pt_error"),

            (
                pl.col("truth_log_pt") - pl.col("truth_pt").log()
            )
            .abs()
            .max()
            .alias("max_logpt_error"),

            (
                (pl.col("truth_phi") < -math.pi)
                | (pl.col("truth_phi") > math.pi)
            )
            .sum()
            .alias("truth_phi_out_of_range"),

            (
                (pl.col("truth_pt") <= 0)
                | (pl.col("truth_energy") <= 0)
            )
            .sum()
            .alias("nonpositive_truth_energy_or_pt"),
        ])
        .collect()
        .to_dicts()[0]
    )

    print(physics)

    if physics["max_pt_error"] < 1e-5:
        ok("truth_pt matches sqrt(px^2 + py^2)")
    else:
        fail("truth_pt does not match sqrt(px^2 + py^2) within tolerance")

    if physics["max_logpt_error"] < 1e-5:
        ok("truth_log_pt matches log(truth_pt)")
    else:
        fail("truth_log_pt does not match log(truth_pt) within tolerance")

    if physics["truth_phi_out_of_range"] == 0:
        ok("truth_phi is within [-pi, pi]")
    else:
        fail(f"{physics['truth_phi_out_of_range']} truth_phi values are outside [-pi, pi]")

    if physics["nonpositive_truth_energy_or_pt"] == 0:
        ok("truth_energy and truth_pt are positive")
    else:
        fail(f"{physics['nonpositive_truth_energy_or_pt']} rows have nonpositive truth_energy or truth_pt")


print("\n=== Checking detector codes ===")

if "cell_detector" in cols:
    detector_codes = (
        pl.scan_parquet(MERGED)
        .select(pl.col("cell_detector").explode().unique().sort())
        .collect()
        .to_series()
        .to_list()
    )

    print(f"Detector codes found: {detector_codes}")

    outside = sorted(set(detector_codes) - EXPECTED_DETECTOR_CODES)
    if outside:
        fail(
            f"Found detector codes not represented by the dataset one-hot encoding: {outside}"
        )
    else:
        ok("Detector codes match the expected one-hot set")


print("\n=== Checking target_stats.json ===")

if not STATS.exists():
    warn(f"No target_stats.json found at {STATS}")
else:
    ok(f"Found target stats: {STATS}")

    stats = json.loads(STATS.read_text())

    missing_stats = [c for c in TARGET_COLS if c not in stats]
    if missing_stats:
        fail(f"target_stats.json is missing stats for: {missing_stats}")
    else:
        ok("target_stats.json contains all target columns")

        train_stats = (
            pl.scan_parquet(MERGED)
            .filter(pl.col("split") == "train")
            .select(
                [
                    pl.col(c).mean().alias(f"{c}_mean")
                    for c in TARGET_COLS
                ]
                +
                [
                    pl.col(c).std().alias(f"{c}_std")
                    for c in TARGET_COLS
                ]
            )
            .collect()
            .to_dicts()[0]
        )

        mismatches = []

        for c in TARGET_COLS:
            mean_file = stats[c]["mean"]
            std_file = stats[c]["std"]

            mean_actual = train_stats[f"{c}_mean"]
            std_actual = train_stats[f"{c}_std"]

            if abs(mean_file - mean_actual) > 1e-6:
                mismatches.append((c, "mean", mean_file, mean_actual))

            if abs(std_file - std_actual) > 1e-6:
                mismatches.append((c, "std", std_file, std_actual))

        if mismatches:
            fail("target_stats.json does not match stats recomputed from train split")
            for item in mismatches:
                print("  mismatch:", item)
        else:
            ok("target_stats.json matches the train split")


print("\n=== Final result ===")

if failures:
    print(f"\nFAILED with {len(failures)} issue(s):")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)

print("\nPASS: merged parquet looks consistent and training-ready.")