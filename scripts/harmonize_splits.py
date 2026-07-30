#!/usr/bin/env python3
"""
harmonize_splits.py
===================

Carry the v1 train/val/test event assignments into a rebuilt (v2) dataset, then
recompute target_stats.json on the harmonized train split.

Why this exists
---------------
`splits.assign_splits()` seeds an RNG and permutes the *sorted unique event_id
array*, then slices by fraction. That procedure is only reproducible for the
exact same set of event_ids: growing the dataset changes the permutation, so an
event that was 'test' in v1 can land in 'train' in v2. Any v2-test comparison
against a checkpoint trained on v1 (e.g. ruche_Jun23_ConstChargeWeight.pt) would
then be contaminated. Rule enforced here:

  * event_id present in v1  -> keep its v1 split verbatim
  * event_id new in v2      -> deterministic fresh assignment (seeded
                               permutation over the sorted NEW event_ids only,
                               same train/val fractions)

The merge stage of fetch_and_cluster.py has already written a `split` column
and a target_stats.json computed on that UN-harmonized split -- both are
overwritten here. Do not use the pre-harmonization target_stats.json.

Usage
-----
    python scripts/harmonize_splits.py \
        --v1 /data/atlas/lambert/processed/zee_pu200_supervised_dbscan.parquet \
        --v2 /data/atlas/lambert/processed_v2/zee_pu200_supervised_dbscan_v2.parquet

Rewrites --v2 in place (or --out) and writes target_stats.json alongside it.
Note: the full v2 table is materialized in RAM once (same as the merge stage).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

# Keep in lock-step with dataset.TARGET_COLS; fall back if torch (imported by
# dataset.py) is unavailable in this environment. Fallback MUST include
# truth_z0 -- z0 is a trained target on both active branches.
try:
    from colliderml_electron.dataset import TARGET_COLS
except Exception:
    TARGET_COLS = [
        "truth_energy", "truth_px", "truth_py", "truth_pz",
        "truth_eta", "truth_phi", "truth_log_pt", "truth_z0",
    ]


def load_v1_mapping(v1_path: Path) -> pl.DataFrame:
    """event_id -> split from the v1 parquet, with integrity checks."""
    m = (
        pl.scan_parquet(v1_path)
        .select(["event_id", "split"])
        .unique()
        .collect()
    )
    dup = m.group_by("event_id").len().filter(pl.col("len") > 1)
    if dup.height:
        sys.exit(
            f"ERROR: {dup.height} v1 event_ids map to more than one split -- "
            f"v1 file is inconsistent; refusing to harmonize."
        )
    return m.rename({"split": "split_v1"})


def assign_new_events(
    new_ids: np.ndarray, train_frac: float, val_frac: float, seed: int
) -> pl.DataFrame:
    """Deterministic split over ONLY the new event_ids (same recipe as v1:
    seeded permutation of the sorted id array, sliced by fraction)."""
    ids = np.sort(np.unique(new_ids))
    rng = np.random.default_rng(seed)
    ids = rng.permutation(ids)
    n = len(ids)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)
    split = np.full(n, "test", dtype=object)
    split[:n_train] = "train"
    split[n_train:n_train + n_val] = "val"
    return pl.DataFrame({"event_id": ids, "split_new": split})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v1", required=True, help="v1 parquet WITH its split column.")
    ap.add_argument("--v2", required=True, help="Merged v2 parquet (from --stage merge).")
    ap.add_argument("--out", default=None,
                    help="Output parquet (default: rewrite --v2 in place).")
    ap.add_argument("--train-frac", type=float, default=0.70)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=1042,
                    help="Seed for NEW-event assignment. Fixed & recorded for "
                         "reproducibility; deliberately distinct from the v1 seed.")
    ap.add_argument("--stats-out", default=None,
                    help="target_stats.json path (default: alongside output).")
    args = ap.parse_args()

    v1_path, v2_path = Path(args.v1), Path(args.v2)
    out = Path(args.out) if args.out else v2_path
    stats_out = Path(args.stats_out) if args.stats_out else out.with_name("target_stats.json")

    v1_map = load_v1_mapping(v1_path)
    print(f"v1: {v1_map.height} events with fixed splits from {v1_path}")

    df = pl.read_parquet(v2_path)
    if "split" in df.columns:
        df = df.drop("split")
    v2_events = df["event_id"].unique().to_numpy()
    print(f"v2: {len(v2_events)} events / {df.height} electrons from {v2_path}")

    # Duplicate-electron guard: catches a shard processed by two tasks
    # (e.g. a part file merged twice) before it poisons training.
    # Row-identity key. The supervised table (pipeline.build_electron_table) keys
    # rows by (event_id, particle_id). The truth-free cluster table
    # (cluster_pipeline.build_cluster_table) has no particle_id -- one row is one
    # DBSCAN cluster -- and keys rows by (event_id, cluster_id). Pick whichever
    # exists so this script harmonizes both datasets.
    if "particle_id" in df.columns:
        row_key = ["event_id", "particle_id"]
    elif "cluster_id" in df.columns:
        row_key = ["event_id", "cluster_id"]
    else:
        sys.exit("ERROR: table has neither particle_id nor cluster_id -- cannot "
                 "verify row uniqueness; refusing to harmonize.")
    dup_e = df.group_by(row_key).len().filter(pl.col("len") > 1)
    if dup_e.height:
        sys.exit(f"ERROR: {dup_e.height} duplicated {tuple(row_key)} rows -- "
                 f"check the parts glob / re-run merge before harmonizing.")

    # Cluster table only: match_clusters_to_electrons() gives each truth electron
    # at most one cluster per event, so a repeated truth electron can only come
    # from a shard merged twice. Catch it here -- it would silently duplicate
    # training targets and break the paired comparison against the supervised set.
    if "cluster_id" in df.columns and "particle_id" not in df.columns:
        dup_t = (df.group_by(["event_id", "truth_px", "truth_py", "truth_pz"])
                   .len().filter(pl.col("len") > 1))
        if dup_t.height:
            sys.exit(f"ERROR: {dup_t.height} truth electrons are claimed by more "
                     f"than one cluster -- duplicated part files in the merge glob.")

    in_v1 = np.isin(v2_events, v1_map["event_id"].to_numpy())
    missing_from_v2 = v1_map.height - int(in_v1.sum())
    if missing_from_v2 > 0:
        print(f"WARNING: {missing_from_v2} v1 events are absent from v2. Expected "
              f"only if v2 does not cover all v1 shards; their fixed assignments "
              f"are simply unused.")

    new_ids = v2_events[~in_v1]
    print(f"Overlap: {int(in_v1.sum())} events keep v1 splits; "
          f"{len(new_ids)} new events get fresh assignment (seed={args.seed}).")
    new_map = assign_new_events(new_ids, args.train_frac, args.val_frac, args.seed)

    # v1 verbatim where available, otherwise the new assignment.
    df = (
        df.join(v1_map, on="event_id", how="left")
          .join(new_map, on="event_id", how="left")
          .with_columns(pl.coalesce([pl.col("split_v1"), pl.col("split_new")]).alias("split"))
          .drop(["split_v1", "split_new"])
    )
    if df["split"].null_count():
        sys.exit("ERROR: some events received no split assignment -- bug, aborting.")

    # No event may span two splits (event-level split => no cell leakage).
    leaks = (df.group_by("event_id").agg(pl.col("split").n_unique().alias("n"))
               .filter(pl.col("n") > 1).height)
    if leaks:
        sys.exit(f"ERROR: {leaks} events span more than one split -- aborting.")

    df.write_parquet(out)
    counts = df.group_by("split").len().sort("split")
    ev_counts = (df.select(["event_id", "split"]).unique()
                   .group_by("split").len().sort("split"))
    print(f"\nWrote harmonized dataset -> {out}")
    print("Electrons per split:"); print(counts)
    print("Events per split:"); print(ev_counts)

    # Recompute normalization stats on the HARMONIZED train split only.
    tr = df.filter(pl.col("split") == "train")
    stats = {c: {"mean": float(tr[c].mean()), "std": float(tr[c].std())}
             for c in TARGET_COLS}
    stats_out.parent.mkdir(parents=True, exist_ok=True)
    stats_out.write_text(json.dumps(stats, indent=2))
    print(f"Recomputed target stats on harmonized train split -> {stats_out}")
    for c in TARGET_COLS:
        print(f"  {c:14s} mean={stats[c]['mean']:+.6g}  std={stats[c]['std']:.6g}")


if __name__ == "__main__":
    main()