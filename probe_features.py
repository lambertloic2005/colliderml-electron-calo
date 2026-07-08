"""Diagnose feature conditioning (extreme values, NaN/inf, or a dead/zero
branch) in x_high_level, applying the padding mask correctly.

v2 fix: the first version flattened (B, L, D) batches WITHOUT applying the
padding mask, and L (the per-batch max cell count) varies batch to batch --
so raw concatenation across batches was invalid and its "all zero" result
is not trustworthy. This version applies `mask` (True = padding, per
collate_pad) before accumulating any statistic, processing one batch at a
time so ragged L across batches is never an issue.

Usage:
    python probe_features_v2.py <checkpoint.pt> [--parquet PATH] [--stats PATH]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "src")
from colliderml_electron.dataset import make_loader, TARGET_COLS  # noqa: E402

RESERVOIR_CAP = 300_000  # rows kept for percentile estimates; min/max/NaN/inf are exact over ALL rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--parquet", default=None)
    ap.add_argument("--stats", default=None)
    args = ap.parse_args()

    cfg = torch.load(args.checkpoint, map_location="cpu")["config"]
    parquet_path = Path(args.parquet or cfg["parquet_path"])
    stats_path = Path(args.stats or cfg["target_stats_path"])
    print(f"parquet: {parquet_path.resolve()}  (exists={parquet_path.exists()})")
    print(f"stats  : {stats_path.resolve()}  (exists={stats_path.exists()})")
    if not parquet_path.exists() or not stats_path.exists():
        sys.exit("Path(s) not found here -- run on the machine where the "
                  "checkpoint's paths resolve, or pass --parquet/--stats.")

    stats = json.loads(stats_path.read_text())
    missing = [c for c in TARGET_COLS if c not in stats]
    if missing:
        sys.exit(f"stats file missing {missing} -- wrong/stale file. "
                  f"Locate the correct one and pass --stats.")

    loader = make_loader(
        parquet_path=parquet_path, split="test", target_stats_path=stats_path,
        batch_size=256, shuffle=False,
        use_angular_features=True, use_cluster_features=True,
        max_abs_eta=cfg.get("max_abs_eta"), min_abs_eta=cfg.get("min_abs_eta"),
    )

    D = None
    n_valid_total = 0
    n_nan_total = 0
    n_inf_total = 0
    running_min = running_max = None
    reservoir = []  # list of small numpy arrays; capped total size
    reservoir_n = 0

    rng = np.random.default_rng(0)
    for bi, batch in enumerate(loader):
        x = batch["x_high_level"].numpy()      # (B, L, D), zero-padded
        pad = batch["mask"].numpy()            # True = padding
        valid = ~pad                            # True = real cell
        if D is None:
            D = x.shape[-1]
            running_min = np.full(D, np.inf)
            running_max = np.full(D, -np.inf)
            print(f"D (feature width) = {D}")

        xv = x[valid]                           # (n_valid_in_batch, D) -- safe: valid is 2D bool on (B,L)
        if xv.shape[0] == 0:
            continue

        n_nan_total += int(np.isnan(xv).sum())
        n_inf_total += int(np.isinf(xv).sum())
        finite = xv[np.isfinite(xv).all(axis=1)]
        if finite.shape[0]:
            running_min = np.minimum(running_min, finite.min(axis=0))
            running_max = np.maximum(running_max, finite.max(axis=0))
        n_valid_total += xv.shape[0]

        # reservoir sample (uniform-ish; fine for a diagnostic, not a proof)
        if reservoir_n < RESERVOIR_CAP:
            take = min(finite.shape[0], RESERVOIR_CAP - reservoir_n)
            if take > 0:
                idx = rng.choice(finite.shape[0], size=take, replace=False) \
                      if finite.shape[0] > take else np.arange(finite.shape[0])
                reservoir.append(finite[idx])
                reservoir_n += take

        if bi % 20 == 0:
            print(f"  batch {bi}: valid cells so far = {n_valid_total}")

    print(f"\ntotal valid (non-padded) cells seen: {n_valid_total}")
    print(f"NaN in valid cells: {n_nan_total}   inf in valid cells: {n_inf_total}")

    if n_valid_total == 0:
        sys.exit("No valid (non-padded) cells found at all -- the mask "
                  "itself may be wrong, or every event has n_cells == 0. "
                  "This would need looking at further; do not conclude "
                  "the feature branch is dead from this alone.")

    absmax = np.maximum(np.abs(running_min), np.abs(running_max))
    order = np.argsort(absmax)[::-1]
    print(f"\nEXACT min/max over ALL {n_valid_total} valid cells:")
    print(f"{'feat':>5s} {'min':>12s} {'max':>12s} {'absmax':>12s}")
    for i in order[:15]:
        print(f"{i:5d} {running_min[i]:12.4f} {running_max[i]:12.4f} {absmax[i]:12.4f}")

    if reservoir:
        R = np.concatenate(reservoir)
        print(f"\npercentiles from a {R.shape[0]}-row reservoir sample:")
        for i in order[:15]:
            p50, p99 = np.percentile(np.abs(R[:, i]), [50, 99])
            print(f"  feat {i:2d}: p50(|x|)={p50:10.4f}  p99(|x|)={p99:10.4f}")

    frac20 = float((np.abs(reservoir[0] if reservoir else np.zeros((1, D))) > 20).any())
    print(f"\nIf absmax is genuinely ~0 for ALL features above, that is a real "
          f"finding worth escalating. If it now shows nonzero values as "
          f"expected (log_e alone should reach O(1-10) magnitude), the v1 "
          f"script's ragged-concat bug was the explanation, not the model.")


if __name__ == "__main__":
    main()