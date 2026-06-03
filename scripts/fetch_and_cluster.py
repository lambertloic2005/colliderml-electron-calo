#!/usr/bin/env python3
"""
fetch_and_cluster.py
====================

Size-capped download of one ColliderML channel (default ``zee_pu200``) followed
by the SUPERVISED per-electron processing (truth selects the shower cells, then
DBSCAN cleans outliers within that truth-selected shower). Built to run
unattended on a remote machine over SSH.

Two stages you can run together or separately:

  download : fetch matching ``particles`` + ``calo_hits`` shard PAIRS into the
             colliderml cache, stopping *before* a hard byte cap (default 500 GB).
             build_electron_table intersects particles ∩ calo_hits shard indices,
             so shards are only useful in matching index pairs -- that is what we
             download.

  process  : run ``pipeline.build_electron_table`` over whatever shard pairs are
             present locally. For each prompt electron (pdg ±11, primary,
             vertex_primary==1) it walks parent_id to gather the electron's full
             shower family, pulls those cells via the MC-truth contrib lists
             (cells_for_particle_set), then -- with mask_kind="dbscan" -- applies
             dbscan_keep_mask to drop outlier cells, keeping the DBSCAN cluster
             nearest the truth direction. Supervised: truth chooses the cells,
             DBSCAN only cleans them. (mask_kind="cone" swaps the clean for a
             fixed dR_max cone instead.)

Where data lives (--data-dir / COLLIDERML_DATA_DIR)
---------------------------------------------------
build_electron_table reads its shards from a fixed cache root. By default that is
``~/.cache/colliderml``; if you set the env var COLLIDERML_DATA_DIR (and have the
matching one-line patch in pipeline.py), it reads from there instead. This script
mirrors the same rule via _builder_cache_root().

Recommended on a shared / quota'd machine: keep nothing in home. Point BOTH the
env var and --data-dir at the same folder you control (scratch or your project
space). Then no symlink is created and no --no-link flag is needed:

    export COLLIDERML_DATA_DIR=/scratch/$USER/colliderml
    export HF_HOME=/scratch/$USER/hf            # keep HF metadata off home too
    python scripts/fetch_and_cluster.py --cap-gb 500 \
        --data-dir /scratch/$USER/colliderml \
        --out /scratch/$USER/myproject/electrons/zee_pu200.parquet

Without the env var, the script falls back to symlinking ~/.cache/colliderml ->
--data-dir so the (unpatched) builder still finds scratch data. Use --no-link to
suppress that.

Cache layout (what build_electron_table globs):
  <root>/CERN__ColliderML-Release-1/<config>/data/<config>/train-NNNNN-of-MMMMM.parquet
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO_ID = "CERN/ColliderML-Release-1"
DATASET_DIRNAME = "CERN__ColliderML-Release-1"  # how colliderml sanitises REPO_ID
SPLIT = "train"
GB = 1024 ** 3
# Approximate events per shard, only used to print a friendly event estimate.
# pu200 ships ~100 events/shard, pu0 ~1000 (per the colliderml download docs).
EVENTS_PER_SHARD = {"pu0": 1000, "pu200": 100}
# Fallback target columns, kept in sync with dataset.TARGET_COLS. We import the
# real list at runtime (so the two never drift); this is only used if that import
# fails -- e.g. running the stats stage in an env without torch installed.
DEFAULT_TARGET_COLS = [
    "truth_energy", "truth_px", "truth_py", "truth_pz",
    "truth_eta", "truth_phi", "truth_log_pt",
]


def _builder_cache_root() -> Path:
    """Root that build_electron_table reads from.

    Mirrors the (patched) pipeline.py rule: COLLIDERML_DATA_DIR if set, else
    ~/.cache/colliderml. Used so this script's symlink/logging logic stays in
    lock-step with where the builder actually looks.
    """
    env = os.environ.get("COLLIDERML_DATA_DIR")
    base = Path(env) if env else Path.home() / ".cache" / "colliderml"
    return base.expanduser().resolve()


BUILDER_CACHE = _builder_cache_root()


# --------------------------------------------------------------------------- #
# shard discovery / planning
# --------------------------------------------------------------------------- #
def _shard_index(path: str) -> int:
    """'data/<cfg>/train-00007-of-01000.parquet' -> 7 (matches pipeline._shard_index)."""
    return int(os.path.basename(path).split("-")[1])


def remote_shards_with_size(api, config: str, revision: str | None) -> dict[int, tuple[str, int]]:
    """Map shard index -> (repo_path, size_bytes) for a config's train shards."""
    prefix = f"data/{config}/"
    out: dict[int, tuple[str, int]] = {}
    for entry in api.list_repo_tree(
        REPO_ID, repo_type="dataset", revision=revision,
        path_in_repo=f"data/{config}", recursive=False,
    ):
        path = getattr(entry, "path", None)
        size = getattr(entry, "size", None)  # RepoFile has .size; RepoFolder doesn't
        if (
            path
            and size is not None
            and path.startswith(prefix)
            and path.endswith(".parquet")
            and os.path.basename(path).startswith(f"{SPLIT}-")
        ):
            out[_shard_index(path)] = (path, int(size))
    return out


def plan_pairs(
    calo: dict[int, tuple[str, int]],
    part: dict[int, tuple[str, int]],
    cap_bytes: int | None,
    max_shards: int | None,
) -> tuple[list[int], list[int], int]:
    """Greedily pick the leading common shard indices that fit under cap_bytes.

    Returns (common_indices, planned_indices, planned_bytes).
    """
    common = sorted(set(calo) & set(part))
    planned: list[int] = []
    total = 0
    for idx in common:
        if max_shards is not None and len(planned) >= max_shards:
            break
        pair_bytes = calo[idx][1] + part[idx][1]
        if cap_bytes is not None and total + pair_bytes > cap_bytes:
            break
        total += pair_bytes
        planned.append(idx)
    return common, planned, total


def local_target(data_dir: Path, config: str, repo_path: str) -> Path:
    """Where a shard lands locally, mirroring colliderml's download_config()."""
    return data_dir / DATASET_DIRNAME / config / repo_path


def human(nbytes: float) -> str:
    return f"{nbytes / GB:.1f} GB"


# --------------------------------------------------------------------------- #
# stages
# --------------------------------------------------------------------------- #
def stage_download(args) -> None:
    from huggingface_hub import HfApi, hf_hub_download

    data_dir = Path(args.data_dir).expanduser().resolve()
    calo_cfg = f"{args.channel}_{args.pileup}_calo_hits"
    part_cfg = f"{args.channel}_{args.pileup}_particles"
    cap_bytes = None if args.cap_gb is None else int(args.cap_gb * GB)

    api = HfApi(token=args.token or None)
    print(f"Listing remote shards for {calo_cfg} and {part_cfg} ...")
    calo = remote_shards_with_size(api, calo_cfg, args.revision)
    part = remote_shards_with_size(api, part_cfg, args.revision)
    if not calo or not part:
        sys.exit(f"ERROR: found {len(calo)} calo_hits / {len(part)} particles shards. "
                 f"Check --channel/--pileup (config names must exist in {REPO_ID}).")

    common, planned, planned_bytes = plan_pairs(calo, part, cap_bytes, args.max_shards)
    if not planned:
        first = common[0] if common else None
        extra = ""
        if first is not None:
            extra = (f" The first pair alone is "
                     f"{human(calo[first][1] + part[first][1])}; raise --cap-gb.")
        sys.exit(f"ERROR: nothing fits the plan (cap={args.cap_gb} GB, "
                 f"common shards={len(common)}).{extra}")

    ev = EVENTS_PER_SHARD.get(args.pileup, 100) * len(planned)
    print(
        f"\nPlan:\n"
        f"  common shard pairs available : {len(common)}\n"
        f"  pairs selected               : {len(planned)}  "
        f"(indices {planned[0]}..{planned[-1]})\n"
        f"  planned download size        : {human(planned_bytes)}"
        + (f"  (cap {args.cap_gb} GB)" if cap_bytes else "  (no cap)") + "\n"
        f"  approx events                : ~{ev:,} ({EVENTS_PER_SHARD.get(args.pileup, 100)}/shard, approx)\n"
        f"  destination                  : {data_dir / DATASET_DIRNAME}\n"
    )
    if args.dry_run:
        print("--dry-run: stopping before any download.")
        return

    _ensure_cache_link(data_dir, do_link=not args.no_link)

    downloaded = 0
    skipped = 0
    on_disk = 0
    t0 = time.time()
    for n, idx in enumerate(planned, 1):
        for cfg, table in ((calo_cfg, calo), (part_cfg, part)):
            repo_path, size = table[idx]
            tgt = local_target(data_dir, cfg, repo_path)
            if tgt.exists() and not args.force and abs(tgt.stat().st_size - size) <= 4096:
                skipped += 1
                on_disk += tgt.stat().st_size
                continue
            # local_dir is the per-config root; filename carries the data/<cfg>/... subpath,
            # so the file lands exactly where build_electron_table globs.
            hf_hub_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                revision=args.revision,
                filename=repo_path,
                local_dir=str(data_dir / DATASET_DIRNAME / cfg),
                force_download=args.force,
                token=args.token or None,
            )
            downloaded += 1
            on_disk += tgt.stat().st_size if tgt.exists() else size
        if n % 10 == 0 or n == len(planned):
            rate = on_disk / max(time.time() - t0, 1e-9) / GB
            print(f"  [{n}/{len(planned)}] idx {idx}: ~{human(on_disk)} on disk "
                  f"({downloaded} files fetched, {skipped} skipped, {rate:.2f} GB/s avg)")

    print(f"\nDownload complete: {human(on_disk)} across {len(planned)} shard pairs "
          f"({downloaded} fetched, {skipped} already present).")


def stage_process(args) -> None:
    data_dir = Path(args.data_dir).expanduser().resolve()
    _ensure_cache_link(data_dir, do_link=not args.no_link)

    try:
        from colliderml_electron.pipeline import build_electron_table
    except ModuleNotFoundError as exc:
        sys.exit(f"ERROR: cannot import colliderml_electron ({exc}). "
                 f"Run `pip install -e .` in the repo root first.")

    out = Path(args.out)
    print(f"Supervised processing (mask_kind={args.mask_kind}) of shards under "
          f"{BUILDER_CACHE / DATASET_DIRNAME} -> {out}")
    build_electron_table(
        channel=args.channel,
        pileup=args.pileup,
        max_events=args.n_events,    # None = all downloaded events
        dR_max=args.dR_max,          # used when mask_kind="cone"
        mask_kind=args.mask_kind,    # "dbscan" (supervised clean) or "cone"
        eps=args.eps,                # DBSCAN neighbourhood radius (dR units)
        min_samples=args.min_samples,
        out_path=str(out),
    )


def stage_stats(args) -> None:
    """Assign event-level train/val/test splits in the electron parquet, then
    write target_stats.json (mean/std of each target column over the TRAIN split).
    This is what ElectronDataset / the training+test scripts consume."""
    try:
        from colliderml_electron.splits import assign_splits, compute_target_stats
    except ModuleNotFoundError as exc:
        sys.exit(f"ERROR: cannot import colliderml_electron.splits ({exc}). "
                 f"Run `pip install -e .` in the repo root first.")

    # Keep target columns in lock-step with the dataset; fall back if torch
    # (a dataset.py import) isn't available in this environment.
    try:
        from colliderml_electron.dataset import TARGET_COLS
    except Exception:
        TARGET_COLS = DEFAULT_TARGET_COLS

    parquet = Path(args.out)
    if not parquet.exists():
        sys.exit(f"ERROR: {parquet} not found. Run the process stage first "
                 f"(it writes the electron parquet that splits/stats operate on).")

    stats_out = Path(args.target_stats_out) if args.target_stats_out else parquet.with_name("target_stats.json")
    stats_out.parent.mkdir(parents=True, exist_ok=True)

    # assign_splits rewrites the parquet in place, adding/overwriting a "split"
    # column at the EVENT level (all electrons of an event share a split), so
    # cells from one event never leak across train/val/test.
    print(f"Assigning {args.train_frac:.0%}/{args.val_frac:.0%}/"
          f"{1 - args.train_frac - args.val_frac:.0%} splits (seed={args.split_seed}) in {parquet}")
    assign_splits(
        parquet,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        seed=args.split_seed,
    )

    print(f"Computing target stats over the train split -> {stats_out}")
    compute_target_stats(parquet, TARGET_COLS, out_path=str(stats_out))


# --------------------------------------------------------------------------- #
# cache-path bridge
# --------------------------------------------------------------------------- #
def _ensure_cache_link(data_dir: Path, do_link: bool) -> None:
    """Make sure the builder's cache root resolves to data_dir.

    If COLLIDERML_DATA_DIR is set to data_dir (with the pipeline.py patch), the
    builder already reads here -> nothing to do. Otherwise, on the home default,
    symlink ~/.cache/colliderml -> data_dir so the unpatched builder still finds
    scratch data.
    """
    data_dir = data_dir.resolve()
    if data_dir == BUILDER_CACHE:
        return  # builder already reads exactly here; no bridge needed.

    if not do_link:
        print(
            f"NOTE: builder reads {BUILDER_CACHE}, but --data-dir is {data_dir} "
            f"and --no-link was set.\n"
            f"      The process stage will NOT see your shards unless you either:\n"
            f"        - export COLLIDERML_DATA_DIR={data_dir}  (with the pipeline.py patch), or\n"
            f"        - ln -s {data_dir} {BUILDER_CACHE}\n"
        )
        return

    BUILDER_CACHE.parent.mkdir(parents=True, exist_ok=True)
    if BUILDER_CACHE.is_symlink():
        if BUILDER_CACHE.resolve() == data_dir:
            return
        print(f"NOTE: repointing existing symlink {BUILDER_CACHE} -> {data_dir}")
        BUILDER_CACHE.unlink()
        BUILDER_CACHE.symlink_to(data_dir, target_is_directory=True)
    elif BUILDER_CACHE.exists():
        sys.exit(
            f"ERROR: {BUILDER_CACHE} exists and is a real directory, not a symlink.\n"
            f"       Preferred fix on a shared machine: export COLLIDERML_DATA_DIR={data_dir}\n"
            f"       (with the pipeline.py patch) and re-run -- no symlink needed.\n"
            f"       Or move/remove {BUILDER_CACHE}, or download straight to it via\n"
            f"       --data-dir {BUILDER_CACHE}."
        )
    else:
        BUILDER_CACHE.symlink_to(data_dir, target_is_directory=True)
        print(f"Linked {BUILDER_CACHE} -> {data_dir}")


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Size-capped ColliderML download + supervised per-electron DBSCAN processing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--stage", choices=["all", "download", "process", "stats"], default="all",
                   help="'all' runs download -> process -> stats.")

    # dataset selection
    p.add_argument("--channel", default="zee")
    p.add_argument("--pileup", default="pu200")
    p.add_argument("--revision", default=None, help="HF dataset revision/tag/commit.")
    p.add_argument("--token", default=os.environ.get("HF_TOKEN"),
                   help="HF token if the repo is gated (or set HF_TOKEN / `hf auth login`).")

    # size control
    p.add_argument("--cap-gb", type=float, default=500.0,
                   help="Hard cap on download size. Set to 0 / negative for no cap.")
    p.add_argument("--max-shards", type=int, default=None,
                   help="Optional cap on number of shard pairs (applied alongside "
                        "--cap-gb, whichever binds first).")
    p.add_argument("--data-dir", default=str(BUILDER_CACHE),
                   help="Where shards live. On a shared machine point this (and "
                        "COLLIDERML_DATA_DIR) at scratch / your project space.")
    p.add_argument("--no-link", action="store_true",
                   help="Do not symlink the home cache to --data-dir (use with "
                        "COLLIDERML_DATA_DIR instead).")
    p.add_argument("--force", action="store_true", help="Re-download present shards.")
    p.add_argument("--dry-run", action="store_true", help="Print the plan, download nothing.")

    # supervised selection knobs (forwarded to build_electron_table)
    p.add_argument("--mask-kind", choices=["dbscan", "cone"], default="dbscan",
                   help="How to clean the truth-selected shower cells: DBSCAN cluster "
                        "nearest the truth direction, or a fixed dR_max cone.")
    p.add_argument("--eps", type=float, default=0.08,
                   help="DBSCAN neighbourhood radius in dR (used when --mask-kind dbscan).")
    p.add_argument("--min-samples", type=int, default=2,
                   help="DBSCAN core-point threshold (used when --mask-kind dbscan).")
    p.add_argument("--dR-max", type=float, default=0.1,
                   help="Cone radius around the truth direction (used when --mask-kind cone).")

    # split + target-stats (stats stage)
    p.add_argument("--train-frac", type=float, default=0.70,
                   help="Event-level train fraction for the split assignment.")
    p.add_argument("--val-frac", type=float, default=0.15,
                   help="Event-level val fraction (test = 1 - train - val).")
    p.add_argument("--split-seed", type=int, default=42,
                   help="RNG seed for the event-level split (keep fixed for reproducibility).")
    p.add_argument("--target-stats-out", default=None,
                   help="Where to write target_stats.json (default: alongside --out).")

    # process output / limiting
    p.add_argument("--n-events", type=int, default=None,
                   help="Limit events processed in the process stage (default: all downloaded).")
    p.add_argument("--out", default="data/electrons/electrons.parquet",
                   help="Output parquet for the process stage; the stats stage adds a "
                        "split column to it and writes target_stats.json next to it. "
                        "Use an absolute path.")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.cap_gb is not None and args.cap_gb <= 0:
        args.cap_gb = None  # no cap

    if args.stage in ("all", "download"):
        stage_download(args)
    if args.stage in ("all", "process"):
        stage_process(args)
    if args.stage in ("all", "stats"):
        stage_stats(args)


if __name__ == "__main__":
    main()