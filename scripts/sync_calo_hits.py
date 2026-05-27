#!/usr/bin/env python3
"""
sync_calo_hits.py — download ColliderML calo_hits Parquet shards, skipping any
already present locally.

ColliderML Release 1 is hosted on the Hugging Face Hub. hf_hub_download already
resumes partial files and skips anything whose hash/etag matches the local cache.
On top of that, this script does an explicit "do I already have this filename?"
check against YOUR data directory (scanned recursively), so files you fetched by
any earlier method are never re-downloaded.

Usage:
    pip install huggingface_hub
    python sync_calo_hits.py --dry-run      # show the plan, download nothing
    python sync_calo_hits.py                # download only the missing shards

If the repo is gated, run `hf auth login` first (or set HF_TOKEN).
"""

import argparse
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

# ---- configure for your setup --------------------------------------------
REPO_ID = "CERN/ColliderML-Release-1"   # dataset repo on the HF Hub
# Substrings every wanted file must contain. Confirm the exact CHANNEL token
# from the --dry-run listing (it may be "zee", "Zee", "z_ee", ...).
MATCH = ["zee", "pu200", "particles"]
# Root under which your existing calo_hits parquet already live AND where new
# shards will be written. Scanned recursively, so sub-folders are fine.
LOCAL_DIR = Path("data/colliderml")
# --------------------------------------------------------------------------


def remote_shards():
    api = HfApi()
    files = api.list_repo_files(REPO_ID, repo_type="dataset")
    return sorted(
        f for f in files
        if f.endswith(".parquet") and all(tok in f for tok in MATCH)
    )


def local_filenames():
    if not LOCAL_DIR.exists():
        return set()
    return {p.name for p in LOCAL_DIR.rglob("*.parquet")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="list the download plan without fetching")
    ap.add_argument("--force", action="store_true",
                    help="re-download even if the filename already exists locally")
    args = ap.parse_args()

    remote = remote_shards()
    if not remote:
        print(f"No remote files matched {MATCH} in {REPO_ID}.")
        print("Inspect the real naming with:")
        print(f'  python -c "from huggingface_hub import HfApi; '
              f'print(*HfApi().list_repo_files(\'{REPO_ID}\', repo_type=\'dataset\'), sep=chr(10))"'
              ' | grep -i calo')
        return

    have = local_filenames()
    todo = remote if args.force else [f for f in remote if Path(f).name not in have]

    matched_present = len(have & {Path(f).name for f in remote})
    print(f"Remote shards matching {MATCH}: {len(remote)}")
    print(f"Already present under {LOCAL_DIR}: {matched_present}")
    print(f"To download: {len(todo)}")
    for f in todo:
        print("   +", f)

    if args.dry_run or not todo:
        return

    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {f}")
        hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=f,
            local_dir=str(LOCAL_DIR),   # reconstructs the repo subpath under here
            force_download=args.force,
        )
    print("Done.")


if __name__ == "__main__":
    main()