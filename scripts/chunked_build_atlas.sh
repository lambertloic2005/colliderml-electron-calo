#!/usr/bin/env bash
# ===========================================================================
# chunked_build_atlas.sh -- disk-bounded full-dataset build on atlas.
#
# Loops over shard-index chunks [LO, HI]; per chunk:
#   1. download  : only that range (skips files already on disk -- your
#                  pre-existing shards 0-184 cost zero download)
#   2. process   : N_TASKS parallel workers, BOUNDED to the range via
#                  --shard-min/--shard-max (requires the fetch_and_cluster.py
#                  and pipeline.py range patches), each writing a uniquely
#                  named part parquet -- parts from all chunks accumulate
#   3. delete    : the chunk's raw shard pairs (only after every worker exits
#                  0 AND the expected part files exist)
#   4. marker    : touch a .done file so re-running resumes where it stopped
#
# Peak raw footprint ~= CHUNK_PAIRS x mean pair size (~2.7 GB/pair at pu200,
# scaled from 185 pairs =~ 500 GB) + the growing (much smaller) parts dir.
# CHUNK_PAIRS=50 => ~135 GB of raw on disk at any time. Adjust to your quota.
#
# RAM sizing unchanged: N_TASKS <= min(free cores, free_RAM_GB / 8).
# Keep N_TASKS <= CHUNK_PAIRS so every worker has at least one shard.
#
# Usage (repo root, colliderml env active, inside tmux):
#     CHUNK_PAIRS=50 N_TASKS=16 bash scripts/chunked_build_atlas.sh
# Resume after any interruption by re-running the same command: completed
# chunks are skipped via markers; a half-downloaded chunk re-skips the files
# it already has; a half-processed chunk is simply re-processed in full
# (part files are overwritten, raw was not yet deleted).
#
# WARNING: DELETE_RAW=1 (default) removes raw shards after processing --
# including your original 0-184 v1 raw copies. They remain re-downloadable
# from HuggingFace. Set DELETE_RAW=0 to keep raw (needs full-dataset disk).
# NEVER point PARTS at the v1 processed directory.
# ===========================================================================
set -eo pipefail

: "${REPO:=$HOME/colliderml-electron-calo}"
: "${RAW:=/data/atlas/lambert/raw}"                   # = COLLIDERML_DATA_DIR
: "${PARTS:=/data/atlas/lambert/processed_v2/parts}"
: "${SHARD_START:=0}"
: "${SHARD_END:=999}"          # zee_pu200 ships 1000 shards: indices 0..999
: "${CHUNK_PAIRS:=50}"
: "${N_TASKS:=16}"
: "${DELETE_RAW:=1}"
LOGS="$PARTS/logs"
MARKS="$PARTS/markers"

mkdir -p "$PARTS" "$LOGS" "$MARKS"
export COLLIDERML_DATA_DIR="$RAW"
export PYTHONUNBUFFERED=1
cd "$REPO"

[ "$N_TASKS" -le "$CHUNK_PAIRS" ] \
    || { echo "ERROR: N_TASKS ($N_TASKS) must be <= CHUNK_PAIRS ($CHUNK_PAIRS)." >&2; exit 1; }
python -c "from colliderml_electron.pipeline import build_electron_table" \
    || { echo "ERROR: colliderml_electron not importable -- 'pip install -e .' first." >&2; exit 1; }
# Refuse to run against unpatched code: without range bounds the process stage
# would sweep up every shard on disk and later chunks would duplicate them.
python scripts/fetch_and_cluster.py --help 2>/dev/null | grep -q -- --shard-min \
    || { echo "ERROR: fetch_and_cluster.py lacks --shard-min/--shard-max -- apply the range patches first." >&2; exit 1; }

count_pairs_in_range () {  # $1=lo $2=hi -> echoes number of matched pairs on disk
    python - "$RAW" "$1" "$2" <<'EOF'
import glob, os, sys
raw, lo, hi = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
def idxs(cfg):
    pat = f"{raw}/CERN__ColliderML-Release-1/{cfg}/data/{cfg}/train-*.parquet"
    return {int(os.path.basename(p).split("-")[1]) for p in glob.glob(pat)}
common = idxs("zee_pu200_particles") & idxs("zee_pu200_calo_hits")
print(sum(1 for i in common if lo <= i <= hi))
EOF
}

LO="$SHARD_START"
while [ "$LO" -le "$SHARD_END" ]; do
    HI=$((LO + CHUNK_PAIRS - 1)); [ "$HI" -gt "$SHARD_END" ] && HI="$SHARD_END"
    MARK="$MARKS/chunk_${LO}_${HI}.done"
    if [ -f "$MARK" ]; then
        echo "=== chunk [$LO,$HI]: already done, skipping ==="
        LO=$((HI + 1)); continue
    fi
    echo "=== chunk [$LO,$HI]: DOWNLOAD ==="
    python -u scripts/fetch_and_cluster.py --stage download \
        --channel zee --pileup pu200 \
        --cap-gb 0 --shard-min "$LO" --shard-max "$HI" \
        --data-dir "$RAW" --no-link \
        2>&1 | tee "$LOGS/download_${LO}_${HI}.log"

    n_pairs=$(count_pairs_in_range "$LO" "$HI")
    [ "$n_pairs" -gt 0 ] || { echo "ERROR: no matched pairs on disk for [$LO,$HI] after download." >&2; exit 1; }
    expected_parts=$(( n_pairs < N_TASKS ? n_pairs : N_TASKS ))

    echo "=== chunk [$LO,$HI]: PROCESS ($n_pairs pairs, $N_TASKS workers) ==="
    pids=()
    for i in $(seq 0 $((N_TASKS - 1))); do
        python -u scripts/fetch_and_cluster.py --stage process \
            --channel zee --pileup pu200 \
            --data-dir "$RAW" --no-link \
            --shard-min "$LO" --shard-max "$HI" \
            --task-id "$i" --n-tasks "$N_TASKS" \
            --out "$PARTS/part_s${LO}-${HI}_t${i}.parquet" \
            > "$LOGS/process_${LO}_${HI}_t${i}.log" 2>&1 &
        pids+=($!)
    done
    fail=0
    for i in "${!pids[@]}"; do
        wait "${pids[$i]}" \
            || { echo "ERROR: worker $i failed -- see $LOGS/process_${LO}_${HI}_t${i}.log" >&2; fail=1; }
    done
    [ "$fail" -eq 0 ] || { echo "Chunk [$LO,$HI] failed; raw NOT deleted. Fix and re-run." >&2; exit 1; }

    n_parts=$(ls "$PARTS"/part_s${LO}-${HI}_t*.parquet 2>/dev/null | wc -l)
    [ "$n_parts" -eq "$expected_parts" ] \
        || { echo "ERROR: chunk [$LO,$HI]: expected $expected_parts part files, found $n_parts. Raw NOT deleted." >&2; exit 1; }

    if [ "$DELETE_RAW" -eq 1 ]; then
        echo "=== chunk [$LO,$HI]: DELETE raw shards ==="
        for cfg in zee_pu200_particles zee_pu200_calo_hits; do
            d="$RAW/CERN__ColliderML-Release-1/$cfg/data/$cfg"
            for idx in $(seq "$LO" "$HI"); do
                printf -v stem 'train-%05d-of-' "$idx"
                rm -f "$d/${stem}"*.parquet
            done
        done
    fi
    touch "$MARK"
    echo "=== chunk [$LO,$HI]: DONE ($n_parts parts) -- disk: $(df -h --output=avail "$RAW" | tail -1 | tr -d ' ') free ==="
    LO=$((HI + 1))
done

echo ""
echo "All chunks complete. Parts in $PARTS. Now merge + harmonize:"
echo "  python -u scripts/fetch_and_cluster.py --stage merge \\"
echo "      --parts-glob '$PARTS/part_*.parquet' \\"
echo "      --out /data/atlas/lambert/processed_v2/zee_pu200_supervised_dbscan_v2.parquet"
echo "  python scripts/harmonize_splits.py \\"
echo "      --v1 /data/atlas/lambert/processed/zee_pu200_supervised_dbscan.parquet \\"
echo "      --v2 /data/atlas/lambert/processed_v2/zee_pu200_supervised_dbscan_v2.parquet"