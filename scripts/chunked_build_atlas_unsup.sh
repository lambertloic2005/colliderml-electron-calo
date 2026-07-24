#!/usr/bin/env bash
# chunked_build_atlas_unsup.sh -- disk-bounded UNSUPERVISED (truth-free DBSCAN)
# full-dataset build. Same chunk/download/delete/marker discipline as
# chunked_build_atlas.sh; process stage = build_cluster_dataset.py.
set -eo pipefail
: "${REPO:=$HOME/colliderml-electron-calo}"
: "${RAW:=/data/atlas/lambert/raw}"
: "${PARTS:=/data/atlas/lambert/unsup_v1/parts}"    # NEW dir -- never a supervised dir
: "${SHARD_START:=0}"
: "${SHARD_END:=999}"
: "${CHUNK_PAIRS:=50}"
: "${N_TASKS:=16}"
: "${DELETE_RAW:=1}"
LOGS="$PARTS/logs"; MARKS="$PARTS/markers"
mkdir -p "$PARTS" "$LOGS" "$MARKS"
export COLLIDERML_DATA_DIR="$RAW"
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_XET=1
cd "$REPO"

[ "$N_TASKS" -le "$CHUNK_PAIRS" ] || { echo "N_TASKS must be <= CHUNK_PAIRS" >&2; exit 1; }
python -c "from colliderml_electron.cluster_pipeline import build_cluster_table" \
    || { echo "ERROR: cluster_pipeline not importable" >&2; exit 1; }
python scripts/build_cluster_dataset.py --help 2>/dev/null | grep -q -- --shard-min \
    || { echo "ERROR: build_cluster_dataset.py lacks --shard-min -- apply Change 8" >&2; exit 1; }

LO="$SHARD_START"
while [ "$LO" -le "$SHARD_END" ]; do
    HI=$((LO + CHUNK_PAIRS - 1)); [ "$HI" -gt "$SHARD_END" ] && HI="$SHARD_END"
    MARK="$MARKS/chunk_${LO}_${HI}.done"
    if [ -f "$MARK" ]; then echo "=== chunk [$LO,$HI]: done, skipping ==="; LO=$((HI + 1)); continue; fi

    echo "=== chunk [$LO,$HI]: download ==="
    python -u scripts/fetch_and_cluster.py --stage download \
        --channel zee --pileup pu200 --data-dir "$RAW" --no-link \
        --shard-min "$LO" --shard-max "$HI" --cap-gb 500 \
        > "$LOGS/download_${LO}_${HI}.log" 2>&1

    echo "=== chunk [$LO,$HI]: process ($N_TASKS workers) ==="
    pids=()
    for i in $(seq 0 $((N_TASKS - 1))); do
        python -u scripts/build_cluster_dataset.py \
            --channel zee --pileup pu200 \
            --shard-min "$LO" --shard-max "$HI" \
            --task-id "$i" --n-tasks "$N_TASKS" \
            --out "$PARTS/part_${LO}_${HI}_task${i}.parquet" \
            > "$LOGS/chunk_${LO}_${HI}_task_${i}.log" 2>&1 &
        pids+=($!)
    done
    fail=0
    for i in "${!pids[@]}"; do
        wait "${pids[$i]}" || { echo "worker $i failed: see $LOGS/chunk_${LO}_${HI}_task_${i}.log" >&2; fail=1; }
    done
    [ "$fail" -eq 0 ] || exit 1
    n_parts=$(ls "$PARTS"/part_${LO}_${HI}_task*.parquet 2>/dev/null | wc -l)
    [ "$n_parts" -eq "$N_TASKS" ] || { echo "expected $N_TASKS parts, found $n_parts -- not deleting raw" >&2; exit 1; }

    if [ "$DELETE_RAW" -eq 1 ]; then
        echo "=== chunk [$LO,$HI]: deleting raw shards ==="
        for cfg in zee_pu200_particles zee_pu200_calo_hits; do
            for idx in $(seq "$LO" "$HI"); do
                printf -v pad "%05d" "$idx"
                rm -f "$RAW"/CERN__ColliderML-Release-1/$cfg/data/$cfg/train-${pad}-of-*.parquet
            done
        done
    fi
    touch "$MARK"
    LO=$((HI + 1))
done
echo "GREEN: all chunks complete. Parts in $PARTS -- ready to merge."