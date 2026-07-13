#!/usr/bin/env bash
# ===========================================================================
# process_all_atlas.sh -- parallel supervised-DBSCAN processing on a plain
# (non-SLURM) machine. The atlas equivalent of slurm/process_array.sbatch:
# N background workers, each processing the strided shard slice
# shards[i :: N_TASKS] via fetch_and_cluster.py --stage process, writing one
# part parquet each. Merge/splits/stats are done AFTERWARDS (see run notes) so
# train/val/test stay globally consistent.
#
# Sizing N_TASKS: each worker holds one particles shard + one calo_hits shard
# in memory plus the per-electron DBSCAN distance matrix (~8 GB budget per
# worker matched the Ruche array sizing). Choose:
#     N_TASKS <= min(free CPU cores, free_RAM_GB / 8)
#
# Usage (from the repo root, inside the colliderml env, ideally under tmux):
#     N_TASKS=16 bash scripts/process_all_atlas.sh
# All knobs are env-overridable; defaults below match the atlas layout used
# for v1 (see scripts/verify_merged_parquet.py) with a NEW processed_v2 dir
# so the v1 dataset -- still needed for split harmonization and as the
# champion baseline's dataset -- is never touched.
# ===========================================================================
set -eo pipefail

: "${REPO:=$HOME/colliderml-electron-calo}"
: "${RAW:=/data/atlas/lambert/colliderml}"            # shard cache root (= COLLIDERML_DATA_DIR)
: "${PARTS:=/data/atlas/lambert/processed_v2/parts}"  # per-task outputs (v2 -- do NOT reuse v1 dir)
: "${N_TASKS:=16}"
LOGS="$PARTS/logs"

mkdir -p "$PARTS" "$LOGS"
export COLLIDERML_DATA_DIR="$RAW"   # pipeline.build_electron_table reads shards from here
export PYTHONUNBUFFERED=1
cd "$REPO"

# ---- green-flag preflight (fail fast, before spawning 16 workers) ----------
python -c "from colliderml_electron.pipeline import build_electron_table" \
    || { echo "ERROR: colliderml_electron not importable -- run 'pip install -e .' first." >&2; exit 1; }
n_calo=$(ls "$RAW"/CERN__ColliderML-Release-1/zee_pu200_calo_hits/data/zee_pu200_calo_hits/train-*.parquet 2>/dev/null | wc -l)
n_part=$(ls "$RAW"/CERN__ColliderML-Release-1/zee_pu200_particles/data/zee_pu200_particles/train-*.parquet 2>/dev/null | wc -l)
echo "Preflight: $n_calo calo_hits shards, $n_part particles shards under $RAW"
[ "$n_calo" -gt 0 ] && [ "$n_part" -gt 0 ] \
    || { echo "ERROR: no shards found -- run the download stage first." >&2; exit 1; }

# ---- launch workers ---------------------------------------------------------
pids=()
for i in $(seq 0 $((N_TASKS - 1))); do
    python -u scripts/fetch_and_cluster.py --stage process \
        --channel zee --pileup pu200 \
        --data-dir "$RAW" --no-link \
        --task-id "$i" --n-tasks "$N_TASKS" \
        --out "$PARTS/part_${i}.parquet" \
        > "$LOGS/task_${i}.log" 2>&1 &
    pids+=($!)
done
echo "Launched $N_TASKS workers (PIDs: ${pids[*]}). Tail progress with:"
echo "    tail -f $LOGS/task_0.log"

# ---- wait and check ---------------------------------------------------------
fail=0
for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
        echo "ERROR: worker $i (PID ${pids[$i]}) exited non-zero -- see $LOGS/task_${i}.log" >&2
        fail=1
    fi
done
[ "$fail" -eq 0 ] || { echo "At least one worker failed. Fix and re-run ONLY the failed --task-id(s) before merging." >&2; exit 1; }

# A crashed worker writes no parquet (the pipeline writes once, at the end),
# so a silent shortfall here would mean silently missing shards in the merge.
n_parts=$(ls "$PARTS"/part_*.parquet 2>/dev/null | wc -l)
echo "All workers exited OK: $n_parts/$N_TASKS part files in $PARTS"
[ "$n_parts" -eq "$N_TASKS" ] \
    || { echo "ERROR: expected $N_TASKS part files, found $n_parts. Do not merge yet." >&2; exit 1; }
echo "GREEN: ready to merge."