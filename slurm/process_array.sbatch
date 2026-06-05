#!/bin/bash
# ===========================================================================
# process_array.sbatch -- parallel supervised-DBSCAN processing on Ruche.
#
# Each array task processes a strided slice of the downloaded shards
# (shards[task_id :: n_tasks]) and writes its own part parquet. A separate
# merge job (merge.sbatch) then concatenates the parts, assigns splits, and
# writes target_stats.json.
#
# Parallelism = the size of the array. To use more/fewer tasks, change ONLY
# the --array range below; --n-tasks is taken from $SLURM_ARRAY_TASK_COUNT so
# the two can never disagree. Keep --array contiguous and 0-based (0-(N-1)).
#
# Submit with:
#   ARRAY_JOB=$(sbatch --parsable slurm/process_array.sbatch)
#   sbatch --dependency=afterok:$ARRAY_JOB slurm/merge.sbatch
# ===========================================================================
#SBATCH --job-name=elec_dbscan
#SBATCH --output=%x.%A_%a.out      # %A = array job id, %a = task id
#SBATCH --partition=cpu_med        # 4 h walltime, no per-user job cap
#SBATCH --time=03:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G           # headroom for the per-electron NxN DBSCAN matrix
#SBATCH --array=0-49               # 50 parallel tasks; edit this to change parallelism

set -eo pipefail

# ---- paths (edit if yours differ) -----------------------------------------
REPO="$HOME/colliderml-electron-calo"
RAW="$WORKDIR/raw"                 # where you rsync'd the raw shards
PARTS="$WORKDIR/processed/parts"   # per-task outputs
ENV_NAME="colliderml"              # conda env built on the login node
mkdir -p "$PARTS"

# ---- environment ----------------------------------------------------------
module purge
module load anaconda3/2023.09-0/none-none
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

export COLLIDERML_DATA_DIR="$RAW"  # so build_electron_table reads from $RAW
export HF_HOME="$WORKDIR/hf"
export PYTHONUNBUFFERED=1

cd "$REPO"

echo "host=$(hostname) task=$SLURM_ARRAY_TASK_ID/$SLURM_ARRAY_TASK_COUNT raw=$RAW"

# ---- this task's slice of shards ------------------------------------------
python -u scripts/fetch_and_cluster.py --stage process \
    --channel zee --pileup pu200 \
    --data-dir "$RAW" \
    --task-id "$SLURM_ARRAY_TASK_ID" \
    --n-tasks "$SLURM_ARRAY_TASK_COUNT" \
    --out "$PARTS/part_${SLURM_ARRAY_TASK_ID}.parquet"
