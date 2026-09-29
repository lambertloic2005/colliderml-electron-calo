#!/bin/bash
# ===========================================================================
# merge.sbatch -- combine the per-task part parquets into one dataset, assign
# event-level splits, and write target_stats.json. Run AFTER the array via:
#   sbatch --dependency=afterok:$ARRAY_JOB slurm/merge.sbatch
# The split is computed here, on the full merged set, so train/val/test are
# globally consistent across all tasks.
# ===========================================================================
#SBATCH --job-name=elec_merge
#SBATCH --output=%x.%j.out
#SBATCH --partition=cpu_short      # 1 h walltime; the merge is light
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G          # holds all parts in memory for the concat

set -eo pipefail

REPO="$HOME/colliderml-electron-calo"
PARTS="$WORKDIR/processed/parts"
OUT="$WORKDIR/processed/zee_pu200_supervisedDBSCAN.parquet"
ENV_NAME="colliderml"

module purge
module load anaconda3/2023.09-0/none-none
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
export PYTHONUNBUFFERED=1

cd "$REPO"

python -u scripts/fetch_and_cluster.py --stage merge \
    --parts-glob "$PARTS/part_*.parquet" \
    --out "$OUT"

echo "Done. Final dataset + target_stats.json in $(dirname "$OUT")"
