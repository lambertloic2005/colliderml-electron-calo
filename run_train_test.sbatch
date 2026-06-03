#!/bin/bash

# ============================================================
# SLURM resource request
# ============================================================

#SBATCH --job-name=trainTestElectronDBSCAN
#SBATCH --output=%x.o%j
#SBATCH --error=%x.e%j
#SBATCH --time=24:00:00
#SBATCH --mem=60G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpua100
#SBATCH --export=ALL
#SBATCH --exclude=ruche-gpu16
#SBATCH --tmp=120G

# Stop the script if something fails, if a variable is missing,
# or if part of a command pipeline fails.
set -euo pipefail

echo "========================================"
echo "Job started"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Submitted from: ${SLURM_SUBMIT_DIR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not set}"
echo "========================================"


# ============================================================
# 1. Define important paths
# ============================================================

# This should be your repo folder.
# Usually SLURM_SUBMIT_DIR is the folder where you ran sbatch.
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$HOME/colliderml-electron-calo}"

# Safety fallback in case you submit from the wrong folder.
if [[ ! -f "$PROJECT_DIR/pyproject.toml" ]]; then
    PROJECT_DIR="$HOME/colliderml-electron-calo"
fi

# Name of your conda environment on Ruche.
CONDA_ENV_NAME="colliderml-electron-calo"

# Main DBSCAN parquet file.
SOURCE_DBSCAN_FILE="$PROJECT_DIR/data/electrons/electrons_dbscan.parquet"

# Target normalization file.
SOURCE_STATS_FILE="$PROJECT_DIR/data/electrons/target_stats.json"

# Final output folder for this specific job.
RUN_DIR="$PROJECT_DIR/runs/ruche_eta_phi_pt_dbscan_${SLURM_JOB_ID}"

mkdir -p "$RUN_DIR"

echo "Project directory: $PROJECT_DIR"
echo "Run output directory: $RUN_DIR"


# ============================================================
# 2. Check that required data files exist
# ============================================================

if [[ ! -f "$SOURCE_DBSCAN_FILE" ]]; then
    echo "ERROR: Could not find DBSCAN parquet file:"
    echo "$SOURCE_DBSCAN_FILE"
    exit 1
fi

if [[ ! -f "$SOURCE_STATS_FILE" ]]; then
    echo "ERROR: Could not find target stats file:"
    echo "$SOURCE_STATS_FILE"
    exit 1
fi


# ============================================================
# 3. Create a temporary working copy
# ============================================================
# The job runs in a temporary copy of the repo.
# This avoids modifying your real repo and lets us use fast local storage.

if [[ -n "${TMPDIR:-}" ]]; then
    WORKDIR="$TMPDIR/colliderml-electron-calo"
else
    WORKDIR="$PROJECT_DIR/.slurm_work_${SLURM_JOB_ID}"
fi

echo "Temporary work directory: $WORKDIR"
mkdir -p "$WORKDIR"

echo "Copying repo code to temporary work directory..."

rsync -a --delete \
    --exclude ".git" \
    --exclude ".venv" \
    --exclude "__pycache__" \
    --exclude ".pytest_cache" \
    --exclude "wandb" \
    --exclude "runs" \
    --exclude "results" \
    --exclude "checkpoints" \
    --exclude "data/electrons/*.parquet" \
    "$PROJECT_DIR/" "$WORKDIR/"

mkdir -p "$WORKDIR/data/electrons"

echo "Copying DBSCAN data and target stats..."
rsync -av "$SOURCE_DBSCAN_FILE" "$WORKDIR/data/electrons/electrons_dbscan.parquet"
rsync -av "$SOURCE_STATS_FILE" "$WORKDIR/data/electrons/target_stats.json"

# Your current training script expects data/electrons/electrons.parquet.
# This link makes electrons.parquet point to electrons_dbscan.parquet.
cd "$WORKDIR/data/electrons"
ln -sf electrons_dbscan.parquet electrons.parquet

cd "$WORKDIR"

echo "Data files available to the job:"
ls -lh data/electrons/


# ============================================================
# 4. Copy outputs back when the job ends
# ============================================================
# This function runs automatically at the end, even if training crashes.
# It saves logs, checkpoints, plots, metrics, and W&B files.

sync_back() {
    exit_code=$?

    echo "========================================"
    echo "Copying outputs back to:"
    echo "$RUN_DIR"
    echo "Original exit code: $exit_code"
    echo "========================================"

    mkdir -p "$RUN_DIR"

    if [[ -d "$WORKDIR/checkpoints" ]]; then
        mkdir -p "$RUN_DIR/checkpoints"
        rsync -av "$WORKDIR/checkpoints/" "$RUN_DIR/checkpoints/" || true
    fi

    if [[ -d "$WORKDIR/results" ]]; then
        mkdir -p "$RUN_DIR/results"
        rsync -av "$WORKDIR/results/" "$RUN_DIR/results/" || true
    fi

    if [[ -d "$WORKDIR/wandb" ]]; then
        mkdir -p "$RUN_DIR/wandb"
        rsync -av "$WORKDIR/wandb/" "$RUN_DIR/wandb/" || true
    fi

    if [[ -f "$WORKDIR/train.log" ]]; then
        cp "$WORKDIR/train.log" "$RUN_DIR/train.log" || true
    fi

    if [[ -f "$WORKDIR/test.log" ]]; then
        cp "$WORKDIR/test.log" "$RUN_DIR/test.log" || true
    fi

    echo "Saved output files:"
    find "$RUN_DIR" -maxdepth 3 -type f | sort || true

    # Clean up fallback workdir if it was created inside the repo.
    if [[ "$WORKDIR" == "$PROJECT_DIR"/.slurm_work_* ]]; then
        rm -rf "$WORKDIR" || true
    fi

    echo "Job finished at: $(date)"
    exit "$exit_code"
}

trap sync_back EXIT


# ============================================================
# 5. Load Python environment
# ============================================================

module purge
module load anaconda3/2023.09-0/none-none

source activate "$CONDA_ENV_NAME"

echo "Python executable:"
which python

echo "Python version:"
python --version

# Install the repo package from this temporary copy.
# --no-deps avoids reinstalling dependencies.
python -m pip install -e . --no-deps

# Use non-interactive plotting for saved figures.
export MPLBACKEND=Agg

# Offline W&B means logs are saved locally.
# Change to online if you already ran wandb login on Ruche.
export WANDB_MODE=offline

# Match CPU thread usage to the CPUs requested from SLURM.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# Print Python output immediately into the log.
export PYTHONUNBUFFERED=1


# ============================================================
# 6. Check GPU and code state
# ============================================================

echo "Checking GPU:"
nvidia-smi || true

echo "Checking PyTorch CUDA:"
python - <<'PY'
import torch
print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

echo "========================================"
echo "Code state"
echo "========================================"

if [[ -d "$PROJECT_DIR/.git" ]]; then
    cd "$PROJECT_DIR"
    echo "Branch:"
    git branch --show-current || true
    echo "Latest commit:"
    git log -1 --oneline || true
fi

cd "$WORKDIR"

echo "Training script:"
ls -lh scripts/train_eta_phi_pt_angular_features.py

echo "Testing script:"
ls -lh scripts/test_eta_phi_pt_angular_features.py


# ============================================================
# 7. Train the model
# ============================================================

echo "========================================"
echo "START TRAINING"
echo "Date: $(date)"
echo "========================================"

python -u scripts/train_eta_phi_pt_angular_features.py 2>&1 | tee train.log

echo "========================================"
echo "TRAINING FINISHED"
echo "Date: $(date)"
echo "========================================"

echo "Checkpoint files after training:"
find checkpoints -maxdepth 2 -type f -print -exec ls -lh {} \; || true


# ============================================================
# 8. Make sure the trained checkpoint exists
# ============================================================

EXPECTED_CKPT="checkpoints/eta_phi_pt_conv_dbscan_energy.pt"

if [[ ! -f "$EXPECTED_CKPT" ]]; then
    echo "ERROR: Expected checkpoint was not created:"
    echo "$EXPECTED_CKPT"
    echo "Available checkpoints:"
    find checkpoints -type f || true
    exit 1
fi


# ============================================================
# 9. Test the trained model
# ============================================================

echo "========================================"
echo "START TESTING"
echo "Date: $(date)"
echo "========================================"

python -u scripts/test_eta_phi_pt_angular_features.py 2>&1 | tee test.log

echo "========================================"
echo "TESTING FINISHED"
echo "Date: $(date)"
echo "========================================"

echo "Result files after testing:"
find results -maxdepth 4 -type f -print -exec ls -lh {} \; || true


# ============================================================
# 10. End of job
# ============================================================

echo "========================================"
echo "SUCCESS"
echo "Outputs will be copied to:"
echo "$RUN_DIR"
echo "========================================"