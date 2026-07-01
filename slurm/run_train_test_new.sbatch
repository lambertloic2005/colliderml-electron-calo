#!/bin/bash

# ============================================================
# SLURM resource request
# ============================================================

#SBATCH --job-name=trainTestEtaPhiPtZ0Charge
#SBATCH --output=%x.o%j
#SBATCH --error=%x.e%j
#SBATCH --time=4:00:00
#SBATCH --mem=40G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --partition=gpua100
#SBATCH --export=ALL
#SBATCH --exclude=ruche-gpu16
#SBATCH --tmp=120G

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
# User/job configuration
# ============================================================

CONDA_ENV_NAME="colliderml"
REGION="${REGION:-barrel}"

RUN_NAME="eta_phi_pt_z0_charge_${REGION}_$(date +%Y%m%d_%H%M%S)"

TRAIN_SCRIPT="scripts/train_all.py"
TEST_SCRIPT="scripts/test_all.py"

# Folder where sbatch was submitted from.
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"

# Safety fallback in case the job was submitted from the wrong place.
if [[ ! -f "$PROJECT_DIR/pyproject.toml" ]]; then
    PROJECT_DIR="$HOME/colliderml-electron-calo"
fi

SOURCE_Z0_CHARGE_FILE="$PROJECT_DIR/data/electrons/eta_phi_pt_z0_charge/zee_pu200_z0_charge.parquet"
SOURCE_STATS_FILE="$PROJECT_DIR/data/electrons/eta_phi_pt_z0_charge/target_stats.json"

RUN_DIR="$PROJECT_DIR/runs/ruche_${RUN_NAME}_${SLURM_JOB_ID}"

EXPECTED_CKPT="checkpoints/ruche_eta_phi_pt_z0_charge.pt"

TEST_CKPT="checkpoints/ruche/ruche_eta_phi_pt_z0_charge.pt"
TEST_PARQUET="data/electrons/testRuche/zee_pu200_z0_charge_TEST.parquet"
TEST_STATS="data/electrons/testRuche/target_stats.json"
TEST_OUTPUT_DIR="results/ruche/${RUN_NAME}"

echo "Project directory: $PROJECT_DIR"
echo "Run output directory: $RUN_DIR"

mkdir -p "$RUN_DIR"

# ============================================================
# Check required files before doing anything expensive
# ============================================================

if [[ ! -f "$PROJECT_DIR/$TRAIN_SCRIPT" ]]; then
    echo "ERROR: Could not find training script:"
    echo "$PROJECT_DIR/$TRAIN_SCRIPT"
    exit 1
fi

if [[ ! -f "$PROJECT_DIR/$TEST_SCRIPT" ]]; then
    echo "ERROR: Could not find testing script:"
    echo "$PROJECT_DIR/$TEST_SCRIPT"
    exit 1
fi

if [[ ! -f "$SOURCE_Z0_CHARGE_FILE" ]]; then
    echo "ERROR: Could not find z0/charge parquet file:"
    echo "$SOURCE_Z0_CHARGE_FILE"
    exit 1
fi

if [[ ! -f "$SOURCE_STATS_FILE" ]]; then
    echo "ERROR: Could not find target stats file:"
    echo "$SOURCE_STATS_FILE"
    exit 1
fi

# ============================================================
# Create temporary working copy
# ============================================================

if [[ -n "${TMPDIR:-}" ]]; then
    JOB_WORKDIR="$TMPDIR/colliderml-electron-calo_${RUN_NAME}_${SLURM_JOB_ID}"
else
    JOB_WORKDIR="$PROJECT_DIR/.slurm_work_${RUN_NAME}_${SLURM_JOB_ID}"
fi

echo "Temporary work directory: $JOB_WORKDIR"
mkdir -p "$JOB_WORKDIR"

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
    --exclude "notebooks" \
    --exclude "tests" \
    --exclude "data/electrons/" \
    "$PROJECT_DIR/" "$JOB_WORKDIR/"

mkdir -p "$JOB_WORKDIR/data/electrons"

echo "Copying z0/charge data and target stats..."
rsync -av "$SOURCE_Z0_CHARGE_FILE" "$JOB_WORKDIR/data/electrons/electrons_z0_charge.parquet"
rsync -av "$SOURCE_STATS_FILE" "$JOB_WORKDIR/data/electrons/target_stats.json"

# The training script currently expects:
#   data/electrons/electrons.parquet
cd "$JOB_WORKDIR/data/electrons"
ln -sf electrons_z0_charge.parquet electrons.parquet
cd "$JOB_WORKDIR"

echo "Data files available to the job:"
ls -lh data/electrons/

# ============================================================
# Copy outputs back even if the job fails
# ============================================================

sync_back() {
    exit_code=$?

    echo "========================================"
    echo "Copying outputs back to:"
    echo "$RUN_DIR"
    echo "Original exit code: $exit_code"
    echo "========================================"

    mkdir -p "$RUN_DIR"

    if [[ -d "$JOB_WORKDIR/checkpoints" ]]; then
        mkdir -p "$RUN_DIR/checkpoints"
        rsync -av "$JOB_WORKDIR/checkpoints/" "$RUN_DIR/checkpoints/" || true
    fi

    if [[ -d "$JOB_WORKDIR/results" ]]; then
        mkdir -p "$RUN_DIR/results"
        rsync -av "$JOB_WORKDIR/results/" "$RUN_DIR/results/" || true
    fi

    if [[ -d "$JOB_WORKDIR/wandb" ]]; then
        mkdir -p "$RUN_DIR/wandb"
        rsync -av "$JOB_WORKDIR/wandb/" "$RUN_DIR/wandb/" || true
    fi

    if [[ -f "$JOB_WORKDIR/train_${RUN_NAME}.log" ]]; then
        cp "$JOB_WORKDIR/train_${RUN_NAME}.log" "$RUN_DIR/" || true
    fi

    if [[ -f "$JOB_WORKDIR/test_${RUN_NAME}.log" ]]; then
        cp "$JOB_WORKDIR/test_${RUN_NAME}.log" "$RUN_DIR/" || true
    fi

    if [[ -f "$JOB_WORKDIR/$TEST_SCRIPT" ]]; then
        cp "$JOB_WORKDIR/$TEST_SCRIPT" "$RUN_DIR/test_eta_phi_pt_z0_charge.used.py" || true
    fi

    echo "Saved output files:"
    find "$RUN_DIR" -maxdepth 4 -type f | sort || true

    if [[ "$JOB_WORKDIR" == "$PROJECT_DIR"/.slurm_work_* ]]; then
        rm -rf "$JOB_WORKDIR" || true
    fi

    echo "Job finished at: $(date)"
    exit "$exit_code"
}

trap sync_back EXIT

# ============================================================
# Load Python environment
# ============================================================

module purge
module load anaconda3/2023.09-0/none-none

# Conda activation can reference unset variables, so temporarily
# disable strict unset-variable checking.
set +u
source activate "$CONDA_ENV_NAME"
set -u

echo "Python executable:"
which python

echo "Python version:"
python --version

python -m pip install -e . --no-deps

export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# Prefer online W&B if available, but do not crash only because W&B is not logged in.
export WANDB_MODE="${WANDB_MODE:-online}"

if [[ "$WANDB_MODE" == "online" && -z "${WANDB_API_KEY:-}" && ! -f "$HOME/.netrc" ]]; then
    echo "WARNING: No WANDB_API_KEY or ~/.netrc found. Switching WANDB_MODE=offline."
    export WANDB_MODE=offline
fi

# ============================================================
# Check GPU and code state
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

cd "$JOB_WORKDIR"

echo "Training script:"
ls -lh "$TRAIN_SCRIPT"

echo "Testing script:"
ls -lh "$TEST_SCRIPT"

# ============================================================
# Preflight data validation
# ============================================================

echo "========================================"
echo "Preflight data validation"
echo "========================================"

python - <<'PY'
import json
from pathlib import Path
import polars as pl

parquet_path = Path("data/electrons/electrons.parquet")
stats_path = Path("data/electrons/target_stats.json")

required_columns = {
    "split",

    "truth_energy",
    "truth_px",
    "truth_py",
    "truth_pz",
    "truth_eta",
    "truth_phi",
    "truth_log_pt",
    "truth_z0",
    "truth_d0",
    "truth_charge",

    "cell_x",
    "cell_y",
    "cell_z",
    "cell_e_calibrated",
    "cell_eta",
    "cell_phi",
    "cell_detector",
}

required_stats = {
    "truth_energy",
    "truth_px",
    "truth_py",
    "truth_pz",
    "truth_eta",
    "truth_phi",
    "truth_log_pt",
    "truth_z0",
    "truth_d0",
}

df_head = pl.read_parquet(parquet_path, n_rows=1)
available_columns = set(df_head.columns)
missing_columns = sorted(required_columns - available_columns)

if missing_columns:
    raise RuntimeError(
        "Parquet file is missing required columns for z0/charge training: "
        + ", ".join(missing_columns)
    )

stats = json.loads(stats_path.read_text())

missing_stats = sorted(required_stats - set(stats))
if missing_stats:
    raise RuntimeError(
        "target_stats.json is missing required target stats: "
        + ", ".join(missing_stats)
    )

bad_stats = sorted(
    key for key in required_stats
    if key in stats and ("mean" not in stats[key] or "std" not in stats[key])
)

if bad_stats:
    raise RuntimeError(
        "These target_stats entries are missing mean/std: "
        + ", ".join(bad_stats)
    )

if missing_stats:
    raise RuntimeError(
        "target_stats.json is missing required target stats: "
        + ", ".join(missing_stats)
    )

split_df = pl.read_parquet(parquet_path, columns=["split", "truth_eta"])

for split in ["train", "val", "test"]:
    n = split_df.filter(
        (pl.col("split") == split) & (pl.col("truth_eta").abs() <= 3)
    ).height
    print(f"{split} rows after |truth_eta| <= 3 cut: {n}")
    if n == 0:
        raise RuntimeError(f"No rows left for split={split} after eta cut.")

print("Preflight validation passed.")
PY

# ============================================================
# Train
# ============================================================

echo "========================================"
echo "START TRAINING: $RUN_NAME"
echo "Date: $(date)"
echo "========================================"

python -u "$TRAIN_SCRIPT" 2>&1 | tee "train_${RUN_NAME}.log"

echo "========================================"
echo "TRAINING FINISHED"
echo "Date: $(date)"
echo "========================================"

echo "Checkpoint files after training:"
find checkpoints -maxdepth 3 -type f -print -exec ls -lh {} \; || true

if [[ ! -f "$EXPECTED_CKPT" ]]; then
    echo "ERROR: Expected checkpoint was not created:"
    echo "$EXPECTED_CKPT"
    echo "Available checkpoints:"
    find checkpoints -type f || true
    exit 1
fi

# ============================================================
# Prepare test paths
# ============================================================

mkdir -p checkpoints/ruche
mkdir -p data/electrons/testRuche

cp "$EXPECTED_CKPT" "$TEST_CKPT"
cp data/electrons/electrons_z0_charge.parquet "$TEST_PARQUET"
cp data/electrons/target_stats.json "$TEST_STATS"

# The current test script hardcodes older Ruche paths.
# Patch only the temporary working copy so this run uses proper z0/charge names.
export TEST_CKPT
export TEST_PARQUET
export TEST_STATS
export TEST_OUTPUT_DIR

python - <<'PY'
import os
import re
from pathlib import Path

script = Path("scripts/test_eta_phi_pt_z0_charge.py")
text = script.read_text()

# Patch by variable name, not by the literal path string.
subs = {
    "checkpoint_path": os.environ["TEST_CKPT"],
    "parquet_path":    os.environ["TEST_PARQUET"],
    "stats_path":      os.environ["TEST_STATS"],
    "output_dir":      os.environ["TEST_OUTPUT_DIR"],
}
for var, newpath in subs.items():
    text, n = re.subn(rf'{var} = Path\("[^"]*"\)',
                      f'{var} = Path("{newpath}")', text)
    if n == 0:
        raise RuntimeError(f"Could not find a Path() assignment to patch for: {var}")

if os.environ["TEST_CKPT"] not in text:
    raise RuntimeError("Test script checkpoint path was not patched.")
if os.environ["TEST_PARQUET"] not in text:
    raise RuntimeError("Test script parquet path was not patched.")
if os.environ["TEST_OUTPUT_DIR"] not in text:
    raise RuntimeError("Test script output directory was not patched.")
if os.environ["TEST_STATS"] not in text:
    raise RuntimeError("Test script stats path was not patched.")

script.write_text(text)
print(f"Patched {script} for this SLURM run.")
PY

echo "Test checkpoint:"
ls -lh "$TEST_CKPT"

echo "Test parquet:"
ls -lh "$TEST_PARQUET"

echo "Test stats:"
ls -lh "$TEST_STATS"

# ============================================================
# Test
# ============================================================

echo "========================================"
echo "START TESTING: $RUN_NAME"
echo "Date: $(date)"
echo "========================================"

python -u "$TEST_SCRIPT" 2>&1 | tee "test_${RUN_NAME}.log"

echo "========================================"
echo "TESTING FINISHED"
echo "Date: $(date)"
echo "========================================"

echo "Result files after testing:"
find results -maxdepth 5 -type f -print -exec ls -lh {} \; || true

echo "========================================"
echo "SUCCESS"
echo "Outputs will be copied to:"
echo "$RUN_DIR"
echo "========================================"