# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

ML experiments for reconstructing prompt-electron kinematics (eta, phi, pT, z0) from CERN ColliderML calorimeter showers. The model predicts **anchored residuals** — small corrections on top of physics-motivated anchors (energy-weighted centroid, pointing fit) rather than absolute values. Python 3.10–3.11 required.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Workflow commands

**Build DBSCAN-cleaned electron table** (skip if `data/electrons/electrons_dbscan.parquet` exists):
```bash
python scripts/build_electron_dataset.py \
  --channel zee --pileup pu200 --mask dbscan \
  --n-events 50 --out data/electrons/electrons_dbscan.parquet
```

**Compute train-split normalization statistics** (run once after building the table):
```bash
python -m colliderml_electron.stats
```

**Train** (barrel by default; set `REGION=endcap` for endcap):
```bash
REGION=barrel python scripts/train_eta_phi_pt_z0_charge.py
REGION=endcap  python scripts/train_eta_phi_pt_z0_charge.py
```

**Evaluate**:
```bash
python scripts/test_eta_phi_pt_z0_charge.py
```

Checkpoints are saved to `checkpoints/ruche_eta_phi_pt_z0_charge.pt`. Evaluation outputs (plots, `test_metrics.json`) land in `results/`. Training is logged to Weights & Biases under project `colliderml-electron-calo`.

## Architecture

### Data pipeline (`src/colliderml_electron/`)

| File | Role |
|---|---|
| `io.py` | Load ColliderML parquet/HF data, select prompt electrons |
| `coords.py` | Coordinate transforms (centroid frame rotation, etc.) |
| `calibration.py` | Cell energy calibration |
| `cluster.py` | DBSCAN shower cleaning |
| `cluster_pipeline.py` | Per-electron cluster feature extraction |
| `pipeline.py` | End-to-end data building pipeline |
| `dataset.py` | PyTorch `Dataset` + `DataLoader` factory (`make_loader`) |
| `splits.py` | Train/val/test split assignment |
| `stats.py` | Compute target normalization statistics (train-only) |
| `model.py` | Model classes (see below) |
| `embedding.py` | `FourierPositionalEncoding` for (x, y, z) cell positions |
| `encoder.py` | `CellEncoder`: per-cell transformer over Fourier-embedded positions + high-level features |
| `resolution.py`, `plots.py` | Evaluation helpers |

### Model (`model.py`)

Two model classes, selected via `config["model_type"]`:

- **`ConvCaloRegressor`** (default, `"conv"`): transformer encoder → Conv1d stack along energy-sorted sequence → masked mean+max pool → MLP head. Better aggregation than flat concat.
- **`ConcatCaloRegressor`** (`"concat"`): transformer encoder → flatten all cell vectors → MLP head. Baseline.

Both share the same front end: top-`max_cells` selection by energy → `FourierPositionalEncoding` (3D position) + 21-dim high-level features → `CellEncoder` (multi-head self-attention).

### Model inputs

- **Per-cell** (12-dim): `(x, y, z)` Fourier-embedded, `log(E)`, `eta`, `sin/cos(phi - phi_centroid)`, `theta`, `cos(theta)`, 6-dim detector one-hot.
- **Cluster-level** (9-dim, broadcast to each cell): log total energy, log total ET, log n-cells, std/skew in phi and eta, `z0` pointing anchor, `dz/dr` slope.

### Outputs and loss (`train_eta_phi_pt_z0_charge.py`)

The model outputs 5 values: `[delta_eta, delta_phi, delta_logpt, delta_z0, charge_logit]`.

- `delta_phi` is a **single signed bend** — its sign encodes the charge deflection direction. In the current script the model also outputs a `charge_logit` (binary classification, positron vs electron), whereas the README describes an older two-head scheme. The train script's `KinematicLoss` implements the single-head + classifier design.
- Loss: Huber on each regression residual + BCE for charge. The four regression tasks are combined with **homoscedastic uncertainty weighting** (learned `log_sigma` per task). Charge loss is added with a fixed `charge_weight`.
- Anchors: `eta_centroid`, `phi_centroid`, `log_sum_et`, `z0_anchor` are passed as batch fields, not model inputs directly — the loss function uses them to decode predictions.

### Barrel / endcap split

The `REGION` environment variable (`"barrel"` or `"endcap"`) controls the eta acceptance cut applied in `make_loader`:

- barrel: `|eta| < 1.7`
- endcap: `1.3 < |eta| < 3` (overlap region 1.3–1.7 is included in both during training)

### SLURM (HPC cluster `ruche`)

Batch scripts are in `slurm/`. The main training job is `slurm/run_train_test.sbatch` / `run_train_test_new.sbatch`.

## Key data files

| Path | Content |
|---|---|
| `data/electrons/electrons_dbscan.parquet` | Main training table (one row per electron) |
| `data/electrons/target_stats.json` | Train-split mean/std for each target column |
| `checkpoints/ruche_eta_phi_pt_z0_charge.pt` | Saved model checkpoint |

Do not recompute `target_stats.json` on the validation or test set.
