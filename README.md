# colliderml-electron-calo

Machine-learning experiments on the calorimeter portion of the CERN ColliderML Release 1 dataset, focused on reconstructing prompt-electron kinematics from calorimeter showers.

The current workflow uses an idealized DBSCAN-cleaned per-electron dataset:

```text
data/electrons/electrons_dbscan.parquet
```

The model is trained to predict:

- electron pseudorapidity, `eta`
- electron azimuthal angle, `phi`
- electron transverse momentum, `pT`, trained through `log(pT)`

The current pT-supervised DBSCAN version uses per-cell calorimeter inputs, an energy-weighted phi centroid, and total shower-energy features.

## Project goal

The long-term goal is to study how much tracker-like information can be inferred from calorimeter showers alone.

In the current workflow, the specific goal is:

```text
DBSCAN-cleaned calorimeter shower -> truth eta, truth phi, truth pT
```

This is not yet a fully realistic detector-level reconstruction pipeline. The DBSCAN cleaning is idealized and supervised: it is used to create a cleaner per-electron dataset so that the model can be tested on how well it learns from calorimeter shower information.

## Current workflow

```text
ColliderML Release 1 data
        |
        v
Load particle and calorimeter-hit tables
        |
        v
Select prompt electrons
        |
        v
Collect calorimeter cells associated with each prompt electron
        |
        v
Apply idealized DBSCAN shower cleaning
        |
        v
Save one row per electron to electrons_dbscan.parquet
        |
        v
Assign train / validation / test splits
        |
        v
Compute target normalization statistics
        |
        v
Train eta / phi / pT model
        |
        v
Evaluate residuals and resolution plots
```

The main training table is:

```text
data/electrons/electrons_dbscan.parquet
```

This is different from the earlier cone-based table:

```text
data/electrons/electrons.parquet
```

For the current workflow, training and evaluation should use:

```text
data/electrons/electrons_dbscan.parquet
```

## What the model learns

Each row in `electrons_dbscan.parquet` corresponds to one prompt electron.

The input is a variable-length set of DBSCAN-cleaned calorimeter cells. The target is the truth prompt-electron kinematics from the ColliderML particle table.

The model does not receive truth `eta`, `phi`, or `pT` as input. Truth values are used only as supervised labels during training and evaluation.

## Model inputs

The model receives two kinds of information:

1. per-cell shower information
2. cluster-level shower information

### Per-cell position inputs

For each selected calorimeter cell, the model receives:

```text
cell_x
cell_y
cell_z
```

These describe the 3D position of the calorimeter cell.

### Per-cell high-level inputs

The model also receives angular, energy, and detector-subsystem information:

```text
log(cell_e_calibrated)
cell_eta
sin(cell_phi)
cos(cell_phi)
theta
cos(theta)
detector one-hot encoding
```

The `sin(phi)` and `cos(phi)` representation is used because `phi` is periodic.

### Cluster-level energy inputs

The current pT-improved version also gives the model global shower-energy information.

These features are computed from the DBSCAN-cleaned shower:

```text
log(total calibrated cluster energy)
log(total transverse-energy proxy)
log(number of cells)
```

These cluster-level features are broadcast to every cell, so each cell carries both local cell information and global shower information.

This matters because pT is not determined only by the local shower shape. The total energy scale of the shower is one of the most important pieces of information for predicting pT.

## Phi-centroid anchoring

The current model uses an energy-weighted phi centroid.

The centroid is computed from the DBSCAN-cleaned shower using the cell energies as weights:

```text
phi_centroid = atan2(
    weighted average of sin(cell_phi),
    weighted average of cos(cell_phi)
)
```

Instead of making the model learn absolute `phi` from scratch, the model learns the correction from the energy-weighted centroid to the true electron phi.

Conceptually:

```text
truth_phi = phi_centroid + predicted_delta_phi
```

This makes the phi task easier because the centroid is already a strong first estimate of the electron direction.

The predicted residual is represented as:

```text
cos(delta_phi)
sin(delta_phi)
```

rather than as a raw angle. This avoids the discontinuity between `-pi` and `+pi`.

## Model output

The model predicts four values:

```text
[eta, phi_cos, phi_sin, log_pt]
```

where:

```text
eta      = normalized eta prediction
phi_cos  = cos(predicted delta phi)
phi_sin  = sin(predicted delta phi)
log_pt   = normalized log(pT) prediction
```

During evaluation:

```text
predicted_delta_phi = atan2(phi_sin, phi_cos)
predicted_phi       = phi_centroid + predicted_delta_phi
predicted_pT        = exp(denormalized predicted log_pt)
```

## Loss function

The total loss combines eta, phi, and log-pT losses:

```text
total_loss = eta_loss + phi_loss + logpt_loss
```

### Eta loss

Eta is trained with mean squared error in normalized space:

```text
eta_loss = MSE(predicted_eta_norm, true_eta_norm)
```

### Phi loss

Phi is trained as a residual from the energy-weighted centroid.

First, the target residual is computed:

```text
delta_phi_target = wrapped_angle_delta(true_phi, phi_centroid)
```

Then it is represented as:

```text
target_phi_cos = cos(delta_phi_target)
target_phi_sin = sin(delta_phi_target)
```

The phi loss compares the predicted and target unit-circle vectors:

```text
phi_loss = MSE(predicted_phi_cos, target_phi_cos)
         + MSE(predicted_phi_sin, target_phi_sin)
```

This handles the periodic nature of phi correctly.

### pT loss

pT is trained as `log(pT)`:

```text
logpt_loss = MSE(predicted_logpt_norm, true_logpt_norm)
```

Using `log(pT)` makes the model focus more on fractional pT error than absolute pT error.

For small errors:

```text
delta log(pT) ≈ delta pT / pT
```

So the log-pT residual is closely related to the fractional pT resolution.

## Why total energy helps pT

For a relativistic electron, energy and momentum are closely related. The transverse momentum is:

```text
pT = p sin(theta)
```

For high-energy electrons:

```text
p ≈ E
```

so approximately:

```text
pT ≈ E sin(theta)
```

This means that the model needs information about both:

```text
the total shower energy
```

and:

```text
the shower direction
```

The energy-weighted phi centroid helps with direction, while the total cluster-energy features help with the energy scale needed for pT.

## The 128 hottest cells

The model does not necessarily pass every cell in a shower.

For batching and speed, the dataset keeps a maximum number of cells per electron, typically:

```text
max_cells = 128
```

The selected cells are the highest-energy cells from the DBSCAN-cleaned shower.

However, the global cluster-energy features are computed before this truncation. That means the model can still receive information about the full DBSCAN-cleaned shower energy even if only the 128 hottest cells are passed as detailed per-cell inputs.

## Repository structure

```text
.
├── checkpoints/                  # Saved model checkpoints
├── data/                         # Local generated data
├── docs/                         # Notes and documentation
├── notebooks/                    # Jupyter notebooks
├── results/                      # Evaluation plots and metrics
├── scripts/                      # Command-line scripts
├── src/colliderml_electron/      # Main Python package
├── pyproject.toml
├── requirements.txt
├── uv.lock
└── README.md
```

Important source files:

```text
src/colliderml_electron/io.py
src/colliderml_electron/coords.py
src/colliderml_electron/calibration.py
src/colliderml_electron/pipeline.py
src/colliderml_electron/cluster.py
src/colliderml_electron/cluster_pipeline.py
src/colliderml_electron/dataset.py
src/colliderml_electron/model.py
src/colliderml_electron/regressor.py
src/colliderml_electron/resolution.py
src/colliderml_electron/splits.py
src/colliderml_electron/stats.py
```

Important scripts:

```text
scripts/build_electron_dataset.py
scripts/build_cluster_dataset.py
scripts/train_eta_phi_pt_angular_features.py
scripts/test_eta_phi_pt_angular_features.py
scripts/plot_predictions.py
scripts/check_predictions.py
scripts/check_splits.py
scripts/diagnose_phi.py
scripts/diagnose_matching.py
```

## Setup

This project expects Python 3.10 or 3.11.

Create and activate a virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

If needed, install from the requirements file:

```bash
pip install -r requirements.txt
```

## Data

The project uses the CERN ColliderML Release 1 dataset.

The default sample is:

```text
zee_pu200
```

This corresponds to Z to ee events with high pileup.

The current generated training table is:

```text
data/electrons/electrons_dbscan.parquet
```

The target statistics file is:

```text
data/electrons/target_stats.json
```

The train, validation, and test split is stored inside the parquet table as a `split` column.

## Build the DBSCAN-cleaned electron dataset

The current workflow trains from:

```text
data/electrons/electrons_dbscan.parquet
```

If this file already exists, you do not need to rebuild it.

A typical dataset-building command is:

```bash
python scripts/build_electron_dataset.py \
  --channel zee \
  --pileup pu200 \
  --n-events 50 \
  --out data/electrons/electrons_dbscan.parquet
```

For a larger run, remove the event limit:

```bash
python scripts/build_electron_dataset.py \
  --channel zee \
  --pileup pu200 \
  --out data/electrons/electrons_dbscan.parquet
```

If your local script has an explicit DBSCAN option, use the DBSCAN mode when writing:

```text
data/electrons/electrons_dbscan.parquet
```

## Compute splits and target statistics

After creating the DBSCAN electron table, assign splits and compute target statistics.

The target statistics should include:

```text
truth_energy
truth_px
truth_py
truth_pz
truth_eta
truth_phi
truth_log_pt
```

The training script expects:

```text
data/electrons/electrons_dbscan.parquet
data/electrons/target_stats.json
```

## Train the eta/phi/pT model

Run:

```bash
python scripts/train_eta_phi_pt_angular_features.py
```

The intended training configuration is:

```text
parquet_path            = data/electrons/electrons_dbscan.parquet
target_stats_path       = data/electrons/target_stats.json
use_angular_features    = True
use_cluster_features    = True
output_dim              = 4
```

The output layout is:

```text
[eta, phi_cos, phi_sin, log_pt]
```

The checkpoint is saved under:

```text
checkpoints/
```

The current DBSCAN-energy model checkpoint is expected to be similar to:

```text
checkpoints/eta_phi_pt_conv_dbscan_energy.pt
```

## Evaluate the model

Run:

```bash
python scripts/test_eta_phi_pt_angular_features.py
```

The evaluation script should use:

```text
checkpoint_path = checkpoints/eta_phi_pt_conv_dbscan_energy.pt
parquet_path    = data/electrons/electrons_dbscan.parquet
stats_path      = data/electrons/target_stats.json
```

Evaluation outputs are saved under:

```text
results/
```

Important plots include:

```text
expected_vs_predicted_eta.png
expected_vs_predicted_phi.png
expected_vs_predicted_pt.png
residuals_eta.png
residuals_phi.png
residuals_pt_rel.png
```

The main metrics are saved in:

```text
test_metrics.json
```

## Important evaluation quantities

### Eta residual

```text
eta_residual = predicted_eta - true_eta
```

This measures how far the predicted pseudorapidity is from the truth pseudorapidity.

### Phi residual

```text
phi_residual = wrapped_angle_delta(predicted_phi, true_phi)
```

This is measured in radians.

The residual is wrapped so that predictions near `+pi` and `-pi` are treated correctly.

### pT relative residual

```text
pt_rel_residual = (predicted_pT - true_pT) / true_pT
```

This is the fractional pT error.

For example:

```text
pt_rel_residual = 0.10
```

means the model predicted pT about 10 percent too high.

```text
pt_rel_residual = -0.10
```

means the model predicted pT about 10 percent too low.

### Log-pT residual

```text
logpt_residual = predicted_logpt - true_logpt
```

This is often a cleaner estimate of fractional pT resolution, especially when low-pT electrons cause large relative residuals.

## Current status

Implemented:

- ColliderML data loading
- prompt-electron selection
- calorimeter cell extraction
- idealized DBSCAN-cleaned electron table generation
- `electrons_dbscan.parquet` workflow
- train / validation / test splitting
- target normalization statistics
- angular per-cell features
- energy-weighted phi-centroid anchoring
- cluster-level total-energy features
- eta / phi / log-pT training
- convolutional calorimeter regressor option
- residual and Gaussian-resolution evaluation plots
- Weights & Biases logging

Current active questions:

- how much pT information is recoverable from calorimeter showers alone
- how strongly the model relies on total cluster energy
- how much phi prediction improves from the energy-weighted phi centroid
- how performance changes as true electron pT changes
- how idealized DBSCAN cleaning compares to more realistic clustering methods

## License

MIT
