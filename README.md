# colliderml-electron-calo

Machine-learning experiments on the calorimeter portion of the CERN ColliderML
Release 1 dataset, focused on reconstructing prompt-electron kinematics from
calorimeter showers.

The current workflow uses an idealized DBSCAN-cleaned per-electron dataset:

```text
data/electrons/electrons_dbscan.parquet
```

The model is trained to predict four physics quantities per electron:

- pseudorapidity, `eta`
- azimuthal angle, `phi`, predicted under **two charge hypotheses** (electron and
  positron), because the magnetic field displaces the shower differently for the
  two charges
- transverse momentum, `pT`, trained through `log(pT)`
- longitudinal impact parameter, `z0`

Every quantity is predicted as a **residual from a physics-motivated anchor**
rather than from scratch (see "Anchored residual predictions" below).

## Project goal

The long-term goal is to study how much tracker-like information can be inferred
from calorimeter showers alone. The four targets above span both the directional
information a calorimeter measures well (`eta`, `phi`) and the
harder, more tracker-like information (`pT`, and especially the longitudinal
impact parameter `z0`).

In the current workflow, the specific goal is:

```text
DBSCAN-cleaned calorimeter shower -> truth eta, truth phi, truth pT, truth z0
```

This is not yet a fully realistic detector-level reconstruction pipeline. The
DBSCAN cleaning is idealized and supervised: it is used to create a cleaner
per-electron dataset so the model can be tested on how well it learns from
calorimeter shower information.

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
Save one row per electron (with truth eta/phi/log_pt/z0/charge)
        |
        v
Assign train / validation / test splits
        |
        v
Compute target normalization statistics
        |
        v
Train eta / phi / pT / z0 model (two phi charge heads)
        |
        v
Evaluate residuals and resolution plots
```

The main training table is `data/electrons/electrons_dbscan.parquet`. This is
different from the earlier cone-based table `data/electrons/electrons.parquet`.
For the current workflow, the training and test tables should both be built with
DBSCAN cleaning so that training and evaluation see the same kind of shower.

## What the model learns

Each row in `electrons_dbscan.parquet` corresponds to one prompt electron.

The input is a variable-length set of DBSCAN-cleaned calorimeter cells. The
target is the truth prompt-electron kinematics from the ColliderML particle
table.

The model does not receive truth `eta`, `phi`, `pT`, or `z0` as input. Truth
values are used only as supervised labels during training and evaluation. The
truth charge is used during training to route the two phi heads, and during
evaluation it is used to select which phi head to read out (in a realistic
pipeline the charge would come from track matching).

## Model inputs

For batching and speed, each electron keeps at most `max_cells = 128` cells, the
highest-energy cells from the DBSCAN-cleaned shower. The cluster-level features
below are computed over the **full** cleaned shower before this truncation, so
the model still sees the total shower energy even when only the 128 hottest
cells are passed individually.

Before features are built, the event is rotated in azimuth so that the
energy-weighted phi centroid sits at `phi = 0`. This gives every shower a
canonical azimuthal frame and keeps the per-cell coordinates in a consistent
range.

### Per-cell positional inputs (Fourier-embedded)

For each selected cell, the 3D position (in the centroid-rotated frame):

```text
cell_x
cell_y
cell_z
```

### Per-cell high-level inputs

```text
log(cell_e_calibrated)
cell_eta
sin(cell_phi - phi_centroid)
cos(cell_phi - phi_centroid)
theta
cos(theta)
detector one-hot (6 subsystems)
```

The `sin`/`cos` of the centroid-relative phi is used because phi is periodic.

### Cluster-level inputs (broadcast to every cell)

Computed from the full DBSCAN-cleaned shower and appended to every cell:

```text
log(total calibrated cluster energy)
log(total transverse-energy proxy)
log(number of cells)
phi shower-shape width (std_phi)
phi shower-shape skewness (skew_phi)
eta shower-shape width (std_eta)
eta shower-shape skewness (skew_eta)
z0 pointing anchor (in metres)
shower pointing slope dz/dr
```

The phi skewness matters physically: the bremsstrahlung tail is asymmetric in a
charge-dependent way, so the sign of `skew_phi` carries information related to
the charge.

The full high-level input is 21-dimensional (12 per-cell + 9 cluster-level).

## Anchored residual predictions

Instead of regressing the absolute kinematics, the model predicts a small
**correction** to a physics-motivated anchor for each target. This makes every
task easier because the anchor is already a strong first estimate.

```text
eta    = eta_centroid + delta_eta
phi    = phi_centroid + delta_phi      (two charge hypotheses, see below)
log_pt = log_sum_et   + delta_log_pt
z0     = z0_anchor    + delta_z0
```

The anchors are:

- `eta_centroid`: the energy-weighted average of the cell pseudorapidities.
- `phi_centroid`: the energy-weighted azimuth, `atan2(<sin phi>, <cos phi>)`.
- `log_sum_et`: the log of the total transverse-energy proxy of the shower. For a
  contained electromagnetic shower this is already close to `log(pT)`.
- `z0_anchor`: an energy-weighted least-squares "pointing" fit of cell `z` versus
  cell `r`, extrapolated to `r = 0`. A straight shower points back along the
  electron flight direction to its production `z`. Because the magnetic field
  bends tracks in the transverse plane, the `r`-`z` projection used here is
  approximately unaffected by bending, which makes this a clean anchor for `z0`.

## Two charge hypotheses for phi

A charged particle is bent in azimuth by the solenoidal field, and the
calorimeter shower (including its bremsstrahlung tail) is displaced from the true
electron direction in a charge-dependent way. An electron and a positron of the
same momentum are displaced in **opposite** azimuthal directions.

The energy-weighted `phi_centroid` therefore sits on the wrong side of the truth
by an amount whose sign depends on the charge. To handle this without knowing the
charge from the calorimeter alone, the model outputs two phi corrections:

```text
delta_phi_e   phi correction under the electron  hypothesis (q = -1)
delta_phi_p   phi correction under the positron  hypothesis (q = +1)
```

The two heads are trained with mirror symmetry. Writing the true residual as
`d = wrapped_delta(true_phi, phi_centroid)`:

```text
electron head target =  d  for true electrons,  -d  for true positrons
positron head target = -d  for true electrons,   d  for true positrons
```

At evaluation, the head matching the (externally supplied) truth charge is read
out as the physics phi. The test script also reports the wrong-charge selection,
which measures the cost of a charge misassignment.

## Longitudinal impact parameter z0

The truth `z0` label is the production-vertex `z` of the prompt electron. For a
prompt particle originating on the beamline, this is the quantity a tracker would
reconstruct as the longitudinal impact parameter.

`z0` is predicted as a correction to the calorimeter pointing anchor described
above. Two baselines are reported at evaluation time:

- the anchor-only `z0` resolution (model head set to zero), and
- the beamspot prior, i.e. the RMS of `z0` with no model at all.

A useful model must beat both: it must improve on the raw pointing anchor, and it
must do better than simply predicting the center of the beamspot.

## Model output

The model produces five values, all of them **residuals** in physical units:

```text
[delta_eta, delta_phi_e, delta_phi_p, delta_log_pt, delta_z0]
```

Decoding at evaluation:

```text
pred_eta    = eta_centroid + delta_eta
pred_phi_e  = wrap(phi_centroid + delta_phi_e)        # electron hypothesis
pred_phi_p  = wrap(phi_centroid + delta_phi_p)        # positron hypothesis
pred_phi    = head selected by truth charge           # physics phi
pred_pt     = exp(log_sum_et + delta_log_pt)          # GeV
pred_z0     = z0_anchor + delta_z0                    # mm
```

## Loss function

Each target uses a Huber (smooth-L1) loss on its anchor residual, which is more
robust to shower outliers than plain MSE. The per-target transition points are:

```text
eta    : Huber(delta=0.1)
phi    : Huber(delta=0.05)   on the wrapped angular error of each head
log_pt : Huber(delta=0.2)    (~20% in pT)
z0     : Huber(delta=20 mm)
```

The phi loss averages the two heads:

```text
phi_loss = 0.5 * ( Huber(wrap(delta_phi_e - electron_head_target)) +
                   Huber(wrap(delta_phi_p - positron_head_target)) )
```

The four task losses are combined as a **weighted geometric mean** rather than a
plain sum:

```text
total_loss = exp( ( w_eta   * log(eta_loss)
                  + w_phi   * log(phi_loss)
                  + w_logpt * log(logpt_loss)
                  + w_z0    * log(z0_loss) ) / (w_eta + w_phi + w_logpt + w_z0) )
```

The geometric mean is scale-invariant, so tasks with very different natural units
(radians, eta units, log-pT, millimetres) are balanced automatically without
hand-tuning per-task weights. Note that because all four tasks share this single
combined loss, adding or removing a target slightly changes the effective
weighting of the others.

## Why total energy helps pT

For a relativistic electron, energy and momentum are closely related:

```text
pT = p sin(theta),    and for high energy   p ≈ E,   so   pT ≈ E sin(theta)
```

The model therefore needs both the total shower energy and the shower direction.
The transverse-energy anchor `log_sum_et` supplies the energy scale, while the
energy-weighted centroid supplies the direction.

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
├── slurm/                        # SLURM batch scripts (ruche)
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
src/colliderml_electron/resolution.py
src/colliderml_electron/splits.py
src/colliderml_electron/stats.py
```

Important scripts:

```text
scripts/build_electron_dataset.py
scripts/train_eta_phi_pt_z0_charge.py
scripts/test_eta_phi_pt_z0_charge.py
scripts/plot_predictions.py
scripts/check_predictions.py
scripts/check_splits.py
scripts/diagnose_phi.py
scripts/diagnose_matching.py
```

## Setup

This project expects Python 3.10 or 3.11.

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

The project uses the CERN ColliderML Release 1 dataset. The default sample is
`zee_pu200` (Z to ee events with high pileup).

The current generated training table is `data/electrons/electrons_dbscan.parquet`
and the target statistics file is `data/electrons/target_stats.json`. The train,
validation, and test split is stored inside the parquet as a `split` column.

Each electron row carries the truth labels `truth_eta`, `truth_phi`,
`truth_log_pt`, `truth_z0`, and `truth_charge`.

## Build the DBSCAN-cleaned electron dataset

If the table already exists you do not need to rebuild it. A typical command:

```bash
python scripts/build_electron_dataset.py \
  --channel zee \
  --pileup pu200 \
  --mask dbscan \
  --n-events 50 \
  --out data/electrons/electrons_dbscan.parquet
```

For a full run, drop `--n-events`. Use `--mask dbscan` so the table matches the
DBSCAN-cleaned test set.

## Compute splits and target statistics

After building the table, assign splits and compute target normalization
statistics over the **train** split only. The statistics file must contain every
target column the dataset normalizes, including `truth_log_pt` and `truth_z0`:

```bash
python -m colliderml_electron.stats
```

The same statistics file should be reused for validation and test. Do not
recompute normalization statistics on the test set.

## Train the model

```bash
python scripts/train_eta_phi_pt_z0_charge.py
```

Key training configuration:

```text
parquet_path         = data/electrons/electrons_dbscan.parquet
target_stats_path    = data/electrons/target_stats.json
use_angular_features = True
use_cluster_features = True
high_level_dim       = 21
max_cells            = 128
max_abs_eta          = 3        # acceptance cut on |truth_eta|
output_dim           = 5        # [d_eta, d_phi_e, d_phi_p, d_log_pt, d_z0]
model_type           = conv
```

The checkpoint is saved to:

```text
checkpoints/ruche_eta_phi_pt_z0_charge.pt
```

## Evaluate the model

```bash
python scripts/test_eta_phi_pt_z0_charge.py
```

The evaluation script should point at the checkpoint written by training and the
same target statistics file used for training:

```text
checkpoint_path = checkpoints/ruche_eta_phi_pt_z0_charge.pt
parquet_path    = <DBSCAN-cleaned test parquet>
stats_path      = data/electrons/target_stats.json
```

Evaluation outputs are saved under `results/`, including expected-vs-predicted
scatter plots and residual/resolution fits for each target, a phi-residual plot
split by truth charge, and `test_metrics.json`.

## Important evaluation quantities

### Eta residual

```text
eta_residual = predicted_eta - true_eta
```

### Phi residual

```text
phi_residual = wrapped_angle_delta(predicted_phi, true_phi)   # radians
```

Wrapped so predictions near `+pi` and `-pi` are handled correctly. Reported
separately for electrons and positrons, and with the wrong-charge head selected,
to quantify the cost of a charge flip.

### pT relative residual

```text
pt_rel_residual = (predicted_pT - true_pT) / true_pT
```

The fractional pT error. The cleaner `log_pt` residual
`predicted_log_pt - true_log_pt` is also reported, since it is not blown up by
low-pT electrons.

### z0 residual

```text
z0_residual = predicted_z0 - true_z0   # mm
```

Compared against the anchor-only resolution and the beamspot-prior RMS.

## Current status

Implemented:

- ColliderML data loading and prompt-electron selection
- calorimeter cell extraction and idealized DBSCAN cleaning
- `electrons_dbscan.parquet` workflow with truth eta/phi/log_pt/z0/charge
- train / validation / test splitting and train-only normalization statistics
- canonical azimuthal frame (centroid rotated to phi = 0)
- angular per-cell features and shower-shape cluster features
- anchored residual predictions for eta, phi, log_pt, z0
- two charge-hypothesis phi heads (mirror-symmetry training)
- calorimeter pointing anchor for z0
- Huber per-target losses combined by a scale-invariant geometric mean
- convolutional calorimeter regressor
- residual and Gaussian-resolution evaluation plots
- Weights & Biases logging

Current active questions:

- how much pT information is recoverable from calorimeter showers alone
- how well z0 can be reconstructed from shower pointing
- how separable the electron and positron phi hypotheses are
- how performance changes with true electron pT
- how idealized DBSCAN cleaning compares to more realistic clustering

## License

MIT
