# colliderml-electron-calo

Machine-learning experiments on the calorimeter portion of the CERN ColliderML
Release 1 dataset, focused on reconstructing prompt-electron kinematics from
calorimeter showers alone.

The scientific question is how much tracker-like information can be recovered
from a calorimeter shower with no tracker input. The model is trained to predict
five per-electron quantities from calorimeter cells only:

- pseudorapidity, `eta`
- azimuthal angle, `phi`
- transverse momentum, `pT`, trained through `log(pT)`
- longitudinal impact parameter, `z0`
- electric charge sign, `q`

`eta`, `phi`, and `pT` are predicted as residuals from anchors computed from the
shower (see "Anchored residual predictions"). `z0` is regressed directly in
z-scored units; its pointing-fit anchor is an input feature, not a residual base.
Charge is a binary classification output.

## Project goal

`eta` and `phi` are directions a calorimeter measures relatively well. `pT`,
`z0`, and the charge sign are the harder, more tracker-like quantities. Studying
all five together is a way to map where calorimeter-only reconstruction saturates
and where the tracker is genuinely needed. The broader aim is understanding
tracker-versus-calorimeter complementarity.

Two reconstruction regimes are studied:

- **Supervised (this README).** Cells are selected using truth association, so
  the model is tested on how well it learns kinematics from a clean shower. This
  is the main training and evaluation path.
- **Truth-free.** A DBSCAN clustering pipeline selects cells with no truth input.
  This measures what is lost moving toward a realistic pipeline. It is documented
  separately in `docs/unsup_clustering_summary.md`.

## Data

The project uses the CERN ColliderML Release 1 dataset. The default sample is
`zee_pu200` (Z to ee events, PU200, OpenDataDetector).

The supervised training table is a per-electron parquet. Each row is one prompt
electron: a variable-length set of calorimeter cells plus the truth labels
`truth_eta`, `truth_phi`, `truth_log_pt`, `truth_z0`, and `truth_charge`. Truth
kinematics are generator-level, taken from the primary electron at its production
vertex.

The train / validation / test split is stored inside the parquet as a `split`
column. Target normalization statistics are computed on the **train** split only
and reused unchanged for validation and test.

Large data and checkpoint files are not tracked in git (see `.gitignore`). The
parquet tables live outside the repository on the compute hosts.

## Model inputs

Each electron keeps at most `max_cells = 128` cells, the highest-energy cells
from the shower. Cluster-level features are computed over the **full** shower
before this truncation, so total shower energy is preserved even when only the
128 hottest cells are passed individually.

Before features are built, the event is rotated in azimuth so the energy-weighted
phi centroid sits at `phi = 0`. This gives every shower a canonical azimuthal
frame.

### Per-cell inputs

Positional (Fourier-embedded), in the centroid-rotated frame:

```text
cell_x, cell_y, cell_z
```

High-level per-cell features include `log(cell_e_calibrated)`, `cell_eta`,
`sin`/`cos` of the centroid-relative phi (phi is periodic), `theta`,
`cos(theta)`, and a detector-subsystem one-hot.

### Cluster-level inputs

Computed from the full shower and broadcast to every cell (29 values): log total
calibrated energy, log transverse-energy proxy, log cell count, phi and eta
energy-weighted widths and skewnesses, the z0 pointing anchor, the r-z pointing
slope, the radial spread and RMS residual of the pointing fit, and a 6-slice
radial profile (mean radius, mean z relative to the anchor, and energy fraction
per slice). With the 12 per-cell features (log E, eta, sin and cos of
centroid-relative phi, theta, cos theta, 6-way detector one-hot), the total is 41. The phi skewness is physically
meaningful: the bremsstrahlung tail is asymmetric in a charge-dependent way.

`phi_slope` is computed but not exposed in the 41-dim set.

The full high-level input vector is **41-dimensional** (`high_level_dim = 41`).
`x_high_level[..., 0]` is the per-cell log-energy, which is also the score used
to select the top `max_cells` cells.

## Model architecture

The current champion is `AttnPoolCaloRegressor` (in
`src/colliderml_electron/model.py`):

```text
top-cell selection (128 highest-energy cells)
    -> Fourier positional embedding of (x, y, z)
    -> transformer cell encoder
    -> learned-query cross-attention pooling (permutation-invariant over cells)
    -> MLP head -> 5 outputs
```

Attention pooling replaced an earlier Conv1d-over-sequence aggregation, which had
an energy-rank ordering artifact. The pooling uses `n_queries = 4` trained query
vectors that attend over the encoder output with padding masked.

**Trained champion hyperparameters** (set in the training-script config, not the
class defaults):

```text
model_type      = attnpool
model_dim       = 128
n_layers        = 3
n_heads         = 4
dim_feedforward = 256
n_queries       = 4
max_cells       = 128
high_level_dim  = 41
output_dim      = 5
```

Note: the `AttnPoolCaloRegressor.__init__` defaults (`model_dim = 256`,
`n_layers = 6`, `n_heads = 8`) are **not** the trained configuration. The config
dict in `scripts/train_eta_phi_pt_z0_charge.py` overrides them to 128 / 3 / 4.
A larger model (256 / 6 / 8) trained for the same number of optimizer steps lost
charge performance, with phi core resolution degrading from about 0.007 to
0.023 rad. The working interpretation is that the larger model needs more steps
before the charge head lifts off; this has not been tested with a longer run.

## Model output

The model produces five values:

```text
[delta_eta, delta_phi, delta_log_pt, z0_norm, charge_logit]
```

The first three are residuals added to their anchors; the fourth is z0 in
z-scored units (train-split mean and std); the fifth is a raw classification
logit. Decoding at evaluation:

```text
pred_eta    = eta_centroid + delta_eta
pred_phi    = wrap(phi_centroid + delta_phi)          # single signed correction
pred_pt     = exp(log_sum_et + delta_log_pt)          # GeV
pred_z0     = z0_mean + z0_norm * z0_std              # mm, direct regression
pred_charge = +1 (positron) if sigmoid(charge_logit) > 0.5 else -1 (electron)
```

## Anchored residual predictions

Instead of regressing absolute kinematics, the model predicts a small correction
to a physics-motivated anchor, which is already a strong first estimate:

- `eta_centroid`: energy-weighted average of cell pseudorapidities.
- `phi_centroid`: energy-weighted azimuth, `atan2(<sin phi>, <cos phi>)`.
- `log_sum_et`: log of the total transverse-energy proxy. For a contained
  electromagnetic shower this is already close to `log(pT)`.
- `z0_anchor` (input feature only, not a residual base): energy-weighted
least-squares fit of cell z versus cell radius r, extrapolated to r = 0. The
solenoid bends only in the transverse plane, so the r-z trajectory is close to
a straight line, but the fit's lever arm (the shower's radial extent) is short
compared with the extrapolation to the beamline. The anchor alone is therefore
poor (299-540 mm RMSE in the tracked runs, worse than the beamspot prior).

## Phi and charge

A charged particle is bent in azimuth by the solenoidal field, and the shower
(including its bremsstrahlung tail) is displaced from the true electron direction
in a charge-dependent way. Electrons and positrons of the same momentum are
displaced in **opposite** azimuthal directions.

The model predicts a **single signed** `delta_phi` correction. Its sign encodes
the azimuthal bend, which is physically the charge handle. Charge is decoded from
a **dedicated** `charge_logit`, trained with binary cross-entropy, rather than
being inferred from a two-hypothesis phi scheme.

An earlier design used two phi heads (electron and positron hypotheses) and
selected between them using the truth charge at evaluation. That leaked truth
information and was retired. The current single-phi-plus-charge-logit design uses
no truth charge at inference: charge comes only from the learned logit.

The charge signal is fundamentally the sign of the azimuthal bend, so phi
resolution is the leading indicator for charge performance. The charge head is
slow to train. How many optimizer steps it needs before lifting off the 0.5
chance line depends on the configuration (about 90,000 for <config>, see
docs/supervised_status.md), so comparisons between configurations should hold
the step count fixed rather than the epoch count.

## Loss function

eta, phi and log(pT) use a Huber loss on the residual from their anchor, in
physical units (Huber delta = 0.1, 0.05 rad, and 0.2 respectively; the phi
residual is wrapped). z0 uses a Huber loss on the z-scored target (delta = 1.0).

The four regression losses are combined with **homoscedastic uncertainty
weighting**: a learned per-task `log_sigma` (four parameters) sets each task's
weight automatically via

```text
total_reg_loss = sum_t ( exp(-2 * log_sigma_t) * loss_t + log_sigma_t )
```

This balances tasks in very different natural units (eta units, radians, log-pT,
z-scored z0) without hand-tuning. The learned `log_sigma` converges to
`sigma^2 ~ 2 * E[Huber loss]`, so it tracks the tail-insensitive **core** of the
resolution rather than the RMS.
Charge is kept outside the learned weighting with a fixed weight of 1.0. In the
Jun23 runs the charge head did not train under the learned weighting and did
train with the fixed weight; the mechanism was not isolated.

## Evaluation quantities

- `eta_residual   = pred_eta - true_eta`
- `phi_residual   = wrapped_angle_delta(pred_phi, true_phi)`  (radians, wrapped)
- `pt_rel_residual = (pred_pT - true_pT) / true_pT`; the cleaner
  `log_pt` residual is also reported since it is not blown up by low-pT electrons.
- `z0_residual    = pred_z0 - true_z0`  (mm), compared against the anchor-only
  resolution and the beamspot-prior RMS. A useful model must beat both.
- charge: ROC AUC and accuracy versus pT, plus calibration.

In the tracked pre-AttnPool runs (pT > 10 GeV), barrel z0 RMSE is about 39 mm
against a 54 mm beamspot prior; over |eta| <= 1.7 it is 42-47 mm. In the endcap
z0 does not beat the prior. Whether the barrel value is a calorimeter-only limit
has not been established. Endcap charge is harder: a forward electron reaches the calorimeter at a
smaller transverse radius, so the charge-dependent azimuthal displacement
(proportional to r / pT) is smaller than in the barrel.

## Repository layout

```text
src/colliderml_electron/   # main package (io, coords, calibration, pipeline,
                           #   cluster, dataset, model, encoder, embedding,
                           #   resolution, splits, stats, plots, ...)
scripts/                   # build / train / test / diagnose / plot scripts
slurm/                     # SLURM batch scripts (Lyon CC-IN2P3)
results/                   # evaluation plots and metrics per run
docs/                      # supervised_status.md, experiment_log.md, unsup_clustering_summary.md
notebooks/                 # exploratory notebooks
pyproject.toml, uv.lock    # environment (uv, Python 3.10-3.11, torch 2.2.2)
```

Key source files:

```text
src/colliderml_electron/io.py           ColliderML loading, prompt-electron selection
src/colliderml_electron/pipeline.py     supervised per-electron row builder
src/colliderml_electron/cluster_pipeline.py   truth-free DBSCAN pipeline
src/colliderml_electron/dataset.py      torch dataset, feature builder, region cuts
src/colliderml_electron/model.py        ConcatCaloRegressor / ConvCaloRegressor /
                                        AttnPoolCaloRegressor
```

Key scripts:

```text
scripts/build_electron_dataset.py       build the supervised per-electron parquet
scripts/train_eta_phi_pt_z0_charge.py   train the AttnPool champion
scripts/test_eta_phi_pt_z0_charge.py    evaluate: residuals, resolutions, charge ROC
scripts/compare_preds_bootstrap.py      paired bootstrap for A/B comparisons
scripts/compare_regions_bootstrap.py    barrel-vs-endcap bootstrap
scripts/make_summary_figs.py            summary figures
scripts/check_dims.py                   preflight: verify high_level_dim / output_dim
```

## Setup

Python 3.10 or 3.11. The environment is managed with `uv` and pinned by
`uv.lock` (`torch == 2.2.2`, `colliderml`, Python 3.11).

```bash
uv sync
```

A plain-pip fallback also works:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Running

Scripts are invoked in `env VAR=value python ...` single-line form. Command-line
`--flags` are **not** parsed by the training and evaluation entry points;
configuration is read from environment variables and the in-script config dict.
Use `$HOME`, not `~`, in cluster paths.

Preflight before submitting to the cluster:

```bash
python -c "from colliderml_electron.model import AttnPoolCaloRegressor"
python scripts/check_dims.py --high-level-dim 41 --output-dim 5
git log --oneline -1
```

Train (region, seed and epoch count are environment-driven; regions are
`full`, `barrel`, `endcap`; `N_EPOCHS=200` is the supervised reference, 324 the
truth-free step-matched run). Locally the script reads
`data/electrons/electrons.parquet`; the Lyon sbatch symlinks the staged parquet
to that name.

```bash
env N_EPOCHS=200 REGION=full SEED=0 python scripts/train_eta_phi_pt_z0_charge.py
```

On the Lyon CC-IN2P3 cluster (H100, SLURM):

```bash
N_EPOCHS=200 REGION=full SEED=0 sbatch slurm/run_train_test_lyon.sbatch
```

The per-region input-projection ablation is gated by an environment flag and is
backward-compatible with existing checkpoints:

```bash
env PER_REGION_PROJ=1 N_EPOCHS=200 REGION=full SEED=0 python scripts/train_eta_phi_pt_z0_charge.py
```

Evaluate a trained checkpoint. Always verify checkpoint provenance first, because
scoring a checkpoint through the wrong branch's dataset code causes silent shape
mismatches:

```bash
env CHECKPOINT=CKPT.pt STATS_PATH=target_stats.json OUTPUT_DIR=results/myrun python scripts/test_eta_phi_pt_z0_charge.py
```

Optional eval cuts: `MIN_PT_EVAL`, `MAX_ABS_ETA_EVAL`, `MIN_ABS_ETA_EVAL`.

Evaluation writes expected-vs-predicted scatter plots, per-target residual and
Gaussian-resolution fits, a phi-residual plot split by truth charge, charge ROC
and calibration plots, and `test_metrics.json` under `results/`.

## Reporting discipline

Any difference reported as a result is first run through the pre-registered
paired bootstrap (`compare_preds_bootstrap.py`, 2000 resamples). Evaluation
criteria and the comparison population are fixed before results are examined.
The between-seed sd of barrel charge AUC on the current dataset generation is
0.0065 (docs/unsup_clustering_summary.md); the seeds behind it should be listed
from W&B. The 0.017 spread of the three tracked July benchmark runs is not a
seed variance: they mix configurations and were selected on the charge head
having trained.

## Status

Implemented: ColliderML loading and prompt-electron selection; supervised
per-electron dataset with truth eta/phi/log_pt/z0/charge; truth-free DBSCAN
pipeline; canonical azimuthal frame; angular and shower-shape features; anchored
residual predictions; single signed phi plus dedicated charge logit; homoscedastic
regression loss with manually-weighted charge BCE; AttnPool regressor; residual,
resolution, and charge-ROC evaluation; paired-bootstrap comparison tooling;
learned-sigma instrumentation; W&B logging.

Open items are tracked in `docs/supervised_status.md`.

## License

MIT
