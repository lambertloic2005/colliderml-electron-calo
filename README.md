# colliderml-electron-calo

Machine-learning experiments on the calorimeter portion of the CERN ColliderML
Release 1 dataset, focused on reconstructing prompt-electron kinematics from
calorimeter showers alone.

The model is trained to predict five physics quantities per electron:

- pseudorapidity, `eta`
- azimuthal angle, `phi`
- transverse momentum, `pT`, trained through `log(pT)`
- longitudinal impact parameter, `z0`
- charge, `q` (binary classification, electron vs. positron)

Every regression quantity is predicted as a **residual from a physics-motivated
anchor** rather than from scratch (see "Anchored residual predictions" below).
Charge is predicted as a dedicated logit, independently of the other four
targets (see "Charge reconstruction" below).

## Project goal

The long-term goal is to study how much tracker-like information can be
inferred from calorimeter showers alone. The five targets above span
directional information the calorimeter measures well (`eta`, `phi`), more
tracker-like information (`pT`, `z0`), and a binary discriminant that depends
on the magnetic-field deflection accumulated before the shower forms (`q`).

```text
DBSCAN-cleaned calorimeter shower -> truth eta, truth phi, truth pT, truth z0, truth charge
```

This is not yet a fully realistic detector-level reconstruction pipeline. The
DBSCAN cleaning is idealized and supervised: it is used to create a cleaner
per-electron dataset so the model can be tested on how well it learns from
calorimeter shower information.

A second, currently active line of work asks whether the calorimeter-only
problem should be solved by **one model across the full detector** or by
**separate barrel and endcap models** — see "Barrel / endcap region split"
below.

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
Train eta / phi / pT / z0 / charge model
        |
        v
Evaluate residuals, resolution plots, and charge ROC/calibration
        |
        v
(active) Split evaluation, then training, by detector region
```

The canonical local training table is
`data/electrons/eta_phi_pt_z0_charge/zee_pu200_z0_charge.parquet`, with target
statistics in the matching `target_stats.json` alongside it. On Ruche, the
SLURM launcher (`slurm/run_train_test_new.sbatch`) copies this file into the
job's private working directory as `data/electrons/electrons_z0_charge.parquet`
and symlinks it to `electrons.parquet`, since the training script expects that
filename. Training and evaluation must use the same DBSCAN-cleaned table so
both see the same kind of shower.

## What the model learns

Each row in the training table corresponds to one prompt electron. The input
is a variable-length set of DBSCAN-cleaned calorimeter cells. The target is
the truth prompt-electron kinematics and charge from the ColliderML particle
table.

The model does not receive truth `eta`, `phi`, `pT`, `z0`, or charge as input
at any point — not even during decoding. Truth values are used only as
supervised labels for the loss. This is a deliberate scientific-integrity
constraint: an earlier version of this model used a two-head phi scheme where
truth charge was used at evaluation time to select which phi head to read out.
That meant the model never actually had to reconstruct charge to get a good
phi resolution, which made the reported phi resolution artificially
optimistic. The current model instead predicts a single signed phi residual
and a dedicated charge logit, both decoded with no truth information
whatsoever (see "Charge reconstruction").

## Model inputs

For batching and speed, each electron keeps at most `max_cells = 128` cells,
the highest-energy cells from the DBSCAN-cleaned shower. The cluster-level
features below are computed over the **full** cleaned shower before this
truncation, so the model still sees the total shower energy and shape even
when only the 128 hottest cells are passed individually.

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

### Per-cell high-level inputs (12 dims)

```text
log(cell_e_calibrated)
cell_eta
sin(cell_phi - phi_centroid)
cos(cell_phi - phi_centroid)
theta
cos(theta)
detector one-hot (6 subsystems: codes 9, 10, 11, 12, 13, 14)
```

The `sin`/`cos` of the centroid-relative phi is used because phi is periodic.

### Cluster-level inputs (29 dims, broadcast to every cell)

Computed from the full DBSCAN-cleaned shower and appended to every cell.

Base cluster scalars (11 dims):

```text
log(total calibrated cluster energy)
log(total transverse-energy proxy)
log(number of cells)
phi shower-shape width (std_phi)
phi shower-shape skewness (skew_phi)
eta shower-shape width (std_eta)
eta shower-shape skewness (skew_eta)
z0 pointing anchor (m)
shower pointing slope dz/dr
r_spread (pointing-fit radial spread, m)
fit_rms (pointing-fit residual RMS, in units of 100 mm)
```

Radial-slice pointing profile (18 dims, `K = 6` slices in cell radius `r`):

```text
<r>_k / 1000          for k = 1..K   (mean radius per slice, m)
(<z>_k - z0_anchor)/100  for k = 1..K  (mean longitudinal offset per slice)
energy_fraction_k      for k = 1..K   (fraction of shower energy per slice)
```

The phi skewness matters physically: the bremsstrahlung tail is asymmetric in
a charge-dependent way, so the sign of `skew_phi` carries charge information.
The radial-slice profile lets the model resolve the shower's longitudinal
pointing direction more finely than a single linear fit, and `r_spread` /
`fit_rms` let the model judge how trustworthy the linear pointing anchor is
for a given shower (a shower with little radial extent has an ill-determined
slope).

The full high-level input is **41-dimensional** (12 per-cell + 11 base
cluster + 18 profile).

## Anchored residual predictions

Instead of regressing the absolute kinematics, the model predicts a small
**correction** to a physics-motivated anchor for each regression target:

```text
eta    = eta_centroid + delta_eta
phi    = phi_centroid + delta_phi
log_pt = log_sum_et   + delta_log_pt
z0     = z0_anchor    + delta_z0
```

The anchors are:

- `eta_centroid`: the energy-weighted average of the cell pseudorapidities.
- `phi_centroid`: the energy-weighted azimuth, `atan2(<sin phi>, <cos phi>)`.
- `log_sum_et`: the log of the total transverse-energy proxy of the shower.
  For a contained electromagnetic shower this is already close to `log(pT)`.
- `z0_anchor`: an energy-weighted least-squares "pointing" fit of cell `z`
  versus cell `r`, extrapolated to `r = 0`. A straight shower points back
  along the electron flight direction to its production `z`. Because the
  magnetic field bends tracks in the transverse plane, the `r`-`z` projection
  used here is approximately unaffected by bending, which makes this a clean
  anchor for `z0`.

## Charge reconstruction

A charged particle is bent in azimuth by the solenoidal field **in the
tracker**, before it ever reaches the calorimeter; the calorimeter only
records the resulting charge-dependent azimuthal offset of the shower, not a
bend inside the calorimeter itself. An electron and a positron of the same
momentum are displaced in **opposite** azimuthal directions, by an amount that
shrinks at high pT (less time in the field) — confirmed empirically as a
~0.014 rad charge-dependent separation in shower centroid phi, with no
corresponding eta separation.

The model predicts charge with a single dedicated logit, trained with binary
cross-entropy:

```text
charge_logit > 0  =>  predict positron (q = +1)
charge_logit < 0  =>  predict electron (q = -1)
```

This logit is produced by the same head as the four regression residuals
(see "Model output"), but its loss is **not** combined into the homoscedastic
regression weighting. An earlier version weighted charge by the same learned
per-task uncertainty scheme used for the regression targets; because the BCE
loss (~0.69 at initialization) is orders of magnitude larger than the Huber
regression losses (~1e-3 after some training), the homoscedastic scheme
collapsed: the learned uncertainty for charge inflated until its gradient was
effectively zero, and charge accuracy stalled near chance. The fix was to
remove charge from the learned-uncertainty pool entirely and give it a fixed,
manually chosen weight (`charge_weight`) added on top of the regression total.
After this fix, full-detector charge accuracy reached ~0.72 and AUC ~0.82.

The azimuthal phi residual is decoded with **no truth charge involved at any
point** — its sign is itself the model's implicit charge signal, but the
explicit charge call comes only from the logit head.

## Longitudinal impact parameter z0

The truth `z0` label is the production-vertex `z` of the prompt electron. For
a prompt particle originating on the beamline, this is the quantity a tracker
would reconstruct as the longitudinal impact parameter.

`z0` is predicted as a correction to the calorimeter pointing anchor described
above, refined by the radial-slice pointing profile features. Two baselines
are reported at evaluation time:

- the beamspot prior, i.e. the RMS of `z0` with no model at all (~56 mm), and
- the model's actual residual RMS / Gaussian-fit sigma.

A useful model must beat the beamspot prior. On the full test set the model
achieves z0 RMSE ~49 mm / sigma ~48 mm against a prior of ~56 mm (R^2 ~ 0.20).
This is interpreted as close to the calorimeter-only information ceiling, not
a model-capacity limitation: calorimeter timing is far too imprecise for
vertex resolution (the fitted shower t0 spread is roughly 5x larger than the
vertex spread), and the radial-slice pointing profile has been pushed close
to what the pointing geometry alone can resolve.

**The ceiling is sharply region-dependent — see "Barrel / endcap region
split" below.** This was discovered by evaluating the trained model's z0
residuals separately for truth-|eta| < 1.5 (barrel) versus > 1.5 (endcap):
the barrel population beats its local prior (sigma ~38.5 mm vs. a barrel
prior ~54.5 mm), while the endcap population sits almost exactly at its
prior (sigma ~56.8 mm vs. an endcap prior ~57.1 mm) — i.e. the model has
learned to fall back to predicting the beamspot mean in the endcap, not
because of insufficient capacity, but because the calorimeter geometry there
does not constrain z-pointing as well as the barrel cylinder does.

A caveat on a related diagnostic plot: conditioning on
`|pred_z0| > 10 mm` and reporting that subset's RMSE (~42 mm) is **not** an
unbiased resolution number, since it selects on the prediction itself, which
correlates with the residual. It is a legitimate look at the non-collapsed
population (the diagonal arm of the z0 scatter, which the region split shows
is mostly the barrel), but should never be quoted as "the model's z0
resolution."

### Considered and parked: transverse impact parameter d0

It was proposed to add the transverse impact parameter `d0` as a sixth
target, with the hypothesis that it would behave as a rough 90-degree
rotation of z0 — recoverable in the endcap (where the calorimeter resolves
transverse position well) and poorly recoverable in the barrel. This
hypothesis does not survive a check of the underlying physics: z0 is
recoverable in the barrel because the longitudinal luminous region is
extended (the beamspot has ~50 mm of spread in z, giving the model something
to resolve), whereas the **transverse** luminous region at the LHC is
extremely narrow (micron scale) for a prompt particle. `d0` for a prompt
electron is therefore essentially zero with negligible true spread in
**either** region — there is no recoverable signal for the calorimeter to
find, regardless of how well it resolves transverse position. A model would
simply learn to output ~0 everywhere, which is correct but uninformative.
`vx`/`vy` are also not currently extracted anywhere in the pipeline (only
`vz`, via `truth_z0`, is read out of the particle record in `io.py`), so `d0`
does not yet exist as a column. Before any code is written for this, run a
direct empirical check of `std(truth_d0)` against the calorimeter's
transverse pointing resolution; if it is well below the ~5.1 mm cell pitch,
this direction is dead and the result itself ("the transverse luminous region
is below calorimeter resolution, unlike z0") is worth stating on a slide
rather than pursued as a training target.

## Barrel / endcap region split

### Motivation

Evaluating the single full-detector model's z0 residuals split by truth
`|eta|` showed two qualitatively different populations: a barrel population
that genuinely points (correlated diagonal band, beats its local prior) and
an endcap population that collapses to the beamspot mean (flat horizontal
band, sits at its local prior). This is a real geometric effect, not an
artifact: barrel cells sit on a cylinder at roughly constant radius and
spread out in `z`, so z-pointing comes from where along the cylinder the
shower lands; endcap cells sit on disks at roughly constant `|z|` and spread
out in radius, so the longitudinal handle the barrel relies on is largely
absent. A single model is being asked to learn two different pointing modes
at once. The active question is whether two region-specialized models do
better than one combined model, particularly for the barrel (where there is
real signal to extract) and for charge (whose azimuthal deflection also
differs with path length through the field before each region).

### Detector geometry

The barrel/endcap boundary was determined empirically rather than assumed,
via `scripts/diagnose_detector_regions.py`, which characterizes each
`cell_detector` code by its radius/`|z|` spread:

| code | subsystem (inferred)     | shape  | `\|eta\|` range |
|------|---------------------------|--------|----------------|
| 9    | ECal endcap               | endcap | 1.51 - 3.08    |
| 10   | **ECal barrel**           | barrel | 0.00 - 1.62    |
| 11   | ECal endcap                | endcap | 1.51 - 3.08    |
| 12   | HCal/forward endcap        | endcap | 1.40 - 3.40    |
| 13   | HCal barrel (low occupancy)| barrel | 0.00 - 1.48    |
| 14   | HCal/forward endcap        | endcap | 1.44 - 3.40    |

The relevant handoff is the **ECal** barrel-to-endcap transition: code 10
(barrel) covers up to `|eta| = 1.62`; codes 9/11 (endcap) begin at
`|eta| = 1.51`. The boundary used everywhere in this project is
**`|eta| = 1.5`**, with the small 1.5-1.6 band being the physical
barrel/endcap crack.

### Evaluation-only region split (no retraining)

`scripts/test_eta_phi_pt_z0_charge.py` evaluates the existing full-detector
checkpoint, then additionally buckets the test set by truth `|eta|` into
`barrel` (< 1.5), `endcap` (1.5-3.0), and `fwd` (> 3.0, empty under the
current `|truth_eta| <= 3` training cut) and recomputes per-target
resolutions, z0 prior/RMSE, and charge AUC/accuracy within each bucket. These
land in `test_metrics.json` under `test/region/<label>/...` and as additional
per-region z0 plots (`expected_vs_predicted_z0_barrel.png`, etc.). Buckets
with `n < 50` are skipped to avoid plotting noise as if it were signal.

The split uses **truth** eta as the routing label; at inference time predicted
eta (sigma ~0.019) would route essentially identically, since a routing error
only matters for showers within ~0.02 of the |eta| = 1.5 boundary.

Single-model results from this evaluation-only split (test set, full-detector
checkpoint, `Jun23_ConstChargeWeight`):

| region | n    | z0 sigma (mm) | z0 prior (mm) | charge AUC | charge acc |
|--------|------|---------------|----------------|------------|------------|
| barrel | 2411 | 38.5          | 54.5           | 0.898      | 0.808      |
| endcap | 2068 | 56.8          | 57.1           | 0.680      | 0.624      |

The barrel clearly beats its prior on both z0 and charge; the endcap sits at
its z0 prior (no pointing recovered) and is markedly worse, though still
above chance, on charge.

### Two-model training (active)

The current experiment trains two **separate** models rather than just
evaluating one model's regions: a barrel-only model and an endcap-only model,
with **overlapping** training acceptance cuts so showers near the seam are
seen by both models during training, while the evaluation routing boundary
stays a sharp `|eta| = 1.5`:

```text
barrel model:  min_abs_eta = None,  max_abs_eta = 1.7
endcap model:  min_abs_eta = 1.3,   max_abs_eta = 3
```

This is controlled by a `REGION` environment variable
(`REGION=barrel` or `REGION=endcap`), read at the top of
`scripts/train_eta_phi_pt_z0_charge.py::main()`, which sets `region`,
`max_abs_eta`, and `min_abs_eta` in the run config. `min_abs_eta` is a new
acceptance cut added to `ElectronDataset` / `make_loader`
(`src/colliderml_electron/dataset.py`) alongside the existing `max_abs_eta`,
filtered with `truth_eta.abs() >= min_abs_eta`. The same `REGION` value
governs both the training and test loaders, and the test script's charge
reload also applies `min_abs_eta`, so a model's own training cut is always
the cut it is evaluated under.

Submit both regions as separate SLURM jobs:

```bash
sbatch --job-name=trainTest_barrel --export=ALL,REGION=barrel slurm/run_train_test_new.sbatch
sbatch --job-name=trainTest_endcap --export=ALL,REGION=endcap slurm/run_train_test_new.sbatch
```

`slurm/run_train_test_new.sbatch` reads `REGION` (default `barrel`) and bakes
it into `RUN_NAME=eta_phi_pt_z0_charge_${REGION}_<timestamp>`. Each job runs
in its own private working-directory copy of the repo (`rsync`'d into
`$TMPDIR` or a `.slurm_work_*` directory), so concurrent barrel/endcap jobs do
not clobber each other's `checkpoints/ruche_eta_phi_pt_z0_charge.pt` while
running. On completion (or failure), outputs are copied back to
`runs/ruche_${RUN_NAME}_${SLURM_JOB_ID}/`, containing that region's
checkpoint, `results/`, training/test logs, and the exact test script used.
There is therefore no single canonical "latest checkpoint" path once both
regions have been trained — look under `runs/` for the region and timestamp
you want.

The fair comparison for this experiment is each specialized model evaluated
on its own region, against the single-model per-region numbers above (not
the two specialized models against each other): does the barrel model beat
38.5 mm z0 sigma / 0.898 charge AUC on barrel data, and does the endcap model
do anything beyond the ~57 mm prior it is already pinned to? A null result in
the endcap (model still pinned at the prior) is an expected, physically
meaningful outcome — it would confirm the endcap z0 ceiling is genuinely a
detector-geometry limit rather than a single-model capacity limit, and is
worth reporting as such rather than treated as a failed experiment.

## Model output

The model produces five values per electron, in this fixed order:

```text
[delta_eta, delta_phi, delta_log_pt, delta_z0, charge_logit]
```

Decoding at evaluation, with no truth information used anywhere in this step:

```text
pred_eta     = eta_centroid + delta_eta
pred_phi     = wrap(phi_centroid + delta_phi)
pred_log_pt  = log_sum_et + delta_log_pt
pred_pt      = exp(pred_log_pt)                       # GeV
pred_z0      = z0_anchor + delta_z0                    # mm
pred_charge  = +1 if charge_logit > 0 else -1
```

## Loss function

Each of the four regression targets uses a Huber (smooth-L1) loss on its
anchor residual, more robust to shower outliers than plain MSE:

```text
eta    : Huber(delta=0.1)   on delta_eta
phi    : Huber(delta=0.05)  on the wrapped angular error of delta_phi
log_pt : Huber(delta=0.2)   on delta_log_pt   (~20% in pT)
z0     : Huber(delta=1.0)   on the z-scored z0 residual
```

The four regression losses are combined with **learned homoscedastic
uncertainty weighting** (Kendall & Gal style): each task `i` has a learned
log-sigma parameter, and the combined loss is

```text
total_reg_loss = sum_i ( exp(-2 * log_sigma_i) * loss_i + log_sigma_i )
```

so a task the model is currently worse at (effectively higher sigma) is
automatically down-weighted, with the `log_sigma` term preventing the trivial
solution of inflating sigma to infinity for every task.

Charge is **not** included in this learned-uncertainty pool (see "Charge
reconstruction" for why) — its binary cross-entropy loss is added on top with
a fixed, manually set weight:

```text
total_loss = total_reg_loss + charge_weight * BCE(charge_logit, charge_label)
```

## Why total energy helps pT

For a relativistic electron, energy and momentum are closely related:

```text
pT = p sin(theta),    and for high energy   p ~ E,   so   pT ~ E sin(theta)
```

The model therefore needs both the total shower energy and the shower
direction. The transverse-energy anchor `log_sum_et` supplies the energy
scale, while the energy-weighted centroid supplies the direction.

## Repository structure

```text
.
├── checkpoints/                  # Saved model checkpoints
├── data/                         # Local generated data
├── docs/                         # Notes and documentation
├── notebooks/                    # Jupyter notebooks
├── results/                      # Evaluation plots and metrics
├── runs/                         # Archived SLURM job outputs (per-run, per-region)
├── scripts/                      # Command-line scripts
├── src/colliderml_electron/      # Main Python package
├── slurm/                        # SLURM batch scripts (Ruche)
├── pyproject.toml
├── requirements.txt
├── uv.lock
└── README.md
```

Important source files:

```text
src/colliderml_electron/io.py            # raw particle/cell extraction (vz -> truth_z0; no vx/vy yet)
src/colliderml_electron/coords.py
src/colliderml_electron/calibration.py
src/colliderml_electron/pipeline.py
src/colliderml_electron/cluster.py
src/colliderml_electron/cluster_pipeline.py
src/colliderml_electron/dataset.py       # ElectronDataset / make_loader, incl. min/max_abs_eta region cuts
src/colliderml_electron/model.py         # ConvCaloRegressor
src/colliderml_electron/resolution.py    # Gaussian-fit resolution helper
src/colliderml_electron/splits.py
src/colliderml_electron/stats.py         # target normalization stats (python -m colliderml_electron.stats)
```

Important scripts (current pipeline):

```text
scripts/build_electron_dataset.py        # builds the DBSCAN-cleaned electron table
scripts/train_eta_phi_pt_z0_charge.py    # training entry point; REGION env var for barrel/endcap
scripts/test_eta_phi_pt_z0_charge.py     # evaluation entry point; per-region breakdown block
scripts/diagnose_detector_regions.py     # measures barrel/endcap |eta| boundary from cell geometry
scripts/check_truth_vertex.py            # verifies truth_z0 is the production-vertex z, not a track-fit value
scripts/check_dims.py                    # verifies emitted high_level_dim matches the train config
scripts/plot_charge_results.py           # ROC, calibration, logit-by-truth, accuracy-vs-pT plots for charge
scripts/plot_resolution_vs_pt.py         # resolution-vs-pT plots, with a --split-eta region option
scripts/export_charge_eval.py            # exports raw per-electron charge predictions for offline analysis
```

`scripts/` also contains a number of one-off diagnostic and exploratory
scripts (`diagnose_*.py`, `plot_*.py`, `check_*.py`, `verify_*.py`) used to
debug specific issues during development (DBSCAN keep-rule tuning, energy
starvation, matching, phi behavior, z0 anchoring, etc.); these are kept for
reference but are not part of the train/evaluate pipeline above.

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

The canonical local training table is
`data/electrons/eta_phi_pt_z0_charge/zee_pu200_z0_charge.parquet` and the
matching target statistics file is
`data/electrons/eta_phi_pt_z0_charge/target_stats.json`. The train,
validation, and test split is stored inside the parquet as a `split` column.

Each electron row carries the truth labels `truth_eta`, `truth_phi`,
`truth_log_pt`, `truth_z0`, and `truth_charge`, plus per-cell columns
including `cell_x`, `cell_y`, `cell_z`, `cell_e_calibrated`, `cell_eta`,
`cell_phi`, and `cell_detector`.

## Build the DBSCAN-cleaned electron dataset

If the table already exists you do not need to rebuild it. A typical command:

```bash
python scripts/build_electron_dataset.py \
  --channel zee \
  --pileup pu200 \
  --mask dbscan \
  --n-events 50 \
  --out data/electrons/eta_phi_pt_z0_charge/zee_pu200_z0_charge.parquet
```

For a full run, drop `--n-events`. Use `--mask dbscan` so the table matches
the DBSCAN-cleaned test set.

## Compute splits and target statistics

After building the table, assign splits and compute target normalization
statistics over the **train** split only:

```bash
python -m colliderml_electron.stats
```

The same statistics file should be reused for validation and test. Do not
recompute normalization statistics on the test set.

## Train the model

Full-detector model (no region cut beyond the standard `|truth_eta| <= 3`):

```bash
python scripts/train_eta_phi_pt_z0_charge.py
```

Region-specialized model:

```bash
REGION=barrel python scripts/train_eta_phi_pt_z0_charge.py
REGION=endcap python scripts/train_eta_phi_pt_z0_charge.py
```

Key training configuration:

```text
parquet_path         = data/electrons/eta_phi_pt_z0_charge/zee_pu200_z0_charge.parquet
target_stats_path    = data/electrons/eta_phi_pt_z0_charge/target_stats.json
use_angular_features = True
use_cluster_features = True
high_level_dim       = 41
max_cells            = 128
max_abs_eta          = 3 (full) / 1.7 (barrel) / 3 (endcap)
min_abs_eta          = None (full) / None (barrel) / 1.3 (endcap)
output_dim           = 5        # [delta_eta, delta_phi, delta_log_pt, delta_z0, charge_logit]
model_type           = conv
```

The checkpoint is saved to `checkpoints/ruche_eta_phi_pt_z0_charge.pt`. When
run via the SLURM launcher, this is archived per-job under `runs/` (see
"Barrel / endcap region split" above for the exact path).

## Evaluate the model

```bash
python scripts/test_eta_phi_pt_z0_charge.py
```

The evaluation script reads `max_abs_eta` / `min_abs_eta` from the
checkpoint's own training config, so a region-specialized checkpoint is
automatically evaluated on its own region's test rows. Independently, it
always also produces the truth-`|eta|`-based barrel/endcap/fwd breakdown
described above, regardless of which checkpoint is being tested — this lets
you compare a full-detector checkpoint's regional behavior against a
region-specialized checkpoint's behavior on the same axis.

Evaluation outputs are saved under `results/`, including expected-vs-predicted
scatter plots and residual/resolution fits for each target (overall and
per-region), charge ROC/calibration/logit-separation plots, and
`test_metrics.json`.

## Important evaluation quantities

### Eta residual

```text
eta_residual = predicted_eta - true_eta
```

### Phi residual

```text
phi_residual = wrapped_angle_delta(predicted_phi, true_phi)   # radians
```

Wrapped so predictions near `+pi` and `-pi` are handled correctly. Decoded
with no truth charge involved. Also reported split by truth charge, since the
charge-dependent deflection means electrons and positrons can have
systematically different phi bias even though both are unbiased on average.

### pT relative residual

```text
pt_rel_residual = (predicted_pT - true_pT) / true_pT
```

The fractional pT error. The cleaner `log_pt` residual
`predicted_log_pt - true_log_pt` is also reported, since it is not blown up
by low-pT electrons.

### z0 residual

```text
z0_residual = predicted_z0 - true_z0   # mm
```

Compared against the beamspot-prior RMS, both overall and per detector
region. See "Longitudinal impact parameter z0" above for the important
caveat about the `z0_outside_pm10` conditioned subset.

### Charge metrics

```text
charge_acc = mean( (charge_logit > 0) == (truth_charge > 0) )
charge_auc = rank-based AUC of charge_logit against truth_charge
```

Reported overall, per truth-pT bin (accuracy floors out near the high-pT
phi-resolution floor of ~0.007 rad, where the charge-dependent deflection
becomes indistinguishable from noise), and per detector region.

## Current status

Implemented:

- ColliderML data loading and prompt-electron selection
- calorimeter cell extraction and idealized DBSCAN cleaning
- train / validation / test splitting and train-only normalization statistics
- canonical azimuthal frame (centroid rotated to phi = 0)
- 41-dim high-level features: angular per-cell features, shower-shape cluster
  moments, and a K=6 radial-slice longitudinal pointing profile
- anchored residual predictions for eta, phi, log_pt, z0
- single signed phi head + dedicated charge-logit head (BCE), with no truth
  charge used anywhere at decode time
- fixed-weight charge loss to avoid homoscedastic-weighting collapse
- calorimeter pointing anchor + radial profile for z0
- Huber per-target regression losses combined by learned homoscedastic
  (Kendall & Gal) uncertainty weighting
- convolutional calorimeter regressor
- residual and Gaussian-resolution evaluation plots, charge ROC/calibration
- empirically-determined barrel/endcap |eta| = 1.5 boundary
  (`diagnose_detector_regions.py`)
- per-region (`barrel`/`endcap`/`fwd`) evaluation breakdown for an existing
  checkpoint, with a minimum-count guard before plotting
- `min_abs_eta` acceptance cut plumbed through the dataset/loader/train/test
  scripts, with a `REGION` env var selecting barrel (1.3-1.7 train overlap,
  1.7 max) vs. endcap (1.3 min, 3 max) cuts
- region-aware SLURM launcher (`REGION` env var, per-region run naming,
  isolated per-job working directories so concurrent region jobs don't
  collide)
- Weights & Biases logging

In progress:

- training and evaluating dedicated barrel-only and endcap-only models, to
  test whether region specialization beats the single full-detector model's
  per-region numbers (barrel z0 sigma 38.5 mm / charge AUC 0.898; endcap z0
  sigma 56.8 mm, at its prior / charge AUC 0.680)

Current active questions:

- does region specialization improve barrel z0/charge beyond the single-model
  numbers, and does it change anything in the endcap, or is the endcap z0
  ceiling purely a detector-geometry limit (expected: little to no change)
- how much pT information is recoverable from calorimeter showers alone
- how separable electron and positron charge is at low pT (the high-pT
  ceiling is explained by the phi resolution floor; the low-pT failure mode
  is not yet understood)
- how idealized DBSCAN cleaning compares to more realistic clustering

Considered and parked (see "Considered and parked: transverse impact
parameter d0" above):

- adding `d0` as a sixth target — physically expected to carry no recoverable
  signal in either region, since the transverse luminous region is far
  smaller than the calorimeter's transverse resolution, unlike z0's
  longitudinal beamspot. Needs `vx`/`vy` to be added to the pipeline and an
  empirical `std(truth_d0)` check before any model work is justified.

## License

MIT