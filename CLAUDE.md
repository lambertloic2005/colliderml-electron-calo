# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project overview

This repository contains machine-learning experiments for reconstructing prompt-electron kinematics from CERN ColliderML calorimeter showers.

Current targets:

* `truth_eta`
* `truth_phi`
* `truth_log_pt`
* `truth_z0`
* `truth_charge`

The active branch is `region-split-experiment`.

The current experiment compares whether the calorimeter-only reconstruction problem is better handled by:

1. one model across the full detector, or
2. separate barrel-specialized and endcap-specialized models.

The model predicts anchored residuals for eta, phi, log-pT, and z0, plus a charge logit:

```text
[delta_eta, delta_phi, delta_log_pt, delta_z0, charge_logit]
```

Decoded predictions:

```text
pred_eta     = eta_centroid + delta_eta
pred_phi     = wrap(phi_centroid + delta_phi)
pred_log_pt  = log_sum_et + delta_log_pt
pred_pt      = exp(pred_log_pt)
pred_z0      = z0_anchor + delta_z0
pred_charge  = +1 if charge_logit > 0 else -1
```

Do not use truth charge, truth eta, truth phi, truth pT, truth log-pT, or truth z0 as model inputs. Truth values are labels only.

## Current branch behavior

`REGION` controls the training region:

```bash
REGION=barrel python scripts/train_eta_phi_pt_z0_charge.py
REGION=endcap python scripts/train_eta_phi_pt_z0_charge.py
```

Important: in this branch, a bare command

```bash
python scripts/train_eta_phi_pt_z0_charge.py
```

defaults to `REGION=barrel`.

Do not describe a bare training command as full-detector training unless the code has been changed to support a full-detector mode.

Region-specialized training cuts:

```text
barrel model:  min_abs_eta = None,  max_abs_eta = 1.7
endcap model:  min_abs_eta = 1.3,   max_abs_eta = 3
```

The sharp evaluation boundary is:

```text
|eta| = 1.5
```

The region split is motivated by the fact that barrel and endcap showers provide different z-pointing information. Barrel geometry gives useful z0 pointing; endcap geometry is expected to sit closer to the beamspot-prior limit.

## Important files

Core package files:

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

Current train/evaluate scripts:

```text
scripts/train_eta_phi_pt_z0_charge.py
scripts/test_eta_phi_pt_z0_charge.py
```

Useful diagnostic scripts:

```text
scripts/check_dims.py
scripts/check_truth_vertex.py
scripts/diagnose_detector_regions.py
scripts/plot_charge_results.py
scripts/plot_resolution_vs_pt.py
scripts/export_charge_eval.py
```

Ruche launcher:

```text
slurm/run_train_test_new.sbatch
```

Longer project explanation may live in:

```text
README.md
docs/project_context_from_old_claude.md
docs/
```

Prefer reading these long docs only when project context is needed.

## Data paths

Canonical generated z0/charge files:

```text
data/electrons/eta_phi_pt_z0_charge/zee_pu200_z0_charge.parquet
data/electrons/eta_phi_pt_z0_charge/target_stats.json
```

Current training script paths:

```text
data/electrons/electrons.parquet
data/electrons/target_stats.json
```

The SLURM launcher creates the expected training paths inside the job working directory by copying/symlinking the z0/charge parquet and stats file.

For local training, either create equivalent symlinks or deliberately update the script paths.

Do not casually inspect, edit, or commit:

```text
data/
*.parquet
*.npz
*.pt
*.pth
*.ckpt
checkpoints/
runs/
results/
wandb/
```

When checking parquet files, prefer reading only the schema, selected columns, or one row. Avoid loading full parquet files unless explicitly asked.

## Model inputs

Each electron row corresponds to one prompt electron and a variable-length set of DBSCAN-cleaned calorimeter cells.

For speed, each electron keeps at most:

```text
max_cells = 128
```

These are the highest-energy cells from the cleaned shower.

The model uses:

1. per-cell positional inputs,
2. per-cell high-level inputs,
3. cluster-level features broadcast to every selected cell.

Current high-level feature dimension:

```text
high_level_dim = 41
```

This includes:

* 12 per-cell high-level features,
* 11 base cluster features,
* 18 radial-slice pointing-profile features.

Current output dimension:

```text
output_dim = 5
```

Output order must remain:

```text
[delta_eta, delta_phi, delta_log_pt, delta_z0, charge_logit]
```

## Physics and modeling conventions

Use these conventions unless explicitly told otherwise:

* `truth_z0` is the production vertex z coordinate from `vz`.
* `truth_charge` is a binary classification target.
* Charge is predicted from a dedicated logit.
* Charge is trained with binary cross-entropy.
* Charge is not part of the homoscedastic regression weighting.
* Charge BCE is added with a fixed manual weight.
* Phi is periodic; phi residuals must use wrapped angular differences.
* The phi residual is decoded without truth charge.
* z0 is predicted as a correction to the calorimeter pointing anchor.
* z0 performance must be compared against the beamspot-prior RMS.
* Do not quote a prediction-selected subset such as `|pred_z0| > 10 mm` as an unbiased z0 resolution.
* Do not add `d0` unless explicitly asked. It is currently parked because `vx`/`vy` are not extracted and prompt-particle transverse spread is expected to be below calorimeter resolution.

## Loss function conventions

The four regression targets use Huber / smooth-L1 style losses:

```text
eta
phi
log_pt
z0
```

The four regression losses are combined with learned homoscedastic uncertainty weighting.

Charge uses binary cross-entropy and is added separately:

```text
total_loss = total_regression_loss + charge_weight * BCE(charge_logit, charge_label)
```

Do not put charge back into the learned homoscedastic uncertainty pool unless explicitly asked.

## Setup commands

Use Python 3.10 or 3.11.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

If needed:

```bash
pip install -r requirements.txt
```

Do not change dependency versions unless explicitly asked.

Be careful with PyTorch compatibility.

## Common commands

Fast syntax check:

```bash
python -m py_compile \
  scripts/train_eta_phi_pt_z0_charge.py \
  scripts/test_eta_phi_pt_z0_charge.py \
  src/colliderml_electron/*.py
```

Dimension check:

```bash
python scripts/check_dims.py
```

Compute target stats:

```bash
python -m colliderml_electron.stats
```

Train barrel model:

```bash
REGION=barrel python scripts/train_eta_phi_pt_z0_charge.py
```

Train endcap model:

```bash
REGION=endcap python scripts/train_eta_phi_pt_z0_charge.py
```

Evaluate current checkpoint:

```bash
python scripts/test_eta_phi_pt_z0_charge.py
```

Submit Ruche jobs only when explicitly approved:

```bash
sbatch --job-name=trainTest_barrel --export=ALL,REGION=barrel slurm/run_train_test_new.sbatch
sbatch --job-name=trainTest_endcap --export=ALL,REGION=endcap slurm/run_train_test_new.sbatch
```

## Ruche workflow notes

The SLURM launcher:

* reads `REGION`,
* creates a region-specific run name,
* copies the repo into a private job working directory,
* copies/symlinks the canonical z0/charge parquet and stats file into the paths expected by training,
* runs training and testing,
* copies outputs back into `runs/`.

Do not assume there is one canonical latest checkpoint after barrel/endcap training. Region-specific outputs should be found under `runs/`.

Do not submit or cancel Ruche jobs unless explicitly approved.

## Workflow rules for Claude Code

Before editing:

* Run `git status`.
* Check the current branch.
* Identify the files that need to be changed.
* Prefer a short plan before changing dataset, model, train, test, loss, evaluation, or SLURM code.

While editing:

* Make minimal targeted changes.
* Do not rewrite large files unnecessarily.
* Do not change unrelated scripts.
* Do not change generated data.
* Do not change checkpoints.
* Do not change W&B runs.
* Do not change result plots or archived run outputs unless explicitly asked.
* Do not run full training unless explicitly approved.
* Do not run testing that loads large datasets unless explicitly approved.
* Do not run `sbatch`, `scancel`, `rsync`, `git push`, `git reset --hard`, `git clean`, or destructive shell commands unless explicitly approved.
* Do not install, upgrade, or remove packages unless explicitly approved.

After editing:

* Run a fast validation command when possible.
* Prefer `py_compile` for Python syntax validation.
* Run `scripts/check_dims.py` when model input dimensions or feature construction change.
* Show `git diff --stat`.
* Summarize exactly what changed.
* State what still needs manual review.

## Git safety

Allowed without special approval:

```bash
git status
git diff
git diff --stat
git branch --show-current
git log --oneline -n 5
```

Require explicit approval:

```bash
git add
git commit
git push
git pull
git merge
git switch
git checkout
git reset
git clean
```

Never use destructive git commands unless explicitly asked and the user confirms the exact intent.

## Preferred Claude behavior

For explanation tasks:

* Read relevant files.
* Explain the code path simply.
* Cite exact files and functions when useful.
* Do not edit files.

For code tasks:

* Start with a plan.
* Make the smallest safe patch.
* Validate quickly.
* Show the diff summary.
* Avoid broad refactors.

For ML experiment tasks:

* Separate scientific changes from infrastructure changes.
* Do not silently change losses, target definitions, normalization, cuts, output ordering, or evaluation metrics.
* Make comparisons fair: region-specialized models should be compared against the corresponding single-model per-region baseline, not only against each other.
* Treat endcap null results as physically meaningful if the model remains near the z0 prior.

## Good first tasks for Claude Code

Useful safe prompts:

```text
Read the train, test, dataset, model, and SLURM files. Summarize how REGION flows through the region-split experiment. Do not edit files.
```

```text
Plan only. Check whether the parquet path, stats path, high_level_dim, output_dim, checkpoint path, and SLURM launcher are internally consistent. Do not edit files.
```

```text
Explain how charge is trained and evaluated. Do not edit files.
```

```text
Explain how z0 is anchored, predicted, and evaluated against the beamspot prior. Do not edit files.
```

```text
Make a minimal patch to add command-line arguments to the test script for checkpoint, parquet, stats, and output directory. Do not change model logic or metrics. After editing, run py_compile and show git diff --stat.
```

Avoid broad prompts such as:

```text
Fix everything.
Make the model better.
Clean up the repo.
Run the experiment.
Refactor the project.
```
