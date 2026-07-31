# Experiment log

Record of what has been tried in this project and what came of it.

Task: reconstruct prompt-electron eta, phi, log pT, z0 and charge from
DBSCAN-cleaned calorimeter cells (ColliderML Release 1, `zee_pu200`), and use the
results to characterise tracker/calorimeter complementarity.

Status labels used below: ADOPTED, RETIRED, NEGATIVE, TIE, OPEN.

Definitions used throughout:

- v1 = the ~30k-electron dataset used up to mid-July 2026.
  v2 = `zee_pu200_supervised_dbscan_v2.parquet`, 178,602 electrons from 96,553
  events, split-harmonised against v1.
- Registered evaluation population: truth pT >= 10 GeV, full detector
  (|eta| <= 3), with barrel and endcap breakdowns. In `train_*.py` the region
  cuts are barrel |eta| <= 1.7 and endcap 1.3 <= |eta| <= 3. These overlap in
  1.3-1.7 and neither matches the ECal transition at |eta| = 1.5, so the
  "barrel" numbers include the transition region.
- Comparisons between runs are decided by paired bootstrap
  (`compare_preds_bootstrap.py`, 2000 resamples), 95% CI excluding zero.
  On v1-scale data, single-run charge AUC differences below about 0.02-0.03
  are inside seed noise and are not evidence of anything.

Numbers below are transcribed from run logs and analysis notes. Re-check them
against the relevant `test_metrics.json` before quoting any of them externally.

---

## 1. Dataset and pipeline

### 1.1 Cone association replaced by supervised DBSCAN cleaning - ADOPTED

The first per-electron table collected cells in a cone. This was replaced by
idealised supervised DBSCAN cleaning. Cells are matched to prompt electrons
(`pdg_id == +-11`, `primary == True`, `vertex_primary == 1`) by walking
`contrib_particle_ids`, then DBSCAN removes pileup, secondaries and noise.
Training and evaluation tables must both be built this way so both see the same
kind of shower. All later work uses DBSCAN tables.

### 1.2 Energy calibration - ADOPTED

`calo_hits.total_energy` is uncalibrated. The per-subsystem scaling factors from
the release paper are applied before anything is quoted in GeV.

### 1.3 v2 build on atlas - ADOPTED

The full `zee_pu200` set was processed on `atlas1.ijclab.in2p3.fr` in 20 chunks,
downloading, processing and deleting raw shards per chunk, since the ~2.7 TB raw
dataset does not fit on disk at once. Output was transferred to Lyon THRONG
storage. Fixes made during the build:

- `DEFAULT_TARGET_COLS` was missing `truth_z0` in the fallback path of
  `fetch_and_cluster.py`.
- `assign_splits()` permuted the full event-id array, so growing the dataset
  would have silently reassigned v1 test events. Fixed by `harmonize_splits.py`,
  which preserves the v1 train/val/test assignments verbatim (verified: zero
  split flips) and recomputes `target_stats.json` on the harmonised train split
  only. v1 stats files are never overwritten.
- Added `--shard-min` / `--shard-max` and `--download-retries` with exponential
  backoff for chunked, resumable processing.
- `HF_HUB_DISABLE_XET=1` is required, otherwise the HuggingFace Xet backend
  hangs on a futex.

### 1.4 Positional alignment in `pipeline.py` - OPEN

The production loop assumes positional row alignment between paired
particles/calo shards without checking `event_id` equality. `io.get_event()`
does check. A `RuntimeError` guard has been written but should be confirmed in
place before the next rebuild.

---

## 2. Targets

### 2.1 eta, phi, pT - ADOPTED

These are what the calorimeter does well. On v1, pT >= 10: eta sigma about
0.019 with negligible bias; pT core resolution about 3% with a non-Gaussian tail
around 10% from pileup contamination and leakage; barrel phi sigma about
0.023 rad. On v2 barrel phi sigma reached 0.0064 rad.

### 2.2 z0 - TIE against the prior overall, geometry-limited

Full detector sigma is about 48-51 mm against a beamspot prior of about 56 mm
(R^2 about 0.20). Split by region the picture is clearer: barrel genuinely beats
its prior (38.5 mm vs 54.5 mm), endcap sits at its prior (56.8 mm). The
eta/z0 coupling diagnostic showed the two share roughly 88-97% of their variance
through the unknown vertex z, with rho between -0.94 and -0.99 per |eta| bin and
an effective shower radius of 1222-1464 mm consistent with shower depth. This is
a detector-geometry ceiling, not a capacity limit. z0 is kept as a target and
reported as the observable that most needs the tracker.

### 2.3 z0 from calorimeter timing - NEGATIVE

`diagnose_z0.py` gave prior 64.9 mm, FAIR (time of flight with t0 profiled out)
336.9 mm, IDEAL (t0 fixed to 0) 111.6 mm. Timing is worse than the prior even
under idealised assumptions, because the fitted t0 spread is much larger than
the vertex spread being resolved. Per-cell time was therefore not adopted as a
z0 feature. The gentler time-consistency cleaning mask is a separate idea and
remains untested as an ablation for the phi and pT tails.

### 2.4 d0 - NEGATIVE, not attempted

Rejected on physics grounds. The transverse luminous region is micron-scale, so
truth d0 for prompt electrons is effectively a delta function at zero, far below
anything the calorimeter could resolve.

### 2.5 Charge as a fifth head - ADOPTED

The charge sign is imprinted upstream of the calorimeter: the solenoid field
deflects e+ and e- in opposite azimuthal directions before they arrive, leaving a
charge-dependent phi offset (about +-0.035 rad in v2, +-0.014 rad in v1). The
calorimeter has no field and does not bend anything; it records where the
already-deflected particle landed.

v1 champion barrel charge AUC is about 0.90. Accuracy vs pT is non-monotonic:
near chance at very low pT where the shower is too poor to measure the bend, a
peak around 0.77-0.78 at 7-9 GeV, then a slow decline at high pT as the 1/pT
bend falls below the centroid resolution. The output probabilities are close to
calibrated (reliability curve near the diagonal, slightly overconfident at the
top end), so the logit is usable as a per-event reliability weight rather than
just a yes/no.

---

## 3. Architecture and decoding

### 3.1 Two-head phi with truth-charge routing - RETIRED

The original phi scheme used truth charge at decode time to pick between two phi
heads, which made the reported phi resolution optimistic since the model never
had to determine the charge itself. Replaced by a single signed phi head plus a
dedicated BCE charge logit, with no truth information used anywhere at decode
time. This is now a fixed constraint on the design.

### 3.2 Homoscedastic weighting applied to the charge head - RETIRED

Putting charge in the learned uncertainty-weighting pool collapsed the charge
gradient. The BCE term sits at a scale of about 0.69 while the Huber terms are
around 1e-3, so the optimiser gains by inflating the charge log-variance, which
suppresses the gradient to near zero. Charge sat at chance (AUC 0.523) with a
by-charge phi residual plot that clearly showed separation, which is what made
the diagnosis possible.

Fix: remove charge from the learned pool and give it a fixed manual
`charge_weight = 1.0`, passed as a constructor argument with a matching
`cfg["charge_weight"]` entry. Charge AUC went to 0.816 and accuracy to 0.723
immediately. Regression targets keep homoscedastic weighting.

This is worth writing up as a methodological result in its own right:
homoscedastic weighting fails silently when one head's loss is on a different
scale from the rest.

### 3.3 Checkpoint selection blind to charge - RETIRED, replaced by charge-aware selection

The charge head does not begin to learn until roughly epoch 47-50 at the
champion configuration, while total validation loss is dominated by the four
regression heads. Two failures followed from that. The best-checkpoint save
always fired on an early regression plateau where charge was still at chance, so
trained charge weights were discarded (one run trained charge to 0.565 by epoch
54 and reported 0.5036 at test). Early stopping fired as soon as `min_epochs`
allowed, because `epochs_no_improve` had been counting from the regression
plateau around epoch 20.

Fix, applied to both the checkpoint save and the patience counter:

```python
charge_acc = val_logs["charge_acc"]
selection_score = val_loss - 2.0 * (charge_acc - 0.5)
```

with `n_epochs = min_epochs = 100`, which disables early stopping entirely so the
charge climb cannot be truncated. This is the change that made the v2 runs work.

### 3.4 Encoder scaling - OPEN, charge did not learn

Scaling to `n_layers=6`, `model_dim=256`, `batch_size=256` (from 3, 128, 64) left
the regression heads healthy (`phi_sigma_rad` 0.0193, matching the champion) but
charge flat at chance for 120 epochs. Two things came out of this:

- The `phi RMSE` column is a broken metric. It sits at 0.196 rad from epoch 1 to
  epoch 120 while the per-head Huber loss on phi improves normally. 0.196 rad is
  approximately the RMS of the raw `dphi_target` distribution, so the reported
  quantity is a constant of the data rather than the model residual. The
  Gaussian-core sigma numbers are the honest ones. This affects the test script
  as well (`phi_rmse` 0.215 vs `phi_sigma_rad` 0.0193).
- The likely cause of the charge failure is that lift-off is a step-count
  phenomenon, not an epoch phenomenon. Batch 256 gives roughly a quarter of the
  optimiser steps per epoch, so the lift-off that happens near epoch 47 at batch
  64 would land near epoch 190. The run stopped at 120. This has not been tested
  yet: check steps per epoch, then either extend the budget or return to batch 64.

An earlier hypothesis for the same failure, that phi resolution had collapsed and
charge was a downstream casualty, was falsified once the broken metric was
understood. Phi was fine.

### 3.5 Fourier position embedding range - OPEN

`model.py` sets `dim_max=[1100.0, 1100.0, 3000.0]`. Barrel cells reach r beyond
1700 mm, so |x| and |y| probably exceed 1100 for a substantial fraction of cells.
If so, even the lowest-frequency positional component wraps past a full cycle and
the encoding is non-injective over the detector volume. Needs the coordinate-range
check; if confirmed, the fix changes the input representation and therefore needs
its own branch and control.

### 3.6 Depth-ordered convolution - OPEN

Flagged in the `ConvCaloRegressor` docstring as an untried lever. Not yet run.

---

## 4. Features

### 4.1 Canonical azimuthal frame and cluster moments - ADOPTED

Each event is rotated in azimuth so the energy-weighted phi centroid sits at 0,
which keeps per-cell coordinates in a consistent range. Cluster scalars cover log
energies, cell count, phi and eta spread and skew, and the pointing anchor and
slope. These are computed over the full cleaned shower before the
`max_cells = 128` top-energy truncation, so total shower energy survives the cut.

### 4.2 K=6 radial-slice pointing profile plus fit-quality scalars - ADOPTED (champion)

Energy-weighted <z>, <r> and energy fraction in K=6 radial slices (`prof_z`,
`prof_r`, `prof_f`, 3K = 18 features), plus `r_spread` and `fit_rms` so the
network can tell when the pointing fit is unreliable and fall back to the
beamspot prior. Width arithmetic: 12 angular and detector + 11 base cluster + 18
profile = 41. This is the Jun23 champion feature set
(`ruche_Jun23_ConstChargeWeight.pt`, barrel charge AUC about 0.9026).

Note that the profile and the two fit-quality scalars have to be restored
together; restoring the slices alone gives 39 and a dimension mismatch.

### 4.3 Azimuthal energy skew as a charge tag - NEGATIVE

The working assumption had been that `skew_phi` tags the direction of the
bremsstrahlung tail and therefore the charge. Direct test: the skew distributions
for electrons and positrons are essentially identical, symmetric and centred on
zero, and a skew-only classifier is at or below chance in every pT bin
(all-cells ceiling 0.542 at high pT, 0.476 at low pT). A classifier below 0.5 on
a subset means no signal at all.

The reconciliation matters, because the model reaches 70-78% charge accuracy on
the same test set. The charge information is in the Delta-phi versus depth
correlation carried by the 3D cell positions, which is a trajectory measurement.
The depth-integrated skew averages that away, which is exactly why the skew is
flat while the model is not. The "skew tags the charge" claim was removed from
the presentation.

### 4.4 Adaptive-depth `phi_slope` - RETIRED

Follow-up to 4.3: a geometry-correct depth coordinate (barrel to r, endcap to
|z|) for the phi-versus-depth slope. Standalone, the adaptive-depth slope peaks
at about 0.72 at mid-pT, tracking the model's own 0.78 peak and confirming that
the earlier problem was the depth coordinate. Profile-LDA on the full per-slice
profile came out at chance, so only the single scalar was added (41 to 42). In
paired bootstrap it did not beat the baseline and was retired with the K=12
bundle in 4.6.

### 4.5 Feature instability fix, commit `96368ca` - ADOPTED

A mask-aware probe found pathological tails from near-degenerate pointing fits:
`prof_z` reaching absmax 326-449 against a p99 of 25-31, and `slope` reaching 123
against a physical ceiling of sinh(3) ~ 10. Fix: variance floors
(`var_r > MIN_VAR_R = 25.0` mm^2 for the z0 pointing fit,
`var_d > MIN_VAR_D = 25.0` mm^2 for the phi_slope fit) and a +-50 clip on the
assembled `cluster_feats` array. Well-conditioned events are unaffected.

Consequence: any checkpoint trained before this commit has an uncertain feature
distribution and its scores are not interpretable. Resolving this for the
pointing-upgrade checkpoints, via wandb commit logs or empirical rescoring, is
still open.

### 4.6 K=12 profile, `phi_slope` and pT floor together (combo-floor) - TIE on v1, partial win on v2

On v1: K=12 underperformed the baseline on barrel z0 (43.2 vs 38.5 mm across two
independent runs), and the full bundle was a statistical tie on charge AUC.
The decisive number was the rep1/rep2 comparison of two identical combo-floor
trainings, which diverged by Delta-AUC -0.024 with the CI excluding zero. That
established the 0.02-0.03 AUC seed-noise floor and is why single-run charge
comparisons on v1-scale data are not trusted. The pre-registered rule retired
K=12, `phi_slope` and the training floor, and the 41-feature configuration went
into v2 as champion.

On v2, with six times the data, combo-floor showed statistically significant
regression improvements over retreat-control on eta, phi and pT in a 2000-resample
paired bootstrap, a tie on z0 (consistent with the geometry ceiling), and a tie on
charge (0.9474 vs 0.9488 barrel AUC).

---

## 5. Population and region studies

### 5.1 pT >= 10 GeV evaluation floor - ADOPTED as a population definition

This is a physics population boundary, not a cosmetic filter. Regression and
charge have opposite pT dependence, since charge is easiest at low pT where the
bend is largest, so the choice of floor changes which result you are optimising.
Decision: train on all pT, where sub-10 GeV acts as charge curriculum, and
evaluate only above 10 GeV. Always cut on truth pT, never predicted pT.
Population integrity is checked by the acceptance chain printed to stdout
(4967 -> 4479 -> 4005, n = 2149 barrel) before any metric is trusted.

### 5.2 Barrel/endcap boundary - ADOPTED

Determined empirically at |eta| = 1.5 with `diagnose_detector_regions.py`,
replacing the assumed 1.2. Note this does not match the region cuts actually used
in the training scripts (see the definitions at the top).

### 5.3 Barrel specialist model - NEGATIVE

A barrel-only model lost to the full-detector baseline evaluated on barrel:
eta 12% worse, pT 6% worse, z0 12% worse. The dominant factor is training
statistics, around 21k electrons at v1 scale. Region specialisation was retired.
It could be revisited at v2 scale if there is a reason to.

### 5.4 Endcap charge ceiling - revised by v2

The v1 endcap charge AUC of 0.68-0.72 was attributed to physics, on the grounds
that forward trajectories are nearly parallel to the solenoid field. v2 reached
0.785, so part of what looked like a ceiling was statistics. The barrel/endcap
gap persists and the geometric argument still holds directionally, but the v1
number should not be quoted as a physics limit.

---

## 6. v1 to v2 summary

| Metric, pT >= 10 | v1 champion (Jun23, A100) | retreat-control, v2 (H100) | combo-floor, v2 |
|---|---|---|---|
| Barrel charge AUC | 0.9026 | 0.9488 | 0.9474 (tie with RC) |
| Endcap charge AUC | 0.68-0.72 | 0.785 | not recorded here |
| Barrel phi sigma (rad) | 0.023 | 0.0064 | better than RC, significant |
| eta, pT | baseline | reproduced | better than RC, significant |
| z0 | barrel 38.5 mm, endcap at prior | ceiling-consistent | tie with RC |

Cross-cluster reproducibility: Lyon H100 reproduced the Ruche A100 champion on v1
within noise (barrel charge AUC 0.9009 vs 0.9026), which validates Lyon as a
compute stack.

A phi/charge coupling diagnostic on v2 confirmed that charge correctness is a
proxy for the quality of the azimuthal bend measurement rather than a causal
driver of phi error: the wrong-charge residual distribution is unimodal and
centred, not bimodal near +-0.068 rad as a sign-flip mechanism would require.

---

## 7. Engineering lessons and the guards that came out of them

The branch is the experiment. `dataset.py` and the training script are forked per
experiment rather than parameterised, so feature-set identity is carried by which
branch you are standing on and not by anything inside the checkpoint. Guards:

- Verify `torch.load(f)["config"]["high_level_dim"]` before scoring any
  checkpoint. This exists because a checkpoint labelled `retreat-control` turned
  out to carry `high_level_dim=60`, identical to combo-floor.
- Always score a checkpoint through its own branch's dataset code. Never mix
  feature sets across branches.
- `check_dims.py` preflight, checking `high_level_dim` and `output_dim`, as a
  hard gate before every submission.
- Separate clone directories per branch on each cluster
  (`~/colliderml-electron-calo`, `~/cc-combo`) so queued jobs cannot race on the
  checked-out branch.

Submission discipline:

- Never submit against unpushed code; verify the commit hash before `sbatch`.
- Run `git status` and the import check
  (`python -c "from colliderml_electron.dataset import ElectronDataset, make_loader, TARGET_COLS"`)
  first. This caught a corrupted `dataset.py` that had been overwritten with the
  contents of the training script, producing a self-import at the top of the file.
- Use `env VAR=val python ...` single-line form with `$HOME`, not `~`.
- Export `REGION`, `MIN_PT_EVAL`, `SOURCE_Z0_CHARGE_FILE` and `SOURCE_STATS_FILE`
  at submit time; `test_eta_phi_pt_z0_charge.py` reads environment variables
  exclusively and has no argparse interface.
- `bash -n` on every sbatch script before launching.
- Confirm GPU architecture with `nvidia-smi` per run, since A100 and H100 runs
  are being compared.

SLURM robustness: `#SBATCH --requeue` (a NODE_FAIL once made a job vanish
silently, because it bypasses bash EXIT traps), `#SBATCH --signal=B:TERM@300`,
and atomic checkpoint saves that write to `.tmp` and rename, with a `"partial"`
marker.

Memory: eager `pl.read_parquet()` on the full merged parquet causes transient
double-copy spikes during multi-split loader construction. Use
`pl.scan_parquet(...).filter(pl.col("split") == split).collect()` instead.

Reporting:

- Report Gaussian-core sigma alongside RMSE, since the tails dominate RMSE.
- Phi scatter plots must be wrap-aware, using `angular_residual()`, or points
  near +-pi appear as large outliers. The underlying metrics were already
  correct; this was a plotting-only bug.
- Dual-axis accuracy against mean |logit| is misleading because of axis scaling
  and the sigmoid nonlinearity. Use per-event mean confidence, max(p_pos, p_neg).
- Show the beamspot prior next to any z0 number. Sigma 48 mm reads as a success
  in isolation and reads correctly against a 56 mm prior.

---

## 8. Open items

1. Test the step-count hypothesis for charge lift-off under the scaled encoder
   (section 3.4): count steps per epoch, then extend the budget or return to
   batch 64.
2. Fix or remove the broken `phi RMSE` metric in the training loop and test
   script (section 3.4).
3. Resolve the provenance of the pointing-upgrade checkpoints relative to commit
   `96368ca` (section 4.5).
4. Add the `event_id` alignment guard to `pipeline.py` (section 1.4).
5. Check the true cell coordinate extents against the Fourier `dim_max`
   (section 3.5).
6. Realistic unsupervised-clustering dataset: build matched-only first, harmonise
   splits against the v2 supervised parquet so supervised and realistic can be
   compared per event on identical events, and leave `--keep-unmatched` to a
   later pass for electron-ID and fake-rejection studies.
7. Update the README on `main`, which still describes the exploration-phase
   scaffolding and lists the loader, geometry helpers and plot gallery as
   unchecked TODOs.
8. Summary table and bootstrap figure for David and Tuan.