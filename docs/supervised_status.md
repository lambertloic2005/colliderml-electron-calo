# Supervised electron reconstruction

## The method

Each prompt electron is reconstructed from its truth-associated calorimeter
cells into eta, phi, log(pT), z0, and charge sign. The network is
AttnPoolCaloRegressor (transformer cell encoder, learned-query cross-attention
pooling), model_dim 128, 3 layers, 4 heads, 41 high-level features, 128 cells
per electron. eta, phi and log(pT) are predicted as corrections to physics
anchors. z0 is regressed directly in z-scored units; the pointing-fit anchor is
an input feature, not a residual base. Charge is a separate BCE logit. The four
regression losses share a learned homoscedastic weighting; charge (BCE) sits outside it on a fixed weight of 1.0. It was originally a fifth task under the learned weighting; in the Jun23 runs the head did not train there and did train on the fixed weight. The mechanism was not isolated: at equilibrium the learned weight for a BCE term is 1/(2L), about 0.72 at chance, so it is not driven to zero by the formula, and the comparison is one run each with loss-based early stopping (stopping epochs not recorded).

## Reference run

Supervised AttnPool reference: branch `attnpool-200ep`, commit `770ba8a`, Lyon
job 55426542, run directory
/pbs/home/l/llambert/cc-attn200/runs/lyon_eta_phi_pt_z0_charge_full_seed0_20260730_104715_55426542
W&B (run offline, synced afterwards): training a7sjsrl7, evaluation i6mf98h5.
The on-disk test_metrics.json is the authoritative copy and is tracked as
results/attnpool_200ep_full/test_metrics.json. The run predates provenance.txt,
so the parquet md5 is not recorded. Checked instead: the test script md5
matches 770ba8a; the checkpoint config is attnpool 128/3/4, high_level_dim 41,
batch 96, seed 0; the acceptance cuts read train 124,957 -> 113,427, val
26,813 -> 24,184, test 26,832 -> 24,196.

The run trained for 200 epochs (236,400 optimizer steps), but the saved
checkpoint is from epoch 49 (about 57,900 steps), the epoch with the lowest
selection score (see README, Checkpoint selection). The runner-up, epoch 38,
scores 0.015 worse, far above log rounding. After epoch 49 train loss keeps
falling and val loss rises. Of the per-task val quantities, the phi loss worsens
by about 10 percent and the ln pT RMSE by about 7 percent by epoch 191, eta is
flat, and val charge accuracy stays near 0.81. Most of the rise in val
loss_total comes from the learned sigmas tracking the falling train loss, not
from val degradation. Every number below is from the epoch-49 weights.

Test split, |eta| <= 3, truth pT >= 10 GeV (n = 21,607, the same population as
the acceptance denominator in unsup_clustering_summary.md). Resolutions are
3-sigma-truncated core RMS, tail fraction in brackets.

- barrel, n = 11,799: charge AUC 0.956 (acc 0.886), phi sigma 0.0054 rad
  (11.0%), eta sigma 0.019 (0.7%), pT sigma 2.9% (2.4%), z0 RMSE 37.0 mm
  against a 54.9 mm prior
- endcap, n = 9,808: charge AUC 0.835 (acc 0.740), phi sigma 0.0072 rad
  (5.7%), eta sigma 0.016 (0.4%), pT sigma 2.7% (6.5%), z0 RMSE 55.4 mm
  against a 55.7 mm prior

All pT (n = 24,196): barrel AUC 0.951, phi 0.0058 rad (15.5%), eta 0.020
(2.9%), pT 3.1% (6.9%), z0 RMSE 39.2 mm (prior 55.2); endcap AUC 0.832, phi
0.0077 rad (11.4%), eta 0.017 (2.3%), pT 2.8% (10.9%), z0 RMSE 58.2 mm (prior
58.5). The all-pT RMSEs (phi 0.20 rad, relative pT 52%) are dominated by
electrons below 5 GeV, 6 percent of the sample, and should not be quoted.

The phi tail is mostly the charge failure mode. In the barrel at pT >= 10 GeV,
wrong-charge electrons are 11.4 percent of the sample but 71 percent of the phi
tail, and their residual carries the sign of the uncorrected bend (mean
q * residual -0.025 rad, against -0.002 rad for correct calls). For correctly
charged barrel electrons phi sigma is 0.0045 rad. The endcap shows the same
pattern (74 percent of the tail, 26 percent of the sample).

pT-cut numbers come from preds.npz via scripts/metrics_from_preds.py, which
reproduces test_metrics.json exactly with no pT cut. One seed; no seed
variance exists for the resolutions. This is the v2 test population; the July
numbers below are on v1, so the two are not a paired comparison.

## What the tracked July runs show

All tracked July runs were trained and tested on the v1 dataset (about 30k
electrons), not the current v2 table (178,602 electrons). Their test populations
(2,452 electrons at pT > 10 GeV, |eta| <= 1.7; 4,479 at |eta| <= 3, all pT) are
therefore not the population behind the AttnPool numbers above, and the two sets
of values should not be compared directly.

`results/ab/` (Jul 06) and `results/benchmark/` (Jul 17) predate AttnPool
(added Jul 28). The ab runs were scored with pT > 10 GeV and, from the region
counts (2149 barrel, 303 endcap), an |eta| <= 1.7 cut, so their endcap is only
the 1.5-1.7 transition. They are not full acceptance.

On |eta| < 1.5, pT > 10 GeV, the three benchmark runs whose charge head trained
(baseline, combo_rep1, combo_rep2) give charge AUC 0.869-0.903, phi RMSE
0.016-0.017 rad, and z0 RMSE about 39 mm against a 54 mm prior.
baseline_barrel_pt10 and ab/baseline are the same model.

z0 beats the beamspot prior in the barrel. It does not in the endcap: the ab
runs give 58.6 and 66.7 mm in the 1.5-1.7 band against a 58.0 mm prior, and the
AttnPool run sits at the prior across the endcap.

## Charge failures

pointing_rep1 and pointing_rep2 have charge AUC 0.52 and 0.69, with phi RMSE
0.023 and 0.021 rad against 0.016-0.017 in the runs that trained (core sigma
0.020-0.023 vs 0.010-0.011). The two Jul08 pointing_upgrade runs also failed
(AUC 0.52, 0.62); their ~0.20 rad phi RMSE is tail-dominated and not comparable,
since they were scored at |eta| <= 3 with no pT cut.

Worse phi alongside a failed charge head fits phi resolution being the leading
indicator for charge. Whether these runs failed from step starvation is not
verified: they use a different configuration from the combo runs and their step
counts are not recorded here.

## Seed variance

The between-seed sd of barrel charge AUC on the current dataset generation is
0.0065. The runs behind this number are not recorded in the repo; the seed list
has to be recovered from W&B. The 0.017 spread of the three July runs is not a
seed variance: they predate AttnPool, appear to mix two configurations
(baseline and combo were scored with different test-script versions), and were
selected on the charge head having trained.

## z0

In the barrel the network gets well below the prior. Whether this is a
calorimeter-only ceiling is open; the poor anchor-only RMSE (300-540 mm) shows
the anchor is noisy, not that the network has saturated.

## Open items

- The champion read train/val counts of 124,957 / 26,813; the table in
  unsup_clustering_summary.md says 124,921 / 26,849, and 124,906 is quoted
  there as the pre-cut train count. Totals and the test count agree. Find the
  parquet the job staged and check its md5 and split counts.
- Checkpoint selection on loss_total mostly tracks the train/val gap through
  the learned sigmas. A selection on fixed-weight val metrics would be cleaner.
- All paired supervised vs truth-free numbers (here and in
  `unsup_clustering_summary.md`) need recomputing with the fixed matching in
  `compare_regions_bootstrap.py`. Before that, decide whether the headline
  resolutions use all pT or pT >= 10 GeV to match the acceptance population.
   `compare_preds_bootstrap.py` still uses rounded keys and should not be used
  with `--cross-dataset` until it gets the same fix. The same applies to fig1 of
  `make_summary_figs.py` (its `match()` also pairs on rounded truth eta, phi and
  pT), so that figure carries the same roughly 15 percent pairing loss.
- The 0.0065 seed sd still needs its seed list from W&B. There is no seed
  variance for any of the resolutions.
- Only the `PER_REGION_PROJ=1` run exists. The flag-off twin (same parquet,
  seed and N_EPOCHS) was never run, so there is no result for this yet.
- The about 90,000 steps to charge lift-off comes from earlier configurations.
  For the AttnPool champion, val charge accuracy was 0.49 at epoch 1 and 0.788
  at epoch 11 (13,002 steps), so lift-off happened within the first 11 epochs;
  the exact epoch was not extracted.
- Supervised electron loss is traced in code: build_electron_row drops an
  electron only when its descendant family has zero truth-linked cells (the DBSCAN
  clean falls back to keeping all cells and cannot empty it); duplicates on
  rounded (px, py, pz, pdg) are also collapsed. The per-worker "skipped by cuts"
  counts in the atlas build logs give the number. Which electrons these are
  (acceptance, soft, gaps) is not characterized.
- Scoring the supervised checkpoint on the truth-free dataset (zero-shot) was
  never run.
