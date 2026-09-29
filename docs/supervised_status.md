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

The test_metrics.json files tracked in results/ are from earlier, pre-AttnPool
July runs; the AttnPool reference below has no tracked metrics file.

Supervised AttnPool reference: branch `attnpool-200ep`, commit `770ba8a`, Lyon
job 55426542, 200 epochs, batch 96, 236,400 optimizer steps. On the paired
population of `unsup_clustering_summary.md` (electrons reconstructed by both
pipelines):

- barrel, n = 10,825: charge AUC 0.955, phi sigma 0.0056 rad, eta sigma 0.019,
  pT sigma 3.0 percent, z0 sigma 34.8 mm
- endcap, n = 3,139: charge AUC 0.882, phi sigma 0.0093 rad, eta sigma 0.017,
  pT sigma 2.9 percent, z0 sigma 55.0 mm (beamspot prior about 57 mm)

These are Gaussian-core sigmas on a selected population, not full-test-set
RMSEs. That population is also incomplete: the matching dropped about 15
percent of genuine pairs, more in the endcap (see the caveats in
`unsup_clustering_summary.md`), so these values need to be recomputed before
they are quoted. Full-population numbers for this checkpoint still need to be
pulled from W&B or regenerated before quoting a headline.

## What the tracked July runs show

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

- The champion run directory (`attnpool-200ep`, `770ba8a`, job 55426542) on
  Lyon is
  /pbs/home/l/llambert/cc-attn200/runs/lyon_eta_phi_pt_z0_charge_full_seed0_20260730_104715_55426542
  Its full test-set metrics are not in the repo yet; rerun
  scripts/test_eta_phi_pt_z0_charge.py on it to get them.
- All paired supervised vs truth-free numbers (here and in
  `unsup_clustering_summary.md`) need recomputing with the fixed matching in
  `compare_regions_bootstrap.py`. Before that, decide whether the headline
  resolutions use all pT or pT >= 10 GeV to match the acceptance population.
  `compare_preds_bootstrap.py` still uses rounded keys and should not be used
  with `--cross-dataset` until it gets the same fix.
- The 0.0065 seed sd still needs its seed list from W&B. There is no seed
  variance for any of the resolutions.
- Only the `PER_REGION_PROJ=1` run exists. The flag-off twin (same parquet,
  seed and N_EPOCHS) was never run, so there is no result for this yet.
- The about 90,000 steps to charge lift-off comes from earlier configurations.
  AttnPool lifted off earlier and its step count at lift-off was not measured.
- Why the supervised pipeline drops some electrons is not traced in the code.
  The guess is zero truth-linked cells. The 14.9 percent figure that seemed to
  support this was a matching artifact (see `unsup_clustering_summary.md`).
- Scoring the supervised checkpoint on the truth-free dataset (zero-shot) was
  never run.
