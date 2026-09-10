# Supervised reconstruction: status note

Status note, 09 September 2026. L. Lambert.
Dataset: ColliderML Release 1, zee_pu200 (Z -> ee, PU200, OpenDataDetector).

This is the supervised counterpart to docs/unsup_clustering_summary.md. It
records the current method and the state of results. It deliberately does not
crown a single champion: see "Champion selection" below.

## Method

Per-electron regression from truth-associated calorimeter cells to five targets:
eta, phi, log(pT), z0, and charge sign. Architecture is AttnPoolCaloRegressor
(transformer cell encoder + learned-query cross-attention pooling), trained
model_dim 128 / 3 layers / 4 heads, high_level_dim 41, max_cells 128, output_dim
5. eta/phi/pT/z0 are anchored residuals; charge is a dedicated BCE logit. The
four regression losses share homoscedastic uncertainty weighting; charge uses a
fixed manual weight, because putting BCE under the learned weight collapses its
gradient. Full architecture detail is in the top-level README.

## Current results

All numbers are from the tracked test_metrics.json of each run. Charge is ROC
AUC; eta, phi (rad), pT (relative), and z0 (mm) are RMSE.

Full acceptance (ab/baseline, ab/candidate):

    AUC   0.89 - 0.90
    eta   0.021 - 0.023
    phi   0.016 - 0.017 rad
    pT    0.052 - 0.054 (relative)
    z0    42 - 47 mm   (beamspot prior 54.6 mm)

Barrel, pT > 10 GeV, three successful seeds (benchmark/baseline, combo_rep1,
combo_rep2):

    AUC   mean 0.888, sample sd 0.017  (0.869 - 0.903)
    eta   0.021 - 0.022
    phi   0.016 - 0.017 rad
    pT    0.051 - 0.090 (relative; combo_rep2 is the high tail)
    z0    39.2 - 39.6 mm  (beamspot prior 54.1 mm)

The model beats the beamspot prior on z0 in every current run and reaches charge
AUC around 0.89 in the barrel. z0 is calorimeter-limited: the pointing anchor
alone is far worse than the beamspot prior in these runs, and the model
recovering to ~39 mm barrel / ~42-47 mm full is close to the expected ceiling
for pointing-based z0.

## Known failure mode in the run set

Two barrel benchmark seeds (pointing_rep1, pointing_rep2) and both full
pointing_upgrade runs show charge AUC near 0.5-0.69: the charge head never
lifted off. This is the documented step-starvation mechanism, not overfitting -
the charge head needs on the order of 90,000+ optimizer steps before lift-off,
and phi resolution is the leading indicator (the failed runs also have degraded
phi, ~0.20 rad vs ~0.016 rad in the successful runs). These runs are retained as
negatives, not deleted.

## Seed variance

The three successful barrel benchmark seeds give charge AUC sample sd ~0.017.
Note: an earlier working figure of sd ~0.0065 has been used for v2 barrel charge
AUC. That tighter value is not reproduced by the three tracked benchmark runs
here and may derive from a different (e.g. W&B-logged) comparison set. This
should be reconciled against W&B before either number is quoted to supervisors.
Any single-run charge comparison must be evaluated against whichever seed-
variance floor is confirmed, using the pre-registered paired bootstrap
(compare_preds_bootstrap.py, 2000 resamples, 95% CI excluding zero).

## Champion selection (open)

No tracked run directory carries a training config or checkpoint hash, so the
exact training commit for each result cannot be recovered from the results files
alone. Metric schema identifies the current-method era but not the precise code.
Finalizing a single reported champion requires cross-referencing the W&B run
configs to confirm architecture, feature dimension, and dataset for the chosen
run. This is the main outstanding step before a headline result is quoted.

## Open items

- Confirm champion provenance against W&B; then fix the reported headline run and
  its exact metrics here and in results/README.md.
- Reconcile the seed-variance figure (0.017 from tracked runs vs 0.0065 working
  value).
- Region-projection ablation: a PER_REGION_PROJ=1 run exists, but the matched
  PER_REGION_PROJ=0 twin on the same parquet and seed has not been run. No
  comparison verdict yet.
- Zero-shot measurement (supervised checkpoint scored on truth-free data with
  supervised stats) has not been executed.
- Confirm exactly why the supervised pipeline drops certain electrons (suspected:
  zero truth-linked cells for soft or geometrically gapped electrons).
