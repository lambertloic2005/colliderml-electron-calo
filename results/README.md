# Results index

Each subdirectory holds evaluation outputs (expected-vs-predicted scatters,
residual and resolution fits, and test_metrics.json) for one run. Prediction
arrays (preds.npz) are not tracked; regenerate from a checkpoint with
scripts/test_eta_phi_pt_z0_charge.py if a bootstrap needs them.

Labels: CURRENT (AttnPool-era current method), RETIRED (superseded method),
REFERENCE (diagnostic/baseline kept for comparison).

Note on champion selection: none of these run directories carry a training
config or checkpoint hash, so the exact training commit cannot be recovered from
the results files alone. Metric schema identifies the current-method era, not
the precise code. Designating a single reported champion requires cross-checking
the W&B run configs. Until that is done, the current-method runs below are
presented as a comparable set, not ranked.

## Current-method runs (AttnPool-era, full target set: eta/phi/pT/z0/charge)

Numbers are from each run's test_metrics.json (charge AUC; eta, phi[rad], pT
relative, z0[mm] RMSE; z0 prior[mm]).

Full acceptance:

- ab/baseline           -- AUC 0.891, eta 0.021, phi 0.017, pT 0.052, z0 42.1 (prior 54.6)
- ab/candidate          -- AUC 0.895, eta 0.023, phi 0.016, pT 0.054, z0 47.3 (prior 54.6)
  (ab/candidate and ruche/Jul04_barrel are the same run under two paths.)

Barrel, pT > 10 GeV (seed-variation benchmark set):

- benchmark/baseline_barrel_pt10   -- AUC 0.903, eta 0.021, phi 0.016, pT 0.051, z0 39.3 (prior 54.1)
- benchmark/combo_rep1_barrel_pt10 -- AUC 0.893, eta 0.021, phi 0.016, pT 0.054, z0 39.2
- benchmark/combo_rep2_barrel_pt10 -- AUC 0.869, eta 0.022, phi 0.017, pT 0.090, z0 39.6
  Three successful seeds: mean AUC 0.888, sample sd 0.017.

- benchmark/pointing_rep1_barrel_pt10 -- AUC 0.521  (charge did not lift off)
- benchmark/pointing_rep2_barrel_pt10 -- AUC 0.688  (charge did not lift off)
  These two are step-starved charge failures, kept as documented negatives.

## z0 / charge development (RETIRED method, kept for provenance)

- ruche/Jul08_pointing_upgrade_full, ..._full_2 -- full-acceptance runs where
  charge failed to lift off (AUC 0.52, 0.62) and the z0 anchor is degraded
  (~1500-1870 mm). RETIRED negatives.
- ruche/Jun23_ConstChargeWeight -- fixed manual charge weight; AUC 0.816 but
  large phi/pT RMSE. RETIRED.
- ruche/Jun23_singlePhi, Jun23_singlePhi_z0Slice -- single-phi-head development.
- ruche/Jun19_first, Jun19_z0AnchorFix_homoscedastic, Jun19_z0Slice -- z0 anchor
  and homoscedastic-weighting development.
- ruche/Jun15_charge -- early charge attempt.
- charge/, resolution/ -- charge ROC and resolution-vs-pT summary figures.

## Earlier supervised progression (RETIRED)

- eta_phi_baseline, eta_phi_conv, eta_phi_conv_theta -- eta/phi only.
- eta_phi_angular_features -- + angular features.
- eta_phi_pt_conv, eta_phi_pt_conv_clusters -- + pT.
- concat_baseline, first-full -- early full-target baselines.
- ruche/Jun12_32etaLim, Jun15_3etaLim -- eta acceptance scans.
- ruche/geometricLoss_wrappedPhi, huber_rotated_skew -- loss-form experiments.

## Truth-free / DBSCAN (REFERENCE)

- eta_phi_pt_dbscan, eta_phi_pt_conv_dbscan_energy.pt -- truth-free runs.
  See docs/unsup_clustering_summary.md.
- cluster_purity -- DBSCAN cluster-purity analysis.
- ruche/.../ruche_eta_phi_pt_supervised_dbscan -- supervised-on-DBSCAN run.

## Diagnostics (REFERENCE)

- diagnostics/ -- energy-starved electron diagnostics.
