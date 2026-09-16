# Results index

Each subdirectory holds evaluation outputs (expected-vs-predicted scatters,
residual and resolution fits, and test_metrics.json) for one run. Prediction
arrays (preds.npz) are not tracked; regenerate from a checkpoint with
scripts/test_eta_phi_pt_z0_charge.py if a bootstrap needs them. Those untracked
in 16030c0 are recoverable from history, e.g.
git show 16030c0^:results/benchmark/baseline_barrel_pt10/preds.npz > preds.npz

Labels: CURRENT (AttnPool-era current method), RETIRED (superseded method),
REFERENCE (diagnostic/baseline kept for comparison).

Note on the champion: none of the run directories below is an AttnPool run.
The supervised AttnPool reference (attnpool-200ep, 770ba8a, Lyon job 55426542)
has no tracked test_metrics.json; its paired numbers are in
docs/unsup_clustering_summary.md and summarized in docs/supervised_status.md.

## Tracked July runs (pre-AttnPool, full target set)

Numbers are from each run's test_metrics.json (charge AUC; eta, phi[rad], pT
relative, z0[mm] RMSE; z0 prior[mm]).

|eta| <= 1.7, pT > 10 GeV (endcap here = 1.5-1.7 only):

- ab/baseline           -- AUC 0.891, eta 0.021, phi 0.016, pT 0.052, z0 42.1 (prior 54.6)
- ab/candidate          -- AUC 0.895, eta 0.023, phi 0.016, pT 0.054, z0 47.3 (prior 54.6)
  (ab/candidate and ruche/Jul04_barrel are the same run under two paths.)

Barrel, pT > 10 GeV (seed-variation benchmark set):

- benchmark/baseline_barrel_pt10   -- AUC 0.903, eta 0.021, phi 0.016, pT 0.051, z0 39.3 (prior 54.1)
- benchmark/combo_rep1_barrel_pt10 -- AUC 0.893, eta 0.021, phi 0.016, pT 0.054, z0 39.2
- benchmark/combo_rep2_barrel_pt10 -- AUC 0.869, eta 0.022, phi 0.017, pT 0.090, z0 39.6
  Spread across these three runs: 0.869-0.903 (not a seed variance, see status note).

- benchmark/pointing_rep1_barrel_pt10 -- AUC 0.521  (charge did not lift off)
- benchmark/pointing_rep2_barrel_pt10 -- AUC 0.688  (charge did not lift off)
  Charge failures kept as documented negatives; cause not verified (see status note).

## z0 / charge development (RETIRED method, kept for provenance)

- ruche/Jul08_pointing_upgrade_full, ..._full_2 -- |eta| <= 3, no pT cut; charge
  failed to lift off (AUC 0.52, 0.62). Anchor-only z0 RMSE is 1500-1870 mm on
  this wider population (not directly comparable to the pT > 10 GeV runs).
  RETIRED negatives.
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

- eta_phi_pt_dbscan, eta_phi_pt_conv_dbscan_energy -- truth-free runs.
  See docs/unsup_clustering_summary.md.
- cluster_purity -- DBSCAN cluster-purity analysis.
- ruche/.../ruche_eta_phi_pt_supervised_dbscan -- supervised-on-DBSCAN run.

## Diagnostics (REFERENCE)

- diagnostics/ -- energy-starved electron diagnostics.
