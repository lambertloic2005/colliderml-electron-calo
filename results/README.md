# Results index

Each subdirectory holds evaluation outputs (expected-vs-predicted scatters,
residual and resolution fits, and test_metrics.json) for one run. Labels:
CHAMPION (current best, supervised), CURRENT (current method, not the single
champion), RETIRED (superseded method), REFERENCE (diagnostic/baseline kept for
comparison).

Provenance note: prediction arrays (preds.npz) are not tracked; regenerate from
the checkpoint with scripts/test_eta_phi_pt_z0_charge.py if a bootstrap needs
them.

## Current / champion

- ruche/Jul08_pointing_upgrade_full, ..._full_2 -- pointing-upgrade full runs.
  CURRENT-era.
- ab/candidate, ab/baseline -- A/B pair used for paired-bootstrap comparison.
- benchmark/*_barrel_pt10 -- barrel, pT>10 GeV benchmark set (baseline,
  combo_rep1/2, pointing_rep1/2). REFERENCE for seed spread.
  << CONFIRM which single directory is the reported CHAMPION and label it here >>

## z0 and charge development (RETIRED method, kept for provenance)

- ruche/Jun19_first, Jun19_z0AnchorFix_homoscedastic, Jun19_z0Slice -- z0 anchor
  and homoscedastic-weighting development.
- ruche/Jun23_singlePhi, Jun23_singlePhi_z0Slice, Jun23_ConstChargeWeight --
  single-phi-head and manual-charge-weight development.
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
