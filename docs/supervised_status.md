# Supervised electron reconstruction: where things stand

Status note, 09 September 2026. L. Lambert.
Dataset: ColliderML Release 1, `zee_pu200` (Z -> ee, PU200, OpenDataDetector).

This is the supervised sibling of the truth-free note
(`unsup_clustering_summary.md`). The short version: the method has converged, the
current runs are consistent, and two things are still open before there is a
headline number to quote — confirming which run is the champion, and finishing
the region-projection ablation.

## The method

Each prompt electron is reconstructed from its truth-associated calorimeter cells
into five quantities: eta, phi, log(pT), z0, and charge sign. The network is
AttnPoolCaloRegressor — a transformer cell encoder followed by learned-query
cross-attention pooling — trained at model_dim 128, 3 layers, 4 heads, with the
41-dimensional feature set and 128 cells per electron. eta, phi, pT, and z0 are
predicted as corrections to physics anchors; charge is a separate BCE logit. The
four regression tasks share a learned homoscedastic weighting, but charge sits
outside it on a fixed manual weight, because putting the BCE term under the
learned weight drives its gradient to zero. The README has the full architecture.

## What the current runs show

All numbers below come straight from the tracked `test_metrics.json` files.
Charge is ROC AUC; eta, phi (radians), pT (relative), and z0 (mm) are RMSE.

At full acceptance (the `ab/baseline` and `ab/candidate` runs), charge AUC lands
at 0.89 to 0.90, eta around 0.021 to 0.023, phi at 0.016 to 0.017 rad, relative
pT near 0.052 to 0.054, and z0 between 42 and 47 mm against a 54.6 mm beamspot
prior.

Restricting to the barrel above 10 GeV, the three seeds that trained cleanly
(`baseline`, `combo_rep1`, `combo_rep2` in `benchmark/`) give a mean charge AUC
of 0.888 with a sample spread of 0.017, ranging 0.869 to 0.903. Their z0 tightens
to about 39 mm against a 54.1 mm prior, and eta and phi are essentially unchanged
from the full-acceptance numbers. pT is stable except for combo_rep2, whose
relative RMSE runs high at 0.09.

The takeaway is that the model beats the beamspot prior on z0 everywhere and
reaches charge AUC around 0.89 in the barrel. z0 is genuinely calorimeter-limited
here: the pointing anchor on its own is far worse than the prior, so recovering
to ~39 mm barrel is close to the ceiling pointing-based z0 can reach.

## A failure mode worth keeping in view

Two of the barrel benchmark seeds (`pointing_rep1`, `pointing_rep2`) and both
full-acceptance `pointing_upgrade` runs have charge AUC sitting between 0.5 and
0.69 — the charge head simply never lifted off. This is the step-starvation
pattern, not overfitting. The charge head needs something like 90,000+ optimizer
steps before it climbs off chance, and phi resolution is the tell: every failed
run also has phi degraded to ~0.20 rad, against ~0.016 rad in the runs that
worked. These are kept as negatives rather than deleted, because they are the
cleanest evidence for why step count governs the whole run design.

## The seed-variance number needs reconciling

The three clean barrel seeds give a charge-AUC spread of about 0.017. I have been
carrying a tighter working figure of 0.0065 for v2 barrel charge AUC, but the
tracked runs here do not reproduce it — that number probably comes from a
different comparison set, most likely something logged in W&B rather than saved
here. Before either figure goes in front of David and Tuan it needs to be pinned
down against W&B. Whichever value holds, single-run charge comparisons get judged
against it through the pre-registered paired bootstrap
(`compare_preds_bootstrap.py`, 2000 resamples, CI excluding zero).

## Why there is no champion named yet

None of the result directories carry a training config or a checkpoint hash, so
I cannot recover the exact training commit for any given run from the results
files alone. The metric schema tells me a run belongs to the current AttnPool era,
but not precisely which code produced it. Picking one run as the reported champion
means going back to the W&B run configs and confirming its architecture, feature
dimension, and dataset. That is the main thing standing between the current state
and a single quotable result.

## Still open

- Confirm champion provenance in W&B, then fix the headline run and its exact
  numbers here and in `results/README.md`.
- Reconcile the seed-variance figure: 0.017 from the tracked runs versus the
  0.0065 working value.
- Finish the region-projection ablation. A `PER_REGION_PROJ=1` run exists, but
  the matching `PER_REGION_PROJ=0` twin on the same parquet and seed has not been
  run, so there is no verdict to report.
- Run the zero-shot measurement (a supervised checkpoint scored on truth-free
  data with supervised stats) — never executed.
- Pin down why the supervised pipeline drops certain electrons. The suspicion is
  zero truth-linked cells for soft or geometrically gapped electrons, but it is
  unconfirmed.
