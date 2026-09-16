# Experiment log
## Data and infrastructure

- building-data-pipeline (May 22) -- splits, target normalization, dataset
  loader. ADOPTED, foundation of the current pipeline.
- atlas-processing (Jun 05) -- merged-parquet preprocessing on the atlas
  machine. ADOPTED.
- build-dbscan (Jun 02) -- first truth-free DBSCAN clustering. ADOPTED as the
  basis of the truth-free line.
- fetch_and_cluster / harmonize_splits / chunked atlas build (Jul) -- cluster
  dataset plumbing and split harmonization. ADOPTED.

## Model progression (supervised)

- building-first-model (May 26) -- first eta/phi learning. RETIRED (scripts now
  in scripts/legacy/).
- angular-features-v1 (May 26) -- angular per-cell features; right direction,
  precision still lacking. ADOPTED (features), scripts RETIRED.
- split-phi-output (May 28) -- 1D convolution aggregation, with/without theta.
  Conv ADOPTED for a period, later RETIRED in favour of attention pooling.
- improve-pt-supervised-dbscan (Jun 02) -- energy-weighted phi centroid vs truth
  analysis. ADOPTED (centroid anchor).
- phi-wrapped-loss-geometric-combine (Jun 12) -- wrapped phi loss + geometric-
  mean task combination + early stopping. Wrapped phi ADOPTED; geometric-mean
  combine later RETIRED in favour of homoscedastic weighting.
- fix-low-pT / fix-residual-fit (Jun 09-15) -- eta acceptance to 3.0, residual
  fit and ruche compatibility. ADOPTED.

## z0 and charge

- Add-z0-and-charge (Jun 22) -- z0 target and first charge attempt. ADOPTED.
- Jun19 z0 anchor fix (pointing anchor, beamspot z-scoring) -- fixed a pointing-
  anchor failure. ADOPTED; z0 now anchored on the pointing fit.
- region-split-experiment (Jun 30) -- single-head charge architecture, 41-dim
  features, barrel/endcap split, d0 notes. ADOPTED (single-head charge, 41-dim).
- pointing-upgrade (Jul 08) -- feature stability: variance floors, cluster-
  feature clipping, min-epochs floor. ADOPTED (code); the Jul08 runs from this
  branch failed on charge and are kept as negatives.
- Jun23 ConstChargeWeight -- fixed manual charge weight after homoscedastic
  weighting collapsed the BCE gradient. ADOPTED (charge uses a manual weight).

## Architecture and scaling

- combo-floor (Jul 23) -- combined-target floor experiments. MERGED.
- scale-v2 / attnpool-v2 (Jul 24) -- attention-pooling aggregation and scaling.
  ADOPTED (AttnPool).
- batch-step-fix / batch-step-fix-scaled (Jul 27) -- step-count accounting fix;
  established total optimizer steps as the governing quantity. ADOPTED.
- champion-headroom-200ep / champion-seeded (Jul 28-29) -- epoch headroom and
  seeded ensembles. ADOPTED (seeding protocol).
- attnpool-200ep / attnpool-champ (Jul 30) -- AttnPool champion at 200 epochs.
  ADOPTED as champion architecture.
- retreat-control (Jul 30) -- 41-feature set (K=6, phi_slope not exposed), no pT
  floor, durable best-checkpoint save. ADOPTED (current dataset/train config).

## Analysis and instrumentation

- attnpool-unsup-324ep (Jul 31) -- truth-free run at step-matched 324 epochs,
  cross-dataset bootstrap, region split, summary figures. ADOPTED; writeup in
  docs/unsup_clustering_summary.md.
- sigma-instrumentation (Aug 28) -- W&B logging of learned log_sigma, best-epoch
  loss_state_dict in checkpoints, uncertainty-vs-resolution check. ADOPTED,
  merged to main.
