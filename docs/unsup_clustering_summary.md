# Calorimeter-only electron reconstruction with truth-free clustering

Status note, 31 July 2026. L. Lambert.
Dataset: ColliderML Release 1, `zee_pu200` (Z -> ee, PU200, OpenDataDetector).

## Summary

We have replaced the truth-seeded cell selection with a fully truth-free
clustering pipeline and measured what the network loses. The result splits
sharply by detector region.

In the barrel (|eta| < 1.5) the truth-free pipeline reconstructs 99.3 percent of
electrons above 10 GeV, and the reconstruction penalty is modest for every
quantity except pT: eta and z0 degrade by about 10 percent, phi by 18 percent,
charge AUC falls by 0.024, and relative pT resolution roughly doubles from 3.0
to 6.2 percent.

In the endcap (1.5 < |eta| < 3.0) the pipeline fails, recovering only 36.2
percent of electrons, and the survivors are also measured worse (charge AUC
-0.116, pT resolution 2.9 -> 7.7 percent).

The endcap failure has been diagnosed. It is not the cell energy threshold, not
shower fragmentation, and not lost bremsstrahlung; three separate measurements
exclude each. It is DBSCAN percolating into pileup: the electron's cluster is
bridged to a neighbouring pileup blob, the merged centroid moves outside the
matching cone, and the electron is lost. The remedy is a splitting step of the
kind used in ATLAS topological clustering, not a threshold adjustment.

## What was built

Truth-free pipeline (`cluster_pipeline.build_cluster_table`, branch
`retreat-control`): all calorimeter cells above `e_thresh_gev = 0.10` are
clustered with DBSCAN in (eta, phi) using `eps = 0.05`, `min_samples = 4`; each
cluster is matched to the nearest prompt truth electron within `dR_max`, and
truth is used only to assign the label after clustering, never to select cells.

The supervised reference (`pipeline.build_electron_row`) instead selects cells
by truth contribution and applies no energy threshold at all. That asymmetry
matters and is quantified below.

| | supervised (truth-seeded) | truth-free |
|---|---|---|
| electrons | 178,602 | 109,983 |
| events | 96,553 | 72,394 |
| file size | 11.1 GB | 524 MB |
| median cells per cluster | 1366 | 98 |
| train / val / test electrons | 124,921 / 26,849 / 26,832 | 77,270 / 16,288 / 16,425 |

Both builds cover shards 0-999 of the release.

## Dataset verification

- Shard pairing: the truth-free builder joins calorimeter to particle rows by
  `event_id` rather than by position. All 480 worker logs report zero events
  skipped, so the paired shards contain identical event sets.
- The supervised build still pairs positionally. This was checked empirically on
  the test split: the energy-weighted cell eta minus truth eta has median 0.016,
  with 2.7 percent beyond 0.1. That tail is uniform across event-id blocks
  (worst block 4.5 percent, +2.8 sigma of binomial expectation for 40 blocks)
  and is entirely a low-momentum population (median 3 cells, E_dep/E_truth =
  0.19, median pT 0.56 GeV). Mispairing would appear as whole blocks near 100
  percent. It does not. Positional pairing is validated.
- Split harmonisation: 72,388 of 72,394 truth-free events inherit their
  supervised split assignment verbatim; 6 new events receive fresh assignments.
  The two test sets are the same events, so all comparisons below are paired.

## Training

Identical architecture on both datasets: `AttnPoolCaloRegressor`, model_dim 128,
3 layers, 4 heads, feedforward 256, high_level_dim 41, batch size 96, top-128
cells by energy into the encoder. `src/colliderml_electron/dataset.py` is
byte-identical between the two branches, so no feature-set drift is possible.

Optimizer steps were matched rather than epochs, because the charge head is
known to require roughly 90,000 steps before lift-off and the truth-free
training set is smaller. Supervised: 200 epochs, 113,427 training electrons
after the |eta| <= 3 cut, 236,400 steps. Truth-free: 324 epochs, 77,196
electrons, 260,820 steps.

Correction to note: the 324-epoch figure was chosen against an assumed
supervised training size of 124,906, which was the pre-cut count. Against the
correct post-cut count the truth-free run received 10.3 percent *more* optimizer
steps than the supervised one, not an equal number. The bias therefore favours
the truth-free model, so the degradation reported below is if anything
conservative. The run also converged: validation loss, phi RMSE, pT resolution
and charge accuracy are flat to four decimal places over the final twenty
epochs, so the extra budget had no effect either way.

Training-set composition confirms the barrel comparison is not confounded by
data volume: 61,823 supervised barrel electrons against 58,752 truth-free, or
95.0 percent. The whole shortfall is in the endcap (51,604 against 18,444, 35.7
percent), which tracks the measured endcap efficiency of 36.2 percent.

## Acceptance

Denominator is the supervised (truth-seeded) electron, which is reconstructed by
construction. Truth pT >= 10 GeV, |eta| <= 3.

Overall 15,267 / 21,607 = 0.7066 +- 0.0031.

| region | efficiency |
|---|---|
| barrel, |eta| <= 1.5 | 0.9930 |
| endcap, |eta| > 1.5 | 0.3621 |

Efficiency is flat in pT: 0.703, 0.729, 0.711, 0.698, 0.714 across
[10,15), [15,20), [20,30), [30,50), [50,inf) GeV. Constant to within a percent
over a factor of five in energy.

## Resolution, paired by region

Paired bootstrap, 2000 resamples, on the 13,964 electrons reconstructed by both
pipelines. Significance means the 95 percent CI of the paired delta excludes
zero. For reference, the between-seed standard deviation on this dataset
generation is 0.0065 in barrel charge AUC.

Barrel, |eta| < 1.5, n = 10,825:

| metric | supervised | truth-free | delta | 95% CI |
|---|---|---|---|---|
| charge AUC | 0.9546 | 0.9304 | -0.0242 | [-0.0283, -0.0201] |
| charge accuracy | 0.8834 | 0.8463 | -0.0371 | [-0.0447, -0.0299] |
| pT sigma (rel) | 0.0303 | 0.0615 | +0.0312 | [+0.0294, +0.0325] |
| phi sigma [rad] | 0.0056 | 0.0066 | +0.0010 | [+0.0002, +0.0015] |
| eta sigma | 0.0194 | 0.0211 | +0.0017 | [+0.0012, +0.0023] |
| z0 sigma [mm] | 34.78 | 38.13 | +3.36 | [+2.48, +4.34] |

Endcap, 1.5 < |eta| < 3.0, n = 3,139:

| metric | supervised | truth-free | delta | 95% CI |
|---|---|---|---|---|
| charge AUC | 0.8821 | 0.7661 | -0.1160 | [-0.1315, -0.1006] |
| charge accuracy | 0.7971 | 0.6789 | -0.1182 | [-0.1370, -0.0997] |
| pT sigma (rel) | 0.0288 | 0.0771 | +0.0483 | [+0.0452, +0.0547] |
| phi sigma [rad] | 0.0093 | 0.0154 | +0.0061 | [+0.0050, +0.0073] |
| eta sigma | 0.0166 | 0.0187 | +0.0021 | [+0.0015, +0.0029] |
| z0 sigma [mm] | 54.96 | 54.74 | -0.22 | [-0.76, +0.35] |

The endcap z0 delta is consistent with zero, and both values sit at the beamspot
prior of about 57 mm. Neither model measures z0 in the endcap at all, so there
was no information available to lose. This serves as an internal consistency
check on the comparison.

pT is the dominant loss everywhere and eta the most robust, which follows from
the containment measurement: log_sum_et is the pT anchor and is an energy sum,
whereas the angular centroids are set by the high-energy core that survives
thresholding.

## Mechanism

Containment. Sub-100 MeV cells are not a negligible halo. Applying the 100 MeV
cut to supervised clusters leaves a median of 124 of 1352 cells carrying only
0.610 of the deposited energy (5th percentile 0.239). The truth-free pipeline at
the same threshold achieves a median E_cluster / E_truth of 0.576. Clustering
therefore costs about 6 percent relative; the threshold costs 39 percent. Not
knowing which cells belong to the electron is nearly free at this working point.
What costs is that one cannot go below 100 MeV without percolating.

Three candidate explanations for the endcap failure were tested and excluded.

1. Energy threshold. Excluded: efficiency is flat in pT across a factor of five
   in energy. A threshold effect would bite hardest at low pT.
2. Shower fragmentation. Excluded: the median distance to the 4th nearest
   neighbour among cells above threshold is 0.0010 in the barrel and 0.0021 at
   |eta| in [2.5, 3.0], both 25 to 50 times below `eps = 0.05`. The core-point
   fraction is 0.999 to 0.824. Endcap showers also carry four times *more* cells
   above threshold than barrel showers (362 against 85). Connectivity is never
   the limitation.
3. Loss of bremsstrahlung handedness. Charge is the sign of the azimuthal bend,
   so if the sub-threshold cells carried an asymmetric radiative fan, removing
   them would destroy the charge signal. Measured on supervised clusters, the
   charge-split bend asymmetry of the energy-weighted centroid is 0.02824 +-
   0.00261 rad using all cells and 0.02873 +- 0.00200 rad using only cells above
   100 MeV. Unchanged. The low-energy cells are isotropic and only dilute the
   centroid (spread 0.1425 -> 0.1089). Excluded.

What survives is merging. In the truth-free test set, matched clusters at
|eta| in [2.0, 2.5] have a median of 279 cells and E_cluster / E_truth = 0.759,
entirely healthy. At |eta| in [2.5, 3.0] the same quantities are 13,111 cells and
9.19: the electron contributes about 11 percent of a percolated pileup blob that
happened to land within the matching cone. The transition is abrupt rather than
gradual. Most merged clusters do not land within the cone at all, which is the
acceptance loss.

This is the same mechanism seen globally in the threshold scan, where lowering
`e_thresh_gev` to 0.02 produced a single connected component of 306,144 cells
and collapsed efficiency to 2.3 percent. In the endcap it occurs at 0.10 because
forward pileup density per unit (eta, phi) is much higher.

## Working point scan

500 events, shards 0-4, truth pT >= 10 GeV. `sigLog` is the half 16-84 interval
of log(E_cluster / E_truth), which is the irreducible noise the clustering
injects into the pT anchor. Counts are matched electrons; the truth denominator
is fixed across thresholds, so counts are a relative efficiency.

| threshold | barrel n | barrel sigLog | endcap n | endcap sigLog |
|---|---|---|---|---|
| 0.05 | 323 | 0.141 | 14 | 1.513 |
| 0.08 | 364 | 0.162 | 87 | 0.167 |
| 0.10 | 365 | 0.172 | 131 | 0.169 |
| 0.13 | 365 | 0.205 | 193 | 0.182 |
| 0.16 | 365 | 0.228 | 239 | 0.220 |
| 0.20 | 365 | 0.246 | 299 | 0.236 |
| 0.30 | 358 | 0.322 | 328 | 0.271 |

Barrel efficiency saturates at 0.08 and is flat to 0.20, so within that range
only containment matters, and 0.08 strictly dominates the 0.10 working point we
built at: same efficiency, better median containment (0.619 against 0.572) and
tighter spread. The gain is about 6 percent on sigLog, which does not justify a
rebuild on its own.

The endcap has no optimum. Efficiency rises monotonically to 0.30 while sigLog
also rises, a pure Pareto trade. The fraction of matched clusters carrying more
than twice the electron energy falls monotonically with threshold (28.6, 3.4,
3.1, 1.0, 0.4, 0.3, 0.0 percent), confirming that raising the threshold removes
percolated blobs and recovers genuine electrons.

## Caveats

- 14.9 percent of truth-free test electrons are absent from the supervised test
  set. These are electrons the supervised pipeline rejected because no
  truth-linked cells survived. Some are genuine recoveries, some are clusters
  matched to electrons that deposited essentially nothing. They are excluded
  from the paired comparison but they inflate the raw metrics in the truth-free
  test log, which is why only the paired numbers should be quoted.
- The endcap efficiency curve in the scan is an upper bound for the same reason.
- The scan holds `eps = 0.05` and `min_samples = 4` fixed while the threshold
  varies. Percolation depends on density and reach jointly, so these are not
  independent knobs and a two-dimensional scan would be more informative.
- All resolutions are quoted without a pT floor, consistent with previous runs
  in this project, which also had no training or evaluation floor.

## Proposed next steps

1. Implement local-maximum splitting in `cluster.py`, following ATLAS
   topological clustering: seed on cells at 4 sigma above noise, grow through
   2 sigma neighbours, then split clusters on local energy maxima. The absence
   of a splitting step is the single structural gap that every diagnostic above
   points at, and five shards are already downloaded for validation.
2. If splitting recovers endcap efficiency, rebuild once with the correct
   algorithm rather than twice with a threshold workaround. Adopt 0.08 in the
   barrel at that point.
3. Add an explicit |eta| <= 3 cut in `build_cluster_dataset.py`. Forward
   electrons currently leak into diagnostics and are poorly reconstructed
   (sigLog 0.47 to 0.66).
4. Zero-shot measurement, still outstanding: score the supervised checkpoint on
   truth-free clusters with the supervised normalisation. This separates the
   deployment penalty from the irreducible one. Expected to show pT low by about
   39 percent from the anchor shift, which is a calibration artifact rather than
   an information loss.

## Figures

- `fig1_resolution_by_region.png` -- phi residual, relative pT residual and
  charge ROC, barrel and endcap, supervised against truth-free, paired.
- `fig2_efficiency.png` -- cluster-matching efficiency against |eta| and pT.

## Reproducibility

Truth-free build: branch `retreat-control`, commit `d9a357d`.
Truth-free training: branch `attnpool-unsup-324ep`, commit `d9c4873`, Lyon job
55460288, dataset md5 `505393c1c8a39b0cafabbac10b632c2c`.
Supervised reference: branch `attnpool-200ep`, commit `770ba8a`, Lyon job
55426542.
