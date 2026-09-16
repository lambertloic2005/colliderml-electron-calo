# Supervised electron reconstruction
## The method

Each prompt electron is reconstructed from its truth-associated calorimeter cells
into five quantities: eta, phi, log(pT), z0, and charge sign. The network is
AttnPoolCaloRegressor, a transformer cell encoder followed by learned-query
cross-attention pooling, trained at model_dim 128, 3 layers, 4 heads, with the
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
reaches charge AUC around 0.89 in the barrel. z0 is calorimeter-limited
here: the pointing anchor on its own is far worse than the prior, so recovering
to ~39 mm barrel is close to the ceiling pointing-based z0 can reach.
