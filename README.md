# colliderml-electron-calo

Machine learning experiments on the calorimeter portion of the CERN ColliderML Release 1 dataset, with a focus on reconstructing prompt-electron kinematics from calorimeter showers.

The current development branch uses DBSCAN to build truth-free calorimeter clusters, then matches those clusters to truth prompt electrons only for supervised training labels. The model is then trained to predict electron direction and transverse momentum from the calorimeter cells.

## Project goal

The long-term goal is to test how much tracker-like information can be inferred from calorimeter showers alone.

Current targets:

- electron pseudorapidity, `eta`
- electron azimuthal angle, `phi`
- electron transverse momentum, `pT`, trained as `log(pT)`

## Current pipeline

The current DBSCAN-based workflow is:

1. Load ColliderML Release 1 particle and calorimeter-hit parquet files.
2. For each event, load all observable calorimeter cells.
3. Calibrate cell energies.
4. Cluster cells in `(eta, phi)` space using DBSCAN.
5. Drop DBSCAN outliers/noise cells.
6. Compute cluster-level observables such as energy-weighted centroid and total calibrated cluster energy.
7. Match clusters to prompt electrons using truth only after clustering.
8. Save one row per matched cluster to `data/clusters/clusters.parquet`.
9. Assign train/validation/test splits.
10. Compute target normalization statistics.
11. Train a neural network to predict `eta`, `phi`, and `log(pT)`.
12. Evaluate the model using residual plots and Gaussian resolution fits.

Important distinction:

- DBSCAN clustering is truth-free.
- Truth information is used only to label clusters for supervised training.

## Repository structure

```text
.
├── docs/                         # Notes and documentation
├── notebooks/                    # Jupyter notebooks for analysis
├── scripts/                      # Command-line scripts
├── src/colliderml_electron/      # Main Python package
├── data/                         # Local generated data; usually not committed
├── checkpoints/                  # Saved model weights
├── results/                      # Test plots and metrics
├── pyproject.toml
├── requirements.txt
└── README.md
