# Baseline A0

Current saved checkpoint for eta/phi/pT model.

Dataset:
- data/electrons/electrons_dbscan.parquet

Stats:
- data/electrons/target_stats.json

Model:
- eta/phi/pT angular features model
- DBSCAN-cleaned electron cells
- conv architecture if using eta_phi_pt_conv_dbscan_energy.pt

Purpose:
This is the current known-good baseline checkpoint before larger-scale training on the remote machine.
