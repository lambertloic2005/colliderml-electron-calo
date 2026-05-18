# colliderml-electron-calo

Exploring the calorimeter portion of the [CERN ColliderML Release 1](https://huggingface.co/datasets/CERN/ColliderML-Release-1) dataset for prompt electrons.

## Goals
1. **v1 — exploration**: load events, isolate prompt-electron calorimeter hits, and produce a gallery of plots to build intuition for the dataset (detector geometry, energy spectra, longitudinal/lateral shower profiles, η–φ lego, etc.).
2. **v2 — outlier rejection**: methods for separating outlier cells (pileup, secondaries, noise) from the central electromagnetic shower.

## Dataset notes
- Default channel: `zee_pu200` (Z→ee with HL-LHC pileup) — guaranteed prompt electrons plus a realistic outlier background.
- Cell energies in `calo_hits.total_energy` are **uncalibrated**. Apply the per-subsystem scaling factors from the release paper before quoting GeV.
- "Electron-associated" cells are identified by walking `contrib_particle_ids` per cell and matching against prompt electrons (`pdg_id == ±11`, `primary == True`, `vertex_primary == 1`) from the `particles` table.

## Quickstart
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Status
- [x] Repo scaffolding
- [ ] Data loader (`src/colliderml_electron/io.py`)
- [ ] Geometry helpers (`src/colliderml_electron/coords.py`)
- [ ] Plot gallery (`src/colliderml_electron/plots.py`)
- [ ] Outlier methods (`src/colliderml_electron/outliers.py`) — stub only

## License
MIT
