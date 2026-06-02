# colliderml-electron-calo

Exploration and early machine-learning experiments on the calorimeter portion of the [CERN ColliderML Release 1](https://huggingface.co/datasets/CERN/ColliderML-Release-1) dataset, focused on prompt electrons.

The project currently studies how electron showers appear in the calorimeter and builds tools for loading events, selecting prompt-electron-associated cells, plotting shower structure, applying simple outlier rejection, and preparing per-electron training tables.

## Goals

The current `main` branch is focused on two stages:

1. **Exploration**
   - Load ColliderML particle and calorimeter-hit data.
   - Identify prompt electrons.
   - Select calorimeter cells associated with each prompt electron and its descendants.
   - Visualize detector geometry, shower shapes, energy spectra, and η–φ structure.

2. **Outlier rejection and early dataset building**
   - Remove cells far from the truth electron direction using a fixed ΔR cone.
   - Experiment with shower-cleaning methods for reducing pileup, secondary, and noise contamination.
   - Build a per-electron parquet table that can later be used for model training.

## Dataset notes

Default dataset configuration:

```text
channel = zee
pileup  = pu200
