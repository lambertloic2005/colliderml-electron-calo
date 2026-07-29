#!/usr/bin/env python3
"""Quick barrel/endcap charge AUC + phi_sigma from one or more preds.npz.

Reports the same quantities as test_metrics.json so an ensemble can be put
side by side with its members without re-running the full test script.
Region split at |eta| = 1.5 (ECal barrel/endcap boundary).

Usage:
    python score_ensemble.py A.npz [B.npz ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def auc(scores: np.ndarray, pos: np.ndarray) -> float:
    pos = np.asarray(pos, dtype=bool)
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def wrap(x):
    return np.arctan2(np.sin(x), np.cos(x))


def sigma(resid: np.ndarray) -> float:
    """Gaussian-core width: iterated 2-sigma-clipped std (tail insensitive)."""
    r = np.asarray(resid, dtype=np.float64)
    s = np.std(r)
    for _ in range(5):
        keep = np.abs(r - np.median(r)) < 2.0 * s
        if keep.sum() < 10:
            break
        s_new = np.std(r[keep])
        if abs(s_new - s) < 1e-12:
            break
        s = s_new
    return float(s)


for p in sys.argv[1:]:
    d = dict(np.load(Path(p)))
    eta = np.asarray(d["truth_eta"], dtype=np.float64)
    barrel = np.abs(eta) <= 1.5
    endcap = ~barrel
    q = np.asarray(d["charge"], dtype=np.float64)
    logit = np.asarray(d["charge_logit"], dtype=np.float64)
    pos = q > 0

    line = [f"{Path(p).name:>42}"]
    for name, m in (("barrel", barrel), ("endcap", endcap)):
        line.append(f"{name}AUC={auc(logit[m], pos[m]):.4f}")
    line.append(f"allAUC={auc(logit, pos):.4f}")

    if "pred_phi" in d and "truth_phi" in d:
        dphi = wrap(np.asarray(d["pred_phi"]) - np.asarray(d["truth_phi"]))
        line.append(f"barrel_phi_sigma={sigma(dphi[barrel]):.5f}")
    print("  ".join(line))