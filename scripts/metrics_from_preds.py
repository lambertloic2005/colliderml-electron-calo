"""Recompute test metrics from a preds.npz, optionally with a truth-pT floor.

Usage: python scripts/metrics_from_preds.py PREDS.npz [PT_MIN_GEV]
With PT_MIN_GEV = 0 this must reproduce the run's test_metrics.json
(same residual definitions, estimator and region boundaries as the test script).
"""
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

from colliderml_electron.resolution import gaussian_resolution, wrap_angle

d = np.load(sys.argv[1])
pt_min = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0

tpt, teta = d["truth_pt"], d["truth_eta"]
res = {
    "eta": d["pred_eta"] - teta,
    "phi": wrap_angle(d["pred_phi"] - d["truth_phi"]),
    "pt_rel": (d["pred_pt"] - tpt) / tpt,
    "z0": d["pred_z0"] - d["truth_z0"],
}
q, logit = d["charge"], d["charge_logit"]
abs_eta = np.abs(teta)
sel = tpt >= pt_min
print(f"pT >= {pt_min} GeV: {int(sel.sum())} of {len(tpt)} electrons")

regions = (
    ("all", np.ones_like(sel)),
    ("barrel", abs_eta < 1.5),
    ("endcap", (abs_eta >= 1.5) & (abs_eta < 3.0)),
)
for label, mreg in regions:
    m = sel & mreg
    n = int(m.sum())
    if n < 50:
        print(f"{label}: n={n}, skipped")
        continue
    lines = [f"{label} n={n}"]
    for k in ("eta", "phi", "pt_rel", "z0"):
        f = gaussian_resolution(res[k][m], wrap=(k == "phi"))
        rmse = float(np.sqrt(np.mean(res[k][m] ** 2)))
        lines.append(f"{k:6s} sigma={f.sigma:.5g}  tail={f.tail_fraction:.2%}  rmse={rmse:.5g}")
    lines.append(f"z0 prior (std of truth z0) = {np.std(d['truth_z0'][m]):.2f} mm")
    y = (q[m] > 0).astype(int)
    acc = float(np.mean((logit[m] > 0) == (q[m] > 0)))
    lines.append(f"charge auc={roc_auc_score(y, logit[m]):.4f}  acc={acc:.4f}")
    print("\n  ".join(lines))
