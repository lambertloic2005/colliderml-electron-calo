#!/usr/bin/env python3
"""Check whether the supervised / truth-free pairing loses electrons to key
rounding. Usage: python scripts/check_pairing.py SUP_preds.npz UNS_preds.npz"""
import sys
import numpy as np


def load(p):
    f = np.load(p)
    return {k: f[k] for k in ("truth_eta", "truth_phi", "truth_pt")}


def keys(d, de, dp):
    return list(zip(np.round(d["truth_eta"], de), np.round(d["truth_phi"], de),
                    np.round(d["truth_pt"], dp)))


def matched_mask(a, b, de, dp):
    ka = set(keys(a, de, dp))
    return np.array([k in ka for k in keys(b, de, dp)], dtype=bool)


def cut(d):
    m = (d["truth_pt"] >= 10.0) & (np.abs(d["truth_eta"]) <= 3.0)
    return {k: v[m] for k, v in d.items()}


a, b = load(sys.argv[1]), load(sys.argv[2])
for name, d in (("sup", a), ("uns", b)):
    print(f"{name}: n={len(d['truth_pt'])}  min pT={d['truth_pt'].min():.3f}  "
          f"n below 10 GeV={int((d['truth_pt'] < 10).sum())}  "
          f"max |eta|={np.abs(d['truth_eta']).max():.4f}")
for label, (x, y) in (("all", (a, b)), ("pT>=10, |eta|<=3", (cut(a), cut(b)))):
    print(f"\n{label}")
    barrel = np.abs(y["truth_eta"]) < 1.5
    for de, dp in ((6, 4), (4, 2)):
        m = matched_mask(x, y, de, dp)
        print(f"  eta/phi {de} dp, pT {dp} dp: matched = {int(m.sum())}  "
              f"(barrel {int((m & barrel).sum())}, endcap {int((m & ~barrel).sum())})")
