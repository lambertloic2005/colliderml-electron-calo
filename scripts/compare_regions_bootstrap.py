#!/usr/bin/env python3
"""Paired bootstrap of two preds.npz, computed separately in |eta| bands.
Matching, residuals and metrics are imported unchanged from
compare_preds_bootstrap.py; only the matched population is partitioned."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_preds_bootstrap import _load_raw, _keys, _residuals, _metrics


def _match(a, b):
    keys_a, keys_b = _keys(a), _keys(b)
    index_a = {}
    for i, k in enumerate(keys_a):
        index_a.setdefault(k, i)
    ia, ib = [], []
    for j, k in enumerate(keys_b):
        i = index_a.get(k)
        if i is not None:
            ia.append(i); ib.append(j)
    return np.asarray(ia, dtype=int), np.asarray(ib, dtype=int)


def main():
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    if len(args) < 2:
        sys.exit(__doc__)
    n_boot = int(args[2]) if len(args) > 2 else 2000
    edges = [0.0, 1.5, 3.0]
    for x in sys.argv[1:]:
        if x.startswith("--edges"):
            edges = [float(v) for v in x.split("=", 1)[1].split(",")]

    a, b = _load_raw(Path(args[0])), _load_raw(Path(args[1]))
    ia, ib = _match(a, b)
    print(f"A: n={len(_keys(a))}   B: n={len(_keys(b))}   matched={len(ia)}")
    if len(ia) == 0:
        sys.exit("no matched events")

    ra, rb = _residuals(a, ia), _residuals(b, ib)
    aeta = np.abs(a["truth_eta"][ia])
    rng = np.random.default_rng(0)

    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = np.where((aeta >= lo) & (aeta < hi))[0]
        n = len(sel)
        print("\n" + "=" * 74)
        print(f"|eta| in [{lo}, {hi})   matched n = {n}")
        print("=" * 74)
        if n < 200:
            print("  too few matched events for a stable bootstrap; skipping")
            continue
        base, cand = _metrics(ra, sel), _metrics(rb, sel)
        keys = sorted(set(base) & set(cand))
        deltas = {k: np.empty(n_boot) for k in keys}
        for t in range(n_boot):
            r = sel[rng.integers(0, n, n)]
            ma, mb = _metrics(ra, r), _metrics(rb, r)
            for k in keys:
                deltas[k][t] = mb[k] - ma[k]
        print(f"{'metric':>16s} {'A (base)':>10s} {'B (cand)':>10s} "
              f"{'delta':>10s} {'95% CI of delta':>24s}  signif")
        for k in keys:
            lo_ci, hi_ci = np.percentile(deltas[k], [2.5, 97.5])
            d = cand[k] - base[k]
            sig = "YES" if (lo_ci > 0) == (hi_ci > 0) else "no"
            print(f"{k:>16s} {base[k]:10.4f} {cand[k]:10.4f} "
                  f"{d:+10.4f} [{lo_ci:+9.4f}, {hi_ci:+9.4f}]  {sig}")

    print("\nsigma-type metrics improve when delta < 0; AUC/acc when delta > 0.")
    print("Matched counts are an intersection, NOT an efficiency.")


if __name__ == "__main__":
    main()
