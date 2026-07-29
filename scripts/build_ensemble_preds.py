#!/usr/bin/env python3
"""Average several single-seed preds.npz into one ensemble preds.npz.

Members must come from the SAME test population (same split, same REGION,
same pT floor). Events are matched across members by their truth values --
never by row position -- so a differing loader order cannot silently
misalign the average.

What gets averaged:
  * every array whose name starts with "pred_"   -> regression outputs
  * "charge_logit"                               -> averaged in LOGIT space

Logit-space averaging is the standard choice for a classifier ensemble: it
corresponds to a geometric mean of the odds, is symmetric in the two
classes, and is monotone, so the ROC/AUC is well defined. Averaging
probabilities instead would give a slightly different (arithmetic-mean)
ensemble; either is defensible, but mixing the two across analyses is not,
so this script commits to logits and records that choice in the output.

Truth arrays are copied from the first member after verifying they agree
across members (they must, since the population is identical).

Usage:
    python build_ensemble_preds.py OUT.npz MEMBER1.npz MEMBER2.npz [MEMBER3.npz ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

MATCH_KEYS = ("truth_eta", "truth_phi")   # extended with a pT-like key if present
ROUND_DP = 6


def _pt_key(d: dict) -> str | None:
    for k in ("truth_pt", "truth_log_pt", "truth_logpt"):
        if k in d:
            return k
    return None


def _keyset(d: dict, keys: tuple[str, ...]) -> np.ndarray:
    cols = [np.round(np.asarray(d[k], dtype=np.float64), ROUND_DP) for k in keys]
    return np.array(list(zip(*cols)), dtype=object)


def main() -> None:
    if len(sys.argv) < 4:
        print(__doc__)
        raise SystemExit(2)

    out_path = Path(sys.argv[1])
    member_paths = [Path(p) for p in sys.argv[2:]]
    members = [dict(np.load(p)) for p in member_paths]

    for p, d in zip(member_paths, members):
        print(f"{p.name}: {len(d[MATCH_KEYS[0]])} events, {len(d)} arrays")

    pt = _pt_key(members[0])
    keys = MATCH_KEYS + ((pt,) if pt else ())
    print(f"matching on: {keys}")

    # index each member by its truth tuple
    indexes = []
    for d in members:
        km = _keyset(d, keys)
        idx = {}
        for i, k in enumerate(km):
            idx.setdefault(tuple(k), i)
        indexes.append(idx)

    common = set(indexes[0])
    for idx in indexes[1:]:
        common &= set(idx)
    common = sorted(common)
    n_min = min(len(d[MATCH_KEYS[0]]) for d in members)
    drop = 1.0 - len(common) / n_min
    print(f"matched {len(common)} / {n_min} events ({drop:.3%} dropped)")
    if drop > 0.01:
        raise SystemExit(
            "ERROR: >1% of events failed to match across members. That is a "
            "population mismatch, not a float boundary effect -- fix it at the "
            "scoring step rather than averaging over it."
        )

    order = [np.array([idx[k] for k in common]) for idx in indexes]

    avg_keys = sorted(
        k for k in members[0]
        if k.startswith("pred_") or k == "charge_logit"
    )
    if not avg_keys:
        raise SystemExit("ERROR: no pred_* or charge_logit arrays found.")
    print(f"averaging: {avg_keys}")

    out: dict[str, np.ndarray] = {}
    for k in avg_keys:
        stack = np.stack([
            np.asarray(d[k], dtype=np.float64)[o] for d, o in zip(members, order)
        ])
        out[k] = stack.mean(axis=0)

    # copy truth / passthrough arrays, verifying agreement
    for k in members[0]:
        if k in out:
            continue
        ref = np.asarray(members[0][k])[order[0]]
        for d, o in zip(members[1:], order[1:]):
            other = np.asarray(d[k])[o]
            if ref.dtype.kind in "fc":
                same = np.allclose(ref, other, rtol=0, atol=1e-6, equal_nan=True)
            else:
                same = np.array_equal(ref, other)
            if not same:
                raise SystemExit(
                    f"ERROR: non-averaged array '{k}' differs between members "
                    f"after matching. The populations are not identical."
                )
        out[k] = ref

    out["ensemble_n_members"] = np.array(len(members))
    out["ensemble_charge_combine"] = np.array("logit_mean")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **out)
    print(f"\nwrote {out_path}  ({len(common)} events, {len(members)} members)")


if __name__ == "__main__":
    main()