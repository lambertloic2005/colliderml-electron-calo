"""Paired bootstrap A/B comparison of two runs on the IDENTICAL test set.

Both preds.npz files must come from the same test population (same split,
same region cuts, same pT floor) so that event i in file A is event i in
file B. The script verifies this by matching the truth arrays, then
bootstrap-resamples EVENTS (paired), recomputing each metric for both runs
on the same resample. The delta distribution gives a confidence interval
that is immune to shared event-selection noise.

Usage:
    python scripts/compare_preds_bootstrap.py \
        results/<baseline_run>/preds.npz results/<candidate_run>/preds.npz \
        [n_boot]
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from colliderml_electron.resolution import gaussian_resolution  # noqa: E402


def _auc(scores: np.ndarray, pos: np.ndarray) -> float:
    pos = np.asarray(pos, dtype=bool)
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _wrap(x: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(x), np.cos(x))


def _metrics(d: dict, idx: np.ndarray) -> dict:
    out = {
        "eta_sigma": gaussian_resolution(d["eta_res"][idx]).sigma,
        "phi_sigma_rad": gaussian_resolution(d["phi_res"][idx], wrap=True).sigma,
        "pt_sigma_rel": gaussian_resolution(d["pt_res"][idx]).sigma,
    }
    if "z0_res" in d:
        out["z0_sigma_mm"] = gaussian_resolution(d["z0_res"][idx]).sigma
    if "charge_logit" in d:
        pos = d["charge"][idx] > 0
        out["charge_auc"] = _auc(d["charge_logit"][idx], pos)
        out["charge_acc"] = float(np.mean((d["charge_logit"][idx] > 0) == pos))
    return out


def _load(path: Path) -> dict:
    f = np.load(path)
    d = {
        "truth": np.stack([f["truth_eta"], f["truth_phi"], f["truth_pt"]], axis=1),
        "eta_res": f["pred_eta"] - f["truth_eta"],
        "phi_res": _wrap(f["pred_phi"] - f["truth_phi"]),
        "pt_res": (f["pred_pt"] - f["truth_pt"]) / f["truth_pt"],
    }
    if "pred_z0" in f.files:
        d["z0_res"] = f["pred_z0"] - f["truth_z0"]
    if "charge_logit" in f.files:
        d["charge_logit"] = f["charge_logit"]
        d["charge"] = f["charge"]
    return d


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    a = _load(Path(sys.argv[1]))
    b = _load(Path(sys.argv[2]))
    n_boot = int(sys.argv[3]) if len(sys.argv) > 3 else 2000

    if a["truth"].shape != b["truth"].shape or not np.allclose(
        a["truth"], b["truth"], atol=1e-5
    ):
        sys.exit(
            "Truth arrays differ: the two runs were NOT scored on the same "
            "test population. Re-score both with identical region and min_pt "
            "cuts (e.g. MIN_PT_EVAL=10) before comparing."
        )

    n = a["truth"].shape[0]
    keys = sorted(set(_metrics(a, np.arange(n))) & set(_metrics(b, np.arange(n))))
    print(f"paired bootstrap on n={n} events, {n_boot} resamples\n")

    full_a = _metrics(a, np.arange(n))
    full_b = _metrics(b, np.arange(n))

    rng = np.random.default_rng(0)
    deltas = {k: np.empty(n_boot) for k in keys}
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        ma, mb = _metrics(a, idx), _metrics(b, idx)
        for k in keys:
            deltas[k][i] = mb[k] - ma[k]

    print(f"{'metric':>14s} {'A (base)':>10s} {'B (cand)':>10s} {'delta':>9s} "
          f"{'95% CI of delta':>22s} {'signif':>7s}")
    for k in keys:
        dv = deltas[k]
        lo, hi = np.percentile(dv, [2.5, 97.5])
        sig = "YES" if (lo > 0 and hi > 0) or (lo < 0 and hi < 0) else "no"
        print(f"{k:>14s} {full_a[k]:10.4f} {full_b[k]:10.4f} "
              f"{full_b[k] - full_a[k]:+9.4f} [{lo:+9.4f}, {hi:+9.4f}] {sig:>7s}")

    print(
        "\n'signif' = 95% bootstrap CI of the paired delta excludes zero.\n"
        "For sigma-type metrics an improvement is delta < 0; for AUC/acc it\n"
        "is delta > 0."
    )


if __name__ == "__main__":
    main()