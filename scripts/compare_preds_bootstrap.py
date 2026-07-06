"""Paired bootstrap A/B comparison of two runs on the IDENTICAL test set.

Both preds.npz files must come from the same test population (same split,
same eta cuts, same pT floor). The script matches events between the two
files by their truth (eta, phi, pT) values, so it also works if orderings
differ. If the populations differ, it prints a forensic report showing
exactly how (counts, cut violations, unmatched events) instead of numbers.

Usage:
    python scripts/compare_preds_bootstrap.py \
        results/ab/baseline/preds.npz results/ab/candidate/preds.npz \
        [n_boot] [--align]

--align: proceed on the INTERSECTION of matched events if a small fraction
(<1%) fails to match (e.g. a single event sitting at a float boundary of
the pT floor). Refuses if more than 1% is unmatched -- that indicates a
wrong population, not a boundary effect, and must be fixed at the scoring
step, never papered over here.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from colliderml_electron.resolution import gaussian_resolution  # noqa: E402

EXPECTED_MAX_ABS_ETA = 1.7   # candidate barrel acceptance
EXPECTED_MIN_PT = 10.0       # candidate pT floor [GeV]
ALIGN_MAX_DROP_FRAC = 0.01


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


def _load_raw(path: Path) -> dict:
    f = np.load(path)
    d = {k: f[k] for k in f.files}
    missing = [k for k in ("truth_z0", "pred_z0", "charge", "charge_logit")
               if k not in d]
    if missing:
        print(f"[warning] {path} is missing {missing} -- it was written by "
              f"the OLD test script (before the extended np.savez). Regenerate "
              f"it with the updated script; z0/charge metrics will be skipped.")
    return d


def _summary(name: str, d: dict) -> None:
    pt, eta = d["truth_pt"], d["truth_eta"]
    n = len(pt)
    n_lowpt = int(np.sum(pt < EXPECTED_MIN_PT))
    n_higheta = int(np.sum(np.abs(eta) > EXPECTED_MAX_ABS_ETA))
    print(f"  {name}: n={n}")
    print(f"    truth pT  [GeV]: min={pt.min():8.3f}  max={pt.max():8.3f}  "
          f"events below {EXPECTED_MIN_PT} GeV floor: {n_lowpt}")
    print(f"    |truth eta|    : min={np.abs(eta).min():8.4f}  "
          f"max={np.abs(eta).max():8.4f}  "
          f"events beyond |eta|={EXPECTED_MAX_ABS_ETA}: {n_higheta}")
    if n_lowpt:
        print(f"    -> the pT floor was NOT applied when this file was scored")
    if n_higheta:
        print(f"    -> the eta cut was NOT applied when this file was scored")


def _keys(d: dict) -> list:
    # truth values decode identically on both branches (same parquet, same
    # stats, same float32 targets), so exact-value matching is safe; rounding
    # only guards against last-bit noise.
    return list(zip(np.round(d["truth_eta"], 6),
                    np.round(d["truth_phi"], 6),
                    np.round(d["truth_pt"], 4)))


def _residuals(d: dict, idx: np.ndarray) -> dict:
    out = {
        "eta_res": (d["pred_eta"] - d["truth_eta"])[idx],
        "phi_res": _wrap(d["pred_phi"] - d["truth_phi"])[idx],
        "pt_res": ((d["pred_pt"] - d["truth_pt"]) / d["truth_pt"])[idx],
    }
    if "pred_z0" in d and "truth_z0" in d:
        out["z0_res"] = (d["pred_z0"] - d["truth_z0"])[idx]
    if "charge_logit" in d and "charge" in d:
        out["charge_logit"] = d["charge_logit"][idx]
        out["charge"] = d["charge"][idx]
    return out


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


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--align"]
    align = "--align" in sys.argv[1:]
    if len(args) < 2:
        sys.exit(__doc__)
    path_a, path_b = Path(args[0]), Path(args[1])
    n_boot = int(args[2]) if len(args) > 2 else 2000

    a, b = _load_raw(path_a), _load_raw(path_b)

    # ---- event matching by truth values ----
    keys_a, keys_b = _keys(a), _keys(b)
    index_a = {}
    dup_a = 0
    for i, k in enumerate(keys_a):
        if k in index_a:
            dup_a += 1
        else:
            index_a[k] = i
    if dup_a:
        print(f"[warning] {dup_a} duplicate truth keys in A (kept first each)")

    ia, ib = [], []
    for j, k in enumerate(keys_b):
        i = index_a.pop(k, None)
        if i is not None:
            ia.append(i)
            ib.append(j)
    unmatched_a = len(keys_a) - dup_a - len(ia)
    unmatched_b = len(keys_b) - len(ib)

    if unmatched_a or unmatched_b:
        print("=" * 70)
        print("POPULATION MISMATCH -- forensic report")
        print("=" * 70)
        _summary(f"A (baseline)  {path_a}", a)
        _summary(f"B (candidate) {path_b}", b)
        print(f"\n  matched events: {len(ia)}")
        print(f"  in A only: {unmatched_a}    in B only: {unmatched_b}")

        def _show_unmatched(name, d, matched_idx, n_show=5):
            mask = np.ones(len(d["truth_pt"]), dtype=bool)
            mask[np.asarray(matched_idx, dtype=int)] = False
            um = np.where(mask)[0]
            if um.size:
                print(f"  first {min(n_show, um.size)} events only in {name} "
                      f"(truth pT [GeV], truth eta):")
                for i in um[:n_show]:
                    print(f"    pT={d['truth_pt'][i]:8.3f}  "
                          f"eta={d['truth_eta'][i]:+7.4f}")

        _show_unmatched("A", a, ia)
        _show_unmatched("B", b, ib)

        drop_frac = max(
            unmatched_a / max(len(keys_a), 1),
            unmatched_b / max(len(keys_b), 1),
        )
        if not align:
            sys.exit(
                "\nRefusing to compare. Diagnose with the report above: "
                "events below the pT floor or beyond the eta cut mean that "
                "file was scored without the MIN_PT_EVAL / MAX_ABS_ETA_EVAL "
                "overrides (or with the old test script). Re-score that file. "
                "Only if the mismatch is a handful of boundary events, re-run "
                "with --align."
            )
        if drop_frac > ALIGN_MAX_DROP_FRAC:
            sys.exit(
                f"\n--align refused: {drop_frac:.1%} of events unmatched "
                f"(limit {ALIGN_MAX_DROP_FRAC:.0%}). This is a wrong "
                "population, not a boundary effect. Fix the scoring step."
            )
        print(f"\n[--align] proceeding on the {len(ia)} matched events "
              f"(dropped {unmatched_a} from A, {unmatched_b} from B)")

    ia = np.asarray(ia, dtype=int)
    ib = np.asarray(ib, dtype=int)
    ra = _residuals(a, ia)
    rb = _residuals(b, ib)

    n = len(ia)
    keys = sorted(set(_metrics(ra, np.arange(n))) & set(_metrics(rb, np.arange(n))))
    print(f"\npaired bootstrap on n={n} events, {n_boot} resamples\n")

    full_a = _metrics(ra, np.arange(n))
    full_b = _metrics(rb, np.arange(n))

    rng = np.random.default_rng(0)
    deltas = {k: np.empty(n_boot) for k in keys}
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        ma, mb = _metrics(ra, idx), _metrics(rb, idx)
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