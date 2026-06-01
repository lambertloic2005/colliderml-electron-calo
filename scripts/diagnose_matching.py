"""Diagnose why prompt electrons fail to match a cluster.

Reports, per truth electron:
  - does the shower survive thresholding? (>=3 cells within dR<0.05 of truth)
  - nearest cluster by energy-weighted CENTROID and by PEAK (max-energy) cell
and breaks survival + peak-match down by |truth eta| and truth pT, so you can
tell tuning loss apart from acceptance loss (forward / low-pT electrons that no
central clustering can recover).
"""
from __future__ import annotations
import argparse, glob, os
import numpy as np
import polars as pl

from colliderml_electron.io import all_event_cells, prompt_electrons
from colliderml_electron.coords import xyz_to_eta_phi, momentum_to_eta_phi
from colliderml_electron.calibration import calibrate
from colliderml_electron.cluster import cluster_event_cells, cluster_centroid, iter_clusters
from colliderml_electron.pipeline import _shard_index


def _dR(e0, p0, e1, p1):
    dphi = (np.asarray(p1) - p0 + np.pi) % (2 * np.pi) - np.pi
    return np.hypot(np.asarray(e1) - e0, dphi)


def _rate_by_bin(d, value_col, edges, label_col):
    """Print mean of `value_col` per bin of `label_col` defined by `edges`."""
    lab = np.asarray(d[label_col].to_list(), float)
    val = np.asarray(d[value_col].to_list(), float)
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (lab >= lo) & (lab < hi)
        n = int(sel.sum())
        rate = float(val[sel].mean()) if n else float("nan")
        print(f"    {lo:>4.1f}-{hi:<4.1f} : {rate:6.1%}  (n={n})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", default="zee")
    ap.add_argument("--pileup", default="pu200")
    ap.add_argument("--n-events", type=int, default=30)
    ap.add_argument("--eps", type=float, default=0.03)
    ap.add_argument("--min-samples", type=int, default=3)
    ap.add_argument("--e-thresh-gev", type=float, default=0.05)
    ap.add_argument("--energy-weighted", action="store_true")
    ap.add_argument("--dR-match", type=float, default=0.1)
    a = ap.parse_args()

    home = os.path.expanduser("~")
    base = f"{home}/.cache/colliderml/CERN__ColliderML-Release-1"
    p_pat = f"{base}/{a.channel}_{a.pileup}_particles/data/{a.channel}_{a.pileup}_particles/train-*.parquet"
    c_pat = f"{base}/{a.channel}_{a.pileup}_calo_hits/data/{a.channel}_{a.pileup}_calo_hits/train-*.parquet"
    p_by = {_shard_index(p): p for p in glob.glob(p_pat)}
    c_by = {_shard_index(p): p for p in glob.glob(c_pat)}
    common = sorted(set(p_by) & set(c_by))

    rec, done = [], 0
    for idx in common:
        if done >= a.n_events:
            break
        p_df = pl.read_parquet(p_by[idx]); c_df = pl.read_parquet(c_by[idx])
        for i in range(p_df.height):
            if done >= a.n_events:
                break
            p_row = p_df.row(i, named=True); c_row = c_df.row(i, named=True)
            cells = all_event_cells(c_row)
            e_cal = calibrate(cells["e_total"], cells["x"], cells["y"], cells["z"])
            eta_all, phi_all = xyz_to_eta_phi(cells["x"], cells["y"], cells["z"])

            labels = cluster_event_cells(cells, eps=a.eps, min_samples=a.min_samples,
                                         e_thresh_gev=a.e_thresh_gev,
                                         energy_weighted=a.energy_weighted)
            clusters = list(iter_clusters(labels))
            cents, peaks = [], []
            for _, m in clusters:
                cents.append(cluster_centroid(eta_all[m], phi_all[m], e_cal[m]))
                k = m[int(np.argmax(e_cal[m]))]
                peaks.append((float(eta_all[k]), float(phi_all[k])))

            seen = set()
            for el in prompt_electrons(p_row):
                key = (round(el["px"],6), round(el["py"],6), round(el["pz"],6), el["pdg_id"])
                if key in seen: continue
                seen.add(key)
                te, tp = (float(v) for v in momentum_to_eta_phi(el["px"], el["py"], el["pz"]))
                pt = float(np.hypot(el["px"], el["py"]))

                near = _dR(te, tp, eta_all, phi_all)
                surv = (near < 0.05) & (e_cal > a.e_thresh_gev)
                dR_cent = min((_dR(te, tp, ce, cp) for ce, cp in cents), default=np.inf)
                dR_peak = min((_dR(te, tp, pe, pp) for pe, pp in peaks), default=np.inf)
                rec.append(dict(
                    abs_eta=abs(te), pt=pt,
                    survives=float(surv.sum() >= 3),
                    matched_peak=float(dR_peak < a.dR_match),
                    dR_centroid=float(dR_cent), dR_peak=float(dR_peak)))
            done += 1
        del p_df, c_df

    d = pl.DataFrame(rec)
    n = d.height
    print(f"\n=== {n} electrons over {done} events  (eps={a.eps}, min_samples={a.min_samples}, "
          f"e_thresh={a.e_thresh_gev}, ew={a.energy_weighted}) ===")
    print(f"shower survives threshold : {d['survives'].mean():.1%}")
    print(f"peak-cell match < {a.dR_match}    : {d['matched_peak'].mean():.1%}")
    print(f"nearest-cluster dR quantiles (peak): "
          f"{[round(float(d['dR_peak'].quantile(q)),3) for q in (.25,.5,.75,.9)]}")

    print("\n-- survival by |eta| --");      _rate_by_bin(d, "survives", [0,1.0,1.5,2.0,2.5,4.0], "abs_eta")
    print("-- peak-match by |eta| --");      _rate_by_bin(d, "matched_peak", [0,1.0,1.5,2.0,2.5,4.0], "abs_eta")
    print("\n-- survival by pT [GeV] --");   _rate_by_bin(d, "survives", [0,10,25,45,70,1e4], "pt")
    print("-- peak-match by pT [GeV] --");   _rate_by_bin(d, "matched_peak", [0,10,25,45,70,1e4], "pt")


if __name__ == "__main__":
    main()