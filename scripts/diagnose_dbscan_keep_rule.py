"""Where do the starved clusters lose their cells: family selection or DBSCAN?

For each prompt electron in a sample of events, compare:
  n_family / sumET_family   -- truth-selected cells, before any DBSCAN
  n_kept   / sumET_kept     -- current rule: only the truth-nearest cluster
  n_union  / sumET_union    -- proposed rule: all non-noise clusters

If sumET_family ~ pT but sumET_kept << pT, the current keep rule is the
culprit and the union rule should recover the energy. If sumET_family
itself is << pT, the loss is upstream (acceptance / genuine leakage).
"""
import numpy as np
from sklearn.cluster import DBSCAN

from colliderml_electron.io import (
    load_frames, get_event, prompt_electrons,
    descendant_pids, cells_for_particle_set, dbscan_keep_mask,
)
from colliderml_electron.coords import xyz_to_eta_phi, delta_eta_phi
from colliderml_electron.calibration import calibrate

N_EVENTS = 50
EPS, MIN_SAMPLES = 0.08, 2
MAX_ABS_ETA = 3.5


def sum_et(cells, mask):
    x, y, z = cells["x"][mask], cells["y"][mask], cells["z"][mask]
    e = calibrate(cells["e_total"][mask], x, y, z).astype(np.float64)
    r3 = np.sqrt(x*x + y*y + z*z)
    st = np.hypot(x, y) / np.clip(r3, 1e-9, None)
    return float((e * st).sum())


def union_keep_mask(cells, eps=EPS, min_samples=MIN_SAMPLES):
    """Proposed rule: DBSCAN as pure noise filter -- keep all non-noise clusters."""
    n = len(cells["x"])
    if n == 0:
        return np.zeros(0, dtype=bool)
    eta_c, phi_c = xyz_to_eta_phi(cells["x"], cells["y"], cells["z"])
    eta_c, phi_c = np.asarray(eta_c, float), np.asarray(phi_c, float)
    deta = eta_c[:, None] - eta_c[None, :]
    _, dphi = delta_eta_phi(eta_c[:, None], phi_c[:, None],
                            eta_c[None, :], phi_c[None, :])
    dist = np.sqrt(deta**2 + dphi**2)
    labels = DBSCAN(eps=eps, min_samples=min(min_samples, max(n, 1)),
                    metric="precomputed").fit_predict(dist)
    keep = labels != -1
    return keep if keep.any() else np.ones(n, dtype=bool)


def main():
    frames = load_frames(channel="zee", pileup="pu200", max_events=N_EVENTS)
    rows = []
    for i in range(N_EVENTS):
        p_row, c_row = get_event(frames, i)
        for e in prompt_electrons(p_row):
            pt = float(np.hypot(e["px"], e["py"]))
            p = float(np.sqrt(e["px"]**2 + e["py"]**2 + e["pz"]**2))
            eta = float(np.arctanh(e["pz"] / p)) if p > 0 else 99.0
            if abs(eta) > MAX_ABS_ETA or pt < 1.0:
                continue
            family = descendant_pids(p_row, e["particle_id"])
            cells = cells_for_particle_set(c_row, family)
            n_fam = len(cells["x"])
            if n_fam == 0:
                continue
            all_mask = np.ones(n_fam, dtype=bool)
            kept = dbscan_keep_mask(cells, e, eps=EPS, min_samples=MIN_SAMPLES)
            union = union_keep_mask(cells)
            rows.append(dict(
                pt=pt, eta=eta, n_fam=n_fam,
                n_kept=int(kept.sum()), n_union=int(union.sum()),
                r_fam=sum_et(cells, all_mask) / pt,
                r_kept=sum_et(cells, kept) / pt,
                r_union=sum_et(cells, union) / pt,
            ))

    r_fam = np.array([r["r_fam"] for r in rows])
    r_kept = np.array([r["r_kept"] for r in rows])
    r_union = np.array([r["r_union"] for r in rows])
    pts = np.array([r["pt"] for r in rows])

    print(f"electrons analysed: {len(rows)}")
    for name, r in (("family (pre-DBSCAN)", r_fam),
                    ("current keep rule  ", r_kept),
                    ("union keep rule    ", r_union)):
        print(f"  {name}: median ratio {np.median(r):.3f}   "
              f"starved(<0.5) {np.mean(r < 0.5):.2%}")

    # the smoking gun: events healthy pre-DBSCAN but starved after the keep rule
    smoking = (r_fam > 0.8) & (r_kept < 0.5)
    print(f"\nhealthy family but starved after current rule: {np.mean(smoking):.2%}")
    if smoking.any():
        rec = np.mean(r_union[smoking] > 0.8)
        print(f"  of those, recovered by union rule: {rec:.2%}")
        print(f"  their median pT: {np.median(pts[smoking]):.1f} GeV "
              f"(all: {np.median(pts):.1f} GeV)")


if __name__ == "__main__":
    main()