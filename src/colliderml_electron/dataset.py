# =============================================================================
# JOB 2 -- COMBO -- branch: combo-floor  (created off pointing-upgrade)
# Paste as: src/colliderml_electron/dataset.py   (keep this exact filename/path)
# Content: K=12 + phi_slope feature (60 features total) + min_pt plumbing so
#          training can apply the pT >= 10 GeV floor; floors/clip retained.
# Pairs with: the combo-floor train script (high_level_dim = 60, min_pt = 10.0).
# Preflight: python scripts/check_dims.py --high-level-dim 60 --output-dim 5
# Submit FROM THIS BRANCH: REGION=full sbatch slurm/run_train_test_new.sbatch
# Log must show: "acceptance cut pT >= 10: N -> M"  (if absent, scancel)
# =============================================================================
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset, DataLoader


use_angular_features: bool = True

# Detector subsystem codes present in the full zee_pu200 data.
DETECTOR_CODES = [9, 10, 11, 12, 13, 14]
N_DETECTORS = len(DETECTOR_CODES)

TARGET_COLS = [
    "truth_energy", "truth_px", "truth_py", "truth_pz",
    "truth_eta", "truth_phi", "truth_log_pt", "truth_z0",
]


def _one_hot_detector(det: np.ndarray) -> np.ndarray:
    """Map integer detector codes to one-hot rows of shape (n_cells, N_DETECTORS)."""
    out = np.zeros((len(det), N_DETECTORS), dtype=np.float32)
    for i, code in enumerate(DETECTOR_CODES):
        out[:, i] = (det == code).astype(np.float32)
    return out


class ElectronDataset(Dataset):
    "One sample per prompt electron."

    def __init__(
        self,
        parquet_path: str | Path,
        split: str | None = None,
        target_stats_path: str | Path | None = None,
        use_angular_features: bool = False,
        use_cluster_features: bool = False,
        max_abs_eta: float | None = None,
        min_abs_eta: float | None = None,
        min_pt: float | None = None,
    ):
        df = pl.read_parquet(parquet_path)

        if split is not None:
            df = df.filter(pl.col("split") == split)

        if max_abs_eta is not None:
            n0 = df.height
            df = df.filter(pl.col("truth_eta").abs() <= max_abs_eta)
            print(f"acceptance cut |truth_eta| <= {max_abs_eta}: {n0} -> {df.height}")
        if min_abs_eta is not None:
            n0 = df.height
            df = df.filter(pl.col("truth_eta").abs() >= min_abs_eta)
            print(f"acceptance cut |truth_eta| >= {min_abs_eta}: {n0} -> {df.height}")
        if min_pt is not None:
            n0 = df.height
            df = df.filter(pl.col("truth_log_pt").exp() >= min_pt)
            print(f"acceptance cut pT >= {min_pt}: {n0} -> {df.height}")

        self.df = df
        self.use_angular_features = use_angular_features
        self.use_cluster_features = use_cluster_features

        self.stats = None
        if target_stats_path is not None:
            self.stats = json.loads(Path(target_stats_path).read_text())

    def __len__(self) -> int:
        return self.df.height

    def __getitem__(self, idx: int) -> dict:
        row = self.df.row(idx, named=True)

        # x_sampled: positional coords for the Fourier embedding
        x_sampled = np.stack([
            np.asarray(row["cell_x"], dtype=np.float32),
            np.asarray(row["cell_y"], dtype=np.float32),
            np.asarray(row["cell_z"], dtype=np.float32),
        ], axis=-1)  # (n_cells, 3)

        # x_high_level: log-energy + optional angular features + detector one-hot
        e_cal = np.asarray(row["cell_e_calibrated"], dtype=np.float32)
        log_e = np.log(np.clip(e_cal, 1e-6, None))[:, None]

        # --- energy-weighted anchors (over ALL cells, before any topk) ---
        cell_phi_all = np.asarray(row["cell_phi"], dtype=np.float64)
        cell_eta_all = np.asarray(row["cell_eta"], dtype=np.float64)
        w = np.clip(e_cal.astype(np.float64), 1e-9, None)
        phi_centroid = float(np.arctan2(
            np.average(np.sin(cell_phi_all), weights=w),
            np.average(np.cos(cell_phi_all), weights=w),
        ))
        eta_centroid = float(np.average(cell_eta_all, weights=w))

        # --- canonical azimuthal frame: rotate event so the centroid sits at phi=0 ---
        cos_o, sin_o = np.cos(phi_centroid), np.sin(phi_centroid)
        x_rot = x_sampled[:, 0] * cos_o + x_sampled[:, 1] * sin_o
        y_rot = -x_sampled[:, 0] * sin_o + x_sampled[:, 1] * cos_o
        x_sampled = np.stack([x_rot, y_rot, x_sampled[:, 2]], axis=-1).astype(np.float32)

        det_oh = _one_hot_detector(np.asarray(row["cell_detector"]))

        if self.use_angular_features:
            cell_eta = np.asarray(row["cell_eta"], dtype=np.float32)[:, None]
            dphi_cell = np.arctan2(
                np.sin(cell_phi_all - phi_centroid),
                np.cos(cell_phi_all - phi_centroid),
            )
            sin_phi = np.sin(dphi_cell)[:, None].astype(np.float32)
            cos_phi = np.cos(dphi_cell)[:, None].astype(np.float32)

            # theta / cos(theta) from the same xyz as x_sampled (coords.py convention)
            cx = x_sampled[:, 0]; cy = x_sampled[:, 1]; cz = x_sampled[:, 2]
            r3d = np.sqrt(cx * cx + cy * cy + cz * cz)
            cos_theta = (cz / np.clip(r3d, 1e-9, None)).astype(np.float32)[:, None]
            theta = np.arctan2(np.hypot(cx, cy), cz).astype(np.float32)[:, None]

            x_high_level = np.concatenate(
                [
                    log_e,
                    cell_eta,
                    sin_phi,
                    cos_phi,
                    theta,
                    cos_theta,
                    det_oh,
                ],
                axis=-1,
            )
        else:
            x_high_level = np.concatenate([log_e, det_oh], axis=-1)

        # --- cluster-level scalars (over ALL cells, before any topk truncation),
        # broadcast to every cell. Appended at the END so x_high_level[..., 0]
        # stays log_e (the topk energy score). ---
        sum_e = float(e_cal.sum())                       # total calibrated E [GeV]
        cx = x_sampled[:, 0]; cy = x_sampled[:, 1]; cz = x_sampled[:, 2]
        r3d = np.sqrt(cx * cx + cy * cy + cz * cz)
        sin_theta = (np.hypot(cx, cy) / np.clip(r3d, 1e-9, None)).astype(np.float64)
        sum_et = float((e_cal.astype(np.float64) * sin_theta).sum())  # transverse-E proxy
        log_sum_et = float(np.log(max(sum_et, 1e-6)))

        # --- pointing-z0 anchor: energy-weighted LS fit of cell z vs cell r,
        # extrapolated to r=0. A straight shower points back to its production z.
        r_cell = np.hypot(cx, cy).astype(np.float64)
        z_cell = cz.astype(np.float64)
        wz = w  # same energy weights as the centroid
        wsum_z = float(wz.sum())
        r_bar = float(np.sum(wz * r_cell) / wsum_z)
        z_bar = float(np.sum(wz * z_cell) / wsum_z)
        var_r = float(np.sum(wz * (r_cell - r_bar) ** 2) / wsum_z)
        cov_rz = float(np.sum(wz * (r_cell - r_bar) * (z_cell - z_bar)) / wsum_z)
        # Variance floor: below ~5mm of radial leverage the LS slope is
        # statistically undetermined and can diverge to unphysical values
        # (observed |slope| up to ~123 in production data, vs a physical
        # ceiling of sinh(3) ~ 10 at |eta_max|=3). The old `> 1e-9` guard only
        # avoided literal div-by-zero, not near-singular fits. Below the
        # floor, fall back to "no slope information" (z0_anchor = z_bar) --
        # exactly the case r_spread/fit_rms exist to let the network flag.
        MIN_VAR_R = 25.0  # mm^2, i.e. sqrt ~ 5 mm minimum radial spread
        slope = cov_rz / var_r if var_r > MIN_VAR_R else 0.0   # dz/dr
        z0_anchor = float(z_bar - slope * r_bar)          # z at r=0 [mm]

        if self.use_cluster_features:
            n = x_high_level.shape[0]

            # E-weighted shower-shape moments around the anchors; the sign of
            # skew_phi tags the bremsstrahlung tail direction (i.e. the charge).
            dphi_all = np.arctan2(np.sin(cell_phi_all - phi_centroid),
                                  np.cos(cell_phi_all - phi_centroid))
            deta_all = cell_eta_all - eta_centroid
            wsum = float(w.sum())
            std_phi = float(np.sqrt(max(np.sum(w * dphi_all**2) / wsum, 1e-12)))
            skew_phi = float(np.sum(w * dphi_all**3) / wsum) / std_phi**3
            std_eta = float(np.sqrt(max(np.sum(w * deta_all**2) / wsum, 1e-12)))
            skew_eta = float(np.sum(w * deta_all**3) / wsum) / std_eta**3

            # --- azimuthal charge curvature: E-weighted LS slope of dphi vs the
            # geometry-correct DEPTH axis. Barrel showers develop along r, endcap
            # along |z|; we fit against whichever coordinate has the larger
            # energy-weighted spread, avoiding a near-zero variance (a pure dphi/dr
            # fit was noise in the endcap). Sign of dphi/d(depth) tracks the charge.
            # Reuses r_bar / var_r / wsum_z from the z0 pointing fit. The per-slice
            # <dphi> profile was tested (Plan B) and carried no signal beyond this
            # slope (profile-LDA ~ chance), so only the scalar slope is kept. ---
            absz_cell = np.abs(z_cell)
            z_bar_a = float(np.sum(wz * absz_cell) / wsum_z)
            var_z = float(np.sum(wz * (absz_cell - z_bar_a) ** 2) / wsum_z)
            if var_z > var_r:
                depth_c, d_bar, var_d = absz_cell, z_bar_a, var_z
            else:
                depth_c, d_bar, var_d = r_cell, r_bar, var_r
            dphi_bar = float(np.sum(wz * dphi_all) / wsum_z)
            cov_dphi = float(np.sum(wz * (depth_c - d_bar) * (dphi_all - dphi_bar)) / wsum_z)
            # Same variance-floor rationale as the z0 slope above (observed
            # |phi_slope*1000| up to ~78 vs a p99 of ~0.25 -- an extreme,
            # rare-event tail from the identical near-singular-fit mechanism).
            MIN_VAR_D = 25.0  # mm^2
            phi_slope = cov_dphi / var_d if var_d > MIN_VAR_D else 0.0   # rad/mm on adaptive depth

            # --- pointing-line quality: lets the net judge when to trust the anchor ---
            z_fit = z_bar + slope * (r_cell - r_bar)
            fit_rms = float(np.sqrt(np.sum(wz * (z_cell - z_fit) ** 2) / wsum_z))
            r_spread = float(np.sqrt(max(var_r, 0.0)))   # mm; small => slope ill-determined

            # --- longitudinal pointing profile: E-weighted <z> in K radial slices ---
            K = 12
            r_lo, r_hi = float(r_cell.min()), float(r_cell.max())
            edges = np.linspace(r_lo, r_hi + 1e-6, K + 1)
            prof_z = np.zeros(K, dtype=np.float32)   # (<z> - anchor)/100 per slice
            prof_r = np.zeros(K, dtype=np.float32)   # <r>/1000 per slice [m]
            prof_f = np.zeros(K, dtype=np.float32)   # energy fraction per slice
            for k in range(K):
                m = (r_cell >= edges[k]) & (r_cell < edges[k + 1])
                if m.any():
                    sk = float(wz[m].sum())
                    prof_z[k] = (np.sum(wz[m] * z_cell[m]) / sk - z0_anchor) / 100.0
                    prof_r[k] = (np.sum(wz[m] * r_cell[m]) / sk) / 1000.0
                    prof_f[k] = sk / wsum_z

            cluster_feats = np.array(
                [np.log(max(sum_e, 1e-6)), np.log(max(sum_et, 1e-6)), np.log(max(n, 1)),
                 std_phi, skew_phi, std_eta, skew_eta,
                 z0_anchor / 1000.0, slope,
                 r_spread / 1000.0, fit_rms / 100.0,
                 phi_slope * 1000.0],          # rad/mm -> rad/m, O(1); adaptive-depth charge curvature
                dtype=np.float32,
            )
            cluster_feats = np.concatenate(
                [cluster_feats, prof_r, prof_z, prof_f]
            ).astype(np.float32)
            # Defense-in-depth: even with the variance floors above, guarantee
            # every exposed high-level feature is bounded. Healthy events sit
            # well within +-30 (measured p99 across all features); this only
            # clips the pathological tail (previously up to ~449), not the
            # normal dynamic range.
            cluster_feats = np.clip(cluster_feats, -50.0, 50.0)
            x_high_level = np.concatenate(
                [x_high_level, np.tile(cluster_feats, (n, 1))], axis=-1
            )
        # truth targets

        target = np.array([row[c] for c in TARGET_COLS], dtype=np.float32)
        if self.stats is not None:
            for i, c in enumerate(TARGET_COLS):
                target[i] = (target[i] - self.stats[c]["mean"]) / self.stats[c]["std"]

        return {
            "x_sampled":    torch.from_numpy(x_sampled),
            "x_high_level": torch.from_numpy(x_high_level),
            "target":       torch.from_numpy(target),
            "n_cells":      x_sampled.shape[0],
            "phi_centroid": torch.tensor(phi_centroid, dtype=torch.float32),
            "eta_centroid": torch.tensor(eta_centroid, dtype=torch.float32),
            "log_sum_et":   torch.tensor(log_sum_et, dtype=torch.float32),
            "z0_anchor":    torch.tensor(z0_anchor, dtype=torch.float32),
            "truth_charge": torch.tensor(float(row["truth_charge"]), dtype=torch.float32),
        }


def collate_pad(batch: list[dict]) -> dict:
    """Pad variable-length cell sequences in a batch.

    Returns:
      x_sampled:    (B, L_max, 3)
      x_high_level: (B, L_max, H)
      mask:         (B, L_max)  — True = padding, False = real cell
      target:       (B, n_targets)
    """
    B = len(batch)
    L = max(item["n_cells"] for item in batch)
    D_sampled = batch[0]["x_sampled"].shape[-1]
    D_high = batch[0]["x_high_level"].shape[-1]
    T = batch[0]["target"].shape[-1]

    x_sampled    = torch.zeros(B, L, D_sampled)
    x_high_level = torch.zeros(B, L, D_high)
    mask         = torch.ones(B, L, dtype=torch.bool)  # True = padding
    target       = torch.zeros(B, T)
    phi_centroid = torch.zeros(B)
    eta_centroid = torch.zeros(B)
    log_sum_et   = torch.zeros(B)
    z0_anchor    = torch.zeros(B)
    truth_charge = torch.zeros(B)

    for i, item in enumerate(batch):
        n = item["n_cells"]
        x_sampled[i, :n] = item["x_sampled"]
        x_high_level[i, :n] = item["x_high_level"]
        mask[i, :n] = False
        target[i] = item["target"]
        phi_centroid[i] = item["phi_centroid"]
        eta_centroid[i] = item["eta_centroid"]
        log_sum_et[i]   = item["log_sum_et"]
        z0_anchor[i]    = item["z0_anchor"]
        truth_charge[i] = item["truth_charge"]
    return {
        "x_sampled": x_sampled,
        "x_high_level": x_high_level,
        "mask": mask,
        "target": target,
        "phi_centroid": phi_centroid,
        "eta_centroid": eta_centroid,
        "log_sum_et": log_sum_et,
        "z0_anchor": z0_anchor,
        "truth_charge": truth_charge,
    }


def make_loader(
    parquet_path: str | Path,
    split: str | None = None,
    target_stats_path: str | Path | None = None,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    use_angular_features: bool = False,
    use_cluster_features: bool = False,
    max_abs_eta: float | None = None,
    min_abs_eta: float | None = None,
    min_pt: float | None = None,
) -> DataLoader:
    ds = ElectronDataset(
        parquet_path,
        split=split,
        target_stats_path=target_stats_path,
        use_angular_features=use_angular_features,
        use_cluster_features=use_cluster_features,
        max_abs_eta=max_abs_eta,
        min_abs_eta=min_abs_eta,
        min_pt=min_pt,
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_pad,
    )