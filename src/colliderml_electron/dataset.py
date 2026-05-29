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
    "truth_eta", "truth_phi", "truth_log_pt",
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
    ):
        df = pl.read_parquet(parquet_path)

        if split is not None:
            df = df.filter(pl.col("split") == split)

        self.df = df
        self.use_angular_features = use_angular_features

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

        det_oh = _one_hot_detector(np.asarray(row["cell_detector"]))

        if self.use_angular_features:
            cell_eta = np.asarray(row["cell_eta"], dtype=np.float32)[:, None]
            cell_phi = np.asarray(row["cell_phi"], dtype=np.float32)

            sin_phi = np.sin(cell_phi)[:, None].astype(np.float32)
            cos_phi = np.cos(cell_phi)[:, None].astype(np.float32)

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

    for i, item in enumerate(batch):
        n = item["n_cells"]
        x_sampled[i, :n] = item["x_sampled"]
        x_high_level[i, :n] = item["x_high_level"]
        mask[i, :n] = False
        target[i] = item["target"]

    return {
        "x_sampled": x_sampled,
        "x_high_level": x_high_level,
        "mask": mask,
        "target": target,
    }


def make_loader(
    parquet_path: str | Path,
    split: str | None = None,
    target_stats_path: str | Path | None = None,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    use_angular_features: bool = False,
) -> DataLoader:
    ds = ElectronDataset(
        parquet_path,
        split=split,
        target_stats_path=target_stats_path,
        use_angular_features=use_angular_features,
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_pad,
    )