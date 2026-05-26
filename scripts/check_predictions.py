# scripts/check_predictions.py
"""Dump model predictions and targets for one batch — diagnostic only.

Run from the repo root:
    python scripts/check_predictions.py --parquet path/to/smoke.parquet
"""
from __future__ import annotations
import argparse
import torch

from colliderml_electron.embedding import FourierPositionalEncoding
from colliderml_electron.encoder import CellEncoder
from colliderml_electron.regressor import Regressor, EtaPhiModel
from colliderml_electron.dataset import make_loader, N_DETECTORS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--eta-idx", type=int, default=4)
    p.add_argument("--phi-idx", type=int, default=5)
    args = p.parse_args()

    embed = FourierPositionalEncoding(
        input_dim=3, high_level_dim=1 + N_DETECTORS,
        num_frequencies=[6, 6, 6], dim_max=[1100.0, 1100.0, 3000.0],
        shift=[0.0, 0.0, 0.0],
    )
    F_FULL = 3 * 6 * 2 + (1 + N_DETECTORS)

    encoder = CellEncoder(fourier_embed=embed, fourier_out_dim=F_FULL,
                          model_dim=128, n_heads=4, n_layers=3,
                          dim_feedforward=256, dropout=0.0)
    regressor = Regressor(model_dim=128, hidden=128, n_outputs=3)
    model = EtaPhiModel(encoder, regressor)

    loader = make_loader(args.parquet, batch_size=args.batch_size)
    batch = next(iter(loader))

    model.eval()
    with torch.no_grad():
        pred = model(batch["x_sampled"], batch["x_high_level"], batch["mask"])

    print("pred [eta, cos, sin]:")
    print(pred[:5])
    print(f"eta_true (col {args.eta_idx}):", batch["target"][:5, args.eta_idx])
    print(f"phi_true (col {args.phi_idx}):", batch["target"][:5, args.phi_idx])
    print("cells/electron      :", (~batch["mask"]).sum(1)[:5])


if __name__ == "__main__":
    main()