# scripts/diagnose_phi.py
"""Isolate why phi won't learn. Uses your real model unchanged; only the loss
is swapped to phi-only, and --n-electrons optionally restricts the dataset.

(A) phi alone, full smoke set:
      python scripts/diagnose_phi.py --parquet data/electrons/smoke.parquet
(B) phi alone, overfit a handful:
      python scripts/diagnose_phi.py --parquet data/electrons/smoke.parquet \
          --n-electrons 4 --batch-size 4 --steps 1000
"""
import argparse
import torch
from torch.utils.data import DataLoader, Subset

from colliderml_electron.embedding import FourierPositionalEncoding
from colliderml_electron.encoder import CellEncoder
from colliderml_electron.regressor import Regressor, EtaPhiModel
from colliderml_electron.dataset import ElectronDataset, collate_pad, N_DETECTORS


def phi_only_loss(pred, phi_true):
    """Plain MSE on the unit-circle coords. pred row = [eta, phi_cos, phi_sin]."""
    cos_t, sin_t = torch.cos(phi_true), torch.sin(phi_true)
    return ((pred[:, 1] - cos_t).pow(2) + (pred[:, 2] - sin_t).pow(2)).mean()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--phi-idx", type=int, default=5)
    p.add_argument("--n-electrons", type=int, default=None,
                   help="restrict to the first N electrons (overfit test)")
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

    ds = ElectronDataset(args.parquet)
    if args.n_electrons is not None:
        ds = Subset(ds, range(args.n_electrons))
        print(f"restricted to {args.n_electrons} electrons (overfit test)")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        collate_fn=collate_pad)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print("phi_loss baseline (predict a constant direction) is ~1.0\n")
    model.train()
    step = 0
    while step < args.steps:
        for batch in loader:
            pred = model(batch["x_sampled"], batch["x_high_level"], batch["mask"])
            phi_true = batch["target"][:, args.phi_idx]
            loss = phi_only_loss(pred, phi_true)
            opt.zero_grad(); loss.backward(); opt.step()
            if step % 20 == 0:
                print(f"step {step:04d}  phi_loss={loss.item():.4f}")
            step += 1
            if step >= args.steps:
                break
    print("\ndone.")


if __name__ == "__main__":
    main()