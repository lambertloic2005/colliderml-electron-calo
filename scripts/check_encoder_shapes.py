import argparse
import torch

from colliderml_electron.embedding import FourierPositionalEncoding
from colliderml_electron.encoder import CellEncoder
from colliderml_electron.dataset import make_loader, N_DETECTORS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", required=True, help="path to a built electron parquet")
    p.add_argument("--batch-size", type=int, default=8)
    args = p.parse_args()

    embed = FourierPositionalEncoding(
        input_dim=3,
        high_level_dim=1 + N_DETECTORS,
        num_frequencies=[6, 6, 6],
        dim_max=[1100.0, 1100.0, 3000.0],
        shift=[0.0, 0.0, 0.0],
    )
    F_FULL = 3 * 6 * 2 + (1 + N_DETECTORS)

    enc = CellEncoder(
        fourier_embed=embed,
        fourier_out_dim=F_FULL,
        model_dim=128, n_heads=4, n_layers=3,
        dim_feedforward=256, dropout=0.0,
    )
    enc.eval()  # no dropout / no grad needed for a shape check

    loader = make_loader(args.parquet, batch_size=args.batch_size)
    batch = next(iter(loader))

    with torch.no_grad():
        h, mask = enc(batch["x_sampled"], batch["x_high_level"], batch["mask"])

    B, L = mask.shape
    real_per_electron = (~mask).sum(dim=1)  # (B,)

    print(f"batch size B            : {B}")
    print(f"padded length L         : {L}")
    print(f"x_sampled               : {tuple(batch['x_sampled'].shape)}")
    print(f"x_high_level            : {tuple(batch['x_high_level'].shape)}")
    print(f"mask dtype / shape      : {mask.dtype} {tuple(mask.shape)}")
    print(f"per-cell encoded h      : {tuple(h.shape)}   (expect (B, L, 128))")
    print(f"real cells per electron : {real_per_electron.tolist()}")
    print(f"any NaNs in h?          : {torch.isnan(h).any().item()}")


if __name__ == "__main__":
    main()