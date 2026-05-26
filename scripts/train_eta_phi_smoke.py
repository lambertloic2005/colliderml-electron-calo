# scripts/train_eta_phi_smoke.py
import argparse
import torch
import wandb

# Adjust import names if your module filenames differ.
from colliderml_electron.embedding import FourierPositionalEncoding
from colliderml_electron.encoder import CellEncoder
from colliderml_electron.regressor import Regressor, EtaPhiModel, eta_phi_geometric_loss
from colliderml_electron.dataset import make_loader, N_DETECTORS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", required=True, help="path to a built electron parquet")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--eta-idx", type=int, default=0, help="column index of truth eta in target")
    p.add_argument("--phi-idx", type=int, default=1, help="column index of truth phi in target")
    args = p.parse_args()

    wandb.init(
        project="colliderml-electron",
        config={
            "model_dim": 128, "n_heads": 4, "n_layers": 3,
            "dim_feedforward": 256, "lr": args.lr,
            "batch_size": args.batch_size,
            "combine": "masked_sum", "loss": "geometric_log",
        },
    )

    # --- build the two stages and compose them ---
    embed = FourierPositionalEncoding(
        input_dim=3,
        high_level_dim=1 + N_DETECTORS,
        num_frequencies=[6, 6, 6],
        dim_max=[1100.0, 1100.0, 3000.0],
        shift=[0.0, 0.0, 0.0],
    )
    F_FULL = 3 * 6 * 2 + (1 + N_DETECTORS)

    encoder = CellEncoder(
        fourier_embed=embed, fourier_out_dim=F_FULL,
        model_dim=128, n_heads=4, n_layers=3,
        dim_feedforward=256, dropout=0.0,
    )
    regressor = Regressor(model_dim=128, hidden=128, n_outputs=3)
    model = EtaPhiModel(encoder, regressor)

    wandb.watch(model, log="all", log_freq=10)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loader = make_loader(args.parquet, batch_size=args.batch_size)

    # --- sanity: confirm which target columns are eta / phi BEFORE training ---
    first = next(iter(loader))
    print(f"target shape : {tuple(first['target'].shape)}")
    print(f"first row    : {[round(v, 4) for v in first['target'][0].tolist()]}")
    print(f"using eta_idx={args.eta_idx}, phi_idx={args.phi_idx} — confirm these are right\n")

    # --- short training loop ---
    model.train()
    step = 0
    while step < args.steps:
        for batch in loader:                      # loops over the tiny set repeatedly
            pred = model(batch["x_sampled"], batch["x_high_level"], batch["mask"])
            eta_true = batch["target"][:, args.eta_idx]
            phi_true = batch["target"][:, args.phi_idx]

            loss, parts = eta_phi_geometric_loss(pred, eta_true, phi_true)

            opt.zero_grad()
            loss.backward()
            opt.step()

            wandb.log({**parts, "loss_logsum": loss.item()}, step=step)

            if step % 10 == 0:
                print(f"step {step:04d}  geo={parts['geo_mean']:.4f}  "
                      f"eta={parts['eta_loss']:.4f}  phi={parts['phi_loss']:.4f}")
            step += 1
            if step >= args.steps:
                break
    wandb.finish()
    print("\ndone.")


if __name__ == "__main__":
    main()