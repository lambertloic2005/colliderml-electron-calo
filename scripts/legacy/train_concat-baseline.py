
from pathlib import Path

import torch
from torch import nn
import wandb

from colliderml_electron.dataset import make_loader
from colliderml_electron.model import ConcatCaloRegressor


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def compute_grad_norm(model: nn.Module) -> float:
    norms = [
        p.grad.detach().norm(2)
        for p in model.parameters()
        if p.grad is not None
    ]

    if len(norms) == 0:
        return 0.0

    return torch.norm(torch.stack(norms), 2).item()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    loss_fn,
    device: torch.device,
) -> float:
    model.eval()

    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        pred = model(
            batch["x_sampled"],
            batch["x_high_level"],
            batch["mask"],
        )

        loss = loss_fn(pred, batch["target"])

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def main():
    config = {
        "architecture": "concat_transformer_baseline",
        "dataset": "colliderml_release1_zee_prompt_electrons",
        "parquet_path": "data/electrons/electrons.parquet",
        "target_stats_path": "data/electrons/target_stats.json",

        "max_cells": 128,
        "model_dim": 128,
        "n_heads": 4,
        "n_layers": 3,
        "dim_feedforward": 256,
        "dropout": 0.1,
        "output_dim": 6,

        "batch_size": 4,
        "n_epochs": 20,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,

        "log_freq_batches": 10,
        "watch_gradients": False,
    }

    with wandb.init(
        project="colliderml-electron-calo",
        name="concat-baseline",
        job_type="training",
        config=config,
    ) as run:
        cfg = run.config

        device = get_device()
        print(f"Using device: {device}")

        parquet_path = Path(cfg["parquet_path"])
        stats_path = Path(cfg["target_stats_path"])

        if not parquet_path.exists():
            raise FileNotFoundError(
                f"Could not find {parquet_path}. Build the electron table first."
            )

        if not stats_path.exists():
            raise FileNotFoundError(
                f"Could not find {stats_path}. Run the split/stat step first."
            )

        train_loader = make_loader(
            parquet_path=parquet_path,
            split="train",
            target_stats_path=stats_path,
            batch_size=cfg["batch_size"],
            shuffle=True,
        )

        val_loader = make_loader(
            parquet_path=parquet_path,
            split="val",
            target_stats_path=stats_path,
            batch_size=cfg["batch_size"],
            shuffle=False,
        )

        model = ConcatCaloRegressor(
            max_cells=cfg["max_cells"],
            model_dim=cfg["model_dim"],
            n_heads=cfg["n_heads"],
            n_layers=cfg["n_layers"],
            dim_feedforward=cfg["dim_feedforward"],
            dropout=cfg["dropout"],
            output_dim=cfg["output_dim"],
        ).to(device)

        if cfg["watch_gradients"]:
            run.watch(
                model,
                log="gradients",
                log_freq=100,
            )

        loss_fn = nn.MSELoss()

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg["learning_rate"],
            weight_decay=cfg["weight_decay"],
        )

        global_step = 0
        best_val_loss = float("inf")

        for epoch in range(1, cfg["n_epochs"] + 1):
            model.train()

            total_train_loss = 0.0
            n_batches = 0

            for batch_idx, batch in enumerate(train_loader):
                batch = move_batch_to_device(batch, device)

                pred = model(
                    batch["x_sampled"],
                    batch["x_high_level"],
                    batch["mask"],
                )

                loss = loss_fn(pred, batch["target"])

                optimizer.zero_grad()
                loss.backward()

                grad_norm = compute_grad_norm(model)

                optimizer.step()

                total_train_loss += loss.item()
                n_batches += 1

                if batch_idx % cfg["log_freq_batches"] == 0:
                    run.log(
                        {
                            "train_loss_batch": loss.item(),
                            "grad_norm": grad_norm,
                            "learning_rate": optimizer.param_groups[0]["lr"],
                            "epoch": epoch,
                            "batch_idx": batch_idx,
                        },
                        step=global_step,
                    )

                global_step += 1

            train_loss = total_train_loss / max(n_batches, 1)
            val_loss = evaluate(model, val_loader, loss_fn, device)

            best_val_loss = min(best_val_loss, val_loss)

            run.log(
                {
                    "train_loss_epoch": train_loss,
                    "val_loss_epoch": val_loss,
                    "best_val_loss": best_val_loss,
                    "epoch": epoch,
                },
                step=global_step,
            )

            print(
                f"epoch {epoch:03d} | "
                f"train loss {train_loss:.6f} | "
                f"val loss {val_loss:.6f} | "
                f"best val loss {best_val_loss:.6f}"
            )

        Path("checkpoints").mkdir(exist_ok=True)

        checkpoint_path = Path("checkpoints/concat_baseline.pt")

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": dict(cfg),
                "best_val_loss": best_val_loss,
            },
            checkpoint_path,
        )

        artifact = wandb.Artifact(
            name="concat_baseline",
            type="model",
            metadata={
                "best_val_loss": best_val_loss,
                "architecture": cfg["architecture"],
            },
        )
        artifact.add_file(str(checkpoint_path))
        run.log_artifact(artifact)

        run.summary["best_val_loss"] = best_val_loss
        run.summary["checkpoint_path"] = str(checkpoint_path)

        print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()