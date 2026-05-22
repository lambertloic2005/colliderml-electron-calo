import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from colliderml_electron.dataset import make_loader, TARGET_COLS
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


def denormalize(values: np.ndarray, stats: dict) -> np.ndarray:
    """
    Convert normalized targets back into real physical units.

    normalized = (true - mean) / std
    so:
    true = normalized * std + mean
    """

    values = values.copy()

    for i, target_name in enumerate(TARGET_COLS):
        mean = stats[target_name]["mean"]
        std = stats[target_name]["std"]
        values[:, i] = values[:, i] * std + mean

    return values


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()

    all_preds = []
    all_targets = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        preds = model(
            batch["x_sampled"],
            batch["x_high_level"],
            batch["mask"],
        )

        all_preds.append(preds.cpu().numpy())
        all_targets.append(batch["target"].cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    return all_preds, all_targets


def plot_expected_vs_predicted(
    true_values: np.ndarray,
    pred_values: np.ndarray,
    target_name: str,
    output_dir: Path,
):
    plt.figure(figsize=(6, 6))

    plt.scatter(true_values, pred_values, s=8, alpha=0.5)

    low = min(true_values.min(), pred_values.min())
    high = max(true_values.max(), pred_values.max())

    plt.plot(
        [low, high],
        [low, high],
        linestyle="--",
        label="perfect prediction",
    )

    plt.xlabel(f"Expected / true {target_name}")
    plt.ylabel(f"Predicted {target_name}")
    plt.title(f"Expected vs predicted: {target_name}")
    plt.legend()
    plt.tight_layout()

    output_path = output_dir / f"expected_vs_predicted_{target_name}.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path


def main():
    device = get_device()
    print(f"Using device: {device}")

    checkpoint_path = Path("checkpoints/concat_baseline.pt")
    parquet_path = Path("data/electrons/electrons.parquet")
    stats_path = Path("data/electrons/target_stats.json")

    output_dir = Path("results/concat_baseline")
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = json.loads(stats_path.read_text())

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]

    test_loader = make_loader(
        parquet_path=parquet_path,
        split="test",
        target_stats_path=stats_path,
        batch_size=config["batch_size"],
        shuffle=False,
    )

    model = ConcatCaloRegressor(
        max_cells=config["max_cells"],
        model_dim=config["model_dim"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        dim_feedforward=config["dim_feedforward"],
        dropout=config["dropout"],
        output_dim=config["output_dim"],
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    pred_norm, true_norm = collect_predictions(model, test_loader, device)

    pred = denormalize(pred_norm, stats)
    true = denormalize(true_norm, stats)

    print("\nMaking expected-vs-predicted plots...")

    for i, target_name in enumerate(TARGET_COLS):
        path = plot_expected_vs_predicted(
            true_values=true[:, i],
            pred_values=pred[:, i],
            target_name=target_name,
            output_dir=output_dir,
        )

        print(f"Saved {path}")

    print(f"\nAll plots saved in: {output_dir}")


if __name__ == "__main__":
    main()