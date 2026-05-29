from pathlib import Path
import numpy as np
import polars as pl
import json


def assign_splits(
    parquet_path: str | Path,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
    out_path: str | Path | None = None,
) -> None:
    df = pl.read_parquet(parquet_path)

    events = df["event_id"].unique().sort().to_numpy()
    rng = np.random.default_rng(seed)
    events = rng.permutation(events)   # was: rng.shuffle(events)

    n = len(events)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)
    train_ids = set(events[:n_train].tolist())
    val_ids = set(events[n_train:n_train + n_val].tolist())

    def which(eid: int) -> str:
        if eid in train_ids:
            return "train"
        if eid in val_ids:
            return "val"
        return "test"
    
    split_col = [which(e) for e in df["event_id"].to_list()]
    df = df.with_columns(pl.Series("split", split_col))

    out = Path(out_path) if out_path else Path(parquet_path)
    df.write_parquet(out)

    counts = df.group_by("split").len().sort("split")
    print(f"Split assignment ({n} events -> {df.height} electrons):")
    print(counts)

def compute_target_stats(
    parquet_path: str | Path,
    target_cols: list[str],
    out_path: str | Path = "data/electrons/target_stats.json",
) -> dict:
    df = pl.read_parquet(parquet_path).filter(pl.col("split") == "train")
    stats = {
        col: {"mean": float(df[col].mean()), "std": float(df[col].std())}
        for col in target_cols
    }
    Path(out_path).write_text(json.dumps(stats, indent=2))
    print(f"Wrote target stats to {out_path}")
    return stats

if __name__ == "__main__":
    assign_splits("data/electrons/electrons.parquet")
    compute_target_stats(
        "data/electrons/electrons.parquet",
        ["truth_energy", "truth_px", "truth_py", "truth_pz", "truth_eta", "truth_phi", "truth_log_pt"],
    )