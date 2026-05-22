import json
from pathlib import Path

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
    compute_target_stats(
        "data/electrons/electrons.parquet",
        ["truth_energy", "truth_px", "truth_py", "truth_pz", "truth_eta", "truth_phi"],
    )

