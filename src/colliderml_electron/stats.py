import json
from pathlib import Path

import polars as pl

def compute_target_stats(
    parquet_path: str | Path,
    target_cols: list[str],
    out_path: str | Path = "data/electrons/eta_phi_pt_z0_charge/target_stats.json",
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
    from colliderml_electron.dataset import TARGET_COLS

    compute_target_stats(
        "data/electrons/eta_phi_pt_z0_charge/zee_pu200_z0_charge.parquet",
        list(TARGET_COLS),
    )