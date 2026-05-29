import polars as pl
path = "data/electrons/electrons.parquet"
df = pl.read_parquet(path)
df = df.with_columns(pl.col("truth_pt").log().alias("truth_log_pt"))
df.write_parquet(path)