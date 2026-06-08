# extract_test.py
import polars as pl

df = pl.read_parquet("data/electrons/zee_pu200_supervised_dbscan.parquet")
test = df.filter(pl.col("split") == "test")

print(f"test electrons: {test.height}")
print(f"test events:    {test['event_id'].n_unique()}")

test.write_parquet("zee_pu200_supervised_dbscan_TEST.parquet")