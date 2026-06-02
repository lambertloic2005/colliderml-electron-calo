import polars as pl, json
from colliderml_electron.dataset import TARGET_COLS

P = "data/electrons/electrons_dbscan.parquet"        # match your real filename
S = "data/electrons/target_stats.json"
df = pl.read_parquet(P)

# 1) split column exists with the three expected values, ~70/15/15
print(df.group_by("split").len().sort("split"))
print({k: round(v/df.height, 3) for k, v in
       zip(df.group_by("split").len().sort("split")["split"],
           df.group_by("split").len().sort("split")["len"])})

# 2) NO event leaks across splits (this number must be 0)
leaks = (df.group_by("event_id").agg(pl.col("split").n_unique().alias("n"))
           .filter(pl.col("n") > 1).height)
print("events spanning >1 split:", leaks)

# 3) stats file has every TARGET_COL incl truth_log_pt, std finite & > 0
stats = json.load(open(S))
import math
for c in TARGET_COLS:
    ok = c in stats and math.isfinite(stats[c]["std"]) and stats[c]["std"] > 0
    print(f"{c:14s} present={c in stats}  std_ok={ok}  "
          f"mean={stats.get(c,{}).get('mean')}  std={stats.get(c,{}).get('std')}")

# 4) stats were computed on TRAIN ONLY — re-derive and compare
tr = df.filter(pl.col("split") == "train")
print("recomputed train mean/std for log_pt:",
      round(tr["truth_log_pt"].mean(), 6), round(tr["truth_log_pt"].std(), 6))

import math
m_file, s_file = stats["truth_log_pt"]["mean"], stats["truth_log_pt"]["std"]
m_tr,   s_tr   = tr["truth_log_pt"].mean(),     tr["truth_log_pt"].std()
print("file:       ", round(m_file, 6), round(s_file, 6))
print("recomputed: ", round(m_tr, 6),   round(s_tr, 6))
print("match:", math.isclose(m_file, m_tr, rel_tol=1e-6)
              and math.isclose(s_file, s_tr, rel_tol=1e-6))