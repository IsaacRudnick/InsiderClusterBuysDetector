"""Re-run run_sharpe_search's grade() for the SHIPPED candidate only, using a
precomputed ensemble score parquet instead of refitting 10 seeds. Reproduces
findings.TOP_BAND_SHARPE_* (the 132-recipe search + its permutation test).
"""
import os
import sys, time
import numpy as np, pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from tools import run_score_lab as lab
from tools import sharpe_lab as sh
from tools import run_sharpe_search as rss

DATASET, SCORES = sys.argv[1], sys.argv[2]
DRAWS = int(sys.argv[3]) if len(sys.argv) > 3 else 120
NAME = "C19_month_vol_rel_a35_live"

df = lab.load_dataset(DATASET)
sc = pd.read_parquet(SCORES)
assert len(sc) == len(df), f"{len(sc)} vs {len(df)}"
df[NAME] = sc["ens"].to_numpy()

prepared = sh.prepare(df, DATASET, score_cols=(NAME,))
key = ["ticker", "event_day"]
prepared = prepared.merge(df[key + [NAME]], on=key, how="left", suffixes=("", "_dup"))
if f"{NAME}_dup" in prepared.columns:
    prepared[NAME] = prepared[f"{NAME}_dup"]
    prepared = prepared.drop(columns=[f"{NAME}_dup"])

t0 = time.time()
r = rss.grade(prepared, NAME, DRAWS)
print(f"graded in {time.time()-t0:.0f}s  ({DRAWS} permutation draws)\n")
for k in ("recipe", "avg_names", "ann_return", "ann_vol", "sharpe", "sortino",
          "max_drawdown", "win_rate", "null_median", "null_p95", "perm_p"):
    if k in r:
        v = r[k]
        print(f"  {k:14s} {v:.4f}" if isinstance(v, float) else f"  {k:14s} {v}")
print(f"\n  TOP_BAND_SHARPE_PERMUTATION_P = {r.get('perm_p', float('nan')):.4f}   (was 0.005)")
print(f"  TOP_BAND_SHARPE_NULL_MEDIAN   = {r.get('null_median', float('nan')):.3f}   (was 0.987)")
print(f"  TOP_BAND_SHARPE_NULL_P95      = {r.get('null_p95', float('nan')):.3f}   (was 1.236)")
