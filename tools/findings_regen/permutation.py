"""Permutation test + band excess, reusing a precomputed ensemble score
parquet instead of refitting 10 members (band_robustness.py's main() refits).
Logic copied verbatim from tools/band_robustness.check_multiple_comparisons.
"""
import os
import sys
import numpy as np, pandas as pd
from scipy import stats

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from tools import run_score_lab as lab
from tools import band_backtest as bb
from tools.band_robustness import _excess, ann

DATASET, SCORES = sys.argv[1], sys.argv[2]

df = lab.load_dataset(DATASET)
sc = pd.read_parquet(SCORES)
assert len(sc) == len(df), f"{len(sc)} vs {len(df)}"
df["ens"] = sc["ens"].to_numpy()
df = bb.attach_benchmarks(df, bb.HORIZON)

for BAND_LO, BAND_HI, tag in ((0.70, 0.90, "70-90 (findings.py top_band)"),
                              (0.80, 0.90, "80-90 (band_robustness default)")):
    print(f"\n########## BAND {tag} ##########")
    per = bb.band_periods(df, "ens", BAND_LO, BAND_HI)
    ex = _excess(per)
    p_raw = float(stats.ttest_1samp(ex, 0.0).pvalue)
    n_bands = len(bb.BANDS)
    print(f"  periods {len(per)}   raw p {p_raw:.4f}   Bonferroni x{n_bands} {min(1.0,p_raw*n_bands):.4f}")

    rng = np.random.default_rng(0)
    work = df.copy()
    base = int(work["entry_idx"].min())
    work["period"] = (work["entry_idx"] - base) // bb.HORIZON
    best_null = []
    for _ in range(200):
        work["_shuf"] = work.groupby("period")["ens"].transform(
            lambda s: rng.permutation(s.to_numpy()))
        best = -9.9
        for lo, hi, _n in bb.BANDS:
            p = bb.band_periods(work, "_shuf", lo, hi)
            if len(p) > 2:
                best = max(best, float(_excess(p).mean()))
        best_null.append(best)
    best_null = np.array(best_null)
    observed = float(ex.mean())
    pperm = float((best_null >= observed).mean())
    print(f"    observed best band excess/period  {observed:+.4f} ({ann(observed):+.2%}/yr)")
    print(f"    null median                       {np.median(best_null):+.4f} ({ann(np.median(best_null)):+.2%}/yr)")
    print(f"    null 95th pct                     {np.percentile(best_null,95):+.4f}")
    print(f"    permutation p                     {pperm:.4f}")
    print(f"\n  ==> TOP_BAND_ANNUALIZED_EXCESS = {ann(observed):.4f}   (was 0.1977)")
    print(f"  ==> TOP_BAND_PERMUTATION_P     = {pperm:.3f}   (was 0.435)")
