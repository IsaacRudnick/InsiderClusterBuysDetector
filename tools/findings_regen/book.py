"""Regenerate findings.BOOK_RESULTS + the multiples from an ensemble score
parquet. Books are band_backtest.band_periods output (equal weight,
non-overlapping 21-day periods, 20bps round trip); metrics are
sharpe_lab._series_stats verbatim.
"""
import os
import sys
import numpy as np, pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from tools import run_score_lab as lab
from tools import band_backtest as bb
from tools.sharpe_lab import _series_stats

DATASET, SCORES = sys.argv[1], sys.argv[2]

df = lab.load_dataset(DATASET)
sc = pd.read_parquet(SCORES)
assert len(sc) == len(df)
df["ens"] = sc["ens"].to_numpy()
df = bb.attach_benchmarks(df, bb.HORIZON)

BOOKS = [("Top band (70-90)", 0.70, 0.90),
         ("All clusters", 0.0, 1.0),
         ("Bottom 30% (elevated risk)", 0.0, 0.30)]
OLD = {"Top band (70-90)":      (0.345,0.230,1.303,1.400,-0.307,0.73),
       "All clusters":          (0.238,0.235,0.917,1.124,-0.322,0.58),
       "Bottom 30% (elevated risk)":(0.149,0.340,0.412,0.668,-0.576,0.51),
       "SPY":                   (0.181,0.141,1.192,1.125,-0.192,0.73),
       "IWM":                   (0.156,0.202,0.723,0.859,-0.273,0.60)}

rows, series = [], {}
for name, lo, hi in BOOKS:
    per = bb.band_periods(df, "ens", lo, hi)
    r = per["ret"]
    series[name] = r
    rows.append((name, _series_stats(r), len(per)))

# benchmarks: same period grid as the all-clusters book
per_all = bb.band_periods(df, "ens", 0.0, 1.0)
for b in ("SPY", "IWM"):
    r = per_all[f"bench_{b}"]
    series[b] = r
    rows.append((b, _series_stats(r), len(per_all)))

print(f"periods: {len(per_all)}  (BOOK_PROVENANCE says 78)\n")
hdr = f"{'book':28s}{'ann':>8}{'vol':>8}{'sharpe':>8}{'sortino':>9}{'maxdd':>8}{'win':>7}   vs published"
print(hdr); print("-"*len(hdr))
out = []
for name, s, n in rows:
    o = OLD[name]
    print(f"{name:28s}{s['ann_return']:>8.3f}{s['ann_vol']:>8.3f}{s['sharpe']:>8.3f}"
          f"{s['sortino']:>9.3f}{s['max_drawdown']:>8.3f}{s['win_rate']:>7.2f}"
          f"   was {o[0]:.3f}/{o[2]:.3f}/{o[4]:.3f}")
    out.append((name, round(s['ann_return'],3), round(s['ann_vol'],3), round(s['sharpe'],3),
                round(s['sortino'],3), round(s['max_drawdown'],3), round(s['win_rate'],2)))

print("\n=== findings.py BOOK_RESULTS block ===")
for name,a,v,sh,so,dd,w in out:
    print(f'    BookResult("{name}", {a}, {v}, {sh}, {so}, {dd}, {w}),')

tb = float((1.0 + series["Top band (70-90)"]).prod())
sp = float((1.0 + series["SPY"]).prod())
print(f"\nTOP_BAND_FINAL_MULTIPLE = {tb:.2f}   (was 5.77)")
print(f"SPY_FINAL_MULTIPLE      = {sp:.2f}   (was 2.76)")

per_tb = bb.band_periods(df, "ens", 0.70, 0.90)
ex = per_tb["ret"] - per_tb["bench_SPY"]
by = per_tb.assign(ex=ex).groupby("year")["ex"].mean()
print(f"TOP_BAND_YEARS_BEAT_SPY = ({int((by>0).sum())}, {len(by)})   (was (4, 7))")
print(f"  by year: {{{', '.join(f'{int(y)}: {v:+.4f}' for y,v in by.items())}}}")
