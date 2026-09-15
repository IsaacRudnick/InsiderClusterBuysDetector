"""Re-derive findings.py's population-level constants.

HORIZON_EXPECTATIONS, FLAT_REFINEMENTS, BENCHMARK_CAGR and SURVIVORSHIP.
None of these depend on a model score.

Usage: python tools/findings_regen/population.py [DATASET] [EVENTS]
Reads price_cache/ from the current directory.
"""
import os
import sys

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from tools import data_paths  # noqa: E402

DATASET = sys.argv[1] if len(sys.argv) > 1 else data_paths.latest_research_dataset(required=True)
EVENTS = sys.argv[2] if len(sys.argv) > 2 else data_paths.latest_events_file(required=True)
BENCH = {"SPY (large cap)": "SPY", "RSP (equal-weight S&P)": "RSP", "MDY (mid cap)": "MDY",
         "IWC (micro cap)": "IWC", "IWM (small cap)": "IWM", "XBI (biotech)": "XBI"}
REFINEMENTS = [("Reacting faster to a fresh filing", "x_median_filing_delay_days"),
               ("More insiders in the cluster", "x_n_insiders"),
               ("Larger total dollars bought", "x_log1p_total_value"),
               ("A higher CEO share of the buying", "x_ceo_value_share"),
               ("More ten-percent owners", "x_n_ten_pct")]


def logex(df, h):
    f, s = df[f"fwd_{h}"].astype(float), df[f"spy_{h}"].astype(float)
    return (np.log1p(f.clip(lower=-.999)) - np.log1p(s.clip(lower=-.999))).where(f.notna() & s.notna())


def bench_ret(tkr, entry_days, h):
    d = pd.read_parquet(f"price_cache/{tkr}.parquet").sort_index()
    o, idx = d["open"].to_numpy(), d.index
    pos = idx.searchsorted(pd.to_datetime(entry_days))
    out = np.full(len(entry_days), np.nan)
    ok = pos < len(idx) - h
    out[ok] = o[np.clip(pos + h, 0, len(idx) - 1)][ok] / o[np.clip(pos, 0, len(idx) - 1)][ok] - 1.0
    return pd.Series(out)


df = pd.read_parquet(DATASET)
print(f"dataset {DATASET} ({len(df)} rows)\nevents  {EVENTS}\n")

print("HORIZON_EXPECTATIONS: (trading_days, vs_spy, vs_iwm, vs_iwc, win_rate)")
for h in (10, 21, 63, 126, 252):
    f = df[f"fwd_{h}"].astype(float)
    le_spy = logex(df, h)
    legs = []
    for t in ("IWM", "IWC"):
        b = bench_ret(t, df["entry_day"], h)
        le = (np.log1p(f.clip(lower=-.999)) - np.log1p(b.clip(lower=-.999))).where(f.notna() & b.notna())
        legs.append(le.mean() * 252 / h if le.notna().sum() > 500 else float("nan"))
    print(f"    HorizonExpectation({h}, {le_spy.mean() * 252 / h:.4f}, {legs[0]:.4f}, "
          f"{legs[1]:.4f}, {(le_spy.dropna() > 0).mean():.3f}),")

print("\nFLAT_REFINEMENTS: median 21-day log excess by quartile, lowest first")
df["le21"] = logex(df, 21)
for label, col in REFINEMENTS:
    d = df[df[col].notna() & df["le21"].notna()]
    q = pd.qcut(d[col].rank(method="first"), 4, labels=False)  # ties broken by order
    meds = tuple(round(float(d.loc[q == i, "le21"].median()), 4) for i in range(4))
    print(f'    ("{label}", {meds}),')

print("\nBENCHMARK_CAGR: open-to-open over the trading days ALL six share")
lo, hi = pd.to_datetime(df["event_day"]).min(), pd.to_datetime(df["event_day"]).max()
frames = {k: pd.read_parquet(f"price_cache/{t}.parquet").sort_index() for k, t in BENCH.items()}
common = None
for d in frames.values():
    idx = d[(d.index >= lo) & (d.index <= hi)].index
    common = idx if common is None else common.intersection(idx)
yrs = (common.max() - common.min()).days / 365.25
print(f"    ({len(common)} common days, {common.min().date()}..{common.max().date()})")
for k, d in frames.items():
    s = d.loc[common, "open"]
    print(f'    "{k}": {(float(s.iloc[-1]) / float(s.iloc[0])) ** (1 / yrs) - 1:.4f},')

print("\nSURVIVORSHIP: event tickers with no file in price_cache/")
ev = pd.read_parquet(EVENTS)
have = {f[:-8] for f in os.listdir("price_cache") if f.endswith(".parquet")}
t = ev["ticker"].astype(str).str.upper()
uniq = set(t.unique())
dead = t.isin({x for x in uniq if x not in have})
v = ev["value"].astype(float).fillna(0)
print(f'    "frac_tickers_unpriceable": {len(set(t[dead])) / len(uniq):.3f},')
print(f'    "frac_buy_rows_unpriceable": {dead.mean():.3f},')
print(f'    "frac_buy_dollars_unpriceable": {v[dead].sum() / v.sum():.3f},')
