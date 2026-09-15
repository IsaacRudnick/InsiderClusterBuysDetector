"""Which slice of the ranking should the screener actually put in front of a
user, and what does holding that slice earn against BOTH yardsticks?

WHY THIS IS A SEPARATE STEP FROM THE RANKING
--------------------------------------------
The ranking audit (tools/ship_candidate.py) answers "does the score order
these events". It does not answer "so what do I buy", and the two answers turn
out to be different. The ensemble's deciles rise monotonically from decile 0
to decile 8 and then FALL BACK at decile 9 -- the very top of the ranking is
not the best place to be, in 5 of 7 out-of-sample years. Sorting descending
and taking the first N, the obvious thing to do and what the screener does
today, therefore lands on the wrong rows.

So the band is chosen here, on evidence, and reported as a percentile window
rather than a top-N cut.

TWO YARDSTICKS, ALWAYS
----------------------
Insider clusters live in microcaps, and over this window the size factor alone
cost about 6 percentage points a year: SPY compounded at +15.2%, IWM at
+9.1%. Measured only against SPY, a perfectly good small-cap selection looks
like a failure; measured only against IWM, an index fund's return looks like
skill. Reporting one and not the other is benchmark shopping, so both are
printed side by side for every band, and neither is called "the" result.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools import data_paths  # noqa: E402

from scipy import stats  # noqa: E402

HORIZON = 21
BENCHMARKS = ("SPY", "IWM", "IWC")

#: Percentile windows put to the test. Deliberately includes the top-N-style
#: cuts the screener uses today (90-100, 95-100) so the comparison is direct,
#: and includes the whole population as the do-nothing control.
BANDS = [
    (0.0, 1.0, "all clusters"),
    (0.0, 0.10, "bottom decile"),
    (0.0, 0.50, "bottom half"),
    (0.50, 1.0, "top half"),
    (0.60, 0.95, "60-95"),
    (0.70, 0.95, "70-95"),
    (0.70, 0.90, "70-90"),
    (0.75, 0.95, "75-95"),
    (0.80, 0.95, "80-95"),
    (0.80, 0.90, "80-90"),
    (0.90, 1.0, "90-100 (top decile)"),
    (0.95, 1.0, "95-100"),
]


def benchmark_forward(ticker: str, horizon: int) -> pd.Series:
    """Open-to-open forward return over `horizon` trading days, by entry date.

    Indexed on the date the position is OPENED, so it lines up directly with
    an event's `entry_day` without any calendar arithmetic on the caller's
    side.
    """
    path = os.path.join(REPO_ROOT, "price_cache", f"{ticker}.parquet")
    px = pd.read_parquet(path)["open"].astype(float)
    fwd = px.shift(-horizon) / px - 1.0
    fwd.index = pd.to_datetime(fwd.index)
    return fwd.dropna()


def attach_benchmarks(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Add one forward-return column per benchmark, aligned on `entry_day`."""
    out = df.copy()
    entry = pd.to_datetime(out["entry_day"])
    for b in BENCHMARKS:
        fwd = benchmark_forward(b, horizon)
        out[f"bench_{b}"] = entry.map(fwd)
    return out


def band_periods(
    df: pd.DataFrame,
    score: str,
    lo: float,
    hi: float,
    *,
    horizon: int = HORIZON,
    cost_bps: float = 20.0,
    min_names: int = 3,
) -> pd.DataFrame:
    """One row per non-overlapping period: what the band earned, and the
    benchmarks over the identical days.

    The band is re-cut WITHIN each period, not once globally. That is the only
    version a live screener can reproduce: on any given day you can rank the
    candidates you can see, and you cannot know where they will sit in a
    distribution that includes the next four years of filings.
    """
    need = [score, "entry_idx", "entry_day", f"fwd_{horizon}"] + [
        f"bench_{b}" for b in BENCHMARKS
    ]
    sub = df[need].dropna().copy()
    if sub.empty:
        return pd.DataFrame()
    base = int(sub["entry_idx"].min())
    sub["period"] = (sub["entry_idx"] - base) // horizon
    cost = cost_bps / 10_000.0

    rows = []
    for p, g in sub.groupby("period"):
        if len(g) < 10:  # too few candidates to cut a percentile band from
            continue
        r = g[score].rank(pct=True)
        pick = g[(r > lo) & (r <= hi)]
        if len(pick) < min_names:
            continue
        row = dict(
            period=int(p),
            year=int(pd.to_datetime(g["entry_day"]).dt.year.median()),
            n_candidates=len(g),
            n_held=len(pick),
            ret=float(pick[f"fwd_{horizon}"].mean()) - cost,
        )
        for b in BENCHMARKS:
            row[f"bench_{b}"] = float(g[f"bench_{b}"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def summarise(periods: pd.DataFrame, label: str) -> dict:
    """Annualised excess over each benchmark, with a p-value and a year count."""
    if len(periods) < 3:
        return {}
    ppy = 252.0 / HORIZON
    out = dict(
        band=label,
        periods=len(periods),
        avg_held=round(float(periods["n_held"].mean()), 1),
        ann_return=float((1 + periods["ret"].mean()) ** ppy - 1),
    )
    for b in BENCHMARKS:
        ex = periods["ret"] - periods[f"bench_{b}"]
        per_year = periods.assign(ex=ex).groupby("year")["ex"].mean()
        out[f"vs_{b}"] = float((1 + ex.mean()) ** ppy - 1)
        out[f"p_{b}"] = float(stats.ttest_1samp(ex, 0.0).pvalue)
        out[f"yrs_{b}"] = f"{int((per_year > 0).sum())}/{len(per_year)}"
    out["worst"] = float(periods["ret"].min())
    out["stdev"] = float(periods["ret"].std(ddof=1))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", required=True, help="parquet with an `ens` column")
    ap.add_argument("--score-col", default="ens")
    ap.add_argument("--dataset", default=data_paths.latest_research_dataset())
    args = ap.parse_args(argv)

    scored = pd.read_parquet(args.scores)
    scored["event_day"] = pd.to_datetime(scored["event_day"])
    base = pd.read_parquet(args.dataset)
    base["event_day"] = pd.to_datetime(base["event_day"])
    keep = ["ticker", "event_day", "entry_day", f"fwd_{HORIZON}"]
    df = scored.merge(base[keep], on=["ticker", "event_day"], how="left")
    df = attach_benchmarks(df, HORIZON)

    print(f"{len(df)} rows; benchmarks attached for "
          + ", ".join(f"{b}:{df[f'bench_{b}'].notna().sum()}" for b in BENCHMARKS))
    print(f"\nEqual-weight, re-cut every {HORIZON} trading days, "
          f"non-overlapping, net of 20bps round trip.\n")

    rows = []
    for lo, hi, label in BANDS:
        per = band_periods(df, args.score_col, lo, hi)
        s = summarise(per, label)
        if s:
            rows.append(s)
    table = pd.DataFrame(rows)
    with pd.option_context("display.width", 220, "display.max_columns", 40):
        print(table.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
