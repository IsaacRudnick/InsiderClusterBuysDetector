"""A less pessimistic, more honest execution model -- replaces
tools/tradeable_universe.py's `tradeable_mask` and its flat round-trip cost.

WHY THIS REPLACES tradeable_mask
=================================
`tradeable_mask` (tools/tradeable_universe.py) required a position of
`capital / n_names` to fit inside 10% of a SINGLE day's dollar volume, and
every book, cheap or expensive, thin or deep, was charged the same flat
round-trip cost (20-75bps depending on the scenario). Both choices were
"the right way to be wrong" in the sense that module's own docstring
claimed -- conservative on purpose -- but conservative-on-purpose is not the
same as CORRECT, and the resulting capacity table ($1M book: Sharpe 0.982,
$25M: 0.750, both below SPY) was the deciding evidence that the strategy was
untradeable. Two things about that were wrong in ways that matter:

  1. LIQUIDITY. A real order is worked over several days, not dumped in one.
     Participation-rate execution (VWAP-style) spread over `days_to_fill`
     days gets `days_to_fill` times the capacity of a one-day fill for the
     same participation rate -- that is arithmetic, not an assumption, and
     the old one-day version understated capacity by exactly that factor.

  2. COST. A flat 20-75bps either overcharges a $30/$20M-ADV name (which
     trades for a few bps) or undercharges a $2/$300k-ADV name (which can
     cost several hundred bps), and the strategy's edge concentrates
     precisely in the cheap, thin names a flat cost misprices worst. A flat
     number cannot get both right, so it was always going to bias the
     conclusion -- in whichever direction the flat number happened to sit
     relative to the true, row-varying cost.

This module fixes both without assuming the answer. If the strategy still
does not clear the bar under an honestly generous execution model, that is
the correct conclusion and this module will say so. If it does clear the
bar, that is the correct conclusion too. Nothing here is tuned to make either
outcome happen -- the cost constants are calibrated to two independent,
literature-typical reference points (see `estimated_cost_bps`) BEFORE any
capacity table was run, not fit to produce a target Sharpe.

CAPACITY: capital / n_names inside participation x ADV x days_to_fill
=======================================================================
`capacity_mask` keeps a row only if a position of `capital / n_names` fits
inside `participation` of the name's 20-day average dollar volume, spread
over `days_to_fill` trading days:

    (participation * ADV * days_to_fill) >= capital / n_names

`participation=0.10, days_to_fill=1` reproduces the old one-day mask
exactly. `days_to_fill=5` -- a full trading week to work one entry, which is
unremarkable for a systematic strategy rebalancing monthly (HORIZON=21
trading days) -- gives 5x the capacity for the same participation rate.

COST: a per-row estimate, not a flat number
============================================
`estimated_cost_bps` prices every row individually from what is already on
disk (`entry_open`, `x_log_adv20`) plus an assumed position size, using the
standard decomposition of a round-trip trading cost into a spread
component (paid regardless of size) and a market-impact component (paid
because of size):

    spread_bps  = clamp(a/sqrt(price) + b/sqrt(adv_millions), 5, 400)
    impact_bps  = c * sqrt(position_size / adv)
    round_trip  = spread_bps + 2 * impact_bps

The spread term widens as price falls (low-price names quote wider relative
spreads -- a one-cent tick is a much bigger fraction of a $2 stock than a
$30 one) and as volume falls (thin names quote wider spreads, full stop).
The impact term is the textbook square-root model (Almgren & Chriss 2000;
the "square-root law" rule of thumb in Grinold & Kahn's *Active Portfolio
Management*: cost ~ Y * sigma * sqrt(order size / ADV)) -- cost grows with
the square root of how much of the day's volume the order represents, not
linearly, because a patient participation-rate execution absorbs liquidity
faster than it refills only up to a point. The constant `c` here bundles the
model's `Y * sigma`: microcaps carry daily volatility several times a
large-cap's, so a single constant calibrated on microcap names is not
directly comparable to a large-cap trading desk's impact coefficient, and is
not meant to be -- it is calibrated on THIS universe, not imported from one.

CALIBRATION (this is the whole prior, stated so it can be checked)
--------------------------------------------------------------------
Two reference points, chosen before any capacity table was run:

  * a $30 stock, $20M ADV, $50k position (a $1M book split 20 ways) should
    round-trip for roughly 15-30bps -- the middle of what a liquid small/mid
    cap name actually costs a modestly-sized order.
  * a $2 stock, $300k ADV, the same $50k position should round-trip for
    roughly 200-400bps -- expensive, because a $50k clip against $300k of
    daily volume is a lot of a thin name's day, not because $2 is special.

With SPREAD_A=4.0, SPREAD_B=20.0, IMPACT_C=200.0, floor=5, cap=400bps:

    (30,  20e6, 50_000)  -> spread  5.2bps, impact 10.0bps, round trip  25.2bps
    (2,  300e3, 50_000)  -> spread 39.3bps, impact 81.6bps, round trip 202.6bps

Both land inside their target bands. This is an ESTIMATE built from two
fields on disk, not a fill simulator or a TAQ-calibrated model -- there is
no bid-ask or trade-print data in this dataset to calibrate against
directly. Treat every number this module reports as "priced under a
standard, cited cost model with stated constants", not as "this is what a
broker would actually charge." The distribution the CLI prints exists so
that claim can be checked against intuition rather than trusted blind.
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

from tools import sharpe_lab as sh  # noqa: E402

#: Assumed number of names the book holds, for sizing a per-name position as
#: `capital / n_names`. 20 matches the name count `tradeable_universe.py`
#: used, so results are comparable to the numbers being redone here.
DEFAULT_N_NAMES = 20.0

#: Fraction of a day's dollar volume a working order is willing to represent,
#: per day of the fill. 10% is the standard conservative participation rate
#: used throughout this repo's prior liquidity work.
DEFAULT_PARTICIPATION = 0.10

#: How many trading days the order is worked over. 1 reproduces the OLD,
#: overly-pessimistic one-day mask exactly; this module's default is 5 (one
#: trading week), which is unremarkable for a strategy that only rebalances
#: every 21 trading days (HORIZON in sharpe_lab.py).
DEFAULT_DAYS_TO_FILL = 5.0

#: Spread model: spread_bps = clamp(A/sqrt(price) + B/sqrt(adv_millions), ...)
SPREAD_A = 4.0
SPREAD_B = 20.0
SPREAD_FLOOR_BPS = 5.0
SPREAD_CAP_BPS = 400.0

#: Impact model: impact_bps = C * sqrt(position_size / adv)
IMPACT_C = 200.0

#: Reference position size used when no explicit size is given to
#: `estimated_cost_bps` -- a $1M book split 20 ways, the same assumption
#: baked into DEFAULT_N_NAMES above. Only matters for callers that want a
#: standalone cost estimate without specifying a capital level.
DEFAULT_POSITION_SIZE = 50_000.0


# ---------------------------------------------------------------------------
# A. Capacity: participation x ADV x days, not one day
# ---------------------------------------------------------------------------

def capacity_mask(
    df: pd.DataFrame,
    capital: float,
    n_names: float = DEFAULT_N_NAMES,
    participation: float = DEFAULT_PARTICIPATION,
    days_to_fill: float = DEFAULT_DAYS_TO_FILL,
) -> pd.Series:
    """Rows a book of `capital` could fill within `days_to_fill` trading days.

    A position of `capital / n_names` must fit inside `participation` of the
    name's 20-day average dollar volume, spread over `days_to_fill` days:

        participation * ADV * days_to_fill  >=  capital / n_names

    This is the SAME crude, conservative shape as the old `tradeable_mask` --
    same participation rate, same fixed `n_names` assumption -- with one
    change: the fill is allowed to take `days_to_fill` days instead of being
    forced into one. Capacity is therefore exactly `days_to_fill` times the
    old mask's capacity at the same participation rate; `days_to_fill=1`
    reproduces the old mask's liquidity condition exactly (it drops the old
    mask's separate price floor, which the per-row cost model below now
    prices directly instead of gating on with an arbitrary cutoff).

    ADV comes from `x_log_adv20`, which is stored log1p-transformed, so it is
    inverted with `np.expm1` before use.
    """
    adv = np.expm1(pd.to_numeric(df["x_log_adv20"], errors="coerce"))
    need = capital / n_names
    capacity = adv * participation * days_to_fill
    return (capacity >= need).fillna(False)


# ---------------------------------------------------------------------------
# B. Cost: a per-row estimate, not a flat number
# ---------------------------------------------------------------------------

def spread_bps(price: np.ndarray, adv: np.ndarray) -> np.ndarray:
    """Half-spread-proxy component: wider for cheap, thin names.

    `price` in dollars, `adv` in dollars (NOT millions -- converted here).
    Clamped to [SPREAD_FLOOR_BPS, SPREAD_CAP_BPS] because the raw formula is
    unbounded in both directions: it explodes as price or volume approaches
    zero, and it can go negative for large, liquid names, which a spread
    never does.
    """
    price = np.maximum(np.asarray(price, dtype=float), 1e-4)
    adv_millions = np.maximum(np.asarray(adv, dtype=float), 1.0) / 1e6
    raw = SPREAD_A / np.sqrt(price) + SPREAD_B / np.sqrt(adv_millions)
    return np.clip(raw, SPREAD_FLOOR_BPS, SPREAD_CAP_BPS)


def market_impact_bps(position_size, adv: np.ndarray) -> np.ndarray:
    """Square-root impact component: bigger for bigger orders in thinner names.

    `position_size` and `adv` both in dollars. `position_size` may be a
    scalar (one assumed clip size for every row) or an array (a different
    clip per row, e.g. `capital / names_actually_held_that_period`).
    """
    adv = np.maximum(np.asarray(adv, dtype=float), 1.0)
    pos = np.asarray(position_size, dtype=float)
    return IMPACT_C * np.sqrt(np.maximum(pos, 0.0) / adv)


def estimated_cost_bps(
    df: pd.DataFrame, position_size=None
) -> np.ndarray:
    """Per-row estimated ROUND-TRIP cost in bps, from price and ADV alone.

    `position_size` is the dollar size of the position being priced -- a
    scalar (same assumed clip for every row) or an array/Series aligned to
    `df` (e.g. a different clip per row because `capital` is split across a
    different number of names each period). Defaults to
    `DEFAULT_POSITION_SIZE` ($50k, a $1M book split 20 ways) when not given,
    for a standalone cost estimate that is not tied to any one capital level.

    round_trip = spread_bps + 2 * impact_bps

    The spread is charged once (entering AND exiting a name cross roughly
    half the spread each, summing to the full quoted spread over the round
    trip); impact is charged once per side, so twice for a round trip. See
    the module docstring for the calibration targets and citations.
    """
    price = pd.to_numeric(df["entry_open"], errors="coerce").to_numpy(dtype=float)
    adv = np.expm1(pd.to_numeric(df["x_log_adv20"], errors="coerce")).to_numpy(dtype=float)
    pos = DEFAULT_POSITION_SIZE if position_size is None else position_size
    sp = spread_bps(price, adv)
    im = market_impact_bps(pos, adv)
    cost = sp + 2.0 * im
    # A row with no usable price or ADV gets no cost estimate at all (NaN)
    # rather than a fabricated one -- callers must decide how to handle that
    # (the CLI below drops such rows, same as every other stage here already
    # requires a finite price and ADV to trade a name at all).
    bad = ~(np.isfinite(price) & np.isfinite(adv) & (price > 0) & (adv > 0))
    cost = np.where(bad, np.nan, cost)
    return cost


# ---------------------------------------------------------------------------
# C. CLI: rebuild the capacity table with real fills and real costs
# ---------------------------------------------------------------------------

CAPITAL_LEVELS = [100_000.0, 250_000.0, 1_000_000.0, 5_000_000.0, 25_000_000.0]
DAYS_TO_FILL_LEVELS = [1.0, 3.0, 5.0, 10.0]


def _fmt_money(x: float) -> str:
    if x >= 1_000_000:
        return f"${x / 1_000_000:g}M"
    return f"${x / 1_000:g}k"


def build_universe(scores: pd.DataFrame, dataset: str, score_col: str) -> pd.DataFrame:
    """Prepared frame (via sharpe_lab.prepare) with `x_log_adv20` attached.

    `sharpe_lab.prepare` does not carry `x_log_adv20` -- it was built for a
    world without a capacity model -- so it is merged back in here from the
    same dataset, on the same (ticker, event_day) key `prepare` itself uses.
    """
    prepared = sh.prepare(scores, dataset, score_cols=(score_col,))
    base = pd.read_parquet(dataset, columns=["ticker", "event_day", "x_log_adv20"])
    base["event_day"] = pd.to_datetime(base["event_day"])
    prepared = prepared.merge(base, on=["ticker", "event_day"], how="left")
    return prepared


def capacity_table(
    df: pd.DataFrame,
    score_col: str,
    *,
    capitals=CAPITAL_LEVELS,
    days_levels=DAYS_TO_FILL_LEVELS,
    n_names: float = DEFAULT_N_NAMES,
    participation: float = DEFAULT_PARTICIPATION,
    lo: float = 0.70,
    hi: float = 0.90,
) -> pd.DataFrame:
    """One row per (capital, days_to_fill): the 70-90 book under real fills.

    For each capital level the per-row cost is recomputed at that level's
    implied position size (`capital / n_names`) -- a $25M book trades bigger
    clips than a $100k book, so it pays more impact, and charging every
    capital level the same cost would understate exactly the effect being
    measured.
    """
    rows = []
    for capital in capitals:
        pos = capital / n_names
        df = df.copy()
        df["_cost_bps"] = estimated_cost_bps(df, position_size=pos)
        for days in days_levels:
            mask = capacity_mask(
                df, capital=capital, n_names=n_names,
                participation=participation, days_to_fill=days,
            )
            sub = df[mask & df["_cost_bps"].notna()]
            if len(sub) < 200:
                rows.append(dict(capital=capital, days_to_fill=days,
                                  rows=len(sub), note="too few rows"))
                continue
            panel = sh.build_panel(sub, score_col, cost_col="_cost_bps")
            spec = sh.BookSpec(score_col=score_col, lo=lo, hi=hi,
                               weighting="equal", use_per_row_cost=True)
            per = sh.panel_returns(panel, spec)
            if len(per) < 5:
                rows.append(dict(capital=capital, days_to_fill=days,
                                  rows=len(sub), note="too few periods held"))
                continue
            m = sh.evaluate(per)
            spy = sh.benchmark_row(per, "SPY")
            rows.append(dict(
                capital=capital, days_to_fill=days, rows=len(sub),
                avg_names=m["avg_names"], ann_return=m["ann_return"],
                ann_vol=m["ann_vol"], sharpe=m["sharpe"],
                max_drawdown=m["max_drawdown"], excess_SPY=m["excess_SPY"],
                spy_sharpe=spy["sharpe"],
            ))
    return pd.DataFrame(rows)


def cost_distribution(
    df: pd.DataFrame, score_col: str, *, capital: float, n_names: float,
    lo: float = 0.70, hi: float = 0.90,
) -> pd.Series:
    """Per-row estimated cost for events ACTUALLY HELD in the lo-hi band.

    This is not "every row in the dataset" -- it is the subset the book in
    question actually would have bought, at one representative capital
    level, so the distribution answers "what did the strategy actually pay"
    rather than "what would some unrelated name have cost".
    """
    pos = capital / n_names
    work = df.copy()
    work["_cost_bps"] = estimated_cost_bps(work, position_size=pos)
    held = []
    for _, g in work.groupby("period"):
        if len(g) < 10:
            continue
        r = g[score_col].rank(pct=True)
        pick = g[(r > lo) & (r <= hi)]
        held.append(pick["_cost_bps"])
    if not held:
        return pd.Series(dtype=float)
    return pd.concat(held).dropna()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--score-col", default="ens")
    ap.add_argument("--dataset", default=data_paths.latest_research_dataset())
    ap.add_argument("--n-names", type=float, default=DEFAULT_N_NAMES)
    ap.add_argument("--participation", type=float, default=DEFAULT_PARTICIPATION)
    args = ap.parse_args(argv)

    scores = pd.read_parquet(args.scores)
    df = build_universe(scores, args.dataset, args.score_col)
    print(f"{len(df)} scored rows, {df['period'].nunique()} periods\n")

    table = capacity_table(df, args.score_col, n_names=args.n_names,
                           participation=args.participation)
    print("===== CAPACITY TABLE: real (multi-day) fills, per-row costs =====")
    print(f"(70-90 band, equal weight, n_names assumption={args.n_names:g}, "
          f"participation={args.participation:.0%})\n")
    show = table.copy()
    show["capital"] = show["capital"].map(_fmt_money)
    with pd.option_context("display.width", 260, "display.max_columns", 40):
        print(show.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))

    print("\n===== COST DISTRIBUTION: events actually held in the 70-90 band =====")
    print("(one representative capital level per row; position size = capital / n_names)\n")
    for capital in (250_000.0, 1_000_000.0, 5_000_000.0):
        c = cost_distribution(df, args.score_col, capital=capital, n_names=args.n_names)
        if len(c) == 0:
            print(f"  {_fmt_money(capital)}: no held rows")
            continue
        q = c.quantile([0.25, 0.5, 0.75])
        frac_over_100 = float((c > 100.0).mean())
        print(f"  {_fmt_money(capital)} book (n={len(c)}): "
              f"median {q[0.5]:.0f}bps, IQR [{q[0.25]:.0f}, {q[0.75]:.0f}]bps, "
              f"{frac_over_100:.0%} exceed 100bps")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
