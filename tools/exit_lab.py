"""Day-by-day trade simulation: what if you stop selling the winners?

THE ERROR THIS EXISTS TO FIX
============================
Every risk-adjusted result measured in this repo so far rebalances the whole
book on a fixed 21-trading-day clock. On a return distribution whose entire
payoff is a fat right tail, that is close to the worst possible exit rule: it
sells the one position in fifty that was going to triple, on a calendar date
chosen for bookkeeping convenience, and it keeps the losers for the full 21
days.

The classic way to harvest a skewed payoff is asymmetric: cut the losers
quickly on a hard stop, and let the winners run behind a trailing stop until
the trend actually breaks. That has never been tested here. Fixed-hold
backtests structurally cannot express it, because a fixed hold has no memory
of what the position did after entry.

So this module simulates each trade DAY BY DAY against the real daily bars in
`price_cache/`, and applies exit rules that depend on the path.

HOW A TRADE IS SIMULATED, AND THE CONSERVATIVE CHOICES IN IT
------------------------------------------------------------
Entry is at the OPEN of `entry_day` -- the same entry every other result here
uses, so exits are the only thing that changes.

Each subsequent day, in this order:
  1. If the day's LOW breaches the hard stop, exit AT THE STOP PRICE. Using
     the low (not the close) means an intraday spike down stops you out, which
     is what would really happen with a resting stop order.
  2. Else if the day's LOW breaches the trailing stop, exit at the trailing
     stop price. The trail is measured from the highest CLOSE seen so far, not
     the highest high -- ratcheting off intraday highs would let a single
     spurious print set a stop level that never really existed.
  3. Else update the high-water mark from today's CLOSE.
  4. If the maximum holding period is reached, exit at the next OPEN.

Gaps are handled the pessimistic way: if a day OPENS below the stop, the fill
is the open, not the stop. A stop does not protect you through a gap and
pretending otherwise is how backtests invent money.

WHAT THIS MODULE DELIBERATELY DOES NOT DO
-----------------------------------------
It does not search for the best stop pair and report it as a result. The exit
grid is swept, the WHOLE table is printed, and the winner is then handed to
the same permutation machinery everything else here goes through. A
best-of-grid exit search on a fat-tailed distribution manufactures impressive
numbers exactly the way the band search did.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

PRICE_DIR = os.path.join(REPO_ROOT, "price_cache")

#: Loaded price frames, keyed by ticker. The sweep touches the same few
#: thousand tickers many times over; re-reading a parquet each time dominates
#: the runtime and changes nothing.
_PX_CACHE: dict[str, Optional[pd.DataFrame]] = {}


def load_prices(ticker: str) -> Optional[pd.DataFrame]:
    if ticker in _PX_CACHE:
        return _PX_CACHE[ticker]
    path = os.path.join(PRICE_DIR, f"{ticker}.parquet")
    if not os.path.exists(path):
        _PX_CACHE[ticker] = None
        return None
    df = pd.read_parquet(path)[["open", "high", "low", "close"]].astype(float)
    df.index = pd.to_datetime(df.index)
    _PX_CACHE[ticker] = df
    return df


@dataclass(frozen=True)
class ExitRule:
    """One exit policy.

    hard_stop     fraction below ENTRY that force-closes the trade (0 = none)
    trail_stop    fraction below the highest close since entry (0 = none)
    max_hold      trading days after which the trade closes regardless
    activate_at   the trail only arms once the position is up by this much.
                  Without it a tight trail behaves like a very tight stop on
                  day one and cuts every trade before it can develop.
    """

    hard_stop: float = 0.0
    trail_stop: float = 0.0
    max_hold: int = 21
    activate_at: float = 0.0

    def label(self) -> str:
        parts = []
        parts.append(f"hard{self.hard_stop:.0%}" if self.hard_stop else "hard-")
        if self.trail_stop:
            a = f"@{self.activate_at:.0%}" if self.activate_at else ""
            parts.append(f"trail{self.trail_stop:.0%}{a}")
        else:
            parts.append("trail-")
        parts.append(f"max{self.max_hold}")
        return " ".join(parts)


@dataclass(frozen=True)
class TradeResult:
    ret: float          # gross return, entry open to exit fill
    days_held: int      # trading days the capital was committed
    exit_reason: str


def simulate_trade(
    px: pd.DataFrame, entry_day: pd.Timestamp, rule: ExitRule
) -> Optional[TradeResult]:
    """Walk one trade forward bar by bar. None if it cannot be priced."""
    idx = px.index.searchsorted(entry_day)
    if idx >= len(px):
        return None
    entry = float(px["open"].iloc[idx])
    if not np.isfinite(entry) or entry <= 0:
        return None

    hard_level = entry * (1.0 - rule.hard_stop) if rule.hard_stop else None
    peak_close = entry
    last = min(idx + rule.max_hold, len(px) - 1)

    for j in range(idx + 1, last + 1):
        o = float(px["open"].iloc[j])
        lo = float(px["low"].iloc[j])
        c = float(px["close"].iloc[j])
        if not np.isfinite(o) or not np.isfinite(lo) or not np.isfinite(c):
            continue

        trail_level = None
        if rule.trail_stop and peak_close >= entry * (1.0 + rule.activate_at):
            trail_level = peak_close * (1.0 - rule.trail_stop)

        # The higher of the two active stops is the one that binds first.
        level = max([x for x in (hard_level, trail_level) if x is not None],
                    default=None)
        if level is not None and lo <= level:
            # A gap through the stop fills at the open, not at the stop. A
            # resting order cannot execute at a price the market never traded.
            fill = min(o, level) if o < level else level
            return TradeResult(fill / entry - 1.0, j - idx,
                               "hard" if (hard_level is not None
                                          and level == hard_level) else "trail")
        peak_close = max(peak_close, c)

    exit_px = float(px["open"].iloc[last]) if last > idx else float(px["close"].iloc[idx])
    if not np.isfinite(exit_px):
        exit_px = float(px["close"].iloc[last])
    return TradeResult(exit_px / entry - 1.0, last - idx, "time")


@dataclass(frozen=True)
class PathSet:
    """Every trade's forward price path, extracted once as flat matrices.

    Rows are trades, columns are trading days after entry. Building this once
    and then evaluating each exit rule as a whole-matrix operation is what
    makes a real exit sweep affordable -- the per-trade Python loop that
    `simulate_trade` uses is correct but roughly a hundred times too slow to
    run ninety rules over a few thousand trades, and a sweep nobody can afford
    to run is a sweep that silently does not get run.

    `simulate_trade` is kept as the readable reference implementation, and
    `tests/test_exit_lab.py` asserts the two agree trade for trade.
    """

    entry: np.ndarray      # (n,)      entry price
    open_: np.ndarray      # (n, days) open on each subsequent day
    low: np.ndarray        # (n, days)
    close: np.ndarray      # (n, days)
    valid: np.ndarray      # (n, days) bar exists and is finite
    meta: pd.DataFrame     # one row per trade, aligned to the matrices


def build_paths(events: pd.DataFrame, max_days: int) -> PathSet:
    """Extract `max_days` of forward bars for every priceable event."""
    entries, opens, lows, closes, valids, keep = [], [], [], [], [], []
    for t in events.itertuples(index=False):
        px = load_prices(t.ticker)
        if px is None:
            continue
        i = px.index.searchsorted(pd.Timestamp(t.entry_day))
        if i >= len(px):
            continue
        e = float(px["open"].iloc[i])
        if not np.isfinite(e) or e <= 0:
            continue
        sl = slice(i + 1, i + 1 + max_days)
        o = px["open"].to_numpy()[sl]
        lo = px["low"].to_numpy()[sl]
        c = px["close"].to_numpy()[sl]
        pad = max_days - len(o)
        if pad > 0:
            # Short paths (a recent event, or a delisting) are padded with NaN
            # and masked out rather than forward-filled. Forward-filling a
            # delisted name's last price would let a dead position sit at par
            # forever, which is the single most flattering bug available here.
            o = np.concatenate([o, np.full(pad, np.nan)])
            lo = np.concatenate([lo, np.full(pad, np.nan)])
            c = np.concatenate([c, np.full(pad, np.nan)])
        entries.append(e)
        opens.append(o)
        lows.append(lo)
        closes.append(c)
        valids.append(np.isfinite(o) & np.isfinite(lo) & np.isfinite(c))
        keep.append(t)
    if not entries:
        return PathSet(np.array([]), np.zeros((0, max_days)),
                       np.zeros((0, max_days)), np.zeros((0, max_days)),
                       np.zeros((0, max_days), bool), pd.DataFrame())
    return PathSet(
        np.asarray(entries, dtype=float),
        np.vstack(opens), np.vstack(lows), np.vstack(closes),
        np.vstack(valids), pd.DataFrame(keep),
    )


def simulate_paths(
    paths: PathSet, rule: ExitRule, *, cost_bps: float = 20.0
) -> pd.DataFrame:
    """Apply one exit rule to every path at once. Same semantics as
    `simulate_trade`, expressed as array operations."""
    if not len(paths.entry):
        return pd.DataFrame()
    n = len(paths.entry)
    d = min(rule.max_hold, paths.open_.shape[1])
    entry = paths.entry[:, None]
    o = paths.open_[:, :d]
    lo = paths.low[:, :d]
    c = paths.close[:, :d]
    ok = paths.valid[:, :d]

    # Peak close STRICTLY BEFORE each day: the trail can only ratchet on
    # information already closed, so today's close cannot raise the stop that
    # today's low is tested against.
    filled = np.where(ok, c, -np.inf)
    running = np.maximum.accumulate(filled, axis=1)
    peak_before = np.concatenate(
        [np.full((n, 1), -np.inf), running[:, :-1]], axis=1
    )
    peak_before = np.maximum(peak_before, entry)

    hard_level = entry * (1.0 - rule.hard_stop) if rule.hard_stop else None
    if rule.trail_stop:
        armed = peak_before >= entry * (1.0 + rule.activate_at)
        trail_level = np.where(armed, peak_before * (1.0 - rule.trail_stop),
                               -np.inf)
    else:
        trail_level = np.full((n, d), -np.inf)
    level = trail_level if hard_level is None else np.maximum(
        trail_level, np.broadcast_to(hard_level, (n, d))
    )

    breach = ok & (lo <= level)
    any_breach = breach.any(axis=1)
    first = np.where(any_breach, breach.argmax(axis=1), d - 1)
    rows = np.arange(n)

    lvl = level[rows, first]
    op = o[rows, first]
    # A gap through the stop fills at the open. A resting order cannot execute
    # at a price the market never traded through.
    stop_fill = np.where(op < lvl, op, lvl)

    # Time exits take the last VALID bar's open; a path that went dead early
    # exits at the last price that actually existed.
    last_valid = np.where(ok.any(axis=1), (d - 1) - ok[:, ::-1].argmax(axis=1), 0)
    time_fill = o[rows, last_valid]
    time_fill = np.where(np.isfinite(time_fill), time_fill, c[rows, last_valid])

    exit_px = np.where(any_breach, stop_fill, time_fill)
    days = np.where(any_breach, first + 1, last_valid + 1)
    reason = np.where(
        ~any_breach, "time",
        np.where(
            (hard_level is not None)
            & (np.abs(lvl - np.broadcast_to(
                hard_level if hard_level is not None else np.zeros((n, 1)),
                (n, d))[rows, first]) < 1e-12),
            "hard", "trail",
        ),
    )

    good = np.isfinite(exit_px) & (exit_px > 0)
    ret = exit_px / paths.entry - 1.0 - cost_bps / 10_000.0

    out = paths.meta.copy()
    out["ret"] = ret
    out["days_held"] = days
    out["exit_reason"] = reason
    out["year"] = pd.to_datetime(out["entry_day"]).dt.year
    out["score"] = out["score"].astype(float) if "score" in out.columns else np.nan
    return out[good].reset_index(drop=True)


def simulate_events(
    events: pd.DataFrame, rule: ExitRule, *, cost_bps: float = 20.0
) -> pd.DataFrame:
    """Reference implementation: run `rule` over every event, one at a time.

    Kept for clarity and as the thing `simulate_paths` is tested against. Use
    `build_paths` + `simulate_paths` for any real sweep.
    """
    cost = cost_bps / 10_000.0
    rows = []
    for t in events.itertuples(index=False):
        px = load_prices(t.ticker)
        if px is None:
            continue
        res = simulate_trade(px, pd.Timestamp(t.entry_day), rule)
        if res is None:
            continue
        rows.append(
            dict(
                ticker=t.ticker,
                entry_day=t.entry_day,
                entry_idx=int(t.entry_idx),
                year=pd.Timestamp(t.entry_day).year,
                score=float(getattr(t, "score")),
                ret=res.ret - cost,
                days_held=res.days_held,
                exit_reason=res.exit_reason,
            )
        )
    return pd.DataFrame(rows)


def daily_marked_portfolio(
    paths: "PathSet",
    trades: pd.DataFrame,
    *,
    n_slots: int = 20,
    cost_bps: float = 20.0,
) -> pd.DataFrame:
    """A slot-limited book marked to market EVERY DAY on real closing prices.

    WHY THIS REPLACED THE EARLIER VERSION. `slot_portfolio` below spreads each
    trade's total return evenly across the days it was held. That is fine for
    ordering rules by return per unit of capital-time and catastrophic for any
    risk number: smearing a lumpy return into a constant daily drip removes
    almost all of the volatility, and it produced Sharpe ratios near 10, which
    is not a good result, it is a broken measurement. Any Sharpe, drawdown or
    vol figure must come from here instead.

    The simulation:
      - positions are taken in score order as they arrive, never exceeding
        `n_slots`, first come first served -- the constraint a real account
        faces, and deliberately not one that peeks at future candidates;
      - each held position is marked daily on its own close-to-close return;
      - the book's daily return is the average across the slots, with EMPTY
        SLOTS EARNING ZERO. That last point is what makes the comparison to an
        index honest: a book that is only half invested does not get to
        annualise as though it were fully invested.
      - the round-trip cost is charged on the position's first day.
    """
    if trades.empty or not len(paths.entry):
        return pd.DataFrame()

    close = paths.close
    entry_px = paths.entry
    # trades carries the row position of each surviving path in `meta`
    pos_of = {(t.ticker, pd.Timestamp(t.entry_day)): i
              for i, t in enumerate(paths.meta.itertuples(index=False))}

    t = trades.sort_values(["entry_idx", "score"], ascending=[True, False])
    occupied: list[int] = []
    taken = []
    for row in t.itertuples(index=False):
        occupied = [e for e in occupied if e > row.entry_idx]
        if len(occupied) >= n_slots:
            continue
        occupied.append(row.entry_idx + max(int(row.days_held), 1))
        taken.append(row)
    if not taken:
        return pd.DataFrame()

    lo = min(int(r.entry_idx) for r in taken)
    hi = max(int(r.entry_idx) + int(r.days_held) for r in taken)
    n_days = hi - lo + 2
    acc = np.zeros(n_days)          # summed position returns per calendar day
    active = np.zeros(n_days)       # how many slots were live that day

    for row in taken:
        i = pos_of.get((row.ticker, pd.Timestamp(row.entry_day)))
        if i is None:
            continue
        d = int(row.days_held)
        path = close[i, :d]
        if not np.isfinite(path).any():
            continue
        # Day 1 is measured from the ENTRY price (an open), every later day
        # from the previous close. The final day's return is overridden with
        # the trade's realised total so a stop fill -- which happens intraday,
        # not at the close -- is what the book actually books.
        prev = np.concatenate([[entry_px[i]], path[:-1]])
        rets = np.where(np.isfinite(path) & np.isfinite(prev) & (prev > 0),
                        path / prev - 1.0, 0.0)
        realised = float(row.ret)
        implied = float(np.prod(1.0 + rets) - 1.0)
        if d >= 1 and np.isfinite(realised):
            adj = (1.0 + realised) / (1.0 + implied) - 1.0 if implied > -1 else 0.0
            rets[-1] = (1.0 + rets[-1]) * (1.0 + adj) - 1.0
        rets[0] -= cost_bps / 10_000.0
        s = int(row.entry_idx) - lo
        acc[s:s + d] += rets[:min(d, n_days - s)][: max(0, min(d, n_days - s))]
        active[s:s + d] += 1.0

    daily = acc / max(n_slots, 1)
    return pd.DataFrame(dict(day=np.arange(lo, lo + n_days), ret=daily,
                             n_active=active))


def slot_portfolio(
    trades: pd.DataFrame, *, n_slots: int = 20, rank_within_days: int = 5
) -> pd.DataFrame:
    """A capital-constrained book: at most `n_slots` positions at any time.

    Variable-length holds mean positions overlap irregularly, so a book has to
    decide what to do when more candidates arrive than there is capital for.
    This takes them in score order within each small batch of entry days and
    refuses any trade that would exceed the slot count -- the same
    first-come-first-served constraint a real account faces, and deliberately
    NOT the version that peeks ahead to pick the best of a future batch.

    Returns a daily equity curve in trading-day index space.
    """
    if trades.empty:
        return pd.DataFrame()
    t = trades.sort_values(["entry_idx", "score"],
                           ascending=[True, False]).reset_index(drop=True)
    t["batch"] = t["entry_idx"] // rank_within_days

    occupied_until: list[int] = []
    taken = []
    for row in t.itertuples(index=False):
        occupied_until = [e for e in occupied_until if e > row.entry_idx]
        if len(occupied_until) >= n_slots:
            continue
        occupied_until.append(row.entry_idx + max(row.days_held, 1))
        taken.append(row)
    if not taken:
        return pd.DataFrame()
    held = pd.DataFrame(taken)

    # Spread each trade's return evenly across the days it was held, then sum
    # across positions per day. This is an approximation of a daily-marked
    # book, and it is stated as one: it gets the timing of capital commitment
    # right without needing every position's full daily path in memory.
    lo, hi = int(held["entry_idx"].min()), int((held["entry_idx"] + held["days_held"]).max())
    daily = np.zeros(hi - lo + 2)
    for row in held.itertuples(index=False):
        d = max(int(row.days_held), 1)
        per_day = (1.0 + row.ret) ** (1.0 / d) - 1.0
        s = int(row.entry_idx) - lo
        daily[s:s + d] += per_day / n_slots
    return pd.DataFrame(dict(day=np.arange(lo, lo + len(daily)), ret=daily))


def summarise_trades(trades: pd.DataFrame, label: str) -> dict:
    """Per-trade statistics, plus return per unit of TIME COMMITTED.

    `ann_per_slot` is the number that makes variable-length exits comparable
    to fixed ones: a rule that earns 4% in 10 days used the capital for half as
    long as one that earns 5% in 21 days, and the second is worse. Comparing
    raw per-trade means across rules with different holding periods would be
    straightforwardly wrong.
    """
    if len(trades) < 50:
        return {}
    r = trades["ret"]
    d = trades["days_held"].clip(lower=1)
    per_day = np.log1p(r.clip(lower=-0.999)) / d
    return dict(
        rule=label,
        n=len(trades),
        median=float(r.median()),
        mean=float(r.mean()),
        win_rate=float((r > 0).mean()),
        avg_days=float(d.mean()),
        p_big_win=float((r > 0.50).mean()),
        p_crash=float((r < -0.30).mean()),
        worst=float(r.min()),
        best=float(r.max()),
        ann_per_slot=float(np.expm1(per_day.mean() * 252)),
        stopped_hard=float((trades["exit_reason"] == "hard").mean()),
        stopped_trail=float((trades["exit_reason"] == "trail").mean()),
    )
