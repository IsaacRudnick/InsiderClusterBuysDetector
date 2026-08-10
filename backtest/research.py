"""Rich per-event research dataset: one row per insider-cluster episode,
with cluster-shape / role / stake / point-in-time history / price-context
features (all prefixed x_) plus SPY-adjusted forward-return labels at
several horizons and the legacy hand-tuned flags for comparison.

This is a research artifact, not something the live engine consumes. It is
meant to be fed to a real model (lightgbm / sklearn) downstream to replace
insider_cluster_buys.DEFAULT_WEIGHTS' 22 hand-tuned binary flags.

Pipeline shape mirrors backtest/signal_fit.py's build_event_dataset:
  1. Walk the trading calendar day by day, ask DailyStateBuilder for the
     day's state, and detect "episode starts" (a ticker crossing from
     < 2 insiders to >= 2 insiders in its rolling window).
  2. For each episode start, find the first priceable entry day (next-open
     execution, same MAX_ENTRY_LOOKAHEAD rule as signal_fit) and reconstruct
     the exact raw transaction window state.py used internally (state.py
     does not expose it), then build one row of features + labels from it.
  3. build_research_dataset() returns the full DataFrame; save/load helpers
     persist it to parquet.

Reuses signal_fit._open_or_fallback / _find_entry_day directly rather than
reimplementing next-open execution semantics.
"""

from __future__ import annotations

import logging
import math
import os
import re
from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

import insider_cluster_buys as ics
from . import splits as splits_mod
from . import ticker_reuse
from .prices import PriceUniverse
from .signal_fit import MAX_ENTRY_LOOKAHEAD, _find_entry_day, _open_or_fallback
from .state import DailyStateBuilder

log = logging.getLogger(__name__)

DEFAULT_HORIZONS: tuple[int, ...] = (10, 21, 63, 126, 252)

# Section D's point-in-time gating rule is fixed at 63 trading days,
# independent of whatever `horizons` the caller asks for as output labels.
# We always compute this internally so the leakage rule below is well
# defined even if 63 is not in `horizons`.
HISTORY_HORIZON = 63

LOG_EVERY = 5000

# Mirrors insider_cluster_buys._component_flags's fractional_shares
# tolerance exactly, so this plain per-transaction fraction stays
# consistent with the binary flag it is a cousin of.
FRACTIONAL_SHARE_TOL = 1e-4


# ---------------------------------------------------------------------------
# Section D leakage note
# ---------------------------------------------------------------------------
# Owner- and issuer-level history features (x_owner_*, x_issuer_*) are the
# highest lookahead risk in this module: a naive "average this owner's past
# returns" feature can leak the future into the past two ways:
#
#   1. Same-day contamination. Two clusters that both start on day D must
#      not see each other, even though "D" is technically "the past" by the
#      time either row is materialized. We enforce this by staging each
#      day's owner/issuer/ticker history updates in local dicts and only
#      merging them into the shared running history AFTER every event for
#      that day has been built (see the day loop in build_research_dataset).
#
#   2. Using a prior event's own forward-return label before that label's
#      63-trading-day window has actually closed. A prior cluster's adj_63
#      is only "known" once its own exit day has passed. We gate every use
#      of a prior event's adj_63 with _label_window_closed(): the prior
#      event's exit day (its entry_idx + HISTORY_HORIZON trading days) must
#      fall strictly before the CURRENT event's event_day. Plain counts
#      (n_prior_buys, n_prior_clusters) don't touch a forward-looking value
#      so they only need condition 1, not this second gate.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Section F leakage note (x_sell_* / x_buy_sell_* / x_*_sold_* -- concurrent
# selling)
# ---------------------------------------------------------------------------
# These features summarize SALE transactions (Form 4/4A code S, disposed)
# filed for the SAME issuer around the same time as the cluster buy, built
# from a separately-cached issuer-keyed sale index (see backtest/
# sales_history.py, sales_df / sale_index below). The identical Form-4
# filing-lag problem that motivates Section D's discipline applies here, with
# an extra wrinkle: a sale's TRANSACTION date can sit comfortably inside the
# cluster's window while its FILING date lands well after event_day (SEC
# allows up to two business days, and late filings routinely take longer).
# Using transaction_date alone to decide visibility would leak a sale nobody
# could have known about yet at decision time.
#
# The rule, mirroring _window_for_ticker_day's buy-side visibility test
# exactly (same function is reused for both, see _prepare_sale_index below):
# a sale is visible to event (ticker, D) iff filing_date <= D AND
# transaction_date falls in [D - (window_days - 1), D]. filing_date gates
# VISIBILITY; transaction_date is only used for the economics (which window
# a visible sale's dollar value falls into).
#
# D (event_day), not entry_day, is the anchor for both the visibility cutoff
# and the window math. This matches Section D's owner/issuer history gate,
# not Section E's price-context helpers (which deliberately use entry_day).
# The two Section-E-style helpers reach forward to entry_day because that is
# the trade's own execution date -- information right up to the open that
# fills the order is fair game. Concurrent-selling features answer a
# different question ("was anyone dumping when this cluster was IDENTIFIED"),
# so anchoring them to entry_day would let them see up to MAX_ENTRY_LOOKAHEAD
# extra trading days of filings the cluster detector itself never had. Using
# D keeps the answer to exactly what was on the tape the moment the episode
# was flagged, which is the more conservative and more defensible choice.
#
# Unlike Section D, no same-day-contamination staging is needed: the sale
# index is a static, externally-cached fact table (every sale's filing_date
# is fixed at load time), not something this loop derives from the events it
# is currently processing. The filing_date <= D check applied fresh at each
# event's own D is sufficient on its own.
# ---------------------------------------------------------------------------

# Trailing baseline window (calendar days) for x_sell_*_trail90 -- roughly
# the same span as the 63-TRADING-day horizons used elsewhere in this module
# (x_drawdown_63, x_vol_63_ann, HISTORY_HORIZON), converted to calendar days.
# Wide enough to establish a stable baseline selling rate for the issuer
# without reaching back so far that it mixes in a stale ownership regime.
# Deliberately NOT exclusive of the (much shorter) cluster window below --
# x_sell_n_trail90 answers "how much selling has this issuer seen lately,
# period", and the cluster window is a subset of "lately".
TRAILING_SALE_WINDOW_DAYS = 90


# ---------------------------------------------------------------------------
# Split-adjustment wiring (see backtest/splits.py)
# ---------------------------------------------------------------------------
# Yahoo's split table is incomplete for many small tickers. The cached price
# series can hide an unadjusted split as a fake overnight jump. Left alone,
# this fabricates huge forward-return labels (see RESEARCH_NOTES.md,
# "Unadjusted splits corrupt the price cache"). build_research_dataset now
# detects each ticker's discontinuities ONCE (see _split_info below, cached
# per ticker for the whole build), and back-adjusts every confirmed "split".
# It drops any event row whose forward-return label window spans a jump that
# is not a confirmed split: an "ambiguous" jump, or a "real_move" past
# splits.REAL_MOVE_CEILING_RATIO. Dropping the row is correct there. A label
# built across a possibly-fabricated price is worse than a missing row.
#
# Future-split-information note. detect_discontinuities scans a ticker's
# WHOLE price history in one pass. A split dated after some EARLIER event's
# entry_day can still back-adjust bars before that entry_day. This looks
# like lookahead at first glance. It is not, for two reasons.
#
#   1. Every feature and label in this module is a RATIO (a forward return,
#      a drawdown, a momentum, price-to-SMA) or a DOLLAR VOLUME (close times
#      volume). Both stay the same under a uniform rescale applied to one
#      side of a split boundary. Two bars on the SAME side of a future split
#      boundary get the SAME scaling factor, so their ratio does not change,
#      no matter what that factor turns out to be. A future split can only
#      change a computed value when the window ITSELF crosses the split
#      boundary. That is exactly the case where the raw data is wrong and
#      the correction is wanted.
#   2. A split is public information from the moment it executes, not
#      before. A point-in-time price vendor retroactively rewrites its OWN
#      historical series the same day a split happens. Every later user of
#      that vendor sees the same rewritten history. Scanning the full
#      series once, then back-adjusting bars strictly before each split's
#      own date, reproduces that same behavior. It never lets a split dated
#      AFTER today change a ratio that depends only on bars up to today.
#
# apply_split_adjustments only ever rescales bars whose index is strictly
# before the split's own date. Point 2 above holds by construction.


# ---------------------------------------------------------------------------
# Ticker-reuse wiring (see ticker_reuse.py, root level)
# ---------------------------------------------------------------------------
# A Form 4 carries the issuer's ticker as of ITS OWN filing date, but prices
# are fetched from Yahoo for that ticker string TODAY. When a ticker has
# changed hands between two unrelated companies (the old occupant delisted,
# a new one later listed under the freed symbol), every event row is priced
# against PriceUniverse's ticker-keyed lookup with no idea which company's
# stock it is actually reading -- a delisted company's insider buys silently
# inherit whichever company holds the ticker now. ticker_reuse.py classifies
# every multi-issuer_cik ticker in events_df as REUSE (drop: an unrelated
# company later claimed the ticker), RENAME (keep: the same company under a
# new CIK, e.g. a bankruptcy emergence or holdco reorg), or AMBIGUOUS
# (treated as unsafe, so also dropped -- see that module's docstring for the
# evidence and thresholds behind each call).
#
# This runs ONCE, up front, on the whole events_df -- before ticker_index is
# built below -- so a dropped row never enters window reconstruction, and
# never pollutes owner_history/issuer_history/ticker_last_event either.
# drop_ticker_reuse=True by default (safe); pass False to keep every row
# unfiltered, e.g. to reproduce a pre-fix artifact exactly.
def _ticker_price_frame(prices: PriceUniverse, ticker: str) -> Optional[pd.DataFrame]:
    """The raw OHLCV frame detect_discontinuities needs, if `prices` exposes
    one. Test doubles (see tests/test_research.py's FakePriceUniverse) may
    carry no `.frames` attribute at all, or no entry for this ticker. Either
    way this returns None, and the caller treats the ticker as having no
    discontinuities, which keeps every existing caller byte-for-byte
    unchanged."""
    frames = getattr(prices, "frames", None)
    if frames is None:
        return None
    return frames.get(ticker)


def _apply_adjusted_prices(prices: PriceUniverse, ticker: str, adjusted: pd.DataFrame) -> None:
    """Push a split-back-adjusted per-ticker frame into PriceUniverse's flat
    lookup dicts (open_by_ticker / close_by_ticker / dv_by_ticker), which is
    what every price lookup in this module actually reads. Bar DATES never
    change under adjustment, only values, so dates_by_ticker needs no
    update. median_dollar_volume() memoizes by (ticker, date, window), so
    any cached entry for this ticker is stale the moment its prices change
    and must be dropped.
    """
    frames = getattr(prices, "frames", None)
    if frames is not None:
        frames[ticker] = adjusted
    d_index = [ts.date() for ts in adjusted.index]
    open_by = getattr(prices, "open_by_ticker", None)
    if open_by is not None:
        open_by[ticker] = dict(zip(d_index, adjusted["open"].to_list()))
    close_by = getattr(prices, "close_by_ticker", None)
    if close_by is not None:
        close_by[ticker] = dict(zip(d_index, adjusted["close"].to_list()))
    dv_by = getattr(prices, "dv_by_ticker", None)
    if dv_by is not None:
        dv_by[ticker] = dict(zip(d_index, adjusted["dollar_volume"].to_list()))
    mdv_cache = getattr(prices, "_mdv_cache", None)
    if mdv_cache is not None:
        for key in [k for k in mdv_cache if k[0] == ticker]:
            del mdv_cache[key]


@dataclass(frozen=True)
class _PriorEvent:
    """One entry in an owner's or issuer's running history."""
    event_day: date
    entry_idx: int
    adj63: float   # SPY-adjusted 63-day forward return, NaN if unknown


def _label_window_closed(
    prior_entry_idx: int, current_event_day: date, calendar: list[date],
    horizon: int = HISTORY_HORIZON,
) -> bool:
    """True iff a prior event's `horizon`-day label window has closed
    strictly before `current_event_day` (see the leakage note above)."""
    exit_idx = prior_entry_idx + horizon
    if exit_idx >= len(calendar):
        return False
    return calendar[exit_idx] < current_event_day


def _weighted_mean(pairs: list[tuple[float, float]]) -> float:
    """Value-weighted mean of (value, weight) pairs.

    NaN if `pairs` is empty (callers are responsible for excluding entries
    with no usable data rather than passing a 0/NaN placeholder — see each
    call site). Falls back to a plain mean if every weight is <= 0, so an
    all-zero-dollar-value cluster doesn't collapse silently to NaN.
    """
    if not pairs:
        return float("nan")
    wsum = sum(w for _, w in pairs)
    if wsum > 0:
        return sum(v * w for v, w in pairs) / wsum
    return sum(v for v, _ in pairs) / len(pairs)


def _median(values: list[float]) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return float(s[mid])
    return float((s[mid - 1] + s[mid]) / 2.0)


# ---------------------------------------------------------------------------
# 1. Window reconstruction (mirrors state.py's DailyStateBuilder._build())
# ---------------------------------------------------------------------------
def _prepare_ticker_index(events_df: pd.DataFrame) -> dict[str, dict]:
    """Pre-sort events per ticker by transaction_date, keeping a parallel
    list of transaction_date for bisect-based window slicing.

    Avoids an O(full events_df) scan per event: build_research_dataset
    calls _window_for_ticker_day() once per episode start, and there can be
    ~48k of those over ~597k rows in the real dataset.
    """
    out: dict[str, dict] = {}
    for ticker, sub in events_df.groupby("ticker", sort=False):
        sub_sorted = sub.sort_values("transaction_date", kind="mergesort")
        out[ticker] = {
            "tx_dates": sub_sorted["transaction_date"].to_list(),
            "records": sub_sorted.to_dict("records"),
        }
    return out


def _window_for_ticker_day(idx_entry: dict, D: date, window_days: int) -> list[dict]:
    """Reconstruct the exact rolling-window transaction list state.py's
    DailyStateBuilder._build() computes internally for (ticker, D).

    A row is visible on day D iff:
      filing_date <= D  AND  transaction_date <= D
      AND  D <= transaction_date + (window_days - 1) days

    Equivalently, transaction_date must be in [D - (window_days-1), D] and
    filing_date <= D. DailyStateBuilder does not expose this raw window
    back to callers, so we rebuild it ourselves directly from events_df
    rather than modifying state.py.
    """
    tx_dates = idx_entry["tx_dates"]
    records = idx_entry["records"]
    lo = D - timedelta(days=window_days - 1)
    lo_i = bisect_left(tx_dates, lo)
    hi_i = bisect_right(tx_dates, D)
    return [r for r in records[lo_i:hi_i] if r["filing_date"] <= D]


def _prepare_sale_index(sales_df: Optional[pd.DataFrame]) -> dict[str, dict]:
    """Same shape/purpose as _prepare_ticker_index, but keyed by issuer_cik
    over the point-in-time sale index (see backtest/sales_history.py and the
    Section F leakage note above). Reusing _window_for_ticker_day for the
    lookup means concurrent-selling features go through the exact same
    filing_date-gated bisect logic as the buy-side window reconstruction,
    not a parallel reimplementation.

    None or empty `sales_df` (no sale data supplied, or the loader found
    nothing) returns {}, which every downstream lookup treats as "no known
    sales for this issuer" -- every x_sell_*/x_buy_sell_*/x_*_sold_* feature
    degrades to its empty-window value (0 / NaN / False), never an error.
    """
    if sales_df is None or sales_df.empty:
        return {}
    out: dict[str, dict] = {}
    for issuer_cik, sub in sales_df.groupby("issuer_cik", sort=False):
        sub_sorted = sub.sort_values("transaction_date", kind="mergesort")
        out[issuer_cik] = {
            "tx_dates": sub_sorted["transaction_date"].to_list(),
            "records": sub_sorted.to_dict("records"),
        }
    return out


def _sale_window_stats(sales: list[dict]) -> tuple[int, int, float]:
    """(n_transactions, n_distinct_sellers, total_dollar_value) for a list of
    visible sale records (as returned by _window_for_ticker_day against the
    sale index). n_distinct_sellers counts unique owner_key values, so one
    seller filing three separate sale transactions counts once there but
    three times in n_transactions -- mirrors n_insiders vs n_transactions on
    the buy side."""
    n = len(sales)
    sellers = {s.get("owner_key") for s in sales if s.get("owner_key")}
    value = float(sum(float(s.get("value") or 0.0) for s in sales))
    return n, len(sellers), value


def _aggregate_owners(window: list[dict]) -> dict[str, dict]:
    """Per-owner aggregation within the window, mirroring state.py's
    _build_state exactly: keyed by owner_cik, falling back to owner_name
    when owner_cik is falsy. Booleans are True if ANY transaction row for
    that owner has them True.

    Unlike state.py's plain setdefault (which freezes owner_roles at
    whatever the FIRST transaction for that owner happened to carry, even
    if empty), here we keep scanning until we see a non-empty owner_roles
    string for that owner. This is a deliberate difference from state.py,
    used only for the role-composition features below.
    """
    owners: dict[str, dict] = {}
    for tx in window:
        key = tx.get("owner_cik") or tx.get("owner_name") or ""
        slot = owners.setdefault(key, {
            "value": 0.0,
            "is_director": False,
            "is_officer": False,
            "is_ten_percent_owner": False,
            "roles": "",
        })
        slot["value"] += float(tx.get("value") or 0)
        if tx.get("is_director"):
            slot["is_director"] = True
        if tx.get("is_officer"):
            slot["is_officer"] = True
        if tx.get("is_ten_percent_owner"):
            slot["is_ten_percent_owner"] = True
        if not slot["roles"] and tx.get("owner_roles"):
            slot["roles"] = tx["owner_roles"]
    return owners


# ---------------------------------------------------------------------------
# 2. Role-string parsing
# ---------------------------------------------------------------------------
_ROLE_PATTERNS: dict[str, re.Pattern] = {
    "has_ceo": re.compile(r"\bCEO\b|Chief\s+Executive", re.IGNORECASE),
    "has_cfo": re.compile(r"\bCFO\b|Chief\s+Financial", re.IGNORECASE),
    "has_chairman": re.compile(r"\bChairman\b", re.IGNORECASE),
    "has_president": re.compile(r"\bPresident\b", re.IGNORECASE),
    "has_coo": re.compile(r"\bCOO\b|Chief\s+Operating", re.IGNORECASE),
}


def _parse_roles(role_str: Optional[str]) -> dict[str, bool]:
    """Detect CEO / CFO / Chairman / President / COO mentions in a free-text
    owner_roles string (e.g. "Director, Officer (President,Chairman & CEO),
    10% Owner"), case-insensitive. Always returns all five keys."""
    s = role_str or ""
    return {name: bool(pat.search(s)) for name, pat in _ROLE_PATTERNS.items()}


# ---------------------------------------------------------------------------
# 3. Forward-return labels (identical open-to-open convention as signal_fit)
# ---------------------------------------------------------------------------
def _fwd_adj_return(
    prices: PriceUniverse, ticker: str, entry_open: Optional[float], entry_idx: int,
    spy_entry_open: Optional[float], calendar: list[date], h: int,
) -> tuple[float, float, float, bool]:
    """Open-to-open forward return from entry_idx to entry_idx + h trading
    days, SPY-adjusted. Returns (fwd_h, spy_h, adj_h, delisted_h).

    delisted_h is True only when fwd_h WAS computed (not NaN) and computing
    the ticker's own exit price required falling back to
    last_close_on_or_before instead of a live open on the exit day. It is
    False, never NaN, when the horizon falls outside the calendar (fwd_h
    itself is NaN then too) or when entry_open is missing so fwd_h can't be
    computed at all — "delisted" is a claim about the exit leg specifically,
    and we don't want an unrelated missing-entry-price row masquerading as
    a delisting.
    """
    exit_idx = entry_idx + h
    if exit_idx >= len(calendar):
        return float("nan"), float("nan"), float("nan"), False

    X = calendar[exit_idx]
    exit_open = prices.open(ticker, X)
    exit_px = exit_open if exit_open is not None else prices.last_close_on_or_before(ticker, X)
    spy_exit_px = _open_or_fallback(prices, "SPY", X)

    if exit_px is None or entry_open is None:
        fwd_h = float("nan")
    else:
        fwd_h = exit_px / entry_open - 1.0
    if spy_exit_px is None or spy_entry_open is None:
        spy_h = float("nan")
    else:
        spy_h = spy_exit_px / spy_entry_open - 1.0

    adj_h = (fwd_h - spy_h) if (fwd_h == fwd_h and spy_h == spy_h) else float("nan")
    delisted_h = bool(fwd_h == fwd_h and exit_open is None)
    return fwd_h, spy_h, adj_h, delisted_h


# ---------------------------------------------------------------------------
# 4. Price-context helpers
# ---------------------------------------------------------------------------
def _prior_dates(prices: PriceUniverse, ticker: str, entry_day: date) -> list[date]:
    """Ascending trading dates strictly before entry_day. dates_by_ticker is
    already ascending and duplicate-free, so a bisect gives us the cut point
    in O(log n) instead of scanning the whole history per event."""
    dates = prices.dates_by_ticker.get(ticker)
    if not dates:
        return []
    cut = bisect_left(dates, entry_day)
    return dates[:cut]


def _drawdown(prices: PriceUniverse, ticker: str, entry_day: date, n: int) -> float:
    """(last known close / max close over the trailing n prior trading days)
    - 1. Requires at least n prior trading days of history (no shorter
    fallback window — a partial-history drawdown isn't comparable across
    rows)."""
    prior = _prior_dates(prices, ticker, entry_day)
    if len(prior) < n:
        return float("nan")
    close_map = prices.close_by_ticker.get(ticker, {})
    last_close = close_map.get(prior[-1])
    highs = [close_map.get(d) for d in prior[-n:]]
    highs = [h for h in highs if h is not None and h == h]
    if last_close is None or last_close != last_close or not highs:
        return float("nan")
    max_close = max(highs)
    if max_close == 0:
        return float("nan")
    return float(last_close) / float(max_close) - 1.0


def _momentum_skip5(prices: PriceUniverse, ticker: str, entry_day: date, n: int) -> float:
    """Skip-the-most-recent-5-trading-days momentum: anchor = close 5
    trading days back from the last known close, momentum = anchor /
    close(anchor - n trading days) - 1. Requires 6 + n prior trading days."""
    prior = _prior_dates(prices, ticker, entry_day)
    need = 6 + n
    if len(prior) < need:
        return float("nan")
    close_map = prices.close_by_ticker.get(ticker, {})
    anchor = close_map.get(prior[-6])
    base = close_map.get(prior[-need])
    if anchor is None or base is None or anchor != anchor or base != base or base == 0:
        return float("nan")
    return float(anchor) / float(base) - 1.0


def _vol_ann(prices: PriceUniverse, ticker: str, entry_day: date, n: int) -> float:
    """Annualized stdev (ddof=1) of the trailing n daily simple returns,
    computed from the n+1 most recent prior closes. Requires n+1 prior
    closes so we have n actual returns."""
    prior = _prior_dates(prices, ticker, entry_day)
    if len(prior) < n + 1:
        return float("nan")
    close_map = prices.close_by_ticker.get(ticker, {})
    tail = [close_map.get(d) for d in prior[-(n + 1):]]
    if any(p is None or p != p or p <= 0 for p in tail):
        return float("nan")
    rets = [tail[i] / tail[i - 1] - 1.0 for i in range(1, len(tail))]
    if len(rets) < 2:
        return float("nan")
    std = float(np.std(np.asarray(rets, dtype=float), ddof=1))
    return std * math.sqrt(252)


def _price_to_sma200(prices: PriceUniverse, ticker: str, entry_day: date) -> float:
    """Last known close / mean of the trailing 200 prior closes (inclusive
    of the anchor, no skip). NaN if fewer than 200 prior closes exist."""
    prior = _prior_dates(prices, ticker, entry_day)
    if len(prior) < 200:
        return float("nan")
    close_map = prices.close_by_ticker.get(ticker, {})
    last_close = close_map.get(prior[-1])
    window_closes = [close_map.get(d) for d in prior[-200:]]
    if last_close is None or last_close != last_close or any(c is None or c != c for c in window_closes):
        return float("nan")
    sma = sum(window_closes) / 200.0
    if sma == 0:
        return float("nan")
    return float(last_close) / float(sma)


def _tx_volume_vs_adv(prices: PriceUniverse, ticker: str, window: list[dict]) -> float:
    """Value-weighted mean, across the window's transactions, of that
    transaction's own dollar volume divided by the 20-day median dollar
    volume as of that transaction's own date. Both sides are point-in-time
    relative to the transaction's own date (historical relative to
    event_day, so this carries no lookahead)."""
    pairs: list[tuple[float, float]] = []
    dv_map = prices.dv_by_ticker.get(ticker, {})
    for t in window:
        td = t.get("transaction_date")
        val = t.get("value")
        if td is None or val is None or val != val:
            continue
        dv = dv_map.get(td)
        adv = prices.median_dollar_volume(ticker, td, window=20)
        if dv is None or dv != dv or adv is None or adv == 0:
            continue
        pairs.append((dv / adv, float(val)))
    if not pairs:
        return float("nan")
    wsum = sum(w for _, w in pairs)
    if wsum <= 0:
        return float("nan")
    return sum(v * w for v, w in pairs) / wsum


def _insider_vwap(window: list[dict]) -> float:
    """Shares-weighted average price_per_share across the window. Falls
    back to a simple mean of valid prices if every share weight is
    zero/invalid. NaN if there are no valid prices at all."""
    num = 0.0
    den = 0.0
    valid_prices: list[float] = []
    for t in window:
        p = t.get("price_per_share")
        if p is None or p != p or p <= 0:
            continue
        valid_prices.append(float(p))
        s = t.get("shares")
        if s is not None and s == s and s > 0:
            num += float(p) * float(s)
            den += float(s)
    if den > 0:
        return num / den
    if valid_prices:
        return sum(valid_prices) / len(valid_prices)
    return float("nan")


# ---------------------------------------------------------------------------
# 5. Column layout
# ---------------------------------------------------------------------------
# entry_idx is the row's position in the trading calendar. Downstream CV
# needs it to purge a training row whose label window overlaps the test
# block. Deriving that from event_day would be wrong, because a calendar
# gap is not a trading day, so keep the index itself.
_IDENTITY_COLS = ["ticker", "issuer_cik", "event_day", "entry_day", "entry_idx", "entry_open"]

_FEATURE_COLS = [
    # A. Cluster shape
    "x_n_insiders", "x_log1p_total_value", "x_log1p_median_value_per_insider",
    "x_log1p_max_insider_value", "x_n_transactions", "x_n_distinct_tx_dates",
    "x_window_span_days", "x_price_dispersion", "x_unique_price_ratio",
    "x_median_filing_delay_days", "x_max_filing_delay_days", "x_frac_10b5_1",
    "x_frac_routine_footnote", "x_frac_fractional_shares",
    # B. Role composition
    "x_n_directors", "x_n_officers", "x_n_ten_pct", "x_n_pure_directors",
    "x_director_value_share", "x_officer_value_share", "x_ten_pct_value_share",
    "x_has_ceo", "x_has_cfo", "x_has_chairman", "x_has_president", "x_has_coo",
    "x_ceo_value_share", "x_cfo_value_share",
    # C. Stake
    "x_stake_pct_max", "x_stake_pct_median", "x_stake_pct_wmean",
    # D. Insider/issuer history (point-in-time)
    "x_owner_prior_buys_wmean", "x_owner_prior_adj63_mean", "x_owner_same_month_frac",
    "x_issuer_n_prior_clusters", "x_issuer_prior_adj63_mean",
    "x_days_since_prior_cluster", "x_is_first_ever_cluster",
    # E. Price context
    "x_buy_value_to_adv", "x_log_adv20", "x_drawdown_252", "x_drawdown_63",
    "x_mom_21_skip5", "x_mom_63_skip5", "x_mom_252_skip5",
    "x_vol_21_ann", "x_vol_63_ann", "x_price_to_sma200",
    "x_tx_volume_vs_adv", "x_entry_vs_insider_vwap",
    # F. Concurrent selling (point-in-time filed sales, see Section F
    # leakage note above)
    "x_sell_n_cluster", "x_sell_n_insiders_cluster", "x_log1p_sell_value_cluster",
    "x_sell_n_trail90", "x_sell_n_insiders_trail90", "x_log1p_sell_value_trail90",
    "x_buy_sell_balance_cluster", "x_buyer_also_sold_nearby",
    "x_officer_or_director_sold_cluster",
]


def _label_cols(horizons: tuple[int, ...]) -> list[str]:
    cols: list[str] = []
    for h in horizons:
        cols += [f"fwd_{h}", f"spy_{h}", f"adj_{h}", f"delisted_{h}"]
    return cols


def _legacy_cols(feature_keys: list[str]) -> list[str]:
    return [f"f_{k}" for k in feature_keys] + ["conviction_score"]


# ---------------------------------------------------------------------------
# 6. Per-event row builder
# ---------------------------------------------------------------------------
def _build_event_row(
    *, ticker: str, D: date, entry_day: date, entry_idx: int, window: list[dict],
    state: dict, prices: PriceUniverse, calendar: list[date], horizons: tuple[int, ...],
    feature_keys: list[str], owner_history: dict[str, list[_PriorEvent]],
    issuer_history: dict[str, list[_PriorEvent]], ticker_last_event: dict[str, date],
    day_owner_updates: list[tuple[str, _PriorEvent]],
    day_issuer_updates: list[tuple[str, _PriorEvent]],
    day_ticker_updates: dict[str, date],
    sale_index: dict[str, dict], window_days: int,
) -> dict:
    row: dict = {}

    # ---- identity ----
    issuer_cik = window[0].get("issuer_cik", "") if window else ""
    entry_open = prices.open(ticker, entry_day)
    row["ticker"] = ticker
    row["issuer_cik"] = issuer_cik
    row["event_day"] = D
    row["entry_day"] = entry_day
    row["entry_idx"] = entry_idx
    row["entry_open"] = entry_open

    # ---- labels ----
    spy_entry_open = prices.open("SPY", entry_day)
    for h in horizons:
        fwd_h, spy_h, adj_h, delisted_h = _fwd_adj_return(
            prices, ticker, entry_open, entry_idx, spy_entry_open, calendar, h
        )
        row[f"fwd_{h}"] = fwd_h
        row[f"spy_{h}"] = spy_h
        row[f"adj_{h}"] = adj_h
        row[f"delisted_{h}"] = delisted_h

    # Internal history label, independent of the caller's `horizons` tuple.
    _, _, hist_adj63, _ = _fwd_adj_return(
        prices, ticker, entry_open, entry_idx, spy_entry_open, calendar, HISTORY_HORIZON
    )

    # ---- legacy columns (pulled straight from the state dict, not recomputed) ----
    fired = set(state.get("component_keys", []))
    for k in feature_keys:
        row[f"f_{k}"] = 1 if k in fired else 0
    row["conviction_score"] = int(state.get("conviction_score", 0))

    # ---- shared owner aggregation ----
    owners = _aggregate_owners(window)
    total_value = float(sum(o["value"] for o in owners.values()))
    n_insiders = int(state.get("num_insiders", len(owners)))

    # ---- A. Cluster shape ----
    owner_values = sorted(o["value"] for o in owners.values())
    median_owner_value = _median(owner_values) if owner_values else 0.0
    max_owner_value = max(owner_values) if owner_values else 0.0
    row["x_n_insiders"] = n_insiders
    row["x_log1p_total_value"] = float(np.log1p(max(total_value, 0.0)))
    row["x_log1p_median_value_per_insider"] = float(np.log1p(max(median_owner_value, 0.0)))
    row["x_log1p_max_insider_value"] = float(np.log1p(max(max_owner_value, 0.0)))
    row["x_n_transactions"] = len(window)

    tx_dates = {t.get("transaction_date") for t in window if t.get("transaction_date") is not None}
    row["x_n_distinct_tx_dates"] = len(tx_dates)
    row["x_window_span_days"] = float((max(tx_dates) - min(tx_dates)).days) if tx_dates else float("nan")

    valid_prices = [
        float(t["price_per_share"]) for t in window
        if t.get("price_per_share") is not None and t["price_per_share"] == t["price_per_share"]
        and t["price_per_share"] > 0
    ]
    if len(valid_prices) >= 2 and max(valid_prices) > 0:
        row["x_price_dispersion"] = (max(valid_prices) - min(valid_prices)) / max(valid_prices)
    else:
        row["x_price_dispersion"] = float("nan")
    row["x_unique_price_ratio"] = (
        len({round(p, 4) for p in valid_prices}) / len(valid_prices) if valid_prices else float("nan")
    )

    delays = [
        (t["filing_date"] - t["transaction_date"]).days for t in window
        if t.get("filing_date") is not None and t.get("transaction_date") is not None
    ]
    row["x_median_filing_delay_days"] = _median([float(d) for d in delays]) if delays else float("nan")
    row["x_max_filing_delay_days"] = float(max(delays)) if delays else float("nan")

    row["x_frac_10b5_1"] = sum(1 for t in window if t.get("is_10b5_1")) / len(window)
    row["x_frac_routine_footnote"] = sum(
        1 for t in window
        if t.get("footnote_text") and ics._ROUTINE_FOOTNOTE_RE.search(t["footnote_text"])
    ) / len(window)
    row["x_frac_fractional_shares"] = sum(
        1 for t in window
        if t.get("shares") and abs(t["shares"] - round(t["shares"])) > FRACTIONAL_SHARE_TOL
    ) / len(window)

    # ---- B. Role composition ----
    n_directors = sum(1 for o in owners.values() if o["is_director"])
    n_officers = sum(1 for o in owners.values() if o["is_officer"])
    n_ten_pct = sum(1 for o in owners.values() if o["is_ten_percent_owner"])
    n_pure_directors = sum(1 for o in owners.values() if o["is_director"] and not o["is_officer"])
    row["x_n_directors"] = n_directors
    row["x_n_officers"] = n_officers
    row["x_n_ten_pct"] = n_ten_pct
    row["x_n_pure_directors"] = n_pure_directors

    def _value_share(pred) -> float:
        if total_value == 0:
            return float("nan")
        return sum(o["value"] for o in owners.values() if pred(o)) / total_value

    row["x_director_value_share"] = _value_share(lambda o: o["is_director"])
    row["x_officer_value_share"] = _value_share(lambda o: o["is_officer"])
    row["x_ten_pct_value_share"] = _value_share(lambda o: o["is_ten_percent_owner"])

    owner_role_flags = {key: _parse_roles(o["roles"]) for key, o in owners.items()}
    for role_key in _ROLE_PATTERNS:
        # role_key is e.g. "has_ceo" -> column "x_has_ceo"
        row[f"x_{role_key}"] = any(flags[role_key] for flags in owner_role_flags.values())

    row["x_ceo_value_share"] = (
        sum(o["value"] for k, o in owners.items() if owner_role_flags[k]["has_ceo"]) / total_value
        if total_value != 0 else float("nan")
    )
    row["x_cfo_value_share"] = (
        sum(o["value"] for k, o in owners.items() if owner_role_flags[k]["has_cfo"]) / total_value
        if total_value != 0 else float("nan")
    )

    # ---- C. Stake ----
    stake_pairs = [
        (float(t["pct_of_prior_stake"]), float(t.get("value") or 0.0))
        for t in window
        if t.get("pct_of_prior_stake") is not None and t["pct_of_prior_stake"] == t["pct_of_prior_stake"]
    ]
    if stake_pairs:
        vals = [p for p, _ in stake_pairs]
        row["x_stake_pct_max"] = max(vals)
        row["x_stake_pct_median"] = _median(vals)
        row["x_stake_pct_wmean"] = _weighted_mean(stake_pairs)
    else:
        row["x_stake_pct_max"] = float("nan")
        row["x_stake_pct_median"] = float("nan")
        row["x_stake_pct_wmean"] = float("nan")

    # ---- D. Insider/issuer history (point-in-time, see module leakage note) ----
    owner_prior_counts: list[tuple[float, float]] = []
    owner_prior_adj63_means: list[tuple[float, float]] = []
    owner_same_month_fracs: list[tuple[float, float]] = []

    for owner_key, o in owners.items():
        w = o["value"]
        prior_events = owner_history.get(owner_key, [])
        owner_prior_counts.append((float(len(prior_events)), w))

        usable_adj63 = [
            pe.adj63 for pe in prior_events
            if pe.adj63 == pe.adj63 and _label_window_closed(pe.entry_idx, D, calendar)
        ]
        if usable_adj63:
            owner_prior_adj63_means.append((sum(usable_adj63) / len(usable_adj63), w))

        if prior_events:
            same_month = sum(1 for pe in prior_events if pe.event_day.month == D.month) / len(prior_events)
            owner_same_month_fracs.append((same_month, w))

    # Value-weighted mean of per-owner prior-buy counts, NOT a cluster total.
    # A total would track cluster size, which x_n_insiders already carries.
    row["x_owner_prior_buys_wmean"] = _weighted_mean(owner_prior_counts)
    row["x_owner_prior_adj63_mean"] = _weighted_mean(owner_prior_adj63_means)
    row["x_owner_same_month_frac"] = _weighted_mean(owner_same_month_fracs)

    issuer_prior = issuer_history.get(issuer_cik, [])
    row["x_issuer_n_prior_clusters"] = len(issuer_prior)
    usable_issuer_adj63 = [
        pe.adj63 for pe in issuer_prior
        if pe.adj63 == pe.adj63 and _label_window_closed(pe.entry_idx, D, calendar)
    ]
    row["x_issuer_prior_adj63_mean"] = (
        sum(usable_issuer_adj63) / len(usable_issuer_adj63) if usable_issuer_adj63 else float("nan")
    )

    last_event = ticker_last_event.get(ticker)
    if last_event is not None:
        row["x_days_since_prior_cluster"] = float((D - last_event).days)
        row["x_is_first_ever_cluster"] = False
    else:
        row["x_days_since_prior_cluster"] = float("nan")
        row["x_is_first_ever_cluster"] = True

    # Stage this event into the running history. NOT merged until the whole
    # day's events are built (see build_research_dataset) so same-day
    # episodes can never see each other.
    for owner_key, o in owners.items():
        day_owner_updates.append(
            (owner_key, _PriorEvent(event_day=D, entry_idx=entry_idx, adj63=hist_adj63))
        )
    day_issuer_updates.append(
        (issuer_cik, _PriorEvent(event_day=D, entry_idx=entry_idx, adj63=hist_adj63))
    )
    day_ticker_updates[ticker] = D

    # ---- E. Price context ----
    # All "as of entry_day" price lookups below use PriceUniverse helpers
    # that are STRICTLY BEFORE dt by convention (median_dollar_volume etc).
    # Calling them with dt=entry_day (not dt=event_day) means event_day's
    # own close/volume IS included (known at decision time) while nothing
    # on/after entry_day leaks in (entry_day's own open is the execution
    # price, not an input feature).
    adv20 = prices.median_dollar_volume(ticker, entry_day, window=20)
    row["x_buy_value_to_adv"] = total_value / adv20 if adv20 is not None and adv20 != 0 else float("nan")
    row["x_log_adv20"] = float(math.log(adv20)) if adv20 is not None and adv20 > 0 else float("nan")

    row["x_drawdown_252"] = _drawdown(prices, ticker, entry_day, 252)
    row["x_drawdown_63"] = _drawdown(prices, ticker, entry_day, 63)

    row["x_mom_21_skip5"] = _momentum_skip5(prices, ticker, entry_day, 21)
    row["x_mom_63_skip5"] = _momentum_skip5(prices, ticker, entry_day, 63)
    row["x_mom_252_skip5"] = _momentum_skip5(prices, ticker, entry_day, 252)

    row["x_vol_21_ann"] = _vol_ann(prices, ticker, entry_day, 21)
    row["x_vol_63_ann"] = _vol_ann(prices, ticker, entry_day, 63)

    row["x_price_to_sma200"] = _price_to_sma200(prices, ticker, entry_day)

    row["x_tx_volume_vs_adv"] = _tx_volume_vs_adv(prices, ticker, window)

    vwap = _insider_vwap(window)
    if entry_open is not None and vwap == vwap and vwap != 0:
        row["x_entry_vs_insider_vwap"] = entry_open / vwap - 1.0
    else:
        row["x_entry_vs_insider_vwap"] = float("nan")

    # ---- F. Concurrent selling (point-in-time, see Section F leakage note) ----
    # Both windows are anchored at D (event_day), not entry_day -- see the
    # Section F note above for why. _window_for_ticker_day is the exact same
    # filing_date-gated bisect the buy-side reconstruction uses, applied
    # here to the issuer-keyed sale index instead of the ticker-keyed
    # events_df index.
    issuer_sale_idx = sale_index.get(issuer_cik)
    cluster_sales = _window_for_ticker_day(issuer_sale_idx, D, window_days) if issuer_sale_idx else []
    trail_sales = (
        _window_for_ticker_day(issuer_sale_idx, D, TRAILING_SALE_WINDOW_DAYS)
        if issuer_sale_idx else []
    )

    n_sell_cluster, n_sell_insiders_cluster, sell_value_cluster = _sale_window_stats(cluster_sales)
    n_sell_trail, n_sell_insiders_trail, sell_value_trail = _sale_window_stats(trail_sales)

    row["x_sell_n_cluster"] = n_sell_cluster
    row["x_sell_n_insiders_cluster"] = n_sell_insiders_cluster
    row["x_log1p_sell_value_cluster"] = float(np.log1p(max(sell_value_cluster, 0.0)))

    row["x_sell_n_trail90"] = n_sell_trail
    row["x_sell_n_insiders_trail90"] = n_sell_insiders_trail
    row["x_log1p_sell_value_trail90"] = float(np.log1p(max(sell_value_trail, 0.0)))

    # Buy-dominance share: 1.0 when nobody sold alongside this cluster buy,
    # trending toward 0 as concurrent selling grows relative to the buy
    # itself. Bounded in [0, 1] (both terms are non-negative dollar values),
    # unlike a raw buy/sell ratio which blows up to +inf as sell_value -> 0.
    # NaN only in the degenerate case where the cluster's own buy value is
    # zero AND no concurrent selling was seen either (nothing to compare).
    balance_denom = total_value + sell_value_cluster
    row["x_buy_sell_balance_cluster"] = (
        total_value / balance_denom if balance_denom > 0 else float("nan")
    )

    # Did any owner who BOUGHT into this cluster also show up as a SELLER in
    # the same (filing_date-gated) cluster window? Same owner_key convention
    # as _aggregate_owners (owner_cik, falling back to owner_name).
    buyer_keys = set(owners.keys())
    seller_keys_cluster = {s.get("owner_key") for s in cluster_sales if s.get("owner_key")}
    row["x_buyer_also_sold_nearby"] = bool(buyer_keys & seller_keys_cluster)

    # Was any of the concurrent sellers themselves an officer or director
    # (per that SALE filing's own owner flags, not the buyer's role) --
    # distinct from x_buyer_also_sold_nearby, which doesn't care about role.
    row["x_officer_or_director_sold_cluster"] = any(
        bool(s.get("is_officer")) or bool(s.get("is_director")) for s in cluster_sales
    )

    return row


# ---------------------------------------------------------------------------
# 7. Public API
# ---------------------------------------------------------------------------
def build_research_dataset(
    states: DailyStateBuilder,
    prices: PriceUniverse,
    calendar: list[date],
    events_df: pd.DataFrame,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    drop_ticker_reuse: bool = True,
    sales_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """One row per episode start (a ticker crossing from < 2 to >= 2
    qualifying insiders in its rolling window), with identity columns,
    fwd_/spy_/adj_/delisted_ label columns per horizon, x_-prefixed research
    features, and f_<key>/conviction_score legacy columns.

    Assumes `prices` has already been .ensure()'d and .finalize()'d for
    every ticker that can appear in `events_df`, including SPY.

    drop_ticker_reuse (default True): run ticker_reuse.filter_unsafe_ticker_
    reuse over events_df first, dropping every event whose issuer_cik did
    not hold its ticker at the time of that event (see the "Ticker-reuse
    wiring" note above _ticker_price_frame). Explicit and controllable so a
    caller can pass False to reproduce a pre-fix artifact exactly.

    sales_df (default None): point-in-time sale-transaction table (see
    backtest/sales_history.py), used to build the x_sell_*/x_buy_sell_*/
    x_*_sold_* concurrent-selling features (Section F). None (the default)
    means "no sale data supplied" -- every Section F feature degrades to its
    empty-window value (0 count, NaN dollar value, False flag), NOT an
    error, and NOT an implicit disk scan. Deliberately never auto-loaded
    here: unlike `prices`/`events_df`, which every caller must already have
    in hand, scanning parse_cache/'s 1.3M+ files is expensive enough that
    triggering it as a side effect of calling this function would silently
    make every existing caller (including the whole test suite) pay for it.
    Callers that want the real cache call
    backtest.sales_history.load_or_build_sales_cache() themselves and pass
    the result in (see run_research.py's run_build_dataset_stage).
    """
    feature_keys = list(ics.DEFAULT_WEIGHTS.keys())
    columns = _IDENTITY_COLS + _label_cols(horizons) + _FEATURE_COLS + _legacy_cols(feature_keys)

    if events_df.empty:
        log.warning("build_research_dataset: empty events_df — returning an empty frame.")
        return pd.DataFrame(columns=columns)

    n_dropped_ticker_reuse = 0
    if drop_ticker_reuse:
        events_df, n_dropped_ticker_reuse, reuse_transitions = ticker_reuse.filter_unsafe_ticker_reuse(events_df)
        if n_dropped_ticker_reuse:
            by_class = (
                reuse_transitions["classification"].value_counts().to_dict()
                if not reuse_transitions.empty else {}
            )
            log.info(
                "build_research_dataset: ticker-reuse guard dropped %d event row(s) "
                "whose issuer_cik did not hold its ticker at the time of the event "
                "(%d transition(s) across %d ticker(s): %s) -- see ticker_reuse.py",
                n_dropped_ticker_reuse, len(reuse_transitions),
                reuse_transitions["ticker"].nunique() if not reuse_transitions.empty else 0,
                by_class,
            )
        if events_df.empty:
            log.warning(
                "build_research_dataset: empty events_df after the ticker-reuse guard "
                "dropped %d row(s) — returning an empty frame.", n_dropped_ticker_reuse,
            )
            return pd.DataFrame(columns=columns)

    ticker_index = _prepare_ticker_index(events_df)
    sale_index = _prepare_sale_index(sales_df)
    if sales_df is None:
        log.info("build_research_dataset: no sales_df supplied -- Section F features are all empty-window defaults.")
    else:
        log.info(
            "build_research_dataset: sale index built from %d sale row(s) across %d issuer(s)",
            len(sales_df), len(sale_index),
        )

    owner_history: dict[str, list[_PriorEvent]] = defaultdict(list)
    issuer_history: dict[str, list[_PriorEvent]] = defaultdict(list)
    ticker_last_event: dict[str, date] = {}

    # Split-adjustment: detect each ticker's discontinuities ONCE. A
    # whole-build scan over roughly 4,916 tickers cannot afford to redo this
    # per event. unsafe_label_horizons folds in HISTORY_HORIZON as well as
    # the caller's own horizons, so the internal hist_adj63 label (used for
    # owner/issuer history features below, not only the output labels) sits
    # under the same safety net.
    _split_cache: dict[str, tuple[list[splits_mod.Discontinuity], set[date]]] = {}
    unsafe_label_horizons = tuple(sorted(set(horizons) | {HISTORY_HORIZON}))
    n_split_adjusted_tickers = 0
    n_split_discontinuities = 0
    n_ambiguous_discontinuities = 0
    n_real_move_discontinuities = 0
    n_dropped_unsafe_label = 0

    def _split_info(ticker: str) -> tuple[list[splits_mod.Discontinuity], set[date]]:
        nonlocal n_split_adjusted_tickers, n_split_discontinuities
        nonlocal n_ambiguous_discontinuities, n_real_move_discontinuities
        cached = _split_cache.get(ticker)
        if cached is not None:
            return cached
        frame = _ticker_price_frame(prices, ticker)
        if frame is None or frame.empty:
            info = ([], set())
            _split_cache[ticker] = info
            return info
        discontinuities = splits_mod.detect_discontinuities(ticker, frame)
        for d in discontinuities:
            if d.classification == "split":
                n_split_discontinuities += 1
            elif d.classification == "ambiguous":
                n_ambiguous_discontinuities += 1
            elif d.classification == "real_move":
                n_real_move_discontinuities += 1
        if any(d.classification == "split" for d in discontinuities):
            adjusted = splits_mod.apply_split_adjustments(frame, discontinuities)
            _apply_adjusted_prices(prices, ticker, adjusted)
            n_split_adjusted_tickers += 1
        unsafe_dates = splits_mod.unsafe_label_dates(
            frame, discontinuities, horizons=unsafe_label_horizons
        )
        info = (discontinuities, unsafe_dates)
        _split_cache[ticker] = info
        return info

    rows: list[dict] = []
    prev_qualifying: set[str] = set()
    n_skipped_no_entry = 0
    n_events = 0

    for idx_D, D in enumerate(calendar):
        day_states = states.state_for_day(D)
        qualifying = {t for t, st in day_states.items() if st.get("num_insiders", 0) >= 2}
        new_tickers = sorted(qualifying - prev_qualifying)

        # Staged per-day updates, merged into the shared history only after
        # every event for D is built (see the module leakage note).
        day_owner_updates: list[tuple[str, _PriorEvent]] = []
        day_issuer_updates: list[tuple[str, _PriorEvent]] = []
        day_ticker_updates: dict[str, date] = {}

        for ticker in new_tickers:
            entry_day, entry_idx = _find_entry_day(prices, ticker, idx_D, calendar)
            if entry_day is None:
                n_skipped_no_entry += 1
                continue

            # Detect (once per ticker, cached) and back-adjust confirmed
            # splits before any price lookup below reads this ticker's
            # prices. Then drop the episode outright if its own label
            # window spans an unsafe jump: ambiguous, or an implausibly
            # large real_move. See the split-adjustment note above
            # _PriorEvent.
            _, unsafe_dates = _split_info(ticker)
            if entry_day in unsafe_dates:
                n_dropped_unsafe_label += 1
                continue

            idx_entry = ticker_index.get(ticker)
            window = _window_for_ticker_day(idx_entry, D, states.window_days) if idx_entry else []
            if not window:
                log.warning(
                    "build_research_dataset: no reconstructed window for %s on %s "
                    "despite a qualifying state — skipping.", ticker, D,
                )
                continue

            row = _build_event_row(
                ticker=ticker, D=D, entry_day=entry_day, entry_idx=entry_idx, window=window,
                state=day_states[ticker], prices=prices, calendar=calendar, horizons=horizons,
                feature_keys=feature_keys, owner_history=owner_history, issuer_history=issuer_history,
                ticker_last_event=ticker_last_event, day_owner_updates=day_owner_updates,
                day_issuer_updates=day_issuer_updates, day_ticker_updates=day_ticker_updates,
                sale_index=sale_index, window_days=states.window_days,
            )
            rows.append(row)
            n_events += 1
            if n_events % LOG_EVERY == 0:
                log.info("build_research_dataset: processed %d events (last event_day %s)", n_events, D)

        for owner_key, pe in day_owner_updates:
            owner_history[owner_key].append(pe)
        for issuer_cik, pe in day_issuer_updates:
            issuer_history[issuer_cik].append(pe)
        ticker_last_event.update(day_ticker_updates)

        prev_qualifying = qualifying

    log.info(
        "build_research_dataset: split adjustment -- %d ticker(s) had a confirmed "
        "split and were back-adjusted (%d split, %d ambiguous, %d real_move "
        "discontinuities seen across %d scanned tickers). %d row(s) dropped for "
        "an unsafe label window (ambiguous, or real_move past a %.0fx ceiling).",
        n_split_adjusted_tickers, n_split_discontinuities, n_ambiguous_discontinuities,
        n_real_move_discontinuities, len(_split_cache), n_dropped_unsafe_label,
        splits_mod.REAL_MOVE_CEILING_RATIO,
    )

    df = pd.DataFrame(rows, columns=columns)
    if df.empty:
        log.warning(
            "build_research_dataset: no events produced (%d skipped for missing entry price)",
            n_skipped_no_entry,
        )
        return df
    log.info(
        "build_research_dataset: %d events spanning %s .. %s (%d skipped for missing "
        "entry price within %d trading days, %d dropped for ticker reuse)",
        len(df), df["event_day"].min(), df["event_day"].max(),
        n_skipped_no_entry, MAX_ENTRY_LOOKAHEAD, n_dropped_ticker_reuse,
    )
    return df


# Data goes to research_data/, not research/. research/ is the model package.
def save_research_dataset(df: pd.DataFrame, out_dir: str = "research_data", tag: str = "") -> str:
    """Write `df` to a parquet file under `out_dir`, named
    research_<tag_>_<n_rows>rows_<YYYYMMDD>.parquet. Returns the path
    written.

    Writes to a `.tmp` sibling first, then os.replace()s it over the real
    path. A same-filesystem rename is atomic, so a run interrupted mid-write
    (Ctrl-C, OOM kill, power loss) can never leave a truncated parquet file
    sitting at the real path where a later load would either error loudly or,
    worse, silently succeed on a partial file. Mirrors
    split_fingerprint._atomic_save / repair_price_cache._atomic_save.
    """
    os.makedirs(out_dir, exist_ok=True)
    tag_part = f"{tag}_" if tag else ""
    fname = f"research_{tag_part}{len(df)}rows_{date.today():%Y%m%d}.parquet"
    path = os.path.join(out_dir, fname)
    tmp_path = path + ".tmp"
    df.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, path)
    log.info("Wrote research dataset: %s (%d rows, %d cols)", path, len(df), len(df.columns))
    return path


def load_research_dataset(path: str) -> pd.DataFrame:
    """Read a research dataset parquet back into a DataFrame."""
    df = pd.read_parquet(path)
    log.info("Loaded research dataset: %s (shape=%s)", path, df.shape)
    return df
