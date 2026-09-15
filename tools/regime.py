"""A market-regime overlay: de-risk on macro/vol conditions, not on stock picking.

WHY THIS EXISTS
================
A pre-registered holdout test (tools/final_search.py) just showed that every
form of SELECTION tried in this project overfits: the search's best
configuration scored Sharpe 1.437 in the selection window and 0.400 out of
sample. The one change that generalised was not a better universe, band or
model -- it was risk management: swapping a fixed 21-day hold for a 15%
trailing stop armed at +10% took the UNSELECTED population from -1.65%/yr to
+12.67%/yr out of sample, with a smaller drawdown, on the exact same events.

So this module tests the next risk-management lever instead of another
picking lever: does scaling exposure down when the market itself looks
dangerous (high vol, widening credit spreads, tight financial conditions)
help, hurt, or do nothing, on top of that same unselected, stop-managed
population? It is deliberately NOT a model. Every state below is a
self-referential rule (a series compared to its own trailing history), not a
threshold chosen because it worked -- see `regime_state`'s docstring.

THE POINT-IN-TIME PROBLEM, STATED PLAINLY
===========================================
FRED's plain CSV endpoint (`fredgraph.csv?id=SERIES`) returns each series'
CURRENT, most-recently-revised vintage -- not what a person actually knew on
any given historical date. Two separate failure modes hide in that one fact,
and this module handles them differently because they are different sizes of
problem:

  1. PUBLICATION LAG. Every series here is dated by its REFERENCE period, not
     its release date. VIXCLS for Tuesday is the Tuesday close, known after
     Tuesday's close -- fine to use starting Wednesday. NFCI is worse: it is
     a WEEKLY series (one observation per Friday-dated week) that the
     Chicago Fed releases the FOLLOWING Friday, roughly five trading days
     after its own reference date. Using the CSV's date column as "known on
     that date" would let a 2019 backtest trade on information that did not
     exist for another week. `SeriesSpec.lag_trading_days` fixes this: every
     raw observation is shifted forward to the trading day on which it
     would ACTUALLY have been public (`_lag_to_calendar`), and
     `tests/test_regime.py::test_lag_prevents_early_read` proves a value is
     never readable before that day.

  2. REVISION. Lag fixes WHEN a number becomes available; it does nothing
     about WHAT NUMBER was available. VIXCLS and T10Y2Y are market prices --
     once printed, they are not revised. BAMLH0A0HYM2 is an index level,
     also effectively not revised (see its SeriesSpec note for a real,
     separate data problem it has instead). NFCI **is** revised: the Chicago
     Fed periodically restates the index as its underlying inputs are
     restated. This module does NOT solve that -- doing so needs FRED's
     ALFRED real-time-vintage API, which is a different, keyed endpoint, not
     the free `fredgraph.csv` URL this task specified. What is shipped
     instead: NFCI gets the largest lag (an extra trading day of margin on
     top of its five-day release delay) as a partial mitigation, its
     SeriesSpec says "revised" explicitly, and `regime_state`'s majority
     vote means NFCI alone can never flip `risk_off` -- at least one other,
     unrevised series has to agree. Anyone using `nfci_tight` in isolation
     for a date before ~2015 (the rough horizon past which NFCI's revisions
     are known to be small) should treat it as an estimate, not a fact.
     This residual risk is real and is not being hidden.

WHAT "regime" MEANS OPERATIONALLY HERE
=========================================
  regime_frame()  raw series, lagged and forward-filled onto a trading
                   calendar -- the point-in-time data layer.
  regime_state()  a handful of self-referential binary flags computed from
                   `regime_frame()`'s trailing history, plus a `risk_off`
                   majority vote -- the decision layer.
  apply_overlay() scales a daily-marked book's exposure by `risk_off`, one
                   more trading day removed for execution causality -- the
                   portfolio layer.
The CLI at the bottom wires these to `tools/exit_lab.py`'s daily-marked,
slot-limited book (the same machinery `tools/final_search.py` uses) over the
UNSELECTED population -- every cluster event, no band, no score-based
filtering -- so the reported effect is the overlay's alone.
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from dataclasses import dataclass

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools import data_paths  # noqa: E402

from tools import exit_lab as el  # noqa: E402
from tools.final_search import (  # noqa: E402  -- reused, not modified
    HOLDOUT_START,
    SELECTION_END,
    spy_returns,
    stats,
)

try:
    from tools import execution_model as em  # noqa: E402
    HAVE_EXEC_MODEL = True
except ImportError:  # pragma: no cover -- exercised only if the module moves
    HAVE_EXEC_MODEL = False

PRICE_DIR = os.path.join(REPO_ROOT, "price_cache")
FRED_CACHE_DIR = os.path.join(REPO_ROOT, "fred_cache")
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"

FLAT_COST_BPS = 50.0  # fallback round-trip cost when execution_model is absent


# ---------------------------------------------------------------------------
# 0. Trading calendar and the entry_idx <-> date translation
# ---------------------------------------------------------------------------

#: exit_lab's daily_marked_portfolio pads its 'day' axis two days past the
#: last taken trade's own (entry_idx + days_held), and a trade entered in
#: the final days of the price cache can still hold for a few days beyond
#: SPY's last cached bar before running out of price data. Neither is a
#: real trading day with a real price -- it is bookkeeping padding, holding
#: at most a handful of already-exiting positions -- but day_index_to_dates
#: still needs a calendar DATE to label it with. This many extra business
#: days appended past SPY's last cached date covers that padding without
#: pretending any position is still open and priced there.
CALENDAR_FORWARD_PAD_DAYS = 30


def load_trading_calendar(
    price_dir: str | None = None, forward_pad_days: int = CALENDAR_FORWARD_PAD_DAYS,
) -> pd.DatetimeIndex:
    """The trading calendar this repo already standardises on: SPY's own
    price index (see backtest/prices.py's `trading_calendar(ref="SPY")`, and
    `tools/final_search.py`'s `entry_idx`, which is a position in exactly
    this calendar -- see `day_index_to_dates` below), extended forward by
    `forward_pad_days` business days so exit_lab's end-of-window bookkeeping
    padding (see CALENDAR_FORWARD_PAD_DAYS) always has a date to land on.
    These padded days are labels only -- nothing here claims SPY or any
    ticker actually traded on them."""
    path = os.path.join(price_dir or PRICE_DIR, "SPY.parquet")
    idx = pd.DatetimeIndex(pd.to_datetime(pd.read_parquet(path, columns=["close"]).index))
    idx = idx.sort_values().unique()
    if forward_pad_days > 0 and len(idx):
        pad = pd.bdate_range(idx[-1], periods=forward_pad_days + 1)[1:]
        idx = idx.append(pad).unique()
    return idx


def day_index_to_dates(
    day_idx: np.ndarray,
    ref_entry_idx: pd.Series,
    ref_entry_day: pd.Series,
    calendar: pd.DatetimeIndex,
) -> pd.DatetimeIndex:
    """Translate exit_lab's integer 'day' axis back into real calendar dates.

    `exit_lab.daily_marked_portfolio` and `final_search`'s `entry_idx` both
    live in an integer offset into a SPY trading calendar -- but the EPOCH
    (which calendar position is day 0) is an implementation detail of the
    historical event build, never written down as a constant. Rather than
    hardcode it (which would silently go stale the moment the calendar or
    build changes), this function DERIVES the offset from the same events
    driving the run: for every event, (entry_idx, entry_day) is a known
    pair, and entry_day's position in `calendar` minus entry_idx is constant
    across every row -- verified here, not assumed, and this raises loudly
    if it is not (see tests/test_regime.py).
    """
    ref_entry_day = pd.to_datetime(pd.Series(ref_entry_day))
    pos = calendar.get_indexer(ref_entry_day)
    good = pos >= 0
    if not good.any():
        raise ValueError("day_index_to_dates: no ref_entry_day found in calendar")
    offsets = pos[good] - pd.Series(ref_entry_idx).to_numpy()[good]
    offset = int(pd.Series(offsets).mode().iloc[0])
    agree = float(np.mean(offsets == offset))
    if agree < 0.99:
        raise ValueError(
            f"day_index_to_dates: entry_idx -> calendar offset is inconsistent "
            f"({agree:.1%} of rows agree on {offset}) -- this calendar does not "
            "match the one entry_idx was built from"
        )
    idx = np.asarray(day_idx, dtype=int) + offset
    if idx.min() < 0 or idx.max() >= len(calendar):
        raise ValueError(
            "day_index_to_dates: translated day index falls outside the supplied "
            "calendar -- pass a calendar with more lookback/lookahead buffer"
        )
    return pd.DatetimeIndex(calendar[idx])


# ---------------------------------------------------------------------------
# 1. regime_frame -- raw series, lagged to their true publication day
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SeriesSpec:
    fred_id: str
    lag_trading_days: int   # trading days after the OBSERVATION date the value becomes public
    revised: bool           # True if FRED's CSV can differ from what was first published
    note: str


#: See the module docstring for the lag/revision reasoning behind each entry.
DEFAULT_SERIES: dict[str, SeriesSpec] = {
    "vix": SeriesSpec(
        "VIXCLS", lag_trading_days=1, revised=False,
        note="CBOE VIX close, a market price. Known after that day's close, "
             "usable from the next trading day. Not revised.",
    ),
    "hy_oas": SeriesSpec(
        "BAMLH0A0HYM2", lag_trading_days=1, revised=False,
        note="ICE BofA US High Yield OAS, an index level, market-price-like "
             "and effectively not revised. SEPARATE PROBLEM, not a lag issue: "
             "FRED's free CSV for this ID only goes back to 2023-08-21 (ICE's "
             "data licence to FRED lapsed 2022-07..2023-08 and the pre-lapse "
             "history was not restored under this ID). There is NO usable "
             "value before that date -- regime_frame leaves it NaN rather "
             "than fabricating one, so any state built on it is INERT over "
             "the entire 2020-2022 selection window.",
    ),
    "nfci": SeriesSpec(
        "NFCI", lag_trading_days=6, revised=True,
        note="Chicago Fed National Financial Conditions Index. WEEKLY, one "
             "Friday-dated observation per week, released the FOLLOWING "
             "Friday (~5 trading days after its own reference date); 6 used "
             "for a one-day margin. REVISED: FRED serves the current "
             "vintage, not the as-first-published one -- see module "
             "docstring's residual-risk note.",
    ),
    "term_spread": SeriesSpec(
        "T10Y2Y", lag_trading_days=1, revised=False,
        note="10y-2y Treasury constant-maturity spread, a market price. "
             "Known after that day's close, usable from the next trading "
             "day. Not revised.",
    ),
}


def _fetch_series(fred_id: str, cache_dir: str, *, refresh: bool = False,
                   timeout: float = 20.0) -> pd.Series:
    """Read `fred_id` from the on-disk cache, fetching the raw CSV from FRED
    only if it is missing or `refresh` is set. The cached file is the raw
    bytes FRED returned -- an explicit, inspectable record of exactly what
    vintage this run used (see the module docstring: it IS the current
    vintage, on purpose stated, not hidden)."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{fred_id}.csv")
    if refresh or not os.path.exists(path):
        req = urllib.request.Request(
            FRED_URL.format(sid=fred_id), headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec B310
            raw = resp.read()
        with open(path, "wb") as fh:
            fh.write(raw)
    df = pd.read_csv(path, na_values=".")
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["value"].astype(float)


def _lag_to_calendar(raw: pd.Series, lag_trading_days: int,
                      calendar: pd.DatetimeIndex) -> pd.Series:
    """Shift every raw observation to the trading day it actually becomes
    knowable on: `lag_trading_days` trading days after its own date, placed
    onto the real calendar (so weekends/holidays are skipped correctly, and
    a Friday-dated weekly observation lands on an actual trading day)."""
    raw = raw.dropna().sort_index()
    if raw.empty:
        return pd.Series(dtype=float)
    # position of the first calendar day STRICTLY AFTER the obs date, then
    # step forward (lag-1) more trading days -- lag=1 means "the very next
    # trading day", which is the correct floor: a value cannot be knowable
    # on the same day it refers to.
    pos = calendar.searchsorted(raw.index.to_numpy(), side="right")
    known_pos = pos + lag_trading_days - 1
    ok = known_pos < len(calendar)
    known_dates = calendar[known_pos[ok]]
    vals = raw.to_numpy()[ok]
    s = pd.Series(vals, index=known_dates)
    # A holiday-shortened calendar can map two raw obs onto the same known
    # day; keep the most recent raw observation, since that is the one a
    # real decision-maker would see when that day arrives.
    return s.groupby(level=0).last().sort_index()


def regime_frame(
    calendar: pd.DatetimeIndex,
    *,
    series: dict[str, SeriesSpec] | None = None,
    cache_dir: str = FRED_CACHE_DIR,
    refresh: bool = False,
) -> pd.DataFrame:
    """Daily frame, indexed by `calendar`, of every series in `series`
    (default `DEFAULT_SERIES`), each one:

      1. fetched from FRED's free CSV (or the on-disk cache under
         `cache_dir`) -- the CURRENT, most-recently-revised vintage, see the
         module docstring for what that does and does not solve;
      2. shifted forward to the trading day it would actually have been
         public on, per that series' `SeriesSpec.lag_trading_days`
         (`_lag_to_calendar`); and
      3. forward-filled onto every trading day in `calendar` from that day
         on -- a value stays "the most recently known reading" until a newer
         one is published, which is what a real trading decision would see.

    Days before a series' first lagged observation are left NaN, not
    back-filled or zeroed -- there is genuinely nothing known yet, and
    pretending otherwise is exactly the lookahead this module exists to
    avoid. `regime_state` and `apply_overlay` both treat NaN as "unknown",
    never as a synonym for "off" or "on".
    """
    series = series or DEFAULT_SERIES
    calendar = pd.DatetimeIndex(pd.to_datetime(calendar)).sort_values().unique()
    out = pd.DataFrame(index=calendar)
    out.index.name = "date"
    for col, spec in series.items():
        raw = _fetch_series(spec.fred_id, cache_dir, refresh=refresh)
        known = _lag_to_calendar(raw, spec.lag_trading_days, calendar)
        out[col] = known.reindex(calendar).ffill()
    return out


# ---------------------------------------------------------------------------
# 2. regime_state -- self-referential flags, never a fitted threshold
# ---------------------------------------------------------------------------

#: ~1 trading year. VIX's own trailing history is the yardstick, not a
#: level (like "VIX > 20") chosen because it happened to work here.
TRAILING_VOL_WINDOW = 252
TRAILING_VOL_MIN_PERIODS = 60

#: ~1 trading month -- matches this repo's own HORIZON convention elsewhere
#: (see tools/final_search.py's 21-day fixed-hold rule). "Widening" is a
#: SIGN, not a level: is the spread higher now than 21 trading days ago.
CREDIT_WIDEN_WINDOW = 21

_FLAG_COLS = ("vix_high", "hy_widening", "nfci_tight", "term_inverted")


def regime_state(frame: pd.DataFrame) -> pd.DataFrame:
    """A small set of binary flags, each computed ONLY from `frame`'s own
    trailing history -- no threshold here was chosen by checking what it did
    to a strategy's return, which is what "self-referential" buys you: swap
    in a different regime dataset or a different date range and every rule
    below still means the same thing.

    Columns (each {0.0, 1.0, NaN}; NaN = not enough history yet, or the
    underlying series has none for that date -- see regime_frame):
      vix_high       VIX above its own trailing 1-year (252 trading day)
                      median. Needs >=60 trading days of history to fire.
      hy_widening    High-yield OAS higher than it was 21 trading days ago
                      -- credit spreads are RISING (deteriorating), judged
                      on the sign of the recent change, not an absolute
                      level. NaN (not False) for the entire 2020-2022 window
                      here -- BAMLH0A0HYM2's free FRED history only starts
                      2023-08-21, see DEFAULT_SERIES['hy_oas'].note.
      nfci_tight     NFCI above zero -- the Chicago Fed's OWN zero line
                      (average conditions since 1971), so no trailing window
                      is needed; the series is already self-referenced by
                      construction. Carries this series' revision caveat.
      term_inverted  10y-2y Treasury spread below zero (the curve inverted).

    risk_off  1.0 iff a STRICT MAJORITY of the flags that are actually known
              (non-NaN) on that day are True; NaN if none are known yet.
              A majority vote, rather than requiring all four, means one
              series with no history for part of the sample (hy_widening,
              pre-2023-08-21) or one that is revised (nfci_tight) can never
              unilaterally decide the regime -- at least two agreeing,
              independently-sourced signals are required. This threshold
              ("more than half") is arithmetic, not fitted: it does not
              change with how many of the four are known on a given day.
      n_known / n_flags  the denominator/numerator behind risk_off, exposed
              so a caller can see how much evidence backed each day's call
              (e.g. n_known=2 for most of 2020-2022, n_known=4 from
              2023-08-21 on).
    """
    out = pd.DataFrame(index=frame.index)

    vix_med = frame["vix"].rolling(
        TRAILING_VOL_WINDOW, min_periods=TRAILING_VOL_MIN_PERIODS
    ).median()
    out["vix_high"] = np.where(vix_med.isna(), np.nan,
                                (frame["vix"] > vix_med).astype(float))

    hy_chg = frame["hy_oas"].diff(CREDIT_WIDEN_WINDOW)
    out["hy_widening"] = np.where(hy_chg.isna(), np.nan,
                                   (hy_chg > 0).astype(float))

    out["nfci_tight"] = np.where(frame["nfci"].isna(), np.nan,
                                  (frame["nfci"] > 0.0).astype(float))

    out["term_inverted"] = np.where(frame["term_spread"].isna(), np.nan,
                                     (frame["term_spread"] < 0.0).astype(float))

    flags = out[list(_FLAG_COLS)]
    known = flags.notna()
    n_known = known.sum(axis=1)
    n_flags = flags.fillna(0.0).sum(axis=1)  # NaN contributes 0 to the True-count
    out["n_known"] = n_known
    out["n_flags"] = n_flags
    # strict majority of what's KNOWN, e.g. 2-of-2 or 2-of-3, not 2-of-4 when
    # only 2 are known -- n_flags*2 > n_known is "more True than not-True".
    out["risk_off"] = np.where(n_known == 0, np.nan,
                                (n_flags * 2 > n_known).astype(float))
    return out


# ---------------------------------------------------------------------------
# 3. apply_overlay -- scale a daily-marked book's exposure by regime
# ---------------------------------------------------------------------------

MODES: tuple[str, ...] = ("flat", "risk_off_half", "risk_off_flat")


def apply_overlay(port_daily: pd.DataFrame, states: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Scale a daily-marked book's exposure by the regime state.

    `port_daily` -- one row per trading day, columns 'date' (a Timestamp on
        the same trading calendar `states` is indexed by) and 'ret' (that
        day's book return, with empty slots already earning zero -- exactly
        `tools/exit_lab.daily_marked_portfolio`'s output once its integer
        'day' axis has been translated to real dates, see
        `day_index_to_dates`).
    `states`     -- `regime_state()`'s output (or anything with a
        {0.0, 1.0, NaN}-valued 'risk_off' column indexed by date).
    `mode`       -- one of MODES:
        "flat"           the control. No overlay at all: exposure is always
                          1.0 and 'ret' comes back unchanged. See
                          tests/test_regime.py::test_flat_mode_reproduces_book.
        "risk_off_half"  exposure 0.5 on a risk-off day, 1.0 otherwise.
        "risk_off_flat"  exposure 0.0 (fully in cash) on a risk-off day,
                          1.0 otherwise.

    CAUSALITY, TWO LAYERS OF IT. `states['risk_off']` on date d is already
    the LAG-ADJUSTED read (see regime_frame) -- but this function adds one
    more trading day of caution on top: exposure for trading DURING day d is
    set from the state as of the CLOSE of day d-1 (`states.shift(1)`), never
    from day d's own value. A decision made at day d's open cannot act on
    information that (even if technically public) only firmed up sometime
    during day d itself; shifting one more day sidesteps having to know each
    series' exact intraday release time, which the CSV alone cannot tell us.

    UNKNOWN STATE (risk_off is NaN that day -- no history yet, or, for
    hy_widening, any date before 2023-08-21). This function does NOT invent
    a signal from missing data: such a day gets exposure 1.0, the same as
    "flat". This means the overlay is INERT wherever its inputs have no
    history -- a documented limitation, not a hidden one; the CLI reports
    how many days in each window fall in this bucket.

    UNINVESTED CAPITAL EARNS ZERO. An exposure of 0.5 means half the book is
    in cash earning 0%, so `ret * exposure` -- not some other blend -- is
    the correct daily return under that assumption.
    """
    if mode not in MODES:
        raise ValueError(f"apply_overlay: unknown mode {mode!r}, expected one of {MODES}")

    out = port_daily.copy()
    out["date"] = pd.to_datetime(out["date"])

    if mode == "flat":
        out["exposure"] = 1.0
    else:
        risk_off_prior = states["risk_off"].reindex(out["date"]).shift(1).to_numpy()
        on = risk_off_prior == 1.0  # NaN compares False -> defaults to "on" (exposure 1.0)
        level = 0.5 if mode == "risk_off_half" else 0.0
        out["exposure"] = np.where(on, level, 1.0)

    out["ret_unlevered"] = out["ret"]
    out["ret"] = out["ret_unlevered"] * out["exposure"]
    return out


# ---------------------------------------------------------------------------
# 4. CLI -- unselected population, stop-managed, every overlay mode, vs SPY
# ---------------------------------------------------------------------------

#: The one exit rule this task is about testing overlays ON TOP OF: the
#: trailing stop that generalised in tools/final_search.py's holdout test
#: (RULES[3] there, reproduced here so this module has no import-order
#: dependency on final_search's grid staying unchanged).
DEFAULT_RULE = el.ExitRule(hard_stop=0.0, trail_stop=0.15, max_hold=126, activate_at=0.10)
DEFAULT_SLOTS = 20


def load_events(dataset: str, scores: str, score_col: str = "ens") -> pd.DataFrame:
    """Every cluster event, scored but NOT band-selected.

    Unscored rows (this dataset's 2018-2019 warm-up, where the ensemble had
    no out-of-fold prediction yet) are kept, not dropped -- dropping them
    would be a second, silent selection on top of the one this CLI is
    explicitly told not to do. They get score = -inf, matching this repo's
    existing convention elsewhere: an unscored candidate never wins a scarce
    slot over a scored one, but it is never excluded from trading either.
    """
    df = pd.read_parquet(dataset)
    df["event_day"] = pd.to_datetime(df["event_day"])
    df["entry_day"] = pd.to_datetime(df["entry_day"])
    sc = pd.read_parquet(scores)
    sc["event_day"] = pd.to_datetime(sc["event_day"])
    df = df.merge(sc[["ticker", "event_day", score_col]], on=["ticker", "event_day"], how="left")
    df["score"] = df[score_col].fillna(-np.inf)

    if HAVE_EXEC_MODEL:
        cost = em.estimated_cost_bps(df)
    else:
        cost = np.full(len(df), np.nan)
    df["cost_bps"] = np.nan_to_num(cost, nan=FLAT_COST_BPS)
    return df.dropna(subset=["entry_day", "entry_idx"])


def build_book(
    df_window: pd.DataFrame, calendar: pd.DatetimeIndex,
    rule: el.ExitRule = DEFAULT_RULE, slots: int = DEFAULT_SLOTS,
) -> pd.DataFrame:
    """The unselected-population, stop-managed, daily-marked book for one
    window, with a real calendar 'date' column instead of exit_lab's raw
    integer day axis. Empty if too few trades survive to price a book."""
    paths = el.build_paths(df_window, rule.max_hold)
    trades = el.simulate_paths(paths, rule, cost_bps=0.0)
    if trades.empty:
        return pd.DataFrame()
    cost = trades[["ticker", "entry_day"]].merge(
        df_window[["ticker", "entry_day", "cost_bps"]],
        on=["ticker", "entry_day"], how="left")["cost_bps"].to_numpy()
    trades = trades.copy()
    trades["ret"] = trades["ret"] - np.nan_to_num(cost, nan=FLAT_COST_BPS) / 10_000.0

    port = el.daily_marked_portfolio(paths, trades, n_slots=slots, cost_bps=0.0)
    if port.empty:
        return port
    dates = day_index_to_dates(
        port["day"].to_numpy(), df_window["entry_idx"], df_window["entry_day"], calendar
    )
    return port.assign(date=dates)


def _window_report(label: str, port: pd.DataFrame, states: pd.DataFrame) -> pd.DataFrame:
    """One row per overlay mode: this window's stats, plus SPY over the
    identical trading days."""
    bench = spy_returns(None, port["date"].min(), port["date"].max())
    rows = []
    for mode in MODES:
        ov = apply_overlay(port[["date", "ret"]], states, mode)
        r = ov["ret"].to_numpy()
        s = stats(r, bench)
        s.update(window=label, mode=mode,
                 mean_exposure=float(ov["exposure"].mean()))
        rows.append(s)
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default=data_paths.latest_research_dataset())
    ap.add_argument("--scores", default=(
        r"C:\Users\Isaac\AppData\Local\Temp\claude\C--Users-Isaac-Documents-Programming-"
        r"Euen-Work-Insider-Cluster-Buys---Simulator\e19eca61-30f6-4a59-aa18-23cbb4a60ba2"
        r"\scratchpad\ship_scores.parquet"))
    ap.add_argument("--score-col", default="ens")
    ap.add_argument("--refresh-fred", action="store_true",
                     help="Re-download every FRED series instead of using the cache.")
    args = ap.parse_args(argv)

    print("loading events and prices...", flush=True)
    df = load_events(args.dataset, args.scores, args.score_col)
    calendar = load_trading_calendar()

    print("fetching/loading regime series from FRED...", flush=True)
    frame = regime_frame(calendar, refresh=args.refresh_fred)
    states = regime_state(frame)

    windows = {
        "SELECTION": df[df["event_day"] <= SELECTION_END],
        "HOLDOUT": df[df["event_day"] >= HOLDOUT_START],
    }

    all_rows = []
    for label, sub in windows.items():
        print(f"\nbuilding the unselected-population book -- {label} "
              f"({len(sub)} events)...", flush=True)
        port = build_book(sub, calendar)
        if port.empty:
            print(f"  {label}: no book could be priced")
            continue
        cov = states.reindex(port["date"])
        print(f"  {label}: {len(port)} days, {sub['event_day'].min():%Y-%m-%d}"
              f"..{sub['event_day'].max():%Y-%m-%d}  "
              f"risk_off known {cov['risk_off'].notna().mean():.0%} of days "
              f"(hy_widening known {cov['hy_widening'].notna().mean():.0%})")
        rep = _window_report(label, port, states)
        all_rows.append(rep)

    if not all_rows:
        print("no window produced a book")
        return 1

    table = pd.concat(all_rows, ignore_index=True)
    show = ["window", "mode", "days", "mean_exposure", "ann_return", "ann_vol",
            "sharpe", "sortino", "max_dd", "final_x",
            "spy_ann", "spy_sharpe", "spy_dd", "spy_final_x"]
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print("\n===== overlay comparison, by window =====")
        print(table[show].to_string(index=False, float_format=lambda v: f"{v:+.3f}"))

    hold = table[table["window"] == "HOLDOUT"]
    if not hold.empty:
        print("\n===== HOLDOUT verdict (scored once, not tuned after seeing it) =====")
        flat = hold[hold["mode"] == "flat"].iloc[0]
        for _, row in hold.iterrows():
            delta = row["sharpe"] - flat["sharpe"]
            print(f"  {row['mode']:<16s} Sharpe {row['sharpe']:+.3f} "
                  f"(vs flat {delta:+.3f})  maxDD {row['max_dd']:+.1%}  "
                  f"ann {row['ann_return']:+.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
