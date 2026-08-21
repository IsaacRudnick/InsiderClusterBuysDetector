"""Has anyone ever swept the definition of a "cluster buy"?

Every strategy, model and backtest in this repo is conditioned on one
inherited choice: a cluster is ">=2 distinct insiders buying the same
issuer within a rolling 14-day window" (insider_cluster_buys.py's
ICB_MIN_INSIDERS=2 / ICB_WINDOW_DAYS=14 defaults). Nobody has ever asked
whether a different threshold, window, dollar-size floor, or buyer-role
requirement produces a materially better RAW forward return. This module
asks that question directly, on the raw local data, with no model in the
loop.

WHAT THIS DOES AND DOES NOT DO
-------------------------------
This is a pure event-definition sweep. It does not fit anything, does not
rank candidates within an event, and does not decide what to trade. It
answers one question per definition: "if you bought every event this
definition would have flagged, at the next open after the window closed,
what did you get?" Everything downstream (scoring, banding, cost/liquidity
audits) is a separate question this script deliberately does not touch.

CLUSTER DETECTION
------------------
Reuses insider_cluster_buys.py::detect_clusters's exact windowing
algorithm (grow a maximal window within `window_days` of an anchor
transaction, check whether it holds >= `min_insiders` distinct owners, and
if so consume the whole window and restart the anchor after it) --
translated here into a form that also carries the per-window aggregates
(total dollar value, officer/CEO-CFO/ten-percent-owner role mix) each
filter axis below needs. The two-pointer scan itself is unchanged.

EVENT DAY / ENTRY CONVENTION
-----------------------------
Event day = the LAST transaction date in the qualifying window (the day
the cluster asserted itself). Entry = the first trading day strictly after
event day with a valid open price for that ticker (searched up to
MAX_ENTRY_LOOKAHEAD trading days ahead, same next-open convention and
lookahead as backtest/signal_fit.py's _find_entry_day / _open_or_fallback,
reimplemented here against a local, in-memory price_cache/ reader instead
of backtest.prices.PriceUniverse so this script never touches the network).
Forward returns are open-to-open from entry to entry+H trading days on a
shared SPY trading calendar, exactly mirroring backtest/research.py's
_fwd_adj_return.

TICKER REUSE
------------
backtest/ticker_reuse.py found that 4.88% of events in this file price
against a company that never actually held that ticker (a delisted name's
symbol picked up years later by an unrelated business). That bias runs
upward (see ticker-reuse-misprices-events.md) and would contaminate a
definition sweep the same way it contaminated everything else, so this
script runs ticker_reuse.filter_unsafe_ticker_reuse over the raw events
before any clustering, exactly as backtest/research.py does.

GRADING
-------
Log excess over SPY (and, at 21 days, over IWM too), NOT arithmetic excess:
np.log1p(fwd) - np.log1p(bench_fwd), the same construction tools/score_lab.
py::log_excess and research/screen_model.py::log_excess use, because
arithmetic excess does not aggregate (see log-excess-flips-the-population-
sign.md: the average adj_63 is +1.02% while the same events compound to
-3.49% against SPY). The headline number in every table is the MEDIAN, not
the mean -- this label has a right tail that runs past +1000%, and prior
notes in RESEARCH_NOTES.md document exactly how badly a mean misleads here.

This script does not modify any existing file and does not require
network access -- clusters_history/events_*.parquet and price_cache/*.parquet
are both already on disk.
"""

from __future__ import annotations

import argparse
import bisect
import itertools
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from backtest import ticker_reuse  # noqa: E402

DEFAULT_EVENTS_PATH = os.path.join(
    REPO_ROOT, "clusters_history", "events_20180813_20260813.parquet"
)
PRICE_CACHE_DIR = os.path.join(REPO_ROOT, "price_cache")

# ---------------------------------------------------------------------------
# Sweep grid
# ---------------------------------------------------------------------------
MIN_INSIDERS_GRID: tuple[int, ...] = (2, 3, 4)
WINDOW_DAYS_GRID: tuple[int, ...] = (3, 7, 14, 30)
MIN_TOTAL_VALUE_GRID: tuple[float, ...] = (0.0, 50_000.0, 250_000.0, 1_000_000.0)
MIN_PRICE_GRID: tuple[float, ...] = (0.0, 5.0)
ROLE_FILTERS: tuple[str, ...] = ("any", "officer", "ceo_cfo", "exclude_ten_pct_only")

#: The inherited, never-swept incumbent. Must appear in every report.
BASELINE = dict(
    min_insiders=2, window_days=14, min_total_value=0.0,
    role_filter="any", min_price=0.0,
)

#: Combinations producing fewer priced events than this are dropped from the
#: grid -- too few to say anything about a median, let alone a tail.
MIN_EVENTS = 300

HORIZONS: tuple[int, ...] = (21, 63)

#: Trading days to search past the cluster's last transaction date for a
#: priceable next-open entry. Mirrors backtest/signal_fit.py's
#: MAX_ENTRY_LOOKAHEAD exactly.
MAX_ENTRY_LOOKAHEAD = 5

#: "Loses more than 30%" is a statement about the raw position, not the
#: benchmark-adjusted return -- it is what actually happened to the dollars.
BIG_LOSS_THRESH = -0.30

_CEO_CFO_RE = re.compile(
    r"\bCEO\b|Chief\s+Executive|\bCFO\b|Chief\s+Financial", re.IGNORECASE
)


# ---------------------------------------------------------------------------
# Local, network-free price reader
# ---------------------------------------------------------------------------
class PriceStore:
    """Lazy, memoized reader over price_cache/{TICKER}.parquet.

    Deliberately reimplements only the two operations backtest/signal_fit.py
    needs from backtest.prices.PriceUniverse (open-on-day, last-close-on-
    or-before) against files already on disk -- no .ensure()/yfinance call
    ever happens, so this sweep is 100% local as the task requires.
    """

    def __init__(self, price_dir: str = PRICE_CACHE_DIR):
        self._dir = price_dir
        self._cache: dict[str, Optional[tuple[list[date], dict, dict]]] = {}

    def _load(self, ticker: str):
        if ticker in self._cache:
            return self._cache[ticker]
        path = os.path.join(self._dir, f"{ticker}.parquet")
        if not os.path.exists(path):
            self._cache[ticker] = None
            return None
        try:
            px = pd.read_parquet(path, columns=["open", "close"])
        except Exception:
            self._cache[ticker] = None
            return None
        if px.empty:
            self._cache[ticker] = None
            return None
        dates = [d.date() for d in px.index.to_pydatetime()]
        opens = px["open"].astype(float).tolist()
        closes = px["close"].astype(float).tolist()
        rec = (dates, dict(zip(dates, opens)), dict(zip(dates, closes)))
        self._cache[ticker] = rec
        return rec

    def open(self, ticker: str, day: date) -> Optional[float]:
        rec = self._load(ticker)
        if rec is None:
            return None
        v = rec[1].get(day)
        if v is None or v != v:
            return None
        return float(v)

    def last_close_on_or_before(self, ticker: str, day: date) -> Optional[float]:
        rec = self._load(ticker)
        if rec is None:
            return None
        dates, _, closes = rec
        i = bisect.bisect_right(dates, day) - 1
        while i >= 0:
            v = closes[dates[i]]
            if v == v:
                return float(v)
            i -= 1
        return None

    def open_or_fallback(self, ticker: str, day: date) -> Optional[float]:
        v = self.open(ticker, day)
        if v is not None:
            return v
        return self.last_close_on_or_before(ticker, day)

    def calendar(self, ticker: str = "SPY") -> list[date]:
        rec = self._load(ticker)
        if rec is None:
            raise RuntimeError(f"Reference ticker {ticker!r} not found in {self._dir}")
        return rec[0]


def _find_entry(
    store: PriceStore, ticker: str, last_date: date, calendar: list[date],
    max_lookahead: int = MAX_ENTRY_LOOKAHEAD,
) -> tuple[Optional[date], Optional[int]]:
    """First trading day strictly after `last_date` with a valid open for
    `ticker`, within `max_lookahead` trading days. Mirrors
    backtest/signal_fit.py::_find_entry_day."""
    idx0 = bisect.bisect_right(calendar, last_date)
    for offset in range(max_lookahead):
        idx = idx0 + offset
        if idx >= len(calendar):
            return None, None
        day = calendar[idx]
        if store.open(ticker, day) is not None:
            return day, idx
    return None, None


# ---------------------------------------------------------------------------
# Loading + cluster detection
# ---------------------------------------------------------------------------
@dataclass
class _Tx:
    d: date
    owner_cik: str
    is_director: bool
    is_officer: bool
    is_ten_pct: bool
    owner_roles: str
    value: float


def load_qualifying_rows(events_path: str) -> pd.DataFrame:
    """Read the events parquet, keep only open-market purchases, and drop
    ticker-reuse-unsafe rows (see module docstring)."""
    df = pd.read_parquet(events_path)
    df = df[(df["transaction_code"] == "P") & (df["acquired_disposed"] == "A")].copy()
    df, n_dropped, transitions = ticker_reuse.filter_unsafe_ticker_reuse(df)
    if n_dropped:
        print(f"ticker-reuse guard: dropped {n_dropped} row(s) across "
              f"{transitions['ticker'].nunique() if not transitions.empty else 0} ticker(s)")
    df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0.0)
    return df


def _group_by_issuer(df: pd.DataFrame) -> dict[str, list[_Tx]]:
    by_issuer: dict[str, list[_Tx]] = defaultdict(list)
    for row in df.itertuples(index=False):
        by_issuer[row.issuer_cik].append(_Tx(
            d=row.transaction_date, owner_cik=row.owner_cik,
            is_director=bool(row.is_director), is_officer=bool(row.is_officer),
            is_ten_pct=bool(row.is_ten_percent_owner),
            owner_roles=row.owner_roles or "", value=float(row.value or 0.0),
        ))
    for txs in by_issuer.values():
        txs.sort(key=lambda t: t.d)
    return by_issuer


def _cluster_aggregates(window: list[_Tx]) -> dict:
    """Per-window rollups every filter axis below reads from."""
    owners: dict[str, dict] = {}
    has_ceo_cfo = False
    total_value = 0.0
    for t in window:
        total_value += t.value
        if _CEO_CFO_RE.search(t.owner_roles):
            has_ceo_cfo = True
        o = owners.setdefault(t.owner_cik, {"officer": False, "director": False, "tenpct": False})
        o["officer"] = o["officer"] or t.is_officer
        o["director"] = o["director"] or t.is_director
        o["tenpct"] = o["tenpct"] or t.is_ten_pct

    has_officer = any(o["officer"] for o in owners.values())
    # An owner is "only" a ten-percent-owner if it is flagged as one and
    # never shows as officer or director anywhere in this window.
    has_non_ten_pct_only_owner = any(
        not (o["tenpct"] and not o["officer"] and not o["director"])
        for o in owners.values()
    )
    return dict(
        n_insiders=len(owners), total_value=total_value,
        has_officer=has_officer, has_ceo_cfo=has_ceo_cfo,
        has_non_ten_pct_only_owner=has_non_ten_pct_only_owner,
    )


def detect_clusters_with_aggregates(
    by_issuer: dict[str, list[_Tx]], min_insiders: int, window_days: int,
) -> list[dict]:
    """insider_cluster_buys.py::detect_clusters's exact maximal-window
    two-pointer scan, extended to carry the per-window aggregates the
    filter axes need. issuer_cik keys and per-owner role bookkeeping are
    the only additions over the original."""
    out: list[dict] = []
    for issuer_cik, txs in by_issuer.items():
        n = len(txs)
        i = 0
        while i < n:
            j = i
            while j + 1 < n and (txs[j + 1].d - txs[i].d).days <= window_days:
                j += 1
            window = txs[i:j + 1]
            distinct = {t.owner_cik for t in window}
            if len(distinct) >= min_insiders:
                agg = _cluster_aggregates(window)
                out.append(dict(
                    issuer_cik=issuer_cik, last_date=window[-1].d, **agg,
                ))
                i = j + 1
            else:
                i += 1
    return out


# ---------------------------------------------------------------------------
# Attach tickers + forward returns
# ---------------------------------------------------------------------------
def build_cluster_frame(
    df: pd.DataFrame, store: PriceStore, calendar: list[date],
    min_insiders_grid: tuple[int, ...] = MIN_INSIDERS_GRID,
    window_days_grid: tuple[int, ...] = WINDOW_DAYS_GRID,
    horizons: tuple[int, ...] = HORIZONS,
    verbose: bool = True,
) -> pd.DataFrame:
    """One row per (min_insiders, window_days, cluster) with price-derived
    forward-return columns. This is the expensive step; every downstream
    filter combination (value / role / min_price) is just a boolean mask
    over this frame, computed once."""
    by_issuer = _group_by_issuer(df)
    # issuer_cik -> ticker (last one filed under, which is what matters for
    # a cluster whose last_date determines it -- ticker_reuse already
    # dropped the ambiguous spans).
    ticker_of = df.drop_duplicates("issuer_cik", keep="last").set_index("issuer_cik")["ticker"]

    all_rows: list[dict] = []
    for min_insiders, window_days in itertools.product(min_insiders_grid, window_days_grid):
        t0 = time.monotonic()
        clusters = detect_clusters_with_aggregates(by_issuer, min_insiders, window_days)
        n_priced = 0
        for c in clusters:
            ticker = ticker_of.get(c["issuer_cik"])
            if not ticker:
                continue
            entry_day, entry_idx = _find_entry(store, ticker, c["last_date"], calendar)
            if entry_day is None:
                continue
            entry_open = store.open(ticker, entry_day)
            spy_entry = store.open("SPY", entry_day)
            iwm_entry = store.open("IWM", entry_day)
            row = dict(
                min_insiders=min_insiders, window_days=window_days,
                ticker=ticker, issuer_cik=c["issuer_cik"], last_date=c["last_date"],
                entry_day=entry_day, entry_price=entry_open,
                n_insiders=c["n_insiders"], total_value=c["total_value"],
                has_officer=c["has_officer"], has_ceo_cfo=c["has_ceo_cfo"],
                has_non_ten_pct_only_owner=c["has_non_ten_pct_only_owner"],
            )
            for h in horizons:
                exit_idx = entry_idx + h
                if exit_idx >= len(calendar):
                    row[f"fwd_{h}"] = np.nan
                    row[f"spy_fwd_{h}"] = np.nan
                    row[f"iwm_fwd_{h}"] = np.nan
                    continue
                exit_day = calendar[exit_idx]
                exit_px = store.open_or_fallback(ticker, exit_day)
                spy_exit = store.open_or_fallback("SPY", exit_day)
                iwm_exit = store.open_or_fallback("IWM", exit_day)
                row[f"fwd_{h}"] = (
                    exit_px / entry_open - 1.0 if exit_px is not None and entry_open else np.nan
                )
                row[f"spy_fwd_{h}"] = (
                    spy_exit / spy_entry - 1.0 if spy_exit is not None and spy_entry else np.nan
                )
                row[f"iwm_fwd_{h}"] = (
                    iwm_exit / iwm_entry - 1.0 if iwm_exit is not None and iwm_entry else np.nan
                )
            all_rows.append(row)
            if row.get(f"fwd_{horizons[0]}") == row.get(f"fwd_{horizons[0]}"):  # not NaN
                n_priced += 1
        if verbose:
            print(f"  min_insiders={min_insiders} window_days={window_days}: "
                  f"{len(clusters)} clusters, {n_priced} priced "
                  f"({time.monotonic() - t0:.1f}s)", flush=True)

    out = pd.DataFrame(all_rows)
    for h in horizons:
        fwd = out[f"fwd_{h}"]
        spy = out[f"spy_fwd_{h}"]
        out[f"log_excess_spy_{h}"] = np.where(
            fwd.notna() & spy.notna(),
            np.log1p(fwd.clip(lower=-0.999)) - np.log1p(spy.clip(lower=-0.999)),
            np.nan,
        )
        iwm = out[f"iwm_fwd_{h}"]
        out[f"log_excess_iwm_{h}"] = np.where(
            fwd.notna() & iwm.notna(),
            np.log1p(fwd.clip(lower=-0.999)) - np.log1p(iwm.clip(lower=-0.999)),
            np.nan,
        )
    return out


# ---------------------------------------------------------------------------
# Definitions -> filters -> summary stats
# ---------------------------------------------------------------------------
def _role_mask(cf: pd.DataFrame, role_filter: str) -> pd.Series:
    if role_filter == "any":
        return pd.Series(True, index=cf.index)
    if role_filter == "officer":
        return cf["has_officer"]
    if role_filter == "ceo_cfo":
        return cf["has_ceo_cfo"]
    if role_filter == "exclude_ten_pct_only":
        return cf["has_non_ten_pct_only_owner"]
    raise ValueError(f"unknown role_filter {role_filter!r}")


def select_definition(
    cf: pd.DataFrame, min_insiders: int, window_days: int,
    min_total_value: float, role_filter: str, min_price: float,
) -> pd.DataFrame:
    mask = (
        (cf["min_insiders"] == min_insiders)
        & (cf["window_days"] == window_days)
        & (cf["total_value"] >= min_total_value)
        & _role_mask(cf, role_filter)
    )
    if min_price > 0:
        mask &= cf["entry_price"] >= min_price
    return cf.loc[mask]


def summarize_definition(sub: pd.DataFrame, horizons: tuple[int, ...] = HORIZONS) -> Optional[dict]:
    priced = sub.dropna(subset=[f"fwd_{horizons[0]}"])
    n = len(priced)
    if n < MIN_EVENTS:
        return None
    stats: dict = dict(n_events=n)
    for h in horizons:
        lx = priced[f"log_excess_spy_{h}"].dropna()
        fwd = priced[f"fwd_{h}"].dropna()
        stats[f"median_lx_spy_{h}"] = float(lx.median()) if len(lx) else np.nan
        stats[f"mean_lx_spy_{h}"] = float(lx.mean()) if len(lx) else np.nan
        stats[f"win_rate_{h}"] = float((lx > 0).mean()) if len(lx) else np.nan
        stats[f"p_lose30_{h}"] = float((fwd < BIG_LOSS_THRESH).mean()) if len(fwd) else np.nan
    lx_iwm_21 = priced[f"log_excess_iwm_{horizons[0]}"].dropna()
    stats[f"median_lx_iwm_{horizons[0]}"] = float(lx_iwm_21.median()) if len(lx_iwm_21) else np.nan
    return stats


def run_sweep(
    cf: pd.DataFrame,
    min_insiders_grid: tuple[int, ...] = MIN_INSIDERS_GRID,
    window_days_grid: tuple[int, ...] = WINDOW_DAYS_GRID,
    min_total_value_grid: tuple[float, ...] = MIN_TOTAL_VALUE_GRID,
    role_filters: tuple[str, ...] = ROLE_FILTERS,
    min_price_grid: tuple[float, ...] = MIN_PRICE_GRID,
) -> pd.DataFrame:
    rows = []
    combos = itertools.product(
        min_insiders_grid, window_days_grid, min_total_value_grid, role_filters, min_price_grid,
    )
    for mi, wd, mv, rf, mp in combos:
        sub = select_definition(cf, mi, wd, mv, rf, mp)
        stats = summarize_definition(sub)
        if stats is None:
            continue
        stats.update(min_insiders=mi, window_days=wd, min_total_value=mv,
                      role_filter=rf, min_price=mp,
                      is_baseline=(mi == BASELINE["min_insiders"]
                                   and wd == BASELINE["window_days"]
                                   and mv == BASELINE["min_total_value"]
                                   and rf == BASELINE["role_filter"]
                                   and mp == BASELINE["min_price"]))
        rows.append(stats)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("median_lx_spy_21", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Time-split sanity check
# ---------------------------------------------------------------------------
def time_split_check(cf: pd.DataFrame, definitions: list[dict]) -> pd.DataFrame:
    """Split the whole priced universe in half by entry_day; recompute each
    definition's headline (21d median log-excess vs SPY) inside each half.
    A definition whose sign or rough size flips between halves is a search
    artifact of the sweep, not a real effect."""
    entry_ts = pd.to_datetime(cf["entry_day"])
    mid_ts = entry_ts.median()
    first_half = cf[entry_ts <= mid_ts]
    second_half = cf[entry_ts > mid_ts]
    rows = []
    for d in definitions:
        row = dict(d)
        for label, half in (("first_half", first_half), ("second_half", second_half)):
            sub = select_definition(
                half, d["min_insiders"], d["window_days"], d["min_total_value"],
                d["role_filter"], d["min_price"],
            )
            stats = summarize_definition(sub)
            row[f"{label}_n"] = stats["n_events"] if stats else len(sub.dropna(subset=["fwd_21"]))
            row[f"{label}_median_lx_spy_21"] = stats["median_lx_spy_21"] if stats else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
_DEF_COLS = ["min_insiders", "window_days", "min_total_value", "role_filter", "min_price"]
_REPORT_COLS = _DEF_COLS + [
    "n_events", "median_lx_spy_21", "mean_lx_spy_21", "win_rate_21", "p_lose30_21",
    "median_lx_spy_63", "mean_lx_spy_63", "win_rate_63", "p_lose30_63",
    "median_lx_iwm_21",
]


def _fmt_table(df: pd.DataFrame) -> str:
    show = df[_REPORT_COLS].copy()
    for c in ["median_lx_spy_21", "mean_lx_spy_21", "median_lx_spy_63", "mean_lx_spy_63",
              "median_lx_iwm_21"]:
        show[c] = show[c].map(lambda v: f"{v:+.4f}")
    for c in ["win_rate_21", "win_rate_63", "p_lose30_21", "p_lose30_63"]:
        show[c] = show[c].map(lambda v: f"{v:.1%}")
    show["min_total_value"] = show["min_total_value"].map(lambda v: f"${v:,.0f}")
    show["min_price"] = show["min_price"].map(lambda v: f"${v:g}")
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        return show.to_string(index=False)


def print_report(sweep: pd.DataFrame, cf: pd.DataFrame, top: int, bottom: int) -> None:
    print(f"\n===== FULL SWEEP: {len(sweep)} definitions with >= {MIN_EVENTS} priced events "
          f"(sorted by 21d median log-excess vs SPY) =====\n")
    print(_fmt_table(sweep))

    print(f"\n===== TOP {top} =====\n")
    print(_fmt_table(sweep.head(top)))

    print(f"\n===== BOTTOM {bottom} =====\n")
    print(_fmt_table(sweep.tail(bottom)))

    base_rows = sweep[sweep["is_baseline"]]
    print("\n===== BASELINE (>=2 insiders / 14-day window / no filters) =====\n")
    if base_rows.empty:
        print(f"  Baseline definition did not clear the {MIN_EVENTS}-event floor "
              f"(should not happen with the shipped grid) -- reporting it directly:")
        sub = select_definition(cf, **BASELINE)
        stats = summarize_definition(sub) or {}
        stats.update(**BASELINE)
        print(pd.DataFrame([stats])[_DEF_COLS + [c for c in _REPORT_COLS if c not in _DEF_COLS
                                                  and c in stats]].to_string(index=False))
    else:
        print(_fmt_table(base_rows))
        rank = int(sweep.index[sweep["is_baseline"]][0]) + 1
        print(f"\n  Baseline rank by 21d median log-excess vs SPY: {rank} of {len(sweep)}")

    print(f"\n===== TIME-SPLIT SANITY CHECK: top 3 definitions =====\n")
    top3 = sweep.head(3)[_DEF_COLS].to_dict("records")
    split = time_split_check(cf, top3)
    split_show = split[_DEF_COLS + ["first_half_n", "first_half_median_lx_spy_21",
                                     "second_half_n", "second_half_median_lx_spy_21"]].copy()
    split_show["min_total_value"] = split_show["min_total_value"].map(lambda v: f"${v:,.0f}")
    split_show["min_price"] = split_show["min_price"].map(lambda v: f"${v:g}")
    for c in ["first_half_median_lx_spy_21", "second_half_median_lx_spy_21"]:
        split_show[c] = split_show[c].map(lambda v: f"{v:+.4f}" if v == v else "n/a")
    print(split_show.to_string(index=False))
    print(
        "\n  A definition only 'works' if the sign and rough magnitude of "
        "first_half and second_half agree -- a large gap or a sign flip is a "
        "search artifact of sweeping 384 combinations, not a real effect."
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events", default=DEFAULT_EVENTS_PATH,
                    help="Path to the events parquet (default: %(default)s)")
    ap.add_argument("--price-cache", default=PRICE_CACHE_DIR,
                    help="Path to price_cache/ (default: %(default)s)")
    ap.add_argument("--top", type=int, default=25, help="Rows to print from the top of the sweep")
    ap.add_argument("--bottom", type=int, default=10, help="Rows to print from the bottom of the sweep")
    ap.add_argument("--out-csv", default=None, help="Optional path to write the full sweep table as CSV")
    args = ap.parse_args(argv)

    t0 = time.monotonic()
    print(f"Loading {args.events} ...")
    df = load_qualifying_rows(args.events)
    print(f"{len(df)} qualifying purchase rows, {df['issuer_cik'].nunique()} issuers "
          f"({time.monotonic() - t0:.1f}s)")

    store = PriceStore(args.price_cache)
    calendar = store.calendar("SPY")
    print(f"SPY trading calendar: {calendar[0]} .. {calendar[-1]} ({len(calendar)} days)")

    print("\nDetecting clusters and pricing forward returns "
          f"for {len(MIN_INSIDERS_GRID)}x{len(WINDOW_DAYS_GRID)} "
          "(min_insiders, window_days) combinations ...")
    cf = build_cluster_frame(df, store, calendar)
    print(f"\n{len(cf)} total priceable (min_insiders, window_days, cluster) rows built "
          f"({time.monotonic() - t0:.1f}s elapsed)")

    sweep = run_sweep(cf)
    print_report(sweep, cf, args.top, args.bottom)

    if args.out_csv:
        sweep.to_csv(args.out_csv, index=False)
        print(f"\nFull sweep table written to {os.path.abspath(args.out_csv)}")

    print(f"\nTotal runtime: {time.monotonic() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
