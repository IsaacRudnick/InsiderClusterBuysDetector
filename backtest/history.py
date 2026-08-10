"""Historical filings scrape and qualifying-event timeline build.

Delegates SEC fetching/parsing to insider_cluster_buys.* (which uses parse_cache/
so repeat runs are cheap). Produces a tidy DataFrame keyed by transaction
date and filing date that the state layer slices into rolling-window views.
"""

from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd

import insider_cluster_buys as ics
from . import tickers as tickers_module
from . import ticker_reuse

log = logging.getLogger(__name__)

EVENTS_CACHE_DIR = "clusters_history"


def _parse_filing_date(raw: str) -> Optional[date]:
    if not raw:
        return None
    for fmt in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    return None


def _parse_tx_date(raw: str) -> Optional[date]:
    if not raw:
        return None
    try:
        return datetime.strptime(raw, "%Y-%m-%d").date()
    except ValueError:
        return None


def scrape_filings(months_back: int) -> tuple[list[dict], list[dict]]:
    """Discover + parse every Form 4/A in the lookback window.

    Returns (parsed_filings, errors). Reuses insider_cluster_buys.parse_cache/.
    """
    lookback_days = int(months_back * 30.44) + 5
    log.info("Discovering Form 4 filings for last %d days (~%d months)",
             lookback_days, months_back)
    filings = ics.discover_filings(lookback_days)

    parsed: list[dict] = []
    errors: list[dict] = []
    total = len(filings)
    log.info("Parsing %d filings (cache-hot ones are free)", total)

    with ThreadPoolExecutor(max_workers=ics.MAX_WORKERS) as ex:
        futures = [ex.submit(ics._parse_one, f) for f in filings]
        for i, fut in enumerate(as_completed(futures), start=1):
            result = fut.result()
            if result.get("_error"):
                result.pop("_error", None)
                errors.append(result)
            else:
                parsed.append(result)
            if i % 500 == 0 or i == total:
                log.info("Parsed %d/%d filings", i, total)

    log.info("Parsed %d filings (%d errors)", len(parsed), len(errors))
    return parsed, errors


def build_events_df(parsed_filings: list[dict]) -> pd.DataFrame:
    """Flatten parsed filings to one row per qualifying (Form 4, 'P', acquisition).

    Normalizes filing_date from YYYYMMDD -> date and transaction_date from
    YYYY-MM-DD -> date. Drops rows missing either date or the ticker.
    """
    rows = ics.extract_qualifying_rows(parsed_filings)
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["transaction_date"] = df["transaction_date"].map(_parse_tx_date)
    df["filing_date"] = df["filing_date"].map(_parse_filing_date)
    df = df.dropna(subset=["transaction_date", "filing_date"]).copy()
    df = df[df["ticker"].astype(str).str.strip().ne("")].copy()
    df["ticker"] = df["ticker"].str.strip().str.upper()
    df["value"] = df["value"].fillna(0.0).astype(float)
    df["shares"] = df["shares"].fillna(0.0).astype(float)
    df["price_per_share"] = df["price_per_share"].fillna(0.0).astype(float)

    df = df.sort_values(["filing_date", "transaction_date"]).reset_index(drop=True)
    log.info("Events DataFrame: %d rows, %d unique tickers, %d unique insiders",
             len(df), df["ticker"].nunique(), df["owner_cik"].nunique())
    return df


def save_events_df(df: pd.DataFrame, window_start: date, window_end: date) -> str:
    os.makedirs(EVENTS_CACHE_DIR, exist_ok=True)
    fname = f"events_{window_start:%Y%m%d}_{window_end:%Y%m%d}.parquet"
    path = os.path.join(EVENTS_CACHE_DIR, fname)
    df.to_parquet(path, index=False)

    manifest_path = os.path.join(EVENTS_CACHE_DIR, "manifest.json")
    manifest = {"scraped_ranges": []}
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, encoding="utf-8") as fh:
                manifest = json.load(fh)
        except (OSError, json.JSONDecodeError):
            pass
    manifest.setdefault("scraped_ranges", []).append({
        "from": window_start.isoformat(),
        "to": window_end.isoformat(),
        "file": fname,
        "n_rows": len(df),
        "scraped_at": datetime.now().isoformat(timespec="seconds"),
    })
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    log.info("Wrote events parquet: %s (%d rows)", path, len(df))
    return path


def _basic_clean_events_df(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with impossible dates and normalize/validate tickers.

    A transaction cannot occur after the filing that reports it. The raw SEC
    data contains a small number of typo dates, some many years in the future
    (the 2018-2026 scrape contains transaction dates up to 2033). These rows
    corrupt the rolling-window index and any forward-return label, so remove
    them here rather than in each consumer.

    Also normalizes ticker symbols to handle malformed cases and drops rows
    whose ticker cannot be normalized to a valid symbol.

    Deliberately does NOT touch ticker reuse (see _apply_ticker_reuse_filter
    below and _clean_events_df's docstring). build_history's fresh-scrape
    branch persists THIS function's output to the events parquet -- the
    cache must hold every row that survives basic cleaning, including rows
    a ticker-reuse drop would remove, so a later run with
    BT_DROP_TICKER_REUSE=0 can still recover them. Only load_events_df /
    build_history's in-memory frame goes on to have ticker-reuse rows
    dropped.
    """
    if df.empty:
        return df
    n_before = len(df)

    # Drop rows with impossible dates
    bad = df["transaction_date"] > df["filing_date"]
    if bad.any():
        log.warning("Dropping %d rows with transaction_date after filing_date", int(bad.sum()))
        df = df[~bad].copy()

    # Normalize tickers and drop rows where ticker is unusable
    df["ticker"] = tickers_module.normalize_ticker_series(df["ticker"])
    n_dropped_tickers = df["ticker"].isna().sum()
    if n_dropped_tickers > 0:
        log.info("Dropping %d rows with malformed/unusable tickers", int(n_dropped_tickers))
        df = df[df["ticker"].notna()].copy()

    log.info("Basic-cleaned events: %d -> %d rows", n_before, len(df))
    return df


def _apply_ticker_reuse_filter(
    df: pd.DataFrame, *, drop_ticker_reuse: bool = True,
) -> tuple[pd.DataFrame, int]:
    """Run ticker_reuse.filter_unsafe_ticker_reuse over df if requested.

    Split out of _clean_events_df (see its docstring) so build_history's
    fresh-scrape branch can persist _basic_clean_events_df's output to the
    events parquet BEFORE this filter ever runs, instead of baking the drop
    into the cache.

    drop_ticker_reuse (default True): run ticker_reuse.filter_unsafe_
    ticker_reuse over the FULL df (see that module and backtest/research.py's
    "Ticker-reuse wiring" note), dropping every event whose issuer_cik did
    not hold its ticker at the time of that event -- a delisted company's
    insider buys must not silently inherit whatever company holds its old
    ticker today. Explicit and controllable so a caller can pass False to
    reproduce a pre-fix artifact exactly, matching build_research_dataset's
    own drop_ticker_reuse kwarg.

    Returns (filtered_df, n_dropped_ticker_reuse).
    """
    if df.empty:
        return df, 0

    n_dropped_ticker_reuse = 0
    if drop_ticker_reuse:
        df, n_dropped_ticker_reuse, reuse_transitions = ticker_reuse.filter_unsafe_ticker_reuse(df)
        if n_dropped_ticker_reuse:
            by_class = (
                reuse_transitions["classification"].value_counts().to_dict()
                if not reuse_transitions.empty else {}
            )
            log.info(
                "_apply_ticker_reuse_filter: ticker-reuse guard dropped %d event row(s) "
                "whose issuer_cik did not hold its ticker at the time of the event "
                "(%d transition(s) across %d ticker(s): %s) -- see ticker_reuse.py",
                n_dropped_ticker_reuse, len(reuse_transitions),
                reuse_transitions["ticker"].nunique() if not reuse_transitions.empty else 0,
                by_class,
            )
    return df, n_dropped_ticker_reuse


def _clean_events_df(
    df: pd.DataFrame, *, drop_ticker_reuse: bool = True,
) -> tuple[pd.DataFrame, int]:
    """Drop rows with impossible dates, normalize/validate tickers, and (by
    default) drop unsafe ticker-reuse rows.

    This is the seam shared by BOTH the cached-parquet load path
    (load_events_df) and the fresh-scrape path (build_history) -- the only
    place a cleaning rule can be applied once and be guaranteed to reach
    every event the backtest or the research pipeline will ever see. It is
    just _basic_clean_events_df followed by _apply_ticker_reuse_filter (see
    both docstrings); build_history's fresh-scrape branch calls those two
    steps separately instead of calling this function, so it can persist the
    basic-cleaned frame to the events parquet before the ticker-reuse filter
    runs, keeping the drop out of the cache.

    drop_ticker_reuse (default True) runs ticker-reuse filtering here, before
    build_history's BT_AS_OF filing-date trim, deliberately: classification
    decides which CIK last (currently) holds a ticker, and trimming to an
    as-of date first would change who counts as "current" and so change
    classifications.

    Returns (cleaned_df, n_dropped_ticker_reuse).
    """
    df = _basic_clean_events_df(df)
    return _apply_ticker_reuse_filter(df, drop_ticker_reuse=drop_ticker_reuse)


def load_events_df(
    path: str, *, drop_ticker_reuse: bool = True,
) -> tuple[pd.DataFrame, date, date, int]:
    """Load a previously scraped events parquet.

    Returns (df, window_start, window_end, n_dropped_ticker_reuse), the
    window taken from the transaction-date range of the file. Use this to
    skip Phase 1 entirely. A full 2018-2026 scrape costs about one day per
    year of history, but the same rows read back from parquet take about 15
    seconds.

    drop_ticker_reuse is forwarded to _clean_events_df -- see its docstring.
    """
    log.info("Loading cached events from %s", path)
    df = pd.read_parquet(path)
    df, n_dropped_ticker_reuse = _clean_events_df(df, drop_ticker_reuse=drop_ticker_reuse)
    if df.empty:
        raise SystemExit(f"Events file {path} has no usable rows")
    window_start = df["transaction_date"].min()
    window_end = df["filing_date"].max()
    log.info(
        "Loaded %d rows, %d tickers, %s -> %s",
        len(df), df["ticker"].nunique(), window_start, window_end,
    )
    return df, window_start, window_end, n_dropped_ticker_reuse


def resolve_events_path(spec: str) -> str:
    """Resolve BT_EVENTS_FROM to a concrete parquet path.

    Accepts an explicit path, or 'latest' to pick the widest file in
    clusters_history/ (most transaction-date coverage, which is what a
    research run wants).
    """
    if spec != "latest":
        if not os.path.exists(spec):
            raise SystemExit(f"BT_EVENTS_FROM={spec!r} does not exist")
        return spec
    if not os.path.isdir(EVENTS_CACHE_DIR):
        raise SystemExit(f"BT_EVENTS_FROM=latest but {EVENTS_CACHE_DIR}/ does not exist")
    candidates = [
        os.path.join(EVENTS_CACHE_DIR, f)
        for f in os.listdir(EVENTS_CACHE_DIR)
        if f.startswith("events_") and f.endswith(".parquet")
    ]
    if not candidates:
        raise SystemExit(f"No events_*.parquet files in {EVENTS_CACHE_DIR}/")
    # Widest file wins, measured by size on disk as a proxy for row count.
    best = max(candidates, key=os.path.getsize)
    log.info("BT_EVENTS_FROM=latest resolved to %s", best)
    return best


def build_history(
    months_back: int,
    as_of: date | None = None,
    events_from: str | None = None,
    *,
    drop_ticker_reuse: bool = True,
) -> tuple[pd.DataFrame, date, date, list[dict], int]:
    """End-to-end: scrape, flatten, persist.

    Returns (df, window_start, window_end, parse_errors, n_dropped_ticker_reuse).

    If `events_from` is set, read that parquet instead of scraping. `as_of`
    pins the window end so a run is reproducible. Without it the window
    tracks date.today() and never matches a cached file.

    drop_ticker_reuse (default True) is forwarded to the cleaning helpers,
    which run on BOTH branches below. ORDERING IS LOAD-BEARING: the
    ticker-reuse classification (see ticker_reuse.py) decides which
    issuer_cik is the ticker's CURRENT occupant by looking at the full,
    untrimmed event history. It must run on the full event set BEFORE the
    BT_AS_OF trim just below runs (cached-parquet branch) or before scrape's
    own window trim (fresh-scrape branch) -- trimming first would make some
    later CIK invisible to the classifier and could change which span looks
    "current", changing a REUSE/RENAME call. Basic cleaning + ticker-reuse
    filtering run before either trim in both branches, so this ordering
    already holds.

    The fresh-scrape branch persists events_*.parquet to disk, and a full
    scrape is expensive (about a day per year of history) so that cache must
    never silently lose rows. It therefore saves the basic-cleaned frame
    (_basic_clean_events_df: bad dates + ticker normalization only) BEFORE
    applying the ticker-reuse filter, and only applies that filter to the
    in-memory frame the run actually returns/trades on -- see
    _basic_clean_events_df's and _apply_ticker_reuse_filter's docstrings. The
    cached-parquet branch (load_events_df) has no such concern: it re-reads
    the untouched source file every run, so it can keep calling
    _clean_events_df (basic clean + ticker-reuse filter together) without
    ever writing the filtered result back.
    """
    if events_from:
        df, window_start, window_end, n_dropped_ticker_reuse = load_events_df(
            resolve_events_path(events_from), drop_ticker_reuse=drop_ticker_reuse,
        )
        if as_of is not None:
            # Honor the pin: hide anything filed after the as-of date so a
            # replay of an older date cannot see later filings. This trim
            # runs AFTER load_events_df's ticker-reuse classification above,
            # which already saw the full, untrimmed file -- see docstring.
            n_before = len(df)
            df = df[df["filing_date"] <= as_of].copy()
            window_end = as_of
            if len(df) != n_before:
                log.info("BT_AS_OF=%s trimmed %d -> %d rows", as_of, n_before, len(df))
        return df, window_start, window_end, [], n_dropped_ticker_reuse

    window_end = as_of or date.today()
    window_start = window_end - timedelta(days=int(months_back * 30.44))
    parsed, errors = scrape_filings(months_back)
    df = build_events_df(parsed)
    # Trim to the requested window (defensive; discover_filings can return a hair more)
    df = df[df["transaction_date"] >= window_start].copy()
    df = _basic_clean_events_df(df)
    # Persist the basic-cleaned (but NOT ticker-reuse-filtered) frame. This is
    # the cache: a later run with BT_DROP_TICKER_REUSE=0 must be able to
    # recover rows this run's filter drops below, and a rescrape is too
    # expensive to redo just to get them back. See build_history's docstring.
    if not df.empty:
        save_events_df(df, window_start, window_end)
    df, n_dropped_ticker_reuse = _apply_ticker_reuse_filter(df, drop_ticker_reuse=drop_ticker_reuse)
    return df, window_start, window_end, errors, n_dropped_ticker_reuse
