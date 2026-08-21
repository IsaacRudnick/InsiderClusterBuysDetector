"""FINRA consolidated short-interest history, and point-in-time short-
interest features attached to a COPY of the research dataset.

WHY this exists. Cluster buys often show up in names that are also heavily
shorted -- a crowded short can be the reason insiders are buying (a squeeze
setup) or a warning the insiders are about to be wrong (the shorts know
something). The research dataset has no way to see either story: nothing in
it describes the stock's short position at all. FINRA's free, unauthenticated
Consolidated Short Interest API fills that gap.

Endpoint (no auth, do not add an API key):
    GET  https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest
    POST https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest
         {"limit": ..., "offset": ..., "compareFilters": [...], "fields": [...]}
It is bi-monthly: FINRA members report a short position as of each
settlement date (~15th and last business day of the month), and FINRA
compiles + publishes the compiled table roughly 8 CALENDAR DAYS later. The
dataset spans every market class FINRA sees (NYSE, Nasdaq, ARCA, OTC, ...),
not just OTC, but OTC is where this project's microcaps concentrate.

How tickers are pulled -- BY SETTLEMENT DATE, not by symbol. Both were
tested against the live endpoint. Querying one settlement date at a time
(an EQUAL filter on `settlementDate`, which is the dataset's declared
partition key) returns every symbol FINRA has for that date -- about 16-17k
rows, i.e. ~4 pages at the API's 5000-row page cap. Covering the ~7,390
tickers in clusters_history/events_20180813_20260813.parquet this way costs
roughly 4 requests x ~192 settlement dates (2018-08 through 2026-08) =
~770 requests total. Querying per-symbol instead (`symbolCode` EQUAL) would
cost one request per ticker just to *ask* -- 7,390 requests, roughly 10x
more, before even accounting for tickers this project doesn't need historic
short data for at all. By-date wins decisively.

Discovering the settlement-date calendar is itself cheap: FINRA does not
expose a "list distinct settlement dates" endpoint, but a `symbolCode`
EQUAL filter on one continuously-reported, always-listed name (AAPL) for
the full history returns exactly the calendar of settlement dates as its
`settlementDate` column, in ONE request (its own short-interest history is
well under the 5000-row page cap). See CALENDAR_PROBE_SYMBOL below.

Caching. Every settlement date's FULL raw pull (all market classes, all
symbols, all pages concatenated) is cached verbatim as one CSV per date
under short_interest_cache/, so a rerun -- even for a different ticker
universe or a different research dataset -- costs zero new requests unless
`--refresh` is passed or a new settlement date has appeared since the last
run. The settlement-date calendar itself is cached the same way. Requests
are throttled to <= 5/sec via a private rate limiter (see _RateLimiter;
this is a DIFFERENT host and a DIFFERENT courtesy limit than SEC EDGAR, so
it deliberately does not reuse insider_cluster_buys.py's SEC-facing
limiter/session -- only its already-configured contact User-Agent string,
reused here purely to identify this project honestly to FINRA too).

POINT-IN-TIME CONTRACT -- the entire reason attach_features() exists as
more than a plain merge. A short-interest report's SETTLEMENT date is not
when the market -- or a backtest standing at event_day -- could have known
that report. FINRA compiles and publishes it roughly PUBLICATION_LAG_DAYS
calendar days later (see that constant for the exact value and its
justification). A feature that uses any report whose settlement date is
merely "before event_day" is not point-in-time: a report that settled 3
days before event_day is still unpublished on event_day and using it would
manufacture a fake edge -- the model would be handed information that, in
live use, does not exist yet. Every feature below is computed using only
reports whose PUBLICATION date (settlement_date + PUBLICATION_LAG_DAYS) is
STRICTLY BEFORE event_day. See `_ticker_short_interest_index` and
`attach_features` for the enforcement mechanism (numpy.searchsorted on the
publication-date axis, mirroring tools/earnings_features.py's technique).

Runnable standalone as `python tools/short_interest.py` (pulls/refreshes
short_interest_cache/, writes research_data/short_interest.parquet, then
attaches features to the research parquet and prints a coverage + median-
adj_21-by-quartile report) or importable, e.g. from a test, via
`attach_features`.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import insider_cluster_buys as ics  # noqa: E402  (reuses only its configured contact User-Agent)

log = logging.getLogger("short_interest")

CACHE_DIR = os.path.join(REPO_ROOT, "short_interest_cache")
FINRA_URL = "https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest"

DEFAULT_EVENTS = os.path.join(REPO_ROOT, "research_data", "research_10861rows_20260813.parquet")
DEFAULT_SHORT_INTEREST_PARQUET = os.path.join(REPO_ROOT, "research_data", "short_interest.parquet")

# FINRA's fair-access ask is <= 10 req/s; the task spec for this module asks
# for a stricter courtesy limit of <= 5 req/s, kept intentionally separate
# from insider_cluster_buys.py's SEC-facing 8 req/s limiter (different host,
# different policy).
RATE_LIMIT = 5.0
# Matches insider_cluster_buys.py's approach: a handful of worker threads so
# network latency (not the limiter) is what's overlapped, not raced.
MAX_WORKERS = 5
# Observed live via the API's `record-max-limit` response header (also its
# documented row cap per page).
PAGE_LIMIT = 5000

# One continuously-reported, always-listed symbol, queried once with NO date
# filter, whose own `settlementDate` column IS the FINRA settlement-date
# calendar (every settlement date in this dataset's history). Far cheaper
# than discovering the calendar by paging full-universe data. AAPL has
# reported a short position at every FINRA settlement cycle on record.
CALENDAR_PROBE_SYMBOL = "AAPL"

# FINRA members must report a short position within 2 business days of the
# settlement date; FINRA then compiles and publishes the consolidated table
# for that cycle. Empirically (per this task's own verified spec) that
# publication lands roughly 8 CALENDAR days after settlement. FINRA does not
# expose the actual publication timestamp in this dataset (only
# `settlementDate`), so there is no way to measure the true lag directly --
# this is a documented, deliberately conservative modeling assumption, and
# the single knob to revisit if FINRA's practice is later found to differ.
PUBLICATION_LAG_DAYS = 8

# FINRA's sentinel for "undefined / effectively infinite days to cover"
# (observed live: a security with averageDailyVolumeQuantity == 0 reports
# daysToCoverQuantity == 999.99, NOT the "default value is 0" the metadata
# endpoint's field description claims). Treated as missing, not as a real
# extreme value -- 999.99 would otherwise read as "practically impossible to
# cover", which is a claim about zero-volume microcaps this data cannot
# actually support.
DAYS_TO_COVER_SENTINEL = 999.99

RAW_TO_CANONICAL = {
    "symbolCode": "symbol",
    "settlementDate": "settlement_date",
    "currentShortPositionQuantity": "short_shares",
    "previousShortPositionQuantity": "prev_short_shares",
    "averageDailyVolumeQuantity": "adv_shares",
    "daysToCoverQuantity": "days_to_cover",
    "marketClassCode": "market_class",
}
OUTPUT_COLUMNS = [
    "symbol", "settlement_date", "short_shares", "prev_short_shares",
    "adv_shares", "days_to_cover", "market_class",
]

X_COLS = [
    "x_short_days_to_cover",
    "x_short_pct_of_adv",
    "x_short_change_2m",
    "x_short_report_age_days",
]


# ---------------------------------------------------------------------------
# Rate-limited HTTP
# ---------------------------------------------------------------------------
class _RateLimiter:
    """Same token-bucket approach as insider_cluster_buys.py's _RateLimiter,
    reimplemented locally (not imported) because it must run at a different
    rate against a different host, and this module is meant to stand alone."""

    def __init__(self, rate_per_sec: float):
        self._interval = 1.0 / rate_per_sec
        self._lock = threading.Lock()
        self._next_time = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = max(0.0, self._next_time - now)
            self._next_time = max(now, self._next_time) + self._interval
        if wait > 0:
            time.sleep(wait)


_LIMITER = _RateLimiter(RATE_LIMIT)

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": ics.USER_AGENT,  # identify this project to FINRA too
    "Content-Type": "application/json",
    "Accept": "text/plain",
})


class _RequestStats:
    """Thread-safe counter so main() can report exactly how many live HTTP
    requests a run made (cache hits are free and do not count)."""

    def __init__(self):
        self._lock = threading.Lock()
        self.count = 0

    def bump(self) -> None:
        with self._lock:
            self.count += 1

    def reset(self) -> None:
        with self._lock:
            self.count = 0


STATS = _RequestStats()


def _post(payload: dict) -> requests.Response:
    """POST with retry/backoff on 429s and 5xxs, mirroring
    insider_cluster_buys.py's `_get`. A 4xx other than 429 is a real bug in
    the request (bad field name, bad filter) and must not be retried."""
    last_exc: Exception | None = None
    for attempt in range(5):
        _LIMITER.acquire()
        STATS.bump()
        resp = SESSION.post(FINRA_URL, json=payload, timeout=30)
        if resp.status_code < 400:
            return resp
        if resp.status_code == 429 or resp.status_code >= 500:
            backoff = 2 ** attempt
            log.warning("FINRA %d on POST %s - retrying in %ds (attempt %d/5)",
                        resp.status_code, payload, backoff, attempt + 1)
            try:
                resp.raise_for_status()
            except requests.HTTPError as exc:
                last_exc = exc
            time.sleep(backoff)
            continue
        resp.raise_for_status()  # 4xx, non-429: raise immediately
    assert last_exc is not None
    raise last_exc


# ---------------------------------------------------------------------------
# Settlement-date calendar
# ---------------------------------------------------------------------------
def _calendar_cache_path() -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, "_settlement_dates.json")


def _fetch_full_settlement_date_history(*, refresh: bool = False) -> list[str]:
    """Every settlement date FINRA has ever reported for
    CALENDAR_PROBE_SYMBOL, sorted ascending as ISO strings -- i.e. the full
    FINRA settlement-date calendar (see module docstring). Cached
    unconditionally (not scoped to a start/end window) so changing
    --start/--end never busts this cache."""
    path = _calendar_cache_path()
    if not refresh and os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)

    payload = {
        "limit": PAGE_LIMIT,
        "fields": ["settlementDate"],
        "compareFilters": [
            {"fieldName": "symbolCode", "fieldValue": CALENDAR_PROBE_SYMBOL, "compareType": "EQUAL"},
        ],
    }
    resp = _post(payload)
    df = pd.read_csv(io.StringIO(resp.text), dtype=str)
    dates = sorted(df["settlementDate"].dropna().unique().tolist())
    if not dates:
        raise RuntimeError(
            f"short_interest: probe symbol {CALENDAR_PROBE_SYMBOL!r} returned no "
            "settlement dates -- FINRA endpoint shape may have changed."
        )
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(dates, fh)
    return dates


def fetch_settlement_dates(start: str, end: str, *, refresh: bool = False) -> list[str]:
    """Settlement dates in [start, end] (inclusive, ISO 'YYYY-MM-DD' strings)."""
    all_dates = _fetch_full_settlement_date_history(refresh=refresh)
    return [d for d in all_dates if start <= d <= end]


# ---------------------------------------------------------------------------
# Per-date raw pull (cached verbatim)
# ---------------------------------------------------------------------------
def _date_cache_path(settlement_date: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{settlement_date}.csv")


def fetch_date(settlement_date: str, *, refresh: bool = False) -> pd.DataFrame:
    """All rows (every market class, every symbol) FINRA has for one
    settlement date, paginated to completion and cached as a single raw CSV
    (FINRA's own column names, untouched) so a rerun is free."""
    path = _date_cache_path(settlement_date)
    if not refresh and os.path.exists(path):
        return pd.read_csv(path, dtype=str)

    pages: list[pd.DataFrame] = []
    offset = 0
    total = None
    while total is None or offset < total:
        payload = {
            "limit": PAGE_LIMIT,
            "offset": offset,
            "compareFilters": [
                {"fieldName": "settlementDate", "fieldValue": settlement_date, "compareType": "EQUAL"},
            ],
        }
        resp = _post(payload)
        if total is None:
            try:
                total = int(resp.headers.get("record-total", "0"))
            except ValueError:
                total = 0
        if not resp.text.strip():
            break
        page = pd.read_csv(io.StringIO(resp.text), dtype=str)
        if page.empty:
            break
        pages.append(page)
        offset += PAGE_LIMIT

    df = pd.concat(pages, ignore_index=True) if pages else pd.DataFrame()
    df.to_csv(path, index=False)
    log.info("short_interest: %s -> %d row(s) (%d page request(s))",
              settlement_date, len(df), len(pages))
    return df


def fetch_all(dates: list[str], *, refresh: bool = False, workers: int = MAX_WORKERS) -> pd.DataFrame:
    """Raw (FINRA column names) pull across every settlement date given,
    threaded across DATES (pagination within one date stays sequential --
    the next offset isn't known until the prior page's record-total header
    is read)."""
    frames: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(fetch_date, d, refresh=refresh): d for d in dates}
        done = 0
        for fut in as_completed(futs):
            d = futs[fut]
            try:
                frames.append(fut.result())
            except Exception as exc:  # noqa: BLE001 -- one bad date must not kill the run
                log.warning("short_interest: %s failed (%s)", d, exc)
            done += 1
            if done % 25 == 0:
                log.info("short_interest: %d/%d settlement date(s) done", done, len(dates))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Raw -> canonical table
# ---------------------------------------------------------------------------
def build_table(raw: pd.DataFrame, *, tickers: set[str] | None = None) -> pd.DataFrame:
    """Normalize FINRA's raw column names/rows into the shipped schema:
    symbol, settlement_date, short_shares, prev_short_shares, adv_shares,
    days_to_cover, market_class.

    `tickers`, if given, restricts the OUTPUT to that ticker set -- the raw
    cache on disk always holds every market class and every symbol FINRA
    reported (see fetch_date), so this is purely a size/scope choice for the
    parquet this project ships, not a limitation of what was fetched.
    """
    if raw.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    df = raw.rename(columns=RAW_TO_CANONICAL)[list(RAW_TO_CANONICAL.values())].copy()
    df["symbol"] = df["symbol"].astype(str).str.strip()
    df["settlement_date"] = pd.to_datetime(df["settlement_date"])
    for col in ("short_shares", "prev_short_shares", "adv_shares", "days_to_cover"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # See DAYS_TO_COVER_SENTINEL: FINRA's "undefined" flag, not a real value.
    df.loc[np.isclose(df["days_to_cover"], DAYS_TO_COVER_SENTINEL), "days_to_cover"] = np.nan
    df["market_class"] = df["market_class"].astype(str).str.strip()

    if tickers is not None:
        df = df[df["symbol"].isin(tickers)]

    df = df.drop_duplicates(subset=["symbol", "settlement_date"], keep="last")
    df = df.sort_values(["symbol", "settlement_date"]).reset_index(drop=True)
    return df[OUTPUT_COLUMNS]


# ---------------------------------------------------------------------------
# Point-in-time feature attachment
# ---------------------------------------------------------------------------
def _to_day_ordinal(series: pd.Series) -> np.ndarray:
    """Mirrors tools/earnings_features.py's helper of the same name
    (reimplemented locally rather than imported, to keep this a standalone
    two-file addition): dates/strings/Timestamps -> proleptic-Gregorian
    ordinal-day floats, NaT-safe."""
    dt = pd.to_datetime(series)
    return dt.map(lambda ts: ts.toordinal() if pd.notna(ts) else np.nan).to_numpy()


def _ticker_short_interest_index(short_interest: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    """Per symbol, ascending-sorted-by-settlement-date parallel arrays,
    including `publication_ord` (settlement_ord + PUBLICATION_LAG_DAYS) --
    the axis attach_features actually searches on. Adding a constant to an
    already-ascending array preserves order, so a single sort suffices."""
    idx: dict[str, dict[str, np.ndarray]] = {}
    if short_interest.empty:
        return idx

    si = short_interest.sort_values(["symbol", "settlement_date"])
    settlement_ord = _to_day_ordinal(si["settlement_date"])
    publication_ord = settlement_ord + PUBLICATION_LAG_DAYS

    si = si.assign(settlement_ord=settlement_ord, publication_ord=publication_ord)
    for symbol, grp in si.groupby("symbol", sort=False):
        idx[symbol] = {
            "settlement_ord": grp["settlement_ord"].to_numpy(),
            "publication_ord": grp["publication_ord"].to_numpy(),
            "short_shares": grp["short_shares"].to_numpy(dtype=float),
            "prev_short_shares": grp["prev_short_shares"].to_numpy(dtype=float),
            "adv_shares": grp["adv_shares"].to_numpy(dtype=float),
            "days_to_cover": grp["days_to_cover"].to_numpy(dtype=float),
        }
    return idx


def attach_features(events_df: pd.DataFrame, short_interest: pd.DataFrame | None = None) -> pd.DataFrame:
    """Compute the point-in-time short-interest feature columns.

    Parameters
    ----------
    events_df : must have `ticker` and `event_day` columns. Row order and
        index are preserved in the output (safe to
        `pd.concat([events_df, result], axis=1)`).
    short_interest : canonical table (OUTPUT_COLUMNS). Defaults to loading
        DEFAULT_SHORT_INTEREST_PARQUET if not given.

    Point-in-time rule (see module docstring): for each (ticker, event_day),
    only the LATEST report whose PUBLICATION date (settlement_date +
    PUBLICATION_LAG_DAYS) is STRICTLY BEFORE event_day may be used.
    Enforced via numpy.searchsorted(..., side='left') on each ticker's
    ascending publication_ord array -- the same technique
    tools/earnings_features.py uses for filing dates, applied to the
    publication axis instead of the settlement axis, which is the whole
    point: settlement date alone is NOT point-in-time here.
    """
    if short_interest is None:
        short_interest = pd.read_parquet(DEFAULT_SHORT_INTEREST_PARQUET)

    n = len(events_df)
    out = {c: np.full(n, np.nan, dtype=np.float64) for c in X_COLS}

    idx = _ticker_short_interest_index(short_interest)
    event_tickers = events_df["ticker"].astype(str).to_numpy()
    event_ords = _to_day_ordinal(events_df["event_day"])

    for i in range(n):
        entry = idx.get(event_tickers[i])
        eo = event_ords[i]
        if entry is None or eo != eo:  # unknown ticker, or NaT event_day
            continue

        pub_ord = entry["publication_ord"]
        # side='left': index of the first report whose publication date is
        # >= event_day. Everything before that index -- and only that -- was
        # actually knowable on event_day.
        boundary = int(np.searchsorted(pub_ord, eo, side="left"))
        if boundary == 0:
            continue  # no report published before event_day yet
        j = boundary - 1  # latest eligible report

        out["x_short_days_to_cover"][i] = entry["days_to_cover"][j]

        adv = entry["adv_shares"][j]
        sh = entry["short_shares"][j]
        if adv == adv and adv > 0:
            out["x_short_pct_of_adv"][i] = sh / adv

        prev = entry["prev_short_shares"][j]
        if sh == sh and prev == prev:
            # log1p rather than log(ratio): a handful of reports have
            # prev_short_shares == 0 (newly-shortable name), where a plain
            # log(current/previous) is undefined; log1p degrades gracefully.
            out["x_short_change_2m"][i] = np.log1p(sh) - np.log1p(prev)

        out["x_short_report_age_days"][i] = eo - entry["settlement_ord"][j]

    return pd.DataFrame(out, index=events_df.index)[X_COLS]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _print_report(events_with_feats: pd.DataFrame) -> None:
    n = len(events_with_feats)
    print(f"\n=== short_interest sanity report ({n} rows) ===")
    print("Coverage (fraction of rows with a non-null value):")
    for c in X_COLS:
        frac = events_with_feats[c].notna().mean()
        print(f"  {c}: {frac:.1%}")

    if "adj_21" not in events_with_feats.columns:
        print("\n(adj_21 not present -- skipping quartile tables)")
        return

    for feat in ("x_short_days_to_cover", "x_short_pct_of_adv"):
        valid = events_with_feats.dropna(subset=[feat, "adj_21"])
        print(f"\nMEDIAN adj_21 by quartile of {feat} (medians, not means -- "
              f"adj_21 has a large right tail that would make a mean misleading), n={len(valid)}:")
        if len(valid) < 4:
            print("  (not enough non-null rows to form quartiles)")
            continue
        try:
            q_labels = pd.qcut(valid[feat], 4, duplicates="drop")
        except ValueError:
            print("  (could not form 4 distinct quartiles)")
            continue
        tbl = valid.groupby(q_labels, observed=True)["adj_21"].median()
        print(tbl.to_string())
        spread = tbl.max() - tbl.min()
        verdict = "NON-FLAT" if spread > 0.005 else "flat"
        print(f"  -> spread across quartiles: {spread:.4f} ({verdict})")
    print()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--events", default=DEFAULT_EVENTS,
                    help="Research/events parquet to take tickers (and, for the report, "
                         "event_day/adj_21) from.")
    p.add_argument("--start", default="2018-08-01", help="Earliest settlement date to pull (inclusive).")
    p.add_argument("--end", default=date.today().isoformat(), help="Latest settlement date to pull (inclusive).")
    p.add_argument("--out", default=DEFAULT_SHORT_INTEREST_PARQUET)
    p.add_argument("--refresh", action="store_true", help="Ignore the disk cache and refetch.")
    p.add_argument("--workers", type=int, default=MAX_WORKERS)
    p.add_argument("--verbose", action="store_true")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                         format="%(asctime)s [%(levelname)s] %(message)s")

    log.info("short_interest: reading tickers from %s", a.events)
    events = pd.read_parquet(a.events)
    tickers = set(events["ticker"].dropna().astype(str).unique())
    log.info("short_interest: %d distinct ticker(s)", len(tickers))

    STATS.reset()
    t0 = time.monotonic()

    dates = fetch_settlement_dates(a.start, a.end, refresh=a.refresh)
    log.info("short_interest: %d settlement date(s) in [%s, %s]", len(dates), a.start, a.end)

    raw = fetch_all(dates, refresh=a.refresh, workers=a.workers)
    table = build_table(raw, tickers=tickers)

    elapsed = time.monotonic() - t0
    log.info("short_interest: %d live HTTP request(s) in %.1fs (%.2f req/s)",
              STATS.count, elapsed, (STATS.count / elapsed) if elapsed > 0 else 0.0)

    out_path = a.out if os.path.isabs(a.out) else os.path.join(REPO_ROOT, a.out)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    table.to_parquet(out_path, index=False)
    log.info("short_interest: wrote %s (%d rows, %d symbols)",
              out_path, len(table), table["symbol"].nunique() if len(table) else 0)

    feats = attach_features(events, table)
    assert len(feats) == len(events), "row count changed -- refusing to report"
    out_df = pd.concat([events, feats], axis=1)
    _print_report(out_df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
