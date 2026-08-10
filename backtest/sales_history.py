"""Point-in-time index of insider SALE transactions, extracted straight from
parse_cache/ -- the same Form 4/4A JSON cache insider_cluster_buys.py already
builds and reuses for the (code-P, acquired) BUY side of the pipeline.

Why this doesn't need a fresh SEC scrape
-----------------------------------------
insider_cluster_buys.extract_qualifying_rows only keeps code-P acquisitions
(QUALIFYING_CODES = {"P"}); everything else in a filing's transaction list is
discarded at that step. But parse_cache/ itself holds the FULL parsed
transaction list per filing -- every code, not just P -- because parsing
happens once per accession and the whole filing is cached. So building a
sale-side dataset is a LOCAL reprocessing pass over parse_cache/*.json, not a
network operation. This module makes zero requests.

Which codes count as a "sale" here
-----------------------------------
Per SEC Form 4 transaction coding, code S is the open-market/private SALE
code (acquired_disposed == "D"). Two other disposition codes show up in the
cache far more often than a stray typo would explain, and are deliberately
EXCLUDED:

  F -- payment of exercise price or tax liability by delivering/withholding
       securities. This fires automatically alongside an option exercise or
       RSU vest (a "sell-to-cover"); the insider did not choose to sell for
       market reasons, the broker/plan did the arithmetic for them. Counting
       it as bearish selling pressure would misread routine plan mechanics
       as a sentiment signal.
  D -- disposition to the issuer (e.g., an option expiring unexercised, or
       shares surrendered back to the company). Also not a discretionary
       market sale -- there is no counterparty buying on the open market.

A sample of 20,000 cached filings (see the research writeup) found S is
already the single most common transaction code in the cache (more common
than A, F, or M), with acquired_disposed == "D" on 9,618/9,630 S rows --
confirming code S is both plentiful and, on its own, already almost entirely
disposals. Requiring acquired_disposed == "D" alongside code == "S" mirrors
exactly how insider_cluster_buys.extract_qualifying_rows requires
acquired_disposed == "A" alongside code == "P" on the buy side, and drops
the small residual (~0.1%) of code-S rows that don't fit the "D" pattern
(most likely data-entry noise in the source filings).

Owner attribution
-------------------
A filing's <reportingOwner> list is not exploded per-transaction in the raw
XML (see insider_cluster_buys.parse_ownership_xml): the parse cache carries
one flat `owners` list per filing, not one owner per transaction. This
module uses owners[0] for every transaction in a filing, identical to
insider_cluster_buys.extract_qualifying_rows's existing "owner = owners[0]"
convention on the buy side -- the overwhelming majority of Form 4s report a
single reporting owner, and this keeps the two attribution rules consistent
with each other rather than introducing a second, different heuristic here.

Output schema (one row per qualifying sale transaction)
-----------------------------------------------------------
adsh, issuer_cik, ticker, owner_cik, owner_name, owner_key (owner_cik or
owner_name, matching backtest.research._aggregate_owners's key), is_director,
is_officer, is_ten_percent_owner, transaction_date (date), filing_date
(date), shares, price_per_share, value (shares * price_per_share).

transaction_date/filing_date are plain datetime.date objects, matching
backtest.history.build_events_df's convention -- python date objects survive
a to_parquet/read_parquet round trip unchanged (pyarrow infers date32), so
no dtype massaging is needed on load.

Point-in-time note: this module only extracts and tags each sale with its
OWN filing_date and transaction_date. It does not itself decide what is
"visible" to any given cluster event -- that gating (filing_date <= D) is
backtest.research's job (see its "Section F leakage note"). Keeping the
extraction and the visibility rule in separate modules means the expensive
part (the 1.3M-file scan) never needs rerunning just because the visibility
rule changes.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from typing import Optional

import pandas as pd

import insider_cluster_buys as ics

log = logging.getLogger(__name__)

PARSE_CACHE_DIR = ics.PARSE_CACHE_DIR  # "parse_cache" -- reuse the existing constant

# Data goes to research_data/, matching backtest.research.save_research_dataset's
# out_dir convention (research_data/, not research/ -- see that module's note).
SALES_OUT_DIR = "research_data"
SALES_GLOB = "sales_cache_*rows_*.parquet"

SALE_CODE = "S"
SALE_ACQUIRED_DISPOSED = "D"

LOG_EVERY = 100_000
DEFAULT_MAX_WORKERS = 16

_SALE_COLUMNS = [
    "adsh", "issuer_cik", "ticker", "owner_cik", "owner_name", "owner_key",
    "is_director", "is_officer", "is_ten_percent_owner",
    "transaction_date", "filing_date", "shares", "price_per_share", "value",
]


def _parse_filing_date(raw: Optional[str]) -> Optional[date]:
    """parse_cache stores filing_date as YYYYMMDD (see insider_cluster_buys.
    parse_ownership_xml's `filing["date_filed"]`), unlike transaction dates."""
    if not raw:
        return None
    try:
        return datetime.strptime(raw, "%Y%m%d").date()
    except ValueError:
        return None


def _parse_tx_date(raw: Optional[str]) -> Optional[date]:
    if not raw:
        return None
    try:
        return datetime.strptime(raw, "%Y-%m-%d").date()
    except ValueError:
        return None


def _load_one(path: str) -> list[dict]:
    """Parse one cached filing JSON and return its qualifying SALE rows
    (possibly empty). Any unreadable/malformed/non-Form-4 file, or one with
    no code-S/acquired_disposed-D transactions, yields an empty list --
    counted in aggregate by the caller's progress log, never raised. A
    handful of parse_cache files are known to have a stray trailing byte
    from an old interrupted write (see the research writeup); skipping those
    silently here is the correct behavior for the same reason
    insider_cluster_buys.py treats a parse failure as a logged error, not a
    crash.
    """
    try:
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
    except (OSError, ValueError):
        return []

    if not (d.get("form_type") or "").startswith("4"):
        return []
    filing_date = _parse_filing_date(d.get("filing_date"))
    if filing_date is None:
        return []

    issuer = d.get("issuer") or {}
    issuer_cik = issuer.get("cik") or ""
    if not issuer_cik:
        return []
    ticker = issuer.get("ticker") or ""

    owners = d.get("owners") or [{}]
    owner = owners[0] or {}
    owner_cik = owner.get("cik") or ""
    owner_name = owner.get("name") or ""
    owner_key = owner_cik or owner_name
    is_director = bool(owner.get("is_director"))
    is_officer = bool(owner.get("is_officer"))
    is_ten_pct = bool(owner.get("is_ten_percent_owner"))

    adsh = d.get("adsh", "")
    rows: list[dict] = []
    for tx in d.get("transactions") or []:
        if (tx.get("code") or "").upper() != SALE_CODE:
            continue
        if (tx.get("acquired_disposed") or "").upper() != SALE_ACQUIRED_DISPOSED:
            continue
        tx_date = _parse_tx_date(tx.get("date"))
        if tx_date is None:
            continue
        shares = float(tx.get("shares") or 0.0)
        price = float(tx.get("price_per_share") or 0.0)
        rows.append({
            "adsh": adsh,
            "issuer_cik": issuer_cik,
            "ticker": ticker,
            "owner_cik": owner_cik,
            "owner_name": owner_name,
            "owner_key": owner_key,
            "is_director": is_director,
            "is_officer": is_officer,
            "is_ten_percent_owner": is_ten_pct,
            "transaction_date": tx_date,
            "filing_date": filing_date,
            "shares": shares,
            "price_per_share": price,
            "value": shares * price,
        })
    return rows


def build_sales_cache(
    parse_cache_dir: str = PARSE_CACHE_DIR, max_workers: int = DEFAULT_MAX_WORKERS,
) -> pd.DataFrame:
    """Scan every cached Form 4/4A filing JSON under `parse_cache_dir` and
    extract discretionary sale transactions (code S, acquired_disposed D --
    see module docstring). Purely local: reads files already on disk, makes
    zero network requests.

    This is the expensive step (1.3M+ small-file reads, a few minutes even
    threaded). Callers should persist the result via save_sales_cache and
    reload it with load_or_build_sales_cache rather than calling this
    directly on every research build.
    """
    paths = glob.glob(os.path.join(parse_cache_dir, "*.json"))
    n_total = len(paths)
    log.info("build_sales_cache: scanning %d cached filing(s) under %s/", n_total, parse_cache_dir)
    t0 = time.monotonic()

    rows: list[dict] = []
    n_done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for result in ex.map(_load_one, paths, chunksize=256):
            if result:
                rows.extend(result)
            n_done += 1
            if n_done % LOG_EVERY == 0:
                log.info(
                    "build_sales_cache: scanned %d/%d filings (%.1fs elapsed, %d sale row(s) so far)",
                    n_done, n_total, time.monotonic() - t0, len(rows),
                )

    elapsed = time.monotonic() - t0
    df = pd.DataFrame(rows, columns=_SALE_COLUMNS)
    log.info(
        "build_sales_cache: done in %.1fs -- %d filing(s) scanned, %d sale transaction(s) "
        "extracted (code=S, acquired_disposed=D), %d distinct issuer(s)",
        elapsed, n_total, len(df), df["issuer_cik"].nunique() if not df.empty else 0,
    )
    return df


def save_sales_cache(df: pd.DataFrame, out_dir: str = SALES_OUT_DIR) -> str:
    """Write `df` to a parquet file under `out_dir`, named
    sales_cache_<n_rows>rows_<YYYYMMDD>.parquet. Returns the path written.

    Writes to a `.tmp` sibling first, then os.replace()s it over the real
    path -- same atomic write-then-rename convention as
    backtest.research.save_research_dataset / split_fingerprint._atomic_save
    / repair_price_cache._atomic_save, so an interrupted write can never
    leave a truncated parquet sitting at the real path.
    """
    os.makedirs(out_dir, exist_ok=True)
    fname = f"sales_cache_{len(df)}rows_{date.today():%Y%m%d}.parquet"
    path = os.path.join(out_dir, fname)
    tmp_path = path + ".tmp"
    df.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, path)
    log.info("Wrote sales cache: %s (%d rows, %d cols)", path, len(df), len(df.columns))
    return path


def load_sales_cache(path: str) -> pd.DataFrame:
    """Read a sales cache parquet back into a DataFrame."""
    df = pd.read_parquet(path)
    log.info("Loaded sales cache: %s (shape=%s)", path, df.shape)
    return df


def _resolve_latest(out_dir: str, pattern: str) -> Optional[str]:
    matches = glob.glob(os.path.join(out_dir, pattern))
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


def load_or_build_sales_cache(
    out_dir: str = SALES_OUT_DIR,
    parse_cache_dir: str = PARSE_CACHE_DIR,
    max_workers: int = DEFAULT_MAX_WORKERS,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """Reuse the most recently written sales_cache_*.parquet under `out_dir`
    (by mtime) when one exists and force_rebuild is False. Otherwise scans
    parse_cache_dir from scratch via build_sales_cache and persists the
    result via save_sales_cache, so the expensive 1.3M-file scan is paid at
    most once -- every later call (this process or a future one) reuses the
    cached parquet instead.
    """
    if not force_rebuild:
        existing = _resolve_latest(out_dir, SALES_GLOB)
        if existing is not None:
            log.info("load_or_build_sales_cache: reusing cached sales index %s", existing)
            return load_sales_cache(existing)

    df = build_sales_cache(parse_cache_dir=parse_cache_dir, max_workers=max_workers)
    save_sales_cache(df, out_dir=out_dir)
    return df
