"""Repair price_cache/*.parquet files that mix adjustment bases.

SUPERSEDED. Do not use this to fix the >=3x overnight jumps in the cache.

The premise below is wrong for that problem. A refetch does NOT remove those
jumps, because Yahoo serves the same broken series again. A single fresh
request for DRIO reproduces its 2019-11-18 jump at ratio 20.77, and one
request cannot contain a merge artifact. The true cause is an incomplete
Yahoo split table, so auto_adjust never back-adjusts the earlier prices.
See RESEARCH_NOTES.md, "Unadjusted splits corrupt the price cache", and use
backtest/splits.py instead.

The scan function `find_suspect_tickers` is still useful as a detector. The
repair path needs --i-know-this-is-superseded before it will touch the cache.

A cache file gets corrupted when a split happens between two fetches and
the old merge code welds stale-basis rows onto fresh-basis rows (see
backtest/prices.py, PriceUniverse._merge_price_frames). This script finds
files with an implausible overnight close-to-close jump and repairs them
by refetching the full date range in one request, so the result sits on a
single consistent basis.

Resumable: a repaired ticker naturally drops out of find_suspect_tickers on
the next scan, so an interrupted or partial run is safe to just run again.

Usage:
    python tools/repair_price_cache.py --dry-run
    python tools/repair_price_cache.py --limit 20
    python tools/repair_price_cache.py
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import shutil
import sys
from datetime import date

import pandas as pd

# Run directly (`python tools/repair_price_cache.py ...`), the interpreter
# puts this file's own directory (tools/) on sys.path[0], not the repo root
# -- so `from backtest.prices import ...` below would fail without this.
# Harmless no-op when this module is instead imported normally (e.g. `from
# tools import repair_price_cache`), since the repo root is already on
# sys.path in that case.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from backtest.prices import (
    CACHE_DIR,
    DOWNLOAD_BATCH,
    _AdaptiveThrottle,
    _download_batch,
    _download_serial,
    _load_needs_refetch,
    _save_needs_refetch,
)

log = logging.getLogger(__name__)

# Coarser than backtest.prices._BASIS_CHANGE_TOLERANCE (2%). That constant
# guards a live merge, catching a split the moment it appears. This constant
# scans an already-written cache file for a jump so large it can only be a
# leftover basis mismatch, not ordinary volatility.
JUMP_RATIO_THRESHOLD = 3.0

# Non-ticker files and directories that live alongside the ticker parquet
# files in price_cache/. Most of these do not match "*.parquet" anyway, but
# the check is kept explicit so the scan never depends on that being true.
_SKIP_NAMES = {"_meta", "_missing.json", "_yf_tz", "_needs_refetch.json"}


def _iter_ticker_parquet_paths(cache_dir: str):
    for path in sorted(glob.glob(os.path.join(cache_dir, "*.parquet"))):
        if os.path.basename(path) in _SKIP_NAMES:
            continue
        yield path


def find_suspect_tickers(cache_dir: str) -> list[tuple[str, str, float]]:
    """Scan every ticker parquet in cache_dir for an implausible overnight jump.

    Returns a list of (ticker, boundary_date_iso, ratio) tuples, one entry
    per affected ticker (the first jump found is enough to flag a ticker for
    a full repair). `ratio` is close[i]/close[i-1] or its reciprocal,
    whichever is at least one, so both a fabricated spike and a fabricated
    collapse are caught. `boundary_date_iso` is the date the jump lands on.
    """
    out: list[tuple[str, str, float]] = []
    for path in _iter_ticker_parquet_paths(cache_dir):
        ticker = os.path.splitext(os.path.basename(path))[0]
        try:
            df = pd.read_parquet(path)
        except Exception as exc:
            log.warning("Could not read %s: %s", path, exc)
            continue
        if df is None or df.empty or "close" not in df.columns:
            continue
        df = df.sort_index()
        closes = df["close"].to_numpy(dtype=float)
        idx = df.index
        for i in range(1, len(closes)):
            prev, cur = closes[i - 1], closes[i]
            if prev != prev or cur != cur or prev <= 0 or cur <= 0:
                continue
            fwd = cur / prev
            ratio = fwd if fwd >= 1.0 else 1.0 / fwd
            if ratio >= JUMP_RATIO_THRESHOLD:
                boundary = idx[i]
                boundary_iso = (
                    boundary.date().isoformat()
                    if hasattr(boundary, "date")
                    else str(boundary)
                )
                out.append((ticker, boundary_iso, float(ratio)))
                break
    return out


def _earliest_cached_date(ticker: str, cache_dir: str) -> date | None:
    path = os.path.join(cache_dir, f"{ticker}.parquet")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        log.warning("Could not read %s for its earliest date: %s", path, exc)
        return None
    if df is None or df.empty:
        return None
    return pd.DatetimeIndex(df.index).min().date()


# A refetch that comes back materially shorter than what is cached is more
# likely a bad response than a real correction, so it does not get to
# overwrite good history. Delisted tickers are the usual cause: yfinance may
# return only a recent stub for a symbol that has since been reused.
MIN_KEEP_ROW_FRACTION = 0.9

BACKUP_DIRNAME = "_pre_repair_backup"


def _backup_original(ticker: str, cache_dir: str) -> None:
    """Copy a ticker's cache file aside before it is overwritten.

    The repair replaces a file outright rather than merging, so without this
    there is no way back from a bad refetch. Keeping the first backup only
    means a second run cannot overwrite the pristine original with an
    already-repaired copy.
    """
    src = os.path.join(cache_dir, f"{ticker}.parquet")
    if not os.path.exists(src):
        return
    backup_dir = os.path.join(cache_dir, BACKUP_DIRNAME)
    os.makedirs(backup_dir, exist_ok=True)
    dst = os.path.join(backup_dir, f"{ticker}.parquet")
    if os.path.exists(dst):
        return
    shutil.copy2(src, dst)


def _cached_row_count(ticker: str, cache_dir: str) -> int:
    path = os.path.join(cache_dir, f"{ticker}.parquet")
    if not os.path.exists(path):
        return 0
    try:
        return len(pd.read_parquet(path, columns=["close"]))
    except Exception:
        return 0


def _atomic_save(ticker: str, df: pd.DataFrame, cache_dir: str) -> None:
    """Write to a temp file then rename over the target.

    A rename is atomic on the same filesystem, so an interrupted run never
    leaves a truncated or half-written parquet file where the real cache
    file is expected. This also keeps repair_price_cache.py from depending
    on backtest.prices' own CACHE_DIR-bound save helpers, so cache_dir stays
    a real parameter instead of a hardcoded path.
    """
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{ticker}.parquet")
    tmp_path = path + ".tmp"
    df.to_parquet(tmp_path)
    os.replace(tmp_path, path)

    meta_dir = os.path.join(cache_dir, "_meta")
    os.makedirs(meta_dir, exist_ok=True)
    meta_path = os.path.join(meta_dir, f"{ticker}.json")
    meta_tmp = meta_path + ".tmp"
    with open(meta_tmp, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "ticker": ticker,
                "first_date": df.index.min().date().isoformat(),
                "last_date": df.index.max().date().isoformat(),
                "repaired": True,
            },
            fh,
        )
    os.replace(meta_tmp, meta_path)


def repair_tickers(
    tickers: list[str], cache_dir: str, throttle: _AdaptiveThrottle
) -> tuple[list[str], list[str]]:
    """Refetch each ticker's full history in one request and overwrite its cache.

    One fetch means one consistent adjustment basis, so this does not merge
    with what is on disk, it replaces it outright. Returns
    (repaired_tickers, failed_tickers). Failed tickers are left alone and
    will be picked up again on the next run.
    """
    repaired: list[str] = []
    failed: list[str] = []
    today = date.today()

    # Bucket by YEAR, not by exact earliest date. Grouping on the exact date
    # put nearly every ticker in a group of its own, so the batch download
    # degenerated to one request per ticker. Starting a fetch earlier than
    # strictly needed is harmless, since the response simply begins at the
    # listing date, so each bucket can use the earliest date it contains.
    year_groups: dict[int, list[str]] = {}
    earliest_in_year: dict[int, date] = {}
    for t in tickers:
        first = _earliest_cached_date(t, cache_dir)
        if first is None:
            # No existing cache to anchor a start date. Fall back to a
            # generous lookback so the refetch still gets full history.
            first = date(1990, 1, 1)
        year_groups.setdefault(first.year, []).append(t)
        if first < earliest_in_year.get(first.year, date(9999, 12, 31)):
            earliest_in_year[first.year] = first

    groups: dict[date, list[str]] = {
        earliest_in_year[y]: ts for y, ts in year_groups.items()
    }

    for first_date, group_tickers in groups.items():
        for i in range(0, len(group_tickers), DOWNLOAD_BATCH):
            batch = group_tickers[i : i + DOWNLOAD_BATCH]
            fetched = _download_batch(batch, first_date, today)
            blanks = [t for t in batch if t not in fetched or fetched[t].empty]
            if blanks:
                fetched.update(_download_serial(blanks, first_date, today))
            for t in batch:
                df = fetched.get(t)
                if df is None or df.empty:
                    log.warning(
                        "%s: repair refetch returned no data, will retry next run", t
                    )
                    failed.append(t)
                    continue
                cached_rows = _cached_row_count(t, cache_dir)
                if cached_rows and len(df) < cached_rows * MIN_KEEP_ROW_FRACTION:
                    # Refusing here is the conservative call. A corrupt file
                    # with full history is still repairable later; a good file
                    # overwritten by a truncated stub is not.
                    log.warning(
                        "%s: refetch returned %d rows against %d cached; "
                        "refusing to overwrite, will retry next run",
                        t, len(df), cached_rows,
                    )
                    failed.append(t)
                    continue
                _backup_original(t, cache_dir)
                _atomic_save(t, df, cache_dir)
                log.info("%s: repaired, %d rows written", t, len(df))
                repaired.append(t)
            throttle.observe(0, len(batch))
            throttle.sleep()

    return repaired, failed


def _print_sample(label: str, items: list, n: int = 10) -> None:
    for item in items[:n]:
        print(f"  {item}")
    if len(items) > n:
        print(f"  ... and {len(items) - n} more")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Find and repair price_cache parquet files whose merge history "
            "mixed adjustment bases (a split merged onto stale data)."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and report only. Changes nothing on disk.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Cap the number of tickers repaired per run. In --dry-run mode "
            "the same limit also caps the worklist shown in the report, so "
            "the report previews exactly what a real run would touch."
        ),
    )
    parser.add_argument(
        "--i-know-this-is-superseded",
        action="store_true",
        help=(
            "Required to actually write. Refetching does not fix the >=3x "
            "jumps; see the module docstring and RESEARCH_NOTES.md."
        ),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cache_dir = CACHE_DIR
    suspects = find_suspect_tickers(cache_dir)
    suspect_tickers = [t for t, _, _ in suspects]

    needs_refetch = _load_needs_refetch()
    needs_refetch_tickers = list(needs_refetch.keys())

    worklist: list[str] = []
    for t in suspect_tickers + needs_refetch_tickers:
        if t not in worklist:
            worklist.append(t)

    print(
        f"Found {len(suspect_tickers)} ticker(s) with a >= "
        f"{JUMP_RATIO_THRESHOLD}x close-to-close jump in the cache."
    )
    _print_sample(
        "suspects",
        [f"{t}: ratio={r:.2f}x at {b}" for t, b, r in suspects],
    )
    print(
        f"Found {len(needs_refetch_tickers)} ticker(s) flagged for a "
        f"disjoint-merge refetch."
    )
    _print_sample(
        "needs_refetch",
        [f"{t}: {needs_refetch[t].get('reason', 'unknown')}" for t in needs_refetch_tickers],
    )

    total = len(worklist)
    if args.limit is not None:
        worklist = worklist[: args.limit]
    capped = args.limit is not None and total > len(worklist)
    print(
        f"Worklist: {len(worklist)} of {total} ticker(s)"
        + (" (capped by --limit)" if capped else "")
    )

    if args.dry_run:
        print("Dry run: no changes made.")
        return 0

    if not args.i_know_this_is_superseded:
        print(
            "REFUSING TO WRITE. Refetching does not remove these jumps: Yahoo\n"
            "serves the same series again, because its split table is missing\n"
            "the split. Use backtest/splits.py. See RESEARCH_NOTES.md,\n"
            "'Unadjusted splits corrupt the price cache'.\n"
            "Pass --i-know-this-is-superseded to override."
        )
        return 1

    if not worklist:
        print("Nothing to repair.")
        return 0

    throttle = _AdaptiveThrottle()
    repaired, failed = repair_tickers(worklist, cache_dir, throttle)

    if repaired:
        remaining = {t: e for t, e in needs_refetch.items() if t not in repaired}
        if len(remaining) != len(needs_refetch):
            _save_needs_refetch(remaining)

    print(
        f"Repaired {len(repaired)} ticker(s). {len(failed)} failed and will "
        f"be retried next run."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
