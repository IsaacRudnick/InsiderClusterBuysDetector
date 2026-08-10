"""Durable manual split corrections that survive a price-cache re-fetch.

The problem: Yahoo's own split table is missing certain reverse splits (see
split_fingerprint.py's docstring for the full fingerprint methodology). When
that happens, `auto_adjust=True` can never back-adjust the pre-split history,
because yfinance has no record the split occurred. A human can fix the
cached parquet by hand -- but the very next backtest run that re-fetches that
ticker overwrites the fix, because `PriceUniverse._merge_price_frames`
compares the (now-corrected) cached frame against a fresh (still-broken)
Yahoo fetch, reads the correction itself as a spurious "basis change", and
falls back to the raw Yahoo data. This is exactly what happened to DKI on
2026-08-07: a hand-applied fix from 2026-08-04 was silently destroyed by a
routine run three days later, and the corrupted DKI series went on to supply
65% of one strategy's backtested P&L.

The fix has to live somewhere a re-fetch cannot touch. This module is that
somewhere: confirmed corrections are recorded in `price_overrides.json` (a
plain, version-controllable file at the project root, NOT inside
price_cache/), and `apply_price_overrides` re-applies them to a ticker's
frame every time one is loaded from disk or freshly fetched from Yahoo. See
backtest/prices.py's `_load_cached` and `_fetch_and_cache` for the two call
sites -- both the load path and the fetch path need the override, because
`_merge_price_frames` compares one side against the other; if only one side
were corrected, the correction would look like a basis change all over
again and get merged away.

Safety properties this module is responsible for:
  - Idempotent: applying an override to an already-corrected frame is a
    no-op. Each event independently re-checks whether the raw jump it
    describes is STILL PRESENT in the frame (see `_jump_present`) before
    adjusting. A silently double-adjusted series is worse than an
    unadjusted one, so when the jump is not found -- because the frame is
    already corrected, or because Yahoo has since filled the split in
    itself -- the event is skipped and logged at INFO, never applied blind.
  - Unknown tickers, and tickers with no override entries, are a no-op:
    `apply_price_overrides` returns the input frame unchanged.
  - Reuses split_fingerprint.py's own adjustment math (`SplitEvent`,
    `apply_adjustment`) rather than re-implementing the back-adjustment
    arithmetic here.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import date
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)

OVERRIDES_PATH = "price_overrides.json"

# How far an observed close-to-close ratio at the override date may depart
# from the recorded price_ratio and still count as "the raw jump is still
# there". Real data carries an ordinary day-to-day wobble on top of the
# split itself -- split_fingerprint.py's own confirmed DKI example departs
# from a perfectly clean ratio (dollar-volume ratio 1.81, not 1.0) for the
# same reason. 15% comfortably clears that kind of noise while still telling
# "the jump is gone" (ratio near 1.0, effectively 100% off) apart from "the
# jump is still there".
RATIO_MATCH_TOLERANCE = 0.15


@dataclass(frozen=True)
class PriceOverride:
    """One confirmed manual split correction, as recorded in price_overrides.json."""

    ticker: str
    date: date
    price_ratio: float
    who: str = ""
    when_added: str = ""
    why: str = ""
    observed_volume_ratio: Optional[float] = None
    observed_dollar_volume_ratio: Optional[float] = None


# Process-local cache of the parsed overrides file, invalidated by mtime so
# a human editing price_overrides.json mid-session (or a test monkeypatching
# OVERRIDES_PATH) is picked up without a process restart.
_cache: Optional[dict[str, list[PriceOverride]]] = None
_cache_path: Optional[str] = None
_cache_mtime: Optional[float] = None


def _parse(raw: dict) -> dict[str, list[PriceOverride]]:
    out: dict[str, list[PriceOverride]] = {}
    for item in raw.get("overrides", []):
        try:
            ticker = str(item["ticker"]).strip().upper()
            ev = PriceOverride(
                ticker=ticker,
                date=date.fromisoformat(str(item["date"])),
                price_ratio=float(item["price_ratio"]),
                who=str(item.get("who", "")),
                when_added=str(item.get("when_added", "")),
                why=str(item.get("why", "")),
                observed_volume_ratio=(
                    float(item["observed_volume_ratio"])
                    if item.get("observed_volume_ratio") is not None
                    else None
                ),
                observed_dollar_volume_ratio=(
                    float(item["observed_dollar_volume_ratio"])
                    if item.get("observed_dollar_volume_ratio") is not None
                    else None
                ),
            )
        except (KeyError, ValueError, TypeError) as exc:
            log.warning("price_overrides.json: skipping malformed entry %r: %s", item, exc)
            continue
        out.setdefault(ticker, []).append(ev)
    return out


def load_overrides(path: str = OVERRIDES_PATH) -> dict[str, list[PriceOverride]]:
    """Load price_overrides.json, keyed by upper-case ticker.

    Cached per-process and refreshed automatically when the file's mtime
    changes. Returns {} (never raises) if the file is missing or malformed,
    so a broken or absent overrides file degrades to "no overrides" rather
    than breaking price loading.
    """
    global _cache, _cache_path, _cache_mtime

    if not os.path.exists(path):
        _cache, _cache_path, _cache_mtime = {}, path, None
        return _cache

    mtime = os.path.getmtime(path)
    if _cache is not None and _cache_path == path and _cache_mtime == mtime:
        return _cache

    try:
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("Could not read %s: %s", path, exc)
        _cache, _cache_path, _cache_mtime = {}, path, None
        return _cache

    _cache = _parse(raw)
    _cache_path = path
    _cache_mtime = mtime
    return _cache


def _jump_present(df: pd.DataFrame, ev: PriceOverride) -> bool:
    """True when `df` still shows the raw, unadjusted jump `ev` describes.

    Looks up the bar at ev.date and the bar immediately before it in df's
    OWN index (not a fixed calendar offset, so weekends/holidays are handled
    correctly), and checks whether their close ratio is still close to
    ev.price_ratio. If the frame has already been adjusted (by this same
    override on an earlier call, by a prior manual fix, or because Yahoo has
    since filled the split in itself), that ratio collapses to ~1.0 and this
    returns False.
    """
    if "close" not in df.columns:
        return False
    ts = pd.Timestamp(ev.date)
    idx = df.index
    if ts not in idx:
        return False
    pos = idx.get_loc(ts)
    if not isinstance(pos, int):
        # A duplicate-timestamp index returns a slice/array; too ambiguous
        # to trust, so treat as "cannot confirm the jump is there".
        return False
    if pos == 0:
        return False

    prev_close = df["close"].iloc[pos - 1]
    cur_close = df["close"].iloc[pos]
    if prev_close != prev_close or cur_close != cur_close or prev_close <= 0:
        return False

    observed_ratio = float(cur_close) / float(prev_close)
    return abs(observed_ratio - ev.price_ratio) <= RATIO_MATCH_TOLERANCE * ev.price_ratio


def apply_price_overrides(
    ticker: str, df: Optional[pd.DataFrame], path: str = OVERRIDES_PATH
) -> Optional[pd.DataFrame]:
    """Re-apply every confirmed manual split correction for `ticker` to `df`.

    Safe to call on every load and every fetch:
      - A ticker with no override entries (including any ticker not in
        price_overrides.json at all) is a no-op; `df` is returned unchanged.
      - Each event independently checks `_jump_present` before adjusting.
        Events whose jump is not present are skipped and logged at INFO,
        never applied -- this is what keeps a re-load of an
        already-corrected frame from silently double-adjusting it.
      - Events whose jump IS present are applied via
        split_fingerprint.apply_adjustment (the same back-adjustment math
        used by the automated fingerprint scanner) and logged at WARNING,
        naming the ticker and date, so an applied override is visible in
        run output.

    Imports split_fingerprint lazily to avoid a circular import: this module
    is imported from backtest/prices.py, and split_fingerprint.py itself
    imports CACHE_DIR from backtest/prices.py.
    """
    if df is None or df.empty:
        return df

    events = load_overrides(path).get(ticker.upper())
    if not events:
        return df

    from .split_fingerprint import SplitEvent, apply_adjustment  # local: see docstring

    to_apply: list[SplitEvent] = []
    for ev in events:
        if not _jump_present(df, ev):
            log.info(
                "%s: manual override at %s not applied -- the raw jump "
                "(price_ratio~%.4f) is not present in this frame (already "
                "adjusted, or the source data no longer shows it)",
                ticker, ev.date, ev.price_ratio,
            )
            continue
        log.warning(
            "%s: applying manual split override at %s (price_ratio=%.4f, "
            "added by %s)",
            ticker, ev.date, ev.price_ratio, ev.who or "unknown",
        )
        to_apply.append(
            SplitEvent(
                date=ev.date,
                price_ratio=ev.price_ratio,
                volume_ratio=(
                    ev.observed_volume_ratio
                    if ev.observed_volume_ratio is not None
                    else float("nan")
                ),
                dollar_volume_ratio=(
                    ev.observed_dollar_volume_ratio
                    if ev.observed_dollar_volume_ratio is not None
                    else float("nan")
                ),
                # Reverse-split convention (price_ratio >= 1): inferred_factor
                # equals price_ratio itself. Every seeded override is a
                # reverse split; see apply_adjustment's own direction logic.
                inferred_factor=ev.price_ratio,
            )
        )

    if not to_apply:
        return df
    return apply_adjustment(df, to_apply)
