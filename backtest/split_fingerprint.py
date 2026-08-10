"""Detect and locally repair unadjusted splits in price_cache/*.parquet.

Some tickers in price_cache carry a fabricated overnight price jump because
Yahoo's own split table is incomplete: `auto_adjust=True` never back-adjusts
the pre-split history for a split Yahoo does not know happened. A backtest
that reads the raw series books that jump as a real +N00% return.

repair_price_cache.py already detects a >=3x overnight jump, but its fix is
to re-fetch from Yahoo -- which cannot work here, because Yahoo serves the
same broken series again. This module repairs LOCALLY instead: it infers the
split from the SHAPE of the jump (price and volume move in opposite
directions, because a split changes share count, not company value) and
back-adjusts the cached file in place. No network access.

Confirmed example, DKI:
    2026-05-08  close 0.336  volume 636,700
    2026-05-11  close 5.610  volume  68,900   (open 5.220)
    price ratio x16.70, volume ratio x0.108, dollar-volume ratio x1.81

Fingerprint used to tell a fake jump from a real one:
    reverse split: price up N x AND volume down ~1/N x; dollar volume
                   (price * volume) is roughly preserved, because the trade
                   value did not change, only how many shares represent it.
    real move:     price up N x AND volume UP -- a real squeeze or news
                   event brings a volume explosion, not a volume collapse.

Four verified real moves that must NOT be touched (see RESEARCH_NOTES.md and
the task that produced this module for the source numbers):
    XHLD 2025-03-19  price x2.72   volume x45.2
    XHLD 2026-01-27  price x2.77   volume x1038.7
    DKI  2026-02-02  price x2.85   volume x3.67
    ADTX 2026-06-30  price x3.00   volume x1.84
In every one of these, volume moved the SAME direction as price (up), which
is exactly the opposite of what a share-count change would produce. That
direction check alone is what keeps this module from "fixing" them; see
_evaluate_jump below.

A second manual review pass (see the task that produced this docstring
update) found a further, more dangerous false-positive class: real crashes
whose price-DOWN/volume-UP shape passes the forward-split fingerprint just
as cleanly as a genuine forward split would. Confirmed real crashes, NOT
splits, all misread by the pre-review forward-split path:
    SBET 2025-06-13  price x0.28   volume x3.13   $vol x0.89  real crash
    NFE  2025-05-15  price x0.37   volume x9.15   $vol x3.39  real crash
    SKIN 2023-11-14  price x0.36   volume x6.41   $vol x2.28  real crash
    ADTX 2026-06-17  price x0.36   volume x1.36   $vol x0.49  real dilution
    MLTX 2025-09-29  price x0.10   volume x18.21  $vol x1.83  trial-failure crash
    ATYR 2025-09-15  price x0.17   volume x3.92   $vol x0.66  trial-failure crash
Note SBET's dollar-volume ratio, 0.89, is CLOSER to a perfectly-preserved
1.0 than DKI's own confirmed real split (0.94, see above). No dollar-volume
band can separate "real reverse split" from "real crash" using this one
signal: tightening it enough to exclude SBET would also exclude DKI. Because
no genuine forward split has turned up anywhere in this cache to calibrate
against, forward-split detection defaults OFF (ALLOW_FORWARD_SPLITS_DEFAULT)
and is opt-in only; see _evaluate_jump for the gate and
FORWARD_DOLLAR_VOLUME_RATIO_TOLERANCE for the (unvalidated) tolerance it
uses if enabled.

A third class, unrelated to direction, is sub-penny OTC noise: tickers like
AAGR/VAXX/ENSV oscillate day to day between two sub-cent closes (e.g. 0.0099
and 0.0005) on a few hundred to a couple thousand shares of volume, which
trips the price-ratio gate every single day without being a split, a crash,
or anything tradeable. MIN_PRE_JUMP_CLOSE and MIN_PRE_JUMP_DOLLAR_VOLUME
screen these out with a distinct "untradeable" verdict before the fingerprint
logic ever runs.

Usage:
    python backtest/split_fingerprint.py --scan                 # report only, writes nothing
    python backtest/split_fingerprint.py --scan --out report.csv
    python backtest/split_fingerprint.py --apply                # backs up, then writes
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import math
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

# Run directly (`python backtest/split_fingerprint.py ...`), the interpreter
# puts this file's own directory (backtest/) on sys.path[0], not the repo
# root -- so the absolute `backtest.prices` import below would fail without
# this. Harmless no-op when this module is instead imported normally (e.g.
# `from backtest import split_fingerprint`), since the repo root is already
# on sys.path in that case.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from backtest.prices import CACHE_DIR

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

# A boundary only gets a closer look once the raw overnight close ratio (or
# its reciprocal, for a forward split) clears this. 2.5x is below
# repair_price_cache.JUMP_RATIO_THRESHOLD / backtest.splits.CLOSE_RATIO_THRESHOLD
# (both 3.0) on purpose: this module has its own, stricter confirmation step
# (the volume fingerprint below), so it can afford to look at more candidates
# without the false-positive rate rising, and the task that specifies this
# module calls for 2.5 explicitly.
PRICE_RATIO_THRESHOLD = 2.5

# Volume on a single bar is noisy: comparing only the bar right before the
# jump to the bar right after can be thrown off by one quiet day or one
# unrelated high-volume day next to the boundary. A short median window
# smooths that out. It is kept SHORT (10 bars, half of the 20-bar lookback
# backtest.prices.median_dollar_volume already uses elsewhere in this repo)
# so that when two splits land close together, the window for one boundary
# does not reach across and swallow bars that belong to the other.
VOLUME_MEDIAN_WINDOW = 10

# A confirmed split needs at least this many bars of clean context on BOTH
# sides of the boundary, so the median window above never degenerates to a
# handful of points (which would just reintroduce the single-bar noise
# problem it exists to fix). A boundary inside the first or last few rows of
# a ticker's history fails this and is reported as "insufficient_context",
# never as "split".
MIN_CONTEXT_BARS = 3

# Below this many total rows, a ticker's history is too short to say
# anything reliable about a "before" and "after" regime at all.
MIN_ROWS = 10

# Reverse-split arithmetic: price_ratio * volume_ratio is exactly 1.0 for a
# split that changes only the share count, since dollar volume traded is the
# same before and after. Real data never lands on exactly 1.0 -- DKI's own
# confirmed fake split measures 1.80 (16.70 * 0.108), not 1.0, because an
# ordinary day-to-day price/volume wobble rides on top of the split itself.
# The tolerance must clear DKI's 1.80 with room to spare while rejecting the
# smallest of the four real moves, ADTX at 3.00 * 1.84 = 5.52. A band of
# [1/3, 3] does both: DKI sits comfortably inside it (1.80 < 3.0) and every
# real move sits clearly outside it (ADTX 5.52, DKI-real 10.46, XHLD 122.9
# and 2876.6), so there is more than 3x of headroom on the accepted side and
# nearly 2x of margin before the nearest rejected case.
DOLLAR_VOLUME_RATIO_TOLERANCE = 3.0

# How far volume must FALL for a price jump of factor N to count as a reverse
# split. A 1-for-N reverse split destroys N-1 of every N shares, so volume
# should drop about N-fold. Requiring the full N-fold drop rejects genuine
# splits, because split days often trade heavily and volume is noisy. This
# exponent requires a drop of N ** exponent instead: at 0.5, volume must fall
# by at least sqrt(N).
#
# Calibrated against every confirmed case in this cache. Kept splits and the
# ceiling each had to clear:
#   DKI  x16.70 needs <=0.245, has 0.056     OPAD x8.34 needs <=0.346, has 0.122
#   FFAI x96.43 needs <=0.102, has 0.030     CMCT x9.88 needs <=0.318, has 0.114
# Rejected real market moves, whose volume barely moved at all:
#   OCGN x3.01 needs <=0.576, has 0.974      CHRD x2.70 needs <=0.608, has 0.996
#   AHT  x2.55 needs <=0.626, has 0.914      CNVS x2.77 needs <=0.601, has 0.836
#   VISL x2.65 needs <=0.614, has 0.841      PFSA x2.85 needs <=0.592, has 0.885
# Every kept case clears by 2x or more; every rejected case misses by a wide
# margin. There is no confirmed case anywhere near the boundary.
VOLUME_FALL_EXPONENT = 0.5

# Forward-split detection (price DOWN, volume UP) is OFF by default. A
# manual review found six confirmed real crashes/dilutions with exactly
# this shape (see module docstring); the closest of them, SBET at a
# dollar-volume ratio of 0.89, sits nearer to a perfectly-preserved 1.0
# than DKI's own confirmed real split (0.94). That means the dollar-volume
# signal this module relies on cannot separate "real forward split" from
# "real crash" here -- tightening the band enough to exclude SBET would
# also exclude DKI-shaped splits. No genuine forward split has been found
# anywhere in this cache to calibrate a safe threshold against, so rather
# than ship a rule that would eventually erase a real crash, forward-split
# detection stays opt-in. Pass allow_forward_splits=True (or
# --allow-forward-splits on the CLI) to turn it on.
ALLOW_FORWARD_SPLITS_DEFAULT = False

# Tolerance used ONLY when allow_forward_splits=True. Deliberately far
# tighter than DOLLAR_VOLUME_RATIO_TOLERANCE (1.15x vs 3.0x) so it at least
# excludes the loosest confirmed false positives (SKIN 2.28, NFE 3.39,
# MLTX 1.83) -- but it still does NOT clear SBET (0.89, inside [0.87,
# 1.15]), which is exactly why the feature defaults off above instead of
# relying on this number to do the job by itself. Treat this as an
# unvalidated starting point for opt-in use, not a proven-safe threshold.
FORWARD_DOLLAR_VOLUME_RATIO_TOLERANCE = 1.15

# Tradeability screen, item 2: sub-penny OTC tickers (AAGR, VAXX, ENSV, ...)
# oscillate between two sub-cent closes on trivial volume, tripping the
# price-ratio gate every day without being a split. $0.01 is the standard
# "sub-penny stock" line and cleanly separates AAGR-shaped noise (closes of
# 0.0099 / 0.0005) from every real ticker in the confirmed-split table
# above (all >= $0.10). 231 of the 721 raw candidates found by --scan had a
# pre-jump close under this line.
MIN_PRE_JUMP_CLOSE = 0.01

# Matches backtest.engine.LIQUIDITY_FLOOR (500_000): that constant is the
# minimum 20-day median dollar volume the backtest already requires before
# it will buy a ticker at all. A "split" repair on a ticker that never
# clears that floor cannot change any trade the backtest actually takes, so
# there is no upside to guessing at its fingerprint -- only downside if the
# guess is wrong. 525 of the 721 raw candidates found by --scan were below
# this floor.
MIN_PRE_JUMP_DOLLAR_VOLUME = 500_000

BACKUP_DIRNAME = "_pre_split_fingerprint_backup"

_SKIP_NAMES = {"_meta", "_missing.json", "_yf_tz", "_needs_refetch.json"}

Verdict = str  # "split" | "real_move" | "ambiguous" | "insufficient_context" | "untradeable"


@dataclass(frozen=True)
class SplitEvent:
    """One confirmed unadjusted split boundary for a single ticker frame.

    `price_ratio` is the raw overnight close ratio (close[i] / close[i-1]):
    >1 for a reverse split (price jumped up), <1 for a forward split (price
    dropped). `inferred_factor` is always >= 1 -- the "N" in a "1-for-N"
    reverse split or an "N-for-1" forward split -- which direction to use in
    apply_adjustment is recovered from the sign of price_ratio (see there).
    """

    date: date
    price_ratio: float
    volume_ratio: float
    dollar_volume_ratio: float
    inferred_factor: float


@dataclass(frozen=True)
class _BoundaryRow:
    """Every candidate boundary considered, confirmed or not.

    Used by the --scan report so a human can see what was rejected and why,
    not just what got flagged. `volume_ratio`, `dollar_volume_ratio`, and
    `inferred_factor` are None whenever the verdict could not compute them
    (ambiguous volume, or insufficient context).
    """

    date: date
    price_ratio: float
    verdict: Verdict
    volume_ratio: Optional[float]
    dollar_volume_ratio: Optional[float]
    inferred_factor: Optional[float]


def _median_valid(window: np.ndarray) -> Optional[float]:
    """Robust volume baseline over a window of bars.

    NaN is truthy in a bare Python `if`, so this filters explicitly with a
    self-equality check rather than relying on truthiness.
    """
    if window is None or len(window) == 0:
        return None
    valid = window[(window == window) & (window >= 0)]
    if valid.size == 0:
        return None
    return float(np.median(valid))


def _evaluate_jump(
    price_ratio: float,
    pre_vol: Optional[float],
    post_vol: Optional[float],
    dollar_volume_tolerance: float,
    allow_forward_splits: bool = ALLOW_FORWARD_SPLITS_DEFAULT,
    forward_dollar_volume_tolerance: float = FORWARD_DOLLAR_VOLUME_RATIO_TOLERANCE,
) -> tuple[Verdict, Optional[float], Optional[float], Optional[float]]:
    """Classify one already-detected price jump using its volume baselines.

    Returns (verdict, volume_ratio, dollar_volume_ratio, inferred_factor).
    """
    if pre_vol is None or post_vol is None or pre_vol <= 0 or post_vol <= 0:
        return "ambiguous", None, None, None

    volume_ratio = post_vol / pre_vol
    dollar_ratio = price_ratio * volume_ratio
    reverse = price_ratio >= 1.0

    # A share-count change moves volume to the OPPOSITE side of 1.0 from
    # price: reverse split (price up) needs volume down; forward split
    # (price down) needs volume up. This one check is what rejects all four
    # original real moves in the module docstring -- in each of them volume
    # moved the SAME direction as price.
    direction_ok = (volume_ratio < 1.0) if reverse else (volume_ratio > 1.0)

    if reverse:
        # Reverse split: a boundary is classified "split" ONLY when price
        # went up AND volume went down. This is the well-supported path --
        # DKI's confirmed real split (dollar-volume ratio 0.94) clears it
        # and no confirmed false positive found in review has this shape.
        magnitude_ok = (
            (1.0 / dollar_volume_tolerance) <= dollar_ratio <= dollar_volume_tolerance
        )
        # direction_ok alone is satisfied by volume_ratio = 0.974, which is
        # volume essentially UNCHANGED. A 1-for-N reverse split destroys N-1
        # of every N shares, so volume must fall roughly N-fold, not merely
        # tick down. Without this, real market moves where volume happened
        # to dip a hair were classified as splits: OCGN 2021-02-08 (price
        # x3.01, volume x0.974 -- the real COVAXIN run), CHRD 2020-03-13 and
        # AHT 2020-03-19 (COVID-crash rebounds), CNVS/VISL 2020-06-04 (the
        # June 2020 meme rally), PFSA 2025-07-28. Six false positives out of
        # thirteen, two of them on tickers the backtest actually traded.
        #
        # Require volume to fall by at least sqrt(N) rather than the full N.
        # Volume is noisy and a split-day often trades heavily, so demanding
        # the full N-fold drop would reject genuine splits; sqrt(N) keeps
        # every confirmed split in this cache (DKI needs <=0.245 and has
        # 0.056; OPAD needs <=0.346 and has 0.122) while rejecting all six
        # false positives above, whose volume barely moved.
        proportional_ok = volume_ratio <= price_ratio ** -VOLUME_FALL_EXPONENT
        if direction_ok and magnitude_ok and proportional_ok:
            return "split", volume_ratio, dollar_ratio, price_ratio
        return "real_move", volume_ratio, dollar_ratio, None

    # Forward split path (price down, volume up): this is the exact shape
    # of a real crash. See ALLOW_FORWARD_SPLITS_DEFAULT above for why it is
    # disabled unless explicitly requested -- SBET, NFE, SKIN, ADTX, MLTX,
    # and ATYR (all real crashes/dilutions, not splits) all have this
    # shape, and the closest of them cannot be reliably separated from a
    # genuine forward split using the dollar-volume signal alone.
    if not allow_forward_splits:
        return "real_move", volume_ratio, dollar_ratio, None

    magnitude_ok = (
        (1.0 / forward_dollar_volume_tolerance)
        <= dollar_ratio
        <= forward_dollar_volume_tolerance
    )
    if direction_ok and magnitude_ok:
        return "split", volume_ratio, dollar_ratio, 1.0 / price_ratio
    return "real_move", volume_ratio, dollar_ratio, None


def _scan_boundaries(
    df: Optional[pd.DataFrame],
    price_ratio_threshold: float = PRICE_RATIO_THRESHOLD,
    volume_window: int = VOLUME_MEDIAN_WINDOW,
    min_context_bars: int = MIN_CONTEXT_BARS,
    min_rows: int = MIN_ROWS,
    dollar_volume_tolerance: float = DOLLAR_VOLUME_RATIO_TOLERANCE,
    allow_forward_splits: bool = ALLOW_FORWARD_SPLITS_DEFAULT,
    forward_dollar_volume_tolerance: float = FORWARD_DOLLAR_VOLUME_RATIO_TOLERANCE,
    min_pre_jump_close: float = MIN_PRE_JUMP_CLOSE,
    min_pre_jump_dollar_volume: float = MIN_PRE_JUMP_DOLLAR_VOLUME,
) -> list[_BoundaryRow]:
    """Scan one ticker's OHLCV frame for every candidate overnight jump.

    A "candidate" is any boundary whose raw close ratio clears
    price_ratio_threshold (or its reciprocal), regardless of how it is later
    classified. This is the shared core behind both detect_unadjusted_splits
    (which keeps only "split" rows) and the --scan report (which keeps all
    of them, so a human can see what got rejected and why).
    """
    if df is None or df.empty or "close" not in df.columns:
        return []

    frame = df.sort_index()
    n = len(frame)
    if n < min_rows or "volume" not in frame.columns:
        return []

    closes = frame["close"].to_numpy(dtype=float)
    volumes = frame["volume"].to_numpy(dtype=float)
    dollar_volumes = closes * volumes
    idx = frame.index

    rows: list[_BoundaryRow] = []
    for i in range(1, n):
        prev_close = closes[i - 1]
        cur_close = closes[i]
        # NaN is truthy, so this uses an explicit self-equality check
        # instead of bare truthiness, matching backtest/splits.py.
        if prev_close != prev_close or cur_close != cur_close:
            continue
        if prev_close <= 0 or cur_close <= 0:
            continue

        price_ratio = cur_close / prev_close
        if not (
            price_ratio >= price_ratio_threshold
            or price_ratio <= 1.0 / price_ratio_threshold
        ):
            continue

        idx_val = idx[i]
        d = idx_val.date() if hasattr(idx_val, "date") else idx_val

        if i < min_context_bars or (n - 1 - i) < min_context_bars:
            rows.append(_BoundaryRow(d, price_ratio, "insufficient_context", None, None, None))
            continue

        pre_window = volumes[max(0, i - volume_window):i]
        post_window = volumes[i:i + volume_window]
        pre_vol = _median_valid(pre_window)
        post_vol = _median_valid(post_window)

        # Tradeability screen (item 2): sub-penny closes and thin pre-jump
        # dollar volume both mean the "before" side of this boundary could
        # never have been traded by the backtest anyway, so there is no
        # value in risking a wrong split/real-move call on it. This is
        # checked BEFORE the fingerprint logic and short-circuits straight
        # to "untradeable", regardless of what the price/volume shape looks
        # like. Pre-jump dollar volume uses the same median-window baseline
        # as the volume fingerprint, for the same single-bar-noise reasons.
        pre_dollar_window = dollar_volumes[max(0, i - volume_window):i]
        pre_dollar_vol = _median_valid(pre_dollar_window)
        untradeable = prev_close < min_pre_jump_close or (
            pre_dollar_vol is not None and pre_dollar_vol < min_pre_jump_dollar_volume
        )
        if untradeable:
            rows.append(_BoundaryRow(d, price_ratio, "untradeable", None, None, None))
            continue

        verdict, volume_ratio, dollar_ratio, factor = _evaluate_jump(
            price_ratio,
            pre_vol,
            post_vol,
            dollar_volume_tolerance,
            allow_forward_splits=allow_forward_splits,
            forward_dollar_volume_tolerance=forward_dollar_volume_tolerance,
        )
        rows.append(_BoundaryRow(d, price_ratio, verdict, volume_ratio, dollar_ratio, factor))

    return rows


def detect_unadjusted_splits(
    df: Optional[pd.DataFrame],
    price_ratio_threshold: float = PRICE_RATIO_THRESHOLD,
    volume_window: int = VOLUME_MEDIAN_WINDOW,
    min_context_bars: int = MIN_CONTEXT_BARS,
    min_rows: int = MIN_ROWS,
    dollar_volume_tolerance: float = DOLLAR_VOLUME_RATIO_TOLERANCE,
    allow_forward_splits: bool = ALLOW_FORWARD_SPLITS_DEFAULT,
    forward_dollar_volume_tolerance: float = FORWARD_DOLLAR_VOLUME_RATIO_TOLERANCE,
    min_pre_jump_close: float = MIN_PRE_JUMP_CLOSE,
    min_pre_jump_dollar_volume: float = MIN_PRE_JUMP_DOLLAR_VOLUME,
) -> list[SplitEvent]:
    """Scan a single ticker's OHLCV frame for confirmed unadjusted splits.

    `df` must have a DatetimeIndex and `close`/`volume` columns. Only
    boundaries that pass the tradeability screen, the price-ratio gate, AND
    the volume-fingerprint check (see _evaluate_jump) come back; real moves,
    ambiguous jumps, untradeable boundaries, and boundaries too close to the
    start/end of the history are silently excluded -- use _scan_boundaries
    via the CLI report to see those too. Forward splits (price down) are
    excluded by default; see allow_forward_splits.
    """
    rows = _scan_boundaries(
        df,
        price_ratio_threshold=price_ratio_threshold,
        volume_window=volume_window,
        min_context_bars=min_context_bars,
        min_rows=min_rows,
        dollar_volume_tolerance=dollar_volume_tolerance,
        allow_forward_splits=allow_forward_splits,
        forward_dollar_volume_tolerance=forward_dollar_volume_tolerance,
        min_pre_jump_close=min_pre_jump_close,
        min_pre_jump_dollar_volume=min_pre_jump_dollar_volume,
    )
    return [
        SplitEvent(
            date=r.date,
            price_ratio=r.price_ratio,
            volume_ratio=r.volume_ratio,
            dollar_volume_ratio=r.dollar_volume_ratio,
            inferred_factor=r.inferred_factor,
        )
        for r in rows
        if r.verdict == "split"
    ]


def apply_adjustment(df: pd.DataFrame, events: list[SplitEvent]) -> pd.DataFrame:
    """Back-adjust bars before each confirmed split, newest split first.

    For a REVERSE split (price_ratio >= 1, inferred_factor == price_ratio),
    bars before the boundary are multiplied by the factor (price) and
    divided by it (volume): the historical price was really that much lower
    in old-share terms, so it is scaled UP to the new-share basis, and old
    volume, recorded in the larger old share count, is scaled DOWN to match.
    For a FORWARD split (price_ratio < 1, inferred_factor == 1/price_ratio),
    it is the reverse: price divided, volume multiplied.

    Multiple splits compound correctly because the newest one is applied
    first: a bar before every split date then picks up each factor in turn,
    equivalent to multiplying by the cumulative factor in one step. Never
    mutates `df`; always returns a copy. dollar_volume is recomputed from
    the adjusted close/volume rather than left stale.
    """
    out = df.copy()
    if not events:
        return out

    # The cached parquet files hold volume as int64; dividing or multiplying
    # by a non-integer factor produces a float, and pandas 2.x refuses to
    # write a float back into an int column. Cast to float64 up front so the
    # assignments below never raise on real cache data (synthetic test
    # frames are float already, so this is a no-op there).
    for col in ("open", "high", "low", "close", "volume"):
        if col in out.columns:
            out[col] = out[col].astype("float64")

    for ev in sorted(events, key=lambda e: e.date, reverse=True):
        factor = ev.inferred_factor
        if factor is None or factor <= 0:
            log.warning("split event at %s has no usable factor, skipped", ev.date)
            continue
        boundary = pd.Timestamp(ev.date)
        mask = out.index < boundary
        if not mask.any():
            continue

        reverse = ev.price_ratio >= 1.0
        price_op = (lambda s: s * factor) if reverse else (lambda s: s / factor)
        volume_op = (lambda s: s / factor) if reverse else (lambda s: s * factor)

        for col in ("open", "high", "low", "close"):
            if col in out.columns:
                out.loc[mask, col] = price_op(out.loc[mask, col])
        if "volume" in out.columns:
            out.loc[mask, "volume"] = volume_op(out.loc[mask, "volume"])

    if "close" in out.columns and "volume" in out.columns:
        out["dollar_volume"] = out["close"] * out["volume"]

    return out


# ---------------------------------------------------------------------------
# Cache scan / report / apply
# ---------------------------------------------------------------------------

def _iter_ticker_parquet_paths(cache_dir: str):
    for path in sorted(glob.glob(os.path.join(cache_dir, "*.parquet"))):
        if os.path.basename(path) in _SKIP_NAMES:
            continue
        yield path


def scan_cache(
    cache_dir: str = CACHE_DIR,
    limit: Optional[int] = None,
    allow_forward_splits: bool = ALLOW_FORWARD_SPLITS_DEFAULT,
) -> pd.DataFrame:
    """Scan every ticker parquet in cache_dir and return one row per
    candidate boundary (every verdict, not just confirmed splits).

    Columns: ticker, date, price_ratio, volume_ratio, dollar_volume_ratio,
    inferred_factor, verdict. Sorted by inferred_factor descending (rows
    with no inferred factor, i.e. not a confirmed split, sort last).
    """
    records: list[dict] = []
    paths = list(_iter_ticker_parquet_paths(cache_dir))
    if limit is not None:
        paths = paths[:limit]

    for path in paths:
        ticker = os.path.splitext(os.path.basename(path))[0]
        try:
            df = pd.read_parquet(path)
        except Exception as exc:
            log.warning("Could not read %s: %s", path, exc)
            continue
        rows = _scan_boundaries(df, allow_forward_splits=allow_forward_splits)
        for r in rows:
            records.append(
                {
                    "ticker": ticker,
                    "date": r.date.isoformat() if hasattr(r.date, "isoformat") else str(r.date),
                    "price_ratio": r.price_ratio,
                    "volume_ratio": r.volume_ratio,
                    "dollar_volume_ratio": r.dollar_volume_ratio,
                    "inferred_factor": r.inferred_factor,
                    "verdict": r.verdict,
                }
            )

    out = pd.DataFrame.from_records(
        records,
        columns=[
            "ticker", "date", "price_ratio", "volume_ratio",
            "dollar_volume_ratio", "inferred_factor", "verdict",
        ],
    )
    if not out.empty:
        out = out.sort_values("inferred_factor", ascending=False, na_position="last")
        out = out.reset_index(drop=True)
    return out


def _backup_original(ticker: str, cache_dir: str) -> None:
    """Copy a ticker's cache file aside before it is overwritten.

    Matches repair_price_cache._backup_original: copy2 into a dedicated
    backup directory, and only the FIRST backup is kept, so a second run
    can never overwrite a pristine original with an already-adjusted copy.
    A separate directory name from repair_price_cache's own
    _pre_repair_backup keeps the two tools' backups from colliding, since a
    ticker could in principle be touched by both at different times.
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


def _atomic_save(ticker: str, df: pd.DataFrame, cache_dir: str, events: list[SplitEvent]) -> None:
    """Write to a temp file then rename over the target.

    Matches repair_price_cache._atomic_save: a same-filesystem rename is
    atomic, so an interrupted run never leaves a truncated parquet file
    where the real cache file is expected.
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
                "split_fingerprint_repaired": True,
                "split_fingerprint_events": [
                    {
                        "date": e.date.isoformat() if hasattr(e.date, "isoformat") else str(e.date),
                        "price_ratio": e.price_ratio,
                        "inferred_factor": e.inferred_factor,
                    }
                    for e in events
                ],
            },
            fh,
        )
    os.replace(meta_tmp, meta_path)


def apply_cache(
    cache_dir: str = CACHE_DIR,
    limit: Optional[int] = None,
    allow_forward_splits: bool = ALLOW_FORWARD_SPLITS_DEFAULT,
) -> tuple[list[str], int]:
    """Apply confirmed split adjustments to every affected ticker on disk.

    Returns (tickers_written, total_events_applied). Backs up each touched
    file before overwriting it (see _backup_original).
    """
    written: list[str] = []
    total_events = 0
    paths = list(_iter_ticker_parquet_paths(cache_dir))
    if limit is not None:
        paths = paths[:limit]

    for path in paths:
        ticker = os.path.splitext(os.path.basename(path))[0]
        try:
            df = pd.read_parquet(path)
        except Exception as exc:
            log.warning("Could not read %s: %s", path, exc)
            continue
        events = detect_unadjusted_splits(df, allow_forward_splits=allow_forward_splits)
        if not events:
            continue
        adjusted = apply_adjustment(df, events)
        _backup_original(ticker, cache_dir)
        _atomic_save(ticker, adjusted, cache_dir, events)
        log.info("%s: applied %d split adjustment(s)", ticker, len(events))
        written.append(ticker)
        total_events += len(events)

    return written, total_events


def _print_report(report: pd.DataFrame, top_n: int = 25) -> None:
    splits = report[report["verdict"] == "split"]
    affected_tickers = splits["ticker"].nunique()
    print(
        f"Scanned candidates: {len(report)} boundary(ies) across "
        f"{report['ticker'].nunique()} ticker(s) with a >= {PRICE_RATIO_THRESHOLD}x "
        f"overnight jump."
    )
    print(
        f"Confirmed splits: {len(splits)} event(s) across {affected_tickers} "
        f"ticker(s)."
    )
    by_verdict = report["verdict"].value_counts()
    for verdict, count in by_verdict.items():
        print(f"  {verdict}: {count}")

    if not splits.empty:
        print(f"\nTop {top_n} by inferred_factor (descending):")
        top = splits.head(top_n)
        for _, row in top.iterrows():
            print(
                f"  {row['ticker']:<8} {row['date']}  "
                f"price_ratio={row['price_ratio']:.3f}  "
                f"volume_ratio={row['volume_ratio']:.4f}  "
                f"dollar_volume_ratio={row['dollar_volume_ratio']:.3f}  "
                f"inferred_factor={row['inferred_factor']:.3f}"
            )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Detect and locally back-adjust unadjusted splits in "
            "price_cache/*.parquet, using a price/volume fingerprint "
            "instead of a network refetch."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--scan", action="store_true",
        help="Report only. Writes the CSV report but changes no cache files.",
    )
    mode.add_argument(
        "--apply", action="store_true",
        help="Write back-adjusted cache files (after backing up originals).",
    )
    parser.add_argument(
        "--cache-dir", default=CACHE_DIR,
        help=f"Directory of ticker parquet files (default: {CACHE_DIR}).",
    )
    parser.add_argument(
        "--out", default="split_fingerprint_report.csv",
        help="CSV report path (default: split_fingerprint_report.csv).",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Cap the number of tickers scanned/applied. Mainly for testing.",
    )
    parser.add_argument(
        "--allow-forward-splits", action="store_true", default=ALLOW_FORWARD_SPLITS_DEFAULT,
        help=(
            "Opt into forward-split (price DOWN, volume UP) detection. OFF "
            "by default because every confirmed false positive found in "
            "this cache has this exact shape and cannot be reliably told "
            "apart from a real crash by dollar volume alone -- see "
            "ALLOW_FORWARD_SPLITS_DEFAULT in split_fingerprint.py."
        ),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    report = scan_cache(
        args.cache_dir, limit=args.limit, allow_forward_splits=args.allow_forward_splits
    )
    report.to_csv(args.out, index=False)
    print(f"CSV report written to {os.path.abspath(args.out)}")
    _print_report(report)

    if args.scan:
        print("\n--scan: no cache files changed.")
        return 0

    written, total_events = apply_cache(
        args.cache_dir, limit=args.limit, allow_forward_splits=args.allow_forward_splits
    )
    print(f"\n--apply: adjusted {len(written)} ticker(s), {total_events} split event(s) applied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
