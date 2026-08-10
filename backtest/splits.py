"""Detect and correct unadjusted splits hiding in the cached price series.

Yahoo's `auto_adjust=True` bars should already back-adjust every split. In
practice Yahoo's split table is incomplete for many small tickers, so some
splits never get applied. The result is a fake overnight close-to-close jump
that looks like a huge one-day return. See DRIO: close 4.14 -> 86.00 on
2019-11-18, a 20.77x jump, with no split listed by `yf.Ticker('DRIO').splits`
before that date.

This module cannot read the missing split metadata, because it does not
exist anywhere we can fetch it. Instead it infers a split from the SHAPE of
the jump: a real corporate action moves price and volume in OPPOSITE
directions, because the share count itself changes. A genuine news-driven
move (buyout, FDA result, halt) tends to move volume the SAME way as price,
or leaves volume mostly unrelated to the price change, because it is driven
by new information rather than a changed share count.

Three outcomes per detected jump:
  - "split": price and volume moved inversely, and the ratio is close to a
    plausible clean split factor. Safe to back-adjust.
  - "real_move": a genuine repricing. Volume shows a spike, not a share-count
    signature. Leave the data alone. This classification carries no size
    limit on its own: see REAL_MOVE_CEILING_RATIO below for the research
    policy that stops an implausibly large "real_move" from feeding a label.
  - "ambiguous": neither signature is clean, or volume data is unusable.
    Never adjust these. Forward-return labels that span one must be dropped
    instead of silently trusting a possibly-fabricated number.

unsafe_label_dates() is the label-safety layer built on top of these three
outcomes. It flags every "ambiguous" jump, plus any "real_move" whose
magnitude exceeds REAL_MOVE_CEILING_RATIO, as unsafe for a forward-return
label. detect_discontinuities() itself never reclassifies a jump because of
its size: it stays a plain description of the SHAPE it saw. The ceiling is
a decision about which of those shapes a label can trust, so it lives in
unsafe_label_dates, not in the classifier.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import date
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)

# An overnight close-to-close move of 3x (or its reciprocal, a 67% drop) is
# already far outside anything a liquid stock does on real news in one day.
# Microcaps do sometimes move this much on real news, so 3x alone is only
# the trigger for a closer look, not proof of a split. Matches the constant
# already used by repair_price_cache.find_suspect_tickers, so a "suspect"
# ticker there and a "discontinuity" here mean the same threshold.
CLOSE_RATIO_THRESHOLD = 3.0

# Single-day volume is very noisy: DRIO's own adjacent-bar volume only fell
# 2.4x (8170 -> 3410) against a 20x price jump, nowhere near the 1/20 a
# share-count halving would predict from that pair alone. A median over many
# bars on each side of the jump smooths out day-to-day noise (a single
# no-news high-volume day, a single thin day) far better than comparing the
# two bars that sit right next to the jump. 20 bars matches the lookback
# PriceUniverse.median_dollar_volume already uses elsewhere in this repo.
VOLUME_BASELINE_WINDOW = 20

# Clean ratios a real split or reverse split actually uses. Reciprocals
# (searched separately, see _nearest_plausible_ratio) cover forward splits.
PLAUSIBLE_SPLIT_RATIOS: tuple[float, ...] = (
    2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0,
    50.0, 100.0,
)

# DRIO's observed ratio was 20.77 for a true 1-for-20 split, a 3.85%
# departure from clean, because a real same-day price move rode on top of
# the split. The tolerance must clear that case with headroom, so 10% is
# used rather than a tighter band that would misclassify DRIO itself as
# "not clean enough". Measured in LOG space so a 10% miss on a ratio of 2
# counts the same as a 10% miss on a ratio of 100.
SPLIT_RATIO_LOG_TOLERANCE = math.log(1.10)

# Volume must move a full 20% away from 1.0, in the direction a share-count
# change would predict, before it counts as a confirmed signal. This sits
# well clear of ordinary volume noise while still being far looser than a
# strict "volume_ratio == 1/close_ratio" test, which DRIO's own numbers
# would fail (0.42 observed vs 0.048 predicted). The margin only checks
# DIRECTION and MAGNITUDE-OF-MOVEMENT, never a match to the exact predicted
# ratio.
VOLUME_DROP_MARGIN = 0.8
VOLUME_RISE_MARGIN = 1.0 / VOLUME_DROP_MARGIN  # 1.25

Classification = str  # "split" | "real_move" | "ambiguous"


@dataclass(frozen=True)
class Discontinuity:
    """One detected overnight close-to-close jump for one ticker."""

    ticker: str
    date: date
    close_ratio: float
    volume_ratio: Optional[float]
    inferred_split_ratio: Optional[float]
    classification: Classification
    confidence: float
    detail: str = ""


def _median_volume(window: pd.Series) -> Optional[float]:
    """Robust volume baseline over a window of bars.

    Returns None when no usable bar exists in the window. A bare truthiness
    check on a possibly-NaN value is wrong, because NaN is truthy in Python,
    so this filters with `notna()` instead.
    """
    if window is None or len(window) == 0:
        return None
    valid = window[window.notna() & (window >= 0)]
    if valid.empty:
        return None
    return float(valid.median())


def _nearest_plausible_ratio(ratio: float) -> tuple[float, float]:
    """Find the plausible split ratio closest to an observed close ratio.

    Searches every ratio in PLAUSIBLE_SPLIT_RATIOS and its reciprocal, since
    a reverse split multiplies price and a forward split divides it.
    Distance is measured in log space so the search treats a small ratio and
    a large ratio with the same relative tolerance.

    Returns (nearest_ratio, log_distance).
    """
    candidates = list(PLAUSIBLE_SPLIT_RATIOS) + [1.0 / r for r in PLAUSIBLE_SPLIT_RATIOS]
    log_ratio = math.log(ratio)
    best = min(candidates, key=lambda c: abs(math.log(c) - log_ratio))
    return best, abs(math.log(best) - log_ratio)


def _classify_jump(
    ticker: str,
    d: date,
    close_ratio: float,
    pre_vol: Optional[float],
    post_vol: Optional[float],
) -> Discontinuity:
    """Classify one already-detected jump using its volume baselines."""
    if pre_vol is None or post_vol is None or pre_vol <= 0 or post_vol <= 0:
        return Discontinuity(
            ticker=ticker, date=d, close_ratio=close_ratio, volume_ratio=None,
            inferred_split_ratio=None, classification="ambiguous", confidence=0.1,
            detail=(
                "the volume baseline around the jump is missing or zero, so "
                "the split signature cannot be checked"
            ),
        )

    volume_ratio = float(post_vol / pre_vol)
    nearest_ratio, log_dist = _nearest_plausible_ratio(close_ratio)
    is_plausible = log_dist <= SPLIT_RATIO_LOG_TOLERANCE

    # A share-count change predicts volume moving to the OPPOSITE side of
    # 1.0 from where price moved: up a lot pairs with a volume DROP, down a
    # lot pairs with a volume RISE. "flat" means volume barely moved either
    # way, too weak a signal to trust.
    if volume_ratio <= VOLUME_DROP_MARGIN:
        vol_state = "down"
    elif volume_ratio >= VOLUME_RISE_MARGIN:
        vol_state = "up"
    else:
        vol_state = "flat"

    price_up = close_ratio >= 1.0
    matches_split_direction = (price_up and vol_state == "down") or (
        not price_up and vol_state == "up"
    )

    if is_plausible and matches_split_direction:
        tightness = max(0.0, 1.0 - log_dist / SPLIT_RATIO_LOG_TOLERANCE)
        confidence = min(0.97, 0.65 + 0.30 * tightness)
        return Discontinuity(
            ticker=ticker, date=d, close_ratio=close_ratio, volume_ratio=volume_ratio,
            inferred_split_ratio=nearest_ratio, classification="split",
            confidence=confidence,
            detail=(
                f"close ratio {close_ratio:.2f} sits near a clean "
                f"{nearest_ratio:g}x ratio, and volume moved the opposite "
                f"way (volume ratio {volume_ratio:.2f})"
            ),
        )

    # A volume SPIKE is the real-move signature regardless of price
    # direction. Price up with volume up is obviously a spike. Price DOWN
    # with volume up (a sell-off) looks, in raw sign terms, like a forward
    # split, but it is not one unless the ratio is also a clean split
    # factor, which the branch above already ruled out here. So an
    # implausible ratio plus a volume rise is read as a crash with a
    # volume spike, not a split.
    if vol_state == "up":
        spike = abs(math.log(volume_ratio))
        confidence = min(0.9, 0.55 + 0.25 * min(1.0, spike / math.log(3.0)))
        return Discontinuity(
            ticker=ticker, date=d, close_ratio=close_ratio, volume_ratio=volume_ratio,
            inferred_split_ratio=None, classification="real_move", confidence=confidence,
            detail=(
                f"volume spiked (volume ratio {volume_ratio:.2f}), which fits "
                "a genuine news-driven move rather than a split"
            ),
        )

    return Discontinuity(
        ticker=ticker, date=d, close_ratio=close_ratio, volume_ratio=volume_ratio,
        inferred_split_ratio=None, classification="ambiguous", confidence=0.3,
        detail=(
            f"close ratio {close_ratio:.2f} and volume ratio {volume_ratio:.2f} "
            "do not cleanly match a split or a real move"
        ),
    )


def detect_discontinuities(
    ticker: str,
    df: pd.DataFrame,
    baseline_window: int = VOLUME_BASELINE_WINDOW,
) -> list[Discontinuity]:
    """Scan one ticker's OHLCV frame for unadjusted splits.

    `df` must have a DatetimeIndex and a `close` column. A `volume` column
    lets the classifier tell a split from a real move. Without one, every
    jump comes back "ambiguous", because there is nothing to check the
    signature against.

    Returns one Discontinuity per bar-to-bar close ratio at or beyond
    CLOSE_RATIO_THRESHOLD (or its reciprocal). An ordinary trading day never
    appears in the result.
    """
    if df is None or df.empty or "close" not in df.columns:
        return []

    frame = df.sort_index()
    closes = frame["close"]
    has_volume = "volume" in frame.columns
    n = len(frame)
    out: list[Discontinuity] = []

    for i in range(1, n):
        prev_close = closes.iloc[i - 1]
        cur_close = closes.iloc[i]
        # NaN is truthy, so every one of these checks uses `!= itself` or an
        # explicit comparison rather than bare truthiness.
        if prev_close is None or prev_close != prev_close or prev_close <= 0:
            continue
        if cur_close is None or cur_close != cur_close or cur_close <= 0:
            continue

        ratio = float(cur_close / prev_close)
        if (1.0 / CLOSE_RATIO_THRESHOLD) < ratio < CLOSE_RATIO_THRESHOLD:
            continue  # an ordinary day, not a discontinuity

        idx_val = frame.index[i]
        d = idx_val.date() if hasattr(idx_val, "date") else idx_val

        if not has_volume:
            out.append(Discontinuity(
                ticker=ticker, date=d, close_ratio=ratio, volume_ratio=None,
                inferred_split_ratio=None, classification="ambiguous", confidence=0.1,
                detail="no volume column is present, so a split cannot be confirmed",
            ))
            continue

        pre_window = frame["volume"].iloc[max(0, i - baseline_window):i]
        post_window = frame["volume"].iloc[i:i + baseline_window]
        pre_vol = _median_volume(pre_window)
        post_vol = _median_volume(post_window)
        out.append(_classify_jump(ticker, d, ratio, pre_vol, post_vol))

    return out


def apply_split_adjustments(
    df: pd.DataFrame, discontinuities: list[Discontinuity]
) -> pd.DataFrame:
    """Back-adjust bars before each detected split, newest split first.

    Only "split" rows are used. "real_move" and "ambiguous" rows are left
    alone, because there is no confirmed ratio to correct with, and
    guessing one risks mangling a genuine price move.

    Multiple splits compound correctly because the newest split is applied
    first. A bar before every split date then picks up every ratio in turn,
    once per split whose date falls after that bar, which is the same as
    multiplying by the cumulative factor in one step.
    """
    out = df.copy()
    splits = [d for d in discontinuities if d.classification == "split"]
    if not splits:
        return out

    # Cached parquet files hold volume as int64, and dividing by a split
    # ratio gives a float. pandas 2.x refuses to write a float back into an
    # int column, so the assignment below raises unless the column becomes
    # float first. Synthetic test frames use float volume already, so this
    # only appears against real cache data.
    for col in ("open", "high", "low", "close", "volume"):
        if col in out.columns:
            out[col] = out[col].astype("float64")

    for disc in sorted(splits, key=lambda d: d.date, reverse=True):
        factor = disc.inferred_split_ratio
        if factor is None or factor <= 0:
            log.warning(
                "%s: split at %s has no usable ratio, skipped", disc.ticker, disc.date
            )
            continue
        boundary = pd.Timestamp(disc.date)
        mask = out.index < boundary
        if not mask.any():
            continue
        for col in ("open", "high", "low", "close"):
            if col in out.columns:
                out.loc[mask, col] = out.loc[mask, col] * factor
        if "volume" in out.columns:
            out.loc[mask, "volume"] = out.loc[mask, "volume"] / factor

    if "close" in out.columns and "volume" in out.columns:
        out["dollar_volume"] = out["close"] * out["volume"]

    return out


DEFAULT_LABEL_HORIZONS: tuple[int, ...] = (10, 21, 63, 126, 252)


# "real_move" has NO upper bound on magnitude by design, because
# detect_discontinuities only checks the SHAPE of a jump (price against
# volume), never its size. Measured across the whole cache, "real_move"
# magnitude runs far past anything a real listed stock does overnight:
# p50 4.4x, p75 9.1x, p90 42.3x, p95 100.0x, p99 301.4x, max 2,666.7x, and
# 151 of 687 detections (22.0%) exceed 10x. A genuine 10x overnight move
# essentially never happens in a real listed equity. NCPL on 2020-11-11
# shows a 2,666x close ratio with only a 6.0x volume ratio, currently
# trusted as "real_move" at confidence 0.80. That is a data glitch, not a
# trade.
#
# This ceiling is a RESEARCH POLICY about which detected shapes are safe to
# trust for labels. It is not a fact about the jump's shape, so it lives
# here, in the label-safety layer, and not inside detect_discontinuities.
# The detector stays a plain description of what it sees: a jump past this
# ceiling still comes back classified "real_move" in the returned
# Discontinuity list. unsafe_label_dates below is the one place that turns
# "real_move past the ceiling" into "unsafe for a label", the same way it
# already treats "ambiguous".
REAL_MOVE_CEILING_RATIO = 10.0


def _exceeds_real_move_ceiling(close_ratio: float) -> bool:
    """True when a close ratio, or its reciprocal, passes the ceiling above.

    A drop reports a close_ratio below 1, so the reciprocal is checked too,
    the same way CLOSE_RATIO_THRESHOLD is checked in detect_discontinuities.
    """
    magnitude = close_ratio if close_ratio >= 1.0 else 1.0 / close_ratio
    return magnitude > REAL_MOVE_CEILING_RATIO


def unsafe_label_dates(
    df: pd.DataFrame,
    discontinuities: list[Discontinuity],
    horizons: tuple[int, ...] = DEFAULT_LABEL_HORIZONS,
) -> set[date]:
    """Return dates whose forward-return label window spans an unsafe jump.

    A jump is unsafe for labels in two cases:
      - classification is "ambiguous". There is no confirmed correction, so
        a label whose horizon window crosses it must be dropped instead of
        trusting a possibly fabricated return. This also covers most
        near-dead microcaps whose volume baseline is missing or zero, where
        the honest reading is "this price series is broken", not "a
        corporate action we cannot classify".
      - classification is "real_move" AND its magnitude exceeds
        REAL_MOVE_CEILING_RATIO (see the comment above that constant).

    A "split" jump gets back-adjusted, so labels crossing it are safe once
    apply_split_adjustments has run, and it never appears in the result.

    Horizons are counted in TRADING bars, matching how research.py computes
    forward returns (adj_10/21/63/126/252), not calendar days. An entry date
    is unsafe for horizon h when the unsafe bar falls anywhere in
    [entry, entry + h] on the ticker's own trading calendar.
    """
    unsafe_source_dates = {
        d.date for d in discontinuities
        if d.classification == "ambiguous"
        or (d.classification == "real_move" and _exceeds_real_move_ceiling(d.close_ratio))
    }
    if not unsafe_source_dates or df is None or df.empty:
        return set()

    frame = df.sort_index()
    idx = frame.index
    position_by_date = {ts.date(): pos for pos, ts in enumerate(idx)}

    unsafe: set[date] = set()
    for src_date in unsafe_source_dates:
        pos = position_by_date.get(src_date)
        if pos is None:
            continue  # not one of this frame's own bars, nothing to anchor on
        for h in horizons:
            lo = max(0, pos - h)
            for i in range(lo, pos + 1):
                unsafe.add(idx[i].date())

    return unsafe
