"""Tests for backtest.splits: unadjusted-split detection and correction.

Follows this repo's pytest conventions: plain pytest, class-grouped tests,
minimal fixtures. Every frame here is synthetic. No network access, no
price_cache reads or writes.
"""

from __future__ import annotations

import math
import os
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from backtest import splits as splits_mod  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------
def _dates(start: date, n: int) -> list[date]:
    return [start + timedelta(days=i) for i in range(n)]


def _frame(dates: list[date], closes: list[float], volumes: list[float]) -> pd.DataFrame:
    assert len(dates) == len(closes) == len(volumes)
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d in dates], name="date")
    df = pd.DataFrame(
        {
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": volumes,
        },
        index=idx,
    )
    df["dollar_volume"] = df["close"] * df["volume"]
    return df


def _flat_segment(n: int, close: float, volume: float, jitter: float = 0.0) -> tuple[list[float], list[float]]:
    """n bars of near-constant close/volume, with tiny deterministic jitter
    so a segment is not perfectly flat (real data never is)."""
    closes = [close * (1.0 + jitter * (((i % 5) - 2) / 10.0)) for i in range(n)]
    volumes = [max(1.0, volume * (1.0 + jitter * (((i % 7) - 3) / 10.0))) for i in range(n)]
    return closes, volumes


# ---------------------------------------------------------------------------
# Clean reverse split
# ---------------------------------------------------------------------------
class TestCleanReverseSplit:
    """Price jumps up by a clean factor, volume drops by roughly the same
    factor: the canonical unadjusted-reverse-split signature."""

    def test_classifies_as_split_with_correct_ratio(self):
        start = date(2020, 1, 1)
        pre_closes, pre_vols = _flat_segment(25, 10.0, 8000.0, jitter=0.3)
        post_closes, post_vols = _flat_segment(25, 40.0, 2000.0, jitter=0.3)
        # Force the exact boundary ratio to a clean 4x.
        pre_closes[-1] = 10.0
        post_closes[0] = 40.0
        closes = pre_closes + post_closes
        volumes = pre_vols + post_vols
        dates = _dates(start, len(closes))
        df = _frame(dates, closes, volumes)

        detections = splits_mod.detect_discontinuities("RSPLIT", df)

        splits = [d for d in detections if d.classification == "split"]
        assert len(splits) == 1
        d = splits[0]
        assert d.date == dates[25]
        assert d.close_ratio == pytest.approx(4.0, rel=1e-6)
        assert d.inferred_split_ratio == pytest.approx(4.0)
        assert d.volume_ratio is not None and d.volume_ratio < 1.0
        assert d.confidence > 0.6


# ---------------------------------------------------------------------------
# Clean forward split
# ---------------------------------------------------------------------------
class TestCleanForwardSplit:
    """Price drops by a clean factor, volume rises by roughly the same
    factor: a forward split (e.g. 4-for-1) that Yahoo failed to adjust."""

    def test_classifies_as_split_with_reciprocal_ratio(self):
        start = date(2021, 3, 1)
        pre_closes, pre_vols = _flat_segment(25, 100.0, 1000.0, jitter=0.2)
        post_closes, post_vols = _flat_segment(25, 25.0, 4000.0, jitter=0.2)
        pre_closes[-1] = 100.0
        post_closes[0] = 25.0
        closes = pre_closes + post_closes
        volumes = pre_vols + post_vols
        dates = _dates(start, len(closes))
        df = _frame(dates, closes, volumes)

        detections = splits_mod.detect_discontinuities("FSPLIT", df)

        splits = [d for d in detections if d.classification == "split"]
        assert len(splits) == 1
        d = splits[0]
        assert d.close_ratio == pytest.approx(0.25, rel=1e-6)
        assert d.inferred_split_ratio == pytest.approx(0.25, rel=1e-2)
        assert d.volume_ratio is not None and d.volume_ratio > 1.0


# ---------------------------------------------------------------------------
# Real move with volume spike
# ---------------------------------------------------------------------------
class TestRealMoveVolumeSpike:
    """Price jumps up on genuine news and volume spikes WITH it, not against
    it. Must not be classified as a split."""

    def test_classifies_as_real_move(self):
        start = date(2022, 6, 1)
        pre_closes, pre_vols = _flat_segment(25, 5.0, 50_000.0, jitter=0.2)
        post_closes, post_vols = _flat_segment(25, 25.0, 150_000.0, jitter=0.2)
        pre_closes[-1] = 5.0
        post_closes[0] = 25.0
        closes = pre_closes + post_closes
        volumes = pre_vols + post_vols
        dates = _dates(start, len(closes))
        df = _frame(dates, closes, volumes)

        detections = splits_mod.detect_discontinuities("MOVER", df)

        assert len(detections) == 1
        d = detections[0]
        assert d.classification == "real_move"
        assert d.inferred_split_ratio is None
        assert d.volume_ratio is not None and d.volume_ratio > 1.0

    def test_crash_with_volume_spike_is_not_forced_into_split(self):
        """A genuine crash (price down hard) with a volume spike looks
        directionally like a forward split (both show volume rising), so the
        implausible-ratio check must be what keeps this out of "split"."""
        start = date(2022, 9, 1)
        pre_closes, pre_vols = _flat_segment(25, 30.0, 20_000.0, jitter=0.2)
        post_closes, post_vols = _flat_segment(25, 6.72, 200_000.0, jitter=0.2)
        pre_closes[-1] = 30.0
        # ratio 0.224: sits in the gap between clean ratios 0.2 (1/5) and
        # 0.25 (1/4), outside SPLIT_RATIO_LOG_TOLERANCE of either one.
        post_closes[0] = 6.72
        closes = pre_closes + post_closes
        volumes = pre_vols + post_vols
        dates = _dates(start, len(closes))
        df = _frame(dates, closes, volumes)

        detections = splits_mod.detect_discontinuities("CRASH", df)
        assert len(detections) == 1
        assert detections[0].classification != "split"


# ---------------------------------------------------------------------------
# Ambiguous: zero / missing volume
# ---------------------------------------------------------------------------
class TestAmbiguousZeroVolume:
    def test_zero_volume_both_sides_is_ambiguous(self):
        start = date(2019, 1, 1)
        closes = [10.0] * 25 + [40.0] * 25
        volumes = [0.0] * 50
        dates = _dates(start, len(closes))
        df = _frame(dates, closes, volumes)

        detections = splits_mod.detect_discontinuities("NOVOL", df)
        assert len(detections) == 1
        assert detections[0].classification == "ambiguous"
        assert detections[0].volume_ratio is None
        assert detections[0].inferred_split_ratio is None

    def test_missing_volume_column_is_ambiguous(self):
        start = date(2019, 1, 1)
        closes = [10.0] * 25 + [40.0] * 25
        dates = _dates(start, len(closes))
        idx = pd.DatetimeIndex([pd.Timestamp(d) for d in dates], name="date")
        df = pd.DataFrame({"open": closes, "high": closes, "low": closes, "close": closes}, index=idx)

        detections = splits_mod.detect_discontinuities("NOVOLCOL", df)
        assert len(detections) == 1
        assert detections[0].classification == "ambiguous"

    def test_nan_volume_baseline_is_ambiguous(self):
        start = date(2019, 1, 1)
        closes = [10.0] * 25 + [40.0] * 25
        volumes = [float("nan")] * 25 + [float("nan")] * 25
        dates = _dates(start, len(closes))
        df = _frame(dates, closes, volumes)

        detections = splits_mod.detect_discontinuities("NANVOL", df)
        assert len(detections) == 1
        assert detections[0].classification == "ambiguous"


# ---------------------------------------------------------------------------
# Multiple splits compounding
# ---------------------------------------------------------------------------
class TestMultipleSplitsCompound:
    def test_two_splits_detected_and_adjustment_compounds(self):
        start = date(2018, 1, 1)
        # Segment A (oldest): close ~2.5, volume ~80000
        # Split 1 (4x): close jumps to ~10, volume drops to ~20000
        # Segment B: close ~10, volume ~20000
        # Split 2 (5x): close jumps to ~50, volume drops to ~4000
        # Segment C (newest): close ~50, volume ~4000
        # Both jump ratios (4x, 5x) clear CLOSE_RATIO_THRESHOLD (3x) on
        # their own, so both register as candidate discontinuities.
        seg_a_c, seg_a_v = _flat_segment(25, 2.5, 80_000.0, jitter=0.2)
        seg_b_c, seg_b_v = _flat_segment(25, 10.0, 20_000.0, jitter=0.2)
        seg_c_c, seg_c_v = _flat_segment(25, 50.0, 4_000.0, jitter=0.2)
        seg_a_c[-1] = 2.5
        seg_b_c[0] = 10.0
        seg_b_c[-1] = 10.0
        seg_c_c[0] = 50.0

        closes = seg_a_c + seg_b_c + seg_c_c
        volumes = seg_a_v + seg_b_v + seg_c_v
        dates = _dates(start, len(closes))
        df = _frame(dates, closes, volumes)

        detections = splits_mod.detect_discontinuities("MULTI", df)
        splits = [d for d in detections if d.classification == "split"]
        assert len(splits) == 2
        ratios = sorted(round(d.inferred_split_ratio) for d in splits)
        assert ratios == [4, 5]

        adjusted = splits_mod.apply_split_adjustments(df, detections)
        adj_closes = adjusted.sort_index()["close"].to_numpy()

        # No jump anywhere >= 3x after adjustment.
        for i in range(1, len(adj_closes)):
            jump = max(adj_closes[i] / adj_closes[i - 1], adj_closes[i - 1] / adj_closes[i])
            assert jump < 3.0, f"discontinuity remains at index {i}"

        # Segment A (before BOTH splits) must be scaled by the full 4 * 5 = 20
        # cumulative factor to land on segment C's basis (~50), not just one
        # of the two factors.
        boundary_a = pd.Timestamp(dates[24])
        assert adjusted.loc[boundary_a, "close"] == pytest.approx(
            df.loc[boundary_a, "close"] * 20.0, rel=1e-6
        )
        assert adjusted.loc[boundary_a, "volume"] == pytest.approx(
            df.loc[boundary_a, "volume"] / 20.0, rel=1e-6
        )
        # dollar_volume stays close * volume, recomputed, not left stale.
        assert adjusted.loc[boundary_a, "dollar_volume"] == pytest.approx(
            adjusted.loc[boundary_a, "close"] * adjusted.loc[boundary_a, "volume"]
        )

        # Segment B (after split 1, before split 2) only picks up the 5x
        # factor from split 2, not split 1's factor too.
        boundary_b = pd.Timestamp(dates[49])
        assert adjusted.loc[boundary_b, "close"] == pytest.approx(
            df.loc[boundary_b, "close"] * 5.0, rel=1e-6
        )

        # Segment C (newest, after every split) is untouched.
        newest = pd.Timestamp(dates[-1])
        assert adjusted.loc[newest, "close"] == pytest.approx(df.loc[newest, "close"])


# ---------------------------------------------------------------------------
# real_move / ambiguous rows must not be adjusted
# ---------------------------------------------------------------------------
class TestAdjustmentLeavesNonSplitsAlone:
    def test_real_move_untouched_by_adjustment(self):
        start = date(2022, 6, 1)
        pre_closes, pre_vols = _flat_segment(25, 5.0, 50_000.0, jitter=0.2)
        post_closes, post_vols = _flat_segment(25, 25.0, 150_000.0, jitter=0.2)
        pre_closes[-1] = 5.0
        post_closes[0] = 25.0
        closes = pre_closes + post_closes
        volumes = pre_vols + post_vols
        dates = _dates(start, len(closes))
        df = _frame(dates, closes, volumes)

        detections = splits_mod.detect_discontinuities("MOVER", df)
        adjusted = splits_mod.apply_split_adjustments(df, detections)

        pd.testing.assert_series_equal(
            adjusted.sort_index()["close"], df.sort_index()["close"]
        )

    def test_ambiguous_untouched_by_adjustment(self):
        start = date(2019, 1, 1)
        closes = [10.0] * 25 + [40.0] * 25
        volumes = [0.0] * 50
        dates = _dates(start, len(closes))
        df = _frame(dates, closes, volumes)

        detections = splits_mod.detect_discontinuities("NOVOL", df)
        adjusted = splits_mod.apply_split_adjustments(df, detections)

        pd.testing.assert_series_equal(
            adjusted.sort_index()["close"], df.sort_index()["close"]
        )

    def test_no_detections_returns_equivalent_frame(self):
        start = date(2023, 1, 1)
        closes, volumes = _flat_segment(30, 20.0, 5000.0, jitter=0.1)
        dates = _dates(start, len(closes))
        df = _frame(dates, closes, volumes)

        detections = splits_mod.detect_discontinuities("FLAT", df)
        assert detections == []
        adjusted = splits_mod.apply_split_adjustments(df, detections)
        pd.testing.assert_frame_equal(adjusted.sort_index(), df.sort_index())


# ---------------------------------------------------------------------------
# NaN handling
# ---------------------------------------------------------------------------
class TestNaNHandling:
    def test_nan_close_rows_do_not_crash_and_are_skipped(self):
        start = date(2020, 1, 1)
        closes = [10.0, 10.1, float("nan"), 10.2, 40.0, 40.1, float("nan"), 40.2]
        volumes = [1000.0] * len(closes)
        dates = _dates(start, len(closes))
        df = _frame(dates, closes, volumes)

        # Must not raise.
        detections = splits_mod.detect_discontinuities("NANCLOSE", df)
        # The NaN bars themselves never appear as a jump endpoint.
        for d in detections:
            assert d.date not in {dates[2], dates[6]}

    def test_zero_and_negative_close_do_not_crash(self):
        start = date(2020, 1, 1)
        closes = [10.0, 0.0, -5.0, 10.0, 40.0]
        volumes = [1000.0] * len(closes)
        dates = _dates(start, len(closes))
        df = _frame(dates, closes, volumes)

        detections = splits_mod.detect_discontinuities("BADCLOSE", df)
        assert isinstance(detections, list)  # no exception raised

    def test_empty_frame_returns_empty_list(self):
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume", "dollar_volume"])
        assert splits_mod.detect_discontinuities("EMPTY", df) == []

    def test_none_frame_returns_empty_list(self):
        assert splits_mod.detect_discontinuities("NONE", None) == []


# ---------------------------------------------------------------------------
# DRIO regression case
# ---------------------------------------------------------------------------
class TestDRIORegression:
    """Mirrors the real DRIO numbers: close 4.14 -> 86.00 (ratio 20.77),
    single-adjacent-bar volume 8170 -> 3410 (only a 2.4x drop, nowhere near
    the 1/20.77 a naive same-bar test would require). Must still classify
    as "split" with inferred ratio 20, because the 20-bar baseline median
    -- not the single adjacent bar -- is what the classifier actually uses."""

    def test_drio_numbers_classify_as_split_ratio_20(self):
        start = date(2019, 10, 1)
        pre_closes, pre_vols = _flat_segment(25, 4.0, 8170.0, jitter=0.15)
        post_closes, post_vols = _flat_segment(25, 90.0, 3410.0, jitter=0.15)
        pre_closes[-1] = 4.14
        post_closes[0] = 86.00
        pre_vols[-1] = 8170.0
        post_vols[0] = 3410.0
        closes = pre_closes + post_closes
        volumes = pre_vols + post_vols
        dates = _dates(start, len(closes))
        df = _frame(dates, closes, volumes)

        detections = splits_mod.detect_discontinuities("DRIO", df)
        splits = [d for d in detections if d.classification == "split"]
        assert len(splits) == 1
        d = splits[0]
        assert d.close_ratio == pytest.approx(86.00 / 4.14, rel=1e-6)
        assert round(d.inferred_split_ratio) == 20
        assert d.date == dates[25]


class TestIntegerVolumeColumn:
    """The cached parquet files hold volume as int64, but every synthetic
    frame in this file builds it as float. That gap hid a real crash: a
    split divides volume by the ratio, which gives a float, and pandas 2.x
    refuses to write a float back into an int column. The whole dataset
    build failed on real data while every test here passed."""

    def test_adjustment_works_on_int64_volume(self):
        start = date(2019, 10, 1)
        pre_closes, pre_vols = _flat_segment(25, 4.0, 8170.0, jitter=0.15)
        post_closes, post_vols = _flat_segment(25, 80.0, 408.0, jitter=0.15)
        closes = pre_closes + post_closes
        volumes = pre_vols + post_vols
        dates = _dates(start, len(closes))
        df = _frame(dates, closes, volumes)

        # This is the only difference from the passing tests: real dtype.
        # dollar_volume must be rebuilt from the rounded volume, or the
        # invariant check below compares against a value the frame itself
        # no longer holds.
        df["volume"] = df["volume"].round().astype("int64")
        df["dollar_volume"] = df["close"] * df["volume"]
        assert df["volume"].dtype == "int64"

        detections = splits_mod.detect_discontinuities("INTVOL", df)
        assert any(d.classification == "split" for d in detections)

        adjusted = splits_mod.apply_split_adjustments(df, detections)

        # Volume must come back scaled, not truncated to an integer.
        ratio = next(
            d.inferred_split_ratio
            for d in detections
            if d.classification == "split"
        )
        assert adjusted["volume"].iloc[0] == pytest.approx(
            float(df["volume"].iloc[0]) / ratio
        )

        # Traded dollars must survive the rescale untouched.
        assert adjusted["dollar_volume"].iloc[0] == pytest.approx(
            float(df["dollar_volume"].iloc[0])
        )

        # And the seam the adjustment exists to remove must be gone.
        r = (adjusted["close"] / adjusted["close"].shift(1)).dropna()
        assert r.max() < 3.0


# ---------------------------------------------------------------------------
# unsafe_label_dates
# ---------------------------------------------------------------------------
class TestUnsafeLabelDates:
    def test_ambiguous_jump_marks_window_unsafe(self):
        start = date(2021, 1, 1)
        closes = [10.0] * 40 + [40.0] * 40
        volumes = [0.0] * 80  # forces ambiguous: no usable volume baseline
        dates = _dates(start, len(closes))
        df = _frame(dates, closes, volumes)

        detections = splits_mod.detect_discontinuities("UNSAFE", df)
        assert len(detections) == 1 and detections[0].classification == "ambiguous"

        unsafe = splits_mod.unsafe_label_dates(df, detections, horizons=(10, 21))
        amb_date = detections[0].date
        amb_pos = dates.index(amb_date)

        # The jump date itself, and entries up to 21 bars before it, are unsafe.
        assert amb_date in unsafe
        assert dates[amb_pos - 21] in unsafe
        assert dates[amb_pos - 5] in unsafe

        # An entry far enough before the jump that even the longest horizon
        # (21) can't reach it is safe.
        assert dates[amb_pos - 22] not in unsafe

        # Entries strictly after the jump are safe: the window looks
        # forward from the entry, so it never reaches back to a jump that
        # already happened.
        assert dates[amb_pos + 1] not in unsafe

    def test_split_jump_is_not_marked_unsafe(self):
        """A "split" gets back-adjusted, so it must not appear in the
        exclusion set even though it is just as large a jump."""
        start = date(2020, 1, 1)
        pre_closes, pre_vols = _flat_segment(25, 10.0, 8000.0, jitter=0.2)
        post_closes, post_vols = _flat_segment(25, 40.0, 2000.0, jitter=0.2)
        pre_closes[-1] = 10.0
        post_closes[0] = 40.0
        closes = pre_closes + post_closes
        volumes = pre_vols + post_vols
        dates = _dates(start, len(closes))
        df = _frame(dates, closes, volumes)

        detections = splits_mod.detect_discontinuities("SAFESPLIT", df)
        assert detections and detections[0].classification == "split"

        unsafe = splits_mod.unsafe_label_dates(df, detections)
        assert unsafe == set()

    def test_no_ambiguous_detections_returns_empty_set(self):
        df = _frame(_dates(date(2022, 1, 1), 10), [10.0] * 10, [1000.0] * 10)
        assert splits_mod.unsafe_label_dates(df, []) == set()


# ---------------------------------------------------------------------------
# real_move ceiling (Part 1): unbounded real_move magnitude is untenable
# ---------------------------------------------------------------------------
class TestRealMoveCeiling:
    """A "real_move" past REAL_MOVE_CEILING_RATIO must be treated as unsafe
    for labels, the same as "ambiguous", even though detect_discontinuities
    still reports it as "real_move" (the detector stays purely descriptive,
    see the module docstring and the comment above REAL_MOVE_CEILING_RATIO
    in splits.py)."""

    def test_real_move_above_ceiling_is_still_classified_real_move(self):
        """The classifier itself does not change: a huge implausible-ratio
        volume-spike jump is still "real_move", not reclassified."""
        start = date(2020, 11, 1)
        pre_closes, pre_vols = _flat_segment(25, 0.05, 20_000.0, jitter=0.2)
        post_closes, post_vols = _flat_segment(25, 1.5, 120_000.0, jitter=0.2)
        pre_closes[-1] = 0.05
        post_closes[0] = 1.5  # ratio 30x, well past the 10x ceiling
        closes = pre_closes + post_closes
        volumes = pre_vols + post_vols
        dates = _dates(start, len(closes))
        df = _frame(dates, closes, volumes)

        detections = splits_mod.detect_discontinuities("NCPLLIKE", df)
        assert len(detections) == 1
        d = detections[0]
        assert d.classification == "real_move"
        assert d.close_ratio > splits_mod.REAL_MOVE_CEILING_RATIO

    def test_real_move_above_ceiling_marks_window_unsafe(self):
        start = date(2020, 11, 1)
        pre_closes, pre_vols = _flat_segment(25, 0.05, 20_000.0, jitter=0.2)
        post_closes, post_vols = _flat_segment(25, 1.5, 120_000.0, jitter=0.2)
        pre_closes[-1] = 0.05
        post_closes[0] = 1.5  # ratio 30x
        closes = pre_closes + post_closes
        volumes = pre_vols + post_vols
        dates = _dates(start, len(closes))
        df = _frame(dates, closes, volumes)

        detections = splits_mod.detect_discontinuities("NCPLLIKE", df)
        assert detections[0].classification == "real_move"

        unsafe = splits_mod.unsafe_label_dates(df, detections, horizons=(10, 21))
        jump_date = detections[0].date
        assert jump_date in unsafe

    def test_real_move_at_or_below_ceiling_stays_safe(self):
        """A plausible real_move (well under the ceiling) must NOT be
        excluded. We must not throw away genuine large returns along with
        the implausible ones."""
        start = date(2022, 6, 1)
        pre_closes, pre_vols = _flat_segment(25, 5.0, 50_000.0, jitter=0.2)
        post_closes, post_vols = _flat_segment(25, 17.0, 150_000.0, jitter=0.2)
        pre_closes[-1] = 5.0
        post_closes[0] = 17.0  # ratio 3.4x, under the 10x ceiling
        closes = pre_closes + post_closes
        volumes = pre_vols + post_vols
        dates = _dates(start, len(closes))
        df = _frame(dates, closes, volumes)

        detections = splits_mod.detect_discontinuities("PLAUSIBLE", df)
        assert detections[0].classification == "real_move"
        assert detections[0].close_ratio <= splits_mod.REAL_MOVE_CEILING_RATIO

        unsafe = splits_mod.unsafe_label_dates(df, detections, horizons=(10, 21))
        assert unsafe == set()
