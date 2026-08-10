"""Tests for split_fingerprint: local (no-network) unadjusted-split repair.

Follows this repo's pytest conventions: plain pytest, class-grouped tests,
minimal fixtures, synthetic frames only (see tests/test_splits.py). No
network access, no price_cache reads or writes.
"""

from __future__ import annotations

import os
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import split_fingerprint as sf  # noqa: E402


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


def _flat_segment(n: int, close: float, volume: float, jitter: float = 0.15) -> tuple[list[float], list[float]]:
    """n bars of near-constant close/volume, with tiny deterministic jitter
    so a segment is not perfectly flat (real data never is)."""
    closes = [close * (1.0 + jitter * (((i % 5) - 2) / 10.0)) for i in range(n)]
    volumes = [max(1.0, volume * (1.0 + jitter * (((i % 7) - 3) / 10.0))) for i in range(n)]
    return closes, volumes


def _boundary_frame(
    pre_close: float, pre_vol: float, post_close: float, post_vol: float,
    seg_len: int = 15, start: date = date(2020, 1, 1),
) -> tuple[pd.DataFrame, date]:
    pre_closes, pre_vols = _flat_segment(seg_len, pre_close, pre_vol)
    post_closes, post_vols = _flat_segment(seg_len, post_close, post_vol)
    pre_closes[-1] = pre_close
    post_closes[0] = post_close
    closes = pre_closes + post_closes
    volumes = pre_vols + post_vols
    dates = _dates(start, len(closes))
    df = _frame(dates, closes, volumes)
    return df, dates[seg_len]


# ---------------------------------------------------------------------------
# Core fingerprint math against the exact numbers from the task spec
# ---------------------------------------------------------------------------
class TestEvaluateJumpAgainstSpecNumbers:
    """Exercises _evaluate_jump directly with the exact ratios given in the
    task: DKI's confirmed fake split must pass, all four confirmed real
    moves must not."""

    def test_dki_fake_split_is_confirmed(self):
        # price ratio 16.70, volume ratio 0.108 (0.336 -> 5.610, 636700 -> 68900)
        verdict, vol_ratio, dollar_ratio, factor = sf._evaluate_jump(
            16.70, pre_vol=636700.0, post_vol=636700.0 * 0.108,
            dollar_volume_tolerance=sf.DOLLAR_VOLUME_RATIO_TOLERANCE,
        )
        assert verdict == "split"
        assert vol_ratio == pytest.approx(0.108, rel=1e-6)
        assert dollar_ratio == pytest.approx(16.70 * 0.108, rel=1e-6)
        assert factor == pytest.approx(16.70)

    @pytest.mark.parametrize(
        "label,price_ratio,volume_ratio",
        [
            ("XHLD_2025-03-19", 2.72, 45.2),
            ("XHLD_2026-01-27", 2.77, 1038.7),
            ("DKI_2026-02-02", 2.85, 3.67),
            ("ADTX_2026-06-30", 3.00, 1.84),
        ],
    )
    def test_real_moves_are_not_confirmed(self, label, price_ratio, volume_ratio):
        verdict, vol_ratio, dollar_ratio, factor = sf._evaluate_jump(
            price_ratio, pre_vol=10_000.0, post_vol=10_000.0 * volume_ratio,
            dollar_volume_tolerance=sf.DOLLAR_VOLUME_RATIO_TOLERANCE,
        )
        assert verdict != "split", label
        assert factor is None


# ---------------------------------------------------------------------------
# End-to-end: clean reverse split
# ---------------------------------------------------------------------------
class TestCleanReverseSplitEndToEnd:
    def test_detected_and_adjustment_removes_discontinuity(self):
        df, boundary = _boundary_frame(10.0, 80_000.0, 40.0, 20_000.0)

        events = sf.detect_unadjusted_splits(df)
        assert len(events) == 1
        ev = events[0]
        assert ev.date == boundary
        assert ev.price_ratio == pytest.approx(4.0, rel=1e-6)
        assert ev.volume_ratio < 1.0
        assert ev.inferred_factor == pytest.approx(4.0, rel=1e-2)

        adjusted = sf.apply_adjustment(df, events)
        closes = adjusted.sort_index()["close"].to_numpy()
        rets = closes[1:] / closes[:-1]
        max_overnight = max(rets.max(), (1.0 / rets).max())
        # A sane bound: nothing close to the raw ~4x jump should remain.
        assert max_overnight < 1.6

    def test_series_is_continuous_around_boundary(self):
        df, boundary = _boundary_frame(10.0, 80_000.0, 40.0, 20_000.0)
        events = sf.detect_unadjusted_splits(df)
        adjusted = sf.apply_adjustment(df, events).sort_index()

        ts = pd.Timestamp(boundary)
        pos = adjusted.index.get_loc(ts)
        before = adjusted["close"].iloc[pos - 1]
        after = adjusted["close"].iloc[pos]
        assert after / before == pytest.approx(1.0, abs=0.3)


# ---------------------------------------------------------------------------
# Forward-split detection: OFF by default, opt-in only
# ---------------------------------------------------------------------------
# Forward splits (price DOWN, volume UP) are the exact shape of a real
# crash -- see split_fingerprint.ALLOW_FORWARD_SPLITS_DEFAULT. Prior to the
# task that added this section, TestCleanForwardSplitEndToEnd asserted that
# a clean forward-split-shaped boundary (dollar ratio == 1.0 exactly) was
# detected by default. That expectation is now WRONG: forward-split
# detection defaults off precisely because this shape is indistinguishable
# from a real crash like SBET (dollar ratio 0.89, see
# TestSpecTableSevenRows below) using only the fingerprint signal. The old
# assertions are kept below but now require allow_forward_splits=True to
# opt in, and a new pair of default-off assertions replaces what the old
# test used to check implicitly.
class TestCleanForwardSplitEndToEnd:
    def test_not_detected_by_default(self):
        df, boundary = _boundary_frame(100.0, 10_000.0, 25.0, 40_000.0)
        events = sf.detect_unadjusted_splits(df)
        assert events == []

    def test_detected_with_reciprocal_ratio_when_opted_in(self):
        df, boundary = _boundary_frame(100.0, 10_000.0, 25.0, 40_000.0)

        events = sf.detect_unadjusted_splits(df, allow_forward_splits=True)
        assert len(events) == 1
        ev = events[0]
        assert ev.price_ratio == pytest.approx(0.25, rel=1e-6)
        assert ev.volume_ratio > 1.0
        assert ev.inferred_factor == pytest.approx(4.0, rel=1e-2)

    def test_adjustment_divides_pre_boundary_price_when_opted_in(self):
        df, boundary = _boundary_frame(100.0, 10_000.0, 25.0, 40_000.0)
        events = sf.detect_unadjusted_splits(df, allow_forward_splits=True)
        adjusted = sf.apply_adjustment(df, events).sort_index()

        ts = pd.Timestamp(boundary)
        pos = adjusted.index.get_loc(ts)
        before = adjusted["close"].iloc[pos - 1]
        after = adjusted["close"].iloc[pos]
        assert after / before == pytest.approx(1.0, abs=0.3)

    def test_sbet_shaped_crash_not_detected_even_when_opted_in(self):
        # SBET 2025-06-13: price x0.28, volume x3.13, dollar ratio 0.89 --
        # closer to a preserved 1.0 than DKI's own real split (0.94). Even
        # with the feature opted in, FORWARD_DOLLAR_VOLUME_RATIO_TOLERANCE
        # (1.15) must not accept this, which is exactly why the feature
        # defaults off instead of relying on this tolerance alone.
        pre_close = 10.0
        pre_vol = 100_000.0
        df, boundary = _boundary_frame(
            pre_close, pre_vol, pre_close * 0.28, pre_vol * 3.13
        )
        events = sf.detect_unadjusted_splits(df, allow_forward_splits=True)
        assert events == []


# ---------------------------------------------------------------------------
# Real moves must not be flagged (full-frame integration version)
# ---------------------------------------------------------------------------
class TestRealMovesNotFlaggedEndToEnd:
    @pytest.mark.parametrize(
        "label,pre_close,post_close,pre_vol,post_vol",
        [
            ("XHLD_2025-03-19", 10.0, 27.2, 10_000.0, 452_000.0),
            ("XHLD_2026-01-27", 10.0, 27.7, 1_000.0, 1_038_700.0),
            ("DKI_2026-02-02", 10.0, 28.5, 10_000.0, 36_700.0),
            ("ADTX_2026-06-30", 10.0, 30.0, 10_000.0, 18_400.0),
        ],
    )
    def test_real_move_produces_no_split_event(self, label, pre_close, post_close, pre_vol, post_vol):
        df, boundary = _boundary_frame(pre_close, pre_vol, post_close, post_vol)
        events = sf.detect_unadjusted_splits(df)
        assert events == [], label


# ---------------------------------------------------------------------------
# Table-driven test against the exact seven manual-review rows (task spec):
# one confirmed real reverse split (DKI) and six confirmed real
# crashes/dilutions that a pre-review fingerprint misread as splits. Built
# as synthetic frames (not the live cache -- DKI's cache file is already
# back-adjusted, see report) using pre_close=10.0 / pre_vol=100_000.0 so
# every row clears the tradeability screen and only the direction/magnitude
# fingerprint logic is under test.
# ---------------------------------------------------------------------------
class TestSpecTableSevenRows:
    _PRE_CLOSE = 10.0
    _PRE_VOL = 100_000.0  # $1,000,000 pre-jump dollar volume, well above the
    # $500k MIN_PRE_JUMP_DOLLAR_VOLUME floor even after _flat_segment jitter.

    @pytest.mark.parametrize(
        "label,price_ratio,volume_ratio,expect_split",
        [
            ("DKI_2026-05-11", 16.70, 0.056, True),   # confirmed REAL reverse split
            ("SBET_2025-06-13", 0.28, 3.13, False),   # real crash
            ("NFE_2025-05-15", 0.37, 9.15, False),    # real crash
            ("SKIN_2023-11-14", 0.36, 6.41, False),   # real crash
            ("ADTX_2026-06-17", 0.36, 1.36, False),   # real dilution, not a split
            ("MLTX_2025-09-29", 0.10, 18.21, False),  # real trial-failure crash
            ("ATYR_2025-09-15", 0.17, 3.92, False),   # real trial-failure crash
        ],
    )
    def test_spec_row_verdict(self, label, price_ratio, volume_ratio, expect_split):
        pre_close, pre_vol = self._PRE_CLOSE, self._PRE_VOL
        post_close = pre_close * price_ratio
        post_vol = pre_vol * volume_ratio
        df, boundary = _boundary_frame(pre_close, pre_vol, post_close, post_vol)

        events = sf.detect_unadjusted_splits(df)
        if expect_split:
            assert len(events) == 1, label
            assert events[0].date == boundary, label
        else:
            assert events == [], label


class TestVolumeMustFallProportionally:
    """A price jump of factor N is only a split if volume fell about N-fold.

    Requiring merely that volume went DOWN is satisfied by a volume ratio of
    0.974, which is volume essentially unchanged. That let six real market
    moves through as "split" in a full-cache scan, two of them on tickers the
    backtest had traded. The rows below are those exact six, plus the eight
    genuine splits that survived, taken from that scan.
    """

    _PRE_CLOSE = 10.0
    _PRE_VOL = 100_000.0  # $1M pre-jump dollar volume, clear of the floor.

    @pytest.mark.parametrize(
        "label,price_ratio,volume_ratio,expect_split",
        [
            # Genuine reverse splits: volume collapses with the share count.
            ("FFAI_2026-07-23", 96.4286, 0.029552, True),
            ("TTOO_2022-10-13", 45.5357, 0.062557, True),
            ("ASBP_2026-05-11", 29.8378, 0.013568, True),
            ("DKI_2026-05-11", 16.6960, 0.056340, True),
            ("CMCT_2025-01-06", 9.88235, 0.113967, True),
            ("OPAD_2026-06-05", 8.33784, 0.122233, True),
            ("NXTP_2023-12-22", 3.016575, 0.213450, True),
            ("AIRJ_2024-03-11", 2.814423, 0.373714, True),
            # Real market moves. Price roughly tripled while volume barely
            # moved, which no share-count change can produce.
            ("OCGN_2021-02-08", 3.011429, 0.974429, False),  # COVAXIN run
            ("PFSA_2025-07-28", 2.852941, 0.884977, False),
            ("CNVS_2020-06-04", 2.770992, 0.835675, False),  # Jun-2020 meme
            ("CHRD_2020-03-13", 2.702702, 0.996429, False),  # COVID rebound
            ("VISL_2020-06-04", 2.651163, 0.841350, False),  # Jun-2020 meme
            ("AHT_2020-03-19", 2.551020, 0.914388, False),   # COVID rebound
        ],
    )
    def test_proportional_volume_fall(self, label, price_ratio, volume_ratio, expect_split):
        post_close = self._PRE_CLOSE * price_ratio
        post_vol = self._PRE_VOL * volume_ratio
        df, boundary = _boundary_frame(self._PRE_CLOSE, self._PRE_VOL, post_close, post_vol)

        events = sf.detect_unadjusted_splits(df)
        if expect_split:
            assert len(events) == 1, f"{label} should be a split"
            assert events[0].date == boundary, label
        else:
            assert events == [], f"{label} is a real move, not a split"

    def test_volume_unchanged_is_never_a_split(self):
        # The degenerate case the old rule missed by the widest margin:
        # price triples, volume is flat. Rejected at any plausible exponent.
        df, _ = _boundary_frame(10.0, 100_000.0, 30.0, 100_000.0)
        assert sf.detect_unadjusted_splits(df) == []


# ---------------------------------------------------------------------------
# Sub-penny OTC noise must be "untradeable", never "split"
# ---------------------------------------------------------------------------
class TestSubPennyScreen:
    def test_aagr_shaped_oscillation_is_not_split(self):
        # AAGR-shaped: closes oscillate between 0.0099 and 0.0005 on
        # 100-2000 share volume, day after day. Every boundary clears
        # PRICE_RATIO_THRESHOLD but both closes are under MIN_PRE_JUMP_CLOSE.
        n = 20
        closes = [0.0099 if i % 2 == 0 else 0.0005 for i in range(n)]
        volumes = [100.0 + 50.0 * (i % 20) for i in range(n)]
        dates = _dates(date(2024, 1, 1), n)
        df = _frame(dates, closes, volumes)

        events = sf.detect_unadjusted_splits(df)
        assert events == []

        rows = sf._scan_boundaries(df)
        flagged = [r for r in rows if r.verdict != "insufficient_context"]
        assert flagged, "expected some boundaries to clear the price-ratio gate"
        assert all(r.verdict == "untradeable" for r in flagged)
        assert not any(r.verdict == "split" for r in rows)

    def test_pre_jump_close_just_under_floor_is_untradeable(self):
        df, boundary = _boundary_frame(0.0099, 5000.0, 0.396, 1250.0)
        events = sf.detect_unadjusted_splits(df)
        assert events == []
        rows = sf._scan_boundaries(df)
        boundary_row = next(r for r in rows if r.date == boundary)
        assert boundary_row.verdict == "untradeable"


# ---------------------------------------------------------------------------
# Thin pre-jump dollar volume must be "untradeable", never "split", even
# when the price/volume shape looks like a textbook clean reverse split.
# ---------------------------------------------------------------------------
class TestDollarVolumeFloorScreen:
    def test_thin_dollar_volume_reverse_split_shape_is_untradeable(self):
        # Clean reverse-split shape (price x4, volume x0.25, dollar ratio
        # 1.0) but pre-jump dollar volume is only 10.0 * 50 = $500, far
        # below MIN_PRE_JUMP_DOLLAR_VOLUME ($500,000).
        df, boundary = _boundary_frame(10.0, 50.0, 40.0, 12.5)
        events = sf.detect_unadjusted_splits(df)
        assert events == []
        rows = sf._scan_boundaries(df)
        boundary_row = next(r for r in rows if r.date == boundary)
        assert boundary_row.verdict == "untradeable"

    def test_dollar_volume_just_above_floor_is_still_evaluated_as_split(self):
        # Same clean shape, but pre-jump dollar volume is comfortably above
        # the $500k floor -- confirms the floor is the reason the previous
        # test is rejected, not some other change.
        df, boundary = _boundary_frame(10.0, 80_000.0, 40.0, 20_000.0)
        events = sf.detect_unadjusted_splits(df)
        assert len(events) == 1
        assert events[0].date == boundary


# ---------------------------------------------------------------------------
# Multiple splits compound
# ---------------------------------------------------------------------------
class TestMultipleSplitsCompound:
    def test_two_splits_detected_and_adjustment_compounds(self):
        # Volumes scaled 10x vs. the smallest reverse-split example elsewhere
        # in this file so every boundary's pre-jump dollar volume (>= $1.6M)
        # clears MIN_PRE_JUMP_DOLLAR_VOLUME ($500k); the volume RATIOS
        # between segments (0.25, then 0.2) are unchanged, so the inferred
        # factors [4, 5] this test asserts on are unaffected.
        seg_a_c, seg_a_v = _flat_segment(15, 2.5, 800_000.0)
        seg_b_c, seg_b_v = _flat_segment(15, 10.0, 200_000.0)
        seg_c_c, seg_c_v = _flat_segment(15, 50.0, 40_000.0)
        seg_a_c[-1] = 2.5
        seg_b_c[0] = 10.0
        seg_b_c[-1] = 10.0
        seg_c_c[0] = 50.0

        closes = seg_a_c + seg_b_c + seg_c_c
        volumes = seg_a_v + seg_b_v + seg_c_v
        dates = _dates(date(2018, 1, 1), len(closes))
        df = _frame(dates, closes, volumes)

        events = sf.detect_unadjusted_splits(df)
        assert len(events) == 2
        factors = sorted(round(e.inferred_factor) for e in events)
        assert factors == [4, 5]

        adjusted = sf.apply_adjustment(df, events).sort_index()
        adj_closes = adjusted["close"].to_numpy()
        for i in range(1, len(adj_closes)):
            jump = max(adj_closes[i] / adj_closes[i - 1], adj_closes[i - 1] / adj_closes[i])
            assert jump < 2.0, f"discontinuity remains at index {i}"

        boundary_a = pd.Timestamp(dates[14])
        assert adjusted.loc[boundary_a, "close"] == pytest.approx(
            df.sort_index().loc[boundary_a, "close"] * 20.0, rel=0.05
        )
        assert adjusted.loc[boundary_a, "volume"] == pytest.approx(
            df.sort_index().loc[boundary_a, "volume"] / 20.0, rel=0.05
        )

        boundary_b = pd.Timestamp(dates[29])
        assert adjusted.loc[boundary_b, "close"] == pytest.approx(
            df.sort_index().loc[boundary_b, "close"] * 5.0, rel=0.05
        )

        newest = pd.Timestamp(dates[-1])
        assert adjusted.loc[newest, "close"] == pytest.approx(df.sort_index().loc[newest, "close"])


# ---------------------------------------------------------------------------
# apply_adjustment invariants
# ---------------------------------------------------------------------------
class TestApplyAdjustmentInvariants:
    def test_does_not_mutate_input(self):
        df, boundary = _boundary_frame(10.0, 80_000.0, 40.0, 20_000.0)
        before = df.copy(deep=True)
        events = sf.detect_unadjusted_splits(df)
        assert events

        sf.apply_adjustment(df, events)

        pd.testing.assert_frame_equal(df, before)

    def test_no_events_returns_equivalent_frame(self):
        closes, volumes = _flat_segment(30, 20.0, 5000.0)
        dates = _dates(date(2023, 1, 1), len(closes))
        df = _frame(dates, closes, volumes)

        adjusted = sf.apply_adjustment(df, [])
        pd.testing.assert_frame_equal(adjusted.sort_index(), df.sort_index())

    def test_dollar_volume_is_recomputed_not_stale(self):
        df, boundary = _boundary_frame(10.0, 80_000.0, 40.0, 20_000.0)
        # Corrupt dollar_volume so it no longer equals close * volume; if
        # apply_adjustment merely rescaled the stale column instead of
        # recomputing it, this corruption would survive into the output.
        df = df.copy()
        df["dollar_volume"] = 999_999.0

        events = sf.detect_unadjusted_splits(df)
        adjusted = sf.apply_adjustment(df, events)

        expected = adjusted["close"] * adjusted["volume"]
        pd.testing.assert_series_equal(
            adjusted["dollar_volume"], expected, check_names=False
        )

    def test_integer_volume_column_does_not_crash(self):
        df, boundary = _boundary_frame(4.0, 163_400.0, 80.0, 8_160.0)
        df = df.copy()
        df["volume"] = df["volume"].round().astype("int64")
        df["dollar_volume"] = df["close"] * df["volume"]

        events = sf.detect_unadjusted_splits(df)
        assert events
        adjusted = sf.apply_adjustment(df, events)
        assert adjusted["volume"].dtype == np.float64


# ---------------------------------------------------------------------------
# Degenerate inputs
# ---------------------------------------------------------------------------
class TestDegenerateInputs:
    def test_zero_volume_both_sides_is_not_flagged(self):
        closes = [10.0] * 15 + [40.0] * 15
        volumes = [0.0] * 30
        dates = _dates(date(2019, 1, 1), len(closes))
        df = _frame(dates, closes, volumes)

        events = sf.detect_unadjusted_splits(df)
        assert events == []

    def test_nan_volume_is_not_flagged_and_does_not_crash(self):
        closes = [10.0] * 15 + [40.0] * 15
        volumes = [float("nan")] * 30
        dates = _dates(date(2019, 1, 1), len(closes))
        df = _frame(dates, closes, volumes)

        events = sf.detect_unadjusted_splits(df)
        assert events == []

    def test_fewer_than_min_rows_does_not_crash(self):
        closes = [10.0, 10.1, 40.0, 40.1, 40.2]
        volumes = [1000.0, 900.0, 250.0, 260.0, 240.0]
        dates = _dates(date(2020, 1, 1), len(closes))
        df = _frame(dates, closes, volumes)

        assert len(df) < sf.MIN_ROWS
        events = sf.detect_unadjusted_splits(df)
        assert events == []

    def test_boundary_at_very_first_row_is_not_flagged(self):
        # 12 rows total (clears MIN_ROWS), but the jump sits at index 1,
        # inside MIN_CONTEXT_BARS of the start of the history.
        closes = [10.0, 40.0] + [40.0 + 0.1 * i for i in range(10)]
        volumes = [8000.0, 2000.0] + [2000.0] * 10
        dates = _dates(date(2020, 1, 1), len(closes))
        df = _frame(dates, closes, volumes)

        events = sf.detect_unadjusted_splits(df)
        assert events == []

        rows = sf._scan_boundaries(df)
        assert any(r.verdict == "insufficient_context" for r in rows)

    def test_zero_prior_close_does_not_crash(self):
        closes = [0.0, 10.0, 10.1, 10.2] + [10.0 + 0.05 * i for i in range(10)]
        volumes = [1000.0] * len(closes)
        dates = _dates(date(2020, 1, 1), len(closes))
        df = _frame(dates, closes, volumes)

        events = sf.detect_unadjusted_splits(df)  # must not raise
        assert isinstance(events, list)

    def test_empty_frame_returns_empty_list(self):
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume", "dollar_volume"])
        assert sf.detect_unadjusted_splits(df) == []

    def test_none_frame_returns_empty_list(self):
        assert sf.detect_unadjusted_splits(None) == []

    def test_missing_volume_column_returns_empty_list(self):
        closes, _ = _flat_segment(15, 10.0, 1.0)
        dates = _dates(date(2020, 1, 1), len(closes))
        idx = pd.DatetimeIndex([pd.Timestamp(d) for d in dates], name="date")
        df = pd.DataFrame({"open": closes, "high": closes, "low": closes, "close": closes}, index=idx)

        assert sf.detect_unadjusted_splits(df) == []
