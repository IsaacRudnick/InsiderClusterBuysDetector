"""Tests for the basis-aware price-cache merge and the repair-scan logic.

Follows this repo's pytest conventions: plain pytest, class-grouped tests,
minimal fixtures. No network access. No writes into the real price_cache/
directory -- every disk-touching test monkeypatches the relevant
module-level path constants to a tmp_path.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date, timedelta

import pandas as pd
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from backtest import prices as prices_mod  # noqa: E402
from tools import repair_price_cache  # noqa: E402


def _frame(dates: list[date], closes: list[float], volumes: list[float] | None = None) -> pd.DataFrame:
    n = len(dates)
    if volumes is None:
        volumes = [1_000_000.0] * n
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


def _dates(start: date, n: int) -> list[date]:
    return [start + timedelta(days=i) for i in range(n)]


@pytest.fixture()
def isolated_cache(tmp_path, monkeypatch):
    """Point every module-level cache path at a tmp dir. No test in this
    file may read or write the real price_cache/ directory."""
    cache_dir = str(tmp_path / "price_cache")
    meta_dir = os.path.join(cache_dir, "_meta")
    missing_path = os.path.join(cache_dir, "_missing.json")
    needs_refetch_path = os.path.join(cache_dir, "_needs_refetch.json")
    monkeypatch.setattr(prices_mod, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(prices_mod, "META_DIR", meta_dir)
    monkeypatch.setattr(prices_mod, "MISSING_PATH", missing_path)
    monkeypatch.setattr(prices_mod, "NEEDS_REFETCH_PATH", needs_refetch_path)
    return cache_dir


# ---------------------------------------------------------------------------
# Part 1: PriceUniverse._merge_price_frames
# ---------------------------------------------------------------------------
class TestMergeSplitBetweenFetches:
    """A real split (~20x, like DRIO) between two fetches must not fabricate
    a jump at the old cache boundary."""

    def test_stale_segment_rescaled_no_discontinuity(self, isolated_cache):
        pu = prices_mod.PriceUniverse()
        start = date(2019, 10, 1)

        existing_dates = _dates(start, 20)
        existing = _frame(existing_dates, [4.00 + 0.01 * i for i in range(20)])
        # Override the last existing close, same shape as the real DRIO row.
        existing.iloc[-1, existing.columns.get_loc("close")] = 4.14
        existing.iloc[-1, existing.columns.get_loc("open")] = 4.14

        overlap_dates = existing_dates[-5:]
        new_dates = overlap_dates + _dates(existing_dates[-1] + timedelta(days=1), 10)
        ratio = 20.0
        overlap_closes = [
            existing.loc[pd.Timestamp(d), "close"] * ratio for d in overlap_dates
        ]
        fresh_closes = [102.80 + i for i in range(10)]
        new = _frame(new_dates, overlap_closes + fresh_closes)

        merged = pu._merge_price_frames("DRIO", existing, new)

        # No >= 3x close-to-close jump anywhere, including across the old
        # existing/new boundary.
        closes = merged.sort_index()["close"].to_numpy()
        for i in range(1, len(closes)):
            jump = max(closes[i] / closes[i - 1], closes[i - 1] / closes[i])
            assert jump < 3.0, f"discontinuity at index {i}: {closes[i - 1]} -> {closes[i]}"

        # The stale, non-overlapping existing rows got rescaled by ~ratio.
        boundary_date = pd.Timestamp(existing_dates[-6])  # last non-overlap day
        assert merged.loc[boundary_date, "close"] == pytest.approx(
            existing.loc[boundary_date, "close"] * ratio, rel=1e-6
        )
        assert merged.loc[boundary_date, "volume"] == pytest.approx(
            existing.loc[boundary_date, "volume"] / ratio, rel=1e-6
        )
        expected_dv = merged.loc[boundary_date, "close"] * merged.loc[boundary_date, "volume"]
        assert merged.loc[boundary_date, "dollar_volume"] == pytest.approx(expected_dv)

        # New wins outright on overlapping dates.
        for d in overlap_dates:
            ts = pd.Timestamp(d)
            assert merged.loc[ts, "close"] == pytest.approx(new.loc[ts, "close"])


class TestMergeNormalIncremental:
    """No split: behavior must stay identical to the pre-fix two-line merge."""

    def test_ordinary_noise_preserves_existing_appends_new(self, isolated_cache):
        pu = prices_mod.PriceUniverse()
        start = date(2023, 1, 2)
        existing_dates = _dates(start, 10)
        existing = _frame(existing_dates, [50.0 + 0.1 * i for i in range(10)])

        overlap_dates = existing_dates[-3:]
        new_dates = overlap_dates + _dates(existing_dates[-1] + timedelta(days=1), 4)
        # Overlap drifts well under 2%: ordinary noise, not a split.
        overlap_closes = [
            existing.loc[pd.Timestamp(d), "close"] * 1.001 for d in overlap_dates
        ]
        fresh_closes = [51.0 + i for i in range(4)]
        new = _frame(new_dates, overlap_closes + fresh_closes)

        merged = pu._merge_price_frames("ABC", existing, new)
        expected = pd.concat(
            [existing, new[~new.index.isin(existing.index)]]
        ).sort_index()

        pd.testing.assert_frame_equal(merged.sort_index(), expected)

    def test_existing_wins_on_overlap(self, isolated_cache):
        pu = prices_mod.PriceUniverse()
        start = date(2023, 1, 2)
        existing_dates = _dates(start, 5)
        existing = _frame(existing_dates, [10.0, 10.1, 10.2, 10.3, 10.4])
        overlap_dates = existing_dates[-2:]
        new_dates = overlap_dates + _dates(existing_dates[-1] + timedelta(days=1), 2)
        new = _frame(new_dates, [10.19, 10.41, 10.5, 10.6])  # basis unchanged

        merged = pu._merge_price_frames("ABC", existing, new)
        for d in overlap_dates:
            ts = pd.Timestamp(d)
            assert merged.loc[ts, "close"] == pytest.approx(existing.loc[ts, "close"])


class TestMergeDisjointRange:
    """Zero overlap: must not silently concatenate, must record for refetch."""

    def test_disjoint_range_recorded_and_not_concatenated(self, isolated_cache):
        pu = prices_mod.PriceUniverse()
        existing_dates = _dates(date(2020, 1, 1), 5)
        existing = _frame(existing_dates, [10.0, 10.1, 10.2, 10.3, 10.4])
        new_dates = _dates(date(2021, 1, 1), 5)  # entirely disjoint
        new = _frame(new_dates, [500.0, 501.0, 502.0, 503.0, 504.0])

        merged = pu._merge_price_frames("XYZ", existing, new)

        # Not a blind concat: merged is `new` alone, existing rows dropped
        # for now rather than welded on with an unknowable basis.
        assert set(merged.index) == set(new.index)
        assert not any(ts in merged.index for ts in existing.index)

        registry = prices_mod._load_needs_refetch()
        assert "XYZ" in registry
        assert registry["XYZ"]["reason"] == "disjoint_merge_range"


# ---------------------------------------------------------------------------
# Part 2: repair_price_cache.find_suspect_tickers
# ---------------------------------------------------------------------------
class TestFindSuspectTickers:
    def _write_parquet(self, cache_dir: str, ticker: str, df: pd.DataFrame) -> None:
        os.makedirs(cache_dir, exist_ok=True)
        df.to_parquet(os.path.join(cache_dir, f"{ticker}.parquet"))

    def test_detects_genuine_large_jump(self, tmp_path):
        cache_dir = str(tmp_path / "price_cache")
        dates = _dates(date(2019, 10, 1), 10)
        closes = [4.0 + 0.02 * i for i in range(9)] + [102.80]  # ~20x jump at end
        self._write_parquet(cache_dir, "DRIO", _frame(dates, closes))

        suspects = repair_price_cache.find_suspect_tickers(cache_dir)
        tickers = [t for t, _, _ in suspects]
        assert "DRIO" in tickers

    def test_does_not_flag_wild_but_real_volatility(self, tmp_path):
        cache_dir = str(tmp_path / "price_cache")
        dates = _dates(date(2020, 1, 1), 15)
        # Genuinely wild: daily moves up to +/-40%, always well under 3x
        # close-to-close, to prove real headroom below the threshold.
        closes = [10.0]
        moves = [0.35, -0.30, 0.40, -0.35, 0.30, -0.25, 0.38, -0.32,
                 0.20, -0.15, 0.10, -0.40, 0.25, -0.10]
        for m in moves:
            closes.append(closes[-1] * (1 + m))
        self._write_parquet(cache_dir, "WILD", _frame(dates, closes))

        suspects = repair_price_cache.find_suspect_tickers(cache_dir)
        tickers = [t for t, _, _ in suspects]
        assert "WILD" not in tickers

    def test_skips_non_ticker_files(self, tmp_path):
        cache_dir = str(tmp_path / "price_cache")
        os.makedirs(os.path.join(cache_dir, "_meta"), exist_ok=True)
        with open(os.path.join(cache_dir, "_missing.json"), "w", encoding="utf-8") as fh:
            json.dump({}, fh)
        dates = _dates(date(2022, 1, 1), 5)
        self._write_parquet(cache_dir, "CLEAN", _frame(dates, [20.0, 20.1, 20.2, 20.15, 20.3]))

        suspects = repair_price_cache.find_suspect_tickers(cache_dir)
        assert suspects == []
