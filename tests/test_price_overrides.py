"""Tests for price_overrides.py: durable manual split corrections that
survive a price-cache re-fetch.

Follows this repo's pytest conventions: plain pytest, class-grouped tests,
minimal fixtures, synthetic frames only (see tests/test_split_fingerprint.py
and tests/test_prices_merge.py). No network access. No reads or writes
against the real price_cache/ directory or the real price_overrides.json --
every disk-touching test uses tmp_path and monkeypatches path constants.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import date, timedelta

import pandas as pd
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from backtest import price_overrides as po  # noqa: E402
from backtest import prices as prices_mod  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------
def _dates(start: date, n: int) -> list[date]:
    return [start + timedelta(days=i) for i in range(n)]


def _frame(dates: list[date], closes: list[float], volumes: list[float]) -> pd.DataFrame:
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


def _split_frame(
    pre_close: float, pre_vol: float, post_close: float, post_vol: float,
    seg_len: int = 10, start: date = date(2020, 1, 1),
) -> tuple[pd.DataFrame, date]:
    """A flat pre-boundary segment followed by a flat post-boundary segment,
    with an unadjusted (raw) jump at the boundary date. Mirrors
    tests/test_split_fingerprint.py's _boundary_frame."""
    dates = _dates(start, 2 * seg_len)
    closes = [pre_close] * seg_len + [post_close] * seg_len
    volumes = [pre_vol] * seg_len + [post_vol] * seg_len
    df = _frame(dates, closes, volumes)
    return df, dates[seg_len]


def _write_overrides(path: str, entries: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"overrides": entries}, fh)


@pytest.fixture(autouse=True)
def _reset_overrides_cache():
    """price_overrides.py memoizes the parsed file in module globals; make
    sure no test's cache leaks into the next one."""
    po._cache = None
    po._cache_path = None
    po._cache_mtime = None
    yield
    po._cache = None
    po._cache_path = None
    po._cache_mtime = None


# ---------------------------------------------------------------------------
# load_overrides / parsing
# ---------------------------------------------------------------------------
class TestLoadOverrides:
    def test_missing_file_returns_empty_dict(self, tmp_path):
        path = str(tmp_path / "does_not_exist.json")
        assert po.load_overrides(path) == {}

    def test_parses_seeded_entry(self, tmp_path):
        path = str(tmp_path / "price_overrides.json")
        _write_overrides(path, [{
            "ticker": "aaa", "date": "2020-06-01", "price_ratio": 4.0,
            "who": "unit test", "why": "synthetic",
            "observed_volume_ratio": 0.25, "observed_dollar_volume_ratio": 1.0,
        }])

        overrides = po.load_overrides(path)
        assert set(overrides) == {"AAA"}  # ticker is upper-cased
        ev = overrides["AAA"][0]
        assert ev.date == date(2020, 6, 1)
        assert ev.price_ratio == pytest.approx(4.0)
        assert ev.observed_volume_ratio == pytest.approx(0.25)

    def test_malformed_entry_is_skipped_not_fatal(self, tmp_path, caplog):
        path = str(tmp_path / "price_overrides.json")
        _write_overrides(path, [
            {"ticker": "AAA", "date": "2020-06-01", "price_ratio": 4.0},
            {"ticker": "BBB", "date": "not-a-date", "price_ratio": 4.0},  # malformed
            {"ticker": "CCC"},  # missing required fields
        ])
        with caplog.at_level(logging.WARNING):
            overrides = po.load_overrides(path)
        assert set(overrides) == {"AAA"}
        assert sum(len(v) for v in overrides.values()) == 1

    def test_cache_invalidates_on_mtime_change(self, tmp_path):
        path = str(tmp_path / "price_overrides.json")
        _write_overrides(path, [{"ticker": "AAA", "date": "2020-06-01", "price_ratio": 4.0}])
        first = po.load_overrides(path)
        assert set(first) == {"AAA"}

        _write_overrides(path, [{"ticker": "BBB", "date": "2021-01-01", "price_ratio": 2.0}])
        os.utime(path, (os.path.getmtime(path) + 5, os.path.getmtime(path) + 5))
        second = po.load_overrides(path)
        assert set(second) == {"BBB"}


# ---------------------------------------------------------------------------
# apply_price_overrides: core behavior
# ---------------------------------------------------------------------------
class TestApplyPriceOverrides:
    def test_unknown_ticker_is_noop(self, tmp_path):
        path = str(tmp_path / "price_overrides.json")
        _write_overrides(path, [{"ticker": "AAA", "date": "2020-01-11", "price_ratio": 4.0}])
        df, _boundary = _split_frame(10.0, 1000.0, 40.0, 250.0)

        out = po.apply_price_overrides("ZZZZ", df, path=path)
        pd.testing.assert_frame_equal(out, df)

    def test_ticker_with_no_overrides_file_is_noop(self, tmp_path):
        path = str(tmp_path / "does_not_exist.json")
        df, _boundary = _split_frame(10.0, 1000.0, 40.0, 250.0)
        out = po.apply_price_overrides("AAA", df, path=path)
        pd.testing.assert_frame_equal(out, df)

    def test_override_applied_when_jump_present(self, tmp_path, caplog):
        path = str(tmp_path / "price_overrides.json")
        df, boundary = _split_frame(10.0, 1000.0, 40.0, 250.0)
        _write_overrides(path, [{
            "ticker": "AAA", "date": boundary.isoformat(), "price_ratio": 4.0,
            "who": "unit test", "why": "synthetic",
            "observed_volume_ratio": 0.25, "observed_dollar_volume_ratio": 1.0,
        }])

        with caplog.at_level(logging.WARNING):
            out = po.apply_price_overrides("AAA", df, path=path)

        assert any(
            r.levelno == logging.WARNING and "applying manual split override" in r.message
            for r in caplog.records
        )
        assert any("AAA" in r.message for r in caplog.records)

        ts = pd.Timestamp(boundary)
        pos = out.index.get_loc(ts)
        before = out["close"].iloc[pos - 1]
        after = out["close"].iloc[pos]
        assert after / before == pytest.approx(1.0, abs=0.05)
        assert out["close"].iloc[0] == pytest.approx(df["close"].iloc[0] * 4.0)

    def test_skip_when_jump_not_present_logs_info(self, tmp_path, caplog):
        # Flat frame -- no jump anywhere -- but an override is recorded for a
        # date inside it. Simulates re-loading an already-corrected frame, or
        # a ticker Yahoo has since fixed itself: apply_price_overrides must
        # NOT re-adjust it a second time.
        path = str(tmp_path / "price_overrides.json")
        dates = _dates(date(2020, 1, 1), 20)
        df = _frame(dates, [10.0] * 20, [1000.0] * 20)
        override_date = dates[10]
        _write_overrides(path, [{
            "ticker": "AAA", "date": override_date.isoformat(), "price_ratio": 4.0,
        }])

        with caplog.at_level(logging.INFO):
            out = po.apply_price_overrides("AAA", df, path=path)

        pd.testing.assert_frame_equal(out, df)
        assert any(
            r.levelno == logging.INFO and "not applied" in r.message for r in caplog.records
        )
        assert not any("applying manual split override" in r.message for r in caplog.records)

    def test_idempotent_second_application_is_noop(self, tmp_path):
        path = str(tmp_path / "price_overrides.json")
        df, boundary = _split_frame(10.0, 1000.0, 40.0, 250.0)
        _write_overrides(path, [{
            "ticker": "AAA", "date": boundary.isoformat(), "price_ratio": 4.0,
        }])

        once = po.apply_price_overrides("AAA", df, path=path)
        twice = po.apply_price_overrides("AAA", once, path=path)

        pd.testing.assert_frame_equal(once.sort_index(), twice.sort_index())

    def test_does_not_mutate_input(self, tmp_path):
        path = str(tmp_path / "price_overrides.json")
        df, boundary = _split_frame(10.0, 1000.0, 40.0, 250.0)
        before = df.copy(deep=True)
        _write_overrides(path, [{
            "ticker": "AAA", "date": boundary.isoformat(), "price_ratio": 4.0,
        }])

        po.apply_price_overrides("AAA", df, path=path)

        pd.testing.assert_frame_equal(df, before)

    def test_none_frame_is_noop(self, tmp_path):
        path = str(tmp_path / "price_overrides.json")
        _write_overrides(path, [{"ticker": "AAA", "date": "2020-01-11", "price_ratio": 4.0}])
        assert po.apply_price_overrides("AAA", None, path=path) is None

    def test_empty_frame_is_noop(self, tmp_path):
        path = str(tmp_path / "price_overrides.json")
        _write_overrides(path, [{"ticker": "AAA", "date": "2020-01-11", "price_ratio": 4.0}])
        empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume", "dollar_volume"])
        out = po.apply_price_overrides("AAA", empty, path=path)
        assert out.empty


# ---------------------------------------------------------------------------
# Integration: the seam inside backtest/prices.py (the load path)
# ---------------------------------------------------------------------------
@pytest.fixture()
def isolated_universe(tmp_path, monkeypatch):
    """Isolate both the price cache and price_overrides.json in tmp_path, so
    tests here never touch the real price_cache/ or the real
    price_overrides.json at the project root."""
    cache_dir = str(tmp_path / "price_cache")
    meta_dir = os.path.join(cache_dir, "_meta")
    monkeypatch.setattr(prices_mod, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(prices_mod, "META_DIR", meta_dir)
    monkeypatch.setattr(prices_mod, "MISSING_PATH", os.path.join(cache_dir, "_missing.json"))
    monkeypatch.setattr(prices_mod, "NEEDS_REFETCH_PATH", os.path.join(cache_dir, "_needs_refetch.json"))
    # price_overrides.apply_price_overrides defaults to the cwd-relative
    # "price_overrides.json"; chdir makes that resolve inside tmp_path.
    monkeypatch.chdir(tmp_path)
    return cache_dir


class TestLoadSeamAppliesOverride:
    def test_load_cached_applies_confirmed_override(self, isolated_universe, caplog):
        cache_dir = isolated_universe
        df, boundary = _split_frame(10.0, 1000.0, 40.0, 250.0)
        os.makedirs(cache_dir, exist_ok=True)
        df.to_parquet(os.path.join(cache_dir, "AAA.parquet"))
        _write_overrides("price_overrides.json", [{
            "ticker": "AAA", "date": boundary.isoformat(), "price_ratio": 4.0,
        }])

        pu = prices_mod.PriceUniverse()
        with caplog.at_level(logging.WARNING):
            loaded = pu._load_cached("AAA")

        assert any("applying manual split override" in r.message for r in caplog.records)
        ts = pd.Timestamp(boundary)
        pos = loaded.index.get_loc(ts)
        before = loaded["close"].iloc[pos - 1]
        after = loaded["close"].iloc[pos]
        assert after / before == pytest.approx(1.0, abs=0.05)

    def test_load_cached_unknown_ticker_is_noop(self, isolated_universe):
        cache_dir = isolated_universe
        df, boundary = _split_frame(10.0, 1000.0, 40.0, 250.0)
        os.makedirs(cache_dir, exist_ok=True)
        df.to_parquet(os.path.join(cache_dir, "ZZZZ.parquet"))
        _write_overrides("price_overrides.json", [{
            "ticker": "AAA", "date": boundary.isoformat(), "price_ratio": 4.0,
        }])

        pu = prices_mod.PriceUniverse()
        loaded = pu._load_cached("ZZZZ")

        # No override applies to ZZZZ, so the raw jump is left untouched.
        ts = pd.Timestamp(boundary)
        pos = loaded.index.get_loc(ts)
        before = loaded["close"].iloc[pos - 1]
        after = loaded["close"].iloc[pos]
        assert after / before == pytest.approx(4.0, rel=1e-6)

    def test_repeated_load_does_not_double_adjust(self, isolated_universe):
        cache_dir = isolated_universe
        df, boundary = _split_frame(10.0, 1000.0, 40.0, 250.0)
        os.makedirs(cache_dir, exist_ok=True)
        df.to_parquet(os.path.join(cache_dir, "AAA.parquet"))
        _write_overrides("price_overrides.json", [{
            "ticker": "AAA", "date": boundary.isoformat(), "price_ratio": 4.0,
        }])

        pu = prices_mod.PriceUniverse()
        first = pu._load_cached("AAA")
        second = pu._load_cached("AAA")

        pd.testing.assert_frame_equal(first.sort_index(), second.sort_index())
        ts = pd.Timestamp(boundary)
        pos = first.index.get_loc(ts)
        # Still ~1x across the boundary, not re-scaled a second time (which
        # would read ~4x again on top of the already-corrected value).
        assert first["close"].iloc[pos] / first["close"].iloc[pos - 1] == pytest.approx(1.0, abs=0.05)


# ---------------------------------------------------------------------------
# Regression: the fetch path must ALSO get the override, or the merge logic
# reads the correction itself as a spurious basis change and discards it --
# this is the exact mechanism behind the 2026-08-07 DKI corruption.
# ---------------------------------------------------------------------------
class TestFetchPathPreventsMergeFromDiscardingOverride:
    def test_uncorrected_refetch_would_discard_the_fix(self, isolated_universe):
        # Negative control: proves the bug this module fixes is real. A
        # corrected `existing` merged against a RAW (un-overridden) `new`
        # over the same full date range reads the correction as a basis
        # change and keeps `new` (the still-broken data) almost everywhere.
        pu = prices_mod.PriceUniverse()
        raw, boundary = _split_frame(10.0, 1000.0, 40.0, 250.0, seg_len=15)
        _write_overrides("price_overrides.json", [{
            "ticker": "AAA", "date": boundary.isoformat(), "price_ratio": 4.0,
        }])
        existing = po.apply_price_overrides("AAA", raw)  # simulates the load path only
        new_raw = raw.copy(deep=True)  # simulates a fresh, still-broken Yahoo fetch

        merged = pu._merge_price_frames("AAA", existing, new_raw)

        ts = pd.Timestamp(boundary)
        pos = merged.sort_index().index.get_loc(ts)
        closes = merged.sort_index()["close"].to_numpy()
        # The raw ~4x discontinuity is back, proving the fix was discarded.
        assert closes[pos] / closes[pos - 1] > 3.0

    def test_corrected_refetch_preserves_the_fix(self, isolated_universe):
        pu = prices_mod.PriceUniverse()
        raw, boundary = _split_frame(10.0, 1000.0, 40.0, 250.0, seg_len=15)
        _write_overrides("price_overrides.json", [{
            "ticker": "AAA", "date": boundary.isoformat(), "price_ratio": 4.0,
        }])
        existing = po.apply_price_overrides("AAA", raw)  # load path
        new_raw = raw.copy(deep=True)  # fresh, still-broken Yahoo fetch
        new_corrected = po.apply_price_overrides("AAA", new_raw)  # fetch-path hook

        merged = pu._merge_price_frames("AAA", existing, new_corrected)

        closes = merged.sort_index()["close"].to_numpy()
        for i in range(1, len(closes)):
            jump = max(closes[i] / closes[i - 1], closes[i - 1] / closes[i])
            assert jump < 1.6, f"discontinuity remains at index {i}"
