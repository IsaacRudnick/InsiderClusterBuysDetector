"""Tests for tools/regime.py -- the market-regime overlay.

Two tests matter most, and they are named for exactly what they prove:

  test_lag_prevents_early_read       a FRED value is never readable on a
                                      trading day before its documented
                                      publication lag has elapsed. This is
                                      the whole point-in-time fix the module
                                      exists for; if this test is wrong,
                                      every regime-based backtest in this
                                      repo would be trading on lookahead.

  test_flat_mode_reproduces_book     apply_overlay(..., mode="flat") is the
                                      control -- it must return the input
                                      book UNCHANGED, so every other mode's
                                      effect can be read as a delta from a
                                      known-neutral baseline.

Everything else here pins the supporting pieces: the causal one-extra-day
shift in apply_overlay, the self-referential state flags, and the
entry_idx <-> calendar-date translation the CLI depends on.

All tests are OFFLINE. Any FRED data a test needs is written directly into a
`tmp_path` cache directory before the call, so `_fetch_series` finds it on
disk and never reaches the network (regime_frame(..., refresh=False) checks
the cache first -- see tools/regime.py).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools import regime as rg  # noqa: E402


def _write_fake_csv(cache_dir: str, fred_id: str, obs: list[tuple[pd.Timestamp, float]]) -> None:
    """Write a cache file in the exact shape fredgraph.csv returns, so
    _fetch_series reads it as if FRED had served it."""
    os.makedirs(cache_dir, exist_ok=True)
    lines = [f"observation_date,{fred_id}"]
    lines += [f"{d.strftime('%Y-%m-%d')},{v}" for d, v in obs]
    with open(os.path.join(cache_dir, f"{fred_id}.csv"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# 1. The point-in-time fix
# ---------------------------------------------------------------------------

def test_lag_prevents_early_read(tmp_path):
    calendar = pd.bdate_range("2024-01-02", periods=40)
    cache_dir = str(tmp_path)
    spec = rg.SeriesSpec("FAKE_A", lag_trading_days=3, revised=False, note="test")

    # One observation dated calendar[10] -- with lag=3 it must first be
    # readable on calendar[13] (the 3rd trading day strictly after it), and
    # nowhere earlier, including calendar[12].
    _write_fake_csv(cache_dir, "FAKE_A", [(calendar[10], 100.0)])

    frame = rg.regime_frame(calendar, series={"x": spec}, cache_dir=cache_dir)

    assert pd.isna(frame["x"].iloc[9])            # obs date itself: not yet known
    assert pd.isna(frame["x"].iloc[12])            # one trading day before publication
    assert frame["x"].iloc[13] == 100.0             # exactly the lagged publication day
    assert frame["x"].iloc[20] == 100.0             # forward-filled onward


def test_lag_prevents_early_read_second_observation(tmp_path):
    """A second, later observation must not leak early either -- the old
    value has to persist right up to (and not one day past) its own lagged
    publication day."""
    calendar = pd.bdate_range("2024-01-02", periods=40)
    cache_dir = str(tmp_path)
    spec = rg.SeriesSpec("FAKE_B", lag_trading_days=2, revised=False, note="test")
    _write_fake_csv(cache_dir, "FAKE_B", [(calendar[10], 100.0), (calendar[20], 200.0)])

    frame = rg.regime_frame(calendar, series={"x": spec}, cache_dir=cache_dir)

    known_day = 20 + spec.lag_trading_days  # pos(20)+1=21, +lag-1=22
    assert frame["x"].iloc[known_day - 1] == 100.0  # still the OLD value the day before
    assert frame["x"].iloc[known_day] == 200.0       # new value exactly on schedule


def test_regime_frame_never_backfills_before_first_observation(tmp_path):
    calendar = pd.bdate_range("2024-01-02", periods=20)
    cache_dir = str(tmp_path)
    spec = rg.SeriesSpec("FAKE_C", lag_trading_days=1, revised=False, note="test")
    _write_fake_csv(cache_dir, "FAKE_C", [(calendar[15], 5.0)])

    frame = rg.regime_frame(calendar, series={"x": spec}, cache_dir=cache_dir)
    assert frame["x"].iloc[:16].isna().all()
    assert frame["x"].iloc[16] == 5.0


# ---------------------------------------------------------------------------
# 2. apply_overlay
# ---------------------------------------------------------------------------

def _port(calendar, rets):
    return pd.DataFrame({"date": calendar, "ret": np.asarray(rets, dtype=float)})


def test_flat_mode_reproduces_book():
    calendar = pd.bdate_range("2024-01-02", periods=10)
    rng = np.random.default_rng(0)
    rets = rng.normal(0.0, 0.01, len(calendar))
    port = _port(calendar, rets)
    states = pd.DataFrame(
        {"risk_off": [1.0, 0.0] * 5}, index=calendar
    )

    out = rg.apply_overlay(port, states, "flat")

    np.testing.assert_array_equal(out["ret"].to_numpy(), rets)
    assert (out["exposure"] == 1.0).all()
    # ret_unlevered is kept too, and equals the same unmodified series
    np.testing.assert_array_equal(out["ret_unlevered"].to_numpy(), rets)


def test_risk_off_half_scales_by_half_with_one_day_lag():
    calendar = pd.bdate_range("2024-01-02", periods=6)
    rets = np.array([0.02, 0.02, 0.02, 0.02, 0.02, 0.02])
    port = _port(calendar, rets)
    # risk_off on days 1 and 3 (0-indexed); everything else risk_on.
    states = pd.DataFrame({"risk_off": [0.0, 1.0, 0.0, 1.0, 0.0, 0.0]}, index=calendar)

    out = rg.apply_overlay(port, states, "risk_off_half")

    # Exposure on day d comes from risk_off on day d-1 (causal shift). Day 0
    # has no prior day -> defaults to full exposure.
    expected_exposure = [1.0, 1.0, 0.5, 1.0, 0.5, 1.0]
    np.testing.assert_allclose(out["exposure"].to_numpy(), expected_exposure)
    np.testing.assert_allclose(out["ret"].to_numpy(), rets * np.array(expected_exposure))


def test_risk_off_flat_zeroes_the_day_after_risk_off():
    calendar = pd.bdate_range("2024-01-02", periods=4)
    rets = np.array([0.05, -0.03, 0.10, -0.10])
    port = _port(calendar, rets)
    states = pd.DataFrame({"risk_off": [1.0, 0.0, 0.0, 0.0]}, index=calendar)

    out = rg.apply_overlay(port, states, "risk_off_flat")

    assert out["exposure"].tolist() == [1.0, 0.0, 1.0, 1.0]
    assert out["ret"].iloc[1] == 0.0
    assert out["ret"].iloc[0] == rets[0]  # today's own risk_off does not affect today


def test_same_day_risk_off_does_not_affect_same_day_exposure():
    """The causal-shift guarantee, isolated: flipping risk_off on day d must
    only ever change exposure on day d+1, never day d itself."""
    calendar = pd.bdate_range("2024-01-02", periods=5)
    rets = np.full(5, 0.01)
    port = _port(calendar, rets)

    states_a = pd.DataFrame({"risk_off": [0.0, 0.0, 0.0, 0.0, 0.0]}, index=calendar)
    states_b = pd.DataFrame({"risk_off": [0.0, 0.0, 1.0, 0.0, 0.0]}, index=calendar)  # day 2 flipped

    out_a = rg.apply_overlay(port, states_a, "risk_off_flat")
    out_b = rg.apply_overlay(port, states_b, "risk_off_flat")

    assert out_a["exposure"].iloc[2] == out_b["exposure"].iloc[2]  # day 2 itself: unaffected
    assert out_a["exposure"].iloc[3] != out_b["exposure"].iloc[3]  # day 3: the flip shows up here


def test_apply_overlay_rejects_unknown_mode():
    calendar = pd.bdate_range("2024-01-02", periods=3)
    port = _port(calendar, [0.0, 0.0, 0.0])
    states = pd.DataFrame({"risk_off": [0.0, 0.0, 0.0]}, index=calendar)
    with pytest.raises(ValueError):
        rg.apply_overlay(port, states, "bogus_mode")


def test_unknown_regime_state_defaults_to_full_exposure():
    """NaN in risk_off (no history yet) must not be silently read as
    'de-risk' -- that would fabricate a signal from missing data."""
    calendar = pd.bdate_range("2024-01-02", periods=4)
    rets = np.full(4, 0.01)
    port = _port(calendar, rets)
    states = pd.DataFrame({"risk_off": [np.nan, np.nan, np.nan, np.nan]}, index=calendar)

    out = rg.apply_overlay(port, states, "risk_off_flat")
    assert (out["exposure"] == 1.0).all()


# ---------------------------------------------------------------------------
# 3. regime_state
# ---------------------------------------------------------------------------

def test_regime_state_flag_directions():
    calendar = pd.bdate_range("2022-01-03", periods=300)
    n = len(calendar)
    frame = pd.DataFrame(index=calendar)

    # VIX: flat at 15 for a year, then jumps to 40 for the tail -- the jump
    # should read vix_high=True only once the trailing median has caught up
    # to reflecting mostly-15 history, i.e. right after the jump.
    vix = np.full(n, 15.0)
    vix[260:] = 40.0
    frame["vix"] = vix

    # HY spread: flat, then widens sharply over the final 21 days.
    hy = np.full(n, 4.0)
    hy[-10:] = 6.0
    frame["hy_oas"] = hy

    # NFCI: negative (loose) for the first half, positive (tight) for the second.
    nfci = np.where(np.arange(n) < n // 2, -0.3, 0.3)
    frame["nfci"] = nfci

    # Term spread: positive (normal) throughout except the very end.
    term = np.full(n, 0.5)
    term[-5:] = -0.2
    frame["term_spread"] = term

    state = rg.regime_state(frame)

    assert state["vix_high"].iloc[265] == 1.0
    assert state["vix_high"].iloc[100] == 0.0
    assert pd.isna(state["vix_high"].iloc[10])  # not enough trailing history yet

    assert state["hy_widening"].iloc[-1] == 1.0
    assert state["hy_widening"].iloc[100] == 0.0

    assert state["nfci_tight"].iloc[0] == 0.0
    assert state["nfci_tight"].iloc[-1] == 1.0

    assert state["term_inverted"].iloc[-1] == 1.0
    assert state["term_inverted"].iloc[0] == 0.0

    # Near the very end, all four flags are known and at least 3 are True
    # (vix_high, hy_widening, term_inverted) -> risk_off must be a majority.
    assert state["n_known"].iloc[-1] == 4
    assert state["risk_off"].iloc[-1] == 1.0


def test_regime_state_missing_series_leaves_flag_nan_not_false():
    """Mirrors the real hy_oas gap before 2023-08-21: an entirely-missing
    input series must produce NaN flags, not False ones, and risk_off must
    still be computable from whatever else is known."""
    calendar = pd.bdate_range("2022-01-03", periods=300)
    n = len(calendar)
    frame = pd.DataFrame(index=calendar)
    # Low for most of history, then a late jump -- so the trailing median
    # (dominated by the low regime) sits well below the final value, and
    # vix_high reads True at the end (a flat, constant VIX would equal its
    # own median forever and never flag -- that's not what "elevated" means).
    vix = np.full(n, 15.0)
    vix[260:] = 40.0
    frame["vix"] = vix
    frame["hy_oas"] = np.nan          # entirely missing, like pre-2023-08-21
    frame["nfci"] = np.full(n, 0.5)   # tight the whole time
    frame["term_spread"] = np.full(n, 0.5)  # never inverted

    state = rg.regime_state(frame)

    assert state["hy_widening"].isna().all()
    assert state["n_known"].iloc[-1] == 3  # vix_high, nfci_tight, term_inverted only
    # 2 of 3 known flags True (vix_high, nfci_tight) -> majority -> risk_off
    assert state["risk_off"].iloc[-1] == 1.0


# ---------------------------------------------------------------------------
# 4. day_index_to_dates
# ---------------------------------------------------------------------------

def test_day_index_to_dates_round_trip():
    calendar = pd.bdate_range("2018-01-02", periods=500)
    offset = 37  # entry_idx = calendar position - offset, i.e. entry_idx 0 is calendar[37]
    positions = np.array([50, 51, 51, 80, 120])
    entry_idx = positions - offset
    entry_day = calendar[positions]

    day_idx = np.array([13, 14, 90])  # arbitrary 'day' values to translate
    dates = rg.day_index_to_dates(day_idx, pd.Series(entry_idx), pd.Series(entry_day), calendar)

    expected = calendar[day_idx + offset]
    assert list(dates) == list(expected)


def test_day_index_to_dates_rejects_inconsistent_offset():
    calendar = pd.bdate_range("2018-01-02", periods=500)
    entry_idx = pd.Series([10, 20, 30])
    # entry_day deliberately does NOT correspond to a constant offset from
    # entry_idx -- this must be caught, not silently averaged over.
    entry_day = pd.Series([calendar[50], calendar[999 % len(calendar)], calendar[45]])
    with pytest.raises(ValueError):
        rg.day_index_to_dates(np.array([0, 1]), entry_idx, entry_day, calendar)
