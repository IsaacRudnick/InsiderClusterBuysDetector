"""Tests for the two buy-side size/price screens in backtest/engine.py:

  - MIN_PRICE_FLOOR (min_price kwarg): refuses a new position in a name
    trading under the floor price.
  - MAX_PARTICIPATION_PCT: truncates (never silently drops) a buy order
    whose share count would exceed a fraction of the ticker's recent median
    daily SHARE volume.

Both screens exist because of a single real case in the 72-month grid run of
2026-08-07 (that output directory has since been deleted, so the numbers below
are the record): SMFL entered 2024-09-20 at $0.000200 for
34,881,540 shares -- about 110x that day's entire tape of 316,699 shares --
because LIQUIDITY_FLOOR only screens dollar volume, and a fabricated-looking
sub-penny wick can clear a $-volume floor on the strength of a few
pre-collapse days while trading almost no actual shares. That single lot
produced $17.04M of P&L across the grid, 95.4% of one strategy's total
return.

These tests use the same fake PriceUniverse / DailyStateBuilder doubles as
tests/test_engine_hold.py and tests/test_engine_rank_fn.py: no disk, no
network, the whole file runs in well under a second. Duplicated here, not
imported, so this file stays runnable on its own (matches those files).
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from backtest import engine
from backtest.strategies import Strategy, ExitMethod


# ---------------------------------------------------------------------------
# Fakes: same minimal PriceUniverse / DailyStateBuilder surface as
# tests/test_engine_hold.py, extended with an overridable median_share_volume
# and a call log so tests can prove the participation screen is (or is not)
# consulted for a given order.
# ---------------------------------------------------------------------------
class FakePrices:
    def __init__(self, calendar, tickers, base_price: float = 10.0,
                 mdv: float = 5_000_000.0, msv: float = 5_000_000.0):
        self.calendar = calendar
        self.idx_by_date = {d: i for i, d in enumerate(calendar)}
        self.tickers = set(tickers)
        self.base_price = base_price
        self.mdv = mdv
        self.msv = msv
        self.close_overrides: dict[tuple[str, int], float] = {}
        self.open_overrides: dict[tuple[str, int], float] = {}
        self.msv_calls: list[tuple[str, date]] = []

    def _price(self, ticker, dt, overrides):
        idx = self.idx_by_date.get(dt)
        if idx is None or ticker not in self.tickers:
            return None
        return overrides.get((ticker, idx), self.base_price)

    def open(self, ticker, dt):
        return self._price(ticker, dt, self.open_overrides)

    def close(self, ticker, dt):
        return self._price(ticker, dt, self.close_overrides)

    def last_close_on_or_before(self, ticker, dt):
        idx = self.idx_by_date.get(dt, len(self.calendar) - 1)
        idx = min(idx, len(self.calendar) - 1)
        return self.close_overrides.get((ticker, idx), self.base_price)

    def is_past_last_bar(self, ticker, dt) -> bool:
        return False

    def price_signals(self, ticker, dt) -> dict:
        return {"momentum_20d": None, "vol_30d": None, "dist_from_high_90d": None}

    def median_dollar_volume(self, ticker, dt, window: int = 20):
        return self.mdv

    def median_share_volume(self, ticker, dt, window: int = 20):
        self.msv_calls.append((ticker, dt))
        return self.msv


class FakeStates:
    def __init__(self, activity: dict[int, dict[str, dict]], calendar):
        self.activity = activity
        self.idx_by_date = {d: i for i, d in enumerate(calendar)}

    def state_for_day(self, D) -> dict:
        return self.activity.get(self.idx_by_date[D], {})


def make_calendar(n_days: int, start: date = date(2024, 1, 2)) -> list[date]:
    out = []
    d = start
    while len(out) < n_days:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def firing_state(score: int = 5) -> dict:
    return {
        "num_insiders": 2,
        "conviction_score": score,
        "is_recent_ipo": False,
        "total_value": 0.0,
    }


def small_target_fn(state, cap):
    """Requests 2% of capital -- small enough to never trip the
    participation cap under the tests' default 5,000,000-share msv."""
    if state.get("num_insiders", 0) >= 2:
        return 0.02 * cap
    return 0.0


def big_target_fn(state, cap):
    """Requests 50% of capital -- deliberately oversized so it exceeds a
    thin ticker's participation cap and exercises truncation."""
    if state.get("num_insiders", 0) >= 2:
        return 0.5 * cap
    return 0.0


# ---------------------------------------------------------------------------
# MIN_PRICE_FLOOR
# ---------------------------------------------------------------------------
def test_price_floor_rejects_subdollar_buy():
    """engine.MIN_PRICE_FLOOR is now the default min_price. A ticker
    trading under it must never open a new position -- SMFL's $0.0002 entry
    is the motivating case."""
    calendar = make_calendar(10)
    strategy = Strategy(
        name="floor_test", description="", max_concurrent_tickers=10,
        target_fn=small_target_fn,
    )
    exit_method = ExitMethod("test", 5)
    prices = FakePrices(calendar, ["PENNY"], base_price=0.50)
    states = FakeStates({0: {"PENNY": firing_state()}}, calendar)
    result = engine.run_strategy(
        strategy=strategy, exit_method=exit_method, starting_capital=100_000.0,
        calendar=calendar, prices=prices, states=states,
    )
    assert result.trades == []
    assert result.skips["price_floor"] >= 1


def test_price_floor_remains_overridable_to_zero():
    """min_price=0 must still be a valid, explicit opt-out for callers that
    want no floor at all -- the default changed, the override did not."""
    calendar = make_calendar(10)
    strategy = Strategy(
        name="floor_off_test", description="", max_concurrent_tickers=10,
        target_fn=small_target_fn,
    )
    exit_method = ExitMethod("test", 5)
    prices = FakePrices(calendar, ["PENNY"], base_price=0.50)
    states = FakeStates({0: {"PENNY": firing_state()}}, calendar)
    result = engine.run_strategy(
        strategy=strategy, exit_method=exit_method, starting_capital=100_000.0,
        calendar=calendar, prices=prices, states=states, min_price=0.0,
    )
    assert len(result.trades) == 1
    assert result.skips["price_floor"] == 0


# ---------------------------------------------------------------------------
# MAX_PARTICIPATION_PCT
# ---------------------------------------------------------------------------
def test_participation_cap_truncates_oversized_order():
    """A thin-volume ticker's order gets shrunk to the cap, not dropped."""
    calendar = make_calendar(10)
    strategy = Strategy(
        name="cap_trunc_test", description="", max_concurrent_tickers=10,
        target_fn=big_target_fn,
    )
    exit_method = ExitMethod("test", 5)
    prices = FakePrices(calendar, ["THIN"], base_price=10.0, msv=1_000.0)
    states = FakeStates({0: {"THIN": firing_state()}}, calendar)
    result = engine.run_strategy(
        strategy=strategy, exit_method=exit_method, starting_capital=100_000.0,
        calendar=calendar, prices=prices, states=states,
    )
    assert result.skips["participation_capped"] >= 1
    assert len(result.trades) == 1
    max_shares = 1_000.0 * engine.MAX_PARTICIPATION_PCT
    assert result.trades[0].shares == pytest.approx(max_shares, rel=1e-6)
    # Uncapped, this order would have wanted ~50,000 / 10.01 ~= 4,995
    # shares -- roughly 50x the allowed size.
    naive_shares = (0.5 * 100_000.0) / (10.0 * (1 + engine.SLIPPAGE))
    assert naive_shares / result.trades[0].shares > 40


def test_participation_cap_rejects_dust_sized_truncation():
    """When the cap would shrink an order to next to nothing, it is skipped
    outright rather than booking a near-zero lot."""
    calendar = make_calendar(10)
    strategy = Strategy(
        name="dust_test", description="", max_concurrent_tickers=10,
        target_fn=big_target_fn,
    )
    exit_method = ExitMethod("test", 5)
    prices = FakePrices(calendar, ["DUST"], base_price=1.0, msv=1.0)
    states = FakeStates({0: {"DUST": firing_state()}}, calendar)
    result = engine.run_strategy(
        strategy=strategy, exit_method=exit_method, starting_capital=100_000.0,
        calendar=calendar, prices=prices, states=states,
    )
    assert result.trades == []
    assert result.skips["participation"] >= 1
    assert result.skips.get("participation_capped", 0) == 0


def test_participation_cap_never_applies_to_sells():
    """The screen only runs in order execution's delta>0 branch. Proven
    directly, not just asserted: track every call to median_share_volume
    and confirm it is never invoked while the position is being trimmed to
    zero on the days after the one-shot signal decays (same one-shot shape
    as tests/test_engine_hold.py's run_one_shot_signal)."""
    calendar = make_calendar(10)
    strategy = Strategy(
        name="sell_untouched_test", description="", max_concurrent_tickers=10,
        target_fn=small_target_fn,
    )
    exit_method = ExitMethod("test", 365)
    prices = FakePrices(calendar, ["ABC"], base_price=10.0, msv=5_000_000.0)
    states = FakeStates({0: {"ABC": firing_state()}}, calendar)
    result = engine.run_strategy(
        strategy=strategy, exit_method=exit_method, starting_capital=100_000.0,
        calendar=calendar, prices=prices, states=states,
    )
    trims = [t for t in result.trades if t.exit_reason == "trim"]
    assert trims, "expected the one-shot signal to decay into a full trim"
    # Exactly one buy order was ever executed (day 0's signal); the trim
    # that follows must not add a second call.
    assert len(prices.msv_calls) == 1


def test_normal_liquid_order_passes_through_untouched():
    """A normally-sized order against an ordinary, liquid ticker must be
    completely unaffected by either new screen."""
    calendar = make_calendar(10)
    strategy = Strategy(
        name="normal_test", description="", max_concurrent_tickers=10,
        target_fn=small_target_fn,
    )
    exit_method = ExitMethod("test", 5)
    prices = FakePrices(calendar, ["ABC"], base_price=10.0, msv=5_000_000.0)
    states = FakeStates({0: {"ABC": firing_state()}}, calendar)
    result = engine.run_strategy(
        strategy=strategy, exit_method=exit_method, starting_capital=100_000.0,
        calendar=calendar, prices=prices, states=states,
    )
    assert len(result.trades) == 1
    expected_shares = (0.02 * 100_000.0) / (10.0 * (1 + engine.SLIPPAGE))
    assert result.trades[0].shares == pytest.approx(expected_shares, rel=1e-9)
    assert result.skips["price_floor"] == 0
    assert result.skips["participation"] == 0
    assert result.skips["participation_capped"] == 0


# ---------------------------------------------------------------------------
# Regression: the SMFL shape itself
# ---------------------------------------------------------------------------
def test_participation_cap_reproduces_smfl_shape():
    """Regression guard for the actual SMFL lot: entry 2024-09-20 at
    $0.000200, 34,881,540 shares bought against a day whose entire tape was
    316,699 shares. min_price=0 is passed explicitly so this test isolates
    the participation cap from the price floor -- the floor alone would
    already reject a $0.0002 entry (covered by
    test_price_floor_rejects_subdollar_buy above); this test proves the
    cap catches the same shape even when the floor is disabled."""
    calendar = make_calendar(10)
    strategy = Strategy(
        name="smfl_test", description="", max_concurrent_tickers=10,
        target_fn=big_target_fn,
    )
    exit_method = ExitMethod("test", 5)
    prices = FakePrices(calendar, ["SMFL"], base_price=0.0002, msv=316_699.0)
    states = FakeStates({0: {"SMFL": firing_state()}}, calendar)
    result = engine.run_strategy(
        strategy=strategy, exit_method=exit_method, starting_capital=100_000.0,
        calendar=calendar, prices=prices, states=states, min_price=0.0,
    )
    assert result.skips["participation_capped"] >= 1
    assert len(result.trades) == 1
    max_shares = 316_699.0 * engine.MAX_PARTICIPATION_PCT
    assert result.trades[0].shares == pytest.approx(max_shares, rel=1e-6)
    naive_shares = (0.5 * 100_000.0) / (0.0002 * (1 + engine.SLIPPAGE))
    # The real lot bought 34,881,540 shares off a 316,699-share tape --
    # about 110x. The cap must keep this analogous order at least three
    # orders of magnitude smaller than the naive, uncapped request.
    assert naive_shares / result.trades[0].shares > 1000
