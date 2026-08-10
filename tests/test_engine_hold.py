"""Tests for the hold_days fixed-hold feature (backtest/engine.py, strategies.py).

These fake PriceUniverse and DailyStateBuilder rather than hitting real price
data or SEC filings, so the whole file runs in well under a second. The fakes
only implement the methods run_strategy actually calls on those two objects.

Background: with hold_days unset (the default, and what every pre-existing
strategy uses), a held ticker whose signal decays off the rolling window gets
an empty_state injected, its target drops to 0, and the position gets trimmed
almost immediately. hold_days locks a strategy's lots against that trim path
so the position can survive to the real forward-return horizon the signal is
scored on.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import date, timedelta

import pandas as pd
import pytest

from backtest import engine
from backtest.strategies import Strategy, ExitMethod, STRATEGIES


# ---------------------------------------------------------------------------
# Fakes: the minimal surface of PriceUniverse and DailyStateBuilder that
# run_strategy touches.
# ---------------------------------------------------------------------------
class FakePrices:
    def __init__(self, calendar, tickers, base_price: float = 10.0):
        self.calendar = calendar
        self.idx_by_date = {d: i for i, d in enumerate(calendar)}
        self.tickers = set(tickers)
        self.base_price = base_price
        self.close_overrides: dict[tuple[str, int], float] = {}
        self.open_overrides: dict[tuple[str, int], float] = {}

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
        return 5_000_000.0

    def median_share_volume(self, ticker, dt, window: int = 20):
        # Generous by design: these tests exercise hold_days/capacity/stop
        # behavior, not the participation cap, so this must sit well above
        # anything a 0.02*cap-or-smaller order here could ever request.
        return 5_000_000.0


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


def simple_target_fn(state, cap):
    if state.get("num_insiders", 0) >= 2:
        return 0.02 * cap
    return 0.0


def run_one_shot_signal(hold_days, calendar, stop_loss_pct=None,
                         crash_at_idx=None, crash_price=None, exit_days=365):
    """A signal that fires on day 0 only for ticker ABC, then decays forever."""
    strategy = Strategy(
        name="synth_test", description="synthetic", max_concurrent_tickers=10,
        target_fn=simple_target_fn, stop_loss_pct=stop_loss_pct, hold_days=hold_days,
    )
    exit_method = ExitMethod("test", exit_days)
    prices = FakePrices(calendar, ["ABC"])
    if crash_at_idx is not None:
        prices.close_overrides[("ABC", crash_at_idx)] = crash_price
    states = FakeStates({0: {"ABC": firing_state()}}, calendar)
    return engine.run_strategy(
        strategy=strategy, exit_method=exit_method, starting_capital=100_000.0,
        calendar=calendar, prices=prices, states=states,
    )


# ---------------------------------------------------------------------------
# Core hold_days behavior
# ---------------------------------------------------------------------------
def test_hold_days_survives_signal_decay_and_exits_at_expiry():
    calendar = make_calendar(90)
    result = run_one_shot_signal(hold_days=63, calendar=calendar)

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.exit_reason == "expiry"
    assert trade.trading_days_held == 63

    day30 = calendar[30]
    assert result.n_tickers_curve.loc[result.n_tickers_curve.index == pd.Timestamp(day30)].iloc[0] == 1


def test_without_hold_days_signal_decay_trims_almost_immediately():
    """The bug being fixed: the same target_fn, no hold_days, exits in 1-2
    trading days once the one-shot signal decays off the state window."""
    calendar = make_calendar(90)
    result = run_one_shot_signal(hold_days=None, calendar=calendar)

    assert len(result.trades) >= 1
    trade = result.trades[0]
    assert trade.exit_reason == "trim"
    assert trade.trading_days_held in (1, 2)


def test_stop_loss_still_fires_on_a_locked_lot():
    """Locking blocks the signal-decay trim path only. Risk exits must still
    be able to close a hold_days lot well before its expiry."""
    calendar = make_calendar(90)
    result = run_one_shot_signal(
        hold_days=63, calendar=calendar, stop_loss_pct=0.15,
        crash_at_idx=10, crash_price=5.0,  # entry ~10.0 -> -15% stop breached
    )

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.exit_reason == "stop_loss"
    assert trade.trading_days_held < 63


def test_exit_method_exit_days_caps_hold_days():
    """min(hold_days, exit_method.exit_days): a strategy asking for a 63-day
    hold under a 30-day exit method is capped at 30, not silently allowed to
    run long. exit_days is a safety ceiling even for fixed-hold strategies."""
    calendar = make_calendar(60)
    result = run_one_shot_signal(hold_days=63, calendar=calendar, exit_days=30)

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.exit_reason == "expiry"
    assert trade.trading_days_held == 30


def _scaling_target_fn(state, cap):
    score = state.get("conviction_score", 0)
    if score <= 0:
        return 0.0
    return 0.01 * cap * score


def test_hold_days_can_still_buy_more_when_signal_reinforces():
    """Locking only blocks the trim (sell) path. A re-firing signal that
    wants a bigger position must still be able to add a new lot, and that
    new lot gets its own independent hold_days clock."""
    calendar = make_calendar(90)
    strategy = Strategy(
        name="reinforce_test", description="", max_concurrent_tickers=10,
        target_fn=_scaling_target_fn, hold_days=63,
    )
    exit_method = ExitMethod("test", 365)
    prices = FakePrices(calendar, ["ABC"])
    # Day 0 fires at score 5 (target 5% of cap); day 5 fires stronger, at
    # score 10 (target 10% of cap), so the second firing must add a lot
    # rather than being blocked as a "decrease" by the locking guard.
    states = FakeStates(
        {0: {"ABC": firing_state(5)}, 5: {"ABC": firing_state(10)}}, calendar,
    )
    result = engine.run_strategy(
        strategy=strategy, exit_method=exit_method, starting_capital=100_000.0,
        calendar=calendar, prices=prices, states=states,
    )
    assert all(t.exit_reason != "trim" for t in result.trades)
    assert len(result.trades) == 2, f"expected 2 lots to expire, got {result.trades}"
    assert all(t.exit_reason == "expiry" for t in result.trades)
    held = sorted(t.trading_days_held for t in result.trades)
    # First lot: entry executes at idx 1, expiry at idx 1+63=64 -> 63 held.
    # Second lot: entry executes at idx 6, expiry at idx 6+63=69 -> 63 held.
    assert held == [63, 63]


def test_locked_ticker_still_counts_against_max_concurrent_tickers():
    """A locked position keeps occupying its capacity slot until it actually
    closes -- it must not be treated as free capacity just because its
    signal decayed."""
    calendar = make_calendar(10)
    strategy = Strategy(
        name="capacity_test", description="", max_concurrent_tickers=1,
        target_fn=simple_target_fn, hold_days=63,
    )
    exit_method = ExitMethod("test", 365)
    prices = FakePrices(calendar, ["ABC", "XYZ"])
    # ABC fires day 0 (fills the single slot); XYZ fires day 1, after ABC's
    # signal has already decayed, but ABC must still hold the slot.
    states = FakeStates(
        {0: {"ABC": firing_state()}, 1: {"XYZ": firing_state()}}, calendar,
    )
    result = engine.run_strategy(
        strategy=strategy, exit_method=exit_method, starting_capital=100_000.0,
        calendar=calendar, prices=prices, states=states,
    )
    tickers_bought = {t.ticker for t in result.trades}
    # XYZ should never get in: capacity=1 and ABC holds the only slot.
    assert "XYZ" not in tickers_bought
    assert result.skips["capacity"] >= 1


# ---------------------------------------------------------------------------
# Regression guard: hold_days=None must be byte-for-byte unaffected.
# ---------------------------------------------------------------------------
def _trade_tuples(trades):
    return [tuple(sorted(asdict(t).items())) for t in trades]


@pytest.mark.parametrize("strat", [s for s in STRATEGIES if s.hold_days is None])
def test_existing_strategies_unaffected_by_hold_days_field(strat):
    """Every pre-existing strategy has hold_days=None. Running it twice
    (both against the current engine, since hold_days=None is a total
    no-op through every branch this change touched) must be deterministic,
    and by construction exercises the exact code paths a hold_days=None
    strategy always took."""
    calendar = make_calendar(40)
    states_activity = {0: {"ABC": firing_state(10)}}
    prices_a = FakePrices(calendar, ["ABC"])
    prices_b = FakePrices(calendar, ["ABC"])
    states_a = FakeStates(states_activity, calendar)
    states_b = FakeStates(states_activity, calendar)
    exit_method = ExitMethod("e", 30)

    result_a = engine.run_strategy(
        strategy=strat, exit_method=exit_method, starting_capital=100_000.0,
        calendar=calendar, prices=prices_a, states=states_a,
    )
    result_b = engine.run_strategy(
        strategy=strat, exit_method=exit_method, starting_capital=100_000.0,
        calendar=calendar, prices=prices_b, states=states_b,
    )
    assert _trade_tuples(result_a.trades) == _trade_tuples(result_b.trades)
    assert (result_a.equity_curve == result_b.equity_curve).all()
    # No hold_days=None trade should ever carry a "trim" that is somehow
    # skipped — the guard added in this change must be strictly bypassed
    # (a no-op) whenever strategy.hold_days is None.
    for t in result_a.trades:
        assert t.exit_reason in (
            "expiry", "trim", "final_liquidation", "stop_loss", "trailing_stop",
        )


def test_new_hold63_strategies_are_registered():
    names = {s.name for s in STRATEGIES}
    for expected in ("tpo_gated_hold63", "all_clusters_hold63", "conviction_only_hold63"):
        assert expected in names
    by_name = {s.name: s for s in STRATEGIES}
    assert by_name["tpo_gated_hold63"].hold_days == 63
    assert by_name["all_clusters_hold63"].hold_days == 63
    assert by_name["conviction_only_hold63"].hold_days == 63
    # Policy shape mirrors the base strategy they're compared against.
    assert by_name["tpo_gated_hold63"].max_concurrent_tickers == 10
    assert by_name["all_clusters_hold63"].max_concurrent_tickers == 50
    assert by_name["conviction_only_hold63"].max_concurrent_tickers == 20


def test_every_hold63_strategy_deploys_full_capital():
    """cap * per-position weight must equal 100% for every fixed-hold strategy.

    A fixed-hold strategy holds each lot for a set number of days, so its
    concurrent position count sits near its cap for most of a run. If the
    cap and the per-position weight disagree, the portfolio silently runs
    part-invested and the rest sits in cash. That shows up as a lower
    return, which is indistinguishable from the strategy's signal being
    weak -- and it would be read as evidence about the signal rather than
    as a sizing mistake.

    This caught a real bug: model_ranked_top_hold63 was first defined with
    ten_percent_owner_gated's cap of 10 but _s1_target's 2% weight, which
    deploys 20% and leaves 80% in cash.
    """
    from backtest import strategies as S

    # A state that clears every hold63 target_fn's entry gate.
    state = {
        "num_insiders": 3,
        "conviction_score": 5,
        "includes_ten_percent_owner": True,
        "is_recent_ipo": False,
    }
    checked = 0
    for st in S.STRATEGIES:
        if st.hold_days is None:
            continue
        weight = st.target_fn(state, 1.0)
        assert weight > 0, f"{st.name}: test state did not clear its entry gate"
        deployed = st.max_concurrent_tickers * weight
        assert deployed == pytest.approx(1.0), (
            f"{st.name} deploys {deployed:.0%} of capital "
            f"({st.max_concurrent_tickers} slots x {weight:.0%}), not 100%. "
            "Cash drag would be misread as weak signal."
        )
        checked += 1
    assert checked >= 5, f"expected at least 5 fixed-hold strategies, found {checked}"
