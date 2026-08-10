"""Tests for Strategy.rank_fn: capacity allocation by score instead of by
arrival order (backtest/engine.py step 2, backtest/strategies.py rank_fn
library).

Background: RESEARCH_NOTES.md ("Capacity is the hidden selector") found that
every existing strategy's max_concurrent_tickers binds far harder than it
looks, and the engine used to grant slots strictly in dict-iteration
(arrival) order, with nothing sorting by score. rank_fn is an opt-in fix:
when a Strategy sets it, capacity goes to the highest-scoring candidates.
When rank_fn is None (every strategy defined before this feature), capacity
allocation must stay exactly as it was.

These tests use the same fake PriceUniverse / DailyStateBuilder doubles as
tests/test_engine_hold.py: no disk, no network, the whole file runs in well
under a second.

Runnable standalone via `python -m pytest tests/test_engine_rank_fn.py -q`
from the repo root.
"""

from __future__ import annotations

import os
import sys
from datetime import date, timedelta

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from backtest import engine  # noqa: E402
from backtest.strategies import (  # noqa: E402
    MODEL_RANKED_SHORT_HOLD_DAYS,
    MODEL_RANKED_SHORT_HOLD_SLOT_COUNTS,
    MODEL_RANKED_SLOT_COUNTS,
    STRATEGIES,
    ExitMethod,
    Strategy,
    rank_by_conviction,
    rank_by_model_score,
)


# ---------------------------------------------------------------------------
# Fakes: same minimal PriceUniverse / DailyStateBuilder surface as
# tests/test_engine_hold.py. Duplicated here, not imported, so this file
# stays runnable on its own (matches tests/test_research.py and
# tests/test_model.py, which do the same).
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
        # Generous by design: these tests exercise rank_fn capacity
        # allocation, not the participation cap.
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


def mk_state(ticker: str, score, **extra) -> dict:
    st = {
        "ticker": ticker,
        "num_insiders": 2,
        "conviction_score": score,
        "is_recent_ipo": False,
        "total_value": 0.0,
    }
    st.update(extra)
    return st


def qualifies_target_fn(state: dict, cap: float) -> float:
    # Every candidate wants the same 2% slice. Sizing never depends on
    # score, so any selection difference below is caused only by capacity
    # allocation (arrival order vs. rank_fn), not by target_fn.
    if state.get("num_insiders", 0) >= 2:
        return 0.02 * cap
    return 0.0


def run_five_candidate_day(rank_fn, scores=(1, 5, 3, 2, 4)):
    """T1..T5 all fire on day 0 with the given conviction_score, in that
    dict-insertion order. max_concurrent_tickers=2, so 3 of the 5 must be
    turned away. Which 3 depends only on rank_fn."""
    calendar = make_calendar(5)
    tickers = ["T1", "T2", "T3", "T4", "T5"]
    prices = FakePrices(calendar, tickers)
    activity = {0: {t: mk_state(t, s) for t, s in zip(tickers, scores)}}
    states = FakeStates(activity, calendar)
    strategy = Strategy(
        name="cap_test", description="", max_concurrent_tickers=2,
        target_fn=qualifies_target_fn, rank_fn=rank_fn,
    )
    exit_method = ExitMethod("test", 365)
    return engine.run_strategy(
        strategy=strategy, exit_method=exit_method, starting_capital=100_000.0,
        calendar=calendar, prices=prices, states=states,
    )


def bought_tickers(result) -> set[str]:
    return {t.ticker for t in result.trades}


# ---------------------------------------------------------------------------
# Part 1: golden regression. rank_fn=None must be a total no-op.
#
# There is no stored pre-change engine.py to byte-diff against (this repo
# has no git history), so this is proven three ways instead, mirroring the
# precedent tests/test_engine_hold.py already sets for the identical class
# of claim (test_existing_strategies_unaffected_by_hold_days_field):
#   1. _rank_capacity_order is structurally never called when rank_fn is
#      None (monkeypatched to raise if it is).
#   2. The observable selection is still plain arrival order, not score
#      order, on a scenario where the two disagree.
#   3. Repeated runs are bit-for-bit identical (trades and equity curve).
# ---------------------------------------------------------------------------
def test_rank_fn_none_never_calls_the_reorder_helper(monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError(
            "_rank_capacity_order must never run when strategy.rank_fn is None"
        )
    monkeypatch.setattr(engine, "_rank_capacity_order", _boom)

    result = run_five_candidate_day(rank_fn=None)

    # If the guard above had fired, run_five_candidate_day would have raised.
    # Reaching here already proves the helper was skipped; the arrival-order
    # assertions below prove nothing else silently reproduced its effect.
    assert bought_tickers(result) == {"T1", "T2"}, (
        "with rank_fn=None, capacity must go to the first two tickers queued "
        "(T1, T2), not the two highest-scored (T2, T5) -- arrival order, "
        "not score, still decides"
    )
    assert result.skips["capacity"] == 3
    # No diagnostic keys should appear at all: they are only written on the
    # rank_fn branch, so a strategy without rank_fn must see the exact same
    # skips Counter shape it always did.
    assert "capacity_rank_score_sum" not in result.skips
    assert "capacity_rank_score_n" not in result.skips
    assert "capacity_rank_score_max" not in result.skips


def test_rank_fn_none_is_deterministic_across_runs():
    result_a = run_five_candidate_day(rank_fn=None)
    result_b = run_five_candidate_day(rank_fn=None)
    assert bought_tickers(result_a) == bought_tickers(result_b) == {"T1", "T2"}
    assert list(result_a.equity_curve) == list(result_b.equity_curve)
    assert result_a.skips == result_b.skips


def test_every_existing_strategy_has_rank_fn_unset():
    """The pre-existing strategies must not have been touched: adding the
    field must not have given any of them a ranking key by accident.

    model_ranked_hold63, model_ranked_top_hold63, the
    model_ranked_n{NN}_hold63 slot-count sweep, and the
    model_ranked_n{NN}_hold{H} short-hold family (backtest/strategies.py)
    are the deliberate exception: they exist specifically to opt into
    rank_fn=rank_by_model_score, so they are excluded here by name rather
    than by weakening this assertion for everyone else."""
    deliberately_ranked = {"model_ranked_hold63", "model_ranked_top_hold63"} | {
        f"model_ranked_n{n:02d}_hold63" for n in MODEL_RANKED_SLOT_COUNTS
    } | {
        f"model_ranked_n{n:02d}_hold{h}"
        for n in MODEL_RANKED_SHORT_HOLD_SLOT_COUNTS
        for h in MODEL_RANKED_SHORT_HOLD_DAYS
    }
    for strat in STRATEGIES:
        if strat.name in deliberately_ranked:
            continue
        assert strat.rank_fn is None, f"{strat.name} unexpectedly has a rank_fn"


# ---------------------------------------------------------------------------
# Part 2: rank_fn set -> capacity goes to the highest scores, not first come.
# ---------------------------------------------------------------------------
def test_rank_fn_selects_the_two_highest_scored_candidates():
    result = run_five_candidate_day(rank_fn=rank_by_conviction)

    # Scores: T1=1, T2=5, T3=3, T4=2, T5=4. Top two are T2 and T5.
    assert bought_tickers(result) == {"T2", "T5"}
    assert result.skips["capacity"] == 3

    # Diagnostics: rejected are T1(1), T3(3), T4(2).
    assert result.skips["capacity_rank_score_n"] == 3
    assert result.skips["capacity_rank_score_sum"] == pytest.approx(6.0)
    assert result.skips["capacity_rank_score_max"] == pytest.approx(3.0)


def test_rank_fn_leaves_sell_relative_order_alone_only_buys_move():
    """A held ticker whose target has dropped (a trim, delta < 0) must not
    be treated as a "buy" candidate for reordering. It must stay at its own
    queued position so a same-day sell-then-buy sequence (the trim frees a
    slot that a fresh candidate then takes) works exactly as it did before
    rank_fn existed. Only which BUY fills the freed slot may change."""
    calendar = make_calendar(5)
    tickers = ["HELD", "NEW1", "NEW2"]
    prices = FakePrices(calendar, tickers)
    # Day 0: HELD fires alone and gets bought on day 1.
    # Day 1: HELD fires again, explicitly decayed (num_insiders=0 -> target
    # 0, a real trim), queued FIRST -- same as its day-0 slot in the dict --
    # while NEW1 (score 1) and NEW2 (score 9) queue after it. With
    # max_concurrent_tickers=1, HELD's trim must still execute before either
    # new buy is attempted (that part of the sequence is untouched), and
    # rank_fn then decides NEW2, not NEW1, takes the freed slot even though
    # NEW1 was queued first.
    activity = {
        0: {"HELD": mk_state("HELD", 5)},
        1: {
            "HELD": mk_state("HELD", 0, num_insiders=0),
            "NEW1": mk_state("NEW1", 1),
            "NEW2": mk_state("NEW2", 9),
        },
    }
    states = FakeStates(activity, calendar)
    strategy = Strategy(
        name="mixed_test", description="", max_concurrent_tickers=1,
        target_fn=qualifies_target_fn, rank_fn=rank_by_conviction,
    )
    exit_method = ExitMethod("test", 365)
    result = engine.run_strategy(
        strategy=strategy, exit_method=exit_method, starting_capital=100_000.0,
        calendar=calendar, prices=prices, states=states,
    )
    # HELD trims out (freeing the slot), and NEW2 -- the higher scored of
    # the two new candidates -- takes it. NEW1 never gets in, even though
    # it was queued before NEW2.
    assert "NEW1" not in bought_tickers(result)
    assert "NEW2" in bought_tickers(result)
    assert "HELD" in bought_tickers(result)  # bought day 1, trimmed day 2


# ---------------------------------------------------------------------------
# Part 3: determinism, including tied scores.
# ---------------------------------------------------------------------------
def test_tied_scores_break_deterministically_by_ticker():
    calendar = make_calendar(5)
    # Insertion order is deliberately the reverse of alphabetical order, so
    # a win for "ATIE" cannot be arrival order in disguise.
    tickers = ["ZTIE", "ATIE"]
    activity = {0: {"ZTIE": mk_state("ZTIE", 5), "ATIE": mk_state("ATIE", 5)}}
    strategy = Strategy(
        name="tie_test", description="", max_concurrent_tickers=1,
        target_fn=qualifies_target_fn, rank_fn=rank_by_conviction,
    )
    exit_method = ExitMethod("test", 365)

    winners = []
    for _ in range(3):
        result = engine.run_strategy(
            strategy=strategy, exit_method=exit_method, starting_capital=100_000.0,
            calendar=calendar, prices=FakePrices(calendar, tickers),
            states=FakeStates(activity, calendar),
        )
        winners.append(bought_tickers(result))

    assert all(w == {"ATIE"} for w in winners), (
        f"tie-break must be deterministic and favor the alphabetically "
        f"first ticker on a tied score, got {winners}"
    )


# ---------------------------------------------------------------------------
# Part 4: NaN / None / raising rank_fn all sort last, never crash the run.
# ---------------------------------------------------------------------------
def _flaky_rank_fn(state: dict) -> float:
    ticker = state.get("ticker")
    if ticker == "NANT":
        return float("nan")
    if ticker == "NONET":
        return None
    if ticker == "BOOMT":
        raise ValueError("synthetic rank_fn failure")
    return float(state.get("conviction_score", 0))


def test_nan_none_and_raising_scores_all_sort_last_and_do_not_crash():
    calendar = make_calendar(5)
    tickers = ["GOODT", "NANT", "NONET", "BOOMT"]
    prices = FakePrices(calendar, tickers)
    # GOODT's raw score (1) is the lowest of the four by value, but it is
    # the only one with a usable score, so it must still win the one slot.
    activity = {0: {
        "GOODT": mk_state("GOODT", 1),
        "NANT": mk_state("NANT", 99),
        "NONET": mk_state("NONET", 99),
        "BOOMT": mk_state("BOOMT", 99),
    }}
    states = FakeStates(activity, calendar)
    strategy = Strategy(
        name="flaky_test", description="", max_concurrent_tickers=1,
        target_fn=qualifies_target_fn, rank_fn=_flaky_rank_fn,
    )
    exit_method = ExitMethod("test", 365)

    result = engine.run_strategy(
        strategy=strategy, exit_method=exit_method, starting_capital=100_000.0,
        calendar=calendar, prices=prices, states=states,
    )

    assert bought_tickers(result) == {"GOODT"}
    assert result.skips["capacity"] == 3
    # None of the three rejected candidates had a usable score, so the
    # sum/max diagnostics (which exclude failed scores) must stay empty.
    assert result.skips.get("capacity_rank_score_n", 0) == 0


# ---------------------------------------------------------------------------
# Part 5: rank_by_model_score, the hook the ranking model will use.
# ---------------------------------------------------------------------------
def test_rank_by_model_score_handles_partial_coverage():
    calendar = make_calendar(5)
    tickers = ["MSA", "MSB", "MSC", "MSD"]
    prices = FakePrices(calendar, tickers)
    activity = {0: {
        "MSA": mk_state("MSA", 0, model_score=10),
        "MSB": mk_state("MSB", 0, model_score=2),
        "MSC": mk_state("MSC", 0),  # no model_score key at all
        "MSD": mk_state("MSD", 0, model_score=float("nan")),
    }}
    states = FakeStates(activity, calendar)
    strategy = Strategy(
        name="model_score_test", description="", max_concurrent_tickers=2,
        target_fn=qualifies_target_fn, rank_fn=rank_by_model_score,
    )
    exit_method = ExitMethod("test", 365)

    result = engine.run_strategy(
        strategy=strategy, exit_method=exit_method, starting_capital=100_000.0,
        calendar=calendar, prices=prices, states=states,
    )

    # MSC (missing) and MSD (NaN) must sort behind both scored candidates.
    assert bought_tickers(result) == {"MSA", "MSB"}
    assert result.skips["capacity"] == 2


def test_rank_by_model_score_and_rank_by_conviction_unit():
    """Direct unit checks on the rank_fn library, independent of the engine."""
    assert rank_by_conviction({"conviction_score": 7}) == 7.0
    assert rank_by_conviction({}) == 0.0

    assert rank_by_model_score({"model_score": 3}) == 3.0
    assert rank_by_model_score({}) == float("-inf")
    assert rank_by_model_score({"model_score": None}) == float("-inf")
    assert rank_by_model_score({"model_score": float("nan")}) == float("-inf")
