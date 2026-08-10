"""Tests for the model_score wiring: backtest/model_scores.py (the OOF
parquet loader), DailyStateBuilder.set_model_scores / the model_score
lookup in backtest/state.py._build_state, the two model_ranked_* strategies,
and the engine.py clamp-warning added alongside them.

Runnable standalone via `python -m pytest tests/test_model_scores.py -q`
from the repo root. No network access: DailyStateBuilder's IPO-date
prefetch is monkeypatched exactly like tests/test_research.py does, since
DailyStateBuilder's constructor always calls it regardless of whether a
test cares about IPO recency.

Background this file is proving:
  - research.py emits ONE row per distinct cluster event, not one per
    visible day, so the score has to be forward-filled across the same
    rolling window the state itself stays visible for (WINDOW_DAYS=14
    calendar days from the event). Without the forward-fill, a candidate
    would be scored on day 1 and silently unscored (None -> -inf in
    rank_by_model_score) on every day after, losing capacity slots for no
    principled reason.
  - No-lookahead is the single most important property: a score whose
    event_day is after the day being decided must never be used.
  - A silent (ticker, event_day) key-type mismatch between the loader and
    DailyStateBuilder would make every lookup miss and read as "the model
    has no edge" -- so the loader's key type is checked directly here.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import date, timedelta

import pandas as pd
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import ipo_lookup  # noqa: E402
from backtest import engine  # noqa: E402
from backtest.model_scores import DEFAULT_SCORE_COL, load_model_scores  # noqa: E402
from backtest.state import DailyStateBuilder, WINDOW_DAYS  # noqa: E402
from backtest.strategies import ExitMethod, Strategy, rank_by_model_score  # noqa: E402


# ---------------------------------------------------------------------------
# Fakes / fixtures -- duplicated from tests/test_research.py rather than
# imported, so this file stays runnable on its own (same convention
# tests/test_engine_rank_fn.py and tests/test_engine_hold.py already use).
# ---------------------------------------------------------------------------
@pytest.fixture()
def no_network_ipo(monkeypatch):
    """DailyStateBuilder._prefetch_ipo_dates calls ipo_lookup.get_first_trade_date
    in a thread pool at construction time, which can hit yfinance. Stub it out
    so tests never touch the network."""
    monkeypatch.setattr(ipo_lookup, "get_first_trade_date", lambda t: None)
    yield


def make_row(*, ticker: str, transaction_date: date, filing_date: date,
             owner_cik: str = "OWNER1") -> dict:
    """A single minimal insider-buy row. Values are unrelated to
    model_score, only present because state.py's _build_state and
    insider_cluster_buys._component_flags read them unconditionally.
    owner_cik is the insiders dict's grouping key (see _build_state), so
    two rows for the same ticker need distinct owner_ciks to count as two
    separate insiders, not one insider who transacted twice."""
    return {
        "ticker": ticker, "owner_cik": owner_cik, "owner_name": owner_cik,
        "owner_roles": "Director", "is_director": True, "is_officer": False,
        "is_ten_percent_owner": False, "transaction_date": transaction_date,
        "filing_date": filing_date, "shares": 1000.0, "value": 10_000.0,
        "pct_of_prior_stake": None, "is_10b5_1": False,
    }


def build_states(rows: list[dict]) -> DailyStateBuilder:
    return DailyStateBuilder(pd.DataFrame(rows))


# ---------------------------------------------------------------------------
# Part 1: model_score forward-fill / no-lookahead, exercised directly
# through DailyStateBuilder._build_state.
#
# _build_state's other fields (is_recent_ipo, etc.) don't depend on the
# window rows lining up with as_of -- only model_score does -- so a single
# fixed one-row window plus a varying `as_of` isolates the lookup logic
# cleanly, without needing a second real event to keep the ticker's
# rolling-window visibility (_events_by_day) alive out to day 14+.
# ---------------------------------------------------------------------------
def test_forward_fill_within_window_and_stops_after_window_days(no_network_ipo):
    rows = [make_row(ticker="TICK", transaction_date=date(2024, 1, 2), filing_date=date(2024, 1, 2))]
    states = build_states(rows)
    states.set_model_scores({("TICK", date(2024, 1, 2)): 7.0})

    event_day = date(2024, 1, 2)
    on_event_day = states._build_state([], "TICK", event_day)
    assert on_event_day["model_score"] == 7.0

    # Still within WINDOW_DAYS (14) calendar days: as_of - event_day == 13.
    still_within = states._build_state([], "TICK", event_day + timedelta(days=WINDOW_DAYS - 1))
    assert still_within["model_score"] == 7.0

    # Exactly at the boundary: as_of - event_day == window_days is allowed
    # per the spec ("<=  window_days"), so this must still be scored.
    at_boundary = states._build_state([], "TICK", event_day + timedelta(days=WINDOW_DAYS))
    assert at_boundary["model_score"] == 7.0

    # One day past the boundary: the score must stop forward-filling.
    past_boundary = states._build_state([], "TICK", event_day + timedelta(days=WINDOW_DAYS + 1))
    assert past_boundary["model_score"] is None


def test_no_lookahead_future_event_day_never_used(no_network_ipo):
    """A score whose event_day is after as_of must never be picked up, even
    though it would otherwise be the "latest" entry in the ticker's sorted
    list. This is the single most important property in this feature: a
    future score leaking into a past decision would invalidate the whole
    backtest."""
    rows = [make_row(ticker="TICK", transaction_date=date(2024, 1, 2), filing_date=date(2024, 1, 2))]
    states = build_states(rows)
    future_day = date(2024, 1, 10)
    states.set_model_scores({("TICK", future_day): 99.0})

    # Decision day is strictly before the scored event_day.
    result = states._build_state([], "TICK", date(2024, 1, 5))
    assert result["model_score"] is None, (
        "a future event_day's score must never be used for a decision made "
        "before that event_day exists"
    )


def test_latest_qualifying_event_wins_not_earliest(no_network_ipo):
    """Two scored events for the same ticker, both <= as_of and both within
    the window: the lookup must take the LATEST one, not the first one
    found or an average."""
    rows = [make_row(ticker="TICK", transaction_date=date(2024, 1, 2), filing_date=date(2024, 1, 2))]
    states = build_states(rows)
    states.set_model_scores({
        ("TICK", date(2024, 1, 2)): 1.0,
        ("TICK", date(2024, 1, 5)): 2.0,
    })
    result = states._build_state([], "TICK", date(2024, 1, 6))
    assert result["model_score"] == 2.0


def test_unscored_ticker_yields_none_and_ranks_last(no_network_ipo):
    rows = [make_row(ticker="TICK", transaction_date=date(2024, 1, 2), filing_date=date(2024, 1, 2))]
    states = build_states(rows)
    # A different ticker is scored; TICK itself never appears in the map.
    states.set_model_scores({("OTHER", date(2024, 1, 2)): 5.0})

    result = states._build_state([], "TICK", date(2024, 1, 2))
    assert result["model_score"] is None
    assert rank_by_model_score(result) == float("-inf")


def test_model_score_none_before_set_model_scores_called(no_network_ipo):
    """Mirrors learned_score/tail_score's default-None behavior before
    their setters run: a builder that never calls set_model_scores must
    report model_score=None, not raise and not default to 0.0."""
    rows = [make_row(ticker="TICK", transaction_date=date(2024, 1, 2), filing_date=date(2024, 1, 2))]
    states = build_states(rows)
    result = states._build_state([], "TICK", date(2024, 1, 2))
    assert result["model_score"] is None


# ---------------------------------------------------------------------------
# Part 2: engine-level -- a scored candidate outranks an unscored one for a
# capacity slot when rank_fn=rank_by_model_score.
# ---------------------------------------------------------------------------
class FakePrices:
    def __init__(self, calendar, tickers, base_price: float = 10.0):
        self.calendar = calendar
        self.idx_by_date = {d: i for i, d in enumerate(calendar)}
        self.tickers = set(tickers)
        self.base_price = base_price

    def _price(self, ticker, dt):
        if self.idx_by_date.get(dt) is None or ticker not in self.tickers:
            return None
        return self.base_price

    def open(self, ticker, dt):
        return self._price(ticker, dt)

    def close(self, ticker, dt):
        return self._price(ticker, dt)

    def last_close_on_or_before(self, ticker, dt):
        return self.base_price

    def is_past_last_bar(self, ticker, dt) -> bool:
        return False

    def price_signals(self, ticker, dt) -> dict:
        return {"momentum_20d": None, "vol_30d": None, "dist_from_high_90d": None}

    def median_dollar_volume(self, ticker, dt, window: int = 20):
        return 5_000_000.0

    def median_share_volume(self, ticker, dt, window: int = 20):
        # Generous by design: these tests exercise rank_by_model_score
        # capacity allocation, not the participation cap.
        return 5_000_000.0


def make_calendar(n_days: int, start: date = date(2024, 1, 2)) -> list[date]:
    out = []
    d = start
    while len(out) < n_days:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def qualifies_target_fn(state: dict, cap: float) -> float:
    if state.get("num_insiders", 0) >= 2:
        return 0.02 * cap
    return 0.0


def test_scored_candidate_outranks_unscored_for_capacity_slot(no_network_ipo):
    calendar = make_calendar(5)
    d0 = calendar[0]
    rows = [
        make_row(ticker="SCORED", transaction_date=d0, filing_date=d0, owner_cik="A1"),
        make_row(ticker="SCORED", transaction_date=d0, filing_date=d0, owner_cik="A2"),
        make_row(ticker="UNSCORED", transaction_date=d0, filing_date=d0, owner_cik="B1"),
        make_row(ticker="UNSCORED", transaction_date=d0, filing_date=d0, owner_cik="B2"),
    ]
    states = build_states(rows)
    states.set_model_scores({("SCORED", d0): 3.0})  # UNSCORED never appears.

    prices = FakePrices(calendar, ["SCORED", "UNSCORED"])
    strategy = Strategy(
        name="cap_test", description="", max_concurrent_tickers=1,
        target_fn=qualifies_target_fn, rank_fn=rank_by_model_score,
    )
    exit_method = ExitMethod("test", 365)
    result = engine.run_strategy(
        strategy=strategy, exit_method=exit_method, starting_capital=100_000.0,
        calendar=calendar, prices=prices, states=states,
    )
    bought = {t.ticker for t in result.trades}
    assert bought == {"SCORED"}
    # UNSCORED keeps re-qualifying and getting turned away on every
    # remaining day of the 5-day calendar (its own window easily outlasts
    # 5 days), so this is >= 1, not necessarily exactly 1.
    assert result.skips["capacity"] >= 1


# ---------------------------------------------------------------------------
# Part 3: engine.py's clamp-warning. hold_days=63 paired with a shorter
# exit_days must warn (naming both values and both names); paired with a
# longer exit_days must not.
# ---------------------------------------------------------------------------
def _run_one_shot(hold_days: int, exit_days: int):
    calendar = make_calendar(90)
    prices = FakePrices(calendar, ["ABC"])

    class OneShotStates:
        def state_for_day(self, D):
            if D == calendar[0]:
                return {"ABC": {"ticker": "ABC", "num_insiders": 2, "conviction_score": 5}}
            return {}

    strategy = Strategy(
        name="clamp_test", description="", max_concurrent_tickers=10,
        target_fn=qualifies_target_fn, hold_days=hold_days,
    )
    exit_method = ExitMethod("test", exit_days)
    return engine.run_strategy(
        strategy=strategy, exit_method=exit_method, starting_capital=100_000.0,
        calendar=calendar, prices=prices, states=OneShotStates(),
    )


def test_clamp_warning_fires_when_exit_days_below_hold_days(caplog):
    with caplog.at_level(logging.WARNING, logger="backtest.engine"):
        _run_one_shot(hold_days=63, exit_days=30)
    messages = [r.message for r in caplog.records]
    assert any(
        "clamp_test" in m and "63" in m and "30" in m and "test" in m
        for m in messages
    ), messages


def test_clamp_warning_does_not_fire_when_exit_days_covers_hold_days(caplog):
    with caplog.at_level(logging.WARNING, logger="backtest.engine"):
        _run_one_shot(hold_days=63, exit_days=90)
    messages = [r.message for r in caplog.records]
    assert not any("clamp_test" in m for m in messages), messages


# ---------------------------------------------------------------------------
# Part 4: the loader. Key type must match what state_for_day passes as
# `as_of` (datetime.date), and the NaN/match-rate accounting must be sane.
# ---------------------------------------------------------------------------
def test_loader_default_score_column_is_oof_tail_classifier():
    assert DEFAULT_SCORE_COL == "oof_tail_classifier"


def test_loader_key_type_matches_state_for_day_as_of_type(tmp_path, no_network_ipo):
    df = pd.DataFrame({
        "ticker": ["AAA", "BBB"],
        "event_day": [date(2024, 1, 2), date(2024, 1, 5)],
        "oof_tail_classifier": [0.7, 0.2],
    })
    path = tmp_path / "oof_scores.parquet"
    df.to_parquet(path, index=False)

    scores = load_model_scores(str(path))
    assert len(scores) == 2

    rows = [make_row(ticker="AAA", transaction_date=date(2024, 1, 2), filing_date=date(2024, 1, 2))]
    states = build_states(rows)
    day_states = states.state_for_day(date(2024, 1, 2))
    as_of_type = type(day_states["AAA"]["as_of"])

    # Every key's event_day component must be the exact same type
    # state_for_day passes as `as_of`, so the dict lookup in _build_state
    # can actually hit. `is date` (not isinstance) because
    # datetime.datetime is itself a subclass of datetime.date but compares
    # unequal to a plain date with the same y/m/d in dict lookups reliably
    # only when both sides are the plain type.
    for ticker, event_day in scores:
        assert type(event_day) is as_of_type is date

    states.set_model_scores(scores)
    result = states._build_state([], "AAA", date(2024, 1, 2))
    assert result["model_score"] == pytest.approx(0.7)


def test_loader_drops_nan_scores_and_logs_match_rate(tmp_path, caplog):
    df = pd.DataFrame({
        "ticker": ["AAA", "BBB"],
        "event_day": [date(2024, 1, 2), date(2024, 1, 5)],
        "oof_tail_classifier": [0.7, float("nan")],
    })
    path = tmp_path / "oof_scores.parquet"
    df.to_parquet(path, index=False)

    with caplog.at_level(logging.INFO, logger="backtest.model_scores"):
        scores = load_model_scores(str(path))

    assert scores == {("AAA", date(2024, 1, 2)): pytest.approx(0.7)}
    messages = [r.message for r in caplog.records]
    assert any("1 NaN dropped" in m for m in messages), messages


def test_loader_respects_explicit_score_column(tmp_path):
    df = pd.DataFrame({
        "ticker": ["AAA"],
        "event_day": [date(2024, 1, 2)],
        "oof_tail_classifier": [0.7],
        "oof_regressor": [1.5],
    })
    path = tmp_path / "oof_scores.parquet"
    df.to_parquet(path, index=False)

    scores = load_model_scores(str(path), score_col="oof_regressor")
    assert scores == {("AAA", date(2024, 1, 2)): pytest.approx(1.5)}
