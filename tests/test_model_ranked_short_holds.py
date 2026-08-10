"""Tests for the model_ranked_n{NN}_hold{H} short-hold family
(backtest/strategies.py).

Background: pool-wide, the SPY-adjusted forward return is lottery-shaped and
the skew WORSENS with horizon (63-day: mean +0.76%, median -2.34%, win rate
43.9%; 252-day: mean +2.58%, median -10.53%, win rate 38.5%; no horizon has
a positive median), but at 10 days the payoff is far more symmetric (48.2%
win rate). MODEL_RANKED_SHORT_HOLD_DAYS (10, 21) x
MODEL_RANKED_SHORT_HOLD_SLOT_COUNTS (10, 25, 50) tests whether a shorter
hold trades better against that payoff, independent of slot count (breadth).

Mirrors tests/test_model_ranked_slots.py's conventions. The one property
that matters most is sizing: every variant in this family holds to
slots * weight == 100% deployed, exactly like the hold63 sweep it reuses
_make_model_ranked_slot_target from -- see _s_model_top_target's comment in
strategies.py for the full argument against a fixed weight across slot
counts.

Runnable standalone via `python -m pytest tests/test_model_ranked_short_holds.py
-q` from the repo root.
"""

from __future__ import annotations

import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import pytest  # noqa: E402

from backtest.strategies import (  # noqa: E402
    MODEL_RANKED_SHORT_HOLD_DAYS,
    MODEL_RANKED_SHORT_HOLD_SLOT_COUNTS,
    STRATEGIES,
    rank_by_model_score,
)

NAME_RE = re.compile(r"^model_ranked_n(\d{2})_hold(\d+)$")

_BY_NAME = {s.name: s for s in STRATEGIES}


def _short_hold_variants():
    variants = []
    for s in STRATEGIES:
        m = NAME_RE.match(s.name)
        if m is not None and int(m.group(2)) in MODEL_RANKED_SHORT_HOLD_DAYS:
            variants.append(s)
    return variants


# ---------------------------------------------------------------------------
# Part 1: the family exists, matches the configured (hold_days, slots) grid,
# and every name is unique / correctly formatted.
# ---------------------------------------------------------------------------
def test_expected_short_hold_config():
    assert MODEL_RANKED_SHORT_HOLD_DAYS == [10, 21]
    assert MODEL_RANKED_SHORT_HOLD_SLOT_COUNTS == [10, 25, 50]


def test_one_variant_registered_per_hold_days_x_slots_combo():
    variants = _short_hold_variants()
    expected_combos = {
        (hold, slots)
        for hold in MODEL_RANKED_SHORT_HOLD_DAYS
        for slots in MODEL_RANKED_SHORT_HOLD_SLOT_COUNTS
    }
    got_combos = {
        (s.hold_days, s.max_concurrent_tickers) for s in variants
    }
    assert got_combos == expected_combos
    assert len(variants) == len(expected_combos)


def test_names_follow_the_existing_nNN_holdH_scheme():
    for hold in MODEL_RANKED_SHORT_HOLD_DAYS:
        for slots in MODEL_RANKED_SHORT_HOLD_SLOT_COUNTS:
            name = f"model_ranked_n{slots:02d}_hold{hold}"
            assert name in _BY_NAME, f"expected generated strategy {name!r} missing"


def test_all_strategy_names_globally_unique():
    names = [s.name for s in STRATEGIES]
    dupes = {n for n in names if names.count(n) > 1}
    assert not dupes, f"duplicate strategy names in registry: {dupes}"


def test_short_hold_names_do_not_collide_with_hold63_family():
    # model_ranked_n{NN}_hold63 uses a distinct suffix ("_hold63") from this
    # family's "_hold10"/"_hold21", so a 10- or 25- or 50-slot short-hold
    # variant can never be mistaken for (or overwrite) its hold63 sibling.
    short_names = {s.name for s in _short_hold_variants()}
    hold63_names = {f"model_ranked_n{n:02d}_hold63"
                     for n in MODEL_RANKED_SHORT_HOLD_SLOT_COUNTS}
    assert short_names.isdisjoint(hold63_names)
    for n in hold63_names:
        assert n in _BY_NAME  # the hold63 sibling still exists, untouched


# ---------------------------------------------------------------------------
# Part 2: the critical sizing invariant -- slots * weight == 1.0 (100%
# deployed) for every generated variant, within floating-point tolerance.
# ---------------------------------------------------------------------------
def test_every_variant_deploys_exactly_100_percent_when_full():
    qualifying_state = {"num_insiders": 2}
    cap = 1.0
    for strat in _short_hold_variants():
        weight = strat.target_fn(qualifying_state, cap)
        assert weight > 0, f"{strat.name} target_fn did not fire on a qualifying state"
        total_deployed = strat.max_concurrent_tickers * weight
        assert total_deployed == pytest.approx(1.0, rel=1e-9), (
            f"{strat.name}: {strat.max_concurrent_tickers} slots x "
            f"{weight!r} weight = {total_deployed!r}, expected 1.0 -- a "
            f"mis-sized variant measures cash drag, not ranking quality"
        )


def test_non_qualifying_state_gets_zero_target():
    for strat in _short_hold_variants():
        assert strat.target_fn({"num_insiders": 1}, 1.0) == 0.0
        assert strat.target_fn({"num_insiders": 0}, 1.0) == 0.0
        assert strat.target_fn({}, 1.0) == 0.0


# ---------------------------------------------------------------------------
# Part 3: every generated variant keeps everything else identical to its
# model_ranked_n{NN}_hold63 sibling -- same rank_fn, no stop-loss, not
# equal-weight, and hold_days is the ONLY thing that differs from that
# sibling at the same slot count.
# ---------------------------------------------------------------------------
def test_every_variant_matches_the_shared_policy_shape():
    for strat in _short_hold_variants():
        assert strat.hold_days in MODEL_RANKED_SHORT_HOLD_DAYS
        assert strat.rank_fn is rank_by_model_score
        assert strat.stop_loss_pct is None
        assert strat.equal_weight is False


def test_short_hold_weight_matches_hold63_sibling_at_same_slot_count():
    for slots in MODEL_RANKED_SHORT_HOLD_SLOT_COUNTS:
        hold63_name = f"model_ranked_n{slots:02d}_hold63"
        hold63_weight = _BY_NAME[hold63_name].target_fn({"num_insiders": 2}, 1.0)
        for hold in MODEL_RANKED_SHORT_HOLD_DAYS:
            short_name = f"model_ranked_n{slots:02d}_hold{hold}"
            short_weight = _BY_NAME[short_name].target_fn({"num_insiders": 2}, 1.0)
            assert short_weight == pytest.approx(hold63_weight), (
                f"{short_name} weight should match {hold63_name} weight -- "
                f"only hold_days should differ between them"
            )


# ---------------------------------------------------------------------------
# Part 4: reference strategies elsewhere in the registry are untouched by
# this additive change.
# ---------------------------------------------------------------------------
def test_model_ranked_n10_hold63_unchanged():
    strat = _BY_NAME["model_ranked_n10_hold63"]
    assert strat.max_concurrent_tickers == 10
    assert strat.hold_days == 63
    assert strat.rank_fn is rank_by_model_score
    assert strat.target_fn({"num_insiders": 2}, 1.0) == pytest.approx(0.10)
    assert strat.target_fn({"num_insiders": 1}, 1.0) == 0.0


def test_model_ranked_top_hold63_unchanged():
    strat = _BY_NAME["model_ranked_top_hold63"]
    assert strat.max_concurrent_tickers == 10
    assert strat.hold_days == 63
    assert strat.rank_fn is rank_by_model_score
    assert strat.target_fn({"num_insiders": 2}, 1.0) == pytest.approx(0.10)
