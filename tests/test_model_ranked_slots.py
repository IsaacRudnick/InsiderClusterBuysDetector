"""Tests for the model_ranked_n{NN}_hold63 slot-count sweep
(backtest/strategies.py).

Background: model_ranked_top_hold63 (10 slots, 10% weight) beats SPY and
beats unranked all_clusters_hold63; model_ranked_hold63 (50 slots, 2%
weight) is WORSE than unranked all_clusters_hold63 despite the identical
rank_fn. MODEL_RANKED_SLOT_COUNTS fills in the curve between those two data
points at 5, 10, 15, 20, 25, 30, 40 slots.

The one property that matters most here is sizing: every hold63 strategy in
this registry holds to slots * weight == 100% deployed (see
_s_model_top_target's comment in strategies.py). If a generated variant's
weight were wrong, the sweep would measure cash drag instead of ranking
quality, so that invariant gets its own test rather than being folded into
a general smoke test.

Runnable standalone via `python -m pytest tests/test_model_ranked_slots.py
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
    MODEL_RANKED_SLOT_COUNTS,
    STRATEGIES,
    rank_by_model_score,
)

NAME_RE = re.compile(r"^model_ranked_n(\d{2,3})_hold63$")

_BY_NAME = {s.name: s for s in STRATEGIES}


def _slot_variants():
    return [s for s in STRATEGIES if NAME_RE.match(s.name)]


# ---------------------------------------------------------------------------
# Part 1: the family exists, one strategy per configured slot count, and
# every name is unique / correctly formatted.
# ---------------------------------------------------------------------------
def test_expected_slot_counts_configured():
    assert MODEL_RANKED_SLOT_COUNTS == [5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 100]


def test_one_variant_registered_per_slot_count():
    variants = _slot_variants()
    assert len(variants) == len(MODEL_RANKED_SLOT_COUNTS)
    got_slots = sorted(s.max_concurrent_tickers for s in variants)
    assert got_slots == sorted(MODEL_RANKED_SLOT_COUNTS)


def test_names_use_02d_slot_formatting_and_parse_back_to_the_right_count():
    # f"{n:02d}" zero-pads two-digit counts (05, 10, ..., 75) but a 3-digit
    # count (100) naturally overflows the width instead of truncating, so
    # names stop being uniform-width once the sweep passes 99 slots. That
    # also means raw lexicographic string sort no longer equals numeric sort
    # on slot count ("n100_..." < "n40_..." as strings, since '1' < '4') --
    # this test checks the thing that actually matters (each name parses
    # back to its own slot count, unambiguously) rather than string order.
    expected_names = [
        f"model_ranked_n{n:02d}_hold63" for n in MODEL_RANKED_SLOT_COUNTS
    ]
    for name, slots in zip(expected_names, MODEL_RANKED_SLOT_COUNTS):
        assert name in _BY_NAME, f"expected generated strategy {name!r} missing"
        m = NAME_RE.match(name)
        assert m is not None
        assert int(m.group(1)) == slots

    # Numeric sort on the parsed slot count matches MODEL_RANKED_SLOT_COUNTS'
    # own (already-sorted) order.
    parsed = [int(NAME_RE.match(n).group(1)) for n in expected_names]
    assert parsed == sorted(MODEL_RANKED_SLOT_COUNTS)


def test_all_strategy_names_globally_unique():
    names = [s.name for s in STRATEGIES]
    dupes = {n for n in names if names.count(n) > 1}
    assert not dupes, f"duplicate strategy names in registry: {dupes}"


# ---------------------------------------------------------------------------
# Part 2: the critical sizing invariant -- slots * weight == 1.0 (100%
# deployed) for every generated variant, within floating-point tolerance.
# ---------------------------------------------------------------------------
def test_every_variant_deploys_exactly_100_percent_when_full():
    qualifying_state = {"num_insiders": 2}
    cap = 1.0
    for strat in _slot_variants():
        weight = strat.target_fn(qualifying_state, cap)
        assert weight > 0, f"{strat.name} target_fn did not fire on a qualifying state"
        total_deployed = strat.max_concurrent_tickers * weight
        assert total_deployed == pytest.approx(1.0, rel=1e-9), (
            f"{strat.name}: {strat.max_concurrent_tickers} slots x "
            f"{weight!r} weight = {total_deployed!r}, expected 1.0 -- a "
            f"mis-sized variant measures cash drag, not ranking quality"
        )


def test_non_qualifying_state_gets_zero_target():
    for strat in _slot_variants():
        assert strat.target_fn({"num_insiders": 1}, 1.0) == 0.0
        assert strat.target_fn({"num_insiders": 0}, 1.0) == 0.0
        assert strat.target_fn({}, 1.0) == 0.0


# ---------------------------------------------------------------------------
# Part 3: every generated variant keeps everything else identical to
# model_ranked_top_hold63 -- same gate, hold_days=63, rank_fn, no stop-loss,
# not equal-weight.
# ---------------------------------------------------------------------------
def test_every_variant_matches_the_shared_policy_shape():
    for strat in _slot_variants():
        assert strat.hold_days == 63
        assert strat.rank_fn is rank_by_model_score
        assert strat.stop_loss_pct is None
        assert strat.equal_weight is False


# ---------------------------------------------------------------------------
# Part 4: the two reference strategies are untouched by this change.
# ---------------------------------------------------------------------------
def test_model_ranked_hold63_unchanged():
    strat = _BY_NAME["model_ranked_hold63"]
    assert strat.max_concurrent_tickers == 50
    assert strat.hold_days == 63
    assert strat.rank_fn is rank_by_model_score
    # all_clusters_equal_weight's target_fn: 2% flat on any qualifying state.
    assert strat.target_fn({"num_insiders": 2}, 1.0) == pytest.approx(0.02)
    assert strat.target_fn({"num_insiders": 1}, 1.0) == 0.0


def test_model_ranked_top_hold63_unchanged():
    strat = _BY_NAME["model_ranked_top_hold63"]
    assert strat.max_concurrent_tickers == 10
    assert strat.hold_days == 63
    assert strat.rank_fn is rank_by_model_score
    # _s_model_top_target: 10% flat on any qualifying state (10 x 10% == 100%).
    assert strat.target_fn({"num_insiders": 2}, 1.0) == pytest.approx(0.10)
    assert strat.target_fn({"num_insiders": 1}, 1.0) == 0.0


def test_reference_strategies_not_double_counted_in_the_swept_family():
    # model_ranked_hold63 (50 slots) and model_ranked_top_hold63 (10 slots)
    # are hand-authored above the sweep and must not also match the
    # generated-name pattern, so the family's count stays exactly len(
    # MODEL_RANKED_SLOT_COUNTS) and nobody accidentally double-registers a
    # 10-slot or 50-slot variant under two different names.
    swept_names = {s.name for s in _slot_variants()}
    assert "model_ranked_hold63" not in swept_names
    assert "model_ranked_top_hold63" not in swept_names


# ---------------------------------------------------------------------------
# Part 5: score-floor variants (model_floor_p50/p75/p90_hold63) were
# deliberately not implemented (see the comment block in strategies.py
# explaining the lookahead problem with target_fn's (state, cap) signature).
# This test only guards against someone re-adding them without re-deriving
# a no-lookahead percentile source -- it intentionally does NOT test their
# behavior, since they don't exist.
# ---------------------------------------------------------------------------
def test_score_floor_variants_are_not_present():
    for pct in ("p50", "p75", "p90"):
        assert f"model_floor_{pct}_hold63" not in _BY_NAME
