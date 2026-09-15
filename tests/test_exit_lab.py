"""Tests for tools/exit_lab.py -- the path-dependent trade simulator.

The most important test here is `test_vectorised_matches_reference`. Every exit
conclusion in RESEARCH_NOTES.md comes from `simulate_paths`, the matrix
implementation, because the readable per-trade loop is about a hundred times
too slow to sweep ninety rules. If the two disagree, the readable one is the
specification and the fast one is a bug wearing its results.

The rest pin the deliberately conservative fill choices, each of which is a
place a backtest can quietly invent money:

  - a gap through a stop fills at the OPEN, not at the stop price
  - the trailing stop ratchets on CLOSES already in the past, so today's close
    cannot raise the level today's low is tested against
  - a trail that has not armed yet cannot trigger
  - a dead/delisted path exits at the last real price, never forward-filled
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

from tools import exit_lab as el  # noqa: E402


def px_frame(closes, opens=None, lows=None, start="2020-01-02"):
    """A price frame with an entry bar at position 0 and `closes` after it."""
    n = len(closes) + 1
    idx = pd.bdate_range(start, periods=n)
    o = [100.0] + (list(opens) if opens is not None else list(closes))
    lo = [100.0] + (list(lows) if lows is not None else list(closes))
    c = [100.0] + list(closes)
    return pd.DataFrame(
        {"open": o, "high": [max(a, b) for a, b in zip(o, c)],
         "low": lo, "close": c}, index=idx
    )


def as_events(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for i, (tkr, f) in enumerate(frames.items()):
        el._PX_CACHE[tkr] = f
        rows.append(dict(ticker=tkr, entry_day=f.index[0], entry_idx=i * 5,
                         score=float(i)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# The fast path is the one that produced every published number
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "rule",
    [
        el.ExitRule(0.00, 0.00, 21),
        el.ExitRule(0.10, 0.00, 63),
        el.ExitRule(0.00, 0.20, 63, 0.00),
        el.ExitRule(0.15, 0.30, 126, 0.10),
        el.ExitRule(0.20, 0.15, 252, 0.10),
    ],
)
def test_vectorised_matches_reference(rule):
    rng = np.random.default_rng(0)
    frames = {}
    for i in range(60):
        steps = rng.normal(0.002, 0.05, 260)
        closes = 100.0 * np.cumprod(1.0 + steps)
        lows = closes * (1.0 - np.abs(rng.normal(0, 0.02, 260)))
        opens = closes * (1.0 + rng.normal(0, 0.01, 260))
        frames[f"S{i:03d}"] = px_frame(closes, opens, lows)
    ev = as_events(frames)

    paths = el.build_paths(ev, rule.max_hold)
    fast = el.simulate_paths(paths, rule, cost_bps=0.0)
    slow = el.simulate_events(ev, rule, cost_bps=0.0)
    m = fast.merge(slow, on=["ticker", "entry_day"], suffixes=("_f", "_s"))
    assert len(m) == len(slow) > 0
    np.testing.assert_allclose(m["ret_f"], m["ret_s"], rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(m["days_held_f"], m["days_held_s"])


# ---------------------------------------------------------------------------
# The conservative fill choices
# ---------------------------------------------------------------------------

def test_gap_through_a_stop_fills_at_the_open_not_the_stop():
    """A resting stop cannot execute at a price the market never traded.

    Day 1 opens at 70 with a stop at 80: the fill must be 70, a -30% loss, not
    a comfortable -20%.
    """
    f = px_frame(closes=[70.0, 71.0], opens=[70.0, 71.0], lows=[69.0, 70.0])
    r = el.simulate_trade(f, f.index[0], el.ExitRule(hard_stop=0.20, max_hold=5))
    assert r is not None
    assert r.ret == pytest.approx(-0.30, abs=1e-9)


def test_stop_fills_at_the_stop_when_the_open_is_above_it():
    f = px_frame(closes=[95.0, 85.0], opens=[99.0, 95.0], lows=[94.0, 79.0])
    r = el.simulate_trade(f, f.index[0], el.ExitRule(hard_stop=0.20, max_hold=5))
    assert r.ret == pytest.approx(-0.20, abs=1e-9)


def test_trailing_stop_ratchets_only_on_past_closes():
    """Today's close must not raise the level today's low is tested against.

    Prices rise to 200, then a day dips to 175 intraday. A 20% trail off the
    prior peak close of 200 sits at 160, so 175 must NOT stop out.
    """
    f = px_frame(closes=[150.0, 200.0, 190.0], opens=[150.0, 200.0, 190.0],
                 lows=[150.0, 200.0, 175.0])
    r = el.simulate_trade(
        f, f.index[0], el.ExitRule(trail_stop=0.20, max_hold=5)
    )
    assert r.exit_reason == "time"


def test_trailing_stop_fills_at_its_level_when_there_is_no_gap():
    """Peak close 200, 20% trail => level 160. The day opens at 165, above the
    level, and trades down through it, so the fill is the level itself."""
    f = px_frame(closes=[150.0, 200.0, 150.0], opens=[150.0, 200.0, 165.0],
                 lows=[150.0, 200.0, 150.0])
    r = el.simulate_trade(
        f, f.index[0], el.ExitRule(trail_stop=0.20, max_hold=5)
    )
    assert r.exit_reason == "trail"
    assert r.ret == pytest.approx(0.60, abs=1e-9)      # 160 / 100 - 1


def test_trailing_stop_gaps_through_to_the_open():
    """Same trail level of 160, but the day OPENS at 158. The stop cannot fill
    at 160 because the market never traded there, so the trade books 158."""
    f = px_frame(closes=[150.0, 200.0, 150.0], opens=[150.0, 200.0, 158.0],
                 lows=[150.0, 200.0, 150.0])
    r = el.simulate_trade(
        f, f.index[0], el.ExitRule(trail_stop=0.20, max_hold=5)
    )
    assert r.exit_reason == "trail"
    assert r.ret == pytest.approx(0.58, abs=1e-9)      # 158 / 100 - 1


def test_unarmed_trail_cannot_trigger():
    """With activate_at=50%, a position that never gets there is untouched by
    the trail no matter how far it falls."""
    f = px_frame(closes=[105.0, 80.0, 70.0], opens=[105.0, 80.0, 70.0],
                 lows=[105.0, 80.0, 70.0])
    armed = el.simulate_trade(
        f, f.index[0], el.ExitRule(trail_stop=0.10, max_hold=5, activate_at=0.0)
    )
    unarmed = el.simulate_trade(
        f, f.index[0], el.ExitRule(trail_stop=0.10, max_hold=5, activate_at=0.50)
    )
    assert armed.exit_reason == "trail"
    assert unarmed.exit_reason == "time"


def test_a_dead_path_is_not_forward_filled():
    """Padding a delisted name to the full holding period must not let it sit
    at par -- that is the single most flattering bug available here."""
    f = px_frame(closes=[90.0, 80.0])
    ev = as_events({"DEAD": f})
    paths = el.build_paths(ev, 252)
    assert np.isnan(paths.close[0, 5])
    out = el.simulate_paths(paths, el.ExitRule(max_hold=252), cost_bps=0.0)
    assert len(out) == 1
    assert out["ret"].iloc[0] == pytest.approx(-0.20, abs=1e-9)


def test_cost_is_charged_once_per_trade():
    f = px_frame(closes=[110.0, 110.0])
    ev = as_events({"A": f})
    paths = el.build_paths(ev, 21)
    free = el.simulate_paths(paths, el.ExitRule(max_hold=2), cost_bps=0.0)
    paid = el.simulate_paths(paths, el.ExitRule(max_hold=2), cost_bps=50.0)
    assert free["ret"].iloc[0] - paid["ret"].iloc[0] == pytest.approx(0.005)


# ---------------------------------------------------------------------------
# The daily-marked book
# ---------------------------------------------------------------------------

def test_daily_book_respects_the_slot_limit():
    frames = {f"S{i}": px_frame(closes=[101.0] * 30) for i in range(10)}
    ev = as_events(frames)
    ev["entry_idx"] = 0                     # all arrive the same day
    paths = el.build_paths(ev, 30)
    trades = el.simulate_paths(paths, el.ExitRule(max_hold=30), cost_bps=0.0)
    port = el.daily_marked_portfolio(paths, trades, n_slots=3, cost_bps=0.0)
    assert port["n_active"].max() <= 3


def test_empty_slots_earn_zero():
    """A half-invested book must not annualise as though it were fully
    invested -- that is what makes the comparison to an index honest."""
    frames = {"A": px_frame(closes=[110.0] * 10)}
    ev = as_events(frames)
    paths = el.build_paths(ev, 10)
    trades = el.simulate_paths(paths, el.ExitRule(max_hold=10), cost_bps=0.0)
    one = el.daily_marked_portfolio(paths, trades, n_slots=1, cost_bps=0.0)
    ten = el.daily_marked_portfolio(paths, trades, n_slots=10, cost_bps=0.0)
    assert one["ret"].sum() == pytest.approx(10 * ten["ret"].sum(), rel=1e-9)


# ---------------------------------------------------------------------------
# Unpriceable events are counted, not silently dropped
# ---------------------------------------------------------------------------

def _with_unpriced(n_priced: int, n_unpriced: int) -> pd.DataFrame:
    ev = as_events({f"P{i}": px_frame(closes=[101.0] * 5) for i in range(n_priced)})
    gone = []
    for i in range(n_unpriced):
        el._PX_CACHE[f"GONE{i}"] = None     # what load_prices caches for no file
        gone.append(dict(ticker=f"GONE{i}", entry_day=ev["entry_day"].iloc[0],
                         entry_idx=0, score=0.0))
    return pd.concat([ev, pd.DataFrame(gone)], ignore_index=True)


def test_unpriced_events_are_reported(capsys):
    paths = el.build_paths(_with_unpriced(9, 1), 5)
    assert len(paths.meta) == 9
    assert "1 of 10 events" in capsys.readouterr().err


def test_full_coverage_is_silent(capsys):
    el.build_paths(_with_unpriced(5, 0), 5)
    assert capsys.readouterr().err == ""


def test_min_coverage_fails_a_sparse_cache():
    """The failure mode behind a fabricated Sharpe of 3.8: most events had no
    price file and the search ran on the few that did."""
    with pytest.raises(RuntimeError, match="below the required 98% coverage"):
        el.build_paths(_with_unpriced(2, 8), 5, min_coverage=0.98)
