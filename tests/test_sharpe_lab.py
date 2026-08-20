"""Tests for tools/sharpe_lab.py -- book construction, risk metrics, and the
permutation test the risk-adjusted conclusions rest on.

Runnable standalone via `python -m pytest tests/test_sharpe_lab.py -q`.

The three that matter most:

  test_panel_path_matches_pandas_path
      The fast flat-array path exists only so the permutation test is cheap
      enough that nobody skips it. If it computes something different from the
      reference path, every Sharpe number in RESEARCH_NOTES.md's 2026-08-20
      risk section is measuring the wrong thing.

  test_permutation_null_recovers_a_known_nothing
      On scores that are pure noise by construction, the observed statistic
      must sit in the middle of its own null. A permutation test that says
      noise is significant is worse than no test.

  test_periods_never_overlap
      Overlapping windows are the standard way to turn seven years of data
      into an impressive and meaningless t-statistic.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools import sharpe_lab as sh  # noqa: E402


def make_df(n: int = 1200, seed: int = 0, signal: float = 0.0) -> pd.DataFrame:
    """Rows shaped like the real prepared frame, with a tunable planted edge.

    `signal` scales how much the score actually predicts the forward return, so
    the same generator produces both the "there is nothing here" case and the
    "there is something here" case.
    """
    rng = np.random.default_rng(seed)
    days = pd.date_range("2020-01-02", periods=n, freq="B")
    score = rng.normal(size=n)
    vol = rng.uniform(0.2, 1.5, n)
    fwd = signal * score + rng.normal(0, 0.10, n)
    return pd.DataFrame(
        {
            "ticker": [f"T{i % 50:02d}" for i in range(n)],
            "event_day": days,
            "entry_day": days,
            "entry_idx": np.arange(n),
            "entry_open": rng.uniform(1.0, 50.0, n),
            "x_vol_63_ann": vol,
            "x_log_adv20": rng.uniform(10, 18, n),
            "fwd_21": fwd,
            "bench_SPY": rng.normal(0.01, 0.03, n),
            "bench_IWM": rng.normal(0.008, 0.04, n),
            "ens": score,
            "period": np.arange(n) // 21,
        }
    )


# ---------------------------------------------------------------------------
# The fast path must equal the reference path
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "spec",
    [
        sh.BookSpec(lo=0.70, hi=0.90),
        sh.BookSpec(lo=0.50, hi=1.00, weighting="inv_vol"),
        sh.BookSpec(lo=0.80, hi=0.95, weighting="inv_var"),
        sh.BookSpec(lo=0.60, hi=0.90, min_price=5.0),
        sh.BookSpec(lo=0.60, hi=0.90, max_weight=0.15),
        sh.BookSpec(lo=0.70, hi=1.00, max_names=5),
    ],
)
def test_panel_path_matches_pandas_path(spec):
    df = make_df()
    panel = sh.build_panel(df, "ens")
    a = sh.period_returns(df, spec, score_col="ens")
    b = sh.panel_returns(panel, spec)
    assert len(a) == len(b), f"{len(a)} vs {len(b)} periods"
    np.testing.assert_allclose(
        a["ret"].to_numpy(), b["ret"].to_numpy(), rtol=1e-9, atol=1e-12
    )
    np.testing.assert_array_equal(
        a["n_held"].to_numpy(), b["n_held"].to_numpy()
    )


# ---------------------------------------------------------------------------
# Selection and period discipline
# ---------------------------------------------------------------------------

def test_periods_never_overlap():
    df = make_df()
    per = sh.period_returns(df, sh.BookSpec(lo=0.0, hi=1.0), score_col="ens")
    assert per["period"].is_unique
    # Every period must cover a disjoint stretch of the trading-day index.
    spans = df.groupby("period")["entry_idx"].agg(["min", "max"]).sort_index()
    assert (spans["min"].shift(-1).dropna() > spans["max"][:-1]).all()


def test_band_is_cut_inside_each_period():
    """A band cut once globally would leak the future into every early period.

    Verified behaviourally: a period whose scores are all far below the global
    distribution must still fill its band from its own rows.
    """
    df = make_df()
    df.loc[df["period"] == 0, "ens"] -= 100.0   # this period is globally awful
    per = sh.period_returns(df, sh.BookSpec(lo=0.70, hi=0.90), score_col="ens")
    assert 0 in set(per["period"]), "period 0 was dropped by a global cut"


def test_weightings_all_sum_to_one_and_differ():
    df = make_df()
    g = df[df["period"] == 3]
    ws = {name: fn(g) for name, fn in sh.WEIGHTINGS.items()}
    for name, w in ws.items():
        assert math.isclose(float(w.sum()), 1.0, rel_tol=1e-9), name
        assert (w > 0).all(), name
    assert not np.allclose(ws["equal"], ws["inv_vol"])


def test_position_cap_binds():
    df = make_df()
    spec = sh.BookSpec(lo=0.90, hi=1.00, weighting="inv_var", max_weight=0.15)
    panel = sh.build_panel(df, "ens")
    capped = sh.panel_returns(panel, spec)
    uncapped = sh.panel_returns(
        panel, sh.BookSpec(**{**spec.__dict__, "max_weight": 1.0})
    )
    assert not np.allclose(capped["ret"], uncapped["ret"])


def test_price_floor_removes_rows_and_can_empty_a_book():
    df = make_df()
    lo = sh.period_returns(df, sh.BookSpec(lo=0.7, hi=0.9), score_col="ens")
    hi = sh.period_returns(
        df, sh.BookSpec(lo=0.7, hi=0.9, min_price=45.0), score_col="ens"
    )
    assert len(hi) < len(lo)


def test_cost_reduces_return_by_exactly_the_cost():
    df = make_df()
    panel = sh.build_panel(df, "ens")
    free = sh.panel_returns(panel, sh.BookSpec(lo=0.7, hi=0.9, cost_bps=0.0))
    paid = sh.panel_returns(panel, sh.BookSpec(lo=0.7, hi=0.9, cost_bps=50.0))
    np.testing.assert_allclose(
        free["ret"].to_numpy() - paid["ret"].to_numpy(), 0.0050, atol=1e-12
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def test_sharpe_matches_its_definition():
    per = pd.DataFrame(
        dict(ret=[0.02, -0.01, 0.03, 0.00, 0.015, -0.005] * 4,
             bench_SPY=0.005, bench_IWM=0.004,
             year=2021, n_held=10, period=range(24))
    )
    m = sh.evaluate(per, "t")
    r = per["ret"]
    expect = (r.mean() / r.std(ddof=1)) * math.sqrt(sh.PERIODS_PER_YEAR)
    assert math.isclose(m["sharpe"], expect, rel_tol=1e-9)


def test_drawdown_is_negative_and_bounded():
    per = pd.DataFrame(
        dict(ret=[0.1, -0.5, 0.2, -0.3, 0.4] * 4, bench_SPY=0.0,
             bench_IWM=0.0, year=2021, n_held=5, period=range(20))
    )
    m = sh.evaluate(per, "t")
    assert -1.0 < m["max_drawdown"] < 0.0


def test_both_benchmarks_are_always_reported():
    """Reporting whichever benchmark flatters the result is benchmark shopping,
    so the report is built to make omitting one impossible."""
    df = make_df()
    m = sh.evaluate(sh.period_returns(df, sh.BookSpec(lo=0.7, hi=0.9),
                                      score_col="ens"), "t")
    for b in ("SPY", "IWM"):
        assert f"excess_{b}" in m and f"yrs_{b}" in m and f"ir_{b}" in m


# ---------------------------------------------------------------------------
# The permutation test
# ---------------------------------------------------------------------------

def _search(panel, scores):
    best = -np.inf
    for lo, hi in ((0.5, 1.0), (0.7, 0.9), (0.8, 1.0), (0.9, 1.0)):
        per = sh.panel_returns(panel, sh.BookSpec(lo=lo, hi=hi), scores)
        if len(per) < 10:
            continue
        m = sh.evaluate(per)
        if m and np.isfinite(m.get("sharpe", np.nan)):
            best = max(best, m["sharpe"])
    return best


def test_permutation_null_recovers_a_known_nothing():
    """With no planted edge, the observed best-of-search must be unremarkable
    against its own null. If this test ever fails, the test is telling you the
    audit would have blessed noise."""
    df = make_df(n=1600, seed=7, signal=0.0)
    panel = sh.build_panel(df, "ens")
    observed = _search(panel, None)
    res = sh.permutation_test_panel(panel, _search, observed, n_draws=120, seed=1)
    assert res["p_value"] > 0.05, (
        f"noise scored p={res['p_value']:.3f}; the null is not calibrated"
    )


def test_permutation_null_detects_a_planted_edge():
    """The mirror of the above: a large planted edge must clear the null,
    otherwise the test has no power and would reject everything."""
    df = make_df(n=1600, seed=7, signal=0.09)
    panel = sh.build_panel(df, "ens")
    observed = _search(panel, None)
    res = sh.permutation_test_panel(panel, _search, observed, n_draws=120, seed=1)
    assert res["p_value"] < 0.05, (
        f"a planted edge scored p={res['p_value']:.3f}; the test has no power"
    )
    assert res["observed"] > res["null_p95"]


def test_permutation_shuffles_within_periods_only():
    """Shuffling across periods would destroy the period structure as well as
    the signal, which is a different and much weaker null."""
    df = make_df(n=600)
    panel = sh.build_panel(df, "ens")
    seen = {}

    def capture(p, scores):
        seen["sizes"] = [len(s) for s in scores]
        return 1.0

    sh.permutation_test_panel(panel, capture, 0.0, n_draws=1)
    assert seen["sizes"] == [len(s) for s in panel.score]
