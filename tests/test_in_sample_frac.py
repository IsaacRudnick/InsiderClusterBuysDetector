"""Tests for backtest/metrics.py's in_sample_frac column (compute()).

Background: backtest/signal_fit.py fits learned_score (FitResult, mean-
return ridge) and tail_score (TailFitResult, P(moonshot) log2-lift) on a
single chronological train/test split of THIS SAME run's events. A strategy
whose target_fn reads learned_score or tail_score then trades the FULL
backtest window (see backtest.py's window_start/window_end), which includes
the fit's own training period -- for the 72-month run this task was built
against, train_end=2024-10-31 against window 2020-08-04..2026-08-04 is
~70% overlap. Nothing in the old summary output flagged this, so a
fit-dependent strategy's alpha/t_alpha looked identical to a genuinely
out-of-sample one.

VERIFIED against backtest/strategies.py and backtest/state.py: the
fit-dependent set is learned_gt_m01, learned_gt_p00, learned_gt_p03,
learned_score_weighted, learned_tpo_gated (all read state["learned_score"])
plus learned_tail_concentrated (reads state["tail_score"], from the
*separate* tail fit). thr_gt_* strategies threshold
state["conviction_score"], which backtest/state.py's _build_state computes
ALWAYS from insider_cluster_buys.DEFAULT_WEIGHTS -- a fixed, hand-tuned
constant never fit on this run's train/test split -- so despite the naming
resemblance to learned_*, thr_gt_* must read 0.0. model_ranked_* strategies
rank by state["model_score"], populated from an out-of-fold parquet file,
not this run's in-run fit, so they must read 0.0 too.

RunResult is built directly (there is no existing test helper for this --
see tests/test_metrics_concentration.py for the same pattern), since
compute()'s in_sample_frac path only reads result.strategy and
result.equity_curve.index.
"""

from __future__ import annotations

import math
from collections import Counter
from datetime import date

import pandas as pd
import pytest

from backtest import metrics
from backtest.engine import RunResult

WINDOW_START = "2020-08-04"
WINDOW_END = "2026-08-04"
# Matches the actual run this task was built against (signal_weights.json's
# fit.train_end), not an arbitrary test date.
TRAIN_END = date(2024, 10, 31)


def _run_result(strategy: str, *, start: str, end: str, n_days: int = 30) -> RunResult:
    idx = pd.date_range(start, end, periods=n_days)
    equity = pd.Series([100_000.0] * n_days, index=idx)
    exposure = pd.Series([0.0] * n_days, index=idx)
    n_tickers = pd.Series([0] * n_days, index=idx)
    return RunResult(
        strategy=strategy, exit_label="test_exit",
        equity_curve=equity, trades=[], skips=Counter(),
        n_rebalances=0, exposure_curve=exposure,
        n_tickers_curve=n_tickers,
    )


def test_learned_strategy_gets_nonzero_fraction_matching_formula():
    r = _run_result("learned_gt_p00", start=WINDOW_START, end=WINDOW_END)
    out = metrics.compute(r, fit_train_end=TRAIN_END)
    ws = pd.Timestamp(r.equity_curve.index.min())
    we = pd.Timestamp(r.equity_curve.index.max())
    te = pd.Timestamp(TRAIN_END)
    expected = (te - ws) / (we - ws)
    assert out["in_sample_frac"] == pytest.approx(expected, abs=1e-6)
    # Roughly 70%, matching the task's own worked example for this run.
    assert 0.65 < out["in_sample_frac"] < 0.75


@pytest.mark.parametrize("name", [
    "learned_gt_m01", "learned_gt_p00", "learned_gt_p03",
    "learned_score_weighted", "learned_tpo_gated",
])
def test_all_mean_fit_dependent_strategies_get_nonzero_fraction(name):
    r = _run_result(name, start=WINDOW_START, end=WINDOW_END)
    out = metrics.compute(r, fit_train_end=TRAIN_END)
    assert out["in_sample_frac"] > 0.5, name


def test_learned_tail_concentrated_uses_tail_fit_not_mean_fit():
    # A different (missing vs present) train_end on the two fits confirms
    # the right FitResult/TailFitResult metadata reaches the right
    # strategy -- not just "any fit_train_end present -> nonzero".
    r = _run_result("learned_tail_concentrated", start=WINDOW_START, end=WINDOW_END)

    out_no_tail = metrics.compute(r, fit_train_end=TRAIN_END, tail_train_end=None)
    assert out_no_tail["in_sample_frac"] == 0.0  # ignores mean fit's date entirely

    out_with_tail = metrics.compute(r, fit_train_end=None, tail_train_end=TRAIN_END)
    assert out_with_tail["in_sample_frac"] > 0.5  # ignores mean fit being None entirely


def test_thr_gt_strategies_are_not_fit_dependent():
    # The core correction this investigation surfaced: thr_gt_* thresholds
    # conviction_score (hand-tuned DEFAULT_WEIGHTS), not learned_score, so
    # it must read 0.0 even with fit metadata available and nonzero.
    for name in ("thr_gt_m05", "thr_gt_p00", "thr_gt_p11", "thr_gt_p13"):
        r = _run_result(name, start=WINDOW_START, end=WINDOW_END)
        out = metrics.compute(r, fit_train_end=TRAIN_END, tail_train_end=TRAIN_END)
        assert out["in_sample_frac"] == 0.0, name


def test_model_ranked_strategies_get_zero():
    for name in ("model_ranked_hold63", "model_ranked_top_hold63"):
        r = _run_result(name, start=WINDOW_START, end=WINDOW_END)
        out = metrics.compute(r, fit_train_end=TRAIN_END, tail_train_end=TRAIN_END)
        assert out["in_sample_frac"] == 0.0, name


def test_hand_built_and_spy_strategies_get_zero():
    for name in ("conviction_only", "ten_percent_owner_gated", "spy_buy_and_hold",
                 "all_clusters_hold63", "vol_scaled_conviction"):
        r = _run_result(name, start=WINDOW_START, end=WINDOW_END)
        out = metrics.compute(r, fit_train_end=TRAIN_END, tail_train_end=TRAIN_END)
        assert out["in_sample_frac"] == 0.0, name


def test_missing_fit_metadata_does_not_crash_and_is_zero():
    # fit_signal=False, or the fit attempted and returned None -- a
    # learned_* strategy is normally dropped from that run's strategy list
    # (see backtest.py's chosen_strategies filter), but compute() must
    # still degrade gracefully rather than crash or emit NaN if called.
    r = _run_result("learned_gt_p00", start=WINDOW_START, end=WINDOW_END)
    out = metrics.compute(r)  # fit_train_end / tail_train_end default to None
    assert out["in_sample_frac"] == 0.0
    assert math.isfinite(out["in_sample_frac"])


def test_cost_sweep_suffix_still_matches_base_strategy_name():
    # backtest.py's cost-sensitivity sweep renames e.g. learned_gt_p00 to
    # learned_gt_p00_5bps (dataclasses.replace) before running it.
    r = _run_result("learned_gt_p00_5bps", start=WINDOW_START, end=WINDOW_END)
    out = metrics.compute(r, fit_train_end=TRAIN_END)
    assert out["in_sample_frac"] > 0.5


def test_train_end_before_window_start_clips_to_zero():
    r = _run_result("learned_gt_p00", start="2024-01-01", end="2024-12-31")
    out = metrics.compute(r, fit_train_end=date(2020, 1, 1))
    assert out["in_sample_frac"] == 0.0


def test_train_end_after_window_end_clips_to_one():
    r = _run_result("learned_gt_p00", start="2020-01-01", end="2020-12-31")
    out = metrics.compute(r, fit_train_end=date(2024, 10, 31))
    assert out["in_sample_frac"] == 1.0


def test_empty_equity_curve_early_return_omits_key_not_crash():
    empty_eq = pd.Series([], dtype=float)
    result = RunResult(
        strategy="learned_gt_p00", exit_label="empty_exit",
        equity_curve=empty_eq, trades=[], skips=Counter(),
        n_rebalances=0, exposure_curve=pd.Series([], dtype=float),
        n_tickers_curve=pd.Series([], dtype=float),
    )
    out = metrics.compute(result, fit_train_end=TRAIN_END)
    # Matches the documented early-return policy for the concentration
    # keys (see test_metrics_concentration.py): in_sample_frac is added in
    # the populated path only, so it's simply absent here, not NaN.
    assert "in_sample_frac" not in out


def test_summary_table_threads_fit_metadata_to_every_row():
    learned = _run_result("learned_gt_p00", start=WINDOW_START, end=WINDOW_END)
    thr = _run_result("thr_gt_p11", start=WINDOW_START, end=WINDOW_END)
    df = metrics.summary_table([learned, thr], fit_train_end=TRAIN_END)
    row_learned = df[df["strategy"] == "learned_gt_p00"].iloc[0]
    row_thr = df[df["strategy"] == "thr_gt_p11"].iloc[0]
    assert row_learned["in_sample_frac"] > 0.5
    assert row_thr["in_sample_frac"] == 0.0
