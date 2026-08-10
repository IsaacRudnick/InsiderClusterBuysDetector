"""Tests for backtest/report.py's REPORT_MAX_CURVES / REPORT_MAX_POINTS_PER_CURVE
downsampling added to fix report.html blowing up to 352MB.

Background: equity_curves_fig, drawdown_fig, and exposure_fig each embed one
full daily time series PER RUN, across every exit method (the exit-method
picker only toggles trace *visibility* client-side -- every trace's data is
still inlined into the page as JSON). A grid run of ~323 runs x ~1507
trading days is ~487k points per chart, x3 charts -- this, not inline
plotly.js (~3MB), is what made the file unopenable. Two independent caps
fix it: _select_top_runs keeps only the top REPORT_MAX_CURVES runs (by
Sharpe, falling back to total_return), and _downsample_uniform /
_downsample_preserve_min cap points per kept line to
REPORT_MAX_POINTS_PER_CURVE.

RunResult is built directly the same way tests/test_in_sample_frac.py and
tests/test_metrics_concentration.py do -- there's no existing test helper
for it.
"""

from __future__ import annotations

from collections import Counter
from datetime import date

import numpy as np
import pandas as pd
import pytest

from backtest import report
from backtest.engine import RunResult, Trade
from backtest.strategies import EXIT_METHODS


def _run(strategy: str, exit_label: str, *, n_days: int = 1000,
        seed: int = 0) -> RunResult:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-02", periods=n_days)
    nav = 100_000.0 * np.cumprod(1 + rng.normal(0.0003, 0.01, size=n_days))
    equity = pd.Series(nav, index=idx)
    exposure = pd.Series(np.clip(rng.normal(0.5, 0.1, size=n_days), 0, 1), index=idx)
    n_tickers = pd.Series(0, index=idx)
    return RunResult(
        strategy=strategy, exit_label=exit_label,
        equity_curve=equity, trades=[], skips=Counter(),
        n_rebalances=0, exposure_curve=exposure,
        n_tickers_curve=n_tickers,
    )


def _trade(*, ticker: str = "ABC", return_pct: float = 0.05, score_at_entry: int = 2,
          exit_reason: str = "expiry", delisted: bool = False) -> Trade:
    """Minimal realized Trade for score_scatter_fig / lot_return_box_fig tests
    -- these two figures only read ticker, entry_date, return_pct,
    score_at_entry, exit_reason, and delisted, but Trade has no defaults for
    the rest, so fill the others with fixed placeholder values the same way
    tests/test_metrics_concentration.py's _trade() does."""
    return Trade(
        strategy="s", exit_label="e", ticker=ticker,
        entry_date=date(2024, 1, 2), exit_date=date(2024, 2, 2),
        shares=10.0, entry_price=10.0, exit_price=10.0,
        cost_basis=100.0, proceeds=100.0, pnl=0.0,
        return_pct=return_pct, score_at_entry=score_at_entry,
        days_held=31, trading_days_held=21, exit_reason=exit_reason,
        delisted=delisted,
    )


def _run_with_trades(strategy: str, exit_label: str, n_lots: int, *,
                     n_days: int = 30) -> RunResult:
    """A run whose only interesting content is n_lots realized (expiry) lots
    -- used for score_scatter_fig / lot_return_box_fig, which don't touch
    equity_curve/exposure_curve at all, so those are kept minimal."""
    idx = pd.bdate_range("2020-01-02", periods=n_days)
    trades = [_trade(ticker=f"T{i}", return_pct=(i % 21 - 10) / 100.0, score_at_entry=i % 5)
              for i in range(n_lots)]
    return RunResult(
        strategy=strategy, exit_label=exit_label,
        equity_curve=pd.Series(100_000.0, index=idx), trades=trades, skips=Counter(),
        n_rebalances=0, exposure_curve=pd.Series(0.0, index=idx),
        n_tickers_curve=pd.Series(0, index=idx),
    )


def _run_with_outlier_trades(strategy: str, exit_label: str, n_core: int, n_outliers: int, *,
                             n_days: int = 30) -> RunResult:
    """A run with a tight cluster of small-return lots (return_pct in
    roughly [-0.2%, 0.2%]) plus n_outliers lots far outside that cluster's
    Tukey fences, each with a DISTINCT magnitude so there's exactly one
    largest-positive and one largest-negative outlier -- used to test that
    lot_return_box_fig computes exact quartiles from the FULL lot set (not
    a downsampled one, see _box_stats) and that _sample_outliers_keep_extremes
    never randomly drops the single most extreme outlier on either side.

    The Tukey rule only classifies the injected extremes as outliers if
    they stay a minority of the sample (roughly: core count > outlier
    count, otherwise the "outliers" are numerous enough to set the
    quartiles themselves) -- so the actual core count used is bumped up to
    3x n_outliers when a caller's n_core is too small for that, regardless
    of what n_core it passed.
    """
    n_core = max(n_core, 3 * n_outliers + 50)
    idx = pd.bdate_range("2020-01-02", periods=n_days)
    core = [_trade(ticker=f"C{i}", return_pct=(i % 5 - 2) / 1000.0, score_at_entry=i % 5)
            for i in range(n_core)]
    outliers = [
        _trade(ticker=f"O{i}",
              return_pct=10.0 * (1 if i % 2 == 0 else -1) * (1 + i / 1000.0),
              score_at_entry=i % 5)
        for i in range(n_outliers)
    ]
    trades = core + outliers
    return RunResult(
        strategy=strategy, exit_label=exit_label,
        equity_curve=pd.Series(100_000.0, index=idx), trades=trades, skips=Counter(),
        n_rebalances=0, exposure_curve=pd.Series(0.0, index=idx),
        n_tickers_curve=pd.Series(0, index=idx),
    )


# ---------------------------------------------------------------------------
# _downsample_uniform
# ---------------------------------------------------------------------------
def test_downsample_uniform_returns_unchanged_when_series_shorter_than_cap():
    s = pd.Series([1.0, 2.0, 3.0], index=pd.bdate_range("2020-01-01", periods=3))
    out = report._downsample_uniform(s, 400)
    assert out is s  # short-circuited, not just equal


def test_downsample_uniform_returns_unchanged_when_series_equals_cap():
    s = pd.Series(range(400), index=pd.bdate_range("2020-01-01", periods=400), dtype=float)
    out = report._downsample_uniform(s, 400)
    assert out is s


def test_downsample_uniform_retains_final_point_exactly():
    idx = pd.bdate_range("2020-01-01", periods=1507)
    s = pd.Series(np.random.default_rng(1).normal(size=1507), index=idx)
    out = report._downsample_uniform(s, 400)
    assert out.index[-1] == s.index[-1]
    assert out.iloc[-1] == s.iloc[-1]


def test_downsample_uniform_retains_first_point():
    idx = pd.bdate_range("2020-01-01", periods=1507)
    s = pd.Series(np.random.default_rng(1).normal(size=1507), index=idx)
    out = report._downsample_uniform(s, 400)
    assert out.index[0] == s.index[0]
    assert out.iloc[0] == s.iloc[0]


def test_downsample_uniform_caps_point_count():
    idx = pd.bdate_range("2020-01-01", periods=1507)
    s = pd.Series(np.random.default_rng(1).normal(size=1507), index=idx)
    out = report._downsample_uniform(s, 400)
    # Stride sampling + a forced final point can land at most one point over
    # ceil(n/max_points) buckets -- nowhere near the original 1507.
    assert len(out) <= 401
    assert len(out) < len(s)


def test_downsample_uniform_values_are_a_subset_of_original():
    idx = pd.bdate_range("2020-01-01", periods=1507)
    s = pd.Series(np.random.default_rng(2).normal(size=1507), index=idx)
    out = report._downsample_uniform(s, 400)
    assert set(out.index).issubset(set(s.index))
    for ts, val in out.items():
        assert val == s.loc[ts]


# ---------------------------------------------------------------------------
# _downsample_preserve_min (used for the drawdown chart)
# ---------------------------------------------------------------------------
def test_downsample_preserve_min_returns_unchanged_when_short():
    s = pd.Series([-0.1, -0.5, -0.02], index=pd.bdate_range("2020-01-01", periods=3))
    out = report._downsample_preserve_min(s, 400)
    assert out is s


def test_downsample_preserve_min_retains_final_point_exactly():
    idx = pd.bdate_range("2020-01-01", periods=1507)
    s = pd.Series(np.random.default_rng(3).normal(size=1507), index=idx)
    out = report._downsample_preserve_min(s, 400)
    assert out.index[-1] == s.index[-1]
    assert out.iloc[-1] == s.iloc[-1]


def test_downsample_preserve_min_retains_global_minimum_off_a_bucket_boundary():
    # The worst drawdown day sits mid-bucket, nowhere near a stride
    # boundary -- a plain uniform stride sample would very likely skip it.
    idx = pd.bdate_range("2020-01-01", periods=1507)
    values = np.random.default_rng(4).normal(0, 0.01, size=1507)
    worst_i = 733  # deliberately not a multiple of any small stride
    values[worst_i] = -99.0
    s = pd.Series(values, index=idx)
    out = report._downsample_preserve_min(s, 400)
    assert s.index[worst_i] in out.index
    assert out.loc[s.index[worst_i]] == -99.0
    assert out.min() == -99.0 == s.min()


# ---------------------------------------------------------------------------
# _select_top_runs
# ---------------------------------------------------------------------------
def _summary_df(rows):
    return pd.DataFrame(rows)


def test_select_top_runs_returns_input_unchanged_when_under_cap():
    results_by_exit = {"30d": [_run("s1", "30d")], "90d": [_run("s2", "90d")]}
    df = _summary_df([
        {"strategy": "s1", "exit_method": "30d", "sharpe": 0.1, "total_return": 0.1},
        {"strategy": "s2", "exit_method": "90d", "sharpe": 0.2, "total_return": 0.2},
    ])
    out, n_plotted, n_total = report._select_top_runs(results_by_exit, df, max_curves=25)
    assert out is results_by_exit
    assert n_plotted == n_total == 2


def test_select_top_runs_keeps_highest_sharpe_runs():
    lo = _run("lo", "30d")
    mid = _run("mid", "90d")
    hi = _run("hi", "30d")
    results_by_exit = {"30d": [lo, hi], "90d": [mid]}
    df = _summary_df([
        {"strategy": "lo", "exit_method": "30d", "sharpe": 0.1, "total_return": 0.1},
        {"strategy": "mid", "exit_method": "90d", "sharpe": 0.5, "total_return": 0.5},
        {"strategy": "hi", "exit_method": "30d", "sharpe": 0.9, "total_return": 0.9},
    ])
    out, n_plotted, n_total = report._select_top_runs(results_by_exit, df, max_curves=2)
    assert n_plotted == 2
    assert n_total == 3
    kept_strategies = {r.strategy for runs in out.values() for r in runs}
    assert kept_strategies == {"hi", "mid"}
    assert "lo" not in kept_strategies


def test_select_top_runs_falls_back_to_total_return_when_sharpe_nan():
    lo = _run("lo", "30d")
    hi = _run("hi", "30d")
    results_by_exit = {"30d": [lo, hi]}
    df = _summary_df([
        {"strategy": "lo", "exit_method": "30d", "sharpe": float("nan"), "total_return": -0.5},
        {"strategy": "hi", "exit_method": "30d", "sharpe": float("nan"), "total_return": 0.9},
    ])
    out, n_plotted, n_total = report._select_top_runs(results_by_exit, df, max_curves=1)
    kept_strategies = {r.strategy for runs in out.values() for r in runs}
    assert kept_strategies == {"hi"}


def test_select_top_runs_does_not_crash_on_missing_columns():
    r1 = _run("s1", "30d")
    results_by_exit = {"30d": [r1]}
    df = _summary_df([{"strategy": "s1", "exit_method": "30d"}])
    out, n_plotted, n_total = report._select_top_runs(results_by_exit, df, max_curves=0)
    # max_curves=0 with 1 run present -> filtered to nothing, no crash.
    assert n_plotted == 0
    assert all(len(v) == 0 for v in out.values())


# ---------------------------------------------------------------------------
# _downsampling_note_html
# ---------------------------------------------------------------------------
def test_downsampling_note_says_all_runs_shown_when_not_filtered():
    html = report._downsampling_note_html(5, 5)
    assert "All 5 runs" in html
    assert "top" not in html.lower()


def test_downsampling_note_mentions_top_n_of_m_when_filtered():
    html = report._downsampling_note_html(25, 323)
    assert "top 25 of 323 runs" in html
    assert "SPY" in html


def test_downsampling_note_omits_spy_mention_when_include_spy_false():
    html = report._downsampling_note_html(25, 323, include_spy=False)
    assert "SPY" not in html


def test_downsampling_note_mentions_worst_point_for_drawdown():
    html = report._downsampling_note_html(25, 323, drawdown=True)
    assert "worst" in html.lower()


# ---------------------------------------------------------------------------
# Integration: the actual figure functions honor SPY-always-included and
# final-point-exactness end to end.
# ---------------------------------------------------------------------------
LABELS = [em.label for em in EXIT_METHODS]


def _spy_run(n_days: int = 1000) -> RunResult:
    idx = pd.bdate_range("2020-01-02", periods=n_days)
    nav = 100_000.0 * np.cumprod(1 + np.random.default_rng(9).normal(0.0002, 0.008, size=n_days))
    return RunResult(
        strategy="spy_buy_and_hold", exit_label="n/a",
        equity_curve=pd.Series(nav, index=idx), trades=[], skips=Counter(),
        n_rebalances=0, exposure_curve=pd.Series(0.0, index=idx),
        n_tickers_curve=pd.Series(0, index=idx),
    )


def test_equity_curves_fig_always_shows_spy_even_when_run_list_is_filtered_empty():
    # Simulates the post-top-N-filter state where every non-SPY run got
    # dropped: results_by_exit still has the right label keys but empty
    # lists. SPY must still render since it's a separate parameter.
    empty_results = {label: [] for label in LABELS}
    fig = report.equity_curves_fig(empty_results, _spy_run(), strategy_order=[])
    names = {t.name for t in fig.data}
    assert names == {"spy_buy_and_hold"}


def test_drawdown_fig_always_shows_spy_even_when_run_list_is_filtered_empty():
    empty_results = {label: [] for label in LABELS}
    fig = report.drawdown_fig(empty_results, _spy_run(), strategy_order=[])
    names = {t.name for t in fig.data}
    assert names == {"spy_buy_and_hold"}


def test_equity_curves_fig_final_plotted_value_matches_full_resolution_series():
    r = _run("s1", LABELS[0], n_days=1000)
    results_by_exit = {label: [] for label in LABELS}
    results_by_exit[LABELS[0]] = [r]
    fig = report.equity_curves_fig(results_by_exit, _spy_run(), strategy_order=["s1"])
    trace = next(t for t in fig.data if t.name == "s1")
    assert float(trace.y[-1]) == float(r.equity_curve.iloc[-1])
    assert len(trace.y) <= report.REPORT_MAX_POINTS_PER_CURVE + 1


def test_drawdown_fig_worst_point_matches_full_resolution_series():
    idx = pd.bdate_range("2020-01-01", periods=1000)
    values = 100_000.0 * (1 + np.random.default_rng(5).normal(0, 0.001, size=1000)).cumprod()
    values[456] = values[:456].min() * 0.3  # inject a deep, off-boundary drawdown
    equity = pd.Series(values, index=idx)
    r = RunResult(
        strategy="s1", exit_label=LABELS[0], equity_curve=equity, trades=[],
        skips=Counter(), n_rebalances=0,
        exposure_curve=pd.Series(0.0, index=idx), n_tickers_curve=pd.Series(0, index=idx),
    )
    results_by_exit = {label: [] for label in LABELS}
    results_by_exit[LABELS[0]] = [r]
    fig = report.drawdown_fig(results_by_exit, _spy_run(), strategy_order=["s1"])
    trace = next(t for t in fig.data if t.name == "s1")

    peak = equity.cummax()
    full_dd_pct = float(((equity / peak) - 1.0).min() * 100)
    assert min(trace.y) == pytest.approx(full_dd_pct)


def test_exposure_fig_final_plotted_value_matches_full_resolution_series():
    r = _run("s1", LABELS[0], n_days=1000)
    results_by_exit = {label: [] for label in LABELS}
    results_by_exit[LABELS[0]] = [r]
    fig = report.exposure_fig(results_by_exit, strategy_order=["s1"])
    trace = next(t for t in fig.data if t.name == "s1")
    assert float(trace.y[-1]) == pytest.approx(float(r.exposure_curve.iloc[-1] * 100))


# ---------------------------------------------------------------------------
# SCATTER_POINT_BUDGET / _per_trace_point_cap / _lot_trace_count
#
# Background: score_scatter_fig and lot_return_box_fig used to cap points at
# a flat MAX_SCATTER_POINTS=20,000 PER TRACE, and draw one trace per
# (strategy, exit method). A 53-strategy x 7-exit-method grid put up to 371
# traces in a single figure -- an effective ceiling of ~7.4M points, not
# 20,000 -- which is what actually produced a 244.88 MB score-vs-return
# figure and a 67.84 MB per-lot-returns figure (measured against the real
# out/backtest_20260809_100107 output), 99% of a 325 MB report.html.
# SCATTER_POINT_BUDGET fixes this by capping the WHOLE FIGURE's point count,
# split evenly across however many traces it ends up drawing.
# ---------------------------------------------------------------------------
def test_per_trace_point_cap_splits_budget_evenly():
    assert report._per_trace_point_cap(4, budget=100) == 25
    assert report._per_trace_point_cap(3, budget=100) == 33  # floor division


def test_per_trace_point_cap_returns_full_budget_when_no_traces():
    assert report._per_trace_point_cap(0, budget=100) == 100


def test_per_trace_point_cap_floors_at_one_point_when_traces_exceed_budget():
    assert report._per_trace_point_cap(500, budget=100) == 1


def test_lot_trace_count_counts_one_per_nonempty_strategy_exit_combo():
    results_by_exit = {label: [] for label in LABELS}
    results_by_exit[LABELS[0]] = [
        _run_with_trades("s1", LABELS[0], n_lots=5),
        _run_with_trades("s2", LABELS[0], n_lots=5),
    ]
    results_by_exit[LABELS[1]] = [_run_with_trades("s1", LABELS[1], n_lots=5)]
    assert report._lot_trace_count(results_by_exit) == 3


def test_lot_trace_count_excludes_runs_with_no_realized_lots():
    # Every trade is a stop_loss exit -- score_scatter_fig / lot_return_box_fig
    # only count exit_reason in (expiry, trim), not delisted, so this run
    # contributes zero traces despite having trades.
    r = RunResult(
        strategy="s1", exit_label=LABELS[0],
        equity_curve=pd.Series([100_000.0]), trades=[_trade(exit_reason="stop_loss")],
        skips=Counter(), n_rebalances=0,
        exposure_curve=pd.Series([0.0]), n_tickers_curve=pd.Series([0]),
    )
    results_by_exit = {label: [] for label in LABELS}
    results_by_exit[LABELS[0]] = [r]
    assert report._lot_trace_count(results_by_exit) == 0


def test_lot_trace_count_excludes_delisted_lots():
    r = RunResult(
        strategy="s1", exit_label=LABELS[0],
        equity_curve=pd.Series([100_000.0]),
        trades=[_trade(exit_reason="expiry", delisted=True)],
        skips=Counter(), n_rebalances=0,
        exposure_curve=pd.Series([0.0]), n_tickers_curve=pd.Series([0]),
    )
    results_by_exit = {label: [] for label in LABELS}
    results_by_exit[LABELS[0]] = [r]
    assert report._lot_trace_count(results_by_exit) == 0


# ---------------------------------------------------------------------------
# score_scatter_fig / lot_return_box_fig -- global budget behavior
# ---------------------------------------------------------------------------
def _results_with_n_strategies(n_strategies: int, lots_per_strategy: int,
                               label: str = LABELS[0]) -> dict[str, list[RunResult]]:
    results_by_exit = {lbl: [] for lbl in LABELS}
    results_by_exit[label] = [
        _run_with_trades(f"s{i}", label, n_lots=lots_per_strategy)
        for i in range(n_strategies)
    ]
    return results_by_exit


def _results_with_n_outlier_strategies(n_strategies: int, n_core: int, n_outliers: int,
                                       label: str = LABELS[0]) -> dict[str, list[RunResult]]:
    results_by_exit = {lbl: [] for lbl in LABELS}
    results_by_exit[label] = [
        _run_with_outlier_trades(f"s{i}", label, n_core=n_core, n_outliers=n_outliers)
        for i in range(n_strategies)
    ]
    return results_by_exit


def test_score_scatter_fig_total_points_never_exceeds_budget_even_with_many_traces():
    # 40 strategies x 3,000 lots each = 120,000 raw points, far more than
    # SCATTER_POINT_BUDGET on its own -- this is the exact shape of bug that
    # made the real report 249.76 MB (many traces, each already under any
    # sane single-trace cap, but the trace COUNT was what blew up the total).
    results_by_exit = _results_with_n_strategies(40, 3_000)
    fig = report.score_scatter_fig(results_by_exit, strategy_order=[f"s{i}" for i in range(40)])
    total_points = sum(len(t.x) for t in fig.data if t.mode == "markers")
    assert total_points <= report.SCATTER_POINT_BUDGET


def test_lot_return_box_fig_total_points_never_exceeds_budget_even_with_many_traces():
    # lot_return_box_fig no longer embeds every raw lot -- the box itself
    # (q1/median/q3/fences) is precomputed exactly server-side, so only the
    # sampled OUTLIER markers count against SCATTER_POINT_BUDGET. Use data
    # with a genuine, large outlier tail (not _run_with_trades' uniform
    # cyclic pattern, which has zero Tukey outliers) so this test actually
    # exercises the budget cap rather than passing on an empty `y`.
    # per_trace_cap at 40 traces is SCATTER_POINT_BUDGET // 40 = 500, so
    # 1,000 genuine outliers per strategy is enough to force sampling
    # without paying for an enormous core cluster on every one of 40 runs.
    results_by_exit = _results_with_n_outlier_strategies(40, n_core=100, n_outliers=1_000)
    fig = report.lot_return_box_fig(results_by_exit, strategy_order=[f"s{i}" for i in range(40)])
    total_points = sum(len(t.y) for t in fig.data)
    assert total_points <= report.SCATTER_POINT_BUDGET
    # Sanity: this scenario really does have outliers to sample from, so a
    # passing assertion above means the cap actually bound something, not
    # that there was nothing to sample.
    assert any(len(t.y) > 0 for t in fig.data)


def test_score_scatter_fig_does_not_sample_when_under_budget():
    # 3 strategies x 10 lots = 30 total points, nowhere near the budget --
    # every point should survive, and the trace name should carry no
    # "lots shown" disclosure since nothing was sampled.
    results_by_exit = _results_with_n_strategies(3, 10)
    fig = report.score_scatter_fig(results_by_exit, strategy_order=["s0", "s1", "s2"])
    markers = [t for t in fig.data if t.mode == "markers"]
    assert len(markers) == 3
    for t in markers:
        assert len(t.x) == 10
        assert "lots shown" not in t.name


def test_lot_return_box_fig_does_not_sample_when_under_budget():
    # A handful of genuine outliers, nowhere near per_trace_cap -- every
    # outlier should survive, and the trace name should carry no "lots
    # shown" disclosure since nothing was sampled.
    results_by_exit = _results_with_n_outlier_strategies(3, n_core=100, n_outliers=5)
    fig = report.lot_return_box_fig(results_by_exit, strategy_order=["s0", "s1", "s2"])
    assert len(fig.data) == 3
    for t in fig.data:
        assert len(t.y) == 5
        assert "lots shown" not in t.name


def test_score_scatter_fig_discloses_sampling_in_trace_name():
    # 2 strategies sharing the budget -> per-trace cap = budget // 2, each
    # strategy has far more lots than that, so both should be sampled and
    # both trace names should disclose it accurately.
    results_by_exit = _results_with_n_strategies(2, report.SCATTER_POINT_BUDGET)
    fig = report.score_scatter_fig(results_by_exit, strategy_order=["s0", "s1"])
    markers = [t for t in fig.data if t.mode == "markers"]
    expected_cap = report.SCATTER_POINT_BUDGET // 2
    for t in markers:
        assert len(t.x) == expected_cap
        assert f"{expected_cap:,} of {report.SCATTER_POINT_BUDGET:,} lots shown" in t.name


def test_lot_return_box_fig_discloses_sampling_in_trace_name():
    # 2 strategies sharing the budget -> per-trace outlier cap =
    # budget // 2; each strategy has far more outliers than that, so both
    # should be sampled and both trace names should disclose it accurately
    # using the "outlier lots" wording (not the old "(N of M lots shown)",
    # which would wrongly imply the box shape itself was sampled).
    expected_cap = report.SCATTER_POINT_BUDGET // 2
    results_by_exit = _results_with_n_outlier_strategies(
        2, n_core=100, n_outliers=expected_cap + 1_000,
    )
    fig = report.lot_return_box_fig(results_by_exit, strategy_order=["s0", "s1"])
    for t in fig.data:
        assert len(t.y) == expected_cap
        assert f"{expected_cap:,} of {expected_cap + 1_000:,} outlier lots shown" in t.name


def test_score_scatter_fig_sampling_is_deterministic_across_calls():
    results_by_exit = _results_with_n_strategies(2, report.SCATTER_POINT_BUDGET)
    fig1 = report.score_scatter_fig(results_by_exit, strategy_order=["s0", "s1"])
    fig2 = report.score_scatter_fig(results_by_exit, strategy_order=["s0", "s1"])
    m1 = [t for t in fig1.data if t.mode == "markers"]
    m2 = [t for t in fig2.data if t.mode == "markers"]
    for t1, t2 in zip(m1, m2):
        assert list(t1.x) == list(t2.x)
        assert list(t1.y) == list(t2.y)


def test_lot_return_box_fig_sampling_is_deterministic_across_calls():
    results_by_exit = _results_with_n_outlier_strategies(
        2, n_core=100, n_outliers=report.SCATTER_POINT_BUDGET // 2 + 1_000,
    )
    fig1 = report.lot_return_box_fig(results_by_exit, strategy_order=["s0", "s1"])
    fig2 = report.lot_return_box_fig(results_by_exit, strategy_order=["s0", "s1"])
    for t1, t2 in zip(fig1.data, fig2.data):
        assert list(t1.y) == list(t2.y)
        assert t1.q1 == t2.q1
        assert t1.median == t2.median
        assert t1.q3 == t2.q3


# ---------------------------------------------------------------------------
# _box_stats / _sample_outliers_keep_extremes -- the exact-box-statistics fix
#
# Background: sampling the y-values fed to go.Box changes the quartiles it
# DRAWS, not merely what's hidden -- a strategy's most extreme moonshot or
# blow-up lot is exactly what a small random sample is least likely to
# retain, and this project has repeatedly found that those extreme lots ARE
# the P&L (see MEMORY: split-artifacts-decide-the-leaderboard,
# sub-dollar-lots-are-the-pnl). The fix computes q1/median/q3/fences/mean/sd
# from the FULL lot set server-side (_box_stats) and only downsamples the
# individual outlier markers, always keeping the largest-magnitude outlier
# on each side (_sample_outliers_keep_extremes).
# ---------------------------------------------------------------------------
def test_box_stats_quartiles_exact_against_numpy_percentile_of_full_data():
    rng = np.random.default_rng(11)
    rets = list(rng.normal(0, 5, 2_000))
    stats = report._box_stats(rets)
    arr = np.asarray(rets)
    assert stats["q1"] == pytest.approx(float(np.percentile(arr, 25)))
    assert stats["median"] == pytest.approx(float(np.percentile(arr, 50)))
    assert stats["q3"] == pytest.approx(float(np.percentile(arr, 75)))


def test_box_stats_mean_and_sd_exact_against_full_data():
    rng = np.random.default_rng(12)
    rets = list(rng.normal(2.0, 3.0, 2_000))
    stats = report._box_stats(rets)
    arr = np.asarray(rets)
    assert stats["mean"] == pytest.approx(float(arr.mean()))
    assert stats["sd"] == pytest.approx(float(arr.std()))


def test_box_stats_outliers_are_exactly_the_points_beyond_the_fences():
    rets = [0.0, 0.1, -0.1, 0.2, -0.2, 100.0, -100.0]
    stats = report._box_stats(rets)
    assert 100.0 in stats["outliers"]
    assert -100.0 in stats["outliers"]
    for v in (0.0, 0.1, -0.1, 0.2, -0.2):
        assert v not in stats["outliers"]


def test_box_stats_no_sampling_error_regardless_of_dataset_size():
    # The whole point of the fix: quartiles computed from 50,000 lots must
    # be exact, not an estimate -- there is no per_trace_cap-sized sample
    # anywhere in _box_stats' own computation.
    rng = np.random.default_rng(13)
    small = list(rng.normal(0, 1, 50))
    big = small + list(rng.normal(0, 1, 49_950))
    stats_big = report._box_stats(big)
    arr = np.asarray(big)
    assert stats_big["q1"] == pytest.approx(float(np.percentile(arr, 25)))
    assert stats_big["q3"] == pytest.approx(float(np.percentile(arr, 75)))


def test_sample_outliers_keep_extremes_returns_unchanged_when_under_cap():
    outliers = [5.0, -3.0, 8.0, -1.0]
    assert report._sample_outliers_keep_extremes(outliers, cap=10) == outliers


def test_sample_outliers_keep_extremes_always_retains_max_and_min():
    rng = np.random.default_rng(14)
    outliers = list(rng.uniform(-1000, 1000, 5_000))
    outliers[123] = 999_999.0   # unique global max
    outliers[456] = -999_999.0  # unique global min
    sampled = report._sample_outliers_keep_extremes(outliers, cap=100)
    assert len(sampled) == 100
    assert 999_999.0 in sampled
    assert -999_999.0 in sampled


def test_sample_outliers_keep_extremes_caps_at_requested_size():
    outliers = list(range(10_000))
    sampled = report._sample_outliers_keep_extremes([float(v) for v in outliers], cap=250)
    assert len(sampled) == 250


def test_sample_outliers_keep_extremes_deterministic_across_calls():
    rng = np.random.default_rng(15)
    outliers = list(rng.uniform(-500, 500, 3_000))
    s1 = report._sample_outliers_keep_extremes(outliers, cap=200)
    s2 = report._sample_outliers_keep_extremes(outliers, cap=200)
    assert s1 == s2


def test_sample_outliers_keep_extremes_returns_empty_for_zero_cap():
    assert report._sample_outliers_keep_extremes([1.0, 2.0, 3.0], cap=0) == []


def test_sample_outliers_keep_extremes_cap_one_keeps_furthest_from_median():
    outliers = [1.0, 2.0, 3.0, -100.0]
    sampled = report._sample_outliers_keep_extremes(outliers, cap=1)
    assert sampled == [-100.0]


# ---------------------------------------------------------------------------
# lot_return_box_fig -- integration: exact box against full data, extremes
# always plotted, even when outlier markers themselves are downsampled.
# ---------------------------------------------------------------------------
def test_lot_return_box_fig_quartiles_exact_against_full_lot_set_even_when_sampled():
    n_outliers = report.SCATTER_POINT_BUDGET  # forces heavy outlier-marker sampling
    r = _run_with_outlier_trades("s1", LABELS[0], n_core=200, n_outliers=n_outliers)
    results_by_exit = {label: [] for label in LABELS}
    results_by_exit[LABELS[0]] = [r]
    fig = report.lot_return_box_fig(results_by_exit, strategy_order=["s1"])
    trace = fig.data[0]

    full_rets = [t.return_pct * 100 for t in r.trades
                 if t.exit_reason in ("expiry", "trim") and not t.delisted]
    arr = np.asarray(full_rets)
    assert trace.q1[0] == pytest.approx(float(np.percentile(arr, 25)))
    assert trace.median[0] == pytest.approx(float(np.percentile(arr, 50)))
    assert trace.q3[0] == pytest.approx(float(np.percentile(arr, 75)))
    # The outlier markers themselves ARE sampled at this scale (that's the
    # part still allowed to be approximate) -- but the box shape above is
    # exact against every one of the 200 + n_outliers lots, not the sample.
    assert len(trace.y) < len(full_rets)


def test_lot_return_box_fig_largest_magnitude_outlier_survives_marker_sampling():
    n_outliers = report.SCATTER_POINT_BUDGET
    r = _run_with_outlier_trades("s1", LABELS[0], n_core=200, n_outliers=n_outliers)
    results_by_exit = {label: [] for label in LABELS}
    results_by_exit[LABELS[0]] = [r]
    fig = report.lot_return_box_fig(results_by_exit, strategy_order=["s1"])
    trace = fig.data[0]

    full_rets = [t.return_pct * 100 for t in r.trades
                 if t.exit_reason in ("expiry", "trim") and not t.delisted]
    assert max(full_rets) in trace.y
    assert min(full_rets) in trace.y


def test_lot_return_box_fig_boxpoints_set_to_outliers_only():
    # Under go.Box's q1/median/q3 signature, boxpoints defaults to "all",
    # not "outliers" -- if this weren't set explicitly, every downsampled
    # marker in `y` (which is already outliers-only) would still render
    # correctly today, but a future change that widened `y` to include
    # non-outlier points would silently start drawing them all. Pin the
    # explicit setting so that stays impossible.
    results_by_exit = _results_with_n_outlier_strategies(1, n_core=50, n_outliers=5)
    fig = report.lot_return_box_fig(results_by_exit, strategy_order=["s0"])
    assert fig.data[0].boxpoints == "outliers"


# ---------------------------------------------------------------------------
# _lot_downsampling_note_html
# ---------------------------------------------------------------------------
def test_lot_downsampling_note_says_all_runs_eligible_when_not_filtered():
    html = report._lot_downsampling_note_html(5, 5, per_trace_cap=800)
    assert "All 5 runs" in html
    assert "top" not in html.lower()


def test_lot_downsampling_note_mentions_top_n_of_m_runs_when_filtered():
    html = report._lot_downsampling_note_html(25, 371, per_trace_cap=800)
    assert "top 25 of 371 runs" in html


def test_lot_downsampling_note_mentions_per_trace_cap_and_source_files():
    html = report._lot_downsampling_note_html(25, 371, per_trace_cap=800)
    assert "800" in html
    assert "trades_*.csv" in html


def test_lot_downsampling_note_default_wording_says_lots_shown_not_outlier_lots():
    # The default (exact_stats=False) wording is what score_scatter_fig
    # still uses, unchanged -- every plotted point there really is a random
    # sample of the full lot set, so "(N of M lots shown)" is accurate for
    # it. Pin the plain default so a future edit can't accidentally give
    # score_scatter_fig the box chart's "exact" framing.
    html = report._lot_downsampling_note_html(25, 371, per_trace_cap=800)
    assert '"(N of M lots shown)"' in html
    assert "exactly" not in html.lower()


def test_lot_downsampling_note_exact_stats_says_box_is_exact_not_sampled():
    html = report._lot_downsampling_note_html(25, 371, per_trace_cap=800, exact_stats=True)
    assert "exactly" in html.lower()
    assert "no sampling" in html.lower()
    assert "outlier lots shown" in html
    # The old blanket framing ("lots are further randomly sampled ... to at
    # most N points per trace") must not survive verbatim -- it implied the
    # whole box was estimated from a sample, which is no longer true.
    assert "lots are further" not in html


def test_lot_downsampling_note_exact_stats_still_mentions_cap_and_source_files():
    html = report._lot_downsampling_note_html(25, 371, per_trace_cap=800, exact_stats=True)
    assert "800" in html
    assert "trades_*.csv" in html


def test_lot_downsampling_note_exact_stats_says_all_eligible_when_not_filtered():
    html = report._lot_downsampling_note_html(5, 5, per_trace_cap=800, exact_stats=True)
    assert "All 5 runs" in html


# ---------------------------------------------------------------------------
# render_html wiring: score_scatter_fig / lot_return_box_fig must receive
# the SAME top-N-filtered results_by_exit as equity/drawdown/exposure, not
# the full grid. Before this task, render_html deliberately passed the full
# results_by_exit to these two figures (see the old comment this replaced);
# that is the exact bug this test guards against regressing to.
# ---------------------------------------------------------------------------
def test_render_html_filters_scatter_and_violin_figs_to_top_n_runs(monkeypatch):
    n_runs = report.REPORT_MAX_CURVES + 5
    label = LABELS[0]
    results_by_exit = {lbl: [] for lbl in LABELS}
    rows = []
    for i in range(n_runs):
        strategy = f"strat{i}"
        results_by_exit[label].append(_run_with_trades(strategy, label, n_lots=5))
        rows.append({
            "strategy": strategy, "exit_method": label,
            "sharpe": float(i), "total_return": float(i),
            # metric_bar_fig indexes these columns unconditionally (unlike
            # the optional-column tables elsewhere in report.py), so a
            # render_html smoke test needs them present regardless of what
            # this test itself cares about.
            "cagr": float(i) / 10, "max_drawdown": -0.1, "win_rate": 0.5,
        })
    summary_df = pd.DataFrame(rows)

    captured: dict[str, int] = {}
    real_scatter = report.score_scatter_fig
    real_box = report.lot_return_box_fig

    def spy_scatter(results_by_exit_arg, strategy_order):
        captured["scatter_n_runs"] = sum(len(v) for v in results_by_exit_arg.values())
        return real_scatter(results_by_exit_arg, strategy_order)

    def spy_box(results_by_exit_arg, strategy_order):
        captured["box_n_runs"] = sum(len(v) for v in results_by_exit_arg.values())
        return real_box(results_by_exit_arg, strategy_order)

    monkeypatch.setattr(report, "score_scatter_fig", spy_scatter)
    monkeypatch.setattr(report, "lot_return_box_fig", spy_box)

    report.render_html(
        summary_df=summary_df,
        results_by_exit=results_by_exit,
        spy_result=_spy_run(),
        strategy_order=[f"strat{i}" for i in range(n_runs)],
        config={},
        offline=False,
        fit_summary=None,
    )

    assert captured["scatter_n_runs"] == report.REPORT_MAX_CURVES
    assert captured["box_n_runs"] == report.REPORT_MAX_CURVES


def test_render_html_gives_violins_section_the_exact_stats_note_not_scatter():
    # render_html must pass exact_stats=True only for the per-lot box chart
    # note (lot_return_box_fig's box is now exact against the full lot set)
    # and leave score_scatter_fig's note in its original "(N of M lots
    # shown)" wording (every plotted scatter point really is a sample).
    label = LABELS[0]
    results_by_exit = {lbl: [] for lbl in LABELS}
    results_by_exit[label] = [_run_with_trades("strat0", label, n_lots=5)]
    summary_df = pd.DataFrame([{
        "strategy": "strat0", "exit_method": label,
        "sharpe": 1.0, "total_return": 1.0,
        "cagr": 0.1, "max_drawdown": -0.1, "win_rate": 0.5,
    }])
    html = report.render_html(
        summary_df=summary_df,
        results_by_exit=results_by_exit,
        spy_result=_spy_run(),
        strategy_order=["strat0"],
        config={},
        offline=False,
        fit_summary=None,
    )
    violins_section = html.split('<section id="violins">')[1].split("</section>")[0]
    scatter_section = html.split('<section id="scatter">')[1].split("</section>")[0]
    assert "no sampling" in violins_section.lower()
    assert "no sampling" not in scatter_section.lower()
    assert '"(N of M lots shown)"' in scatter_section
