"""Tests for backtest/report.py's in-sample-fraction warning banner and
leaderboard column.

Background: a strategy whose target_fn depends on backtest/signal_fit.py's
fitted learned_score/tail_score can trade back through its own training
window (see backtest/metrics.py's in_sample_frac, tested in
tests/test_in_sample_frac.py). Nothing forced a reader skimming the HTML
report to notice this before trusting a high alpha or t_alpha -- the same
blind spot _concentration_warnings_html was built to close for pnl
concentration (see CONCENTRATION_WARNING_THRESHOLD). These tests exercise
the mirrored _in_sample_warnings_html() banner and the in_sample_frac
column added to both _summary_table_html (the main leaderboard) and
_alpha_beta_table_html (where t_alpha lives, so a t-alpha sort can't miss
it).
"""

from __future__ import annotations

import pandas as pd

from backtest import report


def _summary_row(**overrides) -> dict:
    row = {
        "strategy": "learned_gt_p00", "exit_method": "90d",
        "total_return": 0.5, "cagr": 0.1, "sharpe": 1.0, "sortino": 1.0,
        "max_drawdown": -0.1, "calmar": 1.0, "win_rate": 0.5,
        "avg_lot_return": 0.05, "hit_rate_vs_spy": 0.5, "avg_excess_vs_spy": 0.01,
        "top_ticker_pnl_share": 0.1, "top_ticker": "AAA",
        "n_lots_gt_300pct": 0, "n_lots": 10, "n_natural_exits": 10,
        "n_rebalances": 10, "mean_exposure": 0.5,
        "n_skipped_capacity": 0, "n_skipped_liquidity": 0,
        "alpha_ann": 0.05, "beta": 1.0, "r_squared": 0.5, "t_alpha": 2.061,
        "tracking_error": 0.1, "info_ratio": 0.5,
        "in_sample_frac": 0.0,
    }
    row.update(overrides)
    return row


def test_warning_fires_above_threshold():
    df = pd.DataFrame([_summary_row(in_sample_frac=0.70, strategy="learned_gt_p00")])
    html = report._in_sample_warnings_html(df)
    assert "learned_gt_p00" in html
    assert "70.00%" in html
    assert "in-sample" in html.lower()


def test_warning_does_not_fire_at_exactly_the_threshold():
    # Comparison is strictly-greater-than, mirroring
    # _concentration_warnings_html's `abs(share) <= THRESHOLD: continue`.
    df = pd.DataFrame([_summary_row(
        in_sample_frac=report.IN_SAMPLE_WARNING_THRESHOLD, strategy="borderline",
    )])
    assert report._in_sample_warnings_html(df) == ""


def test_warning_does_not_fire_below_threshold():
    df = pd.DataFrame([_summary_row(in_sample_frac=0.05, strategy="mostly_oos")])
    assert report._in_sample_warnings_html(df) == ""


def test_warning_silent_for_zero_fraction_hand_built_strategies():
    # thr_gt_p11 with its t_alpha=2.061 -- verified NOT fit-dependent, so
    # in_sample_frac is 0.0 and must not trip the banner.
    df = pd.DataFrame([_summary_row(
        in_sample_frac=0.0, strategy="thr_gt_p11", t_alpha=2.061,
    )])
    assert report._in_sample_warnings_html(df) == ""


def test_warning_html_missing_column_returns_empty_not_crash():
    df = pd.DataFrame([_summary_row()]).drop(columns=["in_sample_frac"])
    assert report._in_sample_warnings_html(df) == ""


def test_warning_ignores_nan_fraction_without_crashing():
    df = pd.DataFrame([_summary_row(in_sample_frac=float("nan"))])
    assert report._in_sample_warnings_html(df) == ""


def test_summary_table_html_includes_in_sample_frac_column_and_value():
    df = pd.DataFrame([_summary_row(in_sample_frac=0.707, strategy="learned_gt_p00")])
    html = report._summary_table_html(df)
    assert "In-Sample Frac" in html
    assert "70.70%" in html


def test_alpha_beta_table_html_includes_in_sample_frac_next_to_t_alpha():
    # This is the table a reader sorts by t(alpha) to find the strongest
    # result. in_sample_frac must be visible in the SAME row, not only in
    # a banner above a different table.
    df = pd.DataFrame([_summary_row(
        in_sample_frac=0.707, strategy="learned_gt_p00", t_alpha=2.061,
    )])
    html = report._alpha_beta_table_html(df)
    assert "In-Sample Frac" in html
    assert "t(alpha)" in html
    assert "70.70%" in html


def test_alpha_beta_table_excludes_spy_row():
    df = pd.DataFrame([
        _summary_row(strategy="spy_buy_and_hold", in_sample_frac=0.0),
        _summary_row(strategy="thr_gt_p11", in_sample_frac=0.0, t_alpha=2.061),
    ])
    html = report._alpha_beta_table_html(df)
    assert "spy_buy_and_hold" not in html
    assert "thr_gt_p11" in html
