"""Tests for backtest/metrics.py's pnl-concentration keys (compute()).

Background: a strategy topped the leaderboard at +161% total return with 83%
of its profit coming from one lot (a fake, unadjusted reverse-split return),
and big_stake_increase showed +80k P&L that was 113% attributable to a
single lot -- remove that lot and the strategy loses money. Nothing in the
old summary output could have caught either case. These tests exercise the
new top_lot_pnl_share / top5_lot_pnl_share / top_ticker_pnl_share /
top_ticker / n_lots_gt_300pct / pnl_share_gt_300pct keys added to
metrics.compute()'s returned dict.

RunResult/Trade are built directly (there is no existing test helper for
this -- other tests build them indirectly via engine.run_strategy against
fake PriceUniverse/DailyStateBuilder objects, which is unnecessary
machinery here since compute() only reads result.equity_curve,
result.trades, and result.exposure_curve).
"""

from __future__ import annotations

import math
from collections import Counter
from datetime import date

import pandas as pd
import pytest

from backtest import metrics
from backtest.engine import RunResult, Trade


def _trade(*, pnl: float, return_pct: float, ticker: str = "ABC",
           exit_reason: str = "expiry", delisted: bool = False,
           entry_date: date = date(2024, 1, 2),
           exit_date: date = date(2024, 2, 2)) -> Trade:
    return Trade(
        strategy="test_strat", exit_label="test_exit", ticker=ticker,
        entry_date=entry_date, exit_date=exit_date,
        shares=100.0, entry_price=10.0, exit_price=10.0 + pnl / 100.0,
        cost_basis=1000.0, proceeds=1000.0 + pnl, pnl=pnl,
        return_pct=return_pct, score_at_entry=5,
        days_held=31, trading_days_held=21, exit_reason=exit_reason,
        delisted=delisted,
    )


def _run_result(trades: list[Trade], *, n_days: int = 30) -> RunResult:
    idx = pd.date_range("2024-01-02", periods=n_days, freq="D")
    # Flat, always-positive equity curve -- these tests care about the
    # trade-level concentration keys, not the equity-curve-derived ones,
    # so keep it simple and non-degenerate (eq.iloc[0] != 0).
    equity = pd.Series([100_000.0] * n_days, index=idx)
    exposure = pd.Series([0.5] * n_days, index=idx)
    n_tickers = pd.Series([1] * n_days, index=idx)
    return RunResult(
        strategy="test_strat", exit_label="test_exit",
        equity_curve=equity, trades=trades, skips=Counter(),
        n_rebalances=len(trades), exposure_curve=exposure,
        n_tickers_curve=n_tickers,
    )


def test_one_dominant_lot_top_lot_share_near_that_value():
    # Total pnl = 1000 + 100 + 100 = 1200; the dominant lot is 1000/1200.
    trades = [
        _trade(pnl=1000.0, return_pct=1.5, ticker="AAA"),
        _trade(pnl=100.0, return_pct=0.1, ticker="BBB"),
        _trade(pnl=100.0, return_pct=0.1, ticker="CCC"),
    ]
    out = metrics.compute(_run_result(trades))
    assert out["top_lot_pnl_share"] == pytest.approx(1000.0 / 1200.0)


def test_big_stake_increase_shape_share_exceeds_one_not_clamped():
    # Total pnl positive (+80k-like shape), but one lot's pnl alone exceeds
    # the total -- other lots net to a loss. Share must be > 1.0, unclamped.
    trades = [
        _trade(pnl=90_000.0, return_pct=5.0, ticker="MEGA"),
        _trade(pnl=-5_000.0, return_pct=-0.3, ticker="LOSER1"),
        _trade(pnl=-5_000.0, return_pct=-0.3, ticker="LOSER2"),
    ]
    out = metrics.compute(_run_result(trades))
    total_pnl = 90_000.0 - 5_000.0 - 5_000.0
    expected = 90_000.0 / total_pnl
    assert expected > 1.0
    assert out["top_lot_pnl_share"] == pytest.approx(expected)
    assert out["top_lot_pnl_share"] > 1.0
    # Confirms the "remove it and the strategy loses money" framing.
    assert (total_pnl - 90_000.0) < 0


def test_negative_total_pnl_is_defined_and_does_not_crash():
    trades = [
        _trade(pnl=-800.0, return_pct=-0.6, ticker="BADCO"),
        _trade(pnl=100.0, return_pct=0.1, ticker="OK1"),
    ]
    out = metrics.compute(_run_result(trades))
    total_pnl = -800.0 + 100.0
    assert out["top_lot_pnl_share"] == pytest.approx(-800.0 / total_pnl)
    # Not clamped, not NaN -- a real, meaningful (if unusual-looking) ratio.
    assert math.isfinite(out["top_lot_pnl_share"])


def test_total_pnl_exactly_zero_gives_nan_no_zero_division():
    trades = [
        _trade(pnl=500.0, return_pct=0.5, ticker="WIN"),
        _trade(pnl=-500.0, return_pct=-0.5, ticker="LOSE"),
    ]
    out = metrics.compute(_run_result(trades))
    assert math.isnan(out["top_lot_pnl_share"])
    assert math.isnan(out["top5_lot_pnl_share"])
    assert math.isnan(out["top_ticker_pnl_share"])
    assert math.isnan(out["pnl_share_gt_300pct"])


def test_top_ticker_pnl_share_aggregates_multiple_lots_same_ticker():
    # AAA has two lots summing to 600; BBB has one lot of 300.
    # Total = 900; AAA's aggregated share should win over any single lot.
    trades = [
        _trade(pnl=400.0, return_pct=0.4, ticker="AAA"),
        _trade(pnl=200.0, return_pct=0.2, ticker="AAA"),
        _trade(pnl=300.0, return_pct=0.3, ticker="BBB"),
    ]
    out = metrics.compute(_run_result(trades))
    assert out["top_ticker"] == "AAA"
    assert out["top_ticker_pnl_share"] == pytest.approx(600.0 / 900.0)
    # The single largest LOT is still BBB's 300 (AAA's lots are 400 and 200
    # individually), so top_lot_pnl_share must differ from the ticker share.
    assert out["top_lot_pnl_share"] == pytest.approx(400.0 / 900.0)


def test_n_lots_gt_300pct_and_pnl_share_count_and_sum_correctly():
    trades = [
        _trade(pnl=1000.0, return_pct=5.0, ticker="MOON1"),   # >300%
        _trade(pnl=500.0, return_pct=3.5, ticker="MOON2"),    # >300%
        _trade(pnl=100.0, return_pct=3.0, ticker="EDGE"),     # exactly 300%, not >
        _trade(pnl=200.0, return_pct=0.5, ticker="NORMAL"),
    ]
    out = metrics.compute(_run_result(trades))
    total_pnl = 1000.0 + 500.0 + 100.0 + 200.0
    assert out["n_lots_gt_300pct"] == 2
    assert out["pnl_share_gt_300pct"] == pytest.approx((1000.0 + 500.0) / total_pnl)


def test_empty_equity_curve_early_return_still_works():
    empty_eq = pd.Series([], dtype=float)
    result = RunResult(
        strategy="empty_strat", exit_label="empty_exit",
        equity_curve=empty_eq, trades=[], skips=Counter(),
        n_rebalances=0, exposure_curve=pd.Series([], dtype=float),
        n_tickers_curve=pd.Series([], dtype=float),
    )
    out = metrics.compute(result)
    assert out == {"strategy": "empty_strat", "exit_method": "empty_exit", "n_lots": 0}
    # New concentration keys are deliberately absent on this path, per the
    # documented policy in metrics.compute() -- callers must not assume
    # they're always present.
    assert "top_lot_pnl_share" not in out
