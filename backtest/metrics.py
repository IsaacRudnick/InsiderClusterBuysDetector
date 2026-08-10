"""Performance metrics from equity curves and trade lists."""

from __future__ import annotations

import re
from collections import Counter
from datetime import date

import numpy as np
import pandas as pd

from .engine import RunResult, Trade, rf_daily_rate


TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# In-sample-fraction: how much of a run's traded window overlaps the
# signal-fit training period.
#
# VERIFIED against backtest/strategies.py and backtest/state.py (not
# inferred from strategy naming): a strategy is "fit-dependent" only if its
# target_fn reads state["learned_score"] or state["tail_score"]. Both are
# populated from backtest/signal_fit.py fits (FitResult / TailFitResult)
# on a chronological train/test split of THIS SAME run's data -- trading
# any part of that train period is partly circular.
#
# The thr_gt_* family is NOT in that set despite the name: its target_fn
# thresholds state["conviction_score"], which backtest/state.py's
# _build_state ALWAYS computes from insider_cluster_buys.DEFAULT_WEIGHTS
# ("conviction_* is ALWAYS derived from ics.DEFAULT_WEIGHTS (never the
# active/loaded weights) ... immune to root signal_weights.json"). That's a
# fixed, hand-tuned constant baked into the source file -- it is never fit
# on any train/test split of this run's data, so thr_gt_* trading through
# the fit's train period is not circular the way learned_* is.
#
# model_ranked_* strategies rank by state["model_score"], populated from
# research_data/oof_scores_*.parquet -- genuine out-of-fold predictions
# from purged walk-forward CV, not this run's in-run train/test split -- so
# they are correctly excluded here too.
FIT_DEPENDENT_MEAN_STRATEGIES = frozenset({
    "learned_gt_m01", "learned_gt_p00", "learned_gt_p03",
    "learned_score_weighted", "learned_tpo_gated",
})
FIT_DEPENDENT_TAIL_STRATEGIES = frozenset({"learned_tail_concentrated"})

# backtest.py's cost-sensitivity sweep suffixes a strategy name with
# "_<bps>bps" (dataclasses.replace(s, name=f"{s.name}{suffix}")) before
# running it, so e.g. "learned_gt_p00_5bps" must still match
# "learned_gt_p00" above.
_SWEEP_SUFFIX_RE = re.compile(r"_\d+bps$")


def _strategy_base_name(name: str) -> str:
    return _SWEEP_SUFFIX_RE.sub("", name)


def _in_sample_frac(
    strategy: str,
    window_start,
    window_end,
    fit_train_end: date | None,
    tail_train_end: date | None,
) -> float:
    """Fraction of [window_start, window_end] that lies at-or-before the
    relevant fit's train_end, for strategies whose selection depends on a
    fitted signal score. 0.0 for every other strategy (including
    model_ranked_* and all hand-built ones) and whenever the relevant fit's
    metadata is unavailable (e.g. this run had fit_signal=False, or the fit
    failed and returned None) -- never a crash, never NaN by surprise.
    """
    base = _strategy_base_name(strategy)
    if base in FIT_DEPENDENT_TAIL_STRATEGIES:
        train_end = tail_train_end
    elif base in FIT_DEPENDENT_MEAN_STRATEGIES:
        train_end = fit_train_end
    else:
        return 0.0

    if train_end is None or window_start is None or window_end is None:
        return 0.0

    ws = pd.Timestamp(window_start)
    we = pd.Timestamp(window_end)
    te = pd.Timestamp(train_end)
    span = (we - ws).total_seconds()
    if span <= 0:
        return 0.0
    overlap = (te - ws).total_seconds()
    frac = overlap / span
    return float(max(0.0, min(1.0, frac)))


# Below this absolute-dollar threshold, total realized pnl is treated as "too
# near zero to divide by" — a pnl-share ratio over a near-zero denominator is
# either a ZeroDivisionError or a huge, meaningless number (e.g. a $0.01
# total pnl makes a single $50 lot look like a 5000x share). This is a small
# absolute epsilon rather than a relative one because pnl can be genuinely
# tiny in dollar terms for a toy/degenerate run without any lot being large.
PNL_EPS = 1e-6


def _daily_returns(equity: pd.Series) -> pd.Series:
    return equity.pct_change().dropna()


def _max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    return float(dd.min()) if not dd.empty else 0.0


def _pnl_concentration(realized: list[Trade]) -> dict:
    """How much of a strategy's total realized pnl rides on a handful of lots.

    Policy on the denominator (total pnl across `realized`):
    - Total pnl can be negative (a losing strategy) or ~zero (wins and
      losses cancel out). Both are real states, not error cases.
    - Negative total pnl is handled the same as positive: the ratios are
      still `component / total`, so e.g. one catastrophic loser that IS
      the entire loss still reports a share near 1.0, and a lot that's
      profitable in an overall-losing strategy reports a NEGATIVE share
      (it's fighting the total, not contributing to it). No sign flip.
    - Total pnl within PNL_EPS of zero makes any ratio numerically
      meaningless (dividing by ~0), so every share is NaN in that case
      rather than raising ZeroDivisionError or emitting a huge/undefined
      number.
    - Shares are RANKED BY ABSOLUTE PNL (so a catastrophic loser is caught
      as readily as a fake winner) but reported SIGNED (not abs()'d),
      because "-150% of total pnl" and "+150% of total pnl" mean very
      different things to a reader.
    - Shares are intentionally NOT clamped to [0, 1]. A share > 1.0 (e.g.
      1.13, as with big_stake_increase) is the exact signal this function
      exists to surface: the strategy is only net-profitable because one
      lot/ticker outran a loss everywhere else. Clamping would hide it.
    """
    nan = float("nan")
    total_pnl = sum(t.pnl for t in realized)
    denom_ok = abs(total_pnl) >= PNL_EPS

    def share(x: float) -> float:
        return (x / total_pnl) if denom_ok else nan

    out = {
        "top_lot_pnl_share": nan,
        "top5_lot_pnl_share": nan,
        "top_ticker_pnl_share": nan,
        "top_ticker": "",
        "n_lots_gt_300pct": sum(1 for t in realized if t.return_pct > 3.0),
        "pnl_share_gt_300pct": nan,
    }
    if not realized:
        return out

    by_abs = sorted(realized, key=lambda t: abs(t.pnl), reverse=True)
    out["top_lot_pnl_share"] = share(by_abs[0].pnl)
    out["top5_lot_pnl_share"] = share(sum(t.pnl for t in by_abs[:5]))

    by_ticker: dict[str, float] = {}
    for t in realized:
        by_ticker[t.ticker] = by_ticker.get(t.ticker, 0.0) + t.pnl
    top_ticker, top_ticker_pnl = max(by_ticker.items(), key=lambda kv: abs(kv[1]))
    out["top_ticker_pnl_share"] = share(top_ticker_pnl)
    out["top_ticker"] = str(top_ticker)

    gt_300_pnl = sum(t.pnl for t in realized if t.return_pct > 3.0)
    out["pnl_share_gt_300pct"] = share(gt_300_pnl)

    return out


def compute(result: RunResult, spy_equity: pd.Series | None = None,
            rf_model: str = "zero", fit_train_end: date | None = None,
            tail_train_end: date | None = None) -> dict:
    eq = result.equity_curve.dropna()
    if eq.empty or eq.iloc[0] == 0:
        # Keep this early-return's shape exactly as it was (3 keys). The
        # concentration keys added below are NOT injected here as NaN —
        # callers that need them must go through the populated path, or
        # use .get(...) with a default. This matches how every other
        # non-early-return key in this function already behaves: nothing
        # else is back-filled into the early-return dict either.
        return {"strategy": result.strategy, "exit_method": result.exit_label, "n_lots": 0}

    rets = _daily_returns(eq)
    n_days = len(eq)
    years = max(1e-9, n_days / TRADING_DAYS_PER_YEAR)
    rf_d = rf_daily_rate(rf_model)
    excess_rets = rets - rf_d

    total_return = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    cagr = float((eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1.0) if eq.iloc[0] > 0 else 0.0
    vol_ann = float(rets.std() * np.sqrt(TRADING_DAYS_PER_YEAR)) if len(rets) > 1 else 0.0
    sharpe = float(excess_rets.mean() / rets.std() * np.sqrt(TRADING_DAYS_PER_YEAR)) \
        if rets.std() > 0 else 0.0
    downside = excess_rets[excess_rets < 0]
    sortino = float(excess_rets.mean() / downside.std() * np.sqrt(TRADING_DAYS_PER_YEAR)) \
        if len(downside) > 1 and downside.std() > 0 else 0.0
    max_dd = _max_drawdown(eq)
    calmar = float(cagr / abs(max_dd)) if max_dd < 0 else float("nan")

    trades = result.trades
    realized = [t for t in trades if t.exit_reason in ("expiry", "trim", "final_liquidation")]
    # Win rate is per natural-exit lot only — end-of-run liquidations and
    # delisted closes are not strategy outcomes and would distort the denominator.
    natural = [t for t in realized if t.exit_reason != "final_liquidation" and not t.delisted]
    n_lots = len(realized)
    n_natural = len(natural)
    wins = sum(1 for t in natural if t.return_pct > 0)
    win_rate = (wins / n_natural) if n_natural else 0.0
    avg_ret = float(np.mean([t.return_pct for t in realized])) if realized else 0.0
    median_ret = float(np.median([t.return_pct for t in realized])) if realized else 0.0
    mean_exposure = float(result.exposure_curve.mean()) if not result.exposure_curve.empty else 0.0
    n_final_liq = sum(1 for t in realized if t.exit_reason == "final_liquidation")
    n_delisted = sum(1 for t in realized if t.delisted)

    out = {
        "strategy": result.strategy,
        "exit_method": result.exit_label,
        "total_return": total_return,
        "cagr": cagr,
        "volatility_ann": vol_ann,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "win_rate": win_rate,
        "avg_lot_return": avg_ret,
        "median_lot_return": median_ret,
        "n_lots": n_lots,
        "n_natural_exits": n_natural,
        "n_final_liquidation": n_final_liq,
        "n_delisted": n_delisted,
        "n_rebalances": result.n_rebalances,
        "n_skipped_capacity": result.skips.get("capacity", 0),
        "n_skipped_liquidity": result.skips.get("liquidity", 0),
        "n_skipped_cash": result.skips.get("cash", 0),
        "n_skipped_no_price": result.skips.get("no_price", 0),
        "n_skipped_price_floor": result.skips.get("price_floor", 0),
        "n_skipped_participation": result.skips.get("participation", 0),
        "n_capped_participation": result.skips.get("participation_capped", 0),
        "mean_exposure": mean_exposure,
    }
    out.update(_pnl_concentration(realized))
    # "This run's traded window" is read off the run's own equity curve
    # (not a caller-supplied config value) so it reflects exactly what this
    # RunResult actually traded.
    out["in_sample_frac"] = _in_sample_frac(
        result.strategy, eq.index.min(), eq.index.max(),
        fit_train_end, tail_train_end,
    )

    # Per-lot SPY comparison: each lot's return vs SPY return over same window
    if spy_equity is not None and not spy_equity.empty and realized:
        spy_lookup = spy_equity / spy_equity.iloc[0]
        spy_dates = spy_lookup.index
        excess = []
        beats = 0
        comp = 0
        for t in realized:
            try:
                ed = pd.Timestamp(t.entry_date)
                xd = pd.Timestamp(t.exit_date)
                # snap to nearest available SPY dates
                ed_i = spy_dates.searchsorted(ed)
                xd_i = spy_dates.searchsorted(xd)
                if ed_i >= len(spy_dates) or xd_i >= len(spy_dates) or ed_i == xd_i:
                    continue
                spy_ret = float(spy_lookup.iloc[xd_i] / spy_lookup.iloc[ed_i] - 1.0)
                ex = t.return_pct - spy_ret
                excess.append(ex)
                comp += 1
                if t.return_pct > spy_ret:
                    beats += 1
            except Exception:
                continue
        out["hit_rate_vs_spy"] = (beats / comp) if comp else 0.0
        out["avg_excess_vs_spy"] = float(np.mean(excess)) if excess else 0.0
    else:
        out["hit_rate_vs_spy"] = float("nan")
        out["avg_excess_vs_spy"] = float("nan")

    # Daily-return OLS regression vs SPY: alpha, beta, R², t(alpha), TE, IR.
    ab = _alpha_beta_vs_spy(eq, spy_equity, rf_d)
    out.update(ab)

    return out


def _alpha_beta_vs_spy(eq: pd.Series, spy_equity: pd.Series | None,
                       rf_daily: float) -> dict:
    """OLS regression of strategy excess returns on SPY excess returns.

    Returns annualized alpha + beta + R² + t(alpha) + tracking error + IR.
    All NaN if the equity / SPY series can't be aligned to ≥2 overlapping days.
    """
    nan = float("nan")
    blank = {
        "alpha_ann": nan, "beta": nan, "r_squared": nan,
        "t_alpha": nan, "tracking_error": nan, "info_ratio": nan,
    }
    if spy_equity is None or spy_equity.empty:
        return blank
    joined = pd.concat(
        [eq.pct_change(), spy_equity.pct_change()], axis=1,
    ).dropna()
    if len(joined) < 5:
        return blank
    joined.columns = ["y_raw", "x_raw"]
    y = (joined["y_raw"] - rf_daily).to_numpy()
    x = (joined["x_raw"] - rf_daily).to_numpy()
    if x.std() == 0:
        return blank
    n = len(y)
    x_mean = x.mean()
    y_mean = y.mean()
    sxx = float(((x - x_mean) ** 2).sum())
    if sxx <= 0:
        return blank
    sxy = float(((x - x_mean) * (y - y_mean)).sum())
    beta = sxy / sxx
    alpha = y_mean - beta * x_mean  # daily
    y_hat = alpha + beta * x
    resid = y - y_hat
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((y - y_mean) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else nan
    df = max(1, n - 2)
    se2 = ss_res / df
    se_alpha = (se2 * (1.0 / n + x_mean * x_mean / sxx)) ** 0.5
    t_alpha = alpha / se_alpha if se_alpha > 0 else nan

    diff = joined["y_raw"].to_numpy() - joined["x_raw"].to_numpy()
    te = float(diff.std() * (TRADING_DAYS_PER_YEAR ** 0.5)) if len(diff) > 1 else nan
    ir = (
        float(diff.mean() / diff.std() * (TRADING_DAYS_PER_YEAR ** 0.5))
        if len(diff) > 1 and diff.std() > 0 else nan
    )
    return {
        "alpha_ann": float(alpha * TRADING_DAYS_PER_YEAR),
        "beta": float(beta),
        "r_squared": float(r2) if r2 == r2 else nan,
        "t_alpha": float(t_alpha) if t_alpha == t_alpha else nan,
        "tracking_error": te,
        "info_ratio": ir,
    }


def summary_table(results: list[RunResult], spy_result: RunResult | None = None,
                  rf_model: str = "zero", fit_train_end: date | None = None,
                  tail_train_end: date | None = None) -> pd.DataFrame:
    spy_eq = spy_result.equity_curve if spy_result else None
    rows = [compute(r, spy_eq, rf_model, fit_train_end, tail_train_end) for r in results]
    if spy_result is not None:
        rows.insert(0, compute(spy_result, spy_eq, rf_model, fit_train_end, tail_train_end))
    return pd.DataFrame(rows)
