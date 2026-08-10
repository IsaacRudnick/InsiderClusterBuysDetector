"""Backtest-derived conviction weights: event dataset, numpy ridge/OLS fit,
coefficient-to-points mapping, weights JSON I/O, and out-of-sample stats.

Pipeline (see backtest.py Phase 4.5):
  1. build_event_dataset() turns the daily state index into one row per
     "episode start" (a ticker first qualifying with >=2 insiders), with
     binary component flags as features and SPY-adjusted forward returns
     over several horizons as candidate targets.
  2. fit_weights() does a single chronological train/test split, fits a
     numpy-only ridge regression (normal equations) on the primary horizon's
     adjusted return, and maps surviving coefficients to small integer
     points comparable to the hand-tuned scale in insider_cluster_buys.py.
  3. weights_payload()/save_weights() persist the learned weights (and fit
     metadata) as JSON for both the backtest report and the live scanner's
     signal_weights.json loader.
  4. oos_strategy_stats() summarizes strategy equity curves restricted to
     the out-of-sample window, for an apples-to-apples comparison against
     the hand-tuned strategies and SPY.

Deliberately numpy + pandas only (no scipy/sklearn) and fully deterministic
(no RNG anywhere in this module).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

import insider_cluster_buys as ics
from . import metrics
from .engine import RunResult
from .prices import PriceUniverse
from .state import DailyStateBuilder

log = logging.getLogger(__name__)

MAX_ENTRY_LOOKAHEAD = 5   # trading days to search for a priceable entry open
VAL_FRACTION = 0.20       # trailing slice of train used for lambda selection
MIN_TRAIN_EVENTS = 100    # below this, fit_weights bails out


@dataclass(frozen=True)
class FitConfig:
    horizons: tuple[int, ...] = (10, 30, 90, 180, 365)   # trading days
    primary_horizon: int = 90
    train_frac: float = 0.70
    ridge_lambdas: tuple[float, ...] = (0.0, 1.0, 10.0, 100.0)
    winsor_lo_pct: float = 0.01
    winsor_hi_pct: float = 0.999
    min_flag_count: int = 20
    t_min: float = 1.5
    point_cap: int = 6
    # -- tail-probability ("moonshot") scoring (see fit_tail_score) --
    tail_horizon: int = 90          # trading days; target column adj_<tail_horizon>
    moonshot_thresh: float = 0.20   # a "big win" = adjusted return above this
    tail_scale: float = 4.0         # points = round(tail_scale * log2_lift)
    tail_min_flag_count: int = 100  # min train firings for a flag to get points
    tail_min_lift: float = 0.1      # |pooled log2 lift| below this -> 0 points


@dataclass
class FitResult:
    split_date: date
    train_events: int
    test_events: int
    primary_horizon: int
    lambda_used: float
    feature_stats: pd.DataFrame     # per-key: hand_weight, coef, raw_uplift, t_stat, n_fired_train, points, coef_<h>...
    weights_int: dict               # all DEFAULT_WEIGHTS keys -> int points
    oos_bucket_table: pd.DataFrame  # OOS event-level bucket comparison, hand vs learned
    target_desc: str
    train_start: Optional[date] = None
    train_end: Optional[date] = None


@dataclass
class TailFitResult:
    """Result of fit_tail_score(): a per-flag log2 lift score on
    P(adjusted return > moonshot_thresh at tail_horizon days), intended to
    rank clusters for research triage rather than to predict mean return
    (see fit_tail_score's docstring for why the mean-return ridge fit can't
    do this)."""
    split_date: date
    train_events: int
    test_events: int
    tail_horizon: int
    moonshot_thresh: float
    base_rate: float                 # train P(target > moonshot_thresh)
    tail_stats: pd.DataFrame         # per-key: lift, lift_h1, lift_h2, p_moonshot, n_fired_train, points
    weights_int: dict                # all DEFAULT_WEIGHTS keys -> int points
    oos_quintile_table: pd.DataFrame # OOS quintile validation of the summed tail score
    target_desc: str
    train_start: Optional[date] = None
    train_end: Optional[date] = None


# ---------------------------------------------------------------------------
# 1. Event dataset
# ---------------------------------------------------------------------------
def _open_or_fallback(prices: PriceUniverse, ticker: str, day: date) -> Optional[float]:
    """Open price, falling back to the last available close on/before `day`
    (handles delistings where the open on the exact exit day is missing)."""
    px = prices.open(ticker, day)
    if px is not None:
        return px
    return prices.last_close_on_or_before(ticker, day)


def _find_entry_day(
    prices: PriceUniverse, ticker: str, idx_D: int, calendar: list[date],
    max_lookahead: int = MAX_ENTRY_LOOKAHEAD,
) -> tuple[Optional[date], Optional[int]]:
    """First trading day strictly after calendar[idx_D] with a valid open
    price for `ticker`, mirroring the engine's next-open execution. Searches
    at most `max_lookahead` trading days ahead; (None, None) if none found."""
    for offset in range(1, max_lookahead + 1):
        idx = idx_D + offset
        if idx >= len(calendar):
            break
        day = calendar[idx]
        if prices.open(ticker, day) is not None:
            return day, idx
    return None, None


def build_event_dataset(
    states: DailyStateBuilder, prices: PriceUniverse, calendar: list[date],
    cfg: FitConfig,
) -> pd.DataFrame:
    """One row per episode start: a ticker qualifying (num_insiders >= 2) on
    trading day D that was not qualifying on the previous trading day. This
    dedups the multi-day rolling window down to the moment the engine can
    first act on the signal.

    Columns: ticker, event_day, entry_day, f_<key> (one per
    insider_cluster_buys.DEFAULT_WEIGHTS key, 1/0 from state["component_keys"]),
    then fwd_<h>/adj_<h> pairs for each horizon in cfg.horizons — the
    open-to-open forward return from entry day E to E+h trading days, and
    that return minus SPY's over the identical window.
    """
    feature_keys = list(ics.DEFAULT_WEIGHTS.keys())
    feature_cols = [f"f_{k}" for k in feature_keys]
    horizon_cols: list[str] = []
    for h in cfg.horizons:
        horizon_cols.extend([f"fwd_{h}", f"adj_{h}"])
    columns = ["ticker", "event_day", "entry_day"] + feature_cols + horizon_cols

    rows: list[dict] = []
    prev_qualifying: set[str] = set()
    n_skipped_no_entry = 0

    for idx_D, D in enumerate(calendar):
        day_states = states.state_for_day(D)
        qualifying = {t for t, st in day_states.items() if st.get("num_insiders", 0) >= 2}
        new_tickers = sorted(qualifying - prev_qualifying)

        for ticker in new_tickers:
            st = day_states[ticker]
            entry_day, entry_idx = _find_entry_day(prices, ticker, idx_D, calendar)
            if entry_day is None:
                n_skipped_no_entry += 1
                prev_qualifying = qualifying
                continue

            fired = set(st.get("component_keys", []))
            row: dict = {"ticker": ticker, "event_day": D, "entry_day": entry_day}
            for k, col in zip(feature_keys, feature_cols):
                row[col] = 1 if k in fired else 0

            entry_open = prices.open(ticker, entry_day)
            spy_entry_open = prices.open("SPY", entry_day)

            for h in cfg.horizons:
                exit_idx = entry_idx + h
                if exit_idx >= len(calendar):
                    row[f"fwd_{h}"] = float("nan")
                    row[f"adj_{h}"] = float("nan")
                    continue
                X = calendar[exit_idx]
                exit_px = _open_or_fallback(prices, ticker, X)
                spy_exit_px = _open_or_fallback(prices, "SPY", X)

                if exit_px is None or entry_open is None:
                    fwd_h = float("nan")
                else:
                    fwd_h = exit_px / entry_open - 1.0
                if spy_exit_px is None or spy_entry_open is None:
                    spy_h = float("nan")
                else:
                    spy_h = spy_exit_px / spy_entry_open - 1.0

                row[f"fwd_{h}"] = fwd_h
                row[f"adj_{h}"] = (fwd_h - spy_h) if (fwd_h == fwd_h and spy_h == spy_h) else float("nan")

            rows.append(row)

        prev_qualifying = qualifying

    df = pd.DataFrame(rows, columns=columns)
    if df.empty:
        log.warning(
            "build_event_dataset: no events produced (%d skipped for missing entry price)",
            n_skipped_no_entry,
        )
        return df
    log.info(
        "build_event_dataset: %d events spanning %s .. %s (%d skipped for missing "
        "entry price within %d trading days)",
        len(df), df["event_day"].min(), df["event_day"].max(),
        n_skipped_no_entry, MAX_ENTRY_LOOKAHEAD,
    )
    return df


# ---------------------------------------------------------------------------
# 2. Fit
# ---------------------------------------------------------------------------
def _design_matrix(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Binary feature matrix with an unpenalized intercept column prepended."""
    X = df[cols].to_numpy(dtype=float)
    intercept = np.ones((X.shape[0], 1))
    return np.hstack([intercept, X])


def _ridge_fit(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """Ridge via normal equations; column 0 (intercept) is unpenalized."""
    n_features = X.shape[1]
    penalty = np.eye(n_features)
    penalty[0, 0] = 0.0
    A = X.T @ X + lam * penalty
    b = X.T @ y
    return np.linalg.solve(A, b)


def _select_lambda(X: np.ndarray, y: np.ndarray, lambdas: tuple[float, ...]) -> tuple[float, float]:
    """Pick lambda by validation MSE on the time-ordered trailing VAL_FRACTION
    of train (X, y are assumed already sorted by event day)."""
    n = X.shape[0]
    val_start = int(np.floor((1.0 - VAL_FRACTION) * n))
    val_start = min(max(val_start, 1), n - 1)
    X_inner, y_inner = X[:val_start], y[:val_start]
    X_val, y_val = X[val_start:], y[val_start:]

    best_lam = lambdas[0]
    best_mse = float("inf")
    for lam in lambdas:
        beta = _ridge_fit(X_inner, y_inner, lam)
        pred = X_val @ beta
        mse = float(np.mean((y_val - pred) ** 2))
        if mse < best_mse:
            best_mse = mse
            best_lam = lam
    return float(best_lam), best_mse


def _ols_tstats(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Classic OLS t-stats (non-intercept columns only) via lstsq + the usual
    sigma^2 * (X'X)^-1 sandwich.

    NOTE: events overlap in time (multiple episodes' H-day forward-return
    windows share calendar days), so residuals are serially correlated and
    these t-stats are optimistic (inflated) versus a "true" iid-residual
    OLS. We treat them as a coarse ranking/gating signal only — hence the
    conservative t_min gate in _coefs_to_points rather than a strict 1.96
    significance cutoff.
    """
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, k = X.shape
    dof = max(1, n - k)
    sigma2 = float(resid @ resid) / dof
    xtx_inv = np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.clip(np.diag(xtx_inv) * sigma2, 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, beta / se, 0.0)
    return t[1:]  # drop intercept


def _coefs_to_points(
    coefs: np.ndarray, t_stats: np.ndarray, n_fired: np.ndarray, cfg: FitConfig,
) -> np.ndarray:
    """Zero out coefficients that fail the significance/support gate, scale
    surviving coefficients so max|coef| maps to 5 points (comparable to the
    hand-tuned -5..+3 scale), round, and clamp to +/- point_cap."""
    survive = (np.abs(t_stats) >= cfg.t_min) & (n_fired >= cfg.min_flag_count)
    filtered = np.where(survive, coefs, 0.0)
    max_abs = float(np.max(np.abs(filtered))) if filtered.size else 0.0
    if max_abs <= 0:
        return np.zeros_like(coefs, dtype=int)
    scale = 5.0 / max_abs
    points = np.round(filtered * scale).astype(int)
    return np.clip(points, -cfg.point_cap, cfg.point_cap)


def _bucket_score(score: np.ndarray) -> np.ndarray:
    return np.select(
        [score < 0, score == 0, score == 1, score == 2, score == 3, score >= 4],
        ["<0", "0", "1", "2", "3", "4+"],
        default="4+",
    )


def _oos_bucket_table(test_df: pd.DataFrame, weights_int: dict, cfg: FitConfig) -> pd.DataFrame:
    """OOS event-level bucket comparison: hand-tuned score vs learned score,
    bucketed into integer conviction tiers (<0, 0, 1, 2, 3, 4+), with n /
    mean adj return / median adj return / win rate per bucket per scoring."""
    if test_df.empty:
        return pd.DataFrame(columns=["scoring", "bucket", "n", "mean_adj", "median_adj", "win_rate"])

    feature_cols = [f"f_{k}" for k in ics.DEFAULT_WEIGHTS]
    hand_w = np.array([ics.DEFAULT_WEIGHTS[k] for k in ics.DEFAULT_WEIGHTS], dtype=float)
    learned_w = np.array([weights_int.get(k, 0) for k in ics.DEFAULT_WEIGHTS], dtype=float)
    flags = test_df[feature_cols].to_numpy(dtype=float)
    hand_score = flags @ hand_w
    learned_score = flags @ learned_w
    adj = test_df[f"adj_{cfg.primary_horizon}"].to_numpy(dtype=float)

    bucket_order = ["<0", "0", "1", "2", "3", "4+"]
    rows = []
    for label, score in (("hand", hand_score), ("learned", learned_score)):
        buckets = _bucket_score(score)
        tmp = pd.DataFrame({"bucket": buckets, "adj": adj})
        for b in bucket_order:
            sub = tmp[tmp["bucket"] == b]
            if sub.empty:
                continue
            rows.append({
                "scoring": label,
                "bucket": b,
                "n": int(len(sub)),
                "mean_adj": float(sub["adj"].mean()),
                "median_adj": float(sub["adj"].median()),
                "win_rate": float((sub["adj"] > 0).mean()),
            })
    return pd.DataFrame(rows)


def _chronological_split(events: pd.DataFrame, cfg: FitConfig) -> tuple:
    """Single chronological train/test split by event_day (cfg.train_frac),
    shared by fit_weights and fit_tail_score. Returns
    (split_date, train_all, test_all) — train_all/test_all still include
    rows with a NaN target; callers dropna() on their own target column."""
    ev_sorted = events.sort_values("event_day", kind="mergesort").reset_index(drop=True)
    n = len(ev_sorted)
    split_idx = min(max(int(np.floor(cfg.train_frac * n)), 0), n - 1)
    split_date = ev_sorted.loc[split_idx, "event_day"]

    train_all = ev_sorted[ev_sorted["event_day"] < split_date].reset_index(drop=True)
    test_all = ev_sorted[ev_sorted["event_day"] >= split_date].reset_index(drop=True)
    return split_date, train_all, test_all


def fit_weights(events: pd.DataFrame, cfg: FitConfig) -> Optional[FitResult]:
    """Single chronological train/test split; ridge-regress the primary
    horizon's SPY-adjusted return on binary component flags; map surviving
    coefficients to integer points. Returns None (and logs a warning) if
    fewer than MIN_TRAIN_EVENTS train events have a valid primary target."""
    if events.empty:
        log.warning("fit_weights: empty event dataset — skipping fit.")
        return None

    feature_keys = list(ics.DEFAULT_WEIGHTS.keys())
    feature_cols = [f"f_{k}" for k in feature_keys]
    target_col = f"adj_{cfg.primary_horizon}"

    split_date, train_all, test_all = _chronological_split(events, cfg)

    train_df = train_all.dropna(subset=[target_col]).reset_index(drop=True)
    test_df = test_all.dropna(subset=[target_col]).reset_index(drop=True)

    if len(train_df) < MIN_TRAIN_EVENTS:
        log.warning(
            "fit_weights: only %d train events with a valid %s target (< %d) — skipping fit.",
            len(train_df), target_col, MIN_TRAIN_EVENTS,
        )
        return None

    # Winsorize at train quantiles; same clip bounds applied to train + test.
    # Asymmetric: the low tail is clipped tightly (1%) to blunt data-error
    # style blowups, but the high tail is kept almost entirely (99.9%) since
    # moonshot winners are real lottery-type outcomes, not errors — some
    # components' entire edge lives in that right tail.
    lo = float(train_df[target_col].quantile(cfg.winsor_lo_pct))
    hi = float(train_df[target_col].quantile(cfg.winsor_hi_pct))
    train_y = train_df[target_col].clip(lo, hi).to_numpy(dtype=float)
    test_y = test_df[target_col].clip(lo, hi).to_numpy(dtype=float) if not test_df.empty else np.array([])

    # Raw (unwinsorized) cohort uplift per component on the TRAIN split, for
    # every DEFAULT_WEIGHTS key regardless of the support/variance filters
    # below — this is the number that survives even when a regression's
    # partial coefficient shrinks or a winsorization clip mutes a
    # tail-driven effect.
    raw_target_train = train_df[target_col]
    raw_uplift_map: dict[str, float] = {}
    for k in ics.DEFAULT_WEIGHTS:
        col = f"f_{k}"
        fired_mask = train_df[col] == 1
        if fired_mask.any() and (~fired_mask).any():
            raw_uplift_map[k] = float(
                raw_target_train[fired_mask].mean() - raw_target_train[~fired_mask].mean()
            )
        else:
            raw_uplift_map[k] = float("nan")

    n_fired_train = train_df[feature_cols].sum(axis=0)
    variance_train = train_df[feature_cols].var(axis=0, ddof=0)
    active_cols = [
        c for c in feature_cols
        if n_fired_train[c] >= cfg.min_flag_count and variance_train[c] > 0
    ]
    if not active_cols:
        log.warning("fit_weights: no features survived the support/variance filters — skipping fit.")
        return None

    X_train = _design_matrix(train_df, active_cols)

    lambda_used, val_mse = _select_lambda(X_train, train_y, cfg.ridge_lambdas)
    beta = _ridge_fit(X_train, train_y, lambda_used)
    t_stats = _ols_tstats(X_train, train_y)
    coefs = beta[1:]

    n_fired_active = n_fired_train[active_cols].to_numpy()
    points = _coefs_to_points(coefs, t_stats, n_fired_active, cfg)

    weights_int = {k: 0 for k in ics.DEFAULT_WEIGHTS}
    for col, p in zip(active_cols, points):
        weights_int[col[2:]] = int(p)

    # Diagnostic per-horizon ridge coefficients (same lambda_used, same
    # active feature set) for the feature_stats report columns.
    horizon_coefs: dict[int, np.ndarray] = {}
    for h in cfg.horizons:
        hcol = f"adj_{h}"
        sub = train_all.dropna(subset=[hcol])
        if len(sub) < len(active_cols) + 5:
            horizon_coefs[h] = np.full(len(active_cols), np.nan)
            continue
        hlo = float(sub[hcol].quantile(cfg.winsor_lo_pct))
        hhi = float(sub[hcol].quantile(cfg.winsor_hi_pct))
        y_h = sub[hcol].clip(hlo, hhi).to_numpy(dtype=float)
        X_h = _design_matrix(sub, active_cols)
        beta_h = _ridge_fit(X_h, y_h, lambda_used)
        horizon_coefs[h] = beta_h[1:]

    feature_rows = []
    for k in ics.DEFAULT_WEIGHTS:
        col = f"f_{k}"
        row = {
            "component": k,
            "hand_weight": ics.DEFAULT_WEIGHTS[k],
            "coef": float("nan"),
            "raw_uplift": raw_uplift_map.get(k, float("nan")),
            "t_stat": float("nan"),
            "n_fired_train": int(n_fired_train.get(col, 0)),
            "points": weights_int[k],
        }
        for h in cfg.horizons:
            row[f"coef_{h}"] = float("nan")
        if col in active_cols:
            i = active_cols.index(col)
            row["coef"] = float(coefs[i])
            row["t_stat"] = float(t_stats[i])
            for h in cfg.horizons:
                hc = horizon_coefs[h]
                row[f"coef_{h}"] = float(hc[i]) if hc.size else float("nan")
        feature_rows.append(row)
    feature_stats = pd.DataFrame(feature_rows)

    oos_bucket_table = _oos_bucket_table(test_df, weights_int, cfg)

    target_desc = (
        f"{cfg.primary_horizon}-trading-day open-to-open forward return minus SPY's "
        f"over the identical window, winsorized to train-quantile bounds "
        f"[{lo:.4f}, {hi:.4f}] at the {cfg.winsor_lo_pct:.1%}/{cfg.winsor_hi_pct:.1%} tails"
    )

    return FitResult(
        split_date=split_date,
        train_events=len(train_df),
        test_events=len(test_df),
        primary_horizon=cfg.primary_horizon,
        lambda_used=lambda_used,
        feature_stats=feature_stats,
        weights_int=weights_int,
        oos_bucket_table=oos_bucket_table,
        target_desc=target_desc,
        train_start=train_df["event_day"].min() if not train_df.empty else None,
        train_end=train_df["event_day"].max() if not train_df.empty else None,
    )


# ---------------------------------------------------------------------------
# 2b. Tail-probability ("moonshot") score
# ---------------------------------------------------------------------------
MIN_TAIL_TRAIN_EVENTS = 500   # below this, fit_tail_score bails out


def _flag_lift(fired: pd.Series, is_moonshot: pd.Series) -> float:
    """log2(P(moonshot | flag fired) / P(moonshot)) within the given slice,
    floored at 1e-3 on both sides to avoid -inf/±inf for extreme rates. NaN
    if the flag never fires in this slice."""
    if not fired.any():
        return float("nan")
    p_flag = float(is_moonshot[fired].mean())
    base = float(is_moonshot.mean()) if len(is_moonshot) else 0.0
    return float(np.log2(max(p_flag, 1e-3) / max(base, 1e-3)))


def _tail_oos_quintile_table(test_df: pd.DataFrame, weights_int: dict, cfg: FitConfig) -> pd.DataFrame:
    """OOS validation of the summed tail score: quintile-bucket the test-set
    events by score and report n / P(moonshot) / mean & median adjusted
    return / bucket score range. Falls back to one row per unique score if
    pd.qcut can't produce at least 3 distinct bins (e.g. most scores are 0)."""
    cols = ["bucket", "n", "p_moonshot", "mean_adj", "median_adj", "score_lo", "score_hi"]
    target_col = f"adj_{cfg.tail_horizon}"
    if test_df.empty:
        return pd.DataFrame(columns=cols)

    feature_cols = [f"f_{k}" for k in ics.DEFAULT_WEIGHTS]
    w = np.array([weights_int.get(k, 0) for k in ics.DEFAULT_WEIGHTS], dtype=float)
    flags = test_df[feature_cols].to_numpy(dtype=float)
    score = flags @ w
    adj = test_df[target_col].to_numpy(dtype=float)
    is_moonshot = adj > cfg.moonshot_thresh

    tmp = pd.DataFrame({"score": score, "adj": adj, "moonshot": is_moonshot})

    quintiles = None
    try:
        quintiles = pd.qcut(tmp["score"], 5, duplicates="drop")
    except ValueError:
        quintiles = None

    rows: list[dict] = []
    if quintiles is not None and quintiles.cat.categories.size >= 3:
        tmp["bucket"] = quintiles
        for interval, sub in tmp.groupby("bucket", observed=True):
            if sub.empty:
                continue
            rows.append({
                "bucket": str(interval),
                "n": int(len(sub)),
                "p_moonshot": float(sub["moonshot"].mean()),
                "mean_adj": float(sub["adj"].mean()),
                "median_adj": float(sub["adj"].median()),
                "score_lo": float(interval.left),
                "score_hi": float(interval.right),
            })
    else:
        # Fewer than 3 distinct score-based bins possible — fall back to one
        # row per unique score value.
        for score_val, sub in tmp.groupby("score"):
            rows.append({
                "bucket": f"score={score_val:g}",
                "n": int(len(sub)),
                "p_moonshot": float(sub["moonshot"].mean()),
                "mean_adj": float(sub["adj"].mean()),
                "median_adj": float(sub["adj"].median()),
                "score_lo": float(score_val),
                "score_hi": float(score_val),
            })
    rows.sort(key=lambda r: r["score_lo"])
    return pd.DataFrame(rows, columns=cols)


def fit_tail_score(events: pd.DataFrame, cfg: FitConfig) -> Optional[TailFitResult]:
    """Per-flag log2 lift on P(adjusted return > moonshot_thresh at
    tail_horizon days) — a research-triage score for ranking clusters by
    upside probability, complementing (not replacing) fit_weights().

    fit_weights() regresses mean SPY-adjusted forward return on the 22
    binary component flags, which correctly identifies "avoid" flags but
    can't rank clusters for research triage: cluster outcomes are
    lottery-distributed (most clusters return roughly nothing; a minority
    are moonshots), so the flags predict P(big win), not mean return. This
    function fits that P(big win) signal directly: for each flag, the log2
    ratio of P(moonshot | flag fired) to the train base rate, gated on
    train support, minimum |lift|, and a stability check requiring the
    same-signed lift on both chronological halves of train.

    Uses the same chronological train/test split as fit_weights (via
    _chronological_split). Returns None (and logs a warning) if fewer than
    MIN_TAIL_TRAIN_EVENTS train events have a valid tail_horizon target.
    """
    if events.empty:
        log.warning("fit_tail_score: empty event dataset — skipping fit.")
        return None

    feature_keys = list(ics.DEFAULT_WEIGHTS.keys())
    feature_cols = [f"f_{k}" for k in feature_keys]
    target_col = f"adj_{cfg.tail_horizon}"
    if target_col not in events.columns:
        log.warning(
            "fit_tail_score: target column %s not present in event dataset — skipping fit.",
            target_col,
        )
        return None

    split_date, train_all, test_all = _chronological_split(events, cfg)

    train_df = train_all.dropna(subset=[target_col]).reset_index(drop=True)
    test_df = test_all.dropna(subset=[target_col]).reset_index(drop=True)

    if len(train_df) < MIN_TAIL_TRAIN_EVENTS:
        log.warning(
            "fit_tail_score: only %d train events with a valid %s target (< %d) — skipping fit.",
            len(train_df), target_col, MIN_TAIL_TRAIN_EVENTS,
        )
        return None

    is_moonshot = train_df[target_col] > cfg.moonshot_thresh
    base_rate = float(is_moonshot.mean())

    # Chronological first/second halves of train, for the stability gate.
    half_idx = len(train_df) // 2
    h1_df = train_df.iloc[:half_idx]
    h2_df = train_df.iloc[half_idx:]
    h1_moonshot = h1_df[target_col] > cfg.moonshot_thresh
    h2_moonshot = h2_df[target_col] > cfg.moonshot_thresh

    weights_int = {k: 0 for k in feature_keys}
    rows = []
    for k, col in zip(feature_keys, feature_cols):
        fired = train_df[col] == 1
        n_fired = int(fired.sum())
        p_moonshot = float(is_moonshot[fired].mean()) if n_fired else float("nan")
        lift = (
            float(np.log2(max(p_moonshot, 1e-3) / max(base_rate, 1e-3)))
            if n_fired else float("nan")
        )

        lift_h1 = _flag_lift(h1_df[col] == 1, h1_moonshot)
        lift_h2 = _flag_lift(h2_df[col] == 1, h2_moonshot)

        points = 0
        stable = (
            lift_h1 == lift_h1 and lift_h2 == lift_h2  # both non-NaN
            and lift_h1 != 0.0 and lift_h2 != 0.0       # exact-0 half-lift -> mismatch
            and np.sign(lift_h1) == np.sign(lift_h2)
        )
        if (
            n_fired >= cfg.tail_min_flag_count
            and lift == lift  # non-NaN
            and abs(lift) >= cfg.tail_min_lift
            and stable
        ):
            points = int(np.clip(round(cfg.tail_scale * lift), -cfg.point_cap, cfg.point_cap))

        weights_int[k] = points
        rows.append({
            "component": k,
            "hand_weight": ics.DEFAULT_WEIGHTS[k],
            "lift": lift,
            "lift_h1": lift_h1,
            "lift_h2": lift_h2,
            "p_moonshot": p_moonshot,
            "n_fired_train": n_fired,
            "points": points,
        })
    tail_stats = pd.DataFrame(rows)

    oos_quintile_table = _tail_oos_quintile_table(test_df, weights_int, cfg)

    target_desc = (
        f"P(adjusted {cfg.tail_horizon}-trading-day open-to-open return minus SPY's > "
        f"{cfg.moonshot_thresh:.0%}) — a 'moonshot' — scored per-flag as the log2 lift over "
        f"the train base rate, gated on train support (>= {cfg.tail_min_flag_count} fires), "
        f"minimum |lift| ({cfg.tail_min_lift}), and same-sign stability across the "
        f"chronological first/second half of train"
    )

    return TailFitResult(
        split_date=split_date,
        train_events=len(train_df),
        test_events=len(test_df),
        tail_horizon=cfg.tail_horizon,
        moonshot_thresh=cfg.moonshot_thresh,
        base_rate=base_rate,
        tail_stats=tail_stats,
        weights_int=weights_int,
        oos_quintile_table=oos_quintile_table,
        target_desc=target_desc,
        train_start=train_df["event_day"].min() if not train_df.empty else None,
        train_end=train_df["event_day"].max() if not train_df.empty else None,
    )


# ---------------------------------------------------------------------------
# 3. Persistence
# ---------------------------------------------------------------------------
def weights_payload(
    fit: FitResult, cfg: FitConfig, tail_fit: Optional[TailFitResult] = None,
) -> dict:
    """JSON-serializable payload for signal_weights.json. Consumed by
    insider_cluster_buys.load_signal_weights() via payload["weights"].

    When `tail_fit` is given, adds a "tail_weights" dict (the research-
    triage P(moonshot) points, keyed the same as "weights") and a
    "tail_fit" metadata block. Callers deciding which weight set should be
    live (payload["weights"]) for the scanner — mean-return vs
    tail-probability — do so themselves (see backtest.py's BT_WRITE_WEIGHTS
    root-write path); this function always reports the mean-return fit
    under "weights" and, if present, the tail fit alongside it."""
    payload = {
        "version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fit": {
            "train_start": fit.train_start.isoformat() if fit.train_start else None,
            "train_end": fit.train_end.isoformat() if fit.train_end else None,
            "split_date": fit.split_date.isoformat() if hasattr(fit.split_date, "isoformat")
                           else str(fit.split_date),
            "primary_horizon": fit.primary_horizon,
            "train_events": fit.train_events,
            "test_events": fit.test_events,
            "lambda_used": fit.lambda_used,
            "target_desc": fit.target_desc,
        },
        "weights": fit.weights_int,
    }
    if tail_fit is not None:
        payload["tail_weights"] = tail_fit.weights_int
        payload["tail_fit"] = {
            "tail_horizon": tail_fit.tail_horizon,
            "moonshot_thresh": tail_fit.moonshot_thresh,
            "base_rate": tail_fit.base_rate,
            "split_date": tail_fit.split_date.isoformat()
                           if hasattr(tail_fit.split_date, "isoformat")
                           else str(tail_fit.split_date),
            "train_start": tail_fit.train_start.isoformat() if tail_fit.train_start else None,
            "train_end": tail_fit.train_end.isoformat() if tail_fit.train_end else None,
            "train_events": tail_fit.train_events,
            "test_events": tail_fit.test_events,
            "target_desc": tail_fit.target_desc,
        }
    return payload


def save_weights(payload: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    log.info("Wrote learned signal weights to %s", path)


# ---------------------------------------------------------------------------
# 4. OOS strategy stats
# ---------------------------------------------------------------------------
def oos_strategy_stats(
    all_results: list[RunResult], spy_result: Optional[RunResult], split_date: date,
) -> pd.DataFrame:
    """Rebase each equity curve to 1.0 at split_date and compute OOS total
    return / CAGR / Sharpe (daily returns, rf=0), for direct comparison of
    learned strategies against hand-tuned ones and SPY on unseen data."""
    split_ts = pd.Timestamp(split_date)
    results = list(all_results) + ([spy_result] if spy_result is not None else [])

    rows = []
    for r in results:
        eq_oos = r.equity_curve[r.equity_curve.index >= split_ts]
        if eq_oos.empty or eq_oos.iloc[0] == 0:
            rows.append({
                "strategy": r.strategy, "exit": r.exit_label,
                "oos_total_return": float("nan"),
                "oos_cagr": float("nan"),
                "oos_sharpe": float("nan"),
            })
            continue
        rebased = eq_oos / eq_oos.iloc[0]
        n_days = len(rebased)
        years = max(1e-9, n_days / metrics.TRADING_DAYS_PER_YEAR)
        total_return = float(rebased.iloc[-1] - 1.0)
        cagr = float(rebased.iloc[-1] ** (1.0 / years) - 1.0)
        rets = rebased.pct_change().dropna()
        sharpe = (
            float(rets.mean() / rets.std() * np.sqrt(metrics.TRADING_DAYS_PER_YEAR))
            if len(rets) > 1 and rets.std() > 0 else 0.0
        )
        rows.append({
            "strategy": r.strategy, "exit": r.exit_label,
            "oos_total_return": total_return,
            "oos_cagr": cagr,
            "oos_sharpe": sharpe,
        })
    return pd.DataFrame(rows)
