"""LightGBM ranking model for insider cluster buys, with a purged
walk-forward validation harness whose job is to report honestly, including
when the honest report is "no edge here".

Design goals, in order:
  1. The cross-validation must not leak. Purged, embargoed, expanding-window
     folds (see make_purged_expanding_folds) are the load-bearing piece.
     tests/test_model.py proves the no-overlap invariant directly on
     synthetic data.
  2. Every metric is reported next to the incumbent baselines (the
     ten-percent-owner flag alone, conviction_score, and random scores),
     never in isolation.
  3. Nothing is fit on data outside its own training fold. Winsorization
     bounds are the main place this bites; they are recomputed every fold.
  4. Two independent leakage controls run automatically: a label-shuffle
     test (mandatory, described in run_label_shuffle_test) and a raw
     correlation guard (find_leaky_features / assert_no_leaky_features).

Input schema note: backtest.research.build_research_dataset persists
`entry_idx` in _IDENTITY_COLS and in each row dict (see backtest/research.py's
_build_event_row). Purging needs an exact trading-day integer index, not a
calendar date, so this module still refuses to run without one
(_validate_schema raises a clear error) -- that guard is now a defensive
check against a future regression, not a live gap against today's dataset.

SHAP note: this environment has no `shap` package installed and no network
access to install one. Feature stability and the ten-percent-owner
interaction analysis use LightGBM's own native TreeSHAP implementation
(Booster.predict(..., pred_contrib=True), part of lightgbm's C++ core, no
external package required) for per-feature attribution. True Shapley
*interaction* values (shap.TreeExplainer.shap_interaction_values) are not
available here; run_shap_interaction_analysis documents the proxy it uses
instead and the gap is repeated in its output so a reader cannot miss it.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr, ttest_1samp

import insider_cluster_buys as ics
from backtest.research import _FEATURE_COLS as _RESEARCH_FEATURE_COLS
from backtest.research import _IDENTITY_COLS as _RESEARCH_IDENTITY_COLS

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema constants, imported from backtest.research so this module cannot
# silently drift from the dataset it is meant to consume.
# ---------------------------------------------------------------------------
FEATURE_COLS: list[str] = list(_RESEARCH_FEATURE_COLS)

TEN_PCT_OWNER_KEY = "ten_percent_owner"
if TEN_PCT_OWNER_KEY not in ics.DEFAULT_WEIGHTS:
    raise RuntimeError(
        f"{TEN_PCT_OWNER_KEY!r} is no longer a key in "
        "insider_cluster_buys.DEFAULT_WEIGHTS. The incumbent baseline this "
        "module benchmarks against needs updating."
    )
TEN_PCT_OWNER_COL = f"f_{TEN_PCT_OWNER_KEY}"

# entry_idx is required for purging but is not (yet) part of
# backtest.research's identity columns -- see the module docstring.
REQUIRED_IDENTITY_COLS: list[str] = list(_RESEARCH_IDENTITY_COLS) + ["entry_idx"]

PRIMARY_HORIZON = 63
LABEL_COL = f"adj_{PRIMARY_HORIZON}"
TAIL_THRESH = 0.20


def label_col_for_horizon(horizon: int) -> str:
    """The adj_<horizon> forward-return column fit_and_validate actually
    fits and validates against for a given horizon.

    Added alongside the fix for a bug where fit_and_validate's `horizon`
    argument controlled only the purge/embargo rule (via
    make_purged_expanding_folds) while every model it fit was silently
    still trained on the module constant LABEL_COL ("adj_63"), regardless
    of what horizon was requested -- e.g. `run_research.py --horizon 21`
    purged as if predicting 21 trading days out but still trained on and
    scored against adj_63. fit_and_validate now calls this at the top of
    the function and uses the result everywhere it used to hardcode
    LABEL_COL.

    horizon=PRIMARY_HORIZON (63, the default everywhere) resolves to
    exactly LABEL_COL ("adj_63"), so every existing default-horizon call
    site is unaffected byte-for-byte."""
    return f"adj_{horizon}"

DEFAULT_N_FOLDS = 5
DEFAULT_EMBARGO_DAYS = 63
WINSOR_LO_PCT = 0.01
WINSOR_HI_PCT = 0.99

# A fold needs at least this many training rows before this module fits
# a model on it. A fold below this count is skipped, not fit.
#
# The value is 50. DEFAULT_LGBM_PARAMS sets min_child_samples=20. A tree
# needs at least 20 rows in one leaf to make a single split. A fold near
# that floor can make at most one or two real splits. Winsorization uses
# the 1st and 99th percentile of the training label. That percentile
# estimate is unstable on a small sample. 50 rows sit safely above the
# single-split floor. 50 rows are still a small share of a typical fold.
# This value is a judgment call, not a statistical proof. A caller may
# pass a different value.
MIN_FOLD_TRAIN_ROWS = 50

# Number of independent shuffle draws run_label_shuffle_test uses to build
# its empirical null distribution of the per-fold-averaged IC.
#
# A real measurement on a 1,172-row dataset used 40 draws and found a
# null with stdev near 0.04. One old single-draw run landed at -0.11,
# about 2.3 standard deviations out, which read as a leak under the old
# fixed |IC| >= 0.10 rule. It was not a leak. It was one draw from a wide
# distribution. Twenty draws bring the standard error of the null mean
# down to about 0.04 / sqrt(20) = 0.009, and let the one-sided
# permutation p-value this test reports resolve down to 1 / 21 = 0.048,
# fine enough to compare against SHUFFLE_TEST_ALPHA. Each draw costs one
# full pass over every fold that ran, so 20 draws cost about 20 times one
# fold pass. This is a deliberate trade of cost against statistical
# power, not a proof. A caller with a slow dataset may pass a smaller
# n_seeds, at the cost of a coarser null.
DEFAULT_N_SHUFFLE_SEEDS = 20

# One-sided permutation-test significance level for the label-shuffle
# test. The real model's IC must sit at or above the (1 - SHUFFLE_TEST_ALPHA)
# tail of the null before this module calls the result "extreme". 0.05 is
# the usual convention for this kind of test. It is not derived from this
# dataset.
SHUFFLE_TEST_ALPHA = 0.05

PRECISION_KS: tuple[int, ...] = (10, 25, 50, 100)
DECILE_BINS = 10
RANDOM_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
STABLE_FRAC = 0.8  # feature counts as stable if its sign agrees in >= 80% of folds

INTERACTION_CANDIDATE_COLS: tuple[str, ...] = (
    "x_buy_value_to_adv", "x_n_insiders", "x_drawdown_63", "x_drawdown_252",
)

DEFAULT_LGBM_PARAMS: dict = dict(
    n_estimators=200,
    num_leaves=15,
    min_child_samples=20,
    learning_rate=0.05,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=0,
    verbosity=-1,
)


# ---------------------------------------------------------------------------
# 0. Loading and schema validation
# ---------------------------------------------------------------------------
def _validate_schema(df: pd.DataFrame, label_col: str = LABEL_COL) -> None:
    """Raise a clear error rather than fail deep inside a fold loop.

    Checked eagerly: identity columns (including entry_idx, see the module
    docstring), every x_ feature column, the ten-percent-owner legacy flag,
    conviction_score, and the label column (`label_col`, LABEL_COL /
    "adj_63" by default -- fit_and_validate passes whatever
    label_col_for_horizon(horizon) resolves to).
    """
    missing_identity = [c for c in REQUIRED_IDENTITY_COLS if c not in df.columns]
    if missing_identity:
        raise ValueError(
            f"research dataset is missing identity column(s) {missing_identity}. "
            "entry_idx in particular is not currently written by "
            "backtest.research.build_research_dataset (it is computed per row "
            "but never stored -- see backtest/research.py's _build_event_row). "
            "Add it to _IDENTITY_COLS and the row dict before running this "
            "harness on a real dataset. Purging needs an exact trading-day "
            "index and must not guess one from event_day."
        )
    missing_features = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_features:
        raise ValueError(f"research dataset is missing x_ feature column(s): {missing_features}")
    if TEN_PCT_OWNER_COL not in df.columns:
        raise ValueError(f"research dataset is missing the legacy flag column {TEN_PCT_OWNER_COL!r}")
    if "conviction_score" not in df.columns:
        raise ValueError("research dataset is missing conviction_score")
    if label_col not in df.columns:
        raise ValueError(f"research dataset is missing label column {label_col!r}")


def load_research_dataset(path: str) -> pd.DataFrame:
    """Read a research-dataset parquet and validate its schema before
    returning it. Raises ValueError with a specific, actionable message on
    any schema mismatch rather than letting a downstream fold fail opaquely.
    """
    df = pd.read_parquet(path)
    _validate_schema(df)
    log.info("load_research_dataset: %s (shape=%s)", path, df.shape)
    return df


# ---------------------------------------------------------------------------
# 1. Purged, embargoed, expanding-window walk-forward folds
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Fold:
    """One walk-forward split. train_idx / test_idx hold df.index values
    (not positions), so callers use df.loc[fold.train_idx].

    The actual retained training row count is len(train_idx). This
    equals n_candidate_train - n_purged - n_embargoed. The purge and
    embargo masks are disjoint by construction. n_embargoed excludes
    rows the purge mask already dropped. No extra field is needed to
    know the retained count cheaply. len(train_idx) can be 0. A caller
    must check for that before it fits a model on train_idx."""

    fold_id: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    test_start_entry_idx: int
    n_candidate_train: int   # train rows before purge/embargo removal
    n_purged: int             # dropped because their label window reaches test start
    n_embargoed: int          # dropped by the embargo gap alone (excludes rows purge already dropped)


def make_purged_expanding_folds(
    df: pd.DataFrame,
    n_folds: int = DEFAULT_N_FOLDS,
    horizon: int = PRIMARY_HORIZON,
    embargo: int = DEFAULT_EMBARGO_DAYS,
    entry_idx_col: str = "entry_idx",
    event_day_col: str = "event_day",
) -> list[Fold]:
    """Chronological, expanding-window, purged, embargoed folds.

    Rows are ordered by (event_day, entry_idx), then cut into n_folds + 1
    contiguous, near-equal blocks. Fold k trains on blocks 0..k and tests on
    block k + 1, so the train window always precedes the test window and
    grows with each fold.

    Two independent filters remove rows from a fold's train set:

      purge: a train row is dropped if its label window, defined as
      [entry_idx, entry_idx + horizon], reaches or crosses the test block's
      first entry_idx. Its label is still "open" when the test period
      starts, so training on it would leak test-period price action into
      the model's target.

      embargo: a train row is dropped if its entry_idx sits within
      `embargo` trading days of the test block's first entry_idx, whether
      or not its own label window has closed. This adds a hard buffer
      around the boundary independent of the label horizon, guarding
      against short-range serial correlation (shared owner/issuer history
      features, sector moves) that a horizon-only purge would not catch.

    Both rules key off entry_idx, an integer trading-day position -- label
    windows are defined in trading days, not calendar days.

    Returns a list of n_folds Fold objects. Raises ValueError if entry_idx
    is missing or if there are not enough rows for n_folds + 1 blocks.
    """
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1")
    if entry_idx_col not in df.columns:
        raise ValueError(f"make_purged_expanding_folds: missing {entry_idx_col!r} column")
    if event_day_col not in df.columns:
        raise ValueError(f"make_purged_expanding_folds: missing {event_day_col!r} column")

    order = df.sort_values([event_day_col, entry_idx_col], kind="mergesort").index.to_numpy()
    n = len(order)
    if n < n_folds + 1:
        raise ValueError(f"need at least {n_folds + 1} rows for {n_folds} folds, got {n}")

    entry_idx_sorted = df.loc[order, entry_idx_col].to_numpy(dtype=float)
    if np.isnan(entry_idx_sorted).any():
        raise ValueError("make_purged_expanding_folds: entry_idx has null values")
    entry_idx_sorted = entry_idx_sorted.astype(np.int64)

    blocks = np.array_split(np.arange(n), n_folds + 1)

    # Warn here, before any fold is built or any model is fit. A block
    # that spans fewer trading days than horizon cannot give a full
    # label window of history. Any fold that trains on this block will
    # lose most or all of its candidate rows to purge. The earliest fold
    # to use a given block feels this the most, because it trains on
    # that block alone.
    #
    # This loop checks blocks 0 through n_folds - 1. It skips the last
    # block, at index n_folds. That block is always a test block. It
    # never trains a fold by itself.
    for b in range(n_folds):
        block_entries = entry_idx_sorted[blocks[b]]
        if len(block_entries) == 0:
            continue
        span = int(block_entries.max() - block_entries.min())
        if span < horizon:
            log.warning(
                "make_purged_expanding_folds: block %d spans %d trading "
                "day(s), entry_idx %d to %d. This is below horizon=%d. "
                "Fold %d trains on this block alone, so its candidate "
                "rows may be purged in full. Expect fold %d to run with "
                "few or zero training rows.",
                b, span, int(block_entries.min()), int(block_entries.max()),
                horizon, b, b,
            )

    folds: list[Fold] = []
    for k in range(n_folds):
        train_positions = np.concatenate(blocks[: k + 1])
        test_positions = blocks[k + 1]

        test_start = int(entry_idx_sorted[test_positions].min())
        train_entry_idx = entry_idx_sorted[train_positions]

        purge_mask = (train_entry_idx + horizon) >= test_start
        embargo_mask = (test_start - train_entry_idx) <= embargo
        keep_mask = ~(purge_mask | embargo_mask)

        kept_positions = train_positions[keep_mask]

        folds.append(Fold(
            fold_id=k,
            train_idx=order[kept_positions],
            test_idx=order[test_positions],
            test_start_entry_idx=test_start,
            n_candidate_train=int(len(train_positions)),
            n_purged=int(purge_mask.sum()),
            n_embargoed=int((embargo_mask & ~purge_mask).sum()),
        ))
    return folds


# ---------------------------------------------------------------------------
# 2. Winsorization (train-fold only)
# ---------------------------------------------------------------------------
def _winsor_bounds(train_y: pd.Series, lo_pct: float = WINSOR_LO_PCT, hi_pct: float = WINSOR_HI_PCT) -> tuple[float, float]:
    lo = float(train_y.quantile(lo_pct))
    hi = float(train_y.quantile(hi_pct))
    return lo, hi


# ---------------------------------------------------------------------------
# 3. Baseline scorers
# ---------------------------------------------------------------------------
def ten_pct_owner_score(df: pd.DataFrame) -> pd.Series:
    """The incumbent to beat: the ten-percent-owner flag used alone."""
    return df[TEN_PCT_OWNER_COL].astype(float)


def conviction_score_baseline(df: pd.DataFrame) -> pd.Series:
    """The current hand-tuned score."""
    return df["conviction_score"].astype(float)


def random_score(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n)


# ---------------------------------------------------------------------------
# 4. Metrics
# ---------------------------------------------------------------------------
def spearman_ic(score: pd.Series, label: pd.Series) -> float:
    """Spearman rank correlation of score vs label. NaN if fewer than 3
    valid pairs or if either series has zero variance."""
    mask = score.notna() & label.notna()
    if mask.sum() < 3:
        return float("nan")
    s = score[mask]
    y = label[mask]
    if s.nunique() < 2 or y.nunique() < 2:
        return float("nan")
    rho, _ = spearmanr(s, y)
    return float(rho)


def precision_at_k(score: pd.Series, label: pd.Series, k: int, hit_thresh: float = 0.0) -> float:
    """Fraction of the top-k scored rows with label > hit_thresh. NaN if
    fewer than k valid rows are available -- a k=100 precision on a 20-row
    fold is not a real number and should read as missing, not zero."""
    mask = score.notna() & label.notna()
    s = score[mask]
    y = label[mask]
    if len(s) < k:
        return float("nan")
    top_idx = s.sort_values(ascending=False).index[:k]
    return float((y.loc[top_idx] > hit_thresh).mean())


def precision_base_rate(score: pd.Series, label: pd.Series, k: int, hit_thresh: float = 0.0) -> float:
    """The no-skill value that precision_at_k should be judged against:
    P(label > hit_thresh) over the SAME mask precision_at_k uses (score and
    label both non-null), NOT over the whole fold. Precision at k is a hit
    rate, and a hit rate's null is the base rate, not zero -- unlike a rank
    correlation like IC, whose null genuinely is zero. Returns NaN under the
    same fewer-than-k-valid-rows condition as precision_at_k, so the two are
    always comparable (either both real or both missing)."""
    mask = score.notna() & label.notna()
    y = label[mask]
    if len(y) < k:
        return float("nan")
    return float((y > hit_thresh).mean())


def _decile_table(score: pd.Series, label: pd.Series, n_bins: int = DECILE_BINS) -> pd.DataFrame:
    """decile_rank (0 = lowest score) x n x mean_label, computed by qcut on
    `score`. A near-binary score (e.g. the ten-percent-owner flag) will
    naturally collapse to 2 bins under duplicates='drop' -- that is the
    honest picture for a binary score, not a bug to paper over."""
    mask = score.notna() & label.notna()
    s = score[mask]
    y = label[mask]
    cols = ["decile_rank", "n", "mean_label"]
    if len(s) < 2 or s.nunique() < 2:
        return pd.DataFrame(columns=cols)
    try:
        bins = pd.qcut(s, n_bins, duplicates="drop")
    except ValueError:
        return pd.DataFrame(columns=cols)
    codes = bins.cat.codes
    tmp = pd.DataFrame({"decile_rank": codes.to_numpy(), "label": y.to_numpy()})
    tmp = tmp[tmp["decile_rank"] >= 0]
    out = (
        tmp.groupby("decile_rank")["label"]
        .agg(n="count", mean_label="mean")
        .reset_index()
        .sort_values("decile_rank")
        .reset_index(drop=True)
    )
    return out[cols]


def _tail_decile_table(score: pd.Series, label: pd.Series, thresh: float = TAIL_THRESH, n_bins: int = DECILE_BINS) -> pd.DataFrame:
    """decile_rank x n x p_tail = P(label > thresh) within the bin."""
    mask = score.notna() & label.notna()
    s = score[mask]
    y = label[mask]
    cols = ["decile_rank", "n", "p_tail"]
    if len(s) < 2 or s.nunique() < 2:
        return pd.DataFrame(columns=cols)
    try:
        bins = pd.qcut(s, n_bins, duplicates="drop")
    except ValueError:
        return pd.DataFrame(columns=cols)
    codes = bins.cat.codes
    tmp = pd.DataFrame({"decile_rank": codes.to_numpy(), "is_tail": (y.to_numpy() > thresh).astype(float)})
    tmp = tmp[tmp["decile_rank"] >= 0]
    out = (
        tmp.groupby("decile_rank")["is_tail"]
        .agg(n="count", p_tail="mean")
        .reset_index()
        .sort_values("decile_rank")
        .reset_index(drop=True)
    )
    return out[cols]


def _aggregate_by_rank(tables: list[pd.DataFrame], value_col: str) -> pd.DataFrame:
    """Stack a list of per-fold decile/tail-decile tables and aggregate
    value_col by decile_rank: mean and t-stat across the folds that had
    that rank present. decile_rank is an ordinal position (0 = lowest
    score in that fold), comparable across folds even though the folds'
    exact score cutpoints differ."""
    non_empty = [t for t in tables if not t.empty]
    if not non_empty:
        return pd.DataFrame(columns=["decile_rank", "n_folds", "mean_value", "tstat", "pvalue"])
    stacked = pd.concat(non_empty, ignore_index=True)
    rows = []
    for rank_, sub in stacked.groupby("decile_rank"):
        vals = sub[value_col].dropna().to_numpy()
        row = {"decile_rank": int(rank_), "n_folds": int(len(vals)), "mean_value": float(np.mean(vals)) if len(vals) else float("nan")}
        if len(vals) >= 2 and np.std(vals, ddof=1) > 0:
            t, p = ttest_1samp(vals, 0.0)
            row["tstat"] = float(t)
            row["pvalue"] = float(p)
        else:
            row["tstat"] = float("nan")
            row["pvalue"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows).sort_values("decile_rank").reset_index(drop=True)


def _monotonicity(mean_by_rank: pd.Series) -> dict:
    vals = mean_by_rank.to_numpy(dtype=float)
    vals = vals[~np.isnan(vals)]
    if len(vals) < 2:
        return {"is_monotonic_nondecreasing": None, "n_violations": None, "rank_corr": float("nan")}
    diffs = np.diff(vals)
    n_violations = int((diffs < 0).sum())
    rank_corr = float(spearmanr(np.arange(len(vals)), vals)[0]) if len(vals) >= 3 else float("nan")
    return {
        "is_monotonic_nondecreasing": n_violations == 0,
        "n_violations": n_violations,
        "rank_corr": rank_corr,
    }


# ---------------------------------------------------------------------------
# 5. Model fitting helpers
# ---------------------------------------------------------------------------
def _fit_regressor(
    train_df: pd.DataFrame, feature_cols: list[str], params: dict, label_col: str = LABEL_COL,
) -> tuple[lgb.LGBMRegressor, float, float]:
    lo, hi = _winsor_bounds(train_df[label_col])
    y_train = train_df[label_col].clip(lo, hi)
    model = lgb.LGBMRegressor(**params).fit(train_df[feature_cols], y_train)
    return model, lo, hi


def _fit_classifier(train_df: pd.DataFrame, feature_cols: list[str], target: pd.Series, params: dict) -> Optional[lgb.LGBMClassifier]:
    """Returns None if train has only one class -- callers fall back to a
    constant base-rate prediction rather than letting lightgbm raise."""
    if target.nunique() < 2:
        return None
    return lgb.LGBMClassifier(**params).fit(train_df[feature_cols], target)


def _predict_proba_positive(model: Optional[lgb.LGBMClassifier], X: pd.DataFrame, fallback_rate: float) -> np.ndarray:
    if model is None:
        return np.full(len(X), fallback_rate, dtype=float)
    proba = model.predict_proba(X)
    classes = list(model.classes_)
    if 1 in classes:
        return proba[:, classes.index(1)]
    return np.zeros(len(X), dtype=float)


# ---------------------------------------------------------------------------
# 6. Leakage controls
# ---------------------------------------------------------------------------
def find_leaky_features(
    df: pd.DataFrame, feature_cols: Optional[list[str]] = None, label_col: str = LABEL_COL, threshold: float = 0.95,
) -> list[tuple[str, float]]:
    """Every x_ feature that correlates (Spearman) with the label above
    |rho| = threshold. An empty list is the expected, healthy result."""
    feature_cols = feature_cols if feature_cols is not None else FEATURE_COLS
    label = df[label_col]
    offenders: list[tuple[str, float]] = []
    for col in feature_cols:
        if col not in df.columns:
            continue
        mask = df[col].notna() & label.notna()
        if mask.sum() < 3:
            continue
        s = df.loc[mask, col]
        y = label[mask]
        if s.nunique() < 2 or y.nunique() < 2:
            continue
        rho, _ = spearmanr(s, y)
        if rho == rho and abs(rho) > threshold:
            offenders.append((col, float(rho)))
    return offenders


def assert_no_leaky_features(
    df: pd.DataFrame, feature_cols: Optional[list[str]] = None, label_col: str = LABEL_COL, threshold: float = 0.95,
) -> None:
    """Raises ValueError, loudly, if any feature is essentially a copy of
    the label. Call this before trusting any other number in this module."""
    offenders = find_leaky_features(df, feature_cols, label_col, threshold)
    if offenders:
        raise ValueError(
            f"Leakage guard tripped: {len(offenders)} feature(s) correlate with "
            f"{label_col} above |rho|={threshold}: {offenders}. This usually means "
            "a feature encodes the forward return directly. Fix the feature "
            "before trusting any other metric in this run."
        )


def run_label_shuffle_test(
    df: pd.DataFrame,
    folds: list[Fold],
    feature_cols: list[str],
    params: dict,
    real_ic: float,
    n_seeds: int = DEFAULT_N_SHUFFLE_SEEDS,
    base_seed: int = 12345,
    label_col: str = LABEL_COL,
) -> dict:
    """Build an empirical null distribution of the label-shuffle IC, then
    test the real model against it.

    A single shuffle draw is a sample of size one. A real measurement on
    a 1,172-row dataset used 40 independent draws. The null had a mean
    near -0.02 and a standard deviation near 0.04. One old single-draw
    run landed at -0.11, about 2.3 standard deviations out. Under the old
    fixed rule (fail if abs(ic) >= 0.10), that one draw read as a leak.
    It was not a leak. It was one draw from a wide distribution. Zero of
    the 40 draws crossed 0.10 in either direction, so the fixed threshold
    gave almost no information about this dataset.

    This function takes n_seeds independent draws instead of one. Each
    draw shuffles the label once, refits the regressor through every
    fold in `folds`, and records the mean IC across those folds. `folds`
    must already exclude any fold that fit_and_validate skipped for a low
    row count. The statistic is the mean IC per fold, the same choice the
    main summary table makes, not a pooled IC across all out-of-fold
    rows. Pooling can create a false correlation from level shifts
    between folds alone, with no leak present.

    The old test asked a weak question. Does a model trained on noise
    show no power? That is true almost all the time and proves little.
    This function asks a stronger question instead. Does the REAL
    model's IC stand out from the noise floor built from this same data
    and fold structure? A model that cannot clear that floor has no shown
    edge, no matter how large its raw IC looks. Pass `real_ic`, the real
    (non-shuffled) regressor's mean per-fold IC over the same fold list,
    to make this comparison.

    Returns a dict with the full null distribution (mean, stdev, min,
    max, and the 5th, 25th, 50th, 75th, and 95th percentiles), the real
    IC's z-score and one-sided permutation p-value against that null, and
    a `passed` flag. `passed` is True only when real_ic is a number and
    its p-value is below SHUFFLE_TEST_ALPHA. That means few or no
    shuffled-label draws reached as high an IC as the real model did.
    `passed` is False when the real model's IC cannot be told apart from
    noise. Treat a False result as a stop-ship finding, not a modeling
    nuance.

    The dict also carries `own_shuffle_ic_mean`, a secondary check. It
    measures, on each draw, whether the out-of-fold prediction correlates
    with that same draw's own shuffled label at the held-out test rows.
    A high value here would point at a fold-construction bug, such as
    train and test rows sharing information, rather than at feature
    quality. The shuffled label at test time is never seen during
    training unless folds overlap. This value does not gate `passed`. It
    is reported so a reader can rule out that specific failure mode on
    their own.
    """
    if n_seeds < 2:
        raise ValueError("run_label_shuffle_test: n_seeds must be at least 2 to build a null distribution")

    # A one-sided permutation p-value can only be as fine as 1 / (n_seeds + 1).
    # If that floor sits at or above SHUFFLE_TEST_ALPHA, no real_ic, however
    # extreme, can ever pass. Warn here, before running a single model fit,
    # so a caller with a small n_seeds does not spend the runtime only to
    # find the test could not have passed.
    finest_p = 1.0 / (n_seeds + 1)
    if finest_p >= SHUFFLE_TEST_ALPHA:
        # Derive the advice with the SAME comparison the guard above uses,
        # rather than a closed form. ceil(1/alpha) - 1 is off by one: at
        # alpha=0.05 it advises 19, whose finest p is exactly 0.0500, which
        # is not < 0.05, so a caller who followed the advice would spend a
        # second full run and fail again. Searching also sidesteps the
        # float edge, where 1.0/alpha can land just under an integer.
        min_n_seeds = next(
            n for n in range(2, 100000) if 1.0 / (n + 1) < SHUFFLE_TEST_ALPHA
        )
        log.warning(
            "run_label_shuffle_test: n_seeds=%d cannot resolve a p-value below "
            "SHUFFLE_TEST_ALPHA=%.3f. The finest p-value %d draws can produce "
            "is %.3f. A real model that beats every null draw will still be "
            "reported as FAILED. Raise n_seeds to at least %d.",
            n_seeds, SHUFFLE_TEST_ALPHA, n_seeds, finest_p, min_n_seeds,
        )

    null_ic_vs_true: list[float] = []
    null_ic_vs_own_shuffle: list[float] = []

    for i in range(n_seeds):
        rng = np.random.default_rng(base_seed + i)
        shuffled_label = pd.Series(rng.permutation(df[label_col].to_numpy()), index=df.index)

        fold_ic_true: list[float] = []
        fold_ic_shuf: list[float] = []
        for fold in folds:
            train_df = df.loc[fold.train_idx]
            test_df = df.loc[fold.test_idx]
            y_train_shuf = shuffled_label.loc[train_df.index]
            lo, hi = _winsor_bounds(y_train_shuf)
            y_train_shuf_w = y_train_shuf.clip(lo, hi)
            model = lgb.LGBMRegressor(**params).fit(train_df[feature_cols], y_train_shuf_w)
            pred = pd.Series(model.predict(test_df[feature_cols]), index=test_df.index)
            fold_ic_true.append(spearman_ic(pred, df.loc[test_df.index, label_col]))
            fold_ic_shuf.append(spearman_ic(pred, shuffled_label.loc[test_df.index]))

        valid_true = [x for x in fold_ic_true if x == x]
        valid_shuf = [x for x in fold_ic_shuf if x == x]
        null_ic_vs_true.append(float(np.mean(valid_true)) if valid_true else float("nan"))
        null_ic_vs_own_shuffle.append(float(np.mean(valid_shuf)) if valid_shuf else float("nan"))

    null_arr = np.array([x for x in null_ic_vs_true if x == x], dtype=float)
    n_valid_draws = int(len(null_arr))
    if n_valid_draws < 2:
        raise ValueError(
            f"run_label_shuffle_test: only {n_valid_draws} of {n_seeds} shuffle "
            "draws produced a usable per-fold IC. This module cannot build a "
            "null distribution from fewer than 2 draws. This usually means a "
            "fold has too few test rows, or a near-constant score."
        )

    null_mean = float(np.mean(null_arr))
    null_stdev = float(np.std(null_arr, ddof=1))
    null_percentiles = {p: float(np.percentile(null_arr, p)) for p in (5, 25, 50, 75, 95)}

    own_shuffle_arr = np.array([x for x in null_ic_vs_own_shuffle if x == x], dtype=float)
    own_shuffle_mean = float(np.mean(own_shuffle_arr)) if len(own_shuffle_arr) else float("nan")

    real_ic_is_number = real_ic == real_ic
    if real_ic_is_number:
        n_as_extreme = int(np.sum(null_arr >= real_ic))
        p_value = float((n_as_extreme + 1) / (n_valid_draws + 1))
        z_score = float((real_ic - null_mean) / null_stdev) if null_stdev > 0 else float("nan")
    else:
        n_as_extreme = n_valid_draws
        p_value = float("nan")
        z_score = float("nan")

    passed = bool(real_ic_is_number and p_value == p_value and p_value < SHUFFLE_TEST_ALPHA)

    if passed:
        note = (
            f"The real model's IC ({real_ic:.4f}) beats {n_valid_draws - n_as_extreme} "
            f"of {n_valid_draws} shuffled-label draws (one-sided p={p_value:.4f}, "
            f"z={z_score:.2f}, against a null with mean={null_mean:.4f}, "
            f"stdev={null_stdev:.4f}). It stands out from the noise floor built "
            "from this same dataset and fold structure."
        )
    else:
        reason = (
            "real_ic is not a number" if not real_ic_is_number
            else f"one-sided p={p_value:.4f} is not below alpha={SHUFFLE_TEST_ALPHA}"
        )
        note = (
            f"FAILED: the real model's IC does not stand out from noise ({reason}). "
            f"Null over {n_valid_draws} shuffled-label draws: mean={null_mean:.4f}, "
            f"stdev={null_stdev:.4f}. Treat every other metric in this run as "
            "untrustworthy until this is understood."
        )

    return {
        "real_ic": float(real_ic) if real_ic_is_number else float("nan"),
        "n_seeds": n_seeds,
        "n_valid_draws": n_valid_draws,
        "null_mean": null_mean,
        "null_stdev": null_stdev,
        "null_min": float(np.min(null_arr)),
        "null_max": float(np.max(null_arr)),
        "null_p5": null_percentiles[5],
        "null_p25": null_percentiles[25],
        "null_p50": null_percentiles[50],
        "null_p75": null_percentiles[75],
        "null_p95": null_percentiles[95],
        "null_ic_values": null_arr.tolist(),
        "z_score": z_score,
        "p_value": p_value,
        "own_shuffle_ic_mean": own_shuffle_mean,
        "passed": passed,
        "note": note,
    }


# ---------------------------------------------------------------------------
# 7. SHAP-based feature stability and the ten-percent-owner interaction test
# ---------------------------------------------------------------------------
def _shap_contrib(model: lgb.LGBMRegressor, X: pd.DataFrame) -> np.ndarray:
    """Native lightgbm TreeSHAP contributions, shape (n, len(cols) + 1),
    last column is the expected-value baseline. No external `shap` package
    involved -- pred_contrib=True is implemented in lightgbm's own C++
    core."""
    return model.predict(X, pred_contrib=True)


def _feature_directions(shap_matrix: np.ndarray, X_test: pd.DataFrame, feature_cols: list[str]) -> dict[str, float]:
    """Per-feature sign of Spearman corr(feature value, that feature's own
    SHAP contribution) on one fold's test rows. +1 means higher feature
    values push the prediction up in this fold, -1 means down, NaN means
    not computable (near-constant feature or contribution column)."""
    directions: dict[str, float] = {}
    for i, col in enumerate(feature_cols):
        fvals = X_test[col].to_numpy(dtype=float)
        svals = shap_matrix[:, i]
        mask = ~np.isnan(fvals) & ~np.isnan(svals)
        if mask.sum() < 5 or np.nanstd(fvals[mask]) == 0 or np.nanstd(svals[mask]) == 0:
            directions[col] = float("nan")
            continue
        rho, _ = spearmanr(fvals[mask], svals[mask])
        directions[col] = float(np.sign(rho)) if rho == rho else float("nan")
    return directions


def _feature_stability_table(direction_by_fold: list[dict[str, float]], feature_cols: list[str], stable_frac: float = STABLE_FRAC) -> pd.DataFrame:
    n_folds = len(direction_by_fold)
    required = math.ceil(stable_frac * n_folds)
    rows = []
    for col in feature_cols:
        signs = [d.get(col, float("nan")) for d in direction_by_fold]
        n_pos = sum(1 for s in signs if s == 1.0)
        n_neg = sum(1 for s in signs if s == -1.0)
        n_agree = max(n_pos, n_neg)
        dominant = "positive" if n_pos >= n_neg else "negative"
        row = {"feature": col, "n_folds": n_folds, "n_positive": n_pos, "n_negative": n_neg, "dominant_sign": dominant, "is_stable": bool(n_agree >= required)}
        for k, s in enumerate(signs):
            row[f"sign_fold{k}"] = s
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.sort_values(["is_stable", "n_positive", "n_negative"], ascending=[False, False, False]).reset_index(drop=True)


def run_shap_interaction_analysis(
    df: pd.DataFrame, folds: list[Fold], feature_cols: list[str], params: dict,
    candidate_cols: tuple[str, ...] = INTERACTION_CANDIDATE_COLS,
    label_col: str = LABEL_COL,
) -> tuple[pd.DataFrame, dict]:
    """Tests the user's central hypothesis: is the ten-percent-owner flag
    more useful in combination with other signals than alone?

    Fits a "combo" regressor per fold on FEATURE_COLS plus the raw
    ten-percent-owner flag, then, restricted to rows where the flag fired,
    checks whether the flag's own out-of-fold SHAP contribution correlates
    with each candidate feature's value. A significant, cross-fold-
    plausible correlation would mean the flag's marginal usefulness is
    conditional on that feature rather than constant.

    This is a SHAP-value stratification proxy, not a formal Shapley
    interaction index. shap.TreeExplainer.shap_interaction_values would
    give the exact quantity the hypothesis asks about; the `shap` package
    is not installed here and there is no network access to install it.
    LightGBM's own pred_contrib gives per-feature (not per-feature-pair)
    attributions, which is what this function uses. Treat the verdict
    below as directional evidence, not a definitive interaction estimate.
    """
    combo_cols = feature_cols + [TEN_PCT_OWNER_COL]
    tpo_pos = combo_cols.index(TEN_PCT_OWNER_COL)

    oof_shap_tpo = pd.Series(np.nan, index=df.index, dtype=float)
    oof_pred = pd.Series(np.nan, index=df.index, dtype=float)

    for fold in folds:
        train_df = df.loc[fold.train_idx]
        test_df = df.loc[fold.test_idx]
        lo, hi = _winsor_bounds(train_df[label_col])
        y_train = train_df[label_col].clip(lo, hi)
        model = lgb.LGBMRegressor(**params).fit(train_df[combo_cols], y_train)
        contrib = _shap_contrib(model, test_df[combo_cols])
        oof_shap_tpo.loc[test_df.index] = contrib[:, tpo_pos]
        oof_pred.loc[test_df.index] = model.predict(test_df[combo_cols])

    fired_mask = df[TEN_PCT_OWNER_COL] == 1
    rows = []
    significant: list[str] = []
    for cand in candidate_cols:
        if cand not in df.columns:
            rows.append({"candidate": cand, "n_fired": 0, "corr": float("nan"), "p_value": float("nan"), "note": "column not present"})
            continue
        sub_idx = df.index[fired_mask & df[cand].notna() & oof_shap_tpo.notna()]
        if len(sub_idx) < 30:
            rows.append({"candidate": cand, "n_fired": int(len(sub_idx)), "corr": float("nan"), "p_value": float("nan"), "note": "insufficient overlapping data"})
            continue
        x = df.loc[sub_idx, cand].to_numpy(dtype=float)
        s = oof_shap_tpo.loc[sub_idx].to_numpy(dtype=float)
        if np.nanstd(x) == 0 or np.nanstd(s) == 0:
            rows.append({"candidate": cand, "n_fired": int(len(sub_idx)), "corr": float("nan"), "p_value": float("nan"), "note": "zero variance"})
            continue
        rho, pval = spearmanr(x, s)
        rows.append({"candidate": cand, "n_fired": int(len(sub_idx)), "corr": float(rho), "p_value": float(pval), "note": ""})
        if pval == pval and pval < 0.05 and abs(rho) > 0.10:
            significant.append(cand)

    interaction_report = pd.DataFrame(rows, columns=["candidate", "n_fired", "corr", "p_value", "note"])

    combo_ic = spearman_ic(oof_pred, df[label_col])
    alone_ic = spearman_ic(ten_pct_owner_score(df), df[label_col])

    any_testable = interaction_report["corr"].notna().any()
    if not any_testable:
        verdict = (
            "too weak to say: no candidate interaction feature had enough "
            "overlapping fired-flag data to test."
        )
    elif significant:
        verdict = (
            f"data supports a conditional-usefulness hypothesis for the "
            f"ten-percent-owner flag against {significant}: its marginal "
            f"SHAP contribution correlates with those features among rows "
            f"where the flag fired. combo-model OOF IC={combo_ic:.4f} vs "
            f"flag-alone IC={alone_ic:.4f}."
        )
    else:
        verdict = (
            "data does not support the conditional-usefulness hypothesis: "
            "no candidate feature showed a significant correlation with the "
            f"flag's marginal SHAP contribution. combo-model OOF IC="
            f"{combo_ic:.4f} vs flag-alone IC={alone_ic:.4f}."
        )

    return interaction_report, {
        "verdict": verdict,
        "combo_ic": combo_ic,
        "alone_ic": alone_ic,
        "n_significant": len(significant),
        "significant_candidates": significant,
        "method_note": (
            "SHAP-value stratification proxy using LightGBM's native "
            "pred_contrib TreeSHAP, not a formal Shapley interaction index. "
            "The `shap` package is unavailable in this environment (not "
            "installed, no network access to install it)."
        ),
    }


# ---------------------------------------------------------------------------
# 8. Main entry point
# ---------------------------------------------------------------------------
@dataclass
class ValidationResult:
    """folds holds every Fold that make_purged_expanding_folds built.
    This includes folds that were skipped. skipped_folds records why
    each skip happened. Every metric table in this result counts only
    folds that ran. This covers fold_metrics, summary_metrics,
    decile_tables, tail_decile_tables, feature_stability, and
    oof_scores. A skipped fold adds nothing to any mean, count, or
    t-stat. n_folds_run plus n_folds_skipped always equals len(folds)."""

    folds: list[Fold]
    fold_metrics: pd.DataFrame
    summary_metrics: pd.DataFrame
    decile_tables: dict[str, pd.DataFrame]
    tail_decile_tables: dict[str, pd.DataFrame]
    feature_stability: pd.DataFrame
    label_shuffle: dict
    leakage_offenders: list
    interaction_report: pd.DataFrame
    interaction_summary: dict
    oof_scores: pd.DataFrame
    n_folds_run: int = 0
    n_folds_skipped: int = 0
    skipped_folds: list[dict] = field(default_factory=list)
    models: dict = field(default_factory=dict)
    feature_cols: list[str] = field(default_factory=list)
    config: dict = field(default_factory=dict)


_MODEL_NAMES = ("regressor", "classifier", "tail_classifier")
_BASELINE_NAMES_STATIC = ("ten_pct_owner", "conviction_score")


def _all_score_names(random_seeds: tuple[int, ...]) -> list[str]:
    return list(_MODEL_NAMES) + list(_BASELINE_NAMES_STATIC) + [f"random_seed{s}" for s in random_seeds]


def _summarize_fold_metrics(
    fold_metrics: pd.DataFrame, value_cols: list[str], precision_cols: set[str]
) -> pd.DataFrame:
    """Aggregate per-fold metrics into one summary row per model: mean,
    t-stat, and p-value across folds.

    `ic` is tested against 0.0 -- a rank correlation with no skill IS zero.

    Each column in `precision_cols` (named f"precision_{k}") is a hit rate,
    not a rank correlation. Its no-skill value is the per-fold base rate
    P(label > hit_thresh), which the fold loop records alongside it in a
    f"{col}_base_rate" column. For these columns the t-stat and p-value
    test the EXCESS (precision - base_rate) against zero instead of raw
    precision against zero. `{col}_mean` still reports raw precision
    unchanged, and `{col}_base_mean` / `{col}_excess_mean` are added so a
    reader can see the raw value, what was subtracted, and the difference.

    Precision and base rate are paired fold-by-fold and dropped together
    BEFORE any arithmetic: `sub[[col, base_col]].dropna(how="any")` builds
    the paired array explicitly and drops a fold if EITHER side is NaN.
    Dropping each column independently and then subtracting would silently
    misalign folds whenever only one side is missing, producing a wrong
    t-stat rather than an obvious error.
    """
    summary_rows = []
    for name, sub in fold_metrics.groupby("model"):
        row: dict = {"model": name, "n_folds": int(len(sub))}
        for col in value_cols:
            vals = sub[col].dropna().to_numpy()
            row[f"{col}_mean"] = float(np.mean(vals)) if len(vals) else float("nan")
            if col in precision_cols:
                base_col = f"{col}_base_rate"
                paired = sub[[col, base_col]].dropna(how="any")
                base_vals = paired[base_col].to_numpy()
                excess_vals = paired[col].to_numpy() - base_vals
                row[f"{col}_base_mean"] = float(np.mean(base_vals)) if len(base_vals) else float("nan")
                row[f"{col}_excess_mean"] = float(np.mean(excess_vals)) if len(excess_vals) else float("nan")
                test_vals = excess_vals
            else:
                test_vals = vals
            if len(test_vals) >= 2 and np.std(test_vals, ddof=1) > 0:
                t, p = ttest_1samp(test_vals, 0.0)
                row[f"{col}_tstat"] = float(t)
                row[f"{col}_pvalue"] = float(p)
            else:
                row[f"{col}_tstat"] = float("nan")
                row[f"{col}_pvalue"] = float("nan")
        summary_rows.append(row)
    return pd.DataFrame(summary_rows).sort_values("model").reset_index(drop=True)


def fit_and_validate(
    df: pd.DataFrame,
    *,
    n_folds: int = DEFAULT_N_FOLDS,
    horizon: int = PRIMARY_HORIZON,
    embargo: int = DEFAULT_EMBARGO_DAYS,
    feature_cols: Optional[list[str]] = None,
    lgbm_params: Optional[dict] = None,
    random_seeds: tuple[int, ...] = RANDOM_SEEDS,
    run_shap_interactions: bool = True,
    label_shuffle_seed: int = 12345,
    n_shuffle_seeds: int = DEFAULT_N_SHUFFLE_SEEDS,
    leakage_threshold: float = 0.95,
    min_fold_train_rows: int = MIN_FOLD_TRAIN_ROWS,
    tail_thresh: float = TAIL_THRESH,
) -> ValidationResult:
    """Fit and validate the ranking model with purged walk-forward CV.

    Steps, in order:
      1. Validate schema.
      2. Run the leakage-correlation guard (raises if it trips).
      3. Drop rows without a usable label or entry_idx.
      4. Build purged, embargoed, expanding-window folds.
      5. Per fold: check the training row count against
         min_fold_train_rows. If it is too low, skip the fold. Log a
         warning and record the skip in ValidationResult.skipped_folds,
         then move to the next fold. This is expected behaviour, not a
         bug. A long horizon against a short early block causes it. See
         the warning make_purged_expanding_folds logs at fold-build
         time. Otherwise, fit the regressor (winsorized label_col), the
         binary classifier (label_col > 0), and the tail classifier
         (label_col > tail_thresh), all on FEATURE_COLS. label_col is
         label_col_for_horizon(horizon) -- "adj_63" at the default
         horizon=63, tracking whatever horizon is actually requested
         otherwise (see that function's docstring for the bug this
         fixes). Score every baseline on the same test rows. Collect
         IC, precision@k, decile and tail-decile tables, and
         per-feature SHAP-sign direction.
      6. If every fold got skipped, raise a clear error. A result with
         zero folds run must never look like a valid answer.
      7. Aggregate fold metrics into a summary table (mean + t-stat),
         counting only folds that ran.
      8. Run the label-shuffle leakage test. This compares the real
         regressor's mean per-fold IC (from step 5, over the folds that
         ran) against an empirical null built from n_shuffle_seeds
         independent label shuffles. See run_label_shuffle_test.
      9. Run the ten-percent-owner interaction analysis (optional).
      10. Refit a final set of models on the full valid dataset for
          persistence (this is a deployment artifact, not a CV fold --
          there is no future data left to hold out at deployment time).

    Returns a ValidationResult with everything a caller needs to write a
    report or inject out-of-fold scores into the portfolio engine.
    """
    label_col = label_col_for_horizon(horizon)
    _validate_schema(df, label_col=label_col)

    feature_cols = list(feature_cols) if feature_cols is not None else list(FEATURE_COLS)
    params = dict(DEFAULT_LGBM_PARAMS)
    if lgbm_params:
        params.update(lgbm_params)

    leakage_offenders = find_leaky_features(df, feature_cols, label_col, leakage_threshold)
    if leakage_offenders:
        raise ValueError(
            f"Leakage guard tripped: {len(leakage_offenders)} feature(s) correlate with "
            f"{label_col} above |rho|={leakage_threshold}: {leakage_offenders}."
        )

    df_valid = df.dropna(subset=[label_col, "entry_idx"]).copy()
    if len(df_valid) < n_folds + 1:
        raise ValueError(f"fit_and_validate: only {len(df_valid)} rows with a usable label -- need at least {n_folds + 1}")

    folds = make_purged_expanding_folds(df_valid, n_folds=n_folds, horizon=horizon, embargo=embargo)

    score_names = _all_score_names(random_seeds)
    fold_metric_rows: list[dict] = []
    decile_tables_per_fold: dict[str, list[pd.DataFrame]] = {name: [] for name in score_names}
    tail_decile_tables_per_fold: dict[str, list[pd.DataFrame]] = {name: [] for name in score_names}
    direction_by_fold: list[dict[str, float]] = []
    oof_rows: list[pd.DataFrame] = []
    skipped_folds: list[dict] = []

    for fold in folds:
        n_train = len(fold.train_idx)
        if n_train < min_fold_train_rows:
            reason = (
                f"only {n_train} training row(s) survived purge and embargo, "
                f"below min_fold_train_rows={min_fold_train_rows}"
            )
            log.warning(
                "fit_and_validate: skipping fold %d (%s). "
                "test_start_entry_idx=%d, n_candidate_train=%d, n_purged=%d, "
                "n_embargoed=%d. This fold contributes to no mean, count, or "
                "t-stat in this run.",
                fold.fold_id, reason, fold.test_start_entry_idx,
                fold.n_candidate_train, fold.n_purged, fold.n_embargoed,
            )
            skipped_folds.append({
                "fold_id": fold.fold_id,
                "n_train": n_train,
                "n_candidate_train": fold.n_candidate_train,
                "n_purged": fold.n_purged,
                "n_embargoed": fold.n_embargoed,
                "test_start_entry_idx": fold.test_start_entry_idx,
                "reason": reason,
            })
            continue

        train_df = df_valid.loc[fold.train_idx]
        test_df = df_valid.loc[fold.test_idx]
        label_test = test_df[label_col]

        reg_model, _, _ = _fit_regressor(train_df, feature_cols, params, label_col=label_col)
        reg_pred = pd.Series(reg_model.predict(test_df[feature_cols]), index=test_df.index)

        cls_target = (train_df[label_col] > 0).astype(int)
        cls_model = _fit_classifier(train_df, feature_cols, cls_target, params)
        cls_pred = pd.Series(
            _predict_proba_positive(cls_model, test_df[feature_cols], float(cls_target.mean())),
            index=test_df.index,
        )

        tail_target = (train_df[label_col] > tail_thresh).astype(int)
        tail_model = _fit_classifier(train_df, feature_cols, tail_target, params)
        tail_pred = pd.Series(
            _predict_proba_positive(tail_model, test_df[feature_cols], float(tail_target.mean())),
            index=test_df.index,
        )

        scores: dict[str, pd.Series] = {
            "regressor": reg_pred,
            "classifier": cls_pred,
            "tail_classifier": tail_pred,
            "ten_pct_owner": ten_pct_owner_score(test_df),
            "conviction_score": conviction_score_baseline(test_df),
        }
        for s in random_seeds:
            scores[f"random_seed{s}"] = pd.Series(random_score(len(test_df), seed=s * 100_000 + fold.fold_id), index=test_df.index)

        for name, score in scores.items():
            ic = spearman_ic(score, label_test)
            row = {"fold": fold.fold_id, "model": name, "n_test": len(test_df), "n_train": len(train_df), "ic": ic}
            for k in PRECISION_KS:
                row[f"precision_{k}"] = precision_at_k(score, label_test, k)
                row[f"precision_{k}_base_rate"] = precision_base_rate(score, label_test, k)
            fold_metric_rows.append(row)
            decile_tables_per_fold[name].append(_decile_table(score, label_test))
            tail_decile_tables_per_fold[name].append(_tail_decile_table(score, label_test, thresh=tail_thresh))

        oof_fold = pd.DataFrame({
            "fold": fold.fold_id,
            "ticker": test_df["ticker"],
            "event_day": test_df["event_day"],
            "entry_day": test_df["entry_day"],
            "entry_idx": test_df["entry_idx"],
            label_col: label_test,
            "oof_regressor": reg_pred,
            "oof_classifier": cls_pred,
            "oof_tail_classifier": tail_pred,
            "ten_pct_owner": scores["ten_pct_owner"],
            "conviction_score": scores["conviction_score"],
        }, index=test_df.index)
        oof_rows.append(oof_fold)

        shap_matrix = _shap_contrib(reg_model, test_df[feature_cols])
        direction_by_fold.append(_feature_directions(shap_matrix, test_df[feature_cols], feature_cols))

    n_folds_skipped = len(skipped_folds)
    n_folds_run = len(folds) - n_folds_skipped
    if n_folds_run == 0:
        raise ValueError(
            f"fit_and_validate: all {len(folds)} fold(s) were skipped. Each fold "
            f"had fewer than min_fold_train_rows={min_fold_train_rows} training "
            f"rows after purge and embargo. This dataset has {len(df_valid)} "
            f"valid row(s). That is too few for horizon={horizon} and "
            f"n_folds={n_folds}. Use fewer folds, use a shorter horizon, add "
            f"more data, or pass a lower min_fold_train_rows. Skipped-fold "
            f"detail: {skipped_folds}."
        )

    fold_metrics = pd.DataFrame(fold_metric_rows)
    value_cols = ["ic"] + [f"precision_{k}" for k in PRECISION_KS]
    precision_cols = {f"precision_{k}" for k in PRECISION_KS}
    summary_metrics = _summarize_fold_metrics(fold_metrics, value_cols, precision_cols)

    decile_tables: dict[str, pd.DataFrame] = {}
    tail_decile_tables: dict[str, pd.DataFrame] = {}
    for name in score_names:
        agg = _aggregate_by_rank(decile_tables_per_fold[name], "mean_label")
        mono = _monotonicity(agg.set_index("decile_rank")["mean_value"]) if not agg.empty else {"is_monotonic_nondecreasing": None, "n_violations": None, "rank_corr": float("nan")}
        agg.attrs["monotonicity"] = mono
        decile_tables[name] = agg
        tail_decile_tables[name] = _aggregate_by_rank(tail_decile_tables_per_fold[name], "p_tail")

    feature_stability = _feature_stability_table(direction_by_fold, feature_cols)

    # The label-shuffle test and the interaction analysis each fit a
    # model per fold. Each one walks the same fold list this function
    # just walked. They must see only the folds that ran. A skipped
    # fold has too few rows to fit, or none at all.
    skipped_ids = {sf["fold_id"] for sf in skipped_folds}
    ran_folds = [f for f in folds if f.fold_id not in skipped_ids]

    reg_fold_ics = fold_metrics.loc[fold_metrics["model"] == "regressor", "ic"].dropna()
    real_ic = float(reg_fold_ics.mean()) if len(reg_fold_ics) else float("nan")

    label_shuffle = run_label_shuffle_test(
        df_valid, ran_folds, feature_cols, params, real_ic=real_ic,
        n_seeds=n_shuffle_seeds, base_seed=label_shuffle_seed, label_col=label_col,
    )
    if not label_shuffle["passed"]:
        log.warning(
            "LEAKAGE SUSPECTED OR NO SHOWN EDGE: the real model's IC (%.4f) "
            "does not stand out from a null of mean=%.4f, stdev=%.4f built "
            "from %d shuffled-label draws (p=%.4f). Treat every other metric "
            "in this run as untrustworthy until this is understood.",
            real_ic, label_shuffle["null_mean"], label_shuffle["null_stdev"],
            label_shuffle["n_valid_draws"], label_shuffle["p_value"],
        )

    if run_shap_interactions:
        interaction_report, interaction_summary = run_shap_interaction_analysis(
            df_valid, ran_folds, feature_cols, params, label_col=label_col,
        )
    else:
        interaction_report = pd.DataFrame()
        interaction_summary = {"verdict": "skipped (run_shap_interactions=False)"}

    oof_scores = pd.concat(oof_rows, axis=0).sort_index()

    # Final deployment models: fit on the FULL valid dataset. This is not a
    # CV fold -- there is no future data left to hold out at deployment
    # time, so using everything here is correct and is not a leakage path
    # back into the metrics above (those were computed strictly out of
    # fold).
    final_reg, final_lo, final_hi = _fit_regressor(df_valid, feature_cols, params, label_col=label_col)
    final_cls_target = (df_valid[label_col] > 0).astype(int)
    final_cls = _fit_classifier(df_valid, feature_cols, final_cls_target, params)
    final_tail_target = (df_valid[label_col] > tail_thresh).astype(int)
    final_tail = _fit_classifier(df_valid, feature_cols, final_tail_target, params)

    models = {
        "regressor": final_reg,
        "classifier": final_cls,
        "tail_classifier": final_tail,
    }
    config = {
        "n_folds": n_folds,
        "horizon": horizon,
        "embargo": embargo,
        "min_fold_train_rows": min_fold_train_rows,
        "lgbm_params": params,
        "random_seeds": list(random_seeds),
        "label_shuffle_seed": label_shuffle_seed,
        "n_shuffle_seeds": n_shuffle_seeds,
        "leakage_threshold": leakage_threshold,
        "winsor_lo_pct": WINSOR_LO_PCT,
        "winsor_hi_pct": WINSOR_HI_PCT,
        "final_winsor_bounds": [final_lo, final_hi],
        "tail_thresh": tail_thresh,
        "label_col": label_col,
    }

    return ValidationResult(
        folds=folds,
        fold_metrics=fold_metrics,
        summary_metrics=summary_metrics,
        decile_tables=decile_tables,
        tail_decile_tables=tail_decile_tables,
        feature_stability=feature_stability,
        label_shuffle=label_shuffle,
        leakage_offenders=leakage_offenders,
        interaction_report=interaction_report,
        interaction_summary=interaction_summary,
        oof_scores=oof_scores,
        n_folds_run=n_folds_run,
        n_folds_skipped=n_folds_skipped,
        skipped_folds=skipped_folds,
        models=models,
        feature_cols=feature_cols,
        config=config,
    )


# ---------------------------------------------------------------------------
# 9. Persistence
# ---------------------------------------------------------------------------
def save_model_bundle(result: ValidationResult, path: str) -> None:
    """Persist the final fitted models plus the exact feature list and
    config used to produce them, so a later run can inject scores into the
    portfolio engine without re-fitting."""
    bundle = {
        "models": result.models,
        "feature_cols": result.feature_cols,
        "config": result.config,
    }
    joblib.dump(bundle, path)
    log.info("save_model_bundle: wrote %s", path)


def load_model_bundle(path: str) -> dict:
    return joblib.load(path)


# ---------------------------------------------------------------------------
# 9b. Production bundle: the one artifact insider_cluster_buys.py's live
# scoring path needs to score a freshly-detected cluster, with no
# dependency on ValidationResult, fit_and_validate, or a research dataset
# at call time.
#
# fit_and_validate's step 10 (see its docstring) already fits a regressor,
# classifier, and tail_classifier on the FULL valid dataset -- not a CV
# fold, a real deployment fit -- and hands them back in
# ValidationResult.models. So the walk-forward result DOES already contain
# a model usable for prediction on new rows; nothing here refits anything.
# This section only adds what a live caller needs and ValidationResult does
# not carry on its own: a single named model to use, a FIXED reference
# distribution of that model's own training-time scores (for percentiles),
# and provenance.
# ---------------------------------------------------------------------------

# Of the three final models, the tail classifier is the one this project's
# own validation actually cleared for ranking use. backtest/model_scores.py's
# DEFAULT_SCORE_COL is "oof_tail_classifier" for the same reason (see that
# module's own comment): out of the three, the tail classifier's top-decile
# lift is the one edge that survived both the label-shuffle leakage test and
# a volatility-matched benchmark (+4.74pp over a risk-matched benchmark,
# p=0.004, positive in 4 of 5 folds -- see research/live_score.py's module
# docstring for the full evidence this project bands its verdict on).
# Bundling any other model here would silently substitute an edge that was
# never actually demonstrated.
PRODUCTION_SCORE_MODEL = "tail_classifier"


def _never_live_default_guard_cols() -> tuple[str, ...]:
    """The never-computable-live feature columns build_production_bundle
    refuses to fit on by default (see that function's
    allow_never_live_features parameter).

    Imports research.live_score LAZILY, inside this function, rather than
    at module level. research.live_score does `from research import model
    as rm` to build its own feature categorization (PRICE_FEATURE_COLS /
    ISSUER_HISTORY_FEATURE_COLS / OWNER_HISTORY_FEATURE_COLS /
    SALE_FEATURE_COLS), so a module-level `from research import
    live_score` here would be a circular import. price_overrides.py's
    apply_price_overrides has the identical shape of problem with
    split_fingerprint.py (that module imports CACHE_DIR from
    backtest/prices.py, which imports price_overrides) and solves it the
    same way: a local import inside the one function that needs it.

    Deliberately live_score.SALE_FEATURE_COLS ONLY, not the full
    OWNER_HISTORY_FEATURE_COLS | SALE_FEATURE_COLS "never computable live"
    set live_score itself tracks. The two categories have different
    evidence behind them:
      - OWNER_HISTORY_FEATURE_COLS (3 cols) is already shipped, in
        production_model_noreuse_10575rows_20260808.joblib, and its cost
        as a permanently-missing input is MEASURED: forcing it to NaN
        changes only 10.2% of the top decile (89.8% overlap, Spearman
        0.9924 against the live-available-only ranking). That is a
        deliberate, already-accepted tradeoff, not an oversight -- gating
        it here would only block re-fitting the status quo, which is the
        opposite of what this guard is for.
      - SALE_FEATURE_COLS (9 cols, Section F concurrent-selling) has never
        shipped and has ZERO measured benefit: all 9 rank 41st-59th of 59
        in LightGBM gain, two with exactly zero gain (see
        research_groupE_10905rows_20260809.parquet's feature-importance
        table). Shipping them would add nine more permanently-missing
        live inputs for a null result. THAT is what this guard exists to
        catch -- a new dataset silently becoming the "latest" one and
        smuggling in a strictly-worse, unvetted always-missing category
        the same way OWNER_HISTORY_FEATURE_COLS was once accepted with
        evidence behind it, but this time with none.
    A future never-live category is guarded by default the same way
    SALE_FEATURE_COLS is: only a category with its own measured,
    documented tradeoff (like OWNER_HISTORY_FEATURE_COLS) should be added
    to the exemption above, and only alongside the numbers that justify it.
    """
    from research import live_score as ls  # local: see docstring -- live_score imports this module
    return ls.SALE_FEATURE_COLS


@dataclass(frozen=True)
class ProductionBundle:
    """Self-contained artifact for scoring ONE new (not-in-training) row at
    a time. No dependency on ValidationResult, fit_and_validate, or the
    dataset it was built from -- everything a caller needs is carried here.

    model: the fitted lgb.LGBMClassifier named by PRODUCTION_SCORE_MODEL.
    feature_cols: the exact column list AND ORDER `model` was fit on. A
      caller must build its scoring row in this order.
    training_scores: `model`'s own predict_proba(...)[:, 1] over every
      training row, sorted ascending. A FIXED reference distribution
      captured once at fit time. research.live_score.score_to_percentile
      ranks a new score against this array, never against whatever batch
      of clusters happens to be on screen that day -- a moving reference
      would make the same stock's rating depend on what else got scraped
      that day, which is absurd (see research/live_score.py's docstring).
    provenance: source_dataset_path, n_rows (== len(training_scores)),
      event_day_min/max, fit_timestamp (UTC ISO 8601) -- enough for a
      caller or a future debugger to answer "which data, how much, how
      old is this model" without re-deriving it from a git log.
    config: result.config, unchanged -- the fold/embargo/lgbm params etc.
      used to produce `model`, kept for reference even though this bundle
      is a single-model artifact with no folds of its own.
    """
    model: object
    feature_cols: list[str]
    training_scores: np.ndarray
    provenance: dict
    config: dict


def build_production_bundle(
    result: ValidationResult, df: pd.DataFrame, *, source_path: str = "",
    allow_never_live_features: bool = False,
) -> ProductionBundle:
    """Wrap fit_and_validate's already-fit full-dataset model
    (result.models[PRODUCTION_SCORE_MODEL]) into a ProductionBundle: adds
    the fixed training-score reference distribution and provenance a live
    caller needs that ValidationResult itself does not carry.

    `df` must be the SAME raw dataset (or an exact copy) passed to
    fit_and_validate to produce `result`. This function re-derives the
    exact rows the final model was fit on with the identical one-line
    filter fit_and_validate itself uses (dropna on the label column and
    entry_idx) rather than trusting a row count passed in separately, so
    the training-score distribution used for percentiles can never
    silently drift from the model's actual training data. The label
    column is read from result.config["label_col"] (label_col_for_horizon
    of whatever horizon `result` was actually fit with), not the module
    constant LABEL_COL, so this still lines up correctly for a `result`
    fit at a non-default horizon -- LABEL_COL ("adj_63") remains the
    fallback for a `result` from before this key existed.

    allow_never_live_features (default False): by default, this function
    REFUSES to bundle a model whose result.feature_cols includes any of
    research.live_score's SALE_FEATURE_COLS (the 9 Section F
    concurrent-selling features -- see _never_live_default_guard_cols'
    docstring for exactly why this set and not others). research/live_score.py
    treats those columns as permanently missing at scoring time -- there is
    no plumbing today to compute them live -- so a bundle fit on them ships
    a model that runs with those inputs NaN on every single live score,
    forever, for a feature category with ZERO measured benefit (see
    _never_live_default_guard_cols). Scoring itself (research.live_score's
    build_live_feature_row / score_cluster) is unaffected either way -- it
    builds its matrix as pd.DataFrame([feature_row], columns=bundle.feature_cols),
    so it stays correctly aligned no matter what a bundle declares. This
    guard is about not FITTING a bundle that ships that gap in the first
    place.

    REFUSE, not silently drop: an earlier design considered dropping the
    offending columns from result.feature_cols and fitting on the
    remainder instead. Rejected -- dropping would mean the fitted model
    result.models[PRODUCTION_SCORE_MODEL] (already trained by
    fit_and_validate, upstream of this function) does not match the
    feature set the bundle claims to use, OR would require silently
    re-fitting a second model inside a function whose docstring promises
    it "adds ... provenance ... that ValidationResult itself does not
    carry", not "fits a different model than the one it was handed". Both
    are worse than refusing: they make it possible to end up with a
    bundle whose feature_cols nobody chose on purpose, discoverable only
    by reading the code, not the call site. Refusing forces the choice
    (exclude the columns before calling fit_and_validate, or pass
    allow_never_live_features=True) to happen at the call site, where the
    caller can see and justify it.

    Raises ValueError if the guard above trips (naming the offending
    columns and the caller's options), or if result.models[PRODUCTION_SCORE_MODEL]
    is None -- that only happens when the full dataset's tail target
    (adj_63 > TAIL_THRESH) had a single class, meaning every cluster in
    the dataset either did or didn't clear the tail threshold. That is not
    a usable production dataset.
    """
    model = result.models.get(PRODUCTION_SCORE_MODEL)
    if model is None:
        raise ValueError(
            f"build_production_bundle: result.models[{PRODUCTION_SCORE_MODEL!r}] is "
            "None -- the full dataset's tail target (adj_63 > TAIL_THRESH) had only "
            "one class present. This is not a usable production dataset."
        )

    if not allow_never_live_features:
        guarded = set(_never_live_default_guard_cols())
        offending = [c for c in result.feature_cols if c in guarded]
        if offending:
            raise ValueError(
                "build_production_bundle: refusing to build a production bundle -- "
                f"{len(offending)} of its {len(result.feature_cols)} feature_cols are "
                "columns research.live_score.SALE_FEATURE_COLS marks as never "
                f"computable live: {offending}. research/live_score.py's "
                "build_live_feature_row has no plumbing to compute these outside "
                "the offline research pipeline (see backtest/sales_history.py), so "
                "a bundle fit on them would ship a model that runs with these "
                "inputs permanently NaN on every live score, for a feature group "
                "with no measured benefit (all 9 rank 41st-59th of 59 in LightGBM "
                "gain). Options: (1) exclude these columns before fitting -- pass "
                "feature_cols=[...] to fit_and_validate that excludes them, then "
                "call build_production_bundle on that result, or (2) if this is a "
                "deliberate experiment, pass allow_never_live_features=True to "
                "build_production_bundle (run_research.py's --fit-production "
                "exposes this as --allow-never-live-features)."
            )

    label_col = result.config.get("label_col", LABEL_COL)
    df_valid = df.dropna(subset=[label_col, "entry_idx"]).copy()
    if len(df_valid) == 0:
        raise ValueError("build_production_bundle: zero rows with a usable label and entry_idx")

    X = df_valid[result.feature_cols]
    training_scores = np.sort(_predict_proba_positive(model, X, fallback_rate=float("nan")))

    provenance = {
        "source_dataset_path": source_path,
        "n_rows": int(len(df_valid)),
        "event_day_min": str(df_valid["event_day"].min()),
        "event_day_max": str(df_valid["event_day"].max()),
        "fit_timestamp": datetime.now(timezone.utc).isoformat(),
    }

    log.info(
        "build_production_bundle: model=%s, %d training rows, event_day %s..%s, source=%s",
        PRODUCTION_SCORE_MODEL, len(df_valid), provenance["event_day_min"], provenance["event_day_max"],
        source_path or "(unspecified)",
    )

    return ProductionBundle(
        model=model,
        feature_cols=list(result.feature_cols),
        training_scores=training_scores,
        provenance=provenance,
        config=dict(result.config),
    )


def save_production_bundle(bundle: ProductionBundle, path: str) -> None:
    """joblib-dump `bundle` to `path`, atomically (tmp file + os.replace),
    matching save_research_dataset / save_oof_scores' write pattern so an
    interrupted write can never leave a truncated bundle sitting at the
    real path."""
    tmp_path = path + ".tmp"
    joblib.dump(bundle, tmp_path)
    os.replace(tmp_path, path)
    log.info(
        "save_production_bundle: wrote %s (%d training scores, %d features)",
        path, len(bundle.training_scores), len(bundle.feature_cols),
    )


def load_production_bundle(path: str) -> ProductionBundle:
    return joblib.load(path)


# Data goes to research_data/, not research/. research/ is the model package
# (mirrors backtest.research.save_research_dataset's own module note).
def save_oof_scores(oof_scores: pd.DataFrame, out_dir: str = "research_data", tag: str = "") -> str:
    """Write `result.oof_scores` to a parquet file under `out_dir`, named
    oof_scores_<tag_><YYYYMMDD>.parquet (no tag_ segment when tag is blank,
    matching the existing oof_scores_<YYYYMMDD>.parquet files this repo
    already has on disk). Returns the path written.

    Writes to a `.tmp` sibling first, then os.replace()s it over the real
    path, for the same reason save_research_dataset does: a same-filesystem
    rename is atomic, so a run interrupted mid-write can never leave a
    truncated parquet at the real path. Mirrors
    backtest.research.save_research_dataset / split_fingerprint._atomic_save.
    """
    os.makedirs(out_dir, exist_ok=True)
    tag_part = f"{tag}_" if tag else ""
    fname = f"oof_scores_{tag_part}{date.today():%Y%m%d}.parquet"
    path = os.path.join(out_dir, fname)
    tmp_path = path + ".tmp"
    oof_scores.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, path)
    log.info("Wrote OOF scores: %s (%d rows, %d cols)", path, len(oof_scores), len(oof_scores.columns))
    return path


def load_oof_scores(path: str) -> pd.DataFrame:
    """Read an OOF-scores parquet back into a DataFrame."""
    df = pd.read_parquet(path)
    log.info("load_oof_scores: %s (shape=%s)", path, df.shape)
    return df


# ---------------------------------------------------------------------------
# 10. Markdown report
# ---------------------------------------------------------------------------
def _df_to_markdown(df: pd.DataFrame, float_fmt: str = "{:.4f}") -> str:
    """Minimal dependency-free markdown table writer. pandas'
    DataFrame.to_markdown requires the `tabulate` package, which is not
    installed in this environment and cannot be installed without network
    access, so this writes the pipe-table format by hand."""
    if df.empty:
        return "(empty)\n"
    cols = list(df.columns)

    def fmt(v) -> str:
        if isinstance(v, float):
            if v != v:
                return "nan"
            return float_fmt.format(v)
        return str(v)

    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(row[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def write_markdown_summary(result: ValidationResult, path: str) -> None:
    """Write every metric table in `result` to a single markdown file."""
    parts: list[str] = []
    parts.append("# Ranking model validation report\n")
    parts.append(f"Config: {result.config}\n")

    parts.append("## Fold coverage\n")
    n_requested = len(result.folds)
    if result.n_folds_skipped > 0:
        parts.append(
            f"**{result.n_folds_run} of {n_requested} folds ran. "
            f"{result.n_folds_skipped} fold(s) were SKIPPED.** "
            "Every metric below counts only the folds that ran. Do not "
            f"read this as a {n_requested}-fold result.\n"
        )
        parts.append("Skipped fold detail:\n")
        for sf in result.skipped_folds:
            parts.append(
                f"- fold {sf['fold_id']}: {sf['reason']} "
                f"(n_candidate_train={sf['n_candidate_train']}, "
                f"n_purged={sf['n_purged']}, n_embargoed={sf['n_embargoed']})\n"
            )
    else:
        parts.append(f"All {n_requested} folds ran. None were skipped.\n")

    parts.append("## Leakage guard\n")
    if result.leakage_offenders:
        parts.append(f"TRIPPED: {result.leakage_offenders}\n")
    else:
        parts.append("No feature exceeded the correlation threshold. Passed.\n")

    parts.append("## Label-shuffle test\n")
    ls = result.label_shuffle
    status = "PASSED" if ls["passed"] else "FAILED, NO SHOWN EDGE OVER NOISE"
    parts.append(f"Status: {status}\n")
    parts.append(
        "The real model's IC is judged against an empirical null built from "
        "many independent label shuffles, not against a fixed constant. A "
        "reader can check every number below directly.\n"
    )
    parts.append(f"Real model, mean per-fold IC: {ls['real_ic']:.4f}\n")
    parts.append(
        f"Null distribution ({ls['n_valid_draws']} of {ls['n_seeds']} shuffle "
        f"draws produced a usable value):\n"
    )
    parts.append(
        f"- mean = {ls['null_mean']:.4f}, stdev = {ls['null_stdev']:.4f}, "
        f"min = {ls['null_min']:.4f}, max = {ls['null_max']:.4f}\n"
    )
    parts.append(
        f"- p5 = {ls['null_p5']:.4f}, p25 = {ls['null_p25']:.4f}, "
        f"p50 = {ls['null_p50']:.4f}, p75 = {ls['null_p75']:.4f}, "
        f"p95 = {ls['null_p95']:.4f}\n"
    )
    parts.append(
        f"- raw draws: {[round(v, 4) for v in ls['null_ic_values']]}\n"
    )
    parts.append(
        f"Real IC vs null: z-score = {ls['z_score']:.4f}, one-sided permutation "
        f"p-value = {ls['p_value']:.4f} (pass requires p < {SHUFFLE_TEST_ALPHA})\n"
    )
    parts.append(
        f"Secondary check, own-shuffle IC mean (a fold-construction leak "
        f"check, does not gate pass/fail): {ls['own_shuffle_ic_mean']:.4f}\n"
    )
    parts.append(f"{ls['note']}\n")

    parts.append("## Per-fold metrics\n")
    parts.append(_df_to_markdown(result.fold_metrics))

    parts.append("## Summary metrics (mean, t-stat across folds)\n")
    parts.append(_df_to_markdown(result.summary_metrics))

    parts.append("## Decile lift (mean adj_63 by score decile, aggregated across folds)\n")
    for name, table in result.decile_tables.items():
        mono = table.attrs.get("monotonicity", {})
        parts.append(f"### {name}\n")
        parts.append(f"Monotonicity: {mono}\n")
        parts.append(_df_to_markdown(table))

    parts.append("## P(adj_63 > 0.20) by score decile\n")
    for name, table in result.tail_decile_tables.items():
        parts.append(f"### {name}\n")
        parts.append(_df_to_markdown(table))

    parts.append("## Feature stability\n")
    stable = result.feature_stability[result.feature_stability["is_stable"]]
    unstable = result.feature_stability[~result.feature_stability["is_stable"]]
    parts.append(f"Stable features ({len(stable)}):\n")
    parts.append(_df_to_markdown(stable[["feature", "n_positive", "n_negative", "dominant_sign"]]))
    parts.append(f"Unstable features ({len(unstable)}):\n")
    parts.append(_df_to_markdown(unstable[["feature", "n_positive", "n_negative", "dominant_sign"]]))

    parts.append("## Ten-percent-owner interaction analysis\n")
    parts.append(f"{result.interaction_summary.get('method_note', '')}\n")
    parts.append(f"Verdict: {result.interaction_summary.get('verdict', '')}\n")
    parts.append(_df_to_markdown(result.interaction_report))

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))
    log.info("write_markdown_summary: wrote %s", path)
