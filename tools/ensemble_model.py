"""Bagged-ensemble scorer for research/model.py's ranking model, built to
test whether averaging across refits fixes the instability refit_stability.py
proved out.

WHY this exists. refit_stability.py ran 20 replicate refits under two
perturbation shapes (a random 1%-of-rows drop and a full bootstrap) and found
every top-k slot count FAILS its retention gate (bar 0.70): k=5 retention
0.35 (one replicate had ZERO overlap with the baseline top-5), k=10 0.39,
k=15 0.37, k=25 0.46, bootstrap mode ~0.22 across the board. Its own
diagnosis: real score separation exists in any SINGLE fit (rank5/6 gap ~13x
the rank10/11 gap), but the model's extreme tail-probability predictions
(oof_tail_classifier, a P(adj_63 > 0.20) estimate) are themselves
high-variance under a 1% perturbation of the training rows, so WHICH events
land in the top slots keeps reshuffling. Separation is not stability.

The classic fix for a high-variance estimator with low bias is bagging:
average many independently-fit copies of the same estimator and let
uncorrelated errors cancel. This module implements exactly that -- an
N-member ensemble (default 20, --n-members) of research.model.fit_and_
validate refits, combined two ways (raw-score averaging and rank averaging),
then re-measures the SAME stability gate refit_stability.py defined, on the
ensemble's scores instead of a single fit's.

This module explicitly does NOT assume bagging works. Sections 3 and 6
below run the honest test and section's docstring on run_ensemble_stability_
gate states plainly whether the gate clears. A stabilized ranking that lost
its edge (vol-matched excess collapsing toward zero) is reported as a
failure, not dressed up as a win -- see the module docstring's "Two ways to
fail" note below.

===========================================================================
OOF DISCIPLINE -- READ THIS BEFORE TOUCHING build_ensemble_oof OR
fit_member_oof. Getting this wrong would manufacture a fake edge, the single
worst outcome this task's brief calls out.
===========================================================================

Each ensemble member is ONE independent, complete call to research.model.
fit_and_validate on (a copy of) the pool being scored. That function's own
purged/embargoed/expanding-window fold loop (make_purged_expanding_folds,
tested directly for the no-overlap invariant in tests/test_model.py) is the
ONLY thing that ever decides which rows a member's model trains on versus
predicts on. This module never reimplements, patches, or reaches inside
that loop -- it only ever reads the FINISHED result.oof_scores DataFrame a
member's fit_and_validate call hands back (see fit_member_oof, a thin
wrapper with zero extra logic between the call and the return).

By construction, result.oof_scores contains a row for event X only when X
sat in the TEST block of some fold that member actually ran -- i.e. a row
that member's model for that fold never trained on. A member has NO row for
X in its oof_scores if X was purged/embargoed out of every fold's test set,
dropped for a missing label, or (under --row-resample-mode drop1pct /
bootstrap) simply not present in that member's resampled pool. In every
case, "no row" is the correct and only failure mode -- fit_and_validate
never emits an in-fold (train-set) prediction into oof_scores.

build_ensemble_oof's combination step runs strictly AFTER every member has
finished fitting and returned its oof_scores. It does one thing: for each
(ticker, event_day) key, average the score across whichever members
produced a row for that key (pandas .mean(skipna=True) over columns that
may be NaN, never treating a missing member as a zero). Averaging finished,
independently-produced OOF-only values together cannot inject an in-fold
prediction into the ensemble, because no in-fold prediction is ever present
in ANY member's contribution to average over. The union of several
OOF-only frames is still OOF-only -- there is no code path here through
which member M's training rows can reach member M's own output (that is
fit_and_validate's job and is unchanged), nor through which member M's
output can reach member M' 's fit (each member is fit independently, before
any combination happens; no member's fit function ever reads another
member's predictions).

With --row-resample-mode none (the module default), there is a second,
stronger guarantee worth stating explicitly: make_purged_expanding_folds is
a deterministic function of (df_pool, n_folds, horizon, embargo) alone --
it draws no randomness -- so when every member is fit on the SAME df_pool,
every member gets the IDENTICAL fold split. Every member's oof_scores then
covers the exact same row universe, purely because a different lgbm
random_state changes what the trees inside each fold's model look like,
never which rows are train vs test. This is textbook bagging: N models
fit on the same data with the same held-out structure, differing only in
their own internal stochastic draws (LightGBM's subsample=0.8/
colsample_bytree=0.8 row/feature sampling and split tie-breaking), averaged
after the fact. --row-resample-mode drop1pct/bootstrap (reusing
refit_stability.resample_dataset -- see its own docstring for the shape of
each) adds a second, coarser source of diversity across members (different
folds too, not just different random_state), at the cost of members no
longer sharing an identical OOF universe; every average in this module is
computed with skipna=True for exactly that reason.

Two ways to fail. Bagging can fail a ranking-stability problem in two
distinct directions, and this module checks both, per the task brief:
  1. It can simply not help enough -- the gate still fails after averaging.
  2. It can "stabilize" the ranking by regressing every score toward the
     pool mean, which mechanically raises k-set overlap while destroying
     the very separation that gave the model any edge in the first place.
     Section 4 below (the vol-matched excess check on the ensemble's OWN
     unperturbed fit) exists specifically to catch this: an ensemble that
     passes the retention gate with a vol-matched excess near zero is not
     a win, it is a differently-shaped failure.

===========================================================================
FEATURE-COLUMN WORKAROUND (read before changing FEATURE_COLS handling)
===========================================================================

As of this module's writing, research.model.FEATURE_COLS (imported at
MODULE LOAD TIME from backtest.research._FEATURE_COLS) carries 9
concurrent-selling feature columns (x_sell_n_cluster, x_sell_n_insiders_
cluster, x_log1p_sell_value_cluster, x_sell_n_trail90, x_sell_n_insiders_
trail90, x_log1p_sell_value_trail90, x_buy_sell_balance_cluster, x_buyer_
also_sold_nearby, x_officer_or_director_sold_cluster) that a different,
concurrently-running task added to backtest/research.py AFTER this
module's pinned inputs (research_noreuse_10908rows_20260808.parquet /
oof_scores_noreuse_20260808.parquet) were built -- those 9 columns do not
exist in the pinned research parquet. research.model.fit_and_validate's
schema validation (_validate_schema) unconditionally checks the CURRENT
global FEATURE_COLS list, not whatever feature_cols a caller passes to
fit_and_validate, so a fit against the pinned dataset would otherwise raise
ValueError outright, unrelated to anything this module is trying to
measure.

_resolve_feature_cols / _pad_for_schema (section 0 below) work around this
without touching research/model.py or backtest/research.py (both off
limits for this task) and without changing what the model actually learns
from: every fit in this module explicitly passes feature_cols=<the subset
that exists in the pinned dataset with real data> to fit_and_validate, and
the 9 missing columns are added to a COPY of the pool as all-NaN purely to
satisfy the presence check -- they are never in the explicit feature_cols
list, so they are never split on, never enter winsorization, and never
enter find_leaky_features. This keeps every fit in this module using
EXACTLY the same 50 features the pinned baseline OOF file itself was fit
on, so the ensemble/baseline comparison stays apples-to-apples and immune
to the other task's in-flight change -- which is the entire reason the task
brief pins this module to the noreuse parquet by name in the first place.

===========================================================================
Combination methods (--combine-methods, default both)
===========================================================================
  score: per-event mean of each member's raw oof_tail_classifier
    probability. Sensitive to any one member's extreme prediction (a
    probability near 0 or 1 pulls the mean hard).
  rank: per-event mean of each member's OWN within-member percentile rank
    (rank / n_scored_by_that_member, 1/n = best), then negated so higher
    values (mean-rank-scores) sort to the top the same way oof_tail_
    classifier does. Percentile (not raw rank position) is used so members
    with different-sized OOF universes under row-resampling remain
    comparable. Rank averaging is often more robust to one member's
    outlier probability than raw-score averaging -- section 3 measures
    this directly rather than assuming it.

===========================================================================
Usage
===========================================================================
    python tools/ensemble_model.py --mode drop1pct --n-replicates 20
    python tools/ensemble_model.py --mode bootstrap --n-replicates 20 --n-members 20
    python tools/ensemble_model.py --baseline-only        # fast: just the unperturbed
                                                      # ensemble fit + vol-match
    python tools/ensemble_model.py --regularized --mode drop1pct   # lever 2 (section 7)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp

# Run directly (`python tools/ensemble_model.py ...`), the interpreter puts
# this file's own directory (tools/) on sys.path[0], not the repo root -- so
# the imports below would fail without this. Harmless no-op when this
# module is instead imported normally (e.g. `from tools import
# ensemble_model`), since the repo root is already on sys.path in that case.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tools import refit_stability as rs
from research import model as rm

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
DEFAULT_RESEARCH_PATH = rs.DEFAULT_RESEARCH_PATH
DEFAULT_OOF_PATH = rs.DEFAULT_OOF_PATH

N_MEMBERS_DEFAULT = 20
BASE_SEED_DEFAULT = 0

ROW_RESAMPLE_MODES: tuple[str, ...] = ("none",) + rs.RESAMPLE_MODES  # ("none", "drop1pct", "bootstrap")
ROW_RESAMPLE_MODE_DEFAULT = "none"

SCORE_COL_DEFAULT = rs.SCORE_COL_DEFAULT  # "oof_tail_classifier" -- what each member emits
VOL_COL_DEFAULT = rs.VOL_COL_DEFAULT

ENSEMBLE_SCORE_AVG_COL = "ensemble_score_avg"
ENSEMBLE_RANK_COL = "ensemble_score_rankavg"
COMBINE_METHOD_COLS: dict[str, str] = {"score": ENSEMBLE_SCORE_AVG_COL, "rank": ENSEMBLE_RANK_COL}
COMBINE_METHODS_DEFAULT: tuple[str, ...] = ("score", "rank")

TOP_KS: tuple[int, ...] = rs.TOP_KS
RETENTION_BAR_DEFAULT = rs.RETENTION_BAR_DEFAULT
VOLMATCH_THRESHOLD_PP_DEFAULT = rs.VOLMATCH_THRESHOLD_PP_DEFAULT

# See refit_stability.py's FAST-FIT NOTE: run_shap_interactions=False and
# n_shuffle_seeds=2 leave oof_tail_classifier byte-identical while cutting a
# fit from ~25s to ~8s. Identical reasoning applies unchanged here -- this
# module never reads label_shuffle or interaction_report from any member.
FAST_N_SHUFFLE_SEEDS = rs.FAST_N_SHUFFLE_SEEDS

# Lever 2 (section 7 of the module docstring): heavier regularization,
# applied only when --regularized is passed. Fewer leaves, a higher
# min_child_samples floor, and 5x the L2 penalty relative to research.model.
# DEFAULT_LGBM_PARAMS (num_leaves=15, min_child_samples=20, reg_lambda=1.0)
# -- the three knobs the task brief names, moved in the direction that
# shrinks an individual tree's capacity to fit noise in the training set,
# which is the direct lever on an estimator's own variance (as opposed to
# bagging, which averages variance away after the fact; this module can
# apply both).
REGULARIZED_LGBM_OVERRIDES: dict = dict(num_leaves=7, min_child_samples=50, reg_lambda=5.0)


# ---------------------------------------------------------------------------
# 0. Feature-column workaround -- see module docstring.
# ---------------------------------------------------------------------------
def _resolve_feature_cols(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """(usable_feature_cols, padded_cols): usable_feature_cols is the subset
    of research.model.FEATURE_COLS actually present in `df` with real data;
    padded_cols is whatever from that (current, module-load-time) list is
    missing. See module docstring's "FEATURE-COLUMN WORKAROUND" section."""
    usable = [c for c in rm.FEATURE_COLS if c in df.columns]
    padded = [c for c in rm.FEATURE_COLS if c not in df.columns]
    return usable, padded


def _pad_for_schema(df: pd.DataFrame, padded_cols: list[str]) -> pd.DataFrame:
    """Return a COPY of `df` with every column in `padded_cols` added as
    all-NaN, so research.model.fit_and_validate's unconditional schema
    presence check passes. Every fit in this module also passes an explicit
    feature_cols= (the USABLE subset, never padded_cols), so these columns
    are never split on, never enter winsorization, and never enter
    find_leaky_features -- see module docstring. A no-op (returns `df`
    unchanged, no copy) when padded_cols is empty."""
    if not padded_cols:
        return df
    out = df.copy()
    for c in padded_cols:
        out[c] = np.nan
    return out


# ---------------------------------------------------------------------------
# 1. One ensemble member
# ---------------------------------------------------------------------------
def fit_member_oof(
    df_pool: pd.DataFrame,
    *,
    member_seed: int,
    feature_cols: list[str],
    n_folds: int = rm.DEFAULT_N_FOLDS,
    horizon: int = rm.PRIMARY_HORIZON,
    embargo: int = rm.DEFAULT_EMBARGO_DAYS,
    min_fold_train_rows: int = rm.MIN_FOLD_TRAIN_ROWS,
    lgbm_overrides: Optional[dict] = None,
) -> pd.DataFrame:
    """One ensemble member: research.model.fit_and_validate on `df_pool`
    with lgbm_params={"random_state": member_seed, **lgbm_overrides},
    run_shap_interactions=False and n_shuffle_seeds=FAST_N_SHUFFLE_SEEDS
    (see refit_stability.py's FAST-FIT NOTE, unchanged here). Returns
    result.oof_scores UNMODIFIED -- see module docstring's OOF DISCIPLINE
    section: this module's leak-safety rests on never touching a member's
    OOF frame before build_ensemble_oof's combination step reads it.
    """
    lgbm_params = {"random_state": member_seed}
    if lgbm_overrides:
        lgbm_params.update(lgbm_overrides)
    result = rm.fit_and_validate(
        df_pool, n_folds=n_folds, horizon=horizon, embargo=embargo,
        min_fold_train_rows=min_fold_train_rows, feature_cols=feature_cols,
        lgbm_params=lgbm_params, run_shap_interactions=False, n_shuffle_seeds=FAST_N_SHUFFLE_SEEDS,
    )
    return result.oof_scores


# ---------------------------------------------------------------------------
# 2. The ensemble: N members, combined two ways
# ---------------------------------------------------------------------------
def build_ensemble_oof(
    df_pool: pd.DataFrame,
    *,
    n_members: int = N_MEMBERS_DEFAULT,
    base_seed: int = BASE_SEED_DEFAULT,
    feature_cols: list[str],
    row_resample_mode: str = ROW_RESAMPLE_MODE_DEFAULT,
    row_resample_drop_frac: float = rs.DROP_FRAC_DEFAULT,
    n_folds: int = rm.DEFAULT_N_FOLDS,
    horizon: int = rm.PRIMARY_HORIZON,
    embargo: int = rm.DEFAULT_EMBARGO_DAYS,
    min_fold_train_rows: int = rm.MIN_FOLD_TRAIN_ROWS,
    score_col: str = SCORE_COL_DEFAULT,
    lgbm_overrides: Optional[dict] = None,
) -> tuple[pd.DataFrame, dict]:
    """Fit `n_members` independent members (fit_member_oof) and combine
    their OOF `score_col` values into one ensemble frame carrying BOTH
    ENSEMBLE_SCORE_AVG_COL (mean raw score) and ENSEMBLE_RANK_COL (negated
    mean within-member percentile rank) -- see module docstring's
    "Combination methods" section. Both are derived from the SAME set of
    member fits, so comparing them costs nothing extra: no member is ever
    refit per combination method.

    row_resample_mode="none" (default): every member is fit on `df_pool`
    itself, varying only lgbm random_state -- classic bagging, identical
    fold split across members (see module docstring). "drop1pct"/
    "bootstrap": each member instead gets its own resample_dataset draw of
    `df_pool` (reusing refit_stability.resample_dataset, seeded per member),
    adding a second source of cross-member diversity at the cost of members
    no longer sharing an identical OOF row universe -- every average below
    is computed with skipna=True for exactly that reason.

    A member whose fit_and_validate call raises is logged and skipped (
    mirrors refit_stability.run_stability_gate's per-replicate fault
    tolerance); raises RuntimeError if every member failed.

    Returns (ensemble_df, meta). ensemble_df has one row per (ticker,
    event_day) present in `df_pool`'s own identity columns (a row is
    present here even if EVERY member's score for it is NaN, e.g. purged
    out of every fold -- downstream callers filter via rs.dedup_oof same as
    any other oof frame), plus per-member `_member{m}_score` /
    `_member{m}_pctrank` columns (kept for inspection/debugging) and the
    two ensemble columns. meta carries n_members_requested, n_members_fit,
    per-member fit_seconds, and feature_cols.
    """
    if n_members < 1:
        raise ValueError("build_ensemble_oof: n_members must be >= 1")
    if row_resample_mode not in ROW_RESAMPLE_MODES:
        raise ValueError(f"build_ensemble_oof: unknown row_resample_mode {row_resample_mode!r}, expected one of {ROW_RESAMPLE_MODES}")

    # rm.label_col_for_horizon(horizon), not the module constant rm.LABEL_COL:
    # fit_member_oof passes this same `horizon` through to fit_and_validate,
    # whose oof_scores now carries a column named for the ACTUAL horizon it
    # fit (see research/model.py's label_col_for_horizon docstring for the
    # bug this fixes). horizon=rm.PRIMARY_HORIZON (the default here) resolves
    # to "adj_63", identical to rm.LABEL_COL, so this is a no-op change at
    # every existing default-horizon call site.
    id_cols = ["ticker", "event_day", "entry_day", "entry_idx", rm.label_col_for_horizon(horizon)]
    base_ids = (
        df_pool[id_cols].drop_duplicates(subset=["ticker", "event_day"], keep="first")
        .set_index(["ticker", "event_day"])
    )

    fit_seconds: list[float] = []
    score_series: dict[int, pd.Series] = {}
    pctrank_series: dict[int, pd.Series] = {}

    for m in range(n_members):
        member_seed = base_seed * 1_000_003 + m
        if row_resample_mode == "none":
            df_member = df_pool
        else:
            df_member = rs.resample_dataset(df_pool, mode=row_resample_mode, seed=member_seed, drop_frac=row_resample_drop_frac)

        t0 = time.monotonic()
        try:
            oof_m = fit_member_oof(
                df_member, member_seed=member_seed, feature_cols=feature_cols,
                n_folds=n_folds, horizon=horizon, embargo=embargo,
                min_fold_train_rows=min_fold_train_rows, lgbm_overrides=lgbm_overrides,
            )
        except Exception as exc:  # noqa: BLE001 -- one bad member must not sink the whole ensemble
            log.warning("member %d/%d: fit failed (%s) -- skipped", m + 1, n_members, exc)
            continue
        elapsed = time.monotonic() - t0
        fit_seconds.append(elapsed)

        dedup_m = rs.dedup_oof(oof_m, score_col, context=f"member {m}")
        ranked_m = rs.ranked_picks(dedup_m, score_col)
        n_m = len(ranked_m)
        key = list(zip(ranked_m["ticker"], ranked_m["event_day"]))
        score_series[m] = pd.Series(ranked_m[score_col].to_numpy(), index=pd.MultiIndex.from_tuples(key, names=["ticker", "event_day"]))
        pctrank_series[m] = pd.Series(
            (ranked_m["rank"].to_numpy() / n_m) if n_m else np.array([]),
            index=pd.MultiIndex.from_tuples(key, names=["ticker", "event_day"]),
        )
        log.info("member %d/%d fit in %.1fs (n_oof=%d)", m + 1, n_members, elapsed, n_m)

    n_members_fit = len(score_series)
    if n_members_fit == 0:
        raise RuntimeError(f"build_ensemble_oof: all {n_members} member(s) failed to fit -- nothing to ensemble")
    if n_members_fit < n_members:
        log.warning("build_ensemble_oof: only %d of %d member(s) produced a usable fit", n_members_fit, n_members)

    ensemble = base_ids.copy()
    score_cols: list[str] = []
    pctrank_cols: list[str] = []
    for m in sorted(score_series):
        score_c, pct_c = f"_member{m}_score", f"_member{m}_pctrank"
        ensemble[score_c] = score_series[m]
        ensemble[pct_c] = pctrank_series[m]
        score_cols.append(score_c)
        pctrank_cols.append(pct_c)

    ensemble = ensemble.reset_index()
    ensemble["n_members_scored"] = ensemble[score_cols].notna().sum(axis=1)
    ensemble[ENSEMBLE_SCORE_AVG_COL] = ensemble[score_cols].mean(axis=1, skipna=True)
    mean_pctrank = ensemble[pctrank_cols].mean(axis=1, skipna=True)
    ensemble[ENSEMBLE_RANK_COL] = -mean_pctrank  # negate: lower mean percentile rank (better) -> higher ensemble score

    meta = {
        "n_members_requested": n_members,
        "n_members_fit": n_members_fit,
        "fit_seconds": fit_seconds,
        "mean_fit_seconds": float(np.mean(fit_seconds)) if fit_seconds else float("nan"),
        "feature_cols": list(feature_cols),
        "row_resample_mode": row_resample_mode,
    }
    return ensemble, meta


# ---------------------------------------------------------------------------
# 3. Vol-match significance across replicates (extends refit_stability's
#    single-number vol_matched_excess with a t-test over the replicate
#    distribution -- see module docstring item 4: "positive AND
#    significant", not just positive on one draw).
# ---------------------------------------------------------------------------
def volmatch_significance(replicate_excess: np.ndarray) -> dict:
    """One-sample t-test of `replicate_excess` (one volmatch_excess value
    per stability-gate replicate) against 0. NaNs are dropped first. Needs
    at least 2 valid values to report a t-stat/p-value; otherwise those come
    back NaN and n_positive/n_valid are still reported."""
    vals = replicate_excess[~np.isnan(replicate_excess)]
    n_valid = int(len(vals))
    n_positive = int((vals > 0).sum())
    if n_valid >= 2 and np.std(vals, ddof=1) > 0:
        t, p = ttest_1samp(vals, 0.0)
        t, p = float(t), float(p)
    else:
        t, p = float("nan"), float("nan")
    return {
        "n_valid": n_valid,
        "n_positive": n_positive,
        "mean": float(np.mean(vals)) if n_valid else float("nan"),
        "std": float(np.std(vals, ddof=1)) if n_valid > 1 else float("nan"),
        "tstat": t,
        "pvalue": p,
    }


# ---------------------------------------------------------------------------
# 4. Orchestration: re-run refit_stability's gate on ensemble scores
# ---------------------------------------------------------------------------
def run_ensemble_stability_gate(
    research_path: str = DEFAULT_RESEARCH_PATH,
    *,
    n_replicates: int = rs.N_REPLICATES_DEFAULT,
    mode: str = rs.MODE_DEFAULT,
    drop_frac: float = rs.DROP_FRAC_DEFAULT,
    seed: int = 0,
    n_members: int = N_MEMBERS_DEFAULT,
    row_resample_mode: str = ROW_RESAMPLE_MODE_DEFAULT,
    row_resample_drop_frac: float = rs.DROP_FRAC_DEFAULT,
    n_folds: int = rm.DEFAULT_N_FOLDS,
    horizon: int = rm.PRIMARY_HORIZON,
    embargo: int = rm.DEFAULT_EMBARGO_DAYS,
    min_fold_train_rows: int = rm.MIN_FOLD_TRAIN_ROWS,
    top_ks: tuple[int, ...] = TOP_KS,
    vol_col: str = VOL_COL_DEFAULT,
    top_frac: float = rs.TOP_FRAC,
    n_vol_buckets: int = rs.N_VOL_BUCKETS,
    n_boot: int = rs.N_BOOT_DEFAULT,
    retention_bar: float = RETENTION_BAR_DEFAULT,
    volmatch_threshold_pp: float = VOLMATCH_THRESHOLD_PP_DEFAULT,
    combine_methods: tuple[str, ...] = COMBINE_METHODS_DEFAULT,
    lgbm_overrides: Optional[dict] = None,
) -> dict[str, dict]:
    """Same protocol as refit_stability.run_stability_gate, with the
    "refit" step replaced by "fit an n_members ensemble and combine its
    scores" (build_ensemble_oof). Reuses refit_stability's resampling
    (resample_dataset), dedup/ranking/overlap/vol-match measurement
    functions and PASS/FAIL summary builder (_build_summary) UNCHANGED --
    this module only ever calls them, never reimplements them.

    The baseline every replicate is compared against is an ensemble fit on
    the UNPERTURBED pool (n_members members, same as every replicate),
    computed once here -- unlike refit_stability, which loads a
    pre-existing single-fit OOF file from disk, there is no pre-existing
    "ensemble baseline" file, so this function builds its own, using the
    exact same n_members/row_resample_mode/lgbm_overrides as every
    replicate. This keeps the comparison strictly ensemble-vs-ensemble,
    mirroring refit_stability's single-fit-vs-single-fit design one level
    up.

    Returns {combine_method: {"replicate_df", "summary_df", "baseline_gap",
    "baseline_vol", "baseline_meta", "volmatch_sig"}} for every method in
    `combine_methods` -- computed from the SAME underlying member fits (see
    build_ensemble_oof), so comparing "score" vs "rank" here costs no extra
    model fits, only extra (cheap) aggregation.
    """
    log.info("run_ensemble_stability_gate: loading research dataset %s", research_path)
    df_raw = pd.read_parquet(research_path)
    feature_cols, padded_cols = _resolve_feature_cols(df_raw)
    if padded_cols:
        log.info(
            "run_ensemble_stability_gate: padding %d feature column(s) missing from %s as all-NaN "
            "(excluded from feature_cols passed to every fit) -- see module docstring's "
            "FEATURE-COLUMN WORKAROUND section: %s",
            len(padded_cols), research_path, padded_cols,
        )
    df_orig = _pad_for_schema(df_raw, padded_cols)
    log.info(
        "run_ensemble_stability_gate: %d research rows, %d usable feature(s), mode=%s, n_replicates=%d, n_members=%d",
        len(df_orig), len(feature_cols), mode, n_replicates, n_members,
    )

    label_col = rm.label_col_for_horizon(horizon)  # see build_ensemble_oof's id_cols comment
    vol_map = (
        df_orig[["ticker", "event_day", vol_col]]
        .dropna(subset=[vol_col])
        .drop_duplicates(subset=["ticker", "event_day"], keep="first")
        .set_index(["ticker", "event_day"])[vol_col]
    )

    t0 = time.monotonic()
    baseline_ensemble, baseline_meta = build_ensemble_oof(
        df_orig, n_members=n_members, base_seed=seed, feature_cols=feature_cols,
        row_resample_mode=row_resample_mode, row_resample_drop_frac=row_resample_drop_frac,
        n_folds=n_folds, horizon=horizon, embargo=embargo, min_fold_train_rows=min_fold_train_rows,
        lgbm_overrides=lgbm_overrides,
    )
    log.info("run_ensemble_stability_gate: baseline ensemble fit in %.1fs", time.monotonic() - t0)

    per_method: dict[str, dict] = {}
    for method in combine_methods:
        score_col = COMBINE_METHOD_COLS[method]
        baseline_dedup = rs.dedup_oof(baseline_ensemble, score_col, context=f"baseline ensemble [{method}]")
        baseline_ranked = rs.ranked_picks(baseline_dedup, score_col)
        baseline_topk = {k: rs.top_k_keys(baseline_ranked, k) for k in top_ks}
        baseline_gap = rs.score_gap_metrics(baseline_ranked, score_col)
        baseline_vol = rs.vol_matched_excess(
            baseline_dedup, vol_map, score_col=score_col, label_col=label_col,
            top_frac=top_frac, n_vol_buckets=n_vol_buckets, n_boot=n_boot,
            rng=np.random.default_rng(seed + 999_000),
        )
        log.info(
            "baseline ensemble [%s]: n_oof=%d gap_5_6_norm=%.3f gap_10_11_norm=%.3f volmatch_excess=%+.4f",
            method, len(baseline_ranked), baseline_gap["gap_5_6_norm"], baseline_gap["gap_10_11_norm"],
            baseline_vol["volmatch_excess"],
        )
        per_method[method] = {
            "score_col": score_col, "baseline_ranked": baseline_ranked, "baseline_topk": baseline_topk,
            "baseline_gap": baseline_gap, "baseline_vol": baseline_vol, "baseline_meta": baseline_meta,
            "records": [],
        }

    for rep in range(n_replicates):
        rep_seed = seed * 1_000_003 + rep
        t0 = time.monotonic()
        try:
            df_pool = rs.resample_dataset(df_orig, mode=mode, seed=rep_seed, drop_frac=drop_frac)
            replicate_ensemble, rep_meta = build_ensemble_oof(
                df_pool, n_members=n_members, base_seed=rep_seed, feature_cols=feature_cols,
                row_resample_mode=row_resample_mode, row_resample_drop_frac=row_resample_drop_frac,
                n_folds=n_folds, horizon=horizon, embargo=embargo, min_fold_train_rows=min_fold_train_rows,
                lgbm_overrides=lgbm_overrides,
            )
        except Exception as exc:  # noqa: BLE001 -- one bad replicate must not sink the whole run
            log.warning("replicate %d/%d: ensemble fit failed (%s) -- skipped", rep + 1, n_replicates, exc)
            continue
        elapsed = time.monotonic() - t0

        for method in combine_methods:
            score_col = per_method[method]["score_col"]
            baseline_ranked = per_method[method]["baseline_ranked"]
            baseline_topk = per_method[method]["baseline_topk"]

            dedup = rs.dedup_oof(replicate_ensemble, score_col, context=f"replicate {rep} [{method}]")
            ranked = rs.ranked_picks(dedup, score_col)
            gap = rs.score_gap_metrics(ranked, score_col)
            churn = rs.rank_churn(baseline_ranked, ranked, top_n=rs.RANK_CHURN_TOP_N)
            vol = rs.vol_matched_excess(
                dedup, vol_map, score_col=score_col, label_col=label_col,
                top_frac=top_frac, n_vol_buckets=n_vol_buckets, n_boot=n_boot,
                rng=np.random.default_rng(rep_seed + 500_000),
            )
            row = {
                "replicate_id": rep, "mode": mode, "combine_method": method, "seed": rep_seed,
                "n_pool_rows": len(df_pool), "n_oof_rows": len(ranked),
                "n_members_fit": rep_meta["n_members_fit"], "fit_seconds": elapsed,
            }
            for k in top_ks:
                repl_set = rs.top_k_keys(ranked, k)
                row[f"retention_k{k}"] = rs.retention_rate(baseline_topk[k], repl_set)
                row[f"jaccard_k{k}"] = rs.jaccard_index(baseline_topk[k], repl_set)
            row.update(gap)
            row.update(churn)
            row.update(vol)
            per_method[method]["records"].append(row)

        log.info(
            "replicate %d/%d done in %.1fs (n_members_fit=%d/%d)",
            rep + 1, n_replicates, elapsed, rep_meta["n_members_fit"], n_members,
        )

    out: dict[str, dict] = {}
    for method in combine_methods:
        state = per_method[method]
        replicate_df = pd.DataFrame.from_records(state["records"])
        if replicate_df.empty:
            raise RuntimeError(f"run_ensemble_stability_gate: every replicate failed to fit for combine_method={method!r}")
        if len(replicate_df) < n_replicates:
            log.warning(
                "run_ensemble_stability_gate [%s]: only %d of %d replicate(s) produced a usable fit",
                method, len(replicate_df), n_replicates,
            )
        summary_df = rs._build_summary(
            replicate_df, baseline_gap=state["baseline_gap"], baseline_vol=state["baseline_vol"],
            top_ks=top_ks, retention_bar=retention_bar, volmatch_threshold_pp=volmatch_threshold_pp,
        )
        volmatch_sig = volmatch_significance(replicate_df["volmatch_excess"].to_numpy(dtype=float))
        out[method] = {
            "replicate_df": replicate_df, "summary_df": summary_df,
            "baseline_gap": state["baseline_gap"], "baseline_vol": state["baseline_vol"],
            "baseline_meta": state["baseline_meta"], "volmatch_sig": volmatch_sig,
        }
    return out


# ---------------------------------------------------------------------------
# 5. Persistence (atomic write -- reuses refit_stability's helpers directly,
#    matching this repo's own precedent of test files reaching into these;
#    see refit_stability.py's own "Persistence" section for the pattern).
# ---------------------------------------------------------------------------
def save_ensemble_oof(ensemble_df: pd.DataFrame, out_dir: str = "research_data", tag: str = "") -> str:
    """Write the baseline (unperturbed) ensemble's full OOF frame (per-
    member score/pctrank columns plus both ensemble score columns).
    Filename: ensemble_oof_scores_<tag_><YYYYMMDD>.parquet."""
    os.makedirs(out_dir, exist_ok=True)
    tag_part = f"{tag}_" if tag else ""
    path = os.path.join(out_dir, f"ensemble_oof_scores_{tag_part}{date.today():%Y%m%d}.parquet")
    rs._atomic_write_parquet(ensemble_df, path)
    log.info("save_ensemble_oof: wrote %s (%d rows, %d cols)", path, len(ensemble_df), len(ensemble_df.columns))
    return path


def save_ensemble_stability_results(
    replicate_df: pd.DataFrame, summary_df: pd.DataFrame, *, out_dir: str, mode: str, combine_method: str, tag: str = "",
) -> tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    tag_part = f"{tag}_" if tag else ""
    today = date.today().strftime("%Y%m%d")
    rep_path = os.path.join(out_dir, f"ensemble_stability_replicates_{mode}_{combine_method}_{tag_part}{today}.parquet")
    sum_path = os.path.join(out_dir, f"ensemble_stability_summary_{mode}_{combine_method}_{tag_part}{today}.csv")
    rs._atomic_write_parquet(replicate_df, rep_path)
    rs._atomic_write_csv(summary_df, sum_path)
    log.info("save_ensemble_stability_results: wrote %s and %s", rep_path, sum_path)
    return rep_path, sum_path


# ---------------------------------------------------------------------------
# 6. CLI
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Ensembled (bagged) scorer for research/model.py's ranking model: fits N "
            "independent refits (varying LightGBM random_state and, optionally, a "
            "per-member row resample), averages their out-of-fold scores two ways "
            "(raw score, rank), and re-runs refit_stability.py's retention/vol-match "
            "gate on the ensemble scores to test whether bagging fixes the instability "
            "that gate found in a single fit. See this module's docstring."
        )
    )
    p.add_argument("--research", default=DEFAULT_RESEARCH_PATH, help="Research-dataset parquet to resample and fit on.")
    p.add_argument("--n-replicates", type=int, default=rs.N_REPLICATES_DEFAULT, help="Outer stability-gate replicates.")
    p.add_argument("--mode", choices=rs.RESAMPLE_MODES, default=rs.MODE_DEFAULT, help="Outer perturbation shape (same as refit_stability.py).")
    p.add_argument("--drop-frac", type=float, default=rs.DROP_FRAC_DEFAULT, help="Row-drop fraction for --mode drop1pct (outer perturbation).")
    p.add_argument("--seed", type=int, default=0, help="Master RNG seed for outer replicates.")
    p.add_argument("--n-members", type=int, default=N_MEMBERS_DEFAULT, help="Ensemble size (number of refits averaged per fit).")
    p.add_argument("--row-resample-mode", choices=ROW_RESAMPLE_MODES, default=ROW_RESAMPLE_MODE_DEFAULT,
                    help="Per-member row resampling: 'none' (default, pure seed bagging on identical folds), 'drop1pct', or 'bootstrap'.")
    p.add_argument("--row-resample-drop-frac", type=float, default=rs.DROP_FRAC_DEFAULT, help="Drop fraction for --row-resample-mode drop1pct.")
    p.add_argument("--n-folds", type=int, default=rm.DEFAULT_N_FOLDS)
    p.add_argument("--horizon", type=int, default=rm.PRIMARY_HORIZON)
    p.add_argument("--embargo", type=int, default=rm.DEFAULT_EMBARGO_DAYS)
    p.add_argument("--min-fold-train-rows", type=int, default=rm.MIN_FOLD_TRAIN_ROWS)
    p.add_argument("--top-ks", default=",".join(str(k) for k in TOP_KS), help="Comma-separated slot counts to test.")
    p.add_argument("--vol-col", default=VOL_COL_DEFAULT)
    p.add_argument("--top-frac", type=float, default=rs.TOP_FRAC)
    p.add_argument("--n-vol-buckets", type=int, default=rs.N_VOL_BUCKETS)
    p.add_argument("--n-boot", type=int, default=rs.N_BOOT_DEFAULT)
    p.add_argument("--retention-bar", type=float, default=RETENTION_BAR_DEFAULT)
    p.add_argument("--volmatch-threshold-pp", type=float, default=VOLMATCH_THRESHOLD_PP_DEFAULT)
    p.add_argument("--combine-methods", default=",".join(COMBINE_METHODS_DEFAULT), help="Comma-separated subset of {score, rank}.")
    p.add_argument("--regularized", action="store_true", help="Apply lever 2's heavier-regularization preset (REGULARIZED_LGBM_OVERRIDES) to every member fit.")
    p.add_argument("--num-leaves", type=int, default=None, help="Override, applied after --regularized if both given.")
    p.add_argument("--min-child-samples", type=int, default=None)
    p.add_argument("--reg-lambda", type=float, default=None)
    p.add_argument("--lgbm-n-jobs", type=int, default=None, help="LightGBM thread count per member fit (perf only, does not change scores).")
    p.add_argument("--baseline-only", action="store_true", help="Skip the n_replicates outer loop: just fit the unperturbed ensemble and report its vol-matched excess.")
    p.add_argument("--out-dir", default="research_data")
    p.add_argument("--tag", default="", help="Optional filename tag.")
    p.add_argument("--verbose", action="store_true", help="DEBUG-level logging.")
    return p


def _parse_top_ks(raw: str) -> tuple[int, ...]:
    try:
        return tuple(int(x.strip()) for x in raw.split(",") if x.strip())
    except ValueError:
        raise SystemExit(f"--top-ks: could not parse {raw!r} as a comma-separated list of ints") from None


def _parse_combine_methods(raw: str) -> tuple[str, ...]:
    methods = tuple(x.strip() for x in raw.split(",") if x.strip())
    bad = [m for m in methods if m not in COMBINE_METHOD_COLS]
    if bad:
        raise SystemExit(f"--combine-methods: unknown method(s) {bad}, expected a subset of {list(COMBINE_METHOD_COLS)}")
    return methods


def _assemble_lgbm_overrides(args: argparse.Namespace) -> dict:
    overrides: dict = {}
    if args.regularized:
        overrides.update(REGULARIZED_LGBM_OVERRIDES)
    if args.num_leaves is not None:
        overrides["num_leaves"] = args.num_leaves
    if args.min_child_samples is not None:
        overrides["min_child_samples"] = args.min_child_samples
    if args.reg_lambda is not None:
        overrides["reg_lambda"] = args.reg_lambda
    if args.lgbm_n_jobs is not None:
        overrides["n_jobs"] = args.lgbm_n_jobs
    return overrides


def _print_gate_report(results: dict[str, dict]) -> None:
    cols = [
        "k", "mean_retention", "min_retention", "max_retention", "mean_jaccard", "retention_bar",
        "retention_pass", "volmatch_range_pp", "volmatch_threshold_pp", "volmatch_pass", "verdict",
    ]
    for method, state in results.items():
        print("\n" + "=" * 100)
        print(f"COMBINE METHOD: {method} ({COMBINE_METHOD_COLS[method]})")
        print("=" * 100)
        with pd.option_context("display.max_rows", None, "display.width", 220):
            print(state["summary_df"][cols].to_string(index=False))
        sig = state["volmatch_sig"]
        bv = state["baseline_vol"]["volmatch_excess"]
        print(
            f"\nBaseline (unperturbed) ensemble vol-matched excess: {bv:+.4f}\n"
            f"Replicate vol-matched excess distribution: mean={sig['mean']:+.4f} std={sig['std']:.4f} "
            f"n_positive={sig['n_positive']}/{sig['n_valid']} t={sig['tstat']:.2f} p={sig['pvalue']:.4f}"
        )


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    top_ks = _parse_top_ks(args.top_ks)
    combine_methods = _parse_combine_methods(args.combine_methods)
    lgbm_overrides = _assemble_lgbm_overrides(args)
    if lgbm_overrides:
        log.info("main: lgbm_overrides applied to every member fit: %s", lgbm_overrides)

    if args.baseline_only:
        df_raw = pd.read_parquet(args.research)
        feature_cols, padded_cols = _resolve_feature_cols(df_raw)
        df_orig = _pad_for_schema(df_raw, padded_cols)
        t0 = time.monotonic()
        ensemble_df, meta = build_ensemble_oof(
            df_orig, n_members=args.n_members, base_seed=args.seed, feature_cols=feature_cols,
            row_resample_mode=args.row_resample_mode, row_resample_drop_frac=args.row_resample_drop_frac,
            n_folds=args.n_folds, horizon=args.horizon, embargo=args.embargo,
            min_fold_train_rows=args.min_fold_train_rows, lgbm_overrides=lgbm_overrides,
        )
        log.info("main: baseline-only ensemble fit in %.1fs (%d members)", time.monotonic() - t0, meta["n_members_fit"])
        path = save_ensemble_oof(ensemble_df, out_dir=args.out_dir, tag=args.tag)
        print(f"Baseline ensemble OOF scores written to {os.path.abspath(path)}")

        vol_map = (
            df_orig[["ticker", "event_day", args.vol_col]].dropna(subset=[args.vol_col])
            .drop_duplicates(subset=["ticker", "event_day"], keep="first")
            .set_index(["ticker", "event_day"])[args.vol_col]
        )
        for method in combine_methods:
            score_col = COMBINE_METHOD_COLS[method]
            dedup = rs.dedup_oof(ensemble_df, score_col, context=f"baseline-only [{method}]")
            vol = rs.vol_matched_excess(
                dedup, vol_map, score_col=score_col, label_col=rm.label_col_for_horizon(args.horizon),
                top_frac=args.top_frac, n_vol_buckets=args.n_vol_buckets, n_boot=args.n_boot,
                rng=np.random.default_rng(args.seed + 999_000),
            )
            print(f"[{method}] vol-matched top-decile excess: {vol['volmatch_excess']:+.4f} (n={vol['volmatch_n']})")
        return 0

    t0 = time.monotonic()
    results = run_ensemble_stability_gate(
        args.research,
        n_replicates=args.n_replicates, mode=args.mode, drop_frac=args.drop_frac, seed=args.seed,
        n_members=args.n_members, row_resample_mode=args.row_resample_mode,
        row_resample_drop_frac=args.row_resample_drop_frac,
        n_folds=args.n_folds, horizon=args.horizon, embargo=args.embargo,
        min_fold_train_rows=args.min_fold_train_rows, top_ks=top_ks,
        vol_col=args.vol_col, top_frac=args.top_frac, n_vol_buckets=args.n_vol_buckets, n_boot=args.n_boot,
        retention_bar=args.retention_bar, volmatch_threshold_pp=args.volmatch_threshold_pp,
        combine_methods=combine_methods, lgbm_overrides=lgbm_overrides,
    )
    elapsed = time.monotonic() - t0
    log.info("run_ensemble_stability_gate: done in %.1fs", elapsed)

    for method, state in results.items():
        rep_path, sum_path = save_ensemble_stability_results(
            state["replicate_df"], state["summary_df"], out_dir=args.out_dir,
            mode=args.mode, combine_method=method, tag=args.tag,
        )
        print(f"[{method}] per-replicate detail: {os.path.abspath(rep_path)}")
        print(f"[{method}] summary + verdicts: {os.path.abspath(sum_path)}")

    _print_gate_report(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
