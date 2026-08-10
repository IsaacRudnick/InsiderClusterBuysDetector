"""Refit-stability gate for the ranking model's top-k picks.

WHY this exists. A 2x2 attribution (2026-08-09) found that refitting
research/model.py's ranking model on 118 FEWER rows out of 11,026 (1.07% of
the dataset) moved the flagship strategy `model_ranked_top_hold63` (10
slots) from +229.6% total return to +151.0%, and its annualized alpha from
+11.2% to +5.5%. The ticker-reuse event filter was proven a no-op for that
same strategy (identical results to six figures) -- the ENTIRE move came
from the refit alone. Meanwhile the 5-slot variant (`model_ranked_n05_hold63`)
barely moved: alpha +11.9% -> +10.7%.

Working hypothesis this module exists to TEST, not assume: the top ~5 picks
are genuinely separated in score and survive a refit, while slots 6-10 sit
in a mass of near-tied scores where a tiny perturbation reshuffles holdings
wholesale. A compounding factor already on record in this codebase
(backtest/engine.py's capacity fill grants slots by plain dict iteration
order, not by score) means that even if slots 6-10 ARE near ties, the
engine's PICK among them has no economic content -- so instability at that
rank band would be doubly dangerous, not just noisy.

This script never runs a backtest (a full backtest is 15+ minutes; a model
refit is ~8-25s depending on settings -- see FAST-FIT NOTE below). It
measures stability at the SCORE level, which is the direct and sufficient
layer for this question: whether the model's own ranking of candidates
holds up under a perturbation of the same shape and size as the one that
was already shown to move the backtest.

Method, for N replicates (default 20, --n-replicates):
  1. Resample the research dataset (resample_dataset). Default mode
     "drop1pct" drops a random 1% of rows, matching the 118/11,026 (1.07%)
     perturbation that caused the real collapse. Mode "bootstrap" instead
     draws len(df) rows WITH replacement -- a much larger, differently
     shaped perturbation (classic bootstrap; on average ~36.8% of unique
     rows are excluded from any one draw, since (1 - 1/n)^n -> 1/e). Both
     are useful: drop1pct reproduces the actual documented failure mode;
     bootstrap is a standard, harsher stability check that does not depend
     on getting the "right" perturbation size.
  2. Refit research/model.py's ranking model on the resampled pool and
     regenerate its out-of-fold (OOF) scores by calling
     research.model.fit_and_validate directly (refit_oof_scores) -- the
     SAME purged/embargoed/expanding-window CV code run_research.py's
     --fit-model stage uses. Nothing about folds, purging or embargo is
     reimplemented here.
  3. Compare the replicate's OOF ranking against the baseline OOF file on
     disk (research_data/oof_scores_noreuse_20260808.parquet by default) at
     the SCORE level: top-k overlap, rank churn, score-gap mechanism check,
     and a volatility-matched top-decile excess re-measurement.
  4. Emit a PASS/FAIL verdict per k against a documented threshold.

FAST-FIT NOTE. fit_and_validate's per-fold OOF loop (the only thing this
module reads from its result) is computed BEFORE run_label_shuffle_test and
run_shap_interaction_analysis run, and neither of those touches
result.oof_scores afterward -- they are independent diagnostics computed
from separate model fits. So this module calls fit_and_validate with
run_shap_interactions=False and n_shuffle_seeds=2 (the lowest value
run_label_shuffle_test accepts) to skip ~90% of the per-replicate fit cost
(the shuffle test alone fits n_shuffle_seeds x n_folds regressors -- 20 x 5
= 100 fits at the default n_shuffle_seeds=20, versus about 15 fits for the
OOF loop itself). This was verified empirically before relying on it: a
fast-settings fit and a full-settings fit on the same input
(research_noreuse_10908rows_20260808.parquet) produce byte-identical
oof_tail_classifier scores (max abs diff 0.0, see this module's test suite).
Measured: full settings ~25s/fit, fast settings ~8s/fit.

Retention bar (--retention-bar, default 0.70), chosen and justified rather
than picked silently. `backtest/strategies.py`'s model_ranked_n{NN}_hold63
family sizes every slot so that slots * weight == 1.0 -- see
tests/test_model_ranked_slots.py -- so each of a strategy's k slots carries
exactly 1/k of the fully-deployed portfolio weight. A mean top-k retention
below 0.70 across replicates means that, on average, MORE THAN 30% of a
strategy's portfolio weight (3+ of 10 slots, at k=10) would be reshuffled by
a perturbation of only 1% of the training data -- a fraction of that
magnitude is already consistent with the real collapse this module exists
to explain (alpha roughly halved, +11.2% -> +5.5%, on exactly this kind of
perturbation). 0.70 is a deliberately loose bar (it tolerates real,
expected churn from near-ties) that still fails decisively on wholesale
reshuffling.

Vol-matched excess threshold (--volmatch-threshold-pp, default 0.02 = 2
percentage points), the alpha-equivalent gate at the score level. This
default is specified directly by the task this module was built for. It is
also independently sane against the reference number: the noreuse baseline
fit's own vol-matched top-decile excess is +4.74pp (p=0.004, positive in 4
of 5 folds -- see research/model.py's PRODUCTION_SCORE_MODEL comment and
this repo's volmatch-style scratch analyses). A 2pp move is >40% of that
entire measured edge -- comparable in scale to the ~50% relative alpha drop
(+11.2% -> +5.5%) the real collapse produced. Gated on the FULL range
spanned by {baseline value} union {every replicate value} (see
_build_summary): this is the most direct, literal reading of "moves less
than ~2pp across replicates" -- it catches both replicate-to-replicate
disagreement and drift away from the original baseline fit in one number.

Inputs (both required, --research / --oof):
  research_data/research_noreuse_10908rows_20260808.parquet (the resampled
    pool -- also supplies x_vol_63_ann for the vol-match step)
  research_data/oof_scores_noreuse_20260808.parquet (the baseline OOF
    scores every replicate is compared against)

Outputs, written atomically (temp file + os.replace, matching
ticker_reuse.py / split_fingerprint.py / research.model.save_oof_scores):
  research_data/stability_replicates_<mode>_<tag_><YYYYMMDD>.parquet
    One row per replicate: retention/jaccard per k, rank churn, score-gap
    mechanism check, vol-matched excess, fit time.
  research_data/stability_summary_<mode>_<tag_><YYYYMMDD>.csv
    One row per k: aggregated retention stats, the shared vol-match gate,
    and a PASS/FAIL verdict.

Usage:
    python tools/refit_stability.py
    python tools/refit_stability.py --n-replicates 20 --mode drop1pct
    python tools/refit_stability.py --mode bootstrap --n-replicates 20
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

# Run directly (`python tools/refit_stability.py ...`), the interpreter puts
# this file's own directory (tools/) on sys.path[0], not the repo root -- so
# `from research import model` below would fail without this. Harmless
# no-op when this module is instead imported normally (e.g. `from tools
# import refit_stability`), since the repo root is already on sys.path in
# that case.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from research import model as rm

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunables -- see the module docstring for the justification behind each of
# the non-obvious ones (retention bar, volmatch threshold, fast-fit settings).
# ---------------------------------------------------------------------------
DEFAULT_RESEARCH_PATH = "research_data/research_noreuse_10908rows_20260808.parquet"
DEFAULT_OOF_PATH = "research_data/oof_scores_noreuse_20260808.parquet"

N_REPLICATES_DEFAULT = 20
RESAMPLE_MODES = ("drop1pct", "bootstrap")
MODE_DEFAULT = "drop1pct"
DROP_FRAC_DEFAULT = 0.01  # matches the 118/11,026 == 1.07% perturbation that caused the real collapse

TOP_KS: tuple[int, ...] = (5, 10, 15, 25)
SCORE_COL_DEFAULT = "oof_tail_classifier"  # matches backtest.model_scores.DEFAULT_SCORE_COL
VOL_COL_DEFAULT = "x_vol_63_ann"

# Vol-match methodology, identical to the scratch volmatch.py this module's
# docstring's "Vol-matched excess" section is reusing: stratify into decile
# vol buckets, take the top 10% of the pool by score, resample the pool to
# the picks' vol-bucket mix, 2000 draws.
TOP_FRAC = 0.10
N_VOL_BUCKETS = 10
N_BOOT_DEFAULT = 2000

RANK_CHURN_TOP_N = 10

# The lowest n_shuffle_seeds run_label_shuffle_test accepts (it raises below
# 2). See the module docstring's FAST-FIT NOTE for why this has zero effect
# on the OOF scores this module actually reads.
FAST_N_SHUFFLE_SEEDS = 2

RETENTION_BAR_DEFAULT = 0.70
VOLMATCH_THRESHOLD_PP_DEFAULT = 0.02


# ---------------------------------------------------------------------------
# 0. Feature-column workaround. research.model.FEATURE_COLS (imported at
# MODULE LOAD TIME from backtest.research._FEATURE_COLS) has drifted ahead
# of this module's pinned inputs (DEFAULT_RESEARCH_PATH /
# research_noreuse_10908rows_20260808.parquet): a later change to
# backtest/research.py added 9 concurrent-selling feature columns that do
# not exist in that pinned file, so calling fit_and_validate on it with the
# default feature_cols=None now raises "missing x_ feature column(s)" --
# confirmed by running `python tools/refit_stability.py` with no arguments after
# that change: every replicate (including the plain production-threshold
# baseline) fails to fit, not just this module's own tail_thresh additions.
#
# ensemble_model.py hit the identical problem against the identical pinned
# file and already carries a documented, proven-safe fix (see that
# module's "FEATURE-COLUMN WORKAROUND" docstring section): resolve the
# USABLE subset of the current FEATURE_COLS list, pass that explicit subset
# to every fit_and_validate call, and pad the missing columns onto a COPY
# of the pool as all-NaN purely so the unconditional schema-presence check
# passes (the padded columns are never in feature_cols, so they are never
# split on, never enter winsorization, never enter find_leaky_features).
# Duplicated here rather than imported, because ensemble_model.py imports
# THIS module (`import refit_stability as rs`) -- importing back would be
# circular.
def _resolve_feature_cols(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """(usable_feature_cols, padded_cols) -- see the block comment above."""
    usable = [c for c in rm.FEATURE_COLS if c in df.columns]
    padded = [c for c in rm.FEATURE_COLS if c not in df.columns]
    return usable, padded


def _pad_for_schema(df: pd.DataFrame, padded_cols: list[str]) -> pd.DataFrame:
    """Return a COPY of `df` with every column in `padded_cols` added as
    all-NaN. A no-op (returns `df` unchanged) when padded_cols is empty."""
    if not padded_cols:
        return df
    out = df.copy()
    for c in padded_cols:
        out[c] = np.nan
    return out


# ---------------------------------------------------------------------------
# 1. Resampling
# ---------------------------------------------------------------------------
def resample_dataset(
    df: pd.DataFrame, mode: str = MODE_DEFAULT, seed: int = 0, drop_frac: float = DROP_FRAC_DEFAULT,
) -> pd.DataFrame:
    """Perturb `df` by row, returning a fresh frame with a clean RangeIndex.

    "drop1pct": drop round(len(df) * drop_frac) rows chosen uniformly at
    random, without replacement -- reproduces the shape of the real
    118/11,026 perturbation.

    "bootstrap": draw len(df) rows WITH replacement from `df`. This can (and
    typically does) produce duplicate (ticker, event_day) rows -- see
    dedup_oof, which every downstream consumer of a resampled pool's OOF
    output goes through specifically to handle that.

    The index is reset (0..len-1) rather than kept from `df`: entry_idx (a
    data COLUMN, not the index) is what fit_and_validate's purging logic
    actually uses, so resetting the pandas index here is purely
    bookkeeping hygiene -- it guarantees make_purged_expanding_folds' and
    fit_and_validate's df.loc[fold.train_idx / test_idx] lookups always hit
    a unique row, even under "bootstrap" where the ORIGINAL row could have
    been drawn twice (a duplicate original index would otherwise make
    df.loc return multiple rows for one label and silently corrupt fold
    sizes).
    """
    if mode not in RESAMPLE_MODES:
        raise ValueError(f"resample_dataset: unknown mode {mode!r}, expected one of {RESAMPLE_MODES}")
    n = len(df)
    if n == 0:
        raise ValueError("resample_dataset: empty input dataframe")

    rng = np.random.default_rng(seed)
    if mode == "drop1pct":
        n_drop = int(round(n * drop_frac))
        n_drop = max(1, min(n_drop, n - 1))
        drop_positions = rng.choice(n, size=n_drop, replace=False)
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[drop_positions] = False
        out = df.iloc[keep_mask]
    else:  # bootstrap
        positions = rng.integers(0, n, size=n)
        out = df.iloc[positions]

    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Refit (thin wrapper -- see module docstring's FAST-FIT NOTE)
# ---------------------------------------------------------------------------
def refit_oof_scores(
    df_pool: pd.DataFrame, *, n_folds: int = rm.DEFAULT_N_FOLDS, horizon: int = rm.PRIMARY_HORIZON,
    embargo: int = rm.DEFAULT_EMBARGO_DAYS, min_fold_train_rows: int = rm.MIN_FOLD_TRAIN_ROWS,
    tail_thresh: float = rm.TAIL_THRESH, feature_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Run research.model.fit_and_validate on `df_pool` and return its
    oof_scores frame. run_shap_interactions=False and n_shuffle_seeds=2 are
    hard-coded (not exposed as knobs) because they change ONLY the cost of
    this call, never its oof_scores output -- see the module docstring.

    tail_thresh (default rm.TAIL_THRESH, 0.20, the production default)
    lets a caller run the stability gate against a candidate objective
    other than the production `adj_63 > 0.20` target -- see
    objective_sweep.py, which compares several thresholds' top-decile
    distribution shape and uses this parameter to stability-test whichever
    ones look better than the incumbent.

    feature_cols (default None -> fit_and_validate's own default,
    rm.FEATURE_COLS): pass the resolved usable subset from
    run_stability_gate's own _resolve_feature_cols/_pad_for_schema
    workaround (see the block comment above those functions) so every
    replicate fit uses the exact same feature set as the baseline."""
    result = rm.fit_and_validate(
        df_pool, n_folds=n_folds, horizon=horizon, embargo=embargo,
        min_fold_train_rows=min_fold_train_rows, feature_cols=feature_cols,
        run_shap_interactions=False, n_shuffle_seeds=FAST_N_SHUFFLE_SEEDS,
        tail_thresh=tail_thresh,
    )
    return result.oof_scores


# ---------------------------------------------------------------------------
# 3. Ranking helpers
# ---------------------------------------------------------------------------
def dedup_oof(oof: pd.DataFrame, score_col: str, *, context: str = "") -> pd.DataFrame:
    """One row per (ticker, event_day), keeping the first occurrence, after
    dropping rows with a null score. (ticker, event_day) is a unique key in
    the real research dataset (see backtest.model_scores.load_model_scores's
    own note), but --mode bootstrap resamples WITH replacement and can draw
    the same original row twice, which would otherwise surface as a genuine
    duplicate key in a replicate's OOF output."""
    scored = oof[oof[score_col].notna()].copy()
    deduped = scored.drop_duplicates(subset=["ticker", "event_day"], keep="first")
    n_dupes = len(scored) - len(deduped)
    if n_dupes:
        log.info(
            "dedup_oof%s: dropped %d duplicate (ticker, event_day) row(s)",
            f" [{context}]" if context else "", n_dupes,
        )
    return deduped.reset_index(drop=True)


def ranked_picks(oof_dedup: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """Sort `oof_dedup` descending by score_col, tie-broken deterministically
    by (ticker, event_day) ascending so repeated calls on identical input
    always produce an identical order (mergesort is stable, but the
    tie-break columns make the result independent of input row order too).
    Adds a 1-indexed `rank` column, rank 1 == highest score."""
    out = oof_dedup.sort_values(
        [score_col, "ticker", "event_day"], ascending=[False, True, True], kind="mergesort",
    ).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def top_k_keys(ranked: pd.DataFrame, k: int) -> set[tuple]:
    """{(ticker, event_day), ...} of the top-k rows of an already-ranked
    frame (see ranked_picks)."""
    top = ranked.head(k)
    return set(zip(top["ticker"], top["event_day"]))


def jaccard_index(a: set, b: set) -> float:
    """|a & b| / |a | b|. NaN if both sets are empty (undefined, not 0)."""
    union = a | b
    if not union:
        return float("nan")
    return len(a & b) / len(union)


def retention_rate(baseline_set: set, other_set: set) -> float:
    """Fraction of `baseline_set` also present in `other_set`. NaN if
    baseline_set is empty."""
    if not baseline_set:
        return float("nan")
    return len(baseline_set & other_set) / len(baseline_set)


# ---------------------------------------------------------------------------
# 4. Score-separation mechanism check
# ---------------------------------------------------------------------------
def score_gap_metrics(ranked: pd.DataFrame, score_col: str) -> dict:
    """Score gap between rank 5/6 and rank 10/11, relative to the score
    distribution's own spread (interquartile range, p75 - p25 -- chosen
    over stdev because oof_tail_classifier is a bounded [0, 1] probability
    that piles up near its extremes, which inflates stdev relative to the
    bulk of the distribution; IQR is not sensitive to that pile-up).

    This is the direct mechanism check for the working hypothesis: if
    slots 6-10 sit in a mass of near-tied scores, gap_10_11_norm should be
    small (close to 0) while gap_5_6_norm should be comparatively larger.
    """
    scores = ranked[score_col].to_numpy(dtype=float)
    n = len(scores)
    if n >= 4:
        q75, q25 = np.percentile(scores, [75, 25])
        iqr = float(q75 - q25)
    else:
        iqr = float("nan")

    def _gap(rank_a: int, rank_b: int) -> float:
        if n < rank_b:
            return float("nan")
        return float(scores[rank_a - 1] - scores[rank_b - 1])

    gap_5_6 = _gap(5, 6)
    gap_10_11 = _gap(10, 11)
    return {
        "score_iqr": iqr,
        "gap_5_6": gap_5_6,
        "gap_10_11": gap_10_11,
        "gap_5_6_norm": (gap_5_6 / iqr) if (iqr == iqr and iqr > 0 and gap_5_6 == gap_5_6) else float("nan"),
        "gap_10_11_norm": (gap_10_11 / iqr) if (iqr == iqr and iqr > 0 and gap_10_11 == gap_10_11) else float("nan"),
    }


# ---------------------------------------------------------------------------
# 5. Rank churn
# ---------------------------------------------------------------------------
def rank_churn(baseline_ranked: pd.DataFrame, replicate_ranked: pd.DataFrame, top_n: int = RANK_CHURN_TOP_N) -> dict:
    """For each of the baseline's top-`top_n` picks, how far did that same
    (ticker, event_day) move in the replicate's ranking?

    A pick entirely absent from the replicate's OOF output (dropped by the
    perturbation itself, or shifted into a different fold's train-only
    block) is NOT skipped -- it is charged a penalty rank of
    len(replicate_ranked) + 1, i.e. treated as having fallen off the bottom
    of the ranking, and counted separately in rank_churn_n_missing. Silently
    excluding missing picks from the mean would understate churn exactly
    when the model is least stable.
    """
    base_top = baseline_ranked.head(top_n)
    repl_rank_by_key = dict(
        zip(zip(replicate_ranked["ticker"], replicate_ranked["event_day"]), replicate_ranked["rank"])
    )
    n_replicate_pool = len(replicate_ranked)

    shifts: list[float] = []
    n_missing = 0
    for _, row in base_top.iterrows():
        key = (row["ticker"], row["event_day"])
        base_rank = int(row["rank"])
        repl_rank = repl_rank_by_key.get(key)
        if repl_rank is None:
            n_missing += 1
            repl_rank = n_replicate_pool + 1
        shifts.append(abs(int(repl_rank) - base_rank))

    shifts_arr = np.array(shifts, dtype=float)
    return {
        "rank_churn_mean_abs": float(np.mean(shifts_arr)) if len(shifts_arr) else float("nan"),
        "rank_churn_median_abs": float(np.median(shifts_arr)) if len(shifts_arr) else float("nan"),
        "rank_churn_n_missing": n_missing,
        "rank_churn_n_baseline": len(base_top),
    }


# ---------------------------------------------------------------------------
# 6. Vol-matched top-decile excess (methodology reused from the scratch
#    volmatch.py analysis this repo's baseline +4.74pp figure comes from --
#    see the module docstring)
# ---------------------------------------------------------------------------
def vol_matched_excess(
    oof_dedup: pd.DataFrame, vol_map: pd.Series, *, score_col: str, label_col: str,
    top_frac: float = TOP_FRAC, n_vol_buckets: int = N_VOL_BUCKETS, n_boot: int = N_BOOT_DEFAULT,
    rng: np.random.Generator,
) -> dict:
    """Mean label of the top `top_frac` of `oof_dedup` by score_col, minus a
    synthetic benchmark drawn (n_boot draws) from the FULL pool with the
    identical volatility-decile composition as the picks. `vol_map` is a
    {(ticker, event_day): x_vol_63_ann} Series built ONCE from the original,
    unperturbed research dataset (see run_stability_gate) rather than
    re-derived from each replicate's own resampled pool -- a bootstrap
    replicate can contain duplicate (ticker, event_day) keys, and merging
    against a per-replicate vol column would multiply rows on every
    duplicate; merging against a fixed, pre-deduplicated mapping cannot.
    """
    df = oof_dedup.merge(vol_map.rename("_vol").reset_index(), on=["ticker", "event_day"], how="left")
    df = df[df[label_col].notna() & df["_vol"].notna()].copy()

    min_usable = n_vol_buckets * 3  # need enough rows per bucket for qcut + resampling to be meaningful
    if len(df) < min_usable:
        log.warning(
            "vol_matched_excess: only %d usable row(s) (< %d) -- returning NaN excess",
            len(df), min_usable,
        )
        return {"volmatch_n": len(df), "volmatch_raw_mean": float("nan"),
                "volmatch_bench_mean": float("nan"), "volmatch_excess": float("nan")}

    df["vol_bucket"] = pd.qcut(df["_vol"].rank(method="first"), n_vol_buckets, labels=False)
    k = max(int(len(df) * top_frac), 30)
    top = df.nlargest(k, score_col, keep="all").head(k)

    raw = float(top[label_col].mean())
    comp = top["vol_bucket"].value_counts()
    by_bucket = {b: g[label_col].to_numpy() for b, g in df.groupby("vol_bucket")}

    draws = np.empty(n_boot)
    for i in range(n_boot):
        vals = [
            rng.choice(by_bucket[b], size=int(cnt), replace=True)
            for b, cnt in comp.items() if b in by_bucket
        ]
        draws[i] = np.concatenate(vals).mean()

    bench = float(draws.mean())
    return {"volmatch_n": len(top), "volmatch_raw_mean": raw, "volmatch_bench_mean": bench, "volmatch_excess": raw - bench}


# ---------------------------------------------------------------------------
# 7. Orchestration
# ---------------------------------------------------------------------------
def run_stability_gate(
    research_path: str = DEFAULT_RESEARCH_PATH,
    oof_path: str = DEFAULT_OOF_PATH,
    *,
    n_replicates: int = N_REPLICATES_DEFAULT,
    mode: str = MODE_DEFAULT,
    drop_frac: float = DROP_FRAC_DEFAULT,
    seed: int = 0,
    n_folds: int = rm.DEFAULT_N_FOLDS,
    horizon: int = rm.PRIMARY_HORIZON,
    embargo: int = rm.DEFAULT_EMBARGO_DAYS,
    min_fold_train_rows: int = rm.MIN_FOLD_TRAIN_ROWS,
    tail_thresh: float = rm.TAIL_THRESH,
    top_ks: tuple[int, ...] = TOP_KS,
    score_col: str = SCORE_COL_DEFAULT,
    vol_col: str = VOL_COL_DEFAULT,
    top_frac: float = TOP_FRAC,
    n_vol_buckets: int = N_VOL_BUCKETS,
    n_boot: int = N_BOOT_DEFAULT,
    retention_bar: float = RETENTION_BAR_DEFAULT,
    volmatch_threshold_pp: float = VOLMATCH_THRESHOLD_PP_DEFAULT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full stability gate. Returns (replicate_df, summary_df):

    replicate_df: one row per replicate that fit successfully (a replicate
      whose fit_and_validate call raises is logged and skipped, not fatal --
      see the loop below).
    summary_df: one row per k in top_ks, with aggregated retention stats,
      the shared vol-match gate, and a PASS/FAIL verdict (see
      _build_summary).

    tail_thresh (default rm.TAIL_THRESH): the SAME threshold every replicate
    refit AND `oof_path`'s baseline must have been fit with -- see
    refit_oof_scores' docstring. This function does not verify that
    `oof_path` actually matches `tail_thresh`; passing a mismatched pair
    (e.g. a thresh=0.20 baseline against tail_thresh=0.05 replicates) would
    silently compare two different objectives' top-k sets. objective_sweep.py
    is responsible for generating and pointing --oof at a baseline fit with
    the SAME tail_thresh being gated.

    Raises RuntimeError if every replicate failed to fit.
    """
    log.info("run_stability_gate: loading research dataset %s", research_path)
    df_raw = pd.read_parquet(research_path)
    feature_cols, padded_cols = _resolve_feature_cols(df_raw)
    if padded_cols:
        log.info(
            "run_stability_gate: padding %d feature column(s) missing from %s as all-NaN "
            "(excluded from feature_cols passed to every fit) -- see this module's "
            "FEATURE-COLUMN WORKAROUND block comment: %s",
            len(padded_cols), research_path, padded_cols,
        )
    df_orig = _pad_for_schema(df_raw, padded_cols)
    log.info("run_stability_gate: loading baseline OOF scores %s", oof_path)
    baseline_oof_raw = pd.read_parquet(oof_path)
    log.info(
        "run_stability_gate: %d research rows, %d baseline OOF rows, mode=%s, n_replicates=%d",
        len(df_orig), len(baseline_oof_raw), mode, n_replicates,
    )

    label_col = rm.LABEL_COL
    vol_map = (
        df_orig[["ticker", "event_day", vol_col]]
        .dropna(subset=[vol_col])
        .drop_duplicates(subset=["ticker", "event_day"], keep="first")
        .set_index(["ticker", "event_day"])[vol_col]
    )

    baseline_dedup = dedup_oof(baseline_oof_raw, score_col, context="baseline")
    baseline_ranked = ranked_picks(baseline_dedup, score_col)
    baseline_topk = {k: top_k_keys(baseline_ranked, k) for k in top_ks}
    baseline_gap = score_gap_metrics(baseline_ranked, score_col)
    baseline_vol = vol_matched_excess(
        baseline_dedup, vol_map, score_col=score_col, label_col=label_col,
        top_frac=top_frac, n_vol_buckets=n_vol_buckets, n_boot=n_boot,
        rng=np.random.default_rng(seed + 999_000),
    )
    log.info(
        "baseline: n_oof=%d gap_5_6_norm=%.3f gap_10_11_norm=%.3f volmatch_excess=%+.4f",
        len(baseline_ranked), baseline_gap["gap_5_6_norm"], baseline_gap["gap_10_11_norm"],
        baseline_vol["volmatch_excess"],
    )

    records: list[dict] = []
    for rep in range(n_replicates):
        rep_seed = seed * 1_000_003 + rep  # large multiplier so nearby replicate ids don't share RNG state
        t0 = time.monotonic()
        try:
            df_pool = resample_dataset(df_orig, mode=mode, seed=rep_seed, drop_frac=drop_frac)
            oof = refit_oof_scores(
                df_pool, n_folds=n_folds, horizon=horizon, embargo=embargo,
                min_fold_train_rows=min_fold_train_rows, tail_thresh=tail_thresh,
                feature_cols=feature_cols,
            )
        except Exception as exc:  # noqa: BLE001 -- one bad replicate must not sink the whole run
            log.warning("replicate %d/%d: fit failed (%s) -- skipped", rep + 1, n_replicates, exc)
            continue

        dedup = dedup_oof(oof, score_col, context=f"replicate {rep}")
        ranked = ranked_picks(dedup, score_col)
        gap = score_gap_metrics(ranked, score_col)
        churn = rank_churn(baseline_ranked, ranked, top_n=RANK_CHURN_TOP_N)
        vol = vol_matched_excess(
            dedup, vol_map, score_col=score_col, label_col=label_col,
            top_frac=top_frac, n_vol_buckets=n_vol_buckets, n_boot=n_boot,
            rng=np.random.default_rng(rep_seed + 500_000),
        )

        row = {
            "replicate_id": rep,
            "mode": mode,
            "seed": rep_seed,
            "n_pool_rows": len(df_pool),
            "n_oof_rows": len(ranked),
            "fit_seconds": time.monotonic() - t0,
        }
        for k in top_ks:
            repl_set = top_k_keys(ranked, k)
            row[f"retention_k{k}"] = retention_rate(baseline_topk[k], repl_set)
            row[f"jaccard_k{k}"] = jaccard_index(baseline_topk[k], repl_set)
        row.update(gap)
        row.update(churn)
        row.update(vol)
        records.append(row)

        log.info(
            "replicate %d/%d done in %.1fs: n_oof=%d retention_k5=%.2f retention_k10=%.2f "
            "gap_10_11_norm=%.3f volmatch_excess=%+.4f",
            rep + 1, n_replicates, row["fit_seconds"], row["n_oof_rows"],
            row.get("retention_k5", float("nan")), row.get("retention_k10", float("nan")),
            row["gap_10_11_norm"], row["volmatch_excess"],
        )

    replicate_df = pd.DataFrame.from_records(records)
    if replicate_df.empty:
        raise RuntimeError("run_stability_gate: every replicate failed to fit -- nothing to report")
    if len(replicate_df) < n_replicates:
        log.warning(
            "run_stability_gate: only %d of %d replicate(s) produced a usable fit",
            len(replicate_df), n_replicates,
        )

    summary_df = _build_summary(
        replicate_df, baseline_gap=baseline_gap, baseline_vol=baseline_vol, top_ks=top_ks,
        retention_bar=retention_bar, volmatch_threshold_pp=volmatch_threshold_pp,
    )
    return replicate_df, summary_df


def _build_summary(
    replicate_df: pd.DataFrame, *, baseline_gap: dict, baseline_vol: dict, top_ks: tuple[int, ...],
    retention_bar: float, volmatch_threshold_pp: float,
) -> pd.DataFrame:
    """One row per k. The vol-match gate is NOT k-specific (vol-matched
    excess is measured over a fixed top-decile of the pool, independent of
    k) -- it is computed once and repeated on every row, so it can gate
    every k's verdict equally: an unstable alpha-equivalent makes every
    slot count untrustworthy, not just the one k happens to test.

    volmatch_range_pp is the full range spanned by {baseline_vol} union
    {every replicate's volmatch_excess} -- see the module docstring's
    "Vol-matched excess threshold" section for why this is the chosen
    "moves less than ~2pp across replicates" statistic.

    KNOWN LIMITATION of that choice, documented 2026-08-09 after the first
    ensemble run: range is max-minus-min, so it GROWS WITH n_replicates even
    for a perfectly stable estimator. For 20 draws from a normal the expected
    range is about 3.7 sigma, so an estimator with sigma=0.57pp is expected to
    span ~2.1pp and will fail a 2pp range bar on nothing but sampling noise.
    Range is also not comparable between runs with different --n-replicates.

    So we ALSO emit volmatch_mean_shift_pp (|replicate mean - baseline|), which
    does not scale with n and is the statistic that actually answers "did the
    edge move". The ensemble run that exposed this had range 2.30pp (FAIL) but
    a mean shift of 0.05pp with sigma 0.57pp -- two very different stories.

    `verdict` DELIBERATELY still keys off the range test. The bar was set
    before the numbers were seen, and switching the criterion afterwards, on a
    run whose verdict it would flip, is exactly the move that makes a gate
    worthless. Both statistics are reported; a human decides whether to
    re-specify the bar, and if so, re-runs everything against it.
    """
    n = len(replicate_df)
    repl_vol_vals = replicate_df["volmatch_excess"].dropna().to_numpy()
    baseline_excess = baseline_vol["volmatch_excess"]
    all_vals = np.concatenate([repl_vol_vals, [baseline_excess]]) if baseline_excess == baseline_excess else repl_vol_vals
    if len(all_vals):
        volmatch_range = float(all_vals.max() - all_vals.min())
    else:
        volmatch_range = float("nan")
    volmatch_pass = bool(volmatch_range == volmatch_range and volmatch_range < volmatch_threshold_pp)

    replicate_mean_volmatch = float(repl_vol_vals.mean()) if len(repl_vol_vals) else float("nan")
    replicate_std_volmatch = float(repl_vol_vals.std(ddof=1)) if len(repl_vol_vals) > 1 else float("nan")

    # Reported alongside the range test, never used for `verdict` -- see the
    # docstring's KNOWN LIMITATION note for why both exist and why the verdict
    # was not switched to this one after the fact.
    if replicate_mean_volmatch == replicate_mean_volmatch and baseline_excess == baseline_excess:
        volmatch_mean_shift = abs(replicate_mean_volmatch - baseline_excess)
    else:
        volmatch_mean_shift = float("nan")
    volmatch_mean_shift_pass = bool(
        volmatch_mean_shift == volmatch_mean_shift
        and volmatch_mean_shift < volmatch_threshold_pp
    )

    rows = []
    for k in top_ks:
        col, jcol = f"retention_k{k}", f"jaccard_k{k}"
        mean_ret = float(replicate_df[col].mean())
        retention_pass = bool(mean_ret == mean_ret and mean_ret >= retention_bar)
        rows.append({
            "k": k,
            "n_replicates": n,
            "mean_retention": mean_ret,
            "min_retention": float(replicate_df[col].min()),
            "max_retention": float(replicate_df[col].max()),
            "mean_jaccard": float(replicate_df[jcol].mean()),
            "retention_bar": retention_bar,
            "retention_pass": retention_pass,
            "baseline_volmatch_excess": baseline_excess,
            "replicate_mean_volmatch_excess": replicate_mean_volmatch,
            "replicate_std_volmatch_excess": replicate_std_volmatch,
            "volmatch_range_pp": volmatch_range,
            "volmatch_mean_shift_pp": volmatch_mean_shift,
            "volmatch_mean_shift_pass": volmatch_mean_shift_pass,
            "volmatch_threshold_pp": volmatch_threshold_pp,
            "volmatch_pass": volmatch_pass,
            "baseline_gap_5_6_norm": baseline_gap["gap_5_6_norm"],
            "baseline_gap_10_11_norm": baseline_gap["gap_10_11_norm"],
            "replicate_mean_gap_5_6_norm": float(replicate_df["gap_5_6_norm"].mean()),
            "replicate_mean_gap_10_11_norm": float(replicate_df["gap_10_11_norm"].mean()),
            "verdict": "PASS" if (retention_pass and volmatch_pass) else "FAIL",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 8. Persistence (atomic write -- matches ticker_reuse.py / split_fingerprint.py
#    / research.model.save_oof_scores' temp-file-then-os.replace pattern)
# ---------------------------------------------------------------------------
def _atomic_write_parquet(df: pd.DataFrame, path: str) -> None:
    tmp_path = path + ".tmp"
    df.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, path)


def _atomic_write_csv(df: pd.DataFrame, path: str) -> None:
    tmp_path = path + ".tmp"
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, path)


def save_stability_results(
    replicate_df: pd.DataFrame, summary_df: pd.DataFrame, *, out_dir: str, mode: str, tag: str = "",
) -> tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    tag_part = f"{tag}_" if tag else ""
    today = date.today().strftime("%Y%m%d")
    rep_path = os.path.join(out_dir, f"stability_replicates_{mode}_{tag_part}{today}.parquet")
    sum_path = os.path.join(out_dir, f"stability_summary_{mode}_{tag_part}{today}.csv")
    _atomic_write_parquet(replicate_df, rep_path)
    _atomic_write_csv(summary_df, sum_path)
    log.info("save_stability_results: wrote %s and %s", rep_path, sum_path)
    return rep_path, sum_path


# ---------------------------------------------------------------------------
# 9. CLI
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Refit-stability gate: refits research/model.py's ranking model N times "
            "under a small perturbation of its training data and measures whether "
            "the top-k picks (k in --top-ks) hold up, or reshuffle -- the score-level "
            "test for whether model_ranked_* strategies at a given slot count are "
            "safe to trust. See this module's docstring for the finding it exists "
            "to operationalize."
        )
    )
    p.add_argument("--research", default=DEFAULT_RESEARCH_PATH, help="Research-dataset parquet to resample.")
    p.add_argument("--oof", default=DEFAULT_OOF_PATH, help="Baseline OOF-scores parquet every replicate is compared against.")
    p.add_argument("--n-replicates", type=int, default=N_REPLICATES_DEFAULT, help="Number of refit replicates.")
    p.add_argument("--mode", choices=RESAMPLE_MODES, default=MODE_DEFAULT, help="Perturbation shape.")
    p.add_argument("--drop-frac", type=float, default=DROP_FRAC_DEFAULT, help="Row-drop fraction for --mode drop1pct.")
    p.add_argument("--seed", type=int, default=0, help="Master RNG seed (each replicate derives its own seed from this).")
    p.add_argument("--n-folds", type=int, default=rm.DEFAULT_N_FOLDS)
    p.add_argument("--horizon", type=int, default=rm.PRIMARY_HORIZON)
    p.add_argument("--embargo", type=int, default=rm.DEFAULT_EMBARGO_DAYS)
    p.add_argument("--min-fold-train-rows", type=int, default=rm.MIN_FOLD_TRAIN_ROWS)
    p.add_argument(
        "--tail-thresh", type=float, default=rm.TAIL_THRESH,
        help=(
            "adj_63 > this value is the tail-classifier target every replicate refits. "
            "Must match whatever threshold --oof's baseline was itself fit with -- see "
            "objective_sweep.py, which generates a matched baseline OOF file per candidate "
            "threshold and is the intended way to gate a non-default threshold."
        ),
    )
    p.add_argument(
        "--top-ks", default=",".join(str(k) for k in TOP_KS),
        help="Comma-separated slot counts to test, e.g. '5,10,15,25'.",
    )
    p.add_argument("--score-col", default=SCORE_COL_DEFAULT)
    p.add_argument("--vol-col", default=VOL_COL_DEFAULT)
    p.add_argument("--top-frac", type=float, default=TOP_FRAC, help="Top fraction of the pool used for vol-matched excess.")
    p.add_argument("--n-vol-buckets", type=int, default=N_VOL_BUCKETS)
    p.add_argument("--n-boot", type=int, default=N_BOOT_DEFAULT, help="Bootstrap draws for the vol-matched benchmark.")
    p.add_argument("--retention-bar", type=float, default=RETENTION_BAR_DEFAULT, help="See module docstring for justification.")
    p.add_argument("--volmatch-threshold-pp", type=float, default=VOLMATCH_THRESHOLD_PP_DEFAULT, help="See module docstring for justification.")
    p.add_argument("--out-dir", default="research_data")
    p.add_argument("--tag", default="", help="Optional filename tag, e.g. stability_replicates_<mode>_<tag>_<date>.parquet.")
    p.add_argument("--verbose", action="store_true", help="DEBUG-level logging.")
    return p


def _parse_top_ks(raw: str) -> tuple[int, ...]:
    try:
        return tuple(int(x.strip()) for x in raw.split(",") if x.strip())
    except ValueError:
        raise SystemExit(f"--top-ks: could not parse {raw!r} as a comma-separated list of ints") from None


def _print_report(replicate_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    print(f"\n{len(replicate_df)} replicate(s) fit successfully.\n")
    print("=" * 100)
    print("PER-K VERDICT")
    print("=" * 100)
    cols = [
        "k", "mean_retention", "min_retention", "max_retention", "mean_jaccard", "retention_bar",
        "retention_pass", "volmatch_range_pp", "volmatch_mean_shift_pp",
        "volmatch_threshold_pp", "volmatch_pass", "verdict",
    ]
    with pd.option_context("display.max_rows", None, "display.width", 220):
        print(summary_df[cols].to_string(index=False))

    if not summary_df.empty:
        r0 = summary_df.iloc[0]
        print("\nScore-separation mechanism check (IQR-normalized gap; near 0 == near-tied ranks):")
        print(f"  baseline   gap_5_6_norm={r0['baseline_gap_5_6_norm']:+.3f}  gap_10_11_norm={r0['baseline_gap_10_11_norm']:+.3f}")
        print(f"  replicates gap_5_6_norm={r0['replicate_mean_gap_5_6_norm']:+.3f}  gap_10_11_norm={r0['replicate_mean_gap_10_11_norm']:+.3f} (mean)")
        print(f"\nVol-matched top-decile excess: baseline={r0['baseline_volmatch_excess']:+.4f}  "
              f"replicate mean={r0['replicate_mean_volmatch_excess']:+.4f}  "
              f"replicate std={r0['replicate_std_volmatch_excess']:.4f}  "
              f"range={r0['volmatch_range_pp']:.4f}")

    print("\nRank churn within the top 10 (baseline picks, abs rank shift in each replicate's ranking):")
    print(
        f"  mean of per-replicate means = {replicate_df['rank_churn_mean_abs'].mean():.1f}  "
        f"mean n_missing = {replicate_df['rank_churn_n_missing'].mean():.2f} / {RANK_CHURN_TOP_N}"
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    top_ks = _parse_top_ks(args.top_ks)

    t0 = time.monotonic()
    replicate_df, summary_df = run_stability_gate(
        args.research, args.oof,
        n_replicates=args.n_replicates, mode=args.mode, drop_frac=args.drop_frac, seed=args.seed,
        n_folds=args.n_folds, horizon=args.horizon, embargo=args.embargo,
        min_fold_train_rows=args.min_fold_train_rows, tail_thresh=args.tail_thresh, top_ks=top_ks,
        score_col=args.score_col, vol_col=args.vol_col, top_frac=args.top_frac,
        n_vol_buckets=args.n_vol_buckets, n_boot=args.n_boot,
        retention_bar=args.retention_bar, volmatch_threshold_pp=args.volmatch_threshold_pp,
    )
    elapsed = time.monotonic() - t0
    log.info(
        "run_stability_gate: done in %.1fs (%d/%d replicate(s) succeeded)",
        elapsed, len(replicate_df), args.n_replicates,
    )

    rep_path, sum_path = save_stability_results(
        replicate_df, summary_df, out_dir=args.out_dir, mode=args.mode, tag=args.tag,
    )
    print(f"Per-replicate detail written to {os.path.abspath(rep_path)}")
    print(f"Summary + verdicts written to {os.path.abspath(sum_path)}")

    _print_report(replicate_df, summary_df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
