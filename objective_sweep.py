"""Objective sweep: does a less extreme training target than
`adj_63 > TAIL_THRESH` (TAIL_THRESH=0.20, an untuned "big win" constant
inherited verbatim from backtest/signal_fit.py's moonshot_thresh) buy a
smaller but more tradeable edge?

WHY this exists. research/model.py's production score, oof_tail_classifier,
is trained to predict P(adj_63 > 0.20) -- a genuine 20-point-or-better
winner. That objective explicitly targets moonshots, so the model's
lottery-shaped output (top decile mean +6.74%, MEDIAN -3.31%, see
tail-model-is-a-lottery-ticket.md) may be a property of what it was ASKED
to predict, not of the underlying signal. This script tests that directly:
refit the same purged/embargoed walk-forward CV (research.model.
fit_and_validate, completely unmodified fold logic -- see that module) at
four thresholds (0, 0.05, 0.10, 0.20) plus the winsorized regressor
(oof_regressor, already computed by every fit_and_validate call), and
compares their top-decile MEDIAN, win rate, vol-matched excess, and P&L
concentration -- never mean lift alone, which structurally favors
tail-chasing (a handful of huge winners inflate a mean while a negative
median is invisible to it).

Every candidate is trained on the SAME label horizon (adj_63) -- only the
threshold that turns that continuous return into a binary "big win" target
changes (or, for the regressor, no threshold at all). This is a different
axis from the horizon/label_col fix in research/model.py (see
label_col_for_horizon's docstring); this script does not touch horizon.

Threshold 0.0 and 0.20 are not new capability -- research.model.
fit_and_validate always produces oof_classifier (label > 0) and
oof_tail_classifier (label > TAIL_THRESH). This script instead always reads
oof_tail_classifier from a fit run WITH THAT THRESHOLD passed as
tail_thresh (a new fit_and_validate parameter added alongside this script,
threaded through the exact same code path oof_classifier / oof_tail_
classifier already used -- see research/model.py), so every threshold in
THRESH_CANDIDATES (including 0.0 and 0.20) is measured identically, via the
identical column name, rather than mixing oof_classifier and oof_tail_
classifier readings from a single call.

Method, per candidate:
  1. Fit research.model.fit_and_validate on the pinned, unperturbed
     research pool (research_noreuse_10908rows_20260808.parquet, the same
     input refit_stability.py's baseline is pinned to) with
     run_shap_interactions=False, n_shuffle_seeds=2 -- refit_stability.py's
     FAST-FIT NOTE verified this leaves oof_tail_classifier byte-identical
     to a full-settings fit while cutting the cost from ~25s to ~8s. This
     script never reimplements fit_and_validate's fold loop, winsorization,
     or classifier fitting -- it only varies the tail_thresh argument (or
     reads oof_regressor, unaffected by tail_thresh) and reads the result.
  2. Dedup + rank the resulting OOF frame (refit_stability.dedup_oof /
     that module's own top-decile selection convention: k =
     max(int(n * 0.10), 30) rows, DataFrame.nlargest, reused unchanged).
  3. Report top-decile mean, MEDIAN, win rate (P(adj_63 > 0)), and P&L
     concentration: the share of the decile's SUMMED adj_63 return
     contributed by its 5 highest-return (not highest-score) events --
     this is the direct lottery-ticket check; a decile whose return is a
     coin flip should have a small share here, one built on a few moonshots
     should not.
  4. Vol-matched excess is measured TWICE, both reusing refit_stability.
     vol_matched_excess unchanged (never reimplemented):
       - pooled: one call over the whole OOF pool (what refit_stability.py
         itself reports as its single baseline number).
       - per-fold: one call per CV fold (grouping oof_scores by its "fold"
         column), then a one-sample t-test of the 5 fold-level excesses
         against 0 -- this is the SAME methodology research/live_score.py's
         module docstring cites for the production model's own headline
         number ("+4.74pp over a risk-matched benchmark, p=0.004, positive
         in 4 of 5 folds"), reproduced here so every candidate in this sweep
         is judged on the identical statistic the incumbent was judged on.

CONSTRAINTS carried over from the task this script was built for: no
network access, no backtest is run here (score-level only -- a full
backtest is 15+ minutes and is being run separately), and this script does
not touch backtest/strategies.py, backtest/engine.py, or backtest/report.py.

Usage:
    python objective_sweep.py
    python objective_sweep.py --n-shuffle-seeds 20   # slower, unnecessary (see FAST-FIT NOTE)
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp

import refit_stability as rs
from ensemble_model import _pad_for_schema, _resolve_feature_cols
from research import model as rm

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
DEFAULT_RESEARCH_PATH = rs.DEFAULT_RESEARCH_PATH  # research_noreuse_10908rows_20260808.parquet
LABEL_COL = rm.LABEL_COL  # "adj_63" -- fixed across this whole sweep, see module docstring
VOL_COL_DEFAULT = rs.VOL_COL_DEFAULT  # "x_vol_63_ann"
TOP_FRAC = rs.TOP_FRAC  # 0.10
N_VOL_BUCKETS = rs.N_VOL_BUCKETS  # 10
N_BOOT_DEFAULT = rs.N_BOOT_DEFAULT  # 2000
FAST_N_SHUFFLE_SEEDS = rs.FAST_N_SHUFFLE_SEEDS  # 2 -- see refit_stability.py's FAST-FIT NOTE

THRESH_CANDIDATES: tuple[float, ...] = (0.0, 0.05, 0.10, 0.20)
REGRESSOR_CANDIDATE_NAME = "winsorized_regressor"
SCORE_COL_TAIL = "oof_tail_classifier"
SCORE_COL_REGRESSOR = "oof_regressor"

TOP5_CONCENTRATION_N = 5


# ---------------------------------------------------------------------------
# 1. One fit per threshold candidate
# ---------------------------------------------------------------------------
def fit_threshold_candidate(
    df: pd.DataFrame, feature_cols: list[str], tail_thresh: float, *,
    n_folds: int = rm.DEFAULT_N_FOLDS, horizon: int = rm.PRIMARY_HORIZON,
    embargo: int = rm.DEFAULT_EMBARGO_DAYS, min_fold_train_rows: int = rm.MIN_FOLD_TRAIN_ROWS,
    n_shuffle_seeds: int = FAST_N_SHUFFLE_SEEDS,
) -> rm.ValidationResult:
    """One research.model.fit_and_validate call at a given tail_thresh.
    Everything else (feature_cols, folds, lgbm params) matches the
    production config exactly, so the only thing that varies between
    candidates in THRESH_CANDIDATES is the objective itself."""
    return rm.fit_and_validate(
        df, n_folds=n_folds, horizon=horizon, embargo=embargo, feature_cols=feature_cols,
        min_fold_train_rows=min_fold_train_rows, run_shap_interactions=False,
        n_shuffle_seeds=n_shuffle_seeds, tail_thresh=tail_thresh,
    )


# ---------------------------------------------------------------------------
# 2. Top-decile selection (identical convention to refit_stability.
#    vol_matched_excess's own k = max(int(n * top_frac), 30) / nlargest)
# ---------------------------------------------------------------------------
def top_decile(oof_dedup: pd.DataFrame, score_col: str, top_frac: float = TOP_FRAC) -> pd.DataFrame:
    k = max(int(len(oof_dedup) * top_frac), 30)
    return oof_dedup.nlargest(k, score_col, keep="all").head(k)


def top5_pnl_concentration(decile_df: pd.DataFrame, label_col: str = LABEL_COL, n: int = TOP5_CONCENTRATION_N) -> dict:
    """Share of the decile's SUMMED label return coming from its `n`
    highest-RETURN (not highest-score) events -- the direct lottery-ticket
    check: are a handful of moonshots carrying the whole decile's P&L?

    Reported against the SIGNED total. If the decile's total return is
    negative or near zero, a "share" number is not a clean read (dividing
    by a small or negative denominator can produce a large or negative
    ratio that does not mean what a positive-denominator share means) --
    the raw total and top-n sum are always returned alongside the ratio so
    a reader is never dependent on the ratio alone."""
    vals = decile_df[label_col].to_numpy(dtype=float)
    vals = vals[~np.isnan(vals)]
    total = float(vals.sum())
    top_n_sum = float(np.sort(vals)[-n:].sum()) if len(vals) >= n else float(vals.sum())
    share = (top_n_sum / total) if total != 0 else float("nan")
    return {"decile_total_return": total, "top5_return_sum": top_n_sum, "top5_share_of_total": share}


# ---------------------------------------------------------------------------
# 3. Vol-matched excess, pooled AND per-fold (reusing refit_stability.
#    vol_matched_excess unchanged in both cases)
# ---------------------------------------------------------------------------
def per_fold_volmatch(
    oof_scores: pd.DataFrame, vol_map: pd.Series, *, score_col: str, label_col: str,
    top_frac: float, n_vol_buckets: int, n_boot: int, seed: int,
) -> pd.DataFrame:
    """One refit_stability.vol_matched_excess call per CV fold (grouping
    oof_scores by its "fold" column, written by fit_and_validate's own
    per-fold loop) -- the same per-fold methodology research/live_score.py
    cites for the production model's own "+4.74pp, p=0.004, positive in 4
    of 5 folds" headline number, reproduced here so every candidate in
    this sweep is judged on the identical statistic."""
    rows = []
    for fold_id, sub in oof_scores.groupby("fold"):
        dedup = rs.dedup_oof(sub, score_col, context=f"fold {fold_id}")
        res = rs.vol_matched_excess(
            dedup, vol_map, score_col=score_col, label_col=label_col,
            top_frac=top_frac, n_vol_buckets=n_vol_buckets, n_boot=n_boot,
            rng=np.random.default_rng(seed + fold_id),
        )
        res["fold"] = fold_id
        rows.append(res)
    return pd.DataFrame(rows)


def volmatch_fold_significance(per_fold_df: pd.DataFrame) -> dict:
    """One-sample t-test of the per-fold volmatch_excess values against 0,
    same construction as ensemble_model.volmatch_significance (not
    imported directly -- that function takes a bare np.ndarray of replicate
    values, this one a per-fold DataFrame; the arithmetic is identical, kept
    local to avoid a needless cross-import for one ttest_1samp call)."""
    vals = per_fold_df["volmatch_excess"].dropna().to_numpy()
    n_valid = int(len(vals))
    n_positive = int((vals > 0).sum())
    if n_valid >= 2 and np.std(vals, ddof=1) > 0:
        t, p = ttest_1samp(vals, 0.0)
        t, p = float(t), float(p)
    else:
        t, p = float("nan"), float("nan")
    return {
        "volmatch_fold_mean": float(np.mean(vals)) if n_valid else float("nan"),
        "volmatch_fold_std": float(np.std(vals, ddof=1)) if n_valid > 1 else float("nan"),
        "volmatch_fold_tstat": t,
        "volmatch_fold_pvalue": p,
        "volmatch_n_positive_folds": n_positive,
        "volmatch_n_folds": n_valid,
        "volmatch_per_fold_values": [round(v, 4) for v in vals.tolist()],
    }


# ---------------------------------------------------------------------------
# 4. Per-candidate summary
# ---------------------------------------------------------------------------
def summarize_candidate(
    name: str, oof_scores: pd.DataFrame, vol_map: pd.Series, *, score_col: str, label_col: str,
    top_frac: float, n_vol_buckets: int, n_boot: int, seed: int,
) -> dict:
    dedup = rs.dedup_oof(oof_scores, score_col, context=name)
    decile = top_decile(dedup, score_col, top_frac=top_frac)

    label_vals = decile[label_col]
    row: dict = {
        "candidate": name,
        "n_pool": len(dedup),
        "n_decile": len(decile),
        "decile_mean": float(label_vals.mean()),
        "decile_median": float(label_vals.median()),
        "win_rate": float((label_vals > 0).mean()),
    }
    row.update(top5_pnl_concentration(decile, label_col=label_col))

    pooled = rs.vol_matched_excess(
        dedup, vol_map, score_col=score_col, label_col=label_col,
        top_frac=top_frac, n_vol_buckets=n_vol_buckets, n_boot=n_boot,
        rng=np.random.default_rng(seed + 999_000),
    )
    row["volmatch_excess_pooled"] = pooled["volmatch_excess"]
    row["volmatch_pooled_n"] = pooled["volmatch_n"]

    per_fold = per_fold_volmatch(
        oof_scores, vol_map, score_col=score_col, label_col=label_col,
        top_frac=top_frac, n_vol_buckets=n_vol_buckets, n_boot=n_boot, seed=seed,
    )
    row.update(volmatch_fold_significance(per_fold))
    return row


# ---------------------------------------------------------------------------
# 5. Orchestration
# ---------------------------------------------------------------------------
def run_objective_sweep(
    research_path: str = DEFAULT_RESEARCH_PATH, *,
    thresh_candidates: tuple[float, ...] = THRESH_CANDIDATES,
    vol_col: str = VOL_COL_DEFAULT, top_frac: float = TOP_FRAC,
    n_vol_buckets: int = N_VOL_BUCKETS, n_boot: int = N_BOOT_DEFAULT,
    n_shuffle_seeds: int = FAST_N_SHUFFLE_SEEDS, seed: int = 0,
) -> tuple[pd.DataFrame, dict]:
    """Fit every threshold candidate plus the (threshold-independent)
    winsorized regressor, and return (summary_df, fits) where fits maps
    candidate name -> the rm.ValidationResult that produced it (kept around
    so a caller, e.g. the stability gate step, can re-fit the SAME
    feature_cols/config without re-deriving them)."""
    log.info("run_objective_sweep: loading research dataset %s", research_path)
    df_raw = pd.read_parquet(research_path)
    feature_cols, padded_cols = _resolve_feature_cols(df_raw)
    if padded_cols:
        log.info(
            "run_objective_sweep: padding %d feature column(s) missing from %s as all-NaN "
            "(excluded from feature_cols passed to every fit) -- see ensemble_model.py's "
            "FEATURE-COLUMN WORKAROUND section: %s",
            len(padded_cols), research_path, padded_cols,
        )
    df = _pad_for_schema(df_raw, padded_cols)
    log.info(
        "run_objective_sweep: %d research rows, %d usable feature(s), thresholds=%s",
        len(df), len(feature_cols), thresh_candidates,
    )

    vol_map = (
        df[["ticker", "event_day", vol_col]].dropna(subset=[vol_col])
        .drop_duplicates(subset=["ticker", "event_day"], keep="first")
        .set_index(["ticker", "event_day"])[vol_col]
    )

    fits: dict[str, rm.ValidationResult] = {}
    summary_rows: list[dict] = []

    for t in thresh_candidates:
        name = f"thresh_{t:.2f}"
        t0 = time.monotonic()
        result = fit_threshold_candidate(df, feature_cols, tail_thresh=t, n_shuffle_seeds=n_shuffle_seeds)
        fits[name] = result
        log.info("%s: fit in %.1fs", name, time.monotonic() - t0)
        row = summarize_candidate(
            name, result.oof_scores, vol_map, score_col=SCORE_COL_TAIL, label_col=LABEL_COL,
            top_frac=top_frac, n_vol_buckets=n_vol_buckets, n_boot=n_boot, seed=seed,
        )
        row["tail_thresh"] = t
        summary_rows.append(row)

    # oof_regressor does not depend on tail_thresh -- reuse whichever
    # candidate fit was just run rather than paying for a 5th fit. Uses the
    # LAST candidate in thresh_candidates (0.20 under the default tuple,
    # i.e. the production config) so the regressor candidate is compared
    # against the same fold/feature/lgbm setup as the incumbent.
    last_result = fits[f"thresh_{thresh_candidates[-1]:.2f}"]
    reg_row = summarize_candidate(
        REGRESSOR_CANDIDATE_NAME, last_result.oof_scores, vol_map, score_col=SCORE_COL_REGRESSOR,
        label_col=LABEL_COL, top_frac=top_frac, n_vol_buckets=n_vol_buckets, n_boot=n_boot, seed=seed,
    )
    reg_row["tail_thresh"] = float("nan")
    summary_rows.append(reg_row)

    summary_df = pd.DataFrame(summary_rows)
    return summary_df, fits


# ---------------------------------------------------------------------------
# 6. Pool-level reference row (the honest floor every candidate is judged
#    against -- the event pool has a NEGATIVE median at every horizon).
# ---------------------------------------------------------------------------
def pool_reference_row(df: pd.DataFrame, label_col: str = LABEL_COL) -> dict:
    vals = df[label_col].dropna()
    return {
        "candidate": "FULL_POOL (reference, not a strategy)",
        "n_pool": int(len(vals)), "n_decile": int(len(vals)),
        "decile_mean": float(vals.mean()), "decile_median": float(vals.median()),
        "win_rate": float((vals > 0).mean()),
    }


# ---------------------------------------------------------------------------
# 7. Persistence + CLI
# ---------------------------------------------------------------------------
def save_candidate_oof_scores(fits: dict[str, rm.ValidationResult], out_dir: str = "research_data") -> dict[str, str]:
    """Persist each threshold candidate's oof_scores to disk via
    research.model.save_oof_scores (unchanged, atomic tmp-then-replace
    write), tagged by candidate name -- these become the matched
    --oof baseline files refit_stability.py's --tail-thresh gate needs
    (see that module's run_stability_gate docstring: the baseline and every
    replicate must be fit at the SAME tail_thresh, or their top-k sets are
    not comparable). Returns {candidate_name: path_written}."""
    paths: dict[str, str] = {}
    for name, result in fits.items():
        path = rm.save_oof_scores(result.oof_scores, out_dir=out_dir, tag=f"objsweep_{name}")
        paths[name] = path
        log.info("save_candidate_oof_scores: %s -> %s", name, path)
    return paths


def save_summary(summary_df: pd.DataFrame, out_dir: str = "research_data", tag: str = "") -> str:
    os.makedirs(out_dir, exist_ok=True)
    tag_part = f"{tag}_" if tag else ""
    path = os.path.join(out_dir, f"objective_sweep_summary_{tag_part}{date.today():%Y%m%d}.csv")
    tmp_path = path + ".tmp"
    summary_df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, path)
    return path


def _print_report(summary_df: pd.DataFrame, pool_row: dict) -> None:
    cols = [
        "candidate", "tail_thresh", "n_decile", "decile_mean", "decile_median", "win_rate",
        "top5_share_of_total", "volmatch_excess_pooled",
        "volmatch_fold_mean", "volmatch_fold_tstat", "volmatch_fold_pvalue",
        "volmatch_n_positive_folds", "volmatch_n_folds",
    ]
    print("\n" + "=" * 120)
    print("OBJECTIVE SWEEP -- top-decile shape and vol-matched excess by candidate")
    print("=" * 120)
    with pd.option_context("display.max_rows", None, "display.width", 240, "display.float_format", "{:+.4f}".format):
        print(summary_df[cols].to_string(index=False))
    print(
        f"\nFULL POOL reference (not a strategy, the floor every candidate is judged against): "
        f"n={pool_row['n_pool']} mean={pool_row['decile_mean']:+.4f} "
        f"median={pool_row['decile_median']:+.4f} win_rate={pool_row['win_rate']:.4f}"
    )
    for _, r in summary_df.iterrows():
        print(f"\n[{r['candidate']}] per-fold volmatch_excess: {r['volmatch_per_fold_values']}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--research", default=DEFAULT_RESEARCH_PATH)
    p.add_argument("--thresholds", default=",".join(str(t) for t in THRESH_CANDIDATES))
    p.add_argument("--vol-col", default=VOL_COL_DEFAULT)
    p.add_argument("--top-frac", type=float, default=TOP_FRAC)
    p.add_argument("--n-vol-buckets", type=int, default=N_VOL_BUCKETS)
    p.add_argument("--n-boot", type=int, default=N_BOOT_DEFAULT)
    p.add_argument("--n-shuffle-seeds", type=int, default=FAST_N_SHUFFLE_SEEDS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="research_data")
    p.add_argument("--tag", default="")
    p.add_argument(
        "--save-oof", action="store_true",
        help="Also persist each threshold candidate's oof_scores to --out-dir, tagged "
        "objsweep_thresh_<t>, for use as refit_stability.py --oof baseline files.",
    )
    p.add_argument("--verbose", action="store_true")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    thresholds = tuple(float(x) for x in args.thresholds.split(",") if x.strip())

    t0 = time.monotonic()
    summary_df, fits = run_objective_sweep(
        args.research, thresh_candidates=thresholds, vol_col=args.vol_col, top_frac=args.top_frac,
        n_vol_buckets=args.n_vol_buckets, n_boot=args.n_boot, n_shuffle_seeds=args.n_shuffle_seeds, seed=args.seed,
    )
    log.info("run_objective_sweep: done in %.1fs", time.monotonic() - t0)

    df_raw = pd.read_parquet(args.research)
    pool_row = pool_reference_row(df_raw)

    path = save_summary(summary_df, out_dir=args.out_dir, tag=args.tag)
    print(f"Summary written to {os.path.abspath(path)}")

    if args.save_oof:
        oof_paths = save_candidate_oof_scores(fits, out_dir=args.out_dir)
        for name, p in oof_paths.items():
            print(f"[{name}] OOF scores written to {os.path.abspath(p)}")

    _print_report(summary_df, pool_row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
