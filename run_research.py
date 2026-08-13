"""Reproducible CLI entrypoint for the two research artifacts backtest's
`model_ranked_*` strategies depend on:

  research_data/research_<tag_><n_rows>rows_<YYYYMMDD>.parquet
      One row per insider-cluster episode, built by
      backtest.research.build_research_dataset. See backtest/research.py's
      module docstring for the feature/label schema.

  research_data/oof_scores_<tag_><YYYYMMDD>.parquet
      Out-of-fold ranking-model scores, built by research.model.fit_and_validate's
      purged walk-forward CV. backtest/model_scores.py reads this file's
      oof_tail_classifier column to rank model_ranked_* strategies. The file
      also carries oof_regressor, oof_classifier and oof_sharpe (the
      risk-adjusted objective -- see research/model.py's SHARPE_VOL_COL), any
      of which a caller can select instead.

Neither artifact previously had a command-line entrypoint -- both were built
ad hoc in an interactive session, which made the pipeline unreproducible.
This script wires the existing pieces of backtest/research.py and
research/model.py together into two stages that can run independently or
back to back:

    python run_research.py --build-dataset   # writes research_*.parquet
    python run_research.py --fit-model        # reads the latest research_*.parquet, writes oof_scores_*.parquet
    python run_research.py                    # both, in order (same as --all)

backtest.bat runs this script before backtest.py so a grid always ranks on a
model fit in the same invocation, and passes --oof-path-out so the backtest
receives the exact parquet this run wrote rather than re-resolving 'latest'
by mtime.

A third stage produces the artifact the LIVE screener needs (see
research/live_score.py), separate from the two above because it is a
deployment artifact, not a research one, and is opt-in only -- it is NOT
part of --all or the no-flags default, so existing callers of this script
see no behavior change:

    python run_research.py --fit-production   # reads the latest research_*.parquet, writes production_model_*.joblib

--dry-run reports what a real run would do (row counts, fold sizes, planned
output paths) without fetching prices, fitting anything, or writing a file.

The --fit-model stage's row-count relationship to --build-dataset's output is
NOT 1:1, and is not a bug: research.model.fit_and_validate (1) drops rows
with no usable label or entry_idx, then (2) cuts the remainder into
n_folds + 1 chronological blocks and trains/tests only n_folds of them --
the earliest block is used purely as training history for fold 0 and is
never itself a test block, so it contributes zero rows to oof_scores. On the
existing artifacts (research_full_11026rows_20260731.parquet ->
oof_scores_20260731.parquet) this is exactly 11,026 -> 10,659 valid rows ->
8,882 OOF rows (the first of 6 blocks, 1,777 rows, is train-only). See
dry_run_fit_model below, which computes this exactly (no model fit needed)
and was checked against those two real files while building this script.
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import time
from datetime import date, datetime, timedelta

import pandas as pd

from backtest import history
from backtest import prices as prices_mod
from backtest import research as research_mod
from backtest import sales_history
from backtest.state import DailyStateBuilder
from research import model as rm

log = logging.getLogger("run_research")

# The existing artifacts (research_data/research_full_11026rows_20260731.parquet,
# whose event_day column spans exactly 2018-07-17 .. 2026-07-17) were built
# from clusters_history/events_20180717_20260717.parquet, an 8-year scrape.
# backtest/history.py's build_history() turns months_back into a day count
# via `months_back * 30.44`; (2026-07-17 - 2018-07-17).days / 30.44 ==
# 95.99, i.e. months_back=96. Inferred from the artifact, not guessed.
DEFAULT_MONTHS = 96

DEFAULT_OUT_DIR = "research_data"
RESEARCH_GLOB = "research_*rows_*.parquet"
OOF_GLOB = "oof_scores*.parquet"
PRODUCTION_GLOB = "production_model_*.joblib"

# Heuristic-estimate reference point for --build-dataset --dry-run (see
# dry_run_build_dataset): the one real build on disk, and the events file
# RESEARCH_NOTES.md's "Data facts" section ties it to (same 2018-07-17 ..
# 2026-07-17 window, 597,278 event rows, scraped 2026-07-18).
_REFERENCE_RESEARCH_ROWS = 11026
_REFERENCE_EVENT_ROWS = 597278


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Rebuild the research dataset and/or the OOF ranking-model "
            "scores that backtest's model_ranked_* strategies read."
        ),
    )
    p.add_argument(
        "--build-dataset", action="store_true",
        help="Build the event-level research dataset and write it via save_research_dataset.",
    )
    p.add_argument(
        "--fit-model", action="store_true",
        help="Run the purged walk-forward CV and write the OOF scores parquet.",
    )
    p.add_argument(
        "--all", action="store_true",
        help="Run both stages in order. Equivalent to passing neither --build-dataset nor --fit-model.",
    )
    p.add_argument(
        "--fit-production", action="store_true",
        help=(
            "Train (or reuse, if --fit-model ran in the same invocation) the purged "
            "walk-forward CV, then bundle its final full-dataset model with a fixed "
            "training-score reference distribution and provenance into "
            "production_model_*.joblib for research.live_score to consume. Opt-in "
            "only -- not part of --all or the no-flags default."
        ),
    )
    p.add_argument(
        "--allow-never-live-features", action="store_true",
        help=(
            "Escape hatch for research.model.build_production_bundle's default guard: "
            "without this flag, --fit-production REFUSES to bundle a model whose "
            "feature_cols include any of research.live_score.SALE_FEATURE_COLS (the 9 "
            "Section F concurrent-selling columns, never computable in the live "
            "screener -- see build_production_bundle's docstring). Pass this only for "
            "a deliberate experiment; the shipped production bundle does not need it."
        ),
    )
    p.add_argument(
        "--months", type=int, default=DEFAULT_MONTHS,
        help=(
            f"History window in months for --build-dataset (default {DEFAULT_MONTHS}, "
            "inferred from the existing research_full_11026rows_20260731.parquet artifact's "
            "2018-07-17..2026-07-17 event_day span)."
        ),
    )
    p.add_argument(
        "--out-dir", default=DEFAULT_OUT_DIR,
        help=f"Directory both artifacts are written to/read from (default {DEFAULT_OUT_DIR!r}).",
    )
    p.add_argument(
        "--tag", default="",
        help="Optional filename tag, e.g. research_<tag>_<n>rows_<date>.parquet / oof_scores_<tag>_<date>.parquet.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Report what each selected stage would do (row counts, planned output paths) and write nothing.",
    )
    p.add_argument(
        "--as-of", default=None,
        help="YYYY-MM-DD; pins the window end for --build-dataset so a run is reproducible (default: today).",
    )
    p.add_argument(
        "--events-from", default="",
        help=(
            "Reuse a cached events parquet (explicit path, or 'latest' to pick the widest file "
            "in clusters_history/) instead of scraping SEC EDGAR. Passed straight through to "
            "backtest.history.build_history."
        ),
    )
    p.add_argument(
        "--dataset-path", default=None,
        help=(
            "Explicit research-dataset parquet for --fit-model to read when it runs standalone "
            "(without --build-dataset in the same invocation). Default: the most recently "
            f"modified {RESEARCH_GLOB} in --out-dir."
        ),
    )
    p.add_argument("--n-folds", type=int, default=rm.DEFAULT_N_FOLDS, help="Passed to make_purged_expanding_folds.")
    p.add_argument("--horizon", type=int, default=rm.PRIMARY_HORIZON, help="Label horizon (trading days) for the CV/purge rule.")
    p.add_argument("--embargo", type=int, default=rm.DEFAULT_EMBARGO_DAYS, help="Embargo gap (trading days) around each fold boundary.")
    p.add_argument("--min-fold-train-rows", type=int, default=rm.MIN_FOLD_TRAIN_ROWS, help="A fold below this many training rows is skipped.")
    p.add_argument("--n-shuffle-seeds", type=int, default=rm.DEFAULT_N_SHUFFLE_SEEDS, help="Label-shuffle leakage-test draw count.")
    p.add_argument(
        "--no-shap-interactions", action="store_true",
        help="Skip the ten-percent-owner SHAP interaction analysis (oof_scores is unaffected either way; saves fit time).",
    )
    p.add_argument(
        "--oof-path-out", default=None,
        help=(
            "Write the absolute path of the oof_scores parquet this run produced to "
            "FILE (one line, no trailing newline decoration). backtest.bat uses this "
            "to hand the exact file it just fit straight to backtest.py via "
            "BT_MODEL_SCORES, instead of letting BT_MODEL_SCORES='latest' re-resolve "
            "it by mtime -- a glob whose tiebreak is arbitrary when several score "
            "files share a timestamp (see backtest/model_scores.py). Nothing is "
            "written unless --fit-model actually ran and produced scores."
        ),
    )
    p.add_argument("--verbose", action="store_true", help="DEBUG-level logging.")
    return p


def selected_stages(args: argparse.Namespace) -> tuple[bool, bool]:
    """(do_build, do_fit). Neither --build-dataset nor --fit-model (or
    --all explicitly) means run both, in order -- UNLESS --fit-production
    was explicitly given instead. --fit-production is opt-in only (see
    build_arg_parser's help text) and reads an already-built research
    dataset; without this exception, `python run_research.py
    --fit-production` would silently also run the full --build-dataset
    scrape/parse and --fit-model CV (both expensive, one of them
    network-bound) because build_dataset/fit_model were both left False --
    exactly the "run everything" default this function documents, but not
    what a caller asking only for --fit-production wants or expects.
    getattr guards a caller that constructs args by hand without the
    --fit-production attribute (e.g. an older test/script)."""
    fit_production = bool(getattr(args, "fit_production", False))
    if args.all or (not args.build_dataset and not args.fit_model and not fit_production):
        return True, True
    return bool(args.build_dataset), bool(args.fit_model)


def _parse_as_of(raw: str | None) -> date:
    if not raw:
        return date.today()
    try:
        return datetime.strptime(raw, "%Y-%m-%d").date()
    except ValueError:
        raise SystemExit(f"Invalid --as-of {raw!r}; expected YYYY-MM-DD") from None


def _write_oof_path_file(dest: str, oof_path: str) -> None:
    """Write `oof_path` to `dest` as a single absolute path.

    Absolute, because backtest.bat reads this back and hands it to
    backtest.py, and a relative path is only correct if both processes share
    a working directory. They do today; a caller who moves either one should
    not silently get a path that resolves to nothing.

    Written last in the fit stage, after save_oof_scores has already
    os.replace()d the parquet into place, so this file never points at a
    parquet that does not exist yet.
    """
    resolved = os.path.abspath(oof_path)
    parent = os.path.dirname(os.path.abspath(dest))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as fh:
        fh.write(resolved)
    log.info("fit-model: wrote OOF score path pointer %s -> %s", dest, resolved)


def _resolve_latest(out_dir: str, pattern: str) -> str | None:
    matches = glob.glob(os.path.join(out_dir, pattern))
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


def _resolve_latest_for_production(out_dir: str, pattern: str) -> str | None:
    """Same newest-by-mtime resolution as _resolve_latest, for --fit-production
    specifically: logs the resolved path AND every candidate it was chosen
    over, at WARNING level, because this pick silently decides a production
    bundle's feature set. A newly-landed research_*.parquet with extra
    feature columns (e.g. research_groupE_10905rows_20260809.parquet,
    9 columns wider than research_noreuse_10908rows_20260808.parquet) becomes
    "the" dataset the next --fit-production run reads with no other visible
    signal that anything changed -- research.model.build_production_bundle's
    allow_never_live_features guard catches the specific failure mode of
    those extra columns being permanently-missing-live ones, but a caller
    who only wanted --dataset-path's default to keep working as before
    still deserves to SEE the auto-pick happen. --dataset-path is the
    escape hatch back to silence: pass it explicitly and this function is
    never called.
    """
    matches = sorted(glob.glob(os.path.join(out_dir, pattern)), key=os.path.getmtime)
    if not matches:
        return None
    resolved = matches[-1]
    log.warning(
        "fit-production: no --dataset-path given -- auto-selected the newest-by-mtime "
        "match for %s in %s/: %s (of %d candidate(s): %s). This choice silently decides "
        "the production bundle's feature set -- pass --dataset-path explicitly if this "
        "is not the dataset you intended.",
        pattern, out_dir, resolved, len(matches), ", ".join(os.path.basename(m) for m in matches),
    )
    return resolved


# ---------------------------------------------------------------------------
# --build-dataset: real run
# ---------------------------------------------------------------------------
def run_build_dataset_stage(
    *, months: int, as_of: date, events_from: str, horizons: tuple[int, ...], out_dir: str = DEFAULT_OUT_DIR,
) -> pd.DataFrame:
    """End-to-end: scrape/load events -> daily state index -> price universe
    (goes through PriceUniverse.ensure(), so price_overrides.json's manual
    split corrections are picked up at both the load and fetch seams -- see
    backtest/prices.py) -> trading calendar -> build_research_dataset.

    Mirrors backtest.py's Phase 1-4 wiring (history.build_history ->
    DailyStateBuilder -> PriceUniverse.ensure/finalize -> trading_calendar)
    plus backtest/research.py's own build_research_dataset call.

    Also loads (or builds, on a cold `out_dir`, from parse_cache/ -- purely
    local, no network) the point-in-time sale-transaction index backtest.
    research's Section F (concurrent-selling) features read -- see
    backtest/sales_history.py. This is the one caller responsible for
    supplying sales_df; build_research_dataset itself never auto-loads it
    (see that function's docstring for why).
    """
    t0 = time.monotonic()
    log.info("build-dataset: loading events (months=%d, as_of=%s, events_from=%r)", months, as_of, events_from or "(scrape)")
    events_df, win_start, win_end, parse_errors, _n_dropped_ticker_reuse = history.build_history(
        months, as_of=as_of, events_from=events_from or None,
    )
    if events_df.empty:
        raise SystemExit("build-dataset: no qualifying events in the window -- aborting.")
    if parse_errors:
        log.warning("build-dataset: %d filings failed to parse and were discarded", len(parse_errors))
    log.info(
        "build-dataset: %d event rows, %d unique tickers, window %s..%s (%.1fs so far)",
        len(events_df), events_df["ticker"].nunique(), win_start, win_end, time.monotonic() - t0,
    )

    log.info("build-dataset: building daily state index")
    state_builder = DailyStateBuilder(events_df)

    log.info("build-dataset: fetching/loading price data (SPY + %d event tickers)", events_df["ticker"].nunique())
    tickers = sorted(set(events_df["ticker"].astype(str).str.upper())) + ["SPY"]
    price_start = win_start - timedelta(days=30)
    price_end = as_of + timedelta(days=400)
    pu = prices_mod.PriceUniverse()
    pu.ensure(tickers, price_start, price_end)
    pu.finalize()
    if "SPY" not in pu.frames:
        raise SystemExit("build-dataset: SPY price data could not be loaded -- aborting.")

    calendar = pu.trading_calendar(win_start, min(as_of, price_end))
    if not calendar:
        raise SystemExit("build-dataset: empty trading calendar -- aborting.")
    log.info(
        "build-dataset: calendar %d trading days (%s..%s), %d/%d tickers priced (%.1fs so far)",
        len(calendar), calendar[0], calendar[-1], len(pu.frames), len(tickers), time.monotonic() - t0,
    )

    log.info("build-dataset: loading/building the concurrent-selling sale index (local, no network)")
    sales_df = sales_history.load_or_build_sales_cache(out_dir=out_dir)
    log.info(
        "build-dataset: sale index ready -- %d sale row(s) across %d issuer(s) (%.1fs so far)",
        len(sales_df), sales_df["issuer_cik"].nunique() if not sales_df.empty else 0,
        time.monotonic() - t0,
    )

    log.info("build-dataset: building event-level research dataset (this is the expensive step)")
    df = research_mod.build_research_dataset(
        state_builder, pu, calendar, events_df, horizons=horizons, sales_df=sales_df,
    )

    elapsed = time.monotonic() - t0
    log.info("build-dataset: done in %.1fs -- %d rows, %d cols", elapsed, len(df), len(df.columns))
    return df


def dry_run_build_dataset(args: argparse.Namespace, as_of: date) -> dict:
    """Reports the planned window and, if --events-from is given, the real
    event-row/ticker counts (cheap: a parquet read, no scrape, no price
    fetch) plus a heuristic row-count estimate. Never fetches prices or
    calls build_research_dataset -- that is the expensive step this script
    exists to make reproducible, not something --dry-run should also pay for.
    """
    window_end = as_of
    window_start = window_end - timedelta(days=int(args.months * 30.44))
    report: dict = {
        "months": args.months,
        "as_of": as_of.isoformat(),
        "planned_window": f"{window_start} .. {window_end}",
        "events_source": args.events_from or "scrape (SEC EDGAR) -- not exercised in --dry-run",
    }

    if args.events_from:
        try:
            path = history.resolve_events_path(args.events_from)
            events_df, ev_start, ev_end, _n_dropped_ticker_reuse = history.load_events_df(path)
        except SystemExit as exc:
            report["events_error"] = str(exc)
        else:
            n_events = len(events_df)
            n_tickers = int(events_df["ticker"].nunique())
            report.update({
                "events_path": path,
                "n_event_rows": n_events,
                "n_tickers": n_tickers,
                "events_window": f"{ev_start} .. {ev_end}",
            })
            ratio = _REFERENCE_RESEARCH_ROWS / _REFERENCE_EVENT_ROWS
            report["expected_research_rows_heuristic"] = int(round(n_events * ratio))
            report["expected_rows_note"] = (
                "heuristic only, scaled from the one real build on disk "
                f"({_REFERENCE_RESEARCH_ROWS} research rows / {_REFERENCE_EVENT_ROWS} event rows). "
                "The real count depends on price data this dry run deliberately does not fetch "
                "(missing entry prices, split-unsafe label windows) -- treat this as a ballpark."
            )
    else:
        report["expected_rows_note"] = (
            "no --events-from given; pass one (a path or 'latest') to get an event-row-based estimate."
        )

    tag_part = f"{args.tag}_" if args.tag else ""
    report["planned_output"] = os.path.join(
        args.out_dir, f"research_{tag_part}<N>rows_{date.today():%Y%m%d}.parquet",
    )
    log.info("[dry-run] build-dataset: %s", report)
    return report


# ---------------------------------------------------------------------------
# --fit-model: real run
# ---------------------------------------------------------------------------
def run_fit_model_stage(
    df: pd.DataFrame, *, n_folds: int, horizon: int, embargo: int,
    min_fold_train_rows: int, n_shuffle_seeds: int, run_shap_interactions: bool,
) -> rm.ValidationResult:
    t0 = time.monotonic()
    log.info(
        "fit-model: running purged walk-forward CV on %d rows (n_folds=%d, horizon=%d, embargo=%d)",
        len(df), n_folds, horizon, embargo,
    )
    result = rm.fit_and_validate(
        df, n_folds=n_folds, horizon=horizon, embargo=embargo,
        min_fold_train_rows=min_fold_train_rows, n_shuffle_seeds=n_shuffle_seeds,
        run_shap_interactions=run_shap_interactions,
    )
    elapsed = time.monotonic() - t0
    log.info(
        "fit-model: done in %.1fs -- %d/%d folds ran, %d OOF rows, real IC=%.4f, label-shuffle %s",
        elapsed, result.n_folds_run, len(result.folds), len(result.oof_scores),
        result.label_shuffle.get("real_ic", float("nan")),
        "PASSED" if result.label_shuffle.get("passed") else "FAILED",
    )
    return result


def dry_run_fit_model(
    args: argparse.Namespace, df: pd.DataFrame | None = None, dataset_path: str | None = None,
) -> dict:
    """Reports the EXACT expected oof_scores row count with no model fit at
    all: make_purged_expanding_folds is cheap (no LightGBM involved), and
    result.oof_scores is nothing more than the concatenation of each
    non-skipped fold's test block. This was checked directly against the
    real research_full_11026rows_20260731.parquet -> oof_scores_20260731.parquet
    pair while building this script: 11,026 rows -> 10,659 with a usable
    label+entry_idx -> exactly 8,882 predicted OOF rows, matching the file
    on disk row for row.
    """
    if df is None:
        resolved = dataset_path if dataset_path is not None else _resolve_latest(args.out_dir, RESEARCH_GLOB)
        if resolved is None:
            report = {"error": f"no dataset given and no {RESEARCH_GLOB} found in {args.out_dir}/"}
            log.info("[dry-run] fit-model: %s", report)
            return report
        dataset_path = resolved
        df = rm.load_research_dataset(dataset_path)

    n_total = len(df)
    df_valid = df.dropna(subset=[rm.LABEL_COL, "entry_idx"]).copy()
    n_valid = len(df_valid)
    report: dict = {
        "dataset_path": dataset_path,
        "n_total_rows": n_total,
        "n_valid_rows": n_valid,
        "n_dropped_missing_label_or_entry_idx": n_total - n_valid,
    }

    if n_valid < args.n_folds + 1:
        report["error"] = f"only {n_valid} valid rows -- need at least {args.n_folds + 1} for n_folds={args.n_folds}"
        log.info("[dry-run] fit-model: %s", report)
        return report

    folds = rm.make_purged_expanding_folds(
        df_valid, n_folds=args.n_folds, horizon=args.horizon, embargo=args.embargo,
    )
    fold_report = []
    expected_oof_rows = 0
    for f in folds:
        will_run = len(f.train_idx) >= args.min_fold_train_rows
        fold_report.append({
            "fold_id": f.fold_id, "n_train": len(f.train_idx), "n_test": len(f.test_idx), "will_run": will_run,
        })
        if will_run:
            expected_oof_rows += len(f.test_idx)

    report["folds"] = fold_report
    report["expected_oof_rows"] = expected_oof_rows

    tag_part = f"{args.tag}_" if args.tag else ""
    report["planned_output"] = os.path.join(args.out_dir, f"oof_scores_{tag_part}{date.today():%Y%m%d}.parquet")
    log.info("[dry-run] fit-model: %s", report)
    return report


# ---------------------------------------------------------------------------
# --fit-production: real run
# ---------------------------------------------------------------------------
def run_fit_production_stage(
    df: pd.DataFrame, *, n_folds: int, horizon: int, embargo: int,
    min_fold_train_rows: int, n_shuffle_seeds: int, source_path: str,
    result: rm.ValidationResult | None = None,
    allow_never_live_features: bool = False,
) -> rm.ProductionBundle:
    """Build a ProductionBundle from `df`. If `result` (a ValidationResult
    from an earlier fit_and_validate call in this same process, e.g.
    --fit-model running in the same invocation) is not given, this runs
    fit_and_validate itself -- the SAME code path --fit-model uses, so the
    model this bundle ships is never a separate, unvalidated reimplementation
    of the fitting logic. run_shap_interactions is forced off here when a
    fresh fit is needed: the interaction analysis is diagnostic output for a
    human reading a validation report, not something build_production_bundle
    reads, so paying its fit-per-fold cost here would be pure waste.

    allow_never_live_features: passed straight through to
    rm.build_production_bundle -- see that function's docstring and
    --allow-never-live-features's help text. Default False, so a dataset
    whose feature_cols carry any of research.live_score.SALE_FEATURE_COLS
    (e.g. one built with a wired-in sales_df, like
    research_groupE_10905rows_20260809.parquet) is refused rather than
    silently bundled.
    """
    t0 = time.monotonic()
    if result is None:
        log.info(
            "fit-production: no cached fit result to reuse -- running purged walk-forward "
            "CV on %d rows (n_folds=%d, horizon=%d, embargo=%d)",
            len(df), n_folds, horizon, embargo,
        )
        result = rm.fit_and_validate(
            df, n_folds=n_folds, horizon=horizon, embargo=embargo,
            min_fold_train_rows=min_fold_train_rows, n_shuffle_seeds=n_shuffle_seeds,
            run_shap_interactions=False,
        )
    else:
        log.info("fit-production: reusing this invocation's --fit-model result -- no refit needed")

    bundle = rm.build_production_bundle(
        result, df, source_path=source_path, allow_never_live_features=allow_never_live_features,
    )
    elapsed = time.monotonic() - t0
    log.info(
        "fit-production: done in %.1fs -- model=%s, %d training rows in the percentile "
        "reference, %d feature columns",
        elapsed, rm.PRODUCTION_SCORE_MODEL, len(bundle.training_scores), len(bundle.feature_cols),
    )
    return bundle


def dry_run_fit_production(
    args: argparse.Namespace, df: pd.DataFrame | None = None, dataset_path: str | None = None,
) -> dict:
    """Reports the dataset that would be used and the exact training-row
    count the percentile reference distribution would have (same dropna
    filter build_production_bundle itself uses) -- no model fit, no write."""
    if df is None:
        resolved = dataset_path if dataset_path is not None else _resolve_latest_for_production(args.out_dir, RESEARCH_GLOB)
        if resolved is None:
            report = {"error": f"no dataset given and no {RESEARCH_GLOB} found in {args.out_dir}/"}
            log.info("[dry-run] fit-production: %s", report)
            return report
        dataset_path = resolved
        df = rm.load_research_dataset(dataset_path)

    df_valid = df.dropna(subset=[rm.LABEL_COL, "entry_idx"])
    report: dict = {
        "dataset_path": dataset_path,
        "n_total_rows": len(df),
        "n_training_rows_in_percentile_reference": len(df_valid),
        "score_model": rm.PRODUCTION_SCORE_MODEL,
    }
    tag_part = f"{args.tag}_" if args.tag else ""
    report["planned_output"] = os.path.join(
        args.out_dir, f"production_model_{tag_part}{len(df_valid)}rows_{date.today():%Y%m%d}.joblib",
    )
    log.info("[dry-run] fit-production: %s", report)
    return report


def save_production_bundle_artifact(bundle: rm.ProductionBundle, out_dir: str, tag: str) -> str:
    """research_data/production_model_<tag_><n_rows>rows_<YYYYMMDD>.joblib,
    matching save_research_dataset / save_oof_scores' naming convention.
    Delegates the actual (atomic) write to research.model.save_production_bundle.
    """
    os.makedirs(out_dir, exist_ok=True)
    tag_part = f"{tag}_" if tag else ""
    fname = f"production_model_{tag_part}{len(bundle.training_scores)}rows_{date.today():%Y%m%d}.joblib"
    path = os.path.join(out_dir, fname)
    rm.save_production_bundle(bundle, path)
    return path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    do_build, do_fit = selected_stages(args)
    do_fit_production = bool(args.fit_production)  # opt-in only, see build_arg_parser's help text
    as_of = _parse_as_of(args.as_of)
    horizons = research_mod.DEFAULT_HORIZONS

    log.info(
        "run_research: build-dataset=%s fit-model=%s fit-production=%s dry-run=%s out-dir=%s tag=%r",
        do_build, do_fit, do_fit_production, args.dry_run, args.out_dir, args.tag,
    )

    df_for_fit: pd.DataFrame | None = None
    dataset_path_for_fit: str | None = args.dataset_path

    if do_build:
        if args.dry_run:
            dry_run_build_dataset(args, as_of)
        else:
            df = run_build_dataset_stage(
                months=args.months, as_of=as_of, events_from=args.events_from, horizons=horizons,
                out_dir=args.out_dir,
            )
            path = research_mod.save_research_dataset(df, out_dir=args.out_dir, tag=args.tag)
            df_for_fit = df
            dataset_path_for_fit = path

    result_for_reuse: rm.ValidationResult | None = None  # shared with --fit-production if both ran here

    if do_fit:
        if args.dry_run:
            dry_run_fit_model(args, df=df_for_fit, dataset_path=dataset_path_for_fit)
        else:
            if df_for_fit is None:
                resolved = dataset_path_for_fit or _resolve_latest(args.out_dir, RESEARCH_GLOB)
                if resolved is None:
                    raise SystemExit(
                        f"fit-model: no dataset given and no {RESEARCH_GLOB} found in "
                        f"{args.out_dir}/. Pass --dataset-path, or run with --build-dataset first."
                    )
                df_for_fit = rm.load_research_dataset(resolved)
                dataset_path_for_fit = resolved
            result = run_fit_model_stage(
                df_for_fit, n_folds=args.n_folds, horizon=args.horizon, embargo=args.embargo,
                min_fold_train_rows=args.min_fold_train_rows, n_shuffle_seeds=args.n_shuffle_seeds,
                run_shap_interactions=not args.no_shap_interactions,
            )
            oof_path = rm.save_oof_scores(result.oof_scores, out_dir=args.out_dir, tag=args.tag)
            result_for_reuse = result
            if not result.portfolio_sharpe.empty:
                log.info(
                    "fit-model: portfolio Sharpe (OOF, top-N equal weight) --\n%s",
                    result.portfolio_sharpe.head(10).to_string(index=False),
                )
            if args.oof_path_out:
                _write_oof_path_file(args.oof_path_out, oof_path)

    if do_fit_production:
        if args.dry_run:
            dry_run_fit_production(args, df=df_for_fit, dataset_path=dataset_path_for_fit)
        else:
            if df_for_fit is None:
                resolved = dataset_path_for_fit or _resolve_latest_for_production(args.out_dir, RESEARCH_GLOB)
                if resolved is None:
                    raise SystemExit(
                        f"fit-production: no dataset given and no {RESEARCH_GLOB} found in "
                        f"{args.out_dir}/. Pass --dataset-path, or run with --build-dataset first."
                    )
                df_for_fit = rm.load_research_dataset(resolved)
                dataset_path_for_fit = resolved
            bundle = run_fit_production_stage(
                df_for_fit, n_folds=args.n_folds, horizon=args.horizon, embargo=args.embargo,
                min_fold_train_rows=args.min_fold_train_rows, n_shuffle_seeds=args.n_shuffle_seeds,
                source_path=dataset_path_for_fit or "", result=result_for_reuse,
                allow_never_live_features=args.allow_never_live_features,
            )
            save_production_bundle_artifact(bundle, out_dir=args.out_dir, tag=args.tag)

    if args.dry_run:
        log.info("run_research: dry-run complete -- nothing written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
