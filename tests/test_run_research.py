"""Tests for run_research.py: the CLI entrypoint that rebuilds the two
research artifacts (research_<...>rows_<date>.parquet and
oof_scores_<date>.parquet) backtest's model_ranked_* strategies depend on.

Runnable standalone via `python -m pytest tests/test_run_research.py -q`
from the repo root. No network access: every test either exercises pure
argument-parsing/reporting logic, or builds a small synthetic research
dataset in memory (matching backtest.research's real schema, same approach
as tests/test_model.py's make_synthetic_dataset) rather than touching
price_cache/, ipo_cache/, or clusters_history/.
"""

from __future__ import annotations

import os
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import insider_cluster_buys as ics  # noqa: E402
import run_research as rr  # noqa: E402
from backtest import research as research_mod  # noqa: E402
from backtest.research import DEFAULT_HORIZONS, _label_cols  # noqa: E402
from research import model as rm  # noqa: E402

FAST_LGBM_PARAMS = dict(
    n_estimators=40, num_leaves=7, min_child_samples=5, learning_rate=0.2,
    subsample=0.9, subsample_freq=1, colsample_bytree=0.9, reg_lambda=1.0,
    random_state=0, verbosity=-1,
)

_FEATURE_KEYS = list(ics.DEFAULT_WEIGHTS.keys())
_BOOL_FEATURES = {
    "x_has_ceo", "x_has_cfo", "x_has_chairman", "x_has_president", "x_has_coo",
    "x_is_first_ever_cluster",
}


# ---------------------------------------------------------------------------
# Synthetic research-dataset builder (mirrors tests/test_model.py's, kept
# small and self-contained here so this file has no cross-test-module
# import dependency).
# ---------------------------------------------------------------------------
def make_synthetic_research_df(n: int = 200, seed: int = 0, start: date = date(2020, 1, 1)) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    entry_idx = np.arange(n)
    event_day = [start + timedelta(days=int(i)) for i in entry_idx]
    entry_day = [d + timedelta(days=1) for d in event_day]

    row: dict = {
        "ticker": [f"T{i:05d}" for i in range(n)],
        "issuer_cik": [f"CIK{i:05d}" for i in range(n)],
        "event_day": event_day,
        "entry_day": entry_day,
        "entry_open": rng.uniform(5, 50, n),
        "entry_idx": entry_idx,
    }
    df = pd.DataFrame(row)

    for col in rm.FEATURE_COLS:
        if col in _BOOL_FEATURES:
            df[col] = rng.random(n) < 0.3
        elif col.startswith("x_n_"):
            df[col] = rng.integers(0, 6, n).astype(float)
        else:
            df[col] = rng.standard_normal(n)

    for k in _FEATURE_KEYS:
        df[f"f_{k}"] = (rng.random(n) < 0.25).astype(int)
    df["conviction_score"] = rng.integers(-5, 8, n)

    for h in DEFAULT_HORIZONS:
        df[f"adj_{h}"] = rng.standard_normal(n) * 0.05
        df[f"spy_{h}"] = rng.standard_normal(n) * 0.02
        df[f"fwd_{h}"] = df[f"adj_{h}"] + df[f"spy_{h}"]
        df[f"delisted_{h}"] = rng.random(n) < 0.01

    assert set(_label_cols(DEFAULT_HORIZONS)).issubset(df.columns)
    return df


# ---------------------------------------------------------------------------
# 1. Argument parsing
# ---------------------------------------------------------------------------
class TestArgParsing:
    def test_defaults(self):
        args = rr.build_arg_parser().parse_args([])
        assert args.build_dataset is False
        assert args.fit_model is False
        assert args.all is False
        assert args.dry_run is False
        assert args.months == rr.DEFAULT_MONTHS
        assert args.out_dir == rr.DEFAULT_OUT_DIR
        assert args.tag == ""
        assert args.dataset_path is None
        assert args.events_from == ""
        assert args.allow_never_live_features is False

    def test_months_default_is_96_inferred_from_the_real_artifact(self):
        # research_full_11026rows_20260731.parquet's event_day column spans
        # exactly 2018-07-17 .. 2026-07-17 (8 years). build_history turns
        # months_back into days via months_back * 30.44, and
        # (2026-07-17 - 2018-07-17).days / 30.44 == 95.99 -> 96. This test
        # pins that inference so a future edit can't silently drift back to
        # a guessed default.
        span_days = (date(2026, 7, 17) - date(2018, 7, 17)).days
        assert round(span_days / 30.44) == rr.DEFAULT_MONTHS == 96

    def test_flags_are_boolean_toggles_not_a_mutually_exclusive_group(self):
        # Both flags together must parse fine -- "runnable together" is a
        # hard requirement, not just the neither-flag default.
        args = rr.build_arg_parser().parse_args(["--build-dataset", "--fit-model"])
        assert args.build_dataset is True
        assert args.fit_model is True

    def test_dry_run_and_tag_and_out_dir_parse(self):
        args = rr.build_arg_parser().parse_args([
            "--dry-run", "--tag", "mytag", "--out-dir", "somewhere",
        ])
        assert args.dry_run is True
        assert args.tag == "mytag"
        assert args.out_dir == "somewhere"

    def test_invalid_as_of_raises_systemexit(self):
        args = rr.build_arg_parser().parse_args(["--as-of", "not-a-date"])
        with pytest.raises(SystemExit):
            rr._parse_as_of(args.as_of)

    def test_valid_as_of_parses_to_date(self):
        assert rr._parse_as_of("2026-01-15") == date(2026, 1, 15)

    def test_blank_as_of_defaults_to_today(self):
        assert rr._parse_as_of(None) == date.today()
        assert rr._parse_as_of("") == date.today()


# ---------------------------------------------------------------------------
# 2. Stage selection
# ---------------------------------------------------------------------------
class TestStageSelection:
    def test_neither_flag_runs_both(self):
        args = rr.build_arg_parser().parse_args([])
        assert rr.selected_stages(args) == (True, True)

    def test_all_flag_runs_both_even_with_no_other_flags(self):
        args = rr.build_arg_parser().parse_args(["--all"])
        assert rr.selected_stages(args) == (True, True)

    def test_build_dataset_only(self):
        args = rr.build_arg_parser().parse_args(["--build-dataset"])
        assert rr.selected_stages(args) == (True, False)

    def test_fit_model_only(self):
        args = rr.build_arg_parser().parse_args(["--fit-model"])
        assert rr.selected_stages(args) == (False, True)

    def test_both_flags_explicit_runs_both(self):
        args = rr.build_arg_parser().parse_args(["--build-dataset", "--fit-model"])
        assert rr.selected_stages(args) == (True, True)


# ---------------------------------------------------------------------------
# 3. _resolve_latest
# ---------------------------------------------------------------------------
class TestResolveLatest:
    def test_no_matches_returns_none(self, tmp_path):
        assert rr._resolve_latest(str(tmp_path), rr.RESEARCH_GLOB) is None

    def test_picks_most_recently_modified(self, tmp_path):
        older = tmp_path / "research_100rows_20260101.parquet"
        newer = tmp_path / "research_200rows_20260201.parquet"
        older.write_bytes(b"old")
        newer.write_bytes(b"new")
        # Force distinct mtimes regardless of filesystem timestamp resolution.
        os.utime(older, (1_000_000_000, 1_000_000_000))
        os.utime(newer, (2_000_000_000, 2_000_000_000))
        assert rr._resolve_latest(str(tmp_path), rr.RESEARCH_GLOB) == str(newer)

    def test_does_not_match_oof_files(self, tmp_path):
        (tmp_path / "oof_scores_20260101.parquet").write_bytes(b"x")
        assert rr._resolve_latest(str(tmp_path), rr.RESEARCH_GLOB) is None


# ---------------------------------------------------------------------------
# 3b. _resolve_latest_for_production: same newest-by-mtime pick as
# _resolve_latest, but --fit-production's auto-selection must be VISIBLE
# (WARNING-level log naming the resolved path and every candidate it beat),
# because an unnoticed auto-pick here silently decides a production
# bundle's feature set -- see research_groupE_10905rows_20260809.parquet
# landing as "the latest" research_*.parquet and nearly shipping 9
# never-live columns for exactly that reason.
# ---------------------------------------------------------------------------
class TestResolveLatestForProduction:
    def test_no_matches_returns_none(self, tmp_path):
        assert rr._resolve_latest_for_production(str(tmp_path), rr.RESEARCH_GLOB) is None

    def test_picks_same_file_as_resolve_latest(self, tmp_path):
        older = tmp_path / "research_noreuse_100rows_20260101.parquet"
        newer = tmp_path / "research_groupE_200rows_20260201.parquet"
        older.write_bytes(b"old")
        newer.write_bytes(b"new")
        os.utime(older, (1_000_000_000, 1_000_000_000))
        os.utime(newer, (2_000_000_000, 2_000_000_000))
        assert rr._resolve_latest_for_production(str(tmp_path), rr.RESEARCH_GLOB) == str(newer)
        assert rr._resolve_latest_for_production(str(tmp_path), rr.RESEARCH_GLOB) == \
            rr._resolve_latest(str(tmp_path), rr.RESEARCH_GLOB)

    def test_logs_resolved_path_and_all_candidates_at_warning(self, tmp_path, caplog):
        older = tmp_path / "research_noreuse_100rows_20260101.parquet"
        newer = tmp_path / "research_groupE_200rows_20260201.parquet"
        older.write_bytes(b"old")
        newer.write_bytes(b"new")
        os.utime(older, (1_000_000_000, 1_000_000_000))
        os.utime(newer, (2_000_000_000, 2_000_000_000))

        with caplog.at_level("WARNING", logger="run_research"):
            resolved = rr._resolve_latest_for_production(str(tmp_path), rr.RESEARCH_GLOB)

        assert resolved == str(newer)
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) == 1
        message = warnings[0].getMessage()
        assert os.path.basename(newer) in message
        assert os.path.basename(older) in message
        assert "--dataset-path" in message


# ---------------------------------------------------------------------------
# 4. Atomic write: temp-file-then-rename (research.save_research_dataset and
#    research.model.save_oof_scores both follow the split_fingerprint.py
#    _atomic_save pattern -- write to `<path>.tmp`, then os.replace()).
# ---------------------------------------------------------------------------
class TestAtomicWrite:
    def test_save_research_dataset_leaves_no_tmp_file_on_success(self, tmp_path):
        df = make_synthetic_research_df(n=5)
        path = research_mod.save_research_dataset(df, out_dir=str(tmp_path), tag="atomictest")
        assert os.path.exists(path)
        assert not os.path.exists(path + ".tmp")

    def test_save_research_dataset_interrupted_write_leaves_no_final_file(self, tmp_path, monkeypatch):
        df = make_synthetic_research_df(n=5)

        def boom(self, path, *a, **kw):
            # Simulate a write that gets partway through the TEMP file
            # before the process dies -- the real-path rename must never
            # have been reached.
            with open(path, "wb") as fh:
                fh.write(b"not a real parquet file")
            raise RuntimeError("simulated interruption")

        monkeypatch.setattr(pd.DataFrame, "to_parquet", boom)
        with pytest.raises(RuntimeError):
            research_mod.save_research_dataset(df, out_dir=str(tmp_path), tag="crash")

        final_files = [f for f in os.listdir(tmp_path) if f.endswith(".parquet") and not f.endswith(".tmp")]
        assert final_files == [], (
            "an interrupted write must never leave a file at the real (non-.tmp) "
            "path -- a later load must not see a half-written parquet as valid"
        )

    def test_save_oof_scores_leaves_no_tmp_file_on_success(self, tmp_path):
        oof = pd.DataFrame({
            "fold": [0, 0], "ticker": ["AAA", "BBB"],
            "event_day": [date(2020, 1, 1), date(2020, 1, 2)],
            "entry_day": [date(2020, 1, 2), date(2020, 1, 3)],
            "entry_idx": [0, 1], "adj_63": [0.1, -0.05],
            "oof_regressor": [0.1, 0.0], "oof_classifier": [0.6, 0.4],
            "oof_tail_classifier": [0.2, 0.1], "ten_pct_owner": [0.0, 1.0],
            "conviction_score": [3, -1],
        })
        path = rm.save_oof_scores(oof, out_dir=str(tmp_path), tag="atomictest")
        assert os.path.exists(path)
        assert not os.path.exists(path + ".tmp")
        assert "oof_scores_atomictest_" in os.path.basename(path)

    def test_save_oof_scores_interrupted_write_leaves_no_final_file(self, tmp_path, monkeypatch):
        oof = pd.DataFrame({"fold": [0], "ticker": ["AAA"], "adj_63": [0.1]})

        def boom(self, path, *a, **kw):
            with open(path, "wb") as fh:
                fh.write(b"partial")
            raise OSError("disk full (simulated)")

        monkeypatch.setattr(pd.DataFrame, "to_parquet", boom)
        with pytest.raises(OSError):
            rm.save_oof_scores(oof, out_dir=str(tmp_path), tag="crash")

        final_files = [f for f in os.listdir(tmp_path) if f.endswith(".parquet") and not f.endswith(".tmp")]
        assert final_files == []

    def test_default_tag_matches_existing_oof_scores_naming_convention(self, tmp_path):
        # The real file on disk is oof_scores_20260731.parquet -- no tag
        # segment. save_oof_scores with tag="" must reproduce that shape
        # exactly, not oof_scores__20260731.parquet or similar.
        oof = pd.DataFrame({"fold": [0], "ticker": ["AAA"], "adj_63": [0.1]})
        path = rm.save_oof_scores(oof, out_dir=str(tmp_path), tag="")
        fname = os.path.basename(path)
        assert fname == f"oof_scores_{date.today():%Y%m%d}.parquet"


# ---------------------------------------------------------------------------
# 5. Dry-run: reports without writing.
# ---------------------------------------------------------------------------
class TestDryRun:
    def test_fit_model_dry_run_writes_nothing(self, tmp_path):
        df = make_synthetic_research_df(n=200, seed=3)
        args = rr.build_arg_parser().parse_args([
            "--fit-model", "--dry-run", "--out-dir", str(tmp_path),
            "--n-folds", "3", "--min-fold-train-rows", "10",
        ])
        report = rr.dry_run_fit_model(args, df=df)
        assert os.listdir(tmp_path) == []
        assert report["n_total_rows"] == 200
        assert "expected_oof_rows" in report

    def test_fit_model_dry_run_expected_oof_rows_matches_real_fold_math(self, tmp_path):
        """The headline claim of dry_run_fit_model: it predicts the EXACT
        oof_scores row count with no model fit, by summing the test-block
        size of every fold whose train_idx clears min_fold_train_rows. This
        pins that arithmetic against make_purged_expanding_folds directly,
        the same check that was run against the real
        research_full_11026rows_20260731.parquet / oof_scores_20260731.parquet
        pair while building this script (11,026 -> 10,659 valid -> 8,882
        OOF rows, matching the file on disk exactly)."""
        df = make_synthetic_research_df(n=500, seed=5)
        args = rr.build_arg_parser().parse_args([
            "--fit-model", "--dry-run", "--out-dir", str(tmp_path),
            "--n-folds", "4", "--horizon", "63", "--embargo", "63",
            "--min-fold-train-rows", "20",
        ])
        report = rr.dry_run_fit_model(args, df=df)

        df_valid = df.dropna(subset=[rm.LABEL_COL, "entry_idx"])
        folds = rm.make_purged_expanding_folds(df_valid, n_folds=4, horizon=63, embargo=63)
        expected = sum(len(f.test_idx) for f in folds if len(f.train_idx) >= 20)

        assert report["expected_oof_rows"] == expected
        assert len(report["folds"]) == 4

    def test_fit_model_dry_run_no_dataset_found_reports_error_not_raise(self, tmp_path):
        args = rr.build_arg_parser().parse_args(["--fit-model", "--dry-run", "--out-dir", str(tmp_path)])
        report = rr.dry_run_fit_model(args)
        assert "error" in report
        assert os.listdir(tmp_path) == []

    def test_fit_model_dry_run_loads_from_dataset_path_when_no_df_given(self, tmp_path):
        df = make_synthetic_research_df(n=150, seed=9)
        path = research_mod.save_research_dataset(df, out_dir=str(tmp_path), tag="src")
        args = rr.build_arg_parser().parse_args([
            "--fit-model", "--dry-run", "--out-dir", str(tmp_path), "--dataset-path", path,
        ])
        report = rr.dry_run_fit_model(args, dataset_path=path)
        assert report["dataset_path"] == path
        assert report["n_total_rows"] == 150
        # dry run must not have written a second file next to the input.
        assert sorted(os.listdir(tmp_path)) == [os.path.basename(path)]

    def test_build_dataset_dry_run_writes_nothing_and_needs_no_price_fetch(self, tmp_path, monkeypatch):
        """dry_run_build_dataset must never touch PriceUniverse / prices.py
        -- that network/disk-heavy fetch is exactly the cost --dry-run
        exists to avoid paying. Patch backtest.prices.PriceUniverse.ensure
        to explode if called, proving the dry-run path never reaches it."""
        from backtest import prices as prices_mod

        def explode(self, *a, **kw):
            raise AssertionError("dry_run_build_dataset must not fetch prices")

        monkeypatch.setattr(prices_mod.PriceUniverse, "ensure", explode)

        args = rr.build_arg_parser().parse_args([
            "--build-dataset", "--dry-run", "--out-dir", str(tmp_path), "--months", "6",
        ])
        report = rr.dry_run_build_dataset(args, as_of=date(2026, 1, 1))
        assert os.listdir(tmp_path) == []
        assert report["months"] == 6
        assert "events_source" in report

    def test_build_dataset_dry_run_reports_expected_rows_from_events_from(self, tmp_path):
        """With --events-from pointing at a real (small) events parquet,
        dry_run_build_dataset reads it (cheap: a parquet read + cleanup,
        no scrape, no prices) and reports a heuristic row-count estimate
        derived from it."""
        # Build a tiny events parquet using backtest.history's own writer so
        # the on-disk shape matches exactly what load_events_df expects.
        from backtest import history as history_mod

        rows = []
        for i in range(20):
            rows.append({
                "ticker": f"T{i:03d}", "issuer_cik": f"CIK{i:03d}",
                "issuer_name": f"Issuer {i} Corp",
                "owner_cik": f"O{i:03d}", "owner_name": f"Owner {i}",
                "transaction_date": date(2024, 1, 2), "filing_date": date(2024, 1, 3),
                "is_director": True, "is_officer": False, "is_ten_percent_owner": False,
                "owner_roles": "Director", "value": 1000.0, "shares": 10.0,
                "price_per_share": 100.0, "is_10b5_1": False, "footnote_text": "",
                "pct_of_prior_stake": 0.01,
            })
        events_df = pd.DataFrame(rows)
        events_dir = tmp_path / "events_src"
        events_dir.mkdir()
        events_path = events_dir / "events_20240101_20240110.parquet"
        events_df.to_parquet(events_path, index=False)

        out_dir = tmp_path / "out"
        args = rr.build_arg_parser().parse_args([
            "--build-dataset", "--dry-run", "--out-dir", str(out_dir),
            "--events-from", str(events_path),
        ])
        report = rr.dry_run_build_dataset(args, as_of=date(2024, 1, 10))
        assert not out_dir.exists() or os.listdir(out_dir) == []
        assert report["n_event_rows"] == 20
        assert report["n_tickers"] == 20
        assert "expected_research_rows_heuristic" in report

    def test_cli_dry_run_end_to_end_writes_nothing(self, tmp_path, monkeypatch):
        """Full main() entrypoint, --all --dry-run: must not touch
        PriceUniverse (build-dataset side) and must not fit any model
        (fit-model side), and must leave out-dir untouched."""
        from backtest import prices as prices_mod

        def explode_ensure(self, *a, **kw):
            raise AssertionError("main() --dry-run must not fetch prices")

        def explode_fit(*a, **kw):
            raise AssertionError("main() --dry-run must not fit a model")

        monkeypatch.setattr(prices_mod.PriceUniverse, "ensure", explode_ensure)
        monkeypatch.setattr(rm, "fit_and_validate", explode_fit)

        rc = rr.main([
            "--all", "--dry-run", "--out-dir", str(tmp_path), "--months", "1",
        ])
        assert rc == 0
        assert os.listdir(tmp_path) == []


# ---------------------------------------------------------------------------
# 6. Real (non-dry-run) fit-model stage wiring, on a small synthetic
#    dataset -- exercises rm.fit_and_validate + rm.save_oof_scores for real,
#    cheaply, without touching price_cache/ipo_cache/clusters_history.
# ---------------------------------------------------------------------------
class TestFitModelStageReal:
    def test_run_fit_model_stage_and_save_round_trip(self, tmp_path):
        df = make_synthetic_research_df(n=260, seed=11)
        result = rr.run_fit_model_stage(
            df, n_folds=3, horizon=63, embargo=63, min_fold_train_rows=15,
            n_shuffle_seeds=3, run_shap_interactions=False,
        )
        assert result.n_folds_run >= 1
        assert len(result.oof_scores) > 0

        path = rm.save_oof_scores(result.oof_scores, out_dir=str(tmp_path), tag="realtest")
        assert os.path.exists(path)
        back = pd.read_parquet(path)
        assert len(back) == len(result.oof_scores)


# ---------------------------------------------------------------------------
# 7. --fit-production argument parsing / stage selection independence
# ---------------------------------------------------------------------------
class TestFitProductionArgParsing:
    def test_default_is_false(self):
        args = rr.build_arg_parser().parse_args([])
        assert args.fit_production is False

    def test_flag_sets_true(self):
        args = rr.build_arg_parser().parse_args(["--fit-production"])
        assert args.fit_production is True

    def test_not_part_of_all_or_no_flags_default(self):
        """--fit-production must be strictly opt-in: neither the bare
        invocation nor --all may imply it, or existing callers of this
        script (which already run bare/--all in CI-like contexts) would
        start paying for a stage they never asked for."""
        for argv in ([], ["--all"], ["--build-dataset", "--fit-model"]):
            args = rr.build_arg_parser().parse_args(argv)
            assert args.fit_production is False, argv
        # selected_stages' own return shape (used directly by other tests
        # with `== (True, True)`) must stay a 2-tuple -- fit-production is
        # wired independently in main(), not through this function.
        assert rr.selected_stages(rr.build_arg_parser().parse_args([])) == (True, True)

    def test_fit_production_alone_does_not_imply_build_or_fit(self):
        """The flip side of the "no flags = run both" rule: passing ONLY
        --fit-production must NOT fall into that same default, or a caller
        asking for just the (cheap-ish, reads-an-existing-dataset)
        production bundle would silently also trigger the expensive
        --build-dataset scrape/parse and --fit-model CV. This was caught by
        actually running `python run_research.py --fit-production
        --dataset-path ...` by hand: it started re-parsing 3 million cached
        SEC filings from scratch instead of just reading the given dataset."""
        args = rr.build_arg_parser().parse_args(["--fit-production", "--dataset-path", "x.parquet"])
        assert rr.selected_stages(args) == (False, False)

    def test_fit_production_combined_with_build_or_fit_still_runs_those(self):
        args = rr.build_arg_parser().parse_args(["--fit-production", "--build-dataset"])
        assert rr.selected_stages(args) == (True, False)
        args = rr.build_arg_parser().parse_args(["--fit-production", "--fit-model"])
        assert rr.selected_stages(args) == (False, True)

    def test_flags_combine_with_build_and_fit(self):
        args = rr.build_arg_parser().parse_args(["--build-dataset", "--fit-model", "--fit-production"])
        assert (args.build_dataset, args.fit_model, args.fit_production) == (True, True, True)


# ---------------------------------------------------------------------------
# 8. --fit-production dry run
# ---------------------------------------------------------------------------
class TestFitProductionDryRun:
    def test_dry_run_writes_nothing_and_reports_expected_keys(self, tmp_path):
        df = make_synthetic_research_df(n=120, seed=21)
        args = rr.build_arg_parser().parse_args(["--fit-production", "--dry-run", "--out-dir", str(tmp_path)])
        report = rr.dry_run_fit_production(args, df=df)

        assert os.listdir(tmp_path) == []
        assert report["n_total_rows"] == len(df)
        assert report["score_model"] == rm.PRODUCTION_SCORE_MODEL
        df_valid = df.dropna(subset=[rm.LABEL_COL, "entry_idx"])
        assert report["n_training_rows_in_percentile_reference"] == len(df_valid)
        assert "planned_output" in report

    def test_dry_run_no_dataset_found_reports_error_not_raise(self, tmp_path):
        args = rr.build_arg_parser().parse_args(["--fit-production", "--dry-run", "--out-dir", str(tmp_path)])
        report = rr.dry_run_fit_production(args)
        assert "error" in report

    def test_cli_dry_run_end_to_end_with_fit_production_writes_nothing(self, tmp_path, monkeypatch):
        """No --dataset-path and an empty --out-dir: dry_run_fit_production
        must fall back to its "no dataset found" report rather than
        crashing, and (like every --dry-run path) must never call
        fit_and_validate or write anything."""
        def explode_fit(*a, **kw):
            raise AssertionError("main() --dry-run must not fit a model")

        monkeypatch.setattr(rm, "fit_and_validate", explode_fit)
        rc = rr.main(["--fit-production", "--dry-run", "--out-dir", str(tmp_path)])
        assert rc == 0
        assert os.listdir(tmp_path) == []


# ---------------------------------------------------------------------------
# 9. Real (non-dry-run) --fit-production stage wiring
# ---------------------------------------------------------------------------
def _tail_signal_research_df(n: int = 220, seed: int = 31) -> pd.DataFrame:
    """make_synthetic_research_df's adj_63 sits at a ~0.05 std, which
    essentially never crosses TAIL_THRESH=0.20 -- the final full-dataset
    tail_classifier fit_and_validate step 10 fits would come back None
    (single class), which build_production_bundle correctly refuses to
    bundle. Scaling adj_63 up guarantees a real mix of both classes."""
    df = make_synthetic_research_df(n=n, seed=seed)
    df[rm.LABEL_COL] = df[rm.LABEL_COL] * 6.0
    return df


class TestFitProductionStageReal:
    # _tail_signal_research_df -> make_synthetic_research_df populates every
    # rm.FEATURE_COLS column, including live_score.SALE_FEATURE_COLS' 9
    # never-live columns -- so every real fit in this class needs
    # allow_never_live_features=True to reach the mechanics under test
    # (dataset wiring, reuse, CLI plumbing). The guard's own behavior is
    # covered separately below (TestFitProductionNeverLiveGuard) and in
    # tests/test_model.py's TestNeverLiveFeatureGuard.
    def test_run_fit_production_stage_fits_and_bundles(self, tmp_path):
        df = _tail_signal_research_df()
        bundle = rr.run_fit_production_stage(
            df, n_folds=3, horizon=63, embargo=63, min_fold_train_rows=15,
            n_shuffle_seeds=3, source_path="synthetic.parquet",
            allow_never_live_features=True,
        )
        assert bundle.feature_cols == rm.FEATURE_COLS
        assert len(bundle.training_scores) > 0
        assert bundle.provenance["source_dataset_path"] == "synthetic.parquet"

        path = rr.save_production_bundle_artifact(bundle, out_dir=str(tmp_path), tag="realtest")
        assert os.path.exists(path)
        assert "production_model_realtest" in os.path.basename(path)
        loaded = rm.load_production_bundle(path)
        np.testing.assert_allclose(loaded.training_scores, bundle.training_scores)

    def test_reuses_a_supplied_result_instead_of_refitting(self, tmp_path, monkeypatch):
        """When --fit-model already produced a ValidationResult in this same
        invocation, --fit-production must reuse it rather than paying for a
        second purged walk-forward CV -- this is what main() does when both
        flags are passed together. Proven here by making a second
        fit_and_validate call raise."""
        df = _tail_signal_research_df()
        result = rm.fit_and_validate(
            df, n_folds=3, horizon=63, embargo=63, min_fold_train_rows=15,
            n_shuffle_seeds=3, run_shap_interactions=False,
        )

        def explode(*a, **kw):
            raise AssertionError("fit_and_validate must not be called again when a result is reused")

        monkeypatch.setattr(rm, "fit_and_validate", explode)
        bundle = rr.run_fit_production_stage(
            df, n_folds=3, horizon=63, embargo=63, min_fold_train_rows=15,
            n_shuffle_seeds=3, source_path="synthetic.parquet", result=result,
            allow_never_live_features=True,
        )
        assert bundle.model is result.models[rm.PRODUCTION_SCORE_MODEL]

    def test_cli_fit_production_alone_writes_one_bundle_file(self, tmp_path):
        df = _tail_signal_research_df()
        dataset_path = research_mod.save_research_dataset(df, out_dir=str(tmp_path))

        rc = rr.main([
            "--fit-production", "--out-dir", str(tmp_path), "--dataset-path", dataset_path,
            "--n-folds", "3", "--horizon", "63", "--embargo", "63",
            "--min-fold-train-rows", "15", "--n-shuffle-seeds", "3",
            "--allow-never-live-features",
        ])
        assert rc == 0
        produced = [f for f in os.listdir(tmp_path) if f.startswith("production_model_")]
        assert len(produced) == 1


# ---------------------------------------------------------------------------
# 10. --allow-never-live-features: the escape hatch for research.model.
# build_production_bundle's default guard (see that function's docstring
# and research/model.py's _never_live_default_guard_cols). Argument
# parsing is covered by TestArgParsing.test_defaults /
# TestFitProductionArgParsing above; this section covers the actual
# refuse-by-default / opt-in-allow wiring through run_fit_production_stage
# and main().
# ---------------------------------------------------------------------------
class TestFitProductionNeverLiveGuard:
    def test_default_refuses_a_dataset_carrying_sale_features(self, tmp_path):
        """_tail_signal_research_df carries all 9 of live_score.SALE_FEATURE_COLS
        (make_synthetic_research_df populates every rm.FEATURE_COLS column) --
        exactly research_groupE_10905rows_20260809.parquet's shape. Without
        --allow-never-live-features, run_fit_production_stage must refuse
        rather than silently bundle it."""
        df = _tail_signal_research_df()
        with pytest.raises(ValueError, match="never computable live"):
            rr.run_fit_production_stage(
                df, n_folds=3, horizon=63, embargo=63, min_fold_train_rows=15,
                n_shuffle_seeds=3, source_path="synthetic.parquet",
            )

    def test_flag_allows_the_same_fit_to_succeed(self, tmp_path):
        df = _tail_signal_research_df()
        bundle = rr.run_fit_production_stage(
            df, n_folds=3, horizon=63, embargo=63, min_fold_train_rows=15,
            n_shuffle_seeds=3, source_path="synthetic.parquet",
            allow_never_live_features=True,
        )
        assert bundle.feature_cols == rm.FEATURE_COLS

    def test_cli_without_flag_exits_nonzero_and_writes_no_bundle(self, tmp_path):
        df = _tail_signal_research_df()
        dataset_path = research_mod.save_research_dataset(df, out_dir=str(tmp_path))

        with pytest.raises(ValueError, match="never computable live"):
            rr.main([
                "--fit-production", "--out-dir", str(tmp_path), "--dataset-path", dataset_path,
                "--n-folds", "3", "--horizon", "63", "--embargo", "63",
                "--min-fold-train-rows", "15", "--n-shuffle-seeds", "3",
            ])
        produced = [f for f in os.listdir(tmp_path) if f.startswith("production_model_")]
        assert produced == []
