"""Tests for ensemble_model.py: the bagged-ensemble scorer built to test
whether averaging across refits fixes the instability refit_stability.py's
gate found in a single fit (top-k selection overlap, rank churn, vol-matched
excess, PASS/FAIL verdicts -- same measurement code, applied to ensemble
scores).

Follows this repo's pytest conventions: plain pytest, class-grouped tests
(see tests/test_ticker_reuse.py), a small self-contained synthetic research
dataset matching backtest.research's real schema (same approach as
tests/test_refit_stability.py's make_synthetic_research_df), no network
access, no dependency on the real research_data/*.parquet files.

Runnable standalone via `python -m pytest tests/test_ensemble_model.py -q`
from the repo root.
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
import ensemble_model as em  # noqa: E402
import refit_stability as rs  # noqa: E402
from research import model as rm  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic dataset builder -- duplicated (not imported) per this repo's
# established per-file self-containment convention (see
# tests/test_refit_stability.py's own copy of this same builder).
# ---------------------------------------------------------------------------
_FEATURE_KEYS = list(ics.DEFAULT_WEIGHTS.keys())
_BOOL_FEATURES = {
    "x_has_ceo", "x_has_cfo", "x_has_chairman", "x_has_president", "x_has_coo",
    "x_is_first_ever_cluster",
}


def make_synthetic_research_df(
    n: int = 400, seed: int = 0, start: date = date(2020, 1, 1), tail_scale: float = 6.0,
) -> pd.DataFrame:
    """One row per synthetic episode -- see tests/test_refit_stability.py's
    identical builder for the full rationale behind each column."""
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
        elif col == "x_vol_63_ann":
            df[col] = rng.uniform(0.1, 1.5, n)
        else:
            df[col] = rng.standard_normal(n)

    for k in _FEATURE_KEYS:
        df[f"f_{k}"] = (rng.random(n) < 0.25).astype(int)
    df["conviction_score"] = rng.integers(-5, 8, n)

    df["adj_63"] = rng.standard_normal(n) * 0.05 * tail_scale
    df["spy_63"] = rng.standard_normal(n) * 0.02
    df["fwd_63"] = df["adj_63"] + df["spy_63"]
    df["delisted_63"] = rng.random(n) < 0.01
    return df


# ---------------------------------------------------------------------------
# 1. Feature-column workaround
# ---------------------------------------------------------------------------
class TestFeatureColumnWorkaround:
    def test_resolve_feature_cols_identifies_missing_columns(self):
        df = make_synthetic_research_df(n=100, seed=1)
        df_short = df.drop(columns=[rm.FEATURE_COLS[0], rm.FEATURE_COLS[-1]])
        usable, padded = em._resolve_feature_cols(df_short)
        assert rm.FEATURE_COLS[0] not in usable
        assert rm.FEATURE_COLS[-1] not in usable
        assert set(padded) == {rm.FEATURE_COLS[0], rm.FEATURE_COLS[-1]}
        assert len(usable) == len(rm.FEATURE_COLS) - 2

    def test_resolve_feature_cols_no_missing(self):
        df = make_synthetic_research_df(n=50, seed=2)
        usable, padded = em._resolve_feature_cols(df)
        assert padded == []
        assert usable == list(rm.FEATURE_COLS)

    def test_pad_for_schema_adds_all_nan_columns_without_mutating_input(self):
        df = make_synthetic_research_df(n=50, seed=3)
        df_short = df.drop(columns=[rm.FEATURE_COLS[0]])
        out = em._pad_for_schema(df_short, [rm.FEATURE_COLS[0]])
        assert rm.FEATURE_COLS[0] not in df_short.columns  # original untouched
        assert rm.FEATURE_COLS[0] in out.columns
        assert out[rm.FEATURE_COLS[0]].isna().all()

    def test_pad_for_schema_noop_when_nothing_missing(self):
        df = make_synthetic_research_df(n=20, seed=4)
        out = em._pad_for_schema(df, [])
        assert out is df  # explicit no-copy fast path


# ---------------------------------------------------------------------------
# 2. fit_member_oof
# ---------------------------------------------------------------------------
class TestFitMemberOof:
    def test_different_seeds_give_different_scores(self):
        df = make_synthetic_research_df(n=400, seed=10)
        oof_a = em.fit_member_oof(df, member_seed=1, feature_cols=rm.FEATURE_COLS, n_folds=3, min_fold_train_rows=15)
        oof_b = em.fit_member_oof(df, member_seed=2, feature_cols=rm.FEATURE_COLS, n_folds=3, min_fold_train_rows=15)
        merged = oof_a.merge(oof_b, on=["ticker", "event_day"], suffixes=("_a", "_b"))
        assert len(merged) > 0
        # Different random_state -> not bitwise identical scores across the board.
        assert not np.allclose(merged["oof_tail_classifier_a"], merged["oof_tail_classifier_b"])

    def test_oof_scores_only_cover_rows_that_were_ever_in_a_test_fold(self):
        df = make_synthetic_research_df(n=400, seed=11)
        oof = em.fit_member_oof(df, member_seed=5, feature_cols=rm.FEATURE_COLS, n_folds=3, min_fold_train_rows=15)
        # Every scored (ticker, event_day) must come from df itself.
        assert set(zip(oof["ticker"], oof["event_day"])).issubset(set(zip(df["ticker"], df["event_day"])))
        assert len(oof) <= len(df)


# ---------------------------------------------------------------------------
# 3. build_ensemble_oof -- shape, combination math, and the OOF-discipline
#    guard (the most important test class in this file per the task brief).
# ---------------------------------------------------------------------------
class TestBuildEnsembleOofShape:
    def test_row_resample_none_every_member_shares_identical_oof_universe(self):
        """The core textbook-bagging guarantee this module's docstring
        claims for the default row_resample_mode="none": since
        make_purged_expanding_folds is deterministic given (df, n_folds,
        horizon, embargo), every member fit on the SAME df gets the SAME
        fold split, so a row is scored by ALL members or NONE -- never a
        partial subset. n_members_scored must therefore only ever take the
        values {0, n_members_fit}."""
        df = make_synthetic_research_df(n=400, seed=20)
        ensemble, meta = em.build_ensemble_oof(
            df, n_members=4, base_seed=0, feature_cols=rm.FEATURE_COLS,
            n_folds=3, min_fold_train_rows=15,
        )
        assert meta["n_members_fit"] == 4
        observed = set(ensemble["n_members_scored"].unique())
        assert observed.issubset({0, 4})
        # At least some rows should actually be scored by all 4 members.
        assert (ensemble["n_members_scored"] == 4).sum() > 0

    def test_ensemble_score_avg_matches_manual_mean(self):
        df = make_synthetic_research_df(n=400, seed=21)
        ensemble, _ = em.build_ensemble_oof(
            df, n_members=3, base_seed=0, feature_cols=rm.FEATURE_COLS,
            n_folds=3, min_fold_train_rows=15,
        )
        fully_scored = ensemble[ensemble["n_members_scored"] == 3]
        assert len(fully_scored) > 0
        member_cols = [c for c in fully_scored.columns if c.startswith("_member") and c.endswith("_score")]
        manual_mean = fully_scored[member_cols].mean(axis=1)
        pd.testing.assert_series_equal(
            fully_scored[em.ENSEMBLE_SCORE_AVG_COL].reset_index(drop=True),
            manual_mean.reset_index(drop=True), check_names=False,
        )

    def test_ensemble_rank_col_orders_opposite_of_mean_percentile_rank(self):
        # Rows with a LOW mean percentile rank (near the top of every
        # member's own ranking) must get a HIGH ensemble_score_rankavg.
        df = make_synthetic_research_df(n=400, seed=22)
        ensemble, _ = em.build_ensemble_oof(
            df, n_members=3, base_seed=0, feature_cols=rm.FEATURE_COLS,
            n_folds=3, min_fold_train_rows=15,
        )
        fully_scored = ensemble[ensemble["n_members_scored"] == 3].copy()
        pctrank_cols = [c for c in fully_scored.columns if c.startswith("_member") and c.endswith("_pctrank")]
        mean_pctrank = fully_scored[pctrank_cols].mean(axis=1)
        # Spearman-style monotonic check: sorting by ensemble_score_rankavg
        # descending must exactly match sorting by mean_pctrank ascending.
        order_by_score = fully_scored.assign(_mp=mean_pctrank).sort_values(
            em.ENSEMBLE_RANK_COL, ascending=False
        )["_mp"].to_numpy()
        assert np.all(np.diff(order_by_score) >= -1e-12)  # non-decreasing

    def test_zero_members_raises(self):
        df = make_synthetic_research_df(n=100, seed=23)
        with pytest.raises(ValueError):
            em.build_ensemble_oof(df, n_members=0, feature_cols=rm.FEATURE_COLS)

    def test_unknown_row_resample_mode_raises(self):
        df = make_synthetic_research_df(n=100, seed=24)
        with pytest.raises(ValueError):
            em.build_ensemble_oof(df, n_members=2, feature_cols=rm.FEATURE_COLS, row_resample_mode="not_a_mode")

    def test_all_members_failing_raises_runtime_error(self, monkeypatch):
        df = make_synthetic_research_df(n=100, seed=25)

        def explode(*a, **kw):
            raise RuntimeError("simulated member failure")

        monkeypatch.setattr(em, "fit_member_oof", explode)
        with pytest.raises(RuntimeError, match="all 3 member"):
            em.build_ensemble_oof(df, n_members=3, feature_cols=rm.FEATURE_COLS)


# ---------------------------------------------------------------------------
# 4. OOF-DISCIPLINE GUARD -- direct unit tests of the combination logic
#    against controlled, monkeypatched member outputs (decoupled from real
#    LightGBM fits), proving build_ensemble_oof's merge/average step can
#    never fabricate a score, never leak an in-fold row into the average,
#    and never lets a missing member silently read as a zero.
# ---------------------------------------------------------------------------
class TestOofDisciplineGuard:
    def _stub_member_frame(self, rows: list[tuple]) -> pd.DataFrame:
        """rows: (ticker, event_day, entry_day, entry_idx, adj_63, score)."""
        return pd.DataFrame(rows, columns=["ticker", "event_day", "entry_day", "entry_idx", rm.LABEL_COL, "oof_tail_classifier"])

    def test_ensemble_average_excludes_members_that_never_scored_a_key(self, monkeypatch):
        """Member 0 scores rows A and B; member 1 (simulating a row purged
        out of every one of ITS fold's test blocks, or dropped by a
        row-resample) scores ONLY row A. The ensemble average for row B
        must equal member 0's own score for B exactly -- not member 0's
        score averaged against an assumed-zero for member 1, and not NaN
        propagating so as to silently drop the row from a naive mean."""
        d0 = date(2020, 1, 1)
        frames = {
            0: self._stub_member_frame([
                ("A", d0, d0, 0, 0.10, 0.80),
                ("B", d0 + timedelta(days=1), d0, 1, 0.20, 0.40),
            ]),
            1: self._stub_member_frame([
                ("A", d0, d0, 0, 0.10, 0.60),
                # no row for B -- B was never in this member's OOF output.
            ]),
        }

        def fake_fit_member_oof(df_pool, *, member_seed, **kw):
            return frames[member_seed]

        monkeypatch.setattr(em, "fit_member_oof", fake_fit_member_oof)

        df_pool = pd.DataFrame({
            "ticker": ["A", "B"], "event_day": [d0, d0 + timedelta(days=1)],
            "entry_day": [d0, d0], "entry_idx": [0, 1], rm.LABEL_COL: [0.10, 0.20],
        })
        ensemble, meta = em.build_ensemble_oof(
            df_pool, n_members=2, base_seed=0, feature_cols=rm.FEATURE_COLS,
        )
        # base_seed=0 -> member_seed for m in {0,1} is 0*1_000_003+m = {0, 1}, matching frames' keys.
        assert meta["n_members_fit"] == 2

        row_a = ensemble[ensemble["ticker"] == "A"].iloc[0]
        row_b = ensemble[ensemble["ticker"] == "B"].iloc[0]
        assert row_a["n_members_scored"] == 2
        assert row_a[em.ENSEMBLE_SCORE_AVG_COL] == pytest.approx((0.80 + 0.60) / 2)
        assert row_b["n_members_scored"] == 1
        assert row_b[em.ENSEMBLE_SCORE_AVG_COL] == pytest.approx(0.40)  # NOT (0.40 + 0) / 2 == 0.20

    def test_ensemble_never_invents_a_key_no_member_scored(self, monkeypatch):
        """A key present in df_pool's identity columns but scored by NO
        member (e.g. purged out of every fold everywhere) must survive in
        the ensemble frame with n_members_scored == 0 and NaN ensemble
        scores -- never silently dropped, and never given a fabricated
        value."""
        d0 = date(2020, 1, 1)
        frames = {0: self._stub_member_frame([("A", d0, d0, 0, 0.10, 0.80)])}

        def fake_fit_member_oof(df_pool, *, member_seed, **kw):
            return frames[0]

        monkeypatch.setattr(em, "fit_member_oof", fake_fit_member_oof)

        df_pool = pd.DataFrame({
            "ticker": ["A", "C"], "event_day": [d0, d0 + timedelta(days=2)],
            "entry_day": [d0, d0], "entry_idx": [0, 2], rm.LABEL_COL: [0.10, 0.30],
        })
        ensemble, _ = em.build_ensemble_oof(df_pool, n_members=1, base_seed=0, feature_cols=rm.FEATURE_COLS)

        row_c = ensemble[ensemble["ticker"] == "C"].iloc[0]
        assert row_c["n_members_scored"] == 0
        assert row_c[em.ENSEMBLE_SCORE_AVG_COL] != row_c[em.ENSEMBLE_SCORE_AVG_COL]  # NaN
        assert row_c[em.ENSEMBLE_RANK_COL] != row_c[em.ENSEMBLE_RANK_COL]  # NaN

    def test_row_resample_none_calls_every_member_with_the_identical_pool_object(self, monkeypatch):
        """Directly asserts the mechanism the "identical fold split across
        members" claim rests on: with row_resample_mode="none",
        build_ensemble_oof must pass the SAME df_pool (not a resampled
        copy) to every member's fit_member_oof call."""
        seen_ids = []
        real_df = make_synthetic_research_df(n=60, seed=30)

        def fake_fit_member_oof(df_pool, *, member_seed, **kw):
            seen_ids.append(id(df_pool))
            return pd.DataFrame(columns=["ticker", "event_day", "entry_day", "entry_idx", rm.LABEL_COL, "oof_tail_classifier"])

        monkeypatch.setattr(em, "fit_member_oof", fake_fit_member_oof)
        # Each stub call succeeds (returns an empty-but-valid OOF frame), so
        # this does not raise -- we only care about what df_pool object each
        # member call received.
        em.build_ensemble_oof(real_df, n_members=3, base_seed=0, feature_cols=rm.FEATURE_COLS, row_resample_mode="none")
        assert len(seen_ids) == 3
        assert len(set(seen_ids)) == 1  # every call got the exact same object

    def test_row_resample_drop1pct_calls_members_with_different_resampled_pools(self, monkeypatch):
        seen_lens = []
        real_df = make_synthetic_research_df(n=500, seed=31)

        def fake_fit_member_oof(df_pool, *, member_seed, **kw):
            seen_lens.append(len(df_pool))
            return pd.DataFrame(columns=["ticker", "event_day", "entry_day", "entry_idx", rm.LABEL_COL, "oof_tail_classifier"])

        monkeypatch.setattr(em, "fit_member_oof", fake_fit_member_oof)
        em.build_ensemble_oof(
            real_df, n_members=3, base_seed=0, feature_cols=rm.FEATURE_COLS,
            row_resample_mode="drop1pct", row_resample_drop_frac=0.01,
        )
        # drop1pct of n=500 drops round(500*0.01)=5 rows -> every member sees 495 rows.
        assert seen_lens == [495, 495, 495]


# ---------------------------------------------------------------------------
# 5. volmatch_significance
# ---------------------------------------------------------------------------
class TestVolmatchSignificance:
    def test_all_positive_gives_significant_result(self):
        vals = np.array([0.04, 0.05, 0.045, 0.052, 0.048])
        out = em.volmatch_significance(vals)
        assert out["n_valid"] == 5
        assert out["n_positive"] == 5
        assert out["pvalue"] < 0.05

    def test_mixed_sign_not_significant(self):
        vals = np.array([0.04, -0.03, 0.01, -0.02, 0.0])
        out = em.volmatch_significance(vals)
        assert out["n_positive"] == 2  # strictly > 0: 0.04, 0.01
        assert not (out["pvalue"] == out["pvalue"] and out["pvalue"] < 0.05)

    def test_drops_nan(self):
        vals = np.array([0.04, float("nan"), 0.05])
        out = em.volmatch_significance(vals)
        assert out["n_valid"] == 2

    def test_single_value_reports_nan_stats(self):
        out = em.volmatch_significance(np.array([0.05]))
        assert out["n_valid"] == 1
        assert out["tstat"] != out["tstat"]  # NaN
        assert out["pvalue"] != out["pvalue"]


# ---------------------------------------------------------------------------
# 6. CLI argument parsing
# ---------------------------------------------------------------------------
class TestArgParsing:
    def test_defaults(self):
        args = em.build_arg_parser().parse_args([])
        assert args.research == em.DEFAULT_RESEARCH_PATH
        assert args.n_members == em.N_MEMBERS_DEFAULT
        assert args.row_resample_mode == em.ROW_RESAMPLE_MODE_DEFAULT
        assert args.combine_methods == ",".join(em.COMBINE_METHODS_DEFAULT)
        assert args.retention_bar == pytest.approx(em.RETENTION_BAR_DEFAULT)
        assert args.volmatch_threshold_pp == pytest.approx(em.VOLMATCH_THRESHOLD_PP_DEFAULT)
        assert not args.regularized
        assert not args.baseline_only

    def test_row_resample_mode_rejects_unknown_choice(self):
        with pytest.raises(SystemExit):
            em.build_arg_parser().parse_args(["--row-resample-mode", "shuffle_everything"])

    def test_row_resample_mode_accepts_bootstrap(self):
        args = em.build_arg_parser().parse_args(["--row-resample-mode", "bootstrap"])
        assert args.row_resample_mode == "bootstrap"

    def test_top_ks_parsing(self):
        assert em._parse_top_ks("5,10,15,25") == (5, 10, 15, 25)

    def test_top_ks_parse_error_raises_systemexit(self):
        with pytest.raises(SystemExit):
            em._parse_top_ks("five,ten")

    def test_combine_methods_parsing(self):
        assert em._parse_combine_methods("score,rank") == ("score", "rank")
        assert em._parse_combine_methods("rank") == ("rank",)

    def test_combine_methods_rejects_unknown(self):
        with pytest.raises(SystemExit):
            em._parse_combine_methods("score,quantile")

    def test_assemble_lgbm_overrides_regularized_preset(self):
        args = em.build_arg_parser().parse_args(["--regularized"])
        overrides = em._assemble_lgbm_overrides(args)
        assert overrides == em.REGULARIZED_LGBM_OVERRIDES

    def test_assemble_lgbm_overrides_explicit_wins_over_preset(self):
        args = em.build_arg_parser().parse_args(["--regularized", "--num-leaves", "3"])
        overrides = em._assemble_lgbm_overrides(args)
        assert overrides["num_leaves"] == 3
        assert overrides["reg_lambda"] == em.REGULARIZED_LGBM_OVERRIDES["reg_lambda"]

    def test_assemble_lgbm_overrides_empty_by_default(self):
        args = em.build_arg_parser().parse_args([])
        assert em._assemble_lgbm_overrides(args) == {}


# ---------------------------------------------------------------------------
# 7. Persistence (atomic write, reusing refit_stability's helpers)
# ---------------------------------------------------------------------------
class TestSaveEnsembleResults:
    def test_save_ensemble_oof_leaves_no_tmp_file(self, tmp_path):
        df = pd.DataFrame({"ticker": ["A"], "event_day": [date(2020, 1, 1)], em.ENSEMBLE_SCORE_AVG_COL: [0.5]})
        path = em.save_ensemble_oof(df, out_dir=str(tmp_path), tag="unittest")
        assert os.path.exists(path)
        assert not os.path.exists(path + ".tmp")
        assert "ensemble_oof_scores_unittest_" in os.path.basename(path)
        back = pd.read_parquet(path)
        assert len(back) == 1

    def test_save_ensemble_stability_results_names_include_mode_and_method(self, tmp_path):
        rep_df = pd.DataFrame({"replicate_id": [0, 1], "retention_k5": [0.5, 0.6]})
        summary_df = pd.DataFrame({"k": [5], "verdict": ["FAIL"]})
        rep_path, sum_path = em.save_ensemble_stability_results(
            rep_df, summary_df, out_dir=str(tmp_path), mode="drop1pct", combine_method="rank", tag="t",
        )
        assert os.path.exists(rep_path)
        assert os.path.exists(sum_path)
        assert "ensemble_stability_replicates_drop1pct_rank_t_" in os.path.basename(rep_path)
        assert "ensemble_stability_summary_drop1pct_rank_t_" in os.path.basename(sum_path)


# ---------------------------------------------------------------------------
# 8. End-to-end: run_ensemble_stability_gate on a small synthetic dataset
#    (real, fast fits -- no network, no real research_data/*.parquet files).
# ---------------------------------------------------------------------------
class TestRunEnsembleStabilityGateEndToEnd:
    def test_produces_expected_shape_and_verdicts(self, tmp_path):
        df = make_synthetic_research_df(n=500, seed=40)
        research_path = str(tmp_path / "research.parquet")
        df.to_parquet(research_path, index=False)

        results = em.run_ensemble_stability_gate(
            research_path, n_replicates=2, mode="drop1pct", seed=0,
            n_members=2, n_folds=3, min_fold_train_rows=15,
            top_ks=(3, 5), n_boot=100, combine_methods=("score", "rank"),
        )
        assert set(results.keys()) == {"score", "rank"}
        for method in ("score", "rank"):
            state = results[method]
            rep_df = state["replicate_df"]
            assert len(rep_df) == 2
            assert (rep_df["combine_method"] == method).all()
            for col in ("retention_k3", "retention_k5", "jaccard_k3", "jaccard_k5", "volmatch_excess"):
                assert col in rep_df.columns
            assert set(state["summary_df"]["k"]) == {3, 5}
            assert set(state["summary_df"]["verdict"]).issubset({"PASS", "FAIL"})
            assert "tstat" in state["volmatch_sig"]

    def test_bootstrap_mode_runs_end_to_end(self, tmp_path):
        df = make_synthetic_research_df(n=500, seed=41)
        research_path = str(tmp_path / "research.parquet")
        df.to_parquet(research_path, index=False)

        results = em.run_ensemble_stability_gate(
            research_path, n_replicates=2, mode="bootstrap", seed=1,
            n_members=2, n_folds=3, min_fold_train_rows=15,
            top_ks=(5,), n_boot=100, combine_methods=("score",),
        )
        assert results["score"]["replicate_df"]["mode"].eq("bootstrap").all()

    def test_row_resample_mode_drop1pct_runs_end_to_end(self, tmp_path):
        """Members get their own resampled pool on top of the outer
        replicate perturbation -- a heavier, but still valid, configuration."""
        df = make_synthetic_research_df(n=500, seed=42)
        research_path = str(tmp_path / "research.parquet")
        df.to_parquet(research_path, index=False)

        results = em.run_ensemble_stability_gate(
            research_path, n_replicates=2, mode="drop1pct", seed=2,
            n_members=2, row_resample_mode="drop1pct", n_folds=3, min_fold_train_rows=15,
            top_ks=(5,), n_boot=100, combine_methods=("score",),
        )
        assert len(results["score"]["replicate_df"]) == 2

    def test_regularized_overrides_change_scores(self, tmp_path):
        df = make_synthetic_research_df(n=500, seed=43)
        research_path = str(tmp_path / "research.parquet")
        df.to_parquet(research_path, index=False)

        results_plain = em.run_ensemble_stability_gate(
            research_path, n_replicates=1, mode="drop1pct", seed=3,
            n_members=2, n_folds=3, min_fold_train_rows=15, top_ks=(5,), n_boot=50,
            combine_methods=("score",),
        )
        results_reg = em.run_ensemble_stability_gate(
            research_path, n_replicates=1, mode="drop1pct", seed=3,
            n_members=2, n_folds=3, min_fold_train_rows=15, top_ks=(5,), n_boot=50,
            combine_methods=("score",), lgbm_overrides=em.REGULARIZED_LGBM_OVERRIDES,
        )
        # Different model capacity -> baseline gap metrics should differ (not
        # a strict correctness requirement, just confirms overrides actually
        # reach the fit).
        assert (
            results_plain["score"]["baseline_gap"]["gap_5_6"]
            != results_reg["score"]["baseline_gap"]["gap_5_6"]
        )

    def test_raises_when_every_replicate_fails(self, tmp_path, monkeypatch):
        df = make_synthetic_research_df(n=500, seed=44)
        research_path = str(tmp_path / "research.parquet")
        df.to_parquet(research_path, index=False)

        def explode(*a, **kw):
            raise RuntimeError("simulated ensemble failure")

        monkeypatch.setattr(em, "build_ensemble_oof", lambda *a, **kw: explode())
        # First call (the baseline ensemble) itself will raise -- that's fine,
        # it should propagate rather than being silently swallowed.
        with pytest.raises(RuntimeError):
            em.run_ensemble_stability_gate(
                research_path, n_replicates=2, mode="drop1pct",
                n_members=2, n_folds=3, min_fold_train_rows=15, top_ks=(5,), n_boot=50,
            )
