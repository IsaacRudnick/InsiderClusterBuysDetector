"""Tests for refit_stability.py: the refit-stability gate for
research/model.py's ranking model (top-k selection overlap, rank churn,
score-gap mechanism check, vol-matched excess, PASS/FAIL verdicts).

Follows this repo's pytest conventions: plain pytest, class-grouped tests
(see tests/test_ticker_reuse.py), a small self-contained synthetic research
dataset matching backtest.research's real schema (same approach as
tests/test_model.py's make_synthetic_dataset / tests/test_run_research.py's
make_synthetic_research_df) rather than touching the real
research_data/*.parquet files, no network access.

Runnable standalone via `python -m pytest tests/test_refit_stability.py -q`
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
import refit_stability as rs  # noqa: E402
from research import model as rm  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic dataset builder -- matches backtest.research's real schema
# (mirrors tests/test_model.py's make_synthetic_dataset / tests/test_run_research.py's
# make_synthetic_research_df, duplicated here rather than imported, per this
# repo's established per-file self-containment convention).
# ---------------------------------------------------------------------------
_FEATURE_KEYS = list(ics.DEFAULT_WEIGHTS.keys())
_BOOL_FEATURES = {
    "x_has_ceo", "x_has_cfo", "x_has_chairman", "x_has_president", "x_has_coo",
    "x_is_first_ever_cluster",
}


def make_synthetic_research_df(
    n: int = 400, seed: int = 0, start: date = date(2020, 1, 1), tail_scale: float = 6.0,
) -> pd.DataFrame:
    """One row per synthetic episode, entry_idx = row position, event_day
    strictly increasing with entry_idx (matches the real dataset's
    construction). adj_63 is scaled by `tail_scale` so a real mix of both
    tail_classifier classes exists (adj_63 > research.model.TAIL_THRESH ==
    0.20 for a meaningful share of rows) -- an un-scaled draw almost never
    crosses 0.20, which would give fit_and_validate's final full-dataset
    tail_classifier a single-class target. x_vol_63_ann (needed for the
    vol-matched excess step) is drawn positive, matching a real annualized
    volatility feature.
    """
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


def make_baseline_oof(research_df: pd.DataFrame, *, n_folds: int = 3, min_fold_train_rows: int = 15) -> pd.DataFrame:
    """A real (fast) fit_and_validate call on `research_df`, used as the
    "baseline" OOF frame in end-to-end tests."""
    result = rm.fit_and_validate(
        research_df, n_folds=n_folds, min_fold_train_rows=min_fold_train_rows,
        run_shap_interactions=False, n_shuffle_seeds=2,
    )
    return result.oof_scores


# ---------------------------------------------------------------------------
# 1. resample_dataset
# ---------------------------------------------------------------------------
class TestResampleDataset:
    def test_drop1pct_drops_the_right_row_count(self):
        df = pd.DataFrame({"x": range(1000)})
        out = rs.resample_dataset(df, mode="drop1pct", seed=0, drop_frac=0.01)
        assert len(out) == 990

    def test_drop1pct_keeps_a_subset_of_original_values_no_duplicates(self):
        df = pd.DataFrame({"x": range(500)})
        out = rs.resample_dataset(df, mode="drop1pct", seed=1, drop_frac=0.02)
        assert out["x"].is_unique
        assert set(out["x"]).issubset(set(df["x"]))
        assert len(out) == 490

    def test_drop1pct_resets_index(self):
        df = pd.DataFrame({"x": range(200)})
        out = rs.resample_dataset(df, mode="drop1pct", seed=2)
        assert list(out.index) == list(range(len(out)))

    def test_bootstrap_preserves_length(self):
        df = pd.DataFrame({"x": range(300)})
        out = rs.resample_dataset(df, mode="bootstrap", seed=0)
        assert len(out) == len(df)

    def test_bootstrap_can_produce_duplicates(self):
        # With n=50 draws from n=50 with replacement, some repeat is all but
        # certain; this is deterministic given the fixed seed.
        df = pd.DataFrame({"x": range(50)})
        out = rs.resample_dataset(df, mode="bootstrap", seed=0)
        assert not out["x"].is_unique

    def test_bootstrap_resets_index(self):
        df = pd.DataFrame({"x": range(50)})
        out = rs.resample_dataset(df, mode="bootstrap", seed=3)
        assert list(out.index) == list(range(len(out)))

    def test_unknown_mode_raises(self):
        df = pd.DataFrame({"x": range(10)})
        with pytest.raises(ValueError):
            rs.resample_dataset(df, mode="not_a_mode")

    def test_empty_input_raises(self):
        with pytest.raises(ValueError):
            rs.resample_dataset(pd.DataFrame({"x": []}), mode="drop1pct")

    def test_drop1pct_at_least_one_row_dropped_even_for_tiny_frac(self):
        df = pd.DataFrame({"x": range(20)})
        out = rs.resample_dataset(df, mode="drop1pct", seed=0, drop_frac=0.001)
        assert len(out) == 19  # round(20*0.001)=0 -> floored up to 1 by max(1, ...)

    def test_same_seed_is_reproducible(self):
        df = pd.DataFrame({"x": range(500)})
        a = rs.resample_dataset(df, mode="drop1pct", seed=7)
        b = rs.resample_dataset(df, mode="drop1pct", seed=7)
        pd.testing.assert_frame_equal(a, b)


# ---------------------------------------------------------------------------
# 2. dedup_oof / ranked_picks / top_k_keys
# ---------------------------------------------------------------------------
def _mini_oof(rows: list[tuple]) -> pd.DataFrame:
    """rows: list of (ticker, event_day, score)."""
    return pd.DataFrame(rows, columns=["ticker", "event_day", "score"])


class TestDedupOof:
    def test_drops_null_score_rows(self):
        oof = _mini_oof([("A", date(2020, 1, 1), 0.5), ("B", date(2020, 1, 2), float("nan"))])
        out = rs.dedup_oof(oof, "score")
        assert len(out) == 1
        assert out["ticker"].iloc[0] == "A"

    def test_drops_duplicate_ticker_event_day_keeps_first(self):
        oof = _mini_oof([
            ("A", date(2020, 1, 1), 0.9),
            ("A", date(2020, 1, 1), 0.1),  # duplicate key, e.g. from bootstrap resampling
            ("B", date(2020, 1, 2), 0.5),
        ])
        out = rs.dedup_oof(oof, "score")
        assert len(out) == 2
        row_a = out[out["ticker"] == "A"].iloc[0]
        assert row_a["score"] == pytest.approx(0.9)  # first occurrence kept

    def test_no_dupes_no_change(self):
        oof = _mini_oof([("A", date(2020, 1, 1), 0.9), ("B", date(2020, 1, 2), 0.5)])
        out = rs.dedup_oof(oof, "score")
        assert len(out) == 2


class TestRankedPicks:
    def test_sorted_descending_by_score(self):
        oof = _mini_oof([("A", date(2020, 1, 1), 0.1), ("B", date(2020, 1, 2), 0.9), ("C", date(2020, 1, 3), 0.5)])
        ranked = rs.ranked_picks(oof, "score")
        assert list(ranked["ticker"]) == ["B", "C", "A"]
        assert list(ranked["rank"]) == [1, 2, 3]

    def test_ties_broken_by_ticker_then_event_day_ascending(self):
        oof = _mini_oof([
            ("Z", date(2020, 1, 1), 0.5),
            ("A", date(2020, 1, 1), 0.5),
            ("A", date(2019, 1, 1), 0.5),
        ])
        ranked = rs.ranked_picks(oof, "score")
        # All tied at 0.5 -- tie-break is (ticker asc, event_day asc).
        assert list(zip(ranked["ticker"], ranked["event_day"])) == [
            ("A", date(2019, 1, 1)), ("A", date(2020, 1, 1)), ("Z", date(2020, 1, 1)),
        ]

    def test_deterministic_regardless_of_input_row_order(self):
        rows = [("A", date(2020, 1, 1), 0.3), ("B", date(2020, 1, 2), 0.7), ("C", date(2020, 1, 3), 0.5)]
        r1 = rs.ranked_picks(_mini_oof(rows), "score")
        r2 = rs.ranked_picks(_mini_oof(list(reversed(rows))), "score")
        pd.testing.assert_frame_equal(r1, r2)


class TestTopKKeys:
    def test_returns_correct_keys(self):
        oof = _mini_oof([("A", date(2020, 1, 1), 0.9), ("B", date(2020, 1, 2), 0.5), ("C", date(2020, 1, 3), 0.1)])
        ranked = rs.ranked_picks(oof, "score")
        top2 = rs.top_k_keys(ranked, 2)
        assert top2 == {("A", date(2020, 1, 1)), ("B", date(2020, 1, 2))}


# ---------------------------------------------------------------------------
# 3. jaccard_index / retention_rate
# ---------------------------------------------------------------------------
class TestOverlapMetrics:
    def test_jaccard_identical_sets_is_one(self):
        s = {("A", date(2020, 1, 1)), ("B", date(2020, 1, 2))}
        assert rs.jaccard_index(s, s) == pytest.approx(1.0)

    def test_jaccard_disjoint_sets_is_zero(self):
        a = {("A", date(2020, 1, 1))}
        b = {("B", date(2020, 1, 2))}
        assert rs.jaccard_index(a, b) == pytest.approx(0.0)

    def test_jaccard_partial_overlap(self):
        a = {("A", date(2020, 1, 1)), ("B", date(2020, 1, 2))}
        b = {("B", date(2020, 1, 2)), ("C", date(2020, 1, 3))}
        # intersection=1, union=3
        assert rs.jaccard_index(a, b) == pytest.approx(1 / 3)

    def test_jaccard_both_empty_is_nan(self):
        assert rs.jaccard_index(set(), set()) != rs.jaccard_index(set(), set())  # NaN != NaN

    def test_retention_rate_full_and_partial(self):
        baseline = {("A", date(2020, 1, 1)), ("B", date(2020, 1, 2)), ("C", date(2020, 1, 3))}
        same = set(baseline)
        assert rs.retention_rate(baseline, same) == pytest.approx(1.0)

        partial = {("A", date(2020, 1, 1)), ("Z", date(2099, 1, 1))}
        assert rs.retention_rate(baseline, partial) == pytest.approx(1 / 3)

    def test_retention_rate_empty_baseline_is_nan(self):
        assert rs.retention_rate(set(), {("A", date(2020, 1, 1))}) != rs.retention_rate(set(), set())


# ---------------------------------------------------------------------------
# 4. score_gap_metrics
# ---------------------------------------------------------------------------
class TestScoreGapMetrics:
    def test_exact_gap_and_iqr_values(self):
        # 20 evenly spaced scores from 1.0 down to 0.05 (rank 1 highest).
        scores = np.linspace(1.0, 0.05, 20)
        oof = _mini_oof([(f"T{i}", date(2020, 1, 1) + timedelta(days=i), float(s)) for i, s in enumerate(scores)])
        ranked = rs.ranked_picks(oof, "score")
        out = rs.score_gap_metrics(ranked, "score")
        expected_gap = float(scores[4] - scores[5])  # rank5 - rank6, 0-indexed 4,5
        assert out["gap_5_6"] == pytest.approx(expected_gap)
        expected_gap_10_11 = float(scores[9] - scores[10])
        assert out["gap_10_11"] == pytest.approx(expected_gap_10_11)
        q75, q25 = np.percentile(scores, [75, 25])
        assert out["score_iqr"] == pytest.approx(q75 - q25)
        assert out["gap_5_6_norm"] == pytest.approx(expected_gap / (q75 - q25))

    def test_too_few_rows_returns_nan_gap(self):
        oof = _mini_oof([("A", date(2020, 1, 1), 0.9), ("B", date(2020, 1, 2), 0.1)])
        ranked = rs.ranked_picks(oof, "score")
        out = rs.score_gap_metrics(ranked, "score")
        assert out["gap_5_6"] != out["gap_5_6"]  # NaN
        assert out["gap_10_11"] != out["gap_10_11"]

    def test_near_tied_ranks_give_small_normalized_gap(self):
        # rank 10/11 are near-identical scores -- gap_10_11_norm should be
        # near zero, the direct signature the working hypothesis predicts
        # for a near-tie band.
        scores = list(np.linspace(1.0, 0.5, 9)) + [0.100, 0.0999, 0.0998] + list(np.linspace(0.05, 0.01, 8))
        oof = _mini_oof([(f"T{i}", date(2020, 1, 1) + timedelta(days=i), float(s)) for i, s in enumerate(scores)])
        ranked = rs.ranked_picks(oof, "score")
        out = rs.score_gap_metrics(ranked, "score")
        assert abs(out["gap_10_11_norm"]) < 0.01


# ---------------------------------------------------------------------------
# 5. rank_churn
# ---------------------------------------------------------------------------
class TestRankChurn:
    def _ranked(self, rows):
        return rs.ranked_picks(_mini_oof(rows), "score")

    def test_identical_ranking_has_zero_churn(self):
        rows = [(f"T{i}", date(2020, 1, 1) + timedelta(days=i), 1.0 - i * 0.01) for i in range(15)]
        base = self._ranked(rows)
        churn = rs.rank_churn(base, base, top_n=10)
        assert churn["rank_churn_mean_abs"] == pytest.approx(0.0)
        assert churn["rank_churn_n_missing"] == 0
        assert churn["rank_churn_n_baseline"] == 10

    def test_missing_pick_gets_penalty_rank(self):
        base_rows = [(f"T{i}", date(2020, 1, 1) + timedelta(days=i), 1.0 - i * 0.01) for i in range(10)]
        base = self._ranked(base_rows)
        # Replicate pool of 5 rows that does NOT contain T0 (baseline rank 1) at all.
        repl_rows = [(f"T{i}", date(2020, 1, 1) + timedelta(days=i), 1.0 - i * 0.01) for i in range(1, 6)]
        repl = self._ranked(repl_rows)
        churn = rs.rank_churn(base, repl, top_n=10)
        assert churn["rank_churn_n_missing"] == 5  # T5..T9 also absent from the 5-row replicate
        # T0's penalty rank = len(repl) + 1 = 6, baseline rank 1 -> shift 5.
        assert churn["rank_churn_mean_abs"] > 0

    def test_swapped_adjacent_ranks_gives_shift_of_two_each(self):
        rows = [(f"T{i}", date(2020, 1, 1) + timedelta(days=i), 1.0 - i * 0.1) for i in range(5)]
        base = self._ranked(rows)
        swapped_rows = list(rows)
        swapped_rows[0], swapped_rows[1] = swapped_rows[1], swapped_rows[0]
        # Force T0 and T1 to swap rank by swapping their scores instead of rows.
        s0, s1 = rows[0][2], rows[1][2]
        swapped = [rows[0][:2] + (s1,), rows[1][:2] + (s0,)] + rows[2:]
        repl = self._ranked(swapped)
        churn = rs.rank_churn(base, repl, top_n=5)
        assert churn["rank_churn_n_missing"] == 0
        assert churn["rank_churn_mean_abs"] == pytest.approx(0.4)  # (1+1+0+0+0)/5


# ---------------------------------------------------------------------------
# 6. vol_matched_excess
# ---------------------------------------------------------------------------
class TestVolMatchedExcess:
    def _pool(self, n=60, seed=0):
        rng = np.random.default_rng(seed)
        return pd.DataFrame({
            "ticker": [f"T{i}" for i in range(n)],
            "event_day": [date(2020, 1, 1) + timedelta(days=i) for i in range(n)],
            "score": rng.random(n),
            "label": rng.standard_normal(n) * 0.1,
        })

    def test_returns_finite_excess_on_a_reasonably_sized_pool(self):
        df = self._pool(n=60)
        vol_map = pd.Series(
            np.random.default_rng(1).uniform(0.1, 1.0, len(df)),
            index=pd.MultiIndex.from_arrays([df["ticker"], df["event_day"]]),
        )
        out = rs.vol_matched_excess(
            df, vol_map, score_col="score", label_col="label",
            top_frac=0.2, n_vol_buckets=3, n_boot=200, rng=np.random.default_rng(2),
        )
        assert out["volmatch_excess"] == out["volmatch_excess"]  # finite, not NaN
        assert out["volmatch_n"] > 0

    def test_too_few_rows_returns_nan(self):
        df = self._pool(n=5)
        vol_map = pd.Series(
            [0.5] * 5, index=pd.MultiIndex.from_arrays([df["ticker"], df["event_day"]]),
        )
        out = rs.vol_matched_excess(
            df, vol_map, score_col="score", label_col="label",
            top_frac=0.2, n_vol_buckets=10, n_boot=50, rng=np.random.default_rng(0),
        )
        assert out["volmatch_excess"] != out["volmatch_excess"]  # NaN

    def test_unmatched_vol_rows_are_excluded(self):
        df = self._pool(n=60)
        # vol_map only covers half the pool -- the other half must be dropped, not crash.
        half = df.iloc[:30]
        vol_map = pd.Series(
            np.random.default_rng(1).uniform(0.1, 1.0, len(half)),
            index=pd.MultiIndex.from_arrays([half["ticker"], half["event_day"]]),
        )
        out = rs.vol_matched_excess(
            df, vol_map, score_col="score", label_col="label",
            top_frac=0.2, n_vol_buckets=3, n_boot=100, rng=np.random.default_rng(2),
        )
        # Only the 30 vol-matched rows survive the inner join; top_frac=0.2
        # of 30 (6) is below the max(..., 30) floor vol_matched_excess uses
        # (mirroring volmatch.py's own floor), so every matched row is kept.
        assert out["volmatch_n"] == 30


# ---------------------------------------------------------------------------
# 7. _build_summary: PASS/FAIL verdict logic
# ---------------------------------------------------------------------------
class TestBuildSummary:
    def _replicate_df(self, retention_k5, volmatch_values):
        n = len(volmatch_values)
        return pd.DataFrame({
            "retention_k5": [retention_k5] * n,
            "jaccard_k5": [retention_k5 / 2] * n,
            "volmatch_excess": volmatch_values,
            "gap_5_6_norm": [0.1] * n,
            "gap_10_11_norm": [0.01] * n,
        })

    def _baseline(self, volmatch_excess=0.05):
        return (
            {"gap_5_6_norm": 0.12, "gap_10_11_norm": 0.01},
            {"volmatch_excess": volmatch_excess},
        )

    def test_pass_when_retention_high_and_volmatch_stable(self):
        rep_df = self._replicate_df(retention_k5=0.9, volmatch_values=[0.048, 0.050, 0.049])
        base_gap, base_vol = self._baseline(volmatch_excess=0.050)
        summary = rs._build_summary(
            rep_df, baseline_gap=base_gap, baseline_vol=base_vol, top_ks=(5,),
            retention_bar=0.70, volmatch_threshold_pp=0.02,
        )
        row = summary.iloc[0]
        assert row["retention_pass"]
        assert row["volmatch_pass"]
        assert row["verdict"] == "PASS"

    def test_fail_when_retention_below_bar(self):
        rep_df = self._replicate_df(retention_k5=0.3, volmatch_values=[0.049, 0.050, 0.051])
        base_gap, base_vol = self._baseline(volmatch_excess=0.050)
        summary = rs._build_summary(
            rep_df, baseline_gap=base_gap, baseline_vol=base_vol, top_ks=(5,),
            retention_bar=0.70, volmatch_threshold_pp=0.02,
        )
        row = summary.iloc[0]
        assert not row["retention_pass"]
        assert row["verdict"] == "FAIL"

    def test_fail_when_volmatch_excess_moves_too_much(self):
        rep_df = self._replicate_df(retention_k5=0.9, volmatch_values=[0.01, 0.09, 0.05])
        base_gap, base_vol = self._baseline(volmatch_excess=0.050)
        summary = rs._build_summary(
            rep_df, baseline_gap=base_gap, baseline_vol=base_vol, top_ks=(5,),
            retention_bar=0.70, volmatch_threshold_pp=0.02,
        )
        row = summary.iloc[0]
        assert row["retention_pass"]
        assert not row["volmatch_pass"]
        assert row["verdict"] == "FAIL"

    def test_volmatch_gate_shared_identically_across_every_k(self):
        rep_df = pd.DataFrame({
            "retention_k5": [0.9, 0.9], "jaccard_k5": [0.8, 0.8],
            "retention_k10": [0.2, 0.2], "jaccard_k10": [0.1, 0.1],
            "volmatch_excess": [0.05, 0.051],
            "gap_5_6_norm": [0.1, 0.1], "gap_10_11_norm": [0.01, 0.01],
        })
        base_gap, base_vol = self._baseline(volmatch_excess=0.050)
        summary = rs._build_summary(
            rep_df, baseline_gap=base_gap, baseline_vol=base_vol, top_ks=(5, 10),
            retention_bar=0.70, volmatch_threshold_pp=0.02,
        )
        assert summary.set_index("k").loc[5, "volmatch_pass"] == summary.set_index("k").loc[10, "volmatch_pass"]
        assert summary.set_index("k").loc[5, "verdict"] == "PASS"
        assert summary.set_index("k").loc[10, "verdict"] == "FAIL"  # retention_k10 too low


# ---------------------------------------------------------------------------
# 8. CLI argument parsing
# ---------------------------------------------------------------------------
class TestArgParsing:
    def test_defaults(self):
        args = rs.build_arg_parser().parse_args([])
        assert args.research == rs.DEFAULT_RESEARCH_PATH
        assert args.oof == rs.DEFAULT_OOF_PATH
        assert args.n_replicates == rs.N_REPLICATES_DEFAULT
        assert args.mode == rs.MODE_DEFAULT
        assert args.retention_bar == pytest.approx(rs.RETENTION_BAR_DEFAULT)
        assert args.volmatch_threshold_pp == pytest.approx(rs.VOLMATCH_THRESHOLD_PP_DEFAULT)

    def test_mode_rejects_unknown_choice(self):
        with pytest.raises(SystemExit):
            rs.build_arg_parser().parse_args(["--mode", "shuffle_everything"])

    def test_mode_accepts_bootstrap(self):
        args = rs.build_arg_parser().parse_args(["--mode", "bootstrap"])
        assert args.mode == "bootstrap"

    def test_top_ks_parsing(self):
        assert rs._parse_top_ks("5,10,15,25") == (5, 10, 15, 25)
        assert rs._parse_top_ks(" 5 , 10 ") == (5, 10)

    def test_top_ks_parse_error_raises_systemexit(self):
        with pytest.raises(SystemExit):
            rs._parse_top_ks("five,ten")

    def test_n_replicates_and_seed_parse(self):
        args = rs.build_arg_parser().parse_args(["--n-replicates", "5", "--seed", "42"])
        assert args.n_replicates == 5
        assert args.seed == 42


# ---------------------------------------------------------------------------
# 9. Atomic write / persistence
# ---------------------------------------------------------------------------
class TestSaveStabilityResults:
    def test_leaves_no_tmp_file_and_names_include_mode(self, tmp_path):
        rep_df = pd.DataFrame({"replicate_id": [0, 1], "retention_k5": [0.5, 0.6]})
        summary_df = pd.DataFrame({"k": [5], "verdict": ["PASS"]})
        rep_path, sum_path = rs.save_stability_results(
            rep_df, summary_df, out_dir=str(tmp_path), mode="drop1pct", tag="unittest",
        )
        assert os.path.exists(rep_path)
        assert os.path.exists(sum_path)
        assert not os.path.exists(rep_path + ".tmp")
        assert not os.path.exists(sum_path + ".tmp")
        assert "stability_replicates_drop1pct_unittest_" in os.path.basename(rep_path)
        assert "stability_summary_drop1pct_unittest_" in os.path.basename(sum_path)

        back = pd.read_parquet(rep_path)
        assert len(back) == 2
        back_summary = pd.read_csv(sum_path)
        assert back_summary["verdict"].iloc[0] == "PASS"

    def test_interrupted_parquet_write_leaves_no_final_file(self, tmp_path, monkeypatch):
        rep_df = pd.DataFrame({"replicate_id": [0]})

        def boom(self, path, *a, **kw):
            with open(path, "wb") as fh:
                fh.write(b"partial")
            raise RuntimeError("simulated interruption")

        monkeypatch.setattr(pd.DataFrame, "to_parquet", boom)
        with pytest.raises(RuntimeError):
            rs._atomic_write_parquet(rep_df, str(tmp_path / "out.parquet"))
        assert not (tmp_path / "out.parquet").exists()


# ---------------------------------------------------------------------------
# 10. End-to-end: run_stability_gate on a small synthetic dataset (real,
#     fast fits -- no network, no real research_data/*.parquet files).
# ---------------------------------------------------------------------------
class TestRunStabilityGateEndToEnd:
    def test_produces_expected_shape_and_verdicts(self, tmp_path):
        df = make_synthetic_research_df(n=500, seed=5)
        baseline_oof = make_baseline_oof(df, n_folds=3, min_fold_train_rows=15)

        research_path = str(tmp_path / "research.parquet")
        oof_path = str(tmp_path / "oof.parquet")
        df.to_parquet(research_path, index=False)
        baseline_oof.to_parquet(oof_path, index=False)

        rep_df, summary_df = rs.run_stability_gate(
            research_path, oof_path,
            n_replicates=2, mode="drop1pct", seed=0,
            n_folds=3, min_fold_train_rows=15,
            top_ks=(3, 5), n_boot=100,
        )

        assert len(rep_df) == 2
        for col in ("retention_k3", "retention_k5", "jaccard_k3", "jaccard_k5",
                    "gap_5_6", "gap_10_11", "rank_churn_mean_abs", "volmatch_excess"):
            assert col in rep_df.columns

        assert set(summary_df["k"]) == {3, 5}
        assert set(summary_df["verdict"]).issubset({"PASS", "FAIL"})

    def test_bootstrap_mode_runs_end_to_end(self, tmp_path):
        df = make_synthetic_research_df(n=500, seed=6)
        baseline_oof = make_baseline_oof(df, n_folds=3, min_fold_train_rows=15)

        research_path = str(tmp_path / "research.parquet")
        oof_path = str(tmp_path / "oof.parquet")
        df.to_parquet(research_path, index=False)
        baseline_oof.to_parquet(oof_path, index=False)

        rep_df, summary_df = rs.run_stability_gate(
            research_path, oof_path,
            n_replicates=2, mode="bootstrap", seed=1,
            n_folds=3, min_fold_train_rows=15,
            top_ks=(5,), n_boot=100,
        )
        assert len(rep_df) == 2
        assert rep_df["mode"].eq("bootstrap").all()

    def test_raises_when_every_replicate_fails(self, tmp_path, monkeypatch):
        df = make_synthetic_research_df(n=500, seed=7)
        baseline_oof = make_baseline_oof(df, n_folds=3, min_fold_train_rows=15)

        research_path = str(tmp_path / "research.parquet")
        oof_path = str(tmp_path / "oof.parquet")
        df.to_parquet(research_path, index=False)
        baseline_oof.to_parquet(oof_path, index=False)

        def explode(*a, **kw):
            raise RuntimeError("simulated refit failure")

        monkeypatch.setattr(rs, "refit_oof_scores", explode)
        with pytest.raises(RuntimeError, match="every replicate failed"):
            rs.run_stability_gate(
                research_path, oof_path, n_replicates=2, mode="drop1pct",
                n_folds=3, min_fold_train_rows=15, top_ks=(5,), n_boot=50,
            )
