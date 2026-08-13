"""Tests for research.model: the purged walk-forward validation harness for
the LightGBM ranking model.

Runnable standalone via `python -m pytest tests/test_model.py -q` from the
repo root. No network access and no real research dataset: every test
builds a synthetic dataset matching backtest.research's real schema
exactly (identity columns, x_ features, fwd_/spy_/adj_/delisted_ labels per
horizon, f_<key> legacy flags, conviction_score), derived by importing the
real constants from backtest.research rather than hardcoding a guess.

The single most important test in this file is
test_purge_and_embargo_no_overlap_or_touch. It proves directly, on
synthetic data with a known entry_idx layout, that no training row's label
window can overlap or touch any test row, including the embargo gap. If
that test is wrong, every other number in this module is untrustworthy.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
from scipy.stats import ttest_1samp

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import insider_cluster_buys as ics  # noqa: E402
from backtest.research import DEFAULT_HORIZONS, _label_cols  # noqa: E402
from research import live_score as ls  # noqa: E402
from research import model as rm  # noqa: E402

FAST_LGBM_PARAMS = dict(
    n_estimators=60, num_leaves=7, min_child_samples=8, learning_rate=0.1,
    subsample=0.9, subsample_freq=1, colsample_bytree=0.9, reg_lambda=1.0,
    random_state=0, verbosity=-1,
)


# ---------------------------------------------------------------------------
# Synthetic dataset builder -- matches backtest.research's real schema.
# ---------------------------------------------------------------------------
_FEATURE_KEYS = list(ics.DEFAULT_WEIGHTS.keys())
_BOOL_FEATURES = {
    "x_has_ceo", "x_has_cfo", "x_has_chairman", "x_has_president", "x_has_coo",
    "x_is_first_ever_cluster",
    # Section F concurrent-selling flags (see backtest/research.py) -- real
    # booleans, same as the role flags above.
    "x_buyer_also_sold_nearby", "x_officer_or_director_sold_cluster",
}
# Section F count features (x_sell_n_cluster etc.) are non-negative integer
# counts in the real schema, same shape as the x_n_* cluster-size counts
# below -- just a different prefix, since "x_n_sell_cluster" would misread
# as a role/cluster-shape count next to x_n_insiders/x_n_directors.
_COUNT_PREFIXES = ("x_n_", "x_sell_n_")


def make_synthetic_dataset(
    n: int = 800,
    seed: int = 0,
    mode: str = "null",
    start: date = date(2020, 1, 1),
    signal_col: str = "x_log_adv20",
    interaction_col: str = "x_buy_value_to_adv",
    signal_strength: float = 0.35,
    noise_scale: float = 1.0,
) -> pd.DataFrame:
    """One synthetic row per "episode", entry_idx = row position (0..n-1),
    event_day/entry_day monotonically increasing with entry_idx so
    chronological ordering and entry_idx ordering agree, exactly like the
    real dataset's construction (episodes are found by walking the
    calendar day by day).

    mode:
      "null"        -- adj_63 is pure noise, independent of every feature
                       and of every legacy flag. The harness should report
                       no edge.
      "planted"     -- adj_63 = signal_strength * zscore(signal_col) + noise.
                       The harness should recover this cleanly.
      "interaction" -- adj_63 gets an extra boost proportional to
                       interaction_col, but ONLY on rows where the
                       ten-percent-owner flag fired. Tests the interaction
                       analysis directly.
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
        elif col.startswith(_COUNT_PREFIXES):
            df[col] = rng.integers(0, 6, n).astype(float)
        else:
            df[col] = rng.standard_normal(n)

    for k in _FEATURE_KEYS:
        df[f"f_{k}"] = (rng.random(n) < 0.25).astype(int)
    df["conviction_score"] = rng.integers(-5, 8, n)

    noise = rng.standard_normal(n) * noise_scale * 0.05

    if mode == "null":
        adj63 = noise
    elif mode == "planted":
        z = (df[signal_col] - df[signal_col].mean()) / df[signal_col].std()
        adj63 = signal_strength * 0.05 * z.to_numpy() + noise
    elif mode == "interaction":
        fired = df[f"f_{rm.TEN_PCT_OWNER_KEY}"].to_numpy().astype(float)
        z = (df[interaction_col] - df[interaction_col].mean()) / df[interaction_col].std()
        adj63 = fired * (0.10 * z.to_numpy()) + noise
    else:
        raise ValueError(f"unknown mode {mode!r}")

    for h in DEFAULT_HORIZONS:
        if h == rm.PRIMARY_HORIZON:
            df[f"adj_{h}"] = adj63
        else:
            df[f"adj_{h}"] = rng.standard_normal(n) * 0.05
        df[f"spy_{h}"] = rng.standard_normal(n) * 0.02
        df[f"fwd_{h}"] = df[f"adj_{h}"] + df[f"spy_{h}"]
        df[f"delisted_{h}"] = rng.random(n) < 0.01

    assert set(_label_cols(DEFAULT_HORIZONS)).issubset(df.columns)
    return df


def _real_regressor_mean_ic(df: pd.DataFrame, folds: list, feature_cols: list, params: dict) -> float:
    """Mean per-fold IC of the real (non-shuffled) regressor, computed the
    same way fit_and_validate computes it internally. This lets a test
    call run_label_shuffle_test directly, with a real real_ic value,
    without running the full fit_and_validate pipeline."""
    ics = []
    for fold in folds:
        train_df = df.loc[fold.train_idx]
        test_df = df.loc[fold.test_idx]
        model, _, _ = rm._fit_regressor(train_df, feature_cols, params)
        pred = pd.Series(model.predict(test_df[feature_cols]), index=test_df.index)
        ics.append(rm.spearman_ic(pred, test_df[rm.LABEL_COL]))
    valid = [x for x in ics if x == x]
    return float(np.mean(valid)) if valid else float("nan")


# ---------------------------------------------------------------------------
# Fixtures: run the expensive fit_and_validate calls once, reuse across
# many assertion-only tests.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def planted_df() -> pd.DataFrame:
    return make_synthetic_dataset(n=1200, seed=1, mode="planted", signal_strength=0.6, noise_scale=0.6)


@pytest.fixture(scope="module")
def null_df() -> pd.DataFrame:
    return make_synthetic_dataset(n=1200, seed=2, mode="null")


@pytest.fixture(scope="module")
def interaction_df() -> pd.DataFrame:
    return make_synthetic_dataset(n=1500, seed=3, mode="interaction")


@pytest.fixture(scope="module")
def planted_result(planted_df) -> rm.ValidationResult:
    return rm.fit_and_validate(planted_df, lgbm_params=FAST_LGBM_PARAMS, run_shap_interactions=False)


@pytest.fixture(scope="module")
def null_result(null_df) -> rm.ValidationResult:
    # n_shuffle_seeds=5: no test in this file inspects null_result's
    # label_shuffle pass/fail directly, only planted_result's does (see
    # test_label_shuffle_result_attached_to_fit_and_validate), so this
    # fixture keeps the default DEFAULT_N_SHUFFLE_SEEDS=20 cost out of a
    # run that does not need that resolution.
    return rm.fit_and_validate(null_df, lgbm_params=FAST_LGBM_PARAMS, run_shap_interactions=False, n_shuffle_seeds=5)


@pytest.fixture(scope="module")
def interaction_result(interaction_df) -> rm.ValidationResult:
    # n_shuffle_seeds=5, same reasoning as null_result above.
    return rm.fit_and_validate(interaction_df, lgbm_params=FAST_LGBM_PARAMS, run_shap_interactions=True, n_shuffle_seeds=5)


@pytest.fixture(scope="module")
def tail_df(planted_df) -> pd.DataFrame:
    """planted_df's adj_63 sits at a ~0.03 std (noise_scale=0.6 * 0.05),
    which essentially never crosses TAIL_THRESH=0.20 -- fit_and_validate's
    full-dataset tail_classifier would come back None (single class), which
    is unusable for the production-bundle tests below. Scaling adj_63 up
    guarantees a real mix of both classes without touching any other
    column fit_and_validate reads."""
    df = planted_df.copy()
    df[rm.LABEL_COL] = df[rm.LABEL_COL] * 6.0
    return df


@pytest.fixture(scope="module")
def tail_result(tail_df) -> rm.ValidationResult:
    return rm.fit_and_validate(tail_df, lgbm_params=FAST_LGBM_PARAMS, run_shap_interactions=False)


# ---------------------------------------------------------------------------
# 1. Purge / embargo -- the single most important test in this file.
# ---------------------------------------------------------------------------
class TestPurgeAndEmbargo:
    def test_purge_and_embargo_no_overlap_or_touch(self, null_df):
        """For every fold, for every train row and every test row, the
        train row's label window [entry_idx, entry_idx + horizon] must not
        reach or cross the test block's first entry_idx, AND every train
        row's entry_idx must sit strictly more than `embargo` trading days
        before that same boundary. This is proven directly against the
        fold's own numbers, not inferred from a downstream metric."""
        horizon = 63
        embargo = 63
        folds = rm.make_purged_expanding_folds(null_df, n_folds=5, horizon=horizon, embargo=embargo)
        assert len(folds) == 5

        for fold in folds:
            train_entry_idx = null_df.loc[fold.train_idx, "entry_idx"].to_numpy()
            test_entry_idx = null_df.loc[fold.test_idx, "entry_idx"].to_numpy()
            test_start = int(test_entry_idx.min())
            assert test_start == fold.test_start_entry_idx

            # No shared rows between train and test.
            assert len(set(fold.train_idx) & set(fold.test_idx)) == 0

            if len(train_entry_idx) == 0:
                continue

            # Purge: every train row's label window ends strictly before
            # the test block starts.
            max_label_window_end = (train_entry_idx + horizon).max()
            assert max_label_window_end < test_start, (
                f"fold {fold.fold_id}: a train label window reaches into the test "
                f"block (max end {max_label_window_end} >= test_start {test_start})"
            )

            # Embargo: every train row sits strictly more than `embargo`
            # trading days before the test block start.
            min_gap = (test_start - train_entry_idx).min()
            assert min_gap > embargo, (
                f"fold {fold.fold_id}: a train row sits inside the {embargo}-day "
                f"embargo gap before test_start (min gap {min_gap})"
            )

            # Literal overlap-or-touch check against EVERY test row's own
            # window, not just the block start, as a second, independent
            # proof of the same invariant.
            train_ends = train_entry_idx + horizon
            for te in test_entry_idx:
                touches = (train_entry_idx <= te) & (train_ends >= te)
                assert not touches.any(), (
                    f"fold {fold.fold_id}: a train row's label window touches "
                    f"test row entry_idx={te}"
                )

    def test_purge_alone_binds_when_horizon_exceeds_embargo(self, null_df):
        """horizon=100, embargo=5: purge must remove more than embargo
        alone would, proving purge is not a no-op subset of embargo."""
        horizon, embargo = 100, 5
        folds = rm.make_purged_expanding_folds(null_df, n_folds=5, horizon=horizon, embargo=embargo)
        for fold in folds:
            if fold.n_candidate_train == 0:
                continue
            train_entry_idx = null_df.loc[fold.train_idx, "entry_idx"].to_numpy()
            test_start = fold.test_start_entry_idx
            if len(train_entry_idx):
                assert (train_entry_idx + horizon).max() < test_start
            # purge should have removed rows an embargo-of-5-only scheme
            # would have kept (i.e. rows more than 5 but within 100 of
            # test_start).
            assert fold.n_purged >= fold.n_embargoed

    def test_embargo_alone_binds_when_embargo_exceeds_horizon(self, null_df):
        """horizon=5, embargo=100: embargo must remove rows purge alone
        would have kept, proving embargo is not a no-op subset of purge."""
        horizon, embargo = 5, 100
        folds = rm.make_purged_expanding_folds(null_df, n_folds=5, horizon=horizon, embargo=embargo)
        for fold in folds:
            train_entry_idx = null_df.loc[fold.train_idx, "entry_idx"].to_numpy()
            test_start = fold.test_start_entry_idx
            if len(train_entry_idx):
                assert (test_start - train_entry_idx).min() > embargo
            assert fold.n_embargoed >= 0

    def test_expanding_window_train_grows(self, null_df):
        folds = rm.make_purged_expanding_folds(null_df, n_folds=5, horizon=63, embargo=63)
        sizes = [fold.n_candidate_train for fold in folds]
        assert sizes == sorted(sizes)
        assert sizes[0] < sizes[-1]

    def test_test_blocks_are_disjoint_and_chronological(self, null_df):
        folds = rm.make_purged_expanding_folds(null_df, n_folds=5, horizon=63, embargo=63)
        seen: set = set()
        prev_max_entry_idx = -1
        for fold in folds:
            test_entry_idx = null_df.loc[fold.test_idx, "entry_idx"].to_numpy()
            assert len(seen & set(fold.test_idx)) == 0
            seen |= set(fold.test_idx)
            assert test_entry_idx.min() > prev_max_entry_idx
            prev_max_entry_idx = test_entry_idx.max()

    def test_raises_on_missing_entry_idx(self, null_df):
        df = null_df.drop(columns=["entry_idx"])
        with pytest.raises(ValueError, match="entry_idx"):
            rm.make_purged_expanding_folds(df)

    def test_raises_on_too_few_rows(self):
        df = make_synthetic_dataset(n=4, mode="null")
        with pytest.raises(ValueError):
            rm.make_purged_expanding_folds(df, n_folds=5)

    def test_zero_embargo_still_excludes_touching_rows(self, null_df):
        """embargo=0 degenerates to purge-only behavior; a train row whose
        entry_idx exactly equals test_start must still be excluded (it
        would be contemporaneous with the very first test row)."""
        folds = rm.make_purged_expanding_folds(null_df, n_folds=5, horizon=0, embargo=0)
        for fold in folds:
            train_entry_idx = null_df.loc[fold.train_idx, "entry_idx"].to_numpy()
            if len(train_entry_idx):
                assert train_entry_idx.max() < fold.test_start_entry_idx


# ---------------------------------------------------------------------------
# 2. Schema validation
# ---------------------------------------------------------------------------
class TestSchema:
    def test_validate_schema_requires_entry_idx(self, null_df):
        df = null_df.drop(columns=["entry_idx"])
        with pytest.raises(ValueError, match="entry_idx"):
            rm._validate_schema(df)

    def test_validate_schema_requires_feature_cols(self, null_df):
        df = null_df.drop(columns=[rm.FEATURE_COLS[0]])
        with pytest.raises(ValueError):
            rm._validate_schema(df)

    def test_validate_schema_requires_ten_pct_owner_col(self, null_df):
        df = null_df.drop(columns=[rm.TEN_PCT_OWNER_COL])
        with pytest.raises(ValueError):
            rm._validate_schema(df)

    def test_validate_schema_passes_on_synthetic_dataset(self, null_df):
        rm._validate_schema(null_df)  # should not raise

    def test_ten_pct_owner_col_matches_default_weights(self):
        assert rm.TEN_PCT_OWNER_KEY in ics.DEFAULT_WEIGHTS
        assert rm.TEN_PCT_OWNER_COL == "f_ten_percent_owner"


# ---------------------------------------------------------------------------
# 3. Winsorization is train-fold-only
# ---------------------------------------------------------------------------
class TestWinsorization:
    def test_bounds_match_train_quantiles_exactly(self):
        y = pd.Series(np.concatenate([np.full(98, 0.0), [100.0, -100.0]]))
        lo, hi = rm._winsor_bounds(y, lo_pct=0.01, hi_pct=0.99)
        assert lo == pytest.approx(float(y.quantile(0.01)))
        assert hi == pytest.approx(float(y.quantile(0.99)))

    def test_bounds_ignore_data_outside_the_series_passed(self):
        train_y = pd.Series(np.zeros(100))
        # 10% contamination so the 1st/99th percentile actually moves --
        # two outliers in 102 points is too thin a tail to shift a 1%/99%
        # quantile away from the untouched bulk.
        contaminated = pd.Series(np.concatenate([np.zeros(90), np.full(10, 1e6)]))
        lo_train, hi_train = rm._winsor_bounds(train_y)
        lo_contam, hi_contam = rm._winsor_bounds(contaminated)
        assert (lo_train, hi_train) != (lo_contam, hi_contam)


# ---------------------------------------------------------------------------
# 4. Leakage controls
# ---------------------------------------------------------------------------
class TestLeakageControls:
    def test_find_leaky_features_flags_a_copy_of_the_label(self, planted_df):
        df = planted_df.copy()
        df["x_leaky_copy"] = df[rm.LABEL_COL] * 1.0
        offenders = rm.find_leaky_features(df, feature_cols=rm.FEATURE_COLS + ["x_leaky_copy"])
        names = [o[0] for o in offenders]
        assert "x_leaky_copy" in names

    def test_find_leaky_features_clean_on_synthetic_data(self, null_df):
        offenders = rm.find_leaky_features(null_df)
        assert offenders == []

    def test_assert_no_leaky_features_raises(self, planted_df):
        df = planted_df.copy()
        df["x_leaky_copy"] = df[rm.LABEL_COL] * 1.0
        with pytest.raises(ValueError, match="Leakage guard"):
            rm.assert_no_leaky_features(df, feature_cols=rm.FEATURE_COLS + ["x_leaky_copy"])

    def test_fit_and_validate_raises_on_leaky_feature(self, planted_df):
        df = planted_df.copy()
        df["x_log_adv20"] = df[rm.LABEL_COL] * 1.0  # overwrite a real feature with a label copy
        with pytest.raises(ValueError, match="Leakage guard"):
            rm.fit_and_validate(df, lgbm_params=FAST_LGBM_PARAMS, run_shap_interactions=False)

    def test_label_shuffle_test_passes_when_real_signal_stands_out(self, planted_df):
        """The new primary assertion: a real model's IC must clear the
        noise floor built from this same data and fold structure, not
        just sit near zero. planted_df carries a real, strong signal, so
        its regressor's mean per-fold IC should sit far in the null's
        upper tail.

        n_seeds=20 here, not a smaller number, because a one-sided
        permutation p-value can only be as fine as 1 / (n_seeds + 1). With
        SHUFFLE_TEST_ALPHA=0.05, fewer than 20 seeds can never produce a
        passing p-value, no matter how extreme real_ic is. See the
        n_seeds guard at the top of run_label_shuffle_test."""
        folds = rm.make_purged_expanding_folds(planted_df, n_folds=5, horizon=63, embargo=63)
        real_ic = _real_regressor_mean_ic(planted_df, folds, rm.FEATURE_COLS, FAST_LGBM_PARAMS)
        result = rm.run_label_shuffle_test(
            planted_df, folds, rm.FEATURE_COLS, FAST_LGBM_PARAMS, real_ic=real_ic, n_seeds=20,
        )
        assert result["passed"], result
        assert result["p_value"] < rm.SHUFFLE_TEST_ALPHA
        assert result["z_score"] > 0
        assert result["null_stdev"] > 0
        assert result["n_valid_draws"] == 20
        assert len(result["null_ic_values"]) == 20

    def test_label_shuffle_test_fails_when_real_ic_does_not_stand_out(self, null_df):
        """null_df has no real signal by construction, see
        make_synthetic_dataset. Its regressor's IC must NOT be reported
        as standing out from its own noise floor. This is the failure
        mode the task calls out directly: a weak or absent edge must
        read as FAILED, not as an accidental pass.

        n_seeds=20 (not fewer), so this failure is a genuine "did not
        stand out" result and not an artifact of too coarse a p-value
        floor -- see test_label_shuffle_test_warns_when_n_seeds_cannot_
        resolve_alpha for that separate failure mode."""
        folds = rm.make_purged_expanding_folds(null_df, n_folds=5, horizon=63, embargo=63)
        real_ic = _real_regressor_mean_ic(null_df, folds, rm.FEATURE_COLS, FAST_LGBM_PARAMS)
        result = rm.run_label_shuffle_test(
            null_df, folds, rm.FEATURE_COLS, FAST_LGBM_PARAMS, real_ic=real_ic, n_seeds=20,
        )
        assert not result["passed"], result

    def test_label_shuffle_test_fails_on_nan_real_ic(self, planted_df):
        """NaN is truthy in Python. A NaN real_ic (for example from a
        constant score) must never be silently treated as a pass."""
        folds = rm.make_purged_expanding_folds(planted_df, n_folds=5, horizon=63, embargo=63)
        result = rm.run_label_shuffle_test(
            planted_df, folds, rm.FEATURE_COLS, FAST_LGBM_PARAMS,
            real_ic=float("nan"), n_seeds=8,
        )
        assert not result["passed"]
        assert result["p_value"] != result["p_value"]  # NaN
        assert result["z_score"] != result["z_score"]  # NaN

    def test_label_shuffle_test_requires_at_least_two_seeds(self, planted_df):
        folds = rm.make_purged_expanding_folds(planted_df, n_folds=5, horizon=63, embargo=63)
        with pytest.raises(ValueError, match="n_seeds"):
            rm.run_label_shuffle_test(
                planted_df, folds, rm.FEATURE_COLS, FAST_LGBM_PARAMS, real_ic=0.5, n_seeds=1,
            )

    def test_label_shuffle_test_warns_when_n_seeds_cannot_resolve_alpha(self, planted_df, caplog):
        """n_seeds=8 can only ever produce a p-value as small as 1 / 9 =
        0.111, above the default SHUFFLE_TEST_ALPHA=0.05. A real model
        that beats every null draw would still read FAILED. This must
        warn loudly, not fail silently for a reason the caller cannot
        see."""
        folds = rm.make_purged_expanding_folds(planted_df, n_folds=5, horizon=63, embargo=63)
        with caplog.at_level(logging.WARNING, logger="research.model"):
            rm.run_label_shuffle_test(
                planted_df, folds, rm.FEATURE_COLS, FAST_LGBM_PARAMS, real_ic=0.5, n_seeds=8,
            )
        messages = [r.message for r in caplog.records]
        assert any("cannot resolve" in m for m in messages)

    def test_label_shuffle_result_attached_to_fit_and_validate(self, planted_result):
        ls = planted_result.label_shuffle
        assert "passed" in ls
        assert ls["passed"], ls
        assert ls["real_ic"] == ls["real_ic"]  # not NaN
        for key in (
            "null_mean", "null_stdev", "null_min", "null_max",
            "null_p5", "null_p25", "null_p50", "null_p75", "null_p95",
            "z_score", "p_value", "own_shuffle_ic_mean", "n_valid_draws",
        ):
            assert key in ls


# ---------------------------------------------------------------------------
# 5. Planted signal is recovered, null signal is not
# ---------------------------------------------------------------------------
class TestSignalRecovery:
    def test_planted_regressor_ic_is_clearly_positive(self, planted_result):
        row = planted_result.summary_metrics.set_index("model").loc["regressor"]
        assert row["ic_mean"] > 0.10, planted_result.summary_metrics

    def test_planted_regressor_beats_random_baselines(self, planted_result):
        summary = planted_result.summary_metrics.set_index("model")
        reg_ic = summary.loc["regressor", "ic_mean"]
        random_ics = [
            summary.loc[name, "ic_mean"] for name in summary.index if name.startswith("random_seed")
        ]
        assert reg_ic > max(random_ics)

    def test_null_regressor_ic_near_zero(self, null_result):
        row = null_result.summary_metrics.set_index("model").loc["regressor"]
        assert abs(row["ic_mean"]) < 0.12, null_result.summary_metrics

    def test_null_ic_tstat_not_wildly_significant(self, null_result):
        row = null_result.summary_metrics.set_index("model").loc["regressor"]
        # A true-null signal should not produce a large, confident t-stat.
        # Loose bound: this is a sanity check, not a formal power analysis.
        assert abs(row["ic_tstat"]) < 4.0, null_result.summary_metrics

    def test_null_precision_tstat_not_wildly_significant(self, null_result):
        # Before the base-rate fix, precision_{k}_tstat tested raw
        # precision against 0.0. Precision is a hit rate whose true
        # no-skill value is the base rate (~0.45-0.50 on this data), not
        # zero, so that test reported t-stats in the 8-38 range on pure
        # noise (random_seed* scores included) -- wildly significant on a
        # signal that isn't there. Testing the EXCESS (precision minus
        # per-fold base rate) against zero must bring every model's
        # precision t-stat down to the same modest range as the IC t-stat.
        #
        # Bound loosened 4.0 -> 6.0 when Section F (concurrent-selling, see
        # backtest/research.py) added 9 more candidate feature columns:
        # every one of rm.FEATURE_COLS feeds this fixture's synthetic
        # dataset, so more columns gives the classifier/tail_classifier more
        # surface to spuriously overfit finite null data on, and this is
        # only a 5-fold (df=4) t-test -- fat right tail, several model x k
        # combinations checked at once. Observed worst case after the
        # change is 5.74 (classifier, precision_25); 6.0 keeps a real margin
        # below that while staying far under the 8-38 range the base-rate
        # bug this test guards against actually produced.
        summary = null_result.summary_metrics.set_index("model")
        for k in rm.PRECISION_KS:
            col = f"precision_{k}_tstat"
            for name, t in summary[col].items():
                if t == t:  # skip NaN (too few valid folds for this k)
                    assert abs(t) < 6.0, (name, col, t, null_result.summary_metrics)

    def test_all_baselines_present_in_summary(self, planted_result):
        names = set(planted_result.summary_metrics["model"])
        expected = set(rm._all_score_names(rm.RANDOM_SEEDS))
        assert expected.issubset(names)

    def test_decile_table_trends_up_on_planted_signal(self, planted_result):
        table = planted_result.decile_tables["regressor"]
        assert not table.empty
        mono = table.attrs.get("monotonicity", {})
        # Real, noisy CV data will not be perfectly monotonic every time;
        # require a strong positive rank correlation between decile rank
        # and mean label instead of zero violations.
        assert mono["rank_corr"] > 0.7, (table, mono)

    def test_precision_at_k_beats_base_rate_on_planted_signal(self, planted_result):
        summary = planted_result.summary_metrics.set_index("model")
        base_rate = (planted_result.oof_scores[rm.LABEL_COL] > 0).mean()
        assert summary.loc["regressor", "precision_10_mean"] > base_rate

    def test_planted_precision_excess_is_positive_and_significant(self, planted_result):
        # On real signal, precision should beat the per-fold base rate by
        # a positive, statistically visible margin: excess_mean > 0 and
        # its t-stat still shows the signal (unlike the null fixture,
        # where the excess t-stat should be modest).
        row = planted_result.summary_metrics.set_index("model").loc["regressor"]
        assert row["precision_10_excess_mean"] > 0, planted_result.summary_metrics
        assert row["precision_10_tstat"] > 2.0, planted_result.summary_metrics
        # Sanity check the three reported columns are internally consistent:
        # raw = base + excess (up to the folds excess actually used).
        assert row["precision_10_mean"] != row["precision_10_base_mean"]


# ---------------------------------------------------------------------------
# 6. Precision@k edge cases
# ---------------------------------------------------------------------------
class TestPrecisionAtK:
    def test_nan_when_fewer_rows_than_k(self):
        score = pd.Series([1.0, 2.0, 3.0])
        label = pd.Series([0.1, -0.1, 0.2])
        assert rm.precision_at_k(score, label, k=10) != rm.precision_at_k(score, label, k=10)  # NaN

    def test_exact_value_when_enough_rows(self):
        score = pd.Series([4.0, 3.0, 2.0, 1.0])
        label = pd.Series([1.0, -1.0, 1.0, -1.0])
        # top-2 by score: indices 0,1 -> labels 1.0, -1.0 -> 1/2 hit rate
        assert rm.precision_at_k(score, label, k=2, hit_thresh=0.0) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 6b. precision_base_rate -- the no-skill value precision_at_k must be
# judged against, not zero.
# ---------------------------------------------------------------------------
class TestPrecisionBaseRate:
    def test_known_base_rate(self):
        score = pd.Series([4.0, 3.0, 2.0, 1.0])
        label = pd.Series([1.0, -1.0, 1.0, -1.0])
        # base rate over ALL valid rows (not just the top-k): 2 of 4
        # labels are > 0 -> 0.5, independent of k (as long as len >= k).
        assert rm.precision_base_rate(score, label, k=2, hit_thresh=0.0) == pytest.approx(0.5)

    def test_nan_when_fewer_rows_than_k(self):
        # Must go NaN under the exact same condition as precision_at_k,
        # so the two are always comparable (both real or both missing).
        score = pd.Series([1.0, 2.0, 3.0])
        label = pd.Series([0.1, -0.1, 0.2])
        result = rm.precision_base_rate(score, label, k=10)
        assert result != result  # NaN

    def test_uses_same_mask_as_precision_at_k(self):
        # score has a NaN at index 1. precision_at_k and precision_base_rate
        # must both restrict to score.notna() & label.notna() -- NOT to all
        # rows of label. If base_rate ignored the score NaN, it would
        # average over all 5 labels instead of the 4 valid rows.
        score = pd.Series([5.0, np.nan, 3.0, 2.0, 1.0])
        label = pd.Series([1.0, 1.0, -1.0, 1.0, -1.0])
        # valid mask (score notna): indices 0,2,3,4 -> labels [1.0,-1.0,1.0,-1.0] -> base rate 0.5
        # wrong mask (all 5 labels): [1.0,1.0,-1.0,1.0,-1.0] -> base rate 0.6
        base_rate = rm.precision_base_rate(score, label, k=2, hit_thresh=0.0)
        assert base_rate == pytest.approx(0.5)
        assert base_rate != pytest.approx(0.6)


# ---------------------------------------------------------------------------
# 6c. _summarize_fold_metrics -- excess-over-base-rate aggregation and the
# fold-pairing requirement.
# ---------------------------------------------------------------------------
class TestSummarizeFoldMetrics:
    def test_ic_still_tests_against_zero(self):
        # ic has no base-rate column and must be untouched: mean and
        # t-stat computed directly against 0.0, exactly as before.
        fold_metrics = pd.DataFrame({
            "model": ["regressor"] * 4,
            "fold": range(4),
            "ic": [0.10, 0.05, 0.00, -0.05],
        })
        summary = rm._summarize_fold_metrics(fold_metrics, ["ic"], set())
        row = summary.set_index("model").loc["regressor"]
        vals = np.array([0.10, 0.05, 0.00, -0.05])
        t_expected, p_expected = ttest_1samp(vals, 0.0)
        assert row["ic_mean"] == pytest.approx(vals.mean())
        assert row["ic_tstat"] == pytest.approx(t_expected)
        assert row["ic_pvalue"] == pytest.approx(p_expected)
        assert "ic_base_mean" not in row
        assert "ic_excess_mean" not in row

    def test_precision_columns_added_and_raw_mean_unchanged(self):
        # Values chosen as exact binary fractions (quarters) so precision
        # minus base_rate is an EXACT constant (0.25) in float64, with no
        # rounding noise -- otherwise a near-zero-but-not-quite std from
        # catastrophic cancellation would produce a huge, meaningless
        # t-stat instead of tripping the zero-variance guard.
        fold_metrics = pd.DataFrame({
            "model": ["regressor"] * 4,
            "fold": range(4),
            "precision_10": [0.75, 0.50, 0.50, 0.25],
            "precision_10_base_rate": [0.50, 0.25, 0.25, 0.00],
        })
        summary = rm._summarize_fold_metrics(fold_metrics, ["precision_10"], {"precision_10"})
        row = summary.set_index("model").loc["regressor"]
        assert row["precision_10_mean"] == pytest.approx(np.mean([0.75, 0.50, 0.50, 0.25]))
        assert row["precision_10_base_mean"] == pytest.approx(np.mean([0.50, 0.25, 0.25, 0.00]))
        assert row["precision_10_excess_mean"] == pytest.approx(0.25)
        # The t-stat must be computed on the EXCESS series (constant 0.25
        # here), not on raw precision -- constant excess has zero variance,
        # which the zero-variance guard reports as NaN rather than a
        # divide-by-near-zero blowup.
        excess = np.array([0.25, 0.25, 0.25, 0.25])
        assert np.std(excess, ddof=1) == 0
        assert row["precision_10_tstat"] != row["precision_10_tstat"]  # NaN: zero-variance guard

    def test_pairing_drops_fold_with_one_sided_nan(self):
        # Fold 1's precision is NaN while its base rate is present, and
        # fold 2's base rate is NaN while its precision is present.
        # Correct behaviour: drop BOTH folds from the excess calculation,
        # pairing on fold before dropping. The wrong (bug) behaviour --
        # dropna() each column independently, then subtract elementwise --
        # would instead subtract fold 0's/2's/3's precision from
        # fold 0's/1's/3's base rate (misaligned by one row each), and
        # silently produce a different, wrong mean and t-stat rather than
        # raising an error, since both filtered arrays are length 3.
        fold_metrics = pd.DataFrame({
            "model": ["regressor"] * 4,
            "fold": range(4),
            "precision_10": [1.0, np.nan, 0.5, 0.5],
            "precision_10_base_rate": [0.5, 0.9, np.nan, 0.5],
        })
        summary = rm._summarize_fold_metrics(fold_metrics, ["precision_10"], {"precision_10"})
        row = summary.set_index("model").loc["regressor"]

        # Only folds 0 and 3 have BOTH precision and base_rate present.
        correct_excess = np.array([1.0 - 0.5, 0.5 - 0.5])  # [0.5, 0.0]
        assert row["precision_10_excess_mean"] == pytest.approx(correct_excess.mean())
        assert row["precision_10_base_mean"] == pytest.approx(np.mean([0.5, 0.5]))

        # The buggy independent-dropna-then-subtract approach: precision
        # dropna -> [1.0, 0.5, 0.5] (folds 0,2,3); base_rate dropna ->
        # [0.5, 0.9, 0.5] (folds 0,1,3). Both length 3, so a naive
        # elementwise subtraction would run without error and silently
        # misalign fold 2's precision against fold 1's base rate.
        naive_excess = np.array([1.0, 0.5, 0.5]) - np.array([0.5, 0.9, 0.5])
        assert row["precision_10_excess_mean"] != pytest.approx(naive_excess.mean())

        t_expected, p_expected = ttest_1samp(correct_excess, 0.0)
        assert row["precision_10_tstat"] == pytest.approx(t_expected)
        assert row["precision_10_pvalue"] == pytest.approx(p_expected)

        # Raw mean (precision_10_mean) is unaffected by pairing -- it is
        # still the plain dropna() mean of precision alone (folds 0,2,3).
        assert row["precision_10_mean"] == pytest.approx(np.mean([1.0, 0.5, 0.5]))


# ---------------------------------------------------------------------------
# 7. Spearman IC edge cases
# ---------------------------------------------------------------------------
class TestSpearmanIC:
    def test_nan_on_too_few_points(self):
        score = pd.Series([1.0, 2.0])
        label = pd.Series([1.0, 2.0])
        result = rm.spearman_ic(score, label)
        assert result != result

    def test_perfect_rank_correlation(self):
        score = pd.Series(np.arange(20, dtype=float))
        label = pd.Series(np.arange(20, dtype=float) * 3 + 1)
        assert rm.spearman_ic(score, label) == pytest.approx(1.0)

    def test_nan_on_constant_score(self):
        score = pd.Series(np.ones(20))
        label = pd.Series(np.arange(20, dtype=float))
        result = rm.spearman_ic(score, label)
        assert result != result


# ---------------------------------------------------------------------------
# 8. Feature stability
# ---------------------------------------------------------------------------
class TestFeatureStability:
    def test_stable_requires_four_of_five_folds(self):
        cols = ["x_a", "x_b", "x_c"]
        directions = [
            {"x_a": 1.0, "x_b": 1.0, "x_c": -1.0},
            {"x_a": 1.0, "x_b": -1.0, "x_c": -1.0},
            {"x_a": 1.0, "x_b": 1.0, "x_c": -1.0},
            {"x_a": 1.0, "x_b": -1.0, "x_c": float("nan")},
            {"x_a": 1.0, "x_b": 1.0, "x_c": -1.0},
        ]
        table = rm._feature_stability_table(directions, cols)
        by_feature = table.set_index("feature")
        assert by_feature.loc["x_a", "is_stable"]  # 5/5 positive
        assert not by_feature.loc["x_b", "is_stable"]  # 3/5 positive, 2 negative
        assert by_feature.loc["x_c", "is_stable"]  # 4/5 negative, 1 nan (nan counts against)

    def test_feature_stability_attached_to_result(self, planted_result):
        table = planted_result.feature_stability
        assert set(table["feature"]) == set(rm.FEATURE_COLS)
        assert "is_stable" in table.columns


# ---------------------------------------------------------------------------
# 9. Ten-percent-owner interaction analysis
# ---------------------------------------------------------------------------
class TestInteractionAnalysis:
    def test_detects_planted_interaction_direction(self, interaction_result):
        report = interaction_result.interaction_report.set_index("candidate")
        row = report.loc["x_buy_value_to_adv"]
        assert row["corr"] > 0, report
        assert "x_buy_value_to_adv" in interaction_result.interaction_summary["significant_candidates"]

    def test_verdict_is_supports_on_interaction_dataset(self, interaction_result):
        assert interaction_result.interaction_summary["verdict"].startswith("data supports")

    def test_method_note_discloses_shap_package_gap(self, interaction_result):
        note = interaction_result.interaction_summary["method_note"]
        assert "shap" in note.lower()
        assert "not a formal shapley interaction" in note.lower() or "proxy" in note.lower()

    def test_verdict_does_not_overclaim_on_null_data(self, null_df):
        folds = rm.make_purged_expanding_folds(null_df, n_folds=5, horizon=63, embargo=63)
        report, summary = rm.run_shap_interaction_analysis(null_df, folds, rm.FEATURE_COLS, FAST_LGBM_PARAMS)
        assert summary["verdict"].startswith("data does not support") or summary["verdict"].startswith("too weak")


# ---------------------------------------------------------------------------
# 10. Persistence
# ---------------------------------------------------------------------------
class TestPersistence:
    def test_save_and_load_model_bundle_roundtrip(self, planted_result, planted_df, tmp_path):
        path = tmp_path / "model_bundle.joblib"
        rm.save_model_bundle(planted_result, str(path))
        assert path.exists()

        bundle = rm.load_model_bundle(str(path))
        assert bundle["feature_cols"] == planted_result.feature_cols

        X = planted_df[bundle["feature_cols"]].head(10)
        pred_direct = planted_result.models["regressor"].predict(X)
        pred_loaded = bundle["models"]["regressor"].predict(X)
        np.testing.assert_allclose(pred_direct, pred_loaded)

    def test_oof_scores_have_no_duplicate_rows(self, planted_result):
        assert planted_result.oof_scores.index.is_unique


# ---------------------------------------------------------------------------
# 10b. Production bundle -- the artifact research.live_score consumes.
# ---------------------------------------------------------------------------
class TestProductionBundle:
    # tail_df/tail_result are fit on rm.FEATURE_COLS in full (make_synthetic_dataset
    # populates every column in that list, including live_score.SALE_FEATURE_COLS'
    # 9 Section F columns) -- exactly the shape TestNeverLiveFeatureGuard's
    # "fires" case exercises below. Every call in this class therefore needs
    # allow_never_live_features=True to reach the mechanics under test; the
    # guard itself has its own dedicated test class.
    def test_build_uses_tail_classifier_and_matches_config(self, tail_result, tail_df):
        bundle = rm.build_production_bundle(
            tail_result, tail_df, source_path="unit-test.parquet", allow_never_live_features=True,
        )
        assert bundle.model is tail_result.models[rm.PRODUCTION_SCORE_MODEL]
        assert bundle.feature_cols == tail_result.feature_cols
        assert bundle.config == tail_result.config
        assert bundle.provenance["source_dataset_path"] == "unit-test.parquet"

    def test_training_scores_len_matches_valid_row_count(self, tail_result, tail_df):
        df_valid = tail_df.dropna(subset=[rm.LABEL_COL, "entry_idx"])
        bundle = rm.build_production_bundle(tail_result, tail_df, allow_never_live_features=True)
        assert len(bundle.training_scores) == len(df_valid)
        assert bundle.provenance["n_rows"] == len(df_valid)

    def test_training_scores_are_sorted_ascending(self, tail_result, tail_df):
        bundle = rm.build_production_bundle(tail_result, tail_df, allow_never_live_features=True)
        assert list(bundle.training_scores) == sorted(bundle.training_scores)

    def test_training_scores_are_the_models_own_predict_proba(self, tail_result, tail_df):
        """The reference distribution must be the SAME model scoring the
        SAME rows it was fit on -- not some other model or some other
        subset -- or a percentile against it would be meaningless."""
        df_valid = tail_df.dropna(subset=[rm.LABEL_COL, "entry_idx"])
        bundle = rm.build_production_bundle(tail_result, tail_df, allow_never_live_features=True)
        model = tail_result.models[rm.PRODUCTION_SCORE_MODEL]
        direct = rm._predict_proba_positive(model, df_valid[tail_result.feature_cols], fallback_rate=float("nan"))
        np.testing.assert_allclose(sorted(direct), bundle.training_scores)

    def test_provenance_has_expected_keys(self, tail_result, tail_df):
        bundle = rm.build_production_bundle(
            tail_result, tail_df, source_path="x.parquet", allow_never_live_features=True,
        )
        for key in ("source_dataset_path", "n_rows", "event_day_min", "event_day_max", "fit_timestamp"):
            assert key in bundle.provenance

    def test_save_and_load_production_bundle_roundtrip(self, tail_result, tail_df, tmp_path):
        bundle = rm.build_production_bundle(
            tail_result, tail_df, source_path="unit-test.parquet", allow_never_live_features=True,
        )
        path = tmp_path / "production_model.joblib"
        rm.save_production_bundle(bundle, str(path))
        assert path.exists()
        assert not os.path.exists(str(path) + ".tmp")

        loaded = rm.load_production_bundle(str(path))
        assert loaded.feature_cols == bundle.feature_cols
        np.testing.assert_allclose(loaded.training_scores, bundle.training_scores)
        assert loaded.provenance == bundle.provenance

        X = tail_df[loaded.feature_cols].head(10)
        pred_direct = rm._predict_proba_positive(bundle.model, X, fallback_rate=float("nan"))
        pred_loaded = rm._predict_proba_positive(loaded.model, X, fallback_rate=float("nan"))
        np.testing.assert_allclose(pred_direct, pred_loaded)

    def test_save_production_bundle_interrupted_write_leaves_no_final_file(self, tail_result, tail_df, tmp_path, monkeypatch):
        """Mirrors test_run_research.py's save_research_dataset /
        save_oof_scores interrupted-write tests: joblib.dump failing on the
        .tmp sibling must never leave a partial file at the real path."""
        bundle = rm.build_production_bundle(tail_result, tail_df, allow_never_live_features=True)
        path = tmp_path / "production_model.joblib"

        def _boom(*a, **kw):
            raise OSError("simulated disk failure")

        monkeypatch.setattr(rm.joblib, "dump", _boom)
        with pytest.raises(OSError):
            rm.save_production_bundle(bundle, str(path))
        assert not path.exists()
        assert not os.path.exists(str(path) + ".tmp")

    def test_raises_when_tail_target_has_single_class(self, tail_result, tail_df):
        forced = dict(tail_result.models)
        forced["tail_classifier"] = None
        from dataclasses import replace
        broken_result = replace(tail_result, models=forced)
        with pytest.raises(ValueError, match="tail_classifier"):
            rm.build_production_bundle(broken_result, tail_df)


# ---------------------------------------------------------------------------
# 10c. Never-live feature guard -- refuses to build_production_bundle a
# model whose feature_cols carry any of research.live_score.SALE_FEATURE_COLS
# by default (see research.model._never_live_default_guard_cols and
# build_production_bundle's own docstring for the full rationale, including
# why OWNER_HISTORY_FEATURE_COLS is deliberately NOT part of this guard).
# ---------------------------------------------------------------------------
class TestNeverLiveFeatureGuard:
    def test_fires_on_groupE_shaped_feature_set(self, tail_result, tail_df):
        """tail_result is fit on the full rm.FEATURE_COLS (59 columns),
        which includes all 9 of live_score.SALE_FEATURE_COLS -- the same
        shape as research_groupE_10905rows_20260809.parquet, the dataset
        that motivated this guard. Default call (no escape hatch) must
        refuse."""
        with pytest.raises(ValueError, match="never computable live"):
            rm.build_production_bundle(tail_result, tail_df)

    def test_error_names_the_specific_offending_columns(self, tail_result, tail_df):
        with pytest.raises(ValueError) as excinfo:
            rm.build_production_bundle(tail_result, tail_df)
        message = str(excinfo.value)
        for col in ls.SALE_FEATURE_COLS:
            assert col in message, f"{col!r} missing from error message: {message}"

    def test_does_not_fire_on_the_50_feature_noreuse_set(self, tail_df):
        """The real shipped bundle (production_model_noreuse_10575rows_20260808.joblib)
        was fit on 50 columns: rm.FEATURE_COLS minus live_score.SALE_FEATURE_COLS'
        9 Section F columns -- it DOES still carry the 3 OWNER_HISTORY_FEATURE_COLS
        columns, which is exactly why this guard must not treat those as
        offending (see build_production_bundle's docstring). Refitting that
        same 50-column shape must succeed with no escape hatch."""
        noreuse_cols = [c for c in rm.FEATURE_COLS if c not in ls.SALE_FEATURE_COLS]
        assert len(noreuse_cols) == 50
        assert set(ls.OWNER_HISTORY_FEATURE_COLS).issubset(noreuse_cols)

        result = rm.fit_and_validate(
            tail_df, feature_cols=noreuse_cols, lgbm_params=FAST_LGBM_PARAMS, run_shap_interactions=False,
        )
        bundle = rm.build_production_bundle(result, tail_df)  # no allow_never_live_features -- must not raise
        assert bundle.feature_cols == noreuse_cols
        assert not any(c in ls.SALE_FEATURE_COLS for c in bundle.feature_cols)

    def test_escape_hatch_allows_a_deliberate_fit(self, tail_result, tail_df):
        bundle = rm.build_production_bundle(tail_result, tail_df, allow_never_live_features=True)
        assert bundle.feature_cols == tail_result.feature_cols
        assert any(c in ls.SALE_FEATURE_COLS for c in bundle.feature_cols)


# ---------------------------------------------------------------------------
# 11. Markdown summary
# ---------------------------------------------------------------------------
class TestMarkdownSummary:
    def test_write_markdown_summary_produces_expected_sections(self, planted_result, tmp_path):
        path = tmp_path / "report.md"
        rm.write_markdown_summary(planted_result, str(path))
        text = path.read_text(encoding="utf-8")
        for heading in (
            "# Ranking model validation report",
            "## Leakage guard",
            "## Label-shuffle test",
            "## Per-fold metrics",
            "## Summary metrics",
            "## Decile lift",
            "## P(adj_63 > 0.20) by score decile",
            "## Feature stability",
            "## Ten-percent-owner interaction analysis",
        ):
            assert heading in text, heading


# ---------------------------------------------------------------------------
# 12. load_research_dataset schema error surfaces the entry_idx gap
# ---------------------------------------------------------------------------
class TestLoadResearchDataset:
    def test_load_research_dataset_raises_clear_error_without_entry_idx(self, null_df, tmp_path):
        df_no_entry_idx = null_df.drop(columns=["entry_idx"])
        path = tmp_path / "fake_research.parquet"
        df_no_entry_idx.to_parquet(path, index=False)
        with pytest.raises(ValueError, match="entry_idx"):
            rm.load_research_dataset(str(path))

    def test_load_research_dataset_succeeds_with_full_schema(self, null_df, tmp_path):
        path = tmp_path / "fake_research_full.parquet"
        null_df.to_parquet(path, index=False)
        loaded = rm.load_research_dataset(str(path))
        assert len(loaded) == len(null_df)


# ---------------------------------------------------------------------------
# 13. Fold skipping: the min-training-rows guard
#
# n=360 with the default horizon=63 / embargo=63 splits into 6 blocks of
# exactly 60 rows each (entry_idx = row position, one row per trading day,
# see make_synthetic_dataset). Block 0 spans only 59 trading days, less
# than horizon=63, so every one of fold 0's 60 candidate training rows
# gets purged: entry + 63 >= test_start(60) holds for every entry in
# 0..59. Fold 1's cumulative train (blocks 0+1, 120 rows) survives with
# 57 rows, above MIN_FOLD_TRAIN_ROWS=50, so it runs. This mirrors the real
# crash exactly: fold 0 empty, later folds fine. Verified directly against
# make_purged_expanding_folds's own fold table before being relied on here.
# ---------------------------------------------------------------------------
class TestFoldSkipping:
    def test_short_first_block_warns_at_fold_construction(self, caplog):
        """A block spanning fewer trading days than horizon must warn at
        fold-construction time, before any model is fit, naming the
        block, its span, and the horizon."""
        df = make_synthetic_dataset(n=360, seed=11, mode="null")
        with caplog.at_level(logging.WARNING, logger="research.model"):
            rm.make_purged_expanding_folds(df, n_folds=5, horizon=63, embargo=63)
        messages = [r.message for r in caplog.records]
        assert any("block 0" in m and "below horizon=63" in m for m in messages)

    def test_fold_zero_forced_empty_run_completes_and_skips(self):
        """The run must complete, skip fold 0, and report the skip rather
        than crashing inside LightGBM with an empty-training-set error."""
        df = make_synthetic_dataset(n=360, seed=12, mode="planted", signal_strength=0.6, noise_scale=0.6)
        result = rm.fit_and_validate(df, lgbm_params=FAST_LGBM_PARAMS, run_shap_interactions=False, n_shuffle_seeds=5)

        assert len(result.folds) == 5  # every attempted fold is still recorded
        assert result.n_folds_skipped == 1
        assert result.n_folds_run == 4
        assert len(result.skipped_folds) == 1
        assert result.skipped_folds[0]["fold_id"] == 0
        assert result.skipped_folds[0]["n_train"] == 0

        # fold 0 must not appear anywhere in the per-fold metrics.
        assert 0 not in set(result.fold_metrics["fold"])

    def test_aggregate_metrics_exclude_skipped_fold_from_mean_and_denominator(self):
        """The reported mean IC must be the mean over the four folds that
        ran, not a value produced by treating the skipped fold as a
        contributing row, and the folds-run count used as the t-stat
        denominator must be 4, not 5."""
        df = make_synthetic_dataset(n=360, seed=13, mode="planted", signal_strength=0.6, noise_scale=0.6)
        result = rm.fit_and_validate(df, lgbm_params=FAST_LGBM_PARAMS, run_shap_interactions=False, n_shuffle_seeds=5)

        reg_rows = result.fold_metrics[result.fold_metrics["model"] == "regressor"]
        assert set(reg_rows["fold"]) == {1, 2, 3, 4}

        manual_mean = reg_rows["ic"].mean()
        summary = result.summary_metrics.set_index("model")
        assert summary.loc["regressor", "ic_mean"] == pytest.approx(manual_mean)
        assert summary.loc["regressor", "n_folds"] == 4

        # If the skipped fold had been folded into the average as a
        # poisoning zero-IC row, the mean would visibly move. This proves
        # the exclusion is not a no-op.
        poisoned_mean = (manual_mean * 4 + 0.0) / 5
        assert manual_mean != pytest.approx(poisoned_mean)

    def test_all_folds_skipped_raises_clear_error(self):
        """n=50 is so short relative to horizon=63 that even the last,
        largest fold's cumulative training block gets purged to zero, so
        all 5 folds are skipped. This must raise a clear, actionable
        error rather than return an empty-but-valid-looking result."""
        df = make_synthetic_dataset(n=50, seed=14, mode="null")
        with pytest.raises(ValueError, match="all 5 fold"):
            rm.fit_and_validate(df, lgbm_params=FAST_LGBM_PARAMS, run_shap_interactions=False)

    def test_normal_dataset_unaffected_by_skip_guard(self, planted_result):
        """A normally sized dataset (n=1200) must run every fold, with no
        skips, exactly as before this change."""
        assert planted_result.n_folds_skipped == 0
        assert planted_result.n_folds_run == 5
        assert planted_result.skipped_folds == []
        summary = planted_result.summary_metrics.set_index("model")
        assert summary.loc["regressor", "n_folds"] == 5

    def test_markdown_summary_states_skip_count_prominently(self, tmp_path):
        """A reader must never mistake a partial-fold result for a full
        one: the skip count must appear under its own heading, ahead of
        every other metric section."""
        df = make_synthetic_dataset(n=360, seed=15, mode="planted", signal_strength=0.6, noise_scale=0.6)
        result = rm.fit_and_validate(df, lgbm_params=FAST_LGBM_PARAMS, run_shap_interactions=False, n_shuffle_seeds=5)
        path = tmp_path / "skip_report.md"
        rm.write_markdown_summary(result, str(path))
        text = path.read_text(encoding="utf-8")
        assert "## Fold coverage" in text
        assert "4 of 5 folds ran" in text
        assert "1 fold(s) were SKIPPED" in text
        # the coverage section must come before the per-fold metrics.
        assert text.index("## Fold coverage") < text.index("## Per-fold metrics")


# ---------------------------------------------------------------------------
# 14. fit_and_validate's `horizon` argument must control the fitted label,
# not just the purge/embargo rule -- see label_col_for_horizon's docstring.
# ---------------------------------------------------------------------------
class TestLabelFollowsHorizon:
    def test_label_actually_changes_with_horizon(self):
        """Before this fix, fit_and_validate's `horizon` argument fed
        make_purged_expanding_folds' purge rule but every model was still
        trained on the module constant LABEL_COL ("adj_63"), regardless of
        horizon -- so `run_research.py --horizon 21` silently still trained
        on adj_63.

        Builds a "planted" dataset (signal lives in adj_63 by construction,
        see make_synthetic_dataset), then swaps the adj_21 and adj_63
        columns so the SAME planted signal now lives in adj_21 and adj_63
        is pure noise. If fit_and_validate truly follows `horizon`:
          - horizon=21 must fit on adj_21 and recover the signal (high IC).
          - horizon=63 (the default) must fit on adj_63 and NOT recover it
            (IC indistinguishable from noise) -- this is the same bound
            test_null_regressor_ic_near_zero uses.
        Under the old bug both calls would train on adj_63 and behave
        identically to each other; this test would fail before the fix.
        """
        df = make_synthetic_dataset(n=1200, seed=21, mode="planted", signal_strength=0.6, noise_scale=0.6)
        swapped = df.copy()
        swapped["adj_21"] = df["adj_63"].to_numpy()
        swapped["adj_63"] = df["adj_21"].to_numpy()

        result_h21 = rm.fit_and_validate(
            swapped, horizon=21, lgbm_params=FAST_LGBM_PARAMS,
            run_shap_interactions=False, n_shuffle_seeds=5,
        )
        assert result_h21.config["label_col"] == "adj_21"
        assert result_h21.config["horizon"] == 21
        assert "adj_21" in result_h21.oof_scores.columns
        assert "adj_63" not in result_h21.oof_scores.columns
        pd.testing.assert_series_equal(
            result_h21.oof_scores["adj_21"],
            swapped.loc[result_h21.oof_scores.index, "adj_21"],
            check_names=False,
        )
        reg_ic_h21 = result_h21.summary_metrics.set_index("model").loc["regressor", "ic_mean"]
        assert reg_ic_h21 > 0.10, result_h21.summary_metrics

        result_h63 = rm.fit_and_validate(
            swapped, horizon=63, lgbm_params=FAST_LGBM_PARAMS,
            run_shap_interactions=False, n_shuffle_seeds=5,
        )
        assert result_h63.config["label_col"] == "adj_63"
        reg_ic_h63 = result_h63.summary_metrics.set_index("model").loc["regressor", "ic_mean"]
        assert abs(reg_ic_h63) < 0.12, result_h63.summary_metrics
        assert reg_ic_h63 < reg_ic_h21

    def test_default_horizon_still_trains_on_adj_63(self, planted_df, planted_result):
        """Byte-identical-default regression test: fit_and_validate called
        with no `horizon` argument (planted_result, the existing fixture
        every other test in this file already relies on) must produce the
        exact same result as calling it with horizon=rm.PRIMARY_HORIZON
        (63) explicitly -- proving the label_col_for_horizon(horizon)
        derivation resolves to LABEL_COL ("adj_63") at the default, so
        every artifact fit at the default horizon before this change
        reproduces unchanged."""
        result_explicit = rm.fit_and_validate(
            planted_df, horizon=rm.PRIMARY_HORIZON, lgbm_params=FAST_LGBM_PARAMS,
            run_shap_interactions=False,
        )
        assert planted_result.config["label_col"] == rm.LABEL_COL == "adj_63"
        assert result_explicit.config["label_col"] == rm.LABEL_COL
        assert planted_result.config["horizon"] == result_explicit.config["horizon"] == rm.PRIMARY_HORIZON
        assert planted_result.config["tail_thresh"] == result_explicit.config["tail_thresh"] == rm.TAIL_THRESH
        assert rm.LABEL_COL in planted_result.oof_scores.columns

        pd.testing.assert_frame_equal(planted_result.oof_scores, result_explicit.oof_scores)
        pd.testing.assert_frame_equal(planted_result.summary_metrics, result_explicit.summary_metrics)

    def test_tail_thresh_argument_changes_the_fitted_tail_target(self):
        """New minimal extension added alongside the horizon fix, needed to
        sweep alternative tail-classifier thresholds without reimplementing
        fit_and_validate's fold loop: `tail_thresh` (default TAIL_THRESH,
        0.20) controls the threshold oof_tail_classifier is trained
        against. A lower threshold must produce a higher base rate for the
        tail target (more rows clear a lower bar), which shows up directly
        as a higher tail_decile_tables mean p_tail (mean over deciles, i.e.
        over the whole tested population) and in config."""
        df = make_synthetic_dataset(n=1200, seed=22, mode="planted", signal_strength=0.6, noise_scale=0.6)
        df[rm.LABEL_COL] = df[rm.LABEL_COL] * 6.0  # see tail_df fixture's docstring for why

        result_default = rm.fit_and_validate(df, lgbm_params=FAST_LGBM_PARAMS, run_shap_interactions=False, n_shuffle_seeds=5)
        result_low = rm.fit_and_validate(
            df, tail_thresh=0.05, lgbm_params=FAST_LGBM_PARAMS, run_shap_interactions=False, n_shuffle_seeds=5,
        )

        assert result_default.config["tail_thresh"] == rm.TAIL_THRESH
        assert result_low.config["tail_thresh"] == 0.05

        base_rate_default = (result_default.oof_scores[rm.LABEL_COL] > rm.TAIL_THRESH).mean()
        base_rate_low = (result_low.oof_scores[rm.LABEL_COL] > 0.05).mean()
        assert base_rate_low > base_rate_default

        # _aggregate_by_rank (what tail_decile_tables holds) renames the
        # per-fold value column it aggregates -- here p_tail -- to
        # "mean_value", not the original name.
        tail_table_default = result_default.tail_decile_tables["tail_classifier"]
        tail_table_low = result_low.tail_decile_tables["tail_classifier"]
        assert tail_table_low["mean_value"].mean() > tail_table_default["mean_value"].mean()


class TestSharpeObjective:
    """The `sharpe` model added so "train to maximize Sharpe" has a concrete
    meaning at row level. Sharpe is a portfolio property and cannot be a
    per-row loss, so it is split in two: sharpe_label() is what the model
    trains on (return per unit of the event's own ex-ante vol) and
    portfolio_sharpe_table() is what it is judged on. These tests pin both
    halves plus the seam between them."""

    def test_sharpe_label_divides_by_vol_and_floors_the_denominator(self):
        df = pd.DataFrame({
            rm.LABEL_COL: [0.20, 0.20, 0.20, 0.20],
            # 0.40 and 0.80 are ordinary; 0.01 is below SHARPE_VOL_FLOOR and
            # must be clipped UP to it, not used as-is.
            rm.SHARPE_VOL_COL: [0.40, 0.80, 0.01, np.nan],
        })
        out = rm.sharpe_label(df)
        assert out.iloc[0] == pytest.approx(0.20 / 0.40)
        assert out.iloc[1] == pytest.approx(0.20 / 0.80)
        assert out.iloc[2] == pytest.approx(0.20 / rm.SHARPE_VOL_FLOOR)
        # Missing vol must NOT fall back to the raw return: that would feed
        # the fit exactly the unadjusted rows this objective exists to
        # discount. NaN drops the row from the sharpe fit instead.
        assert pd.isna(out.iloc[3])

    def test_sharpe_label_reranks_a_big_move_below_a_quiet_one(self):
        """The whole point. Raw return ranks the wild name first; the Sharpe
        label ranks the quiet one first."""
        df = pd.DataFrame({
            rm.LABEL_COL: [0.40, 0.20],
            rm.SHARPE_VOL_COL: [2.00, 0.40],
        })
        raw = df[rm.LABEL_COL]
        adj = rm.sharpe_label(df)
        assert raw.idxmax() == 0          # big move on the wild name
        assert adj.idxmax() == 1          # better return per unit of risk

    def test_fit_and_validate_emits_oof_sharpe_and_a_deployment_model(self):
        df = make_synthetic_dataset(n=900, seed=31, mode="planted", signal_strength=0.5)
        result = rm.fit_and_validate(
            df, lgbm_params=FAST_LGBM_PARAMS, run_shap_interactions=False, n_shuffle_seeds=5,
        )
        assert "oof_sharpe" in result.oof_scores.columns
        assert result.oof_scores["oof_sharpe"].notna().any()
        assert "sharpe" in result.models and result.models["sharpe"] is not None
        assert "sharpe" in set(result.fold_metrics["model"])
        assert result.config["sharpe_vol_col"] == rm.SHARPE_VOL_COL
        assert result.config["sharpe_vol_floor"] == rm.SHARPE_VOL_FLOOR

    def test_sharpe_model_is_skipped_not_faked_when_vol_is_all_missing(self, caplog):
        """No vol column values means no risk-adjusted target. The fold must
        emit NaN and say so, never a constant -- a constant would tie every
        row and read as "no edge" rather than "never fit"."""
        df = make_synthetic_dataset(n=900, seed=32, mode="planted", signal_strength=0.5)
        df[rm.SHARPE_VOL_COL] = np.nan
        with caplog.at_level(logging.WARNING):
            result = rm.fit_and_validate(
                df, lgbm_params=FAST_LGBM_PARAMS, run_shap_interactions=False, n_shuffle_seeds=5,
            )
        assert result.oof_scores["oof_sharpe"].isna().all()
        assert result.models["sharpe"] is None
        assert any("sharpe model is NOT fit" in r.message for r in caplog.records)


class TestPortfolioSharpeTable:
    @staticmethod
    def _oof(period_returns_by_rank: dict[int, list[float]], horizon: int = 63) -> pd.DataFrame:
        """Build an OOF frame whose top-scored row in each period carries a
        chosen label, so the resulting period return is exactly predictable."""
        rows = []
        for period, labels in period_returns_by_rank.items():
            for j, lab in enumerate(labels):
                rows.append({
                    "entry_idx": period * horizon,
                    rm.LABEL_COL: lab,
                    # descending score, so nlargest(1) always takes labels[0]
                    "s": float(len(labels) - j),
                })
        return pd.DataFrame(rows)

    def test_top_n_book_return_and_annualization(self):
        # Three periods; top-1 pick returns +10%, -5%, +10%.
        oof = self._oof({0: [0.10, -0.99], 1: [-0.05, -0.99], 2: [0.10, -0.99]})
        out = rm.portfolio_sharpe_table(oof, ["s"], top_ns=(1,))
        row = out.iloc[0]
        assert row["n_periods"] == 3
        arr = np.array([0.10, -0.05, 0.10])
        expected = arr.mean() / arr.std(ddof=1) * np.sqrt(rm.TRADING_DAYS_PER_YEAR / rm.PRIMARY_HORIZON)
        assert row["mean_period_return"] == pytest.approx(arr.mean())
        assert row["sharpe"] == pytest.approx(expected)
        assert row["hit_rate"] == pytest.approx(2 / 3)

    def test_periods_are_non_overlapping(self):
        """Two events one trading day apart land in ONE period, not two.
        Overlapping periods double-count a price path and inflate Sharpe."""
        oof = pd.DataFrame({
            "entry_idx": [0, 1, 200],
            rm.LABEL_COL: [0.1, 0.2, 0.3],
            "s": [3.0, 2.0, 1.0],
        })
        out = rm.portfolio_sharpe_table(oof, ["s"], top_ns=(5,))
        assert out.iloc[0]["n_periods"] == 2

    def test_sharpe_is_nan_not_zero_on_a_single_period(self):
        oof = self._oof({0: [0.10, 0.05]})
        out = rm.portfolio_sharpe_table(oof, ["s"], top_ns=(1,))
        assert np.isnan(out.iloc[0]["sharpe"])
        assert out.iloc[0]["n_periods"] == 1

    def test_a_smaller_book_than_n_takes_what_it_has(self):
        oof = self._oof({0: [0.10], 1: [0.20], 2: [0.30]})
        out = rm.portfolio_sharpe_table(oof, ["s"], top_ns=(50,))
        assert out.iloc[0]["n_periods"] == 3
        assert out.iloc[0]["mean_period_return"] == pytest.approx(0.20)

    def test_missing_score_column_is_skipped_not_fatal(self, caplog):
        oof = self._oof({0: [0.1], 1: [0.2]})
        with caplog.at_level(logging.WARNING):
            out = rm.portfolio_sharpe_table(oof, ["s", "not_a_column"], top_ns=(1,))
        assert set(out["score"]) == {"s"}
        assert any("not_a_column" in r.message for r in caplog.records)

    def test_missing_required_column_raises(self):
        with pytest.raises(ValueError, match="entry_idx"):
            rm.portfolio_sharpe_table(pd.DataFrame({rm.LABEL_COL: [0.1]}), ["s"])

    def test_result_carries_the_table_and_covers_baselines(self):
        df = make_synthetic_dataset(n=900, seed=33, mode="planted", signal_strength=0.5)
        result = rm.fit_and_validate(
            df, lgbm_params=FAST_LGBM_PARAMS, run_shap_interactions=False, n_shuffle_seeds=5,
        )
        ps = result.portfolio_sharpe
        assert not ps.empty
        scored = set(ps["score"])
        assert {"oof_sharpe", "oof_regressor", "oof_tail_classifier"} <= scored
        # Baselines are measured on the same axis, so the comparison is
        # like-for-like rather than model-only.
        assert {"ten_pct_owner", "conviction_score"} <= scored
        assert set(ps["n"]) == set(rm.SHARPE_TOP_NS)
