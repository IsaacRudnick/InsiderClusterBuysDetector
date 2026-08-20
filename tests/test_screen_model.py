"""Tests for research.screen_model and the banding it feeds, plus the
walk-forward discipline in tools.score_lab that selected it.

Runnable standalone via `python -m pytest tests/test_screen_model.py -q` from
the repo root. No network and no real research dataset: every test builds a
synthetic frame matching backtest.research's schema, derived from the real
constants rather than a hardcoded guess.

The two tests that matter most here:

  test_year_folds_never_leak_a_label_window
      proves no training row's 21-day label window can reach into, or even
      touch, the test year it is being used to predict. If this is wrong,
      every number in RESEARCH_NOTES.md's 2026-08-20 section is worthless.

  test_screen_target_cohort_centre_is_local_to_month_and_bucket
      proves the relative target subtracts the right cohort's centre. A
      target that leaked its cohort across months would let the model learn
      which months were good, which is not a stock-picking skill and would
      not survive contact with a live run.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from research import live_score as ls  # noqa: E402
from research import model as rm  # noqa: E402
from research import screen_model as sm  # noqa: E402
from tools import score_lab as sl  # noqa: E402

FAST_LGBM = dict(n_estimators=25, num_leaves=7, min_child_samples=5)


def make_df(n: int = 900, seed: int = 0) -> pd.DataFrame:
    """A synthetic dataset with the real column names and a planted signal.

    One feature (`x_mom_63_skip5`) genuinely predicts the forward return so
    that a fit has something to find; everything else is noise. The point is
    not to measure an edge here -- it is that the machinery runs end to end
    on a frame shaped exactly like the real one.
    """
    rng = np.random.default_rng(seed)
    days = pd.date_range("2019-01-02", periods=n, freq="B")
    df = pd.DataFrame(
        {
            "ticker": [f"T{i % 40:02d}" for i in range(n)],
            "issuer_cik": [f"{i % 40:010d}" for i in range(n)],
            "event_day": days,
            "entry_day": days,
            "entry_idx": np.arange(n),
            "entry_open": rng.uniform(2.0, 60.0, n),
        }
    )
    for c in rm.FEATURE_COLS:
        df[c] = rng.normal(size=n)
    df["x_vol_63_ann"] = rng.uniform(0.2, 1.8, n)
    signal = 0.05 * df["x_mom_63_skip5"]
    for h in (10, 21, 63, 126, 252):
        df[f"spy_{h}"] = rng.normal(0.005, 0.02, n)
        df[f"fwd_{h}"] = df[f"spy_{h}"] + signal + rng.normal(0, 0.05, n)
        df[f"adj_{h}"] = df[f"fwd_{h}"] - df[f"spy_{h}"]
        df[f"delisted_{h}"] = False
    return df


# ---------------------------------------------------------------------------
# Fold discipline
# ---------------------------------------------------------------------------

def test_year_folds_never_leak_a_label_window():
    """No training row's label window may reach the test year, embargo included.

    Checked on entry_idx, the shared trading-day index, so the assertion is
    exact in trading days rather than an approximation via calendar dates --
    which is precisely where an off-by-a-week leak would hide.
    """
    df = make_df(1200)
    folds = sl.make_year_folds(df, horizon=21, embargo=21)
    assert folds, "expected at least one usable fold"
    entry = df["entry_idx"].to_numpy()
    for fold in folds:
        assert len(np.intersect1d(fold.train_idx, fold.test_idx)) == 0
        test_start = entry[fold.test_idx].min()
        latest_close = (entry[fold.train_idx] + 21).max()
        assert latest_close < test_start, (
            f"fold {fold.year}: a training label window closes at "
            f"{latest_close}, at or after the test block start {test_start}"
        )
        assert (entry[fold.train_idx] + 21 + 21).max() < test_start


def test_year_folds_skip_a_year_with_too_little_history():
    """The first year has nothing in front of it and must not be fit at all."""
    df = make_df(400)
    folds = sl.make_year_folds(df, horizon=21, embargo=21, min_train_rows=300)
    assert all(f.year > df["event_day"].dt.year.min() for f in folds)


def test_build_oof_leaves_untested_rows_missing_not_zero():
    """A row no fold tested must come back NaN.

    Filling it with 0.0 would silently place it mid-ranking, which is how
    unscored candidates previously stayed eligible for selection.
    """
    df = make_df(900)
    cand = sl.Candidate(
        name="t", target=lambda d: d["adj_21"], objective="quantile",
        alpha=0.35, lgbm=FAST_LGBM,
    )
    out = sl.build_oof(df, cand, horizon=21, seed=0)
    assert out.isna().any(), "expected the earliest year to go unscored"
    assert out.notna().any(), "expected later years to be scored"


# ---------------------------------------------------------------------------
# The training target
# ---------------------------------------------------------------------------

def test_screen_target_cohort_centre_is_local_to_month_and_bucket():
    """Each cohort's target must be centred on that cohort alone.

    Constructed so the answer is known: within any one month-and-volatility
    cohort the demeaned target has a median of zero. If the centre were
    pooled across months, a month with an unusual return level would leave a
    non-zero median behind.
    """
    df = make_df(1000)
    t = sm.screen_target(df)
    month = pd.to_datetime(df["event_day"]).dt.to_period("M").astype(str)
    ok = t.notna()
    med = t[ok].groupby(month[ok]).median()
    # Each month is the union of its volatility buckets, each individually
    # centred, so the per-month median is at most a bucket-boundary artifact
    # away from zero -- never a whole month's return level.
    assert med.abs().max() < 0.02


def test_screen_target_is_nan_where_the_label_is():
    df = make_df(400)
    df.loc[df.index[:50], "fwd_21"] = np.nan
    t = sm.screen_target(df)
    assert t.iloc[:50].isna().all()


def test_screen_target_survives_a_month_too_small_to_bucket():
    """A month with fewer rows than volatility buckets must not raise.

    One bucket is the correct degenerate answer for a thin month; an
    exception here would take down a whole production fit over a quiet
    December.
    """
    df = make_df(60)
    df.loc[:, "x_vol_63_ann"] = 0.5  # no distinct quantiles at all
    t = sm.screen_target(df)
    assert t.notna().any()


# ---------------------------------------------------------------------------
# The ensemble and its bundle
# ---------------------------------------------------------------------------

def test_screen_feature_cols_excludes_everything_never_computable_live():
    """The fit must not use an input production will never have.

    Derived from live_score's own categorisation rather than a copy of it, so
    this test also fails if those two ever drift apart.
    """
    cols = sm.screen_feature_cols()
    never_live = set(ls.OWNER_HISTORY_FEATURE_COLS) | set(ls.SALE_FEATURE_COLS)
    assert never_live, "expected live_score to declare never-live columns"
    assert not (set(cols) & never_live)
    assert set(cols) | never_live == set(rm.FEATURE_COLS)


def test_ensemble_scores_are_percentile_ranks_in_unit_interval():
    df = make_df(900)
    ens = sm.fit_screen_ensemble(df, n_members=3, lgbm_params=FAST_LGBM)
    X = df[ens.feature_cols].astype(float)
    s = ens.score_rows(X)
    assert len(s) == len(df)
    assert np.isfinite(s).all()
    assert s.min() >= 0.0 and s.max() <= 1.0


def test_ensemble_ordering_is_stable_across_member_count():
    """Adding members must refine the ordering, not reinvent it.

    An ensemble whose ranking depended on how many members it happened to
    have would be the seed lottery again in a different costume.
    """
    df = make_df(900)
    a = sm.fit_screen_ensemble(df, n_members=3, lgbm_params=FAST_LGBM)
    b = sm.fit_screen_ensemble(df, n_members=6, lgbm_params=FAST_LGBM)
    X = df[a.feature_cols].astype(float)
    corr = pd.Series(a.score_rows(X)).corr(
        pd.Series(b.score_rows(X)), method="spearman"
    )
    assert corr > 0.9, f"ordering moved too much between sizes: {corr:.3f}"


def test_bundle_round_trips_through_disk(tmp_path):
    df = make_df(900)
    ens = sm.fit_screen_ensemble(df, n_members=2, lgbm_params=FAST_LGBM)
    bundle = sm.build_screen_bundle(ens, df, source_path="synthetic")
    path = tmp_path / "screen.joblib"
    rm.save_production_bundle(bundle, str(path))
    back = rm.load_production_bundle(str(path))

    assert sm.is_screen_bundle(back)
    assert back.feature_cols == bundle.feature_cols
    X = df[bundle.feature_cols].astype(float).head(20)
    np.testing.assert_allclose(
        back.model.score_rows(X), bundle.model.score_rows(X)
    )


def test_a_single_classifier_bundle_is_not_mistaken_for_a_screen_bundle():
    """The branch in live_score keys off this, and a wrong answer would band
    one score's percentiles with the other score's measured cuts."""
    class OldModel:
        def predict_proba(self, X):
            return np.zeros((len(X), 2))

    bundle = rm.ProductionBundle(
        model=OldModel(), feature_cols=["x_n_insiders"],
        training_scores=np.array([0.0, 1.0]), provenance={}, config={},
    )
    assert not sm.is_screen_bundle(bundle)


def test_fit_refuses_a_dataset_too_small_to_be_a_production_model():
    with pytest.raises(ValueError, match="refusing"):
        sm.fit_screen_ensemble(make_df(120), n_members=1, lgbm_params=FAST_LGBM)


# ---------------------------------------------------------------------------
# Banding
# ---------------------------------------------------------------------------

def _avail(frac_missing: float = 0.0):
    class A:
        pass

    a = A()
    a.frac_missing = frac_missing
    return a


@pytest.mark.parametrize(
    "pct,expected",
    [
        (0.0, ls.Verdict.ELEVATED_RISK),
        (29.9, ls.Verdict.ELEVATED_RISK),
        (30.0, ls.Verdict.MIDDLE),
        (69.9, ls.Verdict.MIDDLE),
        (70.0, ls.Verdict.TOP_BAND),
        (89.9, ls.Verdict.TOP_BAND),
        (90.0, ls.Verdict.ABOVE_BAND),
        (100.0, ls.Verdict.ABOVE_BAND),
    ],
)
def test_screen_bands_cut_where_they_were_measured(pct, expected):
    assert ls.band_verdict(pct, _avail(), screen=True) is expected


def test_the_very_top_is_not_the_top_band():
    """The finding this whole change rests on: a higher percentile is not a
    better candidate. 95 must not band better than 80."""
    assert ls.band_verdict(95.0, _avail(), screen=True) is ls.Verdict.ABOVE_BAND
    assert ls.band_verdict(80.0, _avail(), screen=True) is ls.Verdict.TOP_BAND


def test_old_bundle_still_bands_the_old_way():
    """An older single-classifier bundle must keep its own two states. The new
    cuts were measured on a different ordering and mean nothing applied to it.
    """
    assert ls.band_verdict(95.0, _avail(), screen=False) is ls.Verdict.TOP_DECILE
    assert ls.band_verdict(50.0, _avail(), screen=False) is ls.Verdict.NO_EDGE


@pytest.mark.parametrize("screen", [True, False])
def test_too_much_missing_beats_any_band(screen):
    assert ls.band_verdict(99.0, _avail(0.9), screen=screen) is ls.Verdict.UNAVAILABLE
    assert ls.band_verdict(float("nan"), _avail(), screen=screen) is ls.Verdict.UNAVAILABLE


# ---------------------------------------------------------------------------
# Availability is judged against the model that was actually fit
# ---------------------------------------------------------------------------

def test_availability_restricted_to_a_models_own_columns():
    """A model must not be marked short of inputs it never asked for.

    The screening ensemble deliberately drops the 12 never-computable-live
    columns. Counting those as missing would report 47/59 available for a
    model that had every input it wanted, and would push a genuinely thin
    cluster nearer the `unavailable` backstop for the wrong reason.
    """
    full = ls.FeatureAvailability(
        computed=["x_n_insiders", "x_vol_63_ann"],
        missing=["x_sell_n_cluster", "x_mom_63_skip5"],
        missing_reasons={
            "x_sell_n_cluster": "never live",
            "x_mom_63_skip5": "no prices",
        },
    )
    assert full.n_total == 4
    assert full.frac_missing == 0.5

    scoped = full.restricted_to(["x_n_insiders", "x_vol_63_ann", "x_mom_63_skip5"])
    assert scoped.n_total == 3
    assert scoped.missing == ["x_mom_63_skip5"]
    assert "x_sell_n_cluster" not in scoped.missing_reasons
    assert scoped.frac_missing == pytest.approx(1 / 3)


def test_a_full_screen_row_reports_nothing_missing():
    """With prices and issuer history supplied, the screening feature set is
    fully computable -- which is the point of restricting the fit to it."""
    cols = set(sm.screen_feature_cols())
    never_live = set(ls.OWNER_HISTORY_FEATURE_COLS) | set(ls.SALE_FEATURE_COLS)
    full = ls.FeatureAvailability(
        computed=sorted(cols),
        missing=sorted(never_live),
        missing_reasons={c: "never live" for c in never_live},
    )
    scoped = full.restricted_to(sm.screen_feature_cols())
    assert scoped.missing == []
    assert scoped.frac_missing == 0.0
