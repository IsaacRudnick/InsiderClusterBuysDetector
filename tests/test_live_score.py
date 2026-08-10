"""Tests for research.live_score: the backend that scores a freshly-
detected live cluster (insider_cluster_buys._build_cluster's shape) with
the trained ranking model.

Runnable standalone via `python -m pytest tests/test_live_score.py -q` from
the repo root. No network access: prices come from a FakePriceUniverse
double (same shape as tests/test_research.py's own fixture, trimmed to
what live_score.py actually calls), and the "historical reference frame"
issuer_history is a small in-memory DataFrame built by hand -- never
price_cache/, ipo_cache/, or clusters_history/.

The single most important test in this file is
test_section_abc_parity_with_research_build_event_row: it proves that
build_live_feature_row's Section A/B/C formulas (duplicated from
backtest.research._build_event_row because that function has no
standalone entry point -- see live_score.py's own module docstring)
produce IDENTICAL values to the real research pipeline on the same
cluster, one feature at a time. If that test is wrong, every score this
module produces is scored on features that silently disagree with what
the model was trained on.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

import insider_cluster_buys as ics
from backtest import research as research_mod
from research import live_score as ls
from research import model as rm


# ---------------------------------------------------------------------------
# Fixtures / builders
# ---------------------------------------------------------------------------
class FakePriceUniverse:
    """Minimal PriceUniverse-shaped double covering exactly what
    research.live_score calls: open, last_close_on_or_before,
    median_dollar_volume, and the three flat lookup dicts _drawdown /
    _momentum_skip5 / _vol_ann / _price_to_sma200 read directly. No disk,
    no network. Trimmed copy of tests/test_research.py's own fixture."""

    def __init__(self) -> None:
        self.open_by_ticker: dict[str, dict[date, float]] = {}
        self.close_by_ticker: dict[str, dict[date, float]] = {}
        self.dv_by_ticker: dict[str, dict[date, float]] = {}
        self.dates_by_ticker: dict[str, list[date]] = {}

    def add_flat_series(self, ticker: str, dates: list[date], price: float, dv: float) -> None:
        opens = self.open_by_ticker.setdefault(ticker, {})
        closes = self.close_by_ticker.setdefault(ticker, {})
        dvs = self.dv_by_ticker.setdefault(ticker, {})
        for d in dates:
            opens[d] = price
            closes[d] = price
            dvs[d] = dv
        self.dates_by_ticker[ticker] = sorted(set(self.dates_by_ticker.get(ticker, [])) | set(dates))

    def open(self, ticker: str, dt: date):
        return self.open_by_ticker.get(ticker, {}).get(dt)

    def close(self, ticker: str, dt: date):
        return self.close_by_ticker.get(ticker, {}).get(dt)

    def last_close_on_or_before(self, ticker: str, dt: date):
        dates = self.dates_by_ticker.get(ticker, [])
        for d in reversed(dates):
            if d <= dt:
                v = self.close_by_ticker.get(ticker, {}).get(d)
                if v is not None:
                    return v
        return None

    def median_dollar_volume(self, ticker: str, dt: date, window: int = 20):
        dates = self.dates_by_ticker.get(ticker, [])
        prior = [d for d in dates if d < dt][-window:]
        if not prior:
            return None
        dv = self.dv_by_ticker.get(ticker, {})
        vals = sorted(v for v in (dv.get(d, 0.0) for d in prior) if v == v)
        if not vals:
            return None
        return float(vals[len(vals) // 2])


def _tx(
    *, ticker: str = "ABC", issuer_cik: str = "CIK1", owner_cik: str = "O1",
    owner_name: str = "Owner One", transaction_date: date, filing_date: date,
    owner_roles: str = "Director", is_director: bool = True, is_officer: bool = False,
    is_ten_pct: bool = False, shares: float = 1000.0, price_per_share: float = 10.0,
    value: float = 10000.0, pct_of_prior_stake=None, footnote_text: str = "",
    is_10b5_1: bool = False, date_strings: bool = False,
) -> dict:
    """One transaction dict. date_strings=True mimics the live scanner's
    own shape (transaction_date "YYYY-MM-DD", filing_date "YYYYMMDD" --
    see insider_cluster_buys.py's Form 4 parser); date_strings=False
    mimics backtest.research's events_df shape (real date objects)."""
    if date_strings:
        td = transaction_date.strftime("%Y-%m-%d")
        fd = filing_date.strftime("%Y%m%d")
    else:
        td = transaction_date
        fd = filing_date
    return {
        "issuer_cik": issuer_cik, "issuer_name": "ISSUER", "ticker": ticker,
        "owner_cik": owner_cik, "owner_name": owner_name, "owner_roles": owner_roles,
        "is_director": is_director, "is_officer": is_officer, "is_ten_percent_owner": is_ten_pct,
        "transaction_date": td, "filing_date": fd,
        "shares": shares, "price_per_share": price_per_share, "value": value,
        "pct_of_prior_stake": pct_of_prior_stake, "footnote_text": footnote_text,
        "is_10b5_1": is_10b5_1,
    }


def make_window(date_strings: bool, n_owners: int = 3, ticker: str = "ABC") -> list[dict]:
    """A small, realistic 3-owner cluster window, spread across a few
    days, with one 10%-owner director and mixed prices/stakes -- enough
    variety to exercise every Section A/B/C formula (dispersion, role
    shares, stake stats, filing delay) with non-degenerate values."""
    base = date(2026, 1, 5)
    out = []
    for i in range(n_owners):
        out.append(_tx(
            ticker=ticker, owner_cik=f"O{i}", owner_name=f"Owner {i}",
            transaction_date=base + timedelta(days=i), filing_date=base + timedelta(days=i + 1),
            price_per_share=10.0 + i, shares=1000.0 + 100 * i,
            value=(10.0 + i) * (1000.0 + 100 * i),
            is_director=(i % 2 == 0), is_officer=(i % 2 == 1), is_ten_pct=(i == 0),
            pct_of_prior_stake=5.0 + i, owner_roles="Director, CEO" if i == 0 else "Officer, CFO",
            date_strings=date_strings,
        ))
    return out


def make_bundle(feature_cols: list[str], *, seed: int = 0, n: int = 300) -> rm.ProductionBundle:
    """A tiny real ProductionBundle (real fitted LGBMClassifier, not a
    stub) so score_live_cluster's predict_proba call path is exercised for
    real, cheaply, with no network and no real research dataset."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.standard_normal(n) for c in feature_cols})
    df["entry_idx"] = np.arange(n)
    df[rm.LABEL_COL] = rng.standard_normal(n) * 0.05
    # Force a real mix of both tail classes -- see tests/test_model.py's
    # tail_df fixture for the same reasoning.
    df.loc[: n // 5, rm.LABEL_COL] = rng.uniform(0.25, 0.6, n // 5 + 1)

    fast_params = dict(
        n_estimators=40, num_leaves=7, min_child_samples=8, learning_rate=0.1,
        subsample=0.9, subsample_freq=1, colsample_bytree=0.9, reg_lambda=1.0,
        random_state=0, verbosity=-1,
    )
    target = (df[rm.LABEL_COL] > rm.TAIL_THRESH).astype(int)
    model = rm._fit_classifier(df, feature_cols, target, fast_params)
    scores = np.sort(rm._predict_proba_positive(model, df[feature_cols], fallback_rate=float("nan")))
    return rm.ProductionBundle(
        model=model, feature_cols=list(feature_cols), training_scores=scores,
        provenance={"source_dataset_path": "synthetic", "n_rows": n}, config={},
    )


# ---------------------------------------------------------------------------
# 1. Feature categorization sanity (also enforced at import time, but
#    worth a direct test so a future edit's failure mode is a clear
#    assertion instead of a module-level RuntimeError at collection time).
# ---------------------------------------------------------------------------
class TestFeatureCategorization:
    def test_categories_partition_feature_cols_exactly(self):
        groups = [
            ls.ALWAYS_AVAILABLE_FEATURE_COLS, ls.PRICE_FEATURE_COLS,
            ls.ISSUER_HISTORY_FEATURE_COLS, ls.OWNER_HISTORY_FEATURE_COLS,
            ls.SALE_FEATURE_COLS,
        ]
        union = set().union(*[set(g) for g in groups])
        assert union == set(rm.FEATURE_COLS)
        assert sum(len(g) for g in groups) == len(rm.FEATURE_COLS)

    def test_reason_panel_factors_are_all_real_feature_columns(self):
        for spec in ls.REASON_PANEL_FACTORS:
            assert spec.feature in rm.FEATURE_COLS
            assert spec.favorable_direction in ("lower", "higher")


# ---------------------------------------------------------------------------
# 2. Section A/B/C parity with the real research pipeline -- the load-
#    bearing test in this file.
# ---------------------------------------------------------------------------
class TestSectionABCParity:
    def test_section_abc_parity_with_research_build_event_row(self):
        research_window = make_window(date_strings=False)
        live_window = make_window(date_strings=True)

        calendar = [date(2026, 1, 1) + timedelta(days=i) for i in range(400)]
        prices = FakePriceUniverse()  # no data on purpose: only labels / Section E read it
        owners = research_mod._aggregate_owners(research_window)

        research_row = research_mod._build_event_row(
            ticker="ABC", D=date(2026, 1, 8), entry_day=date(2026, 1, 9), entry_idx=5,
            window=research_window,
            state={"component_keys": [], "conviction_score": 0, "num_insiders": len(owners)},
            prices=prices, calendar=calendar, horizons=(63,),
            feature_keys=list(ics.DEFAULT_WEIGHTS.keys()),
            owner_history={}, issuer_history={}, ticker_last_event={},
            day_owner_updates=[], day_issuer_updates=[], day_ticker_updates={},
            # No sale data on either side of this parity check: Section F
            # features aren't in ALWAYS_AVAILABLE_FEATURE_COLS (see
            # research.live_score.SALE_FEATURE_COLS), so this loop's
            # assertions never touch them regardless of sale_index's content.
            sale_index={}, window_days=14,
        )

        live_row, availability = ls.build_live_feature_row(
            cluster={"ticker": "ABC", "issuer_cik": "CIK1"}, window=live_window,
        )

        assert set(ls.ALWAYS_AVAILABLE_FEATURE_COLS) == set(availability.computed)
        for feature in ls.ALWAYS_AVAILABLE_FEATURE_COLS:
            expected = research_row[feature]
            actual = live_row[feature]
            expected_f = float(expected)
            actual_f = float(actual)
            if expected_f != expected_f:  # NaN
                assert actual_f != actual_f, f"{feature}: research=NaN but live={actual_f}"
            else:
                assert actual_f == pytest.approx(expected_f, rel=1e-9, abs=1e-9), (
                    f"{feature}: research={expected_f} != live={actual_f}"
                )

    def test_parity_holds_with_a_ten_percent_owner_and_missing_stake_pct(self):
        """A second, differently-shaped cluster (fewer owners, one missing
        pct_of_prior_stake, no ten-percent owner) -- guards against the
        first test's parity happening to hold only for one convenient shape."""
        research_window = [
            _tx(owner_cik="A", transaction_date=date(2026, 2, 1), filing_date=date(2026, 2, 2),
                is_director=True, is_officer=False, price_per_share=5.0, shares=2000.0, value=10000.0,
                pct_of_prior_stake=None, owner_roles="Director"),
            _tx(owner_cik="B", transaction_date=date(2026, 2, 1), filing_date=date(2026, 2, 3),
                is_director=False, is_officer=True, price_per_share=5.0, shares=4000.0, value=20000.0,
                pct_of_prior_stake=12.5, owner_roles="Officer (President)"),
        ]
        live_window = [
            _tx(owner_cik="A", transaction_date=date(2026, 2, 1), filing_date=date(2026, 2, 2),
                is_director=True, is_officer=False, price_per_share=5.0, shares=2000.0, value=10000.0,
                pct_of_prior_stake=None, owner_roles="Director", date_strings=True),
            _tx(owner_cik="B", transaction_date=date(2026, 2, 1), filing_date=date(2026, 2, 3),
                is_director=False, is_officer=True, price_per_share=5.0, shares=4000.0, value=20000.0,
                pct_of_prior_stake=12.5, owner_roles="Officer (President)", date_strings=True),
        ]
        calendar = [date(2026, 1, 1) + timedelta(days=i) for i in range(400)]
        prices = FakePriceUniverse()
        owners = research_mod._aggregate_owners(research_window)

        research_row = research_mod._build_event_row(
            ticker="XYZ", D=date(2026, 2, 3), entry_day=date(2026, 2, 4), entry_idx=10,
            window=research_window,
            state={"component_keys": [], "conviction_score": 0, "num_insiders": len(owners)},
            prices=prices, calendar=calendar, horizons=(63,),
            feature_keys=list(ics.DEFAULT_WEIGHTS.keys()),
            owner_history={}, issuer_history={}, ticker_last_event={},
            day_owner_updates=[], day_issuer_updates=[], day_ticker_updates={},
            sale_index={}, window_days=14,
        )
        live_row, _ = ls.build_live_feature_row(cluster={"ticker": "XYZ"}, window=live_window)

        for feature in ls.ALWAYS_AVAILABLE_FEATURE_COLS:
            expected_f = float(research_row[feature])
            actual_f = float(live_row[feature])
            if expected_f != expected_f:
                assert actual_f != actual_f, feature
            else:
                assert actual_f == pytest.approx(expected_f, rel=1e-9, abs=1e-9), feature

    def test_raises_on_empty_window(self):
        with pytest.raises(ValueError):
            ls.build_live_feature_row(cluster={"ticker": "ABC"}, window=[])


# ---------------------------------------------------------------------------
# 3. Feature availability reporting
# ---------------------------------------------------------------------------
class TestFeatureAvailability:
    def test_no_optional_inputs_leaves_price_and_history_missing(self):
        window = make_window(date_strings=True)
        row, availability = ls.build_live_feature_row(cluster={"ticker": "ABC"}, window=window)

        assert set(availability.computed) == set(ls.ALWAYS_AVAILABLE_FEATURE_COLS)
        expected_missing = (
            set(ls.PRICE_FEATURE_COLS) | set(ls.ISSUER_HISTORY_FEATURE_COLS)
            | set(ls.OWNER_HISTORY_FEATURE_COLS) | set(ls.SALE_FEATURE_COLS)
        )
        assert set(availability.missing) == expected_missing
        assert availability.n_total == len(rm.FEATURE_COLS)
        assert availability.frac_missing == pytest.approx(len(expected_missing) / len(rm.FEATURE_COLS))
        for c in ls.PRICE_FEATURE_COLS:
            assert row[c] != row[c]  # NaN

    def test_owner_history_always_missing_even_with_everything_else_supplied(self):
        window = make_window(date_strings=True)
        prices = FakePriceUniverse()
        prices.add_flat_series("ABC", [date(2025, 1, 1) + timedelta(days=i) for i in range(300)], price=10.0, dv=1_000_000.0)
        issuer_history = pd.DataFrame({
            "ticker": ["ABC"], "issuer_cik": ["CIK1"], "event_day": [date(2025, 6, 1)], rm.LABEL_COL: [0.1],
        })
        _, availability = ls.build_live_feature_row(
            cluster={"ticker": "ABC", "issuer_cik": "CIK1"}, window=window,
            prices=prices, issuer_history=issuer_history, as_of=date(2026, 1, 10),
        )
        expected_missing = set(ls.OWNER_HISTORY_FEATURE_COLS) | set(ls.SALE_FEATURE_COLS)
        assert set(availability.missing) == expected_missing
        assert availability.frac_missing == pytest.approx(len(expected_missing) / len(rm.FEATURE_COLS))
        for c in ls.OWNER_HISTORY_FEATURE_COLS:
            assert availability.missing_reasons[c] == ls._OWNER_HISTORY_REASON
        for c in ls.SALE_FEATURE_COLS:
            assert availability.missing_reasons[c] == ls._SALE_REASON

    def test_prices_only_unlocks_price_features_but_not_history(self):
        window = make_window(date_strings=True)
        prices = FakePriceUniverse()
        prices.add_flat_series("ABC", [date(2025, 1, 1) + timedelta(days=i) for i in range(300)], price=10.0, dv=1_000_000.0)
        _, availability = ls.build_live_feature_row(
            cluster={"ticker": "ABC"}, window=window, prices=prices, as_of=date(2026, 1, 10),
        )
        assert set(ls.PRICE_FEATURE_COLS).issubset(availability.computed)
        assert set(ls.ISSUER_HISTORY_FEATURE_COLS).issubset(availability.missing)


# ---------------------------------------------------------------------------
# 4. Issuer/ticker-level history features
# ---------------------------------------------------------------------------
class TestIssuerTickerHistory:
    def test_first_ever_cluster_when_no_prior_ticker_rows(self):
        issuer_history = pd.DataFrame({
            "ticker": ["OTHER"], "issuer_cik": ["CIKX"], "event_day": [date(2020, 1, 1)], rm.LABEL_COL: [0.1],
        })
        row = ls._issuer_ticker_history_features("CIK1", "ABC", date(2026, 1, 1), issuer_history)
        assert row["x_is_first_ever_cluster"] == 1.0
        assert row["x_issuer_n_prior_clusters"] == 0.0
        assert row["x_days_since_prior_cluster"] != row["x_days_since_prior_cluster"]  # NaN

    def test_prior_cluster_at_same_ticker_and_issuer(self):
        issuer_history = pd.DataFrame({
            "ticker": ["ABC", "ABC"],
            "issuer_cik": ["CIK1", "CIK1"],
            "event_day": [date(2024, 1, 1), date(2024, 6, 1)],
            rm.LABEL_COL: [0.10, 0.20],
        })
        as_of = date(2026, 1, 1)
        row = ls._issuer_ticker_history_features("CIK1", "ABC", as_of, issuer_history)
        assert row["x_is_first_ever_cluster"] == 0.0
        assert row["x_issuer_n_prior_clusters"] == 2.0
        assert row["x_days_since_prior_cluster"] == float((as_of - date(2024, 6, 1)).days)
        # Both prior events are well over 91 days old -- both contribute.
        assert row["x_issuer_prior_adj63_mean"] == pytest.approx(0.15)

    def test_recent_prior_cluster_excluded_from_adj63_mean_label_window_gate(self):
        """A prior cluster inside the ~91-day approximate label-window gate
        must not contribute to x_issuer_prior_adj63_mean (its own 63-day
        return would not be "known" yet as of as_of), but it MUST still
        count in x_issuer_n_prior_clusters (a plain count needs no gate --
        see backtest/research.py's own leakage note, reused verbatim in
        live_score.py's docstring for _ISSUER_LABEL_WINDOW_APPROX_DAYS)."""
        as_of = date(2026, 1, 1)
        issuer_history = pd.DataFrame({
            "ticker": ["ABC"], "issuer_cik": ["CIK1"],
            "event_day": [as_of - timedelta(days=10)],
            rm.LABEL_COL: [0.99],
        })
        row = ls._issuer_ticker_history_features("CIK1", "ABC", as_of, issuer_history)
        assert row["x_issuer_n_prior_clusters"] == 1.0
        assert row["x_issuer_prior_adj63_mean"] != row["x_issuer_prior_adj63_mean"]  # NaN, gated out

    def test_future_or_same_day_rows_never_count_as_prior(self):
        as_of = date(2026, 1, 1)
        issuer_history = pd.DataFrame({
            "ticker": ["ABC", "ABC"], "issuer_cik": ["CIK1", "CIK1"],
            "event_day": [as_of, as_of + timedelta(days=5)],
            rm.LABEL_COL: [0.5, 0.5],
        })
        row = ls._issuer_ticker_history_features("CIK1", "ABC", as_of, issuer_history)
        assert row["x_issuer_n_prior_clusters"] == 0.0
        assert row["x_is_first_ever_cluster"] == 1.0


# ---------------------------------------------------------------------------
# 5. Percentile against a fixed reference distribution
# ---------------------------------------------------------------------------
class TestScoreToPercentile:
    TRAIN = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

    def test_below_min_is_zero(self):
        assert ls.score_to_percentile(0.05, self.TRAIN) == 0.0

    def test_at_or_above_max_is_100(self):
        assert ls.score_to_percentile(0.5, self.TRAIN) == 100.0
        assert ls.score_to_percentile(0.9, self.TRAIN) == 100.0

    def test_mid_value_matches_side_right_semantics(self):
        # 3 of 5 training scores are <= 0.3 -> 60th percentile.
        assert ls.score_to_percentile(0.3, self.TRAIN) == pytest.approx(60.0)
        # Between two training points: still 3 of 5 are <=.
        assert ls.score_to_percentile(0.35, self.TRAIN) == pytest.approx(60.0)

    def test_nan_score_is_nan(self):
        assert math.isnan(ls.score_to_percentile(float("nan"), self.TRAIN))

    def test_empty_reference_is_nan(self):
        assert math.isnan(ls.score_to_percentile(0.3, np.array([])))

    def test_monotonic_nondecreasing_over_increasing_scores(self):
        rng = np.random.default_rng(0)
        train = np.sort(rng.standard_normal(500))
        probes = np.sort(rng.standard_normal(200))
        percentiles = [ls.score_to_percentile(float(p), train) for p in probes]
        assert all(a <= b + 1e-9 for a, b in zip(percentiles, percentiles[1:]))

    def test_reference_distribution_is_fixed_not_recomputed_from_a_batch(self):
        """The whole point of score_to_percentile taking an explicit
        `training_scores` array: scoring the SAME raw_score against two
        different reference arrays (e.g. "today's batch" vs "the model's
        real training distribution") gives different answers -- proving the
        function has no hidden global state that could let a caller
        accidentally rank against a moving target."""
        batch_today = np.array([0.4, 0.4, 0.4, 0.4, 0.4])  # a pathological, all-identical "batch"
        p_vs_training = ls.score_to_percentile(0.3, self.TRAIN)
        p_vs_batch = ls.score_to_percentile(0.3, batch_today)
        assert p_vs_training != p_vs_batch


# ---------------------------------------------------------------------------
# 6. Banded verdict
# ---------------------------------------------------------------------------
class _Avail:
    def __init__(self, frac_missing: float):
        self.frac_missing = frac_missing


class TestBandVerdict:
    def test_percentile_at_cutoff_is_top_decile(self):
        avail = _Avail(0.0)
        assert ls.band_verdict(ls.TOP_DECILE_PERCENTILE_CUTOFF, avail) == ls.Verdict.TOP_DECILE

    def test_just_below_cutoff_is_no_edge(self):
        avail = _Avail(0.0)
        assert ls.band_verdict(ls.TOP_DECILE_PERCENTILE_CUTOFF - 0.01, avail) == ls.Verdict.NO_EDGE

    def test_nan_percentile_is_unavailable(self):
        avail = _Avail(0.0)
        assert ls.band_verdict(float("nan"), avail) == ls.Verdict.UNAVAILABLE

    def test_too_much_missing_data_is_unavailable_even_at_top_score(self):
        avail = _Avail(0.9)
        assert ls.band_verdict(99.9, avail) == ls.Verdict.UNAVAILABLE

    def test_default_no_optional_input_missing_frac_still_scoreable(self):
        """38% missing (no PriceUniverse, no issuer_history -- the
        every-day case until the live tool is wired up further) must NOT
        trip the backstop by itself; DEFAULT_MAX_MISSING_FEATURE_FRAC=0.5
        is a backstop against an extreme case, not the normal gate."""
        avail = _Avail(19 / 50)
        assert ls.band_verdict(50.0, avail) == ls.Verdict.NO_EDGE

    def test_custom_threshold_is_respected(self):
        avail = _Avail(0.3)
        assert ls.band_verdict(99.0, avail, max_missing_frac=0.2) == ls.Verdict.UNAVAILABLE
        assert ls.band_verdict(99.0, avail, max_missing_frac=0.5) == ls.Verdict.TOP_DECILE


# ---------------------------------------------------------------------------
# 7. Per-factor reasons panel
# ---------------------------------------------------------------------------
class TestFactorReport:
    def test_factor_report_carries_value_and_availability(self):
        window = make_window(date_strings=True)
        row, availability = ls.build_live_feature_row(cluster={"ticker": "ABC"}, window=window)
        factors = ls.factor_report(row, availability)
        assert len(factors) == len(ls.REASON_PANEL_FACTORS)
        by_feature = {f.feature: f for f in factors}
        # x_n_ten_pct is Section B -- always available.
        assert by_feature["x_n_ten_pct"].available is True
        assert by_feature["x_n_ten_pct"].value == row["x_n_ten_pct"]
        # x_entry_vs_insider_vwap needs prices -- unavailable with none supplied.
        assert by_feature["x_entry_vs_insider_vwap"].available is False


# ---------------------------------------------------------------------------
# 8. score_live_cluster end to end
# ---------------------------------------------------------------------------
class TestScoreLiveCluster:
    def test_end_to_end_returns_a_real_score_and_verdict(self):
        bundle = make_bundle(rm.FEATURE_COLS)
        window = make_window(date_strings=True)
        cluster = {"ticker": "ABC", "issuer_cik": "CIK1"}
        result = ls.score_live_cluster(cluster, window, bundle, as_of=date(2026, 1, 10))

        assert 0.0 <= result.raw_score <= 1.0
        assert 0.0 <= result.percentile <= 100.0
        assert result.verdict in (ls.Verdict.TOP_DECILE, ls.Verdict.NO_EDGE, ls.Verdict.UNAVAILABLE)
        assert len(result.factors) == len(ls.REASON_PANEL_FACTORS)
        assert result.as_of == date(2026, 1, 10)

    def test_verdict_is_top_decile_iff_percentile_at_or_above_cutoff(self):
        bundle = make_bundle(rm.FEATURE_COLS)
        window = make_window(date_strings=True)
        cluster = {"ticker": "ABC", "issuer_cik": "CIK1"}
        result = ls.score_live_cluster(cluster, window, bundle, as_of=date(2026, 1, 10))
        if result.percentile >= ls.TOP_DECILE_PERCENTILE_CUTOFF:
            assert result.verdict == ls.Verdict.TOP_DECILE
        else:
            assert result.verdict == ls.Verdict.NO_EDGE

    def test_low_max_missing_frac_forces_unavailable_without_optional_inputs(self):
        bundle = make_bundle(rm.FEATURE_COLS)
        window = make_window(date_strings=True)
        cluster = {"ticker": "ABC", "issuer_cik": "CIK1"}
        result = ls.score_live_cluster(
            cluster, window, bundle, as_of=date(2026, 1, 10), max_missing_frac=0.1,
        )
        assert result.verdict == ls.Verdict.UNAVAILABLE

    def test_supplying_prices_and_history_reduces_missing_features(self):
        bundle = make_bundle(rm.FEATURE_COLS)
        window = make_window(date_strings=True)
        cluster = {"ticker": "ABC", "issuer_cik": "CIK1"}
        prices = FakePriceUniverse()
        prices.add_flat_series("ABC", [date(2025, 1, 1) + timedelta(days=i) for i in range(300)], price=10.0, dv=1_000_000.0)
        issuer_history = pd.DataFrame({
            "ticker": ["ABC"], "issuer_cik": ["CIK1"], "event_day": [date(2024, 1, 1)], rm.LABEL_COL: [0.1],
        })
        result = ls.score_live_cluster(
            cluster, window, bundle, prices=prices, issuer_history=issuer_history, as_of=date(2026, 1, 10),
        )
        assert result.availability.missing == list(ls.OWNER_HISTORY_FEATURE_COLS) + list(ls.SALE_FEATURE_COLS)
