"""Tests for build_html.py's model-based rendering.

conviction_score is gone from this project's screener presentation (see
insider_cluster_buys.py's own "Model-based live scoring" section docstring);
render_html/write_excel now sort and label clusters from
cluster["model_score"], the dict research.live_score.score_live_cluster
produces. These tests cover the small pure-function display logic
(factor_favorable, verdict_label, model_sort_key) and a handful of
render_html smoke tests that never touch the network or research_data/.
"""

from __future__ import annotations

import math

import build_html as bh


# ---------------------------------------------------------------------------
# factor_favorable -- the zero-pivot favorable/unfavorable call
# ---------------------------------------------------------------------------
class TestFactorFavorable:
    def test_lower_is_favorable_below_zero(self):
        assert bh.factor_favorable("x_entry_vs_insider_vwap", -0.03, "lower") is True

    def test_lower_is_unfavorable_above_zero(self):
        assert bh.factor_favorable("x_entry_vs_insider_vwap", 0.03, "lower") is False

    def test_lower_at_exactly_zero_is_favorable(self):
        # <=0 favorable for a "lower" feature -- 0 itself (bought at exactly
        # the insiders' own VWAP) counts as favorable, not a tie.
        assert bh.factor_favorable("x_entry_vs_insider_vwap", 0.0, "lower") is True

    def test_higher_is_favorable_above_zero(self):
        assert bh.factor_favorable("x_issuer_n_prior_clusters", 2.0, "higher") is True

    def test_higher_at_zero_is_unfavorable(self):
        assert bh.factor_favorable("x_issuer_n_prior_clusters", 0.0, "higher") is False

    def test_ten_pct_count_zero_is_favorable(self):
        assert bh.factor_favorable("x_n_ten_pct", 0.0, "lower") is True

    def test_ten_pct_count_positive_is_unfavorable(self):
        assert bh.factor_favorable("x_n_ten_pct", 1.0, "lower") is False

    def test_ten_pct_value_share_zero_is_favorable(self):
        assert bh.factor_favorable("x_ten_pct_value_share", 0.0, "lower") is True

    def test_is_first_ever_cluster_zero_favorable_one_unfavorable(self):
        assert bh.factor_favorable("x_is_first_ever_cluster", 0.0, "lower") is True
        assert bh.factor_favorable("x_is_first_ever_cluster", 1.0, "lower") is False

    def test_no_zero_pivot_feature_returns_none(self):
        # x_vol_21_ann is always >= 0 -- there is no defensible "0 is bad,
        # positive is good" split, only the directional "lower is better".
        # Inventing a threshold here would be unsupported precision.
        assert bh.factor_favorable("x_vol_21_ann", 0.35, "lower") is None
        assert bh.factor_favorable("x_vol_21_ann", 0.0, "lower") is None

    def test_none_value_returns_none(self):
        assert bh.factor_favorable("x_entry_vs_insider_vwap", None, "lower") is None

    def test_nan_value_returns_none(self):
        assert bh.factor_favorable("x_entry_vs_insider_vwap", float("nan"), "lower") is None

    def test_every_reason_panel_factor_is_categorized_consistently(self):
        """Cross-check against research.live_score.REASON_PANEL_FACTORS so a
        future edit to that constant cannot silently drift from this
        module's favorable/unfavorable logic without a test noticing."""
        from research import live_score as ls

        for spec in ls.REASON_PANEL_FACTORS:
            # Every factor must at least be callable without raising, for
            # both a "favorable-looking" and "unfavorable-looking" value.
            bh.factor_favorable(spec.feature, 0.0, spec.favorable_direction)
            bh.factor_favorable(spec.feature, 1.0, spec.favorable_direction)


# ---------------------------------------------------------------------------
# verdict_label
# ---------------------------------------------------------------------------
class TestVerdictLabel:
    def test_known_verdicts(self):
        assert "Top decile" in bh.verdict_label("top_decile")
        assert "No measured edge" == bh.verdict_label("no_edge")
        assert "Unavailable" in bh.verdict_label("unavailable")
        assert bh.verdict_label("not_scored") == "Not scored"

    def test_none_and_unknown_default_to_not_scored(self):
        assert bh.verdict_label(None) == "Not scored"
        assert bh.verdict_label("some_future_verdict") == "Not scored"


# ---------------------------------------------------------------------------
# model_sort_key -- default cluster ordering
# ---------------------------------------------------------------------------
def _cluster(*, verdict=None, percentile=None, total_value=0.0, scored=True):
    if not scored:
        return {"total_value": total_value, "model_score": None}
    return {
        "total_value": total_value,
        "model_score": {"verdict": verdict, "percentile": percentile},
    }


class TestModelSortKey:
    def test_top_decile_sorts_before_no_edge(self):
        top = _cluster(verdict="top_decile", percentile=95.0)
        no_edge = _cluster(verdict="no_edge", percentile=50.0)
        ordered = sorted([no_edge, top], key=bh.model_sort_key)
        assert ordered == [top, no_edge]

    def test_no_edge_sorts_before_unavailable(self):
        no_edge = _cluster(verdict="no_edge", percentile=10.0)
        unavailable = _cluster(verdict="unavailable", percentile=99.0)
        ordered = sorted([unavailable, no_edge], key=bh.model_sort_key)
        assert ordered == [no_edge, unavailable]

    def test_unscored_sorts_last(self):
        top = _cluster(verdict="top_decile", percentile=95.0)
        unscored = _cluster(scored=False)
        ordered = sorted([unscored, top], key=bh.model_sort_key)
        assert ordered == [top, unscored]

    def test_within_same_verdict_higher_percentile_first(self):
        hi = _cluster(verdict="top_decile", percentile=99.0)
        lo = _cluster(verdict="top_decile", percentile=90.5)
        ordered = sorted([lo, hi], key=bh.model_sort_key)
        assert ordered == [hi, lo]

    def test_total_value_is_the_final_tiebreak(self):
        big = _cluster(verdict="no_edge", percentile=50.0, total_value=5_000_000)
        small = _cluster(verdict="no_edge", percentile=50.0, total_value=1_000)
        ordered = sorted([small, big], key=bh.model_sort_key)
        assert ordered == [big, small]

    def test_nan_percentile_does_not_raise_and_sorts_low(self):
        nan_pctl = _cluster(verdict="no_edge", percentile=float("nan"))
        real_pctl = _cluster(verdict="no_edge", percentile=10.0)
        ordered = sorted([nan_pctl, real_pctl], key=bh.model_sort_key)
        assert ordered == [real_pctl, nan_pctl]


# ---------------------------------------------------------------------------
# render_html smoke tests -- no network, no research_data/, hand-built
# payloads only.
# ---------------------------------------------------------------------------
def _payload(clusters, model_info=None):
    return {
        "generated_at": "2026-08-08T00:00:00",
        "scanned_from": "2026-07-01",
        "scanned_to": "2026-08-08",
        "qualifying_codes": ["P"],
        "clusters": clusters,
        "model_info": model_info or {},
    }


def _minimal_cluster(**overrides) -> dict:
    c = {
        "issuer_cik": "1",
        "issuer_name": "Test Issuer",
        "ticker": "TST",
        "cluster_start": "2026-08-01",
        "cluster_end": "2026-08-03",
        "num_insiders": 2,
        "insiders": [
            {"name": "A", "roles": "Director", "shares": 100, "value": 1000},
            {"name": "B", "roles": "Officer", "shares": 200, "value": 2000},
        ],
        "total_shares": 300,
        "total_value": 3000,
        "max_pct_of_prior_stake": 10.0,
        "includes_ten_percent_owner": False,
        "includes_director": True,
        "includes_officer": True,
        "num_transactions": 2,
        "transactions": [],
        "edgar_url": "https://example.com",
        "is_recent_ipo": False,
        "model_score": None,
    }
    c.update(overrides)
    return c


def test_render_html_with_no_model_bundle_says_so_in_the_banner():
    html_out = bh.render_html(_payload([_minimal_cluster()], model_info={"model_available": False}))
    assert "No production model bundle was found" in html_out
    assert "verdict-not_scored" in html_out


def test_render_html_top_decile_cluster_shows_edge_badge_and_percentile():
    ms = {
        "raw_score": 0.8, "percentile": 96.0, "verdict": "top_decile",
        "n_features_available": 47, "n_features_total": 50,
        "missing_features": [], "n_training_scores": 10693,
        "factors": [
            {"feature": "x_entry_vs_insider_vwap", "value": -0.02, "ic": -0.119,
             "favorable_direction": "lower", "description": "test desc", "available": True},
        ],
    }
    payload = _payload(
        [_minimal_cluster(model_score=ms)],
        model_info={"model_available": True, "n_training_scores": 10693,
                    "price_fetch_enabled": True, "prices_loaded": True,
                    "n_tickers_priced": 1, "price_fetch_elapsed_sec": 1.2,
                    "issuer_history_available": True, "n_issuer_history_rows": 11026},
    )
    html_out = bh.render_html(payload)
    assert "verdict-top_decile" in html_out
    assert "96th percentile" in html_out
    assert "Favorable" in html_out
    assert "only the top decile" in html_out.lower() or "top decile (percentile" in html_out.lower()


def test_render_html_degraded_cluster_shows_reduced_features_chip():
    ms = {
        "raw_score": 0.4, "percentile": 55.0, "verdict": "no_edge",
        "n_features_available": 31, "n_features_total": 50,
        "missing_features": ["x_vol_21_ann"], "n_training_scores": 10693,
        "factors": [],
    }
    html_out = bh.render_html(_payload([_minimal_cluster(model_score=ms)]))
    assert "reduced features" in html_out
    assert "31/50" in html_out


def test_render_html_does_not_reference_conviction_score():
    """The whole point of this migration: conviction_score/conviction_label
    must not appear anywhere in the rendered output."""
    html_out = bh.render_html(_payload([_minimal_cluster()]))
    assert "conviction" not in html_out.lower()
    assert "signal-conviction" not in html_out
    assert "Signal breakdown" not in html_out


def test_render_html_scored_clusters_sort_before_unscored():
    scored = _minimal_cluster(
        ticker="AAA",
        model_score={"raw_score": 0.9, "percentile": 95.0, "verdict": "top_decile",
                     "n_features_available": 50, "n_features_total": 50,
                     "missing_features": [], "n_training_scores": 100, "factors": []},
    )
    unscored = _minimal_cluster(ticker="ZZZ", model_score=None)
    html_out = bh.render_html(_payload([unscored, scored]))
    assert html_out.index("AAA") < html_out.index("ZZZ")


def test_render_html_empty_clusters_does_not_crash():
    html_out = bh.render_html(_payload([]))
    assert "No flagged clusters" in html_out


def test_extract_embedded_payload_round_trips_nan_scores():
    """model_score fields can legitimately be NaN (e.g. percentile when
    raw_score wasn't finite) -- the embedded-payload round trip used by the
    CLI re-render mode must survive that."""
    ms = {
        "raw_score": float("nan"), "percentile": float("nan"), "verdict": "unavailable",
        "n_features_available": 10, "n_features_total": 50,
        "missing_features": [], "n_training_scores": 100, "factors": [],
    }
    html_out = bh.render_html(_payload([_minimal_cluster(model_score=ms)]))
    payload = bh.extract_embedded_payload(html_out)
    got = payload["clusters"][0]["model_score"]["percentile"]
    assert got != got  # NaN
