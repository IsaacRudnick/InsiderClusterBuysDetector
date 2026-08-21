"""Tests for tools.fundamentals, the point-in-time company-size/fundamentals
feature set built from SEC's bulk XBRL data.

Runnable standalone via `python -m pytest tests/test_fundamentals.py -q`
from the repo root. No network, no real bulk XBRL data: every test builds a
synthetic `fundamentals_df` (the shape build_fundamentals() produces) and/or
a synthetic raw num.txt-shaped frame directly.

The single most important test here is
`test_same_day_and_future_filings_have_zero_influence` -- mirrors
tests/test_earnings_features.py's test of the same name. Every x_ feature
in tools/fundamentals.py exists to describe an issuer's size and balance
sheet as it was knowable at the moment of the event -- not as it turned out
three months later when the next 10-Q landed. If a same-day or future-dated
filing were allowed to leak into any of these columns, the model would be
quietly handed a peek at the future dressed up as a "feature", and every
backtest number built on it would be unearned.

The second is `test_tag_fallback_chain_picks_preferred_source`, which
proves _resolve_group_facts actually honors TAG_CHAINS's priority order
instead of picking arbitrarily when a filing tags a concept more than one
way.
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

from tools import fundamentals as fnd  # noqa: E402


CIK_A = "0000000001"
CIK_B = "0000000002"


def make_fundamentals(rows) -> pd.DataFrame:
    """rows: list of (cik, filed_date_str, period_end_str, tag_group, value)."""
    df = pd.DataFrame(rows, columns=["cik", "filed_date", "period_end", "tag_group", "value"])
    df["filed_date"] = pd.to_datetime(df["filed_date"])
    df["period_end"] = pd.to_datetime(df["period_end"])
    df["value"] = df["value"].astype(float)
    return df


def make_events(rows, with_entry_open: bool = False) -> pd.DataFrame:
    """rows: list of (issuer_cik, event_day_str[, entry_open])."""
    cols = ["issuer_cik", "event_day"] + (["entry_open"] if with_entry_open else [])
    df = pd.DataFrame(rows, columns=cols)
    df["event_day"] = pd.to_datetime(df["event_day"]).dt.date
    return df


# ---------------------------------------------------------------------------
# The test that matters most: point-in-time enforcement
# ---------------------------------------------------------------------------

def test_same_day_and_future_filings_have_zero_influence():
    """Fundamentals filed on or after event_day must not move any computed
    feature -- proven by deleting them and checking every output is
    bit-for-bit identical."""
    event_day = "2020-06-15"
    events = make_events([(CIK_A, event_day, 10.0)], with_entry_open=True)

    prior_facts = [
        (CIK_A, "2020-03-01", "2019-12-31", "shares_outstanding", 1_000_000.0),
        (CIK_A, "2020-03-01", "2019-12-31", "revenue", 4_000_000.0),
        (CIK_A, "2020-03-01", "2019-12-31", "cash", 500_000.0),
        (CIK_A, "2020-03-01", "2019-12-31", "equity", 2_000_000.0),
        (CIK_A, "2020-03-01", "2019-12-31", "net_income", -400_000.0),
        (CIK_A, "2020-03-01", "2019-12-31", "assets", 3_000_000.0),
    ]
    # These must have ZERO influence: one exactly on event_day, one after it,
    # for every tag_group.
    contaminating = [
        (CIK_A, event_day, "2020-06-15", "shares_outstanding", 9_999_999.0),
        (CIK_A, event_day, "2020-06-15", "revenue", 999_000_000.0),
        (CIK_A, "2020-06-16", "2020-06-16", "cash", 999_000_000.0),
        (CIK_A, "2020-06-20", "2020-06-20", "equity", 999_000_000.0),
        (CIK_A, "2020-07-01", "2020-06-30", "net_income", 999_000_000.0),
        (CIK_A, "2020-07-01", "2020-06-30", "assets", 999_000_000.0),
    ]

    fund_with_future = make_fundamentals(prior_facts + contaminating)
    fund_without_future = make_fundamentals(prior_facts)

    feats_with = fnd.attach_fundamentals_features(events, fund_with_future)
    feats_without = fnd.attach_fundamentals_features(events, fund_without_future)

    pd.testing.assert_frame_equal(feats_with, feats_without)

    # And sanity-check the values are the ones the PRIOR-only facts imply.
    row = feats_without.iloc[0]
    assert row["x_market_cap_log"] == pytest.approx(np.log(1_000_000.0 * 10.0))
    assert row["x_revenue_log"] == pytest.approx(np.log1p(4_000_000.0))
    assert row["x_cash_to_assets"] == pytest.approx(500_000.0 / 3_000_000.0)
    assert row["x_equity_to_assets"] == pytest.approx(2_000_000.0 / 3_000_000.0)
    assert row["x_net_margin"] == pytest.approx(-400_000.0 / 4_000_000.0)
    # net_income (annualized) = -400,000 -> quarterly burn = 100,000
    assert row["x_cash_runway_quarters"] == pytest.approx(500_000.0 / 100_000.0)
    assert row["x_fundamentals_age_days"] == (pd.Timestamp(event_day) - pd.Timestamp("2020-03-01")).days


def test_same_day_filing_excluded_even_when_it_is_the_only_one():
    """An issuer whose ONLY fundamentals filing is dated exactly on
    event_day must get NaN everywhere, not a leaked same-day value."""
    events = make_events([(CIK_A, "2020-06-15", 10.0)], with_entry_open=True)
    fund = make_fundamentals([
        (CIK_A, "2020-06-15", "2020-06-15", "assets", 1_000_000.0),
        (CIK_A, "2020-06-15", "2020-06-15", "cash", 100_000.0),
    ])
    feats = fnd.attach_fundamentals_features(events, fund)
    row = feats.iloc[0]
    assert pd.isna(row["x_cash_to_assets"])
    assert pd.isna(row["x_fundamentals_age_days"])


# ---------------------------------------------------------------------------
# Tag fallback chain
# ---------------------------------------------------------------------------

def _facts_row(cik, adsh, filed, period_end, tag, rank, qtrs, value):
    return {
        "cik": cik, "adsh": adsh, "filed": filed, "period_end": period_end,
        "tag_group": tag, "_rank": rank, "qtrs": qtrs, "value": value,
    }


def test_tag_fallback_chain_picks_preferred_source():
    """A filing that tags a concept with BOTH a preferred and a fallback
    tag must resolve to the PREFERRED (lower _rank) one, never the
    fallback, and never both."""
    facts = pd.DataFrame([
        # revenue: chain is Revenues(0) -> RevenueFromContract...(1) -> SalesRevenueNet(2).
        # This filing tags both Revenues and SalesRevenueNet -- Revenues must win.
        _facts_row(CIK_A, "acc-1", "20200301", "20191231", "revenue", 0, 4, 5_000_000.0),
        _facts_row(CIK_A, "acc-1", "20200301", "20191231", "revenue", 2, 4, 4_900_000.0),
        # shares_outstanding: only the rank-1 (us-gaap:CommonStockSharesOutstanding)
        # and rank-2 (CommonStockSharesIssued) tags present -- rank 1 must win.
        _facts_row(CIK_A, "acc-1", "20200301", "20191231", "shares_outstanding", 1, 0, 1_000_000.0),
        _facts_row(CIK_A, "acc-1", "20200301", "20191231", "shares_outstanding", 2, 0, 1_100_000.0),
        # A second issuer's filing tags ONLY the least-preferred shares tag
        # -- the fallback must still be usable, not dropped.
        _facts_row(CIK_B, "acc-2", "20200401", "20200101", "shares_outstanding", 2, 0, 2_000_000.0),
    ])

    resolved = fnd._resolve_group_facts(facts)

    rev = resolved[(resolved["cik"] == CIK_A) & (resolved["tag_group"] == "revenue")]
    assert len(rev) == 1
    assert rev.iloc[0]["value"] == 5_000_000.0  # Revenues (rank 0), not SalesRevenueNet (rank 2)

    shares_a = resolved[(resolved["cik"] == CIK_A) & (resolved["tag_group"] == "shares_outstanding")]
    assert len(shares_a) == 1
    assert shares_a.iloc[0]["value"] == 1_000_000.0  # rank 1, not rank 2

    shares_b = resolved[(resolved["cik"] == CIK_B) & (resolved["tag_group"] == "shares_outstanding")]
    assert len(shares_b) == 1
    assert shares_b.iloc[0]["value"] == 2_000_000.0  # only the fallback was tagged; still used


def test_duration_tie_break_prefers_larger_qtrs():
    """When the same (adsh, tag_group) at the same rank has more than one
    qtrs span (e.g. a Q3 10-Q tagging both the 3-month and 9-month figure),
    _resolve_group_facts must keep the LARGER qtrs -- see its docstring."""
    facts = pd.DataFrame([
        _facts_row(CIK_A, "acc-1", "20201101", "20200930", "net_income", 0, 1, 100_000.0),
        _facts_row(CIK_A, "acc-1", "20201101", "20200930", "net_income", 0, 3, 900_000.0),
    ])
    resolved = fnd._resolve_group_facts(facts)
    assert len(resolved) == 1
    assert resolved.iloc[0]["qtrs"] == 3
    assert resolved.iloc[0]["value"] == 900_000.0


# ---------------------------------------------------------------------------
# Annualization (build_fundamentals's *4/qtrs contract)
# ---------------------------------------------------------------------------

def test_duration_values_are_annualized_at_write_time():
    facts = pd.DataFrame([
        _facts_row(CIK_A, "acc-1", "20200301", "20191231", "revenue", 0, 1, 1_000_000.0),  # 1 qtr -> *4
        _facts_row(CIK_A, "acc-2", "20200601", "20200430", "revenue", 0, 4, 8_000_000.0),  # FY -> *1
        _facts_row(CIK_A, "acc-1", "20200301", "20191231", "assets", 0, 0, 5_000_000.0),   # instant -> unchanged
    ])
    fund = fnd._resolve_group_facts(facts)
    is_duration = fund["tag_group"].isin(fnd.DURATION_GROUPS)
    fund.loc[is_duration, "value"] = fund.loc[is_duration, "value"] * (4.0 / fund.loc[is_duration, "qtrs"])

    rev_q1 = fund[(fund["adsh"] == "acc-1") & (fund["tag_group"] == "revenue")].iloc[0]
    assert rev_q1["value"] == pytest.approx(4_000_000.0)
    rev_fy = fund[(fund["adsh"] == "acc-2") & (fund["tag_group"] == "revenue")].iloc[0]
    assert rev_fy["value"] == pytest.approx(8_000_000.0)
    assets = fund[fund["tag_group"] == "assets"].iloc[0]
    assert assets["value"] == pytest.approx(5_000_000.0)


# ---------------------------------------------------------------------------
# Feature-level behavior
# ---------------------------------------------------------------------------

def test_market_cap_log_skipped_without_entry_open():
    """If events_df has no entry_open column, x_market_cap_log must be
    ABSENT from the output entirely -- not present as an all-NaN column."""
    events = make_events([(CIK_A, "2020-06-15")], with_entry_open=False)
    fund = make_fundamentals([(CIK_A, "2020-03-01", "2019-12-31", "shares_outstanding", 1_000_000.0)])
    feats = fnd.attach_fundamentals_features(events, fund)
    assert "x_market_cap_log" not in feats.columns
    assert set(feats.columns) == set(c for c in fnd.NEW_FEATURE_COLS if c != "x_market_cap_log")


def test_cash_runway_nan_when_profitable():
    """A profitable (or breakeven) company reports NaN runway, per spec --
    not an infinite or negative "runway"."""
    events = make_events([(CIK_A, "2020-06-15", 10.0)], with_entry_open=True)
    fund = make_fundamentals([
        (CIK_A, "2020-03-01", "2019-12-31", "cash", 1_000_000.0),
        (CIK_A, "2020-03-01", "2019-12-31", "net_income", 400_000.0),  # profitable
    ])
    feats = fnd.attach_fundamentals_features(events, fund)
    assert pd.isna(feats.iloc[0]["x_cash_runway_quarters"])


def test_cash_runway_capped_at_ceiling():
    """Near-zero burn must not blow the ratio up past the documented
    ceiling."""
    events = make_events([(CIK_A, "2020-06-15", 10.0)], with_entry_open=True)
    fund = make_fundamentals([
        (CIK_A, "2020-03-01", "2019-12-31", "cash", 1_000_000_000.0),
        (CIK_A, "2020-03-01", "2019-12-31", "net_income", -4.0),  # essentially breakeven, tiny loss
    ])
    feats = fnd.attach_fundamentals_features(events, fund)
    assert feats.iloc[0]["x_cash_runway_quarters"] == pytest.approx(fnd._RUNWAY_CEILING_QUARTERS)


def test_no_prior_fundamentals_at_all_gives_nan_not_zero():
    events = make_events([(CIK_A, "2020-01-01", 10.0)], with_entry_open=True)
    fund = make_fundamentals([
        (CIK_A, "2020-06-01", "2020-03-31", "assets", 1_000_000.0),  # only AFTER the event
    ])
    feats = fnd.attach_fundamentals_features(events, fund)
    row = feats.iloc[0]
    for c in fnd.NEW_FEATURE_COLS:
        assert pd.isna(row[c])


def test_row_count_and_order_preserved():
    events = make_events(
        [(CIK_A, "2020-06-15", 10.0), (CIK_B, "2019-01-01", 20.0), (CIK_A, "2018-01-01", 5.0)],
        with_entry_open=True,
    )
    fund = make_fundamentals([(CIK_A, "2019-01-01", "2018-12-31", "assets", 1_000_000.0)])
    feats = fnd.attach_fundamentals_features(events, fund)
    assert len(feats) == len(events)
    assert list(feats.index) == list(events.index)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
