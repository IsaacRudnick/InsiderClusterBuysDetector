"""Tests for tools.short_interest -- FINRA short-interest normalization and
the point-in-time short-interest feature set.

Runnable standalone via `python -m pytest tests/test_short_interest.py -q`
from the repo root. No network: every test builds a synthetic short-interest
table directly in the canonical shape tools/short_interest.build_table
produces (symbol, settlement_date, short_shares, prev_short_shares,
adv_shares, days_to_cover, market_class), or a synthetic raw FINRA-shaped
CSV for the normalization tests. attach_features() is pure computation over
whatever table it is handed, so it needs no network to test.

The single most important test here is
`test_settled_and_published_on_or_after_event_has_zero_influence`. FINRA
compiles and publishes a settlement date's table roughly
PUBLICATION_LAG_DAYS calendar days after that settlement date -- a report
that "happened" before event_day is still not knowable on event_day if its
PUBLICATION date has not arrived yet. If a feature let such a report
through, the model would be handed short-interest information that, at
decision time, did not exist yet -- exactly the kind of leak that
manufactures a fake edge tools/short_interest.py's own module docstring
warns about. This test proves the opposite for the literal case the task
specifies (a report settled AND published on/after event_day), and the test
right after it proves the subtler case that motivates PUBLICATION_LAG_DAYS
existing at all: a report settled BEFORE event_day but not yet PUBLISHED by
event_day must also have zero influence, i.e. settlement date alone is not
a sufficient point-in-time cutoff.
"""

from __future__ import annotations

import io
import os
import sys

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools import short_interest as si  # noqa: E402


TICKER_A = "AAAA"
TICKER_B = "BBBB"


def make_short_interest(rows) -> pd.DataFrame:
    """rows: list of (symbol, settlement_date_str, short_shares,
    prev_short_shares, adv_shares, days_to_cover, market_class)."""
    df = pd.DataFrame(rows, columns=si.OUTPUT_COLUMNS)
    df["settlement_date"] = pd.to_datetime(df["settlement_date"])
    return df


def make_events(rows) -> pd.DataFrame:
    """rows: list of (ticker, event_day_str)."""
    df = pd.DataFrame(rows, columns=["ticker", "event_day"])
    return df


# ---------------------------------------------------------------------------
# The test that matters most: the literal task spec
# ---------------------------------------------------------------------------

def test_settled_and_published_on_or_after_event_has_zero_influence():
    """A report settled exactly ON event_day (so it is also published well
    after event_day, since publication = settlement + lag) must not move a
    single computed feature."""
    event_day = "2021-06-15"
    events = make_events([(TICKER_A, event_day)])

    earlier = [
        (TICKER_A, "2021-04-15", 100_000, 90_000, 50_000, 2.0, "OTC"),
        # Published 2021-04-30 + 8d = 2021-05-08, well before event_day -> eligible.
        (TICKER_A, "2021-04-30", 120_000, 100_000, 60_000, 2.0, "OTC"),
    ]
    # Settled (and therefore published) ON OR AFTER event_day -- must have
    # zero influence no matter what implausible values it carries.
    contaminating = [
        (TICKER_A, event_day, 999_999_999, 1, 1, 0.01, "OTC"),
    ]

    si_with_future = make_short_interest(earlier + contaminating)
    si_without_future = make_short_interest(earlier)

    feats_with = si.attach_features(events, si_with_future)
    feats_without = si.attach_features(events, si_without_future)

    pd.testing.assert_frame_equal(feats_with, feats_without)

    # And sanity-check the surviving value really is the 2021-04-30 report,
    # not NaN or some default.
    row = feats_without.iloc[0]
    assert row["x_short_pct_of_adv"] == pytest.approx(120_000 / 60_000)
    assert row["x_short_days_to_cover"] == 2.0


# ---------------------------------------------------------------------------
# The subtler case: settled before event_day, but not yet PUBLISHED
# ---------------------------------------------------------------------------

def test_settled_before_but_not_yet_published_has_zero_influence():
    """A report settled a few days before event_day, but whose PUBLICATION
    date (settlement + PUBLICATION_LAG_DAYS) has not arrived by event_day,
    must be excluded exactly as if it had not settled yet. This is the case
    that makes settlement-date-alone filtering wrong: naive code that only
    checked `settlement_date < event_day` would let this row through."""
    lag = si.PUBLICATION_LAG_DAYS
    event_day = pd.Timestamp("2022-03-01")

    # Settled 3 days before event_day: publication = settle + lag, which for
    # lag=8 lands 5 days AFTER event_day. Must be excluded.
    not_yet_published = event_day - pd.Timedelta(days=3)
    # Settled far enough before event_day that publication lands strictly
    # before event_day. Must be the row features are computed from.
    eligible = event_day - pd.Timedelta(days=lag + 5)

    rows_with_leak = [
        (TICKER_A, eligible.strftime("%Y-%m-%d"), 50_000, 40_000, 25_000, 2.0, "OTC"),
        (TICKER_A, not_yet_published.strftime("%Y-%m-%d"), 9_000_000, 1, 1, 0.01, "OTC"),
    ]
    rows_without_leak = rows_with_leak[:1]

    events = make_events([(TICKER_A, event_day.strftime("%Y-%m-%d"))])
    feats_with = si.attach_features(events, make_short_interest(rows_with_leak))
    feats_without = si.attach_features(events, make_short_interest(rows_without_leak))

    pd.testing.assert_frame_equal(feats_with, feats_without)
    assert feats_without.iloc[0]["x_short_pct_of_adv"] == pytest.approx(50_000 / 25_000)


def test_report_becomes_eligible_exactly_when_published_strictly_before_event():
    """Boundary check: publication_date == event_day - 1 day is eligible;
    publication_date == event_day is not. Constructed so settlement_date
    lands exactly on those publication boundaries once PUBLICATION_LAG_DAYS
    is subtracted back out."""
    lag = si.PUBLICATION_LAG_DAYS
    event_day = pd.Timestamp("2023-01-20")

    settle_pub_equals_event = event_day - pd.Timedelta(days=lag)          # publication == event_day
    settle_pub_one_day_before = event_day - pd.Timedelta(days=lag + 1)    # publication == event_day - 1

    events = make_events([(TICKER_A, event_day.strftime("%Y-%m-%d"))])

    excluded_table = make_short_interest([
        (TICKER_A, settle_pub_equals_event.strftime("%Y-%m-%d"), 10_000, 10_000, 10_000, 1.0, "OTC"),
    ])
    included_table = make_short_interest([
        (TICKER_A, settle_pub_one_day_before.strftime("%Y-%m-%d"), 30_000, 10_000, 10_000, 3.0, "OTC"),
    ])

    feats_excluded = si.attach_features(events, excluded_table)
    assert feats_excluded.iloc[0].isna().all()

    feats_included = si.attach_features(events, included_table)
    assert feats_included.iloc[0]["x_short_days_to_cover"] == 3.0


# ---------------------------------------------------------------------------
# NaN behaviour with no eligible report
# ---------------------------------------------------------------------------

def test_unknown_ticker_gives_nan_across_the_board():
    events = make_events([(TICKER_B, "2020-01-01")])
    table = make_short_interest([(TICKER_A, "2019-01-01", 1, 1, 1, 1.0, "OTC")])
    feats = si.attach_features(events, table)
    row = feats.iloc[0]
    for c in si.X_COLS:
        assert pd.isna(row[c]), c


def test_no_eligible_report_at_all_gives_nan_not_zero():
    events = make_events([(TICKER_A, "2018-01-01")])
    table = make_short_interest([(TICKER_A, "2019-06-01", 1, 1, 1, 1.0, "OTC")])  # only AFTER
    feats = si.attach_features(events, table)
    row = feats.iloc[0]
    for c in si.X_COLS:
        assert pd.isna(row[c]), c


# ---------------------------------------------------------------------------
# Feature math
# ---------------------------------------------------------------------------

def test_x_short_pct_of_adv_guards_zero_adv():
    """adv_shares == 0 must produce NaN, not inf -- FINRA's own zero-ADV
    rows exist for real (see DAYS_TO_COVER_SENTINEL's docstring)."""
    event_day = "2021-01-01"
    events = make_events([(TICKER_A, event_day)])
    table = make_short_interest([
        (TICKER_A, "2020-11-01", 5_000, 4_000, 0, np.nan, "OTC"),
    ])
    feats = si.attach_features(events, table)
    row = feats.iloc[0]
    assert pd.isna(row["x_short_pct_of_adv"])
    assert pd.isna(row["x_short_days_to_cover"])  # NaN carried straight through


def test_x_short_change_2m_is_log1p_difference():
    event_day = "2021-01-01"
    events = make_events([(TICKER_A, event_day)])
    current, previous = 30_000, 10_000
    table = make_short_interest([
        (TICKER_A, "2020-11-01", current, previous, 15_000, 2.0, "OTC"),
    ])
    feats = si.attach_features(events, table)
    expected = np.log1p(current) - np.log1p(previous)
    assert feats.iloc[0]["x_short_change_2m"] == pytest.approx(expected)


def test_x_short_report_age_days_is_calendar_days_since_settlement():
    event_day = pd.Timestamp("2021-03-01")
    settle = event_day - pd.Timedelta(days=30)
    events = make_events([(TICKER_A, event_day.strftime("%Y-%m-%d"))])
    table = make_short_interest([
        (TICKER_A, settle.strftime("%Y-%m-%d"), 1_000, 900, 500, 1.0, "OTC"),
    ])
    feats = si.attach_features(events, table)
    assert feats.iloc[0]["x_short_report_age_days"] == 30


def test_latest_eligible_report_wins_over_older_ones():
    """Two eligible reports -- the more recent (but still eligible) one's
    values must be used, not the older one's."""
    lag = si.PUBLICATION_LAG_DAYS
    event_day = pd.Timestamp("2022-06-01")
    older = event_day - pd.Timedelta(days=lag + 40)
    newer = event_day - pd.Timedelta(days=lag + 5)

    events = make_events([(TICKER_A, event_day.strftime("%Y-%m-%d"))])
    table = make_short_interest([
        (TICKER_A, older.strftime("%Y-%m-%d"), 1_000, 900, 500, 1.0, "OTC"),
        (TICKER_A, newer.strftime("%Y-%m-%d"), 8_000, 7_000, 4_000, 4.0, "OTC"),
    ])
    feats = si.attach_features(events, table)
    row = feats.iloc[0]
    assert row["x_short_days_to_cover"] == 4.0
    assert row["x_short_pct_of_adv"] == pytest.approx(8_000 / 4_000)


# ---------------------------------------------------------------------------
# Output shape contract
# ---------------------------------------------------------------------------

def test_output_length_and_index_match_input():
    events = make_events([(TICKER_A, "2020-01-01"), (TICKER_B, "2020-02-01"), (TICKER_A, "2020-03-01")])
    table = make_short_interest([(TICKER_A, "2019-01-01", 1, 1, 1, 1.0, "OTC")])
    feats = si.attach_features(events, table)
    assert len(feats) == len(events)
    assert list(feats.index) == list(events.index)
    assert list(feats.columns) == si.X_COLS


def test_empty_short_interest_table_gives_all_nan_no_crash():
    events = make_events([(TICKER_A, "2020-01-01")])
    empty = pd.DataFrame(columns=si.OUTPUT_COLUMNS)
    feats = si.attach_features(events, empty)
    assert len(feats) == 1
    assert feats.iloc[0].isna().all()


# ---------------------------------------------------------------------------
# Raw FINRA CSV -> canonical table (build_table)
# ---------------------------------------------------------------------------

_RAW_CSV = (
    "accountingYearMonthNumber,symbolCode,issueName,issuerServicesGroupExchangeCode,"
    "marketClassCode,currentShortPositionQuantity,previousShortPositionQuantity,"
    "stockSplitFlag,averageDailyVolumeQuantity,daysToCoverQuantity,revisionFlag,"
    "changePercent,changePreviousNumber,settlementDate\n"
    '20200415,A,Agilent Technologies Inc.,A,NYSE,4851353,4767556,,2012318,2.41,,1.76,83797,2020-04-15\n'
    # Zero-ADV, FINRA's 999.99 "undefined days to cover" sentinel.
    '20200415,ZZZZ,Zero Volume Co,S,OTC,79128,100172,,0,999.99,,-21.01,-21044,2020-04-15\n'
)


def test_build_table_renames_and_selects_canonical_columns():
    raw = pd.read_csv(io.StringIO(_RAW_CSV))
    table = si.build_table(raw)
    assert list(table.columns) == si.OUTPUT_COLUMNS
    a_row = table[table["symbol"] == "A"].iloc[0]
    assert a_row["short_shares"] == 4851353
    assert a_row["prev_short_shares"] == 4767556
    assert a_row["adv_shares"] == 2012318
    assert a_row["days_to_cover"] == pytest.approx(2.41)
    assert a_row["market_class"] == "NYSE"
    assert a_row["settlement_date"] == pd.Timestamp("2020-04-15")


def test_build_table_converts_days_to_cover_sentinel_to_nan():
    """999.99 is FINRA's "undefined" flag (zero ADV), not a real value --
    it must become NaN, not a literal ~1000-days-to-cover reading."""
    raw = pd.read_csv(io.StringIO(_RAW_CSV))
    table = si.build_table(raw)
    z_row = table[table["symbol"] == "ZZZZ"].iloc[0]
    assert z_row["adv_shares"] == 0
    assert pd.isna(z_row["days_to_cover"])


def test_build_table_filters_to_requested_tickers():
    raw = pd.read_csv(io.StringIO(_RAW_CSV))
    table = si.build_table(raw, tickers={"A"})
    assert set(table["symbol"]) == {"A"}


def test_build_table_empty_input_returns_empty_canonical_frame():
    table = si.build_table(pd.DataFrame())
    assert list(table.columns) == si.OUTPUT_COLUMNS
    assert len(table) == 0
