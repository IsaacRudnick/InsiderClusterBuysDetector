"""Tests for tools.earnings_features, the point-in-time earnings/filing-
proximity feature set.

Runnable standalone via `python -m pytest tests/test_earnings_features.py -q`
from the repo root. No network and no real research dataset: every test
builds a synthetic events frame and a synthetic filing calendar directly, in
the shapes tools/earnings_features.compute_earnings_features expects.

The single most important test here is
`test_same_day_and_future_filings_have_zero_influence`. Every feature in
this module exists to describe an issuer's filing rhythm as it was knowable
at the moment of the event -- not as it turned out. If a same-day or
future-dated filing were allowed to leak into any of these columns, the
model would be quietly handed a peek at the future (did an 8-K happen today
or tomorrow?) dressed up as a "feature", and every backtest number built on
it would be unearned. This test proves the opposite: deleting those filings
from the calendar entirely must not change a single computed value.
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

from tools import earnings_features as ef  # noqa: E402


CIK_A = "0000000001"
CIK_B = "0000000002"


def make_calendar(rows) -> pd.DataFrame:
    """rows: list of (cik, form, filing_date_str, report_date_str_or_None)."""
    df = pd.DataFrame(rows, columns=["cik", "form", "filing_date", "report_date"])
    df["filing_date"] = pd.to_datetime(df["filing_date"])
    df["report_date"] = pd.to_datetime(df["report_date"])
    return df


def make_events(rows) -> pd.DataFrame:
    """rows: list of (issuer_cik, event_day_str)."""
    df = pd.DataFrame(rows, columns=["issuer_cik", "event_day"])
    df["event_day"] = pd.to_datetime(df["event_day"])
    return df


# ---------------------------------------------------------------------------
# The test that matters most
# ---------------------------------------------------------------------------

def test_same_day_and_future_filings_have_zero_influence():
    """Filings on or after event_day must not move any computed feature."""
    event_day = "2020-06-15"
    events = make_events([(CIK_A, event_day)])

    earlier_periodic = [
        (CIK_A, "10-K", "2019-03-01", None),
        (CIK_A, "10-Q", "2019-06-01", None),
        (CIK_A, "10-Q", "2019-09-01", None),
        (CIK_A, "10-Q", "2019-12-01", None),
        (CIK_A, "10-Q", "2020-03-01", None),
    ]
    earlier_8k = [
        (CIK_A, "8-K", "2020-05-20", None),
        (CIK_A, "8-K", "2020-06-01", None),
        (CIK_A, "8-K", "2020-06-10", None),
    ]
    # These must have zero influence: one exactly on event_day, one after it.
    contaminating = [
        (CIK_A, "10-Q", event_day, None),     # same-day periodic
        (CIK_A, "8-K", event_day, None),      # same-day 8-K
        (CIK_A, "10-Q", "2020-06-20", None),  # future periodic
        (CIK_A, "8-K", "2020-06-16", None),   # future 8-K
    ]

    calendar_with_future = make_calendar(earlier_periodic + earlier_8k + contaminating)
    calendar_without_future = make_calendar(earlier_periodic + earlier_8k)

    feats_with = ef.compute_earnings_features(events, calendar_with_future)
    feats_without = ef.compute_earnings_features(events, calendar_without_future)

    pd.testing.assert_frame_equal(feats_with, feats_without)

    # And sanity-check the actual values are the ones the earlier-only
    # filings imply, not NaN or some default.
    row = feats_without.iloc[0]
    assert row["x_days_since_last_periodic"] == (pd.Timestamp("2020-06-15") - pd.Timestamp("2020-03-01")).days
    assert row["x_days_since_last_8k"] == (pd.Timestamp("2020-06-15") - pd.Timestamp("2020-06-10")).days
    # 8-Ks strictly before event_day, within [event_day - 30, event_day):
    # 2020-05-20, 2020-06-01, 2020-06-10 all qualify (event_day - 30 = 2020-05-16).
    assert row["x_n_8k_trail30"] == 3


# ---------------------------------------------------------------------------
# NaN behaviour with no prior filings
# ---------------------------------------------------------------------------

def test_no_prior_filings_at_all_gives_nan_not_zero():
    """An issuer with zero history before the event must report NaN for the
    filing-recency features, not a default like 0 -- 0 would falsely read as
    'a filing happened today'."""
    events = make_events([(CIK_A, "2020-01-01")])
    calendar = make_calendar([
        (CIK_A, "10-Q", "2020-06-01", None),  # only AFTER the event
    ])
    feats = ef.compute_earnings_features(events, calendar)
    row = feats.iloc[0]
    assert pd.isna(row["x_days_since_last_periodic"])
    assert pd.isna(row["x_days_since_last_8k"])
    assert pd.isna(row["x_days_to_expected_periodic"])
    assert pd.isna(row["x_earnings_inside_horizon"])
    # No prior 8-Ks is a real, countable zero, not an unknown quantity.
    assert row["x_n_8k_trail30"] == 0


def test_unknown_issuer_gives_nan_across_the_board():
    events = make_events([(CIK_B, "2020-01-01")])
    calendar = make_calendar([(CIK_A, "10-Q", "2019-01-01", None)])
    feats = ef.compute_earnings_features(events, calendar)
    row = feats.iloc[0]
    for c in ef.NEW_FEATURE_COLS:
        assert pd.isna(row[c]), c
    assert row["x_n_8k_trail30"] == 0 or pd.isna(row["x_n_8k_trail30"])


# ---------------------------------------------------------------------------
# x_earnings_inside_horizon boundary logic
# ---------------------------------------------------------------------------

def test_earnings_inside_horizon_boundaries():
    """x_days_to_expected_periodic in [0, 31] -> 1.0, else 0.0, NaN only when
    the estimate itself is NaN. Constructed so the expected gap lands exactly
    at the boundaries: last periodic filing + median gap - event_day."""
    # Two periodic filings 91 days apart -> median gap 91 days.
    calendar = make_calendar([
        (CIK_A, "10-K", "2019-01-01", None),
        (CIK_A, "10-Q", "2019-04-02", None),  # 91 days after 2019-01-01
    ])
    last = pd.Timestamp("2019-04-02")
    gap = 91

    # event_day chosen so expected - event_day == 0 (inside horizon, lower edge)
    event_at_zero = last + pd.Timedelta(days=gap)
    # event_day chosen so expected - event_day == 31 (inside horizon, upper edge)
    event_at_31 = last + pd.Timedelta(days=gap - 31)
    # event_day chosen so expected - event_day == 32 (just outside)
    event_at_32 = last + pd.Timedelta(days=gap - 32)
    # event_day chosen so expected - event_day == -1 (just outside, past due)
    event_at_neg1 = last + pd.Timedelta(days=gap + 1)

    events = make_events([
        (CIK_A, event_at_zero.strftime("%Y-%m-%d")),
        (CIK_A, event_at_31.strftime("%Y-%m-%d")),
        (CIK_A, event_at_32.strftime("%Y-%m-%d")),
        (CIK_A, event_at_neg1.strftime("%Y-%m-%d")),
    ])
    feats = ef.compute_earnings_features(events, calendar)

    assert feats.iloc[0]["x_days_to_expected_periodic"] == 0
    assert feats.iloc[0]["x_earnings_inside_horizon"] == 1.0

    assert feats.iloc[1]["x_days_to_expected_periodic"] == 31
    assert feats.iloc[1]["x_earnings_inside_horizon"] == 1.0

    assert feats.iloc[2]["x_days_to_expected_periodic"] == 32
    assert feats.iloc[2]["x_earnings_inside_horizon"] == 0.0

    assert feats.iloc[3]["x_days_to_expected_periodic"] == -1
    assert feats.iloc[3]["x_earnings_inside_horizon"] == 0.0


# ---------------------------------------------------------------------------
# Fallback to a 91-day median gap
# ---------------------------------------------------------------------------

def test_fallback_91_day_gap_with_fewer_than_two_periodic_filings():
    """With exactly one (or zero) periodic filing before the event, there is
    no real gap to measure, so the estimate must fall back to a 91-day
    assumed cadence rather than being NaN or zero."""
    calendar = make_calendar([
        (CIK_A, "10-K", "2019-01-01", None),
    ])
    event_day = "2019-06-01"  # some arbitrary date after the single filing
    events = make_events([(CIK_A, event_day)])
    feats = ef.compute_earnings_features(events, calendar)
    row = feats.iloc[0]

    expected = pd.Timestamp("2019-01-01") + pd.Timedelta(days=91)
    expected_days_to = (expected - pd.Timestamp(event_day)).days
    assert row["x_days_to_expected_periodic"] == expected_days_to
    # x_days_since_last_periodic must still be computed normally.
    assert row["x_days_since_last_periodic"] == (pd.Timestamp(event_day) - pd.Timestamp("2019-01-01")).days


def test_zero_periodic_filings_before_event_gives_nan_estimate():
    """No periodic filing before the event at all -> nothing to anchor the
    estimate on -> NaN, not the 91-day fallback (which needs at least one
    filing to add the fallback gap to)."""
    calendar = make_calendar([
        (CIK_A, "8-K", "2019-01-01", None),  # only an 8-K, no periodic filing
    ])
    events = make_events([(CIK_A, "2019-06-01")])
    feats = ef.compute_earnings_features(events, calendar)
    row = feats.iloc[0]
    assert pd.isna(row["x_days_to_expected_periodic"])
    assert pd.isna(row["x_earnings_inside_horizon"])


# ---------------------------------------------------------------------------
# main()'s row-preservation contract (exercised at the compute_earnings_features level)
# ---------------------------------------------------------------------------

def test_output_length_and_index_match_input():
    events = make_events([(CIK_A, "2020-01-01"), (CIK_B, "2020-02-01"), (CIK_A, "2020-03-01")])
    calendar = make_calendar([(CIK_A, "10-K", "2019-01-01", None)])
    feats = ef.compute_earnings_features(events, calendar)
    assert len(feats) == len(events)
    assert list(feats.index) == list(events.index)
    assert list(feats.columns) == ef.NEW_FEATURE_COLS
