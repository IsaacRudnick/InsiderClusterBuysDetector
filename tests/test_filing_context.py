"""Tests for tools.filing_context, the point-in-time 13D/13G + 8-K item-code
feature set.

Runnable standalone via `python -m pytest tests/test_filing_context.py -q`
from the repo root. No network: every test builds synthetic ownership_df /
eightk_df / events_df directly, in the shapes
tools.filing_context.compute_filing_context_features expects, and calls that
pure function directly -- fetch_one/fetch_many (the network+cache layer) are
exercised only implicitly through their pure helpers (_parse_item_codes,
normalize_cik), matching how tools/filing_calendar.py and tools/issuer_meta.py
(this module's siblings) are tested: the network layer itself is untested,
the pure compute layer is tested thoroughly.

The single most important test here is
`test_same_day_and_future_filings_have_zero_influence`. A 13D/13G or 8-K
dated on or after event_day must not move any computed feature -- if it did,
the model would be handed a peek at a filing that, at decision time, had not
happened yet, and every backtest number touching that feature would be
unearned.
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

from tools import filing_context as fc  # noqa: E402


CIK_A = "0000000001"
CIK_B = "0000000002"


def make_events(rows) -> pd.DataFrame:
    """rows: list of (issuer_cik, event_day_str)."""
    df = pd.DataFrame(rows, columns=["issuer_cik", "event_day"])
    df["event_day"] = pd.to_datetime(df["event_day"])
    return df


def make_ownership(rows) -> pd.DataFrame:
    """rows: list of (cik, form, filing_date_str)."""
    df = pd.DataFrame(rows, columns=["cik", "form", "filing_date"])
    if not df.empty:
        df["filing_date"] = pd.to_datetime(df["filing_date"])
    return df


def make_eightk(rows) -> pd.DataFrame:
    """rows: list of (cik, filing_date_str, items_str_or_None)."""
    df = pd.DataFrame(rows, columns=["cik", "filing_date", "items"])
    if not df.empty:
        df["filing_date"] = pd.to_datetime(df["filing_date"])
    return df


# ---------------------------------------------------------------------------
# The test that matters most
# ---------------------------------------------------------------------------

def test_same_day_and_future_filings_have_zero_influence():
    """13D/13G/8-K filings on or after event_day must not move any feature."""
    event_day = "2020-06-15"
    events = make_events([(CIK_A, event_day)])

    earlier_ownership = [
        (CIK_A, "SC 13G", "2020-01-10"),
        (CIK_A, "SC 13D", "2020-05-01"),
    ]
    earlier_8k = [
        (CIK_A, "2020-04-01", "1.01,9.01"),
        (CIK_A, "2020-05-20", "5.02"),
        (CIK_A, "2020-06-10", "2.02,9.01"),
    ]
    # Must have zero influence: one exactly on event_day, one after it.
    contaminating_ownership = [
        (CIK_A, "SC 13D", event_day),          # same-day 13D
        (CIK_A, "SC 13D", "2020-06-20"),        # future 13D
    ]
    contaminating_8k = [
        (CIK_A, event_day, "5.02"),             # same-day 8-K
        (CIK_A, "2020-06-16", "1.01"),          # future 8-K
    ]

    own_with = make_ownership(earlier_ownership + contaminating_ownership)
    own_without = make_ownership(earlier_ownership)
    ek_with = make_eightk(earlier_8k + contaminating_8k)
    ek_without = make_eightk(earlier_8k)

    covered = {CIK_A}
    feats_with = fc.compute_filing_context_features(events, own_with, ek_with, covered)
    feats_without = fc.compute_filing_context_features(events, own_without, ek_without, covered)

    pd.testing.assert_frame_equal(feats_with, feats_without)

    # Sanity-check the values are the ones the earlier-only filings imply.
    row = feats_without.iloc[0]
    assert row["x_days_since_13d"] == (pd.Timestamp(event_day) - pd.Timestamp("2020-05-01")).days
    assert row["x_days_since_13g"] == (pd.Timestamp(event_day) - pd.Timestamp("2020-01-10")).days
    assert row["x_8k_departure_trail90"] == 1  # 2020-05-20
    assert row["x_8k_results_trail90"] == 1    # 2020-06-10
    assert row["x_8k_material_agmt_trail90"] == 1  # 2020-04-01


# ---------------------------------------------------------------------------
# days_since_13d / days_since_13g basic correctness
# ---------------------------------------------------------------------------

def test_days_since_uses_most_recent_prior_filing_per_family():
    events = make_events([(CIK_A, "2021-01-01")])
    ownership = make_ownership([
        (CIK_A, "SC 13D", "2020-10-01"),
        (CIK_A, "SC 13D/A", "2020-12-01"),   # more recent 13D-family filing
        (CIK_A, "SC 13G", "2020-06-01"),
    ])
    eightk = make_eightk([])
    feats = fc.compute_filing_context_features(events, ownership, eightk, {CIK_A})
    row = feats.iloc[0]
    assert row["x_days_since_13d"] == (pd.Timestamp("2021-01-01") - pd.Timestamp("2020-12-01")).days
    assert row["x_days_since_13g"] == (pd.Timestamp("2021-01-01") - pd.Timestamp("2020-06-01")).days


# ---------------------------------------------------------------------------
# trail180 / has_recent_13d window boundaries
# ---------------------------------------------------------------------------

def test_trail180_and_has_recent_13d_window_boundaries():
    """A 13D exactly 180 days before event_day is IN the trail180 window
    (inclusive lower bound); one exactly 90 days before is still "recent"."""
    event_day = pd.Timestamp("2021-01-01")
    at_180 = (event_day - pd.Timedelta(days=180)).strftime("%Y-%m-%d")
    at_181 = (event_day - pd.Timedelta(days=181)).strftime("%Y-%m-%d")
    at_90 = (event_day - pd.Timedelta(days=90)).strftime("%Y-%m-%d")
    at_91 = (event_day - pd.Timedelta(days=91)).strftime("%Y-%m-%d")

    events = make_events([(CIK_A, "2021-01-01")])
    ownership = make_ownership([
        (CIK_A, "SC 13D", at_180),
        (CIK_A, "SC 13D", at_181),
    ])
    feats = fc.compute_filing_context_features(events, ownership, make_eightk([]), {CIK_A})
    # Both are within trail180 (>= event_day - 180 catches at_180 but not at_181).
    assert feats.iloc[0]["x_n_13d_trail180"] == 1

    events2 = make_events([(CIK_A, "2021-01-01")])
    ownership_90 = make_ownership([(CIK_A, "SC 13D", at_90)])
    feats_90 = fc.compute_filing_context_features(events2, ownership_90, make_eightk([]), {CIK_A})
    assert feats_90.iloc[0]["x_has_recent_13d"] == 1.0

    ownership_91 = make_ownership([(CIK_A, "SC 13D", at_91)])
    feats_91 = fc.compute_filing_context_features(events2, ownership_91, make_eightk([]), {CIK_A})
    assert feats_91.iloc[0]["x_has_recent_13d"] == 0.0
    # Still counted in trail180 even though not "recent" (90-day window).
    assert feats_91.iloc[0]["x_n_13d_trail180"] == 1


# ---------------------------------------------------------------------------
# 8-K item-code bucketing, including the "other" bucket
# ---------------------------------------------------------------------------

def test_8k_item_codes_bucket_correctly_including_multi_item_and_other():
    events = make_events([(CIK_A, "2021-01-01")])
    eightk = make_eightk([
        (CIK_A, "2020-12-01", "5.02"),          # departure only
        (CIK_A, "2020-12-05", "2.02,9.01"),     # results + untracked -> results bucket
        (CIK_A, "2020-12-10", "1.01"),          # material agreement only
        (CIK_A, "2020-12-15", "5.02,2.02"),     # BOTH departure and results
        (CIK_A, "2020-12-20", "7.01,8.01"),     # neither tracked code -> other
        (CIK_A, "2020-12-22", None),            # no items at all -> other
        (CIK_A, "2020-12-24", ""),              # empty items -> other
    ])
    feats = fc.compute_filing_context_features(events, make_ownership([]), eightk, {CIK_A})
    row = feats.iloc[0]
    assert row["x_8k_departure_trail90"] == 2   # 12-01 and 12-15
    assert row["x_8k_results_trail90"] == 2     # 12-05 and 12-15
    assert row["x_8k_material_agmt_trail90"] == 1  # 12-10
    assert row["x_8k_other_trail90"] == 3        # 12-20, 12-22, 12-24


def test_8k_trail90_window_excludes_older_filings():
    event_day = pd.Timestamp("2021-01-01")
    at_90 = (event_day - pd.Timedelta(days=90)).strftime("%Y-%m-%d")
    at_91 = (event_day - pd.Timedelta(days=91)).strftime("%Y-%m-%d")
    events = make_events([(CIK_A, "2021-01-01")])
    eightk = make_eightk([
        (CIK_A, at_90, "5.02"),
        (CIK_A, at_91, "5.02"),
    ])
    feats = fc.compute_filing_context_features(events, make_ownership([]), eightk, {CIK_A})
    assert feats.iloc[0]["x_8k_departure_trail90"] == 1  # only at_90 is in [event-90, event)


# ---------------------------------------------------------------------------
# NaN (unknown) vs 0 (confirmed zero) via covered_ciks
# ---------------------------------------------------------------------------

def test_covered_issuer_with_no_filings_gives_zero_not_nan():
    """An issuer confirmed covered (found on SEC) but with zero 13D/13G/8-K
    history is a real 0 for count features, not a missing value."""
    events = make_events([(CIK_A, "2021-01-01")])
    feats = fc.compute_filing_context_features(
        events, make_ownership([]), make_eightk([]), covered_ciks={CIK_A},
    )
    row = feats.iloc[0]
    assert row["x_n_13d_trail180"] == 0
    assert row["x_has_recent_13d"] == 0.0
    assert row["x_8k_departure_trail90"] == 0
    assert row["x_8k_results_trail90"] == 0
    assert row["x_8k_material_agmt_trail90"] == 0
    assert row["x_8k_other_trail90"] == 0
    # "days since" quantities are undefined with no filing ever -> NaN.
    assert pd.isna(row["x_days_since_13d"])
    assert pd.isna(row["x_days_since_13g"])


def test_uncovered_issuer_gives_nan_across_the_board():
    """An issuer NOT in covered_ciks (never fetched, or SEC 404'd it) must
    report NaN everywhere -- including the count features, which would
    otherwise be indistinguishable from a confirmed zero."""
    events = make_events([(CIK_B, "2021-01-01")])
    ownership = make_ownership([(CIK_A, "SC 13D", "2020-01-01")])
    eightk = make_eightk([(CIK_A, "2020-01-01", "5.02")])
    feats = fc.compute_filing_context_features(events, ownership, eightk, covered_ciks={CIK_A})
    row = feats.iloc[0]
    for c in fc.NEW_FEATURE_COLS:
        assert pd.isna(row[c]), c


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def test_parse_item_codes():
    assert fc._parse_item_codes("5.02,9.01") == {"5.02", "9.01"}
    assert fc._parse_item_codes(" 5.02 , 9.01 ") == {"5.02", "9.01"}
    assert fc._parse_item_codes(None) == set()
    assert fc._parse_item_codes("") == set()


def test_normalize_cik():
    assert fc.normalize_cik("1800") == "0000001800"
    assert fc.normalize_cik(1800) == "0000001800"
    assert fc.normalize_cik("0000001800") == "0000001800"
    assert fc.normalize_cik(1800.0) == "0000001800"
    assert fc.normalize_cik(None) is None
    assert fc.normalize_cik("") is None
    assert fc.normalize_cik("not-a-cik") is None


# ---------------------------------------------------------------------------
# records_to_frames: flattening + coverage extraction
# ---------------------------------------------------------------------------

def test_records_to_frames_extracts_coverage_and_rows():
    records = [
        {
            "cik": CIK_A,
            "found": True,
            "ownership": [{"form": "SC 13D", "filing_date": "2020-01-01"}],
            "eightk": [{"filing_date": "2020-02-01", "items": "5.02"}],
        },
        {"cik": CIK_B, "found": False},
        None,  # a failed fetch that returned nothing
    ]
    own_df, ek_df, covered = fc.records_to_frames(records)
    assert covered == {CIK_A}
    assert len(own_df) == 1
    assert len(ek_df) == 1
    assert own_df.iloc[0]["form"] == "SC 13D"
    assert ek_df.iloc[0]["items"] == "5.02"


# ---------------------------------------------------------------------------
# main()'s row-preservation contract, exercised at the compute level
# ---------------------------------------------------------------------------

def test_output_length_and_index_match_input():
    events = make_events([(CIK_A, "2020-01-01"), (CIK_B, "2020-02-01"), (CIK_A, "2020-03-01")])
    ownership = make_ownership([(CIK_A, "SC 13D", "2019-01-01")])
    feats = fc.compute_filing_context_features(events, ownership, make_eightk([]), {CIK_A})
    assert len(feats) == len(events)
    assert list(feats.index) == list(events.index)
    assert list(feats.columns) == fc.NEW_FEATURE_COLS


def test_attach_features_row_and_column_contract_no_network():
    """attach_features(fetch_missing=False) must not touch the network and
    must preserve row count/order while adding exactly NEW_FEATURE_COLS."""
    events = make_events([(CIK_A, "2020-01-01"), (CIK_B, "2020-02-01")])
    out = fc.attach_features(events, fetch_missing=False)
    assert len(out) == len(events)
    assert list(out.index) == list(events.index)
    for c in fc.NEW_FEATURE_COLS:
        assert c in out.columns
