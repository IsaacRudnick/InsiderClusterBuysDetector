"""Tests for tools.form4_extras, the point-in-time feature set mined from the
FULL (not just code-P) Form 4/4A transaction record in parse_cache/.

Runnable standalone via `python -m pytest tests/test_form4_extras.py -q` from
the repo root. No network, and no dependency on the real parse_cache/
directory -- scan-level tests (_scan_one / build_form4_scan) write small
crafted JSON files to tmp_path and point the scanner at that directory;
feature-level tests (attach_features) pass small hand-built tx_df/
owner_first_df frames directly, matching backtest.sales_history's tests'
"synthetic frame, not a real scan" convention.

The single most important test in this file is
test_filing_after_event_day_is_invisible_even_if_transaction_is_earlier: it
proves the point-in-time discipline the module docstring requires -- a
transaction whose TRANSACTION date sits inside every trailing window but
whose FILING date lands on/after event_day must not move any x_ feature,
matching the same discipline backtest.research (Section D/F) and
tools.earnings_features already enforce elsewhere in this codebase.
"""

from __future__ import annotations

import json
import os
from datetime import date, timedelta

import pandas as pd
import pytest

from tools import form4_extras as fx

CIK_A = "0000000001"
CIK_B = "0000000002"


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def make_tx_row(
    *, issuer_cik: str = CIK_A, owner_key: str = "OWNER_X",
    code: str, acquired_disposed: str,
    transaction_date: date, filing_date: date,
    shares: float = 100.0, price_per_share: float = 10.0, value: float | None = None,
    is_10b5_1: bool = False, adsh: str = "0001",
) -> dict:
    val = value if value is not None else shares * price_per_share
    return {
        "adsh": adsh, "issuer_cik": issuer_cik, "owner_key": owner_key,
        "code": code, "acquired_disposed": acquired_disposed,
        "transaction_date": transaction_date, "filing_date": filing_date,
        "shares": shares, "price_per_share": price_per_share, "value": val,
        "is_10b5_1": is_10b5_1,
    }


def make_tx_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=fx._TX_COLUMNS)


def make_owner_row(*, issuer_cik: str = CIK_A, owner_key: str, filing_date: date) -> dict:
    return {"issuer_cik": issuer_cik, "owner_key": owner_key, "filing_date": filing_date}


def make_owner_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=fx._OWNER_COLUMNS)


def make_events(rows: list[tuple[str, date]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["issuer_cik", "event_day"])


# ---------------------------------------------------------------------------
# 1. THE load-bearing test: point-in-time / no leakage from filing lag
# ---------------------------------------------------------------------------
def test_filing_after_event_day_is_invisible_even_if_transaction_is_earlier():
    """A sale/grant/exercise dated well inside the trailing window, but
    FILED on or after event_day, must leave every feature untouched -- the
    module's core point-in-time rule (STRICTLY BEFORE event_day, not <=)."""
    event_day = date(2022, 6, 15)
    tx_date = event_day - timedelta(days=10)  # well inside any trailing window

    late_filed = make_tx_df([
        make_tx_row(code="S", acquired_disposed="D", transaction_date=tx_date, filing_date=event_day),  # ON event_day -- must NOT count
        make_tx_row(code="A", acquired_disposed="A", transaction_date=tx_date,
                    filing_date=event_day + timedelta(days=3)),  # AFTER event_day -- must NOT count
        make_tx_row(code="M", acquired_disposed="A", transaction_date=tx_date, filing_date=event_day),
        make_tx_row(code="P", acquired_disposed="A", transaction_date=tx_date, filing_date=event_day,
                    is_10b5_1=True),
    ])
    on_time_filed = make_tx_df([
        make_tx_row(code="S", acquired_disposed="D", transaction_date=tx_date,
                    filing_date=event_day - timedelta(days=1)),
    ])
    events = make_events([(CIK_A, event_day)])
    owner_first_empty = make_owner_df([])

    feats_late = fx.attach_features(events, tx_df=late_filed, owner_first_df=owner_first_empty)
    row = feats_late.iloc[0]
    assert row["x_insider_grants_trail180"] == 0.0
    assert row["x_option_exercises_trail180"] == 0.0
    assert row["x_exercise_and_hold_trail180"] == 0.0
    assert row["x_insider_sales_trail180"] == 0.0
    assert pd.isna(row["x_sale_to_buy_ratio"])  # no visible buys either
    assert pd.isna(row["x_frac_buys_10b5_1_trail180"])

    # Control: the same shape, but filed strictly before event_day, IS visible.
    feats_ontime = fx.attach_features(events, tx_df=on_time_filed, owner_first_df=owner_first_empty)
    assert feats_ontime.iloc[0]["x_insider_sales_trail180"] == 1.0


def test_new_insider_first_filing_on_event_day_does_not_count():
    """x_n_new_insiders_trail365 uses the SAME strictly-before rule on the
    owner's first-ever filing date -- a first filing dated exactly on
    event_day is not yet known as of event_day."""
    event_day = date(2022, 6, 15)
    owner_first = make_owner_df([
        make_owner_row(owner_key="NEW1", filing_date=event_day - timedelta(days=30)),  # visible
        make_owner_row(owner_key="NEW2", filing_date=event_day),  # NOT visible (same day)
        make_owner_row(owner_key="NEW3", filing_date=event_day + timedelta(days=1)),  # NOT visible (future)
    ])
    events = make_events([(CIK_A, event_day)])
    feats = fx.attach_features(events, tx_df=make_tx_df([]), owner_first_df=owner_first)
    assert feats.iloc[0]["x_n_new_insiders_trail365"] == 1.0


# ---------------------------------------------------------------------------
# 2. Trailing-window counts
# ---------------------------------------------------------------------------
def test_grants_exercises_sales_counted_within_trail180():
    event_day = date(2022, 6, 15)
    tx_df = make_tx_df([
        make_tx_row(code="A", acquired_disposed="A",
                    transaction_date=event_day - timedelta(days=10), filing_date=event_day - timedelta(days=5)),
        make_tx_row(code="A", acquired_disposed="A",
                    transaction_date=event_day - timedelta(days=200), filing_date=event_day - timedelta(days=195)),  # outside 180d
        make_tx_row(code="M", acquired_disposed="A",
                    transaction_date=event_day - timedelta(days=20), filing_date=event_day - timedelta(days=1)),
        make_tx_row(code="S", acquired_disposed="D",
                    transaction_date=event_day - timedelta(days=2), filing_date=event_day - timedelta(days=1)),
        make_tx_row(code="S", acquired_disposed="D",
                    transaction_date=event_day - timedelta(days=3), filing_date=event_day - timedelta(days=1)),
    ])
    events = make_events([(CIK_A, event_day)])
    row = fx.attach_features(events, tx_df=tx_df, owner_first_df=make_owner_df([])).iloc[0]
    assert row["x_insider_grants_trail180"] == 1.0  # the 200-day-old one is excluded
    assert row["x_option_exercises_trail180"] == 1.0
    assert row["x_insider_sales_trail180"] == 2.0


def test_transaction_exactly_180_days_old_is_excluded_boundary():
    """Window is [event_day - 180, event_day) -- a transaction dated exactly
    event_day - 180 is INSIDE (>=), one dated event_day is OUTSIDE (< only)."""
    event_day = date(2022, 6, 15)
    tx_df = make_tx_df([
        make_tx_row(code="A", acquired_disposed="A",
                    transaction_date=event_day - timedelta(days=180), filing_date=event_day - timedelta(days=1)),
    ])
    events = make_events([(CIK_A, event_day)])
    row = fx.attach_features(events, tx_df=tx_df, owner_first_df=make_owner_df([])).iloc[0]
    assert row["x_insider_grants_trail180"] == 1.0


def test_different_issuer_never_counted():
    event_day = date(2022, 6, 15)
    tx_df = make_tx_df([
        make_tx_row(issuer_cik=CIK_B, code="S", acquired_disposed="D",
                    transaction_date=event_day - timedelta(days=2), filing_date=event_day - timedelta(days=1)),
    ])
    events = make_events([(CIK_A, event_day)])
    row = fx.attach_features(events, tx_df=tx_df, owner_first_df=make_owner_df([])).iloc[0]
    assert row["x_insider_sales_trail180"] == 0.0


def test_issuer_with_no_history_gives_real_zero_counts_but_nan_ratios():
    event_day = date(2022, 6, 15)
    events = make_events([(CIK_A, event_day)])
    row = fx.attach_features(events, tx_df=make_tx_df([]), owner_first_df=make_owner_df([])).iloc[0]
    assert row["x_insider_grants_trail180"] == 0.0
    assert row["x_option_exercises_trail180"] == 0.0
    assert row["x_exercise_and_hold_trail180"] == 0.0
    assert row["x_insider_sales_trail180"] == 0.0
    assert pd.isna(row["x_sale_to_buy_ratio"])
    assert pd.isna(row["x_frac_buys_10b5_1_trail180"])
    assert row["x_n_new_insiders_trail365"] == 0.0


# ---------------------------------------------------------------------------
# 3. x_sale_to_buy_ratio
# ---------------------------------------------------------------------------
def test_sale_to_buy_ratio_normal_case():
    event_day = date(2022, 6, 15)
    tx_df = make_tx_df([
        make_tx_row(code="P", acquired_disposed="A", value=1000.0,
                    transaction_date=event_day - timedelta(days=5), filing_date=event_day - timedelta(days=1)),
        make_tx_row(code="S", acquired_disposed="D", value=500.0,
                    transaction_date=event_day - timedelta(days=3), filing_date=event_day - timedelta(days=1)),
    ])
    events = make_events([(CIK_A, event_day)])
    row = fx.attach_features(events, tx_df=tx_df, owner_first_df=make_owner_df([])).iloc[0]
    assert row["x_sale_to_buy_ratio"] == pytest.approx(0.5)


def test_sale_to_buy_ratio_nan_when_no_trailing_buys():
    event_day = date(2022, 6, 15)
    tx_df = make_tx_df([
        make_tx_row(code="S", acquired_disposed="D", value=500.0,
                    transaction_date=event_day - timedelta(days=3), filing_date=event_day - timedelta(days=1)),
    ])
    events = make_events([(CIK_A, event_day)])
    row = fx.attach_features(events, tx_df=tx_df, owner_first_df=make_owner_df([])).iloc[0]
    assert pd.isna(row["x_sale_to_buy_ratio"])


# ---------------------------------------------------------------------------
# 4. x_frac_buys_10b5_1_trail180
# ---------------------------------------------------------------------------
def test_frac_buys_10b5_1_computed_correctly():
    event_day = date(2022, 6, 15)
    tx_df = make_tx_df([
        make_tx_row(code="P", acquired_disposed="A", is_10b5_1=True,
                    transaction_date=event_day - timedelta(days=5), filing_date=event_day - timedelta(days=1)),
        make_tx_row(code="P", acquired_disposed="A", is_10b5_1=False,
                    transaction_date=event_day - timedelta(days=4), filing_date=event_day - timedelta(days=1)),
        make_tx_row(code="P", acquired_disposed="A", is_10b5_1=False,
                    transaction_date=event_day - timedelta(days=3), filing_date=event_day - timedelta(days=1)),
        make_tx_row(code="P", acquired_disposed="A", is_10b5_1=False,
                    transaction_date=event_day - timedelta(days=3), filing_date=event_day - timedelta(days=1)),
    ])
    events = make_events([(CIK_A, event_day)])
    row = fx.attach_features(events, tx_df=tx_df, owner_first_df=make_owner_df([])).iloc[0]
    assert row["x_frac_buys_10b5_1_trail180"] == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# 5. x_exercise_and_hold_trail180
# ---------------------------------------------------------------------------
def test_exercise_followed_by_quick_sale_is_not_counted_as_held():
    event_day = date(2022, 6, 15)
    ex_date = event_day - timedelta(days=20)
    tx_df = make_tx_df([
        make_tx_row(owner_key="OWNER_X", code="M", acquired_disposed="A",
                    transaction_date=ex_date, filing_date=ex_date + timedelta(days=1)),
        make_tx_row(owner_key="OWNER_X", code="S", acquired_disposed="D",
                    transaction_date=ex_date + timedelta(days=2), filing_date=ex_date + timedelta(days=3)),
    ])
    events = make_events([(CIK_A, event_day)])
    row = fx.attach_features(events, tx_df=tx_df, owner_first_df=make_owner_df([])).iloc[0]
    assert row["x_option_exercises_trail180"] == 1.0
    assert row["x_exercise_and_hold_trail180"] == 0.0  # sold, not held


def test_exercise_with_no_followon_sale_is_counted_as_held():
    event_day = date(2022, 6, 15)
    ex_date = event_day - timedelta(days=20)  # comfortably 5+ days before event_day
    tx_df = make_tx_df([
        make_tx_row(owner_key="OWNER_X", code="M", acquired_disposed="A",
                    transaction_date=ex_date, filing_date=ex_date + timedelta(days=1)),
    ])
    events = make_events([(CIK_A, event_day)])
    row = fx.attach_features(events, tx_df=tx_df, owner_first_df=make_owner_df([])).iloc[0]
    assert row["x_exercise_and_hold_trail180"] == 1.0


def test_exercise_too_recent_to_judge_is_excluded_not_counted_as_held():
    """An exercise whose 5-day follow-on window has not fully elapsed as of
    event_day must NOT be counted as 'held' just because no sale has been
    seen yet -- that would be an unearned peek at the future (see module
    docstring). It also must not be counted toward x_exercise_and_hold, even
    though it IS counted toward the raw x_option_exercises_trail180."""
    event_day = date(2022, 6, 15)
    ex_date = event_day - timedelta(days=2)  # only 2 days before event_day -- window incomplete
    tx_df = make_tx_df([
        make_tx_row(owner_key="OWNER_X", code="M", acquired_disposed="A",
                    transaction_date=ex_date, filing_date=ex_date),
    ])
    events = make_events([(CIK_A, event_day)])
    row = fx.attach_features(events, tx_df=tx_df, owner_first_df=make_owner_df([])).iloc[0]
    assert row["x_option_exercises_trail180"] == 1.0
    assert row["x_exercise_and_hold_trail180"] == 0.0


def test_followon_sale_by_a_different_owner_does_not_count_against_the_exercise():
    event_day = date(2022, 6, 15)
    ex_date = event_day - timedelta(days=20)
    tx_df = make_tx_df([
        make_tx_row(owner_key="OWNER_X", code="M", acquired_disposed="A",
                    transaction_date=ex_date, filing_date=ex_date + timedelta(days=1)),
        make_tx_row(owner_key="OWNER_Y", code="S", acquired_disposed="D",  # a DIFFERENT owner sold
                    transaction_date=ex_date + timedelta(days=1), filing_date=ex_date + timedelta(days=2)),
    ])
    events = make_events([(CIK_A, event_day)])
    row = fx.attach_features(events, tx_df=tx_df, owner_first_df=make_owner_df([])).iloc[0]
    assert row["x_exercise_and_hold_trail180"] == 1.0  # OWNER_X's exercise is unaffected


def test_followon_sale_filed_late_does_not_count_against_the_exercise():
    """The follow-on sale itself must also obey the filing-lag gate: a sale
    filed on/after event_day cannot be used to prove the exercise was sold,
    even if its transaction_date falls inside the 5-day window."""
    event_day = date(2022, 6, 15)
    ex_date = event_day - timedelta(days=20)
    tx_df = make_tx_df([
        make_tx_row(owner_key="OWNER_X", code="M", acquired_disposed="A",
                    transaction_date=ex_date, filing_date=ex_date + timedelta(days=1)),
        make_tx_row(owner_key="OWNER_X", code="S", acquired_disposed="D",
                    transaction_date=ex_date + timedelta(days=1), filing_date=event_day + timedelta(days=1)),
    ])
    events = make_events([(CIK_A, event_day)])
    row = fx.attach_features(events, tx_df=tx_df, owner_first_df=make_owner_df([])).iloc[0]
    assert row["x_exercise_and_hold_trail180"] == 1.0  # late-filed sale is invisible -> looks held


# ---------------------------------------------------------------------------
# 6. Row order / index preservation
# ---------------------------------------------------------------------------
def test_output_length_and_index_match_input():
    events = make_events([
        (CIK_A, date(2022, 1, 1)), (CIK_B, date(2022, 2, 1)), (CIK_A, date(2022, 3, 1)),
    ])
    feats = fx.attach_features(events, tx_df=make_tx_df([]), owner_first_df=make_owner_df([]))
    assert len(feats) == len(events)
    assert list(feats.index) == list(events.index)
    assert list(feats.columns) == fx.NEW_FEATURE_COLS


# ---------------------------------------------------------------------------
# 7. Local parse_cache/ scanning
# ---------------------------------------------------------------------------
def _write_filing(
    tmp_path, adsh: str, *, form_type: str = "4", filing_date: str = "20220610",
    issuer_cik: str = CIK_A, owner_cik: str = "0000000099",
    transactions: list[dict] | None = None,
) -> str:
    payload = {
        "adsh": adsh, "form_type": form_type, "filing_date": filing_date,
        "issuer": {"cik": issuer_cik, "name": "ISSUER CO", "ticker": "ABC"},
        "owners": [{
            "cik": owner_cik, "name": "Some Insider", "is_director": True,
            "is_officer": False, "is_ten_percent_owner": False, "is_other": False,
            "officer_title": "",
        }],
        "transactions": transactions or [],
        "holdings": [], "filing_url": "https://example.com", "source": "xml",
    }
    path = os.path.join(str(tmp_path), f"{adsh}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    return path


def test_scan_one_keeps_only_the_four_qualifying_code_pairs(tmp_path):
    path = _write_filing(tmp_path, "A1", transactions=[
        {"date": "2022-06-01", "code": "P", "acquired_disposed": "A", "shares": 10.0, "price_per_share": 1.0},
        {"date": "2022-06-01", "code": "A", "acquired_disposed": "A", "shares": 10.0, "price_per_share": 1.0},
        {"date": "2022-06-01", "code": "M", "acquired_disposed": "A", "shares": 10.0, "price_per_share": 1.0},
        {"date": "2022-06-01", "code": "S", "acquired_disposed": "D", "shares": 10.0, "price_per_share": 1.0},
        {"date": "2022-06-01", "code": "F", "acquired_disposed": "D", "shares": 10.0, "price_per_share": 1.0},
        {"date": "2022-06-01", "code": "G", "acquired_disposed": "A", "shares": 10.0, "price_per_share": 1.0},
    ])
    rows, appearance = fx._scan_one(path)
    assert {(r["code"], r["acquired_disposed"]) for r in rows} == {("P", "A"), ("A", "A"), ("M", "A"), ("S", "D")}
    assert appearance == {"issuer_cik": CIK_A, "owner_key": "0000000099", "filing_date": date(2022, 6, 10)}


def test_scan_one_appearance_survives_even_with_zero_qualifying_transactions(tmp_path):
    """An owner's first-ever filing might carry only a non-qualifying code
    (e.g. a gift) or no transactions at all -- the appearance record must
    still be produced so x_n_new_insiders_trail365 doesn't undercount."""
    path = _write_filing(tmp_path, "A2", transactions=[
        {"date": "2022-06-01", "code": "G", "acquired_disposed": "A", "shares": 10.0, "price_per_share": 1.0},
    ])
    rows, appearance = fx._scan_one(path)
    assert rows == []
    assert appearance is not None
    assert appearance["owner_key"] == "0000000099"


def test_scan_one_skips_non_form4(tmp_path):
    path = _write_filing(tmp_path, "F3", form_type="3", transactions=[
        {"date": "2022-06-01", "code": "P", "acquired_disposed": "A", "shares": 10.0, "price_per_share": 1.0},
    ])
    rows, appearance = fx._scan_one(path)
    assert rows == []
    assert appearance is None


def test_scan_one_handles_4a_amendment(tmp_path):
    path = _write_filing(tmp_path, "AMEND", form_type="4/A", transactions=[
        {"date": "2022-06-01", "code": "P", "acquired_disposed": "A", "shares": 10.0, "price_per_share": 1.0},
    ])
    rows, appearance = fx._scan_one(path)
    assert len(rows) == 1
    assert appearance is not None


def test_scan_one_skips_malformed_json(tmp_path):
    path = os.path.join(str(tmp_path), "BAD.json")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write('{"adsh": "BAD", "form_type": "4"}}')  # trailing stray brace
    rows, appearance = fx._scan_one(path)
    assert rows == []
    assert appearance is None


def test_scan_one_skips_missing_issuer_cik(tmp_path):
    path = _write_filing(tmp_path, "NOCIK", issuer_cik="", transactions=[
        {"date": "2022-06-01", "code": "P", "acquired_disposed": "A", "shares": 10.0, "price_per_share": 1.0},
    ])
    rows, appearance = fx._scan_one(path)
    assert rows == []
    assert appearance is None


def test_build_form4_scan_aggregates_directory_and_computes_first_filing(tmp_path):
    _write_filing(tmp_path, "S1", issuer_cik=CIK_A, owner_cik="0000000001",
                   filing_date="20220101", transactions=[
                       {"date": "2022-01-01", "code": "P", "acquired_disposed": "A",
                        "shares": 10.0, "price_per_share": 1.0},
                   ])
    _write_filing(tmp_path, "S2", issuer_cik=CIK_A, owner_cik="0000000001",
                   filing_date="20220301", transactions=[
                       {"date": "2022-03-01", "code": "S", "acquired_disposed": "D",
                        "shares": 5.0, "price_per_share": 2.0},
                   ])
    _write_filing(tmp_path, "S3", issuer_cik=CIK_A, owner_cik="0000000002",
                   filing_date="20220215", transactions=[])  # owner appears, no qualifying tx

    tx_df, owner_first_df = fx.build_form4_scan(parse_cache_dir=str(tmp_path), max_workers=2)
    assert len(tx_df) == 2
    assert set(tx_df["adsh"]) == {"S1", "S2"}

    first = owner_first_df.set_index("owner_key")["filing_date"]
    assert first["0000000001"] == date(2022, 1, 1)  # earliest of its two filings
    assert first["0000000002"] == date(2022, 2, 15)


def test_save_and_load_form4_scan_round_trip(tmp_path):
    tx_df = make_tx_df([
        make_tx_row(code="P", acquired_disposed="A",
                    transaction_date=date(2022, 1, 1), filing_date=date(2022, 1, 2)),
    ])
    owner_df = make_owner_df([make_owner_row(owner_key="O1", filing_date=date(2022, 1, 1))])
    out_dir = str(tmp_path / "out")
    tx_path, owner_path = fx.save_form4_scan(tx_df, owner_df, out_dir=out_dir)
    assert os.path.exists(tx_path) and os.path.exists(owner_path)
    assert not os.path.exists(tx_path + ".tmp")

    loaded_tx = pd.read_parquet(tx_path)
    loaded_owner = pd.read_parquet(owner_path)
    pd.testing.assert_frame_equal(loaded_tx, tx_df.reset_index(drop=True))
    pd.testing.assert_frame_equal(loaded_owner, owner_df.reset_index(drop=True))
    assert isinstance(loaded_tx["transaction_date"].iloc[0], date)


def test_load_or_build_form4_scan_reuses_existing_pair_without_rescanning(tmp_path, monkeypatch):
    out_dir = str(tmp_path / "out")
    tx_df = make_tx_df([make_tx_row(code="P", acquired_disposed="A",
                                     transaction_date=date(2022, 1, 1), filing_date=date(2022, 1, 2))])
    owner_df = make_owner_df([make_owner_row(owner_key="O1", filing_date=date(2022, 1, 1))])
    fx.save_form4_scan(tx_df, owner_df, out_dir=out_dir)

    def _boom(*a, **k):
        raise AssertionError("build_form4_scan should not be called when a cached pair exists")
    monkeypatch.setattr(fx, "build_form4_scan", _boom)

    loaded_tx, loaded_owner = fx.load_or_build_form4_scan(out_dir=out_dir, parse_cache_dir="unused")
    assert len(loaded_tx) == 1
    assert len(loaded_owner) == 1


def test_load_or_build_form4_scan_builds_when_none_cached(tmp_path):
    out_dir = str(tmp_path / "out")
    parse_dir = str(tmp_path / "parse_cache")
    os.makedirs(parse_dir, exist_ok=True)
    _write_filing(parse_dir, "X1", transactions=[
        {"date": "2022-01-01", "code": "P", "acquired_disposed": "A", "shares": 10.0, "price_per_share": 1.0},
    ])
    tx_df, owner_df = fx.load_or_build_form4_scan(out_dir=out_dir, parse_cache_dir=parse_dir, max_workers=2)
    assert len(tx_df) == 1
    assert len(owner_df) == 1
    assert fx._resolve_latest(out_dir, fx.TX_CACHE_GLOB) is not None
    assert fx._resolve_latest(out_dir, fx.OWNER_CACHE_GLOB) is not None
