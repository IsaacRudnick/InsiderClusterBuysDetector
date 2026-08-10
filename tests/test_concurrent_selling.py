"""Tests for the concurrent-selling (Section F) feature group:
  - backtest/sales_history.py: the local parse_cache/ scanner that extracts
    code-S/acquired_disposed-D sale transactions into a point-in-time index.
  - backtest/research.py: the x_sell_*/x_buy_sell_*/x_*_sold_* features
    built from that index (see its "Section F leakage note").

Runnable standalone via `python -m pytest tests/test_concurrent_selling.py -q`
from the repo root. No network access, and no dependency on the real
parse_cache/ directory -- backtest.sales_history tests write small crafted
JSON files to tmp_path and point the scanner at that directory; research.py
tests use the same FakePriceUniverse-style doubles tests/test_research.py
uses, plus a small hand-built sales_df.

The single most important test in this file is
test_sale_filed_after_event_day_is_invisible_even_if_transaction_is_earlier:
it proves the point-in-time discipline Task 3 of the concurrent-selling work
required -- a sale whose TRANSACTION date sits inside the cluster window but
whose FILING date lands after the cluster's own event_day must not move any
Section F feature, matching backtest.research's existing Section D leakage
discipline for owner/issuer history.
"""

from __future__ import annotations

import json
import math
import os
from datetime import date, timedelta

import pandas as pd
import pytest

import ipo_lookup
from backtest import research
from backtest import sales_history as sh
from backtest.state import DailyStateBuilder

CAL_START = date(2020, 1, 1)


# ---------------------------------------------------------------------------
# Fixtures / builders (mirrors tests/test_research.py's own style; kept
# self-contained per this project's "runnable standalone" test convention)
# ---------------------------------------------------------------------------
class FakePriceUniverse:
    def __init__(self) -> None:
        self.open_by_ticker: dict[str, dict[date, float]] = {}
        self.close_by_ticker: dict[str, dict[date, float]] = {}
        self.dv_by_ticker: dict[str, dict[date, float]] = {}
        self.dates_by_ticker: dict[str, list[date]] = {}
        self.frames: dict[str, pd.DataFrame] = {}

    def add_flat_series(self, ticker: str, dates: list[date], price: float, dv: float) -> None:
        opens = self.open_by_ticker.setdefault(ticker, {})
        closes = self.close_by_ticker.setdefault(ticker, {})
        dvs = self.dv_by_ticker.setdefault(ticker, {})
        for d in dates:
            opens[d] = price
            closes[d] = price
            dvs[d] = dv
        self.dates_by_ticker.setdefault(ticker, [])
        self.dates_by_ticker[ticker] = sorted(set(self.dates_by_ticker[ticker]) | set(dates))

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


def make_calendar(n_days: int, start: date = CAL_START) -> list[date]:
    return [start + timedelta(days=i) for i in range(n_days)]


def make_buy_row(
    *, ticker: str, issuer_cik: str, owner_cik: str, owner_name: str,
    transaction_date: date, filing_date: date,
    is_director: bool = True, is_officer: bool = False, is_ten_pct: bool = False,
    shares: float = 1000.0, price_per_share: float = 10.0, value: float = 10000.0,
) -> dict:
    return {
        "adsh": "0001", "form_type": "4", "filing_date": filing_date,
        "filing_url": "", "issuer_cik": issuer_cik, "issuer_name": "ISSUER",
        "ticker": ticker, "owner_cik": owner_cik, "owner_name": owner_name,
        "owner_roles": "Director", "is_director": is_director, "is_officer": is_officer,
        "is_ten_percent_owner": is_ten_pct, "transaction_date": transaction_date,
        "transaction_code": "P", "acquired_disposed": "A",
        "shares": shares, "price_per_share": price_per_share, "value": value,
        "shares_owned_after": shares * 5, "pct_of_prior_stake": None,
        "footnote_text": "", "is_10b5_1": False,
    }


def two_owner_cluster(
    *, ticker: str, issuer_cik: str, event_day: date,
    owner_a: str, owner_b: str, price: float = 10.0, value_each: float = 10000.0,
) -> list[dict]:
    """Same shape as tests/test_research.py's helper of the same name: two
    transactions filed exactly on event_day for a transaction the day
    before, so the ticker's rolling window becomes visible starting exactly
    on event_day -- a clean "new episode start"."""
    tx_date = event_day - timedelta(days=1)
    return [
        make_buy_row(
            ticker=ticker, issuer_cik=issuer_cik, owner_cik=owner_a, owner_name=owner_a,
            transaction_date=tx_date, filing_date=event_day,
            price_per_share=price, shares=value_each / price, value=value_each,
        ),
        make_buy_row(
            ticker=ticker, issuer_cik=issuer_cik, owner_cik=owner_b, owner_name=owner_b,
            transaction_date=tx_date, filing_date=event_day,
            price_per_share=price, shares=value_each / price, value=value_each,
        ),
    ]


def make_sale_row(
    *, issuer_cik: str, owner_cik: str, transaction_date: date, filing_date: date,
    shares: float = 500.0, price_per_share: float = 20.0, value: float | None = None,
    is_director: bool = False, is_officer: bool = False, ticker: str = "ABC",
) -> dict:
    val = value if value is not None else shares * price_per_share
    return {
        "adsh": "9999", "issuer_cik": issuer_cik, "ticker": ticker,
        "owner_cik": owner_cik, "owner_name": f"NAME_{owner_cik}", "owner_key": owner_cik,
        "is_director": is_director, "is_officer": is_officer, "is_ten_percent_owner": False,
        "transaction_date": transaction_date, "filing_date": filing_date,
        "shares": shares, "price_per_share": price_per_share, "value": val,
    }


def make_sales_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=sh._SALE_COLUMNS)


@pytest.fixture()
def no_network_ipo(monkeypatch):
    monkeypatch.setattr(ipo_lookup, "get_first_trade_date", lambda t: None)
    yield


def build_states(events_df: pd.DataFrame) -> DailyStateBuilder:
    return DailyStateBuilder(events_df)


def build_one_cluster_row(
    *, event_day: date, sales_df: pd.DataFrame | None, ticker: str = "ABC", issuer_cik: str = "ISS1",
    owner_a: str = "OWNER_X", owner_b: str = "OWNER_Y",
) -> pd.Series:
    """Common scaffolding for the Section F tests below: a single two-owner
    cluster on `ticker`/`issuer_cik`, flat prices, and whatever `sales_df`
    the caller supplies. Returns the single resulting row."""
    calendar = make_calendar(400)
    events_df = pd.DataFrame(two_owner_cluster(
        ticker=ticker, issuer_cik=issuer_cik, event_day=event_day,
        owner_a=owner_a, owner_b=owner_b, price=10.0, value_each=10000.0,
    ))
    prices = FakePriceUniverse()
    prices.add_flat_series("SPY", calendar, price=100.0, dv=1_000_000.0)
    prices.add_flat_series(ticker, calendar, price=10.0, dv=1_000_000.0)

    states = build_states(events_df)
    df = research.build_research_dataset(
        states, prices, calendar, events_df, horizons=(10,), sales_df=sales_df,
    )
    assert len(df) == 1
    return df.iloc[0]


# ---------------------------------------------------------------------------
# 1. backtest.sales_history: local parse_cache scanning
# ---------------------------------------------------------------------------
def _write_filing(
    tmp_path, adsh: str, *, form_type: str = "4", filing_date: str = "20260110",
    issuer_cik: str = "0000000001", ticker: str = "ABC",
    owners: list[dict] | None = None, transactions: list[dict] | None = None,
) -> str:
    payload = {
        "adsh": adsh, "form_type": form_type, "filing_date": filing_date,
        "issuer": {"cik": issuer_cik, "name": "ISSUER CO", "ticker": ticker},
        "owners": owners if owners is not None else [{
            "cik": "0000000099", "name": "Some Insider", "is_director": True,
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


def test_load_one_extracts_qualifying_sale(tmp_path):
    path = _write_filing(
        tmp_path, "A1", transactions=[
            {"date": "2026-01-08", "code": "S", "acquired_disposed": "D",
             "shares": 100.0, "price_per_share": 12.5},
        ],
    )
    rows = sh._load_one(path)
    assert len(rows) == 1
    r = rows[0]
    assert r["issuer_cik"] == "0000000001"
    assert r["ticker"] == "ABC"
    assert r["owner_cik"] == "0000000099"
    assert r["owner_key"] == "0000000099"
    assert r["is_director"] is True
    assert r["transaction_date"] == date(2026, 1, 8)
    assert r["filing_date"] == date(2026, 1, 10)
    assert r["shares"] == pytest.approx(100.0)
    assert r["price_per_share"] == pytest.approx(12.5)
    assert r["value"] == pytest.approx(1250.0)


@pytest.mark.parametrize("code,ad", [("F", "D"), ("D", "D"), ("A", "A"), ("M", "A"), ("S", "A")])
def test_load_one_excludes_non_qualifying_codes(tmp_path, code, ad):
    """Only code S with acquired_disposed D counts. F (tax withholding) and
    D (disposition to issuer) are non-discretionary; A/M are the buy side;
    S/A is the small residual data-quality noise this module also drops
    (see module docstring)."""
    path = _write_filing(
        tmp_path, f"EXCL_{code}_{ad}", transactions=[
            {"date": "2026-01-08", "code": code, "acquired_disposed": ad,
             "shares": 100.0, "price_per_share": 12.5},
        ],
    )
    assert sh._load_one(path) == []


def test_load_one_skips_non_form4(tmp_path):
    path = _write_filing(
        tmp_path, "FORM3", form_type="3", transactions=[
            {"date": "2026-01-08", "code": "S", "acquired_disposed": "D",
             "shares": 100.0, "price_per_share": 12.5},
        ],
    )
    assert sh._load_one(path) == []


def test_load_one_handles_4a_amendment(tmp_path):
    path = _write_filing(
        tmp_path, "AMEND1", form_type="4/A", transactions=[
            {"date": "2026-01-08", "code": "S", "acquired_disposed": "D",
             "shares": 100.0, "price_per_share": 12.5},
        ],
    )
    assert len(sh._load_one(path)) == 1


def test_load_one_skips_malformed_json(tmp_path):
    path = os.path.join(str(tmp_path), "BAD.json")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write('{"adsh": "BAD", "form_type": "4"}}')  # trailing stray brace
    assert sh._load_one(path) == []


def test_load_one_skips_missing_issuer_cik(tmp_path):
    path = _write_filing(
        tmp_path, "NOCIK", issuer_cik="", transactions=[
            {"date": "2026-01-08", "code": "S", "acquired_disposed": "D",
             "shares": 100.0, "price_per_share": 12.5},
        ],
    )
    assert sh._load_one(path) == []


def test_build_sales_cache_scans_directory_and_aggregates(tmp_path):
    _write_filing(tmp_path, "S1", transactions=[
        {"date": "2026-01-01", "code": "S", "acquired_disposed": "D", "shares": 10.0, "price_per_share": 2.0},
    ])
    _write_filing(tmp_path, "S2", transactions=[
        {"date": "2026-01-02", "code": "F", "acquired_disposed": "D", "shares": 10.0, "price_per_share": 2.0},
        {"date": "2026-01-02", "code": "S", "acquired_disposed": "D", "shares": 20.0, "price_per_share": 3.0},
    ])
    _write_filing(tmp_path, "S3", form_type="3", transactions=[
        {"date": "2026-01-03", "code": "S", "acquired_disposed": "D", "shares": 5.0, "price_per_share": 1.0},
    ])

    df = sh.build_sales_cache(parse_cache_dir=str(tmp_path), max_workers=2)
    assert len(df) == 2  # S1's one sale + S2's one sale; S2's F row and S3 (form 3) excluded
    assert set(df["adsh"]) == {"S1", "S2"}
    assert list(df.columns) == sh._SALE_COLUMNS


def test_save_and_load_sales_cache_round_trip(tmp_path):
    df = make_sales_df([
        make_sale_row(issuer_cik="ISS1", owner_cik="O1",
                      transaction_date=date(2026, 1, 1), filing_date=date(2026, 1, 2)),
    ])
    out_dir = str(tmp_path / "out")
    path = sh.save_sales_cache(df, out_dir=out_dir)
    assert os.path.exists(path)
    assert not os.path.exists(path + ".tmp")
    assert "sales_cache_" in os.path.basename(path)
    assert f"{len(df)}rows" in os.path.basename(path)

    loaded = sh.load_sales_cache(path)
    pd.testing.assert_frame_equal(loaded, df.reset_index(drop=True))
    # Date columns must survive the parquet round trip as plain
    # datetime.date (not pandas Timestamp) -- research.py compares them
    # directly against `date` calendar entries.
    assert isinstance(loaded["transaction_date"].iloc[0], date)
    assert not isinstance(loaded["transaction_date"].iloc[0], pd.Timestamp)


def test_load_or_build_sales_cache_reuses_existing_without_rescanning(tmp_path, monkeypatch):
    out_dir = str(tmp_path / "out")
    df = make_sales_df([
        make_sale_row(issuer_cik="ISS1", owner_cik="O1",
                      transaction_date=date(2026, 1, 1), filing_date=date(2026, 1, 2)),
    ])
    sh.save_sales_cache(df, out_dir=out_dir)

    def _boom(*a, **k):
        raise AssertionError("build_sales_cache should not be called when a cached file exists")
    monkeypatch.setattr(sh, "build_sales_cache", _boom)

    loaded = sh.load_or_build_sales_cache(out_dir=out_dir, parse_cache_dir="unused")
    assert len(loaded) == 1


def test_load_or_build_sales_cache_builds_when_none_cached(tmp_path):
    out_dir = str(tmp_path / "out")
    parse_dir = str(tmp_path / "parse_cache")
    os.makedirs(parse_dir, exist_ok=True)
    _write_filing(parse_dir, "X1", transactions=[
        {"date": "2026-01-01", "code": "S", "acquired_disposed": "D", "shares": 10.0, "price_per_share": 2.0},
    ])
    df = sh.load_or_build_sales_cache(out_dir=out_dir, parse_cache_dir=parse_dir, max_workers=2)
    assert len(df) == 1
    # Must have persisted a cache file so a second call would reuse it.
    assert sh._resolve_latest(out_dir, sh.SALES_GLOB) is not None


# ---------------------------------------------------------------------------
# 2. backtest.research: Section F feature values
# ---------------------------------------------------------------------------
def test_no_sales_df_gives_empty_window_defaults(no_network_ipo):
    """sales_df=None (the default) must never error and must leave every
    Section F feature at its documented empty-window value."""
    event_day = make_calendar(200)[100]
    row = build_one_cluster_row(event_day=event_day, sales_df=None)

    assert row["x_sell_n_cluster"] == 0
    assert row["x_sell_n_insiders_cluster"] == 0
    assert row["x_log1p_sell_value_cluster"] == pytest.approx(0.0)
    assert row["x_sell_n_trail90"] == 0
    assert row["x_sell_n_insiders_trail90"] == 0
    assert row["x_log1p_sell_value_trail90"] == pytest.approx(0.0)
    assert row["x_buy_sell_balance_cluster"] == pytest.approx(1.0)  # no selling -> full buy dominance
    assert row["x_buyer_also_sold_nearby"] == False  # noqa: E712
    assert row["x_officer_or_director_sold_cluster"] == False  # noqa: E712


def test_concurrent_sale_within_cluster_window_is_counted(no_network_ipo):
    event_day = make_calendar(200)[100]
    sales_df = make_sales_df([
        make_sale_row(
            issuer_cik="ISS1", owner_cik="SELLER_Z",
            transaction_date=event_day - timedelta(days=1), filing_date=event_day,
            shares=200.0, price_per_share=10.0,  # value = 2000
        ),
    ])
    row = build_one_cluster_row(event_day=event_day, sales_df=sales_df)

    assert row["x_sell_n_cluster"] == 1
    assert row["x_sell_n_insiders_cluster"] == 1
    assert row["x_log1p_sell_value_cluster"] == pytest.approx(math.log1p(2000.0))
    assert row["x_sell_n_trail90"] == 1
    # cluster buy value = 20,000 (two owners x 10,000 each); sale value = 2,000
    assert row["x_buy_sell_balance_cluster"] == pytest.approx(20000.0 / 22000.0)


def test_sale_outside_cluster_window_only_shows_in_trailing(no_network_ipo):
    """window_days defaults to 14 (backtest.state.WINDOW_DAYS): a sale 30
    days before event_day sits outside the 14-day cluster window but inside
    the 90-day trailing window."""
    event_day = make_calendar(200)[100]
    sale_tx_date = event_day - timedelta(days=30)
    sales_df = make_sales_df([
        make_sale_row(
            issuer_cik="ISS1", owner_cik="SELLER_Z",
            transaction_date=sale_tx_date, filing_date=sale_tx_date + timedelta(days=1),
            shares=100.0, price_per_share=5.0,
        ),
    ])
    row = build_one_cluster_row(event_day=event_day, sales_df=sales_df)

    assert row["x_sell_n_cluster"] == 0
    assert row["x_log1p_sell_value_cluster"] == pytest.approx(0.0)
    assert row["x_sell_n_trail90"] == 1
    assert row["x_log1p_sell_value_trail90"] == pytest.approx(math.log1p(500.0))


def test_sale_older_than_trailing_window_is_invisible_everywhere(no_network_ipo):
    event_day = make_calendar(300)[200]
    sale_tx_date = event_day - timedelta(days=120)  # older than TRAILING_SALE_WINDOW_DAYS=90
    sales_df = make_sales_df([
        make_sale_row(
            issuer_cik="ISS1", owner_cik="SELLER_Z",
            transaction_date=sale_tx_date, filing_date=sale_tx_date + timedelta(days=1),
        ),
    ])
    row = build_one_cluster_row(event_day=event_day, sales_df=sales_df)
    assert row["x_sell_n_cluster"] == 0
    assert row["x_sell_n_trail90"] == 0


def test_sale_for_a_different_issuer_is_never_counted(no_network_ipo):
    event_day = make_calendar(200)[100]
    sales_df = make_sales_df([
        make_sale_row(
            issuer_cik="SOME_OTHER_ISSUER", owner_cik="SELLER_Z",
            transaction_date=event_day - timedelta(days=1), filing_date=event_day,
        ),
    ])
    row = build_one_cluster_row(event_day=event_day, sales_df=sales_df, issuer_cik="ISS1")
    assert row["x_sell_n_cluster"] == 0
    assert row["x_sell_n_trail90"] == 0
    assert row["x_buy_sell_balance_cluster"] == pytest.approx(1.0)


def test_buyer_also_sold_nearby_true_when_seller_matches_a_buyer(no_network_ipo):
    event_day = make_calendar(200)[100]
    sales_df = make_sales_df([
        make_sale_row(
            issuer_cik="ISS1", owner_cik="OWNER_X",  # matches one of the cluster's buyers
            transaction_date=event_day - timedelta(days=1), filing_date=event_day,
        ),
    ])
    row = build_one_cluster_row(
        event_day=event_day, sales_df=sales_df, owner_a="OWNER_X", owner_b="OWNER_Y",
    )
    assert row["x_buyer_also_sold_nearby"] == True  # noqa: E712


def test_buyer_also_sold_nearby_false_when_seller_is_an_outsider(no_network_ipo):
    event_day = make_calendar(200)[100]
    sales_df = make_sales_df([
        make_sale_row(
            issuer_cik="ISS1", owner_cik="SOME_OUTSIDER",
            transaction_date=event_day - timedelta(days=1), filing_date=event_day,
        ),
    ])
    row = build_one_cluster_row(
        event_day=event_day, sales_df=sales_df, owner_a="OWNER_X", owner_b="OWNER_Y",
    )
    assert row["x_buyer_also_sold_nearby"] == False  # noqa: E712
    # Sanity: the sale IS visible (counted), just from a non-buyer -- proves
    # the False above is about owner identity, not visibility.
    assert row["x_sell_n_cluster"] == 1


def test_officer_or_director_sold_cluster_true_for_officer_seller(no_network_ipo):
    event_day = make_calendar(200)[100]
    sales_df = make_sales_df([
        make_sale_row(
            issuer_cik="ISS1", owner_cik="SELLER_Z",
            transaction_date=event_day - timedelta(days=1), filing_date=event_day,
            is_officer=True, is_director=False,
        ),
    ])
    row = build_one_cluster_row(event_day=event_day, sales_df=sales_df)
    assert row["x_officer_or_director_sold_cluster"] == True  # noqa: E712


def test_officer_or_director_sold_cluster_false_for_plain_seller(no_network_ipo):
    event_day = make_calendar(200)[100]
    sales_df = make_sales_df([
        make_sale_row(
            issuer_cik="ISS1", owner_cik="SELLER_Z",
            transaction_date=event_day - timedelta(days=1), filing_date=event_day,
            is_officer=False, is_director=False,
        ),
    ])
    row = build_one_cluster_row(event_day=event_day, sales_df=sales_df)
    assert row["x_officer_or_director_sold_cluster"] == False  # noqa: E712


# ---------------------------------------------------------------------------
# 3. THE load-bearing test: point-in-time / no leakage from filing lag
# ---------------------------------------------------------------------------
def test_sale_filed_after_event_day_is_invisible_even_if_transaction_is_earlier(no_network_ipo):
    """A sale whose transaction_date sits comfortably inside the cluster
    window, but whose filing_date lands AFTER event_day, must be invisible
    to every Section F feature -- Form 4's filing lag means nobody watching
    the tape on event_day could have known about it yet. This is the
    concurrent-selling analogue of backtest.research's Section D leakage
    discipline (owner/issuer history), applied here to sale visibility.

    The control case (same sale, filed ON event_day) proves the harness
    really can see a visible sale in this exact configuration -- so the
    invisible case failing to show up is about the filing-lag gate, not
    about the sale being malformed or out of window some other way.
    """
    event_day = make_calendar(200)[100]
    tx_date = event_day - timedelta(days=1)  # well inside the cluster window either way

    invisible_sales_df = make_sales_df([
        make_sale_row(
            issuer_cik="ISS1", owner_cik="OWNER_X",  # also a cluster buyer, and an officer --
            transaction_date=tx_date, filing_date=event_day + timedelta(days=5),  # ...but filed LATE
            is_officer=True, shares=1000.0, price_per_share=50.0,  # large value, would dominate if seen
        ),
    ])
    visible_sales_df = make_sales_df([
        make_sale_row(
            issuer_cik="ISS1", owner_cik="OWNER_X",
            transaction_date=tx_date, filing_date=event_day,  # filed ON event_day -- visible
            is_officer=True, shares=1000.0, price_per_share=50.0,
        ),
    ])

    row_invisible = build_one_cluster_row(
        event_day=event_day, sales_df=invisible_sales_df, owner_a="OWNER_X", owner_b="OWNER_Y",
    )
    row_visible = build_one_cluster_row(
        event_day=event_day, sales_df=visible_sales_df, owner_a="OWNER_X", owner_b="OWNER_Y",
    )

    # Control: the visible sale really does move every Section F feature.
    assert row_visible["x_sell_n_cluster"] == 1
    assert row_visible["x_sell_n_trail90"] == 1
    assert row_visible["x_buyer_also_sold_nearby"] == True  # noqa: E712
    assert row_visible["x_officer_or_director_sold_cluster"] == True  # noqa: E712
    assert row_visible["x_buy_sell_balance_cluster"] < 1.0

    # The late-filed sale must leave every Section F feature at its
    # empty-window default -- as if the sale never happened.
    assert row_invisible["x_sell_n_cluster"] == 0
    assert row_invisible["x_sell_n_insiders_cluster"] == 0
    assert row_invisible["x_log1p_sell_value_cluster"] == pytest.approx(0.0)
    assert row_invisible["x_sell_n_trail90"] == 0
    assert row_invisible["x_sell_n_insiders_trail90"] == 0
    assert row_invisible["x_log1p_sell_value_trail90"] == pytest.approx(0.0)
    assert row_invisible["x_buy_sell_balance_cluster"] == pytest.approx(1.0)
    assert row_invisible["x_buyer_also_sold_nearby"] == False  # noqa: E712
    assert row_invisible["x_officer_or_director_sold_cluster"] == False  # noqa: E712


def test_sale_filed_after_event_day_is_invisible_to_trailing_window_too(no_network_ipo):
    """Same filing-lag gate, exercised on the trailing-90-day window
    specifically: a sale 30 days before event_day (inside the 90-day
    trailing window on transaction_date alone) filed 5 days after event_day
    must not appear in x_sell_n_trail90 either."""
    event_day = make_calendar(200)[100]
    tx_date = event_day - timedelta(days=30)
    sales_df = make_sales_df([
        make_sale_row(
            issuer_cik="ISS1", owner_cik="SELLER_Z",
            transaction_date=tx_date, filing_date=event_day + timedelta(days=5),
        ),
    ])
    row = build_one_cluster_row(event_day=event_day, sales_df=sales_df)
    assert row["x_sell_n_trail90"] == 0
    assert row["x_log1p_sell_value_trail90"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 4. Low-level helpers
# ---------------------------------------------------------------------------
def test_prepare_sale_index_groups_by_issuer_and_sorts_by_transaction_date():
    sales_df = make_sales_df([
        make_sale_row(issuer_cik="ISS1", owner_cik="A",
                      transaction_date=date(2026, 1, 10), filing_date=date(2026, 1, 11)),
        make_sale_row(issuer_cik="ISS1", owner_cik="B",
                      transaction_date=date(2026, 1, 5), filing_date=date(2026, 1, 6)),
        make_sale_row(issuer_cik="ISS2", owner_cik="C",
                      transaction_date=date(2026, 1, 7), filing_date=date(2026, 1, 8)),
    ])
    idx = research._prepare_sale_index(sales_df)
    assert set(idx.keys()) == {"ISS1", "ISS2"}
    assert idx["ISS1"]["tx_dates"] == [date(2026, 1, 5), date(2026, 1, 10)]
    assert [r["owner_key"] for r in idx["ISS1"]["records"]] == ["B", "A"]


def test_prepare_sale_index_empty_or_none_returns_empty_dict():
    assert research._prepare_sale_index(None) == {}
    assert research._prepare_sale_index(make_sales_df([])) == {}


def test_sale_window_stats_counts_distinct_sellers_and_sums_value():
    sales = [
        {"owner_key": "A", "value": 100.0},
        {"owner_key": "A", "value": 50.0},   # same seller, second transaction
        {"owner_key": "B", "value": 25.0},
    ]
    n, n_sellers, value = research._sale_window_stats(sales)
    assert n == 3
    assert n_sellers == 2
    assert value == pytest.approx(175.0)


def test_sale_window_stats_empty_list():
    n, n_sellers, value = research._sale_window_stats([])
    assert (n, n_sellers, value) == (0, 0, 0.0)
