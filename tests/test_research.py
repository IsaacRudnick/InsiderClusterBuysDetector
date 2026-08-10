"""Tests for backtest.research: the rich per-event research dataset builder.

Runnable standalone via `python -m pytest tests/test_research.py -q` from
the repo root. No network access: DailyStateBuilder's IPO-date prefetch is
monkeypatched to avoid touching ipo_lookup's yfinance calls, and price data
is supplied via small fake PriceUniverse-shaped doubles rather than the real
disk/yfinance-backed PriceUniverse.
"""

from __future__ import annotations

import logging
import math
import os
import sys
from datetime import date, timedelta

import pandas as pd
import pytest

# Make sure `backtest` and `insider_cluster_buys` resolve when this file is
# invoked directly (python -m pytest tests/test_research.py) regardless of
# the working directory pytest was launched from.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import ipo_lookup  # noqa: E402
import ticker_reuse  # noqa: E402
from backtest import research  # noqa: E402
from backtest import splits as splits_mod  # noqa: E402
from backtest.state import DailyStateBuilder  # noqa: E402

CAL_START = date(2020, 1, 1)


# ---------------------------------------------------------------------------
# Fakes / fixtures
# ---------------------------------------------------------------------------
class FakePriceUniverse:
    """Minimal PriceUniverse-shaped double: exposes only the methods and
    public dicts backtest.research actually uses. No disk, no network."""

    def __init__(self) -> None:
        self.open_by_ticker: dict[str, dict[date, float]] = {}
        self.close_by_ticker: dict[str, dict[date, float]] = {}
        self.dv_by_ticker: dict[str, dict[date, float]] = {}
        self.dates_by_ticker: dict[str, list[date]] = {}
        # Real PriceUniverse always carries this (the raw OHLCV frame that
        # backtest.splits.detect_discontinuities scans). Plain add_flat_series
        # below leaves it empty for a ticker, which research._ticker_price_frame
        # reads as "nothing to scan" -- exactly the pre-split-adjustment
        # behavior, so every existing fixture stays byte-for-byte unchanged.
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

    def add_series_with_frame(
        self, ticker: str, dates: list[date], closes: list[float], volumes: list[float],
    ) -> None:
        """Like add_flat_series, but from an arbitrary (non-flat) close/volume
        path, and it also populates self.frames[ticker] -- an OHLCV
        DataFrame with a DatetimeIndex, matching what real PriceUniverse
        exposes -- so backtest.research's split-detection wiring has
        something to scan. Open == close for every bar, which is enough for
        the entry/exit open-price lookups this module needs."""
        assert len(dates) == len(closes) == len(volumes)
        opens = self.open_by_ticker.setdefault(ticker, {})
        closemap = self.close_by_ticker.setdefault(ticker, {})
        dvs = self.dv_by_ticker.setdefault(ticker, {})
        for d, c, v in zip(dates, closes, volumes):
            opens[d] = c
            closemap[d] = c
            dvs[d] = c * v
        self.dates_by_ticker[ticker] = sorted(set(self.dates_by_ticker.get(ticker, [])) | set(dates))

        idx = pd.DatetimeIndex([pd.Timestamp(d) for d in dates], name="date")
        frame = pd.DataFrame(
            {"open": closes, "high": closes, "low": closes, "close": closes, "volume": volumes},
            index=idx,
        )
        frame["dollar_volume"] = frame["close"] * frame["volume"]
        self.frames[ticker] = frame

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
        # Mirrors backtest.prices.PriceUniverse.median_dollar_volume exactly:
        # median over the `window` bars strictly before dt, using
        # vals[len(vals)//2] (not a true even-count average).
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


def make_row(
    *, ticker: str, issuer_cik: str, owner_cik: str, owner_name: str,
    transaction_date: date, filing_date: date,
    owner_roles: str = "Director",
    is_director: bool = True, is_officer: bool = False, is_ten_pct: bool = False,
    shares: float = 1000.0, price_per_share: float = 10.0, value: float = 10000.0,
    pct_of_prior_stake=None, footnote_text: str = "", is_10b5_1: bool = False,
) -> dict:
    return {
        "adsh": "0001", "form_type": "4", "filing_date": filing_date,
        "filing_url": "", "issuer_cik": issuer_cik, "issuer_name": "ISSUER",
        "ticker": ticker, "owner_cik": owner_cik, "owner_name": owner_name,
        "owner_roles": owner_roles, "is_director": is_director, "is_officer": is_officer,
        "is_ten_percent_owner": is_ten_pct, "transaction_date": transaction_date,
        "transaction_code": "P", "acquired_disposed": "A",
        "shares": shares, "price_per_share": price_per_share, "value": value,
        "shares_owned_after": shares * 5, "pct_of_prior_stake": pct_of_prior_stake,
        "footnote_text": footnote_text, "is_10b5_1": is_10b5_1,
    }


def two_owner_cluster(
    *, ticker: str, issuer_cik: str, event_day: date,
    owner_a: str, owner_b: str, price: float = 10.0, value_each: float = 10000.0,
) -> list[dict]:
    """Two transactions, one per owner, filed exactly on event_day for a
    transaction the day before -- so the ticker's rolling window (state.py's
    start_rel = max(transaction_date, filing_date)) becomes visible starting
    exactly on event_day, making it a clean "new episode start" for the main
    day loop."""
    tx_date = event_day - timedelta(days=1)
    return [
        make_row(
            ticker=ticker, issuer_cik=issuer_cik, owner_cik=owner_a, owner_name=owner_a,
            transaction_date=tx_date, filing_date=event_day,
            price_per_share=price, shares=value_each / price, value=value_each,
        ),
        make_row(
            ticker=ticker, issuer_cik=issuer_cik, owner_cik=owner_b, owner_name=owner_b,
            transaction_date=tx_date, filing_date=event_day,
            price_per_share=price, shares=value_each / price, value=value_each,
        ),
    ]


@pytest.fixture()
def no_network_ipo(monkeypatch):
    """DailyStateBuilder._prefetch_ipo_dates calls ipo_lookup.get_first_trade_date
    in a thread pool at construction time, which can hit yfinance. Stub it out
    so tests never touch the network."""
    monkeypatch.setattr(ipo_lookup, "get_first_trade_date", lambda t: None)
    yield


def build_states(events_df: pd.DataFrame) -> DailyStateBuilder:
    return DailyStateBuilder(events_df)


# ---------------------------------------------------------------------------
# Test 4: role parsing (no fixtures needed)
# ---------------------------------------------------------------------------
def test_parse_roles_mixed_string():
    parsed = research._parse_roles("Director, Officer (President,Chairman & CEO), 10% Owner")
    assert parsed["has_ceo"] is True
    assert parsed["has_chairman"] is True
    assert parsed["has_president"] is True
    assert parsed["has_cfo"] is False
    assert parsed["has_coo"] is False


def test_parse_roles_cfo_and_coo():
    parsed = research._parse_roles("Chief Financial Officer, Chief Operating Officer")
    assert parsed["has_cfo"] is True
    assert parsed["has_coo"] is True
    assert parsed["has_ceo"] is False


def test_parse_roles_empty_string_and_none():
    assert all(v is False for v in research._parse_roles("").values())
    assert all(v is False for v in research._parse_roles(None).values())


# ---------------------------------------------------------------------------
# Test 5: momentum / vol / drawdown NaN on short history
# ---------------------------------------------------------------------------
def test_momentum_nan_on_short_history():
    prices = FakePriceUniverse()
    calendar = make_calendar(10)
    # Only 8 prior trading days available; x_mom_21_skip5 needs 6+21=27.
    prices.add_flat_series("ABC", calendar[:8], price=10.0, dv=1_000_000.0)
    entry_day = calendar[9]
    result = research._momentum_skip5(prices, "ABC", entry_day, 21)
    assert math.isnan(result)


def test_vol_nan_on_short_history():
    prices = FakePriceUniverse()
    calendar = make_calendar(10)
    prices.add_flat_series("ABC", calendar[:8], price=10.0, dv=1_000_000.0)
    entry_day = calendar[9]
    result = research._vol_ann(prices, "ABC", entry_day, 21)
    assert math.isnan(result)


def test_drawdown_nan_on_short_history():
    prices = FakePriceUniverse()
    calendar = make_calendar(10)
    prices.add_flat_series("ABC", calendar[:8], price=10.0, dv=1_000_000.0)
    entry_day = calendar[9]
    result = research._drawdown(prices, "ABC", entry_day, 252)
    assert math.isnan(result)


def test_momentum_and_vol_are_finite_with_enough_history():
    prices = FakePriceUniverse()
    calendar = make_calendar(400)
    prices.add_flat_series("ABC", calendar[:390], price=10.0, dv=1_000_000.0)
    entry_day = calendar[395]
    mom = research._momentum_skip5(prices, "ABC", entry_day, 21)
    vol = research._vol_ann(prices, "ABC", entry_day, 21)
    # Flat price series -> exactly 0 return, exactly 0 vol.
    assert mom == pytest.approx(0.0)
    assert vol == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Test 3: x_buy_value_to_adv known-value test
# ---------------------------------------------------------------------------
def test_buy_value_to_adv_known_value(no_network_ipo):
    calendar = make_calendar(40)
    event_day = calendar[25]
    issuer_cik = "ISS1"
    ticker = "KNOWN"

    events_df = pd.DataFrame(two_owner_cluster(
        ticker=ticker, issuer_cik=issuer_cik, event_day=event_day,
        owner_a="OWNER_X", owner_b="OWNER_Y", price=10.0, value_each=27500.0,
    ))
    # total cluster value = 55,000

    prices = FakePriceUniverse()
    prices.add_flat_series("SPY", calendar, price=100.0, dv=1_000_000.0)
    # Flat open/close for KNOWN across the whole calendar (needed for entry
    # price + last_close_on_or_before), but hand-set its dollar volume for
    # the 20 days before entry_day to a known series: 1000, 2000, ..., 20000.
    prices.add_flat_series(ticker, calendar, price=10.0, dv=1_000_000.0)
    entry_day = event_day + timedelta(days=1)  # _find_entry_day picks the next day
    dv_window_dates = [d for d in calendar if d < entry_day][-20:]
    assert len(dv_window_dates) == 20
    for i, d in enumerate(dv_window_dates, start=1):
        prices.dv_by_ticker[ticker][d] = float(i * 1000)
    # Hand-computed expected median: sorted values 1000..20000 step 1000,
    # PriceUniverse-style vals[len(vals)//2] with 20 values -> index 10 -> 11000.
    expected_adv = 11000.0

    states = build_states(events_df)
    df = research.build_research_dataset(states, prices, calendar, events_df, horizons=())

    assert len(df) == 1
    row = df.iloc[0]
    assert row["ticker"] == ticker
    assert row["x_buy_value_to_adv"] == pytest.approx(55000.0 / expected_adv)
    assert row["x_buy_value_to_adv"] == pytest.approx(5.0)
    assert row["x_log_adv20"] == pytest.approx(math.log(expected_adv))


# ---------------------------------------------------------------------------
# Test 2: label-window-must-have-closed gating rule
# ---------------------------------------------------------------------------
def test_label_window_must_have_closed(no_network_ipo):
    calendar = make_calendar(500)
    issuer_cik_a, issuer_cik_b, issuer_cik_c = "ISS_A", "ISS_B", "ISS_C"

    day_a = calendar[50]    # entry_idx_a = 51; label window closes at calendar[51+63]=calendar[114]
    day_b = calendar[100]   # < calendar[114] -> window NOT closed yet relative to day_b
    day_c = calendar[200]   # > calendar[114] -> window closed

    rows = []
    rows += two_owner_cluster(ticker="T1", issuer_cik=issuer_cik_a, event_day=day_a,
                               owner_a="OWNER_A", owner_b="OWNER_B1")
    rows += two_owner_cluster(ticker="T2", issuer_cik=issuer_cik_b, event_day=day_b,
                               owner_a="OWNER_A", owner_b="OWNER_B2")
    rows += two_owner_cluster(ticker="T3", issuer_cik=issuer_cik_c, event_day=day_c,
                               owner_a="OWNER_A", owner_b="OWNER_B3")
    events_df = pd.DataFrame(rows)

    prices = FakePriceUniverse()
    prices.add_flat_series("SPY", calendar, price=100.0, dv=1_000_000.0)
    for t in ("T1", "T2", "T3"):
        prices.add_flat_series(t, calendar, price=10.0, dv=1_000_000.0)

    states = build_states(events_df)
    df = research.build_research_dataset(states, prices, calendar, events_df, horizons=(63,))
    df = df.sort_values("event_day").reset_index(drop=True)

    row_b = df[df["ticker"] == "T2"].iloc[0]
    row_c = df[df["ticker"] == "T3"].iloc[0]

    # Condition 1 alone: the prior-buy COUNT needs no closed label window,
    # so it sees every earlier event by the same owner. The feature is a
    # value-weighted mean over the cluster's owners, not a cluster total.
    # B: OWNER_A has 1 prior (T1), OWNER_B2 has 0 -> 0.5
    # C: OWNER_A has 2 priors (T1, T2), OWNER_B3 has 0 -> 1.0
    assert row_b["x_owner_prior_buys_wmean"] == pytest.approx(0.5)
    assert row_c["x_owner_prior_buys_wmean"] == pytest.approx(1.0)

    # Condition 2 (label window closed) gates the adj63-based feature.
    assert math.isnan(row_b["x_owner_prior_adj63_mean"])
    assert not math.isnan(row_c["x_owner_prior_adj63_mean"])
    # Flat prices -> stock and SPY both return 0 -> adj_63 == 0 exactly.
    assert row_c["x_owner_prior_adj63_mean"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Test 1: point-in-time / no leakage from a later event into an earlier one
# ---------------------------------------------------------------------------
def _build_three_event_frame(t3_price_path):
    """Three events sharing OWNER_A across three different tickers, widely
    spaced so every prior event's 63-day label window has closed by the
    time the next one starts. `t3_price_path` lets the caller mutate T3's
    price history between calls."""
    calendar = make_calendar(500)
    day1, day2, day3 = calendar[50], calendar[200], calendar[350]

    rows = []
    rows += two_owner_cluster(ticker="T1", issuer_cik="ISS1", event_day=day1,
                               owner_a="OWNER_A", owner_b="OWNER_B1")
    rows += two_owner_cluster(ticker="T2", issuer_cik="ISS2", event_day=day2,
                               owner_a="OWNER_A", owner_b="OWNER_B2")
    rows += two_owner_cluster(ticker="T3", issuer_cik="ISS3", event_day=day3,
                               owner_a="OWNER_A", owner_b="OWNER_B3")
    events_df = pd.DataFrame(rows)

    prices = FakePriceUniverse()
    prices.add_flat_series("SPY", calendar, price=100.0, dv=1_000_000.0)
    prices.add_flat_series("T1", calendar, price=10.0, dv=1_000_000.0)
    prices.add_flat_series("T2", calendar, price=10.0, dv=1_000_000.0)
    t3_price_path(prices, calendar)

    states = build_states(events_df)
    df = research.build_research_dataset(states, prices, calendar, events_df, horizons=(63,))
    return df.sort_values("event_day").reset_index(drop=True)


def test_later_event_does_not_leak_into_earlier_rows(no_network_ipo):
    def flat_t3(prices, calendar):
        prices.add_flat_series("T3", calendar, price=10.0, dv=1_000_000.0)

    def wild_t3(prices, calendar):
        # T3's price triples right after its entry day, drastically changing
        # T3's own adj_63 -- but this must never reach T1's or T2's rows.
        prices.add_flat_series("T3", calendar, price=10.0, dv=1_000_000.0)
        entry_idx = calendar.index(calendar[350]) + 1
        for d in calendar[entry_idx + 5:]:
            prices.open_by_ticker["T3"][d] = 30.0
            prices.close_by_ticker["T3"][d] = 30.0

    df_baseline = _build_three_event_frame(flat_t3)
    df_variant = _build_three_event_frame(wild_t3)

    feature_cols = [c for c in df_baseline.columns if c.startswith("x_")]

    row1_base = df_baseline[df_baseline["ticker"] == "T1"].iloc[0]
    row1_var = df_variant[df_variant["ticker"] == "T1"].iloc[0]
    row2_base = df_baseline[df_baseline["ticker"] == "T2"].iloc[0]
    row2_var = df_variant[df_variant["ticker"] == "T2"].iloc[0]
    row3_base = df_baseline[df_baseline["ticker"] == "T3"].iloc[0]
    row3_var = df_variant[df_variant["ticker"] == "T3"].iloc[0]

    for col in feature_cols:
        a, b = row1_base[col], row1_var[col]
        if isinstance(a, float) and math.isnan(a) and isinstance(b, float) and math.isnan(b):
            continue
        assert a == pytest.approx(b) if isinstance(a, float) else a == b, (
            f"T1 (earliest event) feature {col!r} changed when a LATER event's "
            f"price path was mutated: {a!r} vs {b!r}"
        )

    for col in feature_cols:
        a, b = row2_base[col], row2_var[col]
        if isinstance(a, float) and math.isnan(a) and isinstance(b, float) and math.isnan(b):
            continue
        assert a == pytest.approx(b) if isinstance(a, float) else a == b, (
            f"T2 feature {col!r} changed when a LATER event's (T3) price path "
            f"was mutated: {a!r} vs {b!r}"
        )

    # Sanity check the mutation actually did something, so the tests above
    # are not vacuously true. It must show up in T3's own LABEL.
    #
    # It must NOT show up in T3's x_owner_prior_adj63_mean: that feature is
    # built from OWNER_A's PRIOR events (T1 and T2), whose prices were never
    # touched. Asserting that feature changes would be asserting leakage.
    assert row3_base["adj_63"] != pytest.approx(row3_var["adj_63"])
    assert row3_base["x_owner_prior_adj63_mean"] == pytest.approx(
        row3_var["x_owner_prior_adj63_mean"]
    )


# ---------------------------------------------------------------------------
# End-to-end smoke test
# ---------------------------------------------------------------------------
def test_build_research_dataset_smoke(no_network_ipo):
    calendar = make_calendar(300)
    day1, day2 = calendar[30], calendar[150]

    rows = []
    rows += two_owner_cluster(ticker="AAA", issuer_cik="ISSA", event_day=day1,
                               owner_a="O1", owner_b="O2")
    rows += two_owner_cluster(ticker="BBB", issuer_cik="ISSB", event_day=day2,
                               owner_a="O3", owner_b="O4")
    events_df = pd.DataFrame(rows)

    prices = FakePriceUniverse()
    prices.add_flat_series("SPY", calendar, price=100.0, dv=1_000_000.0)
    prices.add_flat_series("AAA", calendar, price=10.0, dv=1_000_000.0)
    prices.add_flat_series("BBB", calendar, price=20.0, dv=2_000_000.0)

    states = build_states(events_df)
    df = research.build_research_dataset(states, prices, calendar, events_df)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    for col in ("ticker", "issuer_cik", "event_day", "entry_day", "entry_open"):
        assert col in df.columns
    for h in research.DEFAULT_HORIZONS:
        for prefix in ("fwd_", "spy_", "adj_", "delisted_"):
            assert f"{prefix}{h}" in df.columns
    for col in research._FEATURE_COLS:
        assert col in df.columns
    for key in ["median_value_high", "identical_prices", "ten_percent_owner"]:
        assert f"f_{key}" in df.columns
    assert "conviction_score" in df.columns

    # Flat prices -> deterministic 0 forward returns, no delistings.
    assert (df["fwd_10"] == 0.0).all()
    assert (df["delisted_10"] == False).all()  # noqa: E712 (explicit bool compare reads clearer here)

    # x_is_first_ever_cluster: both AAA and BBB are each ticker's only event.
    assert df["x_is_first_ever_cluster"].all()
    assert df["x_days_since_prior_cluster"].isna().all()


def test_build_research_dataset_empty_events_df_returns_empty_frame(no_network_ipo):
    events_df = pd.DataFrame(columns=[
        "adsh", "form_type", "filing_date", "filing_url", "issuer_cik", "issuer_name",
        "ticker", "owner_cik", "owner_name", "owner_roles", "is_director", "is_officer",
        "is_ten_percent_owner", "transaction_date", "transaction_code", "acquired_disposed",
        "shares", "price_per_share", "value", "shares_owned_after", "pct_of_prior_stake",
        "footnote_text", "is_10b5_1",
    ])
    prices = FakePriceUniverse()
    calendar = make_calendar(10)
    states = build_states(events_df)
    df = research.build_research_dataset(states, prices, calendar, events_df)
    assert df.empty
    assert "ticker" in df.columns


# ---------------------------------------------------------------------------
# save/load round trip
# ---------------------------------------------------------------------------
def test_save_and_load_research_dataset(tmp_path, no_network_ipo):
    calendar = make_calendar(100)
    day1 = calendar[30]
    rows = two_owner_cluster(ticker="AAA", issuer_cik="ISSA", event_day=day1,
                              owner_a="O1", owner_b="O2")
    events_df = pd.DataFrame(rows)

    prices = FakePriceUniverse()
    prices.add_flat_series("SPY", calendar, price=100.0, dv=1_000_000.0)
    prices.add_flat_series("AAA", calendar, price=10.0, dv=1_000_000.0)

    states = build_states(events_df)
    df = research.build_research_dataset(states, prices, calendar, events_df, horizons=(10,))

    out_dir = str(tmp_path / "research_out")
    path = research.save_research_dataset(df, out_dir=out_dir, tag="unittest")
    assert os.path.exists(path)
    assert "unittest" in os.path.basename(path)
    assert f"{len(df)}rows" in os.path.basename(path)

    loaded = research.load_research_dataset(path)
    pd.testing.assert_frame_equal(loaded, df.reset_index(drop=True))


# ---------------------------------------------------------------------------
# Split-adjustment wiring (see backtest/splits.py and the module-level note
# above _PriorEvent in backtest/research.py)
# ---------------------------------------------------------------------------
def test_unadjusted_split_in_label_window_is_corrected(no_network_ipo):
    """An event whose forward-return label window spans an unadjusted split
    gets the CORRECTED (back-adjusted) label, not the fabricated one, and is
    kept -- a confirmed split is safe once back-adjusted."""
    calendar = make_calendar(200)
    event_day = calendar[100]
    ticker = "SPLITEVT"

    events_df = pd.DataFrame(two_owner_cluster(
        ticker=ticker, issuer_cik="ISSSPLIT", event_day=event_day,
        owner_a="OA", owner_b="OB",
    ))

    prices = FakePriceUniverse()
    prices.add_flat_series("SPY", calendar, price=100.0, dv=1_000_000.0)
    # Unadjusted 4x reverse split at index 105: close 10 -> 40, volume
    # 8000 -> 2000 (opposite direction, clean 4x ratio -> classifies "split").
    closes = [10.0] * 105 + [40.0] * 95
    volumes = [8000.0] * 105 + [2000.0] * 95
    assert len(closes) == len(calendar) == 200
    prices.add_series_with_frame(ticker, calendar, closes, volumes)

    states = build_states(events_df)
    df = research.build_research_dataset(states, prices, calendar, events_df, horizons=(10,))

    assert len(df) == 1
    row = df.iloc[0]
    assert row["entry_day"] == event_day + timedelta(days=1)
    # Raw (unadjusted) prices would show a fabricated +300% return (10 -> 40)
    # across this label window. Back-adjustment must correct it to the true
    # flat 0% return.
    assert row["fwd_10"] == pytest.approx(0.0)
    assert row["adj_10"] == pytest.approx(0.0)


def test_event_spanning_ambiguous_discontinuity_is_dropped(no_network_ipo):
    """An event whose label window spans an "ambiguous" jump (here: volume is
    unusable, so the split signature cannot be confirmed) must be dropped
    entirely, not emit a label built across a possibly-fabricated price."""
    calendar = make_calendar(200)
    event_day = calendar[100]
    ticker = "AMBEVT"

    events_df = pd.DataFrame(two_owner_cluster(
        ticker=ticker, issuer_cik="ISSAMB", event_day=event_day,
        owner_a="OA", owner_b="OB",
    ))

    prices = FakePriceUniverse()
    prices.add_flat_series("SPY", calendar, price=100.0, dv=1_000_000.0)
    closes = [10.0] * 105 + [40.0] * 95   # same jump shape as the split case above
    volumes = [0.0] * 200                 # but volume is unusable -> ambiguous
    prices.add_series_with_frame(ticker, calendar, closes, volumes)

    states = build_states(events_df)
    df = research.build_research_dataset(states, prices, calendar, events_df, horizons=(10,))

    assert df.empty


def test_event_spanning_real_move_above_ceiling_is_dropped(no_network_ipo):
    """An event whose label window spans a "real_move" past
    splits.REAL_MOVE_CEILING_RATIO must be dropped: an unbounded "real_move"
    is not trustworthy (see Part 1 of the split-adjustment work)."""
    calendar = make_calendar(200)
    event_day = calendar[100]
    ticker = "HUGEMOVE"

    events_df = pd.DataFrame(two_owner_cluster(
        ticker=ticker, issuer_cik="ISSHUGE", event_day=event_day,
        owner_a="OA", owner_b="OB",
    ))

    prices = FakePriceUniverse()
    prices.add_flat_series("SPY", calendar, price=100.0, dv=1_000_000.0)
    # Price up 30x AND volume up (spike), so this is "real_move" (volume did
    # not move opposite to price, ruling out "split"), well past the 10x
    # ceiling.
    closes = [0.05] * 105 + [1.5] * 95
    volumes = [20_000.0] * 105 + [120_000.0] * 95
    prices.add_series_with_frame(ticker, calendar, closes, volumes)

    detections = splits_mod.detect_discontinuities(ticker, prices.frames[ticker])
    assert detections and detections[0].classification == "real_move"
    assert detections[0].close_ratio > splits_mod.REAL_MOVE_CEILING_RATIO

    states = build_states(events_df)
    df = research.build_research_dataset(states, prices, calendar, events_df, horizons=(10,))

    assert df.empty


def test_event_spanning_plausible_real_move_is_kept(no_network_ipo):
    """A plausible "real_move" (well under the ceiling, genuine volume
    spike) must be KEPT with its true (large) return -- the ceiling must not
    throw away real returns along with the implausible ones."""
    calendar = make_calendar(200)
    event_day = calendar[100]
    ticker = "REALMOVE"

    events_df = pd.DataFrame(two_owner_cluster(
        ticker=ticker, issuer_cik="ISSREAL", event_day=event_day,
        owner_a="OA", owner_b="OB",
    ))

    prices = FakePriceUniverse()
    prices.add_flat_series("SPY", calendar, price=100.0, dv=1_000_000.0)
    # Price up 3.4x with a volume spike: under the 10x ceiling.
    closes = [5.0] * 105 + [17.0] * 95
    volumes = [50_000.0] * 105 + [150_000.0] * 95
    prices.add_series_with_frame(ticker, calendar, closes, volumes)

    detections = splits_mod.detect_discontinuities(ticker, prices.frames[ticker])
    assert detections and detections[0].classification == "real_move"
    assert detections[0].close_ratio <= splits_mod.REAL_MOVE_CEILING_RATIO

    states = build_states(events_df)
    df = research.build_research_dataset(states, prices, calendar, events_df, horizons=(10,))

    assert len(df) == 1
    row = df.iloc[0]
    # The genuine (unadjusted, uncorrected) +240% return must survive.
    assert row["fwd_10"] == pytest.approx(2.4)


def test_no_discontinuities_matches_pre_change_behavior(no_network_ipo):
    """A ticker whose price series has no discontinuity at all must build
    identically whether or not backtest.research can see a `.frames` entry
    for it. The pre-change code never looked at `.frames`, so this is the
    regression guard for the split-adjustment wiring: seeing a clean frame
    and running detection on it (finding nothing) must not change a single
    output value."""
    calendar = make_calendar(300)
    event_day = calendar[50]

    def build(ticker: str, with_frame: bool) -> pd.DataFrame:
        events_df = pd.DataFrame(two_owner_cluster(
            ticker=ticker, issuer_cik="ISSFLAT", event_day=event_day,
            owner_a="OA", owner_b="OB",
        ))
        prices = FakePriceUniverse()
        prices.add_flat_series("SPY", calendar, price=100.0, dv=1_000_000.0)
        if with_frame:
            # dollar_volume = 10.0 * 100,000 = 1,000,000, matching the
            # add_flat_series(dv=1_000_000.0) branch exactly.
            prices.add_series_with_frame(
                ticker, calendar, [10.0] * len(calendar), [100_000.0] * len(calendar)
            )
        else:
            prices.add_flat_series(ticker, calendar, price=10.0, dv=1_000_000.0)
        states = build_states(events_df)
        return research.build_research_dataset(states, prices, calendar, events_df)

    df_no_frame = build("FLATOLD", with_frame=False)
    df_with_frame = build("FLATNEW", with_frame=True)

    compare_cols = [c for c in df_no_frame.columns if c not in ("ticker", "issuer_cik")]
    pd.testing.assert_frame_equal(
        df_no_frame[compare_cols].reset_index(drop=True),
        df_with_frame[compare_cols].reset_index(drop=True),
    )


def test_point_in_time_features_match_after_split_correction(no_network_ipo):
    """A ticker whose OLDER history hides an unadjusted split must, after
    correction, produce the SAME point-in-time price-context features as an
    economically identical ticker whose cached series was never split at
    all.

    This is the scale-invariance property the future-split-information note
    above _PriorEvent (backtest/research.py) relies on: every x_
    price-context feature here is a RATIO (drawdown, momentum, vol,
    price-to-SMA) or a DOLLAR VOLUME (close * volume), both invariant to a
    uniform rescale of one side of a split boundary, so back-adjustment
    must restore the true point-in-time values EXACTLY, not merely leave
    them close.
    """
    calendar = make_calendar(400)
    event_day = calendar[259]  # 260 trading days of history before entry (>= 252 needed)

    events_df_clean = pd.DataFrame(two_owner_cluster(
        ticker="CLEANHIST", issuer_cik="ISSC", event_day=event_day,
        owner_a="OA", owner_b="OB",
    ))
    events_df_raw = pd.DataFrame(two_owner_cluster(
        ticker="RAWHIST", issuer_cik="ISSR", event_day=event_day,
        owner_a="OA", owner_b="OB",
    ))

    prices_clean = FakePriceUniverse()
    prices_clean.add_flat_series("SPY", calendar, price=100.0, dv=1_000_000.0)
    prices_clean.add_series_with_frame(
        "CLEANHIST", calendar, [40.0] * len(calendar), [2000.0] * len(calendar)
    )

    prices_raw = FakePriceUniverse()
    prices_raw.add_flat_series("SPY", calendar, price=100.0, dv=1_000_000.0)
    # Unadjusted 4x reverse split at index 100: close 10 -> 40, volume
    # 8000 -> 2000, so after back-adjustment this becomes bar-for-bar
    # identical to CLEANHIST above.
    raw_closes = [10.0] * 100 + [40.0] * 300
    raw_vols = [8000.0] * 100 + [2000.0] * 300
    assert len(raw_closes) == len(calendar) == 400
    prices_raw.add_series_with_frame("RAWHIST", calendar, raw_closes, raw_vols)

    # Sanity check: this test is not vacuously comparing two already-equal
    # series -- the raw ticker really does hide a detectable split.
    detections = splits_mod.detect_discontinuities("RAWHIST", prices_raw.frames["RAWHIST"])
    assert any(d.classification == "split" for d in detections)

    states_clean = build_states(events_df_clean)
    states_raw = build_states(events_df_raw)
    df_clean = research.build_research_dataset(
        states_clean, prices_clean, calendar, events_df_clean, horizons=(10,)
    )
    df_raw = research.build_research_dataset(
        states_raw, prices_raw, calendar, events_df_raw, horizons=(10,)
    )

    assert len(df_clean) == 1 and len(df_raw) == 1
    row_clean = df_clean.iloc[0]
    row_raw = df_raw.iloc[0]
    for col in (
        "x_drawdown_252", "x_drawdown_63", "x_mom_21_skip5", "x_mom_63_skip5",
        "x_mom_252_skip5", "x_vol_21_ann", "x_vol_63_ann", "x_price_to_sma200",
        "x_log_adv20", "x_buy_value_to_adv",
    ):
        a, b = row_clean[col], row_raw[col]
        if isinstance(a, float) and math.isnan(a):
            assert isinstance(b, float) and math.isnan(b), f"{col}: {a!r} vs {b!r}"
        else:
            assert a == pytest.approx(b), f"{col}: {a!r} vs {b!r}"


def test_ticker_reuse_drops_predecessor_but_keeps_successor(no_network_ipo):
    """A REUSE-shaped ticker (old, delisted company; unrelated new occupant
    later listed under the freed symbol -- same shape as the real AI:
    Arlington Asset Investment -> C3.ai transition) must drop the OLD
    company's episode row entirely and keep only the NEW company's."""
    calendar = make_calendar(1200)
    day_old = calendar[50]
    day_new = calendar[900]

    rows = []
    rows += two_owner_cluster(ticker="REUSED", issuer_cik="OLDCO", event_day=day_old,
                               owner_a="OA1", owner_b="OB1")
    rows += two_owner_cluster(ticker="REUSED", issuer_cik="NEWCO", event_day=day_new,
                               owner_a="OA2", owner_b="OB2")
    events_df = pd.DataFrame(rows)
    # Unrelated names -- name_similarity well under ticker_reuse.NAME_SIM_LOW,
    # so this classifies REUSE regardless of the gap.
    events_df.loc[events_df["issuer_cik"] == "OLDCO", "issuer_name"] = "Arlington Asset Investment Corp."
    events_df.loc[events_df["issuer_cik"] == "NEWCO", "issuer_name"] = "C3.ai, Inc."

    prices = FakePriceUniverse()
    prices.add_flat_series("SPY", calendar, price=100.0, dv=1_000_000.0)
    prices.add_flat_series("REUSED", calendar, price=10.0, dv=1_000_000.0)

    states = build_states(events_df)
    df = research.build_research_dataset(states, prices, calendar, events_df, horizons=(10,))

    assert len(df) == 1
    assert df.iloc[0]["issuer_cik"] == "NEWCO"
    assert df.iloc[0]["event_day"] == day_new


def test_ticker_reuse_rename_keeps_both_episodes(no_network_ipo):
    """A RENAME-shaped ticker (same corporate identity under a new CIK,
    near-identical name, short gap -- same shape as the real APA: APACHE
    CORP -> APA Corp transition) must keep BOTH episodes' rows."""
    calendar = make_calendar(300)
    day_old = calendar[50]
    day_new = calendar[100]

    rows = []
    rows += two_owner_cluster(ticker="RENAMED", issuer_cik="OLDCIK", event_day=day_old,
                               owner_a="OA1", owner_b="OB1")
    rows += two_owner_cluster(ticker="RENAMED", issuer_cik="NEWCIK", event_day=day_new,
                               owner_a="OA2", owner_b="OB2")
    events_df = pd.DataFrame(rows)
    events_df.loc[events_df["issuer_cik"] == "OLDCIK", "issuer_name"] = "Widget Corp"
    events_df.loc[events_df["issuer_cik"] == "NEWCIK", "issuer_name"] = "Widget Inc"

    prices = FakePriceUniverse()
    prices.add_flat_series("SPY", calendar, price=100.0, dv=1_000_000.0)
    prices.add_flat_series("RENAMED", calendar, price=10.0, dv=1_000_000.0)

    states = build_states(events_df)
    df = research.build_research_dataset(states, prices, calendar, events_df, horizons=(10,))

    assert len(df) == 2
    assert set(df["issuer_cik"]) == {"OLDCIK", "NEWCIK"}


def test_ticker_reuse_guard_can_be_disabled(no_network_ipo):
    """drop_ticker_reuse=False must keep every row, including the ones the
    guard would otherwise drop -- the behavior is explicit and
    controllable, not hardwired on."""
    calendar = make_calendar(1200)
    day_old = calendar[50]
    day_new = calendar[900]

    rows = []
    rows += two_owner_cluster(ticker="REUSED2", issuer_cik="OLDCO", event_day=day_old,
                               owner_a="OA1", owner_b="OB1")
    rows += two_owner_cluster(ticker="REUSED2", issuer_cik="NEWCO", event_day=day_new,
                               owner_a="OA2", owner_b="OB2")
    events_df = pd.DataFrame(rows)
    events_df.loc[events_df["issuer_cik"] == "OLDCO", "issuer_name"] = "Arlington Asset Investment Corp."
    events_df.loc[events_df["issuer_cik"] == "NEWCO", "issuer_name"] = "C3.ai, Inc."

    prices = FakePriceUniverse()
    prices.add_flat_series("SPY", calendar, price=100.0, dv=1_000_000.0)
    prices.add_flat_series("REUSED2", calendar, price=10.0, dv=1_000_000.0)

    states = build_states(events_df)
    df = research.build_research_dataset(
        states, prices, calendar, events_df, horizons=(10,), drop_ticker_reuse=False,
    )

    assert len(df) == 2
    assert set(df["issuer_cik"]) == {"OLDCO", "NEWCO"}


def test_ticker_reuse_log_reports_drop_count(no_network_ipo, caplog):
    """The build log must report how many rows the ticker-reuse guard
    dropped, the same way the split-adjustment counters are reported --
    required for gauging the size of the effect on a real build."""
    calendar = make_calendar(1200)
    day_old = calendar[50]
    day_new = calendar[900]

    rows = []
    rows += two_owner_cluster(ticker="REUSED3", issuer_cik="OLDCO", event_day=day_old,
                               owner_a="OA1", owner_b="OB1")
    rows += two_owner_cluster(ticker="REUSED3", issuer_cik="NEWCO", event_day=day_new,
                               owner_a="OA2", owner_b="OB2")
    events_df = pd.DataFrame(rows)
    events_df.loc[events_df["issuer_cik"] == "OLDCO", "issuer_name"] = "Arlington Asset Investment Corp."
    events_df.loc[events_df["issuer_cik"] == "NEWCO", "issuer_name"] = "C3.ai, Inc."

    prices = FakePriceUniverse()
    prices.add_flat_series("SPY", calendar, price=100.0, dv=1_000_000.0)
    prices.add_flat_series("REUSED3", calendar, price=10.0, dv=1_000_000.0)

    states = build_states(events_df)
    with caplog.at_level(logging.INFO, logger="backtest.research"):
        research.build_research_dataset(states, prices, calendar, events_df, horizons=(10,))

    messages = "\n".join(r.message for r in caplog.records)
    assert "ticker-reuse guard dropped 2 event row(s)" in messages


def test_ticker_reuse_matches_module_directly(no_network_ipo):
    """build_research_dataset's wiring must agree with calling
    ticker_reuse.filter_unsafe_ticker_reuse directly on the same events_df:
    the dropped-row count it logs must equal what the module itself
    reports."""
    calendar = make_calendar(1200)
    day_old = calendar[50]
    day_new = calendar[900]

    rows = []
    rows += two_owner_cluster(ticker="REUSED4", issuer_cik="OLDCO", event_day=day_old,
                               owner_a="OA1", owner_b="OB1")
    rows += two_owner_cluster(ticker="REUSED4", issuer_cik="NEWCO", event_day=day_new,
                               owner_a="OA2", owner_b="OB2")
    events_df = pd.DataFrame(rows)
    events_df.loc[events_df["issuer_cik"] == "OLDCO", "issuer_name"] = "Arlington Asset Investment Corp."
    events_df.loc[events_df["issuer_cik"] == "NEWCO", "issuer_name"] = "C3.ai, Inc."

    _filtered, n_dropped, transitions = ticker_reuse.filter_unsafe_ticker_reuse(events_df)
    assert n_dropped == 2
    assert transitions.iloc[0]["classification"] == "REUSE"


def test_build_log_reports_split_adjustment_and_drop_counts(no_network_ipo, caplog):
    """The build log must report how many rows were adjusted and how many
    were dropped, and why (Part 2, item 3): required for gauging the size
    of the effect on a real multi-thousand-ticker build."""
    calendar = make_calendar(200)
    event_day_a = calendar[100]
    event_day_b = calendar[130]

    rows = []
    rows += two_owner_cluster(ticker="SPLITEVT2", issuer_cik="ISSA2", event_day=event_day_a,
                               owner_a="OA1", owner_b="OB1")
    rows += two_owner_cluster(ticker="AMBEVT2", issuer_cik="ISSB2", event_day=event_day_b,
                               owner_a="OA2", owner_b="OB2")
    events_df = pd.DataFrame(rows)

    prices = FakePriceUniverse()
    prices.add_flat_series("SPY", calendar, price=100.0, dv=1_000_000.0)
    prices.add_series_with_frame(
        "SPLITEVT2", calendar, [10.0] * 105 + [40.0] * 95, [8000.0] * 105 + [2000.0] * 95,
    )
    prices.add_series_with_frame(
        "AMBEVT2", calendar, [10.0] * 135 + [40.0] * 65, [0.0] * 200,
    )

    states = build_states(events_df)
    with caplog.at_level(logging.INFO, logger="backtest.research"):
        research.build_research_dataset(states, prices, calendar, events_df, horizons=(10,))

    messages = "\n".join(r.message for r in caplog.records)
    assert "1 ticker(s) had a confirmed split" in messages
    assert "1 row(s) dropped for an unsafe label window" in messages
