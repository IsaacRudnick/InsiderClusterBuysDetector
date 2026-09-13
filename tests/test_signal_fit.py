"""Unit tests for the learned-signal-weights feature set.

Covers:
  - insider_cluster_buys._component_flags / _score_cluster parameterization
    (candidate components, weight injection, zero-weight omission)
  - insider_cluster_buys.load_signal_weights (precedence, merge, corrupt/
    missing file handling) and _coerce_date (the filing-delay bug fix)
  - backtest.state.DailyStateBuilder dual scoring (learned_score / cache
    invalidation) and the updated empty_state shape
  - backtest.signal_fit.build_event_dataset (episode-start dedup, entry-day
    resolution, forward/adjusted-return computation, missing-price skip)
  - backtest.signal_fit.fit_weights (sign recovery, support/t gates) and the
    _coefs_to_points gates in isolation
  - backtest/strategies.py learned_* strategies (learned_gt_p00/p03,
    learned_score_weighted)
  - backtest.report.render_html's conditional "Learned signal weights"
    section and backtest.signal_fit.oos_strategy_stats

All synthetic — no network, no SEC fetch, no yfinance.

Run with:
    python -m unittest tests.test_signal_fit -v
or:
    python tests/test_signal_fit.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import unittest
from collections import Counter
from datetime import date, datetime, timedelta
from unittest import mock

import numpy as np
import pandas as pd

# Make the project importable when run as a script.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import insider_cluster_buys as ics
from backtest import engine, metrics, report, signal_fit, strategies
from backtest import state as state_mod
from backtest.prices import PriceUniverse


# ---------------------------------------------------------------------------
# Synthetic fixtures (mirrors tests/test_backtest_additions.py's style)
# ---------------------------------------------------------------------------
def _make_calendar(start: date, n_days: int) -> list[date]:
    """Strictly business-day calendar (Mon-Fri). 5 days/week."""
    out: list[date] = []
    d = start
    while len(out) < n_days:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def _make_price_universe(
    ticker_prices: dict[str, list[float]],
    calendar: list[date],
    volume: float = 5_000_000.0,
) -> PriceUniverse:
    """Build a PriceUniverse from synthetic close prices (open == close)."""
    pu = PriceUniverse()
    for t, closes in ticker_prices.items():
        assert len(closes) == len(calendar), f"{t} prices/calendar length mismatch"
        df = pd.DataFrame(
            {
                "open": closes,
                "high": closes,
                "low": closes,
                "close": closes,
                "volume": [volume] * len(closes),
                "dollar_volume": [c * volume for c in closes],
            },
            index=pd.DatetimeIndex([pd.Timestamp(d) for d in calendar], name="date"),
        )
        pu.frames[t] = df
    pu.finalize()
    return pu


class _StubStates:
    """Stand-in for DailyStateBuilder.state_for_day."""

    def __init__(self, by_day: dict[date, dict[str, dict]]):
        self._by_day = by_day

    def state_for_day(self, D: date) -> dict[str, dict]:
        return dict(self._by_day.get(D, {}))


def _signal_state(ticker: str, as_of: date, **overrides) -> dict:
    """Start from `empty_state` and overlay overrides."""
    base = state_mod.empty_state(ticker, as_of)
    base.update(overrides)
    return base


def _window_row(**overrides) -> dict:
    """A minimal insider-transaction row for _component_flags(window, cluster).
    Defaults are inert (won't accidentally fire conditions); override what a
    given test needs to exercise."""
    row = {
        "value": 0.0,
        "price_per_share": None,
        "transaction_date": None,
        "filing_date": None,
        "shares": 0.0,
        "owner_cik": "X",
        "owner_name": "X",
        "is_director": False,
        "is_officer": False,
        "is_ten_percent_owner": False,
        "footnote_text": "",
        "is_10b5_1": False,
        "pct_of_prior_stake": None,
    }
    row.update(overrides)
    return row


def _cluster(**overrides) -> dict:
    base = {
        "insiders": [],
        "includes_ten_percent_owner": False,
        "includes_director": False,
        "includes_officer": False,
        "is_recent_ipo": False,
        "max_pct_of_prior_stake": None,
        "num_insiders": 0,
        "total_value": 0.0,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# (a) _component_flags / _score_cluster parameterization
# ---------------------------------------------------------------------------
class TestComponentFlagsAndScoreCluster(unittest.TestCase):
    def setUp(self):
        # 3-insider window: all directors (no officers), one is also a 10%
        # owner, $200K median value, 3 distinct transaction dates, prompt
        # filings (1-day delay), modest stake increases (well under the
        # big/huge-stake thresholds).
        self.window = [
            _window_row(
                value=200_000.0, price_per_share=20.0,
                transaction_date=date(2024, 1, 2), filing_date=date(2024, 1, 3),
                shares=10_000, owner_cik="A", owner_name="Alice",
                is_director=True, is_ten_percent_owner=True, pct_of_prior_stake=10.0,
            ),
            _window_row(
                value=200_000.0, price_per_share=21.0,
                transaction_date=date(2024, 1, 3), filing_date=date(2024, 1, 4),
                shares=9_500, owner_cik="B", owner_name="Bob",
                is_director=True, pct_of_prior_stake=5.0,
            ),
            _window_row(
                value=200_000.0, price_per_share=22.0,
                transaction_date=date(2024, 1, 4), filing_date=date(2024, 1, 5),
                shares=9_000, owner_cik="C", owner_name="Carol",
                is_director=True, pct_of_prior_stake=5.0,
            ),
        ]
        self.cluster = _cluster(
            insiders=[{"value": 200_000.0}, {"value": 200_000.0}, {"value": 200_000.0}],
            includes_ten_percent_owner=True,
            includes_director=True,
            includes_officer=False,
            max_pct_of_prior_stake=10.0,
            num_insiders=3,
            total_value=600_000.0,
        )

    def test_component_flags_fires_expected_keys(self):
        flags = ics._component_flags(self.window, self.cluster)
        keys = {f["key"] for f in flags}
        expected = {
            "median_value_high", "ten_percent_owner", "directors_only",
            "three_plus_insiders", "multi_date_spread", "filed_promptly",
        }
        self.assertTrue(expected.issubset(keys), keys)
        # And every flag has non-empty text.
        for f in flags:
            self.assertTrue(f["text"])

    def test_score_cluster_with_default_weights_reproduces_legacy_deltas(self):
        contributions = ics._score_cluster(self.window, self.cluster, weights=ics.DEFAULT_WEIGHTS)
        by_key = {c["key"]: c["delta"] for c in contributions}
        self.assertEqual(by_key["median_value_high"], 3)
        self.assertEqual(by_key["ten_percent_owner"], 3)
        self.assertEqual(by_key["directors_only"], 2)
        self.assertEqual(by_key["multi_date_spread"], 1)
        self.assertEqual(by_key["filed_promptly"], 1)
        # Candidate component has a zero default weight -> dropped entirely.
        self.assertNotIn("three_plus_insiders", by_key)

    def test_custom_weights_change_deltas_and_omit_zero_weight_components(self):
        custom = dict(ics.DEFAULT_WEIGHTS)
        custom["three_plus_insiders"] = 5
        custom["ten_percent_owner"] = 0
        contributions = ics._score_cluster(self.window, self.cluster, weights=custom)
        by_key = {c["key"]: c["delta"] for c in contributions}
        self.assertEqual(by_key["three_plus_insiders"], 5)
        self.assertNotIn("ten_percent_owner", by_key)
        # Untouched components keep their default delta.
        self.assertEqual(by_key["median_value_high"], 3)

    def test_filed_promptly_fires_with_date_object_delay(self):
        # Regression test for the strptime bug: filing/transaction dates as
        # datetime.date objects (as the backtest passes them) with a <=2-day
        # delay must now fire filed_promptly.
        window = [
            _window_row(transaction_date=date(2024, 1, 2), filing_date=date(2024, 1, 3)),
            _window_row(transaction_date=date(2024, 1, 3), filing_date=date(2024, 1, 4)),
        ]
        cluster = _cluster(num_insiders=2)
        flags = ics._component_flags(window, cluster)
        keys = {f["key"] for f in flags}
        self.assertIn("filed_promptly", keys)


# ---------------------------------------------------------------------------
# (b) load_signal_weights + _coerce_date
# ---------------------------------------------------------------------------
class TestLoadSignalWeights(unittest.TestCase):
    def test_missing_file_returns_defaults(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "does_not_exist.json")
            weights, source = ics.load_signal_weights(path)
            self.assertEqual(weights, ics.DEFAULT_WEIGHTS)
            self.assertEqual(source, "defaults")

    def test_partial_payload_merges_over_defaults(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "signal_weights.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"version": 1, "weights": {"ten_percent_owner": 7}}, fh)
            weights, source = ics.load_signal_weights(path)
            self.assertEqual(weights["ten_percent_owner"], 7)
            for k, v in ics.DEFAULT_WEIGHTS.items():
                if k != "ten_percent_owner":
                    self.assertEqual(weights[k], v)
            self.assertEqual(source, path)

    def test_bare_flat_dict_payload_also_merges(self):
        # load_signal_weights accepts either the full fit payload
        # ({"weights": {...}}) or a bare {key: int} override file.
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "flat.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"routine_footnote": -9}, fh)
            weights, source = ics.load_signal_weights(path)
            self.assertEqual(weights["routine_footnote"], -9)
            self.assertEqual(source, path)

    def test_unknown_key_ignored(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "unknown.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"weights": {"not_a_real_component": 99}}, fh)
            weights, source = ics.load_signal_weights(path)
            self.assertEqual(weights, ics.DEFAULT_WEIGHTS)
            self.assertNotIn("not_a_real_component", weights)

    def test_corrupt_json_falls_back_to_defaults(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "corrupt.json")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("{not valid json")
            weights, source = ics.load_signal_weights(path)
            self.assertEqual(weights, ics.DEFAULT_WEIGHTS)
            self.assertEqual(source, "defaults")


class TestCoerceDate(unittest.TestCase):
    def test_date_object_passthrough(self):
        d = date(2024, 1, 5)
        self.assertEqual(ics._coerce_date(d), d)

    def test_datetime_object_truncated_to_date(self):
        dt = datetime(2024, 1, 5, 10, 30)
        self.assertEqual(ics._coerce_date(dt), date(2024, 1, 5))

    def test_iso_string(self):
        self.assertEqual(ics._coerce_date("2024-01-05"), date(2024, 1, 5))

    def test_compact_string(self):
        self.assertEqual(ics._coerce_date("20240105"), date(2024, 1, 5))

    def test_garbage_string_returns_none(self):
        self.assertIsNone(ics._coerce_date("not-a-date"))

    def test_none_returns_none(self):
        self.assertIsNone(ics._coerce_date(None))


# ---------------------------------------------------------------------------
# (c) DailyStateBuilder dual scoring
# ---------------------------------------------------------------------------
class TestDailyStateBuilderLearnedScore(unittest.TestCase):
    def _build(self):
        events = pd.DataFrame([
            {
                "ticker": "ABC",
                "transaction_date": date(2024, 1, 5),
                "filing_date": date(2024, 1, 6),
                "owner_cik": "A",
                "owner_name": "Alice",
                "owner_roles": "Director",
                "shares": 100.0,
                "value": 200_000.0,
                "price_per_share": 2000.0,
                "is_director": True,
                "is_officer": False,
                "is_ten_percent_owner": True,
                "pct_of_prior_stake": 30.0,
                "is_10b5_1": True,
                "footnote_text": "",
            },
            {
                "ticker": "ABC",
                "transaction_date": date(2024, 1, 6),
                "filing_date": date(2024, 1, 7),
                "owner_cik": "B",
                "owner_name": "Bob",
                "owner_roles": "Officer",
                "shares": 50.0,
                "value": 150_000.0,
                "price_per_share": 3000.0,
                "is_director": False,
                "is_officer": True,
                "is_ten_percent_owner": False,
                "pct_of_prior_stake": None,
                "is_10b5_1": False,
                "footnote_text": "",
            },
        ])
        with mock.patch("ipo_lookup.get_first_trade_date", return_value=None):
            return state_mod.DailyStateBuilder(events, window_days=14)

    def test_learned_score_none_without_weights(self):
        builder = self._build()
        st = builder.state_for_day(date(2024, 1, 7))["ABC"]
        self.assertIsNone(st["learned_score"])
        self.assertGreater(len(st["component_keys"]), 0)

    def test_learned_score_sums_learned_weights_over_fired_keys(self):
        builder = self._build()
        day = date(2024, 1, 7)
        st = builder.state_for_day(day)["ABC"]
        keys = st["component_keys"]
        weights_map = {k: (i + 1) for i, k in enumerate(keys)}

        builder.set_learned_weights(weights_map)
        st2 = builder.state_for_day(day)["ABC"]

        # Fired flags are unaffected by which score is applied to them.
        self.assertEqual(st2["component_keys"], keys)
        expected = sum(weights_map.get(k, 0) for k in st2["component_keys"])
        self.assertEqual(st2["learned_score"], expected)

    def test_set_learned_weights_invalidates_day_cache(self):
        builder = self._build()
        day = date(2024, 1, 7)
        builder.state_for_day(day)  # populate cache
        self.assertIn(day, builder._state_cache)
        cached_before = builder._state_cache[day]

        builder.set_learned_weights({"ten_percent_owner": 9})
        self.assertEqual(builder._state_cache, {})

        builder.state_for_day(day)  # rebuild
        cached_after = builder._state_cache[day]
        self.assertIsNot(cached_before, cached_after)
        self.assertIsNotNone(cached_after["ABC"]["learned_score"])

    def test_empty_state_shape(self):
        st = state_mod.empty_state("ABC", date(2024, 1, 1))
        self.assertEqual(st["learned_score"], 0)
        self.assertEqual(st["tail_score"], 0)
        self.assertEqual(st["component_keys"], [])

    def test_tail_score_none_without_weights(self):
        builder = self._build()
        st = builder.state_for_day(date(2024, 1, 7))["ABC"]
        self.assertIsNone(st["tail_score"])

    def test_set_tail_weights_populates_tail_score_and_invalidates_cache(self):
        builder = self._build()
        day = date(2024, 1, 7)
        builder.state_for_day(day)  # populate cache
        self.assertIn(day, builder._state_cache)
        cached_before = builder._state_cache[day]

        st = cached_before["ABC"]
        keys = st["component_keys"]
        weights_map = {k: (i + 1) for i, k in enumerate(keys)}

        builder.set_tail_weights(weights_map)
        self.assertEqual(builder._state_cache, {})

        st2 = builder.state_for_day(day)["ABC"]
        cached_after = builder._state_cache[day]
        self.assertIsNot(cached_before, cached_after)
        expected = sum(weights_map.get(k, 0) for k in st2["component_keys"])
        self.assertEqual(st2["tail_score"], expected)


# ---------------------------------------------------------------------------
# (d) build_event_dataset
# ---------------------------------------------------------------------------
class TestBuildEventDataset(unittest.TestCase):
    def test_consecutive_qualifying_run_yields_one_event_with_correct_returns(self):
        n = 40
        cal = _make_calendar(date(2024, 1, 1), n)
        pu = _make_price_universe(
            {"ABC": [100.0 + i for i in range(n)], "SPY": [400.0] * n}, cal,
        )
        # ABC qualifies (>=2 insiders) on trading days 5..12 inclusive - a
        # single contiguous run driven by the rolling window, which should
        # dedup down to exactly one event at day 5.
        by_day = {
            cal[i]: {"ABC": _signal_state(
                "ABC", cal[i], num_insiders=2, component_keys=["ten_percent_owner"],
            )}
            for i in range(5, 13)
        }
        states = _StubStates(by_day)
        cfg = signal_fit.FitConfig(horizons=(5, 10), primary_horizon=5)

        df = signal_fit.build_event_dataset(states, pu, cal, cfg)

        self.assertEqual(len(df), 1)
        row = df.iloc[0]
        self.assertEqual(row["ticker"], "ABC")
        self.assertEqual(row["event_day"], cal[5])
        self.assertEqual(row["entry_day"], cal[6])  # next trading day after event
        self.assertEqual(row["f_ten_percent_owner"], 1)
        feature_cols = [f"f_{k}" for k in ics.DEFAULT_WEIGHTS if k != "ten_percent_owner"]
        self.assertEqual(sum(row[c] for c in feature_cols), 0)

        entry_open = 100.0 + 6  # cal[6]
        exit_5 = 100.0 + 11     # cal[6+5]
        exit_10 = 100.0 + 16    # cal[6+10]
        fwd_5 = exit_5 / entry_open - 1.0
        fwd_10 = exit_10 / entry_open - 1.0
        self.assertAlmostEqual(row["fwd_5"], fwd_5, places=9)
        self.assertAlmostEqual(row["fwd_10"], fwd_10, places=9)
        # SPY is flat -> spy_h == 0 for every horizon -> adj == fwd.
        self.assertAlmostEqual(row["adj_5"], fwd_5, places=9)
        self.assertAlmostEqual(row["adj_10"], fwd_10, places=9)

    def test_event_skipped_when_ticker_has_no_price_history(self):
        n = 40
        cal = _make_calendar(date(2024, 1, 1), n)
        pu = _make_price_universe({"SPY": [400.0] * n}, cal)  # no XYZ frame at all
        states = _StubStates({
            cal[3]: {"XYZ": _signal_state("XYZ", cal[3], num_insiders=2, component_keys=[])},
        })
        cfg = signal_fit.FitConfig(horizons=(5,), primary_horizon=5)
        df = signal_fit.build_event_dataset(states, pu, cal, cfg)
        self.assertTrue(df.empty)

    def test_event_skipped_when_qualifying_too_close_to_calendar_end(self):
        n = 40
        cal = _make_calendar(date(2024, 1, 1), n)
        pu = _make_price_universe(
            {"DEF": [50.0] * n, "SPY": [400.0] * n}, cal,
        )
        # Qualifies on the very last calendar day - no trading days remain
        # for an entry within the lookahead window.
        states = _StubStates({
            cal[-1]: {"DEF": _signal_state("DEF", cal[-1], num_insiders=2, component_keys=[])},
        })
        cfg = signal_fit.FitConfig(horizons=(5,), primary_horizon=5)
        df = signal_fit.build_event_dataset(states, pu, cal, cfg)
        self.assertTrue(df.empty)


# ---------------------------------------------------------------------------
# (e) fit_weights + _coefs_to_points gates
# ---------------------------------------------------------------------------
class TestCoefsToPointsGates(unittest.TestCase):
    def test_t_gate_zeroes_low_significance_feature(self):
        cfg = signal_fit.FitConfig(t_min=1.5, min_flag_count=20, point_cap=6)
        coefs = np.array([0.05, -0.05, 0.001])
        t_stats = np.array([3.0, -3.0, 0.5])   # third feature fails the t gate
        n_fired = np.array([50, 50, 50])       # all have plenty of support
        points = signal_fit._coefs_to_points(coefs, t_stats, n_fired, cfg)
        self.assertGreater(points[0], 0)
        self.assertLess(points[1], 0)
        self.assertEqual(points[2], 0)

    def test_support_gate_zeroes_low_count_feature(self):
        cfg = signal_fit.FitConfig(t_min=1.5, min_flag_count=20, point_cap=6)
        coefs = np.array([0.05, -0.05])
        t_stats = np.array([5.0, 5.0])   # both comfortably significant
        n_fired = np.array([50, 5])      # second fails the support gate
        points = signal_fit._coefs_to_points(coefs, t_stats, n_fired, cfg)
        self.assertNotEqual(points[0], 0)
        self.assertEqual(points[1], 0)


class TestFitWeights(unittest.TestCase):
    def _synthetic_events(self, n=200):
        """Deterministic (no RNG) event frame: f_ten_percent_owner adds a
        clean +5% to the target, f_routine_footnote subtracts 5%, and
        f_five_plus_insiders fires on only 3 rows (below the default
        min_flag_count=20 support gate). A small deterministic sawtooth
        residual keeps OLS t-stats finite (a perfectly-fit design would force
        se == 0 -> t forced to 0 by the code's zero-division guard)."""
        start = date(2020, 1, 1)
        rows = []
        for i in range(n):
            day = start + timedelta(days=i)
            fA = 1 if i % 2 == 0 else 0          # fires 100/200 times
            fB = 1 if i % 3 == 0 else 0           # fires ~67/200 times
            f_low_support = 1 if i < 3 else 0     # fires 3/200 times
            noise = 0.0007 * (i % 7) - 0.0021     # small, deterministic
            adj = 0.05 * fA - 0.05 * fB + noise
            row = {"ticker": "T", "event_day": day, "entry_day": day + timedelta(days=1)}
            for k in ics.DEFAULT_WEIGHTS:
                row[f"f_{k}"] = 0
            row["f_ten_percent_owner"] = fA
            row["f_routine_footnote"] = fB
            row["f_five_plus_insiders"] = f_low_support
            row["fwd_90"] = adj
            row["adj_90"] = adj
            rows.append(row)
        return pd.DataFrame(rows)

    def test_recovers_correct_signs_and_gates_low_support_feature(self):
        events = self._synthetic_events()
        cfg = signal_fit.FitConfig(
            horizons=(90,), primary_horizon=90, train_frac=0.70,
            ridge_lambdas=(0.0,), winsor_lo_pct=0.0, winsor_hi_pct=1.0, min_flag_count=20, t_min=1.5,
        )
        fit = signal_fit.fit_weights(events, cfg)
        self.assertIsNotNone(fit)

        stats_by_key = fit.feature_stats.set_index("component")
        self.assertGreater(stats_by_key.loc["ten_percent_owner", "coef"], 0.01)
        self.assertLess(stats_by_key.loc["routine_footnote", "coef"], -0.01)
        self.assertGreater(fit.weights_int["ten_percent_owner"], 0)
        self.assertLess(fit.weights_int["routine_footnote"], 0)

        # Fires only 3 times (< default min_flag_count=20) -> excluded from
        # the design matrix entirely -> 0 points regardless of any residual
        # correlation with the target.
        self.assertEqual(fit.weights_int["five_plus_insiders"], 0)

        # raw_uplift is reported alongside the (possibly shrunk/clipped)
        # learned coefficient.
        self.assertIn("raw_uplift", fit.feature_stats.columns)
        self.assertGreater(stats_by_key.loc["ten_percent_owner", "raw_uplift"], 0.0)
        self.assertLess(stats_by_key.loc["routine_footnote", "raw_uplift"], 0.0)

        # No test event before split_date: train/test counts must exactly
        # match a manual split of the (noiseless) synthetic frame.
        n_train_expected = int((events["event_day"] < fit.split_date).sum())
        n_test_expected = int((events["event_day"] >= fit.split_date).sum())
        self.assertEqual(fit.train_events, n_train_expected)
        self.assertEqual(fit.test_events, n_test_expected)


# ---------------------------------------------------------------------------
# (e.1) fit_tail_score (tail-probability / "moonshot" research-triage score)
# ---------------------------------------------------------------------------
class TestFitTailScore(unittest.TestCase):
    """Synthetic frame sized for FitConfig() defaults: train_frac=0.70 over
    800 rows gives exactly 560 train / 240 test events, and the 560 train
    rows split into two 280-row chronological halves.

    Three flags are engineered:
      - ten_percent_owner ("A"): fires 60x/half (120 train total, clears
        tail_min_flag_count=100). P(moonshot|A)=40% in BOTH halves against
        a lower per-half base rate (20% / ~11.8%) -> a stable, positive
        log2 lift in both halves -> should earn positive tail points.
      - routine_footnote ("B"): also fires 60x/half (120 train total).
        P(moonshot|B)=40% in the first half (positive lift, same as A) but
        collapses to ~1.7% in the second half (strongly negative lift).
        The pooled/train-wide lift is still net positive, but the sign
        flip between halves must zero its points (stability gate).
      - five_plus_insiders ("rare"): fires on only 30 rows total in train
        (15/half, < tail_min_flag_count=100) -> zero points regardless of
        its lift.
    """

    @staticmethod
    def _pool(n_total: int, n_moon: int, fA: int = 0, fB: int = 0,
              frare_first_n: int = 0) -> list[tuple]:
        return [
            (fA, fB, 1 if idx < frare_first_n else 0, idx < n_moon)
            for idx in range(n_total)
        ]

    def _synthetic_events(self) -> pd.DataFrame:
        blocks: list[tuple] = []
        # -- train half 1 (280 rows) --
        blocks += self._pool(60, 24, fA=1)                    # pool A: P(moonshot|A)=40%
        blocks += self._pool(60, 24, fB=1)                    # pool B: P(moonshot|B)=40% in h1
        blocks += self._pool(160, 8, frare_first_n=15)        # background (+15 rare fires)
        # -- train half 2 (280 rows) --
        blocks += self._pool(60, 24, fA=1)                    # pool A: P(moonshot|A)=40% again -> stable
        blocks += self._pool(60, 1, fB=1)                     # pool B: P(moonshot|B)~1.7% -> sign flip
        blocks += self._pool(160, 8, frare_first_n=15)        # background (+15 rare fires; 30 total)
        # -- test (240 rows) --
        blocks += self._pool(50, 20, fA=1)
        blocks += self._pool(50, 10, fB=1)
        blocks += self._pool(140, 7)

        start = date(2020, 1, 1)
        rows = []
        for i, (fA, fB, frare, is_moon) in enumerate(blocks):
            row = {"event_day": start + timedelta(days=i)}
            for k in ics.DEFAULT_WEIGHTS:
                row[f"f_{k}"] = 0
            row["f_ten_percent_owner"] = fA
            row["f_routine_footnote"] = fB
            row["f_five_plus_insiders"] = frare
            row["adj_90"] = 0.25 if is_moon else 0.01
            rows.append(row)
        return pd.DataFrame(rows)

    def test_stable_flag_positive_unstable_flag_and_rare_flag_zeroed(self):
        events = self._synthetic_events()
        cfg = signal_fit.FitConfig()  # defaults: tail_horizon=90, moonshot_thresh=0.20, ...
        tf = signal_fit.fit_tail_score(events, cfg)
        self.assertIsNotNone(tf)

        self.assertEqual(tf.train_events, 560)
        self.assertEqual(tf.test_events, 240)

        stats = tf.tail_stats.set_index("component")

        # Flag A: same-signed (positive) lift in both chronological halves
        # -> passes the stability gate -> positive tail points.
        self.assertGreater(stats.loc["ten_percent_owner", "lift_h1"], 0)
        self.assertGreater(stats.loc["ten_percent_owner", "lift_h2"], 0)
        self.assertGreater(tf.weights_int["ten_percent_owner"], 0)

        # Flag B: positive lift in h1, negative in h2 -> despite a net
        # positive pooled/train-wide lift, the stability gate zeroes it.
        self.assertGreater(stats.loc["routine_footnote", "lift_h1"], 0)
        self.assertLess(stats.loc["routine_footnote", "lift_h2"], 0)
        self.assertGreater(stats.loc["routine_footnote", "lift"], 0)
        self.assertEqual(tf.weights_int["routine_footnote"], 0)

        # Rare flag: fires only 30x in train (< tail_min_flag_count=100)
        # -> zero points regardless of lift.
        self.assertEqual(int(stats.loc["five_plus_insiders", "n_fired_train"]), 30)
        self.assertEqual(tf.weights_int["five_plus_insiders"], 0)

        # OOS quintile table rows must account for every test event with a
        # valid target (all 240 rows here, since adj_90 is never NaN).
        self.assertEqual(int(tf.oos_quintile_table["n"].sum()), tf.test_events)

    def test_none_when_too_few_train_rows(self):
        cfg = signal_fit.FitConfig()
        events = self._synthetic_events().iloc[:100].copy()  # far below the 500 train-row floor
        tf = signal_fit.fit_tail_score(events, cfg)
        self.assertIsNone(tf)


# ---------------------------------------------------------------------------
# (e.2) Asymmetric winsorization
# ---------------------------------------------------------------------------
class TestAsymmetricWinsorization(unittest.TestCase):
    def test_default_config_winsor_bounds(self):
        cfg = signal_fit.FitConfig()
        self.assertEqual(cfg.winsor_lo_pct, 0.01)
        self.assertEqual(cfg.winsor_hi_pct, 0.999)

    def test_extreme_positive_outlier_mostly_retained_not_clipped_to_99th_percentile(self):
        # A tight cluster of small baseline returns plus a single moonshot
        # outlier deep in the train split. Under the old symmetric 1%/99%
        # winsorization the outlier would have been crushed down to
        # roughly the 99th-percentile value (near the tiny baseline
        # cluster); under the new 1%/99.9% config it should survive
        # almost intact, since 99.9% of ~210 train rows lands past its rank.
        n = 300
        start = date(2020, 1, 1)
        rows = []
        for i in range(n):
            day = start + timedelta(days=i)
            row = {"ticker": "T", "event_day": day, "entry_day": day + timedelta(days=1)}
            for k in ics.DEFAULT_WEIGHTS:
                row[f"f_{k}"] = 0
            row["f_ten_percent_owner"] = 1 if i % 2 == 0 else 0
            adj = 0.001 * (i % 5)  # tight baseline cluster in [0, 0.004]
            row["fwd_90"] = adj
            row["adj_90"] = adj
            rows.append(row)
        # Moonshot outlier, early enough to land in the train split.
        rows[10]["adj_90"] = 5.0
        rows[10]["fwd_90"] = 5.0
        events = pd.DataFrame(rows)

        cfg = signal_fit.FitConfig(horizons=(90,), primary_horizon=90)
        fit = signal_fit.fit_weights(events, cfg)
        self.assertIsNotNone(fit)

        m = re.search(r"\[(-?[\d.]+), (-?[\d.]+)\]", fit.target_desc)
        self.assertIsNotNone(m, fit.target_desc)
        hi = float(m.group(2))
        # Far above the tight [0, 0.004] baseline cluster -> the outlier
        # survives the clip almost intact, unlike a 99th-percentile bound
        # (which would sit right around 0.004).
        self.assertGreater(hi, 1.0)
        self.assertIn("raw_uplift", fit.feature_stats.columns)


# ---------------------------------------------------------------------------
# (e.3) weights_payload with/without tail_fit
# ---------------------------------------------------------------------------
class TestWeightsPayloadTailFit(unittest.TestCase):
    def setUp(self):
        self.cfg = signal_fit.FitConfig()
        self.fit = signal_fit.FitResult(
            split_date=date(2024, 1, 15), train_events=100, test_events=40,
            primary_horizon=90, lambda_used=1.0,
            feature_stats=pd.DataFrame([{
                "component": "ten_percent_owner", "hand_weight": 3, "coef": 0.05,
                "raw_uplift": 0.09, "t_stat": 4.0, "n_fired_train": 50, "points": 5,
            }]),
            weights_int={**{k: 0 for k in ics.DEFAULT_WEIGHTS}, "ten_percent_owner": 5},
            oos_bucket_table=pd.DataFrame(
                columns=["scoring", "bucket", "n", "mean_adj", "median_adj", "win_rate"],
            ),
            target_desc="mean-return target",
            train_start=date(2024, 1, 1), train_end=date(2024, 1, 14),
        )
        self.tail_fit = signal_fit.TailFitResult(
            split_date=date(2024, 1, 15), train_events=560, test_events=240,
            tail_horizon=90, moonshot_thresh=0.20, base_rate=0.1589,
            tail_stats=pd.DataFrame([{
                "component": "ten_percent_owner", "hand_weight": 3,
                "lift": 1.331, "lift_h1": 1.0, "lift_h2": 1.763,
                "p_moonshot": 0.4, "n_fired_train": 120, "points": 5,
            }]),
            weights_int={**{k: 0 for k in ics.DEFAULT_WEIGHTS}, "ten_percent_owner": 5},
            oos_quintile_table=pd.DataFrame(
                columns=["bucket", "n", "p_moonshot", "mean_adj", "median_adj", "score_lo", "score_hi"],
            ),
            target_desc="tail target",
            train_start=date(2024, 1, 1), train_end=date(2024, 1, 14),
        )

    def test_without_tail_fit_payload_unchanged(self):
        payload = signal_fit.weights_payload(self.fit, self.cfg)
        self.assertNotIn("tail_weights", payload)
        self.assertNotIn("tail_fit", payload)
        self.assertEqual(payload["weights"], self.fit.weights_int)

    def test_with_tail_fit_adds_tail_weights_and_metadata(self):
        payload = signal_fit.weights_payload(self.fit, self.cfg, tail_fit=self.tail_fit)
        self.assertIn("tail_weights", payload)
        self.assertEqual(payload["tail_weights"], self.tail_fit.weights_int)
        self.assertIn("tail_fit", payload)
        self.assertEqual(payload["tail_fit"]["tail_horizon"], 90)
        self.assertEqual(payload["tail_fit"]["moonshot_thresh"], 0.20)
        self.assertAlmostEqual(payload["tail_fit"]["base_rate"], 0.1589)
        # The base "weights" key still reports the mean-return fit, unaffected.
        self.assertEqual(payload["weights"], self.fit.weights_int)


# ---------------------------------------------------------------------------
# (f) Learned strategies
# ---------------------------------------------------------------------------
class TestLearnedStrategies(unittest.TestCase):
    CAP = 100_000.0

    def _by_name(self, name: str) -> strategies.Strategy:
        for s in strategies.STRATEGIES:
            if s.name == name:
                return s
        self.fail(f"strategy {name!r} not registered")

    def test_zero_when_learned_score_is_none(self):
        for name in ("learned_gt_p00", "learned_gt_p03", "learned_score_weighted"):
            fn = self._by_name(name).target_fn
            st = _signal_state("X", date(2024, 1, 1), num_insiders=2, learned_score=None)
            self.assertEqual(fn(st, self.CAP), 0.0, name)

    def test_learned_gt_p00_gates_on_positive_score(self):
        fn = self._by_name("learned_gt_p00").target_fn
        st = _signal_state("X", date(2024, 1, 1), num_insiders=2, learned_score=1)
        self.assertEqual(fn(st, self.CAP), self.CAP)
        st["learned_score"] = 0
        self.assertEqual(fn(st, self.CAP), 0.0)

    def test_learned_gt_p03_requires_score_above_three(self):
        fn = self._by_name("learned_gt_p03").target_fn
        st = _signal_state("X", date(2024, 1, 1), num_insiders=2, learned_score=3)
        self.assertEqual(fn(st, self.CAP), 0.0)  # strictly greater than
        st["learned_score"] = 4
        self.assertEqual(fn(st, self.CAP), self.CAP)

    def test_empty_state_gated_even_with_positive_learned_score(self):
        # empty_state has num_insiders=0; a stale positive learned_score
        # (e.g. from a decayed held ticker) must not re-qualify it.
        st = state_mod.empty_state("X", date(2024, 1, 1))
        st["learned_score"] = 5
        for name in ("learned_gt_p00", "learned_gt_p03"):
            fn = self._by_name(name).target_fn
            self.assertEqual(fn(st, self.CAP), 0.0, name)

    def test_learned_score_weighted_scales_then_clamps(self):
        fn = self._by_name("learned_score_weighted").target_fn
        st = _signal_state("X", date(2024, 1, 1), num_insiders=2, learned_score=2)
        self.assertAlmostEqual(fn(st, self.CAP), 0.015 * self.CAP * 2)  # 0.03*cap, unclamped
        st["learned_score"] = 100  # would be 1.5*cap -> clamp at 0.08*cap
        self.assertAlmostEqual(fn(st, self.CAP), 0.08 * self.CAP)
        st["learned_score"] = 0
        self.assertEqual(fn(st, self.CAP), 0.0)
        st["learned_score"] = -5
        self.assertEqual(fn(st, self.CAP), 0.0)


# ---------------------------------------------------------------------------
# (f.1) learned_tpo_gated / learned_tail_concentrated — apples-to-apples
# comparisons against ten_percent_owner_gated (same 10%/10-slot policy
# shape, learned gate).
# ---------------------------------------------------------------------------
class TestLearnedTpoAndTailConcentrated(unittest.TestCase):
    CAP = 100_000.0

    def _by_name(self, name: str) -> strategies.Strategy:
        for s in strategies.STRATEGIES:
            if s.name == name:
                return s
        self.fail(f"strategy {name!r} not registered")

    # ---- learned_tpo_gated ----
    def test_learned_tpo_gated_zero_when_learned_score_none(self):
        fn = self._by_name("learned_tpo_gated").target_fn
        st = _signal_state(
            "X", date(2024, 1, 1),
            includes_ten_percent_owner=True, learned_score=None,
        )
        self.assertEqual(fn(st, self.CAP), 0.0)

    def test_learned_tpo_gated_fires_when_gate_passes(self):
        fn = self._by_name("learned_tpo_gated").target_fn
        st = _signal_state(
            "X", date(2024, 1, 1),
            includes_ten_percent_owner=True, learned_score=0,
        )
        self.assertAlmostEqual(fn(st, self.CAP), 0.10 * self.CAP)
        st["learned_score"] = 5
        self.assertAlmostEqual(fn(st, self.CAP), 0.10 * self.CAP)

    def test_learned_tpo_gated_zero_without_ten_percent_owner(self):
        # Must not fire on learned_score alone even when strongly positive.
        fn = self._by_name("learned_tpo_gated").target_fn
        st = _signal_state(
            "X", date(2024, 1, 1),
            includes_ten_percent_owner=False, learned_score=5,
        )
        self.assertEqual(fn(st, self.CAP), 0.0)

    def test_learned_tpo_gated_zero_on_empty_state(self):
        fn = self._by_name("learned_tpo_gated").target_fn
        st = state_mod.empty_state("X", date(2024, 1, 1))
        self.assertEqual(fn(st, self.CAP), 0.0)

    # ---- learned_tail_concentrated ----
    def test_learned_tail_concentrated_zero_when_tail_score_none(self):
        fn = self._by_name("learned_tail_concentrated").target_fn
        st = _signal_state("X", date(2024, 1, 1), num_insiders=2, tail_score=None)
        self.assertEqual(fn(st, self.CAP), 0.0)

    def test_learned_tail_concentrated_fires_when_gate_passes(self):
        fn = self._by_name("learned_tail_concentrated").target_fn
        st = _signal_state("X", date(2024, 1, 1), num_insiders=2, tail_score=3)
        self.assertAlmostEqual(fn(st, self.CAP), 0.10 * self.CAP)

    def test_learned_tail_concentrated_zero_below_threshold(self):
        # Below-threshold tail_score (2 < 3) must not qualify.
        fn = self._by_name("learned_tail_concentrated").target_fn
        st = _signal_state("X", date(2024, 1, 1), num_insiders=2, tail_score=2)
        self.assertEqual(fn(st, self.CAP), 0.0)

    def test_learned_tail_concentrated_zero_on_empty_state(self):
        fn = self._by_name("learned_tail_concentrated").target_fn
        st = state_mod.empty_state("X", date(2024, 1, 1))
        self.assertEqual(fn(st, self.CAP), 0.0)


# ---------------------------------------------------------------------------
# (g) report.render_html "Learned signal weights" section + oos_strategy_stats
# ---------------------------------------------------------------------------
class TestReportFitSummarySection(unittest.TestCase):
    def setUp(self):
        self._orig_exit_methods = list(strategies.EXIT_METHODS)
        strategies.EXIT_METHODS[:] = [strategies.ExitMethod("30d", 30)]
        self.addCleanup(lambda: strategies.EXIT_METHODS.__setitem__(
            slice(None), self._orig_exit_methods,
        ))
        idx = pd.date_range("2024-01-01", periods=30, freq="D")
        eq = pd.Series([100_000.0 + i * 10 for i in range(30)], index=idx)
        spy_eq = pd.Series([100_000.0 + i * 5 for i in range(30)], index=idx)
        self.rr = engine.RunResult(
            strategy="learned_gt_p00", exit_label="30d", equity_curve=eq, trades=[],
            skips=Counter(), n_rebalances=0,
            exposure_curve=pd.Series(0.0, index=idx), n_tickers_curve=pd.Series(0, index=idx),
        )
        self.spy = engine.RunResult(
            strategy="spy_buy_and_hold", exit_label="n/a", equity_curve=spy_eq, trades=[],
            skips=Counter(), n_rebalances=0,
            exposure_curve=pd.Series(1.0, index=idx), n_tickers_curve=pd.Series(1, index=idx),
        )
        self.results_by_exit = {"30d": [self.rr]}
        self.summary_df = metrics.summary_table([self.rr], self.spy)

        self.fit = signal_fit.FitResult(
            split_date=date(2024, 1, 15),
            train_events=100,
            test_events=40,
            primary_horizon=90,
            lambda_used=1.0,
            feature_stats=pd.DataFrame([{
                "component": "ten_percent_owner", "hand_weight": 3, "coef": 0.05,
                "raw_uplift": 0.09, "t_stat": 4.0, "n_fired_train": 50, "points": 5,
                "coef_10": 0.01, "coef_30": 0.02, "coef_90": 0.05, "coef_180": 0.06, "coef_365": 0.08,
            }]),
            weights_int={"ten_percent_owner": 5},
            oos_bucket_table=pd.DataFrame(
                columns=["scoring", "bucket", "n", "mean_adj", "median_adj", "win_rate"],
            ),
            target_desc="test target",
            train_start=date(2024, 1, 1),
            train_end=date(2024, 1, 14),
        )
        self.oos_df = pd.DataFrame([{
            "strategy": "learned_gt_p00", "exit": "30d",
            "oos_total_return": 0.1, "oos_cagr": 0.2, "oos_sharpe": 1.0,
        }])

        self.tail_fit = signal_fit.TailFitResult(
            split_date=date(2024, 1, 15), train_events=560, test_events=240,
            tail_horizon=90, moonshot_thresh=0.20, base_rate=0.1589,
            tail_stats=pd.DataFrame([{
                "component": "ten_percent_owner", "hand_weight": 3,
                "lift": 1.331, "lift_h1": 1.0, "lift_h2": 1.763,
                "p_moonshot": 0.4, "n_fired_train": 120, "points": 5,
            }]),
            weights_int={"ten_percent_owner": 5},
            oos_quintile_table=pd.DataFrame([{
                "bucket": "(-1.0, 5.0]", "n": 240, "p_moonshot": 0.25,
                "mean_adj": 0.05, "median_adj": 0.02, "score_lo": -1.0, "score_hi": 5.0,
            }]),
            target_desc="tail target",
            train_start=date(2024, 1, 1), train_end=date(2024, 1, 14),
        )

    def test_section_present_with_fit_summary(self):
        out = report.render_html(
            summary_df=self.summary_df, results_by_exit=self.results_by_exit,
            spy_result=self.spy, strategy_order=["learned_gt_p00"],
            config={"Generated": "test"}, offline=True,
            fit_summary={"fit": self.fit, "oos_df": self.oos_df},
        )
        # The actual rendered section (id + heading), not just a caveat that
        # mentions the section by name (CAVEATS_HTML always refers to it).
        self.assertIn('id="signal_fit"', out)
        self.assertIn("<h2>Learned signal weights</h2>", out)
        # No tail_fit given -> the tail-score subsection must not render.
        self.assertNotIn("Tail-probability score", out)
        self.assertNotIn("Tail points", out)

    def test_section_absent_without_fit_summary(self):
        out = report.render_html(
            summary_df=self.summary_df, results_by_exit=self.results_by_exit,
            spy_result=self.spy, strategy_order=["learned_gt_p00"],
            config={"Generated": "test"}, offline=True,
        )
        self.assertNotIn('id="signal_fit"', out)
        self.assertNotIn("<h2>Learned signal weights</h2>", out)

    def test_tail_subsection_present_when_tail_fit_given(self):
        out = report.render_html(
            summary_df=self.summary_df, results_by_exit=self.results_by_exit,
            spy_result=self.spy, strategy_order=["learned_gt_p00"],
            config={"Generated": "test"}, offline=True,
            fit_summary={"fit": self.fit, "oos_df": self.oos_df, "tail_fit": self.tail_fit},
        )
        self.assertIn('id="signal_fit"', out)
        self.assertIn("Tail-probability score", out)
        self.assertIn("Tail points", out)
        self.assertIn("OOS quintiles", out)

    def test_tail_subsection_absent_when_tail_fit_none(self):
        out = report.render_html(
            summary_df=self.summary_df, results_by_exit=self.results_by_exit,
            spy_result=self.spy, strategy_order=["learned_gt_p00"],
            config={"Generated": "test"}, offline=True,
            fit_summary={"fit": self.fit, "oos_df": self.oos_df, "tail_fit": None},
        )
        self.assertIn('id="signal_fit"', out)
        self.assertNotIn("Tail-probability score", out)
        self.assertNotIn("Tail points", out)


class TestOosStrategyStats(unittest.TestCase):
    def test_rebases_at_split_date_and_ignores_pre_split_history(self):
        idx = pd.date_range("2024-01-01", periods=20, freq="D")
        split_date = idx[10].date()

        # Pre-split values are deliberately wild (999 -> 100 is a huge drop)
        # to prove the OOS stats come only from the rebased tail.
        pre = [999.0] * 10
        oos = [100.0 + 2.0 * i for i in range(10)]  # 100,102,...,118
        eq = pd.Series(pre + oos, index=idx)
        rr = engine.RunResult(
            strategy="learned_gt_p00", exit_label="90d", equity_curve=eq, trades=[],
            skips=Counter(), n_rebalances=0,
            exposure_curve=pd.Series(0.0, index=idx), n_tickers_curve=pd.Series(0, index=idx),
        )

        spy_oos = [200.0] * 10  # flat OOS -> zero return, zero sharpe
        spy_eq = pd.Series(pre + spy_oos, index=idx)
        spy = engine.RunResult(
            strategy="spy_buy_and_hold", exit_label="n/a", equity_curve=spy_eq, trades=[],
            skips=Counter(), n_rebalances=0,
            exposure_curve=pd.Series(1.0, index=idx), n_tickers_curve=pd.Series(1, index=idx),
        )

        df = signal_fit.oos_strategy_stats([rr], spy, split_date)

        row = df[df["strategy"] == "learned_gt_p00"].iloc[0]
        expected_total = oos[-1] / oos[0] - 1.0
        self.assertAlmostEqual(row["oos_total_return"], expected_total, places=9)
        years = 10 / metrics.TRADING_DAYS_PER_YEAR
        expected_cagr = (oos[-1] / oos[0]) ** (1.0 / years) - 1.0
        self.assertAlmostEqual(row["oos_cagr"], expected_cagr, places=6)

        spy_row = df[df["strategy"] == "spy_buy_and_hold"].iloc[0]
        self.assertAlmostEqual(spy_row["oos_total_return"], 0.0, places=9)
        self.assertAlmostEqual(spy_row["oos_sharpe"], 0.0, places=9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
