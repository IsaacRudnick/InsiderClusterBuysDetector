"""Unit + end-to-end tests for the backtester additions.

Covers:
  - New state fields (max_pct_of_prior_stake, any_10b5_1, role_mix)
  - prices.price_signals (momentum_20d, vol_30d, dist_from_high_90d, memoization)
  - New strategy target functions (state-only and price-derived)
  - Strategy.stop_loss_pct field defaults
  - Engine end-to-end: stop-loss exit, min-price entry filter, price-signal
    injection into state visible to target_fn
  - Trailing-stop exit methods (ExitMethod.trailing_stop_pct): fires on a
    running-peak drawdown, peak is a true running max (not prior close or
    entry price alone), and the exit_days cap forces an "expiry" exit when
    the trailing stop never fires
  - Equal-weight sizing (Strategy.equal_weight): capital split evenly across
    same-day qualifiers
  - Signal-threshold strategies (thr_gt_*): strict-greater-than gating +
    num_insiders floor
  - Registry sanity: STRATEGIES/EXIT_METHODS counts and shapes
  - metrics._alpha_beta_vs_spy + summary.compute exposes n_skipped_price_floor
  - report.py table builders (smoke + stop_loss conditional rendering)
  - backtest._parse_subset CLI helper (strategy + exit-method subsets)

All synthetic — no network, no SEC fetch, no yfinance.

Run with:
    python -m unittest tests.test_backtest_additions -v
or:
    python tests/test_backtest_additions.py
"""

from __future__ import annotations

import os
import sys
import unittest
from datetime import date, timedelta
from unittest import mock

import pandas as pd

# Make the project importable when run as a script.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from backtest import engine, metrics, report, state as state_mod, strategies
from backtest.prices import PriceUniverse


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------
def _make_calendar(start: date, n_days: int) -> list[date]:
    """Strictly business-day calendar (Mon–Fri). 5 days/week."""
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
    """Start from `empty_state` and overlay overrides — ensures all expected
    keys are present, matching what the real builder produces."""
    base = state_mod.empty_state(ticker, as_of)
    base.update(overrides)
    return base


def _make_world(abc_closes, n=60, signal_from=5, signal_state_kwargs=None):
    """Build a price universe + stub state that keeps firing the signal every
    day from `signal_from` onward — that way the signal-driven trim doesn't
    unwind the lot before our test condition (stop/trailing-stop/expiry) hits.

    Module-level (rather than a TestCase method) so both the stop-loss tests
    and the new trailing-stop tests can share it without subclassing
    unittest.TestCase (which would re-run inherited test_ methods)."""
    cal = _make_calendar(date(2024, 1, 1), n)
    pu = _make_price_universe(
        {"SPY": [400.0] * n, "ABC": abc_closes}, cal,
    )
    signal_state_kwargs = signal_state_kwargs or {
        "num_insiders": 3, "conviction_score": 3,
        "includes_director": True, "includes_officer": True,
        "max_pct_of_prior_stake": 30.0, "any_10b5_1": False,
    }
    by_day = {
        cal[i]: {"ABC": _signal_state("ABC", cal[i], **signal_state_kwargs)}
        for i in range(signal_from, n)
    }
    states = _StubStates(by_day)
    return cal, pu, states


# ---------------------------------------------------------------------------
# State extensions (B3)
# ---------------------------------------------------------------------------
class TestStateFields(unittest.TestCase):
    def test_empty_state_has_new_fields(self):
        st = state_mod.empty_state("ABC", date(2024, 1, 1))
        self.assertIn("max_pct_of_prior_stake", st)
        self.assertIsNone(st["max_pct_of_prior_stake"])
        self.assertIn("any_10b5_1", st)
        self.assertFalse(st["any_10b5_1"])
        self.assertIn("role_mix", st)
        self.assertEqual(
            st["role_mix"], {"directors": 0, "officers": 0, "ten_percent": 0}
        )

    def test_build_state_populates_new_fields(self):
        # Build a 2-insider window. Insider A is a director with a 30% stake
        # increase and a 10b5-1 plan trade. Insider B is an officer.
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
                "is_ten_percent_owner": False,
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

        # _prefetch_ipo_dates hits the network — stub it.
        with mock.patch("ipo_lookup.get_first_trade_date", return_value=None):
            builder = state_mod.DailyStateBuilder(events, window_days=14)

        states_for_day = builder.state_for_day(date(2024, 1, 7))
        self.assertIn("ABC", states_for_day)
        st = states_for_day["ABC"]
        self.assertEqual(st["num_insiders"], 2)
        self.assertEqual(st["max_pct_of_prior_stake"], 30.0)
        self.assertTrue(st["any_10b5_1"])
        self.assertEqual(
            st["role_mix"], {"directors": 1, "officers": 1, "ten_percent": 0}
        )
        self.assertTrue(st["includes_director"])
        self.assertTrue(st["includes_officer"])
        self.assertFalse(st["includes_ten_percent_owner"])


# ---------------------------------------------------------------------------
# Price signals (B1)
# ---------------------------------------------------------------------------
class TestPriceSignals(unittest.TestCase):
    def setUp(self):
        # 100 trading days: linear ramp from 100 -> 200 over the first 90 days,
        # then drop to 150 on day 91+. Gives non-trivial momentum/vol/distance.
        self.cal = _make_calendar(date(2024, 1, 1), 100)
        closes = [100.0 + i for i in range(90)] + [150.0] * 10
        self.pu = _make_price_universe({"ABC": closes}, self.cal)

    def test_insufficient_history_returns_none(self):
        s = self.pu.price_signals("ABC", self.cal[5])
        self.assertIsNone(s["momentum_20d"])  # < 21 prior days
        self.assertIsNone(s["vol_30d"])       # < 11 prior days

    def test_signals_populated_after_enough_history(self):
        # Pick a day past the price drop so signals fall in interesting ranges.
        s = self.pu.price_signals("ABC", self.cal[95])
        self.assertIsNotNone(s["momentum_20d"])
        self.assertIsNotNone(s["vol_30d"])
        self.assertIsNotNone(s["dist_from_high_90d"])
        # Momentum negative because we dropped from 189 to 150 over ~20 days.
        self.assertLess(s["momentum_20d"], 0)
        # Distance from 90-day high: latest price 150 vs high ~189 → ≈ -20%.
        self.assertLess(s["dist_from_high_90d"], -0.15)
        # Vol positive.
        self.assertGreater(s["vol_30d"], 0)

    def test_memoization(self):
        d = self.cal[95]
        # Force cache miss.
        self.pu._signal_cache.clear()
        s1 = self.pu.price_signals("ABC", d)
        self.assertIn(("ABC", d), self.pu._signal_cache)
        s2 = self.pu.price_signals("ABC", d)
        self.assertIs(s1, s2)  # cached object, same identity

    def test_unknown_ticker_returns_none_dict(self):
        s = self.pu.price_signals("XYZ", self.cal[50])
        self.assertEqual(
            s, {"momentum_20d": None, "vol_30d": None, "dist_from_high_90d": None}
        )


# ---------------------------------------------------------------------------
# New strategy target functions (A)
# ---------------------------------------------------------------------------
class TestNewStrategies(unittest.TestCase):
    CAP = 100_000.0

    def _by_name(self, name: str) -> strategies.Strategy:
        for s in strategies.STRATEGIES:
            if s.name == name:
                return s
        self.fail(f"strategy {name!r} not registered")

    # ---- state-only ----
    def test_officer_director_combo(self):
        fn = self._by_name("officer_director_combo").target_fn
        # Fires
        st = _signal_state("X", date(2024, 1, 1),
                           includes_officer=True, includes_director=True,
                           num_insiders=2)
        self.assertAlmostEqual(fn(st, self.CAP), 0.06 * self.CAP)
        # Misses without officer
        st = _signal_state("X", date(2024, 1, 1),
                           includes_officer=False, includes_director=True,
                           num_insiders=2)
        self.assertEqual(fn(st, self.CAP), 0.0)

    def test_big_stake_increase(self):
        fn = self._by_name("big_stake_increase").target_fn
        st = _signal_state("X", date(2024, 1, 1),
                           max_pct_of_prior_stake=50.0, conviction_score=2)
        self.assertAlmostEqual(fn(st, self.CAP), 0.07 * self.CAP)
        # Misses with low stake increase
        st["max_pct_of_prior_stake"] = 10.0
        self.assertEqual(fn(st, self.CAP), 0.0)
        # Misses when None
        st["max_pct_of_prior_stake"] = None
        self.assertEqual(fn(st, self.CAP), 0.0)

    def test_non_10b5_1_only(self):
        fn = self._by_name("non_10b5_1_only").target_fn
        st = _signal_state("X", date(2024, 1, 1),
                           num_insiders=3, conviction_score=2, any_10b5_1=False)
        self.assertAlmostEqual(fn(st, self.CAP), 0.04 * self.CAP)
        # Misses when 10b5-1 present
        st["any_10b5_1"] = True
        self.assertEqual(fn(st, self.CAP), 0.0)

    # ---- price-derived ----
    def test_momentum_confirmed_cluster(self):
        fn = self._by_name("momentum_confirmed_cluster").target_fn
        st = _signal_state("X", date(2024, 1, 1),
                           conviction_score=2, momentum_20d=0.05)
        self.assertAlmostEqual(fn(st, self.CAP), 0.05 * self.CAP)
        # Misses when momentum negative
        st["momentum_20d"] = -0.01
        self.assertEqual(fn(st, self.CAP), 0.0)
        # Misses when momentum None (no price history yet)
        st["momentum_20d"] = None
        self.assertEqual(fn(st, self.CAP), 0.0)

    def test_oversold_cluster_reversion(self):
        fn = self._by_name("oversold_cluster_reversion").target_fn
        st = _signal_state("X", date(2024, 1, 1),
                           conviction_score=2,
                           dist_from_high_90d=-0.25, momentum_20d=-0.05)
        self.assertAlmostEqual(fn(st, self.CAP), 0.05 * self.CAP)
        # Misses if pullback shallow
        st["dist_from_high_90d"] = -0.10
        self.assertEqual(fn(st, self.CAP), 0.0)
        # Misses if momentum in freefall
        st["dist_from_high_90d"] = -0.25
        st["momentum_20d"] = -0.20
        self.assertEqual(fn(st, self.CAP), 0.0)

    def test_low_vol_conviction(self):
        fn = self._by_name("low_vol_conviction").target_fn
        st = _signal_state("X", date(2024, 1, 1),
                           conviction_score=3, vol_30d=0.30)
        self.assertAlmostEqual(fn(st, self.CAP), 0.06 * self.CAP)
        # Misses when high vol
        st["vol_30d"] = 0.60
        self.assertEqual(fn(st, self.CAP), 0.0)

    def test_vol_scaled_conviction(self):
        fn = self._by_name("vol_scaled_conviction").target_fn
        st = _signal_state("X", date(2024, 1, 1),
                           conviction_score=2, vol_30d=0.30)
        # 0.02 * cap * 2 * (0.30 / 0.30) = 0.04 * cap
        self.assertAlmostEqual(fn(st, self.CAP), 0.04 * self.CAP)
        # Vol higher → smaller size
        st["vol_30d"] = 0.60
        # 0.02 * cap * 2 * 0.5 = 0.02 * cap
        self.assertAlmostEqual(fn(st, self.CAP), 0.02 * self.CAP)
        # Vol None → 0
        st["vol_30d"] = None
        self.assertEqual(fn(st, self.CAP), 0.0)
        # Score capped
        st["vol_30d"] = 0.05
        st["conviction_score"] = 10
        # raw target would exceed 0.08*cap; should clamp
        self.assertAlmostEqual(fn(st, self.CAP), 0.08 * self.CAP)

    def test_default_stop_loss_is_none(self):
        for s in strategies.STRATEGIES:
            if "stopped" not in s.name:
                self.assertIsNone(
                    s.stop_loss_pct, f"{s.name} should not have stop_loss"
                )

    def test_stopped_variants_have_correct_pct(self):
        m = {s.name: s for s in strategies.STRATEGIES}
        self.assertEqual(m["conviction_only_stopped_15"].stop_loss_pct, 0.15)
        self.assertEqual(m["multi_insider_stopped_20"].stop_loss_pct, 0.20)
        self.assertEqual(m["score_weighted_stopped_15"].stop_loss_pct, 0.15)


# ---------------------------------------------------------------------------
# Signal-threshold strategies (thr_gt_*)
# ---------------------------------------------------------------------------
class TestThresholdStrategies(unittest.TestCase):
    CAP = 100_000.0

    def _by_name(self, name: str) -> strategies.Strategy:
        for s in strategies.STRATEGIES:
            if s.name == name:
                return s
        self.fail(f"strategy {name!r} not registered")

    def test_thr_gt_p03_rejects_score_equal_to_threshold(self):
        # Gate is strictly-greater-than: score == threshold must NOT qualify.
        fn = self._by_name("thr_gt_p03").target_fn
        st = _signal_state("X", date(2024, 1, 1), num_insiders=2, conviction_score=3)
        self.assertEqual(fn(st, self.CAP), 0.0)

    def test_thr_gt_p03_qualifies_above_threshold(self):
        fn = self._by_name("thr_gt_p03").target_fn
        st = _signal_state("X", date(2024, 1, 1), num_insiders=2, conviction_score=4)
        self.assertEqual(fn(st, self.CAP), self.CAP)

    def test_thr_gt_p03_rejects_thin_insider_count(self):
        # empty_state shape: num_insiders=0 (e.g. a decayed held ticker) must
        # not qualify even with an inflated score -- otherwise negative
        # thresholds would fire forever on empty_state and never trim.
        fn = self._by_name("thr_gt_p03").target_fn
        st = state_mod.empty_state("X", date(2024, 1, 1))
        st["conviction_score"] = 10
        self.assertEqual(fn(st, self.CAP), 0.0)


# ---------------------------------------------------------------------------
# Registry sanity: STRATEGIES / EXIT_METHODS shapes
# ---------------------------------------------------------------------------
class TestRegistrySanity(unittest.TestCase):
    def test_registry_counts(self):
        # A floor, not an exact count. The strategy registry grows as
        # strategies are added -- it was 41 when this was written and is 63
        # now -- so an exact assertion fails on every addition while catching
        # nothing the floor misses. What is worth guarding is the registry
        # silently losing entries. EXIT_METHODS stays exact: it is a small
        # fixed set, and a new exit method is a deliberate enough change to
        # be worth re-reading this test.
        self.assertGreaterEqual(len(strategies.STRATEGIES), 41)
        self.assertEqual(len(strategies.EXIT_METHODS), 7)

    def test_learned_strategies_registered_with_expected_shapes(self):
        m = {s.name: s for s in strategies.STRATEGIES}
        for name in ("learned_gt_m01", "learned_gt_p00", "learned_gt_p03",
                     "learned_score_weighted", "learned_tpo_gated",
                     "learned_tail_concentrated"):
            self.assertIn(name, m, f"{name} should be registered")

        for name in ("learned_gt_m01", "learned_gt_p00", "learned_gt_p03"):
            s = m[name]
            self.assertTrue(s.equal_weight, f"{name} should be equal_weight")
            self.assertIsNone(
                s.max_concurrent_tickers,
                f"{name} should have no ticker cap (divisor would undercount)",
            )

        weighted = m["learned_score_weighted"]
        self.assertFalse(weighted.equal_weight)
        self.assertEqual(weighted.max_concurrent_tickers, 30)

        # Apples-to-apples comparisons against ten_percent_owner_gated: same
        # 10%/10-slot concentrated policy shape, learned gate.
        for name in ("learned_tpo_gated", "learned_tail_concentrated"):
            s = m[name]
            self.assertFalse(s.equal_weight, f"{name} should not be equal_weight")
            self.assertEqual(
                s.max_concurrent_tickers, 10,
                f"{name} should mirror ten_percent_owner_gated's 10-ticker cap",
            )

    def test_threshold_strategies_are_equal_weight_uncapped(self):
        thr = [s for s in strategies.STRATEGIES if s.name.startswith("thr_gt_")]
        self.assertEqual(len(thr), 19)
        for s in thr:
            self.assertTrue(s.equal_weight, f"{s.name} should be equal_weight")
            self.assertIsNone(
                s.max_concurrent_tickers,
                f"{s.name} should have no ticker cap (divisor would undercount)",
            )

    def test_exit_method_labels_and_shapes(self):
        labels = [e.label for e in strategies.EXIT_METHODS]
        self.assertEqual(
            labels,
            ["30d", "90d", "180d", "365d", "trail10", "trail20", "trail30"],
        )
        by_label = {e.label: e for e in strategies.EXIT_METHODS}
        for lbl, days in (("30d", 30), ("90d", 90), ("180d", 180), ("365d", 365)):
            self.assertEqual(by_label[lbl].exit_days, days)
            self.assertIsNone(by_label[lbl].trailing_stop_pct)
        for lbl, pct in (("trail10", 0.10), ("trail20", 0.20), ("trail30", 0.30)):
            self.assertEqual(by_label[lbl].exit_days, 365)
            self.assertEqual(by_label[lbl].trailing_stop_pct, pct)


# ---------------------------------------------------------------------------
# Engine end-to-end with synthetic data
# ---------------------------------------------------------------------------
class TestEngineEndToEnd(unittest.TestCase):
    """Tiny simulations exercising stop-loss, min-price filter, and signal
    injection. Uses a stub state builder + synthetic price universe.

    Slippage is zeroed for these tests so a fixed-% target stays exactly
    matched to current value — otherwise the per-side slippage creates a
    persistent $5 delta that re-buys daily, spawning extra lots that
    confuse stop-loss accounting."""

    def setUp(self):
        self._orig_slip = engine.SLIPPAGE
        engine.SLIPPAGE = 0.0
        self.addCleanup(lambda: setattr(engine, "SLIPPAGE", self._orig_slip))

    def _build_world(self, abc_closes, n=60, signal_from=5,
                     signal_state_kwargs=None):
        """Thin wrapper around the module-level `_make_world` fixture (kept
        as a method so existing calls below read `self._build_world(...)`)."""
        return _make_world(
            abc_closes, n=n, signal_from=signal_from,
            signal_state_kwargs=signal_state_kwargs,
        )

    def test_stop_loss_fires(self):
        # 60 days: $100 for the first 25 days, then a hard 25% drop to $75.
        closes = [100.0] * 25 + [75.0] * 35
        cal, pu, states = self._build_world(closes, n=60)
        stopped = strategies.Strategy(
            name="conviction_test_stopped_15",
            description="t",
            max_concurrent_tickers=5,
            target_fn=lambda st, cap: 0.05 * cap if st.get("conviction_score", 0) >= 3 else 0.0,
            stop_loss_pct=0.15,
        )
        result = engine.run_strategy(
            strategy=stopped, exit_method=strategies.ExitMethod("90d", 90),
            starting_capital=100_000.0, calendar=cal, prices=pu, states=states,
        )
        stop_trades = [t for t in result.trades if t.exit_reason == "stop_loss"]
        self.assertEqual(len(stop_trades), 1)
        stop = stop_trades[0]
        # Stop fires the day after the close that breached -15% from entry.
        # Entry was day 6 open ($100 + slippage). Drop on day 25. Stop checks
        # day 25's close (which is the first $75 close, ~ -25%), executes day 26.
        self.assertLessEqual(stop.return_pct, -0.10)

    def test_no_stop_loss_when_unset(self):
        closes = [100.0] * 25 + [75.0] * 35
        cal, pu, states = self._build_world(closes, n=60)
        plain = strategies.Strategy(
            name="conviction_test_plain",
            description="t",
            max_concurrent_tickers=5,
            target_fn=lambda st, cap: 0.05 * cap if st.get("conviction_score", 0) >= 3 else 0.0,
            stop_loss_pct=None,
        )
        result = engine.run_strategy(
            strategy=plain, exit_method=strategies.ExitMethod("90d", 90),
            starting_capital=100_000.0, calendar=cal, prices=pu, states=states,
        )
        self.assertEqual(
            sum(1 for t in result.trades if t.exit_reason == "stop_loss"), 0
        )

    def test_min_price_blocks_entry(self):
        # Ticker priced at $3 — should be blocked by min_price=5.
        closes = [3.0] * 60
        cal, pu, states = self._build_world(closes, n=60)
        s = strategies.Strategy(
            name="cheap_test",
            description="t",
            max_concurrent_tickers=5,
            target_fn=lambda st, cap: 0.05 * cap if st.get("conviction_score", 0) >= 3 else 0.0,
            stop_loss_pct=None,
        )
        blocked = engine.run_strategy(
            strategy=s, exit_method=strategies.ExitMethod("90d", 90),
            starting_capital=100_000.0, calendar=cal, prices=pu, states=states,
            min_price=5.0,
        )
        self.assertEqual(len(blocked.trades), 0)
        self.assertGreaterEqual(blocked.skips.get("price_floor", 0), 1)
        # Without min_price, the entry goes through.
        allowed = engine.run_strategy(
            strategy=s, exit_method=strategies.ExitMethod("90d", 90),
            starting_capital=100_000.0, calendar=cal, prices=pu, states=states,
            min_price=0.0,
        )
        self.assertGreater(len(allowed.trades), 0)
        self.assertEqual(allowed.skips.get("price_floor", 0), 0)

    def test_price_signals_injected_into_state(self):
        # Build a stock with known momentum so a strategy reading momentum_20d
        # actually fires. Then verify the engine injected the value.
        n = 60
        closes = [100.0 + i for i in range(n)]  # steadily rising
        cal, pu, _ = self._build_world(closes, n=n, signal_from=40)
        states = _StubStates({
            cal[40]: {"ABC": _signal_state(
                "ABC", cal[40], conviction_score=2,
            )},
        })
        # Use the registered momentum_confirmed_cluster (score≥2 AND mom>0).
        s = next(x for x in strategies.STRATEGIES
                 if x.name == "momentum_confirmed_cluster")
        result = engine.run_strategy(
            strategy=s, exit_method=strategies.ExitMethod("10d", 10),
            starting_capital=100_000.0, calendar=cal, prices=pu, states=states,
        )
        # The strategy would NOT fire if momentum_20d were missing (None),
        # because target_fn returns 0 for None. A non-zero trade count proves
        # the engine injected momentum_20d > 0.
        self.assertGreater(len(result.trades), 0)


# ---------------------------------------------------------------------------
# Trailing-stop exit methods (block 1b): peak-tracking + exit_days cap
# ---------------------------------------------------------------------------
class TestTrailingStop(unittest.TestCase):
    """Exercises the new trailing-stop mechanics: the reference price is the
    prior trading day's close; `peak_price` ratchets as a running max seeded
    from the entry execution price; the stop triggers when
    prior_close / peak - 1 <= -trailing_stop_pct and executes at *today's*
    open. `exit_days` (365 for the registered trail* methods) still caps the
    hold via the ordinary expiry machinery if the trailing stop never fires."""

    def setUp(self):
        self._orig_slip = engine.SLIPPAGE
        engine.SLIPPAGE = 0.0
        self.addCleanup(lambda: setattr(engine, "SLIPPAGE", self._orig_slip))

    @staticmethod
    def _strategy(name="trailing_test"):
        return strategies.Strategy(
            name=name,
            description="t",
            max_concurrent_tickers=5,
            target_fn=lambda st, cap: 0.05 * cap if st.get("conviction_score", 0) >= 3 else 0.0,
            stop_loss_pct=None,
        )

    def test_trailing_stop_fires_on_drop_from_peak(self):
        # Flat $100 through entry, rally to a $150 peak, then fall to $130 --
        # 13.3% below the $150 peak, past the 10% trailing threshold.
        closes = [100.0] * 25 + [150.0] * 10 + [130.0] * 25  # 60 days
        cal, pu, states = _make_world(closes, n=60)
        exit_method = strategies.ExitMethod("trail10", 365, 0.10)
        result = engine.run_strategy(
            strategy=self._strategy(), exit_method=exit_method,
            starting_capital=100_000.0, calendar=cal, prices=pu, states=states,
        )
        stops = [t for t in result.trades if t.exit_reason == "trailing_stop"]
        self.assertEqual(len(stops), 1)
        stop = stops[0]
        # The close first prints $130 on cal[35]; the stop is detected the
        # next trading day (using cal[35]'s close as the no-lookahead
        # reference) and executes at that day's open.
        self.assertEqual(stop.exit_date, cal[36])
        self.assertAlmostEqual(stop.exit_price, 130.0, places=2)

    def test_trailing_stop_peak_is_running_max_not_prior_close(self):
        # P1=$120, dip to $115 (< 10% off P1 -> no trigger), then a *higher*
        # peak P2=$160, then a drop to $130: that's -18.75% off P2 (triggers)
        # but +8.3% *above* P1 -- proving the running peak tracked P2 rather
        # than the immediately preceding close or the original entry price.
        #
        # Note: the shallow dip to $115 pulls the fixed-%-of-capital target's
        # mark-to-market value below its dollar target, so the engine's
        # ordinary rebalancing machinery opens a second, smaller lot at the
        # dip (independent of trailing-stop mechanics). That lot rides the
        # same P2 peak and is correctly caught by the same breach -- so we
        # assert on *all* trailing_stop trades sharing the expected exit day
        # and price, rather than requiring exactly one trade.
        closes = (
            [100.0] * 25       # cal[0:25]   entry basis
            + [120.0] * 9       # cal[25:34]  P1
            + [115.0] * 5       # cal[34:39]  shallow dip, no trigger
            + [160.0] * 9       # cal[39:48]  P2 (new running peak)
            + [130.0] * 22      # cal[48:70]  -18.75% off P2, +8.3% off P1
        )
        cal, pu, states = _make_world(closes, n=70)
        exit_method = strategies.ExitMethod("trail10", 365, 0.10)
        result = engine.run_strategy(
            strategy=self._strategy(), exit_method=exit_method,
            starting_capital=100_000.0, calendar=cal, prices=pu, states=states,
        )
        stops = [t for t in result.trades if t.exit_reason == "trailing_stop"]
        self.assertGreaterEqual(len(stops), 1)
        for stop in stops:
            self.assertEqual(stop.exit_date, cal[49])
            self.assertAlmostEqual(stop.exit_price, 130.0, places=2)

    def test_trailing_stop_never_fires_expires_at_exit_days_cap(self):
        # Monotonically rising prices never fall 10% off the running peak, so
        # the trailing stop never fires. exit_days is the same cap mechanism
        # used for the registered 365-day trail* methods; use a short
        # exit_days (10) here instead of 365 days of stub data -- same code
        # path, just named for what it's proving.
        n = 40
        closes = [100.0 + i for i in range(n)]  # strictly increasing
        cal, pu, states = _make_world(closes, n=n)
        exit_method = strategies.ExitMethod("trail10_short_cap", 10, 0.10)
        result = engine.run_strategy(
            strategy=self._strategy(), exit_method=exit_method,
            starting_capital=100_000.0, calendar=cal, prices=pu, states=states,
        )
        stops = [t for t in result.trades if t.exit_reason == "trailing_stop"]
        self.assertEqual(len(stops), 0)
        expiries = [t for t in result.trades if t.exit_reason == "expiry"]
        self.assertGreaterEqual(len(expiries), 1)


# ---------------------------------------------------------------------------
# Equal-weight sizing (Strategy.equal_weight)
# ---------------------------------------------------------------------------
class TestEqualWeightSizing(unittest.TestCase):
    def setUp(self):
        self._orig_slip = engine.SLIPPAGE
        engine.SLIPPAGE = 0.0
        self.addCleanup(lambda: setattr(engine, "SLIPPAGE", self._orig_slip))

    def test_equal_weight_splits_capital_across_qualifiers(self):
        # 4 tickers/day; T1 and T3 qualify (gate=True), T2 and T4 don't.
        # Equal-weight sizing should split starting_capital across the 2
        # qualifiers (~cap/2 each) and leave the non-qualifiers untouched.
        n = 20
        cal = _make_calendar(date(2024, 1, 1), n)
        tickers = ["T1", "T2", "T3", "T4"]
        pu = _make_price_universe({t: [100.0] * n for t in tickers}, cal)
        qualifies = {"T1": True, "T2": False, "T3": True, "T4": False}
        by_day = {
            cal[i]: {
                t: _signal_state(t, cal[i], qualifies=qualifies[t])
                for t in tickers
            }
            for i in range(n)
        }
        states = _StubStates(by_day)
        cap = 100_000.0
        strat = strategies.Strategy(
            name="ew_test",
            description="t",
            max_concurrent_tickers=None,
            target_fn=lambda st, c: c if st.get("qualifies") else 0.0,
            equal_weight=True,
        )
        result = engine.run_strategy(
            strategy=strat, exit_method=strategies.ExitMethod("ew_exit", 90),
            starting_capital=cap, calendar=cal, prices=pu, states=states,
        )
        by_ticker: dict[str, list] = {}
        for t in result.trades:
            by_ticker.setdefault(t.ticker, []).append(t)
        self.assertIn("T1", by_ticker)
        self.assertIn("T3", by_ticker)
        self.assertNotIn("T2", by_ticker)
        self.assertNotIn("T4", by_ticker)
        for t in ("T1", "T3"):
            total_cost = sum(tr.cost_basis for tr in by_ticker[t])
            self.assertAlmostEqual(total_cost, cap / 2, delta=5.0)


# ---------------------------------------------------------------------------
# Metrics: alpha/beta + new price_floor skip column
# ---------------------------------------------------------------------------
class TestMetrics(unittest.TestCase):
    def test_alpha_beta_perfect_correlation(self):
        # y = x + 0 (alpha=0, beta=1) — random walk for SPY, strategy identical.
        idx = pd.date_range("2024-01-01", periods=200, freq="D")
        rng_returns = pd.Series(
            [0.001 * ((-1) ** i) for i in range(200)], index=idx,
        )
        spy_eq = (1 + rng_returns).cumprod() * 100
        strat_eq = spy_eq.copy()
        ab = metrics._alpha_beta_vs_spy(strat_eq, spy_eq, rf_daily=0.0)
        self.assertAlmostEqual(ab["beta"], 1.0, places=5)
        self.assertAlmostEqual(ab["alpha_ann"], 0.0, places=5)
        self.assertAlmostEqual(ab["r_squared"], 1.0, places=5)

    def test_alpha_beta_empty_spy(self):
        idx = pd.date_range("2024-01-01", periods=50, freq="D")
        eq = pd.Series(100.0, index=idx)
        ab = metrics._alpha_beta_vs_spy(eq, pd.Series(dtype=float), rf_daily=0.0)
        for key, val in ab.items():
            self.assertNotEqual(val, val, f"{key} should be NaN, got {val}")

    def test_compute_includes_price_floor_skip(self):
        # Build a minimal RunResult with price_floor in skips.
        from collections import Counter
        idx = pd.date_range("2024-01-01", periods=10, freq="D")
        eq = pd.Series([100_000.0] * 10, index=idx)
        rr = engine.RunResult(
            strategy="test", exit_label="30d",
            equity_curve=eq, trades=[],
            skips=Counter({"price_floor": 7, "cash": 2}),
            n_rebalances=0,
            exposure_curve=pd.Series(0.0, index=idx),
            n_tickers_curve=pd.Series(0, index=idx),
        )
        out = metrics.compute(rr)
        self.assertEqual(out["n_skipped_price_floor"], 7)
        self.assertEqual(out["n_skipped_cash"], 2)
        self.assertEqual(out["exit_method"], "30d")


# ---------------------------------------------------------------------------
# Report builders
# ---------------------------------------------------------------------------
class TestReportBuilders(unittest.TestCase):
    def setUp(self):
        from collections import Counter
        # Narrow EXIT_METHODS so render_html only iterates 30d. Restore on
        # teardown.
        self._orig_exit_methods = list(strategies.EXIT_METHODS)
        strategies.EXIT_METHODS[:] = [strategies.ExitMethod("30d", 30)]
        self.addCleanup(lambda: strategies.EXIT_METHODS.__setitem__(
            slice(None), self._orig_exit_methods,
        ))
        idx = pd.date_range("2024-01-01", periods=50, freq="D")
        eq = pd.Series([100_000.0 + i * 100 for i in range(50)], index=idx)
        spy_eq = pd.Series([100_000.0 + i * 50 for i in range(50)], index=idx)

        def _trade(strat, exit_reason, score, ret_pct, exit_date):
            return engine.Trade(
                strategy=strat, exit_label="30d", ticker="ABC",
                entry_date=date(2024, 1, 1), exit_date=exit_date,
                shares=100.0, entry_price=10.0,
                exit_price=10.0 * (1 + ret_pct),
                cost_basis=1000.0, proceeds=1000.0 * (1 + ret_pct),
                pnl=1000.0 * ret_pct, return_pct=ret_pct,
                score_at_entry=score, days_held=20, trading_days_held=15,
                exit_reason=exit_reason, delisted=False,
            )

        self.rr_plain = engine.RunResult(
            strategy="plain", exit_label="30d",
            equity_curve=eq,
            trades=[
                _trade("plain", "expiry", 2, 0.05, date(2024, 1, 20)),
                _trade("plain", "expiry", 3, -0.02, date(2024, 1, 25)),
            ],
            skips=Counter({"capacity": 3, "liquidity": 1, "cash": 2,
                           "no_price": 0, "price_floor": 4}),
            n_rebalances=2,
            exposure_curve=pd.Series(0.5, index=idx),
            n_tickers_curve=pd.Series(1, index=idx),
        )
        self.rr_stopped = engine.RunResult(
            strategy="stopped_15", exit_label="30d",
            equity_curve=eq,
            trades=[
                _trade("stopped_15", "stop_loss", 2, -0.16, date(2024, 1, 22)),
                _trade("stopped_15", "expiry", 3, 0.04, date(2024, 1, 28)),
            ],
            skips=Counter(),
            n_rebalances=2,
            exposure_curve=pd.Series(0.4, index=idx),
            n_tickers_curve=pd.Series(1, index=idx),
        )
        self.spy = engine.RunResult(
            strategy="spy_buy_and_hold", exit_label="n/a",
            equity_curve=spy_eq, trades=[], skips=Counter(),
            n_rebalances=0,
            exposure_curve=pd.Series(1.0, index=idx),
            n_tickers_curve=pd.Series(1, index=idx),
        )
        self.results_by_exit = {"30d": [self.rr_plain, self.rr_stopped]}
        self.summary_df = metrics.summary_table(
            [self.rr_plain, self.rr_stopped], self.spy,
        )

    def test_score_cohort_table(self):
        html = report._score_cohort_table_html(self.results_by_exit)
        self.assertIn("<table", html)
        self.assertIn("Score bucket", html)
        # Score=2 and score=3 buckets should appear at least once.
        self.assertIn(">2<", html)
        self.assertIn(">3<", html)

    def test_capacity_table(self):
        html = report._capacity_table_html(self.results_by_exit)
        self.assertIn("Cap skip", html)
        self.assertIn("PriceFloor skip", html)

    def test_alpha_beta_table(self):
        html = report._alpha_beta_table_html(self.summary_df)
        self.assertIn("Alpha", html)
        self.assertIn("Beta", html)
        # SPY row should be excluded.
        self.assertNotIn(">spy_buy_and_hold<", html)

    def test_calendar_year_table(self):
        html = report._calendar_year_table_html(self.results_by_exit)
        self.assertIn("Year", html)
        # 2024 should show up.
        self.assertIn(">2024<", html)

    def test_stop_loss_table_present_when_stops_exist(self):
        html = report._stop_loss_table_html(self.results_by_exit)
        self.assertIsNotNone(html)
        self.assertIn("stop_loss", html)

    def test_stop_loss_table_absent_when_no_stops(self):
        html = report._stop_loss_table_html({"30d": [self.rr_plain]})
        self.assertIsNone(html)

    def test_render_html_includes_diagnostics(self):
        out = report.render_html(
            summary_df=self.summary_df,
            results_by_exit=self.results_by_exit,
            spy_result=self.spy,
            strategy_order=["plain", "stopped_15"],
            config={"Generated": "test"},
            offline=True,
        )
        self.assertIn("Per-score cohort", out)
        self.assertIn("Capacity / skip rates", out)
        self.assertIn("Alpha / beta vs SPY", out)
        self.assertIn("Calendar-year contribution", out)
        # because rr_stopped has stops
        self.assertIn("Stop-loss / trailing-stop exits", out)


# ---------------------------------------------------------------------------
# CLI subset parsing
# ---------------------------------------------------------------------------
class TestCLIParseSubset(unittest.TestCase):
    def setUp(self):
        # Import via the script module so we mirror what main() uses.
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "_bt_entry", os.path.join(_ROOT, "backtest.py"),
        )
        self.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mod)

    def test_all_returns_full(self):
        out = self.mod._parse_subset("all", strategies.STRATEGIES, "strategy")
        self.assertEqual([s.name for s in out],
                         [s.name for s in strategies.STRATEGIES])

    def test_strategy_subset_order_preserved(self):
        out = self.mod._parse_subset(
            "multi_insider_positive, conviction_only",
            strategies.STRATEGIES, "strategy",
        )
        self.assertEqual(
            [s.name for s in out],
            ["multi_insider_positive", "conviction_only"],
        )

    def test_unknown_strategy_raises(self):
        with self.assertRaises(SystemExit):
            self.mod._parse_subset(
                "no_such_strategy", strategies.STRATEGIES, "strategy",
            )

    def test_exit_subset_order_preserved(self):
        out = self.mod._parse_subset(
            "90d, 30d", strategies.EXIT_METHODS, "exit",
        )
        self.assertEqual([e.label for e in out], ["90d", "30d"])

    def test_exit_all_returns_full(self):
        out = self.mod._parse_subset("all", strategies.EXIT_METHODS, "exit")
        self.assertEqual(
            [e.label for e in out],
            [e.label for e in strategies.EXIT_METHODS],
        )

    def test_unknown_exit_raises(self):
        with self.assertRaises(SystemExit):
            self.mod._parse_subset("45d", strategies.EXIT_METHODS, "exit")

    def test_trailing_exit_label_recognized(self):
        # trail10/20/30 must resolve like any other label (not just Nd forms).
        out = self.mod._parse_subset(
            "trail10, 90d", strategies.EXIT_METHODS, "exit",
        )
        self.assertEqual([e.label for e in out], ["trail10", "90d"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
