"""Per-day rolling-window cluster state for every ticker with activity.

For a given day D, returns {ticker -> state_dict} where state_dict has:
  - num_insiders, total_value, insiders list
  - includes_{ten_percent_owner,director,officer}
  - is_recent_ipo  (recency computed against D, NOT today's date)
  - component_keys  (fired condition keys, via insider_cluster_buys._component_flags)
  - conviction_score / conviction_contributions  (always scored with
    insider_cluster_buys.DEFAULT_WEIGHTS - immune to any root signal_weights.json)
  - learned_score  (sum of learned weights over component_keys, or None
    until set_learned_weights() has been called)
  - tail_score  (sum of tail-fit weights over component_keys, or None
    until set_tail_weights() has been called)
  - model_score  (an externally-fit ranking model's score for the cluster
    event currently visible on this ticker, forward-filled across the
    event's whole rolling-window visibility; None until set_model_scores()
    has been called, or when no scored event qualifies -- see
    set_model_scores() and backtest/model_scores.py)

We mimic _build_cluster's shape so the upstream scorer works unchanged.
"""

from __future__ import annotations

import bisect
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta

import pandas as pd

import insider_cluster_buys as ics
import ipo_lookup

log = logging.getLogger(__name__)

WINDOW_DAYS = 14


def empty_state(ticker: str, as_of: date) -> dict:
    """A zero-signal state — used when a held ticker has no current activity,
    so strategies can compute target=0 and the engine trims the position."""
    return {
        "ticker": ticker,
        "as_of": as_of,
        "num_insiders": 0,
        "insiders": [],
        "total_value": 0.0,
        "includes_ten_percent_owner": False,
        "includes_director": False,
        "includes_officer": False,
        "is_recent_ipo": False,
        "first_trade_date": None,
        "conviction_score": 0,
        "conviction_contributions": [],
        "learned_score": 0,
        "tail_score": 0,
        "model_score": None,
        "component_keys": [],
        "max_pct_of_prior_stake": None,
        "any_10b5_1": False,
        "role_mix": {"directors": 0, "officers": 0, "ten_percent": 0},
    }


class DailyStateBuilder:
    """Pre-buckets events so state lookup per day is O(events visible that day).

    Results are memoized per day so 24 (strategy, horizon) engine runs share
    the work — `state_for_day(D)` does the groupby+score work once per D.
    """

    def __init__(
        self,
        events_df: pd.DataFrame,
        window_days: int = WINDOW_DAYS,
        learned_weights: dict[str, int] | None = None,
    ) -> None:
        self.window_days = window_days
        self.events = events_df.reset_index(drop=True)
        self.ipo_dates: dict[str, date | None] = {}
        self._events_by_day: dict[date, list[int]] = defaultdict(list)
        self._state_cache: dict[date, dict[str, dict]] = {}
        self.learned_weights = learned_weights
        self.tail_weights: dict[str, int] | None = None
        # model_scores mirrors tail_weights: no constructor param, defaults
        # to "not attached" so a caller who never runs the ranking-model
        # phase gets model_score=None everywhere, same as tail_score/
        # learned_score before their setters are called. Indexed into
        # parallel per-ticker (event_days, scores) lists -- both sorted
        # ascending by event_day -- so _build_state's per-day lookup is a
        # bisect over one ticker's own events, not a scan of the whole
        # mapping; state_for_day runs for every trading day across many
        # strategy/horizon runs, so a linear scan here would be a real cost.
        self.model_scores: dict[tuple[str, date], float] | None = None
        self._model_score_days: dict[str, list[date]] = {}
        self._model_score_values: dict[str, list[float]] = {}
        self._build()

    def set_learned_weights(self, weights: dict[str, int] | None) -> None:
        """Attach (or replace) the learned weight set and invalidate the
        per-day memoization cache so already-built days re-score on next
        access (state_for_day recomputes conviction/learned scores lazily)."""
        self.learned_weights = weights
        self._state_cache.clear()

    def set_tail_weights(self, weights: dict[str, int] | None) -> None:
        """Attach (or replace) the tail-fit weight set and invalidate the
        per-day memoization cache so already-built days re-score on next
        access (state_for_day recomputes conviction/learned/tail scores
        lazily)."""
        self.tail_weights = weights
        self._state_cache.clear()

    def set_model_scores(self, scores: dict[tuple[str, date], float] | None) -> None:
        """Attach (or replace) a ranking model's OOF scores and invalidate
        the per-day memoization cache so already-built days re-score on
        next access (state_for_day recomputes model_score lazily).

        `scores` maps (ticker, event_day) -> score, one entry per distinct
        cluster event (see backtest/model_scores.py for the parquet loader
        that builds this mapping). We re-index it here into parallel
        per-ticker (event_days, scores) lists, both sorted ascending by
        event_day, so _build_state can bisect instead of scanning.
        """
        self.model_scores = scores
        self._model_score_days = {}
        self._model_score_values = {}
        if scores:
            by_ticker: dict[str, list[tuple[date, float]]] = defaultdict(list)
            for (ticker, event_day), score in scores.items():
                by_ticker[ticker].append((event_day, score))
            for ticker, pairs in by_ticker.items():
                pairs.sort(key=lambda p: p[0])
                self._model_score_days[ticker] = [p[0] for p in pairs]
                self._model_score_values[ticker] = [p[1] for p in pairs]
        self._state_cache.clear()

    def _build(self) -> None:
        if self.events.empty:
            return
        self._prefetch_ipo_dates()
        # Bucket each event into the days where it's visible (filing_date <= D)
        # AND in the rolling transaction-date window (D - W + 1 <= tx_date <= D).
        tx_dates = self.events["transaction_date"].to_list()
        fil_dates = self.events["filing_date"].to_list()
        W = self.window_days
        for i, (t_t, t_f) in enumerate(zip(tx_dates, fil_dates)):
            start_rel = max(t_t, t_f)
            end_rel = t_t + timedelta(days=W - 1)
            if start_rel > end_rel:
                # Late filing — never visible within its own rolling window.
                continue
            d = start_rel
            while d <= end_rel:
                self._events_by_day[d].append(i)
                d += timedelta(days=1)
        log.info("Daily state index: %d distinct days, %d total bucket entries",
                 len(self._events_by_day),
                 sum(len(v) for v in self._events_by_day.values()))

    def _prefetch_ipo_dates(self) -> None:
        tickers = sorted(set(self.events["ticker"].dropna().astype(str).str.upper()))
        log.info("Pre-fetching IPO dates for %d tickers (cached lookups are free)",
                 len(tickers))
        with ThreadPoolExecutor(max_workers=8) as ex:
            futures = {ex.submit(ipo_lookup.get_first_trade_date, t): t for t in tickers}
            for fut in as_completed(futures):
                t = futures[fut]
                try:
                    self.ipo_dates[t] = fut.result()
                except Exception:
                    self.ipo_dates[t] = None

    def days_with_activity(self) -> set[date]:
        return set(self._events_by_day.keys())

    def state_for_day(self, D: date) -> dict[str, dict]:
        cached = self._state_cache.get(D)
        if cached is not None:
            return cached
        idxs = self._events_by_day.get(D)
        if not idxs:
            self._state_cache[D] = {}
            return self._state_cache[D]
        rows = self.events.iloc[idxs]
        states: dict[str, dict] = {}
        for ticker, sub in rows.groupby("ticker", sort=False):
            window = sub.to_dict("records")
            states[ticker] = self._build_state(window, ticker, D)
        self._state_cache[D] = states
        return states

    def _build_state(self, window: list[dict], ticker: str, as_of: date) -> dict:
        insiders: dict[str, dict] = {}
        for tx in window:
            key = tx.get("owner_cik") or tx.get("owner_name") or ""
            slot = insiders.setdefault(key, {
                "name": tx.get("owner_name", ""),
                "roles": tx.get("owner_roles", ""),
                "shares": 0.0,
                "value": 0.0,
                "is_director": False,
                "is_officer": False,
                "is_ten_percent_owner": False,
            })
            slot["shares"] += float(tx.get("shares") or 0)
            slot["value"] += float(tx.get("value") or 0)
            if tx.get("is_director"):
                slot["is_director"] = True
            if tx.get("is_officer"):
                slot["is_officer"] = True
            if tx.get("is_ten_percent_owner"):
                slot["is_ten_percent_owner"] = True

        first_trade = self.ipo_dates.get(ticker)
        is_recent_ipo = (
            first_trade is not None
            and (as_of - first_trade).days < ipo_lookup.RECENT_IPO_DAYS
        )

        pcts = [
            float(t["pct_of_prior_stake"]) for t in window
            if t.get("pct_of_prior_stake") is not None
        ]
        max_pct = max(pcts) if pcts else None

        role_mix = {
            "directors": sum(1 for v in insiders.values() if v["is_director"]),
            "officers": sum(1 for v in insiders.values() if v["is_officer"]),
            "ten_percent": sum(1 for v in insiders.values() if v["is_ten_percent_owner"]),
        }

        state = {
            "ticker": ticker,
            "as_of": as_of,
            "num_insiders": len(insiders),
            "insiders": [
                {"name": v["name"], "roles": v["roles"],
                 "shares": v["shares"], "value": v["value"]}
                for v in insiders.values()
            ],
            "total_value": float(sum(v["value"] for v in insiders.values())),
            "includes_ten_percent_owner": any(t.get("is_ten_percent_owner") for t in window),
            "includes_director": any(t.get("is_director") for t in window),
            "includes_officer": any(t.get("is_officer") for t in window),
            "is_recent_ipo": is_recent_ipo,
            "first_trade_date": first_trade,
            "max_pct_of_prior_stake": max_pct,
            "any_10b5_1": any(t.get("is_10b5_1") for t in window),
            "role_mix": role_mix,
        }
        # Fired condition keys are evaluated once and shared by both scores.
        # conviction_* is ALWAYS derived from ics.DEFAULT_WEIGHTS (never the
        # active/loaded weights) so a root signal_weights.json can never
        # poison the hand-tuned baseline the engine compares learned scores
        # against, and so in-sample-fit weights can't leak into it either.
        flags = ics._component_flags(window, state)
        state["component_keys"] = [f["key"] for f in flags]
        contributions = [
            {"key": f["key"], "delta": ics.DEFAULT_WEIGHTS.get(f["key"], 0), "text": f["text"]}
            for f in flags
            if ics.DEFAULT_WEIGHTS.get(f["key"], 0) != 0
        ]
        state["conviction_contributions"] = contributions
        state["conviction_score"] = int(sum(c["delta"] for c in contributions))
        state["learned_score"] = (
            int(sum(self.learned_weights.get(k, 0) for k in state["component_keys"]))
            if self.learned_weights is not None
            else None
        )
        state["tail_score"] = (
            int(sum(self.tail_weights.get(k, 0) for k in state["component_keys"]))
            if self.tail_weights is not None
            else None
        )
        # model_score: forward-fill the LATEST scored event_day for this
        # ticker that is <= as_of AND within window_days calendar days of
        # as_of. Forward-fill matters because research.py emits ONE score
        # row per distinct cluster event, not one per day it stays visible
        # (only ~3% of consecutive same-ticker events land within 13 days
        # of each other) -- without this, a candidate would be scored on
        # the event's first day and silently unscored on every day after,
        # making it lose capacity slots to rank_by_model_score erratically
        # even though its cluster is still visible in every other field of
        # this same state dict.
        #
        # NO LOOKAHEAD: bisect_right(days, as_of) finds the insertion point
        # just past every event_day <= as_of, so index i-1 (the candidate
        # we take) can never point at an event_day > as_of. This is the
        # single most important correctness property here -- a future
        # score leaking into a past decision would invalidate the whole
        # backtest, not just this feature.
        model_score = None
        days = self._model_score_days.get(ticker)
        if days:
            i = bisect.bisect_right(days, as_of) - 1
            if i >= 0 and (as_of - days[i]).days <= self.window_days:
                model_score = self._model_score_values[ticker][i]
        state["model_score"] = model_score
        return state
