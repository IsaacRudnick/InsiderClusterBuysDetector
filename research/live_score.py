"""Backend that connects a freshly-detected cluster (the dict
insider_cluster_buys.py's _build_cluster produces) to the trained ranking
model persisted by research.model.build_production_bundle /
save_production_bundle.

This module is backend only. It has no opinion on HTML/display -- it
returns a LiveScoreResult and lets the caller decide how to render it.

--------------------------------------------------------------------------
Why a live cluster is NOT a research feature row
--------------------------------------------------------------------------
backtest.research.build_research_dataset builds each x_ feature from THREE
kinds of input the offline pipeline has and the live screener does not,
by default, have:

  1. The cluster's own transactions (window) and aggregated owner/role/
     stake data. The live screener already has this -- it is exactly what
     insider_cluster_buys._build_cluster works from.
  2. Price history for the ticker (backtest.prices.PriceUniverse):
     realized vol, drawdown, momentum, ADV, price-to-SMA200, and the
     entry-price-vs-insider-VWAP feature. insider_cluster_buys.py fetches
     NO price data today (grep the file: no PriceUniverse, no yfinance
     import). This module can compute these IF a PriceUniverse is passed
     in and has enough history for the ticker.
  3. Cross-cluster history: how many times has this owner/issuer bought
     before, and when. The offline pipeline builds this by replaying every
     historical episode day by day (see backtest/research.py's module
     docstring, "Section D leakage note"). The live screener has no
     persistent record of past clusters. This module can approximate the
     ISSUER/TICKER-level history features (x_issuer_n_prior_clusters,
     x_issuer_prior_adj63_mean, x_days_since_prior_cluster,
     x_is_first_ever_cluster) IF a historical reference frame (e.g. the
     latest research_data/research_*.parquet, which already has issuer_cik/
     ticker/event_day/adj_63 for every issuer this project has ever seen)
     is passed in. It CANNOT compute the OWNER-level history features
     (x_owner_prior_buys_wmean, x_owner_prior_adj63_mean,
     x_owner_same_month_frac): those need per-owner-CIK cluster
     PARTICIPATION history, which is not persisted anywhere outside the
     offline day-by-day replay. There is no shortcut available today.

Feature availability, of research.model.FEATURE_COLS' 59 columns:

  - 31 (Sections A/B/C: cluster shape, role composition, stake) are ALWAYS
    computable from `window`/`cluster` alone. See ALWAYS_AVAILABLE_FEATURE_COLS.
  - 12 (Section E: price context) need a PriceUniverse. See PRICE_FEATURE_COLS.
  - 4 (Section D, issuer/ticker-level) need a historical reference frame.
    See ISSUER_HISTORY_FEATURE_COLS.
  - 3 (Section D, owner-level) are NEVER computable live today, regardless
    of what is passed in. See OWNER_HISTORY_FEATURE_COLS.
  - 9 (Section F, concurrent selling) are ALSO NEVER computable live today.
    See SALE_FEATURE_COLS. Unlike the issuer/ticker history features, this
    is not fundamentally uncomputable -- backtest/sales_history.py's
    point-in-time sale index is a plain cached parquet a live caller COULD
    load and pass in -- but no plumbing to thread it through
    build_live_feature_row exists yet, so today it behaves like owner
    history: always missing, with its own explicit reason
    (_SALE_REASON) so a future caller building that plumbing has a single
    place to wire it in.

So a live cluster scored with BOTH a PriceUniverse and a historical
reference frame gets 47/59 features (80%); a live cluster scored with
NEITHER gets 31/59 (53%). build_live_feature_row's returned
FeatureAvailability always reports the real count -- this module never
substitutes a NaN and stays quiet about it. LightGBM tolerates NaN inputs,
which is exactly why silently passing NaN would be dangerous: the score
would still come out a number, indistinguishable from a real one, and the
caller would have no way to know 28 of 59 inputs were never computed.

--------------------------------------------------------------------------
Why the verdict is banded, not smoothed (see also Verdict below)
--------------------------------------------------------------------------
This is deliberate and it is not a simplification for its own sake. The
evidence:

  - The model FAILS a label-shuffle test on rank correlation across its
    full range (real IC -0.0067 vs a null of mean 0.0061, sd 0.0157,
    p=0.857 -- see research.model.run_label_shuffle_test). It has NO
    reliable BROAD ordering skill: score 40th percentile vs score 60th
    percentile carries no shown information.
  - Its TOP DECILE was once measured at a volatility-matched +4.74pp over a
    risk-matched benchmark (p=0.004, positive in 4 of 5 folds; oof_classifier
    +2.98pp, oof_regressor +3.34pp). THAT CLAIM IS RETIRED. A later audit of
    the same shipped score -- research.model.PRODUCTION_SCORE_MODEL,
    backtest.model_scores.DEFAULT_SCORE_COL, which is what this module scores
    with -- found pooled rank IC -0.053 against forward return, positive in
    only 1 of 7 years, and a top decile carrying roughly 6x the 30%-loss rate
    of the bottom decile (16.8% vs 2.5%). Both can hold at once: the band has
    a fat right tail AND a fat left tail. Read a high score as "volatile",
    not "good". The banding below is still justified -- the model has no
    broad ordering skill either way -- but it no longer marks an edge.
    The numbers above are frozen history; findings.py is the live source for
    anything the product states, and it is what the dashboard renders.
  - Portfolio confirmation, and its limits. The 10-slot result once quoted
    here (+229.6% over 72 months, Sharpe 1.107) DID NOT SURVIVE A REFIT and
    has been retired. On the current noreuse fit the same strategy returns
    +151.0% (Sharpe 0.801, alpha +5.5%, t_alpha 0.72) against SPY's +152.9%
    / 1.007 -- i.e. it no longer beats the benchmark. A 2x2 attribution
    (out/backtest_20260809_114828 and _120356) proved the ticker-reuse event
    filter was a no-op for this strategy and that the entire -78.6pp came
    from refitting on 1.07% fewer training rows.

    What DOES survive: the 5-slot variant, +211.1% with alpha +10.7% on the
    new fit vs +11.9% on the old -- a 1.2pp move where 10 slots moved 5.7pp.
    The reading is that the top ~5 picks are genuinely separated in score
    while slots 6-10 sit in a mass of near-ties that a small perturbation
    reshuffles wholesale. Everything at 15+ slots is noise around zero alpha
    and swings several points between fits.

    Consequence for THIS module: treat the top-decile verdict as a screen,
    not a ranking. Ordering WITHIN the top decile is not reliable evidence,
    which is the same reason model_sort_key refuses to render a smoothed
    confidence number.

A smooth percentile or a five-tier "confidence" scale would imply the
model can distinguish, say, the 55th from the 65th percentile. It cannot
-- the label-shuffle test says so directly. Reporting only two meaningful
states (TOP_DECILE vs NO_EDGE) is the honest summary of what was actually
measured. Do not add intermediate tiers without new evidence to support
them.

--------------------------------------------------------------------------
Per-factor reasons panel (see REASON_PANEL_FACTORS)
--------------------------------------------------------------------------
Verified Spearman ICs of each x_ feature against adj_63 (the 63-trading-day
SPY-adjusted forward return, research.model.LABEL_COL), computed on the
8,812 out-of-fold rows in research_data/oof_scores_noreuse_20260808.parquet
joined back to research_data/research_noreuse_10908rows_20260808.parquet for
the feature values (recomputed on the ticker-reuse-filtered dataset that the
shipping production bundle was fit on). Negative IC means a LOWER feature
value is associated with a better forward return. The prior column is the
same measurement on the superseded *_clean_20260807 pair, kept to show the
ICs are stable under the refit that broke the 10-slot portfolio result:

                                 clean    noreuse
    x_entry_vs_insider_vwap    -0.119     -0.122   (buy at/below what insiders
                                                     paid -- strongest feature)
    x_issuer_n_prior_clusters  +0.072     +0.072   (repeat clusters are good)
    x_vol_21_ann               -0.065     -0.066   (low realized vol is better)
    x_n_ten_pct                -0.062     -0.064   (more ten-pct owners is worse)
    x_ten_pct_value_share      -0.058     -0.060   (more $ from them is worse)
    x_is_first_ever_cluster    -0.058     -0.060   (first-ever cluster is a warning)

Every sign and magnitude survives the refit (max move 0.003), which is worth
contrasting with the portfolio result above: these per-feature relationships
are stable even though the MODEL'S RANKING built on top of them is not.

These are individually weak (|IC| < 0.12) -- none of them is a strategy on
its own -- but they are the handful with a directionally consistent,
independently-checkable relationship to the label, which is what makes them
useful as a "why" panel alongside a banded verdict that itself carries no
finer-grained information.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

import insider_cluster_buys as ics
from backtest import research as research_mod
from backtest.prices import PriceUniverse
from backtest.signal_fit import _open_or_fallback
from research import model as rm

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature categorization -- drives both the row-builder and the honesty
# report (FeatureAvailability). Every one of research.model.FEATURE_COLS'
# columns falls into exactly one of these five groups; the assertion
# block right after the constants trips at import time if backtest.research
# ever adds/removes/renames a feature and this module falls out of sync,
# rather than letting the availability report silently go stale.
# ---------------------------------------------------------------------------
PRICE_FEATURE_COLS: tuple[str, ...] = (
    "x_buy_value_to_adv", "x_log_adv20", "x_drawdown_252", "x_drawdown_63",
    "x_mom_21_skip5", "x_mom_63_skip5", "x_mom_252_skip5",
    "x_vol_21_ann", "x_vol_63_ann", "x_price_to_sma200",
    "x_tx_volume_vs_adv", "x_entry_vs_insider_vwap",
)

ISSUER_HISTORY_FEATURE_COLS: tuple[str, ...] = (
    "x_issuer_n_prior_clusters", "x_issuer_prior_adj63_mean",
    "x_days_since_prior_cluster", "x_is_first_ever_cluster",
)

# Never computable live today: these need per-owner-CIK cluster
# PARTICIPATION history (not just per-owner transaction history), which is
# only ever built by replaying the full offline episode-detection day loop
# (backtest/research.py's build_research_dataset). Nothing short of running
# that whole pipeline reproduces it, so there is no "pass an extra frame in"
# shortcut the way there is for the issuer-level features above.
OWNER_HISTORY_FEATURE_COLS: tuple[str, ...] = (
    "x_owner_prior_buys_wmean", "x_owner_prior_adj63_mean", "x_owner_same_month_frac",
)

# Section F (concurrent selling, see backtest/research.py's "Section F
# leakage note" and backtest/sales_history.py). Also never computable live
# today, but for a plumbing reason rather than a fundamental one: unlike
# OWNER_HISTORY above (which needs a full offline replay, full stop), the
# point-in-time sale index these features read is a plain cached parquet
# (backtest.sales_history.load_or_build_sales_cache) a live caller COULD
# load once and pass in cheaply. No code path here does that yet, so treat
# it as always-missing -- same shape as OWNER_HISTORY_FEATURE_COLS -- rather
# than silently mis-scoring 9 features that were never actually attempted.
SALE_FEATURE_COLS: tuple[str, ...] = (
    "x_sell_n_cluster", "x_sell_n_insiders_cluster", "x_log1p_sell_value_cluster",
    "x_sell_n_trail90", "x_sell_n_insiders_trail90", "x_log1p_sell_value_trail90",
    "x_buy_sell_balance_cluster", "x_buyer_also_sold_nearby",
    "x_officer_or_director_sold_cluster",
)

ALWAYS_AVAILABLE_FEATURE_COLS: tuple[str, ...] = tuple(
    c for c in rm.FEATURE_COLS
    if c not in PRICE_FEATURE_COLS
    and c not in ISSUER_HISTORY_FEATURE_COLS
    and c not in OWNER_HISTORY_FEATURE_COLS
    and c not in SALE_FEATURE_COLS
)

_categorized = (
    set(PRICE_FEATURE_COLS) | set(ISSUER_HISTORY_FEATURE_COLS)
    | set(OWNER_HISTORY_FEATURE_COLS) | set(SALE_FEATURE_COLS)
    | set(ALWAYS_AVAILABLE_FEATURE_COLS)
)
_n_categorized = (
    len(PRICE_FEATURE_COLS) + len(ISSUER_HISTORY_FEATURE_COLS)
    + len(OWNER_HISTORY_FEATURE_COLS) + len(SALE_FEATURE_COLS)
    + len(ALWAYS_AVAILABLE_FEATURE_COLS)
)
if _categorized != set(rm.FEATURE_COLS) or _n_categorized != len(rm.FEATURE_COLS):
    raise RuntimeError(
        "research.live_score's feature categorization has drifted from "
        "research.model.FEATURE_COLS (backtest.research._FEATURE_COLS). "
        f"categorized={sorted(_categorized)} vs actual={sorted(rm.FEATURE_COLS)}. "
        "Update PRICE_FEATURE_COLS / ISSUER_HISTORY_FEATURE_COLS / "
        "OWNER_HISTORY_FEATURE_COLS / SALE_FEATURE_COLS / "
        "ALWAYS_AVAILABLE_FEATURE_COLS to match."
    )

_PRICE_REASON = "no PriceUniverse was supplied to build_live_feature_row"
_ISSUER_HISTORY_REASON = "no historical reference frame (issuer_history) was supplied"
_OWNER_HISTORY_REASON = (
    "owner-level cluster-participation history is not computable live -- it "
    "requires replaying the full offline episode-detection pipeline"
)
_SALE_REASON = (
    "concurrent-selling features need backtest.sales_history's point-in-time "
    "sale index, which build_live_feature_row has no parameter to accept yet"
)

# Section D's "label window closed" gate (see backtest/research.py's module
# leakage note) needs an exact trading-day count. The live path has no
# guaranteed trading calendar handy (a PriceUniverse may not be supplied at
# all, and even when it is, a calendar spanning "the last several years to
# today" is a bigger ask than this one gate justifies). 63 trading days is
# approximately 63 * 365.25/252 =~ 91 calendar days -- 365.25/252 is the
# average number of calendar days per trading day over a year (252 trading
# days/year is this project's own convention, see research.model's
# annualization in _vol_ann). This is a deliberate approximation, not the
# exact calendar-index method backtest/research.py uses when it has a real
# trading calendar; it can be off by a few days around holidays/weekends,
# which does not matter for a single mean-of-prior-returns feature.
_ISSUER_LABEL_WINDOW_APPROX_DAYS = round(research_mod.HISTORY_HORIZON * 365.25 / 252)


# ---------------------------------------------------------------------------
# Feature availability reporting
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FeatureAvailability:
    """Which of the model's feature columns were actually computed for one
    live cluster, and why the rest were not. `missing` is about STRUCTURAL
    availability (no PriceUniverse / no history frame / never-computable),
    not about a feature legitimately coming back NaN because e.g. a ticker
    has under 200 days of price history -- that is a normal, trained-on
    NaN and does not count against availability here (research.model's own
    training data has plenty of it). Conflating the two would hide exactly
    the failure mode this class exists to surface.
    """
    computed: list[str]
    missing: list[str]
    missing_reasons: dict[str, str]

    @property
    def n_total(self) -> int:
        return len(self.computed) + len(self.missing)

    @property
    def frac_missing(self) -> float:
        total = self.n_total
        return (len(self.missing) / total) if total else 1.0


# ---------------------------------------------------------------------------
# Section A/B/C: cluster shape, role composition, stake.
#
# Deliberately duplicates backtest.research._build_event_row's Section
# A/B/C formulas (lines ~584-676 as of writing) rather than importing them:
# that function is wired unconditionally into build_research_dataset's
# day-loop (state dict, owner/issuer history dicts, prices, calendar) and
# has no standalone entry point for a single ad hoc cluster. The owner-
# aggregation and small numeric helpers it calls ARE imported and reused
# unchanged (backtest.research._aggregate_owners, _parse_roles, _median,
# _weighted_mean) so only the row-assembly glue is duplicated. If
# backtest/research.py's Section A/B/C formulas change, update this
# function to match -- tests/test_live_score.py's parity tests compare
# this function's output against backtest.research on a shared synthetic
# window and will fail if the two drift apart.
# ---------------------------------------------------------------------------
def _coerce_window_dates(window: list[dict]) -> list[dict]:
    """Copy `window`, coercing transaction_date/filing_date to
    datetime.date via insider_cluster_buys._coerce_date.

    The live scanner stores transaction_date as "YYYY-MM-DD" and
    filing_date as "YYYYMMDD" strings (see insider_cluster_buys.py's
    _build_cluster call site); every backtest.research helper this module
    reuses expects real date objects (that is what events_df carries).
    Coercing once, here, keeps every reused helper byte-for-byte unchanged
    instead of teaching each one to accept either shape.
    """
    out = []
    for t in window:
        t2 = dict(t)
        t2["transaction_date"] = ics._coerce_date(t.get("transaction_date"))
        t2["filing_date"] = ics._coerce_date(t.get("filing_date"))
        out.append(t2)
    return out


def _shape_role_stake_features(window: list[dict]) -> dict[str, float]:
    owners = research_mod._aggregate_owners(window)
    total_value = float(sum(o["value"] for o in owners.values()))
    n_insiders = len(owners)

    row: dict[str, float] = {}
    owner_values = sorted(o["value"] for o in owners.values())
    median_owner_value = research_mod._median(owner_values) if owner_values else 0.0
    max_owner_value = max(owner_values) if owner_values else 0.0
    row["x_n_insiders"] = float(n_insiders)
    row["x_log1p_total_value"] = float(np.log1p(max(total_value, 0.0)))
    row["x_log1p_median_value_per_insider"] = float(np.log1p(max(median_owner_value, 0.0)))
    row["x_log1p_max_insider_value"] = float(np.log1p(max(max_owner_value, 0.0)))
    row["x_n_transactions"] = float(len(window))

    tx_dates = {t.get("transaction_date") for t in window if t.get("transaction_date") is not None}
    row["x_n_distinct_tx_dates"] = float(len(tx_dates))
    row["x_window_span_days"] = float((max(tx_dates) - min(tx_dates)).days) if tx_dates else float("nan")

    valid_prices = [
        float(t["price_per_share"]) for t in window
        if t.get("price_per_share") is not None and t["price_per_share"] == t["price_per_share"]
        and t["price_per_share"] > 0
    ]
    if len(valid_prices) >= 2 and max(valid_prices) > 0:
        row["x_price_dispersion"] = (max(valid_prices) - min(valid_prices)) / max(valid_prices)
    else:
        row["x_price_dispersion"] = float("nan")
    row["x_unique_price_ratio"] = (
        len({round(p, 4) for p in valid_prices}) / len(valid_prices) if valid_prices else float("nan")
    )

    delays = [
        (t["filing_date"] - t["transaction_date"]).days for t in window
        if t.get("filing_date") is not None and t.get("transaction_date") is not None
    ]
    row["x_median_filing_delay_days"] = research_mod._median([float(d) for d in delays]) if delays else float("nan")
    row["x_max_filing_delay_days"] = float(max(delays)) if delays else float("nan")

    row["x_frac_10b5_1"] = sum(1 for t in window if t.get("is_10b5_1")) / len(window)
    row["x_frac_routine_footnote"] = sum(
        1 for t in window
        if t.get("footnote_text") and ics._ROUTINE_FOOTNOTE_RE.search(t["footnote_text"])
    ) / len(window)
    row["x_frac_fractional_shares"] = sum(
        1 for t in window
        if t.get("shares") and abs(t["shares"] - round(t["shares"])) > research_mod.FRACTIONAL_SHARE_TOL
    ) / len(window)

    n_directors = sum(1 for o in owners.values() if o["is_director"])
    n_officers = sum(1 for o in owners.values() if o["is_officer"])
    n_ten_pct = sum(1 for o in owners.values() if o["is_ten_percent_owner"])
    n_pure_directors = sum(1 for o in owners.values() if o["is_director"] and not o["is_officer"])
    row["x_n_directors"] = float(n_directors)
    row["x_n_officers"] = float(n_officers)
    row["x_n_ten_pct"] = float(n_ten_pct)
    row["x_n_pure_directors"] = float(n_pure_directors)

    def _value_share(pred) -> float:
        if total_value == 0:
            return float("nan")
        return sum(o["value"] for o in owners.values() if pred(o)) / total_value

    row["x_director_value_share"] = _value_share(lambda o: o["is_director"])
    row["x_officer_value_share"] = _value_share(lambda o: o["is_officer"])
    row["x_ten_pct_value_share"] = _value_share(lambda o: o["is_ten_percent_owner"])

    owner_role_flags = {key: research_mod._parse_roles(o["roles"]) for key, o in owners.items()}
    for role_key in research_mod._ROLE_PATTERNS:
        row[f"x_{role_key}"] = float(any(flags[role_key] for flags in owner_role_flags.values()))

    row["x_ceo_value_share"] = (
        sum(o["value"] for k, o in owners.items() if owner_role_flags[k]["has_ceo"]) / total_value
        if total_value != 0 else float("nan")
    )
    row["x_cfo_value_share"] = (
        sum(o["value"] for k, o in owners.items() if owner_role_flags[k]["has_cfo"]) / total_value
        if total_value != 0 else float("nan")
    )

    stake_pairs = [
        (float(t["pct_of_prior_stake"]), float(t.get("value") or 0.0))
        for t in window
        if t.get("pct_of_prior_stake") is not None and t["pct_of_prior_stake"] == t["pct_of_prior_stake"]
    ]
    if stake_pairs:
        vals = [p for p, _ in stake_pairs]
        row["x_stake_pct_max"] = max(vals)
        row["x_stake_pct_median"] = research_mod._median(vals)
        row["x_stake_pct_wmean"] = research_mod._weighted_mean(stake_pairs)
    else:
        row["x_stake_pct_max"] = float("nan")
        row["x_stake_pct_median"] = float("nan")
        row["x_stake_pct_wmean"] = float("nan")

    return row


# ---------------------------------------------------------------------------
# Section D (issuer/ticker-level history only -- see OWNER_HISTORY_FEATURE_COLS)
# ---------------------------------------------------------------------------
def _issuer_ticker_history_features(
    issuer_cik: str, ticker: str, as_of: date, issuer_history: pd.DataFrame,
) -> dict[str, float]:
    """x_issuer_n_prior_clusters / x_issuer_prior_adj63_mean are keyed by
    issuer_cik; x_days_since_prior_cluster / x_is_first_ever_cluster are
    keyed by ticker -- this matches backtest/research.py exactly (issuer_
    history is a dict[issuer_cik, ...], ticker_last_event is a
    dict[ticker, ...]; the two are not interchangeable when an issuer has
    changed ticker).

    `issuer_history` must have at least ticker, issuer_cik, event_day, and
    research.model.LABEL_COL columns -- the shape of research_data's own
    research_*.parquet, so the simplest way to get one is
    research.model.load_research_dataset(<latest research parquet>).
    """
    event_day_ts = pd.to_datetime(issuer_history["event_day"])
    as_of_ts = pd.Timestamp(as_of)

    row: dict[str, float] = {}

    issuer_mask = (issuer_history["issuer_cik"] == issuer_cik) & (event_day_ts < as_of_ts)
    issuer_sub = issuer_history.loc[issuer_mask]
    row["x_issuer_n_prior_clusters"] = float(len(issuer_sub))
    if len(issuer_sub):
        closed_cutoff = as_of_ts - pd.Timedelta(days=_ISSUER_LABEL_WINDOW_APPROX_DAYS)
        closed = issuer_sub.loc[event_day_ts.loc[issuer_sub.index] < closed_cutoff]
        usable = closed[rm.LABEL_COL].dropna()
        row["x_issuer_prior_adj63_mean"] = float(usable.mean()) if len(usable) else float("nan")
    else:
        row["x_issuer_prior_adj63_mean"] = float("nan")

    ticker_mask = (issuer_history["ticker"] == ticker) & (event_day_ts < as_of_ts)
    ticker_sub = issuer_history.loc[ticker_mask]
    if len(ticker_sub):
        last_event_day = event_day_ts.loc[ticker_sub.index].max()
        row["x_days_since_prior_cluster"] = float((as_of_ts - last_event_day).days)
        row["x_is_first_ever_cluster"] = 0.0
    else:
        row["x_days_since_prior_cluster"] = float("nan")
        row["x_is_first_ever_cluster"] = 1.0

    return row


# ---------------------------------------------------------------------------
# Section E: price context. Every formula here is a direct, unmodified
# reuse of backtest.research's own helpers -- they already take
# (prices, ticker, entry_day[, n]) and need no adaptation.
#
# `as_of` plays the role backtest/research.py's `entry_day` plays: every
# reused helper looks strictly BEFORE its date argument, so as_of's own
# (possibly still-forming, if scored intraday) bar is excluded and the
# most recent CLOSED bar is included -- matching research.py's own
# documented convention ("All 'as of entry_day' price lookups... are
# STRICTLY BEFORE dt"). Pass as_of=<tomorrow> if you want to include
# today's own now-closed bar (e.g. scoring after today's market close for
# tomorrow's session).
# ---------------------------------------------------------------------------
def _price_context_features(
    prices: PriceUniverse, ticker: str, window: list[dict], as_of: date,
) -> dict[str, float]:
    row: dict[str, float] = {}

    adv20 = prices.median_dollar_volume(ticker, as_of, window=20)
    total_value = float(sum(float(t.get("value") or 0.0) for t in window))
    row["x_buy_value_to_adv"] = total_value / adv20 if adv20 is not None and adv20 != 0 else float("nan")
    row["x_log_adv20"] = float(math.log(adv20)) if adv20 is not None and adv20 > 0 else float("nan")

    row["x_drawdown_252"] = research_mod._drawdown(prices, ticker, as_of, 252)
    row["x_drawdown_63"] = research_mod._drawdown(prices, ticker, as_of, 63)

    row["x_mom_21_skip5"] = research_mod._momentum_skip5(prices, ticker, as_of, 21)
    row["x_mom_63_skip5"] = research_mod._momentum_skip5(prices, ticker, as_of, 63)
    row["x_mom_252_skip5"] = research_mod._momentum_skip5(prices, ticker, as_of, 252)

    row["x_vol_21_ann"] = research_mod._vol_ann(prices, ticker, as_of, 21)
    row["x_vol_63_ann"] = research_mod._vol_ann(prices, ticker, as_of, 63)

    row["x_price_to_sma200"] = research_mod._price_to_sma200(prices, ticker, as_of)

    row["x_tx_volume_vs_adv"] = research_mod._tx_volume_vs_adv(prices, ticker, window)

    vwap = research_mod._insider_vwap(window)
    entry_open = _open_or_fallback(prices, ticker, as_of)
    if entry_open is not None and vwap == vwap and vwap != 0:
        row["x_entry_vs_insider_vwap"] = entry_open / vwap - 1.0
    else:
        row["x_entry_vs_insider_vwap"] = float("nan")

    return row


# ---------------------------------------------------------------------------
# Public row builder
# ---------------------------------------------------------------------------
def build_live_feature_row(
    cluster: dict, window: list[dict], *,
    prices: Optional[PriceUniverse] = None,
    issuer_history: Optional[pd.DataFrame] = None,
    as_of: Optional[date] = None,
) -> tuple[dict[str, float], FeatureAvailability]:
    """Build one research.model.FEATURE_COLS-shaped row from a live
    cluster, honestly reporting which columns could actually be computed.

    `cluster`/`window` are exactly insider_cluster_buys._build_cluster's
    return value and the `window` list it was built from.
    `prices`/`issuer_history` are optional; see the module docstring for
    what each unlocks. `as_of` defaults to date.today().

    Returns (row, availability). `row` always has every key in
    research.model.FEATURE_COLS -- columns this function could not compute
    are NaN, exactly like a genuinely-NaN research-time feature would be
    (LightGBM was trained on plenty of those). `availability` is what
    distinguishes "structurally unavailable" from "computed, happens to be
    NaN" -- use it, do not infer availability from row's NaN-ness alone.
    """
    if not window:
        raise ValueError("build_live_feature_row: window is empty -- a cluster needs at least one transaction")

    as_of = as_of or date.today()
    ticker = cluster.get("ticker") or window[0].get("ticker")
    issuer_cik = cluster.get("issuer_cik") or window[0].get("issuer_cik") or ""
    coerced = _coerce_window_dates(window)

    row: dict[str, float] = {}
    row.update(_shape_role_stake_features(coerced))

    if issuer_history is not None:
        row.update(_issuer_ticker_history_features(issuer_cik, ticker, as_of, issuer_history))
    if prices is not None and ticker:
        row.update(_price_context_features(prices, ticker, coerced, as_of))

    # Fill in every column this function did not attempt, so `row` always
    # has the full FEATURE_COLS shape a caller can hand straight to a model.
    for c in rm.FEATURE_COLS:
        row.setdefault(c, float("nan"))

    computed = list(ALWAYS_AVAILABLE_FEATURE_COLS)
    missing: list[str] = list(OWNER_HISTORY_FEATURE_COLS) + list(SALE_FEATURE_COLS)
    missing_reasons: dict[str, str] = {c: _OWNER_HISTORY_REASON for c in OWNER_HISTORY_FEATURE_COLS}
    missing_reasons.update({c: _SALE_REASON for c in SALE_FEATURE_COLS})

    if issuer_history is not None:
        computed += list(ISSUER_HISTORY_FEATURE_COLS)
    else:
        missing += list(ISSUER_HISTORY_FEATURE_COLS)
        missing_reasons.update({c: _ISSUER_HISTORY_REASON for c in ISSUER_HISTORY_FEATURE_COLS})

    if prices is not None and ticker:
        computed += list(PRICE_FEATURE_COLS)
    else:
        missing += list(PRICE_FEATURE_COLS)
        missing_reasons.update({c: _PRICE_REASON for c in PRICE_FEATURE_COLS})

    availability = FeatureAvailability(computed=computed, missing=missing, missing_reasons=missing_reasons)
    return row, availability


# ---------------------------------------------------------------------------
# Percentile against a FIXED reference distribution (item 3)
# ---------------------------------------------------------------------------
def score_to_percentile(raw_score: float, training_scores: np.ndarray) -> float:
    """% of `training_scores` at or below `raw_score`, i.e. this score's
    percentile rank against the FIXED reference distribution captured at
    fit time (ProductionBundle.training_scores) -- never against the
    current batch of clusters being scored. See the module docstring:
    ranking against today's batch would make the same stock's rating
    change depending on what else got scraped that day.

    Returns NaN if raw_score is not finite or training_scores is empty.
    Returns 0.0 for a score below every training score and 100.0 for a
    score at or above every training score (searchsorted side="right", so
    ties with the training distribution count in the caller's favor --
    equalling the max training score is reported as the 100th percentile,
    not just below it).
    """
    if training_scores is None or len(training_scores) == 0:
        return float("nan")
    if raw_score != raw_score or not math.isfinite(raw_score):
        return float("nan")
    rank = int(np.searchsorted(training_scores, raw_score, side="right"))
    return 100.0 * rank / len(training_scores)


# ---------------------------------------------------------------------------
# Banded verdict (item 4) -- see module docstring for the evidence this is
# based on. Exactly two meaningful states plus "unavailable". Do not add a
# third meaningful tier without new evidence backing it.
# ---------------------------------------------------------------------------
class Verdict(str, Enum):
    TOP_DECILE = "top_decile"   # percentile >= TOP_DECILE_PERCENTILE_CUTOFF: the most volatile band, NOT a measured edge
    NO_EDGE = "no_edge"         # everything else: the model does not reliably separate these
    UNAVAILABLE = "unavailable"  # too many features missing, or no usable score, to render any verdict


# Decile boundary: research.model's own decile analysis (DECILE_BINS=10,
# _decile_table) buckets scores into 10 equal-count bins via qcut, so
# "top decile" means the top 10% by score -- percentile >= 90 on the SAME
# fixed reference distribution the decile analysis itself was measured
# against (ProductionBundle.training_scores). This is not a separately
# chosen threshold; it is the same cut the (now retired) +4.74pp evidence was
# measured at, and the same cut findings.SHIPPED_MODEL_CRASH_RATE_BY_DECILE
# reports the elevated crash rate for.
TOP_DECILE_PERCENTILE_CUTOFF = 90.0

# Backstop only, not the primary way to reason about data quality --
# `FeatureAvailability` reports the real missing-feature list and count for
# a caller to inspect regardless of this threshold. This exists to catch
# the extreme case (most of the model's inputs never computed) where the
# raw score is closer to LightGBM's learned missing-value defaults than to
# a real read on the cluster. Half is a judgment call, not a statistical
# proof -- research.model's own training data routinely carries a handful
# of NaN price-context features (e.g. a ticker with under 200 days of
# history) without anyone treating those rows as unscoreable, so the bar
# for "too many to score at all" here is set well above that normal rate.
DEFAULT_MAX_MISSING_FEATURE_FRAC = 0.5


def band_verdict(
    percentile: float, availability: FeatureAvailability,
    max_missing_frac: float = DEFAULT_MAX_MISSING_FEATURE_FRAC,
) -> Verdict:
    if availability.frac_missing > max_missing_frac:
        return Verdict.UNAVAILABLE
    if percentile != percentile or not math.isfinite(percentile):
        return Verdict.UNAVAILABLE
    if percentile >= TOP_DECILE_PERCENTILE_CUTOFF:
        return Verdict.TOP_DECILE
    return Verdict.NO_EDGE


# ---------------------------------------------------------------------------
# Per-factor reasons panel (item 5)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FactorSpec:
    feature: str
    ic: float                    # Spearman IC vs adj_63, see module docstring for source
    favorable_direction: str     # "lower" or "higher"
    description: str


# See the module docstring's "Per-factor reasons panel" section for the
# source of these numbers (research_data/oof_scores_clean_20260807.parquet
# joined to research_data/research_clean_11026rows_20260807.parquet, 8,910
# out-of-fold rows). This is the module-level constant the task's own
# spec calls for -- not a magic list buried inside a function -- so a
# display layer or future analysis can import it directly.
REASON_PANEL_FACTORS: tuple[FactorSpec, ...] = (
    FactorSpec(
        "x_entry_vs_insider_vwap", -0.122, "lower",
        "Entry price vs. the insiders' own VWAP -- buying at or below what "
        "insiders paid is the strongest single-feature signal in this model.",
    ),
    FactorSpec(
        "x_issuer_n_prior_clusters", 0.072, "higher",
        "Repeat cluster buys at this issuer are a good sign.",
    ),
    FactorSpec(
        "x_vol_21_ann", -0.066, "lower",
        "Low realized volatility (21-day annualized) is better.",
    ),
    FactorSpec(
        "x_n_ten_pct", -0.064, "lower",
        "Ten-percent-owner involvement is bad -- more ten-percent owners in "
        "the cluster is worse.",
    ),
    FactorSpec(
        "x_ten_pct_value_share", -0.060, "lower",
        "Ten-percent-owner involvement is bad -- more of the cluster's "
        "dollar value coming from ten-percent owners is worse.",
    ),
    FactorSpec(
        "x_is_first_ever_cluster", -0.060, "lower",
        "A first-ever cluster at an issuer, with no track record, is a "
        "warning sign (0 = not first-ever is favorable; 1 = first-ever).",
    ),
)


@dataclass(frozen=True)
class FactorReading:
    feature: str
    value: float
    ic: float
    favorable_direction: str
    description: str
    available: bool


def factor_report(feature_row: dict[str, float], availability: FeatureAvailability) -> list[FactorReading]:
    """One FactorReading per REASON_PANEL_FACTORS entry, in the same
    (highest-|IC|-first) order the constant is defined in. `available`
    reflects the SAME structural availability build_live_feature_row
    reported, not whether `value` happens to be non-NaN."""
    computed_set = set(availability.computed)
    out = []
    for spec in REASON_PANEL_FACTORS:
        out.append(FactorReading(
            feature=spec.feature,
            value=feature_row.get(spec.feature, float("nan")),
            ic=spec.ic,
            favorable_direction=spec.favorable_direction,
            description=spec.description,
            available=spec.feature in computed_set,
        ))
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LiveScoreResult:
    raw_score: float
    percentile: float
    verdict: Verdict
    availability: FeatureAvailability
    factors: list[FactorReading]
    feature_row: dict[str, float]
    as_of: date


def score_live_cluster(
    cluster: dict, window: list[dict], bundle: "rm.ProductionBundle", *,
    prices: Optional[PriceUniverse] = None,
    issuer_history: Optional[pd.DataFrame] = None,
    as_of: Optional[date] = None,
    max_missing_frac: float = DEFAULT_MAX_MISSING_FEATURE_FRAC,
) -> LiveScoreResult:
    """Score one freshly-detected cluster end to end: build the feature
    row (honestly reporting availability), run the production model, rank
    against the model's fixed training-score distribution, and band the
    result into a Verdict. raw_score/percentile are always computed when
    mechanically possible (LightGBM tolerates NaN inputs) -- `verdict` is
    what tells the caller whether to trust them, not a None/null field.
    See the module docstring for why the verdict has exactly two
    meaningful states plus "unavailable".
    """
    as_of = as_of or date.today()
    feature_row, availability = build_live_feature_row(
        cluster, window, prices=prices, issuer_history=issuer_history, as_of=as_of,
    )

    X = pd.DataFrame([feature_row], columns=bundle.feature_cols).astype(float)
    raw_score = float(
        rm._predict_proba_positive(bundle.model, X, fallback_rate=float("nan"))[0]
    )
    percentile = score_to_percentile(raw_score, bundle.training_scores)
    verdict = band_verdict(percentile, availability, max_missing_frac=max_missing_frac)
    factors = factor_report(feature_row, availability)

    if availability.missing:
        log.info(
            "score_live_cluster: %s -- %d/%d feature(s) unavailable (%s), verdict=%s",
            cluster.get("ticker", "?"), len(availability.missing), availability.n_total,
            sorted(set(availability.missing_reasons.values())), verdict.value,
        )

    return LiveScoreResult(
        raw_score=raw_score,
        percentile=percentile,
        verdict=verdict,
        availability=availability,
        factors=factors,
        feature_row=feature_row,
        as_of=as_of,
    )
