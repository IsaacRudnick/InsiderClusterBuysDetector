"""Measured facts about this strategy, in one place, with provenance.

WHY THIS MODULE EXISTS. The live screener's dashboard used to state one
headline ("only the top decile has a measured edge: +4.74pp, p=0.004") that
was true of the run it was measured on and has since been contradicted by
later work. A claim that lives as a string literal inside a rendering
function is a claim nobody re-checks. Everything a user is told about what
the ranking means now lives here, next to where it came from, so that
correcting the research corrects the product.

EVERY number below is measured, not assumed, and carries its source. If you
change a number, change its provenance line in the same edit.

The headline, stated once: **this pipeline is a skip list, not a pick
list.** The reliable, repeatable result is identifying insider cluster buys
to AVOID. No configuration tested produced a portfolio that beats an index
fund in a way that survives a random-seed sweep.
"""

from __future__ import annotations

from dataclasses import dataclass

# Source for every number in this module unless stated otherwise:
# RESEARCH_NOTES.md, section "Does a ranking exist? Yes. Does it beat an ETF?
# No. (2026-08-13)". Underlying data: research_groupE_10905rows_20260809
# .parquet, 10,905 cluster episodes, event days 2018-08 .. 2026-08,
# expanding-window out-of-sample.
PROVENANCE = (
    "research_groupE_10905rows_20260809.parquet, 10,905 cluster episodes, "
    "2018-08..2026-08, expanding-window out-of-sample"
)
MEASURED_ON = "2026-08-13"


@dataclass(frozen=True)
class HorizonExpectation:
    """What the WHOLE population of insider cluster buys did, by hold length.

    `vs_spy` / `vs_iwm` / `vs_iwc` are annualized log-excess returns over
    SPY, IWM (small cap) and IWC (micro cap), computed open-to-open on the
    same entry days the research labels use. Both yardsticks are kept
    deliberately: IWM/IWC answers "does the insider signal add value over
    the universe it trades in", SPY answers "would you have been better off
    in an index fund". Reporting only the flattering one is benchmark
    shopping.
    """
    trading_days: int
    vs_spy: float
    vs_iwm: float
    vs_iwc: float
    win_rate: float


# The decay curve. This is the single most useful thing the screener can
# tell a user: a fresh cluster is worth about a small-cap index fund if you
# hold it ~2 weeks, and progressively less the longer you hold. There is no
# post-filing drift to capture -- the curve only slopes down.
HORIZON_EXPECTATIONS: tuple[HorizonExpectation, ...] = (
    HorizonExpectation(10, -0.0789, -0.0069, -0.0145, 0.482),
    HorizonExpectation(21, -0.1282, -0.0652, -0.0694, 0.463),
    HorizonExpectation(63, -0.1417, -0.1073, -0.1145, 0.439),
    HorizonExpectation(126, -0.1654, float("nan"), float("nan"), 0.410),
    HorizonExpectation(252, -0.1702, float("nan"), float("nan"), 0.385),
)

# Benchmark CAGRs over the measured window (2018-08-13..2026-08-12), from
# price_cache. Kept so the dashboard can explain WHY the SPY column looks so
# much worse than the IWM column: small caps trailed SPY by ~6pp/yr for
# eight years, and roughly half the apparent "insiders pick badly" result is
# that size factor, not insider skill.
BENCHMARK_CAGR: dict[str, float] = {
    "SPY (large cap)": 0.1516,
    "RSP (equal-weight S&P)": 0.1182,
    "MDY (mid cap)": 0.1020,
    "IWC (micro cap)": 0.0942,
    "IWM (small cap)": 0.0910,
    "XBI (biotech)": 0.0688,
}

# Refinements that DO NOT work. Each is a median 21-day log-excess vs SPY by
# quartile of the named quantity, worst-quartile-first. Every one is flat.
# Shown in the product because these are exactly the intuitions a user will
# reach for, and "we checked, it doesn't help" is more useful than silence.
FLAT_REFINEMENTS: tuple[tuple[str, tuple[float, ...]], ...] = (
    ("Reacting faster to a fresh filing", (-0.0066, -0.0069, -0.0101, -0.0094)),
    ("More insiders in the cluster", (-0.0075, -0.0122, -0.0054, -0.0079)),
    ("Larger total dollars bought", (-0.0109, -0.0081, -0.0063, -0.0074)),
    ("A higher CEO share of the buying", (-0.0105, -0.0118, -0.0031, -0.0075)),
    ("More ten-percent owners", (-0.0092, -0.0069, -0.0094, -0.0074)),
)

# The shipped production model is research.model.PRODUCTION_SCORE_MODEL,
# "tail_classifier" -- a P(adj_63 > 0.20) estimate. Measured on 8,810
# out-of-fold rows, its score is POSITIVELY associated with blow-up risk:
# P(63-day return < -30%) by score decile, lowest score first.
#
# This is the reason the old "top decile has a measured edge" banner had to
# go. That claim came from a volatility-matched top-decile test on an
# earlier fit; a later audit found the same score has pooled rank IC -0.053
# against forward return, positive in only 1 of 7 years, and that its top
# decile carries roughly 6x the crash rate of its bottom decile. Both
# measurements can be true at once -- the top decile is a lottery ticket
# with a fat right tail AND a fat left tail -- but a product that reports
# only the first half is misleading.
SHIPPED_MODEL_CRASH_RATE_BY_DECILE: tuple[float, ...] = (
    0.025, 0.027, 0.027, 0.037, 0.101, 0.125, 0.142, 0.168, 0.168, 0.159,
)
SHIPPED_MODEL_POOLED_IC = -0.053
SHIPPED_MODEL_YEARS_POSITIVE = (1, 7)

# The ranking that DOES hold up, for reference. Not yet shipped: it is a
# 21-day quantile-regression ranker and research/model.py still fits against
# adj_63. Recorded here so the dashboard can describe what "reliable" would
# look like and not overclaim for what is currently running.
RELIABLE_RANKER_AVAILABLE = False
RELIABLE_RANKER_NOTE = (
    "A 21-day median-targeting ranker does hold up out of sample (monotone "
    "deciles, positive in 7 of 7 years, survives a volatility-neutral audit "
    "that four higher-headline candidates failed). It is NOT what is running "
    "here -- see RESEARCH_NOTES.md."
)

# Survivorship, re-measured on the current events file. Stated in the
# product because it caps how much any absolute number here can be trusted.
SURVIVORSHIP = {
    "frac_tickers_unpriceable": 0.332,
    "frac_buy_rows_unpriceable": 0.249,
    "frac_buy_dollars_unpriceable": 0.271,
}

# Round-trip cost assumed in every net figure quoted above (backtest.py's
# own default slippage, 10bps per side).
ROUND_TRIP_COST = 0.0020


def headline() -> str:
    """One sentence, for a console line or a banner title."""
    return (
        "This is a skip list, not a pick list: the reliable result is which "
        "insider buys to avoid, not which to buy."
    )


def expectation_rows() -> list[tuple[str, str, str, str, str]]:
    """(hold, vs SPY, vs small cap, vs micro cap, win rate) as display strings."""
    def pct(v: float) -> str:
        return "n/a" if v != v else f"{v * 100:+.1f}%/yr"
    return [
        (f"{h.trading_days} trading days", pct(h.vs_spy), pct(h.vs_iwm),
         pct(h.vs_iwc), f"{h.win_rate * 100:.1f}%")
        for h in HORIZON_EXPECTATIONS
    ]


def key_points() -> list[str]:
    """The short list a user should read before acting on this dashboard."""
    best = HORIZON_EXPECTATIONS[0]
    lo, hi = SHIPPED_MODEL_CRASH_RATE_BY_DECILE[0], SHIPPED_MODEL_CRASH_RATE_BY_DECILE[-1]
    return [
        "The cluster-buy event itself is not a buy signal. Held about two "
        f"weeks, the average flagged cluster returns {best.vs_iwm * 100:+.1f}%/yr "
        "against a small-cap index fund -- statistically indistinguishable from "
        "just owning the index. Held longer, it does worse, not better.",

        "There is no speed advantage. Clusters filed fastest performed the "
        "same as clusters filed slowest, so there is nothing to be gained by "
        "reacting to a filing sooner.",

        "None of the obvious quality filters work. More insiders, bigger "
        "dollar amounts, CEO participation and ten-percent-owner involvement "
        "were all tested and all came back flat.",

        f"The score shown here rises with crash risk, not against it. In "
        f"backtesting, the highest-scoring decile had a {hi * 100:.0f}% chance "
        f"of losing more than 30% in 63 days, against {lo * 100:.0f}% for the "
        "lowest-scoring decile. Treat a high score as 'volatile', not 'good'.",

        f"About {SURVIVORSHIP['frac_tickers_unpriceable'] * 100:.0f}% of the "
        "companies in the historical data have no price history at all, "
        "because they stopped trading and the data provider deleted them. "
        "Every figure above is therefore optimistic by an unknown margin.",

        "No configuration tested produced a portfolio that beat an index fund "
        "in a way that survived changing the model's random seed. Treat any "
        "such claim from this pipeline with suspicion unless it reports a "
        "seed sweep.",
    ]
