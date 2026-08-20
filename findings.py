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

# RETIRED 2026-08-20. The single-cutoff "top_decile"/"no_edge" verdict this
# section describes has been replaced product-wide by the four-band verdict
# below (BANDS). SHIPPED_MODEL_* is kept, unedited, only so build_html.py can
# still render a "top_decile"/"no_edge" verdict if an old score bundle is
# ever loaded -- do not use these numbers for anything new.

# The ranking that DOES hold up, now SHIPPED. It was a 21-day
# quantile-regression / month-cohort ranker, not yet running when this note
# was first written; as of 2026-08-20 it IS what is running here, as
# C19_month_vol_rel_a35_live, rendered as the four bands below (BANDS).
RELIABLE_RANKER_AVAILABLE = True
RELIABLE_RANKER_NOTE = (
    "A 21-day median-targeting ranker does hold up out of sample (monotone "
    "deciles, positive in 7 of 7 years, survives a volatility-neutral audit "
    "that four higher-headline candidates failed). As of 2026-08-20 this IS "
    "what is running here -- see BANDS and the *_SCORE_* constants below."
)

# ---------------------------------------------------------------------------
# The live score's new banded verdict (research/live_score.py's Verdict
# enum). Replaces the retired top_decile/no_edge cutoff above. Source for
# every number in this section: measured out of sample, 2026-08-20, on
# 9,095 cluster episodes (2020-2026), log excess over SPY 21 trading days
# after entry, research_10861rows_20260813.parquet, a 10-seed ensemble of
# C19_month_vol_rel_a35_live.
# ---------------------------------------------------------------------------
BAND_PROVENANCE = (
    "research_10861rows_20260813.parquet, 9,095 cluster episodes, "
    "2020-2026, log excess over SPY at 21 trading days, out-of-sample"
)
BAND_MEASURED_ON = "2026-08-20"
BAND_MODEL_NAME = "C19_month_vol_rel_a35_live (10-seed ensemble)"


@dataclass(frozen=True)
class Band:
    """One row of the live screener's banded verdict. `verdict` matches
    research/live_score.py's Verdict enum value exactly."""
    verdict: str
    label: str
    pctl_lo: float
    pctl_hi: float
    median_excess: float  # median 21-day log excess vs SPY
    win_rate: float
    p_loses_30pct: float  # P(loses more than 30% within 21 days)


# Percentile is against the model's own training-score distribution, low to
# high. Note the shape: measured risk falls from elevated_risk to top_band,
# then rises again in above_band -- a higher percentile is NOT a better
# candidate past the 70-90 mark. That non-monotonicity is the whole reason
# this ships as four bands and not "sort by percentile descending", which is
# exactly what the retired score above did.
BANDS: tuple[Band, ...] = (
    Band("elevated_risk", "Elevated risk", 0, 30, -0.0268, 0.436, 0.0788),
    Band("middle", "Middle", 30, 70, -0.0083, 0.461, 0.0264),
    Band("top_band", "Top band", 70, 90, -0.0009, 0.495, 0.0121),
    Band("above_band", "Above band", 90, 100, -0.0052, 0.481, 0.0297),
)
BANDS_BY_VERDICT: dict[str, "Band"] = {b.verdict: b for b in BANDS}

# New score's quality, for the banner. Monthly-cohort IC (grouped by entry
# month so no single high-volume month dominates), against forward 21-day
# log excess over SPY.
NEW_SCORE_MONTHLY_IC = 0.0883
NEW_SCORE_IC_T = 5.08
NEW_SCORE_IC_CI = (0.0537, 0.1196)
NEW_SCORE_YEARS_POSITIVE = (7, 7)
NEW_SCORE_VOL_NEUTRAL_IC = 0.0630
LOW_VOL_RANKER_VOL_NEUTRAL_IC = 0.0145  # plain sort-by-low-volatility, same test
NEW_SCORE_DECILE_MONOTONICITY = 0.93

# The score this replaces (the retired top_decile/no_edge verdict),
# re-measured on the SAME monthly-cohort / volatility-neutral methodology as
# the new score above, for a fair comparison. (SHIPPED_MODEL_POOLED_IC above
# is an earlier, pooled-IC measurement of that same old score and is kept
# only for legacy verdict rendering -- the two are not the same number.)
PREV_SCORE_MONTHLY_IC = -0.0368
PREV_SCORE_YEARS_POSITIVE = (2, 7)
PREV_SCORE_VOL_NEUTRAL_IC = 0.0130  # below LOW_VOL_RANKER_VOL_NEUTRAL_IC
PREV_SCORE_CRASH_RATE_BOTTOM_TO_TOP_DECILE = (0.029, 0.071)

# The single most durable number this project has produced: the
# elevated-risk band's chance of a >30% loss within 21 days, by out-of-sample
# year. Never below 7.3%, never above 9.2%, in any of the 7 years measured.
ELEVATED_RISK_CRASH_RATE_RANGE = (0.073, 0.092)
ELEVATED_RISK_CRASH_RATE_BY_YEAR: dict[int, float] = {
    2020: 0.0807, 2021: 0.0791, 2022: 0.0745, 2023: 0.0731,
    2024: 0.0922, 2025: 0.0737, 2026: 0.0766,
}

# A 70-90 (top_band) book looked like an index-beater and was not: it
# measured +19.77%/yr over SPY, then failed a permutation test -- re-running
# the same band search on randomly shuffled scores produces a result this
# large or larger 43.5% of the time. Kept here so the product never states an
# index-beating claim from this ranking.
TOP_BAND_ANNUALIZED_EXCESS = 0.1977
TOP_BAND_PERMUTATION_P = 0.435

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
    """One sentence, for a console line or a banner title. Describes the
    WHOLE POPULATION of cluster buys regardless of score -- see band_headline()
    for the sentence about what the model's own ranking means."""
    return (
        "This is a skip list, not a pick list: the reliable result is which "
        "insider buys to avoid, not which to buy."
    )


def band_headline() -> str:
    """One plain, non-technical sentence about what the banded score means.
    For the model banner -- see headline() for the population-level line."""
    return (
        "This ranking reliably identifies which insider cluster buys have "
        "historically gone wrong most often. It does NOT identify which "
        "ones go up."
    )


def band_rows() -> list[tuple[str, str, str, str, str]]:
    """(band, percentile, median excess, win rate, P(loses >30%)) as display
    strings, in the product's own sort order: top_band, above_band, middle,
    elevated_risk (see build_html._VERDICT_SORT_RANK)."""
    order = ("top_band", "above_band", "middle", "elevated_risk")
    return [
        (
            BANDS_BY_VERDICT[v].label,
            f"{BANDS_BY_VERDICT[v].pctl_lo:.0f}-{BANDS_BY_VERDICT[v].pctl_hi:.0f}",
            f"{BANDS_BY_VERDICT[v].median_excess * 100:+.2f}%",
            f"{BANDS_BY_VERDICT[v].win_rate * 100:.1f}%",
            f"{BANDS_BY_VERDICT[v].p_loses_30pct * 100:.2f}%",
        )
        for v in order
    ]


def new_score_quality_note() -> str:
    """Score-quality sentence for the banner -- monthly-cohort IC, years
    positive, volatility-neutral IC vs a low-vol ranker, decile monotonicity."""
    lo, hi = NEW_SCORE_IC_CI
    yrs_pos, yrs_tot = NEW_SCORE_YEARS_POSITIVE
    return (
        f"Measured out of sample: monthly-cohort rank correlation with forward "
        f"return {NEW_SCORE_MONTHLY_IC:+.4f} (t={NEW_SCORE_IC_T:.2f}, 95% CI "
        f"[{lo:+.4f}, {hi:+.4f}]), positive in {yrs_pos} of {yrs_tot} "
        f"out-of-sample years, volatility-neutral IC {NEW_SCORE_VOL_NEUTRAL_IC:+.4f} "
        f"against {LOW_VOL_RANKER_VOL_NEUTRAL_IC:+.4f} for a plain "
        f"sort-by-low-volatility ranker, decile monotonicity "
        f"{NEW_SCORE_DECILE_MONOTONICITY:.2f}."
    )


def prev_score_contrast_note() -> str:
    """Contrast sentence: how the retired score scored on this same test."""
    yrs_pos, yrs_tot = PREV_SCORE_YEARS_POSITIVE
    lo, hi = PREV_SCORE_CRASH_RATE_BOTTOM_TO_TOP_DECILE
    return (
        f"The score it replaces measured monthly IC {PREV_SCORE_MONTHLY_IC:+.4f}, "
        f"positive in {yrs_pos} of {yrs_tot} years, volatility-neutral IC "
        f"{PREV_SCORE_VOL_NEUTRAL_IC:+.4f} (below the low-vol ranker), and its "
        f"crash rate rose {lo * 100:.1f}% to {hi * 100:.1f}% from bottom decile "
        "to top."
    )


def elevated_risk_crash_note() -> str:
    """The durable number: elevated-risk band crash rate, every year."""
    lo, hi = ELEVATED_RISK_CRASH_RATE_RANGE
    n_years = len(ELEVATED_RISK_CRASH_RATE_BY_YEAR)
    return (
        f"The elevated-risk band's chance of losing more than 30% within 21 "
        f"days stayed between {lo * 100:.1f}% and {hi * 100:.1f}% in every one "
        f"of the {n_years} out-of-sample years measured -- the most durable "
        "number this project has produced."
    )


def top_band_permutation_note() -> str:
    """Why no index-beating claim is made, even though the top band looks
    good in the table above."""
    return (
        f"A top-band-only book measured {TOP_BAND_ANNUALIZED_EXCESS * 100:+.2f}%/yr "
        f"over SPY, then failed a permutation test (p={TOP_BAND_PERMUTATION_P:.3f} "
        "-- re-running the same band search on shuffled scores produces a result "
        "this large nearly as often as not). No index-beating claim is made."
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


# ---------------------------------------------------------------------------
# What a HOLDER of each band would have experienced, not just a
# ranking-quality metric. Source for every number in this section:
# research_10861rows_20260813.parquet, 10-seed ensemble of
# C19_month_vol_rel_a35_live, 78 non-overlapping 21-trading-day periods,
# equal weight, 20bps round trip. Measured 2026-08-20.
# ---------------------------------------------------------------------------
BOOK_PROVENANCE = (
    "research_10861rows_20260813.parquet, 10-seed ensemble of "
    "C19_month_vol_rel_a35_live, 78 non-overlapping 21-trading-day periods, "
    "equal weight, 20bps round trip"
)
BOOK_MEASURED_ON = "2026-08-20"


@dataclass(frozen=True)
class BookResult:
    """What a holder of one book (a band, the whole population, or a
    benchmark) would have experienced over the measured window -- a
    portfolio-outcome metric, not a ranking-quality metric like IC."""
    name: str
    ann_return: float
    ann_vol: float
    sharpe: float
    sortino: float
    max_drawdown: float
    win_rate: float


BOOK_RESULTS: tuple[BookResult, ...] = (
    BookResult("Top band (70-90)", 0.345, 0.230, 1.303, 1.400, -0.307, 0.73),
    BookResult("All clusters", 0.238, 0.235, 0.917, 1.124, -0.322, 0.58),
    BookResult("Bottom 30% (elevated risk)", 0.149, 0.340, 0.412, 0.668, -0.576, 0.51),
    BookResult("SPY", 0.181, 0.141, 1.192, 1.125, -0.192, 0.73),
    BookResult("IWM", 0.156, 0.202, 0.723, 0.859, -0.273, 0.60),
)

# The top-band book, over the full 6.5-year window (2020-08 .. 2026-08).
TOP_BAND_FINAL_MULTIPLE = 5.77
SPY_FINAL_MULTIPLE = 2.76
TOP_BAND_YEARS_BEAT_SPY = (4, 7)

# Permutation test on the Sharpe statistic: the null re-runs the ENTIRE
# 132-recipe search on shuffled scores. This is a different test, on a
# different statistic, than TOP_BAND_PERMUTATION_P above (which tested raw
# annualized excess RETURN and failed, p=0.435). This one, on Sharpe, passes.
TOP_BAND_SHARPE_PERMUTATION_P = 0.005
TOP_BAND_SHARPE_NULL_MEDIAN = 0.987
TOP_BAND_SHARPE_NULL_P95 = 1.236

# Why the two tests disagree even though they are run on the same book: a
# fat right tail inflates the mean return AND the volatility of a
# shuffled-score book together, so it can inflate a raw excess-return
# statistic by chance -- but it cannot inflate a RATIO of the two the same
# way, because the tail's contribution to the numerator is normalized by its
# own contribution to the denominator. That is why Sharpe survives shuffling
# where raw excess return did not.

# Harvestability caveats. Every one of these MUST travel with the headline
# Sharpe/return numbers above wherever they are shown -- this result is a
# risk screen, not a harvestable portfolio, and these are why.
COST_SHARPE_BY_BPS: dict[int, float] = {20: 1.303, 50: 1.147, 100: 0.886}
# Position capped at 10% of a name's trailing 20-day dollar volume. SPY's
# Sharpe (1.192) is above every capital size tested here.
LIQUIDITY_SHARPE_BY_CAPITAL: dict[str, float] = {
    "$100k": 1.107, "$1M": 0.982, "$25M": 0.750,
}
PRICE_FLOOR_EXCESS_BEFORE = 0.141  # annual excess over SPY, no price floor
PRICE_FLOOR_EXCESS_AFTER = 0.032   # annual excess over SPY, $5 minimum entry price
# Model refit inside a realistic tradeable universe ($5+ price floor, $1M
# book, 50bps round trip) rather than the full universe measured above.
REFIT_TRADEABLE_UNIVERSE_SHARPE = 0.650
REFIT_TRADEABLE_UNIVERSE_SPY_SHARPE = 1.237


def book_rows() -> list[tuple[str, str, str, str, str, str, str]]:
    """(book, ann return, ann vol, Sharpe, Sortino, max drawdown, win rate)
    as display strings, in BOOK_RESULTS order."""
    return [
        (
            b.name,
            f"{b.ann_return * 100:+.1f}%",
            f"{b.ann_vol * 100:.1f}%",
            f"{b.sharpe:.3f}",
            f"{b.sortino:.3f}",
            f"{b.max_drawdown * 100:.1f}%",
            f"{b.win_rate * 100:.0f}%",
        )
        for b in BOOK_RESULTS
    ]


def risk_adjusted_note() -> str:
    """The headline sentence for this section: the top band DID beat SPY on
    a risk-adjusted basis this time, and that result passes a permutation
    test where the earlier raw-return claim failed -- with the mechanism."""
    top = next(b for b in BOOK_RESULTS if b.name == "Top band (70-90)")
    spy = next(b for b in BOOK_RESULTS if b.name == "SPY")
    yrs, yrs_tot = TOP_BAND_YEARS_BEAT_SPY
    return (
        f"Over this window, a top-band-only book returned {top.ann_return * 100:+.1f}%/yr "
        f"(Sharpe {top.sharpe:.2f}, Sortino {top.sortino:.2f}) against SPY's "
        f"{spy.ann_return * 100:+.1f}%/yr (Sharpe {spy.sharpe:.2f}) -- a "
        f"{TOP_BAND_FINAL_MULTIPLE:.2f}x final multiple against SPY's "
        f"{SPY_FINAL_MULTIPLE:.2f}x, beating SPY in {yrs} of {yrs_tot} years. "
        f"This Sharpe result passes a permutation test (p={TOP_BAND_SHARPE_PERMUTATION_P:.3f}; "
        f"null median {TOP_BAND_SHARPE_NULL_MEDIAN:.3f}, 95th percentile "
        f"{TOP_BAND_SHARPE_NULL_P95:.3f}) -- unlike the raw annualized-excess-return "
        f"claim above, which failed the same style of test "
        f"(TOP_BAND_PERMUTATION_P={TOP_BAND_PERMUTATION_P:.3f}). The two disagree because a "
        "fat right tail inflates mean return and volatility together, so it can "
        "inflate a raw return statistic by chance but not a ratio of the two."
    )


def harvestability_note() -> str:
    """Why this is a risk screen, not a portfolio: cost, liquidity, price
    floor, and refit-instability caveats, all in one place."""
    bps_20, bps_50, bps_100 = (
        COST_SHARPE_BY_BPS[20], COST_SHARPE_BY_BPS[50], COST_SHARPE_BY_BPS[100]
    )
    liq_100k, liq_1m, liq_25m = (
        LIQUIDITY_SHARPE_BY_CAPITAL["$100k"], LIQUIDITY_SHARPE_BY_CAPITAL["$1M"],
        LIQUIDITY_SHARPE_BY_CAPITAL["$25M"],
    )
    return (
        f"Cost: Sharpe falls below SPY's somewhere under 50bps round trip "
        f"({bps_20:.3f} at 20bps, {bps_50:.3f} at 50bps, {bps_100:.3f} at 100bps). "
        f"Liquidity: capping a position at 10% of a name's 20-day dollar volume gives "
        f"Sharpe {liq_100k:.3f} at $100k of capital, {liq_1m:.3f} at $1M, "
        f"{liq_25m:.3f} at $25M -- below SPY at every size tested. "
        f"Price floor: with a $5 minimum entry price, excess over SPY falls from "
        f"{PRICE_FLOOR_EXCESS_BEFORE * 100:+.1f}%/yr to {PRICE_FLOOR_EXCESS_AFTER * 100:+.1f}%/yr. "
        f"Refit: rebuilding the model inside a tradeable universe ($5+, $1M book, "
        f"50bps) gives Sharpe {REFIT_TRADEABLE_UNIVERSE_SHARPE:.3f} against SPY's "
        f"{REFIT_TRADEABLE_UNIVERSE_SPY_SHARPE:.3f}. Max drawdown is worse than SPY's too "
        "(-30.7% vs -19.2%). Treat this as a risk screen, not a portfolio: the top "
        "band beat SPY only in a universe that includes sub-$5, thinly-traded names "
        "at costs and account sizes that make it unrealistic to actually harvest."
    )


def top_band_holding_summary() -> str:
    """One-line summary for the model banner: top band's Sharpe and
    annualized return, with the harvestability caveat in the SAME sentence
    -- never the return alone. See risk_adjusted_note()/harvestability_note()
    for the full picture."""
    top = next(b for b in BOOK_RESULTS if b.name == "Top band (70-90)")
    spy = next(b for b in BOOK_RESULTS if b.name == "SPY")
    return (
        f"A top-band-only book measured {top.ann_return * 100:+.1f}%/yr (Sharpe "
        f"{top.sharpe:.2f}) against SPY's {spy.ann_return * 100:+.1f}%/yr (Sharpe "
        f"{spy.sharpe:.2f}) and passes a permutation test "
        f"(p={TOP_BAND_SHARPE_PERMUTATION_P:.3f}) -- but that edge requires "
        "sub-$5, thinly-traded names, and disappears under realistic cost, "
        "liquidity, and price-floor limits, so it is a risk screen, not a "
        "harvestable portfolio (see the panel below)."
    )


def key_points() -> list[str]:
    """The short list a user should read before acting on this dashboard."""
    best = HORIZON_EXPECTATIONS[0]
    crash_lo, crash_hi = ELEVATED_RISK_CRASH_RATE_RANGE
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

        "The score shown here is banded, not a straight ranking -- a higher "
        "percentile is not a better candidate. The bottom 30% ('elevated "
        f"risk') reliably has the highest chance of a large loss, between "
        f"{crash_lo * 100:.1f}% and {crash_hi * 100:.1f}% in every out-of-sample "
        "year measured. The 70th-90th percentile ('top band') is the "
        "best-measured band, but a book built only from it failed a "
        "permutation test, so treat this as a guide to what to AVOID, not "
        "a stock-picking signal.",

        f"About {SURVIVORSHIP['frac_tickers_unpriceable'] * 100:.0f}% of the "
        "companies in the historical data have no price history at all, "
        "because they stopped trading and the data provider deleted them. "
        "Every figure above is therefore optimistic by an unknown margin.",

        "No configuration tested produced a portfolio that beat an index fund "
        "in a way that survived changing the model's random seed. Treat any "
        "such claim from this pipeline with suspicion unless it reports a "
        "seed sweep.",
    ]
