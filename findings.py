"""Measured facts about this strategy, in one place, with provenance.

WHY THIS MODULE EXISTS. The dashboard once stated one headline: "only the
top decile has a measured edge: +4.74pp, p=0.004". That claim was true of
the run that produced it. Later work contradicted it. A claim that lives as
a string literal inside a rendering function is a claim nobody re-checks.
Everything a user reads about the ranking now lives here, next to its
source. Correct the research here and the product corrects with it.

Every number below is measured, not assumed, and carries its source. If you
change a number, change its provenance line in the same edit.

THE HEADLINE, STATED ONCE. This pipeline is a skip list, not a pick list.
It tells you which insider cluster buys are worth manual research and which
are not. It does not tell you which ones go up. No configuration tested
produced a portfolio that beats an index fund out of sample.

READ THESE TWO SECTIONS BEFORE QUOTING ANY RETURN FIGURE. The holdout
section shows that the searched results do not survive a period they were
not chosen on. The survivorship section shows that every absolute return
here counts only companies that still trade, so all of them are too high.
"""

from __future__ import annotations

from dataclasses import dataclass

# Source for every number in this module unless stated otherwise:
# RESEARCH_NOTES.md, section "Does a ranking exist? Yes. Does it beat an ETF?
# No. (2026-08-13)", RE-MEASURED 2026-09-14 on research_11158rows_20260914
# .parquet, 11,158 cluster episodes, event days 2018-09 .. 2026-09,
# expanding-window out-of-sample.
#
# WHY THE RE-MEASUREMENT. Every figure in this module up to 2026-09-14 was
# computed on data in which each Form 4 was counted about twice: the SEC
# daily index lists one row per CIK involved in a filing, and discovery
# deduplicated by path rather than by accession, so every dollar amount was
# roughly doubled. The fix and its regression test are in
# insider_cluster_buys.discover_filings. Returns and win rates barely moved
# (they never depended on the dollar columns); the dollar-weighted
# survivorship figure moved materially. See each section for before/after.
PROVENANCE = (
    "research_11158rows_20260914.parquet, 11,158 cluster episodes, "
    "2018-09..2026-09, expanding-window out-of-sample"
)
MEASURED_ON = "2026-09-14"


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
# The 126- and 252-day IWM/IWC cells were float("nan") until 2026-09-14 --
# not because the question was unanswerable but because the IWM/IWC price
# cache did not reach far enough to answer it. It does now, so they are
# measured rather than "n/a"; the curve slopes down against all three
# yardsticks, which is a stronger statement than the one this table used to
# be able to make.
HORIZON_EXPECTATIONS: tuple[HorizonExpectation, ...] = (
    HorizonExpectation(10, -0.0805, -0.0040, -0.0145, 0.484),
    HorizonExpectation(21, -0.1252, -0.0625, -0.0679, 0.466),
    HorizonExpectation(63, -0.1354, -0.1060, -0.1145, 0.441),
    HorizonExpectation(126, -0.1623, -0.1205, -0.1280, 0.411),
    HorizonExpectation(252, -0.1671, -0.1278, -0.1394, 0.390),
)

# Benchmark CAGRs over the measured window (2018-08-13..2026-08-12), from
# price_cache. Kept so the dashboard can explain WHY the SPY column looks so
# much worse than the IWM column: small caps trailed SPY by ~6pp/yr for
# eight years, and roughly half the apparent "insiders pick badly" result is
# that size factor, not insider skill.
# Re-measured 2026-09-14 over 2018-09-14..2026-09-11. All six are now taken
# over the SAME 2,008 trading days: previously SPY's cache ran a month past
# the others', so SPY was being credited with 7.99 years of compounding and
# the rest with 7.91 -- a free advantage to the benchmark the strategy is
# most often compared against. The size-factor gap this table exists to
# document survives the correction at 6.4pp/yr.
BENCHMARK_CAGR: dict[str, float] = {
    "SPY (large cap)": 0.1458,
    "RSP (equal-weight S&P)": 0.1098,
    "MDY (mid cap)": 0.0916,
    "IWC (micro cap)": 0.0882,
    "IWM (small cap)": 0.0822,
    "XBI (biotech)": 0.0656,
}

# Refinements that DO NOT work. Each is a median 21-day log-excess vs SPY by
# quartile of the named quantity, worst-quartile-first. Every one is flat.
# Shown in the product because these are exactly the intuitions a user will
# reach for, and "we checked, it doesn't help" is more useful than silence.
FLAT_REFINEMENTS: tuple[tuple[str, tuple[float, ...]], ...] = (
    ("Reacting faster to a fresh filing", (-0.0074, -0.0053, -0.0093, -0.0089)),
    ("More insiders in the cluster", (-0.0057, -0.0124, -0.0049, -0.0075)),
    ("Larger total dollars bought", (-0.0105, -0.0074, -0.0063, -0.0065)),
    ("A higher CEO share of the buying", (-0.0095, -0.0128, -0.0016, -0.0067)),
    ("More ten-percent owners", (-0.0079, -0.0070, -0.0088, -0.0076)),
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
#
# NOT RE-MEASURED in the 2026-09-14 pass. Every SHIPPED_MODEL_* and
# PREV_SCORE_* figure was computed on the double-counted data described at
# the top of this module, and they are deliberately left that way: they
# describe a model that no longer runs, and re-fitting a retired score to
# refresh numbers nothing consults would cost an OOF pass for no product
# benefit. They stay quotable ONLY as the historical record of why that
# score was retired -- which is a claim about its RANKING, and the ranking
# is what survived the double-count (see the module header). Do not compare
# them like-for-like against the NEW_SCORE_* constants below, which were
# measured on corrected data.

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
    "research_11158rows_20260914.parquet, 9,371 cluster episodes, "
    "2020-2026, log excess over SPY at 21 trading days, out-of-sample"
)
BAND_MEASURED_ON = "2026-09-14"
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
    Band("elevated_risk", "Elevated risk", 0, 30, -0.0268, 0.429, 0.0793),
    Band("middle", "Middle", 30, 70, -0.0070, 0.467, 0.0243),
    Band("top_band", "Top band", 70, 90, -0.0003, 0.497, 0.0133),
    Band("above_band", "Above band", 90, 100, -0.0030, 0.490, 0.0277),
)
BANDS_BY_VERDICT: dict[str, "Band"] = {b.verdict: b for b in BANDS}

# New score's quality, for the banner. Monthly-cohort IC (grouped by entry
# month so no single high-volume month dominates), against forward 21-day
# log excess over SPY.
NEW_SCORE_MONTHLY_IC = 0.0874
NEW_SCORE_IC_T = 5.30
NEW_SCORE_IC_CI = (0.0544, 0.1183)
NEW_SCORE_YEARS_POSITIVE = (7, 7)
NEW_SCORE_VOL_NEUTRAL_IC = 0.0666
LOW_VOL_RANKER_VOL_NEUTRAL_IC = 0.0023  # plain sort-by-low-volatility, same test
NEW_SCORE_DECILE_MONOTONICITY = 0.90

# The score this replaces (the retired top_decile/no_edge verdict),
# re-measured on the SAME monthly-cohort / volatility-neutral methodology as
# the new score above, for a fair comparison. (SHIPPED_MODEL_POOLED_IC above
# is an earlier, pooled-IC measurement of that same old score and is kept
# only for legacy verdict rendering -- the two are not the same number.)
PREV_SCORE_MONTHLY_IC = -0.0368
PREV_SCORE_YEARS_POSITIVE = (2, 7)
PREV_SCORE_VOL_NEUTRAL_IC = 0.0130  # below LOW_VOL_RANKER_VOL_NEUTRAL_IC
PREV_SCORE_CRASH_RATE_BOTTOM_TO_TOP_DECILE = (0.029, 0.071)

# The most durable number this project has produced: the elevated-risk band's
# chance of a >30% loss within 21 days, by out-of-sample year. Between 6.6%
# and 9.9% in every one of the 7 years measured.
#
# The pre-2026-09-14 version of this comment claimed a tighter range --
# "never below 7.3%, never above 9.2%" -- measured on the double-counted
# data. On corrected data the spread is wider at both ends (2022 fell to
# 6.6%, 2024 rose to 9.9%). The band still separates crash risk from the
# rest of the population by roughly 3x in every year, which is the claim the
# product actually rests on; the old narrow range was tighter than the
# evidence supports and should not be restated.
ELEVATED_RISK_CRASH_RATE_RANGE = (0.066, 0.099)
ELEVATED_RISK_CRASH_RATE_BY_YEAR: dict[int, float] = {
    2020: 0.0905, 2021: 0.0804, 2022: 0.0664, 2023: 0.0701,
    2024: 0.0993, 2025: 0.0753, 2026: 0.0672,
}

# A 70-90 (top_band) book looked like an index-beater and was not: it
# measures +5.78%/yr over SPY and fails a permutation test outright --
# re-running the same band search on randomly shuffled scores produces a
# result this large or larger 100% of the time. Kept here so the product
# never states an index-beating claim from this ranking.
#
# TWO CORRECTIONS LANDED HERE ON 2026-09-14, and they compound.
#
# First, ATTRIBUTION. The old +19.77%/yr and p=0.435 were never the 70-90
# band's numbers. tools/band_robustness.py tested the 80-90 slice
# (BAND_LO/BAND_HI = 0.80, 0.90) while top_band has shipped as 70-90 since
# 2026-08-20, and its output was transcribed here under a 70-90 label. On
# the old data the shipped 70-90 band measured +14.07%/yr at p=0.745, not
# +19.77%/yr at p=0.435. band_robustness now defaults to the shipped band so
# the two cannot drift apart again.
#
# Second, the DOUBLE-COUNT correction (see the module header). On corrected
# data the shipped band measures +5.78%/yr at p=1.000 -- below the null
# median of +18.80%/yr, meaning the shuffled-score search beats the real
# score every time.
#
# The product's conclusion never depended on either error: no index-beating
# claim was being made, and none is made now. But the claim is no longer a
# close call. It is not that the edge fails a significance bar; it is that
# there is no edge to test.
TOP_BAND_ANNUALIZED_EXCESS = 0.0578
TOP_BAND_PERMUTATION_P = 1.000

# Survivorship, re-measured on the current events file. Stated in the
# product because it caps how much any absolute number here can be trusted.
# Re-measured 2026-09-14 on events_20180914_20260914.parquet against the
# current price cache. The DOLLAR figure moved the most of any number in this
# module: 0.271 -> 0.322. The old double-count was not spread evenly over
# priceable and unpriceable tickers, so it was masking how much of the
# insider dollar flow sits on companies whose prices cannot be recovered.
# Survivorship bias here is worse than this project previously reported, and
# every absolute return in this module is correspondingly more optimistic.
SURVIVORSHIP = {
    "frac_tickers_unpriceable": 0.318,
    "frac_buy_rows_unpriceable": 0.211,
    "frac_buy_dollars_unpriceable": 0.322,
}

# Round-trip cost assumed in every net figure quoted above (backtest.py's
# own default slippage, 10bps per side).
ROUND_TRIP_COST = 0.0020


def headline() -> str:
    """One sentence, for a console line or a banner title. Describes the
    WHOLE POPULATION of cluster buys regardless of score -- see band_headline()
    for the sentence about what the model's own ranking means."""
    return (
        "This is a skip list, not a pick list. The reliable result is which "
        "insider buys to avoid, not which to buy."
    )


def band_headline() -> str:
    """One plain, non-technical sentence about what the banded score means.
    For the model banner -- see headline() for the population-level line."""
    return (
        "Use these bands to decide where to spend research time. The ranking "
        "reliably identifies which insider cluster buys went wrong most often. "
        "It does NOT identify which ones go up."
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
        "this large or larger every time). No index-beating claim is made."
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
    "research_11158rows_20260914.parquet, 10-seed ensemble of "
    "C19_month_vol_rel_a35_live, 79 non-overlapping 21-trading-day periods, "
    "equal weight, 20bps round trip"
)
BOOK_MEASURED_ON = "2026-09-14"


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


# RE-MEASURED 2026-09-14 on corrected data, and this is the section the
# correction hit hardest. The top band fell from 34.5%/yr at Sharpe 1.303 to
# 24.8%/yr at Sharpe 1.119, and its final multiple from 5.77x to 3.74x.
#
# The controls say the move is real and not a window artifact: "All
# clusters" (0.238 -> 0.238), SPY (0.181 -> 0.180) and IWM (0.156 -> 0.153)
# barely changed over the same periods. Only the BAND moved, because only
# the band depends on the ranking, and correcting the double-count turned
# over 36% of top_band's membership (Spearman 0.935 between the old and new
# ensemble score; top_band retained 1173 of 1824 names).
#
# THE HEADLINE CONSEQUENCE: the top band no longer beats SPY on Sharpe
# (1.119 vs 1.193). It did before (1.303 vs 1.192), and that was the single
# apparent exception to this module's "no configuration beats an index fund"
# statement. The exception is gone. The statement is now unqualified.
BOOK_RESULTS: tuple[BookResult, ...] = (
    BookResult("Top band (70-90)", 0.248, 0.199, 1.119, 1.078, -0.298, 0.72),
    BookResult("All clusters", 0.238, 0.229, 0.942, 1.148, -0.312, 0.61),
    BookResult("Bottom 30% (elevated risk)", 0.166, 0.320, 0.482, 0.820, -0.520, 0.52),
    BookResult("SPY", 0.180, 0.140, 1.193, 1.118, -0.191, 0.73),
    BookResult("IWM", 0.153, 0.201, 0.715, 0.858, -0.273, 0.58),
)

# The top-band book, over the full 6.5-year window (2020-09 .. 2026-09).
TOP_BAND_FINAL_MULTIPLE = 3.74
SPY_FINAL_MULTIPLE = 2.79
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
# Re-measured 2026-09-14. These are cut on sharpe_lab's period grid, which
# is not byte-identical to band_backtest's grid used for BOOK_RESULTS above
# -- the two agreed on the old data (SPY Sharpe 1.192 either way) and differ
# slightly on the new (1.193 vs 1.163). Each constant is kept on the grid
# that originally produced it, and both grids agree on the conclusion.
#
# The cost breakeven moved from "under 50bps" to "under 20bps": the book now
# fails to beat SPY's Sharpe even at the repo's own optimistic 20bps
# convention, where it previously cleared it.
COST_SHARPE_BY_BPS: dict[int, float] = {20: 1.123, 50: 0.941, 100: 0.639}
# Position capped at 10% of a name's trailing 20-day dollar volume. SPY's
# Sharpe (1.163) is above every capital size tested here.
LIQUIDITY_SHARPE_BY_CAPITAL: dict[str, float] = {
    "$100k": 1.063, "$1M": 0.943, "$25M": 0.857,
}
PRICE_FLOOR_EXCESS_BEFORE = 0.062  # annual excess over SPY, no price floor
PRICE_FLOOR_EXCESS_AFTER = 0.026   # annual excess over SPY, $5 minimum entry price
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
    """Lead text for the book table. The in-sample warning comes first, on
    purpose. See band_holdout_note() for what happens out of sample."""
    top = next(b for b in BOOK_RESULTS if b.name == "Top band (70-90)")
    spy = next(b for b in BOOK_RESULTS if b.name == "SPY")
    yrs, yrs_tot = TOP_BAND_YEARS_BEAT_SPY
    return (
        "Read this warning before the table. Every row below covers the same "
        "window that chose the band, so each row is in-sample for the "
        "selection it describes. On a pre-registered holdout the band stops "
        "helping. Every row is also survivors-only, so every return is too "
        "high. "
        f"Over this window a top-band book returned "
        f"{top.ann_return * 100:+.1f}%/yr at Sharpe {top.sharpe:.2f} and "
        f"Sortino {top.sortino:.2f}. SPY returned "
        f"{spy.ann_return * 100:+.1f}%/yr at Sharpe {spy.sharpe:.2f}. That is "
        f"a {TOP_BAND_FINAL_MULTIPLE:.2f}x final multiple against SPY's "
        f"{SPY_FINAL_MULTIPLE:.2f}x, and it beat SPY in {yrs} of {yrs_tot} "
        "years. "
        f"The Sharpe result passes a permutation test "
        f"(p={TOP_BAND_SHARPE_PERMUTATION_P:.3f}, null median "
        f"{TOP_BAND_SHARPE_NULL_MEDIAN:.3f}, 95th percentile "
        f"{TOP_BAND_SHARPE_NULL_P95:.3f}). The raw return claim failed the "
        f"same style of test (p={TOP_BAND_PERMUTATION_P:.3f}). The two "
        "disagree because a fat right tail inflates mean return and "
        "volatility together. That can inflate a return statistic by chance, "
        "but not a ratio of the two."
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
        f"{REFIT_TRADEABLE_UNIVERSE_SPY_SHARPE:.3f}. Max drawdown is worse "
        "than SPY's too, at -30.7% against -19.2%. "
        "Holdout: on a period the band was not chosen on, the band stops "
        "helping. See band_holdout_note() and holdout_note(). "
        "Survivorship: every return here counts only companies that still "
        "trade, so all of them are too high. See "
        "survivorship_correction_note(). "
        "Treat this as a risk screen and not a portfolio. The top band beat "
        "SPY only in a universe of sub-$5, thinly traded names, at costs and "
        "account sizes nobody can use."
    )


def top_band_holding_summary() -> str:
    """One-line summary for the model banner: top band's Sharpe and
    annualized return, with the harvestability caveat in the SAME sentence
    -- never the return alone. See risk_adjusted_note()/harvestability_note()
    for the full picture."""
    top = next(b for b in BOOK_RESULTS if b.name == "Top band (70-90)")
    spy = next(b for b in BOOK_RESULTS if b.name == "SPY")
    return (
        f"In sample, a top-band book measured {top.ann_return * 100:+.1f}%/yr "
        f"at Sharpe {top.sharpe:.2f}, against SPY's "
        f"{spy.ann_return * 100:+.1f}%/yr at Sharpe {spy.sharpe:.2f}. Do not "
        "trade on that number. It needs sub-$5, thinly traded names. It "
        "disappears under real cost, liquidity and price-floor limits. On a "
        "pre-registered holdout the band stops helping at all. These bands "
        "rank what to research first, not what to buy. See the panel below."
    )


def key_points() -> list[str]:
    """The short list a user should read before acting on this dashboard."""
    best = HORIZON_EXPECTATIONS[0]
    crash_lo, crash_hi = ELEVATED_RISK_CRASH_RATE_RANGE
    helped, slot_total = BAND_HOLDOUT_SLOTS_HELPED
    cells, cell_total = EXIT_RULE_CELLS_IMPROVED
    cut_lo, cut_hi = MEAN_TRADE_HAIRCUT_RANGE
    return [
        "Use this list to decide where to spend research time. It sorts "
        "insider cluster buys into bands by how badly they have gone before. "
        "It does not tell you which ones go up.",

        "The cluster-buy event itself is not a buy signal. Held about two "
        f"weeks, the average flagged cluster returns {best.vs_iwm * 100:+.1f}%/yr "
        "against a small-cap index fund. That is indistinguishable from owning "
        "the index. Held longer it does worse, not better.",

        "A higher percentile is not a better candidate. The 70th to 90th "
        "percentile band measures best. The top 10% measures worse than that "
        "band on both median return and crash rate, in 5 of 7 out-of-sample "
        "years. Do not sort descending and take the top rows.",

        "The bottom 30% is the durable result. Its chance of a loss worse "
        f"than 30% in 21 days stayed between {crash_lo * 100:.1f}% and "
        f"{crash_hi * 100:.1f}% in every out-of-sample year measured. Skip "
        "those names, or research them knowing that.",

        "None of the obvious quality filters work. More insiders, bigger "
        "dollar amounts, CEO participation, ten-percent-owner involvement and "
        "reacting faster to a filing were all tested. All came back flat.",

        "The bands do not improve a traded book. On a pre-registered holdout "
        f"the top band helped at {helped} of {slot_total} position counts "
        "tested. Treat the bands as a research queue, not a portfolio.",

        f"One mechanical change did survive that holdout: {EXIT_RULE_LABEL}. "
        f"It beat a fixed 21-day hold in {cells} of {cell_total} tested "
        f"combinations, by about {EXIT_RULE_MEAN_RETURN_GAIN * 100:.1f} "
        "percentage points a year. It uses no score from this model.",

        f"About {SURVIVORSHIP['frac_tickers_unpriceable'] * 100:.0f}% of the "
        "companies in the historical data have no price history, because they "
        f"stopped trading. All {DEAD_TICKERS_RESOLVED:,} were traced. "
        f"{DEAD_TICKER_FATES['Acquired'] * 100:.0f}% were acquired and "
        f"{DEAD_TICKER_FATES['Bankrupt'] * 100:.0f}% went bankrupt. Adding "
        f"them back cuts the mean trade by {abs(cut_lo) * 100:.1f} to "
        f"{abs(cut_hi) * 100:.1f} percentage points, so every return figure "
        "here is too high.",

        "No configuration tested produced a portfolio that beat an index fund "
        "out of sample. Treat any such claim from this pipeline with suspicion "
        "unless it reports a holdout and a seed sweep.",
    ]


# ---------------------------------------------------------------------------
# THE PRE-REGISTERED HOLDOUT, AND WHAT SURVIVED IT.
#
# Read this section before you quote any number from BOOK_RESULTS. Those book
# numbers are measured over the same window that chose the band, so they are
# in-sample for the selection they describe. This section is not.
#
# Source: RESEARCH_NOTES.md, "The holdout test: the search overfits, the exit
# rule survives (2026-08-20)" and "Settling the two holdout tables
# (2026-08-25)". Data: research_10861rows_20260813.parquet, 10-seed ensemble
# of C19_month_vol_rel_a35_live. Selection 2020-01..2022-12, 4,161 events.
# Holdout 2023-01..2026-08, 5,016 events. Daily-marked slot-limited book,
# per-row estimated trading costs. Reproduce with tools/final_search.py and
# tools/settle_band.py.
# ---------------------------------------------------------------------------
HOLDOUT_PROVENANCE = (
    "research_10861rows_20260813.parquet, selection 2020-01..2022-12 "
    "(4,161 events), holdout 2023-01..2026-08 (5,016 events), daily-marked "
    "slot-limited book, per-row estimated trading costs"
)
HOLDOUT_MEASURED_ON = "2026-08-25"

# The search that got caught. 7,560 configurations of universe filter, band,
# exit rule and slot count were scored on the selection window. The single
# best one was then scored once on the holdout, and the top ten were scored
# to show whether the whole region survived. None of them did.
SEARCH_CONFIGS_TRIED = 7560
SEARCH_SELECTION_SHARPE = 1.437
SEARCH_HOLDOUT_SHARPE = 0.400
SEARCH_TOP10_HOLDOUT_SHARPE_RANGE = (-0.009, 0.676)
HOLDOUT_SPY_ANN = 0.2315
HOLDOUT_SPY_SHARPE = 1.456

# What generalized. A trailing stop replaced the fixed 21-day hold on the
# WHOLE unselected population, with no model score involved.
EXIT_RULE_LABEL = (
    "a 15% trailing stop, armed once a position is up 10%, capped at 126 days"
)
EXIT_RULE_FIXED_ANN = -0.0165
EXIT_RULE_FIXED_SHARPE = 0.002
EXIT_RULE_TRAIL_ANN = 0.1267
EXIT_RULE_TRAIL_SHARPE = 0.892
# tools/settle_band.py swept 2 exit rules x 5 bands x 5 slot counts over the
# holdout. The trailing stop beat the fixed hold in every cell.
EXIT_RULE_CELLS_IMPROVED = (25, 25)
EXIT_RULE_MEAN_SHARPE_GAIN = 0.637
EXIT_RULE_MEAN_RETURN_GAIN = 0.1075

# What did NOT generalize: the band. In that same 50-cell sweep the shipped
# 70-90 band beat an unbanded book at 5 slots and lost at 10, 15, 20 and 30.
# A percentile band must not depend on how many positions a book carries, so
# the sign flip means noise. This is why the product ranks for triage and
# makes no portfolio claim.
BAND_HOLDOUT_MEAN_SHARPE_DELTA = -0.137
BAND_HOLDOUT_SLOTS_HELPED = (1, 5)


# ---------------------------------------------------------------------------
# THE SURVIVORSHIP CORRECTION.
#
# SURVIVORSHIP above states the size of the hole. This states what fell into
# it. All 2,455 tickers with no price history were resolved against EDGAR.
# The result overturns an older assumption in RESEARCH_NOTES.md that the
# missing rows would all land in the bottom deciles. They would not.
# Acquisitions close at a premium, so the bias runs in both directions.
#
# Source: RESEARCH_NOTES.md, "Six new free data sources, and the survivorship
# correction that matters most (2026-08-21)". Reproduce with
# tools/delisting_fate.py, tools/survivorship_remeasure.py and
# tools/survivorship_bound.py.
# ---------------------------------------------------------------------------
SURVIVORSHIP_MEASURED_ON = "2026-08-21"
DEAD_TICKERS_RESOLVED = 2455
DEAD_TICKER_FATES: dict[str, float] = {
    "Acquired": 0.316,
    "Renamed, still trading": 0.264,
    "Still filing, no ticker": 0.203,
    "Bankrupt": 0.130,
    "Delisted, unexplained": 0.083,
    "Unknown": 0.004,
}
MEASURED_POPULATION = 10861
RECOVERED_EVENTS = 4214
TRUE_POPULATION = 15075
UNPRICEABLE_EVENTS = 3060
# Mean trade under the book's exit rule, survivors only, then the bound once
# the unpriceable rows get outcomes under three explicit scenarios. Only the
# mean is informative here. The median is pinned by a constant assigned to
# 1,299 acquisitions and is an artifact of that assumption.
MEAN_TRADE_SURVIVORS_ONLY = 0.0452
MEAN_TRADE_HAIRCUT_RANGE = (-0.0454, -0.1239)  # optimistic .. pessimistic


def holdout_note() -> str:
    """What a pre-registered holdout did to every searched result."""
    lo, hi = SEARCH_TOP10_HOLDOUT_SHARPE_RANGE
    return (
        f"A search over {SEARCH_CONFIGS_TRIED:,} configurations scored Sharpe "
        f"{SEARCH_SELECTION_SHARPE:.3f} on the window that chose it. On a "
        f"pre-registered holdout the same configuration scored "
        f"{SEARCH_HOLDOUT_SHARPE:.3f}. The whole top-ten region failed with it, "
        f"at Sharpe {lo:+.3f} to {hi:+.3f}. Every one landed below SPY's "
        f"{HOLDOUT_SPY_SHARPE:.3f} over those same years. One bad draw is "
        "noise. A whole region collapsing is the search getting caught."
    )


def exit_rule_note() -> str:
    """The one change that survived the holdout. It needs no model."""
    cells, total = EXIT_RULE_CELLS_IMPROVED
    return (
        f"One mechanical change did survive: {EXIT_RULE_LABEL}. On the "
        f"unselected population it moved the book from "
        f"{EXIT_RULE_FIXED_ANN * 100:+.2f}%/yr to "
        f"{EXIT_RULE_TRAIL_ANN * 100:+.2f}%/yr, and Sharpe from "
        f"{EXIT_RULE_FIXED_SHARPE:.3f} to {EXIT_RULE_TRAIL_SHARPE:.3f}, with a "
        f"smaller drawdown. Across {total} band and slot-count combinations it "
        f"improved {cells}, by an average of "
        f"{EXIT_RULE_MEAN_RETURN_GAIN * 100:.2f} percentage points a year. It "
        "uses no model score at all."
    )


def band_holdout_note() -> str:
    """Why the bands are a triage tool and not a portfolio rule."""
    helped, total = BAND_HOLDOUT_SLOTS_HELPED
    return (
        "The band does not improve a traded book out of sample. Against an "
        f"unbanded book on the holdout, the 70-90 band helped at {helped} of "
        f"{total} slot counts tested. Its mean Sharpe difference was "
        f"{BAND_HOLDOUT_MEAN_SHARPE_DELTA:+.3f}. A percentile band must not "
        "depend on how many positions a book carries, so read that sign flip "
        "as noise. Use the bands to choose what to research and what to skip."
    )


def survivorship_correction_note() -> str:
    """What happened to the companies the price data lost."""
    lo, hi = MEAN_TRADE_HAIRCUT_RANGE
    return (
        f"All {DEAD_TICKERS_RESOLVED:,} tickers with no price history were "
        f"resolved against EDGAR. {DEAD_TICKER_FATES['Acquired'] * 100:.1f}% "
        f"were acquired and {DEAD_TICKER_FATES['Bankrupt'] * 100:.1f}% went "
        "bankrupt, so the bias runs both ways and not only downward. "
        f"Recovering them raises the true population from "
        f"{MEASURED_POPULATION:,} events to {TRUE_POPULATION:,}. Under three "
        f"explicit scenarios the mean trade falls by {abs(lo) * 100:.1f} to "
        f"{abs(hi) * 100:.1f} percentage points from its survivors-only value "
        f"of {MEAN_TRADE_SURVIVORS_ONLY * 100:+.2f}%. Even the most generous "
        "scenario erases the mean trade profit."
    )


def dead_ticker_rows() -> list[tuple[str, str]]:
    """(fate, share) as display strings, largest share first."""
    return [
        (fate, f"{share * 100:.1f}%")
        for fate, share in sorted(
            DEAD_TICKER_FATES.items(), key=lambda kv: -kv[1]
        )
    ]
