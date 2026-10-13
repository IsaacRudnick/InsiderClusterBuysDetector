# Findings

Everything the research measured, with numbers and caveats. Read this before
trusting any figure the screener or backtester prints. For a plain-language
version, see the [README](../README.md#what-it-found).

**Where the numbers come from.** `findings.py` is the source of truth: every
figure the dashboard states lives there, next to its provenance.
`RESEARCH_NOTES.md` is the full record of how each one was reached. This page
is a snapshot of both as of **2026-09-14**. If it ever disagrees with
`findings.py`, `findings.py` is right.

Data: 11,158 cluster episodes, event days 2018-09 to 2026-09, scored
out of sample on an expanding window. Band figures use the 9,371 episodes from
2020-2026. Returns are log excess over SPY at 21 trading days unless stated.

## The short version

This is a **skip list, not a pick list**. The ranking reliably identifies
which insider cluster buys went wrong most often. It does not identify which
ones go up, and no configuration tested produced a portfolio that beats an
index fund out of sample.

- **The ranking is real.** Monthly-cohort rank IC +0.0874 (t=5.30, 95% CI
  +0.054 to +0.118), positive in 7 of 7 out-of-sample years. Volatility-neutral
  IC +0.0666, against +0.0023 for a plain sort-by-low-volatility ranker.
  Decile monotonicity 0.90.
- **What it buys you is risk triage.** The bottom 30% (`elevated_risk`) lost
  more than 30% within 21 days 7.9% of the time. That rate stayed between
  6.6% and 9.9% in every out-of-sample year, roughly 3x the rest of the
  population. The 70-90th percentile band (`top_band`) carries 1.3%.
- **What it does not buy you is a portfolio.** No test in this project now
  supports a portfolio claim. See [Portfolio claims](#portfolio-claims-none-survive).
- **One mechanical change survived a holdout:** a 15% trailing stop, armed once
  a position is up 10% and capped at 126 days. It beat a fixed 21-day hold in
  25 of 25 tested combinations. It uses no model.
- **Every absolute return here is too high**, because roughly a third of the
  companies in the data no longer trade and have no price history. See
  [Survivorship](#survivorship-every-return-is-too-high).

## The signal itself

- **The cluster-buy event is not a buy signal.** Annualized excess return of
  the whole population, by holding period:

  | Hold | vs SPY | vs IWM (small cap) | vs IWC (micro cap) | Win rate |
  |---|---|---|---|---|
  | 10 trading days | -8.1%/yr | -0.4%/yr | -1.5%/yr | 48.4% |
  | 21 | -12.5%/yr | -6.3%/yr | -6.8%/yr | 46.6% |
  | 63 | -13.5%/yr | -10.6%/yr | -11.5%/yr | 44.1% |
  | 126 | -16.2%/yr | -12.1%/yr | -12.8%/yr | 41.1% |
  | 252 | -16.7%/yr | -12.8%/yr | -13.9%/yr | 39.0% |

  Held about two weeks, a flagged cluster is indistinguishable from owning a
  small-cap index. The curve only slopes down, so there is no post-filing
  drift to capture.
- **Half the "insiders pick badly" result is the size factor**, not insider
  skill. Small caps trailed SPY by about 6.4pp/yr over the window (SPY +14.6%
  CAGR, IWM +8.2%). Both yardsticks are kept for that reason; reporting only
  the flattering one is benchmark shopping.
- **None of the obvious quality filters work.** More insiders, bigger dollar
  amounts, a higher CEO share of the buying, ten-percent-owner involvement and
  reacting faster to a fresh filing were each tested by quartile. All flat.

## The ranking

- **The score that ships holds up.** A 21-day, quantile-regression ensemble
  (`research/screen_model.py`), chosen from 19 pre-registered candidates on a
  ranking gauntlet, never on a portfolio return. Figures are under
  [The short version](#the-short-version).
- **It is price context, not evidence that insiders pick well.** Strip its 12
  price, momentum and volatility features and the ranking disappears: IC about
  zero, positive in 3 of 7 years.
- **The top of the ranking is not the best part of it.** The 90-100th
  percentile (`above_band`) measured worse than the 70-90th on both median
  return and crash rate, in 5 of 7 out-of-sample years. Sorting descending and
  taking the top N lands on the wrong rows.
- **The old shipped score ranked crash risk upward**, on every horizon it was
  graded at: positive IC in only 1 or 2 of 7 years, and its top decile carried
  several times the crash rate of its bottom decile. It has been replaced
  throughout. These figures were measured before the 2026-09-14 correction and
  are kept only as the record of why it was retired.
- **`conviction_score` does not rank.** No monotonicity; a score of -1 beat +9
  and +10. Retained for comparison only, and gone from the dashboard.
- **Sector-relative training, an earlier finding, did not replicate.**
  Month-relative alone beats it on every axis of the gauntlet.
- **Earnings-proximity features were built point-in-time and did not help.**
  They lowered the ranking's IC slightly. Not shipped.
- **Six free data sources were added and five made the holdout book worse.**
  FINRA short interest, SEC bulk XBRL fundamentals, 13D/13G and 8-K item codes,
  a FRED regime overlay, and unused Form 4 fields were each attached
  point-in-time and tested. The 13D/8-K set had the best in-sample ranking of
  anything measured here and still lost out of sample. One descriptive lead
  survived: companies closest to running out of cash do worst, monotonically.
  Adding data is adding selection, and selection is what does not survive.

## Portfolio claims: none survive

- **The top band does not beat SPY, even in sample.** Over 79 non-overlapping
  21-day periods at 20bps round trip, a top-band book returned +24.8%/yr at
  Sharpe 1.119, against SPY's +18.0%/yr at Sharpe 1.193. Higher return, worse
  risk-adjusted. Max drawdown -29.8% against SPY's -19.1%.
- **Its return is noise.** The band's +5.78%/yr over SPY fails a permutation
  test at p=1.000: re-running the same band search on shuffled scores beats it
  every time (null median +18.80%/yr). A best-of-N band search on a fat-tailed
  return distribution manufactures a return from noise, so any band or top-N
  return from this repo needs a permutation test, not just a bootstrap.
- **Its Sharpe is noise too.** This test passed until 2026-09-14 (p=0.005)
  and was the project's one positive portfolio result. On corrected data it
  reads p=0.185 (null median 1.004). It no longer passes.
- **It is not harvestable even if it were real.** Against SPY's Sharpe of
  1.163 on the same grid:
  - Cost: Sharpe 1.123 at 20bps round trip, 0.941 at 50bps, 0.639 at 100bps.
  - Liquidity: capping a position at 10% of a name's 20-day dollar volume
    gives 1.063 at $100k of capital, 0.943 at $1M, 0.857 at $25M.
  - Price floor: a $5 minimum entry price cuts annual excess over SPY from
    +6.2% to +2.6%.
  - Refitting inside a tradeable universe ($5 floor, $1M book, 50bps) gives
    Sharpe 0.630 against SPY's 1.229.
- **A pre-registered holdout caught the whole search.** 7,560 configurations
  of universe filter, band, exit rule and slot count were scored on 2020-2022.
  The best scored Sharpe 1.225 there, then -0.176 on 2023-2026, losing 2.8%/yr
  and ending at 0.90x against SPY's 2.11x. The whole top-ten region failed
  with it, at holdout Sharpe -0.176 to +0.442, every one below SPY's 1.442.
  One bad draw is noise. A whole region collapsing is the search getting
  caught. `tools/final_search.py`.
- **The band does not improve a traded book out of sample.** Against an
  unbanded book on the holdout, the 70-90 band helped at 0 of 5 slot counts
  (mean Sharpe -0.499). `tools/settle_band.py`.
- **Long the top band, short the bottom band does not work either.** The
  bottom band still rose (+16.6%/yr) over the window. It is the worst-measured
  book, not a losing one, so shorting it loses money.

## What did survive: the exit rule

In the same holdout grid (2 exit rules x 5 bands x 5 slot counts), a 15%
trailing stop, armed once a position is up 10% and capped at 126 days, beat
the fixed 21-day hold in **25 of 25 cells**, by a mean +0.98 Sharpe and
+17.1pp/yr. On the whole unselected population at 20 slots it moved the book
from +3.3%/yr (Sharpe 0.267) to +25.7%/yr (Sharpe 1.662).

That is the most consistent result the project has produced, and it requires
no model. On paper that book also beat SPY over the same holdout (+22.7%/yr,
Sharpe 1.442). The project still makes no index-beating claim from it: the
figure is one cell picked from 25, of exactly the kind the permutation
discipline exists to reject, and it counts only companies that still trade, a
bias large enough on its own to erase the mean trade's profit (below). What
the result supports is the comparison it was built for: trailing stop over
fixed hold, every time.

## Survivorship: every return is too high

- 31.8% of tickers in the data have no price history because they stopped
  trading. They hold 21.1% of buy rows and 32.2% of buy dollars. Those rows are
  dropped, not zeroed.
- All 2,353 were resolved against EDGAR: 32.2% acquired, 26.4% renamed and
  still trading, 19.5% still filing without a ticker, 13.0% bankrupt, 8.6%
  delisted without explanation. Deals close at a premium, so the bias runs in
  both directions, not only downward.
- Recovering the priceable ones raises the true population from 11,158 events
  to 15,033. Bounding the rest under three explicit scenarios cuts the mean
  trade by 4.3 to 11.5 percentage points from its survivors-only +4.43%. Even
  the most generous scenario erases the mean trade profit.

## Mistakes the project caught in itself

Kept because each one changed a headline, and each is the reason a guard now
exists.

- **Every Form 4 was counted twice until 2026-09-14.** SEC's daily index lists
  one row per party to a filing, and discovery deduplicated by path rather
  than by accession, so every dollar amount was about 2x. Rankings barely
  moved, but top-band membership turned over 36%, and that is what retired
  the passing Sharpe test above. Fixed, with a regression test, and everything
  re-measured.
- **A band was quoted under the wrong label.** The +19.77%/yr at p=0.435 once
  quoted for the 70-90 band was the 80-90 band's result. `band_robustness.py`
  now defaults to the shipped band.
- **Sub-dollar lots and unadjusted splits used to decide the leaderboard.**
  0.7% of lots once produced 58% of grid P&L, and a single unadjusted reverse
  split gave a "winning" strategy 83% of its P&L. `BT_MIN_PRICE` and
  `backtest/split_fingerprint.py` exist because of this. Audit P&L
  concentration before believing any ranking.
- **Refit instability decided a headline.** 1.07% fewer training rows moved a
  flagship result from +229.6% to +151.0%, and a single fit swung +7.98%/yr to
  +0.81%/yr on the random seed alone. The shipped score is a 10-seed ensemble
  for this reason.
- **The `learned_*` backtest strategies are in-sample.** They are in the grid
  for completeness. Their numbers are not evidence.
