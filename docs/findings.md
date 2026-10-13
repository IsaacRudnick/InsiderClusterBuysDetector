# Findings

This page gives all of the research results, with their numbers and limits.
Read it before you trust a number from the screener or the backtester. For a
short version in plain language, refer to the
[README](../README.md#results).

## Source of the numbers

`findings.py` is the primary source. It contains each number that the
dashboard shows, next to the source of that number. `RESEARCH_NOTES.md` is the
full record of how the research found each number. This page is a copy of the
two files on **2026-09-14**.

> **NOTE:** If this page and `findings.py` are different, `findings.py` is
> correct.

The data contains 11,158 cluster episodes, with event days from 2018-09 to
2026-09. The research scored them out of sample, on a window that grows with time. The
band values use the 9,371 episodes from 2020 to 2026. Unless a section gives a
different value, each return is the log excess return over SPY at 21 trading
days.

## Summary

The screener gives a list of stocks to avoid. It does not give a list of
stocks to buy. The rank order can identify the insider cluster buys that most
often went wrong. It cannot identify the cluster buys that go up. No tested
configuration gave a portfolio that did better than an index fund out of
sample.

- **The rank order is real.** The monthly-cohort rank IC is +0.0874 (t=5.30, 95%
  CI +0.054 to +0.118). It was positive in 7 of 7 out-of-sample years. The
  volatility-neutral IC is +0.0666. A ranker that only sorts by low volatility
  has +0.0023. The decile monotonicity is 0.90.
- **The rank order identifies risk.** In the bottom 30% (`elevated_risk`), 7.9%
  of stocks lost more than 30% in 21 days. In each out-of-sample year, this
  rate was between 6.6% and 9.9%. That is approximately 3 times the rate for
  the other stocks. The 70th to 90th percentile band (`top_band`) has a rate
  of 1.3%.
- **The rank order does not make a portfolio.** No test in this project now
  supports a portfolio claim. Refer to
  [Portfolio claims](#portfolio-claims).
- **One sell rule passed a holdout test.** This rule is a 15% trailing stop.
  It starts when a position increases by 10%, and it sells after a maximum of
  126 days. It did better than a fixed 21-day hold in 25 of 25 tested
  combinations. It does not use the model.
- **Each absolute return in this project is too high.** Approximately one
  third of the companies in the data do not trade now, and they have no price
  history. Refer to [Survivorship bias](#survivorship-bias).

## The signal

**A cluster buy is not a signal to buy.** This table shows the annualized
excess return of all cluster buys, for each hold period:

| Hold | vs SPY | vs IWM (small cap) | vs IWC (micro cap) | Win rate |
|---|---|---|---|---|
| 10 trading days | -8.1%/yr | -0.4%/yr | -1.5%/yr | 48.4% |
| 21 | -12.5%/yr | -6.3%/yr | -6.8%/yr | 46.6% |
| 63 | -13.5%/yr | -10.6%/yr | -11.5%/yr | 44.1% |
| 126 | -16.2%/yr | -12.1%/yr | -12.8%/yr | 41.1% |
| 252 | -16.7%/yr | -12.8%/yr | -13.9%/yr | 39.0% |

After approximately two weeks, a cluster buy gives the same result as a
small-cap index fund. The return decreases as the hold period increases.
Thus, the stock price does not continue to move up after the report.

**The size factor causes half of the bad result.** Insiders do not choose
badly. Small companies did worse than large companies for the full period. The
difference was approximately 6.4 percentage points each year: SPY +14.6%
CAGR, and IWM +8.2%. The research uses the two benchmarks for this reason. If
you report only the better benchmark, the result is not correct.

**The usual quality filters do not help.** The research tested these filters
by quartile:

- More insiders in the cluster.
- Larger dollar amounts.
- A larger share of the purchase from the CEO.
- Purchases by owners of more than 10% of the stock.
- A faster response to a new report.

Each filter gave flat results.

## The rank order

- **The score that the screener uses is valid.** It is a 21-day
  quantile-regression ensemble (`research/screen_model.py`). The research
  selected it from 19 pre-registered candidates. The selection used a rank
  test, not a portfolio return. For its values, refer to
  [Summary](#summary).
- **The rank order comes from the price, not from insider skill.** When the
  research removed the 12 price, momentum and volatility features, the rank
  order went away. The IC was approximately zero and positive in only 3 of 7 years.
- **The top of the rank order is not the best part.** The 90th to 100th
  percentile (`above_band`) did worse than the 70th to 90th percentile. This
  was true for the median return and for the crash rate, in 5 of 7
  out-of-sample years. Thus, do not sort by score and take the top N.
- **The old score put the most crash risk at the top.** Its IC was positive
  in only 1 or 2 of 7 years. Its top decile had several times the crash rate of
  its bottom decile. The project removed it. The research measured these
  values before the correction of 2026-09-14. They are only a record of why
  the project removed the old score.
- **`conviction_score` does not rank.** Its results do not increase with the
  score. A score of -1 did better than +9 and +10. The project keeps it only
  for comparison. The dashboard does not show it.
- **A sector-relative target did not give the same result again.** A target
  relative to the month alone is better on each test.
- **Earnings-date features did not help.** The research built them
  point-in-time. They decreased the IC of the rank order slightly. The screener
  does not use them.
- **Five of six new free data sources made the holdout result worse.** These
  sources are FINRA short interest, SEC XBRL financial data, 13D/13G and 8-K
  item codes, a FRED regime overlay, and unused Form 4 fields. The 13D/8-K set
  gave the best in-sample rank order in the project. It still failed out of
  sample. One result stayed true: the companies with the least cash did the
  worst. More data adds more selection, and selection does not pass the
  holdout.

## Portfolio claims

No portfolio claim passed the tests.

### The top band in sample

The top band does not do better than SPY, also in sample. The test used 79
periods of 21 days, with no overlap, and a 20bps round trip.

| Portfolio | Return | Sharpe ratio | Maximum drawdown |
|---|---|---|---|
| Top band (70-90) | +24.8%/yr | 1.119 | -29.8% |
| SPY | +18.0%/yr | 1.193 | -19.1% |

The return of the band is higher, but its risk-adjusted return is lower.

### Permutation tests

**The return of the band is not real.** The band gave +5.78%/yr over SPY. It
failed a permutation test at p=1.000. The same band search on shuffled scores
gave a better result each time, with a median of +18.80%/yr.

A best-of-N band search on returns with fat tails can make a return from
random data. Thus, each band return and top-N return in this project must pass
a permutation test. A bootstrap is not sufficient.

**The Sharpe ratio of the band is not real.** Until 2026-09-14, this test
passed at p=0.005. It was the one positive portfolio result of the project. On
the corrected data, the result is p=0.185, with a median of 1.004. The test
does not pass now.

### Limits for a real investor

An investor cannot use the result. On the same grid, SPY has a Sharpe ratio of
1.163. Each of these limits puts the band below SPY:

- **Cost:** The Sharpe ratio is 1.123 at a 20bps round trip, 0.941 at 50bps
  and 0.639 at 100bps.
- **Liquidity:** Each position is limited to 10% of the 20-day dollar volume
  of the stock. The Sharpe ratio is 1.063 with $100k of capital, 0.943 with $1M
  and 0.857 with $25M.
- **Price limit:** With a minimum price of $5 at entry, the excess return over
  SPY decreases from +6.2%/yr to +2.6%/yr.
- **Universe:** The research fitted the model again with these limits: a $5
  minimum price, a $1M portfolio and 50bps costs. The Sharpe ratio was 0.630.
  For SPY, it was 1.229.

### The pre-registered holdout

A pre-registered holdout found that the search was overfitted. Refer to
`tools/final_search.py`.

- The search scored 7,560 configurations on the years 2020 to 2022. Each
  configuration had a universe filter, a band, an exit rule and a slot count.
- The best configuration had a Sharpe ratio of 1.225 on those years.
- On the years 2023 to 2026, its Sharpe ratio was -0.176. It lost 2.8% each
  year and ended at 0.90x. SPY ended at 2.11x.
- The full top ten failed also, with holdout Sharpe ratios from -0.176 to
  +0.442. Each one was below the 1.442 of SPY.

One bad result can be random. When a full group of results fails, the search
was overfitted.

**The band does not make a portfolio better out of sample.** On the holdout,
the 70-90 band did better than a portfolio without a band at 0 of 5 slot
counts. The mean change in the Sharpe ratio was -0.499. Refer to
`tools/settle_band.py`.

**A long position in the top band and a short position in the bottom band do
not help.** The bottom band still increased by +16.6%/yr in the period. It has
the worst measured results, but it did not lose money. Thus, a short position
in it loses money.

## The exit rule

The holdout grid had 2 exit rules, 5 bands and 5 slot counts. In this grid,
the trailing stop did better than the fixed 21-day hold in **25 of 25 cells**.
The mean increase was +0.98 in the Sharpe ratio and +17.1 percentage points
each year. The trailing stop is 15%. It starts when a position increases by
10%, and it sells after a maximum of 126 days.

Consider all of the events, with no band and 20 slots. The fixed hold gave
+3.3%/yr at Sharpe 0.267. The trailing stop gave +25.7%/yr at Sharpe 1.662.
This is the most consistent result of the project. It does not use the model.

That portfolio also did better than SPY on the same holdout, which gave
+22.7%/yr at Sharpe 1.442. The project does not claim that it does better than
an index fund, for two reasons:

- The result is one cell from 25. A permutation test exists to reject this
  type of result.
- The result includes only the companies that continue to trade. This error
  alone is sufficient to remove the profit of the mean trade. Refer to
  [Survivorship bias](#survivorship-bias).

The result supports one comparison only: a trailing stop is better than a
fixed hold, in each cell.

## Survivorship bias

- Of the tickers in the data, 31.8% have no price history, because they
  stopped trading. They have 21.1% of the purchase rows and 32.2% of the
  purchase dollars. The research removes these rows. It does not set them to
  zero.
- The research found all 2,353 of these tickers in EDGAR:

  | Result | Share |
  |---|---|
  | Another company bought it | 32.2% |
  | New name, still trades | 26.4% |
  | Still sends reports, no ticker | 19.5% |
  | Bankrupt | 13.0% |
  | Delisted, no known reason | 8.6% |

  A company usually sells at a premium. Thus, the error goes in the two
  directions, not only down.
- When the research adds the tickers with a new name, the true population
  increases from 11,158 events to 15,033. The research then estimated the
  other tickers with three specified scenarios. The mean trade decreases by
  4.3 to 11.5 percentage points from its survivors-only value of +4.43%. Each
  scenario removes the profit of the mean trade.

## Errors that the project found in its own work

Each of these errors changed a main result. Each one is the reason for a check
that the project uses now.

- **Until 2026-09-14, the project counted each Form 4 two times.** The SEC
  daily index has one row for each party to a report. The project removed
  duplicates by path, not by accession. Thus, each dollar amount was
  approximately two times too large.

  The rank order did not change much, but 36% of the stocks in the top band
  changed. The Sharpe test then failed. The project corrected the error, added
  a test, and measured all of the results again.
- **The project gave a band result the wrong label.** The value +19.77%/yr at
  p=0.435 was for the 80-90 band, not for the 70-90 band.
  `tools/band_robustness.py` now uses the band that the screener uses.
- **Lots below one dollar and splits without adjustment decided the
  results.** At one time, 0.7% of the lots gave 58% of the grid profit. One
  reverse split without adjustment gave 83% of the profit of a strategy that
  looked like a winner. `BT_MIN_PRICE` and `backtest/split_fingerprint.py` prevent this
  now. Before you trust a rank order, examine how much of the profit comes from a
  small number of trades.
- **A small change in the data for the fit changed a main result.** With 1.07%
  fewer rows, a main result decreased from +229.6% to +151.0%. One fit
  changed from +7.98%/yr to +0.81%/yr when only the random seed changed. For
  this reason, the score that the screener uses is an ensemble of 10 seeds.
- **The `learned_*` backtest strategies are in-sample.** The grid includes
  them only for completeness. Their results are not evidence.
