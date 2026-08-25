# Insider Cluster-Buy Detector & Backtester

Two programs over one signal: **clusters of SEC Form 4 open-market insider
purchases**, at least N distinct reporting owners buying the same issuer
inside a rolling window.

| | What it does | Entry point |
|---|---|---|
| **Live screener** | Scans recent EDGAR filings, scores today's clusters, writes a shareable dashboard | `run.bat` → `insider_cluster_buys.py` |
| **Backtester** | Fits the ranking model, then runs 63 strategies × 7 exit methods over history | `backtest.bat` → `run_research.py` + `backtest.py` |

Informational tooling, not financial advice.

**The headline, up front. This is a triage tool. It tells you which insider
cluster buys are worth manual research and which are not. It is not a trading
system. Nothing tested here beats an index fund out of sample.**

The ranking is real. Monthly-cohort rank IC is +0.0883 (t=5.08), positive in
7 of 7 out-of-sample years, with a volatility-neutral IC of +0.0630 against
+0.0145 for a plain sort-by-low-volatility ranker. Two disjoint 5-seed halves
rank-correlate +0.964. This is the first score measured here that does not
move on the random seed.

What the ranking buys you is risk triage. The bottom 30% of the list carries
a 7.9% chance of a loss worse than 30% in 21 days. That rate stayed between
7.3% and 9.2% in every out-of-sample year. The 70th to 90th percentile band
carries 1.2%. Use the bands to choose what to read, not what to buy.

What the ranking does not buy you is a portfolio. A top-band book measured
+34.5%/yr at Sharpe 1.303, against SPY's +18.1%/yr at Sharpe 1.192, and that
Sharpe result passes a permutation test (p = 0.005). It does not survive
contact with reality. The edge falls below SPY's under 50bps of round-trip
cost, at every account size once positions are capped at 10% of a name's
daily volume, and mostly disappears under a $5 minimum entry price. On a
pre-registered holdout the band stops helping at all.

One mechanical change did survive that holdout. A 15% trailing stop, armed
once a position is up 10% and capped at 126 days, moved the whole unselected
population from -1.65%/yr to +12.67%/yr with a smaller drawdown. It beat a
fixed 21-day hold in 25 of 25 tested combinations. It uses no model at all.

Every absolute return above counts only companies that still trade. All 2,455
delisted tickers were traced. 31.6% were acquired and 13.0% went bankrupt.
Adding them back cuts the mean trade by 4.5 to 12.4 percentage points, so
every return figure in this repo is too high.

Read [Findings](#findings-read-this-before-trusting-a-number) before you
believe any number this produces.

---

## Setup

```
copy .env.example .env
```

Then edit `.env` and set `SEC_USER_AGENT` to your real name and email. SEC
fair-access rules require it; without one they throttle or block you. The
scraper self-limits to 8 requests/second against SEC's published ceiling of 10.

Both `.bat` files create `.venv`, install `requirements.txt`, and run from
there. Nothing else to install. Python 3.11+ (developed on 3.14).

---

## Live screener

```
run.bat
```

Prompts for lookback days (30), minimum distinct insiders (2), and cluster
window days (14). Set `ICB_LOOKBACK`, `ICB_MIN_INSIDERS`, `ICB_WINDOW_DAYS`
to skip the prompts, necessary for a scheduled or piped run, where a bare
`input()` with no stdin would block forever.

Writes to `out/`, overwritten each run:

- `dashboard.html`, self-contained, shareable as a single attachment. The
  full payload is embedded as a `<script type="application/json">` block, so
  the file is both human- and machine-readable.
- `insider_cluster_buys.xlsx`, Flagged Clusters / All Tx / Errors sheets.

Clusters are sorted and badged by the **trained model**, no longer the old
hand-tuned `conviction_score`, which measured at zero risk-adjusted edge, and
no longer `oof_tail_classifier`, which ranked crash risk upward (see
Findings). The score is `research/screen_model.py`: a 10-seed ensemble of
quantile-regression rankers, built with `python run_research.py --fit-screen`,
which writes `research_data/screen_model_<nrows>rows_<YYYYMMDD>.joblib`. The
screener prefers that bundle and falls back to the older single-classifier
`production_model_*.joblib` with a warning if it is missing. With neither on
disk, rows render as `not_scored` rather than falling back to a score that
does not work.

| Verdict | Percentile | What it means, and what to do |
|---|---|---|
| `top_band` | 70-90 | The best-measured band. Median -0.09% vs SPY, 49.5% win rate, 1.21% chance of losing more than 30% in 21 days. **Research these first.** The median is roughly flat against SPY, so this is a reading queue and not an expected winner. |
| `above_band` | 90-100 | Scored higher, and measured WORSE than `top_band` on both median (-0.52%) and crash rate (2.97%), in 5 of 7 out-of-sample years. A higher score is not a better candidate. Research these second. |
| `middle` | 30-70 | Median -0.83%, crash rate 2.64%. Low priority. |
| `elevated_risk` | 0-30 | Median -2.68%, 43.6% win rate, and a 7.88% chance of losing more than 30% in three weeks. That rate sat between 7.3% and 9.2% in every single out-of-sample year. The most durable result this project has. **Skip these,** or research them knowing that. |
| `unavailable` | - | Too many of the model's inputs could not be computed. |
| `not_scored` | - | No bundle on disk this run. |

Every evidence claim the dashboard makes comes from `findings.py`, which keeps
each number next to its provenance so that correcting the research corrects the
product. Do not hard-code an evidence claim into a rendering function.

To re-render the dashboard after editing CSS or legend copy, without paying
for a fresh EDGAR scan:

```
python build_html.py
```

It reads the payload back out of `out/dashboard.html` and overwrites the file.

---

## Backtester

```
backtest.bat
```

Asks once for the history window in months (default 96), then runs two stages
with that same window:

1. **`run_research.py --all`**, rebuilds the event dataset from EDGAR and
   refits the ranking model, writing `research_data/oof_scores_*.parquet`.
2. **`backtest.py`**, the strategy grid, ranking on the scores stage 1 just
   wrote.

Stage 1 hands stage 2 the exact parquet it produced via a pointer file, rather
than letting `backtest.py` re-resolve `latest` by mtime, that glob's tiebreak
is arbitrary when several score files share a timestamp, and it has silently
selected the wrong model before.

A full `--all` run re-scrapes EDGAR and takes **hours** (roughly a day per year
of history). To refit from the cached events parquet in minutes instead:

```
set BT_TRAIN_ARGS=--fit-model --events-from latest
backtest.bat
```

Or skip training entirely with `BT_TRAIN=0` and backtest against whatever
`BT_MODEL_SCORES` already points at.

### Research pipeline on its own

```
python run_research.py --build-dataset    # research_*.parquet   (one row per cluster episode)
python run_research.py --fit-model        # oof_scores_*.parquet (purged walk-forward CV)
python run_research.py --all              # both, in order
python run_research.py --fit-screen       # screen_model_*.joblib, the score the live screener sorts by
python run_research.py --fit-production   # production_model_*.joblib, retired single-classifier bundle
python run_research.py --dry-run          # row counts and planned paths, writes nothing
```

`--fit-screen` and `--fit-production` are opt-in and deliberately not part of
`--all`: they are deployment artifacts, not research ones. `--fit-production`
builds the retired single-classifier bundle (`oof_tail_classifier`, which
ranked crash risk upward, see Findings) and is kept only for reproducing old
results; the live screener no longer prefers it.

Row counts between `--build-dataset` and `--fit-model` are not 1:1. The fit
stage drops rows with no usable label or entry index, then cuts the remainder
into purged folds. That gap is expected.

### Strategies

63 registered in `backtest/strategies.py`, in families:

- **Hand-built gates**, `all_clusters_equal_weight`, `conviction_only`,
  `ten_percent_owner_gated`, `big_money_director`, `low_vol_conviction`, and
  friends, plus stop-loss and fixed-63-day-hold variants.
- **`thr_gt_*`**, equal-weight above each `conviction_score` threshold, −5
  through +13. A sweep, not 19 ideas.
- **`learned_*`**, six strategies fit inside the backtest. **In-sample; do
  not trust their results.** Fitting costs about 8 minutes; `BT_FIT=0` skips
  them.
- **`model_ranked_*`**, the real ones. Rank by the trained model's OOF score
  and hold the top N (5 … 100) for a fixed 63, 21, or 10 trading days.

Each runs as an independent portfolio against 7 exit methods: fixed 30/90/180/
365-day holds, and 365-day-capped trailing stops at 10/20/30% per FIFO lot.
SPY buy-and-hold is the benchmark.

Note that the exit grid barely binds. Over 99% of exits are *trims* triggered
by signal decay, not expiries, median hold is 7–10 days even under the "365d"
method. Fixed-hold strategies exist to address this.

### Environment variables

Every `BT_*` variable, read by `backtest.py`:

| Variable | Meaning |
|---|---|
| `BT_MONTHS` | History window. `backtest.bat` asks once and passes it to both stages. |
| `BT_AS_OF` | `YYYY-MM-DD` window end. Pin it or a run is not reproducible. |
| `BT_EVENTS_FROM` | Path to an events parquet, or `latest`. Skips the scrape. |
| `BT_MODEL_SCORES` | Path to an OOF scores parquet, or `latest`. |
| `BT_CAPITAL` | Starting capital per strategy. |
| `BT_STRATEGIES`, `BT_EXITS` | Comma lists, or `all`. |
| `BT_RF` | `zero` or `tbills`. |
| `BT_OFFLINE` | `0`/`1`. Cache-only, no network. |
| `BT_SLIPPAGE_BPS`, `BT_LIQUIDITY_FLOOR`, `BT_MIN_PRICE` | Trading frictions. Keep `BT_MIN_PRICE`, see Findings. |
| `BT_COST_SWEEP` | `none` or `low_med_high`. |
| `BT_FIT`, `BT_FIT_HORIZON` | Fit the in-sample `learned_*` strategies (30/90/180/365). |
| `BT_DROP_TICKER_REUSE` | Default `1`. Drop events whose ticker later changed hands to an unrelated company. |
| `BT_WRITE_WEIGHTS`, `BT_VERBOSE` | Diagnostics. |

Output lands in `out/backtest_<timestamp>/`: `report.html` plus
`equity_*.csv` and `trades_*.csv` per strategy × exit pair. A full grid run is
hundreds of MB.

### Research: re-running the candidate sweep

```
python tools/run_score_lab.py --seeds 0,1,2      # every pre-registered candidate
python tools/ship_candidate.py --seeds 10        # the chosen score, deep audit
python tools/band_robustness.py                  # the audits that kill a band claim
```

Every candidate is reported, pass or fail, the 19 in `tools/run_score_lab.py`
are all there is, and adding a 20th means re-running and re-reporting the
whole list, not quoting the new one alone.

---

## Layout

```
backtest.py                 entry point, the strategy grid
run_research.py             entry point, dataset build + model fit
insider_cluster_buys.py     entry point, the live EDGAR screener
build_html.py               dashboard renderer (one consumer: the screener)
findings.py                 every measured claim the product states, next to its
                            provenance. Correcting the research here corrects
                            the dashboard and the workbook. Never inline a
                            number into a rendering function.
ipo_lookup.py               first-trade dates; shared by the screener AND backtest/

backtest/                   the backtester's own modules
  engine, strategies, metrics, report, state, history, prices, splits,
  research, model_scores, signal_fit, legend, sales_history, tickers,
  ticker_reuse, price_overrides, split_fingerprint

research/                   the ranking model
  model.py                  fit_and_validate, purged walk-forward CV
  live_score.py             scores one live cluster from a production bundle
  screen_model.py           the score the live screener sorts by

tools/                      standalone CLIs, each `python tools/<name>.py`
  warm_prices               bulk-fill the price cache
  repair_price_cache        (see Findings, rests on a false premise)
  issuer_meta               fetch SIC / exchange per issuer CIK from SEC
  ensemble_model            blend model objectives
  refit_stability           re-fit under perturbation, measure retention
  objective_sweep           sweep TAIL_THRESH and objective choice
  score_lab                 out-of-fold score construction plus the fixed
                            pre-registered gauntlet every candidate is graded on
  run_score_lab             the 19 pre-registered candidates and the runner
  ship_candidate            deep audit of the chosen score against the one it
                            replaces
  band_backtest             which percentile band to surface, measured against
                            SPY, IWM and IWC
  band_robustness           the permutation, concentration, price-floor,
                            split-half and leave-one-year-out audits that killed
                            the band's return claim
  settle_band               band vs no band on the holdout, sweeping the slot
                            count both disputed tables left unfixed. The band's
                            sign flips; the exit rule's does not.

tests/                      pytest; `python -m pytest tests -q`
  test_findings.py          guards the claims the product makes about itself.
                            Not that the numbers are right, which only the
                            research can say, but that every headline number
                            travels with the limit that applies to it.
```

Everything in `tools/` is dual-purpose, importable as a library and runnable
as `python tools/name.py`. Python puts a script's *own* directory on
`sys.path[0]`, not the repo root, so each carries a guarded `REPO_ROOT`
bootstrap to find its siblings.

### Caches (all gitignored, all reproducible, none cheap)

| Directory | Contents | Cost to rebuild |
|---|---|---|
| `parse_cache/` | One JSON per SEC accession, ~1.3M files, ~2 GB | ~1 day per year of history |
| `price_cache/` | Per-ticker parquets, ~420 MB | Refetched from Yahoo on demand |
| `clusters_history/` | Scraped event parquets, ~100 MB | The scrape above |
| `research_data/` | Datasets, OOF scores, fitted bundles | Minutes, given the above |
| `ipo_cache/`, `issuer_meta_cache/` | Per-ticker / per-CIK JSON | Cheap |
| `out/` | Reports and dashboards | Cheap |

Back up `parse_cache/` and `clusters_history/`. They are the expensive ones.

---

## Findings, read this before trusting a number

`RESEARCH_NOTES.md` is the full record. The short version, because several of
these invalidate the obvious reading of a report:

- **The cluster-buy event is not a buy signal.** Held ~2 weeks, the average
  flagged cluster returns −0.69%/yr against a small-cap index fund -
  indistinguishable from just owning the index. Held longer it does *worse*:
  −6.5%/yr at 21 days, −10.7%/yr at 63. The curve only slopes down, so there is
  no post-filing drift to capture.
- **The old shipped score ranked crash risk upward, on every horizon it was
  graded at.** At 63 days: pooled rank IC −0.053, positive in only 1 of 7
  years, top decile ~6x the 30%-loss rate of the bottom decile. An earlier
  volatility-matched test found +4.74pp (p=0.004) for that same top decile;
  both can be true, fat right tail *and* fat left tail, but the encouraging
  half alone is misleading. Graded again at 21 days on the pre-registered
  gauntlet built for its replacement: monthly IC −0.0368, positive in 2 of 7
  years, volatility-neutral IC +0.0130, below a plain sort-by-low-volatility
  ranker. Replaced throughout the codebase.
- **None of the obvious quality filters work.** More insiders, bigger dollar
  amounts, CEO share of buying, ten-percent-owner involvement, and reacting
  faster to a fresh filing were each tested by quartile. All flat.
- **Survivorship bias caps everything above.** 33.2% of tickers are
  unpriceable, they stopped trading and the provider deleted them, along with
  24.9% of buy rows and 27.1% of buy dollars. Those rows are *dropped*, not
  zeroed, so every absolute return is optimistic by an unknown margin.
- **Half the "insiders pick badly" result is the size factor**, not insider
  skill: small caps trailed SPY by ~6pp/yr over the measured window. Both SPY
  and IWM/IWC yardsticks are kept for that reason, reporting only the
  flattering one is benchmark shopping.
- **A ranking that holds up now ships to the live screener.** The 21-day,
  quantile-regression ensemble in `research/screen_model.py` gives monotone
  deciles, positive in 7 of 7 years, and survives a volatility-neutral audit
  that four higher-headline candidates failed. The backtester's own model,
  `research/model.py`, still fits against `adj_63`.
- **The top of the ranking is not the best part of it.** Decile 9 is worse
  than decile 8 on both median and crash rate, in 5 of 7 out-of-sample years.
  Sorting descending and taking the top N, what the old score's verdicts
  did, lands on the wrong rows.
- **Sector-relative training, an earlier finding, did not replicate.**
  Month-relative alone beats it on every axis of the gauntlet.
- **Strip the 12 price/momentum/volatility features and the ranking
  disappears.** IC −0.004. This is price context, not evidence that insiders
  pick well.
- **A best-of-N band search on a fat-tailed return distribution manufactures
  a return from noise.** The 70-90th percentile book's +19.77%/yr over SPY
  produced a permutation-test null median of +18.46%/yr (p = 0.435) when the
  same 12-band search ran on shuffled scores. Any band or top-N return figure
  from this repo needs a permutation test, not just a bootstrap, the
  bootstrap prices the sampling of periods but not the selection of the band.
- **`conviction_score` does not rank.** No monotonicity; score −1 beats +9 and
  +10. Retained for comparison only, and gone from the dashboard.
- **The top band's Sharpe, not just its return, now passes a permutation
  test.** Holding a 70-90th-percentile book measured Sharpe 1.303 (Sortino
  1.400, +34.5%/yr) against SPY's Sharpe 1.192 (+18.1%/yr), a 5.77x final
  multiple against SPY's 2.76x, beating SPY in 4 of 7 years, and that Sharpe
  result survives re-running the same 132-recipe search on shuffled scores
  (p = 0.005; null median 0.987, 95th percentile 1.236). The earlier raw-return
  version of this same book (+19.77%/yr) failed an identical test (p = 0.435).
  The two disagree because a fat right tail inflates mean return and
  volatility together, that can inflate a raw return statistic by chance, but
  not a ratio of the two.
- **Sharpe rises with score, not just return.** Sharpe climbs from 0.21 in
  the bottom decile to 1.33 in the top, a risk-adjusted confirmation of the
  band ordering, not only a return-based one.
- **The Sharpe result is not harvestable.** Cost: Sharpe falls below SPY's
  somewhere under 50bps round trip (1.303 at 20bps, 1.147 at 50bps, 0.886 at
  100bps). Liquidity: capping a position at 10% of a name's 20-day dollar
  volume gives Sharpe 1.107 at $100k of capital, 0.982 at $1M, 0.750 at $25M -
  below SPY (1.192) at every size tested. Price floor: a $5 minimum entry
  price cuts annual excess over SPY from +14.1% to +3.2%. These are not
  optional footnotes, they travel with the number everywhere it is shown.
- **Refitting inside a tradeable universe does not rescue it.** Rebuilding the
  model with a $5 price floor, a $1M book, and 50bps costs gives Sharpe 0.650
  against SPY's 1.237 over the same window.
- **Long the top band, short the bottom band does not work either.** The
  bottom band ("elevated risk") still rises in absolute terms over this
  window, it is the worst-*measured* book, not a losing one, so shorting it
  loses money on the short leg regardless of the long leg's edge.
- **Point-in-time earnings-proximity features were built and tested, and they
  do not help.** Adding them to the live score's feature set moved
  monthly-cohort IC from +0.0885 to +0.0804, a decline, not an improvement.
  Not shipped.
- **Sub-dollar lots and unadjusted splits used to decide the leaderboard.**
  0.7% of lots once produced 58% of grid P&L, and a single unadjusted reverse
  split gave a "winning" strategy 83% of its P&L. `BT_MIN_PRICE` and
  `backtest/split_fingerprint.py` exist because of this. Audit P&L
  concentration before believing any ranking.
- **Refit instability decides the headline.** 1.07% fewer training rows moved a
  flagship result from +229.6% to +151.0%. Only the 5-slot variant survived.
- **A pre-registered holdout caught the whole search.** 7,560
  configurations of universe filter, band, exit rule and slot count were
  scored on a selection window. The best scored Sharpe 1.437 there and 0.400
  on a holdout it had never seen. The entire top-ten region failed with it,
  at holdout Sharpe -0.009 to +0.676, every one below SPY's 1.456. One bad
  draw is noise. A whole region collapsing is the search getting caught.
  `tools/final_search.py`.
- **Every absolute return in this repo is too high, and the bias runs both
  ways.** 33.2% of tickers have no price history because they stopped
  trading. All 2,455 were resolved against EDGAR: 31.6% acquired, 26.4%
  renamed and still trading, 20.3% still filing without a ticker, 13.0%
  bankrupt, 8.3% delisted without explanation. This overturns a standing
  assumption in `RESEARCH_NOTES.md` that the missing rows would all land in
  the bottom deciles. Deals close at a premium. Recovering the priceable ones
  raises the true population from 10,861 events to 15,075, and bounding the
  rest cuts the mean trade by 4.5 to 12.4 percentage points. Even the most
  generous assumption erases the mean trade profit.
- **Six free data sources were added and five made the holdout book worse.**
  FINRA short interest, SEC bulk XBRL fundamentals, 13D/13G and 8-K item
  codes, a FRED regime overlay and unused Form 4 fields were each attached
  point-in-time and tested. The 13D/8-K set had the best in-sample ranking of
  anything measured here and still lost out of sample. One descriptive lead
  survived: companies closest to running out of cash do worst, monotonically.
  Adding data is adding selection, and selection is what does not survive.
- **The score band does not improve a traded book out of sample; the exit
  rule does.** Two holdout tables in `RESEARCH_NOTES.md` disagreed on this.
  Re-run with only the band and the slot count varying (`tools/settle_band.py`),
  the shipped 70-90 band beats an unbanded book at 5 slots and loses at 10, 15,
  20 and 30 -- mean Sharpe -0.137, helping in 1 of 5 slot counts. A percentile
  band should not care how many positions the book holds, so read it as noise.
  In the same 50-cell grid the trailing stop beats the fixed 21-day hold in
  **25 of 25 cells** (mean +0.637 Sharpe, +10.75pp/yr). Rank for risk triage,
  not for portfolio construction.
- **The `learned_*` strategies are in-sample.** They are in the grid for
  completeness. Their numbers are not evidence.

Numbers above are sourced from `findings.py` (provenance:
`research_groupE_10905rows_20260809.parquet`, 10,905 cluster episodes,
2018-08…2026-08, expanding-window out-of-sample, measured 2026-08-13) and
`RESEARCH_NOTES.md`. The `screen_model.py` gauntlet and band numbers are from
`research_10861rows_20260813.parquet`, 9,095 scored cluster episodes,
out-of-sample 2020-2026, measured 2026-08-20, `RESEARCH_NOTES.md`, "A ranking
that beats the shipped one, and a portfolio claim that dies". The
risk-adjusted book results (`findings.BOOK_RESULTS`) are from the same
parquet, a 10-seed ensemble of `C19_month_vol_rel_a35_live`, 78
non-overlapping 21-trading-day periods, equal weight, 20bps round trip,
measured 2026-08-20.
