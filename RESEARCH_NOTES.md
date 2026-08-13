# Ranking-model rebuild: working notes

Status file for the effort to replace the hand-tuned cluster score. Written
2026-07-30. Update it as findings land.

## Diagnosis (all numbers from `out/backtest_20260718_010152/`)

### 1. The conviction score does not rank
Mean lot return by `score_at_entry`, from `trades_all_clusters_equal_weight_365d.csv`
(48,557 lots, gate = any 2-insider cluster):

| score | n | mean % | score | n | mean % |
|---|---|---|---|---|---|
| -10 | 257 | +2.63 | 5 | 4,321 | +1.63 |
| -1 | 2,889 | +2.31 | 9 | 1,884 | +0.93 |
| 0 | 2,636 | +0.90 | 10 | 1,266 | +0.57 |
| 4 | 6,196 | +1.25 | 12 | 283 | +2.62 |

No monotonicity. Score -1 beats score +9 and +10.

### 2. No strategy has significant alpha
| strategy | CAGR | alpha_ann | t_alpha | beta | mean_exposure |
|---|---|---|---|---|---|
| spy_buy_and_hold | 0.147 | - | - | 1.00 | 1.00 |
| ten_percent_owner_gated | 0.173 | 0.093 | **0.99** | 0.66 | 0.56 |
| all_clusters_equal_weight | 0.051 | -0.042 | -0.86 | 0.71 | 0.47 |
| conviction_only | 0.051 | -0.061 | -1.12 | 0.88 | 0.76 |
| thr_gt_p11 | 0.113 | 0.110 | **1.91** | 0.06 | 0.10 |

The best t-stat in all 245 runs is 1.91, on the most selective strategy
(421 lots). Ranking the grid by CAGR rewards exposure, not selection.

### 3. ten_percent_owner_gated is a regime bet, not a classifier
Per-entry-year mean lot return: 2018 +1.72, 2019 +4.24, 2020 +6.31,
2021 +0.76, 2022 **-1.11**, 2023 +1.60, 2024 +0.76, 2025 +4.97, 2026 +4.12.
Baseline over the same years stays inside +0.49 .. +2.02. TPO touches only
881 unique tickers over 8 years and enters on 1,978 of 2,011 trading days.

### 4. The last run never fit anything
`config.json` has `"fit_signal": false`. 35 of 41 strategies ran. All six
`learned_*` strategies were dropped. TPO won a race the learned models were
not entered in.

### 5. Engine and labels disagree on holding period
99%+ of exits are `trim`, not `expiry`. Median hold is 7 days
(all_clusters_equal_weight) to 10 days (TPO), even under the "365d" exit.
The 7-method exit grid is nearly a no-op. `signal_fit.py` trains on 90
trading-day forward returns. Fixed-hold mode addresses this.

### 6. Severe survivorship bias (biggest validity problem)
`n_delisted` is 0-6 lots per strategy, but `n_skipped_no_price` is huge:

| strategy | n_lots | n_delisted | n_skipped_no_price |
|---|---|---|---|
| all_clusters_equal_weight | 48,557 | 1 | 37,956 |
| conviction_only | 29,101 | 3 | 79,314 |
| ten_percent_owner_gated | 16,204 | 6 | 41,965 |
| score_weighted | 23,922 | 6 | 108,004 |

Tickers without price data are excluded from the universe, not held to zero.
Insider clusters concentrate in micro-caps, the population that delists.
Every absolute return number here is an upper bound.

No free fix found. Stooq is behind a JavaScript bot check and refuses
automated requests for every symbol including AAPL. Plan: bound the bias by
re-pricing missing-price events at -100% and at -30% and reporting the range
next to every model result.

## Data facts

- `clusters_history/events_20180717_20260717.parquet`: 597,278 rows,
  126,128 filings, 7,467 tickers, 28,784 owners, 2018-07-17 .. 2026-07-17.
  Reads in about 15 seconds.
- Raw data has transaction dates up to **2033-11-18**. 58 rows have
  transaction_date after filing_date. `_clean_events_df` drops them.
- 121 malformed tickers over 5,121 rows. The normalizer recovers 107 of them
  (2,250 rows).
- `_clean_events_df` drops **15,117 rows (2.5%)** whose ticker cannot be
  normalized, across only **21 distinct symbols**. This is correct, not data
  loss. The bulk are placeholders that mean "no ticker known":

  | symbol | rows | | symbol | rows |
  |---|---|---|---|---|
  | `NONE` | 11,666 | | `N.A.` | 106 |
  | `N/A` | 2,790 | | `ACRG.A.U` | 28 |
  | `NA` | 330 | | hash-like junk | ~77 |
  | `--` | 120 | | | |

- **Recovery opportunity, not yet done:** the 11,666 `NONE` rows are 2% of the
  dataset. These filings still carry an `issuer_cik`. A CIK to ticker map
  would recover them. Worth doing only if the model shows real edge and more
  events would help.
- Ticker universe after normalization: **7,382** distinct symbols, down from
  7,466 raw. The normalizer drops 125 junk forms such as `"OMEX"`, `(CALX)`
  and `[N/A]`, and recovers 40 symbols that the raw list never reached at
  all, for example `ASAQ`, `BBX`, `CALX` and `FBC`.
- **Price coverage as of 2026-07-30: 4,909 of 7,382 cached, 2,551 missing
  (33.5%), 40 rate-limited (0.5%, recoverable on a re-run).** That missing
  third is the survivorship exposure. It is concentrated in the delisted
  micro-caps that insider clusters favour, so it is not a random third.
- `tools/warm_prices.py` used to read the events parquet directly and apply its own
  crude cleanup, which bypassed `backtest/tickers.normalize_ticker`. It
  therefore spent requests on junk and never fetched the recovered form of a
  malformed symbol. Fixed. Note that `normalize_ticker_series` returns None
  or NaN for unusable input, and **NaN is truthy**, so filter these with
  `isinstance(x, str)` and never with bare truthiness.
- `parse_cache/` holds 324,523 files but covers only **24.2%** of the 126,128
  event filings. Sell-side context needs all Form 4s per issuer per window,
  so concurrent-selling features are out of scope until a targeted scrape
  runs (about 210k fetches at the 8 req/s limiter, roughly 7 hours).
- `owner_roles` is free text like
  "Director, Officer (President,Chairman & CEO), 10% Owner". Parse it for
  CEO/CFO/Chair. The current code collapses all of it into one `is_officer`
  bit, which is the largest single information loss.

## Environment

- Python 3.14. pandas **3.0.5** (major bump from the 2.3.3 this code was
  written against), numpy 2.5.1, pyarrow 25.0.0, scikit-learn 1.9.0,
  lightgbm 4.7.0, yfinance 1.5.2, plotly 6.9.0.
- `requirements.txt` did not exist before this work. `backtest.bat` referenced
  it and would exit. `build_readme.bat` called a missing `render_readme.py` to
  render a `README.md` that had never been written; it is now deleted, and
  `README.md` and `run.bat` (the screener launcher it also assumed) exist.

## Changes landed

- `requirements.txt` created.
- `backtest/history.py`: `load_events_df`, `resolve_events_path`,
  `_clean_events_df`, and `build_history(months, as_of, events_from)`.
  `clusters_history/` was written but never read before this.
- `backtest.py`: `BT_EVENTS_FROM` and `BT_AS_OF`. `date.today()` no longer
  leaks into the price window or the calendar.
- `backtest/prices.py`: yfinance tz cache moved to `price_cache/_yf_tz`.
  The shared default location caused "database is locked", and a locked
  cache makes `yf.download` return an empty frame with no error, so tickers
  looked delisted. Added `_download_serial` fallback so a ticker is only
  recorded missing after it fails on its own. `median_dollar_volume` is now
  memoized and binary-searched (verified 297/297 parity with the old scan).
- `backtest/tickers.py` + `tests/test_tickers.py` (78 tests).
- `tools/warm_prices.py`: one-time bulk price cache fill.

## Phase 2 landed: event-level research dataset

`backtest/research.py` builds one row per cluster episode: `x_`-prefixed
features, `fwd_/spy_/adj_/delisted_` labels per horizon, and the legacy
`f_<key>` flags plus `conviction_score` for comparison.

Point-in-time correctness was audited directly, not just by its own tests:

- Same-day events cannot see each other. Owner, issuer and ticker history
  updates are staged per day and merged only after every event for that day
  is built.
- A prior event's `adj_63` is usable only after `_label_window_closed()`
  confirms its exit day falls strictly before the current event day. Plain
  counts are not gated, which is correct, because a past buy is public at
  once.
- Price context has no lookahead. `_prior_dates()` cuts strictly before the
  entry day.
- `_fwd_adj_return()` sets `delisted_h` only when the exit leg fell back to
  the last close, and never treats a missing entry price as a delisting.
  This gives a direct handle on the survivorship problem above.

`x_owner_prior_buys_wmean` is a value-weighted mean over the cluster's
owners, NOT a cluster total. A total would track cluster size, which
`x_n_insiders` already carries. The name was changed from
`x_owner_n_prior_buys` so feature-importance output cannot be misread.

## Phase 4 note: fixed-hold capacity

`Strategy.hold_days` holds a lot for exactly N trading days and blocks the
signal-decay trim path. Stops still fire. Lot life is
`min(hold_days, exit_method.exit_days)`, so a short exit method stays a hard
ceiling across the whole grid.

Watch capacity when adding any fixed-hold strategy. The first three
hold-63 strategies inherited the `max_concurrent_tickers` values of the
short-hold strategies they mirror. Those caps assumed positions trim out in
about 7 to 10 days. A 63-day hold raises concurrent demand several times
over against an unchanged cap, so the engine drops signals by arrival order
and the strategy selects on arrival time rather than quality. Size the cap
against measured concurrent demand, and keep per-name weight times cap at or
below 1.0.

## Smoke run on real 2019 data (2026-07-30)

1,172 episodes, 630 tickers, 90 columns, built in 181s from cache with no
network. Findings:

**No leakage.** Max |Spearman rho| across every `x_` feature against every
label column is **0.186**, far under the 0.95 alarm. The strongest single
features are weak and sensible: `x_drawdown_252` +0.106, `x_vol_63_ann`
-0.104, `x_price_to_sma200` +0.098.

**The conviction score still does not rank.** Mean `adj_63` by score is
non-monotonic on **13 of 26 steps**. Score +7 gives +10.2% while score +9
gives -12.5%. This reproduces the original finding on a fresh sample the
hand-tuned weights were not built from.

**Labels are heavily right-skewed.** `adj_63` mean -0.83%, median -2.87%,
stdev 0.445, min -0.92, max **+10.39**. Mean and median disagree in size and
the max is a 1,000% outlier, so rank metrics such as Spearman IC are the
right choice and a mean-squared-error view would chase outliers.

**Four features are over half NaN**, all in the history group:
`x_owner_prior_adj63_mean` 74.3%, `x_issuer_prior_adj63_mean` 72.6%,
`x_owner_same_month_frac` 56.1%, `x_days_since_prior_cluster` 53.8%. This
is expected in a one-year sample, because most owners have no prior event
yet and the label-window gate hides recent ones. Re-check the rate on the
full 8-year build before treating any of them as broken. LightGBM handles
NaN natively, so they stay usable either way.

## Survivorship: the real number

The smoke run reported a 7.0% skip rate, but that figure is **misleadingly
low** and must not be quoted. It restricted the universe to cache-present
tickers up front, so it had already removed the missing-price population
before it measured anything.

Measured properly on the same 2019 window:

| quantity | value |
|---|---|
| distinct tickers with events | 2,705 |
| with price data | 1,633 (60.4%) |
| **no price data at all** | **1,072 (39.6%)** |
| **event rows on no-price tickers** | **28,365 of 90,823 (31.2%)** |
| episodes skipped despite having prices | 88 (7.0% of the priceable set) |

So roughly **40% of tickers and 31% of event rows are excluded outright**,
and only then does the 7% skip apply to what is left. The `delisted_63`
reprice bound moves the mean by just 0.09%, because it touches a single row.
That bound is therefore almost meaningless on its own. It measures the small
channel and misses the large one.

Every absolute return from this pipeline is an upper bound, and the gap is
wide. Rank-based conclusions survive far better than level-based ones.

## Capacity is the hidden selector (measured 2026-07-30)

Simultaneous held-ticker demand per gate, with no cap, over 2,020 trading
days. Compare each to the cap the strategy actually uses:

| gate | cap | hold | mean | median | p90 | max |
|---|---|---|---|---|---|---|
| ten_percent_owner_gated | 10 | 63d | 266 | 265 | 317 | 382 |
| ten_percent_owner_gated | 10 | 10d | **96** | 95 | 129 | 197 |
| all_clusters_equal_weight | 50 | 63d | 434 | 402 | 568 | 1004 |
| all_clusters_equal_weight | 50 | 10d | **129** | 113 | 213 | 708 |
| conviction_only | 20 | 63d | 835 | 799 | 1079 | 1537 |
| conviction_only | 20 | 10d | **268** | 255 | 422 | 1035 |

The fixed-hold variants would have dropped about 96% of signals, which is
why their caps must be resized. But the more important point is the 10-day
rows: **the existing strategies are already badly capacity-bound.**
`ten_percent_owner_gated` holds 10 names out of about 96 concurrent
candidates, so it takes roughly a tenth of its own gate.

**How the engine picks those 10 is the problem.** Step 3c queues orders by
iterating `day_states.items()`, which is plain dict order. Step 2 then
grants slots first come, first served (`engine.py`, the
`skips["capacity"]` branch). Nothing sorts by score, conviction or size. So
the surviving 10 names are an arbitrary subsample of the qualifying set,
not the best 10.

This changes how the headline result should be read. The apparent strength
of `ten_percent_owner_gated` is not evidence that the 10-percent-owner gate
selects well. It is a near-arbitrary tenth of that gate's candidates, which
is consistent with its weak `t_alpha` of 0.99. Ranking the grid by CAGR then
rewards exposure on top of that.

**Implication for the model comparison.** Capacity must bind on SCORE, not
on dict order, or a ranked strategy cannot show any benefit from ranking.
The `model_ranked_topN` strategies need the engine to sort candidates by
score and take the top N. That is a required engine change, not an
optional refinement.

Caveat on the conviction row: the measurement ran with IPO lookup restricted
to cache, which returns None on a miss, so recent-IPO names were not
filtered out. Conviction demand is therefore overstated. The direction holds
for all three gates.

## Unadjusted splits corrupt the price cache

**Corrected on 2026-07-31. The first diagnosis below was wrong. Read the
correction before you act on this section.**

The price data contains fabricated overnight jumps. Measured across the
whole cache: **668 of 4,916 tickers (13.6%)** contain a >=3x or <=1/3x
overnight close-to-close jump.

Verified on `DRIO`: close $4.14 on 2019-11-15, open $102.80 on 2019-11-18,
with volume falling from 22,420 to 3,410. That is a reverse split, not a
move. DRIO's 2019-08-29 event reported `fwd_63` of +1047%.

Measured on a 1,172-row sample: only **4 rows** had a label window crossing
a jump, but dropping those four moved mean `adj_63` from **-0.83% to
-2.41%** and cut stdev from **44.5% to 26.5%**. Three of the four were in
the top five `adj_63` values.

This matters far more than 0.34% of rows suggests, because the tail model
trains on P(`adj_63` > 0.20). Fabricated 600% to 1000% winners are exactly
the rows it would learn from.

### The original diagnosis, which was WRONG

The first theory said `prices.py` caused this when it merged a fresh fetch:
keep every old row, append only dates the old frame lacked. A split between
two fetches would then weld a stale basis onto post-split rows.

### What is actually happening

The jumps come from Yahoo, not from our merge. Proof: a single fresh
`_download_serial(['DRIO'], ...)` request reproduces the 2019-11-18 jump
exactly, at ratio 20.77. One request cannot contain a merge artifact.
Refetching `AAGR`, `ABTC` and `ACCR` whole left their jumps unchanged at
199x, 3.35x and 100x.

The root cause is an INCOMPLETE Yahoo split table. `yf.Ticker('DRIO').splits`
lists only a 2025-08-28 split. The 2019-11-18 reverse split that broke the
series is absent, so `auto_adjust` never back-adjusted the earlier prices.
`AAGR` and `ACCR` list no splits at all and still jump.

So `.splits` cannot drive a fix. The missing metadata is exactly the thing
we need. The discontinuity has to be detected from the price and volume
series itself. A split moves volume INVERSELY to price. A real move, which
microcaps do make, comes with a volume spike.

### What this means for each piece of work

- `_merge_price_frames` in `prices.py` is still correct and stays. It
  prevents a real and separate corruption, and the orchestrator verified it
  independently (18 of 18 checks, including the DRIO shape, `dollar_volume`
  invariance, forward splits and trailing stale rows). It just does not fix
  the problem above.
- `tools/repair_price_cache.py` rests on a false premise. Refetching does not
  help, because Yahoo serves the same broken series again. Six tickers were
  repaired before this was understood. That was harmless: originals are
  backed up under `price_cache/_pre_repair_backup/`, and the refetched data
  matches the old data except for one extra recent bar. **Do not run it
  against the remaining 662.**
- The real fix is self-detection and back-adjustment. See `backtest/splits.py`.

## Single-ticker fetches silently returned nothing

Found while testing the repair script. `_download_batch` and
`_download_serial` both call `yf.download(..., group_by="ticker")`, which
returns MultiIndex columns EVEN FOR ONE TICKER. `_normalize_history` then
lowercased the tuple `('AAGR', 'Open')` into a string, matched no OHLCV
name, and returned an empty frame. Confirmed on `AAPL`, so it was never
about the ticker.

This is worse than it looks. `_download_serial` is the rescue path for
every ticker that blanks in a batch, and it could never rescue anything. A
ticker that blanked went straight to "missing". This probably inflates the
**2,551 missing tickers**, and missing tickers are the main survivorship
channel.

Fixed in `_normalize_history`, which now flattens a MultiIndex by choosing
the level that holds the OHLCV names, rather than assuming an order.
Verified: single-ticker batch went from 0 rows to 42 for `AAPL`, serial
went from 0 rows to 662 for `AAGR`, multi-ticker results unchanged.

The fix did NOT recover the missing tickers. Measured after the fix:

- **48 of 48** sampled missing-list symbols still fail to fetch.
- **7 of 8** control symbols already in the cache fetch fine, including
  microcaps such as `XTNT`, `UAMY`, `WORX` and `BETR`. The eighth, `GRAL`,
  correctly returns nothing, because it spun off after the test window.

So the missing list is not an artifact of the fetch bug. Those symbols are
genuinely unavailable from Yahoo.

## Survivorship is structural, and it runs one way

The reason the missing symbols are missing is the problem. Spot checks show
they are mostly DEAD companies: `USAK` acquired in 2022, `SHOS` acquired in
2019, plus names such as `AESE`, `VLCN`, `BCEI` and `YRCW`. Yahoo drops the
history of a symbol once it stops trading.

This is the worst shape survivorship can take. The excluded set is not a
random 33.5% of the universe. It is concentrated in companies that went
bankrupt, got taken under, or delisted. Their true forward returns are
disproportionately very bad, and every one of them is absent from the
dataset.

Consequences, which apply to every result in this repo:

- **Absolute returns are upper bounds, and the margin is large.** Not a
  small correction. 31.2% of 2019 event rows sit on symbols we cannot
  price, and that set is selected for failure.
- **Rank metrics are the primary evidence.** Spearman IC, decile lift and
  precision@k are far more robust here, because the bias hits all
  strategies through the same channel.
- Free data cannot fix this. A real fix needs point-in-time data with
  delisting returns, for example CRSP, which is out of scope by decision.
- Any headline CAGR must carry this caveat. Do not quote one without it.

## The purged CV is sound, but short blocks empty a fold

Verified independently on the 1,172-row smoke dataset. Across all 5 folds:
**0 label-window violations and 0 train/test overlap.** The purge and
embargo rules in `make_purged_expanding_folds` do what they claim.

The fold table exposed a separate structural problem:

```
 id   train   test    cand  purged  embargo  test_start
  0       0    196     196     196        0          56
  1      93    195     392     299        0          98
  2     256    195     587     331        0         140
  3     402    195     782     380        0         163
  4     644    195     977     333        0         215
```

Fold 0 purged all 196 of its 196 candidate rows, and LightGBM then raised
`Input data must be 2 dimensional and non empty`.

The cause is structural. Purge drops a train row when
`entry_idx + horizon >= test_start`. With `test_start = 56` and
`horizon = 63`, that is true for every row. **Whenever the first block
spans fewer trading days than the horizon, fold 0 is empty by
construction.** The purge is correct. Nothing detected the degenerate
result.

The full dataset spans 2,155 trading days, so its blocks run near 360 days
and comfortably exceed the 63-day horizon. This mainly bites small runs,
but it must fail loudly rather than crash, and a partly-skipped run must
never look like a complete one.

## Splits: what the whole-cache scan found

`backtest/splits.py` classifies each discontinuity by whether volume moves
INVERSELY to price, which a share-count change forces, or spikes, which
news drives.

| class | discontinuities | distinct tickers |
|---|---|---|
| split | 660 | 251 |
| real_move | 687 | 359 |
| ambiguous | 2,972 | 343 |
| any | 4,319 | 668 |

`DRIO` classifies correctly on the real cache file: close_ratio 20.77,
volume_ratio 0.648, inferred ratio 20.0, confidence 0.83.

**`real_move` needs a magnitude ceiling.** Measured distribution:

- p50 4.4x, p75 9.1x, p90 42.3x, p95 100.0x, p99 301.4x, max 2,666.7x
- **151 of 687 (22.0%) exceed 10x**

A genuine 10x overnight move does not happen in a real listed equity.
`NCPL` on 2020-11-11 shows close_ratio 2,666x with volume_ratio 6.0x and
currently classifies as `real_move` at confidence 0.80. Left alone, these
fabricate exactly the tail returns the adjustment is meant to remove.

Many `ambiguous` entries carry `volume_ratio=None`. Those are near-dead
microcaps with no usable volume baseline, so the honest reading is that the
price series itself is broken.

## The label-shuffle test was never calibrated

`run_label_shuffle_test` shuffled the label with ONE seed and failed the run
when `abs(ic)` reached a hard-coded 0.10. The threshold was arbitrary.

Measured null distribution, 40 shuffle seeds, per-fold IC averaged each
time, on the 1,172-row smoke dataset:

```
mean = -0.0183   stdev = 0.0402   min = -0.0924   max = +0.0632
p5 = -0.0752  p25 = -0.0487  p50 = -0.0247  p75 = +0.0133  p95 = +0.0478
```

Findings:

1. **The test is one draw from a distribution with stdev 0.040.** The
   harness seed (12345) drew -0.1121, about -2.3 sigma. That is bad luck,
   not proof of a leak. ZERO of 40 null shuffles approached 0.10, so the
   threshold neither catches realistic leaks nor fits the data.
2. **The null centres on -0.018, not 0.** A noise-trained model shows a
   small systematic NEGATIVE correlation with the true label. Unexplained.
   Candidates: near-constant predictions where Spearman is driven by
   tie-breaking, or skew in a heavily right-tailed label.
3. NOT a pooling artifact. Per-fold mean IC (-0.1121) matches pooled IC
   (-0.1184), so measuring per fold does not change the picture.

Scripts: `tmp/shuffle_null.py`, `tmp/check_pooled_ic.py` — ad-hoc, never
committed, and no longer on disk. The numbers above are therefore a record,
not something you can re-run. Rebuild them under `tools/` if this needs
re-testing.

The real test should ask whether the TRUE model's IC is extreme against
this empirical null, not whether a noise model clears a fixed constant.

### Fixed in `research/model.py`

`run_label_shuffle_test` now takes `n_seeds` (default `DEFAULT_N_SHUFFLE_SEEDS
= 20`) independent shuffle draws, builds the empirical null of the
per-fold-averaged IC directly from them, and judges the REAL regressor's
mean per-fold IC (`real_ic`) against that null with a one-sided permutation
p-value (pass requires `p < SHUFFLE_TEST_ALPHA = 0.05`). Twenty draws is the
smallest count whose finest achievable p-value (`1 / 21 = 0.048`) still
clears 0.05. `run_label_shuffle_test` warns loudly if a caller passes fewer
seeds than that floor allows, since no real IC, however extreme, could ever
pass in that case. The function now measures IC per fold and averages,
matching the main summary table, not a pooled IC.

**The negative centring investigated (item 2 above).** Neither candidate
explanation held up:

- **Not tie-breaking.** Out-of-fold predictions from a noise-fit model have
  real spread (per-fold stdev 0.09 to 0.14) and are 100% unique per fold, so
  Spearman is not driven by duplicate ranks.
- **Not the winsorization/skew interaction.** Removing winsorization
  entirely, and separately rank-transforming the label to remove its skew
  before shuffling, left the null centred the same way.
- **New finding: it needs the real features.** Replacing FEATURE_COLS with
  pure random-noise columns of the same shape collapsed the offset (mean
  +0.0073, SE 0.0055, over 40 draws). The offset appears to come from the
  real (already-documented, non-leaky, |rho| <= 0.19) feature-label
  correlations surviving into a noise-fit model's out-of-fold predictions
  through ordinary finite-sample overfitting. Nothing about the true label
  reaches the model in this path, since the label is fully permuted before
  any fold sees it.
- **Magnitude is small and only marginal at high N.** The original 40-draw
  estimate was -0.0183. A 120-draw re-measurement narrowed it to -0.0086
  (SE 0.0039, t=-2.19, p=0.03), about 0.2x the null's own stdev. This is
  much smaller than the scale a real model needs to clear this test (order
  0.1+, several null-stdevs out).

**Verdict: benign for this test.** It does not indicate a leak. It survives
even with the true label fully permuted, and it disappears when features
carry no information at all. It is small next to the null's own spread and
far smaller than what a real edge needs to look like. It also makes the new
permutation test slightly more conservative, not less, since the null's
center sits a hair below zero rather than above it. Not fully explained
beyond "features carry real information, in-sample overfitting under
permutation partly recovers it" -- flagged here rather than papered over.

## The split adjustment works, measured

Rebuilt the 1,172-row smoke dataset with the adjustment wired in. Same
events, same window, only the price correction differs.

| | rows | mean adj_63 | median | stdev | max |
|---|---|---|---|---|---|
| before | 1172 | -0.83% | -2.87% | 44.5% | +1039% |
| after | 1170 | **-2.33%** | -2.93% | **26.7%** | **+282%** |

Build log: 30 tickers back-adjusted, 71 split / 234 ambiguous / 71
real_move discontinuities across 630 scanned tickers, and only **2 rows
dropped** for an unsafe label window.

Two independent methods agree. The earlier manual check, which simply
dropped the 4 rows whose label window crossed a jump, gave mean -2.41% and
stdev 26.5%. The principled adjustment gives -2.33% and 26.7%, and it gets
there by CORRECTING 30 tickers instead of discarding data.

The median hardly moves, which is the expected signature. The corruption
sat in the tail, not the centre. Rows above +100% fell from 9 to 7, and
the fabricated +1039% return is gone.

**A real bug reached production data here.** `apply_split_adjustments`
divided volume by the split ratio, but cached parquet files store volume as
int64, and pandas 2.x refuses to write a float back into an int column.
Every synthetic test frame built volume as float, so 19 tests passed while
the real build crashed. `tests/test_splits.py::TestIntegerVolumeColumn`
now covers the real dtype.

## Known gaps

- Concurrent-selling features blocked on the targeted scrape above.
- Survivorship bounding not yet implemented.
- No pytest config in the repo. Tests run via `python -m pytest tests/ -q`.
