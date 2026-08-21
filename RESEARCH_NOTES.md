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

## Review of `out/backtest_20260812_234954` (2026-08-13)

The first full 64-strategy grid since the ticker-reuse fix. 96 months,
2018-08-12 .. 2026-08-12, `fit_signal: true`, `drop_ticker_reuse: true`,
8,950 events dropped for reuse. Three things came out of it: the run's
model-ranked arm is not measuring what it claims, the shipped ranking score
is actively harmful, and the one objective that does work is not a return
objective.

### 1. The run scored itself with the wrong model, chosen by mtime

`config.json` has `"model_scores": "latest"`. `resolve_model_scores_path`
globs `oof_scores*.parquet` and takes `max(mtime)`. The four objective-sweep
files were written **within 40 ms of each other**, so the winner is
effectively arbitrary. It resolved to
`oof_scores_objsweep_thresh_0.20_20260809.parquet` -- the TAIL_THRESH 0.20
variant that the objective sweep had already found worse than 0.05 on four
of five dimensions.

Worse, the runs this grid gets compared against (`backtest_20260809_232548`,
`backtest_20260810_000228`) pointed at `oof_t005_asdefault_20260809.parquet`
explicitly. That filename does not match `oof_scores*.parquet` at all, so
`latest` could never have picked it. **The model silently changed between
the runs being compared.** Do not read the 8/12 model-ranked numbers as a
continuation of the 8/9-8/10 ones.

Fix: make `latest` refuse to guess when several candidates share a mtime
within a second, and record the resolved path plus its score column in
`config.json` rather than the spec string.

### 2. A fifth of the window had no scores at all

OOF scores cover **2020-01-02 .. 2026-05-06**. The backtest window is
2018-08-12 .. 2026-08-12. `rank_by_model_score` maps an unscored candidate
to `-inf`, which sorts it last but leaves it **eligible**, so in the 17
unscored months every candidate ties at `-inf` and `_rank_capacity_order`'s
ticker tie-break picks the book. The last three months forward-fill stale
scores. Measured share of lots in the blind region:

| strategy | pre-2020 lots | scored lots | stale-ff lots | pre-2020 mean ret |
|---|---|---|---|---|
| model_ranked_n05_hold63 | 64 (17%) | 311 | 10 | +1.98% |
| model_ranked_n10_hold63 | 128 (17%) | 622 | 18 | **-4.93%** |
| model_ranked_n10_hold21 | 405 (19%) | 1,685 | 61 | +4.69% |
| model_ranked_n25_hold63 | 346 (16%) | 1,704 | 54 | +0.30% |

So roughly a sixth of every model-ranked strategy is an alphabetical
selector wearing the model's name. Either clamp the backtest window to the
score window, or gate unscored candidates out instead of sorting them last.

### 3. The leaderboard is three trades deep

Nothing in 442 rows clears significance. Best `t_alpha` is **1.65**
(`thr_gt_p11`, 391 lots, 10% exposure) -- **down from the 1.91 the same
strategy posted a year of work ago**. The model-ranked ladder is not
monotone in slot count and straddles zero: n05 +0.65, n10 +0.36, n15 +0.14,
n20 -0.12, n25 -0.17, n30 -0.35, n40 -0.03, n50 +0.02, n60 -0.44, n75 -0.42,
n100 -0.24. That shape is noise, not a capacity curve.

P&L concentration, per `trades_*_90d.csv`:

| strategy | top-1 lot | top-10 lots | P&L ex. top 1% of lots | worst year share |
|---|---|---|---|---|
| model_ranked_n05_hold63 | 26.1% | **92.3%** | +53% | 2020 = 50.4% |
| model_ranked_n10_hold63 | 15.8% | **92.7%** | +28% | **2020 = 93.8%** |
| model_ranked_n10_hold21 | 17.9% | 71.9% | **-12.3%** | 2025 = 39.1% |
| model_ranked_n25_hold63 | 12.1% | 65.9% | **-6.8%** | 2020 = 78.6% |
| all_clusters_hold63 (baseline) | 10.4% | 44.0% | +15.0% | 2020 = 57.5% |
| thr_gt_p11 (best t_alpha) | 15.9% | 86.0% | +61% | 2024 = 51.8% |

Two of the four model-ranked strategies **lose money once the top 1% of
lots is removed**, and the unranked baseline is less concentrated than any
of them. `model_ranked_n10_hold63` earns 93.8% of its lifetime P&L in 2020
and is negative in 2021 and 2023. This is the same lottery-ticket shape as
[[tail-model-is-a-lottery-ticket]], now visible at portfolio level.

### 4. Why the score fails: it is a volatility factor with the sign flipped

Spearman of `oof_tail_classifier` against its own inputs:
`x_vol_63_ann` **+0.689**, `x_vol_21_ann` +0.666, `x_drawdown_252` -0.623,
`x_entry_vs_insider_vwap` +0.436. The score is two-thirds a realized-vol
factor. But vol's own relationship to forward return is *negative*
(`x_vol_63_ann` IC -0.060, negative in 7 of 9 years), and
`x_entry_vs_insider_vwap` is the strongest single feature in the dataset at
IC **-0.109, negative in 8 of 9 years**. The tail objective inverts both.

Result, measured against the compounding-correct label
`log(1+fwd_63) - log(1+spy_63)`:

| decile of `oof_tail_classifier` | 0 | 3 | 5 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|
| log-excess vs SPY | -0.030 | -0.010 | -0.042 | **-0.064** | **-0.064** | -0.012 |
| P(adj_63 < -30%) | 0.02 | 0.04 | 0.12 | 0.17 | 0.17 | 0.16 |

Pooled IC **-0.053**, positive in **1 of 7 years** (2020). Deciles 4-8 are
all worse than not selecting at all. Screening out its *bottom* deciles
makes results worse in 5 of 7 years. Only decile 9 recovers, and only on the
mean -- its median is still negative. **Ranking by this score selects for
blowup risk.** Every `model_ranked_*` strategy in this grid did exactly that.

The unused `oof_classifier` (P(adj_63 > 0)) is strictly better on the same
data: IC +0.029, positive in 5 of 7 years. It was never wired in.

### 5. The population itself is negative-alpha once compounding is honest

`adj_63` mean is **+1.02%**, which is the number every prior note quotes.
The same events in log space give **-3.49%** vs SPY over 63 days, median
-2.40%, and **-7.99% at 126 days**. The positive arithmetic mean is entirely
the right tail; an equal-weight portfolio does not collect it.

**No decile of any ranking tested reaches positive log-excess.** The best
cell found anywhere in this analysis is -0.012. Read every CAGR in
`summary.csv` against this: the strategies that beat SPY do it with beta
(0.73-0.99 on the model-ranked arm) plus a handful of lots, not selection.

### 6. What does work: predict the loss, not the return

Expanding-window out-of-sample (train on everything ending 95 calendar days
before the test year starts), same 59 `x_` features, LightGBM, four
objectives:

| objective | IC vs log-excess | years positive | drop-bottom-3 lift |
|---|---|---|---|
| rank(log-excess) | +0.014 | 5/6 | +0.011 |
| P(log-excess > 0) | -0.001 | 5/6 | +0.007 |
| huber on winsorized log-excess | +0.007 | 5/6 | +0.010 |
| **P(adj_63 > -0.30)** | **+0.092** | **6/6** | **+0.018** |
| shipped `oof_tail_classifier` | -0.053 | 1/7 | -0.008 |

Every return-shaped objective collapses in 2021 (IC about -0.22 in all
three). The downside-avoidance objective does not. Over the full 7-year
sample it holds IC +0.063 at 6/7 years positive, and it is **stable**:
across 4 seeds x 3 thresholds (-0.20/-0.30/-0.40) IC stays in
0.050 .. 0.066 and the drop-bottom-3 lift in +0.014 .. +0.020. Compare that
to the refit instability in [[refit-instability-decides-the-headline]].

Its decile structure is monotone in exactly the thing it was asked to
predict: P(adj_63 < -30%) runs 0.27, 0.18, 0.13, 0.13, 0.09, 0.08, 0.05,
0.05, 0.03, **0.02**. Top importances are `x_entry_vs_insider_vwap`,
`x_log_adv20`, `x_vol_63_ann`, `x_price_to_sma200`, `x_vol_21_ann`.

An unfitted, sign-constrained rank composite of eight features
(`x_entry_vs_insider_vwap`-, `x_vol_63_ann`-, `x_n_ten_pct`-,
`x_issuer_n_prior_clusters`+, `x_window_span_days`+,
`x_n_distinct_tx_dates`+, `x_owner_prior_buys_wmean`+, `x_drawdown_252`+)
matches it: IC +0.102, 6 of 7 years positive, P(<-30%) 0.27 -> 0.02, and the
lift scales correctly with horizon (+0.004 at 21d, +0.020 at 63d, +0.041 at
126d), which is the signature of a real effect rather than a fitting
artifact. Its signs were picked on this data, so treat it as an upper bound
and the fitted model as the honest number -- but a hand composite with no
parameters tying the tuned model is itself the finding.

`x_n_ten_pct` and `x_ten_pct_value_share` are both negative in 8 of 9 years.
`ten_percent_owner_gated` gates on the wrong sign, confirming
[[ten-pct-owner-ranks-negative]] with cross-year evidence.

**Survivorship cuts the right way for this conclusion.** The excluded third
of tickers is concentrated in dead companies, which would land in the bottom
deciles of a downside model. So the risk ranking is understated here, not
overstated -- the one conclusion in this repo that bias makes *safer*. The
level numbers stay upper bounds as always.

### Verdict

There is no better *return* model to be had from this feature set: four
objectives, none reaches positive log-excess in any decile. There is a
better *risk* model, it is stable, and it is not what the pipeline ships.
The honest product is a downside filter over an equal-weight book, not a
top-N ranker -- which is the portfolio shape
[[edge-exists-only-where-it-cannot-be-measured]] already pointed at from the
other direction.

Ordered next steps:

1. Make `latest` fail loudly on ambiguous mtimes; log the resolved path and
   score column into `config.json`. One-line class of bug, invalidated a
   64-strategy run.
2. Gate unscored candidates out of `model_ranked_*` instead of sorting them
   last, or clamp `BT_AS_OF`/months to the score window.
3. Refit with objective `P(adj_63 > -0.30)`, publish as a separate score
   column, and add `model_screened_*` strategies that use it as a **filter**
   (drop bottom 3 deciles, equal-weight the rest) rather than a rank_fn.
4. Report log-excess alongside `adj_63` everywhere. The arithmetic mean is
   +1.02% and the compounding truth is -3.49%; every note written so far
   quotes the flattering one.

Scripts: `tmp/an*.py` under the session scratchpad (not committed).

## backtest.bat now trains the model it backtests (2026-08-13)

Training and backtesting were two commands with nothing tying them together.
They are one command now, and the seam between them is explicit rather than
inferred from the filesystem.

```
backtest.bat
  1. run_research.py --all --months %BT_MONTHS% [--as-of ...]
        --oof-path-out <tempfile>          rebuild dataset + refit model
  2. backtest.py    with BT_MODEL_SCORES = the path stage 1 wrote
```

Three things this fixes, all of them failures the 8/12 run actually hit:

- **The handoff is a path, not a glob.** `run_research.py --oof-path-out FILE`
  writes the absolute path of the parquet it just saved; the batch file reads
  it with `set /p` and exports `BT_MODEL_SCORES`. Stage 2 can no longer
  re-resolve `latest` and land on a different model than stage 1 fit.
- **One window for both stages.** `BT_MONTHS` is asked once, up front, and
  passed to training and to the backtest. Previously `run_research.py`
  defaulted to 96 months and `backtest.py` prompted with a default of 36, so
  the two could silently disagree -- which is how a model scored 2020-2026
  ended up ranking a 2018-2026 grid.
- **`latest` refuses to guess.** `resolve_model_scores_path` now raises when
  the newest candidate is within `AMBIGUOUS_MTIME_WINDOW_S` (1.0s) of
  another, listing the tied files. The four `oof_scores_objsweep_thresh_*`
  files are 40 ms apart, so a bare `BT_MODEL_SCORES=latest` against the
  current `research_data/` now stops with an actionable message instead of
  picking one at random. Explicit paths bypass the check entirely.

`config.json` gained `model_scores_resolved`, `model_scores_column`,
`model_scores_first_day` and `model_scores_last_day`, so a finished run
records which model it ranked on. `backtest.py` also logs a MODEL-SCORE
COVERAGE GAP warning naming the offending strategies when the run window
extends past the score window in either direction.

Escape hatches, since a full `--all` re-scrapes EDGAR and takes hours:

| variable | effect |
|---|---|
| `BT_TRAIN=0` | skip stage 1, backtest against existing `BT_MODEL_SCORES` |
| `BT_TRAIN_ARGS=--fit-model --events-from latest` | refit from the cached events parquet, minutes not hours |
| `BT_MONTHS`, `BT_AS_OF` | pin the window for both stages |

## The Sharpe objective

`fit_and_validate` now fits a fourth model, `sharpe`, and emits `oof_sharpe`
alongside the existing three OOF columns.

Sharpe is a portfolio property -- mean over stdev of a return stream -- and
this model scores one event at a time, so it cannot be a per-row loss. The
translation has two halves and both are needed:

1. **Train** on `sharpe_label()`: `adj_63` divided by the event's own
   `x_vol_63_ann`, floored at `SHARPE_VOL_FLOOR = 0.10`. This is the direct
   antidote to the failure in section 4 above -- a model fit on raw `adj_63`
   is rewarded for finding big moves, big moves live in high-vol names, and
   that is exactly how `tail_classifier` ended up +0.689 correlated with
   `x_vol_63_ann` while ranking negatively against forward return. The ratio
   removes the reward channel: +20% on a quiet name now outranks +40% on a
   name twice as wild.
2. **Judge** on `portfolio_sharpe_table()`: rebuild the equal-weight top-N
   book each score would have held and measure its realized Sharpe.

The vol column is an `x_` feature, so the denominator is point-in-time by
construction and adds no information the model could not see at decision
time. The floor clips 136 rows (1.3%) and sits below the 1st percentile
(0.087) -- it catches stale/identical-close series, not genuinely quiet
names. A row with no vol reading gets a NaN label and drops out of the
sharpe fit only; it is deliberately NOT backfilled with the raw return,
which would feed the fit the very rows the objective exists to discount. A
fold with too few vol-labelled rows emits NaN and logs it, rather than a
constant that would read as "no edge" instead of "never fit".

`portfolio_sharpe_table` buckets `entry_idx` into **non-overlapping**
`horizon`-day periods. Overlapping entries reuse the same price path across
periods and inflate Sharpe; the bucketing costs sample size and buys a
number that means what it says. It reports `n_periods` next to every figure
for exactly that reason, and returns NaN rather than a finite value when
fewer than two periods survive.

### First measurement (8,810 OOF rows, groupE dataset, 27 periods)

| score | n | Sharpe | stdev | max DD |
|---|---|---|---|---|
| oof_tail_classifier | 10 | **1.102** | 0.215 | -0.328 |
| oof_regressor | 10 | 1.077 | 0.261 | -0.520 |
| oof_sharpe | 5 | 0.974 | 0.250 | -0.497 |
| oof_sharpe | 25 | 0.814 | 0.135 | -0.350 |
| oof_sharpe | 50 | 0.641 | 0.113 | -0.338 |
| oof_tail_classifier | 50 | 0.776 | 0.131 | -0.373 |
| conviction_score | 25 | 0.161 | 0.096 | -0.239 |
| ten_pct_owner | 50 | -0.011 | 0.108 | -0.556 |

**The sharpe model does not win the headline Sharpe, and 27 periods cannot
tell these apart anyway.** A gap of 0.1-0.3 in Sharpe on 27 observations is
noise. What is visible and consistent is the shape: at every book size the
sharpe model runs a LOWER stdev and a shallower drawdown than the tail
classifier at the same N (at n=100, stdev 0.089 vs 0.114 and maxDD -0.319 vs
-0.455). That is the objective doing what it was asked to do. Whether that
converts into a better book is a question for the engine, not this table.

All four models clear both hand-built baselines by a wide margin on this
axis, which is the first time any score in this repo has separated from
`conviction_score` on a risk-adjusted measure.

Caveats that still apply, unchanged: these are `adj_63` levels on the
priceable universe, so they are upper bounds per the survivorship section,
and the table has no costs, no capacity cap and no cash leg -- it ranks
scores, it does not size a strategy.

## Does a ranking exist? Yes. Does it beat an ETF? No. (2026-08-13)

An earlier version of these notes concluded there was no usable ranking, only
a good/bad filter. **That conclusion was wrong**, and the reason it was wrong
is worth recording: it came from reading pooled decile MEANS at a single
63-day horizon. On a label with a 1,000% right tail the mean is noise. Four
methodology fixes -- read MEDIANS, test 21 days, use a quantile objective,
and measure rank correlation WITHIN monthly cohorts -- surface a ranking that
was there the whole time.

### The ranking that survives

Quantile regression at alpha 0.4-0.5, 21-day horizon, expanding-window
out-of-sample, evaluated on log-excess vs SPY:

| decile | median excess | win rate |
|---|---|---|
| 0 (worst) | **-2.85%** | 43.0% |
| 4 | -0.46% | 47.5% |
| 9 (best) | **+0.09%** | 50.4% |

Monotone in median (rank corr +0.93). Positive in **7 of 7** OOS years.
Monthly-cohort IC +0.062, t=4.09, block bootstrap over months gives 95% CI
[+0.031, +0.094], P(<=0) = 0.0000. Stable across 5 seeds.

**It passes the volatility audit that killed the tail model.** Inside a vol
quintile it keeps +0.034 of its +0.056, and it beats a pure low-vol ranker
head to head (top decile +0.09% vs -0.38%). Every other candidate tested --
63d, 126d and 252d downside models, some with much larger headline IC --
FAILED exactly here: within-vol IC collapsed to zero or negative and a plain
"sort by low volatility" matched or beat them. Those were the vol factor
wearing a hat. Always run this audit before believing a new score.

Two hard limits:

- **No resolution at the top.** IC *within* the top decile is -0.0007. It
  separates top-10% from bottom-10% reliably and cannot order the top 30
  against each other.
- **It is mostly price context, not insider quality.** Dropping the 12
  price/momentum/vol features collapses it from 7/7 years to 4/7 and
  vol-neutral IC to +0.001. Top drivers are `x_mom_63_skip5`,
  `x_vol_21_ann`, `x_entry_vs_insider_vwap`, `x_price_to_sma200`.

### Benchmark choice was a real error

Insider clusters live in microcaps; the labels compare them to SPY. Over
2018-08..2026-08 the size factor alone accounts for half the apparent
underperformance:

| ETF | CAGR | vs SPY |
|---|---|---|
| SPY | +15.16% | - |
| RSP (EW S&P) | +11.82% | -3.3pp |
| IWC (microcap) | +9.42% | -5.7pp |
| IWM (smallcap) | +9.10% | -6.1pp |
| XBI (biotech) | +6.88% | -8.3pp |

Report both yardsticks, always. The universe-matched one (IWM/IWC) answers
"does the insider signal add value"; SPY answers "would you have been better
off in an index fund". Picking whichever flatters the result is benchmark
shopping.

### The ETF bar: not cleared, and the reason is instructive

Best configuration found (sector-relative training label, see below), 10-name
equal-weight book, non-overlapping 21-day periods, net of 20bps:

| test | result |
|---|---|
| headline | **+7.98%/yr over SPY** |
| p-value | **0.466** |
| 95% CI | **[-13.4%, +29.3%]** |
| beat SPY | 4 of 7 years |
| drop 2025 | +7.98% -> **+0.06%** |
| **seeds 0-4** | **+7.98, +0.81, +1.26, +6.72, +5.91** |

**The headline is the random seed.** Nothing but the RNG start moves it
tenfold. Seed 0 first reads as a triumph; seed 1 first reads as a failure.
Any future ETF-beating claim from this repo must report the seed sweep and
the leave-one-year-out table, or it is not a claim.

Same instability elsewhere: adding two beta features moved the top-5 book
from +2.33% to +15.86%/yr while moving top-10 the OTHER way (+4.70% ->
+1.74%). A real effect does not behave like that.

**Roughly 40 model configurations were tried in this effort.** That count is
itself the warning. Keep searching and something will clear +15%/yr; it will
be seed 3 of test 47 and it will mean nothing. The six sector tests below
were pre-registered before any result was seen, and all six are reported.

### Issuer reference data added: `tools/issuer_meta.py`

The dataset had 59 features and none described the company -- no industry,
no size, no listing venue. SEC serves industry free at
`data.sec.gov/submissions/CIK##########.json`, keyed on the `issuer_cik`
already in the events parquet. All **6,787 issuers fetched, 0 missing**,
cached in `issuer_meta_cache/`, written to `research_data/issuer_meta.parquet`.
92.9% of research rows get a SIC major group; 67 distinct groups (top: 60
banks 2813, 28 pharma 1476, 73 software 661).

Pre-registered results, all evaluated on the tradeable SPY-relative label:

| test | monthly IC | yrs+ | vol-neutral | top-10 vs SPY |
|---|---|---|---|---|
| T1 baseline | +0.0622 | 7/7 | +0.0336 | +4.70% |
| T2 + beta features | +0.0577 | 7/7 | +0.0113 | +1.74% |
| T3 + sector as a FEATURE | +0.0541 | 7/7 | +0.0166 | +0.81% |
| **T4 sector-relative LABEL** | **+0.0716** | **7/7** | **+0.0508** | +7.98% |
| T5 everything | +0.0377 | 6/7 | +0.0222 | -3.62% |
| T6 downside + everything | +0.0520 | 5/7 | -0.0150 | -10.76% |

**Giving the model the industry made it worse; judging each buy against its
industry made it better.** T4 has the highest vol-neutral IC ever measured
here (+0.051, a 50% improvement on baseline) -- the metric hardest to fake.
The ranking genuinely improved. The portfolio still did not survive the seed
sweep above.

Beta (`x_beta_252`, `x_idio_vol_252`, computed point-in-time from
`price_cache`) was a clean negative: beta-adjusting the label moved its
stdev from 0.1691 to 0.1693. Idiosyncratic risk dominates at these sizes;
market beta explains almost nothing. Kept in the scratch dataset, not
promoted.

### Answer: is the live scraper worth acting on?

The direct question -- does scraping the last few days/weeks of Form 4s
surface anything tradeable -- measured on all 10,905 cluster episodes:

| hold | vs SPY | vs IWM (small) | vs IWC (micro) | win rate |
|---|---|---|---|---|
| **10 days** | -7.89% | **-0.69%** | **-1.45%** | 48.2% |
| 21 days | -12.82% | -6.52% | -6.94% | 46.3% |
| 63 days | -14.17% | -10.73% | -11.45% | 43.9% |
| 252 days | -17.02% | - | - | 38.5% |

**At the shortest horizon a fresh insider cluster is worth exactly a
small-cap index fund** (-0.69% vs IWM, inside noise). Every day you hold
beyond that, you lose. There is no post-filing drift to capture -- the curve
only slopes down.

Nor does any intuitive refinement rescue it. Median 21-day excess by quartile:

- filing speed (fastest -> slowest): -0.66%, -0.69%, -1.01%, -0.94%. Reacting
  fast to a fresh filing buys nothing.
- number of insiders: -0.75%, -1.22%, -0.54%, -0.79%. No pattern.
- total dollars bought: -1.09%, -0.81%, -0.63%, -0.74%. No pattern.
- CEO share of the buying: -1.05%, -1.18%, -0.31%, -0.75%. No pattern.
- 10%-owner count: -0.92%, -0.69%, -0.94%, -0.74%. No pattern, consistent
  with [[ten-pct-owner-ranks-negative]].

**So: the scraper is not a buy signal.** The cluster-buy event itself carries
approximately zero information about forward returns. What the pipeline DOES
produce reliably is the bottom of the ranking -- the worst decile loses 2.85%
in three weeks, in every year, under every seed. Knowing which insider buys
to skip is the durable output. Knowing which to buy is not.

## Known gaps

- Concurrent-selling features blocked on the targeted scrape above.
- Survivorship bounding not yet implemented.
- No pytest config in the repo. Tests run via `python -m pytest tests/ -q`.
- Steps 2-4 of the review's next-steps list are still open: gating unscored
  candidates out of `model_ranked_*` (only warned about, not fixed), the
  `P(adj_63 > -0.30)` downside score, and reporting log-excess alongside
  `adj_63` in the engine's own output. Step 1 is done.
- `PRODUCTION_SCORE_MODEL` is still `tail_classifier`, so `--fit-production`
  bundles the score the review found harmful. Left alone deliberately:
  changing it silently reaims the live screener. Decide it on purpose.
- `portfolio_sharpe_table` gets 27 non-overlapping periods out of the 8-year
  dataset. That is enough to rank scores coarsely and not enough to call a
  0.2 Sharpe difference real.
- The 21-day quantile ranker and the sector-relative label are NOT in the
  pipeline. Both live in scratch scripts only. `research/model.py` still
  fits regressor/classifier/tail/sharpe against `adj_63`.
- `tools/issuer_meta.py` is standalone: nothing in `backtest/research.py`
  joins `issuer_meta.parquet` yet, so `sic_major` is not an available
  feature or label input inside the real pipeline.
- `exchange` from the submissions API is CURRENT, not point-in-time.
  Uplisting/delisting are exactly the events that move a stock, so it must
  not become a feature without a point-in-time source. `sic_major` is much
  safer (reassignment is rare, the 2-digit cut is coarse) but is still
  today's classification.
- Still no size/market-cap, valuation, short interest, or earnings-date
  data. Earnings proximity is the most obviously missing one for a 21-day
  horizon.
- Survivorship, measured again on the current events file: **33.2% of buy
  tickers have no price file at all, covering 24.9% of buy rows and 27.1%
  of buy dollars.** Any ETF-beating claim smaller than this bias is not
  measurable with free data.

## A ranking that beats the shipped one, and a portfolio claim that dies (2026-08-20)

**Method.** New `tools/score_lab.py` builds out-of-fold scores with one
expanding-window fold per calendar year (purge = horizon + 21-trading-day
embargo, compared on `entry_idx` so it is exact in trading days), then runs
a fixed pre-registered gauntlet: monthly-cohort IC with a block bootstrap
over months, years positive, volatility-neutral IC inside vol quintiles
head-to-head against a plain low-volatility ranker, decile MEDIANS with
crash rates, and a top-N book over non-overlapping periods.
`tools/run_score_lab.py` holds 19 pre-registered candidates; all were run
and all are reported. Grading label throughout: log excess over SPY at 21
trading days. Dataset `research_10861rows_20260813.parquet`, 9,095 scored
rows, out-of-sample years 2020-2026.

**Candidate results** (seed 0 unless noted), monthly IC / years positive /
vol-neutral IC:

| candidate | IC | yrs+ | vol-neutral | verdict |
|---|---|---|---|---|
| C1 quantile a0.45 on adj_21 (reproduces the earlier T1) | +0.0617 | 7/7 | +0.0326 | pass |
| C2 quantile on sector-and-month-relative (reproduces T4) | +0.0399 | 6/7 | +0.0246 | pass, worse than C1 |
| C3 quantile on log excess | +0.0572 | 7/7 | +0.0265 | pass |
| C4 LambdaRank, month as query group | -0.0165 | 2/7 | +0.0167 | fail |
| C5 quantile on month-relative | +0.0617 | 6/7 | +0.0377 | pass |
| C7 P(no 15% loss in 21d) | +0.0793 | 7/7 | +0.0105 | FAIL the vol audit |
| C8 P(beat SPY by 10%) = shipped score's shape | -0.0581 | 1/7 | +0.0148 | fail |
| C9 same target, plain squared-error loss | -0.0046 | 3/7 | +0.0069 | fail |
| C10 C2 minus the 12 price/momentum/vol features | -0.0038 | 3/7 | -0.0021 | fail |
| C16 quantile alpha 0.55 | +0.0211 | 5/7 | +0.0249 | fail |
| C19 quantile a0.35, month-and-vol-relative, live features only | +0.0808 | 7/7 | +0.0613 | pass |

Reference: a plain "sort by low volatility" ranker scores +0.0145
vol-neutral.

Four things fell out of this that are worth keeping:

1. Sector-relative training (the earlier T4 result) did NOT replicate.
   Month-relative alone beats it on every axis. What helps is removing the
   month; adding the industry on top hurts.
2. The quantile alpha has a monotone gradient -- 0.25 and 0.35 pass, 0.45
   is weaker, 0.55 fails. A lower alpha puts the loss on the left tail.
   The informative part of this signal is which buys go badly.
3. Learning to rank (C4) failed outright despite optimising the exact
   quantity the IC measures. It also produced the single BEST top-10 book
   number in the whole sweep (+34.7%/yr, p=0.05) while having a NEGATIVE
   IC -- a standing demonstration of why the portfolio test cannot be the
   gate.
4. C10 confirms the uncomfortable earlier finding: strip the 12
   price/momentum/vol features and the ranking is gone (IC -0.004). This
   is price context, not insider quality.

**The shipped score, graded on the same gauntlet for the first time at 21
days:** monthly IC -0.0368 (t=-1.86), positive in 2 of 7 years,
vol-neutral IC +0.0130 which is BELOW the low-volatility ranker's
+0.0145, decile monotonicity -0.81, and its crash rate P(-30% in 21d)
climbs 2.9% -> 7.1% from bottom decile to top. It ranks crash risk
upward. Confirmed again, on a new horizon and a new metric.

**The new score: C19 as a 10-seed rank-average ensemble.** Every one of
the 10 members individually scores 7/7 years positive with IC between
+0.0772 and +0.0884 and vol-neutral IC between +0.0483 and +0.0626. The
ensemble: monthly IC +0.0883, t=5.08, 95% CI [+0.0537, +0.1196], positive
in 7 of 7 years, vol-neutral IC +0.0630 (4.3x the low-vol ranker), decile
monotonicity +0.93. Two disjoint 5-seed halves rank-correlate +0.964.
This is the first score measured here that does not move on the seed.

**Decile table** (ensemble, 9,095 rows, log excess vs SPY at 21 days):

| decile | n | median | win rate | P(-30%) |
|---|---|---|---|---|
| 0 | 910 | -4.10% | 41.7% | 12.75% |
| 1 | 909 | -2.82% | 44.1% | 6.05% |
| 2 | 910 | -1.64% | 45.1% | 4.84% |
| 3 | 909 | -1.46% | 43.9% | 3.52% |
| 4 | 910 | -1.06% | 44.8% | 3.30% |
| 5 | 909 | -0.64% | 47.3% | 2.20% |
| 6 | 909 | -0.42% | 48.4% | 1.54% |
| 7 | 910 | -0.15% | 48.5% | 1.32% |
| 8 | 909 | +0.09% | 50.6% | 1.10% |
| 9 | 910 | -0.52% | 48.1% | 2.97% |

**The top decile is not the best band.** Decile 9 is worse than decile 8
on both median and crash rate, in 5 of 7 out-of-sample years. Grouped
into bands: 0-29th percentile median -2.68% / crash 7.88%; 30-69th
-0.83% / 2.64%; 70-89th -0.09% / 1.21%; 90-100th -0.52% / 2.97%. So "sort
descending, take the top N", which is what the screener does today,
lands on the wrong rows.

**The band result, and why it does not survive.** Holding the 70-90th
percentile band, equal weight, re-cut every 21 trading days,
non-overlapping, net of 20bps, returned +19.77%/yr over SPY (p=0.040,
5/7 years) and +22.34%/yr over IWM (p=0.016, 6/7 years). It does not
survive its audits:

- 12 bands were tested and the best reported. A permutation test --
  shuffle scores within each period, re-run the entire best-of-12 band
  search on the shuffled scores, 200 draws -- gives a null median of
  +18.46%/yr against the observed +19.77%/yr. **Permutation p = 0.435.**
  The band search alone manufactures this number.
- 9 of 906 positions (top 1%) produced 45.7% of gross gain; the top 3%
  produced 79.8%. Largest: CABA +246%, CRVW +167%, TKNO +164%.
- With a $3 entry-price floor the excess collapses from +19.77% to
  +5.76% (p=0.374). Most of it is sub-$3 stocks.
- Leave-one-year-out: dropping 2022 takes it from +19.77% to +9.69%/yr.
  2022 alone was +92.08%.

State plainly that the bootstrap CI over periods (95% [+2.47%, +42.63%],
P(<=0)=0.010) looks convincing and is wrong, because it prices the
sampling of periods but not the selection of the band. That is the
lesson of this section.

**What DOES survive, and is what ships.** The risk ordering. Crash rate
P(-30% in 21 days) by band, per year, bottom 30% vs top 30% of the
score: 2020 8.07 vs 6.36, 2021 7.91 vs 1.00, 2022 7.45 vs 1.18, 2023
7.31 vs 0.71, 2024 9.22 vs 0.63, 2025 7.37 vs 0.28, 2026 7.66 vs 0.00.
The bottom-30% rate sits between 7.3% and 9.2% in every single year.
Decile monotonicity in median holds at +0.93 / +0.94 / +0.93 under
entry-price floors of $0 / $3 / $5, and crash monotonicity at -0.88 /
-0.84 / -0.76. Unlike the band's return number, this is a left-tail
FREQUENCY over ~900 rows per decile, not an average dragged by a
handful of winners, which is why the price floor and the year split do
not move it.

**How this changes the product.** The live screener's score is replaced
by this ensemble, its bands are redrawn to 0-30 / 30-70 / 70-90 / 90-100
with the 70-90 band presented as the top candidates and the 90-100 band
explicitly flagged as NOT better, and no portfolio or index-beating
claim is made anywhere.

What remains open:

- No market-cap/size or earnings-date data. Earnings proximity is still
  the most obviously missing feature for a 21-day horizon.
- Survivorship still caps everything: 33.2% of tickers are unpriceable.
- The ranking is mostly price context, so it is not evidence that
  insiders pick well.
- The BACKTESTER was not re-aimed. `backtest/model_scores.py` still
  defaults to `oof_tail_classifier`, so every `model_ranked_*` strategy
  in the grid still ranks on the score this section measures at IC
  -0.0368 and 2/7 years. That is deliberate for now, not an oversight:
  those strategies are top-N books, and the top-N structure is exactly
  what the band work above shows to be the wrong shape for this signal.
  Re-pointing them would produce a new set of top-N numbers carrying the
  same best-of-N selection bias the permutation test just exposed.
  Decide it on purpose, the way `PRODUCTION_SCORE_MODEL` was decided.
- The screening ensemble is fit on the full dataset with no holdout,
  which is correct for a production artifact and means the bundle itself
  carries no out-of-sample evidence. All of it comes from the
  walk-forward folds in `tools/score_lab.py`. Anyone re-fitting on a new
  dataset should re-run `tools/ship_candidate.py` rather than assume the
  numbers above transfer.

## Risk-adjusted return: the signal is real and mostly unbuyable (2026-08-20)

**The methodological error being corrected.** The previous section graded
every candidate on rank IC, which measures ORDERING. A holder does not
experience an ordering, they experience a return stream with a
volatility and a drawdown. Measured on Sharpe, the shipped score turns
out to order risk-adjusted return strongly -- something the IC and
median-excess tables could not show. New tooling: `tools/sharpe_lab.py`
(book construction, risk metrics, and a permutation test that replays
the WHOLE recipe search), `tools/run_sharpe_lab.py`,
`tools/run_sharpe_search.py`, `tools/sharpe_robustness.py`,
`tools/tradeable_universe.py`, `tools/horizon_sweep.py`,
`tools/long_short.py`.

**Sharpe by decile of the shipped score** (equal weight, non-overlapping
21-day periods, 20bps): 0.21, 0.52, 0.26, 0.68, 0.71, 0.71, 0.81, 1.02,
1.33, 0.69. Decile 9 falls back, consistent with the previous section.

**The headline book** -- 70th-90th percentile, equal weight, 20bps, 78
non-overlapping periods, ~23 names:

| book | ann return | ann vol | Sharpe | Sortino | maxDD | win rate |
|---|---|---|---|---|---|---|
| top band 70-90 | +34.5% | 23.0% | 1.303 | 1.400 | -30.7% | 73% |
| all clusters | +23.8% | 23.5% | 0.917 | 1.124 | -32.2% | 58% |
| bottom 30% | +14.9% | 34.0% | 0.412 | 0.668 | -57.6% | 51% |
| SPY | +18.1% | 14.1% | 1.192 | 1.125 | -19.2% | 73% |
| IWM | +15.6% | 20.2% | 0.723 | 0.859 | -27.3% | 60% |

Final multiple over the 6.5-year window: 5.77x versus SPY's 2.76x. Beat
SPY in 4 of 7 years, and the wins are large while the losses are small
(2022 +56.8pp when SPY was -11.7%; 2024 +32.9pp; 2025 +19.3pp; 2026
+26.4pp; losses -2.6, -4.1, -10.8pp).

**Crucially, this one passes its permutation test.** The null re-runs
the entire 132-recipe grid search (band x weighting x price floor x
position cap) on scores shuffled within each period: null best-of-grid
median Sharpe 0.987, 95th percentile 1.236, observed 1.303, **p =
0.005** on the standalone run and **p = 0.000** over 120 draws in the
candidate sweep. Why Sharpe survives where the return claim died: a
handful of 200%+ winners inflate the mean AND the volatility, so they
cannot inflate a ratio of the two. The return search was measuring the
fat tail; the Sharpe search is not.

**Objectives aimed explicitly at risk-adjusted return did NOT beat it.**
Five new pre-registered candidates, each fit as a 10-seed ensemble and
put through its own permutation test (best-of-grid Sharpe, then p):

| candidate | Sharpe | ann return | maxDD | perm p |
|---|---|---|---|---|
| C19 (incumbent, month-and-vol-relative) | 1.303 | +34.5% | -30.7% | 0.000 |
| S2 log excess / vol, month-relative | 1.286 | +30.1% | -26.0% | 0.033 |
| S5 winsorized target | 1.264 | +33.0% | -31.0% | 0.033 |
| S1 log excess / vol | 1.224 | +50.0% | -43.2% | 0.050 |
| S3 P(beats SPY) | 1.169 | +37.7% | -33.4% | 0.092 |
| S4 P(up in absolute terms) | 1.058 | +32.9% | -35.0% | 0.267 |

Two things worth keeping: dividing the target by volatility is WORSE
than neutralising volatility through cohort demeaning (dividing creates
a heavy-tailed target the quantile loss then chases); and S1 has by far
the highest return (+50%/yr, beating IWM in 7 of 7 years) at the cost of
a -43% drawdown. Several candidates passing independently is stronger
evidence than one search passing.

**Then the implementation audit, which is where it ends.**

- COST. Sharpe by round-trip cost: 0bps 1.408, 20bps 1.303, 50bps 1.147,
  100bps 0.886, 200bps 0.365, 300bps -0.157. It stops beating SPY's
  Sharpe somewhere under 50bps. The book rebalances ~23 microcap names
  twelve times a year.
- LIQUIDITY. Capping a position at 10% of the name's 20-day dollar
  volume: $100k capital Sharpe 1.107, $1M 0.982, $5M 0.908, $25M 0.750.
  Below SPY's 1.192 at every size tested.
- PRICE FLOOR. $0 Sharpe 1.303 / excess +14.1%; $3 0.977 / +3.5%; $5
  0.991 / +3.2%; $10 1.069 / +4.0%.
- BAND EDGES. Sharpe across neighbouring cuts runs 0.94 to 1.30 with no
  spike -- a plateau, which is the one robustness check it passes
  cleanly.

**Refitting INSIDE the tradeable universe does not rescue it.** The
model was refit from scratch on each restricted universe rather than
filtering at the end, because the band cuts were calibrated on a
distribution that no longer exists once the illiquid names are removed:

| universe | rows | Sharpe | SPY Sharpe | perm p |
|---|---|---|---|---|
| unrestricted, 20bps | 10,861 | 1.220 | 1.192 | 0.025 |
| $1+, 50bps | 10,349 | 1.045 | 1.186 | 0.013 |
| $5+, 50bps | 8,595 | 0.851 | 1.204 | 0.025 |
| $5+ & $250k book, 50bps | 7,265 | 0.788 | 1.226 | 0.050 |
| $5+ & $1M book, 50bps | 5,892 | 0.650 | 1.237 | 0.263 |
| $5+ & $5M book, 75bps | 4,061 | 0.531 | 1.255 | 0.463 |
| $10+ & $5M book, 75bps | 3,435 | 0.877 | 1.297 | 0.037 |

Note the important nuance: the permutation p stays significant in most
restricted universes. The score still contains real information among
buyable names. What it does not do is produce a book that beats an
index fund there.

**Two further levers, both dead.**

- HOLDING PERIOD. On the $5+/$1M universe at 50bps, longer holds raise
  absolute Sharpe and collapse drawdown (21d Sharpe 0.650 / maxDD
  -39.1%; 63d 1.342 / -14.6%; 126d 1.360 / -3.4%) -- but SPY's Sharpe
  measured over those same longer periods rises too (1.237, 1.907,
  1.572), so excess stays negative (-0.3%, -1.7%, -3.7%) and years-beat
  falls from 4/7 to 2/7.
- LONG/SHORT. Shorting the bottom band LOSES money: -20.6%/yr
  standalone. The reason is worth recording -- the bottom band
  underperforms the market but still rises in absolute terms over this
  window, so shorting it fights the market's drift rather than
  harvesting the signal. The market-hedged variant (long 70-90, short
  SPY) does produce genuine near-zero-beta alpha unrestricted (+15.3%/yr,
  Sharpe 0.821, beta 0.081, p=0.040, 4/7 years) but is below SPY's
  Sharpe, and in the tradeable universe it returns -1.6%/yr.

**New data, tested and rejected.** Earnings proximity was built
point-in-time from SEC's submissions API (`tools/filing_calendar.py`,
`tools/earnings_features.py`, 2,828 issuers, ~498,000 10-Q/10-K/8-K
records, 5 new features, 92-98% coverage). Adding it moved monthly IC
from +0.0885 to +0.0804 and vol-neutral IC from +0.0650 to +0.0638. It
does not help. The most-flagged missing feature in this repo's own gap
list has now been built and measured, and is not the answer.

**What this means.** The score orders risk-adjusted return genuinely
and reproducibly, and the top band beat SPY on return AND Sharpe over
this window -- but only in a universe containing sub-$5, thinly traded
names, at costs no one pays, at a size close to zero. Under any
realistic constraint it lands at or below an index fund. The deliverable
is unchanged: a screen that identifies which insider cluster buys carry
the worst risk, not a portfolio that beats the market.

Closed off: objective choice, horizon, weighting, band selection,
long/short, earnings data, universe restriction. Remaining and
untested: company size/valuation from XBRL, and the survivorship bound
(33.2% of tickers unpriceable), which caps everything above.

## The execution model was rigged pessimistic; redoing it does not change the verdict (2026-08-20)

The "untradeable" call above rested on `tools/tradeable_universe.py`'s
`tradeable_mask`: a position had to fit inside 10% of a SINGLE day's
dollar volume, and every name, cheap or expensive, thin or deep, was
charged the same flat round-trip cost (20-75bps by scenario). Both were
stated as "the right way to be wrong" -- conservative on purpose. On
review that is not the same as correct, and both push the same
direction, so they deserved to be redone rather than trusted.

`tools/execution_model.py` replaces both pieces:

- **`capacity_mask`** spreads the fill over `days_to_fill` trading days
  instead of forcing it into one: `participation * ADV * days_to_fill
  >= capital / n_names`. `days_to_fill=1` reproduces the old mask's
  liquidity condition exactly; capacity scales linearly from there
  (proven in `tests/test_execution_model.py`), so 5 days is 5x the old
  capacity at the same 10% participation rate.
- **`estimated_cost_bps`** prices every row individually from
  `entry_open` and `x_log_adv20` instead of one flat number: a
  half-spread proxy that widens as price and volume fall, plus a
  square-root market-impact term (Almgren & Chriss; the Grinold & Kahn
  "cost ~ sigma * sqrt(size/ADV)" rule of thumb), calibrated so a $30
  stock on $20M ADV round-trips near 25bps and a $2 stock on $300k ADV
  round-trips near 203bps -- both inside the target bands the module's
  docstring states, fixed before any capacity table was run.
- `sharpe_lab.py` gained an optional per-row cost path
  (`PeriodPanel.cost`, `BookSpec.use_per_row_cost`) so a book can be
  charged the weighted-average of what each held name actually costs.
  Every existing call site defaults to the old flat-cost behaviour
  unchanged -- `tests/test_sharpe_lab.py`'s 18 tests pass byte-for-byte
  as before.

**Capacity table, corrected: 70-90 band, equal weight, shipped `ens`
score, real per-row costs, real multi-day fills:**

| capital | days=1 | days=3 | days=5 | days=10 | SPY Sharpe (same periods) |
|---|---|---|---|---|---|
| $100k | 0.942 | 0.878 | 0.869 | 0.868 | ~1.19 |
| $250k | 0.879 | 0.871 | 0.811 | 0.756 | ~1.19-1.21 |
| $1M | 0.670 | 0.748 | 0.634 | 0.652 | ~1.20 |
| $5M | 0.640 | 0.477 | 0.365 | 0.418 | ~1.19-1.24 |
| $25M | 0.658 | 0.565 | 0.348 | 0.175 | ~0.99-1.24 |

Sharpe stays below SPY's at every capital level and every days-to-fill
tested. More striking: giving the order MORE days to fill usually makes
the book worse, not better, especially at size ($25M: 0.658 at
days=1 falling to 0.175 at days=10). The reason is mechanical and
matches how the pieces were built: more days admits more thin names
into the eligible universe (rows surviving the mask: $25M book, 2,132
at days=1 rising to 4,621 at days=10), and those marginal names are
exactly the ones the corrected cost model prices worst -- a bigger
position against thinner volume is more market impact, not less. Error
#1 (one-day fill) was real, but fixing it does not rescue the strategy,
because error #2 (flat cost) had been masking how expensive the
marginal names actually are.

**Cost distribution, events actually held in the 70-90 band** (median,
quartiles, share over 100bps round trip -- printed so the cost model
itself can be sanity-checked rather than trusted blind):

| book | median | IQR | share > 100bps |
|---|---|---|---|
| $250k | 51bps | 18-125bps | 31% |
| $1M | 85bps | 29-211bps | 44% |
| $5M | 170bps | 58-422bps | 65% |

These medians all sit ABOVE the flat 20-75bps this project was charging
before. The old flat-cost analysis was not uniformly too harsh -- it was
too harsh on liquidity (one-day fills) and too lenient on cost (a flat
number that underprices exactly the cheap, thin names the strategy's
edge concentrates in) at the same time, and the two errors partly
canceled in the old headline numbers rather than one dominating.

**Verdict: unchanged, and now for the right reason.** The signal
survives its permutation test and the unrestricted book beats SPY on
both return and Sharpe. Under a corrected, real-fill, real-cost
execution model it still does not clear an index fund at any capital
level from $100k to $25M. The honest statement stands: the edge is real
and, on this evidence, unharvestable at the prices and depths this
market actually offers.
