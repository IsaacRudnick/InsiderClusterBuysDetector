# Insider Cluster-Buy Detector & Backtester

Two programs over one signal: **clusters of SEC Form 4 open-market insider
purchases** — at least N distinct reporting owners buying the same issuer
inside a rolling window.

| | What it does | Entry point |
|---|---|---|
| **Live screener** | Scans recent EDGAR filings, scores today's clusters, writes a shareable dashboard | `run.bat` → `insider_cluster_buys.py` |
| **Backtester** | Fits the ranking model, then runs 63 strategies × 7 exit methods over history | `backtest.bat` → `run_research.py` + `backtest.py` |

Informational tooling, not financial advice.

**The headline, up front: the ranking is now real, but it is still not an
index-beating claim.** Monthly-cohort rank IC +0.0883 (t=5.08), positive in 7
of 7 out-of-sample years, volatility-neutral IC +0.0630 against +0.0145 for a
plain sort-by-low-volatility ranker, decile monotonicity +0.93, and two
disjoint 5-seed halves that rank-correlate +0.964 — this is the first score
measured here that does not move on the RNG seed. But a 70-90th percentile
band book that measured +19.77%/yr over SPY failed a permutation test at
p = 0.435: re-running the same 12-band search on shuffled scores produces the
same +18.46%/yr. What survived is the risk ordering, not a return. Read
[Findings](#findings-read-this-before-trusting-a-number) before you believe any
number this produces.

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
to skip the prompts — necessary for a scheduled or piped run, where a bare
`input()` with no stdin would block forever.

Writes to `out/`, overwritten each run:

- `dashboard.html` — self-contained, shareable as a single attachment. The
  full payload is embedded as a `<script type="application/json">` block, so
  the file is both human- and machine-readable.
- `insider_cluster_buys.xlsx` — Flagged Clusters / All Tx / Errors sheets.

Clusters are sorted and badged by the **trained model** — no longer the old
hand-tuned `conviction_score`, which measured at zero risk-adjusted edge, and
no longer `oof_tail_classifier`, which ranked crash risk upward (see
Findings). The score is `research/screen_model.py`: a 10-seed ensemble of
quantile-regression rankers, built with `python run_research.py --fit-screen`,
which writes `research_data/screen_model_<nrows>rows_<YYYYMMDD>.joblib`. The
screener prefers that bundle and falls back to the older single-classifier
`production_model_*.joblib` with a warning if it is missing. With neither on
disk, rows render as `not_scored` rather than falling back to a score that
does not work.

| Verdict | Percentile | Meaning |
|---|---|---|
| `top_band` | 70-90 | The best-measured band. Median -0.09% vs SPY, 49.5% win rate, 1.21% chance of losing more than 30% in 21 days. This is the top-candidates list. |
| `above_band` | 90-100 | Scored higher, and measured WORSE than `top_band` on both median (-0.52%) and crash rate (2.97%), in 5 of 7 out-of-sample years. A higher score is not a better candidate. |
| `middle` | 30-70 | Median -0.83%, crash rate 2.64%. |
| `elevated_risk` | 0-30 | Median -2.68%, 43.6% win rate, and a 7.88% chance of losing more than 30% in three weeks — between 7.3% and 9.2% in every single out-of-sample year. The most durable result this project has. |
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

1. **`run_research.py --all`** — rebuilds the event dataset from EDGAR and
   refits the ranking model, writing `research_data/oof_scores_*.parquet`.
2. **`backtest.py`** — the strategy grid, ranking on the scores stage 1 just
   wrote.

Stage 1 hands stage 2 the exact parquet it produced via a pointer file, rather
than letting `backtest.py` re-resolve `latest` by mtime — that glob's tiebreak
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
ranked crash risk upward — see Findings) and is kept only for reproducing old
results; the live screener no longer prefers it.

Row counts between `--build-dataset` and `--fit-model` are not 1:1. The fit
stage drops rows with no usable label or entry index, then cuts the remainder
into purged folds. That gap is expected.

### Strategies

63 registered in `backtest/strategies.py`, in families:

- **Hand-built gates** — `all_clusters_equal_weight`, `conviction_only`,
  `ten_percent_owner_gated`, `big_money_director`, `low_vol_conviction`, and
  friends, plus stop-loss and fixed-63-day-hold variants.
- **`thr_gt_*`** — equal-weight above each `conviction_score` threshold, −5
  through +13. A sweep, not 19 ideas.
- **`learned_*`** — six strategies fit inside the backtest. **In-sample; do
  not trust their results.** Fitting costs about 8 minutes; `BT_FIT=0` skips
  them.
- **`model_ranked_*`** — the real ones. Rank by the trained model's OOF score
  and hold the top N (5 … 100) for a fixed 63, 21, or 10 trading days.

Each runs as an independent portfolio against 7 exit methods: fixed 30/90/180/
365-day holds, and 365-day-capped trailing stops at 10/20/30% per FIFO lot.
SPY buy-and-hold is the benchmark.

Note that the exit grid barely binds. Over 99% of exits are *trims* triggered
by signal decay, not expiries — median hold is 7–10 days even under the "365d"
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
| `BT_SLIPPAGE_BPS`, `BT_LIQUIDITY_FLOOR`, `BT_MIN_PRICE` | Trading frictions. Keep `BT_MIN_PRICE` — see Findings. |
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

Every candidate is reported, pass or fail — the 19 in `tools/run_score_lab.py`
are all there is, and adding a 20th means re-running and re-reporting the
whole list, not quoting the new one alone.

---

## Layout

```
backtest.py                 entry point — the strategy grid
run_research.py             entry point — dataset build + model fit
insider_cluster_buys.py     entry point — the live EDGAR screener
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
  repair_price_cache        (see Findings — rests on a false premise)
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

tests/                      pytest; `python -m pytest tests -q`
```

Everything in `tools/` is dual-purpose — importable as a library and runnable
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
  flagged cluster returns −0.69%/yr against a small-cap index fund —
  indistinguishable from just owning the index. Held longer it does *worse*:
  −6.5%/yr at 21 days, −10.7%/yr at 63. The curve only slopes down, so there is
  no post-filing drift to capture.
- **The old shipped score ranked crash risk upward, on every horizon it was
  graded at.** At 63 days: pooled rank IC −0.053, positive in only 1 of 7
  years, top decile ~6x the 30%-loss rate of the bottom decile. An earlier
  volatility-matched test found +4.74pp (p=0.004) for that same top decile;
  both can be true — fat right tail *and* fat left tail — but the encouraging
  half alone is misleading. Graded again at 21 days on the pre-registered
  gauntlet built for its replacement: monthly IC −0.0368, positive in 2 of 7
  years, volatility-neutral IC +0.0130 — below a plain sort-by-low-volatility
  ranker. Replaced throughout the codebase.
- **None of the obvious quality filters work.** More insiders, bigger dollar
  amounts, CEO share of buying, ten-percent-owner involvement, and reacting
  faster to a fresh filing were each tested by quartile. All flat.
- **Survivorship bias caps everything above.** 33.2% of tickers are
  unpriceable — they stopped trading and the provider deleted them — along with
  24.9% of buy rows and 27.1% of buy dollars. Those rows are *dropped*, not
  zeroed, so every absolute return is optimistic by an unknown margin.
- **Half the "insiders pick badly" result is the size factor**, not insider
  skill: small caps trailed SPY by ~6pp/yr over the measured window. Both SPY
  and IWM/IWC yardsticks are kept for that reason — reporting only the
  flattering one is benchmark shopping.
- **A ranking that holds up now ships to the live screener.** The 21-day,
  quantile-regression ensemble in `research/screen_model.py` gives monotone
  deciles, positive in 7 of 7 years, and survives a volatility-neutral audit
  that four higher-headline candidates failed. The backtester's own model,
  `research/model.py`, still fits against `adj_63`.
- **The top of the ranking is not the best part of it.** Decile 9 is worse
  than decile 8 on both median and crash rate, in 5 of 7 out-of-sample years.
  Sorting descending and taking the top N — what the old score's verdicts
  did — lands on the wrong rows.
- **Sector-relative training, an earlier finding, did not replicate.**
  Month-relative alone beats it on every axis of the gauntlet.
- **Strip the 12 price/momentum/volatility features and the ranking
  disappears.** IC −0.004. This is price context, not evidence that insiders
  pick well.
- **A best-of-N band search on a fat-tailed return distribution manufactures
  a return from noise.** The 70-90th percentile book's +19.77%/yr over SPY
  produced a permutation-test null median of +18.46%/yr (p = 0.435) when the
  same 12-band search ran on shuffled scores. Any band or top-N return figure
  from this repo needs a permutation test, not just a bootstrap — the
  bootstrap prices the sampling of periods but not the selection of the band.
- **`conviction_score` does not rank.** No monotonicity; score −1 beats +9 and
  +10. Retained for comparison only, and gone from the dashboard.
- **Sub-dollar lots and unadjusted splits used to decide the leaderboard.**
  0.7% of lots once produced 58% of grid P&L, and a single unadjusted reverse
  split gave a "winning" strategy 83% of its P&L. `BT_MIN_PRICE` and
  `backtest/split_fingerprint.py` exist because of this. Audit P&L
  concentration before believing any ranking.
- **Refit instability decides the headline.** 1.07% fewer training rows moved a
  flagship result from +229.6% to +151.0%. Only the 5-slot variant survived.
- **The `learned_*` strategies are in-sample.** They are in the grid for
  completeness. Their numbers are not evidence.

Numbers above are sourced from `findings.py` (provenance:
`research_groupE_10905rows_20260809.parquet`, 10,905 cluster episodes,
2018-08…2026-08, expanding-window out-of-sample, measured 2026-08-13) and
`RESEARCH_NOTES.md`. The `screen_model.py` gauntlet and band numbers are from
`research_10861rows_20260813.parquet`, 9,095 scored cluster episodes,
out-of-sample 2020-2026, measured 2026-08-20 — `RESEARCH_NOTES.md`, "A ranking
that beats the shipped one, and a portfolio claim that dies".
