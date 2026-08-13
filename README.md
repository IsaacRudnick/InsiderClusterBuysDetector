# Insider Cluster-Buy Detector & Backtester

Two programs over one signal: **clusters of SEC Form 4 open-market insider
purchases** — at least N distinct reporting owners buying the same issuer
inside a rolling window.

| | What it does | Entry point |
|---|---|---|
| **Live screener** | Scans recent EDGAR filings, scores today's clusters, writes a shareable dashboard | `run.bat` → `insider_cluster_buys.py` |
| **Backtester** | Fits the ranking model, then runs 63 strategies × 7 exit methods over history | `backtest.bat` → `run_research.py` + `backtest.py` |

Informational tooling, not financial advice. Read
[Findings](#findings-read-this-before-trusting-a-number) before you believe
any number this produces.

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

Clusters are sorted and badged by the **trained model**, not by the old
hand-tuned `conviction_score`, which measured at zero risk-adjusted edge.
Scoring needs a production bundle on disk (`run_research.py --fit-production`,
below); with no bundle, rows render as `not_scored` rather than falling back
to a score that does not work.

| Verdict | Meaning |
|---|---|
| `top_decile` | Percentile ≥ 90. The only state with a measured, volatility-matched edge. |
| `no_edge` | Scored, below the top decile. Not "bad" — just no measured edge. |
| `unavailable` | Too few features resolved to score it. |
| `not_scored` | No production bundle was on disk this run. |

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
python run_research.py --fit-production   # production_model_*.joblib, for the live screener
python run_research.py --dry-run          # row counts and planned paths, writes nothing
```

`--fit-production` is opt-in and deliberately not part of `--all`: it is a
deployment artifact, not a research one.

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

---

## Layout

```
backtest.py                 entry point — the strategy grid
run_research.py             entry point — dataset build + model fit
insider_cluster_buys.py     entry point — the live EDGAR screener
build_html.py               dashboard renderer (one consumer: the screener)
ipo_lookup.py               first-trade dates; shared by the screener AND backtest/

backtest/                   the backtester's own modules
  engine, strategies, metrics, report, state, history, prices, splits,
  research, model_scores, signal_fit, legend, sales_history, tickers,
  ticker_reuse, price_overrides, split_fingerprint

research/                   the ranking model
  model.py                  fit_and_validate, purged walk-forward CV
  live_score.py             scores one live cluster from a production bundle

tools/                      standalone CLIs, each `python tools/<name>.py`
  warm_prices               bulk-fill the price cache
  repair_price_cache        (see Findings — rests on a false premise)
  issuer_meta               fetch SIC / exchange per issuer CIK from SEC
  ensemble_model            blend model objectives
  refit_stability           re-fit under perturbation, measure retention
  objective_sweep           sweep TAIL_THRESH and objective choice

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

- **Survivorship bias is the biggest validity problem.** Tickers with no price
  data are *dropped*, not zeroed. `n_skipped_no_price` runs to 37,956 lots
  against 48,557 kept. The missing third is concentrated in the delisted
  micro-caps that insider clusters favour, so it is not a random third. Every
  absolute return is an upper bound.
- **Raw top-decile lift is volatility, not skill.** Real alpha (+5.97pp,
  positive in 5 of 5 folds) appears only once picks are matched to the
  benchmark on risk.
- **Alpha decays to zero by 25 slots**, and shorter holds are worse. The edge
  exists only in the most concentrated configurations — which is to say,
  mostly where it cannot be measured with confidence.
- **`conviction_score` does not rank.** No monotonicity; score −1 beats +9 and
  +10. It is retained for comparison only, and is gone from the dashboard.
- **The `ten_percent_owner` flag is significantly harmful** (t_alpha −2.75),
  which answers the question the project was started to ask.
- **Sub-dollar lots and unadjusted splits used to decide the leaderboard.**
  0.7% of lots once produced 58% of grid P&L, and a single unadjusted reverse
  split gave a "winning" strategy 83% of its P&L. `BT_MIN_PRICE` and
  `backtest/split_fingerprint.py` exist because of this. Audit P&L
  concentration before believing any ranking.
- **Refit instability decides the headline.** 1.07% fewer training rows moved
  a flagship result from +229.6% to +151.0%. Only the 5-slot variant survived
  the refit.
- **The `learned_*` strategies are in-sample.** They are in the grid for
  completeness. Their numbers are not evidence.
