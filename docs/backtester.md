# The backtester and research pipeline

Technical reference for `backtest.bat`, `run_research.py` and `backtest.py`.
For what the project is and what it found, see the [README](../README.md).

## The two stages

```
backtest.bat
```

Asks once for the history window in months (default 96), then runs two stages
with that same window:

1. **`run_research.py --all`** rebuilds the event dataset from EDGAR and
   refits the ranking model, writing `research_data/oof_scores_*.parquet`.
2. **`backtest.py`** runs the strategy grid, ranking on the scores stage 1 just
   wrote.

Stage 1 hands stage 2 the exact parquet it produced through a pointer file,
rather than letting `backtest.py` re-resolve `latest` by mtime. That glob's
tiebreak is arbitrary when several score files share a timestamp, and it has
silently selected the wrong model before.

A full `--all` run re-scrapes EDGAR and takes **hours**, roughly a day per year
of history. A full rebuild's parse peaks near 10 GB of memory, so a 16 GB
machine needs swap. To refit from the cached events parquet in minutes instead:

```
set BT_TRAIN_ARGS=--fit-model --events-from latest
backtest.bat
```

Or skip training entirely with `BT_TRAIN=0` and backtest against whatever
`BT_MODEL_SCORES` already points at.

## Research pipeline on its own

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
ranked crash risk upward; see [findings.md](findings.md)) and is kept only for
reproducing old results. The live screener no longer prefers it.

Row counts between `--build-dataset` and `--fit-model` are not 1:1. The fit
stage drops rows with no usable label or entry index, then cuts the remainder
into purged folds whose earliest block is training-only. That gap is expected.

## Strategies

63 are registered in `backtest/strategies.py`, in families:

- **Hand-built gates:** `all_clusters_equal_weight`, `conviction_only`,
  `ten_percent_owner_gated`, `big_money_director`, `low_vol_conviction` and
  friends, plus stop-loss and fixed-63-day-hold variants.
- **`thr_gt_*`:** equal weight above each `conviction_score` threshold, -5
  through +13. A sweep, not 19 ideas.
- **`learned_*`:** six strategies fit inside the backtest. **In-sample; do not
  trust their results.** Fitting costs about 8 minutes; `BT_FIT=0` skips them.
- **`model_ranked_*`:** the real ones. Rank by the trained model's
  out-of-fold score and hold the top N (5 to 100) for a fixed 63, 21 or 10
  trading days.

Each runs as an independent portfolio against 7 exit methods: fixed 30, 90,
180 and 365-day holds, and 365-day-capped trailing stops at 10, 20 and 30% per
FIFO lot. SPY buy-and-hold is the benchmark.

The exit grid barely binds. Over 99% of exits are *trims* triggered by signal
decay, not expiries; the median hold is 7-10 days even under the "365d"
method. The fixed-hold strategies exist to address this.

## Environment variables

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
| `BT_SLIPPAGE_BPS`, `BT_LIQUIDITY_FLOOR`, `BT_MIN_PRICE` | Trading frictions. Keep `BT_MIN_PRICE`; see [findings.md](findings.md). |
| `BT_COST_SWEEP` | `none` or `low_med_high`. |
| `BT_FIT`, `BT_FIT_HORIZON` | Fit the in-sample `learned_*` strategies (30/90/180/365). |
| `BT_DROP_TICKER_REUSE` | Default `1`. Drop events whose ticker later changed hands to an unrelated company. |
| `BT_WRITE_WEIGHTS`, `BT_VERBOSE` | Diagnostics. |

`backtest.bat` also reads `BT_TRAIN` and `BT_TRAIN_ARGS` (above).

Output lands in `out/backtest_<timestamp>/`: `report.html` plus `equity_*.csv`
and `trades_*.csv` per strategy and exit pair. A full grid run is hundreds of
MB.

## Re-running the candidate sweep

```
python tools/run_score_lab.py --seeds 0,1,2      # every pre-registered candidate
python tools/ship_candidate.py --seeds 10        # the chosen score, deep audit
python tools/band_robustness.py                  # the audits that kill a band claim
```

Every candidate is reported, pass or fail. The 19 in `tools/run_score_lab.py`
are all there is, and adding a 20th means re-running and re-reporting the whole
list, not quoting the new one alone.

`tools/findings_regen/` holds the scripts that re-derive each number in
`findings.py`. Each was required to reproduce the published value on the old
data before being trusted on new data.
