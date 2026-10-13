# The backtester and the research procedure

This page is the technical reference for `backtest.bat`, `run_research.py` and
`backtest.py`. For an overview of the project and its results, refer to the
[README](../README.md).

## The two stages

```
backtest.bat
```

`backtest.bat` asks one time for the history window in months. The default is
96. Then it runs two stages with that window:

1. **`run_research.py --all`** makes the event dataset again from EDGAR. Then
   it fits the model again and writes `research_data/oof_scores_*.parquet`.
2. **`backtest.py`** runs the strategy grid. It ranks on the scores that stage
   1 wrote.

Stage 1 gives stage 2 the exact parquet file through a pointer file.
`backtest.py` does not find the latest file by its modification time. When
two score files have the same time, that method selects one at random. It
selected the wrong model in the past.

> **NOTE:** A full `--all` run downloads EDGAR data again. It takes
> **hours**: approximately one day for each year of history. The parse step
> uses up to approximately 10 GB of memory. On a computer with 16 GB of
> memory, enable swap space.

To fit the model again from the cached events file in a few minutes, type:

```
set BT_TRAIN_ARGS=--fit-model --events-from latest
backtest.bat
```

To skip stage 1, set `BT_TRAIN=0`. The backtester then uses the scores that
`BT_MODEL_SCORES` points to.

## The research procedure without the backtester

```
python run_research.py --build-dataset    # research_*.parquet: one row for each cluster episode
python run_research.py --fit-model        # oof_scores_*.parquet: purged walk-forward CV
python run_research.py --all              # the two steps above, in sequence
python run_research.py --fit-screen       # screen_model_*.joblib: the score that the screener uses
python run_research.py --fit-production   # production_model_*.joblib: the retired single-classifier bundle
python run_research.py --dry-run          # shows row counts and paths, and writes no files
```

`--all` does not include `--fit-screen` and `--fit-production`. These two
options make files for the screener, not for the research. You must select
them yourself.

`--fit-production` makes the retired single-classifier bundle
(`oof_tail_classifier`). That score gave its highest values to the stocks
with the most crash risk. Refer to [findings.md](findings.md). Use it only to make old results
again. The screener does not use it if a `screen_model_*.joblib` is
available.

The row counts of `--build-dataset` and `--fit-model` are different. The fit
stage removes rows with no usable label or entry index. Then it divides the
other rows into purged folds. The model uses the first fold only to learn,
so that fold has no scores. This difference is correct.

## Strategies

`backtest/strategies.py` contains 63 strategies in four groups:

- **Hand-built gates:** `all_clusters_equal_weight`, `conviction_only`,
  `ten_percent_owner_gated`, `big_money_director`, `low_vol_conviction` and
  others. There are also stop-loss variants and fixed 63-day hold variants.
- **`thr_gt_*`:** Equal weight above each `conviction_score` threshold, from
  -5 to +13. This group is one sweep, not 19 different ideas.
- **`learned_*`:** Six strategies that the backtest fits during the run.
  **These strategies are in-sample. Do not trust their results.** The fit
  takes approximately 8 minutes. To skip it, set `BT_FIT=0`.
- **`model_ranked_*`:** The strategies that matter. Each one ranks by the
  out-of-fold score of the model. It holds the top N (5 to 100) for a fixed
  63, 21 or 10 trading days.

Each strategy is an independent portfolio. The backtester tests it with 7
exit methods:

- A fixed hold of 30, 90, 180 or 365 days.
- A trailing stop of 10%, 20% or 30% for each FIFO lot, with a maximum of 365
  days.

The benchmark is a purchase of SPY that the strategy holds for the full
period.

> **NOTE:** The exit method seldom has an effect. More than 99% of exits are
> *trims* that occur when the signal decreases. They are not expiries. The
> median hold is 7 to 10 days, also with the 365-day method. The fixed-hold
> strategies are there to correct this.

## Environment variables

`backtest.py` reads these `BT_*` variables:

| Variable | Function |
|---|---|
| `BT_MONTHS` | The history window. `backtest.bat` asks for it one time and gives it to the two stages. |
| `BT_AS_OF` | The last day of the window, as `YYYY-MM-DD`. Set this value, or you cannot do the same run again. |
| `BT_EVENTS_FROM` | The path to an events parquet file, or `latest`. This skips the download. |
| `BT_MODEL_SCORES` | The path to an out-of-fold scores parquet file, or `latest`. |
| `BT_CAPITAL` | The capital at the start of each strategy. |
| `BT_STRATEGIES`, `BT_EXITS` | Lists with commas between the items, or `all`. |
| `BT_RF` | The risk-free rate: `zero` or `tbills`. |
| `BT_OFFLINE` | `0` or `1`. With `1`, the backtester uses only the cache and no network. |
| `BT_SLIPPAGE_BPS`, `BT_LIQUIDITY_FLOOR`, `BT_MIN_PRICE` | Trade costs and limits. Keep `BT_MIN_PRICE`. Refer to [findings.md](findings.md). |
| `BT_COST_SWEEP` | `none` or `low_med_high`. |
| `BT_FIT`, `BT_FIT_HORIZON` | Fit the in-sample `learned_*` strategies at a horizon of 30, 90, 180 or 365 days. |
| `BT_DROP_TICKER_REUSE` | The default is `1`. This removes events whose ticker later went to a different company. |
| `BT_WRITE_WEIGHTS`, `BT_VERBOSE` | Diagnostic output. |

`backtest.bat` also reads `BT_TRAIN` and `BT_TRAIN_ARGS`. Refer to
[The two stages](#the-two-stages).

The backtester writes its output in `out/backtest_<timestamp>/`. The output
contains `report.html`, and an `equity_*.csv` file and a `trades_*.csv` file
for each strategy and exit method. A full grid run makes hundreds of MB.

## The candidate sweep

```
python tools/run_score_lab.py --seeds 0,1,2      # each pre-registered candidate
python tools/ship_candidate.py --seeds 10        # a full audit of the selected score
python tools/band_robustness.py                  # the audits that a band claim must pass
```

The project reports each candidate, also the candidates that fail. The 19
candidates in `tools/run_score_lab.py` are the full list. If you add a 20th
candidate, run and report the full list again. Do not report only the new
candidate.

`tools/findings_regen/` contains the scripts that calculate each number in
`findings.py` again. Each script first had to give the published value from
the old data. Only then did the project use it on the new data.
