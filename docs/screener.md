# The live screener

Technical reference for `insider_cluster_buys.py`. For what the project is and
how to get started, see the [README](../README.md).

## Running it

```
run.bat                              # Windows: creates .venv, installs, runs
python insider_cluster_buys.py       # anywhere, inside your own virtualenv
```

It prompts for lookback days (30), minimum distinct insiders (2), and cluster
window days (14). Set `ICB_LOOKBACK`, `ICB_MIN_INSIDERS` and `ICB_WINDOW_DAYS`
to skip the prompts. That is necessary for a scheduled or piped run, where a
bare `input()` with no stdin would block forever.

`SEC_USER_AGENT` is required, in `.env` (see `.env.example`). SEC's fair-access
rules require a descriptive User-Agent with a real contact email; without one
SEC throttles or blocks you. The scraper limits itself to 8 requests/second
against SEC's published ceiling of 10.

Other variables, all optional and documented in `.env.example`:

| Variable | Meaning |
|---|---|
| `LIVE_SCORE_FETCH_PRICES` | `0` skips the price fetch for scored tickers. Faster; fewer model inputs resolve. |
| `LIVE_SCORE_MODEL_PATH` | Pin a model bundle instead of using the newest in `research_data/`. |
| `LIVE_SCORE_HISTORY_PATH` | Pin the research dataset used for score percentiles. |

## Outputs

Written to `out/`, overwritten each run:

- `dashboard.html`: self-contained, shareable as a single attachment. The full
  payload is embedded as a `<script type="application/json">` block, so the
  file is both human- and machine-readable.
- `insider_cluster_buys.xlsx`: Flagged Clusters, All Tx and Errors sheets.

To re-render the dashboard after editing CSS or legend copy, without paying for
a fresh EDGAR scan:

```
python build_html.py
```

It reads the payload back out of `out/dashboard.html` and overwrites the file.

## The model

Clusters are sorted and badged by a trained model, `research/screen_model.py`:
a 10-seed ensemble of quantile-regression rankers (alpha 0.35, so the fit
leans on the loss side), at a 21-trading-day horizon, against a target made
relative to other events in the same calendar month and volatility quintile,
using only features the screener can compute at run time. Its docstring
explains each of those choices.

Build it with:

```
python run_research.py --all          # scrape 96 months of EDGAR, build the dataset
python run_research.py --fit-screen   # writes research_data/screen_model_<nrows>rows_<YYYYMMDD>.joblib
```

The screener uses the newest `screen_model_*.joblib`. If none is present it
falls back, with a warning, to the retired single-classifier
`production_model_*.joblib`. With neither on disk every row renders as
`not_scored`, rather than falling back to the hand-tuned `conviction_score`,
which measured at zero risk-adjusted edge.

That fallback does not fail the run: the screener exits 0 with nothing scored.
Check `model_info.model_available` in the embedded payload before trusting an
unattended run.

## The verdicts

A cluster's percentile is measured against the model's own training-score
distribution. Measured out of sample on 9,371 cluster episodes, 2020-2026,
against log excess return over SPY at 21 trading days (`findings.BANDS`,
2026-09-14):

| Verdict | Percentile | Median excess | Win rate | P(loses >30% in 21 days) | Use |
|---|---|---|---|---|---|
| `top_band` | 70-90 | -0.03% | 49.7% | 1.33% | Research first. Roughly flat against SPY: a reading queue, not an expected winner. |
| `above_band` | 90-100 | -0.30% | 49.0% | 2.77% | Scored higher and measured worse than `top_band`, in 5 of 7 out-of-sample years. Research second. |
| `middle` | 30-70 | -0.70% | 46.7% | 2.43% | Low priority. |
| `elevated_risk` | 0-30 | -2.68% | 42.9% | 7.93% | Skip, or research knowing the risk. The crash rate stayed between 6.6% and 9.9% in every out-of-sample year. |
| `unavailable` | - | | | | Too many of the model's inputs could not be computed. |
| `not_scored` | - | | | | No model bundle on disk this run. |

The bands are non-monotonic on purpose: past the 90th percentile, measured risk
rises again. Sorting descending and taking the top rows lands on the wrong
names. See [findings.md](findings.md).

Every evidence claim the dashboard makes comes from `findings.py`, which keeps
each number next to its provenance so that correcting the research corrects the
product. Do not hard-code an evidence claim into a rendering function. The
table above is a snapshot of `findings.py`; when they disagree, `findings.py` is
right.

## Running it unattended

For a scheduled run (cron, a systemd timer, Task Scheduler):

```
ICB_LOOKBACK=30 ICB_MIN_INSIDERS=2 ICB_WINDOW_DAYS=14 \
    python insider_cluster_buys.py </dev/null
```

- **Give it no stdin.** If a prompt is ever added without an env override, the
  run then fails at once instead of hanging until the scheduler's timeout.
- **Check that the run was actually scored.** A missing or unloadable model
  bundle does not fail the run: it exits 0 with every row `not_scored`. Anything
  that consumes the output unattended should check
  `model_info.model_available` in the embedded payload, or a broken run reads
  as a quiet week.
- **Carry the caveat with any exported list.** A `top_band` list passed on
  without the dashboard loses its banner and legend. Put
  `findings.top_band_caveat()` in the same file.
- **Do not `source .env` from a shell.** `SEC_USER_AGENT` is an unquoted value
  with spaces, so sourcing assigns the first word and runs the second as a
  command. The screener reads `.env` itself; a wrapper does not need to.
- **Run from a checkout nothing else writes to.** The run can take the better
  part of an hour and the full parse cache is over a million files. A
  file-sync tool sharing the working tree with another machine can rewrite
  files underneath a live run, and the cache churn propagates to every peer.
- **A missing day fails the run.** A daily index that cannot be fetched after
  retries, or a lookback whose weekdays all come back empty (SEC answers a
  blocked client with 403), exits non-zero rather than publishing a report
  that is short a day.
