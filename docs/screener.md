# The screener

This page is the technical reference for `insider_cluster_buys.py`. For an
overview of the project, refer to the [README](../README.md).

## Start the screener

```
run.bat                              # Windows: makes .venv, installs the packages, starts the screener
python insider_cluster_buys.py       # All systems: use your own virtual environment
```

The screener asks for three values:

| Value | Default |
|---|---|
| The number of days to examine (lookback) | 30 |
| The minimum number of different insiders in a cluster | 2 |
| The cluster window, in days | 14 |

To skip these questions, set `ICB_LOOKBACK`, `ICB_MIN_INSIDERS` and
`ICB_WINDOW_DAYS`. A scheduled run must set these variables. Without them,
`input()` waits for input that does not come, and the run does not stop.

Set `SEC_USER_AGENT` in `.env`. Refer to `.env.example`. The SEC fair-access
rules tell each program to send a User-Agent with a name and a real email
address. Without it, the SEC can slow or block your requests. The screener
sends a maximum of 8 requests each second. The SEC limit is 10.

These variables are optional. `.env.example` also describes them.

| Variable | Function |
|---|---|
| `LIVE_SCORE_FETCH_PRICES` | Set to `0` to skip the price download for scored tickers. The run is faster, but fewer inputs are available to the model. |
| `LIVE_SCORE_MODEL_PATH` | Use this model bundle, not the newest bundle in `research_data/`. |
| `LIVE_SCORE_HISTORY_PATH` | Use this research dataset for the score percentiles, not the newest dataset. |

## Output files

The screener writes these files in `out/`. Each run replaces them.

- `dashboard.html`: One self-contained file. You can send it as one
  attachment. It contains all of the data in a
  `<script type="application/json">` block. Thus, people and programs can
  both read it.
- `insider_cluster_buys.xlsx`: A workbook with three sheets. The sheets are
  Flagged Clusters, All Tx and Errors.

After you change the CSS or the legend text, you can make the dashboard again
without a new EDGAR scan. Type this command:

```
python build_html.py
```

This command reads the data from `out/dashboard.html`. Then it writes the file
again.

## The model

The screener uses the model in `research/screen_model.py` to sort and label
the clusters. The model has these properties:

- It is an ensemble of 10 quantile-regression rankers. Each ranker has a
  different random seed.
- The quantile is 0.35. Thus, the model gives more weight to losses.
- The horizon is 21 trading days.
- The target is the return relative to other events in the same calendar
  month and the same volatility quintile.
- The model uses only the features that the screener can calculate during a
  live run.

The docstring of `research/screen_model.py` gives the reason for each
property.

To make the model, type these commands:

```
python run_research.py --all          # downloads 96 months of EDGAR data and makes the dataset
python run_research.py --fit-screen   # writes research_data/screen_model_<nrows>rows_<YYYYMMDD>.joblib
```

The screener uses the newest `screen_model_*.joblib`. If this file is not
available, the screener uses the retired `production_model_*.joblib` and
shows a warning. If neither file is available, the screener shows each row as
`not_scored`. It does not use `conviction_score`, because that score has no
risk-adjusted edge.

> **CAUTION:** If the model is not available, the run does not fail. The
> screener exits with code 0 and does not score a cluster. Before you use the
> result of an unattended run, examine `model_info.model_available` in the
> embedded data.

## The verdicts

The percentile of a cluster compares its score with the scores that the model
gave to the historical data. The research measured each band out of sample, on 9,371
cluster episodes from 2020 to 2026. The return is the log excess return over
SPY at 21 trading days. The values come from `findings.BANDS`, measured on
2026-09-14.

| Verdict | Percentile | Median excess | Win rate | P(loss > 30% in 21 days) | Use |
|---|---|---|---|---|---|
| `top_band` | 70-90 | -0.03% | 49.7% | 1.33% | Examine first. The median is approximately equal to SPY. Use this band as a list to read, not as a list to buy. |
| `above_band` | 90-100 | -0.30% | 49.0% | 2.77% | Examine second. The score is higher, but the results were worse than `top_band` in 5 of 7 out-of-sample years. |
| `middle` | 30-70 | -0.70% | 46.7% | 2.43% | Low priority. |
| `elevated_risk` | 0-30 | -2.68% | 42.9% | 7.93% | Ignore, or examine carefully. In each out-of-sample year, the crash rate was between 6.6% and 9.9%. |
| `unavailable` | - | | | | The model could not calculate too many of its inputs. |
| `not_scored` | - | | | | No model bundle was available for this run. |

The risk does not decrease in the same order as the bands. Above the 90th
percentile, the measured risk increases again. Thus, do not sort the list by
score and take the top rows. Refer to [findings.md](findings.md).

Each claim on the dashboard comes from `findings.py`. This file keeps each
number next to its source. When the research corrects a number in
`findings.py`, the dashboard also changes. Do not write a claim directly into
a function that makes HTML.

> **NOTE:** The table on this page is a copy of the values in `findings.py`.
> If the two are different, `findings.py` is correct.

## Unattended runs

For a scheduled run, use this command. The scheduler can be cron, a systemd
timer or Windows Task Scheduler.

```
ICB_LOOKBACK=30 ICB_MIN_INSIDERS=2 ICB_WINDOW_DAYS=14 \
    python insider_cluster_buys.py </dev/null
```

Obey these rules:

- **Give the screener no stdin.** If a new question has no environment
  variable, the run then stops immediately. It does not wait until the
  scheduler stops it.
- **Make sure that the run scored the clusters.** If the model is not
  available, the screener exits with code 0 and shows each row as `not_scored`. A program
  that uses the output must examine `model_info.model_available`. If it does
  not, a failed run looks like a week with no results.
- **Send the caveat with each exported list.** A `top_band` list without the
  dashboard does not have the banner and the legend. Put the text from
  `findings.top_band_caveat()` in the same file.
- **Do not use `source .env` in a shell.** The value of `SEC_USER_AGENT`
  contains spaces and has no quotes. Thus, the shell sets the first word as the
  value and runs the second word as a command. The screener reads `.env`
  itself, so a wrapper script does not have to read it.
- **Run the screener from a copy of the repository that no other program
  changes.** A run can take almost one hour, and the full parse cache contains
  more than one million files. A file-sync program can change files during a
  run. It can also copy the cache changes to each other computer.
- **Expect a failure if the data for a day is not available.** The run fails if the screener
  cannot download a daily index after it tries again. The run also fails if
  each weekday in the period gives an empty index. The SEC sends a 403
  response to a blocked program, and the screener reads that as a holiday.
  Thus, the screener does not publish a report that does not include each
  day.
