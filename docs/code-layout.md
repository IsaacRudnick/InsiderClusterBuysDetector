# Code layout

This page shows the location of each module and each data cache. For an
overview of the project, refer to the [README](../README.md).

```
insider_cluster_buys.py     entry point, the live EDGAR screener
run_research.py             entry point, dataset build + model fit
backtest.py                 entry point, the strategy grid
run.bat, backtest.bat       Windows launchers for the screener and the backtester
build_html.py               dashboard renderer (one consumer: the screener)
findings.py                 every measured claim the product states, next to its
                            provenance. Correcting the research here corrects
                            the dashboard and the workbook. Never inline a
                            number into a rendering function.
ipo_lookup.py               first-trade dates; shared by the screener AND backtest/
render_readme.py            README.md -> README.html, for reading offline

backtest/                   the backtester's own modules
  engine, strategies, metrics, report, state, history, prices, splits,
  research, model_scores, signal_fit, legend, sales_history, tickers,
  ticker_reuse, price_overrides, split_fingerprint

research/                   the ranking model
  model.py                  fit_and_validate, purged walk-forward CV
  live_score.py             scores one live cluster from a production bundle
  screen_model.py           the score the live screener sorts by

tools/                      standalone CLIs, each `python tools/<name>.py`
  data_paths                resolves the newest dataset / events file by the
                            date in its name; every tool defaults through it
  warm_prices               bulk-fill the price cache
  repair_price_cache        (see RESEARCH_NOTES.md; rests on a false premise)
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
  final_search              the 7,560-configuration search and its
                            pre-registered holdout
  settle_band               band vs no band on the holdout, sweeping the slot
                            count. The band does not help; the exit rule does.
  findings_regen/           scripts that re-derive each number in findings.py

tests/                      pytest; `python -m pytest tests -q`
  test_findings.py          guards the claims the product makes about itself.
                            Not that the numbers are right, which only the
                            research can say, but that every headline number
                            travels with the limit that applies to it.

docs/                       the technical reference this page belongs to
RESEARCH_NOTES.md           the lab notebook, in the order things happened
```

You can import each module in `tools/` as a library. You can also run it as
`python tools/name.py`. Python puts the directory of the script in
`sys.path[0]`, not the root of the repository. Thus, each tool contains a
`REPO_ROOT` bootstrap that finds the other modules.

## Caches

Git ignores all of these caches. You can make each cache again, but this
takes time.

| Directory | Contents | Time to make again |
|---|---|---|
| `parse_cache/` | One JSON file for each SEC accession, more than one million files | Approximately one day for each year of history |
| `price_cache/` | One parquet file for each ticker, hundreds of MB | The project downloads prices from Yahoo again when necessary |
| `clusters_history/` | Event parquet files from the EDGAR download, approximately 100 MB | The same as `parse_cache/` |
| `research_data/` | Datasets, out-of-fold scores and model bundles | Minutes, if the caches above are available |
| `ipo_cache/`, `issuer_meta_cache/` | One JSON file for each ticker or CIK | Short |
| `filing_cache/`, `delist_cache/`, `short_interest_cache/`, `xbrl_cache/`, `filing_context_cache/`, `fred_cache/` | Inputs for the research tools in `tools/` | Run the tool that made the cache |
| `out/` | Reports and dashboards | Short |

Make a backup of `parse_cache/` and `clusters_history/`. These two caches take
the most time to make again.
