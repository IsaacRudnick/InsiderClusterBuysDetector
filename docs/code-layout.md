# Code layout

Where everything lives, and the data caches it builds. For what the project
is, see the [README](../README.md).

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

Everything in `tools/` is dual-purpose: importable as a library and runnable as
`python tools/name.py`. Python puts a script's *own* directory on
`sys.path[0]`, not the repo root, so each carries a guarded `REPO_ROOT`
bootstrap to find its siblings.

## Caches

All gitignored, all reproducible, none cheap.

| Directory | Contents | Cost to rebuild |
|---|---|---|
| `parse_cache/` | One JSON per SEC accession, over a million files | About a day per year of history |
| `price_cache/` | Per-ticker parquets, hundreds of MB | Refetched from Yahoo on demand |
| `clusters_history/` | Scraped event parquets, ~100 MB | The scrape above |
| `research_data/` | Datasets, OOF scores, fitted model bundles | Minutes, given the above |
| `ipo_cache/`, `issuer_meta_cache/` | Per-ticker / per-CIK JSON | Cheap |
| `filing_cache/`, `delist_cache/`, `short_interest_cache/`, `xbrl_cache/`, `filing_context_cache/`, `fred_cache/` | Inputs for the research tools in `tools/` | Re-run the tool that built them |
| `out/` | Reports and dashboards | Cheap |

Back up `parse_cache/` and `clusters_history/`. They are the expensive ones.
