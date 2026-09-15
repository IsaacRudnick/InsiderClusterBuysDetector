"""Scripts that re-derive the numbers in findings.py.

findings.py exists so every claim sits next to its source and gets
re-checked. These are the scripts that did the 2026-09-14 re-measurement.
Each one was first run on the pre-correction data and had to reproduce the
published value exactly before it was trusted on new data; the "(was ...)"
figures they print are those pre-correction values.

Run from a directory holding the data caches (price_cache/,
issuer_meta_cache/, ...), with the dataset and score paths as arguments:

  population.py    [DATASET] [EVENTS]   HORIZON_EXPECTATIONS, FLAT_REFINEMENTS,
                                        BENCHMARK_CAGR, SURVIVORSHIP
  bands.py         DATASET SCORES       BANDS, NEW_SCORE_*, ELEVATED_RISK_*
  permutation.py   DATASET SCORES       TOP_BAND_ANNUALIZED_EXCESS / _PERMUTATION_P
  book.py          DATASET SCORES       BOOK_RESULTS, *_FINAL_MULTIPLE, YEARS_BEAT_SPY
  sharpe_search.py DATASET SCORES [N]   TOP_BAND_SHARPE_*
  harvestability.py DATASET SCORES      COST_ / LIQUIDITY_ / PRICE_FLOOR_*
  tradeable.py     DATASET [SEEDS DRAWS] REFIT_TRADEABLE_UNIVERSE_*

SCORES is the parquet tools/ship_candidate.py writes with --out. The holdout
and exit-rule constants come from tools/final_search.py and
tools/settle_band.py directly; the survivorship correction from
tools/delisting_fate.py, survivorship_bound.py and survivorship_remeasure.py.
"""
