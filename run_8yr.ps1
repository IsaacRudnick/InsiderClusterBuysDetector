# 8-year backtest on cached events. No prompts, no scrape.
#
# 8 years is the widest cached history (clusters_history/events_20180717_20260717
# .parquet, 2018-07-17 -> 2026-07-17). A 10-year run would need a fresh scrape
# back to 2016 at roughly a day per year, and the model has no scores before
# 2018 anyway, so model_ranked_* would sit idle for the first two years.
#
# Usage:  .\run_8yr.ps1

$env:BT_MONTHS            = '96'
$env:BT_CAPITAL           = '100000'
$env:BT_RF                = 'zero'
$env:BT_OFFLINE           = '0'
$env:BT_STRATEGIES        = 'all'
$env:BT_EXITS             = 'all'
$env:BT_SLIPPAGE_BPS      = '10'
$env:BT_LIQUIDITY_FLOOR   = '500000'
$env:BT_MIN_PRICE         = '1.0'
$env:BT_COST_SWEEP        = 'none'
$env:BT_VERBOSE           = '0'
$env:BT_WRITE_WEIGHTS     = '0'
$env:BT_DROP_TICKER_REUSE = '1'

# 0 = skip fitting the learned_* strategies. They are in-sample and their
# results are not trustworthy; fitting them costs about 8 minutes. Set to '1'
# if you want them anyway.
$env:BT_FIT         = '0'
$env:BT_FIT_HORIZON = '90'

# The line that prevents an 8-day scrape.
$env:BT_EVENTS_FROM = 'clusters_history/events_20180717_20260717.parquet'

# Ranking scores. oof_scores_noreuse is the shipped model (TAIL_THRESH 0.20).
# For the better objective found on 2026-08-09 -- better median, better win
# rate, lower concentration, 5/5 folds -- swap in:
#   research_data/oof_t005_asdefault_20260809.parquet
$env:BT_MODEL_SCORES = 'research_data/oof_scores_noreuse_20260808.parquet'

# Pin the window end so the run is reproducible.
$env:BT_AS_OF = '2026-07-17'

python backtest.py
