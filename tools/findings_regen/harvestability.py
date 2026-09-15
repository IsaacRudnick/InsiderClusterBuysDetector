"""sharpe_robustness's checks, with x_log_adv20 attached (sh.prepare does not
carry it, so the stock CLI dies in check_liquidity)."""
import os
import sys
import pandas as pd
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from tools import sharpe_lab as sh
from tools import sharpe_robustness as sr

DATASET, SCORES = sys.argv[1], sys.argv[2]
COL = "ens"

scores = pd.read_parquet(SCORES)
df = sh.prepare(scores, DATASET, score_cols=(COL,))
base = pd.read_parquet(DATASET)
base["event_day"] = pd.to_datetime(base["event_day"])
df = df.merge(base[["ticker", "event_day", "x_log_adv20"]],
              on=["ticker", "event_day"], how="left")

spec = sh.BookSpec(score_col=COL, lo=0.70, hi=0.90, weighting="equal")
panel = sh.build_panel(df, COL)
ref = sh.panel_returns(panel, sh.BookSpec(score_col=COL, lo=0.0, hi=1.0))
spy = sh.benchmark_row(ref, "SPY")
base_m = sh.evaluate(sh.panel_returns(panel, spec), spec.label())
print(f"Book: {spec.label()}   ann {base_m['ann_return']:+.1%}  Sharpe {base_m['sharpe']:+.3f}")
print(f"SPY:  ann {spy['ann_return']:+.1%}  Sharpe {spy['sharpe']:+.3f}")
sr.check_costs(panel, spec, spy["sharpe"])
sr.check_liquidity(df, spec, COL)
sr.check_price_floor(df, spec, COL)
sr.check_per_year(panel, spec)
