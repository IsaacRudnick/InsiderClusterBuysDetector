"""Just the one scenario findings.REFIT_TRADEABLE_UNIVERSE_* describes
($5+ price floor, $1M book, 50bps), instead of tradeable_universe's full
7-scenario sweep (which would refit 6 seeds seven times over)."""
import os
import sys
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from tools import run_score_lab as lab
from tools import tradeable_universe as tu

DATASET = sys.argv[1]
SEEDS = int(sys.argv[2]) if len(sys.argv) > 2 else 6
DRAWS = int(sys.argv[3]) if len(sys.argv) > 3 else 80

df = lab.load_dataset(DATASET)
sub = df[tu.tradeable_mask(df, min_price=5.0, capital=1_000_000)]
print(f"dataset {len(df)} rows -> tradeable universe {len(sub)} rows", flush=True)
r = tu.evaluate_universe(sub, "$5+ & $1M book, 50bps", seeds=SEEDS, draws=DRAWS,
                         cost_bps=50.0, dataset=DATASET)
print()
for k, v in r.items():
    print(f"  {k:16s} {v:.4f}" if isinstance(v, float) else f"  {k:16s} {v}")
print(f"\n  REFIT_TRADEABLE_UNIVERSE_SHARPE     = {r.get('sharpe', float('nan')):.3f}   (was 0.650)")
print(f"  REFIT_TRADEABLE_UNIVERSE_SPY_SHARPE = {r.get('spy_sharpe', float('nan')):.3f}   (was 1.237)")
