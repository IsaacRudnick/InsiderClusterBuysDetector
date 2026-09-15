"""Sweep book recipes for risk-adjusted return, then permutation-test the sweep.

Prints three things, in this order, and the order is the point:

  1. The full grid of band x weighting x price-floor recipes, ranked by Sharpe.
  2. The benchmarks over the identical periods.
  3. A permutation test whose null is THE WHOLE GRID SEARCH -- the best Sharpe
     the same sweep finds after the scores are shuffled within each period.

Reading (1) without (3) is how this project previously convinced itself of a
+19.77%/yr result that a shuffled-score search reproduced at +18.46%.
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools import data_paths  # noqa: E402

from tools import sharpe_lab as sh  # noqa: E402

#: The recipe grid. Bands are the ones the decile evidence points at plus the
#: obvious naive cuts, so the sweep contains what a person would try by hand.
BANDS = [
    (0.00, 1.00), (0.50, 1.00), (0.60, 0.95), (0.70, 1.00),
    (0.70, 0.90), (0.75, 0.95), (0.80, 0.90), (0.80, 1.00),
    (0.90, 1.00), (0.60, 0.90), (0.50, 0.90),
]
WEIGHTINGS = ["equal", "inv_vol", "inv_var"]
PRICE_FLOORS = [0.0, 3.0]
MAX_WEIGHTS = [1.0, 0.15]


def grid(score_col: str) -> list[sh.BookSpec]:
    out = []
    for (lo, hi), w, pf, mw in itertools.product(
        BANDS, WEIGHTINGS, PRICE_FLOORS, MAX_WEIGHTS
    ):
        out.append(
            sh.BookSpec(score_col=score_col, lo=lo, hi=hi, weighting=w,
                        min_price=pf, max_weight=mw)
        )
    return out


def best_sharpe(panel: sh.PeriodPanel, scores) -> float:
    """The number the sweep would report. This is exactly what gets permuted.

    Every recipe in `grid` is evaluated and the best Sharpe returned, so the
    permutation null includes the search itself and not just its winner.
    """
    best = -np.inf
    for spec in grid("ens"):
        per = sh.panel_returns(panel, spec, scores)
        if len(per) < 20:
            continue
        m = sh.evaluate(per)
        if m and np.isfinite(m.get("sharpe", np.nan)):
            best = max(best, m["sharpe"])
    return best


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--score-col", default="ens")
    ap.add_argument("--dataset", default=data_paths.latest_research_dataset())
    ap.add_argument("--draws", type=int, default=150)
    ap.add_argument("--top", type=int, default=18)
    args = ap.parse_args(argv)

    scores = pd.read_parquet(args.scores)
    df = sh.prepare(scores, args.dataset, score_cols=(args.score_col,))
    print(f"{len(df)} scored rows, {df['period'].nunique()} periods\n")

    panel = sh.build_panel(df, args.score_col)

    # The flat-array path must agree with the pandas path exactly, or every
    # number below is measuring something other than what period_returns
    # defines. Checked once, on a real recipe, rather than assumed.
    probe = sh.BookSpec(score_col=args.score_col, lo=0.70, hi=0.90)
    a = sh.period_returns(df, probe, score_col=args.score_col)["ret"].to_numpy()
    b = sh.panel_returns(panel, probe)["ret"].to_numpy()
    assert a.shape == b.shape and np.allclose(a, b), (
        "fast panel path disagrees with the pandas reference path"
    )
    print("panel path agrees with the pandas reference path\n")

    rows = []
    for spec in grid(args.score_col):
        per = sh.panel_returns(panel, spec)
        if len(per) < 20:
            continue
        m = sh.evaluate(per, spec.label())
        if m:
            rows.append(m)
    table = pd.DataFrame(rows).sort_values("sharpe", ascending=False)

    show = ["book", "periods", "avg_names", "ann_return", "ann_vol", "sharpe",
            "sortino", "max_drawdown", "win_rate", "excess_SPY", "ir_SPY",
            "yrs_SPY", "excess_IWM", "yrs_IWM"]
    print("===== 1. RECIPE GRID, best Sharpe first =====")
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(table[show].head(args.top).to_string(index=False,
              float_format=lambda v: f"{v:+.3f}"))

    print("\n===== 2. BENCHMARKS, same periods =====")
    ref = sh.panel_returns(panel, sh.BookSpec(score_col=args.score_col,
                                              lo=0.0, hi=1.0))
    bench = pd.DataFrame([sh.benchmark_row(ref, b) for b in ("SPY", "IWM")])
    print(bench.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))

    print(f"\n===== 3. PERMUTATION TEST OF THE WHOLE SWEEP "
          f"({len(grid(args.score_col))} recipes x {args.draws} draws) =====")
    observed = float(table["sharpe"].max())
    res = sh.permutation_test_panel(
        panel, best_sharpe, observed, n_draws=args.draws
    )
    if res:
        print(f"  observed best Sharpe      {res['observed']:+.3f}")
        print(f"  null best-of-grid median  {res['null_median']:+.3f}")
        print(f"  null 95th percentile      {res['null_p95']:+.3f}")
        print(f"  null max                  {res['null_max']:+.3f}")
        print(f"  permutation p             {res['p_value']:.4f}   "
              f"({res['n_draws']} draws)")
        print("\n  A p above ~0.10 means the grid search found this on its own"
              " and the score contributed nothing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
