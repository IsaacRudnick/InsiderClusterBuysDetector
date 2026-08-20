"""Everything that could make the risk-adjusted result untrue, tested one at a
time.

The Sharpe result differs from the return result this project already retired:
its permutation test passes. That earns it a serious audit rather than a
dismissal, and a serious audit is harder than a dismissal.

  1. COST. The book rebalances ~23 microcap names every 21 trading days. 20bps
     round trip is the repo's convention and is optimistic for this universe.
     Reported here as a BREAKEVEN: how expensive can trading get before the
     book stops beating SPY's Sharpe? A strategy is only as good as the
     cost assumption it needs.
  2. LIQUIDITY. A position is capped at a share of the name's 20-day dollar
     volume; anything that cannot be filled is dropped. If the result needs
     names you cannot actually buy, it is not a result.
  3. PRICE FLOOR. Sub-dollar quotes are mostly bid-ask bounce. The return
     version of this work lost two thirds of its edge at a $3 floor.
  4. BAND EDGES. Is the chosen band a plateau or a spike? A spike means the
     grid found a coincidence.
  5. PER YEAR. Sharpe and return year by year, and the same for SPY.
  6. SPLIT-HALF SEEDS. Two disjoint 5-seed ensembles, same recipe.
  7. DRAWDOWN PATH. The worst stretch, in full, because a Sharpe of 1.3 with a
     -30% drawdown is a different product from a Sharpe of 1.3 with -12%.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools import sharpe_lab as sh  # noqa: E402

PPY = sh.PERIODS_PER_YEAR


def _fmt(v: float, pct: bool = False) -> str:
    if not np.isfinite(v):
        return "  n/a"
    return f"{v * 100:+7.1f}%" if pct else f"{v:+7.3f}"


def check_costs(panel: sh.PeriodPanel, spec: sh.BookSpec, spy_sharpe: float):
    """Sharpe as a function of round-trip cost, and where it stops beating SPY."""
    print("\n===== 1. COST =====")
    print("  round trip   ann return     Sharpe   vs SPY Sharpe")
    breakeven = None
    for bps in (0, 20, 50, 75, 100, 150, 200, 300):
        per = sh.panel_returns(panel, sh.BookSpec(**{**spec.__dict__, "cost_bps": bps}))
        m = sh.evaluate(per)
        if not m:
            continue
        beats = m["sharpe"] >= spy_sharpe
        if breakeven is None and not beats:
            breakeven = bps
        print(f"  {bps:5d} bps   {_fmt(m['ann_return'], True)}   "
              f"{_fmt(m['sharpe'])}   {'beats' if beats else 'below'}")
    if breakeven is None:
        print("  Still beats SPY's Sharpe at 300bps a side.")
    else:
        print(f"  Stops beating SPY's Sharpe somewhere under {breakeven}bps.")


def check_liquidity(df: pd.DataFrame, spec: sh.BookSpec, score_col: str):
    """Drop names too thin to absorb a position of a given size.

    `x_log_adv20` is log(1 + 20-day average dollar volume). A book of `capital`
    spread over `n` names needs capital/n dollars in each; requiring that to be
    at most `participation` of one day's volume is a crude but honest filter,
    and crude in the conservative direction since a real order would be worked
    over several days.
    """
    print("\n===== 2. LIQUIDITY =====")
    print("  A position must be <= 10% of the name's 20-day average dollar volume.")
    print("  capital     avg names     ann return     Sharpe")
    adv = np.expm1(pd.to_numeric(df["x_log_adv20"], errors="coerce"))
    for capital in (100_000, 1_000_000, 5_000_000, 25_000_000):
        need = capital / 25.0          # rough per-name dollars at ~25 names
        ok = (adv * 0.10) >= need
        sub = df[ok.fillna(False)]
        if len(sub) < 500:
            print(f"  ${capital:>10,}  too few investable names")
            continue
        panel = sh.build_panel(sub, score_col)
        m = sh.evaluate(sh.panel_returns(panel, spec))
        if m:
            print(f"  ${capital:>10,}  {m['avg_names']:9.1f}   "
                  f"{_fmt(m['ann_return'], True)}   {_fmt(m['sharpe'])}")


def check_price_floor(df: pd.DataFrame, spec: sh.BookSpec, score_col: str):
    print("\n===== 3. PRICE FLOOR =====")
    print("  floor   avg names     ann return     Sharpe    maxDD   vs SPY")
    for floor in (0.0, 1.0, 3.0, 5.0, 10.0):
        panel = sh.build_panel(df, score_col)
        per = sh.panel_returns(
            panel, sh.BookSpec(**{**spec.__dict__, "min_price": floor})
        )
        m = sh.evaluate(per)
        if not m:
            print(f"  ${floor:<5.0f} too few periods")
            continue
        print(f"  ${floor:<5.0f} {m['avg_names']:9.1f}   "
              f"{_fmt(m['ann_return'], True)}   {_fmt(m['sharpe'])}  "
              f"{_fmt(m['max_drawdown'], True)}  {_fmt(m['excess_SPY'], True)}")


def check_band_edges(panel: sh.PeriodPanel, spec: sh.BookSpec):
    """Sharpe across neighbouring band edges. A plateau is real; a spike is not."""
    print("\n===== 4. BAND EDGES =====")
    los = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    his = [0.85, 0.90, 0.95, 1.00]
    print("        " + "".join(f"  hi={h:.2f}" for h in his))
    for lo in los:
        cells = []
        for hi in his:
            if hi <= lo:
                cells.append("      -")
                continue
            m = sh.evaluate(sh.panel_returns(
                panel, sh.BookSpec(**{**spec.__dict__, "lo": lo, "hi": hi})
            ))
            cells.append(f"{m['sharpe']:+7.2f}" if m else "      -")
        print(f"  lo={lo:.2f}" + "".join(cells))


def check_per_year(panel: sh.PeriodPanel, spec: sh.BookSpec):
    print("\n===== 5. PER YEAR =====")
    per = sh.panel_returns(panel, spec)
    print("  year   periods   book ann   book Sharpe    SPY ann   excess")
    for y, g in per.groupby("year"):
        if len(g) < 4:
            continue
        m, s = g["ret"].mean(), g["ret"].std(ddof=1)
        sharpe = (m / s) * math.sqrt(PPY) if s > 0 else float("nan")
        print(f"  {y}   {len(g):7d}   "
              f"{_fmt((1 + m) ** PPY - 1, True)}   {_fmt(sharpe)}    "
              f"{_fmt((1 + g['bench_SPY'].mean()) ** PPY - 1, True)}  "
              f"{_fmt((1 + (g['ret'] - g['bench_SPY']).mean()) ** PPY - 1, True)}")


def check_split_half(df: pd.DataFrame, members: dict, spec: sh.BookSpec):
    print("\n===== 6. SPLIT-HALF SEEDS =====")
    from tools.ship_candidate import rank_average

    for label, keys in (("seeds 0-4", range(5)), ("seeds 5-9", range(5, 10))):
        work = df.copy()
        work["_half"] = rank_average([members[k] for k in keys])
        work = work.dropna(subset=["_half"])
        m = sh.evaluate(sh.panel_returns(sh.build_panel(work, "_half"), spec))
        if m:
            print(f"  {label}: ann {_fmt(m['ann_return'], True)}  "
                  f"Sharpe {_fmt(m['sharpe'])}  "
                  f"maxDD {_fmt(m['max_drawdown'], True)}  "
                  f"vs SPY {_fmt(m['excess_SPY'], True)}")


def check_drawdown(panel: sh.PeriodPanel, spec: sh.BookSpec):
    print("\n===== 7. DRAWDOWN PATH =====")
    per = sh.panel_returns(panel, spec)
    eq = (1 + per["ret"]).cumprod()
    dd = eq / eq.cummax() - 1.0
    trough = int(dd.idxmin())
    print(f"  worst drawdown {dd.min() * 100:+.1f}%, "
          f"trough in {int(per.loc[trough, 'year'])}")
    print(f"  worst single period {per['ret'].min() * 100:+.1f}%, "
          f"best {per['ret'].max() * 100:+.1f}%")
    print(f"  periods below water: {int((dd < -0.05).sum())} of {len(dd)}")
    spy_eq = (1 + per["bench_SPY"]).cumprod()
    spy_dd = (spy_eq / spy_eq.cummax() - 1.0).min()
    print(f"  SPY over the same periods: worst drawdown {spy_dd * 100:+.1f}%")
    print(f"  final multiple: book {eq.iloc[-1]:.2f}x, SPY {spy_eq.iloc[-1]:.2f}x")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--score-col", required=True)
    ap.add_argument("--dataset", default=os.path.join(
        REPO_ROOT, "research_data", "research_10861rows_20260813.parquet"))
    ap.add_argument("--lo", type=float, default=0.70)
    ap.add_argument("--hi", type=float, default=0.90)
    ap.add_argument("--weighting", default="equal")
    ap.add_argument("--members", default="", help="parquet of per-seed members")
    args = ap.parse_args(argv)

    scores = pd.read_parquet(args.scores)
    df = sh.prepare(scores, args.dataset, score_cols=(args.score_col,))
    spec = sh.BookSpec(score_col=args.score_col, lo=args.lo, hi=args.hi,
                       weighting=args.weighting)
    panel = sh.build_panel(df, args.score_col)

    ref = sh.panel_returns(panel, sh.BookSpec(score_col=args.score_col,
                                              lo=0.0, hi=1.0))
    spy = sh.benchmark_row(ref, "SPY")
    base = sh.evaluate(sh.panel_returns(panel, spec), spec.label())
    print(f"Book under test: {args.score_col}  {spec.label()}")
    print(f"  ann {_fmt(base['ann_return'], True)}  vol {_fmt(base['ann_vol'], True)}  "
          f"Sharpe {_fmt(base['sharpe'])}  Sortino {_fmt(base['sortino'])}  "
          f"maxDD {_fmt(base['max_drawdown'], True)}")
    print(f"  SPY: ann {_fmt(spy['ann_return'], True)}  Sharpe {_fmt(spy['sharpe'])}"
          f"  maxDD {_fmt(spy['max_drawdown'], True)}")

    check_costs(panel, spec, spy["sharpe"])
    check_liquidity(df, spec, args.score_col)
    check_price_floor(df, spec, args.score_col)
    check_band_edges(panel, spec)
    check_per_year(panel, spec)
    check_drawdown(panel, spec)

    if args.members and os.path.exists(args.members):
        mem = pd.read_parquet(args.members)
        members = {i: mem[f"member_{i}"] for i in range(10)
                   if f"member_{i}" in mem.columns}
        if len(members) == 10:
            check_split_half(df, members, spec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
