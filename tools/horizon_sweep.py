"""Does holding longer rescue the book from its own trading costs?

THE PROBLEM THIS ATTACKS
========================
The 21-day book rebalances twelve times a year. At 50bps round trip that is
600bps a year of cost drag, and the cost audit found the book stops beating
SPY's Sharpe somewhere under 50bps. Every restricted, realistically-costed
universe came out below SPY.

Turnover is the one lever left that acts directly on that arithmetic. A
63-day hold pays the same per-trade cost a third as often. If the score's
information decays more slowly than the cost saving accrues, a longer hold
wins; if the signal is genuinely a three-week effect, it does not. That is a
real question about the world and it has not been asked at this horizon on a
TRADEABLE universe.

WHAT IS HELD FIXED
------------------
The score is refit at each horizon (its target is horizon-specific), but the
band grid, weighting choices, permutation protocol and cost model are the same
ones every other result here went through. The comparison is horizon against
horizon, not horizon against a differently-audited alternative.

Costs are charged PER REBALANCE, so a longer horizon automatically pays less
per year -- that is the whole point, and it is why the cost number must be
per-round-trip and not annualised.
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
import sys

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools import band_backtest as bb  # noqa: E402
from tools import run_score_lab as lab  # noqa: E402
from tools import score_lab as sl  # noqa: E402
from tools import tradeable_universe as tu  # noqa: E402
from tools.ship_candidate import rank_average  # noqa: E402

BANDS = [(0.50, 0.90), (0.60, 0.95), (0.70, 0.90), (0.70, 1.00),
         (0.80, 0.90), (0.80, 1.00), (0.90, 1.00), (0.0, 1.0)]
WEIGHTINGS = ["equal", "inv_vol"]


def horizon_target(df: pd.DataFrame, horizon: int) -> pd.Series:
    """The shipped target shape, moved to `horizon`.

    Log excess over SPY, median-demeaned inside month x within-month
    volatility quintile -- identical construction to research.screen_model's
    target, just at a different holding period.
    """
    fwd = df[f"fwd_{horizon}"].astype(float)
    spy = df[f"spy_{horizon}"].astype(float)
    lx = (np.log1p(fwd.clip(lower=-0.999)) - np.log1p(spy.clip(lower=-0.999))
          ).where(fwd.notna() & spy.notna())
    return sl.cohort_demean(lx, [df["month"], lab._vol_bucket(df)])


def book_periods(
    df: pd.DataFrame, score: str, horizon: int, lo: float, hi: float,
    weighting: str, cost_bps: float,
) -> pd.DataFrame:
    """Non-overlapping `horizon`-day periods, band cut within each one."""
    need = [score, "entry_idx", "entry_day", f"fwd_{horizon}", "x_vol_63_ann"]
    sub = df[need + ["bench_SPY"]].dropna(subset=[score, f"fwd_{horizon}",
                                                  "entry_idx", "bench_SPY"]).copy()
    if sub.empty:
        return pd.DataFrame()
    base = int(sub["entry_idx"].min())
    sub["period"] = (sub["entry_idx"] - base) // horizon
    cost = cost_bps / 10_000.0
    rows = []
    for p, g in sub.groupby("period"):
        if len(g) < 10:
            continue
        r = g[score].rank(pct=True)
        pick = g[(r > lo) & (r <= hi)]
        if len(pick) < 3:
            continue
        if weighting == "equal":
            w = np.ones(len(pick))
        else:
            v = pd.to_numeric(pick["x_vol_63_ann"], errors="coerce")
            v = v.fillna(v.median() if np.isfinite(v.median()) else 1.0).clip(lower=0.05)
            w = 1.0 / v.to_numpy()
        w = w / w.sum()
        rows.append(dict(
            period=int(p),
            year=int(pd.to_datetime(g["entry_day"]).dt.year.median()),
            n_held=len(pick),
            ret=float((pick[f"fwd_{horizon}"].to_numpy() * w).sum()) - cost,
            bench=float(g["bench_SPY"].mean()),
        ))
    return pd.DataFrame(rows)


def summarise(per: pd.DataFrame, horizon: int) -> dict:
    if len(per) < 8:
        return {}
    ppy = 252.0 / horizon
    m, s = per["ret"].mean(), per["ret"].std(ddof=1)
    eq = (1 + per["ret"]).cumprod()
    dd = float((eq / eq.cummax() - 1).min())
    bm, bs = per["bench"].mean(), per["bench"].std(ddof=1)
    ex = per["ret"] - per["bench"]
    yrs = per.assign(e=ex).groupby("year")["e"].mean()
    return dict(
        periods=len(per), avg_names=float(per["n_held"].mean()),
        ann_return=(1 + m) ** ppy - 1,
        ann_vol=s * math.sqrt(ppy),
        sharpe=(m / s) * math.sqrt(ppy) if s > 0 else float("nan"),
        max_dd=dd,
        spy_sharpe=(bm / bs) * math.sqrt(ppy) if bs > 0 else float("nan"),
        spy_ann=(1 + bm) ** ppy - 1,
        excess=(1 + ex.mean()) ** ppy - 1,
        yrs_beat=f"{int((yrs > 0).sum())}/{len(yrs)}",
        cost_drag_per_yr=None,
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default=os.path.join(
        REPO_ROOT, "research_data", "research_10861rows_20260813.parquet"))
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--min-price", type=float, default=5.0)
    ap.add_argument("--capital", type=float, default=1_000_000)
    ap.add_argument("--cost-bps", type=float, default=50.0)
    args = ap.parse_args(argv)

    full = lab.load_dataset(args.dataset)
    mask = tu.tradeable_mask(full, min_price=args.min_price,
                             capital=args.capital)
    df = full[mask].copy()
    print(f"tradeable universe: {len(df)} of {len(full)} rows "
          f"(>= ${args.min_price:g}, ${args.capital:,.0f} book), "
          f"{args.cost_bps:g}bps per rebalance\n")

    rows = []
    for horizon in (21, 63, 126, 252):
        if f"fwd_{horizon}" not in df.columns:
            continue
        cand = sl.Candidate(
            name=f"h{horizon}", target=lambda d, h=horizon: horizon_target(d, h),
            objective="quantile", alpha=0.35,
            features=lab.f_live_only,
        )
        members = [sl.build_oof(df, cand, horizon=horizon, seed=s)
                   for s in range(args.seeds)]
        work = df.copy()
        work[f"s{horizon}"] = rank_average(members)
        work = bb.attach_benchmarks(work, horizon)

        best = None
        for (lo, hi), w in itertools.product(BANDS, WEIGHTINGS):
            per = book_periods(work, f"s{horizon}", horizon, lo, hi, w,
                               args.cost_bps)
            m = summarise(per, horizon)
            if not m or not np.isfinite(m.get("sharpe", np.nan)):
                continue
            m.update(horizon=horizon, band=f"{lo:.0%}-{hi:.0%}", weighting=w)
            if best is None or m["sharpe"] > best["sharpe"]:
                best = m
        # The whole-universe hold at the same horizon, as the do-nothing
        # control: it says how much of any Sharpe is selection and how much is
        # simply owning insider-cluster names for longer.
        allper = book_periods(work, f"s{horizon}", horizon, 0.0, 1.0, "equal",
                              args.cost_bps)
        allm = summarise(allper, horizon)
        if best:
            best["all_names_sharpe"] = allm.get("sharpe", float("nan"))
            rows.append(best)
            print(f"  h={horizon:3d}: best Sharpe {best['sharpe']:+.3f} "
                  f"({best['band']} {best['weighting']}) vs SPY "
                  f"{best['spy_sharpe']:+.3f}  |  hold-everything "
                  f"{allm.get('sharpe', float('nan')):+.3f}", flush=True)

    print("\n\n===== HOLDING PERIOD SWEEP, TRADEABLE UNIVERSE =====")
    show = ["horizon", "band", "weighting", "periods", "avg_names",
            "ann_return", "ann_vol", "sharpe", "spy_sharpe", "all_names_sharpe",
            "max_dd", "excess", "yrs_beat"]
    table = pd.DataFrame(rows)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(table[show].to_string(index=False,
                                    float_format=lambda v: f"{v:+.3f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
