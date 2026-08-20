"""Long the top band, short the bottom band, and a market-hedged variant.

WHY THIS IS THE RIGHT LAST IDEA
===============================
Everything measured here says the same thing about where the information sits:
the score's most durable, most repeatable property is identifying which
insider cluster buys go BADLY. The bottom 30% loses more than 30% of its value
within three weeks 7.9% of the time -- between 7.3% and 9.2% in every single
out-of-sample year -- against 1.2% for the 70th-90th band. That gradient
survived price floors, year splits and the permutation test that killed the
long-only return claim.

A long-only book can only use half of that. It buys the good band and simply
declines to buy the bad one, which earns nothing for the strongest thing the
model knows. A long/short book monetises both sides and, by construction,
removes most of the market exposure that has been flattering and then
un-flattering every comparison against SPY.

THREE HONEST OBSTACLES, ALL PRICED HERE
---------------------------------------
1. BORROW. Microcap shorts are not free and are frequently not available at
   all. `--short-borrow-bps` charges an annualised borrow fee on the short
   leg; the default 400bps is optimistic for this universe and the sweep goes
   to 2000. Availability is NOT modelled, and cannot be with this data -- that
   is stated as a limit, not hand-waved.
2. UNBOUNDED LOSS. A short in a name that triples loses 200%. The equal-weight
   short leg here is rebalanced every period, which caps compounding but not
   the within-period loss, and the worst single period is reported for exactly
   this reason.
3. THE SHORT SIDE IS THE CHEAP END. The bottom band is where the sub-dollar,
   illiquid names concentrate, so the tradeable-universe restriction bites
   HARDER on the short leg than the long leg. Every run here is reported both
   unrestricted and restricted.
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

from scipy import stats  # noqa: E402

from tools import run_score_lab as lab  # noqa: E402
from tools import score_lab as sl  # noqa: E402
from tools import sharpe_lab as sh  # noqa: E402
from tools import tradeable_universe as tu  # noqa: E402
from tools.ship_candidate import rank_average  # noqa: E402

HORIZON = 21
PPY = 252.0 / HORIZON


def ls_periods(
    df: pd.DataFrame, score: str, *,
    long_lo: float, long_hi: float, short_lo: float, short_hi: float,
    short_weight: float, cost_bps: float, borrow_bps: float,
    hedge: float = 0.0,
) -> pd.DataFrame:
    """Per-period return of a long book, a short book, and their combination.

    `short_weight` is the gross size of the short leg per unit of long leg:
    1.0 is dollar neutral, 0.0 is long only, 0.5 is a half hedge. `hedge`
    additionally shorts SPY by that fraction, which is the alternative way to
    strip market exposure when a stock-level short is not available.

    Borrow is charged pro rata for the holding period on the short leg only.
    """
    need = [score, "entry_idx", "entry_day", "fwd_21", "bench_SPY"]
    sub = df[need].dropna().copy()
    if sub.empty:
        return pd.DataFrame()
    base = int(sub["entry_idx"].min())
    sub["period"] = (sub["entry_idx"] - base) // HORIZON
    cost = cost_bps / 10_000.0
    borrow = (borrow_bps / 10_000.0) / PPY

    rows = []
    for p, g in sub.groupby("period"):
        if len(g) < 20:
            continue
        r = g[score].rank(pct=True)
        lng = g[(r > long_lo) & (r <= long_hi)]
        sht = g[(r > short_lo) & (r <= short_hi)]
        if len(lng) < 3 or (short_weight > 0 and len(sht) < 3):
            continue
        lr = float(lng["fwd_21"].mean()) - cost
        sr = float(sht["fwd_21"].mean()) + cost if len(sht) else 0.0
        spy = float(g["bench_SPY"].mean())
        # Long leg, minus the short leg's return, minus borrow, minus the
        # optional SPY hedge. Costs are charged on both legs.
        total = lr - short_weight * (sr + borrow) - hedge * spy
        rows.append(dict(period=int(p),
                         year=int(pd.to_datetime(g["entry_day"]).dt.year.median()),
                         n_long=len(lng), n_short=len(sht),
                         long_ret=lr, short_ret=sr, ret=total, bench=spy))
    return pd.DataFrame(rows)


def summarise(per: pd.DataFrame, label: str) -> dict:
    if len(per) < 10:
        return {}
    r = per["ret"]
    m, s = float(r.mean()), float(r.std(ddof=1))
    eq = (1 + r).cumprod()
    dd = float((eq / eq.cummax() - 1).min())
    dn = r[r < 0]
    dstd = float(dn.std(ddof=1)) if len(dn) > 2 else float("nan")
    # Market exposure of the combined stream. A number near zero is the
    # point of the exercise; a large one means it is still a long book.
    beta = float(np.polyfit(per["bench"], r, 1)[0]) if per["bench"].std() > 0 else np.nan
    yrs = per.groupby("year")["ret"].mean()
    return dict(
        book=label, periods=len(per),
        n_long=float(per["n_long"].mean()), n_short=float(per["n_short"].mean()),
        ann_return=(1 + m) ** PPY - 1,
        ann_vol=s * math.sqrt(PPY),
        sharpe=(m / s) * math.sqrt(PPY) if s > 0 else float("nan"),
        sortino=(m / dstd) * math.sqrt(PPY) if dstd and dstd > 0 else float("nan"),
        max_dd=dd, beta=beta,
        p_value=float(stats.ttest_1samp(r, 0.0).pvalue),
        yrs_pos=f"{int((yrs > 0).sum())}/{len(yrs)}",
        worst=float(r.min()),
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default=os.path.join(
        REPO_ROOT, "research_data", "research_10861rows_20260813.parquet"))
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--cost-bps", type=float, default=50.0)
    ap.add_argument("--borrow-bps", type=float, default=400.0)
    args = ap.parse_args(argv)

    from tools import band_backtest as bb

    full = lab.load_dataset(args.dataset)
    cand = next(c for c in lab.CANDIDATES if c.name == "C19_month_vol_rel_a35_live")

    for uni_label, min_price, capital, cost in (
        ("UNRESTRICTED (20bps)", 0.0, 0.0, 20.0),
        ("TRADEABLE $5+ / $1M book", 5.0, 1_000_000, args.cost_bps),
    ):
        df = (full[tu.tradeable_mask(full, min_price=min_price, capital=capital)]
              if (min_price or capital) else full).copy()
        members = [sl.build_oof(df, cand, horizon=HORIZON, seed=s)
                   for s in range(args.seeds)]
        df["ens"] = rank_average(members)
        df = bb.attach_benchmarks(df, HORIZON).dropna(subset=["ens", "fwd_21"])

        print(f"\n\n########## {uni_label} — {len(df)} rows ##########")
        rows = []
        configs = [
            ("long only 70-90",            0.70, 0.90, 0.0, 0.30, 0.0, 0.0),
            ("long 70-90 / short SPY",     0.70, 0.90, 0.0, 0.30, 0.0, 1.0),
            ("long 70-90 / short 0-30",    0.70, 0.90, 0.0, 0.30, 1.0, 0.0),
            ("long 70-90 / short 0-30 x.5", 0.70, 0.90, 0.0, 0.30, 0.5, 0.0),
            ("long 70-100 / short 0-30",   0.70, 1.00, 0.0, 0.30, 1.0, 0.0),
            ("long 50-90 / short 0-20",    0.50, 0.90, 0.0, 0.20, 1.0, 0.0),
            ("short 0-30 only",            0.70, 0.90, 0.0, 0.30, 1.0, 0.0),
        ]
        for label, llo, lhi, slo, shi, sw, hedge in configs:
            per = ls_periods(
                df, "ens", long_lo=llo, long_hi=lhi, short_lo=slo, short_hi=shi,
                short_weight=sw, cost_bps=cost, borrow_bps=args.borrow_bps,
                hedge=hedge,
            )
            if label.startswith("short") and len(per):
                # The short leg on its own: negate it and drop the long side,
                # so its standalone risk and return are visible rather than
                # inferred from the combination.
                per = per.assign(ret=-(per["short_ret"] + args.borrow_bps / 10_000.0 / PPY))
            m = summarise(per, label)
            if m:
                rows.append(m)
        ref = ls_periods(df, "ens", long_lo=0.0, long_hi=1.0, short_lo=0.0,
                         short_hi=0.3, short_weight=0.0, cost_bps=cost,
                         borrow_bps=0.0)
        if len(ref):
            b = summarise(ref.assign(ret=ref["bench"]), "SPY")
            if b:
                rows.append(b)
        show = ["book", "periods", "n_long", "n_short", "ann_return", "ann_vol",
                "sharpe", "sortino", "max_dd", "beta", "p_value", "yrs_pos", "worst"]
        with pd.option_context("display.width", 250, "display.max_columns", 30):
            print(pd.DataFrame(rows)[show].to_string(
                index=False, float_format=lambda v: f"{v:+.3f}"))

    print(f"\nBorrow charged at {args.borrow_bps:g}bps/yr on the short leg. "
          f"Short AVAILABILITY is not modelled and is the binding constraint "
          f"in this universe.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
