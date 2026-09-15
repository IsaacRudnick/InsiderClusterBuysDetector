"""Sweep exit rules over the top band, and report the whole table.

The question: on a payoff this skewed, does cutting losers fast and letting
winners run beat selling everything on a 21-day clock?

The sweep is a search, so the whole grid is printed and the winner is
compared against the fixed-hold baseline that every other result in this repo
used. A rule is only interesting if it improves BOTH the per-slot return and
the crash rate, or improves one without wrecking the other.
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

from tools import exit_lab as el  # noqa: E402

#: Pre-registered exit grid. The first entry is the incumbent: no stops, sell
#: on the 21-day clock. Everything else is measured against it.
RULES = [
    el.ExitRule(0.00, 0.00, 21),                       # the incumbent
    el.ExitRule(0.00, 0.00, 63),
    el.ExitRule(0.00, 0.00, 126),
    el.ExitRule(0.00, 0.00, 252),
]
for hard, trail, hold, arm in itertools.product(
    (0.00, 0.10, 0.15, 0.20, 0.30),
    (0.00, 0.15, 0.20, 0.25, 0.30, 0.40),
    (63, 126, 252),
    (0.00, 0.10),
):
    if trail == 0.0 and hard == 0.0:
        continue          # already covered by the plain time-exit rows
    if trail == 0.0 and arm > 0:
        continue          # arming level is meaningless with no trail
    RULES.append(el.ExitRule(hard, trail, hold, arm))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--score-col", default="ens")
    ap.add_argument("--dataset", default=data_paths.latest_research_dataset())
    ap.add_argument("--lo", type=float, default=0.70)
    ap.add_argument("--hi", type=float, default=0.90)
    ap.add_argument("--cost-bps", type=float, default=20.0)
    ap.add_argument("--min-price", type=float, default=0.0)
    ap.add_argument("--slots", type=int, default=20)
    args = ap.parse_args(argv)

    scores = pd.read_parquet(args.scores)
    scores["event_day"] = pd.to_datetime(scores["event_day"])
    base = pd.read_parquet(args.dataset)
    base["event_day"] = pd.to_datetime(base["event_day"])
    # Pull across only what the scores file is missing. Merging a column that
    # already exists would suffix both copies and silently break the dropna
    # below rather than raising anywhere useful.
    want = ["entry_day", "entry_idx", "entry_open"]
    missing = [c for c in want if c not in scores.columns]
    df = scores.merge(
        base[["ticker", "event_day"] + missing],
        on=["ticker", "event_day"], how="left",
    ).dropna(subset=[args.score_col, "entry_idx", "entry_day"])
    if args.min_price > 0:
        df = df[df["entry_open"] >= args.min_price]

    # The band is cut WITHIN each 21-day batch of entries, the same way every
    # other result here cuts it -- a globally-ranked cut would use knowledge of
    # where a name sits in a distribution that includes the next four years.
    df["batch"] = (df["entry_idx"] - df["entry_idx"].min()) // 21
    r = df.groupby("batch")[args.score_col].rank(pct=True)
    band = df[(r > args.lo) & (r <= args.hi)].copy()
    band["score"] = band[args.score_col]
    print(f"{len(df)} scored events, {len(band)} in the "
          f"{args.lo:.0%}-{args.hi:.0%} band, cost {args.cost_bps:g}bps, "
          f"min price ${args.min_price:g}\n")

    # Extract every trade's forward path once, then evaluate each rule over
    # the matrices. Re-reading price files per rule made the sweep unrunnable.
    max_days = max(r.max_hold for r in RULES)
    paths = el.build_paths(band, max_days)
    print(f"  built paths for {len(paths.entry)} trades, "
          f"{max_days} days forward\n", flush=True)

    rows = []
    for rule in RULES:
        trades = el.simulate_paths(paths, rule, cost_bps=args.cost_bps)
        s = el.summarise_trades(trades, rule.label())
        if not s:
            continue
        port = el.slot_portfolio(trades, n_slots=args.slots)
        if len(port) > 50:
            d = port["ret"]
            eq = (1 + d).cumprod()
            s["port_ann"] = float(eq.iloc[-1] ** (252 / len(d)) - 1)
            s["port_sharpe"] = float(
                d.mean() / d.std(ddof=1) * np.sqrt(252)
            ) if d.std(ddof=1) > 0 else float("nan")
            s["port_maxdd"] = float((eq / eq.cummax() - 1).min())
        rows.append(s)
        print(f"  {rule.label():28s} n={s['n']:5d} med={s['median']:+.3f} "
              f"days={s['avg_days']:5.1f} ann/slot={s['ann_per_slot']:+.1%} "
              f"bigwin={s['p_big_win']:.3f} crash={s['p_crash']:.3f}",
              flush=True)

    table = pd.DataFrame(rows)
    show = ["rule", "n", "median", "mean", "win_rate", "avg_days",
            "p_big_win", "p_crash", "ann_per_slot", "port_ann", "port_sharpe",
            "port_maxdd", "stopped_hard", "stopped_trail"]
    show = [c for c in show if c in table.columns]

    print("\n\n===== EXIT SWEEP, ranked by return per slot-year =====")
    with pd.option_context("display.width", 260, "display.max_columns", 40):
        print(table.sort_values("ann_per_slot", ascending=False)[show]
              .to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

    inc = table[table["rule"] == RULES[0].label()]
    if len(inc):
        print("\n===== THE INCUMBENT, for comparison =====")
        print(inc[show].to_string(index=False,
                                  float_format=lambda v: f"{v:+.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
