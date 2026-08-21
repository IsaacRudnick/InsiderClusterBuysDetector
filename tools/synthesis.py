"""Put the two independent advances together and mark the result daily.

TWO THINGS CHANGED AT ONCE, AND THEY ARE INDEPENDENT
====================================================
1. THE EVENT. The inherited definition (">=2 distinct insiders inside a
   rolling 14-day window") had never been swept. Across 384 definitions it
   ranks 329th. Requiring four insiders inside three days, at least $1M of
   total buying, with an officer among the buyers, moves the median 21-day
   log excess over SPY from -0.56% to +0.66% -- and unlike most of the grid,
   it keeps its sign and size across a first-half/second-half time split
   (+0.64% / +0.75%).

2. THE EXIT. Every prior result sold the whole book on a fixed 21-day clock,
   which on a fat right tail systematically sells the winners. A 15% trailing
   stop that arms only once a position is up 10% raises the median trade from
   +1.10% to +4.52% and the rate of >50% winners from 1.2% to 7.2%. A HARD
   stop does the opposite and is badly harmful -- cutting at -10% gives a
   -10.2% median, because these names whipsaw and a fixed floor guarantees
   selling into the whipsaw.

Neither result knows about the other, so this module runs them together.

WHAT IS MEASURED, AND WHY DAILY
-------------------------------
A daily-marked, slot-limited book (`exit_lab.daily_marked_portfolio`) with
empty slots earning zero. Variable-length exits make period-based accounting
meaningless -- positions no longer line up on a common clock -- and an earlier
version of this that smeared each trade's return across its holding days
produced Sharpe ratios near 10, which is a broken measurement rather than a
good result.

THE NULL
--------
Taking every event of a definition means there is no band selection left to
permute. The relevant question becomes whether the DEFINITION earns its
result, so the null draws the same number of events, on the same dates, from
the full unfiltered cluster population, and re-runs the identical book. If a
random draw of equal size does as well, the definition contributed nothing.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools import event_definition_sweep as eds  # noqa: E402
from tools import exit_lab as el  # noqa: E402

#: The definition that survived the time split. Named so the choice is
#: greppable and so nothing downstream has to guess.
BEST_DEF = dict(min_insiders=4, window_days=3, min_total_value=1_000_000,
                role_filter="officer", min_price=0.0)
BASELINE_DEF = dict(min_insiders=2, window_days=14, min_total_value=0.0,
                    role_filter="any", min_price=0.0)

#: Exit rules worth carrying forward, best-first from the exit sweep, plus the
#: incumbent fixed hold as the control.
RULES = [
    el.ExitRule(0.00, 0.15, 252, 0.10),
    el.ExitRule(0.00, 0.20, 252, 0.10),
    el.ExitRule(0.00, 0.15, 126, 0.10),
    el.ExitRule(0.00, 0.00, 252),
    el.ExitRule(0.00, 0.00, 63),
    el.ExitRule(0.00, 0.00, 21),          # the incumbent
]


def spy_daily() -> pd.Series:
    px = pd.read_parquet(os.path.join(REPO_ROOT, "price_cache", "SPY.parquet"))
    return px["close"].astype(float).pct_change().dropna()


def book_stats(port: pd.DataFrame, spy: np.ndarray, label: str) -> dict:
    """Annualised return, vol, Sharpe and drawdown from DAILY marks."""
    if len(port) < 250:
        return {}
    r = port["ret"].to_numpy()
    eq = np.cumprod(1.0 + r)
    dd = float((eq / np.maximum.accumulate(eq) - 1.0).min())
    ann = float(eq[-1] ** (252.0 / len(r)) - 1.0)
    vol = float(np.std(r, ddof=1) * np.sqrt(252))
    sharpe = float(np.mean(r) / np.std(r, ddof=1) * np.sqrt(252)) if np.std(r) > 0 else np.nan
    dn = r[r < 0]
    sortino = (float(np.mean(r) / np.std(dn, ddof=1) * np.sqrt(252))
               if len(dn) > 2 else np.nan)
    b = spy[: len(r)]
    beq = np.cumprod(1.0 + b)
    bann = float(beq[-1] ** (252.0 / len(b)) - 1.0)
    bsharpe = float(np.mean(b) / np.std(b, ddof=1) * np.sqrt(252))
    bdd = float((beq / np.maximum.accumulate(beq) - 1.0).min())
    return dict(
        book=label, days=len(r), invested=float(port["n_active"].mean()),
        ann_return=ann, ann_vol=vol, sharpe=sharpe, sortino=sortino,
        max_dd=dd, final_x=float(eq[-1]),
        spy_ann=bann, spy_sharpe=bsharpe, spy_dd=bdd, spy_final_x=float(beq[-1]),
    )


def run_book(events: pd.DataFrame, rule: el.ExitRule, *, slots: int,
             cost_bps: float, spy: np.ndarray) -> tuple[dict, pd.DataFrame]:
    paths = el.build_paths(events, rule.max_hold)
    if not len(paths.entry):
        return {}, pd.DataFrame()
    trades = el.simulate_paths(paths, rule, cost_bps=cost_bps)
    if trades.empty:
        return {}, pd.DataFrame()
    port = el.daily_marked_portfolio(paths, trades, n_slots=slots,
                                     cost_bps=cost_bps)
    return book_stats(port, spy, rule.label()), trades


def definition_null(
    cf: pd.DataFrame, best: pd.DataFrame, rule: el.ExitRule, *,
    slots: int, cost_bps: float, spy: np.ndarray, calendar, n_draws: int,
    seed: int = 0,
) -> dict:
    """Is the winning DEFINITION worth anything, or is it the search?

    384 definitions were compared and roughly 90 exit rules on top of that.
    A search that wide finds impressive books in noise; this project has
    already watched a band search manufacture +18%/yr from shuffled scores.

    The null here holds everything constant except the thing being credited.
    Each draw takes the SAME NUMBER of events, on the SAME CALENDAR DATES, from
    the full unfiltered cluster population, and runs the identical exit rule
    and slot logic. Matching the dates matters: the winning definition's events
    are not uniformly spread, and a null that ignored when they happened would
    be testing calendar luck rather than the definition.

    If a random same-size, same-dates draw does as well, then "four insiders,
    three days, $1M, an officer" selected nothing that mattered.
    """
    rng = np.random.default_rng(seed)
    pool = cf[(cf["min_insiders"] == 2) & (cf["window_days"] == 14)].copy()
    pool = pool.dropna(subset=["ticker", "entry_day"])
    pool["entry_day"] = pd.to_datetime(pool["entry_day"])
    cal_pos = {d_: i for i, d_ in enumerate(calendar)}
    pool["entry_idx"] = [cal_pos.get(x.date(), -1) for x in pool["entry_day"]]
    pool = pool[pool["entry_idx"] >= 0]
    pool["score"] = 0.0

    # Bucket the pool by calendar quarter so a draw can match the winner's
    # timing without needing an exact date match, which would rarely exist.
    pool["q"] = pool["entry_day"].dt.to_period("Q")
    want = pd.to_datetime(best["entry_day"]).dt.to_period("Q").value_counts()
    by_q = {q: g for q, g in pool.groupby("q")}

    stats = []
    for _ in range(n_draws):
        parts = []
        for q, k in want.items():
            g = by_q.get(q)
            if g is None or not len(g):
                continue
            take = min(int(k), len(g))
            parts.append(g.iloc[rng.choice(len(g), take, replace=False)])
        if not parts:
            continue
        draw = pd.concat(parts, ignore_index=True)
        s, _ = run_book(draw, rule, slots=slots, cost_bps=cost_bps, spy=spy)
        if s:
            stats.append((s["ann_return"], s["sharpe"], s["final_x"]))
    if not stats:
        return {}
    arr = np.asarray(stats, dtype=float)
    return dict(
        n_draws=len(arr),
        ann_median=float(np.median(arr[:, 0])),
        ann_p95=float(np.percentile(arr[:, 0], 95)),
        sharpe_median=float(np.median(arr[:, 1])),
        sharpe_p95=float(np.percentile(arr[:, 1], 95)),
        final_median=float(np.median(arr[:, 2])),
        _arr=arr,
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--events", default=os.path.join(
        REPO_ROOT, "clusters_history", "events_20180813_20260813.parquet"))
    ap.add_argument("--slots", type=int, default=15)
    ap.add_argument("--cost-bps", type=float, default=50.0)
    ap.add_argument("--draws", type=int, default=60)
    args = ap.parse_args(argv)

    print("building the cluster frame (this is the expensive step)...",
          flush=True)
    df = eds.load_qualifying_rows(args.events)
    store = eds.PriceStore()
    calendar = store.calendar()
    cf = eds.build_cluster_frame(
        df, store, calendar,
        min_insiders_grid=(2, 4), window_days_grid=(3, 14), verbose=False,
    )

    spy = spy_daily().to_numpy()

    best_events, best_rows = None, []
    for name, d in (("BEST DEFINITION", BEST_DEF), ("BASELINE", BASELINE_DEF)):
        # `build_cluster_frame` already emits entry_day and entry_idx on the
        # shared SPY calendar, which is the same index space exit_lab's slot
        # accounting works in. `score` is a required column for the slot
        # ordering; with no model score here every event ranks equal, so slots
        # fill purely first-come-first-served.
        sub = eds.select_definition(cf, **d).copy()
        sub["score"] = 0.0
        sub = sub.dropna(subset=["ticker", "entry_day"])
        sub["entry_day"] = pd.to_datetime(sub["entry_day"])
        # `build_cluster_frame` emits entry_day but not its calendar position,
        # and exit_lab's slot accounting works in trading-day index space.
        # Recover it from the same SPY calendar the frame was built against so
        # the two agree exactly rather than approximately.
        cal_pos = {d_: i for i, d_ in enumerate(calendar)}
        sub["entry_idx"] = [cal_pos.get(x.date(), -1) for x in sub["entry_day"]]
        sub = sub[sub["entry_idx"] >= 0]
        print(f"\n\n########## {name} — {len(sub)} events ##########")
        rows = []
        for rule in RULES:
            s, _ = run_book(sub, rule, slots=args.slots,
                            cost_bps=args.cost_bps, spy=spy)
            if s:
                rows.append(s)
                print(f"  {rule.label():28s} ann {s['ann_return']:+7.1%} "
                      f"vol {s['ann_vol']:6.1%} Sharpe {s['sharpe']:+6.3f} "
                      f"maxDD {s['max_dd']:+7.1%} "
                      f"invested {s['invested']:.1f}/{args.slots}", flush=True)
        if rows:
            show = ["book", "days", "invested", "ann_return", "ann_vol",
                    "sharpe", "sortino", "max_dd", "final_x",
                    "spy_ann", "spy_sharpe", "spy_dd", "spy_final_x"]
            with pd.option_context("display.width", 250,
                                   "display.max_columns", 30):
                print(pd.DataFrame(rows)[show].to_string(
                    index=False, float_format=lambda v: f"{v:+.3f}"))
        if name == "BEST DEFINITION":
            best_events = sub
            best_rows = rows

    # --- the null -------------------------------------------------------
    rule = RULES[2]   # trail15%@10% max126, the best daily-marked book
    observed = next((r for r in best_rows if r["book"] == rule.label()), None)
    if observed is None:
        return 0
    print("\n\n########## NULL: same count, same quarters, drawn from "
          "the FULL cluster population ##########")
    print(f"  rule under test: {rule.label()}, {args.draws} draws")
    nul = definition_null(cf, best_events, rule, slots=args.slots,
                          cost_bps=args.cost_bps, spy=spy, calendar=calendar,
                          n_draws=args.draws)
    if not nul:
        print("  null produced no usable draws")
        return 0
    arr = nul.pop("_arr")
    p_ann = float((arr[:, 0] >= observed["ann_return"]).mean())
    p_sharpe = float((arr[:, 1] >= observed["sharpe"]).mean())
    print(f"  observed        ann {observed['ann_return']:+7.2%}  "
          f"Sharpe {observed['sharpe']:+.3f}  final {observed['final_x']:.2f}x")
    print(f"  null median     ann {nul['ann_median']:+7.2%}  "
          f"Sharpe {nul['sharpe_median']:+.3f}  final {nul['final_median']:.2f}x")
    print(f"  null 95th pct   ann {nul['ann_p95']:+7.2%}  "
          f"Sharpe {nul['sharpe_p95']:+.3f}")
    print(f"  p(return)       {p_ann:.4f}")
    print(f"  p(Sharpe)       {p_sharpe:.4f}")
    print("\n  A p above ~0.10 means the definition selected nothing: a "
          "random")
    print("  same-size draw from all clusters, run through the same exit rule,")
    print("  does just as well.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
