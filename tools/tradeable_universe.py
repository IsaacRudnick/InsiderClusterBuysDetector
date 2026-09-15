"""Can this be traded with real money? Refit on a tradeable universe and see.

WHY
===
The unrestricted book posts Sharpe 1.303 against SPY's 1.192, and its
permutation test passes at p=0.000, so the signal is real. Then the
implementation audit lands:

  cost      Sharpe falls below SPY's somewhere under 50bps round trip
  liquidity capping a position at 10% of the name's 20-day dollar volume
            takes Sharpe to 0.98 at $1M of capital and 0.75 at $25M
  price     with a $3 entry-price floor, excess over SPY falls +14.1% -> +3.5%

Every one of those says the same thing: the edge is concentrated in names too
small, too cheap or too thin to actually buy.

There is one honest way to find out whether that is fatal, and it is NOT to
apply the filters at the end. The model was fit on the whole universe, so it
learned to rank names it will now never be allowed to hold, and the band cuts
were calibrated on a distribution that no longer exists once the junk is
removed. This module therefore RESTRICTS FIRST and refits inside the
restriction -- a different model, trained and evaluated only on names a
person could have bought.

If the edge survives that, it is tradeable. If it does not, the honest
statement is that the signal is real and unharvestable, and that is a result
worth having rather than one to bury.
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

from tools import data_paths  # noqa: E402

from tools import run_score_lab as lab  # noqa: E402
from tools import run_sharpe_lab as rsl  # noqa: E402
from tools import score_lab as sl  # noqa: E402
from tools import sharpe_lab as sh  # noqa: E402
from tools.ship_candidate import rank_average  # noqa: E402



def tradeable_mask(
    df: pd.DataFrame, *, min_price: float, capital: float,
    n_names: float = 20.0, participation: float = 0.10,
) -> pd.Series:
    """Rows a book of `capital` could actually have taken a position in.

    Two conditions, both crude in the conservative direction:
      price      entry price at or above `min_price`
      liquidity  the per-name position (capital / n_names) is at most
                 `participation` of one day's average dollar volume

    A real order would be worked over several days, so requiring it to fit
    inside 10% of a SINGLE day understates what is executable. That is the
    right way to be wrong here.
    """
    price_ok = pd.to_numeric(df["entry_open"], errors="coerce") >= min_price
    adv = np.expm1(pd.to_numeric(df["x_log_adv20"], errors="coerce"))
    need = capital / n_names
    liq_ok = (adv * participation) >= need
    return (price_ok & liq_ok).fillna(False)


def evaluate_universe(
    df: pd.DataFrame, label: str, *, seeds: int, draws: int, cost_bps: float,
    dataset: str | None = None,
) -> dict:
    """Refit the score inside this universe, then grid-search and permute it."""
    if len(df) < 2000:
        return dict(universe=label, rows=len(df), note="too few rows to fit")

    cand = next(c for c in lab.CANDIDATES
                if c.name == "C19_month_vol_rel_a35_live")
    members = [
        sl.build_oof(df, cand, horizon=lab.HORIZON, seed=s) for s in range(seeds)
    ]
    work = df.copy()
    work["refit"] = rank_average(members)
    work = work.dropna(subset=["refit"])

    # Join the forward returns from the SAME dataset the refit above was
    # fitted on. This was hardcoded to research_10861rows_20260813.parquet,
    # so passing --dataset refit the score on the new data and then silently
    # graded it against the old file's returns.
    prepared = sh.prepare(
        work[["ticker", "event_day", "refit"]],
        dataset or data_paths.latest_research_dataset(),
        score_cols=("refit",),
    )
    panel = sh.build_panel(prepared, "refit")

    best, best_spec = None, None
    for spec in rsl.grid("refit"):
        spec = sh.BookSpec(**{**spec.__dict__, "cost_bps": cost_bps})
        per = sh.panel_returns(panel, spec)
        if len(per) < 20:
            continue
        m = sh.evaluate(per)
        if m and np.isfinite(m.get("sharpe", np.nan)):
            if best is None or m["sharpe"] > best["sharpe"]:
                best, best_spec = m, spec
    if best is None:
        return dict(universe=label, rows=len(df), note="no viable recipe")

    def search(p, scores):
        b = -np.inf
        for s in rsl.grid("refit"):
            s = sh.BookSpec(**{**s.__dict__, "cost_bps": cost_bps})
            per = sh.panel_returns(p, s, scores)
            if len(per) < 20:
                continue
            m = sh.evaluate(per)
            if m and np.isfinite(m.get("sharpe", np.nan)):
                b = max(b, m["sharpe"])
        return b

    perm = sh.permutation_test_panel(panel, search, best["sharpe"], n_draws=draws)
    ref = sh.panel_returns(panel, sh.BookSpec(score_col="refit", lo=0.0, hi=1.0))
    spy = sh.benchmark_row(ref, "SPY")

    return dict(
        universe=label,
        rows=len(df),
        recipe=best_spec.label(),
        avg_names=best["avg_names"],
        ann_return=best["ann_return"],
        ann_vol=best["ann_vol"],
        sharpe=best["sharpe"],
        max_dd=best["max_drawdown"],
        excess_SPY=best["excess_SPY"],
        yrs_SPY=best["yrs_SPY"],
        spy_sharpe=spy["sharpe"],
        beats_spy=best["sharpe"] > spy["sharpe"],
        perm_p=perm.get("p_value", float("nan")),
        null_p95=perm.get("null_p95", float("nan")),
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default=data_paths.latest_research_dataset())
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--draws", type=int, default=80)
    args = ap.parse_args(argv)

    df = lab.load_dataset(args.dataset)

    # Each row is a plausible answer to "how much money is being run, and how
    # fussy is the execution". Costs rise with how thin the universe is
    # allowed to be, because that is how trading actually works.
    scenarios = [
        ("unrestricted, 20bps", 0.0, 0.0, 20.0),
        ("$1+, 50bps", 1.0, 0.0, 50.0),
        ("$5+, 50bps", 5.0, 0.0, 50.0),
        ("$5+ & $250k book, 50bps", 5.0, 250_000, 50.0),
        ("$5+ & $1M book, 50bps", 5.0, 1_000_000, 50.0),
        ("$5+ & $5M book, 75bps", 5.0, 5_000_000, 75.0),
        ("$10+ & $5M book, 75bps", 10.0, 5_000_000, 75.0),
    ]

    rows = []
    for label, min_price, capital, cost in scenarios:
        sub = df[tradeable_mask(df, min_price=min_price, capital=capital)] \
            if (min_price or capital) else df
        r = evaluate_universe(sub, label, seeds=args.seeds, draws=args.draws,
                              cost_bps=cost, dataset=args.dataset)
        rows.append(r)
        print(f"  {label}: {r.get('rows')} rows, "
              f"Sharpe {r.get('sharpe', float('nan')):+.3f} "
              f"vs SPY {r.get('spy_sharpe', float('nan')):+.3f}, "
              f"p={r.get('perm_p', float('nan')):.3f}", flush=True)

    print("\n\n===== REFIT INSIDE EACH TRADEABLE UNIVERSE =====")
    table = pd.DataFrame(rows)
    with pd.option_context("display.width", 260, "display.max_columns", 40):
        print(table.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
