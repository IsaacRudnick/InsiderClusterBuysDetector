"""The full search, with the holdout enforced in code.

THE PROTOCOL, PRE-REGISTERED
===========================
By this point the project has searched 384 event definitions, ~90 exit rules,
132 book recipes, 24 model candidates and several universes. Any "best" number
drawn from all of that, measured on all of the data, is a number about the
search rather than about the world -- this repo has already produced two of
those and retired both.

So the data is split once, here, and the split is enforced by the code rather
than by intention:

    SELECTION  2018-08-13 .. 2022-12-31   every choice is made here
    HOLDOUT    2023-01-01 .. 2026-08-12   scored once, at the end

`search()` never sees a holdout row. `evaluate_holdout()` is called exactly
once, on the single configuration `search()` returned, and whatever it says is
the answer -- including if it says the thing does not work. The holdout is not
a checkpoint to iterate against; re-running the search after seeing it would
destroy the only unbiased estimate available.

Note the holdout is the harder half on purpose: SPY compounded at +23.3% with
a Sharpe of 1.46 over those years, against +10.9% and 0.59 in the selection
window. A strategy that only beat a weak market will be caught here.

WHAT IS SEARCHED
----------------
  universe   which cluster events are eligible at all -- insider count, total
             dollars, officer participation, how tight the buying window was,
             and a minimum share price
  band       whether to rank the eligible events by the model score and hold a
             percentile band, or simply hold all of them
  exit       fixed holding period, or a trailing stop that lets winners run
  slots      how many positions the book carries

All of it is scored on a daily-marked, slot-limited book with empty slots
earning zero, net of per-row estimated trading costs.
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools import data_paths  # noqa: E402

from tools import exit_lab as el  # noqa: E402

SELECTION_END = pd.Timestamp("2022-12-31")
HOLDOUT_START = pd.Timestamp("2023-01-01")

#: A configuration must clear this many trades in the selection window before
#: it is eligible to win. Without a floor the search reliably picks a
#: definition with nine trades and a spectacular Sharpe.
MIN_TRADES = 80


@dataclass(frozen=True)
class Config:
    min_insiders: int
    min_value: float
    require_officer: bool
    max_span: float
    min_price: float
    lo: float
    hi: float
    rule_idx: int
    slots: int

    def label(self) -> str:
        u = (f"n>={self.min_insiders}"
             f",${self.min_value/1e6:g}M" if self.min_value else
             f"n>={self.min_insiders}")
        if self.require_officer:
            u += ",officer"
        if self.max_span < 99:
            u += f",span<={self.max_span:g}"
        if self.min_price:
            u += f",${self.min_price:g}+"
        band = "all" if (self.lo, self.hi) == (0.0, 1.0) else \
            f"{self.lo:.0%}-{self.hi:.0%}"
        return f"{u} | {band} | {RULES[self.rule_idx].label()} | {self.slots}sl"


RULES = [
    el.ExitRule(0.00, 0.00, 21),
    el.ExitRule(0.00, 0.00, 63),
    el.ExitRule(0.00, 0.00, 126),
    el.ExitRule(0.00, 0.15, 126, 0.10),
    el.ExitRule(0.00, 0.15, 252, 0.10),
    el.ExitRule(0.00, 0.20, 252, 0.10),
    el.ExitRule(0.00, 0.25, 252, 0.10),
]

GRID = dict(
    min_insiders=(2, 3, 4),
    min_value=(0.0, 250_000.0, 1_000_000.0),
    require_officer=(False, True),
    max_span=(99.0, 7.0, 3.0),
    min_price=(0.0, 5.0),
    band=((0.0, 1.0), (0.5, 1.0), (0.7, 1.0), (0.5, 0.9), (0.7, 0.9)),
    slots=(10, 20),
)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load(dataset: str, scores: str, score_col: str) -> pd.DataFrame:
    df = pd.read_parquet(dataset)
    df["event_day"] = pd.to_datetime(df["event_day"])
    df["entry_day"] = pd.to_datetime(df["entry_day"])
    sc = pd.read_parquet(scores)
    sc["event_day"] = pd.to_datetime(sc["event_day"])
    keep = ["ticker", "event_day", score_col]
    df = df.merge(sc[keep], on=["ticker", "event_day"], how="left")
    df["score"] = df[score_col]
    df["total_value"] = np.expm1(df["x_log1p_total_value"])
    df["adv"] = np.expm1(pd.to_numeric(df["x_log_adv20"], errors="coerce"))

    from tools import execution_model as em
    df["cost_bps"] = em.estimated_cost_bps(df)
    return df.dropna(subset=["entry_day", "entry_idx", "score"])


def precompute(df: pd.DataFrame) -> dict:
    """Every trade's outcome under every exit rule, computed once.

    The search evaluates thousands of configurations, but a trade's outcome
    under a given exit rule does not depend on which universe or band selected
    it -- only on its own price path. Simulating each rule once over all events
    and then filtering is what makes the search affordable, and it is exactly
    equivalent to simulating each configuration separately.
    """
    # The research dataset only holds events that had an entry price, so near
    # full coverage is expected; anything less means a sparse price_cache/.
    paths = el.build_paths(df, max(r.max_hold for r in RULES), min_coverage=0.98)
    out = {}
    for i, rule in enumerate(RULES):
        tr = el.simulate_paths(paths, rule, cost_bps=0.0)
        # Charge each trade its OWN estimated round-trip cost rather than one
        # flat rate: a $2 microcap and a $30 listed name do not cost the same
        # to trade, and averaging them hides the whole problem.
        cost = tr[["ticker", "entry_day"]].merge(
            df[["ticker", "entry_day", "cost_bps"]],
            on=["ticker", "entry_day"], how="left")["cost_bps"].to_numpy()
        tr["ret"] = tr["ret"] - np.nan_to_num(cost, nan=50.0) / 10_000.0
        out[i] = tr
    return dict(paths=paths, trades=out)


def book(pre: dict, rule_idx: int, keys: set, slots: int) -> np.ndarray | None:
    """Daily returns of a slot-limited book over the selected trades."""
    tr = pre["trades"][rule_idx]
    mask = [(t, d) in keys for t, d in zip(tr["ticker"], tr["entry_day"])]
    sub = tr[np.asarray(mask)]
    if len(sub) < 5:
        return None
    port = el.daily_marked_portfolio(pre["paths"], sub, n_slots=slots,
                                     cost_bps=0.0)
    if len(port) < 200:
        return None
    return port["ret"].to_numpy()


def stats(r: np.ndarray, bench: np.ndarray) -> dict:
    eq = np.cumprod(1.0 + r)
    ann = float(eq[-1] ** (252.0 / len(r)) - 1.0)
    sd = float(np.std(r, ddof=1))
    dn = r[r < 0]
    b = bench[: len(r)]
    beq = np.cumprod(1.0 + b)
    return dict(
        days=len(r), ann_return=ann,
        ann_vol=sd * np.sqrt(252),
        sharpe=float(np.mean(r) / sd * np.sqrt(252)) if sd > 0 else np.nan,
        sortino=(float(np.mean(r) / np.std(dn, ddof=1) * np.sqrt(252))
                 if len(dn) > 2 else np.nan),
        max_dd=float((eq / np.maximum.accumulate(eq) - 1.0).min()),
        final_x=float(eq[-1]),
        spy_ann=float(beq[-1] ** (252.0 / len(b)) - 1.0),
        spy_sharpe=float(np.mean(b) / np.std(b, ddof=1) * np.sqrt(252)),
        spy_dd=float((beq / np.maximum.accumulate(beq) - 1.0).min()),
        spy_final_x=float(beq[-1]),
    )


def universe_keys(df: pd.DataFrame, c: Config) -> set:
    m = (
        (df["x_n_insiders"] >= c.min_insiders)
        & (df["total_value"] >= c.min_value)
        & (df["x_window_span_days"] <= c.max_span)
    )
    if c.require_officer:
        m &= df["x_n_officers"] >= 1
    if c.min_price:
        m &= df["entry_open"] >= c.min_price
    sub = df[m]
    if (c.lo, c.hi) != (0.0, 1.0):
        # Rank WITHIN the eligible universe, in 21-day batches, because that
        # is what a screener can do on the day: rank what it can see.
        batch = (sub["entry_idx"] - sub["entry_idx"].min()) // 21
        r = sub.groupby(batch)["score"].rank(pct=True)
        sub = sub[(r > c.lo) & (r <= c.hi)]
    return set(zip(sub["ticker"], sub["entry_day"]))


# ---------------------------------------------------------------------------
# Search (selection window only) and the single holdout scoring
# ---------------------------------------------------------------------------

def all_configs() -> list[Config]:
    out = []
    for mi, mv, ro, ms, mp, (lo, hi), sl in itertools.product(
        GRID["min_insiders"], GRID["min_value"], GRID["require_officer"],
        GRID["max_span"], GRID["min_price"], GRID["band"], GRID["slots"],
    ):
        for ri in range(len(RULES)):
            out.append(Config(mi, mv, ro, ms, mp, lo, hi, ri, sl))
    return out


def search(df_sel: pd.DataFrame, pre_sel: dict, spy_sel: np.ndarray,
           min_trades: int = MIN_TRADES) -> pd.DataFrame:
    """Score every configuration on the SELECTION window only."""
    rows = []
    cache: dict[tuple, set] = {}
    for c in all_configs():
        ukey = (c.min_insiders, c.min_value, c.require_officer, c.max_span,
                c.min_price, c.lo, c.hi)
        keys = cache.get(ukey)
        if keys is None:
            keys = universe_keys(df_sel, c)
            cache[ukey] = keys
        if len(keys) < min_trades:
            continue
        r = book(pre_sel, c.rule_idx, keys, c.slots)
        if r is None:
            continue
        s = stats(r, spy_sel)
        s.update(config=c.label(), n_events=len(keys), _c=c)
        rows.append(s)
    return pd.DataFrame(rows)


def evaluate_holdout(df_hold: pd.DataFrame, pre_hold: dict,
                     spy_hold: np.ndarray, c: Config) -> dict:
    """Score ONE configuration on the holdout. Called exactly once."""
    keys = universe_keys(df_hold, c)
    r = book(pre_hold, c.rule_idx, keys, c.slots)
    if r is None:
        return {}
    s = stats(r, spy_hold)
    s.update(config=c.label(), n_events=len(keys))
    return s


def spy_returns(index: pd.DatetimeIndex, lo, hi) -> np.ndarray:
    px = pd.read_parquet(os.path.join(REPO_ROOT, "price_cache", "SPY.parquet"))
    r = px["close"].astype(float).pct_change().dropna()
    r.index = pd.to_datetime(r.index)
    return r[(r.index >= lo) & (r.index <= hi)].to_numpy()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default=data_paths.latest_research_dataset())
    ap.add_argument("--scores", required=True)
    ap.add_argument("--score-col", default="ens")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    df = load(args.dataset, args.scores, args.score_col)
    sel = df[df["event_day"] <= SELECTION_END].copy()
    hold = df[df["event_day"] >= HOLDOUT_START].copy()
    print(f"selection {len(sel)} events  ({sel.event_day.min():%Y-%m-%d}"
          f"..{sel.event_day.max():%Y-%m-%d})")
    print(f"holdout   {len(hold)} events  ({hold.event_day.min():%Y-%m-%d}"
          f"..{hold.event_day.max():%Y-%m-%d})\n")

    # The benchmark must span the same days the book does. These were
    # hardcoded to "2023-06-30" and "2026-08-12", which was correct while the
    # dataset ended 2026-08-12 and silently wrong afterwards: a dataset built
    # later gives the BOOK extra months of return that SPY is not credited
    # with, flattering every excess figure below. Derive the end from the data
    # instead so the two can never drift apart again.
    sel_end = sel["entry_day"].max() + pd.Timedelta(days=182)
    hold_end = hold["entry_day"].max()
    spy_sel = spy_returns(None, "2018-08-13", sel_end.strftime("%Y-%m-%d"))
    spy_hold = spy_returns(None, "2023-01-01", hold_end.strftime("%Y-%m-%d"))

    print("simulating exit rules over the selection window...", flush=True)
    pre_sel = precompute(sel)
    table = search(sel, pre_sel, spy_sel)
    if table.empty:
        print("no configuration cleared the trade floor")
        return 1
    table = table.sort_values("sharpe", ascending=False).reset_index(drop=True)

    show = ["config", "n_events", "days", "ann_return", "ann_vol", "sharpe",
            "sortino", "max_dd", "final_x"]
    print(f"\n===== SELECTION WINDOW: {len(table)} configurations, "
          f"best Sharpe first =====")
    print(f"  (SPY over this window: ann {table['spy_ann'].iloc[0]:+.1%}, "
          f"Sharpe {table['spy_sharpe'].iloc[0]:+.3f})")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(table[show].head(args.top).to_string(
            index=False, float_format=lambda v: f"{v:+.3f}"))

    winner = table["_c"].iloc[0]
    print(f"\n\n===== LOCKED IN, from the selection window alone =====")
    print(f"  {winner.label()}")

    print("\nsimulating the same exit rules over the holdout...", flush=True)
    pre_hold = precompute(hold)
    hs = evaluate_holdout(hold, pre_hold, spy_hold, winner)
    print(f"\n===== HOLDOUT {HOLDOUT_START:%Y-%m-%d} .. "
          f"{hold['event_day'].max():%Y-%m-%d}, scored once =====")
    if not hs:
        print("  the winning configuration produced no book in the holdout")
        return 0
    print(f"  events {hs['n_events']}   days {hs['days']}")
    print(f"  BOOK   ann {hs['ann_return']:+7.2%}  vol {hs['ann_vol']:6.2%}  "
          f"Sharpe {hs['sharpe']:+.3f}  Sortino {hs['sortino']:+.3f}  "
          f"maxDD {hs['max_dd']:+7.2%}  final {hs['final_x']:.2f}x")
    print(f"  SPY    ann {hs['spy_ann']:+7.2%}                  "
          f"Sharpe {hs['spy_sharpe']:+.3f}                  "
          f"maxDD {hs['spy_dd']:+7.2%}  final {hs['spy_final_x']:.2f}x")

    # A single locked configuration can still get lucky. The top ten from the
    # selection window are scored too -- not to pick a new winner, but because
    # if only the first one survives out of sample that is a coincidence,
    # whereas a whole neighbourhood surviving is a result.
    print(f"\n===== the selection window's top 10, all scored on the holdout ====")
    print("  (reported to show whether the WHOLE neighbourhood survives, not")
    print("   to choose a new winner after the fact)")
    rows = []
    for i in range(min(10, len(table))):
        c = table["_c"].iloc[i]
        h = evaluate_holdout(hold, pre_hold, spy_hold, c)
        if h:
            rows.append(dict(rank=i + 1, config=c.label(),
                             sel_sharpe=table["sharpe"].iloc[i],
                             sel_ann=table["ann_return"].iloc[i],
                             hold_sharpe=h["sharpe"], hold_ann=h["ann_return"],
                             hold_dd=h["max_dd"], hold_x=h["final_x"]))
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(pd.DataFrame(rows).to_string(
            index=False, float_format=lambda v: f"{v:+.3f}"))

    if args.out:
        table.drop(columns=["_c"]).to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
