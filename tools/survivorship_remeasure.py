"""How optimistic were the published numbers? Measured, not assumed.

The research dataset holds 10,861 cluster events and every one of them is on a
company that still trades today. `tools/delisting_fate.py` and
`tools/survivorship_bound.py` recover the 4,214 events that were silently
dropped, so the true population is 15,075 -- 28% larger than anything this
project has ever measured on.

Crucially the dropped events are NOT mostly failures:

    acquired (incl. delisted)  1,299   deals close at a PREMIUM
    renamed and alive          1,157   real prices recovered under the new symbol
    still filing, no ticker      957   alive, venue unknown
    bankruptcy                   579   worthless, and that is a fact not a guess
    delisted, unexplained        216   genuinely unknown
    unknown                        6

So the bias runs BOTH ways, and the correction is not simply "subtract
something". 1,154 of these can be priced for real. The rest are bounded by
re-running under three explicit assumptions and reporting the range.

The trades that cannot be priced are given a fixed outcome and a fixed
occupancy of DEAD_HOLD_DAYS. Holding the slot longer is the conservative
choice for a capital-constrained book: a fast wipeout would free the capital
sooner and hurt less.
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

from tools import exit_lab as el  # noqa: E402
from tools import survivorship_bound as sb  # noqa: E402

BOOK_RULE = el.ExitRule(0.00, 0.15, 126, 0.10)


def priceable_dead(dead: pd.DataFrame, fate: pd.DataFrame) -> pd.DataFrame:
    """Dead events whose company is alive under a new symbol.

    The event keeps its original date but is priced under the CURRENT ticker,
    because providers backfill a renamed symbol's whole history. This converts
    an assumption into a measurement for a quarter of the dropped population.
    """
    sub = dead[dead["mapped_ticker"].notna()].copy()
    sub["orig_ticker"] = sub["ticker"]
    sub["ticker"] = sub["mapped_ticker"]
    sub["entry_day"] = pd.to_datetime(sub["last_date"])
    sub["entry_idx"] = 0
    sub["score"] = 0.0
    return sub


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cost-bps", type=float, default=50.0)
    args = ap.parse_args(argv)

    fate = sb.load_fate()
    dead = pd.read_parquet(os.path.join(
        REPO_ROOT, "research_data", "dead_cluster_events.parquet"))
    surv = pd.read_parquet(os.path.join(
        REPO_ROOT, "research_data", "research_10861rows_20260813.parquet"))
    surv["entry_day"] = pd.to_datetime(surv["entry_day"])
    surv["score"] = 0.0

    print(f"survivors {len(surv)}   dropped {len(dead)}   "
          f"true population {len(surv) + len(dead)}")

    # 1. The survivors, under the book's exit rule -- the number every
    #    published result rests on.
    ps = el.build_paths(surv, BOOK_RULE.max_hold)
    ts = el.simulate_paths(ps, BOOK_RULE, cost_bps=args.cost_bps)
    print(f"\nsurvivors priced: {len(ts)}")

    # 2. The dropped events that are alive under a new symbol, priced for real.
    rec = priceable_dead(dead, fate)
    pr = el.build_paths(rec, BOOK_RULE.max_hold)
    tr = el.simulate_paths(pr, BOOK_RULE, cost_bps=args.cost_bps)
    print(f"renamed-and-recovered priced: {len(tr)} of {len(rec)} attempted")

    def describe(x, label, n_extra=0, extra_ret=None):
        r = list(x["ret"].to_numpy())
        if extra_ret is not None:
            r += list(extra_ret)
        r = np.asarray(r, dtype=float)
        print(f"  {label:34s} n={len(r):5d}  median {np.median(r):+7.2%}  "
              f"mean {r.mean():+7.2%}  win {(r > 0).mean():6.1%}  "
              f"P(>50%) {(r > 0.50).mean():5.1%}  "
              f"P(<-30%) {(r < -0.30).mean():5.1%}")

    print("\n===== TRADE OUTCOMES UNDER THE BOOK'S EXIT RULE =====")
    describe(ts, "survivors only (the status quo)")
    describe(tr, "the recovered renamed alone")
    both = pd.concat([ts, tr], ignore_index=True)
    describe(both, "survivors + recovered renamed")

    # 3. The genuinely unpriceable, bounded.
    unpriced = dead[dead["mapped_ticker"].isna()].copy()
    cls = unpriced["classification"].fillna("unknown")
    print(f"\nunpriceable dropped events: {len(unpriced)}")
    print(cls.value_counts().to_string())

    print("\n===== THE BOUND: whole population, three assumptions =====")
    print("  (acquisitions are held at 0% rather than a premium, which is the")
    print("   conservative direction for them)")
    for name, assume in sb.SCENARIOS.items():
        synth = cls.map(assume).fillna(assume.get("unknown", -0.25)).to_numpy()
        describe(both, f"{name:12s}", extra_ret=synth)

    print("\n===== WHAT THIS DOES TO THE HEADLINE =====")
    surv_med = float(ts["ret"].median())
    for name, assume in sb.SCENARIOS.items():
        synth = cls.map(assume).fillna(assume.get("unknown", -0.25)).to_numpy()
        allr = np.concatenate([both["ret"].to_numpy(), synth])
        print(f"  {name:12s} median trade {np.median(allr):+7.2%}  "
              f"vs survivors-only {surv_med:+7.2%}  "
              f"(shift {np.median(allr) - surv_med:+.2%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
