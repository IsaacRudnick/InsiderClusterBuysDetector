"""Does the model-score band improve a traded book out of sample? No.

WHY THIS EXISTS
===============
RESEARCH_NOTES.md carried two holdout tables, written a day apart, that
disagreed. "The holdout test" (2026-08-20) found that adding the 70-90 band to
the trailing-stop book made it WORSE (+12.67%/yr -> +9.51%). "Holdout verdict
on the new features" (2026-08-21) showed a control of +11.47% and a top band
of +12.85% -- the band helping. Both were labelled "the holdout", so the file
could answer "does your ranking help when you trade it?" either way.

The books were not identical, so this re-runs the comparison with EVERYTHING
held constant except the two things that actually differed between them: the
percentile band and the slot count.

WHAT IT SHOWS
-------------
1. The 08-20 table reproduces exactly, at 20 slots, on all four of its rows.
2. The 08-21 table does not reproduce at any slot count, because its
   "baseline" is not the shipped score -- `integrated_model.py` defaults to
   6 seeds (not 10) and refits on the feature-attached frame, a different
   model on a different row set. Its internal comparisons stay valid; its
   baseline row is not comparable to the 08-20 rows.
3. The band effect FLIPS SIGN on slot count. The shipped 70-90 band helps at
   5 slots and hurts at 10, 15, 20 and 30. Nothing about a percentile band
   should depend on how many positions the book carries. Read it as noise.
4. The exit rule is robust: the trailing stop beats the fixed 21-day hold in
   25 of 25 cells, mean +0.637 Sharpe and +10.75pp of annual return.

Run:  python tools/settle_band.py --scores research_data/<ens scores>.parquet

The scores parquet is the one `tools/ship_candidate.py --seeds 10 --out ...`
writes; it carries the `ens` column and is not checked in.
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools import data_paths  # noqa: E402

from tools import final_search as fs  # noqa: E402

#: Index into fs.RULES. 0 is the fixed 21-day hold every earlier result used;
#: 3 is the 15% trailing stop armed at +10%, capped at 126 days -- the one
#: change that generalised out of sample.
RULE_NAMES = {0: "21d fixed hold", 3: "15% trail armed +10%, max126"}

BANDS = [((0.0, 1.0), "all events"),
         ((0.5, 1.0), "50-100 band"),
         ((0.7, 1.0), "70-100 band"),
         ((0.5, 0.9), "50-90 band"),
         ((0.7, 0.9), "70-90 band")]
BAND_ORDER = [n for _, n in BANDS]

#: The slot count was never held constant between the two disputed tables, so
#: it is swept rather than chosen. A real band effect would keep its sign here.
SLOTS = [5, 10, 15, 20, 30]


def run(dataset: str, scores: str, score_col: str = "ens") -> pd.DataFrame:
    df = fs.load(dataset, scores, score_col)
    hold = df[df["event_day"] >= fs.HOLDOUT_START].copy()
    print(f"holdout events: {len(hold)}  "
          f"({hold.event_day.min():%Y-%m-%d}..{hold.event_day.max():%Y-%m-%d})")
    spy = fs.spy_returns(None, "2023-01-01", "2026-08-12")
    print("simulating exit rules over the holdout...", flush=True)
    pre = fs.precompute(hold)

    rows = []
    for ri, rname in RULE_NAMES.items():
        for (lo, hi), bname in BANDS:
            for sl in SLOTS:
                # min_insiders=2, no dollar/officer/span/price filter: the
                # unfiltered population, so the BAND is the only selection.
                c = fs.Config(2, 0.0, False, 99.0, 0.0, lo, hi, ri, sl)
                r = fs.book(pre, ri, fs.universe_keys(hold, c), sl)
                if r is None:
                    continue
                s = fs.stats(r, spy)
                rows.append(dict(exit=rname, band=bname, slots=sl,
                                 ann=s["ann_return"], sharpe=s["sharpe"],
                                 maxdd=s["max_dd"]))
    return pd.DataFrame(rows), fs.stats(spy, spy)


def report(t: pd.DataFrame, spy: dict) -> None:
    pd.set_option("display.width", 200)
    for rname in RULE_NAMES.values():
        sub = t[t.exit == rname]
        print(f"\n\n===== EXIT: {rname} =====")
        for col, fmt in (("ann", "{:+.2%}"), ("sharpe", "{:+.3f}"),
                         ("maxdd", "{:+.1%}")):
            print(f"\n{col} by band x slots")
            print(sub.pivot(index="band", columns="slots", values=col)
                  .reindex(BAND_ORDER)
                  .to_string(float_format=lambda v: fmt.format(v)))

    print(f"\n\nSPY, same holdout: ann {spy['ann_return']:+.2%}  "
          f"Sharpe {spy['sharpe']:+.3f}  maxDD {spy['max_dd']:+.1%}")

    print("\n\n===== DOES THE BAND HELP? (band minus 'all events') =====")
    b = t[t.exit == RULE_NAMES[3]].set_index(["band", "slots"])["sharpe"].unstack()
    base = b.loc["all events"]
    for name in BAND_ORDER[1:]:
        d = b.loc[name] - base
        print(f"  {name:12s} mean {d.mean():+.3f}   "
              f"helped in {int((d > 0).sum())}/{len(d)} slot counts")

    print("\n===== DOES THE EXIT RULE HELP? (trail minus 21d fixed) =====")
    p1 = t[t.exit == RULE_NAMES[0]].set_index(["band", "slots"])
    p2 = t[t.exit == RULE_NAMES[3]].set_index(["band", "slots"])
    ds, da = (p2["sharpe"] - p1["sharpe"]), (p2["ann"] - p1["ann"])
    print(f"  Sharpe   mean {ds.mean():+.3f}   positive in "
          f"{int((ds > 0).sum())}/{len(ds)} cells")
    print(f"  return   mean {da.mean():+.2%}   positive in "
          f"{int((da > 0).sum())}/{len(da)} cells")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default=data_paths.latest_research_dataset())
    ap.add_argument("--scores", required=True)
    ap.add_argument("--score-col", default="ens")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    t, spy = run(args.dataset, args.scores, args.score_col)
    report(t, spy)
    if args.out:
        t.to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
