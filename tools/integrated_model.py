"""Every new data source at once, judged by the holdout that has caught
everything else.

WHAT THIS TESTS
===============
Five new point-in-time feature sets were built from free sources:

  short_interest     FINRA bi-monthly short interest, days-to-cover, % of ADV
  fundamentals       SEC bulk XBRL: market cap, cash runway, margins, equity
  earnings_features  SEC submissions: days since/to a periodic filing
  filing_context     13D/13G beneficial-ownership filings and 8-K item codes
  form4_extras       grants, option exercises, exercise-and-hold, new insiders

Each is attached only if its module and data are present, so a missing one
degrades the test rather than breaking it.

WHY THE EXPECTATION HERE IS LOW, AND SAYING SO UP FRONT
-------------------------------------------------------
The pre-registered holdout in `tools/final_search.py` established that every
form of SELECTION in this project overfits: a 7,560-configuration search
scored Sharpe 1.437 in the selection window and 0.400 out of sample, and the
whole top-ten neighbourhood failed. More features are more selection. The one
thing that generalised was a mechanical risk change (a trailing stop), which
no feature set can improve.

So this is run as a genuine test rather than a hopeful one, and the holdout
number is the answer even when it is disappointing. Two things are reported
for every feature set: the ranking quality (monthly IC and the volatility-
neutral IC, the metric that has repeatedly separated real scores from the
volatility factor in a hat), and the holdout book.

THE ONE DESCRIPTIVE LEAD WORTH CHECKING
---------------------------------------
Of everything built, only one raw feature looked non-flat on medians: cash
runway. Median 21-day excess by runway quartile runs -2.33%, -2.13%, -1.81%,
-0.77% -- companies closest to running out of money do worst, monotonically.
That is a plausible mechanism rather than a coincidence hunt, so it is also
tested on its own, as a single hand-specified feature, which is far harder to
overfit than a 60-feature model.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools import data_paths  # noqa: E402

from tools import run_score_lab as lab  # noqa: E402
from tools import score_lab as sl  # noqa: E402
from tools.ship_candidate import rank_average  # noqa: E402

#: (module, friendly name). Each module must expose attach_features(df).
FEATURE_MODULES = [
    ("tools.short_interest", "short interest"),
    ("tools.fundamentals", "fundamentals"),
    ("tools.earnings_features", "earnings proximity"),
    ("tools.filing_context", "13D/13G and 8-K items"),
    ("tools.form4_extras", "unused Form 4 fields"),
]


def attach_all(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Attach every feature module that is importable and has its data."""
    before = set(df.columns)
    added = {}
    for mod_name, friendly in FEATURE_MODULES:
        try:
            mod = importlib.import_module(mod_name)
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {friendly}: not importable ({exc})")
            continue
        # Modules were written independently and do not all agree on the name
        # of their entry point. Try the conventional one first, then the known
        # aliases, rather than silently skipping a whole data source over a
        # naming difference -- which is what happened the first time this ran.
        fn = None
        for attr in ("attach_features", "compute_earnings_features",
                     "compute_filing_context_features"):
            fn = getattr(mod, attr, None)
            if fn is not None:
                break
        if fn is None:
            print(f"  skip {friendly}: no attach-features entry point")
            continue
        try:
            out = fn(df)
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {friendly}: attach_features failed ({exc})")
            continue
        if not isinstance(out, pd.DataFrame) or len(out) != len(df):
            print(f"  skip {friendly}: returned the wrong shape")
            continue
        # Modules are chained, and a module that quietly drops a column it does
        # not care about silently disables a LATER module that needs it. That
        # already happened once here: an upstream attach dropped `entry_open`,
        # so fundamentals could not compute market cap and said so in a log
        # line that was easy to miss. Restore anything lost rather than
        # depending on every module being tidy.
        lost = [c for c in df.columns if c not in out.columns]
        if lost:
            print(f"    ({friendly} dropped {len(lost)} column(s); restoring)")
            # Restore POSITIONALLY, not by index: a module that reset or
            # reordered its index would make a label-join silently misalign
            # every row, which is far worse than the missing column it fixes.
            # Row count was already checked equal above.
            out = out.reset_index(drop=True)
            out[lost] = df[lost].reset_index(drop=True)
        new = [c for c in out.columns if c.startswith("x_") and c not in before]
        if not new:
            print(f"  skip {friendly}: added no x_ columns")
            continue
        df = out
        before |= set(new)
        added[friendly] = new
        cov = {c: float(df[c].notna().mean()) for c in new}
        worst = min(cov.values())
        print(f"  + {friendly}: {len(new)} features, "
              f"coverage {worst:.0%}-{max(cov.values()):.0%}")
    return df, added


def audit_score(df: pd.DataFrame, cand: sl.Candidate, seeds: int,
                horizon: int, name: str) -> tuple[pd.Series, dict]:
    members = [sl.build_oof(df, cand, horizon=horizon, seed=s)
               for s in range(seeds)]
    score = rank_average(members)
    work = df.copy()
    work["_s"] = score
    a = sl.audit(work, "_s", lab.GRADE_LABEL, horizon=horizon, name=name)
    return score, dict(
        name=name,
        ic=a.ic_mean, ic_t=a.ic_t,
        yrs=f"{a.years_positive}/{a.years_total}",
        vol_neutral=a.vol_neutral,
        lowvol_ref=a.lowvol_benchmark,
        mono=a.decile_rank_corr,
        gate="PASS" if a.passes() else "FAIL",
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default=data_paths.latest_research_dataset())
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    base = lab.load_dataset(args.dataset)
    base["entry_day"] = pd.to_datetime(base["entry_day"])
    n_base = len([c for c in base.columns if c.startswith("x_")])
    print(f"{len(base)} events, {n_base} baseline features\n")
    print("attaching new feature sets:")
    rich, added = attach_all(base.copy())
    n_rich = len([c for c in rich.columns if c.startswith("x_")])
    print(f"\n{n_rich} features after attachment "
          f"(+{n_rich - n_base} new)\n")

    never_live = set(lab.NEVER_LIVE_FEATURES)

    def live_cols(d):
        return [c for c in sl.feature_cols(d) if c not in never_live]

    def baseline_cols(d):
        keep = set(sl.feature_cols(base))
        return [c for c in live_cols(d) if c in keep]

    rows, scores = [], {}
    specs = [
        ("baseline (no new data)", baseline_cols, rich),
        ("everything", live_cols, rich),
    ]
    # One feature set at a time, so a winner can be attributed rather than
    # guessed at, and so a single harmful set cannot hide inside the total.
    for friendly, cols_added in added.items():
        keep = set(sl.feature_cols(base)) | set(cols_added)
        specs.append((f"baseline + {friendly}",
                      (lambda d, k=keep: [c for c in live_cols(d) if c in k]),
                      rich))

    for name, colfn, frame in specs:
        cand = sl.Candidate(
            name=name, target=lab.t_month_rel, objective="quantile",
            alpha=0.35, features=colfn,
        )
        s, row = audit_score(frame, cand, args.seeds, lab.HORIZON, name)
        scores[name] = s
        rows.append(row)
        print(f"  {name:38s} IC {row['ic']:+.4f}  yrs {row['yrs']}  "
              f"vol-neutral {row['vol_neutral']:+.4f}  {row['gate']}",
              flush=True)

    print("\n\n===== RANKING QUALITY, every feature set =====")
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(pd.DataFrame(rows).to_string(
            index=False, float_format=lambda v: f"{v:+.4f}"))

    if args.out:
        keep = rich[["ticker", "event_day", "entry_day", "entry_idx"]].copy()
        for k, v in scores.items():
            keep[k.replace(" ", "_").replace("+", "plus")] = v
        keep.to_parquet(args.out, index=False)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
