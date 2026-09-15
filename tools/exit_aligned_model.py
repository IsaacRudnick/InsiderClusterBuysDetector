"""Train the model on the outcome the book actually harvests.

THE MISALIGNMENT
================
Every model in this repo predicts a 21-trading-day excess return, because
that is the horizon at which a ranking was first found. The book that came out
of the exit work does something else entirely: it holds behind a 15% trailing
stop, armed at +10%, for up to 126 days, and the realised holding period
averages around 96 days.

So the score is answering a question nobody is asking any more. An event that
looks mediocre over exactly 21 calendar-fixed days may be exactly the one that
runs for four months behind a trailing stop, and the 21-day label cannot see
the difference.

This module builds the label from the SIMULATED TRADE OUTCOME under the exit
rule the book actually uses, and trains on that.

THE LEAKAGE HAZARD, WHICH IS LARGER HERE
----------------------------------------
A 21-day label closes 21 days after entry. A trailing-stop label can stay open
for 126. The purge and embargo must therefore be sized to the MAXIMUM possible
holding period, not the average one -- otherwise a training row whose trade
happened to run long would still have been open when the test year started,
and the model would be learning from the future. `build_oof` is called with
horizon=126 for exactly this reason, which costs a chunk of training data and
is not optional.

The target keeps the same shape that won on the 21-day work: the log outcome,
median-demeaned inside its calendar month and within-month volatility
quintile, fit with a quantile objective below the median. That shape was
chosen on a different label, so it is carried forward as a prior rather than
re-searched here.
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

from tools import data_paths  # noqa: E402

from tools import exit_lab as el  # noqa: E402
from tools import run_score_lab as lab  # noqa: E402
from tools import score_lab as sl  # noqa: E402
from tools.ship_candidate import rank_average  # noqa: E402

#: The exit the book uses. The label is this rule's realised outcome.
BOOK_RULE = el.ExitRule(0.00, 0.15, 126, 0.10)

#: Purge/embargo horizon. Must be the rule's MAXIMUM hold, not its average.
LABEL_HORIZON = BOOK_RULE.max_hold


def attach_exit_outcome(df: pd.DataFrame, rule: el.ExitRule) -> pd.DataFrame:
    """Add `exit_ret` and `exit_days`: what each event earned under `rule`."""
    paths = el.build_paths(df, rule.max_hold)
    trades = el.simulate_paths(paths, rule, cost_bps=0.0)
    out = df.merge(
        trades[["ticker", "entry_day", "ret", "days_held"]].rename(
            columns={"ret": "exit_ret", "days_held": "exit_days"}),
        on=["ticker", "entry_day"], how="left",
    )
    return out


def spy_over_hold(df: pd.DataFrame) -> pd.Series:
    """SPY's return over each trade's OWN holding window.

    A variable-length trade cannot be judged against a fixed-length benchmark
    window. Holding for 96 days in a market that rose 8% is not the same
    achievement as holding for 12 days while it rose 8%, and comparing both to
    a 21-day SPY number would silently reward long holds during rallies.
    """
    px = pd.read_parquet(os.path.join(REPO_ROOT, "price_cache", "SPY.parquet"))
    opens = px["open"].astype(float)
    opens.index = pd.to_datetime(opens.index)
    pos = {d: i for i, d in enumerate(opens.index)}
    vals = opens.to_numpy()
    out = []
    for day, days_held in zip(df["entry_day"], df["exit_days"]):
        i = pos.get(pd.Timestamp(day))
        if i is None or not np.isfinite(days_held):
            out.append(np.nan)
            continue
        j = min(i + int(days_held), len(vals) - 1)
        out.append(vals[j] / vals[i] - 1.0 if vals[i] > 0 else np.nan)
    return pd.Series(out, index=df.index)


def build_target(df: pd.DataFrame) -> pd.Series:
    """Log excess over SPY across the trade's own window, cohort-demeaned."""
    r = pd.to_numeric(df["exit_ret"], errors="coerce")
    b = pd.to_numeric(df["spy_hold"], errors="coerce")
    lx = (np.log1p(r.clip(lower=-0.999)) - np.log1p(b.clip(lower=-0.999))
          ).where(r.notna() & b.notna())
    return sl.cohort_demean(lx, [df["month"], lab._vol_bucket(df)])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default=data_paths.latest_research_dataset())
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    df = lab.load_dataset(args.dataset)
    df["entry_day"] = pd.to_datetime(df["entry_day"])
    print(f"{len(df)} events; simulating {BOOK_RULE.label()} to build the label",
          flush=True)
    df = attach_exit_outcome(df, BOOK_RULE)
    df["spy_hold"] = spy_over_hold(df)
    df["exit_target"] = build_target(df)
    ok = df["exit_target"].notna()
    print(f"  labelled {int(ok.sum())} events; median hold "
          f"{df.loc[ok, 'exit_days'].median():.0f} days, "
          f"mean {df.loc[ok, 'exit_days'].mean():.0f}")

    cand = sl.Candidate(
        name="exit_aligned",
        target=lambda d: d["exit_target"],
        objective="quantile", alpha=0.35,
        features=lab.f_live_only,
        notes="Trained on the realised trailing-stop outcome rather than a "
              "fixed 21-day return.",
    )
    members = [
        sl.build_oof(df, cand, horizon=LABEL_HORIZON, seed=s)
        for s in range(args.seeds)
    ]
    df["ens"] = rank_average(members)
    scored = int(df["ens"].notna().sum())
    print(f"  scored {scored} events out of fold "
          f"(purge/embargo sized to {LABEL_HORIZON} days)")

    # Graded against the SAME 21-day label every other score in this repo was
    # graded on, so the comparison is like for like even though the training
    # label changed.
    a = sl.audit(df, "ens", lab.GRADE_LABEL, horizon=21,
                 name="exit-aligned model, graded on the usual 21-day label")
    print("\n" + sl.format_audit(a))

    df[["ticker", "event_day", "entry_day", "entry_idx", "ens",
        "exit_ret", "exit_days"]].to_parquet(args.out, index=False)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
