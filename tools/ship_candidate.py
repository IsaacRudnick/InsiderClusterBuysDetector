"""Deep audit of the score chosen for shipping, against the score it replaces.

This is the evidence file for one decision: which score `run.bat` should sort
its dashboard by. It does four things and prints all four:

1. Builds the winning candidate as a SEED ENSEMBLE -- the same configuration
   refit under N different RNG seeds, combined by averaging RANKS. Ranks, not
   raw scores, because the members are quantile regressions whose raw output
   is a predicted return; one member with a wider spread would otherwise
   dominate the average.
2. Runs the full gauntlet on the ensemble AND on each member, so the ensemble
   can be seen to be at least as good as its parts rather than assumed to be.
3. Runs the same gauntlet on `oof_tail_classifier` -- the score the live
   screener sorts by today -- graded identically. This is the number the new
   score has to beat, and until this file existed nobody had put the shipped
   score through this particular gauntlet at 21 days.
4. Reports the top-N book at several N with a per-year breakdown, and the
   crash rate by decile.

WHY A SEED ENSEMBLE
-------------------
RESEARCH_NOTES.md's most expensive finding is that a 10-name book built from
a single fit swung +7.98%/yr to +0.81%/yr on the RNG seed alone. Averaging
ranks over many seeds does not make a weak score strong -- it makes the
score's composition reproducible, so that whatever edge it has is the same
edge tomorrow. A score that needs a particular seed is not a score.
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

from tools import run_score_lab as lab  # noqa: E402
from tools import score_lab as sl  # noqa: E402

#: The configuration chosen after the pre-registered sweep in run_score_lab.
#: Named so the choice is greppable and so nothing downstream has to guess.
SHIP_CANDIDATE = "C19_month_vol_rel_a35_live"

DEFAULT_ENSEMBLE_SEEDS = tuple(range(10))


def rank_average(scores: list[pd.Series]) -> pd.Series:
    """Combine member scores by averaging their within-dataset ranks.

    NaN handling matters here: a member that has no score for a row (its fold
    never tested that row) must not be counted as a zero, which would push the
    row down the ensemble ranking for a reason that has nothing to do with the
    row. Members are averaged only over the ones that produced a rank.
    """
    ranks = [s.rank(pct=True) for s in scores]
    stacked = pd.concat(ranks, axis=1)
    out = stacked.mean(axis=1, skipna=True)
    return out.where(stacked.notna().any(axis=1))


def load_shipped_scores(df: pd.DataFrame, path: str) -> pd.Series:
    """Join the currently-shipped out-of-fold score onto the research rows.

    Keyed on (ticker, event_day), the same key the rest of the pipeline uses
    to identify a cluster episode.
    """
    if not os.path.exists(path):
        return pd.Series(np.nan, index=df.index)
    oof = pd.read_parquet(path)[["ticker", "event_day", "oof_tail_classifier"]]
    oof["event_day"] = pd.to_datetime(oof["event_day"])
    merged = df[["ticker", "event_day"]].merge(
        oof, on=["ticker", "event_day"], how="left"
    )
    return pd.Series(merged["oof_tail_classifier"].to_numpy(), index=df.index)


def portfolio_by_n(
    df: pd.DataFrame, score: str, ns=(5, 10, 15, 25, 40)
) -> pd.DataFrame:
    """Top-N book at several breadths, over non-overlapping 21-day periods.

    Breadth is reported rather than optimised. An earlier finding here is that
    alpha decays monotonically to zero as N goes 5 -> 100, and that the only
    N with a real number attached is the one too narrow to be reproducible.
    Showing the whole curve is what stops that trap being sprung again.
    """
    rows = []
    for n in ns:
        p = sl.nonoverlapping_portfolio(
            df, score, horizon=lab.HORIZON, top_n=n
        )
        if len(p) < 3:
            continue
        per_year = p.groupby("year")["excess"].mean()
        from scipy import stats

        rows.append(
            dict(
                top_n=n,
                periods=len(p),
                mean_excess=float(p["excess"].mean()),
                ann_excess=float(
                    (1 + p["excess"].mean()) ** (252 / lab.HORIZON) - 1
                ),
                p_value=float(stats.ttest_1samp(p["excess"], 0.0).pvalue),
                years_beat=int((per_year > 0).sum()),
                years=int(len(per_year)),
                worst_period=float(p["excess"].min()),
            )
        )
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dataset",
        default=data_paths.latest_research_dataset(),
    )
    ap.add_argument(
        "--shipped-oof",
        default=os.path.join(
            REPO_ROOT, "research_data", "oof_scores_20260813.parquet"
        ),
    )
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    df = lab.load_dataset(args.dataset)
    cand = next(c for c in lab.CANDIDATES if c.name == SHIP_CANDIDATE)
    print(f"ship candidate: {cand.name}\n  {cand.notes}\n")

    seeds = list(range(args.seeds))
    members = []
    for s in seeds:
        members.append(sl.build_oof(df, cand, horizon=lab.HORIZON, seed=s))
        print(f"  fitted member seed={s}", flush=True)

    df["ens"] = rank_average(members)
    for s, m in zip(seeds, members):
        df[f"member_{s}"] = m

    print("\n\n########## 1. THE SCORE THAT SHIPS TODAY ##########")
    df["shipped"] = load_shipped_scores(df, args.shipped_oof)
    if df["shipped"].notna().sum() > 100:
        a_old = sl.audit(
            df, "shipped", lab.GRADE_LABEL, horizon=lab.HORIZON,
            name="oof_tail_classifier (currently shipped)",
        )
        print(sl.format_audit(a_old))
        print(a_old.deciles.to_string(index=False))
    else:
        a_old = None
        print("  no overlapping rows -- cannot grade the shipped score")

    print("\n\n########## 2. EACH ENSEMBLE MEMBER ##########")
    for s in seeds:
        a = sl.audit(
            df, f"member_{s}", lab.GRADE_LABEL, horizon=lab.HORIZON,
            name=f"{SHIP_CANDIDATE} seed={s}",
        )
        print(
            f"  seed {s}: IC {a.ic_mean:+.4f}  yrs {a.years_positive}/"
            f"{a.years_total}  vol-neutral {a.vol_neutral:+.4f}  "
            f"mono {a.decile_rank_corr:+.2f}  "
            f"{'PASS' if a.passes() else 'FAIL'}"
        )

    print("\n\n########## 3. THE SEED ENSEMBLE ##########")
    a_new = sl.audit(
        df, "ens", lab.GRADE_LABEL, horizon=lab.HORIZON,
        name=f"{SHIP_CANDIDATE} rank-ensemble of {len(seeds)} seeds",
    )
    print(sl.format_audit(a_new))
    print("\n  decile detail (median/mean log excess over SPY at 21 days,")
    print("  win rate, and P(loses more than 30%) -- the column the shipped")
    print("  score got backwards):")
    print(a_new.deciles.to_string(index=False))

    print("\n\n########## 4. BOOK BREADTH ##########")
    print(portfolio_by_n(df, "ens").to_string(index=False))

    print("\n\n########## VERDICT ##########")
    if a_old is not None:
        print(
            f"  shipped   IC {a_old.ic_mean:+.4f}  "
            f"yrs {a_old.years_positive}/{a_old.years_total}  "
            f"vol-neutral {a_old.vol_neutral:+.4f}  "
            f"top-decile crash rate "
            f"{a_old.deciles['p_crash'].iloc[-1]:.3f}"
        )
    print(
        f"  ensemble  IC {a_new.ic_mean:+.4f}  "
        f"yrs {a_new.years_positive}/{a_new.years_total}  "
        f"vol-neutral {a_new.vol_neutral:+.4f}  "
        f"top-decile crash rate {a_new.deciles['p_crash'].iloc[-1]:.3f}"
    )

    if args.out:
        keep = ["ticker", "event_day", "entry_idx", "ens", "shipped",
                lab.GRADE_LABEL, "adj_21", "adj_63"]
        df[keep].to_parquet(args.out, index=False)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
