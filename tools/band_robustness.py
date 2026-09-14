"""Every way the chosen band could be an artifact, tested one at a time.

The band backtest says holding the 80th-90th percentile slice of the ensemble
ranking beat SPY by ~20%/yr. (That slice is NOT the one the product ships --
findings.BANDS' top_band is 70-90, which this module now audits by default.
See the comment on BAND_LO.) RESEARCH_NOTES.md contains four separate
occasions on which a number that good turned out to be an artifact of the
measurement rather than a property of the world. This module runs the specific
checks those occasions produced, and a result is only worth quoting if it
survives all of them.

  1. MULTIPLE COMPARISONS. Twelve bands were tested and the best one was
     reported. A p-value of 0.04 out of twelve tries is roughly what pure
     noise produces. The honest version is reported here.
  2. P&L CONCENTRATION. 0.7% of lots once produced 58% of this grid's profit,
     and a single unadjusted reverse split once gave a "winning" strategy 83%
     of its P&L. If a handful of positions carry the band, the band is those
     positions and not a strategy.
  3. PRICE FLOOR. Sub-dollar stocks cannot absorb a real order and their
     quoted returns are mostly bid-ask bounce. The result has to survive
     throwing them out.
  4. SPLIT-HALF ENSEMBLES. Two disjoint sets of RNG seeds produce two
     independent ensembles. A real band appears in both.
  5. LEAVE ONE YEAR OUT. One good year should not be the whole result.
  6. BOOTSTRAP OVER PERIODS. A confidence interval that does not assume the
     per-period excesses are normal.
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

from scipy import stats  # noqa: E402

from tools import band_backtest as bb  # noqa: E402
from tools import run_score_lab as lab  # noqa: E402
from tools import score_lab as sl  # noqa: E402

# The band under test. This MUST track the band the product actually ships
# (findings.BANDS' top_band), or this module audits a slice no user ever
# sees. It said 0.80, 0.90 until 2026-09-14 while top_band shipped as
# 70-90, and this module's output was transcribed into findings.py's
# TOP_BAND_ANNUALIZED_EXCESS / TOP_BAND_PERMUTATION_P under a 70-90 label --
# so the product quoted the 80-90 band's +19.77%/yr and p=0.435 as though
# they described the band it ranks by. The shipped 70-90 band measures
# +14.07%/yr at p=0.745 on the same data: a weaker apparent edge that fails
# the same test more decisively, so the product's "no index-beating claim"
# conclusion was right for the wrong numbers. Override on the command line
# to audit a different slice; do not change the default without changing
# findings.BANDS to match.
DEFAULT_BAND_LO, DEFAULT_BAND_HI = 0.70, 0.90
BAND_LO, BAND_HI = DEFAULT_BAND_LO, DEFAULT_BAND_HI
PPY = 252.0 / bb.HORIZON


def ann(x: float) -> float:
    return (1 + x) ** PPY - 1


def _excess(per: pd.DataFrame, bench: str = "SPY") -> pd.Series:
    return per["ret"] - per[f"bench_{bench}"]


def check_multiple_comparisons(df: pd.DataFrame, score: str) -> None:
    """Re-price the winning band's p-value against the number of bands tried.

    Two corrections, because they bound the truth from both sides: Bonferroni
    is the conservative one (assumes every band was an independent try, which
    overstates the penalty since overlapping bands are highly correlated), and
    a permutation test is the honest one -- shuffle the scores WITHIN each
    period, re-run the entire band search including the "pick the best" step,
    and see how often noise produces a winner this good.
    """
    print("\n===== 1. MULTIPLE COMPARISONS =====")
    n_bands = len(bb.BANDS)
    per = bb.band_periods(df, score, BAND_LO, BAND_HI)
    ex = _excess(per)
    p_raw = float(stats.ttest_1samp(ex, 0.0).pvalue)
    print(f"  {n_bands} bands were tested; the reported one was the best.")
    print(f"  raw p                 {p_raw:.4f}")
    print(f"  Bonferroni x{n_bands}         {min(1.0, p_raw * n_bands):.4f}")

    rng = np.random.default_rng(0)
    work = df.copy()
    base = int(work["entry_idx"].min())
    work["period"] = (work["entry_idx"] - base) // bb.HORIZON
    best_null = []
    for _ in range(200):
        # Shuffle the score inside each period. This destroys any skill the
        # score has while leaving the period structure, the candidate pool and
        # the return distribution exactly as they are -- so anything the search
        # still finds is search, not signal.
        work["_shuf"] = work.groupby("period")[score].transform(
            lambda s: rng.permutation(s.to_numpy())
        )
        best = -9.9
        for lo, hi, _ in bb.BANDS:
            p = bb.band_periods(work, "_shuf", lo, hi)
            if len(p) > 2:
                best = max(best, float(_excess(p).mean()))
        best_null.append(best)
    best_null = np.array(best_null)
    observed = float(ex.mean())
    print(f"  permutation test: best-of-{n_bands} band search on shuffled "
          f"scores, 200 draws")
    print(f"    observed best band excess/period  {observed:+.4f} "
          f"({ann(observed):+.2%}/yr)")
    print(f"    null best-of-{n_bands}, median          "
          f"{np.median(best_null):+.4f} ({ann(np.median(best_null)):+.2%}/yr)")
    print(f"    null 95th percentile              "
          f"{np.percentile(best_null, 95):+.4f}")
    print(f"    permutation p                     "
          f"{float((best_null >= observed).mean()):.4f}")


def check_concentration(df: pd.DataFrame, score: str) -> None:
    """How much of the band's profit comes from how few positions."""
    print("\n===== 2. P&L CONCENTRATION =====")
    base = int(df["entry_idx"].min())
    work = df.copy()
    work["period"] = (work["entry_idx"] - base) // bb.HORIZON
    held = []
    for _, g in work.groupby("period"):
        if len(g) < 10:
            continue
        r = g[score].rank(pct=True)
        held.append(g[(r > BAND_LO) & (r <= BAND_HI)])
    if not held:
        print("  no positions")
        return
    h = pd.concat(held).dropna(subset=[f"fwd_{bb.HORIZON}"])
    pnl = h[f"fwd_{bb.HORIZON}"].sort_values(ascending=False)
    total = float(pnl.sum())
    print(f"  {len(pnl)} positions, summed raw return {total:+.2f}")
    for k in (1, 3, 5, 10, 25):
        n = max(1, int(len(pnl) * k / 100))
        share = float(pnl.iloc[:n].sum()) / total if total else float("nan")
        print(f"    top {k:2d}% of positions ({n:4d}) = {share:6.1%} of gross gain")
    print("  largest single positions:")
    top = h.loc[pnl.index[:5], ["ticker", "entry_day", f"fwd_{bb.HORIZON}"]]
    print(top.to_string(index=False))


def check_price_floor(df: pd.DataFrame, score: str) -> None:
    """Does the band survive dropping stocks too cheap to trade honestly?"""
    print("\n===== 3. PRICE FLOOR =====")
    print("  floor   periods  avg_held   vs_SPY    p      vs_IWM   yrs_IWM")
    for floor in (0.0, 1.0, 3.0, 5.0, 10.0):
        sub = df[df["entry_open"] >= floor]
        per = bb.band_periods(sub, score, BAND_LO, BAND_HI)
        if len(per) < 3:
            print(f"  ${floor:<6.0f} too few periods")
            continue
        ex_s, ex_i = _excess(per, "SPY"), _excess(per, "IWM")
        yrs = per.assign(e=ex_i).groupby("year")["e"].mean()
        print(f"  ${floor:<6.0f} {len(per):7d}  {per['n_held'].mean():7.1f}  "
              f"{ann(ex_s.mean()):+7.2%}  "
              f"{float(stats.ttest_1samp(ex_s, 0).pvalue):.3f}  "
              f"{ann(ex_i.mean()):+7.2%}   "
              f"{int((yrs > 0).sum())}/{len(yrs)}")


def check_split_half(df: pd.DataFrame, members: dict) -> None:
    """Two disjoint seed sets, two independent ensembles, same question."""
    print("\n===== 4. SPLIT-HALF ENSEMBLES =====")
    from tools.ship_candidate import rank_average

    halves = {
        "seeds 0-4": [members[s] for s in range(5)],
        "seeds 5-9": [members[s] for s in range(5, 10)],
    }
    for name, cols in halves.items():
        work = df.copy()
        work["_half"] = rank_average(cols)
        per = bb.band_periods(work, "_half", BAND_LO, BAND_HI)
        ex_s, ex_i = _excess(per, "SPY"), _excess(per, "IWM")
        yrs = per.assign(e=ex_s).groupby("year")["e"].mean()
        print(f"  {name}: vs SPY {ann(ex_s.mean()):+7.2%}  "
              f"p={float(stats.ttest_1samp(ex_s, 0).pvalue):.3f}  "
              f"vs IWM {ann(ex_i.mean()):+7.2%}  "
              f"years positive {int((yrs > 0).sum())}/{len(yrs)}")
    agree = float(
        pd.concat(
            [rank_average(halves["seeds 0-4"]), rank_average(halves["seeds 5-9"])],
            axis=1,
        ).corr(method="spearman").iloc[0, 1]
    )
    print(f"  rank correlation between the two halves: {agree:+.3f}")


def check_leave_one_year_out(df: pd.DataFrame, score: str) -> None:
    print("\n===== 5. LEAVE ONE YEAR OUT =====")
    per = bb.band_periods(df, score, BAND_LO, BAND_HI)
    ex = _excess(per, "SPY")
    full = ann(ex.mean())
    print(f"  full sample vs SPY {full:+.2%}/yr")
    for y in sorted(per["year"].unique()):
        keep = per[per["year"] != y]
        e = _excess(keep, "SPY")
        print(f"    drop {y}: {ann(e.mean()):+7.2%}/yr   "
              f"(that year alone: {ann(_excess(per[per.year == y]).mean()):+7.2%})")


def check_bootstrap(df: pd.DataFrame, score: str) -> None:
    print("\n===== 6. BOOTSTRAP OVER PERIODS =====")
    per = bb.band_periods(df, score, BAND_LO, BAND_HI)
    rng = np.random.default_rng(0)
    for bench in ("SPY", "IWM"):
        ex = _excess(per, bench).to_numpy()
        draws = rng.choice(ex, size=(5000, len(ex)), replace=True).mean(axis=1)
        print(f"  vs {bench}: mean {ann(ex.mean()):+7.2%}/yr   "
              f"95% CI [{ann(np.percentile(draws, 2.5)):+7.2%}, "
              f"{ann(np.percentile(draws, 97.5)):+7.2%}]   "
              f"P(<=0)={float((draws <= 0).mean()):.4f}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default=os.path.join(
        REPO_ROOT, "research_data", "research_10861rows_20260813.parquet"))
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--band-lo", type=float, default=DEFAULT_BAND_LO)
    ap.add_argument("--band-hi", type=float, default=DEFAULT_BAND_HI)
    args = ap.parse_args(argv)

    global BAND_LO, BAND_HI
    BAND_LO, BAND_HI = args.band_lo, args.band_hi

    from tools.ship_candidate import SHIP_CANDIDATE, rank_average

    df = lab.load_dataset(args.dataset)
    cand = next(c for c in lab.CANDIDATES if c.name == SHIP_CANDIDATE)
    members = {}
    for s in range(args.seeds):
        members[s] = sl.build_oof(df, cand, horizon=lab.HORIZON, seed=s)
        print(f"  fitted member seed={s}", flush=True)
    df["ens"] = rank_average(list(members.values()))
    df = bb.attach_benchmarks(df, bb.HORIZON)

    print(f"\nBand under test: {BAND_LO:.0%}-{BAND_HI:.0%} of "
          f"{SHIP_CANDIDATE}, {args.seeds}-seed rank ensemble")

    check_multiple_comparisons(df, "ens")
    check_concentration(df, "ens")
    check_price_floor(df, "ens")
    check_split_half(df, members)
    check_leave_one_year_out(df, "ens")
    check_bootstrap(df, "ens")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
