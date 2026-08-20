"""Fit each candidate as a seed ensemble, then grade it on RISK-ADJUSTED return.

WHAT THIS ANSWERS THAT tools/run_score_lab.py DOES NOT
======================================================
run_score_lab grades a score on rank IC -- whether it orders events correctly.
A holder does not experience an ordering; they experience a return stream with
a volatility and a drawdown. The two come apart badly here: the shipped score
looks almost flat on median excess over SPY and yet its Sharpe by decile runs
0.21 to 1.33. Grading only on IC found the right score for the wrong stated
reason, and would have missed a better one aimed at risk-adjusted return.

So every candidate here is fit the same way (10-seed rank-averaged ensemble,
walk-forward out of fold) and then run through the SAME book-recipe grid, and
each one gets its OWN permutation test.

WHY EACH CANDIDATE IS PERMUTED SEPARATELY
-----------------------------------------
The recipe grid is a search: 132 combinations of band, weighting, price floor
and position cap, best Sharpe reported. On a fat-tailed return distribution a
search like that finds impressive numbers in pure noise -- the return-based
version of it produced +18.46%/yr from shuffled scores. Permuting each
candidate separately means a candidate is judged against what ITS OWN search
finds in noise, and it means several candidates passing independently is
evidence, whereas one candidate passing out of six is a sixth search.

Read the output as: a candidate is interesting only if its observed Sharpe
clears its own null's 95th percentile, and it is only believable if the
mechanism for why is stated.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools import run_score_lab as lab  # noqa: E402
from tools import run_sharpe_lab as rsl  # noqa: E402
from tools import score_lab as sl  # noqa: E402
from tools import sharpe_lab as sh  # noqa: E402
from tools.ship_candidate import rank_average  # noqa: E402

DEFAULT_CANDIDATES = [
    "C19_month_vol_rel_a35_live",   # the incumbent, chosen on IC
    "S1_vol_scaled_excess",
    "S2_vol_scaled_month_rel",
    "S3_prob_beats_spy",
    "S4_prob_up_absolute",
    "S5_winsorized_month_vol_rel",
    "C17_month_rel_a35_live",       # runner-up on IC, as a control
]


def build_ensemble(df: pd.DataFrame, name: str, seeds: int) -> pd.Series:
    cand = next(c for c in lab.CANDIDATES if c.name == name)
    members = [
        sl.build_oof(df, cand, horizon=lab.HORIZON, seed=s) for s in range(seeds)
    ]
    return rank_average(members)


def grade(df: pd.DataFrame, score_col: str, draws: int) -> dict:
    """Best recipe by Sharpe, its full risk report, and its permutation p."""
    panel = sh.build_panel(df.dropna(subset=[score_col]), score_col)

    best_spec, best_metrics = None, None
    for spec in rsl.grid(score_col):
        per = sh.panel_returns(panel, spec)
        if len(per) < 20:
            continue
        m = sh.evaluate(per, spec.label())
        if not m or not np.isfinite(m.get("sharpe", np.nan)):
            continue
        if best_metrics is None or m["sharpe"] > best_metrics["sharpe"]:
            best_spec, best_metrics = spec, m

    if best_metrics is None:
        return {}
    perm = sh.permutation_test_panel(
        panel, rsl.best_sharpe, best_metrics["sharpe"], n_draws=draws
    )
    out = dict(best_metrics)
    out["recipe"] = best_spec.label()
    out.update(
        null_median=perm.get("null_median", float("nan")),
        null_p95=perm.get("null_p95", float("nan")),
        perm_p=perm.get("p_value", float("nan")),
    )
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default=os.path.join(
        REPO_ROOT, "research_data", "research_10861rows_20260813.parquet"))
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--draws", type=int, default=120)
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    df = lab.load_dataset(args.dataset)
    names = (
        [n.strip() for n in args.only.split(",")] if args.only
        else DEFAULT_CANDIDATES
    )

    score_cols = []
    for name in names:
        t0 = time.time()
        df[name] = build_ensemble(df, name, args.seeds)
        score_cols.append(name)
        print(f"  fitted {name} ({args.seeds} seeds, {time.time() - t0:.0f}s)",
              flush=True)

    prepared = sh.prepare(df, args.dataset, score_cols=("C19_month_vol_rel_a35_live",))
    # `prepare` drops rows missing the reference score; re-attach every
    # candidate's score by the same (ticker, event_day) key so all candidates
    # are graded over an IDENTICAL set of rows and periods. Comparing scores
    # measured over different row sets would make the comparison meaningless.
    key = ["ticker", "event_day"]
    prepared = prepared.merge(df[key + score_cols], on=key, how="left",
                              suffixes=("", "_dup"))
    for c in score_cols:
        dup = f"{c}_dup"
        if dup in prepared.columns:
            prepared[c] = prepared[dup]
            prepared = prepared.drop(columns=[dup])

    rows = []
    for name in names:
        t0 = time.time()
        r = grade(prepared, name, args.draws)
        if r:
            r["candidate"] = name
            rows.append(r)
            print(f"  graded {name}: Sharpe {r['sharpe']:+.3f} "
                  f"(null p95 {r['null_p95']:+.3f}, p={r['perm_p']:.4f}) "
                  f"[{time.time() - t0:.0f}s]", flush=True)

    table = pd.DataFrame(rows)
    show = ["candidate", "recipe", "avg_names", "ann_return", "ann_vol",
            "sharpe", "sortino", "max_drawdown", "win_rate", "excess_SPY",
            "yrs_SPY", "excess_IWM", "yrs_IWM", "null_median", "null_p95",
            "perm_p"]
    print("\n\n===== EVERY CANDIDATE, GRADED ON RISK-ADJUSTED RETURN =====")
    with pd.option_context("display.width", 260, "display.max_columns", 40):
        print(table[show].sort_values("sharpe", ascending=False)
              .to_string(index=False, float_format=lambda v: f"{v:+.3f}"))

    print("\n===== BENCHMARKS, same periods =====")
    panel = sh.build_panel(
        prepared.dropna(subset=["C19_month_vol_rel_a35_live"]),
        "C19_month_vol_rel_a35_live",
    )
    ref = sh.panel_returns(panel, sh.BookSpec(lo=0.0, hi=1.0))
    print(pd.DataFrame([sh.benchmark_row(ref, b) for b in ("SPY", "IWM")])
          .to_string(index=False, float_format=lambda v: f"{v:+.3f}"))

    if args.out:
        prepared.to_parquet(args.out, index=False)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
