"""Pre-registered candidate scores, and the runner that grades all of them.

READ THIS BEFORE ADDING A CANDIDATE
===================================
Every idea below was written down BEFORE any of them was run, and every one
is reported afterwards whether it worked or not. That protocol is not
ceremony: RESEARCH_NOTES.md records that ~40 configurations were tried by
hand here and that the best of them cleared +15%/yr on nothing but a lucky
random seed. With enough candidates and a private stopping rule, something
always looks good. Reporting the whole list is what keeps the winner
meaningful.

If you add a candidate later, add it to the registry, re-run everything, and
report the full table again. Do not add a candidate, like the result, and
quote it on its own.

WHAT IS BEING RANKED
--------------------
One row = one cluster episode (>=2 distinct insiders buying the same issuer
inside a rolling 14-day window). The score orders those episodes. It is
graded on log excess over SPY 21 trading days after entry -- 21 days because
that is the horizon at which a ranking was previously found to exist, and log
excess because arithmetic excess does not aggregate honestly (see
score_lab.log_excess).
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

from tools import data_paths  # noqa: E402

from tools import score_lab as sl  # noqa: E402

HORIZON = 21
GRADE_LABEL = "logex_21"


# --------------------------------------------------------------------------
# Dataset assembly
# --------------------------------------------------------------------------

def load_dataset(path: str, meta_path: str | None = None) -> pd.DataFrame:
    """The research parquet, plus the columns the candidates need to exist.

    Adds three derived columns and nothing else:
      `logex_21`   the grading label (see module docstring)
      `sic_major`  2-digit SIC industry, joined from the SEC submissions API
                   cache. Coarse on purpose -- reassignment at 2 digits is
                   rare, so using today's classification for an old event is
                   only mildly anachronistic. The 4-digit code and `exchange`
                   are NOT joined: exchange is current-only, and uplisting is
                   exactly the kind of event that moves a stock, so it would
                   leak.
      `month`      calendar month of the event, the cohort key everything
                   relative is computed within.
    """
    df = pd.read_parquet(path).reset_index(drop=True)
    df["event_day"] = pd.to_datetime(df["event_day"])
    df[GRADE_LABEL] = sl.log_excess(df, HORIZON)
    df["month"] = df["event_day"].dt.to_period("M").astype(str)

    meta_path = meta_path or os.path.join(
        REPO_ROOT, "research_data", "issuer_meta.parquet"
    )
    if os.path.exists(meta_path):
        meta = pd.read_parquet(meta_path)[["cik", "sic_major"]]
        meta = meta.rename(columns={"cik": "issuer_cik"})
        df = df.merge(meta, on="issuer_cik", how="left")
    else:
        df["sic_major"] = np.nan
    df["sic_major"] = df["sic_major"].fillna("NA").astype(str)
    return df


def _vol_bucket(df: pd.DataFrame, n: int = 5) -> pd.Series:
    """Volatility quintile, computed WITHIN each month.

    Within-month, not pooled, because market-wide volatility moves so much
    between 2019 and 2020 that a pooled quintile would mostly encode the date.
    """
    def q(g):
        try:
            return pd.qcut(g, n, labels=False, duplicates="drop")
        except ValueError:
            return pd.Series(0, index=g.index)

    return (
        df.groupby("month")["x_vol_63_ann"]
        .transform(q)
        .fillna(-1)
        .astype(int)
    )


# --------------------------------------------------------------------------
# Training targets
# --------------------------------------------------------------------------

def t_adj21(df):
    return df["adj_21"].astype(float)


def t_logex21(df):
    return df[GRADE_LABEL].astype(float)


def t_sector_rel(df):
    """Log excess judged against the same industry in the same month.

    The reasoning: a January 2021 biotech that returned +4% did NOT show
    insider skill if every biotech returned +9% that month. Removing the
    industry-month centre leaves the part of the return the insiders might
    plausibly have known something about.
    """
    return sl.cohort_demean(df[GRADE_LABEL], [df["sic_major"], df["month"]])


def t_month_rel(df):
    """Same idea with the industry left in -- isolates the sector's contribution."""
    return sl.cohort_demean(df[GRADE_LABEL], [df["month"]])


def t_sector_vol_rel(df):
    """Industry, month AND volatility bucket removed from the target.

    The most aggressive de-factoring tested. If a score still ranks after its
    target has had the two factors known to contaminate this dataset
    subtracted out, whatever is left is much harder to dismiss as a factor in
    disguise.
    """
    return sl.cohort_demean(
        df[GRADE_LABEL], [df["sic_major"], df["month"], _vol_bucket(df)]
    )


def t_downside21(df):
    """P(the position does not lose 15% in three weeks).

    At 63 days the equivalent (P(adj_63 > -0.30)) was the only objective that
    stayed positive in every out-of-sample year while all four return-shaped
    objectives collapsed in 2021. This is that idea moved to the horizon where
    a ranking is known to exist.
    """
    y = df["adj_21"].astype(float)
    return (y > -0.15).astype(float).where(y.notna())


def t_upside21(df):
    """P(the position beats SPY by 10% in three weeks) -- the mirror of the above.

    Included because the shipped score is exactly this shape at 63 days and is
    measured harmful. Running it here separates "the objective is wrong" from
    "the horizon is wrong".
    """
    y = df["adj_21"].astype(float)
    return (y > 0.10).astype(float).where(y.notna())


def t_rank_within_month(df):
    """Within-month relevance grade 0-4, for the learning-to-rank objective.

    LambdaRank needs a small non-negative integer relevance per row and a
    query group. Quintile of the sector-relative label inside the month is
    that grade, so the model optimises the ORDER of this month's candidates --
    which is the only ordering a screener can act on, and the exact quantity
    the monthly-cohort IC scores.
    """
    rel = t_sector_rel(df)
    frame = pd.DataFrame({"v": rel.to_numpy(), "m": df["month"].to_numpy()})

    def q(g):
        try:
            return pd.qcut(g, 5, labels=False, duplicates="drop")
        except ValueError:
            return pd.Series(2, index=g.index)

    out = frame.groupby("m")["v"].transform(q)
    return pd.Series(out.fillna(2).to_numpy(), index=df.index)


# --------------------------------------------------------------------------
# Feature subsets
# --------------------------------------------------------------------------

#: The 12 price/momentum/volatility columns. An earlier finding is that
#: dropping these collapses the ranking -- i.e. the score is mostly price
#: context, not insider quality. Candidate C9 re-measures that claim.
PRICE_FEATURES = [
    "x_buy_value_to_adv", "x_log_adv20", "x_drawdown_252", "x_drawdown_63",
    "x_mom_21_skip5", "x_mom_63_skip5", "x_mom_252_skip5", "x_vol_21_ann",
    "x_vol_63_ann", "x_price_to_sma200", "x_tx_volume_vs_adv",
    "x_entry_vs_insider_vwap",
]

#: The 12 columns that CANNOT be computed when the live screener runs: three
#: owner-history features need a full offline day-by-day replay, and nine
#: concurrent-selling features have no live plumbing. A score that leans on
#: these ranks beautifully in the lab and cannot be reproduced in run.bat.
NEVER_LIVE_FEATURES = [
    "x_owner_prior_buys_wmean", "x_owner_prior_adj63_mean",
    "x_owner_same_month_frac",
    "x_sell_n_cluster", "x_sell_n_insiders_cluster",
    "x_log1p_sell_value_cluster", "x_sell_n_trail90",
    "x_sell_n_insiders_trail90", "x_log1p_sell_value_trail90",
    "x_buy_sell_balance_cluster", "x_buyer_also_sold_nearby",
    "x_officer_or_director_sold_cluster",
]


def f_all(df):
    return sl.feature_cols(df)


def f_no_price(df):
    return [c for c in sl.feature_cols(df) if c not in PRICE_FEATURES]


def f_live_only(df):
    return [c for c in sl.feature_cols(df) if c not in NEVER_LIVE_FEATURES]


# --------------------------------------------------------------------------
# Helpers for the risk-adjusted targets
# --------------------------------------------------------------------------

#: Volatilities below this are treated as this. Dividing a return by a
#: near-zero volatility produces an enormous target value for what is usually
#: a data artifact (a barely-traded ticker whose price did not move), and a
#: quantile loss would then chase those rows.
VOL_FLOOR = 0.15


def _vol_floor(df: pd.DataFrame) -> pd.Series:
    v = pd.to_numeric(df["x_vol_63_ann"], errors="coerce")
    return v.clip(lower=VOL_FLOOR).fillna(VOL_FLOOR)


def _winsor(s: pd.Series, lo: float = 0.02, hi: float = 0.98) -> pd.Series:
    """Pull both tails of the TARGET in to its own 2nd/98th percentiles.

    Applied to the training target only, never to the label anything is graded
    on. The intent is to stop a handful of 1,000% outcomes dominating the loss
    surface; hiding them from the evaluation would be a different and much
    worse thing to do.
    """
    ok = s.dropna()
    if ok.empty:
        return s
    return s.clip(lower=ok.quantile(lo), upper=ok.quantile(hi))


# --------------------------------------------------------------------------
# The pre-registered registry
# --------------------------------------------------------------------------

CANDIDATES: list[sl.Candidate] = [
    # --- reproductions of what is already known, as controls -------------
    sl.Candidate(
        name="C1_quantile_adj21",
        target=t_adj21,
        objective="quantile",
        alpha=0.45,
        notes="Known-good baseline: quantile regression at 21 days on the "
              "raw arithmetic excess. Reproduces the earlier T1 result.",
    ),
    sl.Candidate(
        name="C2_quantile_sector_rel",
        target=t_sector_rel,
        objective="quantile",
        alpha=0.45,
        notes="Reproduces T4, the best score previously measured here: same "
              "model, target judged against industry-and-month.",
    ),
    # --- new targets -----------------------------------------------------
    sl.Candidate(
        name="C3_quantile_logexcess",
        target=t_logex21,
        objective="quantile",
        alpha=0.45,
        notes="Train on the aggregation-honest label instead of arithmetic "
              "excess. Isolates how much the label's own definition mattered.",
    ),
    sl.Candidate(
        name="C4_lambdarank_month",
        target=t_rank_within_month,
        objective="lambdarank",
        grouped_by_month=True,
        notes="NEW. Learning to rank with the month as the query group, so "
              "the loss optimises within-month ordering directly rather than "
              "predicting a number and hoping the order falls out.",
    ),
    sl.Candidate(
        name="C5_quantile_month_rel",
        target=t_month_rel,
        objective="quantile",
        alpha=0.45,
        notes="Month-relative but industry-blind. Read against C2, this "
              "isolates how much of C2 comes from the industry adjustment.",
    ),
    sl.Candidate(
        name="C6_quantile_sector_vol_rel",
        target=t_sector_vol_rel,
        objective="quantile",
        alpha=0.45,
        notes="NEW. Industry, month and volatility bucket all removed from "
              "the target, so the model cannot win by re-learning either "
              "known factor.",
    ),
    sl.Candidate(
        name="C7_downside_21d",
        target=t_downside21,
        objective="binary",
        notes="NEW at this horizon. P(no 15% loss in 21 days). The 63-day "
              "version was the only objective that stayed positive in every "
              "out-of-sample year.",
    ),
    sl.Candidate(
        name="C8_upside_21d",
        target=t_upside21,
        objective="binary",
        notes="The shipped score's shape (a right-tail probability) moved to "
              "21 days. Separates a wrong objective from a wrong horizon.",
    ),
    sl.Candidate(
        name="C9_median_l2_sector_rel",
        target=t_sector_rel,
        objective="l2",
        notes="C2's target with an ordinary squared-error loss, to confirm "
              "the quantile objective is doing real work and is not "
              "decoration.",
    ),
    # --- feature-set ablations ------------------------------------------
    sl.Candidate(
        name="C10_no_price_features",
        target=t_sector_rel,
        objective="quantile",
        alpha=0.45,
        features=f_no_price,
        notes="C2 with the 12 price/momentum/vol columns removed. Answers "
              "the uncomfortable question of whether ANY of this is about "
              "insider behaviour.",
    ),
    sl.Candidate(
        name="C11_live_features_only",
        target=t_sector_rel,
        objective="quantile",
        alpha=0.45,
        features=f_live_only,
        notes="C2 restricted to the 47 features the live screener can "
              "actually compute. This is the one that decides what run.bat "
              "can ship, so it matters more than C2's own number.",
    ),

    # --- round two ------------------------------------------------------
    # Added after round one, and the whole registry was re-run and re-
    # reported rather than these being quoted on their own. Round one said
    # month-demeaning the target helps and adding industry on top of it
    # hurts, which is the opposite of what the earlier notes concluded; these
    # follow that up and check it survives the live feature restriction.
    sl.Candidate(
        name="C12_month_rel_live_only",
        target=t_month_rel,
        objective="quantile",
        alpha=0.45,
        features=f_live_only,
        notes="THE SHIPPABLE ONE. Round one's best target, restricted to the "
              "47 features run.bat can compute. If this fails, nothing "
              "measured in this lab can reach a user.",
    ),
    sl.Candidate(
        name="C13_logexcess_live_only",
        target=t_logex21,
        objective="quantile",
        alpha=0.45,
        features=f_live_only,
        notes="The runner-up target under the same live restriction.",
    ),
    sl.Candidate(
        name="C14_month_vol_rel",
        target=lambda df: sl.cohort_demean(
            df[GRADE_LABEL], [df["month"], _vol_bucket(df)]
        ),
        objective="quantile",
        alpha=0.45,
        notes="Month and volatility removed, industry left in. Read against "
              "C6, separates whether C6's damage came from the volatility "
              "adjustment or from the industry one.",
    ),
    sl.Candidate(
        name="C15_month_rel_a35",
        target=t_month_rel,
        objective="quantile",
        alpha=0.35,
        notes="Alpha sensitivity below the median. A result that only exists "
              "at one alpha is a tuning artifact.",
    ),
    sl.Candidate(
        name="C16_month_rel_a55",
        target=t_month_rel,
        objective="quantile",
        alpha=0.55,
        notes="Alpha sensitivity above the median.",
    ),

    # --- round three ----------------------------------------------------
    # Round two found a clean, monotone gradient in the quantile alpha:
    # 0.35 > 0.45 > 0.55, with 0.55 failing outright. That is not a tuning
    # wiggle, it is a mechanism -- a lower alpha puts the loss on the left
    # tail, so the model is being paid to know which of these buys goes
    # badly. These candidates push on that and re-check it under the live
    # feature restriction.
    sl.Candidate(
        name="C17_month_rel_a35_live",
        target=t_month_rel,
        objective="quantile",
        alpha=0.35,
        features=f_live_only,
        notes="Round two's best target and alpha, restricted to what run.bat "
              "can compute. The leading ship candidate.",
    ),
    sl.Candidate(
        name="C18_month_rel_a25_live",
        target=t_month_rel,
        objective="quantile",
        alpha=0.25,
        features=f_live_only,
        notes="Pushes the tilt further into the left tail, to find where the "
              "gradient stops helping.",
    ),
    # --- round four: targets aimed at RISK-ADJUSTED return -------------
    # Round three's winner was selected on rank IC, which scores ordering and
    # says nothing about what a holder experiences. Measured afterwards, that
    # score's Sharpe by decile runs 0.21 to 1.33 -- a strong risk-adjusted
    # gradient that the IC table could not show. These candidates aim at that
    # quantity directly instead of discovering it by accident, and they are
    # graded on Sharpe in tools/run_sharpe_search.py as well as on IC here.
    sl.Candidate(
        name="S1_vol_scaled_excess",
        target=lambda df: t_logex21(df) / _vol_floor(df),
        objective="quantile",
        alpha=0.35,
        features=f_live_only,
        notes="Per-trade Sharpe proxy: log excess divided by the event's own "
              "ex-ante annualised volatility. The most direct statement of "
              "'return per unit of risk taken' available before the trade.",
    ),
    sl.Candidate(
        name="S2_vol_scaled_month_rel",
        target=lambda df: sl.cohort_demean(t_logex21(df), [df["month"]])
                          / _vol_floor(df),
        objective="quantile",
        alpha=0.35,
        features=f_live_only,
        notes="S1 with the month removed first, so the model is not rewarded "
              "for knowing which months were calm.",
    ),
    sl.Candidate(
        name="S3_prob_beats_spy",
        target=lambda df: (df["adj_21"] > 0).astype(float)
                          .where(df["adj_21"].notna()),
        objective="binary",
        features=f_live_only,
        notes="P(beats SPY over 21 days). The plainest reading of 'does it go "
              "up' relative to the alternative of just owning the index.",
    ),
    sl.Candidate(
        name="S4_prob_up_absolute",
        target=lambda df: (df["fwd_21"] > 0).astype(float)
                          .where(df["fwd_21"].notna()),
        objective="binary",
        features=f_live_only,
        notes="P(the position is up at all in 21 days), ignoring the "
              "benchmark. Included because a high win rate is what actually "
              "drives Sharpe when the payoff is this skewed.",
    ),
    sl.Candidate(
        name="S5_winsorized_month_vol_rel",
        target=lambda df: _winsor(
            sl.cohort_demean(t_logex21(df), [df["month"], _vol_bucket(df)])
        ),
        objective="quantile",
        alpha=0.35,
        features=f_live_only,
        notes="The shipped target with the extreme 2% of each tail pulled in. "
              "Tests whether the handful of enormous winners were teaching "
              "the model anything, or just adding variance to its loss.",
    ),
    sl.Candidate(
        name="C19_month_vol_rel_a35_live",
        target=lambda df: sl.cohort_demean(
            df[GRADE_LABEL], [df["month"], _vol_bucket(df)]
        ),
        objective="quantile",
        alpha=0.35,
        features=f_live_only,
        notes="Adds the volatility de-factoring to the ship candidate. Its "
              "all-feature form had the widest seed spread of any passing "
              "candidate, so it is being watched for exactly that.",
    ),
]


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------

def run(
    df: pd.DataFrame,
    candidates,
    *,
    seeds=(0,),
    top_n: int = 10,
) -> tuple[pd.DataFrame, dict]:
    """Score every candidate under every seed and grade all of them."""
    rows = []
    scores: dict[str, pd.Series] = {}
    for cand in candidates:
        for seed in seeds:
            key = f"{cand.name}__s{seed}"
            t0 = time.time()
            s = sl.build_oof(df, cand, horizon=HORIZON, seed=seed)
            df[key] = s
            scores[key] = s
            a = sl.audit(
                df, key, GRADE_LABEL, horizon=HORIZON, top_n=top_n, name=key
            )
            rows.append(
                dict(
                    candidate=cand.name,
                    seed=seed,
                    ic=a.ic_mean,
                    ic_t=a.ic_t,
                    p_le0=a.ic_p_le_zero,
                    yrs_pos=f"{a.years_positive}/{a.years_total}",
                    vol_neutral=a.vol_neutral,
                    lowvol_ref=a.lowvol_benchmark,
                    dec_mono=a.decile_rank_corr,
                    top_med=a.top_decile_median,
                    bot_med=a.bottom_decile_median,
                    port_ann=a.port_excess_ann,
                    port_p=a.port_p,
                    gate="PASS" if a.passes() else "FAIL",
                    secs=round(time.time() - t0, 1),
                )
            )
            print(sl.format_audit(a), flush=True)
    return pd.DataFrame(rows), scores


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dataset",
        default=data_paths.latest_research_dataset(),
    )
    ap.add_argument("--seeds", default="0", help="comma list of RNG seeds")
    ap.add_argument("--only", default="", help="comma list of candidate names")
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--out", default="", help="write the summary table here")
    args = ap.parse_args(argv)

    df = load_dataset(args.dataset)
    print(f"{len(df)} rows, {df.event_day.min():%Y-%m-%d}..{df.event_day.max():%Y-%m-%d}")
    print(f"grading on {GRADE_LABEL}, {df[GRADE_LABEL].notna().sum()} labelled\n")

    cands = CANDIDATES
    if args.only:
        want = {n.strip() for n in args.only.split(",")}
        cands = [c for c in CANDIDATES if c.name in want]
    seeds = tuple(int(s) for s in args.seeds.split(","))

    table, _ = run(df, cands, seeds=seeds, top_n=args.top_n)
    print("\n\n===== SUMMARY (every candidate, pass or fail) =====")
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(table.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    if args.out:
        table.to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
