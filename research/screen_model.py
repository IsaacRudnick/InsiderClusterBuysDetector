"""The score the live screener sorts by.

WHAT THIS IS
============
A seed ensemble of quantile-regression rankers, fit at a 21-trading-day
horizon against a month-and-volatility-relative target, using only the
features the live screener can actually compute. It replaces
`oof_tail_classifier` as the production score.

Chosen out of 19 pre-registered candidates in `tools/run_score_lab.py`, all
of which were run and all of which are reported in RESEARCH_NOTES.md. The
selection was made on the ranking gauntlet in `tools/score_lab.py`, never on
a portfolio return -- see WHAT THIS IS NOT.

The four choices, each with its reason:

  21 trading days      The horizon at which a ranking exists. At 63 days the
                       label's right tail is wide enough that the ordering
                       drowns; at 10 days there is barely anything to order.

  quantile, alpha 0.35 Not a mean. The alpha gradient is monotone -- 0.25 and
                       0.35 pass the gauntlet, 0.45 is weaker, 0.55 fails
                       outright -- and it points one way: a low alpha puts the
                       loss on the LEFT tail. What this data supports
                       predicting is which insider buys go badly.

  month-and-vol-       The target is each event's log excess minus the median
  relative target      log excess of the events in the same calendar month AND
                       the same within-month volatility quintile. Both
                       subtractions remove a factor the model would otherwise
                       be rewarded for re-learning: "which month was good" is
                       not a stock-picking skill, and "which names are risky"
                       is available for free by sorting on volatility. What is
                       left is the part that has to be earned.

  live features only   The 12 columns the screener cannot compute at run time
                       (3 owner-history, 9 concurrent-selling) are excluded
                       from the fit. A model that leans on inputs production
                       never has ranks beautifully in the lab and cannot be
                       reproduced by the thing users run.

WHY AN ENSEMBLE, NOT ONE FIT
----------------------------
The most expensive finding in this project's history is that a book built
from a single fit swung +7.98%/yr to +0.81%/yr on the RNG seed alone.
Averaging ranks across seeds does not make a weak score strong; it makes the
score's composition reproducible, so the ordering a user sees today is the
ordering the research measured. Measured here: two disjoint 5-seed halves
rank-correlate +0.964, and all 10 members individually score 7/7 years
positive.

Ranks are averaged, not raw predictions. The members are quantile
regressions whose raw output is a predicted return, so one member with a
wider spread would otherwise dominate the average for no good reason.

WHAT THIS IS NOT
================
It is NOT a claim that following it beats an index fund. A 70th-90th
percentile book measured +19.77%/yr over SPY and then failed its own audit: a
permutation test that re-runs the whole band search on shuffled scores
produces a null median of +18.46%/yr, i.e. p = 0.435. The return number is
the band search, not the signal.

What survived every audit is the RISK ordering: the chance of losing more
than 30% in three weeks falls monotonically across the deciles, sits between
7.3% and 9.2% in the bottom band in every single out-of-sample year, and
holds its shape under $3 and $5 entry-price floors. That is a left-tail
frequency over ~900 rows per decile rather than an average dragged by a
handful of winners, which is why it does not evaporate the way the return
number did.

Anything built on this module should say "these are the ones that have
historically gone wrong least often", never "these are the ones that go up".
"""

from __future__ import annotations

import datetime as _dt
import os
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

import lightgbm as lgb

from research import model as rm

# --------------------------------------------------------------------------
# Configuration -- the chosen candidate, in one place
# --------------------------------------------------------------------------

#: Trading days between entry and the label's exit.
SCREEN_HORIZON = 21

#: Quantile level. Below the median on purpose; see the module docstring.
SCREEN_ALPHA = 0.35

#: How many independently-seeded members the ensemble holds. Ten is where the
#: split-half rank correlation (+0.964 at five per half) stopped improving;
#: more members cost fit time and change nothing.
SCREEN_N_MEMBERS = 10

#: Volatility quintiles, cut WITHIN each month. Pooled quintiles would mostly
#: encode the date -- market-wide volatility in 2020 has little to do with
#: 2019's -- so the cut has to be local to the month it is neutralising.
SCREEN_VOL_BINS = 5

SCREEN_LGBM = dict(
    n_estimators=200,
    num_leaves=15,
    min_child_samples=20,
    learning_rate=0.05,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    verbosity=-1,
    n_jobs=-1,
    objective="quantile",
    alpha=SCREEN_ALPHA,
)


def screen_feature_cols() -> list[str]:
    """The 47 columns the live screener can compute, in a fixed order.

    Derived from `research.live_score`'s own categorisation rather than
    hard-coded, so that if a feature ever becomes live-computable (the nine
    concurrent-selling columns need only plumbing, not new data) this set
    follows automatically instead of silently going stale.
    """
    from research import live_score as ls  # local: live_score imports rm

    never_live = set(ls.OWNER_HISTORY_FEATURE_COLS) | set(ls.SALE_FEATURE_COLS)
    return [c for c in rm.FEATURE_COLS if c not in never_live]


# --------------------------------------------------------------------------
# The training target
# --------------------------------------------------------------------------

def log_excess(df: pd.DataFrame, horizon: int = SCREEN_HORIZON) -> pd.Series:
    """Log return of the position minus log return of SPY over the same days.

    Arithmetic excess is what one trade earns; log excess is the version that
    adds up across trades. Averaging arithmetic excess over these events gives
    +1.02% while the same events compound to -3.49% against SPY, so the
    arithmetic version flatters any aggregate built from it.
    """
    fwd = df[f"fwd_{horizon}"].astype(float)
    spy = df[f"spy_{horizon}"].astype(float)
    out = np.log1p(fwd.clip(lower=-0.999)) - np.log1p(spy.clip(lower=-0.999))
    return out.where(fwd.notna() & spy.notna())


def screen_target(
    df: pd.DataFrame, horizon: int = SCREEN_HORIZON
) -> pd.Series:
    """Log excess, minus the median log excess of its month-and-vol cohort.

    Median rather than mean as the cohort centre: one 1,000% winner in a
    cohort of forty would drag a mean far enough to flip the sign of every
    other member's target.

    No leakage: cohorts are keyed on calendar month, and the walk-forward
    folds cut on calendar year, so every row that contributes to a training
    row's cohort centre is itself in the training set.
    """
    day = pd.to_datetime(df["event_day"])
    month = day.dt.to_period("M").astype(str)
    vol = pd.to_numeric(df["x_vol_63_ann"], errors="coerce")

    def _q(g: pd.Series) -> pd.Series:
        try:
            return pd.qcut(g, SCREEN_VOL_BINS, labels=False, duplicates="drop")
        except ValueError:
            # A month with too few distinct volatilities to cut. One bucket is
            # the correct degenerate answer, not an error.
            return pd.Series(0, index=g.index)

    bucket = (
        vol.groupby(month).transform(_q).fillna(-1).astype(int)
    )
    frame = pd.DataFrame(
        {
            "v": log_excess(df, horizon).to_numpy(),
            "m": month.to_numpy(),
            "b": bucket.to_numpy(),
        }
    )
    centre = frame.groupby(["m", "b"], dropna=False)["v"].transform("median")
    return pd.Series((frame["v"] - centre).to_numpy(), index=df.index)


# --------------------------------------------------------------------------
# The ensemble
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ScreenEnsemble:
    """N fitted members plus the reference distribution each one is ranked in.

    `score_rows` returns the average of the members' percentile ranks, in
    0..1. Each member's raw prediction is converted to a rank against that
    member's OWN training predictions (`member_reference`) before averaging,
    which is what makes averaging meaningful across members whose raw outputs
    are on different scales.
    """

    members: list
    feature_cols: list[str]
    member_reference: list[np.ndarray]

    def score_rows(self, X: pd.DataFrame) -> np.ndarray:
        """Average percentile rank across members, one value per input row."""
        frame = pd.DataFrame(X, columns=self.feature_cols).astype(float)
        acc = np.zeros(len(frame), dtype=float)
        for member, ref in zip(self.members, self.member_reference):
            raw = np.asarray(member.predict(frame), dtype=float)
            acc += np.searchsorted(ref, raw, side="left") / max(len(ref), 1)
        return acc / max(len(self.members), 1)


def fit_screen_ensemble(
    df: pd.DataFrame,
    *,
    horizon: int = SCREEN_HORIZON,
    n_members: int = SCREEN_N_MEMBERS,
    feature_cols: Sequence[str] | None = None,
    lgbm_params: dict | None = None,
) -> ScreenEnsemble:
    """Fit every member on the full dataset.

    This is the PRODUCTION fit, so there is deliberately no holdout: a live
    caller scores rows the model has never seen by construction, because they
    have not happened yet. Out-of-sample evidence for this configuration comes
    from the walk-forward folds in `tools/score_lab.py`, not from here.
    """
    cols = list(feature_cols) if feature_cols else screen_feature_cols()
    params = dict(SCREEN_LGBM)
    params.update(lgbm_params or {})

    y = screen_target(df, horizon)
    usable = y.notna() & df["entry_idx"].notna()
    X = df.loc[usable, cols].astype(float)
    y = y.loc[usable]
    if len(X) < 500:
        raise ValueError(
            f"only {len(X)} usable training rows; refusing to fit a "
            f"production score on that little history"
        )

    members, refs = [], []
    for seed in range(n_members):
        m = lgb.LGBMRegressor(**dict(params, random_state=seed))
        m.fit(X, y)
        members.append(m)
        refs.append(np.sort(np.asarray(m.predict(X), dtype=float)))
    return ScreenEnsemble(
        members=members, feature_cols=cols, member_reference=refs
    )


def build_screen_bundle(
    ensemble: ScreenEnsemble, df: pd.DataFrame, *, source_path: str = ""
) -> rm.ProductionBundle:
    """Wrap the ensemble in the same ProductionBundle shape live_score reads.

    Reusing `research.model.ProductionBundle` rather than inventing a second
    artifact type keeps one loader, one saver and one provenance format in the
    codebase. The only difference a caller sees is that `model` here answers
    to `score_rows` instead of `predict_proba`; `research.live_score` branches
    on exactly that.

    `training_scores` is the ensemble's own score over every training row,
    sorted. It is the FIXED reference a live percentile is measured against,
    captured once here -- never recomputed from whatever clusters happen to be
    on screen, which would make a stock's rating depend on what else got
    scraped that morning.
    """
    y = screen_target(df, SCREEN_HORIZON)
    usable = y.notna() & df["entry_idx"].notna()
    X = df.loc[usable, ensemble.feature_cols].astype(float)
    scores = np.sort(ensemble.score_rows(X))

    day = pd.to_datetime(df.loc[usable, "event_day"])
    return rm.ProductionBundle(
        model=ensemble,
        feature_cols=list(ensemble.feature_cols),
        training_scores=scores,
        provenance=dict(
            source_dataset_path=source_path,
            n_rows=int(usable.sum()),
            event_day_min=str(day.min().date()),
            event_day_max=str(day.max().date()),
            fit_timestamp=_dt.datetime.now(_dt.timezone.utc).isoformat(),
            score_kind="screen_ensemble",
        ),
        config=dict(
            kind="screen_ensemble",
            horizon=SCREEN_HORIZON,
            alpha=SCREEN_ALPHA,
            n_members=len(ensemble.members),
            vol_bins=SCREEN_VOL_BINS,
            target="log excess vs SPY, median-demeaned within month x "
                   "within-month volatility quintile",
            lgbm=dict(SCREEN_LGBM),
        ),
    )


def is_screen_bundle(bundle: rm.ProductionBundle) -> bool:
    """True when `bundle` carries the ensemble rather than a single classifier.

    Duck-typed on the scoring method rather than on the provenance string, so
    an older bundle written before `score_kind` existed is still classified
    correctly by what it can actually do.
    """
    return hasattr(getattr(bundle, "model", None), "score_rows")


def default_bundle_path(df_rows: int, out_dir: str) -> str:
    stamp = _dt.datetime.now().strftime("%Y%m%d")
    return os.path.join(out_dir, f"screen_model_{df_rows}rows_{stamp}.joblib")
