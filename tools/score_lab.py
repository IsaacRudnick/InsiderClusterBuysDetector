"""Score laboratory: build candidate ranking scores out of fold, and put every
one of them through the SAME pre-registered gauntlet.

WHY THIS EXISTS
===============
RESEARCH_NOTES.md records that roughly 40 model configurations were tried by
hand, that one of them cleared +15%/yr, and that the number meant nothing --
it was the random seed. The lesson recorded there is not "try fewer things",
it is "fix the measurement before you try anything". This module is that
fixed measurement.

Two halves:

  build_oof(...)   -- fit ONE model per out-of-sample YEAR on an expanding
                      window, with a purge and an embargo, and return the
                      out-of-fold score for every row. Never fits on a row it
                      later scores.
  audit(...)       -- run the full gauntlet on a score column and return an
                      AuditResult. The gauntlet is fixed in code, identical
                      for every candidate, and reports every test it ran
                      including the ones the candidate failed.

THE GAUNTLET, and why each test is in it
----------------------------------------
1. Monthly-cohort IC. Spearman(score, label) computed WITHIN each calendar
   month, then averaged. Pooling across months lets a score win by knowing
   which months were good, which is not a stock-picking skill and is not
   tradeable. t-stat and a block bootstrap over months come with it.
2. Years positive. The same monthly IC, grouped by year. A score that works
   in 4 of 7 years is a regime bet.
3. Volatility-neutral IC. Spearman recomputed INSIDE each volatility
   quintile. This is the test that killed four earlier candidates which had
   LARGER headline IC: their ranking was the volatility factor wearing a hat.
   Reported alongside a plain "sort by low volatility" ranker so the
   comparison is head to head.
4. Decile medians. Medians, not means. The label has a 1000% right tail and
   on that the mean is noise -- reading pooled decile means at 63 days is the
   specific error that produced an earlier wrong conclusion.
5. Seed sweep. The same config refit under 5 RNG seeds. If the spread across
   seeds exceeds the effect, there is no effect. (Driven by the caller; see
   tools/run_score_lab.py.)
6. Top-N portfolio over NON-OVERLAPPING periods, with a p-value and a
   per-year column. Overlapping windows manufacture significance.

A candidate that clears every one of these is still only a ranking, not a
promise about money. See RESEARCH_NOTES.md.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import pandas as pd

# Python puts a script's OWN directory on sys.path[0], not the repo root, so a
# `python tools/score_lab.py` invocation cannot see its siblings without this.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import lightgbm as lgb  # noqa: E402
from scipy import stats  # noqa: E402

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

#: Trading days held between the entry and the exit the label measures.
DEFAULT_HORIZON = 21

#: Extra trading days dropped from the training window on top of the purge.
#: The purge alone only guarantees a training row's label window closed before
#: the test block opens; the embargo additionally buffers against short-range
#: serial correlation between neighbouring events in the same issuer.
DEFAULT_EMBARGO = 21

#: A fold is skipped rather than fit on too little history.
MIN_FOLD_TRAIN_ROWS = 400

DEFAULT_SEEDS = (0, 1, 2, 3, 4)

BASE_LGBM = dict(
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
)


def feature_cols(df: pd.DataFrame) -> list[str]:
    """Every model input column. Convention: model inputs are prefixed `x_`."""
    return [c for c in df.columns if c.startswith("x_")]


# --------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------

def log_excess(df: pd.DataFrame, horizon: int) -> pd.Series:
    """Log return of the position minus log return of SPY over the same window.

    Arithmetic excess (`adj_h`, what the dataset ships) is what a single trade
    earns, but it does not aggregate: averaging it across events overstates
    what a book compounding through those events would have made. The average
    `adj_63` is +1.02% while the same events compound to -3.49% against SPY.
    Log excess is the version that adds up, so it is what candidates are
    graded against here.
    """
    fwd = df[f"fwd_{horizon}"].astype(float)
    spy = df[f"spy_{horizon}"].astype(float)
    out = np.log1p(fwd.clip(lower=-0.999)) - np.log1p(spy.clip(lower=-0.999))
    return out.where(fwd.notna() & spy.notna())


def cohort_demean(values: pd.Series, keys: Sequence[pd.Series]) -> pd.Series:
    """Subtract each row's cohort median from it.

    Used to build a *relative* training target: judging a buy against the
    other buys in its own industry and month, rather than against the whole
    market. Median rather than mean, because a single 1000% winner would drag
    a mean cohort centre far enough to flip the sign of everything else in it.
    """
    frame = pd.DataFrame({"v": np.asarray(values, dtype=float)})
    kcols = []
    for i, k in enumerate(keys):
        col = f"k{i}"
        frame[col] = np.asarray(k)
        kcols.append(col)
    med = frame.groupby(kcols, dropna=False)["v"].transform("median")
    return pd.Series((frame["v"] - med).to_numpy(), index=values.index)


# --------------------------------------------------------------------------
# Out-of-fold score construction
# --------------------------------------------------------------------------

@dataclass
class FoldSpec:
    """One out-of-sample year and the training rows that may legally see it."""

    year: int
    train_idx: np.ndarray
    test_idx: np.ndarray


def make_year_folds(
    df: pd.DataFrame,
    *,
    horizon: int = DEFAULT_HORIZON,
    embargo: int = DEFAULT_EMBARGO,
    min_train_rows: int = MIN_FOLD_TRAIN_ROWS,
) -> list[FoldSpec]:
    """Expanding-window folds, one test block per calendar year.

    A training row survives only if its own label window closed strictly
    before the test year opened, with `embargo` further trading days of
    clearance:

        entry_idx + horizon + embargo  <  min(entry_idx of the test year)

    `entry_idx` is a position in a shared trading-day calendar, so this
    comparison is exact in trading days -- no calendar-day approximation of a
    trading-day horizon, which is where off-by-a-week leaks come from.
    """
    day = pd.to_datetime(df["event_day"])
    years = sorted(day.dt.year.unique())
    entry = df["entry_idx"].to_numpy()
    folds: list[FoldSpec] = []
    for y in years:
        test_mask = (day.dt.year == y).to_numpy()
        if not test_mask.any():
            continue
        test_start = entry[test_mask].min()
        train_mask = (entry + horizon + embargo) < test_start
        if int(train_mask.sum()) < min_train_rows:
            continue
        folds.append(
            FoldSpec(
                year=int(y),
                train_idx=np.flatnonzero(train_mask),
                test_idx=np.flatnonzero(test_mask),
            )
        )
    return folds


@dataclass
class Candidate:
    """One pre-registered idea: a target, an objective, and a feature set."""

    name: str
    #: Builds the TRAINING target from the dataset. May differ from the label
    #: the candidate is later graded on -- training on a sector-relative
    #: target and grading on the plain tradeable one is a legitimate and
    #: measured improvement, not a mismatch.
    target: Callable[[pd.DataFrame], pd.Series]
    objective: str = "quantile"
    alpha: float = 0.45
    features: Callable[[pd.DataFrame], list[str]] | None = None
    lgbm: dict = field(default_factory=dict)
    #: Learning-to-rank objectives need the month cohorts as query groups.
    grouped_by_month: bool = False
    notes: str = ""


def _fit_predict(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_te: pd.DataFrame,
    cand: Candidate,
    seed: int,
    groups_tr: np.ndarray | None,
) -> np.ndarray:
    params = dict(BASE_LGBM)
    params.update(cand.lgbm)
    params["random_state"] = seed

    if cand.objective == "binary":
        model = lgb.LGBMClassifier(**params)
        model.fit(X_tr, y_tr.astype(int))
        return model.predict_proba(X_te)[:, 1]

    if cand.objective == "lambdarank":
        # Learning to rank, with each calendar month as one query group. This
        # optimises the ordering WITHIN a month, which is exactly the quantity
        # the monthly-cohort IC measures and the only ordering a screener can
        # act on -- you choose among this month's candidates, not against
        # candidates from three years ago.
        params.pop("subsample", None)
        params.pop("subsample_freq", None)
        model = lgb.LGBMRanker(objective="lambdarank", **params)
        order = np.argsort(groups_tr, kind="stable")
        _, counts = np.unique(groups_tr[order], return_counts=True)
        model.fit(
            X_tr.iloc[order],
            y_tr.iloc[order].astype(int),
            group=counts,
        )
        return model.predict(X_te)

    if cand.objective == "quantile":
        params.update(objective="quantile", alpha=cand.alpha)
    elif cand.objective == "huber":
        params.update(objective="huber")
    elif cand.objective == "l2":
        pass
    else:  # pragma: no cover - guarded by the registry
        raise ValueError(f"unknown objective {cand.objective!r}")
    model = lgb.LGBMRegressor(**params)
    model.fit(X_tr, y_tr)
    return model.predict(X_te)


def build_oof(
    df: pd.DataFrame,
    cand: Candidate,
    *,
    horizon: int = DEFAULT_HORIZON,
    seed: int = 0,
    embargo: int = DEFAULT_EMBARGO,
) -> pd.Series:
    """Out-of-fold scores for every row a fold could legally test.

    Returns a float Series aligned to `df.index`, NaN where no fold tested the
    row (the earliest year has no training history in front of it).
    """
    folds = make_year_folds(df, horizon=horizon, embargo=embargo)
    cols = cand.features(df) if cand.features else feature_cols(df)
    X = df[cols].astype(float).reset_index(drop=True)
    y = pd.Series(
        np.asarray(cand.target(df), dtype=float), index=range(len(df))
    )
    month = pd.to_datetime(df["event_day"]).dt.to_period("M").astype(str)
    month_code = pd.factorize(month)[0]

    out = np.full(len(df), np.nan, dtype=float)
    for fold in folds:
        yv = y.to_numpy()
        tr = fold.train_idx[np.isfinite(yv[fold.train_idx])]
        if len(tr) < MIN_FOLD_TRAIN_ROWS:
            continue
        te = fold.test_idx
        groups = month_code[tr] if cand.grouped_by_month else None
        out[te] = _fit_predict(
            X.iloc[tr], y.iloc[tr], X.iloc[te], cand, seed, groups
        )
    return pd.Series(out, index=df.index)


# --------------------------------------------------------------------------
# The gauntlet
# --------------------------------------------------------------------------

def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 5:
        return float("nan")
    if np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan")
    return float(stats.spearmanr(a, b).statistic)


def monthly_ic(
    df: pd.DataFrame, score: str, label: str, *, min_per_month: int = 8
) -> pd.Series:
    """Spearman(score, label) computed within each calendar month.

    Months with fewer than `min_per_month` scored events are dropped: a rank
    correlation over 3 points is mostly noise and would widen every downstream
    confidence interval for no information.
    """
    sub = df[[score, label, "event_day"]].dropna()
    if sub.empty:
        return pd.Series(dtype=float)
    m = pd.to_datetime(sub["event_day"]).dt.to_period("M")
    ics = {}
    for key, g in sub.groupby(m):
        if len(g) < min_per_month:
            continue
        ics[key] = _spearman(g[score].to_numpy(), g[label].to_numpy())
    s = pd.Series(ics, dtype=float).dropna()
    if not s.empty:
        s.index = pd.PeriodIndex(s.index, freq="M")
    return s


def block_bootstrap_ci(
    values: pd.Series, *, n: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    """(lo, hi, P(mean<=0)) from resampling whole months with replacement."""
    if values.empty:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    arr = values.to_numpy()
    draws = rng.choice(arr, size=(n, len(arr)), replace=True).mean(axis=1)
    return (
        float(np.percentile(draws, 2.5)),
        float(np.percentile(draws, 97.5)),
        float((draws <= 0).mean()),
    )


def vol_neutral_ic(
    df: pd.DataFrame,
    score: str,
    label: str,
    *,
    vol_col: str = "x_vol_63_ann",
    n_bins: int = 5,
) -> float:
    """IC recomputed inside volatility quintiles, then averaged.

    The single most important test here. A score that merely reorders stocks
    by risk shows a healthy pooled IC and nothing at all once the risk axis is
    held fixed. Four earlier candidates with bigger headline IC than the
    eventual winner died exactly here.
    """
    sub = df[[score, label, vol_col]].dropna()
    if len(sub) < n_bins * 20:
        return float("nan")
    try:
        bins = pd.qcut(sub[vol_col], n_bins, labels=False, duplicates="drop")
    except ValueError:
        return float("nan")
    vals = [
        _spearman(g[score].to_numpy(), g[label].to_numpy())
        for _, g in sub.groupby(bins)
    ]
    vals = [v for v in vals if not math.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def decile_table(df: pd.DataFrame, score: str, label: str) -> pd.DataFrame:
    """Per-decile medians, win rates and crash rates.

    Medians lead because the mean of this label is dominated by a handful of
    multi-hundred-percent winners. `p_crash` (P(label < -30%)) is here because
    the previously shipped score ranked crash risk UPWARDS while looking
    acceptable on returns.
    """
    sub = df[[score, label]].dropna()
    if len(sub) < 100:
        return pd.DataFrame()
    d = pd.qcut(sub[score].rank(method="first"), 10, labels=False)
    rows = []
    for k, g in sub.groupby(d):
        rows.append(
            dict(
                decile=int(k),
                n=len(g),
                median=float(g[label].median()),
                mean=float(g[label].mean()),
                win_rate=float((g[label] > 0).mean()),
                p_crash=float((g[label] < -0.30).mean()),
            )
        )
    return pd.DataFrame(rows)


def nonoverlapping_portfolio(
    df: pd.DataFrame,
    score: str,
    *,
    horizon: int = DEFAULT_HORIZON,
    top_n: int = 10,
    cost_bps: float = 20.0,
) -> pd.DataFrame:
    """Equal-weight top-N book over non-overlapping `horizon`-day periods.

    Periods are cut on `entry_idx`, the shared trading-day calendar, so no two
    periods share a day and no return is counted twice. Overlapping windows
    are the classic way to turn 7 years of data into an impressive and
    meaningless t-statistic.
    """
    need = [score, "entry_idx", f"fwd_{horizon}", f"spy_{horizon}", "event_day"]
    sub = df[need].dropna().copy()
    if sub.empty:
        return pd.DataFrame()
    base = int(sub["entry_idx"].min())
    sub["period"] = (sub["entry_idx"] - base) // horizon
    cost = cost_bps / 10_000.0
    rows = []
    for p, g in sub.groupby("period"):
        if len(g) < top_n:
            continue
        pick = g.nlargest(top_n, score)
        r = float(pick[f"fwd_{horizon}"].mean()) - cost
        b = float(pick[f"spy_{horizon}"].mean())
        rows.append(
            dict(
                period=int(p),
                year=int(pd.to_datetime(g["event_day"]).dt.year.median()),
                n=len(g),
                ret=r,
                bench=b,
                excess=r - b,
            )
        )
    return pd.DataFrame(rows)


@dataclass
class AuditResult:
    name: str
    n_scored: int
    ic_mean: float
    ic_t: float
    ic_lo: float
    ic_hi: float
    ic_p_le_zero: float
    n_months: int
    years_positive: int
    years_total: int
    ic_by_year: dict
    vol_neutral: float
    lowvol_benchmark: float
    deciles: pd.DataFrame
    top_decile_median: float
    bottom_decile_median: float
    decile_rank_corr: float
    port_excess_ann: float
    port_p: float
    port_years_beat: int
    port_years: int

    def passes(self) -> bool:
        """The pre-registered bar, fixed before any candidate was run.

        Deliberately about the RANKING, not about beating an index: the
        portfolio test is reported but not gated on, because RESEARCH_NOTES.md
        already establishes that no portfolio here survives a seed sweep, and
        gating on it would only select the luckiest seed.
        """
        return (
            self.years_positive >= self.years_total - 1
            and self.ic_mean > 0
            and self.ic_p_le_zero < 0.05
            and self.vol_neutral > 0.02
            and self.vol_neutral > self.lowvol_benchmark
            and self.decile_rank_corr > 0.5
        )


def audit(
    df: pd.DataFrame,
    score: str,
    label: str,
    *,
    horizon: int = DEFAULT_HORIZON,
    top_n: int = 10,
    name: str = "",
) -> AuditResult:
    """Run the whole gauntlet on one score column."""
    ics = monthly_ic(df, score, label)
    lo, hi, p0 = block_bootstrap_ci(ics)
    t = (
        float(ics.mean() / (ics.std(ddof=1) / math.sqrt(len(ics))))
        if len(ics) > 2 and ics.std(ddof=1) > 0
        else float("nan")
    )
    by_year = (
        ics.groupby(ics.index.year).mean() if len(ics) else pd.Series(dtype=float)
    )

    dec = decile_table(df, score, label)
    if not dec.empty:
        rc = _spearman(dec["decile"].to_numpy(), dec["median"].to_numpy())
        top_med = float(dec["median"].iloc[-1])
        bot_med = float(dec["median"].iloc[0])
    else:
        rc, top_med, bot_med = float("nan"), float("nan"), float("nan")

    # Head-to-head against the dumbest possible risk ranker. If "buy the
    # low-volatility ones" matches the model, the model is that and nothing
    # more.
    tmp = df[[score, label, "x_vol_63_ann"]].copy()
    tmp["_lowvol"] = -tmp["x_vol_63_ann"]
    lowvol = vol_neutral_ic(tmp, "_lowvol", label)

    port = nonoverlapping_portfolio(df, score, horizon=horizon, top_n=top_n)
    if len(port) > 2:
        tt = stats.ttest_1samp(port["excess"], 0.0)
        per_year = port.groupby("year")["excess"].mean()
        periods_per_year = 252.0 / horizon
        ann = float((1.0 + port["excess"].mean()) ** periods_per_year - 1.0)
        p_val = float(tt.pvalue)
        beat = int((per_year > 0).sum())
        n_years = int(len(per_year))
    else:
        ann, p_val, beat, n_years = (float("nan"), float("nan"), 0, 0)

    return AuditResult(
        name=name or score,
        n_scored=int(df[score].notna().sum()),
        ic_mean=float(ics.mean()) if len(ics) else float("nan"),
        ic_t=t,
        ic_lo=lo,
        ic_hi=hi,
        ic_p_le_zero=p0,
        n_months=int(len(ics)),
        years_positive=int((by_year > 0).sum()),
        years_total=int(len(by_year)),
        ic_by_year={int(k): float(v) for k, v in by_year.items()},
        vol_neutral=vol_neutral_ic(df, score, label),
        lowvol_benchmark=lowvol,
        deciles=dec,
        top_decile_median=top_med,
        bottom_decile_median=bot_med,
        decile_rank_corr=rc,
        port_excess_ann=ann,
        port_p=p_val,
        port_years_beat=beat,
        port_years=n_years,
    )


def format_audit(a: AuditResult) -> str:
    return "\n".join(
        [
            f"=== {a.name} ===",
            f"  scored rows      {a.n_scored}",
            f"  monthly IC       {a.ic_mean:+.4f}  t={a.ic_t:.2f}  "
            f"95% CI [{a.ic_lo:+.4f}, {a.ic_hi:+.4f}]  "
            f"P(<=0)={a.ic_p_le_zero:.4f}  over {a.n_months} months",
            f"  years positive   {a.years_positive}/{a.years_total}   "
            + " ".join(f"{y}:{v:+.3f}" for y, v in sorted(a.ic_by_year.items())),
            f"  vol-neutral IC   {a.vol_neutral:+.4f}   "
            f"(low-vol-only ranker: {a.lowvol_benchmark:+.4f})",
            f"  decile medians   bottom {a.bottom_decile_median:+.4f}  "
            f"top {a.top_decile_median:+.4f}  "
            f"monotonicity {a.decile_rank_corr:+.2f}",
            f"  top-10 book      {a.port_excess_ann:+.2%}/yr vs SPY  "
            f"p={a.port_p:.3f}  beat in {a.port_years_beat}/{a.port_years} yrs",
            f"  GATE             {'PASS' if a.passes() else 'FAIL'}",
        ]
    )
