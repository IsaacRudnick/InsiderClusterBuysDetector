"""Risk-adjusted portfolio evaluation, and the permutation test that keeps it
honest.

WHY THIS EXISTS, AND WHY IT IS SEPARATE FROM tools/score_lab.py
==============================================================
score_lab grades a score on its ORDERING (rank IC). That is the right test for
"does this score know anything", and it is the wrong test for "what should I
buy". A score can order events perfectly by expected return and still produce
an untradeable book, and -- as it turned out here -- a score can look flat on
excess return while ordering RISK-ADJUSTED return strongly. Measured on the
shipped ensemble, Sharpe by decile runs 0.21 at the bottom to 1.33 at decile 8
while median excess over SPY barely moves. Reading only the excess-return
table missed that entirely.

So this module measures what a holder actually experiences: annualised return,
volatility, Sharpe, Sortino, maximum drawdown, and the same numbers for SPY and
IWM over the identical periods.

THE PERMUTATION TEST IS NOT OPTIONAL
------------------------------------
This project has already been burned once by exactly the failure this module
invites. A 70th-90th percentile band showed +19.77%/yr over SPY, p=0.040 and a
bootstrap CI excluding zero -- and a permutation test that re-ran the whole
band search on SHUFFLED scores produced +18.46%/yr from pure noise. p=0.435.
The band search manufactured the number, and the bootstrap could not see it
because a bootstrap prices the sampling of periods and not the selection of
the band.

`permutation_test` therefore takes the ENTIRE selection procedure as a
callable -- band choice, weighting scheme, name count, everything -- and
re-runs all of it on scores shuffled within each period. Whatever the search
would have found in noise is what it reports. Any Sharpe number from this
module that has not been through it is not a result.

WEIGHTING IS A LEVER IN ITS OWN RIGHT
-------------------------------------
Equal weight is not neutral. In a universe whose return distribution has a
1,000% right tail and whose volatilities range over an order of magnitude,
equal weight silently concentrates risk in the most volatile names. Inverse
volatility and inverse variance weighting are offered here because they can
raise Sharpe without the score improving at all -- and because reporting a
Sharpe without saying how the book was weighted is not reporting a Sharpe.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scipy import stats  # noqa: E402

HORIZON = 21
PERIODS_PER_YEAR = 252.0 / HORIZON

#: Round-trip cost charged to every position, every period. 20bps is the
#: figure the rest of this repo uses; microcaps would plausibly cost more, so
#: treat every number here as an upper bound on what was achievable.
DEFAULT_COST_BPS = 20.0

#: Positions below this are dropped before weighting. Sub-dollar stocks once
#: produced 58% of this grid's P&L from 0.7% of its lots; their quoted returns
#: are largely bid-ask bounce and no real order fills at the printed price.
DEFAULT_MIN_PRICE = 0.0


# ---------------------------------------------------------------------------
# Weighting schemes
# ---------------------------------------------------------------------------

def w_equal(g: pd.DataFrame) -> np.ndarray:
    return np.ones(len(g)) / len(g)


def w_inverse_vol(g: pd.DataFrame) -> np.ndarray:
    """Weight inversely to each name's own ex-ante annualised volatility.

    The textbook risk-parity move, and the honest reason it is here: it can
    lift Sharpe with no improvement in the score at all, so any Sharpe gain
    has to be attributed to it rather than credited to the model.
    """
    v = pd.to_numeric(g["x_vol_63_ann"], errors="coerce").to_numpy(dtype=float)
    v = np.where(np.isfinite(v) & (v > 0.05), v, np.nan)
    # A name with no usable volatility estimate gets the median weight rather
    # than being dropped: dropping it would silently change which names the
    # band holds, confounding the weighting comparison with a selection change.
    v = np.where(np.isnan(v), np.nanmedian(v) if np.isfinite(np.nanmedian(v)) else 1.0, v)
    w = 1.0 / v
    return w / w.sum()


def w_inverse_var(g: pd.DataFrame) -> np.ndarray:
    v = pd.to_numeric(g["x_vol_63_ann"], errors="coerce").to_numpy(dtype=float)
    v = np.where(np.isfinite(v) & (v > 0.05), v, np.nan)
    v = np.where(np.isnan(v), np.nanmedian(v) if np.isfinite(np.nanmedian(v)) else 1.0, v)
    w = 1.0 / (v ** 2)
    return w / w.sum()


WEIGHTINGS: dict[str, Callable[[pd.DataFrame], np.ndarray]] = {
    "equal": w_equal,
    "inv_vol": w_inverse_vol,
    "inv_var": w_inverse_var,
}


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BookSpec:
    """One complete, reproducible recipe for a portfolio.

    Everything that could be searched over lives here, so `permutation_test`
    can replay the entire recipe -- not just its last step -- against noise.
    """

    score_col: str = "ens"
    lo: float = 0.70            # percentile band, exclusive lower bound
    hi: float = 0.90            # percentile band, inclusive upper bound
    weighting: str = "equal"
    max_names: int | None = None
    min_names: int = 3
    min_price: float = DEFAULT_MIN_PRICE
    cost_bps: float = DEFAULT_COST_BPS
    #: Cap on any single position's weight, applied after the weighting
    #: scheme. Without one, inverse-variance weighting can put half the book
    #: in a single low-volatility name and quietly stop being a portfolio.
    max_weight: float = 1.0

    def label(self) -> str:
        n = f",n<={self.max_names}" if self.max_names else ""
        p = f",${self.min_price:g}+" if self.min_price else ""
        c = f",cap{self.max_weight:g}" if self.max_weight < 1.0 else ""
        return f"{self.lo:.0%}-{self.hi:.0%} {self.weighting}{n}{p}{c}"


def select(g: pd.DataFrame, spec: BookSpec, score_col: str) -> pd.DataFrame:
    """The candidates held in one period, band cut WITHIN that period.

    Cutting the band inside the period is the only version a live screener can
    reproduce: on any given day you can rank what you can see, and you cannot
    know where those names will sit in a distribution that includes the next
    four years of filings.
    """
    if spec.min_price > 0:
        g = g[g["entry_open"] >= spec.min_price]
    if len(g) < 10:
        return g.iloc[0:0]
    r = g[score_col].rank(pct=True)
    pick = g[(r > spec.lo) & (r <= spec.hi)]
    if spec.max_names and len(pick) > spec.max_names:
        pick = pick.nlargest(spec.max_names, score_col)
    return pick


@dataclass(frozen=True)
class PeriodPanel:
    """The dataset pre-sliced into per-period numpy arrays.

    Purely a performance structure, and it is load-bearing for honesty rather
    than convenience. The permutation test has to re-run an ENTIRE recipe grid
    a few hundred times; through pandas `groupby` that is tens of minutes and
    the test quietly stops being run. As flat arrays the same sweep is
    seconds, so there is no incentive to skip it.

    Every field is a list indexed by period, holding that period's rows.
    """

    score: list[np.ndarray]
    fwd: list[np.ndarray]
    vol: list[np.ndarray]
    price: list[np.ndarray]
    spy: np.ndarray
    iwm: np.ndarray
    year: np.ndarray

    @property
    def n_periods(self) -> int:
        return len(self.score)


def build_panel(df: pd.DataFrame, score_col: str) -> PeriodPanel:
    score, fwd, vol, price, spy, iwm, year = [], [], [], [], [], [], []
    for _, g in df.groupby("period", sort=True):
        score.append(g[score_col].to_numpy(dtype=float))
        fwd.append(g["fwd_21"].to_numpy(dtype=float))
        v = pd.to_numeric(g["x_vol_63_ann"], errors="coerce").to_numpy(dtype=float)
        # A missing or nonsensically small volatility becomes the period's
        # median rather than being dropped, so a weighting change never
        # silently changes WHICH names the book holds.
        med = np.nanmedian(v[np.isfinite(v) & (v > 0.05)]) if np.isfinite(v).any() else 1.0
        med = med if np.isfinite(med) and med > 0 else 1.0
        vol.append(np.where(np.isfinite(v) & (v > 0.05), v, med))
        price.append(pd.to_numeric(g["entry_open"], errors="coerce")
                     .fillna(0.0).to_numpy(dtype=float))
        spy.append(float(g["bench_SPY"].mean()))
        iwm.append(float(g["bench_IWM"].mean()))
        year.append(int(pd.to_datetime(g["entry_day"]).dt.year.median()))
    return PeriodPanel(score, fwd, vol, price,
                       np.asarray(spy), np.asarray(iwm), np.asarray(year))


def panel_returns(
    panel: PeriodPanel, spec: BookSpec, scores: list[np.ndarray] | None = None
) -> pd.DataFrame:
    """`period_returns` over the flat panel. Same recipe, same answer, fast."""
    scores = scores if scores is not None else panel.score
    cost = spec.cost_bps / 10_000.0
    rows = []
    for i in range(panel.n_periods):
        s, f, v, px = scores[i], panel.fwd[i], panel.vol[i], panel.price[i]
        keep = px >= spec.min_price if spec.min_price > 0 else np.ones(len(s), bool)
        if keep.sum() < 10:
            continue
        s, f, v = s[keep], f[keep], v[keep]
        n = len(s)
        # Percentile rank within the period, ties broken by position -- the
        # same "rank(pct=True)" convention the pandas path uses.
        order = np.argsort(s, kind="stable")
        pct = np.empty(n, dtype=float)
        pct[order] = (np.arange(n) + 1) / n
        sel = (pct > spec.lo) & (pct <= spec.hi)
        if sel.sum() < spec.min_names:
            continue
        idx = np.flatnonzero(sel)
        if spec.max_names and len(idx) > spec.max_names:
            idx = idx[np.argsort(-s[idx], kind="stable")[: spec.max_names]]
        if spec.weighting == "equal":
            w = np.ones(len(idx))
        elif spec.weighting == "inv_vol":
            w = 1.0 / v[idx]
        else:
            w = 1.0 / (v[idx] ** 2)
        w = w / w.sum()
        if spec.max_weight < 1.0:
            w = np.minimum(w, spec.max_weight)
            w = w / w.sum()
        rows.append(
            dict(period=i, year=int(panel.year[i]), n_held=len(idx),
                 ret=float((f[idx] * w).sum()) - cost,
                 bench_SPY=float(panel.spy[i]), bench_IWM=float(panel.iwm[i]))
        )
    return pd.DataFrame(rows)


def period_returns(
    df: pd.DataFrame, spec: BookSpec, *, score_col: str | None = None
) -> pd.DataFrame:
    """One row per non-overlapping period: the book's return and the benchmarks'.

    Periods are cut on `entry_idx`, the shared trading-day calendar, so no two
    periods share a day and no return is counted twice.
    """
    score_col = score_col or spec.score_col
    cost = spec.cost_bps / 10_000.0
    rows = []
    for p, g in df.groupby("period"):
        if len(g) < 10:
            continue
        pick = select(g, spec, score_col)
        if len(pick) < spec.min_names:
            continue
        w = WEIGHTINGS[spec.weighting](pick)
        if spec.max_weight < 1.0:
            w = np.minimum(w, spec.max_weight)
            w = w / w.sum()
        r = float((pick["fwd_21"].to_numpy(dtype=float) * w).sum()) - cost
        rows.append(
            dict(
                period=int(p),
                year=int(pd.to_datetime(g["entry_day"]).dt.year.median()),
                n_held=len(pick),
                ret=r,
                bench_SPY=float(g["bench_SPY"].mean()),
                bench_IWM=float(g["bench_IWM"].mean()),
            )
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _series_stats(r: pd.Series) -> dict:
    m = float(r.mean())
    s = float(r.std(ddof=1))
    dn = r[r < 0]
    dstd = float(dn.std(ddof=1)) if len(dn) > 2 else float("nan")
    eq = (1.0 + r).cumprod()
    dd = float((eq / eq.cummax() - 1.0).min())
    ann = (1.0 + m) ** PERIODS_PER_YEAR - 1.0
    return dict(
        ann_return=ann,
        ann_vol=s * math.sqrt(PERIODS_PER_YEAR),
        sharpe=(m / s) * math.sqrt(PERIODS_PER_YEAR) if s > 0 else float("nan"),
        sortino=(m / dstd) * math.sqrt(PERIODS_PER_YEAR) if dstd > 0 else float("nan"),
        max_drawdown=dd,
        calmar=ann / abs(dd) if dd < 0 else float("nan"),
        win_rate=float((r > 0).mean()),
        worst=float(r.min()),
    )


def evaluate(per: pd.DataFrame, label: str = "") -> dict:
    """Full risk report for a book, plus its excess over both yardsticks.

    Both benchmarks, always. These are microcaps: measured against SPY alone a
    good small-cap selection looks like failure, and against IWM alone an index
    fund looks like skill. Reporting whichever flatters is benchmark shopping.
    """
    if len(per) < 5:
        return {}
    out = dict(book=label, periods=len(per), avg_names=float(per["n_held"].mean()))
    out.update(_series_stats(per["ret"]))
    for b in ("SPY", "IWM"):
        ex = per["ret"] - per[f"bench_{b}"]
        out[f"excess_{b}"] = float((1 + ex.mean()) ** PERIODS_PER_YEAR - 1)
        out[f"p_{b}"] = float(stats.ttest_1samp(ex, 0.0).pvalue)
        # Information ratio: excess return per unit of TRACKING error, which is
        # the right risk denominator when the question is "versus the index"
        # rather than "in absolute terms".
        te = float(ex.std(ddof=1))
        out[f"ir_{b}"] = (
            (float(ex.mean()) / te) * math.sqrt(PERIODS_PER_YEAR)
            if te > 0 else float("nan")
        )
        per_year = per.assign(e=ex).groupby("year")["e"].mean()
        out[f"yrs_{b}"] = f"{int((per_year > 0).sum())}/{len(per_year)}"
    return out


def benchmark_row(per: pd.DataFrame, bench: str) -> dict:
    out = dict(book=bench, periods=len(per), avg_names=float("nan"))
    out.update(_series_stats(per[f"bench_{bench}"]))
    return out


# ---------------------------------------------------------------------------
# The permutation test
# ---------------------------------------------------------------------------

def permutation_test_panel(
    panel: PeriodPanel,
    search: Callable[[PeriodPanel, list], float],
    observed: float,
    *,
    n_draws: int = 200,
    seed: int = 0,
) -> dict:
    """Re-run the whole selection procedure on scores shuffled within periods.

    `search` must be the ENTIRE thing that produced `observed` -- every band,
    weighting and floor that was compared, with the best returned. Handing this
    only the winning recipe will bless a number the search invented.

    Permuting inside a period destroys the score's information while leaving
    the period structure, the candidate pool, the volatilities and the return
    distribution untouched, so whatever the search still finds is search.
    """
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_draws):
        shuffled = [rng.permutation(s) for s in panel.score]
        null.append(search(panel, shuffled))
    null = np.asarray([v for v in null if np.isfinite(v)], dtype=float)
    if not len(null):
        return {}
    return dict(
        observed=observed,
        null_median=float(np.median(null)),
        null_p95=float(np.percentile(null, 95)),
        null_max=float(null.max()),
        p_value=float((null >= observed).mean()),
        n_draws=int(len(null)),
    )


def permutation_test(
    df: pd.DataFrame,
    search: Callable[[pd.DataFrame, str], float],
    observed: float,
    *,
    score_col: str = "ens",
    n_draws: int = 200,
    seed: int = 0,
) -> dict:
    """Re-run the WHOLE selection procedure on scores shuffled within periods.

    `search` must be the entire thing that was done to arrive at `observed` --
    if six bands and three weightings were compared and the best taken, then
    `search` compares six bands and three weightings and returns the best. Pass
    only the final recipe and this test will happily bless a number that the
    search invented.

    Shuffling WITHIN each period is what makes this the right null: it destroys
    the score's information while leaving the period structure, the candidate
    pool, the volatilities and the return distribution exactly as they are. So
    anything the search still finds is search, not signal.
    """
    rng = np.random.default_rng(seed)
    work = df.copy()
    null = []
    for _ in range(n_draws):
        work["_perm"] = work.groupby("period")[score_col].transform(
            lambda s: rng.permutation(s.to_numpy())
        )
        null.append(search(work, "_perm"))
    null = np.asarray([v for v in null if np.isfinite(v)], dtype=float)
    if not len(null):
        return {}
    return dict(
        observed=observed,
        null_median=float(np.median(null)),
        null_p95=float(np.percentile(null, 95)),
        null_max=float(null.max()),
        p_value=float((null >= observed).mean()),
        n_draws=int(len(null)),
    )


# ---------------------------------------------------------------------------
# Dataset prep
# ---------------------------------------------------------------------------

def prepare(
    scores: pd.DataFrame,
    dataset_path: str,
    *,
    score_cols: Sequence[str] = ("ens",),
) -> pd.DataFrame:
    """Join scores to the fields a book needs and cut non-overlapping periods."""
    from tools import band_backtest as bb

    scores = scores.copy()
    scores["event_day"] = pd.to_datetime(scores["event_day"])
    base = pd.read_parquet(dataset_path)
    base["event_day"] = pd.to_datetime(base["event_day"])
    need = ["ticker", "event_day", "entry_day", "entry_idx", "fwd_21",
            "entry_open", "x_vol_63_ann"]
    have = [c for c in need if c not in scores.columns] + ["ticker", "event_day"]
    df = scores.merge(base[list(dict.fromkeys(have))], on=["ticker", "event_day"],
                      how="left")
    df = bb.attach_benchmarks(df, HORIZON)
    df = df.dropna(subset=list(score_cols) + ["fwd_21", "entry_idx", "bench_SPY"])
    start = int(df["entry_idx"].min())
    df["period"] = (df["entry_idx"] - start) // HORIZON
    return df
