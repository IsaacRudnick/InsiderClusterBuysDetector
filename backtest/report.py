"""Single-file interactive Plotly HTML report for the backtest run.

Layout principles:
- One full-width chart per section (no cramped 2×2 grids).
- Exit-method picker buttons at the top of each multi-exit chart.
- Legend always on the right, vertical — never overlaps axis labels.
- Summary table is a sortable HTML table (click any header), not a Plotly Table.
- Consistent color palette across all charts so a strategy is one color everywhere.
"""

from __future__ import annotations

import json
import logging
import random
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .engine import RunResult
from .legend import LEGEND_CSS, LEGEND_JS, strategy_legend_html
from .strategies import EXIT_METHODS

log = logging.getLogger(__name__)


# Total point BUDGET for an entire figure (score_scatter_fig or
# lot_return_box_fig combined across ALL of their traces), not a per-trace
# cap. Plotly inlines every marker's x/y/hover-text (score_scatter_fig) or
# every plotted y-value (lot_return_box_fig) into the page as JSON, so file
# size scales with n_traces * points_per_trace -- not points_per_trace alone.
#
# This used to be a flat per-trace cap of 20,000, which is what actually
# blew this report up to 655 MB: both figures draw one trace per (strategy,
# exit method) combination, so a grid of 53 strategies x 7 exit methods put
# up to 371 traces in a single figure -- an effective ceiling of
# ~371 x 20,000 = 7.4M points, not 20,000. Measured on the real
# out/backtest_20260809_100107 output (372 runs), that made score_scatter_fig
# 244.88 MB and lot_return_box_fig 67.84 MB on their own -- 99% of the
# 325 MB report.html.
#
# lot_return_box_fig no longer sends every raw y-value at all: it passes
# exact q1/median/q3/whisker-fence/mean/sd statistics computed server-side
# from the FULL lot set (see _box_stats), so the box shape itself is never
# subject to sampling error. SCATTER_POINT_BUDGET now only bounds how many
# individual OUTLIER markers get drawn on top of each box (see
# _sample_outliers_keep_extremes, which always keeps the largest-magnitude
# outlier on each side rather than uniformly sampling them away).
#
# Both figures are now also restricted to the _select_top_runs top-N run
# list (see REPORT_MAX_CURVES below), the same way the equity/drawdown/
# exposure charts already were, which bounds n_traces to REPORT_MAX_CURVES
# on its own. SCATTER_POINT_BUDGET is a second, independent safety net: it
# is divided evenly across however many traces a figure actually ends up
# drawing (see _per_trace_point_cap), so total figure size stays roughly
# constant even if REPORT_MAX_CURVES is raised later or a run's trade
# filter (exit_reason in expiry/trim, not delisted) happens to leave more
# non-empty traces than expected.
SCATTER_POINT_BUDGET = 20_000


def _per_trace_point_cap(n_traces: int, budget: int = SCATTER_POINT_BUDGET) -> int:
    """Split a whole-figure point budget evenly across n_traces.

    Plain integer division: every trace gets the same cap regardless of its
    own lot count (a run with 500 lots and a run with 50,000 lots both get
    `budget // n_traces`), which is the simplest allocation to reason about
    and to test -- and keeps the guarantee that total plotted points across
    the whole figure never exceeds `budget` (n_traces * (budget // n_traces)
    <= budget). A smarter scheme that gives small traces their full point
    count and redistributes the leftover to big traces would look denser,
    but couples every trace's sample size to every other trace's lot count
    and is harder to reason about or unit test in isolation. Floors at 1
    point/trace so a degenerate case (more traces than budget) never divides
    to zero or goes negative.
    """
    if n_traces <= 0:
        return budget
    return max(1, budget // n_traces)


def _lot_trace_count(results_by_exit: dict[str, list[RunResult]]) -> int:
    """Count how many (exit_label, strategy) combinations will draw a trace
    in score_scatter_fig / lot_return_box_fig. Both figures use the
    identical realized-lot filter (exit_reason in expiry/trim, not
    delisted), so this one count is valid for splitting
    SCATTER_POINT_BUDGET in both figures AND for the downsampling note
    render_html shows above them -- see _lot_downsampling_note_html."""
    n = 0
    for runs in results_by_exit.values():
        for r in runs:
            if any(t.exit_reason in ("expiry", "trim") and not t.delisted
                   for t in r.trades):
                n += 1
    return n


# The equity/drawdown/exposure charts each plot one full daily line per run,
# across every exit method (not just the one visible by default -- the
# exit-method picker toggles trace *visibility*, so every trace's data is
# still inlined into the page). A grid run of ~323 runs x ~1507 trading days
# is ~487k points PER chart, x3 charts, serialized as JSON text -- this (not
# inline plotly.js, which is only ~3MB) is what turned report.html into a
# 352MB, browser-choking file. Two independent, composable caps fix it:
#
#   REPORT_MAX_CURVES caps how many *runs* get a line at all, per chart --
#     ranked by Sharpe (falling back to total_return when Sharpe is
#     NaN/missing). The SPY benchmark line is unaffected: equity_curves_fig
#     and drawdown_fig plot it from a separate `spy` parameter, not from the
#     ranked run list, so it is always shown regardless of rank.
#
#   REPORT_MAX_POINTS_PER_CURVE caps how many *points* each kept line gets,
#     via uniform stride sampling that always retains the series' FINAL
#     point -- so the plotted ending value matches the summary table's
#     total_return, which is computed from the full series -- and, for the
#     drawdown chart specifically, the single worst (most negative) point
#     too. A plain stride sample can step right over the one actual
#     max-drawdown day and plot a shallower low than the table's
#     max_drawdown number, which would make the chart contradict the table
#     it sits next to.
#
# Both are independent of SCATTER_POINT_BUDGET above, which caps raw per-lot
# scatter/box points, not per-run daily time series. The full daily series
# for every run is still written to equity_<strategy>_<exit>.csv regardless
# of what gets plotted here.
REPORT_MAX_CURVES = 25
REPORT_MAX_POINTS_PER_CURVE = 400


# Color-blind-friendly palette (Okabe-Ito + neutrals)
PALETTE = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#CC79A7",  # pink
    "#56B4E9",  # sky
    "#D55E00",  # vermillion
    "#F0E442",  # yellow
    "#999999",  # gray
]
SPY_COLOR = "#000000"


def _strategy_colors(strategy_order: list[str]) -> dict[str, str]:
    m = {name: PALETTE[i % len(PALETTE)] for i, name in enumerate(strategy_order)}
    m["spy_buy_and_hold"] = SPY_COLOR
    return m


def _pct(x: float, signed: bool = True) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "—"
    fmt = "{:+.2f}%" if signed else "{:.2f}%"
    return fmt.format(x * 100)


def _num(x: float, fmt: str = "{:.2f}") -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "—"
    return fmt.format(x)


def _layout_defaults(*, title: str, height: int = 540,
                     yaxis_title: str = "", xaxis_title: str = "") -> dict:
    return dict(
        title=dict(text=title, x=0.02, xanchor="left", font=dict(size=16)),
        height=height,
        margin=dict(l=70, r=230, t=110, b=60),
        legend=dict(
            orientation="v", yanchor="top", y=1, xanchor="left", x=1.02,
            bgcolor="rgba(255,255,255,0.6)", bordercolor="#ddd", borderwidth=1,
            font=dict(size=11),
        ),
        plot_bgcolor="#fafafa",
        font=dict(family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif",
                  size=12, color="#333"),
        xaxis=dict(title=xaxis_title, gridcolor="#e9e9e9", zeroline=False),
        yaxis=dict(title=yaxis_title, gridcolor="#e9e9e9", zeroline=False),
        hovermode="x unified",
    )


def _exit_buttons(visibility_per_exit: dict[str, list[bool]],
                  y: float = 1.10) -> list[dict]:
    """Build a horizontal updatemenus button row that toggles trace visibility per exit method."""
    return [dict(
        type="buttons",
        direction="right",
        x=0.5, xanchor="center",
        y=y, yanchor="top",
        showactive=True,
        bgcolor="white",
        bordercolor="#bbb",
        borderwidth=1,
        font=dict(size=12),
        buttons=[
            dict(
                label=em.label,
                method="update",
                args=[{"visible": visibility_per_exit[em.label]}],
            ) for em in EXIT_METHODS
        ],
    )]


# Any strategy whose top single ticker accounts for more than this share of
# its total realized pnl gets a visible warning banner above the summary
# table — the leaderboard-topping strategy that turned out to be 83% one
# fake reverse-split lot, and big_stake_increase (113% from one lot, i.e.
# money-losing without it), are exactly the cases this threshold is tuned to
# catch. Checked against the raw (non-abs) share per the task spec: a
# strategy that's net-losing because of one ticker also shows a share above
# this bar (its total pnl is negative and the ticker's pnl is a still-more-
# negative share of it), so the same check covers both directions in
# practice for realistic runs.
CONCENTRATION_WARNING_THRESHOLD = 0.30


def _concentration_warnings_html(df: pd.DataFrame) -> str:
    """Visible WARNING banner for any strategy whose pnl leans on one ticker.

    Nothing in the sortable summary table forces a reader to notice a
    top_ticker_pnl_share outlier — a human skimming a leaderboard has no
    reason to sort by that column first. This renders unmissably instead.
    """
    if "top_ticker_pnl_share" not in df.columns:
        return ""
    rows = []
    for _, row in df.iterrows():
        share = row.get("top_ticker_pnl_share")
        if share is None or (isinstance(share, float) and (np.isnan(share) or np.isinf(share))):
            continue
        # Compare on MAGNITUDE. The share is signed, so a bare `share > t`
        # only catches a ticker pushing the total in its own direction. It
        # misses the mirror case: a strategy at +$10k total whose worst
        # ticker lost $50k (share -5.0) is even more concentrated -- without
        # that one name it made $60k -- yet a signed test waves it through.
        # Both directions mean "this ranking is one ticker's story".
        if abs(share) <= CONCENTRATION_WARNING_THRESHOLD:
            continue
        strategy = row.get("strategy", "?")
        exit_method = row.get("exit_method", "?")
        ticker = row.get("top_ticker", "?")
        # A negative share means the ticker moved the total the OTHER way,
        # so "came from" would read as nonsense. Word each case for what it
        # actually is.
        if share > 0:
            detail = (
                f"{_pct(share, signed=False)} of this strategy's P&amp;L came "
                f"from <b>{ticker}</b>"
            )
        else:
            detail = (
                f"<b>{ticker}</b> moved this strategy's P&amp;L against its total "
                f"by {_pct(abs(share), signed=False)}"
            )
        rows.append(
            f'<div class="concentration-warning">'
            f"&#9888; <b>{strategy}</b> ({exit_method}): {detail}; "
            f"verify before trusting this ranking."
            f"</div>"
        )
    if not rows:
        return ""
    return f'<div class="concentration-warnings">{"".join(rows)}</div>'


# Any strategy whose in_sample_frac exceeds this threshold gets a visible
# warning banner. in_sample_frac (see backtest/metrics.py's
# _in_sample_frac) is the fraction of a run's traded window that overlaps
# the signal-fit's own training period — nonzero only for strategies whose
# target_fn reads learned_score or tail_score (backtest/signal_fit.py's
# fitted weights). thr_gt_p11 posting the single t_alpha > 2 in a 322-run
# grid is exactly the kind of result this exists to catch *if* it were
# fit-dependent — verified it is not (it thresholds conviction_score,
# always the hand-tuned insider_cluster_buys.DEFAULT_WEIGHTS, never a
# fitted score), but any learned_* strategy topping a leaderboard the same
# way would be trading on a training-period echo. 0.20 is a soft trip-wire
# the same way CONCENTRATION_WARNING_THRESHOLD is: any nonzero overlap is
# worth a second look, but flagging every sliver would bury the signal.
IN_SAMPLE_WARNING_THRESHOLD = 0.20


def _in_sample_warnings_html(df: pd.DataFrame) -> str:
    """Visible WARNING banner for any strategy whose traded window
    substantially overlaps the period its own signal score was fit on.

    Nothing in the sortable summary table forces a reader to notice
    in_sample_frac before trusting a high t_alpha or alpha_ann — a human
    scanning a leaderboard has no reason to sort by that column first.
    This renders unmissably instead, the same way
    _concentration_warnings_html does for pnl concentration.
    """
    if "in_sample_frac" not in df.columns:
        return ""
    rows = []
    for _, row in df.iterrows():
        frac = row.get("in_sample_frac")
        if frac is None or (isinstance(frac, float) and (np.isnan(frac) or np.isinf(frac))):
            continue
        if frac <= IN_SAMPLE_WARNING_THRESHOLD:
            continue
        strategy = row.get("strategy", "?")
        exit_method = row.get("exit_method", "?")
        rows.append(
            f'<div class="concentration-warning">'
            f"&#9888; <b>{strategy}</b> ({exit_method}): "
            f"{_pct(frac, signed=False)} of its traded window overlaps the "
            f"signal-fit training period; its alpha and t(alpha) are "
            f"<b>partly in-sample</b> and are not evidence of out-of-sample "
            f"skill; check the OOS tables in the Learned signal weights "
            f"section before trusting this ranking."
            f"</div>"
        )
    if not rows:
        return ""
    return f'<div class="concentration-warnings">{"".join(rows)}</div>'


# ---------------------------------------------------------------------------
# 1. Summary table — HTML, sortable
# ---------------------------------------------------------------------------
def _summary_table_html(df: pd.DataFrame) -> str:
    cols = [
        ("strategy", "Strategy", "text"),
        ("exit_method", "Exit method", "text"),
        ("in_sample_frac", "In-Sample Frac", "pct_unsigned"),
        ("total_return", "Total Return", "pct"),
        ("cagr", "CAGR", "pct"),
        ("sharpe", "Sharpe", "num"),
        ("sortino", "Sortino", "num"),
        ("max_drawdown", "Max DD", "pct"),
        ("calmar", "Calmar", "num"),
        ("win_rate", "Win Rate", "pct_unsigned"),
        ("avg_lot_return", "Avg Lot Ret", "pct"),
        ("hit_rate_vs_spy", "Hit vs SPY", "pct_unsigned"),
        ("avg_excess_vs_spy", "Avg Excess vs SPY", "pct"),
        ("top_ticker_pnl_share", "Top Ticker PnL Share", "pct"),
        ("top_ticker", "Top Ticker", "text"),
        ("n_lots_gt_300pct", "Lots >300%", "int"),
        ("n_lots", "Lots", "int"),
        ("n_natural_exits", "Natural Exits", "int"),
        ("n_rebalances", "Rebalances", "int"),
        ("mean_exposure", "Avg Exposure", "pct_unsigned"),
        ("n_skipped_capacity", "Skip Cap", "int"),
        ("n_skipped_liquidity", "Skip Liq", "int"),
        ("n_skipped_participation", "Skip Part", "int"),
    ]
    cols = [(k, label, kind) for (k, label, kind) in cols if k in df.columns]

    def fmt_cell(val, kind):
        if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
            return "—", float("nan")
        if kind == "pct":
            return _pct(val), float(val)
        if kind == "pct_unsigned":
            return _pct(val, signed=False), float(val)
        if kind == "num":
            return _num(val), float(val)
        if kind == "int":
            return f"{int(val)}", float(val)
        return str(val), 0.0

    # Color scale bounds
    cagr_vals = df["cagr"].dropna().tolist() if "cagr" in df else []
    sharpe_vals = df["sharpe"].dropna().tolist() if "sharpe" in df else []
    cagr_max = max([abs(v) for v in cagr_vals] + [0.0001])
    sharpe_max = max([abs(v) for v in sharpe_vals] + [0.0001])

    def diverge_bg(val, vmax):
        if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
            return ""
        t = max(-1.0, min(1.0, val / vmax))
        if t >= 0:
            r, g, b = 220 - int(60 * t), 240, 220 - int(60 * t)
        else:
            r, g, b = 250, 220 + int(60 * t), 220 + int(60 * t)
        return f"background-color: rgb({r},{g},{b});"

    def in_sample_bg(val):
        # Same visible-in-the-row-itself principle as top_ticker_pnl_share's
        # concentration warning: a reader sorting this table by CAGR/Sharpe
        # should not have to separately notice the warning banner above to
        # learn a top result is partly in-sample.
        if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
            return ""
        if val > IN_SAMPLE_WARNING_THRESHOLD:
            return "background-color: rgb(255,224,224); font-weight: 600;"
        return ""

    header_html = "".join(
        f'<th data-col="{k}" data-kind="{kind}">{label}'
        f'<span class="sort-indicator"></span></th>'
        for (k, label, kind) in cols
    )
    rows_html: list[str] = []
    for _, row in df.iterrows():
        cells = []
        for (k, _label, kind) in cols:
            disp, raw = fmt_cell(row.get(k), kind)
            style = ""
            if k == "cagr":
                style = diverge_bg(raw, cagr_max)
            elif k == "sharpe":
                style = diverge_bg(raw, sharpe_max)
            elif k == "in_sample_frac":
                style = in_sample_bg(raw)
            align = "left" if kind == "text" else "right"
            cells.append(
                f'<td data-sort="{raw}" style="{style} text-align:{align};">{disp}</td>'
            )
        # Highlight SPY row
        cls = "spy" if row.get("strategy") == "spy_buy_and_hold" else ""
        rows_html.append(f'<tr class="{cls}">{"".join(cells)}</tr>')

    return f"""
{_concentration_warnings_html(df)}
{_in_sample_warnings_html(df)}
<div class="table-wrap">
  <table class="summary-table">
    <thead><tr>{header_html}</tr></thead>
    <tbody>{"".join(rows_html)}</tbody>
  </table>
</div>
"""


# ---------------------------------------------------------------------------
# 1b. Downsampling / top-N helpers for the big daily time-series figures
#     (equity_curves_fig, drawdown_fig, exposure_fig). See
#     REPORT_MAX_CURVES / REPORT_MAX_POINTS_PER_CURVE above for why.
# ---------------------------------------------------------------------------
def _downsample_uniform(s: pd.Series, max_points: int) -> pd.Series:
    """Stride-sample s to at most max_points rows, always keeping the LAST
    row exact -- the plotted ending value must match the summary table's
    total_return, which is computed off the full-resolution series."""
    n = len(s)
    if n <= max_points or max_points <= 0:
        return s
    stride = -(-n // max_points)  # ceil division, no float rounding
    idx = list(range(0, n, stride))
    if idx[-1] != n - 1:
        idx.append(n - 1)
    return s.iloc[idx]


def _downsample_preserve_min(s: pd.Series, max_points: int) -> pd.Series:
    """Like _downsample_uniform, but keeps the LOWEST-valued point in each
    stride bucket instead of the first, plus the series' global minimum and
    its final point. Used for the drawdown chart: a plain stride sample can
    step right over the single worst day, and a report whose chart shows a
    shallower max drawdown than the summary table's max_drawdown number is a
    bug, not a cosmetic issue."""
    n = len(s)
    if n <= max_points or max_points <= 0:
        return s
    stride = -(-n // max_points)
    values = s.to_numpy()
    idx_set: set[int] = set()
    for start in range(0, n, stride):
        end = min(start + stride, n)
        idx_set.add(start + int(np.argmin(values[start:end])))
    idx_set.add(int(np.argmin(values)))  # global worst DD, belt-and-suspenders
    idx_set.add(n - 1)  # exact ending value
    return s.iloc[sorted(idx_set)]


def _rank_metric_lookup(summary_df: pd.DataFrame) -> dict[tuple, float]:
    """(strategy, exit_method) -> ranking score for REPORT_MAX_CURVES.
    Prefers sharpe (risk-adjusted, already computed for every run); falls
    back to total_return when sharpe is missing/NaN (e.g. too few trading
    days for a stable estimate) so such a run still ranks by something
    instead of being silently sorted to the very bottom."""
    lookup: dict[tuple, float] = {}
    has_sharpe = "sharpe" in summary_df.columns
    has_total_return = "total_return" in summary_df.columns
    for _, row in summary_df.iterrows():
        key = (row.get("strategy"), row.get("exit_method"))
        val = row.get("sharpe") if has_sharpe else None
        if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
            val = row.get("total_return") if has_total_return else None
        if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
            val = float("-inf")
        lookup[key] = float(val)
    return lookup


def _select_top_runs(results_by_exit: dict[str, list[RunResult]],
                     summary_df: pd.DataFrame,
                     max_curves: int = REPORT_MAX_CURVES,
                     ) -> tuple[dict[str, list[RunResult]], int, int]:
    """Keep only the top `max_curves` runs (by _rank_metric_lookup) across
    all exit methods, for the big per-day time-series charts. The SPY
    benchmark is plotted from a separate parameter in those figures (not
    from this run list), so it is unaffected by this filter and is always
    shown regardless of rank.

    Returns (filtered_results_by_exit, n_plotted, n_total) so callers can
    render an honest "showing N of M" note next to the charts.
    """
    all_runs = [r for runs in results_by_exit.values() for r in runs]
    n_total = len(all_runs)
    if n_total <= max_curves:
        return results_by_exit, n_total, n_total
    lookup = _rank_metric_lookup(summary_df)
    ranked = sorted(
        all_runs,
        key=lambda r: lookup.get((r.strategy, r.exit_label), float("-inf")),
        reverse=True,
    )
    # Compare by identity, not dataclass equality: RunResult's generated
    # __eq__ would compare equity_curve/exposure_curve pd.Series field-by-
    # field, and Series.__eq__ returns an elementwise Series, not a bool --
    # `x in some_set_of_RunResult` would raise, not just be slow.
    keep_ids = {id(r) for r in ranked[:max_curves]}
    filtered = {
        label: [r for r in runs if id(r) in keep_ids]
        for label, runs in results_by_exit.items()
    }
    return filtered, max_curves, n_total


def _downsampling_note_html(n_plotted: int, n_total: int, *,
                            drawdown: bool = False, include_spy: bool = True) -> str:
    """Visible note above a downsampled time-series chart so nobody mistakes
    a sampled 25-of-323-runs, 400-point line for the complete daily record."""
    if n_plotted >= n_total:
        scope = f"All {n_total} runs are plotted below"
    else:
        scope = (
            f"Only the top {n_plotted} of {n_total} runs (ranked by Sharpe, "
            f"or total return where Sharpe isn't available) are plotted below"
        )
        if include_spy:
            scope += " — SPY is always shown regardless of rank"
    exact_bits = "the final point of each line is always exact"
    if drawdown:
        exact_bits += " and the worst (most negative) point is always exact"
    return (
        f'<p class="note">{scope}. Each line is downsampled to at most '
        f'{REPORT_MAX_POINTS_PER_CURVE} points to keep the file size '
        f'manageable — {exact_bits}, matching the summary table. Full daily '
        f'granularity for every run is in this output folder\'s '
        f'<code>equity_*.csv</code> files.</p>'
    )


def _lot_downsampling_note_html(n_plotted: int, n_total: int, per_trace_cap: int, *,
                                exact_stats: bool = False) -> str:
    """Visible note above the score-vs-return scatter and per-lot box charts.

    Unlike the daily-curve charts above, these two are downsampled TWICE —
    by run (same top-N filter as the equity/drawdown/exposure charts) AND by
    lot within each kept run. A reader must not come away thinking either
    sampling step alone describes what's on screen.

    `exact_stats` switches the second part of the wording for
    lot_return_box_fig, whose box itself (quartiles/median/whisker fences)
    is computed from every realized lot with no sampling error — only the
    individual outlier markers drawn on top are downsampled, unlike
    score_scatter_fig where every plotted point is a random sample of the
    full lot set. Saying "(N of M lots shown)" for the box chart would
    wrongly imply the box shape itself was estimated from a sample."""
    if n_plotted >= n_total:
        scope = f"All {n_total} runs are eligible to be plotted below"
    else:
        scope = (
            f"Only the top {n_plotted} of {n_total} runs (ranked by Sharpe, "
            f"or total return where Sharpe isn't available) are plotted below"
        )
    if exact_stats:
        return (
            f'<p class="note">{scope}. Within each plotted run, the box itself — '
            f'quartiles, median, and whisker fences — is computed exactly from '
            f'every realized lot, with no sampling. Only the individual outlier '
            f'markers drawn on top are randomly sampled (fixed seed, so '
            f're-rendering is reproducible) to at most {per_trace_cap:,} points '
            f'per trace — a shared budget of {SCATTER_POINT_BUDGET:,} points '
            f'split evenly across every trace in the chart — to keep the file '
            f'size manageable, and the single largest-magnitude outlier on '
            f'each side is always kept. A trace\'s name shows "(N of M outlier '
            f'lots shown)" whenever its outliers were sampled. Full per-lot '
            f'detail for every run is in this output folder\'s '
            f'<code>trades_*.csv</code> files.</p>'
        )
    return (
        f'<p class="note">{scope}. Within each plotted run, lots are further '
        f'randomly sampled (fixed seed, so re-rendering is reproducible) to '
        f'at most {per_trace_cap:,} points per trace — a shared budget of '
        f'{SCATTER_POINT_BUDGET:,} points split evenly across every trace in '
        f'the chart — to keep the file size manageable. A trace\'s name shows '
        f'"(N of M lots shown)" whenever it was sampled. Full per-lot detail '
        f'for every run is in this output folder\'s <code>trades_*.csv</code> '
        f'files.</p>'
    )


# ---------------------------------------------------------------------------
# 2. Equity curves (single chart + exit-method picker)
# ---------------------------------------------------------------------------
def equity_curves_fig(results_by_exit: dict[str, list[RunResult]],
                      spy: RunResult, strategy_order: list[str]) -> go.Figure:
    color = _strategy_colors(strategy_order)
    labels = [em.label for em in EXIT_METHODS]
    fig = go.Figure()
    spy_curve = _downsample_uniform(spy.equity_curve, REPORT_MAX_POINTS_PER_CURVE)
    fig.add_trace(go.Scatter(
        x=spy_curve.index, y=spy_curve.values,
        name="spy_buy_and_hold",
        line=dict(color=SPY_COLOR, width=2.5, dash="dash"),
        hovertemplate="$%{y:,.0f}<extra>SPY</extra>",
    ))
    default_label = labels[1] if len(labels) > 1 else labels[0]
    for label in labels:
        for r in results_by_exit[label]:
            curve = _downsample_uniform(r.equity_curve, REPORT_MAX_POINTS_PER_CURVE)
            fig.add_trace(go.Scatter(
                x=curve.index, y=curve.values,
                name=r.strategy,
                line=dict(color=color[r.strategy], width=2),
                visible=(label == default_label),
                hovertemplate=("$%{y:,.0f}"
                               f"<extra>{r.strategy} ({label})</extra>"),
            ))

    visibility = {}
    for selected_label in labels:
        vis = [True]  # SPY
        for label in labels:
            for _ in results_by_exit[label]:
                vis.append(label == selected_label)
        visibility[selected_label] = vis

    layout = _layout_defaults(
        title="Equity curves — pick an exit method",
        height=560,
        yaxis_title="NAV ($)",
        xaxis_title="",
    )
    layout["yaxis"]["type"] = "log"
    layout["updatemenus"] = _exit_buttons(visibility)
    fig.update_layout(**layout)
    return fig


# ---------------------------------------------------------------------------
# 3. Drawdowns
# ---------------------------------------------------------------------------
def drawdown_fig(results_by_exit: dict[str, list[RunResult]],
                 spy: RunResult, strategy_order: list[str]) -> go.Figure:
    color = _strategy_colors(strategy_order)
    labels = [em.label for em in EXIT_METHODS]

    def dd(s: pd.Series) -> pd.Series:
        peak = s.cummax()
        return (s / peak) - 1.0

    fig = go.Figure()
    spy_dd = _downsample_preserve_min(dd(spy.equity_curve), REPORT_MAX_POINTS_PER_CURVE)
    fig.add_trace(go.Scatter(
        x=spy_dd.index, y=spy_dd.values * 100,
        name="spy_buy_and_hold",
        line=dict(color=SPY_COLOR, width=2.5, dash="dash"),
        hovertemplate="%{y:.2f}%<extra>SPY</extra>",
    ))
    default_label = labels[1] if len(labels) > 1 else labels[0]
    for label in labels:
        for r in results_by_exit[label]:
            d = _downsample_preserve_min(dd(r.equity_curve), REPORT_MAX_POINTS_PER_CURVE)
            fig.add_trace(go.Scatter(
                x=d.index, y=d.values * 100,
                name=r.strategy,
                line=dict(color=color[r.strategy], width=2),
                visible=(label == default_label),
                hovertemplate=("%{y:.2f}%"
                               f"<extra>{r.strategy} ({label})</extra>"),
            ))

    visibility = {}
    for selected_label in labels:
        vis = [True]
        for label in labels:
            for _ in results_by_exit[label]:
                vis.append(label == selected_label)
        visibility[selected_label] = vis

    layout = _layout_defaults(
        title="Drawdown — pick an exit method",
        height=480, yaxis_title="Drawdown %",
    )
    layout["updatemenus"] = _exit_buttons(visibility)
    fig.update_layout(**layout)
    fig.add_hline(y=0, line=dict(dash="dot", color="gray"))
    return fig


# ---------------------------------------------------------------------------
# 4. Per-lot return distribution (box plot, exit-method selector)
# ---------------------------------------------------------------------------
def _box_stats(rets: list[float]) -> dict[str, float | list[float]]:
    """Compute EXACT box-plot statistics from the FULL lot-return list, for
    passing to go.Box's precomputed-statistics parameters (q1/median/q3/
    lowerfence/upperfence/mean/sd) instead of letting Plotly derive them
    client-side from a downsampled `y` array.

    Quartiles use linear interpolation (numpy's default percentile method)
    -- a standard, well-defined choice; since we now compute these ourselves
    from the complete dataset rather than letting Plotly estimate them from
    a sample, there is no prior on-screen quartile behavior to bit-for-bit
    reproduce. Whisker fences and outliers follow the standard Tukey rule
    (1.5x IQR beyond Q1/Q3), which is also what Plotly's own client-side box
    calc uses -- `lowerfence`/`upperfence` are the most extreme NON-outlier
    data points (not the raw 1.5xIQR threshold value itself), matching how
    Plotly draws the whisker end caps.
    """
    arr = np.asarray(rets, dtype=float)
    q1 = float(np.percentile(arr, 25))
    median = float(np.percentile(arr, 50))
    q3 = float(np.percentile(arr, 75))
    iqr = q3 - q1
    lo_thresh = q1 - 1.5 * iqr
    hi_thresh = q3 + 1.5 * iqr
    inside = arr[(arr >= lo_thresh) & (arr <= hi_thresh)]
    lowerfence = float(inside.min()) if inside.size else q1
    upperfence = float(inside.max()) if inside.size else q3
    outliers = [float(v) for v in arr if v < lowerfence or v > upperfence]
    return {
        "q1": q1, "median": median, "q3": q3,
        "lowerfence": lowerfence, "upperfence": upperfence,
        "mean": float(arr.mean()),
        "sd": float(arr.std()),  # population sd -- always defined, even for n=1
        "outliers": outliers,
    }


def _sample_outliers_keep_extremes(outliers: list[float], cap: int,
                                   seed: int = 0) -> list[float]:
    """Downsample a box trace's outlier markers to at most `cap` points,
    but ALWAYS keep the single largest-magnitude outlier on each side (the
    most negative and the most positive value) -- a uniform random sample
    is exactly the sampling strategy least likely to retain a strategy's
    one moonshot or blow-up lot, which is precisely the tail extremity this
    project's box chart needs to show honestly (see lot_return_box_fig's
    docstring). The rest of the budget is filled by a fixed-seed random
    sample of the remaining outliers, so results are reproducible across
    re-renders like the rest of this module's sampling."""
    n = len(outliers)
    if n <= cap:
        return list(outliers)
    if cap <= 0:
        return []
    order = sorted(range(n), key=lambda i: outliers[i])
    if cap == 1:
        # Only room for one point -- keep whichever single value deviates
        # furthest from the median, not an arbitrary side.
        med = float(np.median(outliers))
        best = max(range(n), key=lambda i: abs(outliers[i] - med))
        return [outliers[best]]
    forced = {order[0], order[-1]}  # most negative, most positive
    pool = [i for i in range(n) if i not in forced]
    extra = random.Random(seed).sample(pool, min(cap - len(forced), len(pool)))
    keep = sorted(forced | set(extra))
    return [outliers[i] for i in keep]


def lot_return_box_fig(results_by_exit: dict[str, list[RunResult]],
                       strategy_order: list[str]) -> go.Figure:
    color = _strategy_colors(strategy_order)
    labels = [em.label for em in EXIT_METHODS]
    fig = go.Figure()
    default_label = labels[1] if len(labels) > 1 else labels[0]

    # Pass 1: build each strategy's full (unsampled) lot-return list once.
    # This used to be built twice (once to add the trace, once more just to
    # count how many traces got added for the visibility toggling below),
    # which meant filtering every trade in every run twice over -- keep that
    # single-pass property, but a genuine two-pass split is still required
    # regardless: SCATTER_POINT_BUDGET is a whole-figure budget, so the
    # per-trace sample cap depends on how many non-empty traces there turn
    # out to be, which isn't known until every run has been scanned.
    entries: list[tuple[str, RunResult, list[float]]] = []  # (label, run, rets)
    for label in labels:
        for r in results_by_exit[label]:
            rets = [t.return_pct * 100 for t in r.trades
                    if t.exit_reason in ("expiry", "trim") and not t.delisted]
            if rets:
                entries.append((label, r, rets))

    # NOTE on box-plot honesty: sampling the y-values fed to go.Box used to
    # change the quartiles/whiskers/outliers IT DRAWS -- unlike the scatter
    # chart, where a sampled point is just a point never shown, a sampled
    # box redraws its own summary statistics from the sample, not the full
    # lot set. A strategy's most extreme moonshot/blowup lots are exactly
    # what a small random sample is least likely to retain, so a sampled
    # box chart understates true tail extremity -- and this project has
    # repeatedly found that those extreme lots ARE the P&L.
    #
    # Fix: compute q1/median/q3/whisker-fences/mean/sd from the FULL lot
    # list server-side (see _box_stats) and pass them to go.Box's
    # precomputed-statistics parameters, so the box itself is always exact
    # against every realized lot -- no sampling error, regardless of budget.
    # Only the individual OUTLIER markers drawn on top are downsampled to
    # stay within SCATTER_POINT_BUDGET, and _sample_outliers_keep_extremes
    # always keeps the single largest-magnitude outlier on each side, so
    # the chart's most extreme point never gets randomly dropped.
    per_trace_cap = _per_trace_point_cap(len(entries))

    trace_labels: list[str] = []
    for label, r, rets in entries:
        stats = _box_stats(rets)
        outliers = stats["outliers"]
        n_outliers = len(outliers)
        shown_outliers = _sample_outliers_keep_extremes(outliers, per_trace_cap)
        name = r.strategy if n_outliers <= per_trace_cap else (
            f"{r.strategy} ({per_trace_cap:,} of {n_outliers:,} outlier lots shown)"
        )
        fig.add_trace(go.Box(
            q1=[stats["q1"]], median=[stats["median"]], q3=[stats["q3"]],
            lowerfence=[stats["lowerfence"]], upperfence=[stats["upperfence"]],
            mean=[stats["mean"]], sd=[stats["sd"]],
            y=shown_outliers, boxpoints="outliers",
            name=name,
            marker_color=color[r.strategy],
            boxmean="sd",
            visible=(label == default_label),
            hovertemplate="%{y:.2f}%<extra></extra>",
        ))
        trace_labels.append(label)

    # For visibility toggling: number of traces per exit method may vary
    # (strategies with zero lots are skipped).
    visibility: dict[str, list[bool]] = {}
    for selected_label in labels:
        visibility[selected_label] = [label == selected_label for label in trace_labels]

    layout = _layout_defaults(
        title="Per-lot return distribution — pick an exit method",
        height=520, yaxis_title="Lot return %",
    )
    layout["updatemenus"] = _exit_buttons(visibility)
    fig.update_layout(**layout, boxmode="group")
    fig.add_hline(y=0, line=dict(dash="dot", color="gray"))
    return fig


# ---------------------------------------------------------------------------
# 5. CAGR / Sharpe bar charts (metric selector)
# ---------------------------------------------------------------------------
def metric_bar_fig(summary_df: pd.DataFrame) -> go.Figure:
    df = summary_df[summary_df["strategy"] != "spy_buy_and_hold"].copy()
    spy_row = summary_df[summary_df["strategy"] == "spy_buy_and_hold"]
    fig = go.Figure()
    exit_labels = [em.label for em in EXIT_METHODS if em.label in set(df["exit_method"])]
    metrics = [
        ("cagr", "CAGR", True),
        ("sharpe", "Sharpe", False),
        ("max_drawdown", "Max Drawdown", True),
        ("win_rate", "Win Rate", True),
    ]
    n_exits = len(exit_labels)
    # Trace ordering: metric_idx * n_exits + exit_idx
    for m_key, m_label, is_pct in metrics:
        for i, label in enumerate(exit_labels):
            sub = df[df["exit_method"] == label]
            y = sub[m_key] * (100 if is_pct else 1)
            fig.add_trace(go.Bar(
                x=sub["strategy"], y=y,
                name=label,
                marker_color=PALETTE[i % len(PALETTE)],
                visible=(m_key == "cagr"),
                hovertemplate=("%{x}<br>" + m_label
                               + ": %{y:.2f}" + ("%" if is_pct else "")
                               + f"<extra>{label}</extra>"),
            ))

    # Buttons: switch metric (visibility) AND yaxis title
    n_metrics = len(metrics)
    buttons = []
    for mi, (m_key, m_label, is_pct) in enumerate(metrics):
        vis = []
        for mi2, _ in enumerate(metrics):
            vis.extend([mi2 == mi] * n_exits)
        # SPY reference line (if available) for this metric
        annotations = []
        if not spy_row.empty and m_key in spy_row:
            spy_val = float(spy_row[m_key].iloc[0])
            if spy_val == spy_val:
                annotations = [{
                    "text": f"SPY {m_label}: " + (_pct(spy_val) if is_pct else _num(spy_val)),
                    "xref": "paper", "yref": "paper",
                    "x": 0.99, "y": 1.06, "xanchor": "right", "yanchor": "top",
                    "showarrow": False, "font": {"size": 11, "color": "#666"},
                }]
        buttons.append(dict(
            label=m_label,
            method="update",
            args=[
                {"visible": vis},
                {"yaxis.title.text": m_label + (" (%)" if is_pct else ""),
                 "annotations": annotations},
            ],
        ))

    layout = _layout_defaults(
        title="Strategy performance — pick a metric",
        height=520, yaxis_title="CAGR (%)",
    )
    layout["barmode"] = "group"
    layout["xaxis"]["tickangle"] = -25
    layout["updatemenus"] = [dict(
        type="buttons", direction="right",
        x=0.5, xanchor="center", y=1.10, yanchor="top",
        showactive=True, bgcolor="white", bordercolor="#bbb",
        borderwidth=1, font=dict(size=12), buttons=buttons,
    )]
    # Initialize the SPY annotation for CAGR
    if not spy_row.empty and "cagr" in spy_row:
        spy_val = float(spy_row["cagr"].iloc[0])
        if spy_val == spy_val:
            layout["annotations"] = [{
                "text": f"SPY CAGR: {_pct(spy_val)}",
                "xref": "paper", "yref": "paper",
                "x": 0.99, "y": 1.06, "xanchor": "right", "yanchor": "top",
                "showarrow": False, "font": {"size": 11, "color": "#666"},
            }]
    fig.update_layout(**layout)
    return fig


# ---------------------------------------------------------------------------
# 6. Score vs return scatter
# ---------------------------------------------------------------------------
def score_scatter_fig(results_by_exit: dict[str, list[RunResult]],
                      strategy_order: list[str]) -> go.Figure:
    color = _strategy_colors(strategy_order)
    labels = [em.label for em in EXIT_METHODS]
    fig = go.Figure()
    default_label = labels[1] if len(labels) > 1 else labels[0]
    # One scatter trace per strategy per exit method, color by strategy.
    # A single run can have tens of thousands of lots, and Plotly inlines
    # every marker's x/y/hover-text into the page as JSON. Pass 1 gathers
    # each (label, strategy)'s full (unsampled) realized-trade list, purely
    # to learn n_traces -- SCATTER_POINT_BUDGET is a whole-figure budget, so
    # the per-trace cap isn't known until every run has been scanned. Pass 2
    # samples each trace down to that cap, deterministically (seed 0), and
    # builds the actual x/y/hover-text arrays. Sampling per trace off a
    # shared budget (not one flat cap applied independently per trace,
    # which is what let a 371-trace grid balloon to ~7.4M total points)
    # keeps every strategy/exit combo visible instead of letting either the
    # biggest run crowd out the rest, or the total run count crowd out the
    # per-run detail.
    realized_by_key: dict[tuple[str, str], list] = {}
    for label in labels:
        for r in results_by_exit[label]:
            realized = [t for t in r.trades
                        if t.exit_reason in ("expiry", "trim") and not t.delisted]
            if realized:
                realized_by_key[(label, r.strategy)] = realized

    per_trace_cap = _per_trace_point_cap(len(realized_by_key))

    points: dict[str, dict[str, tuple[list, list, list, int]]] = {label: {} for label in labels}
    for (label, strategy), realized in realized_by_key.items():
        n_total = len(realized)
        if n_total > per_trace_cap:
            sample = random.Random(0).sample(realized, per_trace_cap)
        else:
            sample = realized
        xs = [t.score_at_entry for t in sample]
        ys = [t.return_pct * 100 for t in sample]
        # Plain ASCII separator, not the unicode middle dot used elsewhere in
        # this file: Plotly's JSON encoder escapes non-ASCII as \uXXXX (6
        # bytes) instead of UTF-8 (2 bytes), and this string gets repeated
        # up to per_trace_cap times per trace, so the difference adds up
        # fast at this point count.
        txt = [f"{t.ticker} | entry {t.entry_date} | score {t.score_at_entry} | {_pct(t.return_pct)}"
               for t in sample]
        points[label][strategy] = (xs, ys, txt, n_total)

    trace_keys: list[tuple[str, str]] = []
    for label in labels:
        for s, (xs, ys, txt, n_total) in points[label].items():
            n_shown = len(xs)
            # Make the downsampling visible in the legend itself so nobody
            # mistakes a sampled trace for the full trade log.
            name = s if n_shown == n_total else f"{s} ({n_shown:,} of {n_total:,} lots shown)"
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="markers",
                name=name,
                marker=dict(size=7, opacity=0.55, color=color[s], line=dict(width=0)),
                text=txt,
                hovertemplate="%{text}<extra></extra>",
                visible=(label == default_label),
                legendgroup=s, showlegend=True,
            ))
            trace_keys.append((label, s))

    # Trend line per exit method (one extra trace per exit method)
    fit_keys: list[str] = []
    for label in labels:
        all_x, all_y = [], []
        for s, (xs, ys, _t, _n) in points[label].items():
            all_x.extend(xs); all_y.extend(ys)
        if len(all_x) >= 5 and len(set(all_x)) >= 2:
            try:
                a, b = np.polyfit(all_x, all_y, 1)
                xr = np.linspace(min(all_x), max(all_x), 50)
                fig.add_trace(go.Scatter(
                    x=xr, y=a * xr + b,
                    mode="lines", name=f"fit (slope {a:.2f})",
                    line=dict(color="#444", width=2, dash="dot"),
                    visible=(label == default_label),
                    showlegend=False,
                    hoverinfo="skip",
                ))
                fit_keys.append(label)
            except Exception:
                pass

    visibility: dict[str, list[bool]] = {}
    for selected_label in labels:
        vis = [label == selected_label for label, _ in trace_keys]
        vis.extend(label == selected_label for label in fit_keys)
        visibility[selected_label] = vis

    layout = _layout_defaults(
        title="Conviction score at entry vs lot return",
        height=520,
        xaxis_title="Conviction score at entry",
        yaxis_title="Lot return %",
    )
    layout["updatemenus"] = _exit_buttons(visibility)
    layout["hovermode"] = "closest"
    fig.update_layout(**layout)
    fig.add_hline(y=0, line=dict(dash="dot", color="gray"))
    return fig


# ---------------------------------------------------------------------------
# 7. Daily exposure (one chart per metric, exit-method selector)
# ---------------------------------------------------------------------------
def exposure_fig(results_by_exit: dict[str, list[RunResult]],
                 strategy_order: list[str]) -> go.Figure:
    color = _strategy_colors(strategy_order)
    labels = [em.label for em in EXIT_METHODS]
    fig = go.Figure()
    default_label = labels[1] if len(labels) > 1 else labels[0]
    for label in labels:
        for r in results_by_exit[label]:
            curve = _downsample_uniform(r.exposure_curve, REPORT_MAX_POINTS_PER_CURVE)
            fig.add_trace(go.Scatter(
                x=curve.index, y=curve.values * 100,
                name=r.strategy,
                line=dict(color=color[r.strategy], width=1.5),
                visible=(label == default_label),
                hovertemplate=("%{y:.1f}%"
                               f"<extra>{r.strategy} ({label})</extra>"),
            ))
    visibility = {}
    for selected_label in labels:
        vis = []
        for label in labels:
            for _ in results_by_exit[label]:
                vis.append(label == selected_label)
        visibility[selected_label] = vis
    layout = _layout_defaults(
        title="Daily NAV exposure — pick an exit method",
        height=460, yaxis_title="% of NAV invested",
    )
    layout["updatemenus"] = _exit_buttons(visibility)
    fig.update_layout(**layout)
    return fig


# ---------------------------------------------------------------------------
# 8. Skip diagnostics
# ---------------------------------------------------------------------------
def skip_diag_fig(results_by_exit: dict[str, list[RunResult]]) -> go.Figure:
    rows = []
    for label, runs in results_by_exit.items():
        for r in runs:
            rows.append({
                "label": f"{r.strategy}<br>({label})",
                "Capacity": r.skips.get("capacity", 0),
                "Liquidity": r.skips.get("liquidity", 0),
                "Cash": r.skips.get("cash", 0),
                "No price": r.skips.get("no_price", 0),
                "Participation": r.skips.get("participation", 0),
                "_total": sum(r.skips.values()),
            })
    df = pd.DataFrame(rows).sort_values("_total", ascending=False)
    fig = go.Figure()
    for col, c in [("Capacity", "#0072B2"), ("Liquidity", "#E69F00"),
                   ("Cash", "#009E73"), ("No price", "#CC79A7"),
                   ("Participation", "#D55E00")]:
        fig.add_trace(go.Bar(
            x=df["label"], y=df[col], name=col, marker_color=c,
            hovertemplate="%{y}<extra>%{x} — " + col + "</extra>",
        ))
    layout = _layout_defaults(
        title="Signals skipped (sorted by total)",
        height=480, yaxis_title="# signals skipped",
    )
    layout["barmode"] = "stack"
    layout["xaxis"]["tickangle"] = -45
    fig.update_layout(**layout)
    return fig


# ---------------------------------------------------------------------------
# D1. Per-score-bucket cohort table
# ---------------------------------------------------------------------------
SCORE_BUCKETS: list[tuple[str, int, int]] = [
    # (label, lo, hi) — interval is [lo, hi). Use ±999 for open ends.
    ("<0", -999, 0),
    ("0",   0,   1),
    ("1",   1,   2),
    ("2",   2,   3),
    ("3",   3,   4),
    ("4+",  4, 999),
]


def _score_cohort_table_html(results_by_exit: dict[str, list[RunResult]]) -> str:
    rows_html: list[str] = []
    for exit_label in sorted(results_by_exit.keys()):
        for r in results_by_exit[exit_label]:
            realized = [t for t in r.trades
                        if t.exit_reason in ("expiry", "trim", "stop_loss", "trailing_stop")
                        and not t.delisted]
            if not realized:
                continue
            for label, lo, hi in SCORE_BUCKETS:
                bucket = [t for t in realized if lo <= t.score_at_entry < hi]
                if not bucket:
                    continue
                n = len(bucket)
                avg = float(np.mean([t.return_pct for t in bucket]))
                med = float(np.median([t.return_pct for t in bucket]))
                wins = sum(1 for t in bucket if t.return_pct > 0)
                wr = wins / n
                rows_html.append(
                    '<tr>'
                    f'<td data-sort="{r.strategy}" style="text-align:left;">{r.strategy}</td>'
                    f'<td data-sort="{exit_label}" style="text-align:left;">{exit_label}</td>'
                    f'<td data-sort="{lo}" style="text-align:right;">{label}</td>'
                    f'<td data-sort="{n}" style="text-align:right;">{n}</td>'
                    f'<td data-sort="{avg}" style="text-align:right;">{_pct(avg)}</td>'
                    f'<td data-sort="{med}" style="text-align:right;">{_pct(med)}</td>'
                    f'<td data-sort="{wr}" style="text-align:right;">{_pct(wr, signed=False)}</td>'
                    '</tr>'
                )
    cols = [
        ("strategy", "Strategy", "text"),
        ("exit_method", "Exit method", "text"),
        ("bucket", "Score bucket", "int"),
        ("n", "Lots", "int"),
        ("avg", "Avg return", "pct"),
        ("median", "Median return", "pct"),
        ("win_rate", "Win rate", "pct_unsigned"),
    ]
    header_html = "".join(
        f'<th data-col="{k}" data-kind="{kind}">{label}'
        f'<span class="sort-indicator"></span></th>'
        for (k, label, kind) in cols
    )
    return f"""
<div class="table-wrap">
  <table class="summary-table">
    <thead><tr>{header_html}</tr></thead>
    <tbody>{"".join(rows_html)}</tbody>
  </table>
</div>
"""


# ---------------------------------------------------------------------------
# D2. Capacity / skip-rate table
# ---------------------------------------------------------------------------
def _capacity_table_html(results_by_exit: dict[str, list[RunResult]]) -> str:
    rows_html: list[str] = []
    for exit_label in sorted(results_by_exit.keys()):
        for r in results_by_exit[exit_label]:
            cap = r.skips.get("capacity", 0)
            liq = r.skips.get("liquidity", 0)
            cash = r.skips.get("cash", 0)
            nop = r.skips.get("no_price", 0)
            pf = r.skips.get("price_floor", 0)
            part = r.skips.get("participation", 0)
            total_signals = (
                sum(1 for t in r.trades if t.exit_reason != "final_liquidation")
                + cap + liq + cash + nop + pf + part
            )
            denom = max(1, total_signals)
            # Truncated-but-executed orders (skips["participation_capped"])
            # are deliberately excluded here: they produced a real trade, so
            # they belong in `executed`, not in the not-executed skip counts
            # below. Only a fully-rejected participation order (`part`) —
            # capped down to a dust-sized amount, see engine.py — counts as
            # not executed, the same way the other skip reasons do.
            executed = total_signals - (cap + liq + cash + nop + pf + part)
            def pct(x): return x / denom
            rows_html.append(
                '<tr>'
                f'<td data-sort="{r.strategy}" style="text-align:left;">{r.strategy}</td>'
                f'<td data-sort="{exit_label}" style="text-align:left;">{exit_label}</td>'
                f'<td data-sort="{total_signals}" style="text-align:right;">{total_signals}</td>'
                f'<td data-sort="{executed}" style="text-align:right;">{executed}</td>'
                f'<td data-sort="{pct(executed)}" style="text-align:right;">{_pct(pct(executed), signed=False)}</td>'
                f'<td data-sort="{pct(cap)}" style="text-align:right;">{_pct(pct(cap), signed=False)}</td>'
                f'<td data-sort="{pct(liq)}" style="text-align:right;">{_pct(pct(liq), signed=False)}</td>'
                f'<td data-sort="{pct(cash)}" style="text-align:right;">{_pct(pct(cash), signed=False)}</td>'
                f'<td data-sort="{pct(nop)}" style="text-align:right;">{_pct(pct(nop), signed=False)}</td>'
                f'<td data-sort="{pct(pf)}" style="text-align:right;">{_pct(pct(pf), signed=False)}</td>'
                f'<td data-sort="{pct(part)}" style="text-align:right;">{_pct(pct(part), signed=False)}</td>'
                '</tr>'
            )
    cols = [
        ("strategy", "Strategy", "text"),
        ("exit_method", "Exit method", "text"),
        ("total", "Total signals", "int"),
        ("exec", "Executed", "int"),
        ("exec_pct", "Executed %", "pct_unsigned"),
        ("cap_pct", "Cap skip %", "pct_unsigned"),
        ("liq_pct", "Liq skip %", "pct_unsigned"),
        ("cash_pct", "Cash skip %", "pct_unsigned"),
        ("nop_pct", "NoPrice skip %", "pct_unsigned"),
        ("pf_pct", "PriceFloor skip %", "pct_unsigned"),
        ("part_pct", "Participation skip %", "pct_unsigned"),
    ]
    header_html = "".join(
        f'<th data-col="{k}" data-kind="{kind}">{label}'
        f'<span class="sort-indicator"></span></th>'
        for (k, label, kind) in cols
    )
    return f"""
<div class="table-wrap">
  <table class="summary-table">
    <thead><tr>{header_html}</tr></thead>
    <tbody>{"".join(rows_html)}</tbody>
  </table>
</div>
"""


# ---------------------------------------------------------------------------
# D3. Alpha / beta / IR vs SPY (read from summary_df)
# ---------------------------------------------------------------------------
def _alpha_beta_table_html(summary_df: pd.DataFrame) -> str:
    df = summary_df[summary_df["strategy"] != "spy_buy_and_hold"].copy()
    cols = [
        ("strategy", "Strategy", "text"),
        ("exit_method", "Exit method", "text"),
        ("in_sample_frac", "In-Sample Frac", "pct_unsigned"),
        ("alpha_ann", "Alpha (ann.)", "pct"),
        ("beta", "Beta", "num"),
        ("r_squared", "R²", "num"),
        ("t_alpha", "t(alpha)", "num"),
        ("tracking_error", "Tracking err (ann.)", "pct_unsigned"),
        ("info_ratio", "Info ratio", "num"),
    ]
    cols = [c for c in cols if c[0] in df.columns or c[0] in ("strategy", "exit_method")]

    def fmt(val, kind):
        if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
            return "—", float("nan")
        if kind == "pct":
            return _pct(val), float(val)
        if kind == "pct_unsigned":
            return _pct(val, signed=False), float(val)
        if kind == "num":
            return _num(val), float(val)
        if kind == "int":
            return f"{int(val)}", float(val)
        return str(val), 0.0

    def in_sample_bg(val):
        # This is the table a reader sorts by t(alpha) to find the
        # strongest result -- thr_gt_p11's t_alpha=2.061 lives here. The
        # in_sample_frac column must carry its own highlight so a
        # t-alpha-sorted read can't miss a partly-in-sample row even
        # without noticing the banner above the main summary table.
        if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
            return ""
        if val > IN_SAMPLE_WARNING_THRESHOLD:
            return "background-color: rgb(255,224,224); font-weight: 600;"
        return ""

    rows_html: list[str] = []
    for _, row in df.iterrows():
        cells = []
        for (k, _label, kind) in cols:
            disp, raw = fmt(row.get(k), kind)
            style = in_sample_bg(raw) if k == "in_sample_frac" else ""
            align = "left" if kind == "text" else "right"
            cells.append(
                f'<td data-sort="{raw}" style="{style} text-align:{align};">{disp}</td>'
            )
        rows_html.append(f"<tr>{''.join(cells)}</tr>")
    header_html = "".join(
        f'<th data-col="{k}" data-kind="{kind}">{label}'
        f'<span class="sort-indicator"></span></th>'
        for (k, label, kind) in cols
    )
    return f"""
{_in_sample_warnings_html(df)}
<div class="table-wrap">
  <table class="summary-table">
    <thead><tr>{header_html}</tr></thead>
    <tbody>{"".join(rows_html)}</tbody>
  </table>
</div>
"""


# ---------------------------------------------------------------------------
# D4. Calendar-year contribution table
# ---------------------------------------------------------------------------
def _calendar_year_table_html(results_by_exit: dict[str, list[RunResult]]) -> str:
    rows_html: list[str] = []
    for exit_label in sorted(results_by_exit.keys()):
        for r in results_by_exit[exit_label]:
            realized = [t for t in r.trades
                        if t.exit_reason in ("expiry", "trim", "stop_loss", "trailing_stop")
                        and not t.delisted]
            if not realized:
                continue
            by_year: dict[int, list] = {}
            for t in realized:
                y = t.exit_date.year
                by_year.setdefault(y, []).append(t.return_pct)
            for y in sorted(by_year.keys()):
                vals = by_year[y]
                n = len(vals)
                avg = float(np.mean(vals))
                wins = sum(1 for v in vals if v > 0)
                wr = wins / n
                rows_html.append(
                    '<tr>'
                    f'<td data-sort="{r.strategy}" style="text-align:left;">{r.strategy}</td>'
                    f'<td data-sort="{exit_label}" style="text-align:left;">{exit_label}</td>'
                    f'<td data-sort="{y}" style="text-align:right;">{y}</td>'
                    f'<td data-sort="{n}" style="text-align:right;">{n}</td>'
                    f'<td data-sort="{avg}" style="text-align:right;">{_pct(avg)}</td>'
                    f'<td data-sort="{wr}" style="text-align:right;">{_pct(wr, signed=False)}</td>'
                    '</tr>'
                )
    cols = [
        ("strategy", "Strategy", "text"),
        ("exit_method", "Exit method", "text"),
        ("year", "Year", "int"),
        ("n", "Lots closed", "int"),
        ("avg", "Avg lot return", "pct"),
        ("win_rate", "Win rate", "pct_unsigned"),
    ]
    header_html = "".join(
        f'<th data-col="{k}" data-kind="{kind}">{label}'
        f'<span class="sort-indicator"></span></th>'
        for (k, label, kind) in cols
    )
    return f"""
<div class="table-wrap">
  <table class="summary-table">
    <thead><tr>{header_html}</tr></thead>
    <tbody>{"".join(rows_html)}</tbody>
  </table>
</div>
"""


# ---------------------------------------------------------------------------
# D5. Stop-loss exit breakdown
# ---------------------------------------------------------------------------
def _stop_loss_table_html(results_by_exit: dict[str, list[RunResult]]) -> str | None:
    rows_html: list[str] = []
    any_stops = False
    for exit_label in sorted(results_by_exit.keys()):
        for r in results_by_exit[exit_label]:
            by_reason: dict[str, list] = {}
            for t in r.trades:
                if t.delisted:
                    continue
                by_reason.setdefault(t.exit_reason, []).append(t.return_pct)
            if "stop_loss" not in by_reason and "trailing_stop" not in by_reason:
                continue
            any_stops = True
            for reason in ("expiry", "trim", "stop_loss", "trailing_stop", "final_liquidation"):
                vals = by_reason.get(reason, [])
                if not vals:
                    continue
                n = len(vals)
                avg = float(np.mean(vals))
                rows_html.append(
                    '<tr>'
                    f'<td data-sort="{r.strategy}" style="text-align:left;">{r.strategy}</td>'
                    f'<td data-sort="{exit_label}" style="text-align:left;">{exit_label}</td>'
                    f'<td data-sort="{reason}" style="text-align:left;">{reason}</td>'
                    f'<td data-sort="{n}" style="text-align:right;">{n}</td>'
                    f'<td data-sort="{avg}" style="text-align:right;">{_pct(avg)}</td>'
                    '</tr>'
                )
    if not any_stops:
        return None
    cols = [
        ("strategy", "Strategy", "text"),
        ("exit_method", "Exit method", "text"),
        ("reason", "Exit reason", "text"),
        ("n", "Lots", "int"),
        ("avg", "Avg return", "pct"),
    ]
    header_html = "".join(
        f'<th data-col="{k}" data-kind="{kind}">{label}'
        f'<span class="sort-indicator"></span></th>'
        for (k, label, kind) in cols
    )
    return f"""
<div class="table-wrap">
  <table class="summary-table">
    <thead><tr>{header_html}</tr></thead>
    <tbody>{"".join(rows_html)}</tbody>
  </table>
</div>
"""


# ---------------------------------------------------------------------------
# D6. Learned signal weights (backtest/signal_fit.py FitResult)
# ---------------------------------------------------------------------------
def _signal_weights_table_html(fit) -> str:
    """Weights table: one row per component, hand weight vs learned
    coefficient/points, plus per-horizon diagnostic coefficients."""
    df = fit.feature_stats
    cols = [
        ("component", "Component", "text"),
        ("hand_weight", "Hand weight", "signed_int"),
        ("coef", f"Learned coef (adj {fit.primary_horizon}d)", "coef_pct"),
        ("raw_uplift", "Raw uplift (train)", "coef_pct"),
        ("t_stat", "t", "num"),
        ("n_fired_train", "Fired (train)", "int"),
        ("points", "Learned points", "points"),
    ]
    # Per-horizon diagnostic coefficients: dynamic over whatever coef_<h>
    # columns are present (FitConfig.horizons is configurable), sorted
    # numerically by horizon rather than hardcoded to a fixed set.
    horizon_col_names = sorted(
        (c for c in df.columns if c.startswith("coef_") and c != "coef"),
        key=lambda c: int(c.split("_", 1)[1]),
    )
    cols += [(c, f"coef @{c.split('_', 1)[1]}d", "coef_pct") for c in horizon_col_names]

    def fmt(val, kind):
        if kind == "text":
            return str(val), 0.0
        if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
            return "—", float("nan")
        if kind == "signed_int":
            return f"{int(val):+d}", float(val)
        if kind == "coef_pct":
            return _pct(val), float(val)
        if kind == "num":
            return _num(val), float(val)
        if kind == "int":
            return f"{int(val)}", float(val)
        if kind == "points":
            return f"<b>{int(val):+d}</b>", float(val)
        return str(val), 0.0

    rows_html: list[str] = []
    for _, row in df.iterrows():
        cells = []
        for (k, _label, kind) in cols:
            disp, raw = fmt(row.get(k), kind)
            align = "left" if kind == "text" else "right"
            cells.append(
                f'<td data-sort="{raw}" style="text-align:{align};">{disp}</td>'
            )
        rows_html.append(f"<tr>{''.join(cells)}</tr>")
    header_html = "".join(
        f'<th data-col="{k}" data-kind="{kind}">{label}'
        f'<span class="sort-indicator"></span></th>'
        for (k, label, kind) in cols
    )

    intro = f"""
<p class="note">
  Weights fit by ridge regression on SPY-adjusted forward returns of cluster-episode starts
  ({fit.primary_horizon}-trading-day horizon), <b>train period only</b>: {fit.train_start} to
  {fit.train_end} (split {fit.split_date}; {fit.train_events} train / {fit.test_events} test events;
  &lambda;={fit.lambda_used:g}). Target: {fit.target_desc}. The learned_* strategies elsewhere in this
  report show <b>full-period</b> equity curves that include these in-sample training days &mdash; judge
  them on the out-of-sample tables below, not the full-period charts.
</p>
<p class="note">
  <b>Raw uplift (train)</b> is the unwinsorized cohort mean difference &mdash;
  mean(adj {fit.primary_horizon}d return | flag fired) minus mean(... | flag not fired) &mdash;
  computed on the raw (unclipped) train-split target. Unlike the learned coefficient, it isn't
  shrunk by ridge regularization, isn't netted against the other fired flags, and isn't muted by
  winsorization, so it's useful for spotting tail-driven effects (e.g. a component whose entire
  edge lives in a handful of moonshot outcomes) that the regression coefficient alone can hide.
</p>
"""
    return f"""
{intro}
<div class="table-wrap">
  <table class="summary-table">
    <thead><tr>{header_html}</tr></thead>
    <tbody>{"".join(rows_html)}</tbody>
  </table>
</div>
"""


def _signal_oos_table_html(fit, oos_df: pd.DataFrame) -> str:
    """Two out-of-sample tables: (a) event-level score buckets, hand vs
    learned scoring, and (b) strategy-level OOS return/CAGR/Sharpe for the
    learned_* strategies against the strongest hand-tuned strategies + SPY."""
    bucket_df = fit.oos_bucket_table
    bucket_cols = [
        ("scoring", "Scoring", "text"),
        ("bucket", "Score bucket", "text"),
        ("n", "Lots", "int"),
        ("mean_adj", "Mean adj return", "pct"),
        ("median_adj", "Median adj return", "pct"),
        ("win_rate", "Win rate", "pct_unsigned"),
    ]

    def fmt(val, kind):
        if kind == "text":
            return str(val), 0.0
        if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
            return "—", float("nan")
        if kind == "pct":
            return _pct(val), float(val)
        if kind == "pct_unsigned":
            return _pct(val, signed=False), float(val)
        if kind == "int":
            return f"{int(val)}", float(val)
        if kind == "num":
            return _num(val), float(val)
        return str(val), 0.0

    bucket_rows: list[str] = []
    for _, row in bucket_df.iterrows():
        cells = []
        for (k, _label, kind) in bucket_cols:
            disp, raw = fmt(row.get(k), kind)
            align = "left" if kind == "text" else "right"
            cells.append(
                f'<td data-sort="{raw}" style="text-align:{align};">{disp}</td>'
            )
        bucket_rows.append(f"<tr>{''.join(cells)}</tr>")
    bucket_header = "".join(
        f'<th data-col="{k}" data-kind="{kind}">{label}'
        f'<span class="sort-indicator"></span></th>'
        for (k, label, kind) in bucket_cols
    )
    bucket_html = f"""
<p style="font-weight:600; color:var(--accent); margin:20px 0 6px;">
  OOS event-level buckets &mdash; hand vs learned score</p>
<p class="note">Test-set events only (event day &ge; {fit.split_date}), bucketed into integer
  conviction tiers under each scoring scheme.</p>
<div class="table-wrap">
  <table class="summary-table">
    <thead><tr>{bucket_header}</tr></thead>
    <tbody>{"".join(bucket_rows)}</tbody>
  </table>
</div>
"""

    keep_strategies = {
        "learned_gt_p00", "learned_gt_p03", "learned_score_weighted",
        "ten_percent_owner_gated", "conviction_only", "score_weighted",
        "spy_buy_and_hold",
    }
    strat_cols = [
        ("strategy", "Strategy", "text"),
        ("exit", "Exit method", "text"),
        ("oos_total_return", "OOS Total Return", "pct"),
        ("oos_cagr", "OOS CAGR", "pct"),
        ("oos_sharpe", "OOS Sharpe", "num"),
    ]
    if oos_df is not None and not oos_df.empty:
        strat_df = oos_df[oos_df["strategy"].isin(keep_strategies)]
    else:
        strat_df = pd.DataFrame(columns=[c for c, _l, _k in strat_cols])

    strat_rows: list[str] = []
    for _, row in strat_df.iterrows():
        cells = []
        for (k, _label, kind) in strat_cols:
            disp, raw = fmt(row.get(k), kind)
            align = "left" if kind == "text" else "right"
            cells.append(
                f'<td data-sort="{raw}" style="text-align:{align};">{disp}</td>'
            )
        cls = "spy" if row.get("strategy") == "spy_buy_and_hold" else ""
        strat_rows.append(f'<tr class="{cls}">{"".join(cells)}</tr>')
    strat_header = "".join(
        f'<th data-col="{k}" data-kind="{kind}">{label}'
        f'<span class="sort-indicator"></span></th>'
        for (k, label, kind) in strat_cols
    )
    strat_html = f"""
<p style="font-weight:600; color:var(--accent); margin:20px 0 6px;">
  OOS strategy performance</p>
<p class="note">Equity curves rebased to 1.0 at the OOS split date {fit.split_date}; return/CAGR/Sharpe
  computed on the test-set window only.</p>
<div class="table-wrap">
  <table class="summary-table">
    <thead><tr>{strat_header}</tr></thead>
    <tbody>{"".join(strat_rows)}</tbody>
  </table>
</div>
"""
    return bucket_html + strat_html


# ---------------------------------------------------------------------------
# D7. Tail-probability ("moonshot") score (backtest/signal_fit.py TailFitResult)
# ---------------------------------------------------------------------------
def _tail_score_section_html(tail_fit) -> str:
    """Research-triage score section: per-component log2-lift table (with
    points) and the OOS quintile validation table. This is a second,
    independent scoring output alongside the mean-return ridge fit above —
    it predicts P(big win), not mean return, because cluster outcomes are
    lottery-distributed (see backtest/signal_fit.py's module docstring and
    fit_tail_score's docstring)."""
    stats = tail_fit.tail_stats
    cols = [
        ("component", "Component", "text"),
        ("hand_weight", "Hand weight", "signed_int"),
        ("lift", "log2 lift", "num"),
        ("lift_h1", "Lift 1st half", "num"),
        ("lift_h2", "Lift 2nd half", "num"),
        ("p_moonshot", "P(moonshot|flag)", "pct_unsigned"),
        ("n_fired_train", "Fired (train)", "int"),
        ("points", "Tail points", "points"),
    ]

    def fmt(val, kind):
        if kind == "text":
            return str(val), 0.0
        if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
            return "—", float("nan")
        if kind == "signed_int":
            return f"{int(val):+d}", float(val)
        if kind == "num":
            return _num(val), float(val)
        if kind == "pct_unsigned":
            return _pct(val, signed=False), float(val)
        if kind == "int":
            return f"{int(val)}", float(val)
        if kind == "points":
            return f"<b>{int(val):+d}</b>", float(val)
        return str(val), 0.0

    rows_html: list[str] = []
    for _, row in stats.iterrows():
        cells = []
        for (k, _label, kind) in cols:
            disp, raw = fmt(row.get(k), kind)
            align = "left" if kind == "text" else "right"
            cells.append(
                f'<td data-sort="{raw}" style="text-align:{align};">{disp}</td>'
            )
        rows_html.append(f"<tr>{''.join(cells)}</tr>")
    header_html = "".join(
        f'<th data-col="{k}" data-kind="{kind}">{label}'
        f'<span class="sort-indicator"></span></th>'
        for (k, label, kind) in cols
    )

    intro = f"""
<p style="font-weight:600; color:var(--accent); margin:28px 0 6px;">
  Tail-probability score (research triage)</p>
<p class="note">
  A second, independent scoring output: predicts P(adjusted return &gt;
  {_pct(tail_fit.moonshot_thresh, signed=False)} at {tail_fit.tail_horizon} trading days) &mdash; a
  "moonshot" &mdash; rather than mean return. Cluster outcomes are lottery-distributed, so the
  mean-return fit above correctly flags "avoid" components but can't rank clusters by upside; this
  score does. Fitted on <b>train only</b>: {tail_fit.train_start} to {tail_fit.train_end}
  (split {tail_fit.split_date}; {tail_fit.train_events} train / {tail_fit.test_events} test events
  with a valid target). Each flag's points are gated on train support, minimum |log2 lift|, and a
  stability check requiring the lift to carry the same sign on both the first and second
  chronological halves of train.
</p>
<div class="table-wrap">
  <table class="summary-table">
    <thead><tr>{header_html}</tr></thead>
    <tbody>{"".join(rows_html)}</tbody>
  </table>
</div>
"""

    q = tail_fit.oos_quintile_table
    q_cols = [
        ("bucket", "Bucket", "text"),
        ("n", "n", "int"),
        ("p_moonshot", "P(moonshot)", "pct_unsigned"),
        ("mean_adj", "Mean adj return", "pct"),
        ("median_adj", "Median adj return", "pct"),
        ("score_lo", "Score range lo", "num"),
        ("score_hi", "Score range hi", "num"),
    ]

    def qfmt(val, kind):
        if kind == "text":
            return str(val), 0.0
        if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
            return "—", float("nan")
        if kind == "pct":
            return _pct(val), float(val)
        if kind == "pct_unsigned":
            return _pct(val, signed=False), float(val)
        if kind == "int":
            return f"{int(val)}", float(val)
        if kind == "num":
            return _num(val), float(val)
        return str(val), 0.0

    q_rows_html: list[str] = []
    for _, row in q.iterrows():
        cells = []
        for (k, _label, kind) in q_cols:
            disp, raw = qfmt(row.get(k), kind)
            align = "left" if kind == "text" else "right"
            cells.append(
                f'<td data-sort="{raw}" style="text-align:{align};">{disp}</td>'
            )
        q_rows_html.append(f"<tr>{''.join(cells)}</tr>")
    q_header_html = "".join(
        f'<th data-col="{k}" data-kind="{kind}">{label}'
        f'<span class="sort-indicator"></span></th>'
        for (k, label, kind) in q_cols
    )
    q_html = f"""
<p style="font-weight:600; color:var(--accent); margin:20px 0 6px;">
  OOS quintiles &mdash; tail score vs P(moonshot)</p>
<p class="note">
  Test-set events only (event day &ge; {tail_fit.split_date}), bucketed into quintiles of the
  summed tail-score points. Train base rate P(adj {tail_fit.tail_horizon}d &gt;
  {_pct(tail_fit.moonshot_thresh, signed=False)}) = {_pct(tail_fit.base_rate, signed=False)} &mdash;
  compare each bucket's P(moonshot) against that baseline.
</p>
<div class="table-wrap">
  <table class="summary-table">
    <thead><tr>{q_header_html}</tr></thead>
    <tbody>{"".join(q_rows_html)}</tbody>
  </table>
</div>
"""
    return intro + q_html


# ---------------------------------------------------------------------------
# Caveats text
# ---------------------------------------------------------------------------
CAVEATS_HTML = """
<ol>
  <li><b>Survivorship bias is asymmetric.</b> Insider filings are point-in-time clean; yfinance's
      ticker universe is current-day, so delisted/bankrupt micro-caps fail silently. Direction:
      likely overstates returns because dropped tickers disproportionately precede bankruptcy.</li>
  <li><b>Lookahead avoided.</b> Each daily decision sees only filings with
      <code>filing_date &le; that day</code>; execution at the <em>next</em> trading day's open with
      10 bps slippage per side.</li>
  <li><b>Scorer drift / in-sample risk.</b> <code>_score_cluster</code> was developed alongside
      data overlapping the backtest window. See the score-vs-return scatter to judge robustness.</li>
  <li><b>No tax modeling.</b> All returns pre-tax; short-term capital gains apply.</li>
  <li><b>Liquidity floor</b> at $500k 20-day median dollar volume excludes some of the
      highest-conviction nano-cap signals.</li>
  <li><b>Fixed-% of starting capital</b> sizing (not NAV-based) — compare CAGRs, not gross dollar PnL.</li>
  <li><b>Daily-rebalance slippage compounds.</b> Strategies that thrash (especially the
      continuous score_weighted) pay slippage on every move.</li>
  <li><b>Signal-weight fit is a single chronological split, not an expanding-window refit.</b>
      The learned weights are fit once on the train period's 70% and held fixed for the rest of
      history (see the Learned signal weights section) — a production refit schedule would
      re-fit periodically as new data arrives. Also, the reported OLS t-stats come from
      overlapping H-day forward-return windows, which inflates significance versus a
      true iid-residual model; treat them as an indicative ranking signal, not a strict
      significance test.</li>
</ol>
"""


# ---------------------------------------------------------------------------
# HTML shell + JS
# ---------------------------------------------------------------------------
SORT_JS = """
(function(){
  document.querySelectorAll('table.summary-table').forEach(function(tbl){
    var headers = tbl.querySelectorAll('th[data-col]');
    var tbody = tbl.querySelector('tbody');
    var rows = Array.from(tbody.querySelectorAll('tr'));
    var sortState = {col: null, dir: 1};
    headers.forEach(function(th, idx){
      th.style.cursor = 'pointer';
      th.addEventListener('click', function(){
        var kind = th.getAttribute('data-kind');
        if(sortState.col === idx){ sortState.dir *= -1; }
        else { sortState.col = idx; sortState.dir = 1; }
        headers.forEach(function(h){
          var s = h.querySelector('.sort-indicator');
          if(s) s.textContent = '';
        });
        var ind = th.querySelector('.sort-indicator');
        if(ind) ind.textContent = sortState.dir > 0 ? ' \\u25B2' : ' \\u25BC';
        rows.sort(function(a, b){
          var av = a.children[idx].getAttribute('data-sort');
          var bv = b.children[idx].getAttribute('data-sort');
          if(kind === 'text'){
            return sortState.dir * av.localeCompare(bv);
          } else {
            var an = parseFloat(av); var bn = parseFloat(bv);
            if(isNaN(an)) an = -Infinity;
            if(isNaN(bn)) bn = -Infinity;
            return sortState.dir * (an - bn);
          }
        });
        rows.forEach(function(r){ tbody.appendChild(r); });
      });
    });
  });
})();
"""

CSS = """
:root {
  --fg: #222; --muted: #666; --border: #e3e3e7; --accent: #2c3e50;
}
* { box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       margin: 0; padding: 0; color: var(--fg); background: #fff; }
header { background: var(--accent); color: white; padding: 22px 32px; }
header h1 { margin: 0 0 6px 0; font-size: 22px; font-weight: 500; }
header .config { font-size: 12px; opacity: 0.9; line-height: 1.7; }
header .config b { color: #f0c674; font-weight: 600; }
.layout { display: flex; min-height: calc(100vh - 110px); }
nav {
  width: 210px; padding: 24px 18px; background: #f7f7f9;
  border-right: 1px solid var(--border); position: sticky; top: 0;
  height: 100vh; overflow-y: auto; font-size: 13px;
}
nav h3 { margin: 0 0 12px 0; font-size: 11px; text-transform: uppercase;
         letter-spacing: 0.08em; color: var(--muted); font-weight: 600; }
nav ul { list-style: none; padding: 0; margin: 0; }
nav li { margin: 8px 0; }
nav a { color: var(--accent); text-decoration: none; display: block;
        padding: 4px 8px; border-radius: 3px; }
nav a:hover { background: #e9e9ee; }
main { flex: 1; padding: 24px 32px; min-width: 0; max-width: 1500px; }
section { margin-bottom: 44px; }
section h2 { margin: 0 0 16px 0; padding-bottom: 8px;
             border-bottom: 2px solid #eaeaea; color: var(--accent);
             font-size: 18px; font-weight: 600; }
section p.note { color: var(--muted); font-size: 12px; margin: -8px 0 12px 0; }
code { background: #f0f0f4; padding: 1px 5px; border-radius: 3px;
       font-size: 0.92em; }
footer { padding: 18px 32px; color: var(--muted); font-size: 11px;
         border-top: 1px solid var(--border); }
#caveats ol { padding-left: 20px; }
#caveats ol li { margin-bottom: 10px; line-height: 1.55; font-size: 13px; }

.concentration-warnings { margin-bottom: 12px; }
.concentration-warning {
  background: #fff3cd; border: 1px solid #e6a817; border-left: 4px solid #d9534f;
  color: #6b4a00; padding: 8px 12px; border-radius: 4px; font-size: 13px;
  margin-bottom: 6px; line-height: 1.5;
}
.table-wrap { overflow-x: auto; border: 1px solid var(--border); border-radius: 4px; }
table.summary-table { width: 100%; border-collapse: collapse; font-size: 12px; }
table.summary-table thead { background: var(--accent); color: white; }
table.summary-table thead th {
  padding: 10px 10px; text-align: left; white-space: nowrap;
  position: sticky; top: 0; user-select: none;
}
table.summary-table thead th:hover { background: #34495e; }
table.summary-table tbody td { padding: 7px 10px; border-bottom: 1px solid #eee;
                                white-space: nowrap; font-variant-numeric: tabular-nums; }
table.summary-table tbody tr:nth-child(odd) { background: #fbfbfd; }
table.summary-table tbody tr.spy { background: #fff5cc; font-weight: 600; }
table.summary-table tbody tr.spy:nth-child(odd) { background: #fff5cc; }
table.summary-table tbody tr:hover { background: #eef5ff; }
table.summary-table tbody tr.spy:hover { background: #ffe89a; }
.sort-indicator { color: #f0c674; font-size: 10px; margin-left: 3px; }
"""


def render_html(*, summary_df: pd.DataFrame,
                results_by_exit: dict[str, list[RunResult]],
                spy_result: RunResult, strategy_order: list[str],
                config: dict, offline: bool = False,
                fit_summary: dict | None = None) -> str:
    log.info("Building Plotly figures…")
    stop_loss_html = _stop_loss_table_html(results_by_exit)
    # Equity/drawdown/exposure/score-scatter/per-lot-box are the figures
    # that plot per-run data at a resolution (daily curve, or raw per-lot
    # values) big enough to be a file-size risk (everything else is a
    # per-strategy aggregate, e.g. metric_bar_fig), so all five get the
    # top-N run filter. score_scatter_fig / lot_return_box_fig additionally
    # get a per-lot point-budget downsample on top of that (see
    # SCATTER_POINT_BUDGET) since a run's trade log can be tens of
    # thousands of lots even after the run itself makes the top N.
    top_results_by_exit, n_curves_plotted, n_curves_total = _select_top_runs(
        results_by_exit, summary_df, REPORT_MAX_CURVES,
    )
    equity_note = _downsampling_note_html(n_curves_plotted, n_curves_total)
    drawdown_note = _downsampling_note_html(n_curves_plotted, n_curves_total, drawdown=True)
    exposure_note = _downsampling_note_html(n_curves_plotted, n_curves_total, include_spy=False)
    lot_trace_cap = _per_trace_point_cap(_lot_trace_count(top_results_by_exit))
    scatter_note = _lot_downsampling_note_html(n_curves_plotted, n_curves_total, lot_trace_cap)
    violins_note = _lot_downsampling_note_html(n_curves_plotted, n_curves_total, lot_trace_cap,
                                               exact_stats=True)
    sections = [
        ("summary", "Summary", None),  # special: HTML table not Plotly
        ("legend", "Strategy legend", None),  # HTML <details> list
        ("score_cohort", "Per-score cohort", None),  # HTML table
    ]
    if fit_summary is not None:
        sections.append(("signal_fit", "Learned signal weights", None))  # HTML table
    sections += [
        ("capacity", "Capacity / skip rates", None),  # HTML table
        ("alpha_beta", "Alpha / beta vs SPY", None),  # HTML table
        ("equity", "Equity curves",
         equity_curves_fig(top_results_by_exit, spy_result, strategy_order)),
        ("drawdown", "Drawdowns",
         drawdown_fig(top_results_by_exit, spy_result, strategy_order)),
        ("violins", "Per-lot returns",
         lot_return_box_fig(top_results_by_exit, strategy_order)),
        ("bars", "Strategy performance",
         metric_bar_fig(summary_df)),
        ("scatter", "Score vs return",
         score_scatter_fig(top_results_by_exit, strategy_order)),
        ("calendar_year", "Calendar-year contribution", None),  # HTML table
        ("exposure", "Daily exposure",
         exposure_fig(top_results_by_exit, strategy_order)),
        ("skips", "Skip diagnostics",
         skip_diag_fig(results_by_exit)),
    ]
    if stop_loss_html is not None:
        sections.append(("stop_loss", "Stop-loss / trailing-stop exits", None))

    plotly_head = ""
    if not offline:
        plotly_head = (
            '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>'
        )

    div_html: dict[str, str] = {}
    plotly_attached = False
    for sid, _label, fig in sections:
        if fig is None:
            continue
        include = False
        if offline and not plotly_attached:
            include = "inline"
            plotly_attached = True
        div_html[sid] = fig.to_html(full_html=False, include_plotlyjs=include)

    div_html["equity"] = equity_note + div_html["equity"]
    div_html["drawdown"] = drawdown_note + div_html["drawdown"]
    div_html["exposure"] = exposure_note + div_html["exposure"]
    div_html["scatter"] = scatter_note + div_html["scatter"]
    div_html["violins"] = violins_note + div_html["violins"]

    div_html["summary"] = _summary_table_html(summary_df)
    div_html["legend"] = strategy_legend_html(strategy_order)
    div_html["score_cohort"] = _score_cohort_table_html(results_by_exit)
    if fit_summary is not None:
        signal_fit_html = (
            _signal_weights_table_html(fit_summary["fit"])
            + _signal_oos_table_html(fit_summary["fit"], fit_summary["oos_df"])
        )
        tail_fit = fit_summary.get("tail_fit")
        if tail_fit is not None:
            signal_fit_html += _tail_score_section_html(tail_fit)
        div_html["signal_fit"] = signal_fit_html
    div_html["capacity"] = _capacity_table_html(results_by_exit)
    div_html["alpha_beta"] = _alpha_beta_table_html(summary_df)
    div_html["calendar_year"] = _calendar_year_table_html(results_by_exit)
    if stop_loss_html is not None:
        div_html["stop_loss"] = stop_loss_html

    cfg_html = "<br>".join(f"<b>{k}</b>: {v}" for k, v in config.items())
    toc = "".join(
        f'<li><a href="#{sid}">{label}</a></li>'
        for sid, label, _ in sections
    ) + '<li><a href="#caveats">Caveats</a></li>'

    body = []
    for sid, label, _fig in sections:
        body.append(f'<section id="{sid}"><h2>{label}</h2>{div_html[sid]}</section>')
    body.append(f'<section id="caveats"><h2>Caveats</h2>{CAVEATS_HTML}</section>')

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Insider Cluster-Buy Backtest Report</title>
{plotly_head}
<style>{CSS}{LEGEND_CSS}</style>
</head>
<body>
<header>
  <h1>Insider Cluster-Buy Backtest Report</h1>
  <div class="config">{cfg_html}</div>
</header>
<div class="layout">
  <nav>
    <h3>Sections</h3>
    <ul>{toc}</ul>
  </nav>
  <main>
    {''.join(body)}
  </main>
</div>
<footer>Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ·
  Insider Cluster-Buy Backtester. Informational tooling, not financial advice.</footer>
<script>{SORT_JS}</script>
<script>{LEGEND_JS}</script>
</body>
</html>
"""
