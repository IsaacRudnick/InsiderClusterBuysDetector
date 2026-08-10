"""
Render the cluster payload to a single self-contained dashboard.html.

The full payload is embedded as a <script id="cluster-data" type="application/json">
block inside the HTML, so the file is the one and only artifact - shareable as
a single attachment, machine-readable via the embedded JSON, human-readable
via the rendered table.

CLI mode (re-render without re-scanning): reads out/dashboard.html, extracts
the embedded payload, and overwrites the file. Useful for iterating on CSS or
legend copy without paying for a fresh SEC scan.

--------------------------------------------------------------------------
Ranking: trained model, not conviction_score
--------------------------------------------------------------------------
Clusters used to be sorted and labelled by insider_cluster_buys.py's
hand-tuned conviction_score ("conviction"/"mixed"/"routine"). That score has
been measured at ZERO risk-adjusted edge (0 of 5 time folds beat a
volatility-matched benchmark) and is gone from this module entirely. Sort
order, the per-row badge, and the expanded-row reasons panel now all come
from cluster["model_score"], the dict research.live_score.score_live_cluster
produces (attached by insider_cluster_buys.attach_model_scores before the
payload reaches render_html):

    {
      "raw_score": float, "percentile": float, "verdict": str,
      "n_features_available": int, "n_features_total": int,
      "missing_features": [str, ...], "n_training_scores": int,
      "factors": [{"feature", "value", "ic", "favorable_direction",
                   "description", "available"}, ...],
    }

`verdict` is one of "top_decile" / "no_edge" / "unavailable" (see
research/live_score.py's Verdict enum); a cluster whose model_score is None
was never scored at all (no production bundle on disk this run) and is
rendered as "not_scored". Only "top_decile" (percentile >= 90 against the
model's fixed training-score distribution) has a measured, volatility-
matched edge (+4.74pp, p=0.004, positive in 4/5 backtested folds) -- every
other state is "no measured edge", not "bad". This module never renders a
smoothed 0-100 confidence number for that reason: see model_sort_key,
factor_favorable, and the always-visible <section class="model-banner"> in
render_html for how that constraint plays out in the actual markup.
"""

import html
import json
import os
import re
import sys
from typing import Optional


OUTPUT_DIR = "out"
DASHBOARD_HTML = os.path.join(OUTPUT_DIR, "dashboard.html")

_EMBEDDED_PAYLOAD_RE = re.compile(
    r'<script id="cluster-data" type="application/json">(.*?)</script>',
    re.DOTALL,
)


def extract_embedded_payload(html_text: str) -> dict:
    m = _EMBEDDED_PAYLOAD_RE.search(html_text)
    if not m:
        raise ValueError("No embedded cluster-data payload found in HTML.")
    raw = m.group(1).replace("<\\/", "</")
    return json.loads(raw)


def _fmt_money(v) -> str:
    try:
        n = float(v or 0)
    except (TypeError, ValueError):
        return ""
    if n >= 1_000_000_000:
        return f"${n / 1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"${n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"${n / 1_000:.1f}K"
    return f"${n:,.0f}"


def _fmt_shares(v) -> str:
    try:
        return f"{int(float(v or 0)):,}"
    except (TypeError, ValueError):
        return ""


def _fmt_price(v) -> str:
    try:
        n = float(v)
    except (TypeError, ValueError):
        return ""
    if n == 0:
        return ""
    return f"${n:,.2f}"


def _fmt_pct(v) -> str:
    """Format a percentage stored as e.g. 25.5 = 25.5%."""
    if v is None:
        return ""
    try:
        n = float(v)
    except (TypeError, ValueError):
        return ""
    if n >= 1000:
        return f"{n/1000:.1f}Kx"
    if n >= 100:
        return f"{n:.0f}%"
    return f"{n:.1f}%"


def _esc(v) -> str:
    if v is None:
        return ""
    return html.escape(str(v))


def _count_color(n: int) -> str:
    """Green at 2 insiders, red at 10+ (linear hue interpolation)."""
    t = max(0.0, min(1.0, (n - 2) / 8.0))
    hue = 120 * (1 - t)
    return f"hsl({hue:.0f}, 65%, 45%)"


# ---------------------------------------------------------------------------
# Model verdict / percentile / factor-reasons rendering.
#
# Shared with insider_cluster_buys.py's write_excel so the xlsx and the HTML
# report the exact same verdict labels, favorable/unfavorable calls, and
# default sort order -- "keep it consistent with the HTML" per this
# project's own constraint. See research/live_score.py's module docstring
# for the evidence (top-decile edge, label-shuffle failure) this all rests on.
# ---------------------------------------------------------------------------
VERDICT_LABELS: dict[str, str] = {
    "top_decile": "Top decile — measured edge",
    "no_edge": "No measured edge",
    "unavailable": "Unavailable (too few features)",
    "not_scored": "Not scored",
}


def verdict_label(verdict: Optional[str]) -> str:
    return VERDICT_LABELS.get(verdict or "not_scored", "Not scored")


# Every REASON_PANEL_FACTORS feature (research/live_score.py) that has a
# defensible zero pivot to call "favorable"/"unfavorable" against. Left out
# on purpose: x_vol_21_ann (an annualized realized volatility, always >= 0 --
# there is no "0 is bad, positive is good" split for it, only the
# directional "lower is better" the factor's own description already
# states). Inventing a numeric threshold there would be exactly the kind of
# unsupported precision this project's own evidence argues against -- see
# research/live_score.py's module docstring, "the verdict is banded, not
# smoothed".
_FACTOR_ZERO_PIVOT_FEATURES = {
    "x_entry_vs_insider_vwap",    # <=0: bought at/below insiders' own VWAP
    "x_issuer_n_prior_clusters",  # >0: at least one prior cluster at this issuer
    "x_n_ten_pct",                 # ==0: no ten-percent owner in the cluster
    "x_ten_pct_value_share",       # ==0: no $ from a ten-percent owner
    "x_is_first_ever_cluster",     # ==0: not the issuer's first-ever cluster
}


def factor_favorable(feature: str, value, favorable_direction: str) -> Optional[bool]:
    """True/False if `value` sits on the favorable side of a natural zero
    pivot for `feature`; None when the feature has no such pivot (show the
    value with no favorable/unfavorable call) or `value` is missing/NaN."""
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v != v:  # NaN
        return None
    if feature not in _FACTOR_ZERO_PIVOT_FEATURES:
        return None
    return v <= 0 if favorable_direction == "lower" else v > 0


# Feature-specific value formatting for the reasons panel. Ratios/fractions
# read as percentages; counts read as whole numbers; the one binary feature
# reads as a plain yes/no rather than a bare "1.0"/"0.0".
_FACTOR_PCT_FEATURES = {"x_entry_vs_insider_vwap", "x_vol_21_ann", "x_ten_pct_value_share"}
_FACTOR_COUNT_FEATURES = {"x_issuer_n_prior_clusters", "x_n_ten_pct"}


def fmt_factor_value(feature: str, value) -> str:
    if value is None:
        return "n/a"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if v != v:  # NaN
        return "n/a"
    if feature == "x_is_first_ever_cluster":
        return "Yes (first-ever cluster)" if v >= 0.5 else "No (issuer has prior clusters)"
    if feature == "x_entry_vs_insider_vwap":
        return f"{v * 100:+.1f}%"
    if feature in _FACTOR_PCT_FEATURES:
        return f"{v * 100:.1f}%"
    if feature in _FACTOR_COUNT_FEATURES:
        return f"{v:.0f}"
    return f"{v:.4g}"


def _fmt_percentile(p) -> str:
    if p is None:
        return "—"
    try:
        pf = float(p)
    except (TypeError, ValueError):
        return "—"
    if pf != pf:
        return "—"
    return f"{pf:.0f}th percentile"


def _percentile_sentence(percentile, n_training, verdict: Optional[str]) -> str:
    """Longer, plain-language version of _fmt_percentile for the expanded
    row -- "top 6% of N historical cluster buys" phrasing per this
    project's own display spec."""
    if percentile is None:
        return "No percentile available -- this cluster was not model-scored."
    try:
        pf = float(percentile)
    except (TypeError, ValueError):
        return "No percentile available -- this cluster was not model-scored."
    if pf != pf:
        return "No percentile available -- this cluster was not model-scored."
    n_str = f"{int(n_training):,}" if n_training else None
    base = f"{pf:.0f}th percentile" + (f" of {n_str} historical cluster buys" if n_str else "")
    if verdict == "top_decile":
        top_pct = max(100.0 - pf, 0.0)
        return f"Top {top_pct:.0f}% of{(' ' + n_str) if n_str else ''} historical cluster buys — the one band with a measured edge."
    return f"{base} — no measured edge at this level (see the legend above)."


_VERDICT_SORT_RANK = {"top_decile": 0, "no_edge": 1, "unavailable": 2}


def model_sort_key(cluster: dict):
    """Default cluster ordering: the proven-edge band first (top_decile,
    ranked by percentile -- the one band where finer ordering is at least
    directionally supported by the evidence), then no-measured-edge
    clusters (percentile is still honest information, just not a proven
    ranking signal outside the top decile, so it is used only as a mild
    tiebreak here -- never displayed as a "confidence" number), then
    feature-starved/unavailable clusters, then unscored ones last. Total $
    value is the final tiebreak, matching the pre-model sort's own tiebreak."""
    ms = cluster.get("model_score")
    total_value = float(cluster.get("total_value") or 0)
    if not ms:
        return (3, 0.0, -total_value)
    verdict = ms.get("verdict")
    percentile = ms.get("percentile")
    try:
        percentile = float(percentile)
        if percentile != percentile:
            percentile = -1.0
    except (TypeError, ValueError):
        percentile = -1.0
    rank = _VERDICT_SORT_RANK.get(verdict, 3)
    return (rank, -percentile, -total_value)


def render_html(payload: dict) -> str:
    clusters = list(payload.get("clusters", []))
    clusters.sort(key=model_sort_key)

    distinct_issuers = {c.get("issuer_cik") for c in clusters}
    distinct_insiders = set()
    total_value = 0.0
    for c in clusters:
        for ins in c.get("insiders", []):
            distinct_insiders.add(ins.get("name"))
            try:
                total_value += float(ins.get("value") or 0)
            except (TypeError, ValueError):
                pass

    n_scored = sum(1 for c in clusters if c.get("model_score"))
    n_top_decile = sum(
        1 for c in clusters if (c.get("model_score") or {}).get("verdict") == "top_decile"
    )

    summary = {
        "clusters": len(clusters),
        "issuers": len(distinct_issuers),
        "insiders": len(distinct_insiders),
        "total_value": _fmt_money(total_value),
        "scanned_from": payload.get("scanned_from", ""),
        "scanned_to": payload.get("scanned_to", ""),
        "generated_at": payload.get("generated_at", ""),
        "qualifying_codes": ", ".join(payload.get("qualifying_codes", []) or []),
        "model_scored": f"{n_scored}/{len(clusters)}" if clusters else "0/0",
        "top_decile": str(n_top_decile),
    }

    # ---- "What this ranking means" banner -- always visible, not tucked
    # inside the collapsible legend, per this project's own display spec
    # ("somewhere prominent ... state plainly what the ranking means and
    # what it does not"). See research/live_score.py's module docstring for
    # the underlying evidence this text reports.
    mi = payload.get("model_info") or {}
    n_training = mi.get("n_training_scores") or 0
    if mi.get("model_available"):
        coverage_bits = []
        if mi.get("price_fetch_enabled"):
            if mi.get("prices_loaded"):
                coverage_bits.append(
                    f"prices for {mi.get('n_tickers_priced', 0)} ticker(s) fetched in "
                    f"{mi.get('price_fetch_elapsed_sec', 0):.1f}s"
                )
            else:
                coverage_bits.append("price fetch failed — price-context features unavailable this run")
        else:
            coverage_bits.append("price fetch disabled — price-context features unavailable this run")
        if mi.get("issuer_history_available"):
            coverage_bits.append(
                f"issuer history from {mi.get('n_issuer_history_rows', 0):,} historical rows"
            )
        else:
            coverage_bits.append("no issuer-history reference frame — issuer-history features unavailable this run")
        coverage_line = "; ".join(coverage_bits) + "."
        banner_body = (
            f"Clusters below are ranked by a trained model's percentile against "
            f"{n_training:,} historical cluster buys — not by the old hand-tuned "
            f"score. <b>Only the top decile (percentile &ge; 90) has a measured edge:</b> "
            f"+4.74pp over a volatility-matched benchmark, p=0.004, positive in 4 of 5 "
            f"backtested folds. Everything else is labelled &ldquo;no measured "
            f"edge,&rdquo; not &ldquo;bad&rdquo; — the model fails a label-shuffle "
            f"test on broad rank skill (IC &minus;0.0067, p=0.857), so percentile "
            f"differences below the top decile carry no shown information. Clusters "
            f"scored on fewer than half the model's inputs are marked "
            f"&ldquo;unavailable&rdquo; rather than given a number that looks precise "
            f"but is not. This run: {coverage_line}"
        )
    else:
        banner_body = (
            "No production model bundle was found this run — clusters below are "
            "unscored (no percentile, no verdict). See the run log for where "
            "insider_cluster_buys.py looked."
        )

    rows_html_parts: list[str] = []
    for idx, c in enumerate(clusters):
        flags = []
        if c.get("includes_ten_percent_owner"):
            flags.append('<span class="flag flag-10">10% Owner</span>')
        if c.get("includes_director"):
            flags.append('<span class="flag">Dir</span>')
        if c.get("includes_officer"):
            flags.append('<span class="flag">Off</span>')
        flag_html = " ".join(flags)

        insider_count = int(c.get("num_insiders") or 0)
        count_color = _count_color(insider_count)

        row_class = "row-10" if c.get("includes_ten_percent_owner") else ""

        edgar_url = _esc(c.get("edgar_url", ""))
        ticker = _esc(c.get("ticker") or "")

        ms = c.get("model_score")
        if ms is None:
            verdict = "not_scored"
            percentile = None
            n_avail = n_total = None
            factors: list[dict] = []
        else:
            verdict = ms.get("verdict") or "not_scored"
            percentile = ms.get("percentile")
            n_avail = ms.get("n_features_available")
            n_total = ms.get("n_features_total")
            factors = ms.get("factors") or []

        is_recent_ipo = bool(c.get("is_recent_ipo"))
        ipo_chip = '<span class="ipo-flag" title="Issuer first traded in the last 6 months">Recent IPO</span>' if is_recent_ipo else ""

        degraded = ms is not None and n_avail is not None and n_total is not None and n_avail < n_total
        degraded_chip = (
            f'<span class="degraded-flag" title="Scored on {n_avail}/{n_total} model inputs '
            f'-- reduced feature set">reduced features</span>'
        ) if degraded else ""

        # Reasons panel: the model's factor-level "why", replacing the old
        # conviction_score contribution breakdown entirely (see module
        # docstring -- conviction_score is gone from this file's output).
        if ms is None:
            reasons_html = (
                "<div class='breakdown'><div class='breakdown-empty'>"
                "Not scored — no production model bundle was available this run."
                "</div></div>"
            )
        else:
            factor_rows = []
            for f in factors:
                feature = f.get("feature", "")
                available = bool(f.get("available"))
                value = f.get("value")
                fav = factor_favorable(feature, value, f.get("favorable_direction", "")) if available else None
                fav_label = "Favorable" if fav is True else "Unfavorable" if fav is False else "—"
                fav_class = "delta-pos" if fav is True else "delta-neg" if fav is False else ""
                value_str = fmt_factor_value(feature, value) if available else "not computed this run"
                factor_rows.append(
                    f"<tr><td class='delta {fav_class}'>{_esc(fav_label)}</td>"
                    f"<td class='delta-text'><b>{_esc(value_str)}</b> — {_esc(f.get('description', ''))}</td></tr>"
                )
            coverage_note = (
                "" if not degraded else
                " — reduced feature set this run, treat with extra caution"
            )
            reasons_html = (
                "<div class='breakdown'>"
                "<div class='breakdown-head'>Why this percentile &mdash; the handful of "
                "factors with an independently-checked link to outcomes</div>"
                "<table class='breakdown-table'>" + "".join(factor_rows) + "</table>"
                f"<div class='breakdown-coverage'>{_percentile_sentence(percentile, n_training, verdict)}<br/>"
                f"Scored on {n_avail}/{n_total} model inputs{coverage_note}.</div>"
                "</div>"
            )

        # Per-insider transaction sub-table (most-relevant columns first; Code last).
        tx_rows = []
        for tx in c.get("transactions", []):
            tx_rows.append(
                "<tr>"
                f"<td>{_esc(tx.get('owner_name'))}</td>"
                f"<td>{_esc(tx.get('owner_roles'))}</td>"
                f"<td class='num'>{_fmt_money(tx.get('value'))}</td>"
                f"<td class='num'>{_fmt_pct(tx.get('pct_of_prior_stake'))}</td>"
                f"<td>{_esc(tx.get('transaction_date'))}</td>"
                f"<td class='num'>{_fmt_shares(tx.get('shares'))}</td>"
                f"<td class='num'>{_fmt_price(tx.get('price_per_share'))}</td>"
                f"<td><a href='{_esc(tx.get('filing_url'))}' target='_blank' rel='noopener'>filing</a></td>"
                f"<td class='code-col'>{_esc(tx.get('transaction_code'))}</td>"
                "</tr>"
            )
        sub_table = (
            "<table class='subtable'>"
            "<thead><tr>"
            "<th>Insider</th><th>Role(s)</th>"
            "<th class='num'>$ Value</th><th class='num'>% Stake</th>"
            "<th>Tx Date</th>"
            "<th class='num'>Shares</th><th class='num'>Price</th>"
            "<th>Filing</th><th class='code-col'>Code</th>"
            "</tr></thead><tbody>"
            + "".join(tx_rows)
            + "</tbody></table>"
        )

        total_value_raw = float(c.get("total_value") or 0)
        max_pct_raw = float(c.get("max_pct_of_prior_stake") or 0)
        num_tx_raw = int(c.get("num_transactions") or 0)
        try:
            percentile_sort = float(percentile) if percentile is not None else -1.0
            if percentile_sort != percentile_sort:
                percentile_sort = -1.0
        except (TypeError, ValueError):
            percentile_sort = -1.0

        model_cell = (
            f'<td data-sort-value="{percentile_sort}">'
            f'<span class="verdict verdict-{_esc(verdict)}">{_esc(verdict_label(verdict))}</span>'
            f'{degraded_chip}'
            f'<div class="pctl-text">{_esc(_fmt_percentile(percentile))}</div>'
            f'</td>'
        )

        rows_html_parts.append(
            f'<tr class="cluster-row {row_class}" data-idx="{idx}" '
            f'data-verdict="{_esc(verdict)}" '
            f'data-ipo="{"1" if is_recent_ipo else ""}" '
            f'data-search="{_esc((c.get("issuer_name") or "") + " " + ticker)}">'
            f'<td class="expander">+</td>'
            f'<td class="ticker-col">{ticker}</td>'
            f'<td><a href="{edgar_url}" target="_blank" rel="noopener">{_esc(c.get("issuer_name"))}</a>'
            f' {flag_html}</td>'
            f'{model_cell}'
            f'<td class="num" data-sort-value="{total_value_raw}">{_fmt_money(c.get("total_value"))}</td>'
            f'<td class="num" data-sort-value="{max_pct_raw}">{_fmt_pct(c.get("max_pct_of_prior_stake"))}</td>'
            f'<td class="num" data-sort-value="{insider_count}"><span class="count-badge" style="background:{count_color}">{insider_count}</span></td>'
            f'<td class="num" data-sort-value="{num_tx_raw}">{num_tx_raw}</td>'
            f'<td>{_esc(c.get("cluster_start"))} &rarr; {_esc(c.get("cluster_end"))}</td>'
            f'</tr>'
            f'<tr class="detail-row" data-idx="{idx}"><td></td><td colspan="8">{reasons_html}{sub_table}</td></tr>'
        )

    rows_html = "\n".join(rows_html_parts) if rows_html_parts else \
        '<tr><td colspan="9" class="empty">No flagged clusters in this run.</td></tr>'

    # Embed full payload for machine-readable consumption + future re-renders.
    embedded_json = json.dumps(payload, ensure_ascii=False, default=str).replace("</", "<\\/")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Insider Cluster Buys - {_esc(summary['scanned_to'])}</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         margin: 0; background: #f5f7fa; color: #1f2937; }}
  header {{ background: #111827; color: #f9fafb; padding: 18px 24px; }}
  header h1 {{ margin: 0; font-size: 20px; }}
  header .sub {{ font-size: 12px; opacity: 0.75; margin-top: 4px; }}
  .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
              gap: 12px; padding: 18px 24px; background: #fff; border-bottom: 1px solid #e5e7eb; }}
  .stat {{ background: #f9fafb; padding: 12px; border-radius: 6px; border: 1px solid #e5e7eb; }}
  .stat .label {{ font-size: 11px; color: #6b7280; text-transform: uppercase; letter-spacing: 0.5px; }}
  .stat .value {{ font-size: 22px; font-weight: 600; margin-top: 4px; }}
  section.model-banner {{ margin: 0; padding: 14px 24px; background: #eff6ff;
                          border-bottom: 1px solid #bfdbfe; font-size: 12.5px;
                          color: #1e3a5f; line-height: 1.55; }}
  .model-banner-title {{ font-weight: 700; color: #1e3a8a; font-size: 13px;
                         text-transform: uppercase; letter-spacing: 0.4px; margin-bottom: 6px; }}
  .model-banner-body b {{ color: #1e3a8a; }}
  details.legend {{ margin: 0; padding: 10px 24px; background: #fff;
                    border-bottom: 1px solid #e5e7eb; font-size: 12px; color: #374151; }}
  details.legend[open] {{ padding-bottom: 16px; }}
  details.legend summary {{ cursor: pointer; font-weight: 600; color: #111827;
                            font-size: 13px; user-select: none;
                            padding: 4px 0; list-style: none; }}
  details.legend summary::-webkit-details-marker {{ display: none; }}
  details.legend summary::before {{ content: "\\25B8"; display: inline-block;
                                    width: 16px; color: #6b7280;
                                    transition: transform 0.15s; }}
  details.legend[open] summary::before {{ transform: rotate(90deg); }}
  .legend-grid {{ display: grid; grid-template-columns: max-content 1fr;
                  column-gap: 14px; row-gap: 6px; margin-top: 12px;
                  align-items: baseline; }}
  .legend-grid h4 {{ grid-column: 1 / -1; margin: 10px 0 2px;
                     font-size: 11px; text-transform: uppercase;
                     letter-spacing: 0.6px; color: #6b7280; font-weight: 600; }}
  .legend-grid h4:first-child {{ margin-top: 0; }}
  .legend-grid .lg-key {{ justify-self: start; min-width: 80px;
                          display: inline-flex; align-items: center; }}
  .legend-grid .lg-key .flag,
  .legend-grid .lg-key .ipo-flag,
  .legend-grid .lg-key .verdict {{ margin: 0; }}
  .legend-grid .lg-desc {{ color: #374151; line-height: 1.4; }}
  .legend-grid code {{ display: inline-block; padding: 1px 8px;
                       background: #f3f4f6; border: 1px solid #e5e7eb;
                       border-radius: 3px; font-size: 11px;
                       font-family: ui-monospace, monospace; color: #111827;
                       min-width: 22px; text-align: center; }}
  .legend-grid .col-name {{ font-weight: 600; color: #111827; font-size: 12px; }}
  .toolbar {{ padding: 12px 24px; background: #fff; border-bottom: 1px solid #e5e7eb; }}
  .toolbar input {{ width: 320px; padding: 8px 10px; font-size: 13px;
                    border: 1px solid #d1d5db; border-radius: 4px; }}
  table.main {{ width: 100%; border-collapse: collapse; background: #fff; }}
  table.main th {{ background: #BDD7EE; text-align: left; padding: 10px 12px;
                   font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px;
                   cursor: pointer; user-select: none; position: sticky; top: 0; }}
  table.main td {{ padding: 10px 12px; border-bottom: 1px solid #e5e7eb;
                   vertical-align: top; font-size: 13px; }}
  table.main td.num, table.main th.num {{ text-align: right; }}
  table.main tr.cluster-row:hover {{ background: #f3f4f6; cursor: pointer; }}
  table.main tr.row-10 {{ background: #FFF2CC; }}
  table.main tr.row-10:hover {{ background: #fde7a0; }}
  table.main tr.detail-row {{ display: none; background: #fafafa; }}
  table.main tr.detail-row.show {{ display: table-row; }}
  table.main td.expander {{ width: 24px; text-align: center;
                            font-family: monospace; color: #6b7280; }}
  td.ticker-col {{ font-weight: 600; }}
  .flag {{ display: inline-block; background: #d1fae5; color: #065f46;
           padding: 1px 6px; border-radius: 3px; font-size: 10px;
           font-weight: 600; margin-left: 4px; }}
  .flag-10 {{ background: #fef3c7; color: #92400e; }}
  .count-badge {{ display: inline-block; padding: 2px 8px; border-radius: 10px;
                  font-weight: 600; font-size: 12px; color: #fff; }}
  .verdict {{ display: inline-block; padding: 2px 10px; border-radius: 10px;
             font-size: 11.5px; font-weight: 700; white-space: nowrap; }}
  .verdict-top_decile {{ background: #d1fae5; color: #065f46; }}
  .verdict-no_edge {{ background: #e5e7eb; color: #4b5563; }}
  .verdict-unavailable {{ background: #fef3c7; color: #92400e; }}
  .verdict-not_scored {{ background: #f3f4f6; color: #9ca3af; }}
  .pctl-text {{ font-size: 11px; color: #6b7280; margin-top: 3px; }}
  .degraded-flag {{ display: inline-block; background: #fee2e2; color: #991b1b;
                    padding: 1px 6px; border-radius: 10px; font-size: 9.5px;
                    font-weight: 600; margin-left: 4px; }}
  .ipo-flag {{ display: inline-block; background: #fee2e2; color: #991b1b;
               padding: 1px 6px; border-radius: 10px; font-size: 10px;
               font-weight: 600; margin-left: 6px; }}
  .breakdown {{ background: #f9fafb; border-left: 3px solid #9ca3af;
                padding: 8px 12px; margin: 4px 0 10px; }}
  .breakdown-head {{ font-weight: 600; color: #111827; font-size: 12px;
                     margin-bottom: 6px; }}
  .breakdown-empty {{ color: #6b7280; font-style: italic; font-size: 12px; }}
  .breakdown-coverage {{ margin-top: 8px; font-size: 11.5px; color: #6b7280; }}
  table.breakdown-table {{ border-collapse: collapse; font-size: 12px; }}
  table.breakdown-table td {{ padding: 2px 10px 2px 0; vertical-align: top;
                              border: none; }}
  td.delta {{ font-weight: 700; font-variant-numeric: tabular-nums;
              text-align: right; width: 78px; }}
  td.delta-pos {{ color: #065f46; }}
  td.delta-neg {{ color: #991b1b; }}
  td.delta-text {{ color: #374151; }}
  .signal-toolbar {{ display: inline-block; margin-left: 16px; }}
  .signal-toolbar button {{ background: #fff; border: 1px solid #d1d5db;
                           padding: 7px 12px; font-size: 12px; cursor: pointer;
                           border-radius: 4px; margin-right: 4px; color: #374151; }}
  .signal-toolbar button.active {{ background: #2563eb; color: #fff;
                                  border-color: #2563eb; }}
  .min-filters {{ font-size: 12px; color: #374151; }}
  .min-filters label {{ margin-right: 16px; white-space: nowrap; }}
  .min-filters input {{ width: 88px; padding: 6px 8px; font-size: 12px;
                       border: 1px solid #d1d5db; border-radius: 4px;
                       margin-left: 4px; }}
  table.subtable {{ width: 100%; border-collapse: collapse; margin: 8px 0; }}
  table.subtable th, table.subtable td {{ padding: 6px 10px; font-size: 12px;
                                          border-bottom: 1px solid #e5e7eb; }}
  table.subtable th {{ background: #f3f4f6; text-align: left; }}
  table.subtable td.num, table.subtable th.num {{ text-align: right; }}
  table.subtable .code-col {{ color: #9ca3af; font-size: 11px;
                              font-family: ui-monospace, monospace; }}
  .empty {{ text-align: center; color: #9ca3af; padding: 40px; font-style: italic; }}
  a {{ color: #2563eb; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  footer {{ padding: 12px 24px; font-size: 11px; color: #6b7280; text-align: center; }}
</style>
</head>
<body>
<header>
  <h1>Insider Cluster Buys</h1>
  <div class="sub">
    Scanned {_esc(summary['scanned_from'])} &rarr; {_esc(summary['scanned_to'])}
    &middot; Qualifying codes: {_esc(summary['qualifying_codes'])}
    &middot; Generated {_esc(summary['generated_at'])}
  </div>
</header>

<section class="summary">
  <div class="stat"><div class="label">Flagged Clusters</div><div class="value">{summary['clusters']}</div></div>
  <div class="stat"><div class="label">Distinct Issuers</div><div class="value">{summary['issuers']}</div></div>
  <div class="stat"><div class="label">Insiders Involved</div><div class="value">{summary['insiders']}</div></div>
  <div class="stat"><div class="label">Total Acquired $</div><div class="value">{summary['total_value']}</div></div>
  <div class="stat"><div class="label">Model-Scored</div><div class="value">{summary['model_scored']}</div></div>
  <div class="stat"><div class="label">Top Decile (Edge)</div><div class="value">{summary['top_decile']}</div></div>
</section>

<section class="model-banner">
  <div class="model-banner-title">What this ranking means</div>
  <div class="model-banner-body">{banner_body}</div>
</section>

<details class="legend">
  <summary>Legend</summary>
  <div class="legend-grid">
    <h4>Model verdict &amp; percentile &mdash; click any row for the per-factor reasons panel</h4>
    <span class="lg-key"><span class="verdict verdict-top_decile">Top decile</span></span>
    <span class="lg-desc">Percentile &ge; 90 against the model's own training-score distribution &mdash; the only band with a measured, volatility-matched edge (+4.74pp, p=0.004, 4/5 folds).</span>
    <span class="lg-key"><span class="verdict verdict-no_edge">No measured edge</span></span>
    <span class="lg-desc">Everything below the top decile. The model fails a broad rank-skill test here (IC &minus;0.0067, p=0.857) &mdash; this means &ldquo;unproven,&rdquo; not &ldquo;bad.&rdquo;</span>
    <span class="lg-key"><span class="verdict verdict-unavailable">Unavailable</span></span>
    <span class="lg-desc">Fewer than half the model's 50 inputs could be computed for this cluster (see &ldquo;reduced features&rdquo; below) &mdash; the score exists but is not trustworthy enough to band.</span>
    <span class="lg-key"><span class="verdict verdict-not_scored">Not scored</span></span>
    <span class="lg-desc">No production model bundle was found this run.</span>
    <span class="lg-key"><span class="degraded-flag">reduced features</span></span>
    <span class="lg-desc">This cluster was scored on fewer than the full 50 model inputs (no live price data and/or no issuer-history reference frame this run) &mdash; treat its percentile with extra caution.</span>

    <h4>Flags</h4>
    <span class="lg-key"><span class="ipo-flag">Recent IPO</span></span>
    <span class="lg-desc">Issuer first traded &lt; 6 months ago &mdash; lockup expiries and S-1 selling shareholders distort cluster signal</span>
    <span class="lg-key"><span class="flag">Dir</span></span>
    <span class="lg-desc">Director</span>
    <span class="lg-key"><span class="flag">Off</span></span>
    <span class="lg-desc">Officer</span>
    <span class="lg-key"><span class="flag flag-10">10% Owner</span></span>
    <span class="lg-desc">Beneficial owner of &gt; 10% of voting shares</span>

    <h4>Transaction codes (sub-table)</h4>
    <span class="lg-key"><code>P</code></span>
    <span class="lg-desc">Open-market purchase &mdash; the only code that qualifies by default</span>
    <span class="lg-key"><code>S</code></span>
    <span class="lg-desc">Sale</span>
    <span class="lg-key"><code>A</code></span>
    <span class="lg-desc">Grant / award (compensation, not a buy)</span>
    <span class="lg-key"><code>M</code></span>
    <span class="lg-desc">Option exercise</span>
    <span class="lg-key"><code>G</code></span>
    <span class="lg-desc">Gift</span>
    <span class="lg-key"><code>F</code></span>
    <span class="lg-desc">Tax withholding</span>

    <h4>Columns</h4>
    <span class="lg-key"><span class="col-name">Model</span></span>
    <span class="lg-desc">Verdict badge plus percentile against historical cluster buys (see above)</span>
    <span class="lg-key"><span class="col-name">Total $ Value</span></span>
    <span class="lg-desc">Aggregate dollars acquired across the cluster</span>
    <span class="lg-key"><span class="col-name">Max % Stake</span></span>
    <span class="lg-desc">Largest single-tx % increase to an insider's prior holdings</span>
    <span class="lg-key"><span class="col-name"># Insiders</span></span>
    <span class="lg-desc">Distinct reporting owners</span>
    <span class="lg-key"><span class="col-name"># Tx</span></span>
    <span class="lg-desc">Transaction count</span>
    <span class="lg-key"><span class="col-name">Window</span></span>
    <span class="lg-desc">Cluster start &rarr; end</span>
  </div>
</details>

<div class="toolbar">
  <input id="filter" type="text" placeholder="Filter by issuer or ticker..." />
  <span class="signal-toolbar">
    <button data-show="all" class="active">All</button>
    <button data-show="top_decile">Top decile only</button>
    <button data-show="not-unavailable">Hide unavailable/unscored</button>
    <button data-show="not-ipo">Hide recent IPOs</button>
  </span>
</div>
<div class="toolbar">
  <span class="min-filters">
    <label>Min # insiders <input type="number" id="min-insiders" min="0" step="1" /></label>
    <label>Min $ value <input type="text" id="min-value" placeholder="e.g. 100k" /></label>
    <label>Min % stake <input type="number" id="min-pct" min="0" step="0.1" placeholder="%" /></label>
  </span>
</div>

<table class="main" id="clusters">
  <thead>
    <tr>
      <th></th>
      <th data-sort="text" data-col="1">Ticker</th>
      <th data-sort="text" data-col="2">Issuer</th>
      <th data-sort="num"  data-col="3">Model</th>
      <th data-sort="num"  data-col="4" class="num">Total $ Value</th>
      <th data-sort="num"  data-col="5" class="num">Max % Stake</th>
      <th data-sort="num"  data-col="6" class="num"># Insiders</th>
      <th data-sort="num"  data-col="7" class="num"># Tx</th>
      <th data-sort="text" data-col="8">Window</th>
    </tr>
  </thead>
  <tbody>
    {rows_html}
  </tbody>
</table>

<footer>
  Informational only - not financial advice. Source: SEC EDGAR Form 4.
</footer>

<script id="cluster-data" type="application/json">{embedded_json}</script>
<script>
  // Expand/collapse a cluster row.
  document.querySelectorAll('tr.cluster-row').forEach(function(row) {{
    row.addEventListener('click', function(ev) {{
      if (ev.target.tagName === 'A') return;
      var idx = row.dataset.idx;
      var detail = document.querySelector("tr.detail-row[data-idx='" + idx + "']");
      if (!detail) return;
      detail.classList.toggle('show');
      var exp = row.querySelector('td.expander');
      if (exp) exp.textContent = detail.classList.contains('show') ? '-' : '+';
    }});
  }});

  // Composable filters. Each filter sets a data flag on the row; one
  // applyVisibility() call collapses all flags into the display decision.
  function applyVisibility(row) {{
    var hidden = row.dataset.textHidden === '1'
              || row.dataset.verdictHidden === '1'
              || row.dataset.numHidden === '1';
    row.style.display = hidden ? 'none' : '';
    var detail = document.querySelector("tr.detail-row[data-idx='" + row.dataset.idx + "']");
    if (!detail) return;
    if (hidden) {{ detail.classList.remove('show'); detail.style.display = 'none'; }}
    else {{ detail.style.display = ''; }}
  }}

  // Accept "100", "100k", "1.5M", "$2b" — returns 0 on blank/invalid.
  function parseMoney(s) {{
    if (!s) return 0;
    var m = String(s).trim().toLowerCase().replace(/[$,_\\s]/g, '').match(/^([0-9.]+)([kmb]?)$/);
    if (!m) return 0;
    var n = parseFloat(m[1]);
    if (m[2] === 'k') n *= 1e3;
    else if (m[2] === 'm') n *= 1e6;
    else if (m[2] === 'b') n *= 1e9;
    return n;
  }}

  function applyNumericFilter() {{
    var minIns = parseFloat(document.getElementById('min-insiders').value) || 0;
    var minVal = parseMoney(document.getElementById('min-value').value);
    var minPct = parseFloat(document.getElementById('min-pct').value) || 0;
    document.querySelectorAll('tr.cluster-row').forEach(function(row) {{
      var cells = row.children;
      // children: 0=expander 1=ticker 2=issuer 3=model 4=$value 5=%stake 6=#insiders 7=#tx 8=window
      var val = parseFloat(cells[4].dataset.sortValue) || 0;
      var pct = parseFloat(cells[5].dataset.sortValue) || 0;
      var ins = parseFloat(cells[6].dataset.sortValue) || 0;
      var ok = ins >= minIns && val >= minVal && pct >= minPct;
      row.dataset.numHidden = ok ? '' : '1';
      applyVisibility(row);
    }});
  }}
  ['min-insiders', 'min-value', 'min-pct'].forEach(function(id) {{
    document.getElementById(id).addEventListener('input', applyNumericFilter);
  }});

  var filterInput = document.getElementById('filter');
  filterInput.addEventListener('input', function() {{
    var q = filterInput.value.toLowerCase().trim();
    document.querySelectorAll('tr.cluster-row').forEach(function(row) {{
      var hay = (row.dataset.search || '').toLowerCase();
      var match = !q || hay.indexOf(q) !== -1;
      row.dataset.textHidden = match ? '' : '1';
      applyVisibility(row);
    }});
  }});

  document.querySelectorAll('.signal-toolbar button').forEach(function(btn) {{
    btn.addEventListener('click', function() {{
      document.querySelectorAll('.signal-toolbar button').forEach(function(b) {{
        b.classList.remove('active');
      }});
      btn.classList.add('active');
      var mode = btn.dataset.show;
      document.querySelectorAll('tr.cluster-row').forEach(function(row) {{
        var verdict = row.dataset.verdict;
        var ipo = row.dataset.ipo === '1';
        var keep = (mode === 'all') ||
                   (mode === 'top_decile' && verdict === 'top_decile') ||
                   (mode === 'not-unavailable' && verdict !== 'unavailable' && verdict !== 'not_scored') ||
                   (mode === 'not-ipo' && !ipo);
        row.dataset.verdictHidden = keep ? '' : '1';
        applyVisibility(row);
      }});
    }});
  }});

  // Sortable headers.
  document.querySelectorAll('th[data-sort]').forEach(function(th) {{
    var asc = false;
    th.addEventListener('click', function() {{
      var tbody = document.querySelector('table.main tbody');
      var col = parseInt(th.dataset.col, 10);
      var sortType = th.dataset.sort;
      var pairs = [];
      var rows = Array.from(tbody.querySelectorAll('tr.cluster-row'));
      rows.forEach(function(row) {{
        var detail = document.querySelector("tr.detail-row[data-idx='" + row.dataset.idx + "']");
        pairs.push([row, detail]);
      }});
      pairs.sort(function(a, b) {{
        var aCell = a[0].children[col];
        var bCell = b[0].children[col];
        if (sortType === 'num') {{
          var aRaw = aCell.dataset.sortValue;
          var bRaw = bCell.dataset.sortValue;
          var an = aRaw !== undefined
            ? parseFloat(aRaw)
            : (parseFloat((aCell.textContent || '').replace(/[^0-9.\\-]/g, '')) || 0);
          var bn = bRaw !== undefined
            ? parseFloat(bRaw)
            : (parseFloat((bCell.textContent || '').replace(/[^0-9.\\-]/g, '')) || 0);
          return asc ? an - bn : bn - an;
        }}
        var av = aCell.textContent.trim();
        var bv = bCell.textContent.trim();
        return asc ? av.localeCompare(bv) : bv.localeCompare(av);
      }});
      asc = !asc;
      pairs.forEach(function(p) {{
        tbody.appendChild(p[0]);
        if (p[1]) tbody.appendChild(p[1]);
      }});
    }});
  }});
</script>
</body>
</html>
"""


def main() -> int:
    src = sys.argv[1] if len(sys.argv) > 1 else DASHBOARD_HTML
    if not os.path.exists(src):
        print(f"No dashboard found at {src}. Run insider_cluster_buys.py first.",
              file=sys.stderr)
        return 1
    with open(src, encoding="utf-8") as fh:
        existing = fh.read()
    try:
        payload = extract_embedded_payload(existing)
    except ValueError as exc:
        print(f"Cannot re-render: {exc}", file=sys.stderr)
        return 1

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR, "dashboard.html")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(render_html(payload))
    print(f"Re-rendered {out} from embedded payload ({len(payload.get('clusters', []))} cluster(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
