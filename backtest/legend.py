"""Human-readable legend for every strategy, exit method, and gate term.

The report's summary table identifies a strategy by its registry name alone
(`non_10b5_1_only`, `thr_gt_m03`, `learned_tail_concentrated`), which is
unreadable to anyone who has not read backtest/strategies.py. This module
renders a collapsible legend section explaining each one twice: once in
technical terms (the literal entry rule, thresholds, and sizing) and once in
plain English.

Two rules keep this from drifting out of sync with the code:

1. Anything already on the Strategy dataclass — capacity cap, stop-loss,
   fixed hold, equal-weight flag, rank function — is read off the registry at
   render time and shown as chips, never retyped here.
2. Only what lives *inside* a target_fn (the entry condition and the position
   size) is hand-written, in ENTRY_RULES below. A strategy with no hand-
   written entry falls back to its registry `description`, so adding a
   strategy without touching this file degrades to the old behavior instead
   of raising.
"""

from __future__ import annotations

import html
import re

from .state import WINDOW_DAYS
from .strategies import EXIT_METHODS, STRATEGIES

# ---------------------------------------------------------------------------
# Shared terms. Every gate in ENTRY_RULES leans on these, so they are
# explained once here rather than repeated per strategy.
# ---------------------------------------------------------------------------
GLOSSARY: list[tuple[str, str]] = [
    (
        "Cluster / rolling window",
        f"A ticker's state on any given day is built from all insider buys filed in the "
        f"previous {WINDOW_DAYS} calendar days. It is a rolling window, not a one-off event: "
        f"as buys age past {WINDOW_DAYS} days the cluster shrinks and eventually empties, "
        f"which is what makes most positions get trimmed back out.",
    ),
    (
        "conviction_score",
        "The hand-tuned scorer's verdict on one cluster (<code>_score_cluster</code> in "
        "<code>insider_cluster_buys.py</code>). It sums signed weights for fired conditions — "
        "credit for things like a high median dollar value per insider (+3), a 10% owner "
        "participating (+3), directors only (+2), a big add to an existing stake (+2); "
        "penalties for tells that the buying is routine or cosmetic, such as a routine "
        "footnote (&minus;5), identical prices across every transaction (&minus;3), a recent "
        "IPO (&minus;3), fractional share counts (&minus;2), or a 10b5-1 plan (&minus;1). "
        "Typical scores run from about &minus;5 to +10. "
        "<b>Plainly:</b> higher means the buying looks more like genuine conviction and less "
        "like paperwork.",
    ),
    (
        "Rule 10b5-1 plan",
        "A trading plan an insider sets up in advance, then executes automatically on a "
        "schedule. It is the insider's legal safe harbor: because the orders were locked in "
        "before they knew anything, trading on the plan is not insider trading. "
        "<b>Plainly:</b> a 10b5-1 buy was decided months ago by a calendar, not today by a "
        "person who just saw something good. That is why several strategies discount or "
        "exclude it.",
    ),
    (
        "Insider roles",
        "<b>Director</b> — board member, oversees management. <b>Officer</b> — an executive "
        "who runs the business day to day (CEO, CFO, and similar). <b>10% owner</b> — anyone "
        "holding more than 10% of the shares. All three must file their trades, and each "
        "carries a different informational weight, so several gates key off which roles "
        "appear in the window.",
    ),
    (
        "Recent IPO",
        "A ticker whose price history starts only shortly before the decision date. Insider "
        "buying around lockup expiry and early trading is noisy, so some gates skip these.",
    ),
    (
        "Position size (% of capital)",
        "Every size below is a percentage of <em>starting</em> capital, not of current NAV. "
        "Sizes are targets: the engine compares the target to what is already held and buys "
        "or sells the difference at the next trading day's open, paying 10 bps of slippage "
        "per side. When the cluster decays the target falls to zero and the position is sold.",
    ),
    (
        "Capacity cap",
        "The maximum number of tickers the strategy holds at once. When more candidates "
        "qualify on a day than there are open slots, the cap binds and something has to "
        "choose — by default arrival order, or by a ranking function where one is set. "
        "<b>Plainly:</b> a tight cap means the strategy is often turning down qualifying "
        "signals, so the cap itself is doing a lot of the picking.",
    ),
    (
        "Equal-weight",
        "Instead of a fixed percentage per name, the strategy spreads 100% of capital evenly "
        "across however many tickers qualify that day. Twenty qualifiers means 5% each; two "
        "qualifiers means 50% each.",
    ),
    (
        "learned_score",
        "A score from weights fit by the backtest's own signal-fit phase (regression of "
        "cluster features on forward returns) rather than hand-tuned. "
        "<b>Plainly:</b> the same idea as conviction_score, except the data chose the "
        "weights instead of a human.",
    ),
    (
        "tail_score",
        "A fitted estimate of the chance of a very large winner, not of the average return. "
        "<b>Plainly:</b> it ranks clusters by lottery-ticket upside rather than by typical "
        "outcome.",
    ),
    (
        "model_score",
        "An out-of-fold score from the separate ranking model (<code>research/model.py</code>, "
        "loaded via <code>backtest/model_scores.py</code>). It is used only to decide who wins "
        "a capacity slot, never whether a ticker qualifies at all. Out-of-fold means each "
        "cluster is scored by a model fit without it, so the score is not self-referential.",
    ),
    (
        "Fixed hold (hold63)",
        "Normally a position is trimmed as soon as its cluster decays, which usually happens "
        "within one to two weeks. A fixed hold pins the lot open for a set number of trading "
        "days regardless (63 &approx; 3 months), so the multi-month drift the signal is "
        "actually scored on has room to show up. Stops still fire; only the signal-decay trim "
        "is blocked.",
    ),
    (
        "Stop-loss vs. trailing stop",
        "A stop-loss exits a lot when it falls a set percentage below its <em>entry</em> "
        "price. A trailing stop exits when it falls a set percentage below its <em>highest "
        "price since entry</em>, so it locks in gains as they accrue.",
    ),
]


# ---------------------------------------------------------------------------
# Per-strategy entry rule + plain-English gloss.
#
# gist      one line shown on the collapsed row
# technical the literal entry condition and position size
# plain     the same thing with no jargon
#
# Structural facts (cap, stop, hold, equal-weight, ranking) are NOT written
# here — they are read off the registry in _chips().
# ---------------------------------------------------------------------------
ENTRY_RULES: dict[str, tuple[str, str, str]] = {
    "all_clusters_equal_weight": (
        "Buy every cluster of 2+ insiders, no quality filter",
        "Enter at 2% of capital whenever <code>num_insiders &ge; 2</code> in the rolling "
        "window. No score gate at all.",
        "The control experiment. If two or more insiders bought recently, take a small "
        "position — no judgment about whether the buying looked good. Everything else in "
        "this report is trying to beat this.",
    ),
    "conviction_only": (
        "Only high-scoring clusters, skipping recent IPOs",
        "Enter at 5% when <code>conviction_score &ge; +3</code> and the ticker is not a "
        "recent IPO.",
        "Trust the scorer. Only buy when the buying pattern looks clearly high-quality, and "
        "stay away from freshly listed companies where the signal is noise.",
    ),
    "ten_percent_owner_gated": (
        "Only clusters a 10% owner took part in — big, concentrated bets",
        "Enter at 10% when <code>includes_ten_percent_owner</code> is true and "
        "<code>conviction_score &ge; 0</code> (i.e. no net red flags).",
        "Wait for someone who already owns more than a tenth of the company to buy more, "
        "then bet big. These signals are rare, so each one gets a large slice of the "
        "portfolio.",
    ),
    "multi_insider_positive": (
        "Three or more insiders, and the score is not negative",
        "Enter at 4% when <code>num_insiders &ge; 3</code> and "
        "<code>conviction_score &gt; 0</code>.",
        "Require a real crowd, not just a pair — three separate people buying — plus a score "
        "that at least leans positive.",
    ),
    "score_weighted": (
        "Size scales with the score instead of a yes/no gate",
        "Enter at <code>min(8%, 1.5% &times; conviction_score)</code>; nothing when the score "
        "is zero or negative.",
        "No on/off switch: the better the cluster scores, the more money it gets, up to a "
        "ceiling of 8%. Note that because the size moves every day the score moves, this one "
        "trades a lot and pays slippage for it.",
    ),
    "big_money_director": (
        "Real dollars, with a board member among the buyers",
        "Enter at 5% when the window's <code>total_value &ge; $500,000</code>, a director is "
        "among the buyers, and <code>conviction_score &ge; 0</code>.",
        "Two filters that are hard to fake: the buying has to add up to serious money, and "
        "at least one board member has to be in it.",
    ),
    "officer_director_combo": (
        "Both an executive and a board member bought",
        "Enter at 6% when <code>includes_officer</code> and <code>includes_director</code> "
        "are both true and <code>num_insiders &ge; 2</code>.",
        "Look for agreement across the two sides of the company — someone running it and "
        "someone overseeing it both putting money in during the same stretch.",
    ),
    "big_stake_increase": (
        "Someone grew their own holding by a quarter or more",
        "Enter at 7% when <code>max_pct_of_prior_stake &ge; 25</code> (some insider's buy "
        "increased their existing position by at least 25%) and "
        "<code>conviction_score &ge; 1</code>.",
        "Measure the buy against what that person already owned. Adding 25% to your own "
        "stake is a meaningful commitment, whereas a token purchase by someone already "
        "loaded up says very little.",
    ),
    "non_10b5_1_only": (
        "Multi-insider buying with no pre-scheduled plan trades in it",
        "Enter at 4% when <code>num_insiders &ge; 3</code> and "
        "<code>conviction_score &gt; 0</code> and <code>any_10b5_1</code> is false — the "
        "<code>multi_insider_positive</code> gate plus a hard exclusion of any window "
        "containing a Rule 10b5-1 transaction.",
        "Same as multi_insider_positive, but throws out any cluster where even one purchase "
        "came from a pre-set automatic trading plan. The point is to keep only buying that "
        "someone actively decided to do now. A 10b5-1 buy was scheduled months in advance and "
        "would have happened whatever the insider thinks today, so it carries no information "
        "— and worse, several of them landing in the same window can make a cluster look like "
        "a crowd when it is really just a calendar.",
    ),
    "momentum_confirmed_cluster": (
        "Good cluster, and the price is already rising",
        "Enter at 5% when <code>conviction_score &ge; +2</code> and the trailing 20-day "
        "return is positive.",
        "Insist that the market already agrees. Wait for a decent cluster on a stock that has "
        "risen over the past month, rather than trying to catch a falling knife.",
    ),
    "oversold_cluster_reversion": (
        "Good cluster on a beaten-down stock that has stopped falling",
        "Enter at 5% when <code>conviction_score &ge; +2</code>, price is at least 20% below "
        "its 90-day high, and the trailing 20-day return is better than &minus;10%.",
        "The opposite bet to momentum: buy the dip, but only a dip that is levelling off. The "
        "stock has to be well off its recent high yet not still in free fall.",
    ),
    "low_vol_conviction": (
        "High-scoring clusters in calm stocks only",
        "Enter at 6% when <code>conviction_score &ge; +3</code> and 30-day annualized "
        "volatility is under 40%.",
        "Take the strong signals but skip the wild ones. A quiet stock's move is more likely "
        "to be about the company than about noise.",
    ),
    "vol_scaled_conviction": (
        "Size by score, then shrink it for jumpy stocks",
        "Enter at <code>min(8%, 2% &times; conviction_score &times; (0.30 / max(vol_30d, "
        "0.30)))</code>. Volatility at or below 30% leaves the size untouched; higher "
        "volatility scales it down proportionally.",
        "Two dials at once: a better score buys more, and a more volatile stock buys less. "
        "The aim is for every position to contribute a similar amount of risk rather than a "
        "similar amount of money.",
    ),
    "conviction_only_stopped_15": (
        "conviction_only, cut at a 15% loss",
        "Identical entry to <code>conviction_only</code> (5% when "
        "<code>conviction_score &ge; +3</code>, skip recent IPOs), with a per-lot stop.",
        "Same picks as conviction_only. The only question being asked here is whether "
        "bailing out of a loser helps or just locks in dips.",
    ),
    "multi_insider_stopped_20": (
        "multi_insider_positive, cut at a 20% loss",
        "Identical entry to <code>multi_insider_positive</code> (4% when "
        "<code>num_insiders &ge; 3</code> and <code>conviction_score &gt; 0</code>), with a "
        "per-lot stop.",
        "Same picks as multi_insider_positive, with a wider bail-out level to suit a looser "
        "gate.",
    ),
    "score_weighted_stopped_15": (
        "score_weighted, cut at a 15% loss",
        "Identical sizing to <code>score_weighted</code> "
        "(<code>min(8%, 1.5% &times; conviction_score)</code>), with a per-lot stop.",
        "Same continuous sizing as score_weighted, plus a floor under each position.",
    ),
    "tpo_gated_hold63": (
        "ten_percent_owner_gated, but held ~3 months regardless",
        "Identical entry to <code>ten_percent_owner_gated</code> (10% when a 10% owner "
        "participates and <code>conviction_score &ge; 0</code>).",
        "Same picks as ten_percent_owner_gated, held on purpose. Normally the position gets "
        "sold within a week or two as the cluster ages out — this version refuses to sell, so "
        "you can see what the pick was actually worth over a quarter.",
    ),
    "all_clusters_hold63": (
        "all_clusters_equal_weight, but held ~3 months regardless",
        "Identical entry to <code>all_clusters_equal_weight</code> (2% on any "
        "<code>num_insiders &ge; 2</code> window).",
        "The no-filter control, held for a full quarter instead of being trimmed as the "
        "cluster fades.",
    ),
    "conviction_only_hold63": (
        "conviction_only, but held ~3 months regardless",
        "Identical entry to <code>conviction_only</code> (5% when "
        "<code>conviction_score &ge; +3</code>, skip recent IPOs).",
        "Same picks as conviction_only, held for a full quarter rather than sold as the "
        "cluster fades.",
    ),
    "learned_gt_m01": (
        "Fitted score shows no net red flags",
        "Equal-weight across every ticker with <code>learned_score &gt; -1</code> and "
        "<code>num_insiders &ge; 2</code>.",
        "A deliberately loose version of the fitted score: rather than demanding the model "
        "like a cluster, it only asks the model not to actively dislike it. This stays usable "
        "even when the fit mostly learns what to avoid.",
    ),
    "learned_gt_p00": (
        "Fitted score is positive",
        "Equal-weight across every ticker with <code>learned_score &gt; 0</code> and "
        "<code>num_insiders &ge; 2</code>.",
        "Spread money evenly over everything the fitted model scores positively.",
    ),
    "learned_gt_p03": (
        "Fitted score is strongly positive",
        "Equal-weight across every ticker with <code>learned_score &gt; 3</code> and "
        "<code>num_insiders &ge; 2</code>.",
        "The same idea as learned_gt_p00 with a much higher bar, so far fewer names get in.",
    ),
    "learned_score_weighted": (
        "Size scales with the fitted score",
        "Enter at <code>min(8%, 1.5% &times; learned_score)</code>; nothing at zero or below.",
        "score_weighted's sizing rule with the fitted score in place of the hand-tuned one.",
    ),
    "learned_tpo_gated": (
        "ten_percent_owner_gated with the fitted score as the quality check",
        "Enter at 10% when <code>includes_ten_percent_owner</code> is true and "
        "<code>learned_score &ge; 0</code>.",
        "A controlled comparison. It copies ten_percent_owner_gated exactly — same 10% bets, "
        "same slot count — and swaps only the quality check from the hand-tuned score to the "
        "fitted one, so any difference in results is down to the score and not to how the "
        "money was spread.",
    ),
    "learned_tail_concentrated": (
        "Concentrated bets on the highest lottery-ticket scores",
        "Enter at 10% when <code>num_insiders &ge; 2</code> and <code>tail_score &ge; 3</code> "
        "(threshold set to be about as selective as the 10%-owner flag, roughly one "
        "qualifying cluster per day).",
        "Borrows ten_percent_owner_gated's shape — a few large positions — but picks the "
        "names by their chance of being a huge winner rather than by their average expected "
        "return.",
    ),
    "model_ranked_hold63": (
        "all_clusters_hold63, but the ranking model decides who gets a slot",
        "Identical entry and hold to <code>all_clusters_hold63</code> (2% on any "
        "<code>num_insiders &ge; 2</code> window); the only change is that capacity contention "
        "is resolved by highest <code>model_score</code> instead of arrival order.",
        "Same buy rule and same holding period as all_clusters_hold63. The difference is who "
        "gets in when the portfolio is full: the model's best-rated candidates instead of "
        "whoever happened to show up first.",
    ),
    "model_ranked_top_hold63": (
        "model_ranked_hold63 squeezed to 10 slots so the ranking actually matters",
        "Same entry as <code>model_ranked_hold63</code> but sized at 10% to stay fully "
        "invested across a much tighter cap. At 50 slots the loose 2-insider gate rarely "
        "produces more qualifiers than there are openings, so the ranking has nothing to "
        "decide.",
        "The real test of the ranking model. By cutting the portfolio to ten names, the model "
        "is forced to turn candidates away every day — which is the only situation where its "
        "opinion changes anything.",
    ),
    "spy_buy_and_hold": (
        "The benchmark: buy SPY on day one and do nothing",
        "Full capital into SPY at the start of the window, held to the end. No signals, no "
        "rebalancing, no exit method.",
        "What you would have made by just owning the S&amp;P 500 for the same period. Any "
        "strategy that does not beat this line did not earn its complexity.",
    ),
}


# thr_gt_p03 / thr_gt_m01 -> +3 / -1
_THRESHOLD_RE = re.compile(r"^thr_gt_([mp])(\d+)$")


def _threshold_entry(name: str) -> tuple[str, str, str] | None:
    """Generate a legend entry for the auto-generated thr_gt_* family."""
    m = _THRESHOLD_RE.match(name)
    if m is None:
        return None
    sign, digits = m.groups()
    threshold = -int(digits) if sign == "m" else int(digits)
    shown = f"{threshold:+d}"
    return (
        f"Equal-weight on everything scoring above {shown}",
        f"Equal-weight across every ticker with <code>conviction_score &gt; {shown}</code> "
        f"and <code>num_insiders &ge; 2</code>; capital is split evenly over however many "
        f"qualify that day.",
        f"One rung of the score sweep. It buys every cluster scoring better than {shown} and "
        f"gives them all the same weight. Reading the whole <code>thr_gt_*</code> family in "
        f"order shows what raising the quality bar actually buys you: fewer, better names, "
        f"but also a smaller and lumpier portfolio.",
    )


def _chips(strategy) -> str:
    """Structural facts read straight off the Strategy dataclass."""
    chips: list[str] = []
    if strategy.equal_weight:
        chips.append("equal-weight")
    if strategy.max_concurrent_tickers is None:
        chips.append("no cap on positions")
    else:
        chips.append(f"max {strategy.max_concurrent_tickers} positions")
    if strategy.hold_days is not None:
        chips.append(f"fixed {strategy.hold_days}-day hold")
    if strategy.stop_loss_pct is not None:
        chips.append(f"&minus;{strategy.stop_loss_pct * 100:.0f}% stop-loss")
    if strategy.rank_fn is not None:
        label = getattr(strategy.rank_fn, "__name__", "custom")
        chips.append(f"slots ranked by {html.escape(label)}")
    return "".join(f'<span class="legend-chip">{c}</span>' for c in chips)


def _entry_html(name: str, strategy) -> str:
    entry = ENTRY_RULES.get(name) or _threshold_entry(name)
    if entry is None:
        # Unknown strategy (newly added, not yet documented here): fall back to
        # the registry's own description rather than dropping it from the
        # legend entirely.
        desc = html.escape(strategy.description) if strategy is not None else ""
        gist, technical, plain = ("", desc, "")
    else:
        gist, technical, plain = entry

    parts = [f'<div class="legend-tech"><b>Technical.</b> {technical}</div>']
    if plain:
        parts.append(f'<div class="legend-plain"><b>In plain English.</b> {plain}</div>')
    if strategy is not None:
        chip_html = _chips(strategy)
        if chip_html:
            parts.append(f'<div class="legend-chips">{chip_html}</div>')

    gist_html = f'<span class="legend-gist">{gist}</span>' if gist else ""
    return (
        f'<details class="legend-item">'
        f'<summary><code class="legend-name">{html.escape(name)}</code>{gist_html}</summary>'
        f'<div class="legend-body">{"".join(parts)}</div>'
        f"</details>"
    )


def _exit_methods_html() -> str:
    rows = []
    for em in EXIT_METHODS:
        if em.trailing_stop_pct is None:
            tech = (
                f"Every lot is closed {em.exit_days} trading days after entry, whatever the "
                f"price has done."
            )
            plain = (
                f"Hold each purchase for about "
                f"{em.exit_days / 21:.0f} month{'s' if em.exit_days / 21 >= 1.5 else ''}, "
                f"then sell regardless."
            )
        else:
            pct = f"{em.trailing_stop_pct * 100:.0f}%"
            tech = (
                f"No fixed horizon. A lot is closed when it falls {pct} below its highest "
                f"close since entry, with a {em.exit_days}-trading-day cap as a safety valve."
            )
            plain = (
                f"Let a winner run and only sell once it has given back {pct} from its best "
                f"level. Tighter percentages sell sooner and more often."
            )
        rows.append(
            f'<details class="legend-item">'
            f'<summary><code class="legend-name">{html.escape(em.label)}</code></summary>'
            f'<div class="legend-body">'
            f'<div class="legend-tech"><b>Technical.</b> {tech}</div>'
            f'<div class="legend-plain"><b>In plain English.</b> {plain}</div>'
            f"</div></details>"
        )
    return "".join(rows)


def _glossary_html() -> str:
    rows = "".join(
        f'<div class="legend-term"><dt>{html.escape(term)}</dt><dd>{body}</dd></div>'
        for term, body in GLOSSARY
    )
    return f'<dl class="legend-glossary">{rows}</dl>'


def strategy_legend_html(strategy_order: list[str]) -> str:
    """Collapsible legend for every strategy in this run, plus shared terms.

    `strategy_order` is the run's own list, so a filtered run (--strategies)
    documents only what it actually ran. spy_buy_and_hold is appended because
    it appears in every summary table but is not a registry strategy.
    """
    by_name = {s.name: s for s in STRATEGIES}
    names = list(strategy_order)
    if "spy_buy_and_hold" not in names:
        names.append("spy_buy_and_hold")

    items = "".join(_entry_html(n, by_name.get(n)) for n in names)

    return (
        '<p class="note">Every strategy in this report, explained twice — the exact rule, '
        "and what it means. Click any name to expand.</p>"
        '<div class="legend-controls">'
        '<button type="button" class="legend-btn" data-legend-action="open">Expand all</button>'
        '<button type="button" class="legend-btn" data-legend-action="close">Collapse all</button>'
        "</div>"
        '<div class="legend-group">'
        '<details class="legend-section"><summary>Shared terms used by every rule below'
        "</summary>"
        f'<div class="legend-body">{_glossary_html()}</div></details>'
        "</div>"
        '<h3 class="legend-heading">Strategies</h3>'
        f'<div class="legend-group">{items}</div>'
        '<h3 class="legend-heading">Exit methods</h3>'
        '<p class="note">Every strategy above is run once under each of these exits; the '
        "buttons on each chart switch between them.</p>"
        f'<div class="legend-group">{_exit_methods_html()}</div>'
    )


LEGEND_CSS = """
.legend-controls { margin: 0 0 12px 0; display: flex; gap: 8px; }
.legend-btn {
  font: inherit; font-size: 12px; padding: 4px 10px; cursor: pointer;
  background: #f7f7f9; color: var(--accent); border: 1px solid var(--border);
  border-radius: 3px;
}
.legend-btn:hover { background: #e9e9ee; }
.legend-heading { font-size: 13px; text-transform: uppercase; letter-spacing: 0.08em;
                  color: var(--muted); font-weight: 600; margin: 26px 0 10px 0; }
.legend-group { border: 1px solid var(--border); border-radius: 4px; overflow: hidden; }
.legend-group > details + details { border-top: 1px solid var(--border); }
.legend-item > summary, .legend-section > summary {
  padding: 9px 12px; cursor: pointer; font-size: 13px; list-style: none;
  display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap;
}
.legend-item > summary::-webkit-details-marker,
.legend-section > summary::-webkit-details-marker { display: none; }
.legend-item > summary::before, .legend-section > summary::before {
  content: '\\25B8'; color: var(--muted); font-size: 10px; flex: none;
}
.legend-item[open] > summary::before, .legend-section[open] > summary::before {
  content: '\\25BE';
}
.legend-item > summary:hover, .legend-section > summary:hover { background: #eef5ff; }
.legend-item[open] > summary, .legend-section[open] > summary { background: #f7f7f9; }
.legend-section > summary { font-weight: 600; color: var(--accent); }
.legend-name { background: none; padding: 0; font-weight: 600; color: var(--accent);
               font-size: 12.5px; }
.legend-gist { color: var(--muted); font-size: 12px; }
.legend-body { padding: 4px 12px 14px 30px; font-size: 13px; line-height: 1.6; }
.legend-tech { margin-bottom: 8px; }
.legend-tech b, .legend-plain b { color: var(--accent); }
.legend-plain { color: #333; }
.legend-chips { margin-top: 10px; display: flex; gap: 6px; flex-wrap: wrap; }
.legend-chip {
  font-size: 11px; padding: 2px 8px; border-radius: 10px;
  background: #eef2f7; color: #45596e; border: 1px solid #dde4ec;
}
dl.legend-glossary { margin: 0; }
dl.legend-glossary .legend-term { margin-bottom: 12px; }
dl.legend-glossary dt { font-weight: 600; color: var(--accent); margin-bottom: 2px; }
dl.legend-glossary dd { margin: 0; }
"""

LEGEND_JS = """
(function(){
  document.querySelectorAll('[data-legend-action]').forEach(function(btn){
    btn.addEventListener('click', function(){
      var open = btn.getAttribute('data-legend-action') === 'open';
      var scope = btn.closest('section') || document;
      scope.querySelectorAll('details.legend-item, details.legend-section')
           .forEach(function(d){ d.open = open; });
    });
  });
})();
"""
