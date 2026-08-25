"""Guard the claims the product makes about itself.

findings.py exists so that correcting the research corrects the product. That
only works if nobody quietly re-introduces a claim the research has retired.
Twice now the dashboard has shipped a number that later work contradicted:
the "top decile has a measured edge" banner, and the top-band book result
that a pre-registered holdout then failed to reproduce.

These tests do not check that the numbers are right. Only the research can do
that. They check that every headline number travels with the limit that
applies to it, which is the failure mode this module was built to prevent.
"""

import findings


NOTE_FUNCS = (
    "headline",
    "band_headline",
    "new_score_quality_note",
    "prev_score_contrast_note",
    "elevated_risk_crash_note",
    "top_band_permutation_note",
    "risk_adjusted_note",
    "harvestability_note",
    "top_band_holding_summary",
    "holdout_note",
    "exit_rule_note",
    "band_holdout_note",
    "survivorship_correction_note",
)


def test_every_note_returns_text():
    for name in NOTE_FUNCS:
        out = getattr(findings, name)()
        assert isinstance(out, str), name
        assert len(out) > 40, name


def test_dead_ticker_fates_account_for_every_ticker():
    assert abs(sum(findings.DEAD_TICKER_FATES.values()) - 1.0) < 0.005
    assert len(findings.dead_ticker_rows()) == len(findings.DEAD_TICKER_FATES)
    # Largest share first, so the acquired row cannot hide below the fold.
    shares = [float(s.rstrip("%")) for _, s in findings.dead_ticker_rows()]
    assert shares == sorted(shares, reverse=True)


def test_acquisitions_outnumber_bankruptcies():
    """The correction that overturned a standing assumption. If this ever
    flips, the 'the missing rows are all disasters' reading becomes correct
    again and the notes need rewriting, not just the number."""
    assert findings.DEAD_TICKER_FATES["Acquired"] > findings.DEAD_TICKER_FATES["Bankrupt"]


def test_survivorship_haircut_is_negative_and_ordered():
    optimistic, pessimistic = findings.MEAN_TRADE_HAIRCUT_RANGE
    assert pessimistic < optimistic < 0
    # Even the generous end must erase the survivors-only mean trade, which is
    # the whole point of quoting the range.
    assert findings.MEAN_TRADE_SURVIVORS_ONLY + optimistic <= 0.001


def test_band_ordering_is_the_one_the_product_depends_on():
    """top_band must measure better than above_band. The product surfaces the
    70-90 band precisely because a higher percentile is NOT better."""
    top = findings.BANDS_BY_VERDICT["top_band"]
    above = findings.BANDS_BY_VERDICT["above_band"]
    worst = findings.BANDS_BY_VERDICT["elevated_risk"]
    assert top.p_loses_30pct < above.p_loses_30pct
    assert top.median_excess > above.median_excess
    assert worst.p_loses_30pct == max(b.p_loses_30pct for b in findings.BANDS)


def test_the_search_failed_its_holdout():
    assert findings.SEARCH_HOLDOUT_SHARPE < findings.SEARCH_SELECTION_SHARPE
    # The whole top-ten region landed below SPY. If any of this stops being
    # true the "selection does not survive" claim must be re-argued.
    _, best_of_top10 = findings.SEARCH_TOP10_HOLDOUT_SHARPE_RANGE
    assert best_of_top10 < findings.HOLDOUT_SPY_SHARPE


def test_the_exit_rule_beat_the_fixed_hold_everywhere():
    improved, total = findings.EXIT_RULE_CELLS_IMPROVED
    assert improved == total
    assert findings.EXIT_RULE_TRAIL_SHARPE > findings.EXIT_RULE_FIXED_SHARPE


def test_the_band_did_not_survive_its_holdout():
    helped, total = findings.BAND_HOLDOUT_SLOTS_HELPED
    assert helped < total / 2
    assert findings.BAND_HOLDOUT_MEAN_SHARPE_DELTA < 0


def test_the_return_headline_never_travels_alone():
    """top_band_holding_summary states +34.5%/yr. It must state the limits in
    the same breath. This is the exact regression that shipped twice."""
    text = findings.top_band_holding_summary().lower()
    assert "in sample" in text
    assert "do not trade on that number" in text
    assert "holdout" in text


def test_the_book_table_leads_with_its_warning():
    """The table shows a book beating SPY. A reader who stops after the first
    sentence must already know it is in-sample and survivors-only."""
    lead = findings.risk_adjusted_note()
    first = lead.split(".")[0].lower()
    assert "warning" in first
    head = lead[: lead.index("Over this window")].lower()
    assert "in-sample" in head
    assert "holdout" in head
    assert "survivors-only" in head


def test_harvestability_note_carries_both_corrections():
    text = findings.harvestability_note().lower()
    for term in ("cost", "liquidity", "price floor", "holdout", "survivorship"):
        assert term in text, term


def test_key_points_state_the_retired_claims():
    joined = " ".join(findings.key_points()).lower()
    # The triage framing, which is what the product is for.
    assert "research time" in joined
    # The two corrections that are newer than the book table.
    assert "holdout" in joined
    assert "acquired" in joined
    # The non-monotonicity a reader will otherwise get wrong.
    assert "higher percentile is not a better candidate" in joined
    # No index-beating claim survives out of sample, so none may be implied.
    assert "beat an index fund out of sample" in joined


def test_no_note_promises_a_tradeable_edge():
    """A blunt string guard. These phrases have each appeared in a shipped
    banner that later work retired."""
    banned = ("measured edge", "beats the market", "index-beating strategy")
    for name in NOTE_FUNCS:
        text = getattr(findings, name)().lower()
        for phrase in banned:
            assert phrase not in text, f"{name} says {phrase!r}"
