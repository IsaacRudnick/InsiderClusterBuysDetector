"""Tests for ticker_reuse: REUSE vs RENAME classification of multi-CIK
tickers, and the events filter it drives.

Follows this repo's pytest conventions: plain pytest, class-grouped tests,
synthetic frames only (see tests/test_splits.py / tests/test_split_fingerprint.py).
No network access, no clusters_history/price_cache reads.
"""

from __future__ import annotations

import os
import sys
from datetime import date, timedelta

import pandas as pd
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from backtest import ticker_reuse as tr  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------
def make_row(*, ticker: str, issuer_cik: str, issuer_name: str, filing_date: date) -> dict:
    """Minimal row shape ticker_reuse actually reads (ticker, issuer_cik,
    issuer_name, filing_date). Other events_df columns are irrelevant to
    this module and omitted."""
    return {
        "ticker": ticker,
        "issuer_cik": issuer_cik,
        "issuer_name": issuer_name,
        "filing_date": filing_date,
    }


def span_rows(ticker: str, cik: str, name: str, dates: list[date]) -> list[dict]:
    return [make_row(ticker=ticker, issuer_cik=cik, issuer_name=name, filing_date=d) for d in dates]


# ---------------------------------------------------------------------------
# normalize_issuer_name / name_similarity
# ---------------------------------------------------------------------------
class TestNameNormalization:
    def test_strips_corporate_suffixes(self):
        assert tr.normalize_issuer_name("APACHE CORP") == "APACHE"
        assert tr.normalize_issuer_name("APA Corp") == "APA"

    def test_strips_state_of_incorporation_tag(self):
        assert tr.normalize_issuer_name("Primo Water Corp /CN/") == "PRIMO WATER"
        assert tr.normalize_issuer_name("QUIDEL CORP /DE/") == "QUIDEL"
        assert tr.normalize_issuer_name("LIFECORE BIOMEDICAL, INC. \\DE\\") == "LIFECORE BIOMEDICAL"

    def test_strips_old_wind_down_tag(self):
        assert tr.normalize_issuer_name(
            "Regional Health Properties,Inc. - Old"
        ) == "REGIONAL HEALTH PROPERTIES"

    def test_case_and_punctuation_insensitive(self):
        a = tr.normalize_issuer_name("HEALTHCARE REALTY TRUST INC")
        b = tr.normalize_issuer_name("Healthcare Realty Trust Inc")
        assert a == b == "HEALTHCARE REALTY TRUST"

    def test_empty_and_none_normalize_to_empty(self):
        assert tr.normalize_issuer_name(None) == ""
        assert tr.normalize_issuer_name("") == ""
        assert tr.normalize_issuer_name("INC CORP") == ""  # nothing but suffixes


class TestNameSimilarity:
    def test_identical_after_normalization_is_1(self):
        assert tr.name_similarity("CIGNA CORP", "Cigna Corp") == pytest.approx(1.0)

    def test_unrelated_names_are_low(self):
        sim = tr.name_similarity("Arlington Asset Investment Corp.", "C3.ai, Inc.")
        assert sim < 0.3

    def test_related_but_distinct_names_are_moderate(self):
        # Real anchor pair: APACHE CORP -> APA Corp.
        sim = tr.name_similarity("APACHE CORP", "APA Corp")
        assert 0.5 < sim < 0.8

    def test_empty_name_gives_zero(self):
        assert tr.name_similarity("", "Something Inc.") == 0.0
        assert tr.name_similarity(None, None) == 0.0


# ---------------------------------------------------------------------------
# classify_ticker_transitions: single-transition scenarios
# ---------------------------------------------------------------------------
class TestClassifyTransitions:
    def test_single_cik_ticker_produces_no_transition(self):
        rows = span_rows("SOLO", "CIK1", "Solo Corp", [date(2020, 1, 1), date(2020, 6, 1)])
        events_df = pd.DataFrame(rows)
        transitions = tr.classify_ticker_transitions(events_df)
        assert transitions.empty

    def test_high_similarity_no_prior_identity_is_rename(self):
        """Same real-world shape as APA/AESI: near-identical name, successor
        never filed under any other ticker first."""
        rows = (
            span_rows("XYZ", "CIK_OLD", "Widget Corp", [date(2019, 1, 1), date(2019, 6, 1)])
            + span_rows("XYZ", "CIK_NEW", "Widget Inc", [date(2019, 8, 1), date(2020, 1, 1)])
        )
        events_df = pd.DataFrame(rows)
        transitions = tr.classify_ticker_transitions(events_df)
        assert len(transitions) == 1
        row = transitions.iloc[0]
        assert row["classification"] == "RENAME"
        assert row["successor_has_prior_identity"] == False  # noqa: E712

    def test_low_similarity_is_reuse_regardless_of_gap(self):
        """Same shape as AI: totally different names, moderate gap."""
        rows = (
            span_rows("ZZZ", "CIK_OLD", "Arlington Asset Investment Corp.",
                      [date(2018, 1, 1), date(2020, 9, 28)])
            + span_rows("ZZZ", "CIK_NEW", "C3.ai, Inc.", [date(2021, 6, 11), date(2022, 1, 1)])
        )
        events_df = pd.DataFrame(rows)
        transitions = tr.classify_ticker_transitions(events_df)
        assert len(transitions) == 1
        assert transitions.iloc[0]["classification"] == "REUSE"

    def test_moderate_similarity_short_gap_is_rename(self):
        """Same shape as APA: moderate similarity (0.65 <= sim < 0.85),
        gap under GAP_RENAME_MAX_DAYS -> rename."""
        rows = (
            span_rows("QQQ", "CIK_A", "APACHE CORP", [date(2018, 8, 29), date(2020, 4, 7)])
            + span_rows("QQQ", "CIK_B", "APA Corp", [date(2021, 3, 12), date(2021, 6, 1)])
        )
        events_df = pd.DataFrame(rows)
        transitions = tr.classify_ticker_transitions(events_df)
        assert len(transitions) == 1
        row = transitions.iloc[0]
        assert row["gap_days"] == (date(2021, 3, 12) - date(2020, 4, 7)).days
        assert row["gap_days"] <= tr.GAP_RENAME_MAX_DAYS
        assert row["classification"] == "RENAME"

    def test_moderate_similarity_long_gap_is_ambiguous(self):
        """Same moderate name-similarity band as the APA case above, but the
        gap is stretched past GAP_RENAME_MAX_DAYS -- neither name evidence
        nor gap evidence is decisive, so this must land AMBIGUOUS, not a
        guess either way."""
        rows = (
            span_rows("QAMB", "CIK_A", "APACHE CORP", [date(2018, 8, 29), date(2018, 9, 1)])
            + span_rows("QAMB", "CIK_B", "APA Corp",
                        [date(2018, 9, 1) + timedelta(days=tr.GAP_RENAME_MAX_DAYS + 30)])
        )
        events_df = pd.DataFrame(rows)
        transitions = tr.classify_ticker_transitions(events_df)
        assert len(transitions) == 1
        row = transitions.iloc[0]
        assert row["gap_days"] > tr.GAP_RENAME_MAX_DAYS
        assert row["classification"] == "AMBIGUOUS"

    def test_prior_identity_overrides_identical_name(self):
        """Same shape as BBBY: identical names (similarity 1.0) would
        normally mean RENAME, but the successor CIK already filed under a
        totally different ticker before ever touching this one -- proof it
        was already a distinct, independent company. Must be REUSE."""
        rows = (
            span_rows("BBBY_T", "CIK_OLDCO", "BED BATH & BEYOND INC",
                      [date(2020, 7, 14), date(2022, 7, 28)])
            # The successor's OWN prior identity, under a different ticker,
            # predates its first filing under BBBY_T.
            + span_rows("OSTK_T", "CIK_NEWCO", "OVERSTOCK.COM, INC",
                        [date(2019, 4, 3), date(2024, 1, 1)])
            + span_rows("BBBY_T", "CIK_NEWCO", "BED BATH & BEYOND, INC.", [date(2026, 3, 12)])
        )
        events_df = pd.DataFrame(rows)
        transitions = tr.classify_ticker_transitions(events_df)
        bbby_row = transitions[transitions["ticker"] == "BBBY_T"].iloc[0]
        assert bbby_row["name_similarity"] == pytest.approx(1.0)
        assert bbby_row["successor_has_prior_identity"] == True  # noqa: E712
        assert bbby_row["classification"] == "REUSE"
        # OSTK_T itself only has one CIK -- no transition for it.
        assert (transitions["ticker"] == "OSTK_T").sum() == 0

    def test_predecessor_continues_elsewhere_is_reported_not_decisive(self):
        """predecessor_continues_elsewhere is informational only (see module
        docstring) -- it must be computed and reported, but must not by
        itself flip a high-similarity, no-prior-identity pair away from
        RENAME."""
        rows = (
            span_rows("MOVED", "CIK_A", "Foo Corp", [date(2019, 1, 1), date(2019, 6, 1)])
            + span_rows("MOVED", "CIK_B", "Foo Inc", [date(2019, 8, 1), date(2020, 1, 1)])
            # CIK_A keeps filing elsewhere AFTER leaving ticker MOVED.
            + span_rows("ELSEWHERE", "CIK_A", "Foo Corp", [date(2019, 9, 1), date(2021, 1, 1)])
        )
        events_df = pd.DataFrame(rows)
        transitions = tr.classify_ticker_transitions(events_df)
        row = transitions[transitions["ticker"] == "MOVED"].iloc[0]
        assert row["predecessor_continues_elsewhere"] == True  # noqa: E712
        assert row["classification"] == "RENAME"

    def test_missing_ticker_or_cik_rows_are_ignored(self):
        rows = span_rows("CLN", "CIK1", "Clean Corp", [date(2020, 1, 1), date(2020, 6, 1)])
        rows.append(make_row(ticker=None, issuer_cik="CIK2", issuer_name="Ghost", filing_date=date(2020, 3, 1)))
        rows.append(make_row(ticker="CLN", issuer_cik="", issuer_name="Ghost2", filing_date=date(2020, 4, 1)))
        events_df = pd.DataFrame(rows)
        transitions = tr.classify_ticker_transitions(events_df)
        assert transitions.empty  # only one real CIK ("CIK1") remains on CLN

    def test_missing_required_column_raises(self):
        events_df = pd.DataFrame({"ticker": ["A"], "issuer_cik": ["1"]})
        with pytest.raises(ValueError):
            tr.classify_ticker_transitions(events_df)


# ---------------------------------------------------------------------------
# unsafe_cik_periods: chain-walk logic
# ---------------------------------------------------------------------------
class TestChainWalk:
    def test_two_cik_reuse_marks_predecessor_unsafe(self):
        rows = (
            span_rows("T", "OLD", "Alpha Corp", [date(2018, 1, 1), date(2018, 6, 1)])
            + span_rows("T", "NEW", "Zebra Industries", [date(2022, 1, 1), date(2022, 6, 1)])
        )
        events_df = pd.DataFrame(rows)
        transitions = tr.classify_ticker_transitions(events_df)
        assert transitions.iloc[0]["classification"] == "REUSE"
        unsafe = tr.unsafe_cik_periods(transitions)
        assert ("T", "OLD") in unsafe
        assert ("T", "NEW") not in unsafe  # current occupant is always safe

    def test_two_cik_rename_marks_nothing_unsafe(self):
        rows = (
            span_rows("T", "OLD", "Widget Corp", [date(2018, 1, 1), date(2018, 6, 1)])
            + span_rows("T", "NEW", "Widget Inc", [date(2018, 8, 1), date(2019, 1, 1)])
        )
        events_df = pd.DataFrame(rows)
        transitions = tr.classify_ticker_transitions(events_df)
        assert transitions.iloc[0]["classification"] == "RENAME"
        unsafe = tr.unsafe_cik_periods(transitions)
        assert unsafe == set()

    def test_three_cik_chain_rename_then_reuse_cuts_off_earliest(self):
        """C1 -[RENAME]-> C2 -[REUSE]-> C3 (current). C3 is safe; C2 is cut
        off from C3 by the REUSE link and must be unsafe; C1 must be unsafe
        too even though C1->C2 was itself a rename, because that whole
        earlier pair is orphaned from the current occupant."""
        rows = (
            span_rows("CHAIN", "C1", "Foo Corp", [date(2015, 1, 1), date(2015, 6, 1)])
            + span_rows("CHAIN", "C2", "Foo Inc", [date(2015, 8, 1), date(2016, 1, 1)])
            + span_rows("CHAIN", "C3", "Totally Different Co", [date(2022, 1, 1), date(2022, 6, 1)])
        )
        events_df = pd.DataFrame(rows)
        transitions = tr.classify_ticker_transitions(events_df)
        assert len(transitions) == 2
        unsafe = tr.unsafe_cik_periods(transitions)
        assert ("CHAIN", "C1") in unsafe
        assert ("CHAIN", "C2") in unsafe
        assert ("CHAIN", "C3") not in unsafe

    def test_three_cik_chain_reuse_then_rename_keeps_current_pair_safe(self):
        """C1 -[REUSE]-> C2 -[RENAME]-> C3 (current). C2 and C3 share
        identity (rename), so both are safe; C1 is cut off and unsafe."""
        rows = (
            span_rows("CHAIN2", "C1", "Ancient Corp", [date(2015, 1, 1), date(2015, 6, 1)])
            + span_rows("CHAIN2", "C2", "Zorp Holdings", [date(2019, 1, 1), date(2019, 6, 1)])
            + span_rows("CHAIN2", "C3", "Zorp Inc", [date(2019, 8, 1), date(2020, 1, 1)])
        )
        events_df = pd.DataFrame(rows)
        transitions = tr.classify_ticker_transitions(events_df)
        assert len(transitions) == 2
        unsafe = tr.unsafe_cik_periods(transitions)
        assert ("CHAIN2", "C1") in unsafe
        assert ("CHAIN2", "C2") not in unsafe
        assert ("CHAIN2", "C3") not in unsafe

    def test_empty_transitions_gives_empty_unsafe_set(self):
        assert tr.unsafe_cik_periods(pd.DataFrame(columns=[
            "ticker", "cik_a", "classification",
        ])) == set()


# ---------------------------------------------------------------------------
# filter_unsafe_ticker_reuse
# ---------------------------------------------------------------------------
class TestFilterUnsafeTickerReuse:
    def test_single_cik_ticker_is_never_touched(self):
        rows = span_rows("KEEP", "CIK1", "Keep Corp", [date(2020, 1, 1), date(2020, 6, 1)])
        events_df = pd.DataFrame(rows)
        filtered, n_dropped, transitions = tr.filter_unsafe_ticker_reuse(events_df)
        assert n_dropped == 0
        assert len(filtered) == len(events_df)
        assert transitions.empty

    def test_reuse_ticker_drops_only_predecessor_rows(self):
        rows = (
            span_rows("DROP", "OLD", "Alpha Corp", [date(2018, 1, 1), date(2018, 6, 1)])
            + span_rows("DROP", "NEW", "Zebra Industries", [date(2022, 1, 1), date(2022, 6, 1)])
            + span_rows("OTHER", "SOLO", "Solo Corp", [date(2020, 1, 1)])
        )
        events_df = pd.DataFrame(rows)
        filtered, n_dropped, transitions = tr.filter_unsafe_ticker_reuse(events_df)
        assert n_dropped == 2  # the two OLD/DROP rows
        assert not (filtered["issuer_cik"] == "OLD").any()
        assert (filtered["issuer_cik"] == "NEW").sum() == 2
        assert (filtered["ticker"] == "OTHER").sum() == 1

    def test_rename_ticker_drops_nothing(self):
        rows = (
            span_rows("SAFE_T", "OLD", "Widget Corp", [date(2018, 1, 1), date(2018, 6, 1)])
            + span_rows("SAFE_T", "NEW", "Widget Inc", [date(2018, 8, 1), date(2019, 1, 1)])
        )
        events_df = pd.DataFrame(rows)
        filtered, n_dropped, _ = tr.filter_unsafe_ticker_reuse(events_df)
        assert n_dropped == 0
        assert len(filtered) == len(events_df)

    def test_does_not_mutate_input(self):
        rows = (
            span_rows("MUT", "OLD", "Alpha Corp", [date(2018, 1, 1)])
            + span_rows("MUT", "NEW", "Zebra Industries", [date(2022, 1, 1)])
        )
        events_df = pd.DataFrame(rows)
        before = events_df.copy(deep=True)
        tr.filter_unsafe_ticker_reuse(events_df)
        pd.testing.assert_frame_equal(events_df, before)

    def test_empty_events_df(self):
        events_df = pd.DataFrame(columns=["ticker", "issuer_cik", "issuer_name", "filing_date"])
        filtered, n_dropped, transitions = tr.filter_unsafe_ticker_reuse(events_df)
        assert n_dropped == 0
        assert filtered.empty
        assert transitions.empty


# ---------------------------------------------------------------------------
# The eight named anchor tickers, reproduced from real issuer names/dates
# (see ticker_reuse.py's module docstring "Calibration" section). Each is
# built as a minimal two-CIK events_df, not read from the real parquet, to
# keep this test suite network-free and independent of clusters_history/.
# ---------------------------------------------------------------------------
class TestRequiredAnchors:
    @pytest.mark.parametrize(
        "ticker, name_a, dates_a, name_b, dates_b, expected",
        [
            ("AI", "Arlington Asset Investment Corp.", [date(2018, 10, 3), date(2020, 9, 28)],
             "C3.ai, Inc.", [date(2021, 6, 11), date(2026, 3, 31)], "REUSE"),
            ("AKTS", "Akoustis Technologies, Inc.", [date(2018, 11, 9), date(2024, 1, 31)],
             "Aktis Oncology, Inc.", [date(2026, 1, 12), date(2026, 1, 14)], "REUSE"),
            ("AERO", "AeroGrow International, Inc.", [date(2019, 2, 25), date(2019, 3, 1)],
             "Grupo Aeromexico, S.A.B. de C.V.", [date(2026, 3, 25)], "REUSE"),
            ("AMC", "AMERICAN SHARED HOSPITAL SERVICES", [date(2019, 5, 17)],
             "AMC ENTERTAINMENT HOLDINGS, INC.", [date(2026, 5, 19)], "REUSE"),
            ("APA", "APACHE CORP", [date(2018, 8, 29), date(2020, 4, 7)],
             "APA Corp", [date(2021, 3, 12), date(2025, 4, 3)], "RENAME"),
            ("AESI", "Atlas Energy Solutions Inc.", [date(2023, 3, 15), date(2023, 6, 6)],
             "Atlas Energy Solutions Inc.", [date(2024, 6, 24), date(2026, 3, 6)], "RENAME"),
        ],
    )
    def test_anchor_ticker_without_prior_identity_evidence(
        self, ticker, name_a, dates_a, name_b, dates_b, expected,
    ):
        """These six anchors classify correctly from name similarity and gap
        alone -- no other-ticker history needed in the fixture. AMR and
        BBBY are deliberately NOT here: both have moderate-to-high name
        similarity that only resolves to REUSE once the successor's real
        prior-identity evidence (Alpha Metallurgical Resources was
        Contura Energy; the BBBY-successor was Overstock.com) is present --
        see test_amr_prior_identity_confirms_reuse and
        test_bbby_needs_prior_identity_override below."""
        rows = (
            span_rows(ticker, "CIK_A", name_a, dates_a)
            + span_rows(ticker, "CIK_B", name_b, dates_b)
        )
        events_df = pd.DataFrame(rows)
        transitions = tr.classify_ticker_transitions(events_df)
        assert len(transitions) == 1
        row = transitions.iloc[0]
        assert row["classification"] == expected, (
            f"{ticker}: expected {expected}, got {row['classification']} "
            f"(name_sim={row['name_similarity']}, gap_days={row['gap_days']}, "
            f"reason={row['reason']!r})"
        )

    def test_bbby_needs_prior_identity_override(self):
        """BBBY's name similarity is 1.000 (identical after normalization) --
        without the successor's real prior identity as Overstock.com/OSTK,
        this would misclassify as RENAME. Reproduces that prior identity in
        the fixture, as it exists in the real events parquet."""
        rows = (
            span_rows("BBBY", "OLDCIK", "BED BATH & BEYOND INC",
                      [date(2020, 7, 14), date(2022, 7, 28)])
            + span_rows("OSTK", "NEWCIK", "OVERSTOCK.COM, INC", [date(2019, 4, 3), date(2022, 6, 1)])
            + span_rows("BYON", "NEWCIK", "BEYOND, INC.", [date(2023, 1, 1), date(2025, 1, 1)])
            + span_rows("BBBY", "NEWCIK", "BED BATH & BEYOND, INC.", [date(2026, 3, 12)])
        )
        events_df = pd.DataFrame(rows)
        transitions = tr.classify_ticker_transitions(events_df)
        bbby_row = transitions[transitions["ticker"] == "BBBY"].iloc[0]
        assert bbby_row["name_similarity"] == pytest.approx(1.0)
        assert bbby_row["successor_has_prior_identity"] == True  # noqa: E712
        assert bbby_row["classification"] == "REUSE"

    def test_amr_prior_identity_confirms_reuse(self):
        """AMR's successor CIK is the real Contura Energy -> Alpha
        Metallurgical Resources rename (see module docstring); it filed
        under ticker CTRA years before its first AMR filing. Confirms the
        override path fires the same way it does for BBBY, on the actual
        evidence shape found in the real dataset."""
        rows = (
            span_rows("AMR", "OLDCIK", "Alta Mesa Resources, Inc. /DE",
                      [date(2018, 8, 21), date(2018, 9, 7)])
            + span_rows("CTRA", "NEWCIK", "Contura Energy, Inc.", [date(2019, 11, 22), date(2020, 3, 27)])
            + span_rows("AMR", "NEWCIK", "Alpha Metallurgical Resources, Inc.",
                        [date(2021, 3, 17), date(2026, 6, 16)])
        )
        events_df = pd.DataFrame(rows)
        transitions = tr.classify_ticker_transitions(events_df)
        amr_row = transitions[transitions["ticker"] == "AMR"].iloc[0]
        assert amr_row["successor_has_prior_identity"] == True  # noqa: E712
        assert amr_row["classification"] == "REUSE"
