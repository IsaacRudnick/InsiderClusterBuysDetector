"""Detect ticker REUSE vs ticker RENAME in an events parquet.

A Form 4 records the issuer's ticker as of the FILING date
(insider_cluster_buys.py's `issuerTradingSymbol` parse). That is
point-in-time correct. But backtest/prices.py fetches Yahoo price history
for that ticker string TODAY, keyed only by the symbol. When a ticker has
changed hands between two unrelated companies, the OLD company's events get
priced against the NEW occupant's entire price history -- a delisted
company's insider buys silently inherit a survivor's returns.

Not every multi-CIK ticker is this bug, though. Some are the same company
reorganizing under a new CIK (bankruptcy emergence, holdco conversion,
redomicile) while the ticker itself never stops representing that company:

    APA   APACHE CORP 2018-2020        -> APA Corp 2021-2025          (RENAME)
    AESI  Atlas Energy Solutions 2023   -> Atlas Energy Solutions 2024-2026 (RENAME)

Others are a delisted company's ticker picked up, months or years later, by
a completely unrelated business:

    AI    Arlington Asset Investment 2018-2020 -> C3.ai 2021-2026     (REUSE)
    AMR   Alta Mesa Resources 2018             -> Alpha Metallurgical 2021-2026 (REUSE)

A naive "drop every event whose CIK is not the ticker's current holder"
rule would throw away the RENAME cases' perfectly good data. This module
tells the two apart using three pieces of evidence per CIK-to-CIK
transition on a ticker:

  1. issuer-name similarity, normalized (strip corporate-form suffixes and
     SEC state-of-incorporation tags, case-fold, then a SequenceMatcher
     ratio). "APACHE CORP" vs "APA Corp" -> 0.667. "Arlington Asset
     Investment Corp." vs "C3.ai, Inc." -> 0.194.

  2. the gap, in days, between the predecessor's last filing on the ticker
     and the successor's first. A rename is usually a fast, contiguous
     handoff; a reuse usually has the ticker sitting unclaimed for a while
     after a delisting before someone else lists under it.

  3. successor_has_prior_identity: does the SUCCESSOR's issuer_cik have
     Form 4 activity under a DIFFERENT ticker that predates its first
     filing under this one? This is the decisive signal, and it is what
     makes this module safe against the single case that defeats (1) and
     (2) together: a successor that deliberately adopts a name matching
     (or resembling) the predecessor's, while actually being a wholly
     separate, already-established company. Every one of the following was
     hand-verified against this repo's own events parquet while calibrating
     this module -- in each, the "successor" traded under its own, unrelated
     ticker for years before it ever touched the ticker in question:

       BBBY  successor CIK 0001130713 filed as OSTK (Overstock.com) from
             2019-04-03, years before its 2 filings under ticker BBBY in
             2026 -- the real Bed Bath & Beyond (CIK 0000886158, 2020-2022)
             went bankrupt; Overstock/Beyond, Inc. licensed the brand and
             later filed under the freed ticker. Name similarity for this
             pair is 1.000 (identical after normalization) -- name
             similarity ALONE would call this a rename. It is reuse.
       CTRA  successor CIK 0000858470 filed as COG (Cabot Oil & Gas) from
             2019-01-02, three years before its first CTRA filing in 2022 --
             Cabot merged with Cimarex to form Coterra Energy, which took
             the ticker Contura Energy (an unrelated coal company, CIK
             0001704715) had vacated when Contura itself renamed to Alpha
             Metallurgical Resources and moved to ticker AMR. Two separate
             reuses share a single CIK's history here: 0001704715 is the
             REUSE successor on ticker AMR (Alta Mesa Resources's old slot)
             and the REUSE predecessor on ticker CTRA (Coterra's new slot).
       DCOM  successor CIK filed as BDGE (Bridge Bancorp) before its first
             DCOM filing -- a "merger of equals" where Bridge Bancorp was
             the surviving legal entity but took Dime Community Bancshares'
             name and ticker.
       HR    successor CIK filed as HTA (Healthcare Trust of America)
             before its first HR filing -- same shape: HTA was the
             surviving entity in a "merger of equals" with Healthcare
             Realty Trust and took its name and ticker.
       TCF   successor CIK filed as CHFC (Chemical Financial Corp) before
             its first TCF filing -- Chemical Financial was the surviving
             entity in a "merger of equals" with TCF Financial and took its
             name and ticker.
       PRMW  successor CIK filed as COT/COTT (Cott Corporation) before its
             first PRMW filing -- Cott acquired the (much smaller) original
             Primo Water Corporation, then renamed itself to Primo Water
             Corp and inherited its ticker.
       LFCR  successor CIK filed as LNDC (Landec Corporation) before its
             first LFCR filing -- Landec acquired a small standalone company
             called Lifecore Biomedical, delisting it, then years later
             renamed ITSELF to "Lifecore Biomedical, Inc." and moved onto
             the ticker its former acquisition target had vacated.

     In every one of these, name similarity is high (0.86-1.00) and would,
     on its own, call the pair a rename. The prior-identity check is what
     catches them. It is treated as a hard override: it fires only on an
     objective, structural fact (a DIFFERENT ticker string, actually filed
     under, before this one), not on any fuzzy comparison, so overriding
     name similarity with it is a deliberate, evidence-backed choice, not a
     tie-break of last resort.

     predecessor_continues_elsewhere (the mirror check on the OTHER side of
     the transition -- does the predecessor keep filing under some other
     ticker after leaving this one) is recorded for every transition too,
     but is NOT used to decide anything here: unlike successor_has_prior_
     identity, no real transition in this dataset was manually confirmed
     against it while calibrating, so it would be an unvalidated override
     dressed up as a validated one. It is reported for a human reviewer,
     that's all.

Calibration of the two name-similarity bands and the gap threshold.
`clusters_history/events_20180717_20260717.parquet` has 160 CIK-to-CIK
transitions across 156 multi-CIK tickers (four tickers cycle through three
CIKs, giving two transitions each). The eight transitions this module is
required to get right (see the task that produced it) span the hardest part
of the name-similarity axis:

    ticker  name_similarity  gap_days  must be
    AI      0.194            256       REUSE
    AMC     0.280            2559      REUSE
    AERO    0.292            2581      REUSE
    AKTS    0.629            712       REUSE
    APA     0.667            339       RENAME
    AMR     0.708            922       REUSE   (via prior-identity override)
    AESI    1.000            384       RENAME
    BBBY    1.000            1323      REUSE   (via prior-identity override)

Note APA's 0.667 is LOWER than AMR's 0.708, yet APA must be RENAME and AMR
must be REUSE: name similarity cannot separate these two with a single
threshold, which is exactly why this module also uses the gap and the
prior-identity check rather than name similarity alone.

NAME_SIM_HIGH = 0.85 and NAME_SIM_LOW = 0.65 were chosen to place APA
(0.667) and AKTS (0.629) on opposite sides of NAME_SIM_LOW with the two
required anchors closest to that boundary in this dataset -- there is only
0.038 of separation between them (0.667 vs 0.629), so a real transition
landing inside that narrow band is a genuine coin flip this module cannot
resolve from name similarity alone; that is disclosed here rather than
hidden behind a threshold that happens to clear both named cases.
NAME_SIM_HIGH=0.85 sits above every hand-checked genuine rename found while
calibrating (NVRI at 0.800 was the highest sub-threshold case, and it is
deliberately left in the moderate band rather than folded into "high").
GAP_RENAME_MAX_DAYS=400 (~13 months) clears APA's 339-day gap with about two
months of headroom while staying well under AMR's 922-day and AKTS's
712-day gaps, the two closest REUSE anchors on the gap axis.

Known blind spot, disclosed rather than patched: serial SPAC sponsors
frequently reuse a near-identical name for a legally distinct follow-on
vehicle (e.g. "Legato Merger Corp" -> "Legato Merger Corp. IV", "Panacea
Acquisition Corp" -> "Panacea Acquisition Corp. II"). These score high name
similarity (the boilerplate "Acquisition Corp" / "Merger Corp" tokens
inflate the SequenceMatcher ratio) and the new vehicle has no prior
ticker history of its own (a fresh SPAC has never filed before), so
successor_has_prior_identity cannot catch it either. This module will
classify most of these RENAME when, strictly, the sequel is a different
capital-raising entity. No confirmed example of this shape was fully
traced to ground truth while building this module (SPAC S-1s were not
pulled), so rather than hand-write a "does the name end in a roman
numeral" carve-out calibrated to zero verified cases, this is left as a
disclosed limitation. It is a small population (a handful of tickers) and
errs toward keeping data (RENAME), not toward the dangerous direction
(mispricing via a false RENAME of two totally unrelated companies) --
except that here it plausibly IS the dangerous direction, since the two
SPAC vehicles genuinely are unrelated companies. Flagged for manual review,
not silently accepted.

Usage:
    python backtest/ticker_reuse.py --scan --events clusters_history/events_20180717_20260717.parquet
    python backtest/ticker_reuse.py --scan --events ... --out report.csv
"""

from __future__ import annotations

import argparse
import difflib
import logging
import os
import re
import sys
from dataclasses import dataclass
from datetime import date
from typing import Optional

import pandas as pd

# Run directly (`python backtest/ticker_reuse.py ...`), the interpreter puts
# this file's own directory (backtest/) on sys.path[0], not the repo root --
# so the absolute `backtest.tickers` import below would fail without this.
# Harmless no-op when this module is instead imported normally (e.g. `from
# backtest import ticker_reuse`), since the repo root is already on sys.path
# in that case.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from backtest.tickers import normalize_ticker

log = logging.getLogger(__name__)

Classification = str  # "REUSE" | "RENAME" | "AMBIGUOUS"

# ---------------------------------------------------------------------------
# Tunables -- see the module docstring's "Calibration" section for the
# evidence behind each of these.
# ---------------------------------------------------------------------------

# Normalized issuer names at or above this similarity are treated as the
# same corporate identity (absent a prior-identity override). See docstring.
NAME_SIM_HIGH = 0.85

# Normalized issuer names below this are treated as unrelated companies
# outright, no matter how short the gap. See docstring -- this is only
# 0.038 above AKTS's 0.629 (must be REUSE) and 0.017 below APA's 0.667
# (must be RENAME); the margin either side of this exact value is thin.
NAME_SIM_LOW = 0.65

# In the moderate name-similarity band (NAME_SIM_LOW <= sim < NAME_SIM_HIGH),
# a gap this short or shorter reads as a fast reorg/bankruptcy-emergence/
# redomicile rather than an abandoned-then-reclaimed ticker. See docstring.
GAP_RENAME_MAX_DAYS = 400

_REQUIRED_COLS = {"ticker", "issuer_cik", "issuer_name", "filing_date"}


# ---------------------------------------------------------------------------
# Name normalization / similarity
# ---------------------------------------------------------------------------
# SEC filers commonly append a state-of-incorporation or wind-down tag:
# "/DE/", "/DE", "\DE\", "/MD/", "/NY/", "/CN/", "- Old". These carry no
# brand information and would otherwise drag down similarity between two
# filings for the very same company (e.g. "Primo Water Corp" vs "Primo
# Water Corp /CN/").
_STATE_TAG_RE = re.compile(r"[/\\]\s*[A-Z]{2,4}\s*[/\\]?\s*$")
_OLD_TAG_RE = re.compile(r"-\s*OLD\s*$", re.IGNORECASE)

# Corporate-form words carry no brand information either ("APACHE CORP" and
# "APA Corp" should be compared as "APACHE" vs "APA", not diluted by two
# copies of "CORP" matching each other).
_CORP_SUFFIX_RE = re.compile(
    r"\b(INCORPORATED|INC|CORPORATION|CORP|COMPANY|CO|HOLDINGS|HOLDING|GROUP|"
    r"LLC|LLP|LP|LTD|LIMITED|PLC|THE)\b"
)


def normalize_issuer_name(name) -> str:
    """Fold an issuer_name down to its comparable "brand" core.

    "APACHE CORP" -> "APACHE"; "APA Corp" -> "APA"; "HEALTHCARE REALTY
    TRUST INC" and "Healthcare Realty Trust Inc" both -> "HEALTHCARE REALTY
    TRUST"; "Primo Water Corp /CN/" -> "PRIMO WATER".
    """
    s = str(name or "").upper()
    s = _STATE_TAG_RE.sub(" ", s)
    s = _OLD_TAG_RE.sub(" ", s)
    s = re.sub(r"[^A-Z0-9 ]", " ", s)
    s = _CORP_SUFFIX_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def name_similarity(a, b) -> float:
    """difflib.SequenceMatcher ratio between two normalized issuer names, in
    [0, 1]. 0.0 if either side normalizes to nothing (no brand to compare)."""
    na, nb = normalize_issuer_name(a), normalize_issuer_name(b)
    if not na or not nb:
        return 0.0
    return difflib.SequenceMatcher(None, na, nb).ratio()


# ---------------------------------------------------------------------------
# CIK spans / transitions
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CikSpan:
    """One issuer_cik's filing activity under a single ticker."""
    ticker: str
    cik: str
    name: str
    first_filing: date
    last_filing: date
    n_events: int


@dataclass(frozen=True)
class Transition:
    """One ticker changing hands from cik_a to cik_b -- two chronologically
    adjacent CikSpans sharing the same ticker."""
    ticker: str
    cik_a: str
    name_a: str
    last_filing_a: date
    n_events_a: int
    cik_b: str
    name_b: str
    first_filing_b: date
    n_events_b: int
    gap_days: int
    name_sim: float
    successor_has_prior_identity: bool
    predecessor_continues_elsewhere: bool
    classification: Classification
    reason: str


def _require_columns(events_df: pd.DataFrame) -> None:
    missing = _REQUIRED_COLS - set(events_df.columns)
    if missing:
        raise ValueError(f"events_df is missing required column(s): {sorted(missing)}")


def _issuer_cik_spans(events_df: pd.DataFrame) -> dict[str, list[CikSpan]]:
    """{ticker -> [CikSpan, ...]} sorted by first_filing ascending.

    Rows with a missing ticker or issuer_cik are dropped first -- they
    cannot be assigned to any span and would otherwise show up as a bogus
    extra "CIK" (empty string / NaN) on whatever ticker they landed on.
    """
    _require_columns(events_df)
    df = events_df[["ticker", "issuer_cik", "issuer_name", "filing_date"]].copy()
    df["filing_date"] = pd.to_datetime(df["filing_date"]).dt.date
    df = df[df["ticker"].notna() & df["issuer_cik"].notna()]
    df = df[(df["ticker"].astype(str).str.strip() != "") & (df["issuer_cik"].astype(str).str.strip() != "")]

    out: dict[str, list[CikSpan]] = {}
    for ticker, sub in df.groupby("ticker", sort=False):
        spans = []
        for cik, sub2 in sub.groupby("issuer_cik", sort=False):
            spans.append(CikSpan(
                ticker=ticker,
                cik=cik,
                name=sub2["issuer_name"].iloc[0],
                first_filing=sub2["filing_date"].min(),
                last_filing=sub2["filing_date"].max(),
                n_events=len(sub2),
            ))
        spans.sort(key=lambda s: s.first_filing)
        out[ticker] = spans
    return out


def _cik_ticker_ranges(events_df: pd.DataFrame) -> dict[str, dict[str, tuple[date, date]]]:
    """{issuer_cik -> {ticker -> (min_filing, max_filing)}} across the WHOLE
    events_df (every ticker that CIK has ever filed under), used to answer
    "did this CIK have an identity under some other ticker already?"."""
    df = events_df[["ticker", "issuer_cik", "filing_date"]].copy()
    df["filing_date"] = pd.to_datetime(df["filing_date"]).dt.date
    df = df[df["ticker"].notna() & df["issuer_cik"].notna()]

    out: dict[str, dict[str, tuple[date, date]]] = {}
    for cik, sub in df.groupby("issuer_cik", sort=False):
        per_ticker: dict[str, tuple[date, date]] = {}
        for ticker, sub2 in sub.groupby("ticker", sort=False):
            per_ticker[ticker] = (sub2["filing_date"].min(), sub2["filing_date"].max())
        out[cik] = per_ticker
    return out


def _successor_has_prior_identity(
    cik_ranges: dict[str, dict[str, tuple[date, date]]], cik_b: str, ticker: str, first_filing_b: date,
) -> bool:
    """True iff cik_b filed under some OTHER ticker before first_filing_b --
    i.e. it was already a distinct, independently identified company before
    it ever touched this ticker. See module docstring for verified examples."""
    for t, (mn, _mx) in cik_ranges.get(cik_b, {}).items():
        if t != ticker and mn < first_filing_b:
            return True
    return False


def _predecessor_continues_elsewhere(
    cik_ranges: dict[str, dict[str, tuple[date, date]]], cik_a: str, ticker: str, last_filing_a: date,
) -> bool:
    """True iff cik_a kept filing under some OTHER ticker after leaving this
    one. Reported only -- see module docstring for why this is not used to
    decide a classification."""
    for t, (_mn, mx) in cik_ranges.get(cik_a, {}).items():
        if t != ticker and mx > last_filing_a:
            return True
    return False


def _classify_pair(
    name_sim: float, gap_days: int, successor_has_prior_identity: bool,
    *, name_sim_high: float, name_sim_low: float, gap_rename_max_days: int,
) -> tuple[Classification, str]:
    if successor_has_prior_identity:
        return "REUSE", (
            "successor issuer_cik filed under a DIFFERENT ticker before its first "
            "filing under this one -- it was already a distinct, independently "
            "identified company, so this cannot be a rename of the predecessor "
            "regardless of name similarity (see BBBY/CTRA/DCOM/HR/TCF/PRMW/LFCR in "
            "the module docstring for verified real-world examples of this shape)."
        )
    if name_sim >= name_sim_high:
        return "RENAME", (
            f"issuer names are near-identical after normalization "
            f"(similarity={name_sim:.3f} >= {name_sim_high}) and the successor has "
            "no prior identity elsewhere -- reads as the same corporate identity "
            "continuing under a new CIK."
        )
    if name_sim < name_sim_low:
        return "REUSE", (
            f"issuer names are dissimilar after normalization "
            f"(similarity={name_sim:.3f} < {name_sim_low}); no positive evidence "
            "links the two companies."
        )
    if gap_days <= gap_rename_max_days:
        return "RENAME", (
            f"moderate name similarity ({name_sim:.3f}) combined with a short gap "
            f"({gap_days}d <= {gap_rename_max_days}d) is consistent with a fast "
            "reorg / bankruptcy emergence / redomicile keeping a related name."
        )
    return "AMBIGUOUS", (
        f"moderate name similarity ({name_sim:.3f}) and a long gap "
        f"({gap_days}d > {gap_rename_max_days}d) -- neither a confident rename nor "
        "a confident reuse call; treated as unsafe."
    )


def classify_ticker_transitions(
    events_df: pd.DataFrame,
    *,
    name_sim_high: float = NAME_SIM_HIGH,
    name_sim_low: float = NAME_SIM_LOW,
    gap_rename_max_days: int = GAP_RENAME_MAX_DAYS,
) -> pd.DataFrame:
    """One row per chronologically-adjacent CIK pair, for every ticker whose
    events carry more than one distinct issuer_cik. A ticker with N distinct
    CIKs produces N-1 transitions (three of this dataset's tickers -- A,
    TPL, RMMZ, LGCY -- have 3 CIKs and so 2 transitions each).

    Columns: ticker, cik_a, name_a, last_filing_a, n_events_a, cik_b, name_b,
    first_filing_b, n_events_b, gap_days, name_similarity,
    successor_has_prior_identity, predecessor_continues_elsewhere,
    classification, reason.

    Read-only: never mutates events_df, writes nothing to disk.
    """
    spans_by_ticker = _issuer_cik_spans(events_df)
    cik_ranges = _cik_ticker_ranges(events_df)

    rows: list[dict] = []
    for ticker, spans in spans_by_ticker.items():
        if len(spans) < 2:
            continue
        for a, b in zip(spans, spans[1:]):
            gap_days = (b.first_filing - a.last_filing).days
            sim = name_similarity(a.name, b.name)
            succ_prior = _successor_has_prior_identity(cik_ranges, b.cik, ticker, b.first_filing)
            pred_continues = _predecessor_continues_elsewhere(cik_ranges, a.cik, ticker, a.last_filing)
            classification, reason = _classify_pair(
                sim, gap_days, succ_prior,
                name_sim_high=name_sim_high, name_sim_low=name_sim_low,
                gap_rename_max_days=gap_rename_max_days,
            )
            rows.append({
                "ticker": ticker,
                "cik_a": a.cik, "name_a": a.name, "last_filing_a": a.last_filing, "n_events_a": a.n_events,
                "cik_b": b.cik, "name_b": b.name, "first_filing_b": b.first_filing, "n_events_b": b.n_events,
                "gap_days": gap_days,
                "name_similarity": round(sim, 4),
                "successor_has_prior_identity": succ_prior,
                "predecessor_continues_elsewhere": pred_continues,
                "classification": classification,
                "reason": reason,
            })

    cols = [
        "ticker", "cik_a", "name_a", "last_filing_a", "n_events_a",
        "cik_b", "name_b", "first_filing_b", "n_events_b",
        "gap_days", "name_similarity", "successor_has_prior_identity",
        "predecessor_continues_elsewhere", "classification", "reason",
    ]
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------------------
# Chain walk: which (ticker, cik) spans are safe to price against today's cache
# ---------------------------------------------------------------------------
def unsafe_cik_periods(transitions: pd.DataFrame) -> set[tuple[str, str]]:
    """(ticker, issuer_cik) pairs that must NOT be priced against that
    ticker's current price cache, given `transitions` (classify_ticker_
    transitions' output).

    A ticker's CIK chain is walked backward from its most recent (current,
    "safe by definition") span. A RENAME link keeps the walk going -- that
    span shares the current occupant's continuous identity. The first
    REUSE or AMBIGUOUS link ends the safe run: that link's predecessor
    span, and every span before it, is unsafe, regardless of how THOSE
    earlier links were themselves classified -- once a chain is cut off
    from the current occupant it can never reconnect to it.
    """
    unsafe: set[tuple[str, str]] = set()
    if transitions.empty:
        return unsafe
    for ticker, sub in transitions.groupby("ticker", sort=False):
        sub = sub.reset_index(drop=True)  # already chronological (see classify_ticker_transitions)
        cut_off = False
        for i in range(len(sub) - 1, -1, -1):
            row = sub.iloc[i]
            if cut_off or row["classification"] != "RENAME":
                unsafe.add((ticker, row["cik_a"]))
                cut_off = True
    return unsafe


def filter_unsafe_ticker_reuse(
    events_df: pd.DataFrame,
    *,
    ticker_col: str = "ticker",
    cik_col: str = "issuer_cik",
    name_sim_high: float = NAME_SIM_HIGH,
    name_sim_low: float = NAME_SIM_LOW,
    gap_rename_max_days: int = GAP_RENAME_MAX_DAYS,
) -> tuple[pd.DataFrame, int, pd.DataFrame]:
    """Drop every event row whose (ticker, issuer_cik) is not safely
    continuous with that ticker's CURRENT occupant (see unsafe_cik_periods).

    Returns (filtered_events_df, n_dropped, transitions_df). Never mutates
    `events_df` -- returns a new frame. A ticker events_df shows only one
    issuer_cik for never appears in transitions_df and is always kept in
    full; this only ever removes rows for a ticker whose OWN events prove
    more than one issuer_cik used it.
    """
    if events_df.empty:
        return events_df, 0, classify_ticker_transitions(events_df)

    transitions = classify_ticker_transitions(
        events_df, name_sim_high=name_sim_high, name_sim_low=name_sim_low,
        gap_rename_max_days=gap_rename_max_days,
    )
    unsafe = unsafe_cik_periods(transitions)
    if not unsafe:
        return events_df, 0, transitions

    pairs = pd.Series(
        list(zip(events_df[ticker_col], events_df[cik_col])), index=events_df.index,
    )
    is_unsafe = pairs.apply(lambda p: p in unsafe)
    n_dropped = int(is_unsafe.sum())
    filtered = events_df.loc[~is_unsafe].copy()
    return filtered, n_dropped, transitions


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _load_events_for_scan(path: str) -> pd.DataFrame:
    """Read a raw events parquet and drop rows with an unusable ticker
    (empty/placeholder/malformed), matching backtest/history.py's own
    ticker-normalization rule (backtest.tickers.normalize_ticker) so the
    --scan report's denominator matches what the backtest pipeline actually
    sees. Real parquet files also carry a handful of "CIK0001234567"-style
    placeholder tickers (used when SEC's XBRL omitted a real symbol);
    normalize_ticker's length-limited regex already rejects those too."""
    df = pd.read_parquet(path)
    n_before = len(df)
    df = df.copy()
    df["ticker"] = df["ticker"].apply(normalize_ticker)
    df = df[df["ticker"].notna()].copy()
    log.info(
        "Loaded %s: %d rows, %d dropped for an unusable ticker, %d usable",
        path, n_before, n_before - len(df), len(df),
    )
    return df


def _print_report(transitions: pd.DataFrame, n_dropped: int, n_events_total: int) -> None:
    n_tickers = transitions["ticker"].nunique() if not transitions.empty else 0
    print(f"{len(transitions)} ticker-CIK transition(s) across {n_tickers} multi-CIK ticker(s).")
    by_class = transitions["classification"].value_counts() if not transitions.empty else {}
    for cls in ("REUSE", "RENAME", "AMBIGUOUS"):
        n = int(by_class.get(cls, 0)) if not transitions.empty else 0
        n_tickers_cls = (
            transitions.loc[transitions["classification"] == cls, "ticker"].nunique()
            if not transitions.empty else 0
        )
        print(f"  {cls:<10} {n:4d} transition(s) across {n_tickers_cls} ticker(s)")

    pct = (100.0 * n_dropped / n_events_total) if n_events_total else 0.0
    print(
        f"\nEvents that would be DROPPED under this rule: {n_dropped} / {n_events_total} "
        f"({pct:.3f}% of usable events)"
    )

    if not transitions.empty:
        print("\nAll transitions (ticker, cik_a -> cik_b, name similarity, gap, classification):")
        with pd.option_context("display.max_rows", None, "display.width", 220):
            print(
                transitions[
                    ["ticker", "name_a", "name_b", "gap_days", "name_similarity",
                     "successor_has_prior_identity", "classification"]
                ].to_string(index=False)
            )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Classify every multi-issuer-CIK ticker in an events parquet as "
            "REUSE (an unrelated company later claimed a delisted ticker) or "
            "RENAME (the same company under a new CIK), and report how many "
            "events a REUSE-drop rule would remove. Read-only: writes only "
            "the CSV report, never touches the events parquet."
        )
    )
    parser.add_argument(
        "--scan", action="store_true", required=True,
        help="Report only (the only mode this module has -- detection is read-only).",
    )
    parser.add_argument(
        "--events", required=True,
        help="Path to an events parquet, e.g. clusters_history/events_20180717_20260717.parquet.",
    )
    parser.add_argument(
        "--out", default="ticker_reuse_report.csv",
        help="CSV report path (default: ticker_reuse_report.csv).",
    )
    parser.add_argument("--name-sim-high", type=float, default=NAME_SIM_HIGH)
    parser.add_argument("--name-sim-low", type=float, default=NAME_SIM_LOW)
    parser.add_argument("--gap-rename-max-days", type=int, default=GAP_RENAME_MAX_DAYS)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    events_df = _load_events_for_scan(args.events)
    _filtered, n_dropped, transitions = filter_unsafe_ticker_reuse(
        events_df,
        name_sim_high=args.name_sim_high,
        name_sim_low=args.name_sim_low,
        gap_rename_max_days=args.gap_rename_max_days,
    )
    transitions.to_csv(args.out, index=False)
    print(f"CSV report written to {os.path.abspath(args.out)}")
    _print_report(transitions, n_dropped, len(events_df))
    print("\n--scan: read-only, no input files changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
