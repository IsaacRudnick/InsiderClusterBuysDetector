"""Company size and fundamentals from SEC's bulk XBRL "Financial Statement
Data Sets", keyed on issuer_cik -- point-in-time, like tools/earnings_features.py.

WHY this exists. The research dataset (see tools/issuer_meta.py's docstring)
has no feature describing how BIG a company is or what shape its balance
sheet is in. A $20M shell and a $2B mid-cap both look identical to every
existing feature except price and volume proxies. Size and balance-sheet
health are exactly the kind of thing that plausibly changes what an insider
cluster buy means -- a CFO buying into a company with 2 quarters of cash
left is a different signal than the same buy at a company sitting on a
fortress balance sheet.

SOURCE. SEC publishes ALL XBRL facts filed each quarter as one ZIP at
https://www.sec.gov/files/dera/data/financial-statement-data-sets/YYYYqN.zip
(no per-issuer API calls, no per-CIK rate-limit budget spent -- one request
per quarter). Each ZIP contains:
  sub.txt -- one row per accession (adsh, cik, name, form, period, filed, ...)
  num.txt -- one row per distinct reported numeric fact (adsh, tag, version,
             ddate, qtrs, uom, segments, coreg, value, ...)
This module downloads 2018q1..2026q2 (our events start 2018-08-13), caches
the raw ZIPs under xbrl_cache/, parses each quarter once (caching the parsed,
filtered result under xbrl_cache/parsed/ so a rerun does not re-walk
multi-million-row num.txt files), and writes a single tidy long-format
research_data/fundamentals.parquet.

TAG FALLBACK CHAINS. Companies do not all tag the same concept with the same
XBRL element -- a smaller or late-adopting filer may use a legacy or coarser
tag where a large-cap uses the current standard one. Each concept below is
therefore resolved by trying tags in priority order and taking the first one
actually present in a given filing (see TAG_CHAINS and _resolve_group_facts):

  shares_outstanding:
    dei:EntityCommonStockSharesOutstanding  -- the cover-page count, i.e. the
        single most current share count as of the filing's cover date. This
        is what a live market-cap calculation actually wants. In practice it
        is reported on nearly every filing's cover page but SURFACES IN
        num.txt (a facts-in-the-financial-statements table) only rarely --
        confirmed empirically on 2023q1 (8 of 3.4M rows) -- so this branch
        almost never fires and the chain falls through to:
    us-gaap:CommonStockSharesOutstanding    -- the balance-sheet count, as of
        the period end rather than the cover date. Slightly stale relative
        to the cover page (by up to one reporting lag) but reported far more
        consistently. This is the workhorse of the chain.
    us-gaap:CommonStockSharesIssued         -- issued (not necessarily all
        outstanding -- may include treasury shares). Least preferred: it can
        overstate outstanding count. Used only when a filer tags neither of
        the above, which does happen for small/micro-cap filers.

  revenue:
    us-gaap:Revenues -- the generic top-line tag most filers use.
    us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax -- the
        ASC 606 (new revenue-recognition standard, effective ~2018 for most
        filers) top-line tag. Some filers moved to this tag exclusively when
        they adopted ASC 606 and stopped tagging `Revenues` at all.
    us-gaap:SalesRevenueNet -- the pre-ASC-606 legacy tag. Still used by a
        minority of filers (small/late adopters) even in recent quarters.

  cash:   us-gaap:CashAndCashEquivalentsAtCarryingValue (single canonical tag)
  equity: us-gaap:StockholdersEquity (single canonical tag)
  net_income: us-gaap:NetIncomeLoss (single canonical tag)
  assets: us-gaap:Assets (single canonical tag)

CONSOLIDATED, CURRENT-PERIOD FACTS ONLY. num.txt mixes in: (a) dimensional
breakdowns (by segment, by legal entity, by class of stock -- anything with a
non-blank `segments` or `coreg` value) and (b) comparative prior-period
values every filing re-reports for context (a ddate that is NOT the filing's
own `period` from sub.txt). Both are excluded here: `segments` and `coreg`
must both be blank (consolidated total, not a slice), and `ddate` must equal
the filing's own `period` (this filing's own reporting date, not a
comparative). Skipping this filtering would silently mix segment totals and
stale comparatives into the same column as consolidated current values.

ANNUALIZATION. `revenue` and `net_income` are duration (flow) facts, tagged
with a `qtrs` span (1 = one quarter, 4 = full fiscal year, etc., depending on
whether the filing is a 10-Q or 10-K and which duration it chose to tag).
Comparing a raw 3-month figure to a raw 12-month figure across filings would
be apples to oranges, so every duration value is annualized at write time as
`value * (4 / qtrs)` before being stored in fundamentals.parquet -- a filer
reporting a 9-month year-to-date total (qtrs=3) is projected onto a 4-quarter
run-rate the same way a single quarter (qtrs=1) is scaled by 4. This is a
simple, documented run-rate approximation, not a seasonally-aware TTM -- it
assumes roughly even quarterly pace within a filing, which is wrong for
seasonal businesses but transparent and cheap. `fundamentals.parquet`'s
`value` column for tag_group in {revenue, net_income} IS this annualized
figure; for the four instant tag_groups it is the raw reported value.

POINT-IN-TIME CONTRACT (the whole ballgame -- see attach_features). Every
x_ feature this module produces must use ONLY filings whose `filed` date is
STRICTLY BEFORE the event's `event_day`. A fundamentals feature built from a
filing made AFTER the event would let the model see, e.g., a cash balance
reported three months later -- manufacturing a fake edge that a live system
could never actually have had at decision time. See attach_features's
docstring and tests/test_fundamentals.py's point-in-time test for the
enforcement mechanism (numpy.searchsorted, `side='left'`, same pattern as
tools/earnings_features.py).

Runnable standalone as `python tools/fundamentals.py` (downloads/parses the
bulk XBRL data, writes research_data/fundamentals.parquet, attaches features
to the default research parquet, and prints a coverage + descriptive report)
or importable, e.g. from a test, via `attach_fundamentals_features`.
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import sys
import threading
import time
import zipfile
from datetime import date

import numpy as np
import pandas as pd
import requests

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools import data_paths  # noqa: E402

log = logging.getLogger("fundamentals")

CACHE_DIR = os.path.join(REPO_ROOT, "xbrl_cache")
PARSED_CACHE_DIR = os.path.join(CACHE_DIR, "parsed")
BULK_URL = "https://www.sec.gov/files/dera/data/financial-statement-data-sets/{q}.zip"

DEFAULT_EVENTS = data_paths.latest_research_dataset()
DEFAULT_FUNDAMENTALS_OUT = os.path.join(REPO_ROOT, "research_data", "fundamentals.parquet")


def _quarters(start_year: int, start_q: int, end_year: int, end_q: int) -> list[str]:
    """['2018q1', '2018q2', ..., '2026q2'] -- inclusive of both ends."""
    out = []
    y, q = start_year, start_q
    while (y, q) <= (end_year, end_q):
        out.append(f"{y}q{q}")
        q += 1
        if q > 4:
            q = 1
            y += 1
    return out


# Our events start 2018-08-13, so 2018q1 (Jan-Mar) already covers filings
# from before the first event (needed for point-in-time lookups on the
# earliest events). 2026q2 is the newest quarter published as of this
# module's writing.
QUARTERS = _quarters(2018, 1, 2026, 2)


# ---------------------------------------------------------------------------
# .env loading + rate-limited download session
#
# This module does NOT reuse insider_cluster_buys.py's SESSION/_get/RATE_LIMIT:
# those run at 8 req/s, tuned for the many small per-CIK EDGAR API calls the
# rest of the pipeline makes. This module makes ~34 large (tens-to-~150MB)
# bulk-file requests total, and per the operator's instruction two OTHER
# agents are hitting SEC concurrently on a shared budget -- so this module
# keeps its own, deliberately slower, 3 req/s limiter rather than adding to
# the 8 req/s pool.
# ---------------------------------------------------------------------------
def _load_dotenv() -> None:
    env_path = os.path.join(REPO_ROOT, ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path) as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


_load_dotenv()

USER_AGENT = os.environ.get("SEC_USER_AGENT", "Your Name your.email@example.com")
RATE_LIMIT = 3.0  # req/s -- see module note above; shared SEC budget with other agents.


class _RateLimiter:
    """Same token-bucket-style limiter as insider_cluster_buys.py's
    _RateLimiter, duplicated locally rather than imported so this module's
    3 req/s is independent of that module's 8 req/s (see note above)."""

    def __init__(self, rate_per_sec: float):
        self._interval = 1.0 / rate_per_sec
        self._lock = threading.Lock()
        self._next_time = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = max(0.0, self._next_time - now)
            self._next_time = max(now, self._next_time) + self._interval
        if wait > 0:
            time.sleep(wait)


_LIMITER = _RateLimiter(RATE_LIMIT)

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": USER_AGENT, "Accept-Encoding": "gzip"})


def _get(url: str) -> requests.Response:
    """Rate-limited GET with retry-on-5xx backoff, mirroring
    insider_cluster_buys.py's _get. Streams the response (caller reads
    .content or .iter_content) since these are large files."""
    last_exc: requests.HTTPError | None = None
    for attempt in range(5):
        _LIMITER.acquire()
        resp = SESSION.get(url, timeout=120, stream=True)
        if resp.status_code < 500:
            resp.raise_for_status()
            return resp
        try:
            resp.raise_for_status()
        except requests.HTTPError as exc:
            last_exc = exc
        backoff = 2 ** attempt
        log.warning("SEC %d on %s - retrying in %ds (attempt %d/5)",
                    resp.status_code, url, backoff, attempt + 1)
        time.sleep(backoff)
    assert last_exc is not None
    raise last_exc


def normalize_cik(cik) -> str | None:
    """Zero-pad to SEC's 10-digit CIK string. Mirrors tools/issuer_meta.py's
    normalize_cik exactly -- same input shapes show up in the same
    events/research parquets, and sub.txt's cik column arrives as a plain
    (unpadded) integer, so this is applied on the parsing side too."""
    if cik is None or (isinstance(cik, float) and cik != cik):
        return None
    s = str(cik).strip()
    if not s:
        return None
    s = s.split(".")[0]
    if not s.isdigit():
        return None
    return s.zfill(10)


def download_quarter(q: str, *, refresh: bool = False) -> str:
    """Download (or reuse the cached) ZIP for one quarter. Returns the local
    path. Streamed to disk rather than loaded into memory -- files run
    tens to ~150MB."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{q}.zip")
    if not refresh and os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    url = BULK_URL.format(q=q)
    log.info("fundamentals: downloading %s", url)
    resp = _get(url)
    tmp_path = path + ".part"
    with open(tmp_path, "wb") as fh:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            if chunk:
                fh.write(chunk)
    os.replace(tmp_path, path)
    log.info("fundamentals: wrote %s (%.1f MB)", path, os.path.getsize(path) / 1e6)
    return path


# ---------------------------------------------------------------------------
# Tag fallback chains -- see module docstring for the WHY behind each one.
# ---------------------------------------------------------------------------
TAG_CHAINS: dict[str, list[tuple[str, str]]] = {
    "shares_outstanding": [
        ("dei", "EntityCommonStockSharesOutstanding"),
        ("us-gaap", "CommonStockSharesOutstanding"),
        ("us-gaap", "CommonStockSharesIssued"),
    ],
    "revenue": [
        ("us-gaap", "Revenues"),
        ("us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax"),
        ("us-gaap", "SalesRevenueNet"),
    ],
    "cash": [("us-gaap", "CashAndCashEquivalentsAtCarryingValue")],
    "equity": [("us-gaap", "StockholdersEquity")],
    "net_income": [("us-gaap", "NetIncomeLoss")],
    "assets": [("us-gaap", "Assets")],
}

# Instant (point-in-time balance-sheet) facts require qtrs == 0.
# Duration (flow, income-statement) facts require qtrs in 1..4 and get
# annualized (value * 4/qtrs) at write time -- see module docstring.
INSTANT_GROUPS = {"shares_outstanding", "cash", "equity", "assets"}
DURATION_GROUPS = {"revenue", "net_income"}

# Flatten TAG_CHAINS into (namespace, tag) -> (group, priority_rank) so a
# single dict lookup per num.txt row resolves both "is this a tag we want"
# and "which chain position is it" in one step.
_TAG_LOOKUP: dict[tuple[str, str], tuple[str, int]] = {}
for _group, _chain in TAG_CHAINS.items():
    for _rank, (_ns, _tag) in enumerate(_chain):
        _TAG_LOOKUP[(_ns, _tag)] = (_group, _rank)
_ALL_TAG_NAMES = {tag for (_ns, tag) in _TAG_LOOKUP}
# Same info as _TAG_LOOKUP, as a small DataFrame for a vectorized merge
# instead of a per-row Python dict lookup (num.txt chunks are ~1M rows).
_TAG_LOOKUP_DF = pd.DataFrame(
    [(ns_, tag_, grp, rank_) for (ns_, tag_), (grp, rank_) in _TAG_LOOKUP.items()],
    columns=["_ns", "tag", "tag_group", "_rank"],
)

# Forms whose XBRL facts represent the issuer's own periodic financial
# statements -- the only forms worth extracting facts from here. Mirrors
# tools/filing_calendar.py's PERIODIC_FORMS/8-K distinction in spirit
# (duplicated locally, same reasoning as normalize_cik above: this module
# does not import from filing_calendar.py to stay decoupled).
_PERIODIC_FORMS = {"10-K", "10-K/A", "10-Q", "10-Q/A"}

NEW_FEATURE_COLS = [
    "x_market_cap_log",
    "x_revenue_log",
    "x_cash_to_assets",
    "x_equity_to_assets",
    "x_net_margin",
    "x_cash_runway_quarters",
    "x_fundamentals_age_days",
]

# x_cash_runway_quarters ceiling -- see attach_fundamentals_features.
_RUNWAY_CEILING_QUARTERS = 40.0
# Divide-by-near-zero guard for cash burn.
_BURN_EPSILON = 1e3  # dollars/quarter; small relative to any real burn rate


def _parse_quarter_zip(zip_path: str) -> pd.DataFrame:
    """Parse one quarter's ZIP into resolved facts:
    [cik, adsh, filed, period_end, tag_group, qtrs, value].

    One row per (adsh, tag_group): the fallback chain has already been
    applied (see _resolve_group_facts) so there is at most one value per
    filing per concept.
    """
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open("sub.txt") as f:
            sub = pd.read_csv(
                f, sep="\t",
                usecols=["adsh", "cik", "form", "filed", "period"],
                dtype=str,
            )
        sub = sub[sub["form"].isin(_PERIODIC_FORMS)]
        sub = sub.dropna(subset=["adsh", "cik", "filed", "period"])
        sub_by_adsh = sub.set_index("adsh")[["cik", "filed", "period"]]
        wanted_adsh = set(sub_by_adsh.index)

        kept_chunks: list[pd.DataFrame] = []
        with zf.open("num.txt") as f:
            reader = pd.read_csv(
                f, sep="\t",
                usecols=["adsh", "tag", "version", "ddate", "qtrs", "uom",
                         "segments", "coreg", "value"],
                dtype=str,
                chunksize=1_000_000,
            )
            for chunk in reader:
                # Cheapest filters first to shrink the frame fast.
                chunk = chunk[chunk["adsh"].isin(wanted_adsh)]
                if chunk.empty:
                    continue
                chunk = chunk[chunk["tag"].isin(_ALL_TAG_NAMES)]
                if chunk.empty:
                    continue
                # Consolidated, non-dimensional facts only (see docstring).
                chunk = chunk[chunk["segments"].isna() & chunk["coreg"].isna()]
                if chunk.empty:
                    continue
                ns = chunk["version"].str.split("/", n=1).str[0]
                chunk = chunk.assign(_ns=ns)
                # Resolve (namespace, tag) -> (group, priority_rank); rows
                # whose namespace/tag combo isn't one we want (e.g. an
                # ifrs-namespace "Assets") are dropped via the merge.
                chunk = chunk.merge(_TAG_LOOKUP_DF, on=["_ns", "tag"], how="inner")
                if chunk.empty:
                    continue
                chunk = chunk.join(sub_by_adsh, on="adsh")
                # Current period only, not a comparative prior period.
                chunk = chunk[chunk["ddate"] == chunk["period"]]
                if chunk.empty:
                    continue
                chunk["value"] = pd.to_numeric(chunk["value"], errors="coerce")
                chunk = chunk.dropna(subset=["value"])
                chunk["qtrs"] = pd.to_numeric(chunk["qtrs"], errors="coerce")
                is_instant = chunk["tag_group"].isin(INSTANT_GROUPS) & (chunk["qtrs"] == 0)
                is_duration = chunk["tag_group"].isin(DURATION_GROUPS) & chunk["qtrs"].between(1, 4)
                chunk = chunk[is_instant | is_duration]
                if chunk.empty:
                    continue
                kept_chunks.append(
                    chunk[["cik", "adsh", "filed", "period", "tag_group", "_rank", "qtrs", "value"]]
                )

    if not kept_chunks:
        return pd.DataFrame(columns=["cik", "adsh", "filed", "period_end", "tag_group", "qtrs", "value"])

    facts = pd.concat(kept_chunks, ignore_index=True)
    facts = facts.rename(columns={"period": "period_end"})
    facts = _resolve_group_facts(facts)
    return facts


def _resolve_group_facts(facts: pd.DataFrame) -> pd.DataFrame:
    """Collapse possibly-multiple candidate rows per (adsh, tag_group) down
    to exactly one, applying the fallback-chain priority and the
    duration-tag tie-break documented in the module docstring.

    - Fallback priority: within one (adsh, tag_group), keep only rows at
      the lowest `_rank` present (rank 0 = first/most-preferred tag in the
      chain). A filing that tags BOTH `Revenues` and
      `RevenueFromContractWithCustomerExcludingAssessedTax` uses `Revenues`
      (rank 0), never both.
    - Duration tie-break: if more than one qtrs span survives at that rank
      (e.g. a Q3 10-Q reporting both the 3-month and 9-month duration for
      the same tag), keep the LARGEST qtrs (<=4) -- it is based on more
      data and the *4/qtrs annualization is unaffected either way (see
      docstring), so more data is strictly better here.
    - Instant tag_groups have qtrs==0 by construction (filtered upstream),
      so this tie-break is a no-op for them; any remaining true duplicate
      (same adsh/tag_group/qtrs, different value -- a rare data anomaly) is
      broken by keeping the largest value, deterministically.
    """
    if facts.empty:
        return facts.drop(columns=["_rank"], errors="ignore")

    best_rank = facts.groupby(["adsh", "tag_group"])["_rank"].transform("min")
    facts = facts[facts["_rank"] == best_rank]

    facts = facts.sort_values(["adsh", "tag_group", "qtrs", "value"], ascending=[True, True, False, False])
    facts = facts.drop_duplicates(subset=["adsh", "tag_group"], keep="first")

    return facts.drop(columns=["_rank"]).reset_index(drop=True)


def _parsed_cache_path(q: str) -> str:
    os.makedirs(PARSED_CACHE_DIR, exist_ok=True)
    return os.path.join(PARSED_CACHE_DIR, f"{q}.parquet")


def parse_quarter(q: str, *, refresh: bool = False) -> pd.DataFrame:
    """Parsed, filtered facts for one quarter, cached to
    xbrl_cache/parsed/{q}.parquet so a rerun does not re-walk num.txt."""
    cache_path = _parsed_cache_path(q)
    if not refresh and os.path.exists(cache_path):
        return pd.read_parquet(cache_path)
    zip_path = download_quarter(q, refresh=refresh)
    facts = _parse_quarter_zip(zip_path)
    facts.to_parquet(cache_path, index=False)
    log.info("fundamentals: parsed %s -> %d fact(s)", q, len(facts))
    return facts


def build_fundamentals(quarters: list[str] | None = None, *, refresh: bool = False) -> pd.DataFrame:
    """Download+parse every quarter and assemble the final long-format
    table: cik, filed_date, period_end, tag_group, value.

    `value` for tag_group in DURATION_GROUPS (revenue, net_income) is the
    ANNUALIZED figure (value * 4/qtrs) -- see module docstring. For the
    instant tag_groups it is the raw reported value.

    Sorted by cik, tag_group, filed_date so a point-in-time lookup for one
    issuer/tag_group is a bisect (numpy.searchsorted) on filed_date, exactly
    like tools/earnings_features.py's per-issuer filing-date arrays.
    """
    quarters = quarters if quarters is not None else QUARTERS
    all_facts = []
    for q in quarters:
        f = parse_quarter(q, refresh=refresh)
        if not f.empty:
            all_facts.append(f)

    if not all_facts:
        return pd.DataFrame(columns=["cik", "filed_date", "period_end", "tag_group", "value"])

    facts = pd.concat(all_facts, ignore_index=True)

    is_duration = facts["tag_group"].isin(DURATION_GROUPS)
    facts.loc[is_duration, "value"] = facts.loc[is_duration, "value"] * (4.0 / facts.loc[is_duration, "qtrs"])

    facts["cik"] = facts["cik"].map(normalize_cik)
    facts = facts.dropna(subset=["cik"])
    facts["filed_date"] = pd.to_datetime(facts["filed"], format="%Y%m%d")
    facts["period_end"] = pd.to_datetime(facts["period_end"], format="%Y%m%d")

    out = facts[["cik", "filed_date", "period_end", "tag_group", "value"]].copy()
    # A given (cik, tag_group, filed_date) could in principle appear twice
    # if the same issuer refiled on the same calendar day (e.g. an original
    # + a same-day amendment both surviving the periodic-form filter); keep
    # the larger value deterministically, mirroring _resolve_group_facts.
    out = out.sort_values(["cik", "tag_group", "filed_date", "value"], ascending=[True, True, True, False])
    out = out.drop_duplicates(subset=["cik", "tag_group", "filed_date"], keep="first")
    out = out.sort_values(["cik", "tag_group", "filed_date"]).reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Point-in-time feature attachment
# ---------------------------------------------------------------------------
def _issuer_tag_index(fundamentals: pd.DataFrame) -> dict:
    """Per (cik, tag_group), ascending-sorted (filed_date ordinal, value)
    arrays for O(log n) point-in-time lookup via numpy.searchsorted -- same
    approach as tools/earnings_features.py's _issuer_filing_index."""
    idx: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    if fundamentals.empty:
        return idx
    fd = fundamentals.copy()
    fd["_ord"] = fd["filed_date"].map(lambda ts: ts.toordinal() if pd.notna(ts) else np.nan)
    fd = fd.dropna(subset=["_ord"])
    for (cik, grp), g in fd.groupby(["cik", "tag_group"]):
        order = np.argsort(g["_ord"].to_numpy())
        idx[(cik, grp)] = {
            "ord": g["_ord"].to_numpy()[order].astype(np.int64),
            "value": g["value"].to_numpy()[order].astype(np.float64),
        }
    return idx


def _to_day_ordinal(series: pd.Series) -> np.ndarray:
    dt = pd.to_datetime(series)
    return dt.map(lambda ts: ts.toordinal() if pd.notna(ts) else np.nan).to_numpy()


def _most_recent_prior(entry: dict | None, event_ord: float) -> tuple[float, float]:
    """(value, filed_ord) of the most recent fact STRICTLY BEFORE
    event_ord, or (nan, nan) if none exists. side='left' puts a same-day
    filing on the "not yet known" side -- see module docstring."""
    if entry is None or event_ord != event_ord:
        return float("nan"), float("nan")
    ords = entry["ord"]
    boundary = np.searchsorted(ords, event_ord, side="left")
    if boundary == 0:
        return float("nan"), float("nan")
    return float(entry["value"][boundary - 1]), float(ords[boundary - 1])


def attach_fundamentals_features(events_df: pd.DataFrame, fundamentals_df: pd.DataFrame) -> pd.DataFrame:
    """Compute the point-in-time x_ fundamentals columns.

    Parameters
    ----------
    events_df : must have `issuer_cik` and `event_day`. `entry_open`, if
        present, is used for x_market_cap_log; if ABSENT, x_market_cap_log
        is skipped entirely (not added as an all-NaN column) and a warning
        is logged -- per the "say so" requirement, silently emitting an
        all-NaN column would look like a computed-and-empty feature rather
        than a feature that could not be computed at all.
    fundamentals_df : the long-format table build_fundamentals() produces
        (cik, filed_date, period_end, tag_group, value).

    Returns
    -------
    A DataFrame with the applicable subset of NEW_FEATURE_COLS, same length
    and index as events_df, safe to `pd.concat([events_df, result], axis=1)`.

    POINT-IN-TIME CONTRACT -- read this before touching this function. For
    each event, a tag_group's value is taken from the most recent filing
    whose filed_date is STRICTLY BEFORE event_day (numpy.searchsorted,
    side='left', identical mechanism to tools/earnings_features.py). A
    filing filed ON OR AFTER event_day must have ZERO influence on any
    output value for that event. This is enforced by construction here
    (every lookup only ever walks backward from the searchsorted boundary),
    and tests/test_fundamentals.py's point-in-time test verifies it by
    proving that deleting all on/after-event_day filings from
    fundamentals_df does not change a single computed value.

    NOTE on net_margin period alignment. `revenue` and `net_income` are each
    looked up independently -- the most recent revenue filing and the most
    recent net-income filing for a given event are USUALLY the same filing
    (both duration facts normally get refiled together in one accession)
    but are not guaranteed to be. Requiring them to come from the identical
    filing would only shrink coverage for no real accuracy gain in the
    common case, so this function does not enforce it; x_net_margin can, in
    the rare misaligned case, mix figures from two different accessions for
    the same issuer.
    """
    n = len(events_df)
    have_entry_open = "entry_open" in events_df.columns
    cols = list(NEW_FEATURE_COLS)
    if not have_entry_open:
        log.warning("fundamentals: events_df has no entry_open column -- "
                    "skipping x_market_cap_log entirely (cannot compute market cap without a price).")
        cols = [c for c in cols if c != "x_market_cap_log"]

    out = {c: np.full(n, np.nan, dtype=np.float64) for c in cols}

    fidx = _issuer_tag_index(fundamentals_df)
    event_ciks = events_df["issuer_cik"].astype(str).to_numpy()
    event_ords = _to_day_ordinal(events_df["event_day"])
    entry_open = events_df["entry_open"].to_numpy(dtype=np.float64) if have_entry_open else None

    for i in range(n):
        cik = event_ciks[i]
        eo = event_ords[i]
        if eo != eo:
            continue

        shares, shares_ord = _most_recent_prior(fidx.get((cik, "shares_outstanding")), eo)
        revenue, revenue_ord = _most_recent_prior(fidx.get((cik, "revenue")), eo)
        cash, cash_ord = _most_recent_prior(fidx.get((cik, "cash")), eo)
        equity, equity_ord = _most_recent_prior(fidx.get((cik, "equity")), eo)
        net_income, ni_ord = _most_recent_prior(fidx.get((cik, "net_income")), eo)
        assets, assets_ord = _most_recent_prior(fidx.get((cik, "assets")), eo)

        if have_entry_open:
            px = entry_open[i]
            if shares == shares and px == px and shares > 0 and px > 0:
                out["x_market_cap_log"][i] = float(np.log(shares * px))

        if revenue == revenue and revenue > -1.0:
            out["x_revenue_log"][i] = float(np.log1p(revenue))

        if cash == cash and assets == assets and assets != 0:
            out["x_cash_to_assets"][i] = cash / assets
        if equity == equity and assets == assets and assets != 0:
            out["x_equity_to_assets"][i] = equity / assets
        if net_income == net_income and revenue == revenue and revenue != 0:
            out["x_net_margin"][i] = net_income / revenue

        # Cash runway: cash / quarterly burn, quarters. `net_income` here is
        # already ANNUALIZED (build_fundamentals's contract); de-annualize
        # back to a single quarter to get a burn rate. Only defined for a
        # company that is actually burning cash (quarterly net income < 0)
        # -- a profitable company has no "runway" to report, per the task
        # spec, and reporting one would misleadingly suggest it could run
        # out. Capped at _RUNWAY_CEILING_QUARTERS: near-zero burn makes the
        # ratio blow up (division by an epsilon-guarded near-zero), and past
        # ~10 years the number is not meaningfully informative -- the point
        # is "distinguish weeks-of-cash from years-of-cash", not to model an
        # a precise multi-decade horizon.
        if cash == cash and net_income == net_income:
            quarterly_net_income = net_income / 4.0
            if quarterly_net_income < 0:
                burn = max(-quarterly_net_income, _BURN_EPSILON)
                out["x_cash_runway_quarters"][i] = min(cash / burn, _RUNWAY_CEILING_QUARTERS)
            # else: profitable or breakeven -> stays NaN, per spec.

        used_ords = [o for o in (shares_ord, revenue_ord, cash_ord, equity_ord, ni_ord, assets_ord) if o == o]
        if used_ords:
            newest = max(used_ords)
            out["x_fundamentals_age_days"][i] = eo - newest

    return pd.DataFrame(out, index=events_df.index)[cols]


def attach_features(df: pd.DataFrame, fundamentals_path: str = DEFAULT_FUNDAMENTALS_OUT) -> pd.DataFrame:
    """Load research_data/fundamentals.parquet (or `fundamentals_path`) and
    attach the point-in-time x_ fundamentals columns to a COPY of `df`.
    See attach_fundamentals_features for the point-in-time contract.
    """
    fundamentals = pd.read_parquet(fundamentals_path)
    feats = attach_fundamentals_features(df, fundamentals)
    out = df.copy()
    for c in feats.columns:
        out[c] = feats[c]
    return out


# ---------------------------------------------------------------------------
# CLI: build fundamentals.parquet, attach to the research dataset, report.
# ---------------------------------------------------------------------------
def _print_report(df: pd.DataFrame, present_cols: list[str]) -> None:
    n = len(df)
    print(f"\n=== fundamentals sanity report ({n} rows) ===")
    print("Coverage (fraction of rows with a non-null value):")
    for c in NEW_FEATURE_COLS:
        if c not in present_cols:
            print(f"  {c}: SKIPPED (entry_open not in events)")
            continue
        frac = df[c].notna().mean()
        print(f"  {c}: {frac:.1%}")

    if "x_market_cap_log" in present_cols:
        mc = df["x_market_cap_log"].dropna()
        if len(mc):
            mcap_dollars = np.exp(mc)
            q = mcap_dollars.quantile([0.25, 0.5, 0.75])
            print(f"\nMarket cap ($ implied by entry_open * shares_outstanding, n={len(mc)}):")
            print(f"  25th pct: ${q.loc[0.25]:,.0f}   median: ${q.loc[0.5]:,.0f}   75th pct: ${q.loc[0.75]:,.0f}")

            if "adj_21" in df.columns:
                valid = df.loc[mc.index]
                try:
                    q_labels = pd.qcut(valid["x_market_cap_log"], 5, duplicates="drop")
                except ValueError:
                    q_labels = None
                print("\nMEDIAN adj_21 by quintile of x_market_cap_log (medians, not means -- "
                      "adj_21 has a large right tail):")
                if q_labels is not None:
                    print(valid.groupby(q_labels, observed=True)["adj_21"].median().to_string())
                else:
                    print("  (could not form 5 distinct quintiles)")

    if "x_cash_runway_quarters" in df.columns:
        rq = df["x_cash_runway_quarters"].dropna()
        if len(rq) and "adj_21" in df.columns:
            valid = df.loc[rq.index]
            try:
                q_labels = pd.qcut(valid["x_cash_runway_quarters"], 4, duplicates="drop")
            except ValueError:
                q_labels = None
            print("\nMEDIAN adj_21 by quartile of x_cash_runway_quarters (medians, not means):")
            if q_labels is not None:
                print(valid.groupby(q_labels, observed=True)["adj_21"].median().to_string())
            else:
                print("  (could not form 4 distinct quartiles)")
    print()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--events", default=DEFAULT_EVENTS, help="Research parquet to attach features to.")
    p.add_argument("--out", default=DEFAULT_FUNDAMENTALS_OUT, help="Output path for fundamentals.parquet.")
    p.add_argument("--refresh", action="store_true", help="Ignore ZIP and parsed caches, refetch/reparse everything.")
    p.add_argument("--skip-build", action="store_true", help="Reuse an existing fundamentals.parquet at --out.")
    p.add_argument("--verbose", action="store_true")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                         format="%(asctime)s [%(levelname)s] %(message)s")

    t0 = time.time()
    if a.skip_build and os.path.exists(a.out):
        log.info("fundamentals: --skip-build, reading existing %s", a.out)
        fundamentals = pd.read_parquet(a.out)
    else:
        log.info("fundamentals: building from %d quarter(s) %s..%s", len(QUARTERS), QUARTERS[0], QUARTERS[-1])
        fundamentals = build_fundamentals(refresh=a.refresh)
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        fundamentals.to_parquet(a.out, index=False)
        log.info("fundamentals: wrote %s (%d rows)", a.out, len(fundamentals))
    elapsed = time.time() - t0
    total_zip_bytes = sum(
        os.path.getsize(os.path.join(CACHE_DIR, f"{q}.zip"))
        for q in QUARTERS if os.path.exists(os.path.join(CACHE_DIR, f"{q}.zip"))
    )
    log.info("fundamentals: build wall time %.1fs, total cached ZIP size %.1f MB",
             elapsed, total_zip_bytes / 1e6)

    log.info("fundamentals: reading events from %s", a.events)
    events = pd.read_parquet(a.events)
    out_df = attach_features(events, fundamentals_path=a.out)
    present_cols = [c for c in NEW_FEATURE_COLS if c in out_df.columns]

    _print_report(out_df, present_cols)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
