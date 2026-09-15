"""Event-context features from two SEC filing sources, keyed on issuer_cik.

WHY this exists. Every insider cluster in the research dataset is currently
treated as the same kind of event regardless of what else was happening at
the issuer around the same time. Two sources fill in context that plausibly
matters:

  (A) 13D / 13G BENEFICIAL OWNERSHIP FILINGS. When a 5%+ holder or an
      activist files SC 13D/SC 13D-A (active, "I intend to influence
      control") or SC 13G/SC 13G-A (passive, "I just crossed 5%") close to
      an insider cluster, that is independent confirmation from someone
      with real money at risk -- a different signal than eight insiders
      trading on the same information the market already has.
  (B) 8-K ITEM CODES. An insider cluster that follows an Item 5.02
      (officer/director departure) is a very different situation from one
      that follows an Item 2.02 (results) or Item 1.01 (a material
      agreement), and today's dataset cannot tell them apart at all.

WHERE the data comes from. Both sources live in the same place:
https://data.sec.gov/submissions/CIK##########.json, keyed on the ISSUER's
CIK -- the exact identifier tools/filing_calendar.py and tools/issuer_meta.py
already fetch. SEC's submissions endpoint lists a filing under an entity's
CIK whenever that entity plays ANY role in it, not just when it's the
filer -- for SC 13D/13G, the filer is the beneficial owner but the SUBJECT
COMPANY is the issuer, and the issuer's own submissions.json carries the
filing anyway (verified directly: CIK0001326380's -- GameStop's --
submissions.json lists dozens of SC 13D/A and SC 13G/A entries filed by
Rand-organized activists and Fidelity, not by GameStop). So one fetch per
issuer, already the unit tools/filing_calendar.py fetches on, covers both
sources.

WHY NOT REUSE filing_cache/. tools/filing_calendar.py already caches this
exact endpoint per issuer_cik under filing_cache/, but its cache keeps only
{form, filing_date, report_date} for 10-Q/10-K/8-K -- it discards SC 13D/13G
entirely (not in its form allowlist) and never reads the `items` field it
otherwise passes over. That cache cannot answer either source without a
fresh fetch, so this module keeps its own cache (filing_context_cache/,
mirrors filing_cache/'s and issuer_meta_cache/'s per-CIK-JSON layout) rather
than mutate filing_calendar.py's.

WAS A DOCUMENT FETCH NEEDED FOR (B)? No. `filings.recent.items` in the raw
submissions payload IS populated -- confirmed directly against SEC (e.g.
CIK0000001800's 2026-04-27 8-K carries items "5.02,5.03,5.07,9.01") -- and
so are the paginated older `filings.files` pages this module follows for
issuers whose `recent` block doesn't reach back to 2018. Every 8-K's item
codes come straight out of the submissions JSON already being fetched for
source (A); no per-filing document or index fetch was needed.

POINT-IN-TIME, the whole ballgame. Every x_ feature here uses ONLY filings
whose filing_date is STRICTLY BEFORE the event's event_day. A same-day
filing is treated as NOT yet known (SEC filings can post intraday after an
insider cluster has already happened) -- enforced the same way
tools/earnings_features.py enforces it: per-issuer ascending-sorted filing
date arrays, `numpy.searchsorted(..., event_day, side="left")` to find the
first index that is NOT strictly earlier, and only entries before that
index are ever read. A feature that let a same-day or future filing leak in
would manufacture a fake edge -- the model would look prescient in backtest
by "predicting" filings that, at decision time, had not happened yet.

NaN vs 0, on purpose. A trailing count feature (e.g. x_n_13d_trail180) is
0.0 when the issuer IS covered by this module's fetch and genuinely has no
such filing in the window -- that is a real fact. It is NaN when the issuer
has no coverage at all (never fetched, or SEC 404'd the CIK) -- that is
missing data, and must not be silently treated as "confirmed zero". The
`covered_ciks` set threaded through `compute_filing_context_features`
carries this distinction; see `_issuer_index`.

8-K "other" bucket. An 8-K can carry more than one item code, so a single
8-K counts toward MULTIPLE buckets when it matches multiple tracked codes
(e.g. items "5.02,2.02" counts in both x_8k_departure_trail90 and
x_8k_results_trail90). x_8k_other_trail90 counts 8-Ks that match NONE of
the three tracked codes (5.02 / 2.02 / 1.01) -- including 8-Ks whose
`items` field is empty (rare, mostly pre-2004 filings predating the item
disclosure requirement), which are unclassifiable and so fall into "other"
rather than being dropped.

Runnable standalone as `python tools/filing_context.py` (fetches/caches per
issuer, reads the research parquet, writes research_data/filing_context.parquet)
or importable, e.g. from a test, via `compute_filing_context_features`
(pure, no I/O) or `attach_features` (fetch/cache + compute).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools import data_paths  # noqa: E402

import insider_cluster_buys as ics  # noqa: E402  (reuses its session/User-Agent only)

log = logging.getLogger("filing_context")

CACHE_DIR = os.path.join(REPO_ROOT, "filing_context_cache")
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
FILES_URL = "https://data.sec.gov/submissions/{name}"
DEFAULT_EVENTS = data_paths.latest_research_dataset()
DEFAULT_OUT = os.path.join(REPO_ROOT, "research_data", "filing_context.parquet")

# ---------------------------------------------------------------------------
# Our own rate limit, deliberately independent of insider_cluster_buys.py's
# RATE_LIMIT (8 req/s). Two other agents are hitting SEC concurrently on
# this same task; this module caps ITSELF at 3 req/s regardless of what
# ics.RATE_LIMIT is configured to, so total load stays polite even though
# there is no cross-process coordination possible.
# ---------------------------------------------------------------------------
OWN_RATE_LIMIT = 3.0
MAX_WORKERS = 4


class _RateLimiter:
    """Same token-bucket shape as insider_cluster_buys._RateLimiter, kept as
    a private copy here so this module's SEC request rate is provably its
    own and not entangled with any other module's limiter instance."""

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


_LIMITER = _RateLimiter(OWN_RATE_LIMIT)


def _get(url: str):
    """Rate-limited GET, reusing insider_cluster_buys.SESSION for the
    User-Agent/connection pool but this module's own (slower) limiter."""
    last_exc = None
    for attempt in range(5):
        _LIMITER.acquire()
        resp = ics.SESSION.get(url, headers={"Accept": "application/json"}, timeout=30)
        if resp.status_code < 500:
            resp.raise_for_status()
            return resp
        try:
            resp.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
        backoff = 2 ** attempt
        log.warning("SEC %d on %s - retrying in %ds (attempt %d/5)",
                    resp.status_code, url, backoff, attempt + 1)
        time.sleep(backoff)
    assert last_exc is not None
    raise last_exc


# Forms kept from source (A). Amendments are kept in the same family as
# their base form -- an activist who has since amended a 13D is still an
# active 13D holder for the purposes of "is someone with real money watching
# this issuer right now".
FORM_13D = {"SC 13D", "SC 13D/A"}
FORM_13G = {"SC 13G", "SC 13G/A"}
_OWNERSHIP_FORMS = FORM_13D | FORM_13G

# Forms kept from source (B). 8-K/A (amended) collapses into 8-K, same as
# tools/filing_calendar.py does for its periodic/8-K forms.
_EIGHTK_FORMS = {"8-K", "8-K/A"}

# Item codes this module tracks explicitly; see the module docstring for the
# "other" bucket's semantics.
_ITEM_DEPARTURE = "5.02"
_ITEM_RESULTS = "2.02"
_ITEM_MATERIAL_AGMT = "1.01"

# Our events start 2018-08-13; mirrors tools/filing_calendar.py's pagination
# cutoff exactly so both modules reach the same depth of history for the
# same reason.
_COVERAGE_CUTOFF = date(2018, 6, 1)
_MAX_EXTRA_PAGES = 15

NEW_FEATURE_COLS = [
    "x_days_since_13d",
    "x_days_since_13g",
    "x_n_13d_trail180",
    "x_has_recent_13d",
    "x_8k_departure_trail90",
    "x_8k_results_trail90",
    "x_8k_material_agmt_trail90",
    "x_8k_other_trail90",
]


# ---------------------------------------------------------------------------
# CIK normalization (mirrors tools/issuer_meta.py / tools/filing_calendar.py)
# ---------------------------------------------------------------------------
def normalize_cik(cik) -> str | None:
    if cik is None or (isinstance(cik, float) and cik != cik):
        return None
    s = str(cik).strip()
    if not s:
        return None
    s = s.split(".")[0]  # an int that round-tripped through float
    if not s.isdigit():
        return None
    return s.zfill(10)


def _cache_path(cik10: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{cik10}.json")


# ---------------------------------------------------------------------------
# Fetch + cache
# ---------------------------------------------------------------------------
def _kept_from_block(block: dict) -> tuple[list[dict], list[dict]]:
    """Extract kept ownership and 8-K records from one parallel-array block
    (`filings.recent`, or one bare `filings.files` page)."""
    forms = block.get("form") or []
    filing_dates = block.get("filingDate") or []
    items_list = block.get("items") or []
    n = len(forms)
    ownership: list[dict] = []
    eightk: list[dict] = []
    for i in range(n):
        f = (forms[i] or "").strip()
        fd = filing_dates[i] if i < len(filing_dates) else None
        if not fd:
            continue
        if f in _OWNERSHIP_FORMS:
            ownership.append({"form": f, "filing_date": fd})
        elif f in _EIGHTK_FORMS:
            it = items_list[i] if i < len(items_list) else None
            eightk.append({"filing_date": fd, "items": (it or None)})
    return ownership, eightk


def _earliest_date_in_block(block: dict) -> date | None:
    """Earliest filingDate literally present, across ALL forms -- used only
    to decide whether older pages are worth fetching, same as
    tools/filing_calendar.py's identically-named helper."""
    filing_dates = [d for d in (block.get("filingDate") or []) if d]
    if not filing_dates:
        return None
    try:
        return min(date.fromisoformat(d) for d in filing_dates)
    except ValueError:
        return None


def fetch_one(cik10: str, *, refresh: bool = False) -> dict | None:
    """Fetch (or read from cache) one issuer's ownership + 8-K item history.

    A 404 is cached as an explicit `{"found": False}` record so a rerun does
    not re-request CIKs SEC does not know about.
    """
    path = _cache_path(cik10)
    if not refresh and os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError):
            log.warning("filing_context: unreadable cache %s, refetching", path)

    try:
        resp = _get(SUBMISSIONS_URL.format(cik=cik10))
        payload = resp.json()
    except Exception as exc:  # noqa: BLE001 -- one bad CIK must not kill the run
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status == 404:
            rec = {"cik": cik10, "found": False}
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(rec, fh)
            return rec
        log.warning("filing_context: %s failed (%s)", cik10, exc)
        return None

    try:
        filings_block = payload.get("filings") or {}
        recent = filings_block.get("recent") or {}
        ownership, eightk = _kept_from_block(recent)
        earliest = _earliest_date_in_block(recent)

        files_meta = filings_block.get("files") or []
        pages_fetched = 0
        if (earliest is None or earliest > _COVERAGE_CUTOFF) and files_meta:
            for finfo in files_meta:
                if pages_fetched >= _MAX_EXTRA_PAGES:
                    break
                name = finfo.get("name")
                if not name:
                    continue
                try:
                    presp = _get(FILES_URL.format(name=name))
                    page = presp.json()
                except Exception as exc:  # noqa: BLE001
                    log.warning("filing_context: %s page %s failed (%s)", cik10, name, exc)
                    continue
                pages_fetched += 1
                o2, e2 = _kept_from_block(page)
                ownership.extend(o2)
                eightk.extend(e2)
                page_earliest = _earliest_date_in_block(page)
                if page_earliest is not None and (earliest is None or page_earliest < earliest):
                    earliest = page_earliest
                if page_earliest is not None and page_earliest <= _COVERAGE_CUTOFF:
                    break  # this page already reaches back far enough

        rec = {
            "cik": cik10,
            "found": True,
            "earliest_filing_date": earliest.isoformat() if earliest else None,
            "ownership": ownership,
            "eightk": eightk,
        }
    except Exception as exc:  # noqa: BLE001 -- malformed payload must not kill the run
        log.warning("filing_context: %s unexpected payload shape (%s)", cik10, exc)
        return None

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rec, fh)
    return rec


def fetch_many(ciks, *, refresh: bool = False, workers: int = MAX_WORKERS) -> list[dict]:
    norm = []
    seen = set()
    for c in ciks:
        n = normalize_cik(c)
        if n and n not in seen:
            seen.add(n)
            norm.append(n)

    cached = [c for c in norm if os.path.exists(_cache_path(c))] if not refresh else []
    todo = [c for c in norm if c not in set(cached)]
    log.info("filing_context: %d CIK(s) -- %d cached, %d to fetch @ %.0f req/s",
              len(norm), len(cached), len(todo), OWN_RATE_LIMIT)

    records: list[dict] = []
    for c in cached:
        r = fetch_one(c)
        if r:
            records.append(r)

    if todo:
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(fetch_one, c, refresh=refresh): c for c in todo}
            for fut in as_completed(futs):
                r = fut.result()
                if r:
                    records.append(r)
                done += 1
                if done % 250 == 0:
                    log.info("filing_context: %d/%d fetched", done, len(todo))

    return records


def load_cached() -> list[dict]:
    """Everything already on disk, without touching the network."""
    if not os.path.isdir(CACHE_DIR):
        return []
    recs = []
    for name in sorted(os.listdir(CACHE_DIR)):
        if not name.endswith(".json"):
            continue
        try:
            with open(os.path.join(CACHE_DIR, name), encoding="utf-8") as fh:
                recs.append(json.load(fh))
        except (json.JSONDecodeError, OSError):
            continue
    return recs


def records_to_frames(records: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame, set[str]]:
    """Flatten cached per-issuer records into (ownership_df, eightk_df,
    covered_ciks). `covered_ciks` is every CIK SEC actually returned data
    for (found=True) -- including issuers with zero ownership/8-K filings,
    which is why it must be tracked separately from "which CIKs appear in
    the two frames" (see the module docstring's NaN-vs-0 note)."""
    own_rows = []
    ek_rows = []
    covered: set[str] = set()
    for rec in records:
        if not rec:
            continue
        cik = rec["cik"]
        if not rec.get("found"):
            continue
        covered.add(cik)
        for f in rec.get("ownership") or []:
            own_rows.append({"cik": cik, "form": f["form"], "filing_date": f["filing_date"]})
        for f in rec.get("eightk") or []:
            ek_rows.append({"cik": cik, "filing_date": f["filing_date"], "items": f.get("items")})

    own_df = pd.DataFrame(own_rows, columns=["cik", "form", "filing_date"])
    ek_df = pd.DataFrame(ek_rows, columns=["cik", "filing_date", "items"])
    if not own_df.empty:
        own_df["filing_date"] = pd.to_datetime(own_df["filing_date"])
    if not ek_df.empty:
        ek_df["filing_date"] = pd.to_datetime(ek_df["filing_date"])
    return own_df, ek_df, covered


# ---------------------------------------------------------------------------
# Point-in-time feature computation (pure -- no I/O, fully testable)
# ---------------------------------------------------------------------------
def _to_day_ordinal(series: pd.Series) -> np.ndarray:
    """Convert dates/timestamps/python `date` objects to int64 day ordinals
    so all date arithmetic here is done on plain integers. Mirrors
    tools/earnings_features.py's identically-named helper: event_day in the
    research parquet is `datetime.date`, filing_date read back from a
    fetched/cached frame is a pandas Timestamp -- both round-trip cleanly
    through `.toordinal()`."""
    dt = pd.to_datetime(series)
    return dt.map(lambda ts: ts.toordinal() if pd.notna(ts) else np.nan).to_numpy()


def _prior_slice(sorted_ord: np.ndarray, event_ord: float) -> np.ndarray:
    """All entries in `sorted_ord` (ascending) strictly before `event_ord`.
    `side="left"` puts a same-day filing on the "not yet known" side -- see
    the module docstring's point-in-time section."""
    if sorted_ord.size == 0 or event_ord != event_ord:  # NaN check
        return sorted_ord[:0]
    boundary = np.searchsorted(sorted_ord, event_ord, side="left")
    return sorted_ord[:boundary]


def _parse_item_codes(items) -> set[str]:
    if not items:
        return set()
    return {tok.strip() for tok in str(items).split(",") if tok.strip()}


def _default_issuer_entry() -> dict:
    return {
        "13d_ord": np.empty(0, dtype=np.int64),
        "13g_ord": np.empty(0, dtype=np.int64),
        "8k_ord": np.empty(0, dtype=np.int64),
        "8k_dep": np.empty(0, dtype=bool),
        "8k_res": np.empty(0, dtype=bool),
        "8k_mat": np.empty(0, dtype=bool),
        "8k_other": np.empty(0, dtype=bool),
    }


def _issuer_index(ownership_df: pd.DataFrame, eightk_df: pd.DataFrame,
                   covered_ciks: set[str] | None) -> dict:
    """Per-issuer ascending-sorted arrays for O(log n) point-in-time lookups.

    If `covered_ciks` is given, every CIK in it gets an entry (all-empty if
    it has no ownership/8-K rows) -- an issuer confirmed to have zero such
    filings is a real 0, not a missing value. A CIK not in `covered_ciks`
    (or not passed at all, and not present in either frame) has no entry,
    and the caller leaves its features as NaN. If `covered_ciks` is None
    (e.g. a test wiring frames directly), coverage is inferred from
    whichever CIKs appear in either frame -- a weaker fallback that cannot
    distinguish "confirmed zero" from "unknown", documented here so a
    caller does not assume otherwise.
    """
    idx: dict[str, dict] = {}
    if covered_ciks is not None:
        for cik in covered_ciks:
            idx[cik] = _default_issuer_entry()

    if not ownership_df.empty:
        own = ownership_df.copy()
        own["ord"] = _to_day_ordinal(own["filing_date"])
        own = own.dropna(subset=["ord"])
        own["ord"] = own["ord"].astype(np.int64)
        is_13d = own["form"].isin(FORM_13D)
        is_13g = own["form"].isin(FORM_13G)
        for cik, grp in own[is_13d].groupby("cik"):
            idx.setdefault(cik, _default_issuer_entry())["13d_ord"] = np.sort(grp["ord"].to_numpy())
        for cik, grp in own[is_13g].groupby("cik"):
            idx.setdefault(cik, _default_issuer_entry())["13g_ord"] = np.sort(grp["ord"].to_numpy())

    if not eightk_df.empty:
        ek = eightk_df.copy()
        ek["ord"] = _to_day_ordinal(ek["filing_date"])
        ek = ek.dropna(subset=["ord"])
        ek["ord"] = ek["ord"].astype(np.int64)
        codes = ek["items"].map(_parse_item_codes)
        ek["dep"] = codes.map(lambda c: _ITEM_DEPARTURE in c)
        ek["res"] = codes.map(lambda c: _ITEM_RESULTS in c)
        ek["mat"] = codes.map(lambda c: _ITEM_MATERIAL_AGMT in c)
        ek["other"] = ~(ek["dep"] | ek["res"] | ek["mat"])
        for cik, grp in ek.groupby("cik"):
            grp = grp.sort_values("ord")
            entry = idx.setdefault(cik, _default_issuer_entry())
            entry["8k_ord"] = grp["ord"].to_numpy()
            entry["8k_dep"] = grp["dep"].to_numpy()
            entry["8k_res"] = grp["res"].to_numpy()
            entry["8k_mat"] = grp["mat"].to_numpy()
            entry["8k_other"] = grp["other"].to_numpy()

    return idx


def compute_filing_context_features(events_df: pd.DataFrame, ownership_df: pd.DataFrame,
                                     eightk_df: pd.DataFrame,
                                     covered_ciks: set[str] | None = None) -> pd.DataFrame:
    """Compute the point-in-time x_ feature columns.

    Parameters
    ----------
    events_df : must have `issuer_cik` and `event_day`. Row order/index
        preserved in the output.
    ownership_df : `cik`, `form` (one of FORM_13D | FORM_13G), `filing_date`.
    eightk_df : `cik`, `filing_date`, `items` (comma-separated string or
        None/empty).
    covered_ciks : see `_issuer_index` -- pass the real coverage set from
        `records_to_frames` in production; may be omitted in tests that only
        care about issuers known to have data.

    Returns
    -------
    A DataFrame with exactly `NEW_FEATURE_COLS`, same length/index as
    events_df, safe to `pd.concat([events_df, result], axis=1)`.
    """
    n = len(events_df)
    out = {c: np.full(n, np.nan, dtype=np.float64) for c in NEW_FEATURE_COLS}

    idx = _issuer_index(ownership_df, eightk_df, covered_ciks)
    event_ciks = events_df["issuer_cik"].astype(str).to_numpy()
    event_ords = _to_day_ordinal(events_df["event_day"])

    for i in range(n):
        cik = event_ciks[i]
        eo = event_ords[i]
        entry = idx.get(cik)
        if entry is None or eo != eo:  # no coverage for this issuer, or NaT event_day
            continue

        prior_13d = _prior_slice(entry["13d_ord"], eo)
        if prior_13d.size:
            out["x_days_since_13d"][i] = eo - prior_13d[-1]
        out["x_n_13d_trail180"][i] = float(np.sum(prior_13d >= (eo - 180)))
        out["x_has_recent_13d"][i] = 1.0 if np.any(prior_13d >= (eo - 90)) else 0.0

        prior_13g = _prior_slice(entry["13g_ord"], eo)
        if prior_13g.size:
            out["x_days_since_13g"][i] = eo - prior_13g[-1]

        ord8k = entry["8k_ord"]
        boundary = np.searchsorted(ord8k, eo, side="left") if ord8k.size else 0
        sub_ord = ord8k[:boundary]
        lo = eo - 90
        within = sub_ord >= lo
        out["x_8k_departure_trail90"][i] = float(np.sum(entry["8k_dep"][:boundary][within]))
        out["x_8k_results_trail90"][i] = float(np.sum(entry["8k_res"][:boundary][within]))
        out["x_8k_material_agmt_trail90"][i] = float(np.sum(entry["8k_mat"][:boundary][within]))
        out["x_8k_other_trail90"][i] = float(np.sum(entry["8k_other"][:boundary][within]))

    return pd.DataFrame(out, index=events_df.index)[NEW_FEATURE_COLS]


def attach_features(df: pd.DataFrame, *, refresh: bool = False,
                     fetch_missing: bool = True) -> pd.DataFrame:
    """Fetch/cache (or load from disk-only cache) ownership + 8-K data for
    every issuer_cik in `df`, then return a COPY of df with x_ columns
    attached. `fetch_missing=False` uses only what's already cached (no
    network), matching `load_cached`."""
    if fetch_missing:
        records = fetch_many(df["issuer_cik"], refresh=refresh)
    else:
        records = load_cached()

    ownership_df, eightk_df, covered = records_to_frames(records)
    feats = compute_filing_context_features(df, ownership_df, eightk_df, covered)

    assert len(feats) == len(df), "row count changed -- refusing to attach"
    assert list(feats.index) == list(df.index), "row order changed -- refusing to attach"

    return pd.concat([df, feats], axis=1)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _print_sanity_report(df: pd.DataFrame, eightk_df: pd.DataFrame) -> None:
    n = len(df)
    print(f"\n=== filing_context sanity report ({n} rows) ===")
    print("Coverage (fraction of rows with a non-null value):")
    for c in NEW_FEATURE_COLS:
        frac = df[c].notna().mean()
        print(f"  {c}: {frac:.1%}")

    if not eightk_df.empty:
        has_items = eightk_df["items"].map(lambda x: bool(_parse_item_codes(x)))
        print(f"\nFraction of fetched 8-K filings with a populated items field: "
              f"{has_items.mean():.1%} (n={len(eightk_df)})")

    if "adj_21" in df.columns:
        print("\nMEDIAN adj_21 by x_has_recent_13d (0 vs 1) -- MEDIANS not means, "
              "adj_21 has a large right tail:")
        valid = df.dropna(subset=["x_has_recent_13d", "adj_21"])
        if not valid.empty:
            tbl = valid.groupby("x_has_recent_13d")["adj_21"].agg(["median", "count"])
            print(tbl.to_string())
        else:
            print("  (no rows with both fields non-null)")

        print("\nMEDIAN adj_21 by x_8k_departure_trail90 > 0 (0 vs 1):")
        valid2 = df.dropna(subset=["x_8k_departure_trail90", "adj_21"]).copy()
        if not valid2.empty:
            valid2["has_departure_8k"] = (valid2["x_8k_departure_trail90"] > 0).astype(int)
            tbl2 = valid2.groupby("has_departure_8k")["adj_21"].agg(["median", "count"])
            print(tbl2.to_string())
        else:
            print("  (no rows with both fields non-null)")
    print()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--events", default=DEFAULT_EVENTS,
                    help="Research/events parquet to attach features to.")
    p.add_argument("--out", default=DEFAULT_OUT, help="Output parquet path.")
    p.add_argument("--refresh", action="store_true", help="Ignore the disk cache and refetch.")
    p.add_argument("--verbose", action="store_true")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                         format="%(asctime)s [%(levelname)s] %(message)s")

    log.info("filing_context: reading events from %s", a.events)
    events = pd.read_parquet(a.events)

    records = fetch_many(events["issuer_cik"], refresh=a.refresh)
    if not records:
        raise SystemExit("filing_context: nothing fetched")
    ownership_df, eightk_df, covered = records_to_frames(records)
    log.info("filing_context: %d issuer(s) covered, %d ownership filing(s), %d 8-K(s)",
              len(covered), len(ownership_df), len(eightk_df))

    feats = compute_filing_context_features(events, ownership_df, eightk_df, covered)
    assert len(feats) == len(events), "row count changed -- refusing to write"
    assert list(feats.index) == list(events.index), "row order changed -- refusing to write"
    out_df = pd.concat([events, feats], axis=1)
    assert len(out_df) == len(events), "concat changed row count -- refusing to write"

    out_path = a.out if os.path.isabs(a.out) else os.path.join(REPO_ROOT, a.out)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    out_df.to_parquet(out_path, index=False)
    log.info("filing_context: wrote %s (%d rows, %d cols)", out_path, len(out_df), out_df.shape[1])

    _print_sanity_report(out_df, eightk_df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
