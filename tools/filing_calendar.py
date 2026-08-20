"""Per-issuer filing calendar (10-Q / 10-K / 8-K dates) from SEC's free
submissions API, keyed on issuer_cik.

WHY this exists. The research dataset knows when an insider cluster bought,
but nothing about where that event sat relative to the issuer's own filing
rhythm. A cluster buy three days before earnings and one three days after a
quiet 8-K are very different situations for the 21-day forward return to
absorb, and the dataset has no feature that can tell them apart. This module
fetches each issuer's actual filing history once, caches it to disk, and
writes a tidy long-format calendar that tools/earnings_features.py turns
into point-in-time features. This module ONLY fetches and normalizes dates;
it does not compute point-in-time features itself, so it does not need to
know about event_day at all.

SEC serves this for free at
https://data.sec.gov/submissions/CIK##########.json, keyed on exactly the
issuer_cik the events/research parquet already carries -- same identifier
tools/issuer_meta.py already uses, but a different endpoint payload (this
module only reads `filings`, issuer_meta.py only reads issuer metadata).
Cached separately (filing_cache/, vs issuer_meta.py's issuer_meta_cache/)
so the two never collide on disk.

Coverage caveat. `filings.recent` is capped at roughly 1000 entries and,
for an issuer with a long or busy filing history, does not necessarily
reach back to 2018 -- our events start 2018-08-13. When the oldest date
actually present in `recent` is later than 2018-06-01, this module also
follows `filings.files` (older filings, paginated as separate JSON files)
until it has paged back before that cutoff or run out of pages. Even so, a
small number of issuers may have their true earliest known filing date
still short of 2018-08-13 (e.g. a very prolific 8-K filer with more history
than a bounded number of pages can reach). The coverage summary printed at
the end of a run reports this honestly rather than silently accepting it.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import insider_cluster_buys as ics  # noqa: E402  (reuses its limiter, session, UA)

log = logging.getLogger("filing_calendar")

CACHE_DIR = os.path.join(REPO_ROOT, "filing_cache")
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
FILES_URL = "https://data.sec.gov/submissions/{name}"
MAX_WORKERS = 8

# Forms we keep, and their normalized (amendment-collapsed) name.
_FORM_MAP = {
    "10-Q": "10-Q", "10-Q/A": "10-Q",
    "10-K": "10-K", "10-K/A": "10-K",
    "8-K": "8-K", "8-K/A": "8-K",
}

# Our events start 2018-08-13. If `filings.recent` alone does not reach
# back before this, older paginated files are worth fetching.
_COVERAGE_CUTOFF = date(2018, 6, 1)
_EVENT_WINDOW_START = date(2018, 8, 13)
# Don't page back forever for a single heavy filer -- 2018-06-01 is usually
# reached within a handful of pages; this is just a runtime backstop.
_MAX_EXTRA_PAGES = 15


def _cache_path(cik10: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{cik10}.json")


def normalize_cik(cik) -> str | None:
    """SEC's endpoint wants a zero-padded 10-digit CIK. Mirrors
    tools/issuer_meta.py's normalize_cik exactly -- same input shapes show
    up in the same events/research parquets."""
    if cik is None or (isinstance(cik, float) and cik != cik):
        return None
    s = str(cik).strip()
    if not s:
        return None
    s = s.split(".")[0]  # an int that round-tripped through float
    if not s.isdigit():
        return None
    return s.zfill(10)


def _normalize_form(form: str) -> str | None:
    return _FORM_MAP.get((form or "").strip())


def _kept_filings_from_block(block: dict) -> list[dict]:
    """Extract kept (form, filingDate, reportDate) triples from one parallel-
    array block, whether that block is `filings.recent` (nested) or one of
    `filings.files`' paginated payloads (bare top-level)."""
    forms = block.get("form") or []
    filing_dates = block.get("filingDate") or []
    report_dates = block.get("reportDate") or []
    out = []
    n = len(forms)
    for i in range(n):
        norm = _normalize_form(forms[i])
        if norm is None:
            continue
        fd = filing_dates[i] if i < len(filing_dates) else None
        rd = report_dates[i] if i < len(report_dates) else None
        if not fd:
            continue
        out.append({"form": norm, "filing_date": fd, "report_date": rd or None})
    return out


def _earliest_date_in_block(block: dict) -> date | None:
    """The earliest filingDate literally present in a parallel-array block,
    regardless of form -- used only to decide whether older pages are
    needed, so it must look at ALL filings, not just the kept forms."""
    filing_dates = [d for d in (block.get("filingDate") or []) if d]
    if not filing_dates:
        return None
    try:
        return min(date.fromisoformat(d) for d in filing_dates)
    except ValueError:
        return None


def fetch_one(cik10: str, *, refresh: bool = False) -> dict | None:
    """Fetch (or read from cache) one issuer's filing calendar.

    A 404 is cached as an explicit `{"found": False}` record, mirroring
    tools/issuer_meta.py, so a rerun does not re-request CIKs SEC does not
    know about.
    """
    path = _cache_path(cik10)
    if not refresh and os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError):
            log.warning("filing_calendar: unreadable cache %s, refetching", path)

    try:
        resp = ics._get(SUBMISSIONS_URL.format(cik=cik10))
        payload = resp.json()
    except Exception as exc:  # noqa: BLE001 -- one bad CIK must not kill the run
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status == 404:
            rec = {"cik": cik10, "found": False}
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(rec, fh)
            return rec
        log.warning("filing_calendar: %s failed (%s)", cik10, exc)
        return None

    try:
        filings_block = payload.get("filings") or {}
        recent = filings_block.get("recent") or {}
        kept = _kept_filings_from_block(recent)
        earliest = _earliest_date_in_block(recent)

        # `recent` alone may not reach back far enough for our event window.
        # Page through `filings.files` (oldest-first is not guaranteed by
        # spec, so we just keep going until either we've passed the cutoff
        # or we run out of files / pages).
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
                    presp = ics._get(FILES_URL.format(name=name))
                    page = presp.json()
                except Exception as exc:  # noqa: BLE001
                    log.warning("filing_calendar: %s page %s failed (%s)", cik10, name, exc)
                    continue
                pages_fetched += 1
                kept.extend(_kept_filings_from_block(page))
                page_earliest = _earliest_date_in_block(page)
                if page_earliest is not None and (earliest is None or page_earliest < earliest):
                    earliest = page_earliest
                if page_earliest is not None and page_earliest <= _COVERAGE_CUTOFF:
                    break  # this page already reaches back far enough

        rec = {
            "cik": cik10,
            "found": True,
            "earliest_filing_date": earliest.isoformat() if earliest else None,
            "filings": kept,
        }
    except Exception as exc:  # noqa: BLE001 -- malformed payload must not kill the run
        log.warning("filing_calendar: %s unexpected payload shape (%s)", cik10, exc)
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
    log.info("filing_calendar: %d CIK(s) -- %d cached, %d to fetch @ %.0f req/s",
             len(norm), len(cached), len(todo), ics.RATE_LIMIT)

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
                    log.info("filing_calendar: %d/%d fetched", done, len(todo))

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


def records_to_frame(records: list[dict]) -> pd.DataFrame:
    """Flatten cached per-issuer records into the long-format calendar with
    exactly the columns filing_calendar.parquet is specified to have."""
    rows = []
    for rec in records:
        if not rec or not rec.get("found"):
            continue
        cik = rec["cik"]
        for f in rec.get("filings") or []:
            rows.append({
                "cik": cik,
                "form": f["form"],
                "filing_date": f["filing_date"],
                "report_date": f.get("report_date"),
            })
    df = pd.DataFrame(rows, columns=["cik", "form", "filing_date", "report_date"])
    if not df.empty:
        df["filing_date"] = pd.to_datetime(df["filing_date"])
        df["report_date"] = pd.to_datetime(df["report_date"])
    return df


def print_coverage_summary(records: list[dict]) -> None:
    n = len(records)
    found = [r for r in records if r.get("found")]
    missing = n - len(found)
    log.info("filing_calendar: coverage -- %d issuer(s), %d found on SEC, %d missing", n, len(found), missing)

    with_earliest = [r for r in found if r.get("earliest_filing_date")]
    no_filings_kept = [r for r in found if not r.get("earliest_filing_date")]
    late_start = []
    for r in with_earliest:
        try:
            d = date.fromisoformat(r["earliest_filing_date"])
        except ValueError:
            continue
        if d > _EVENT_WINDOW_START:
            late_start.append((r["cik"], d))

    log.info("filing_calendar: %d issuer(s) have NO 10-Q/10-K/8-K on record at all "
              "(found on SEC but zero kept filings)", len(no_filings_kept))
    log.info(
        "filing_calendar: %d/%d issuer(s) with filings have their EARLIEST known "
        "filing date AFTER our event window start (%s) -- our coverage for those "
        "issuers is INCOMPLETE for early events, honestly flagged, not hidden.",
        len(late_start), len(with_earliest), _EVENT_WINDOW_START.isoformat(),
    )
    if late_start:
        worst = sorted(late_start, key=lambda t: t[1], reverse=True)[:10]
        log.info("filing_calendar: latest-starting examples (cik, earliest_filing_date): %s", worst)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--events", default="",
                    help="Research/events parquet to take issuer_cik from. "
                         "Default: research_data/research_10861rows_20260813.parquet.")
    p.add_argument("--out", default="research_data/filing_calendar.parquet")
    p.add_argument("--refresh", action="store_true", help="Ignore the disk cache and refetch.")
    p.add_argument("--verbose", action="store_true")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                         format="%(asctime)s [%(levelname)s] %(message)s")

    events = a.events
    if not events:
        default_events = os.path.join(REPO_ROOT, "research_data", "research_10861rows_20260813.parquet")
        if os.path.exists(default_events):
            events = default_events
        else:
            cands = glob.glob(os.path.join(REPO_ROOT, "research_data", "research_*rows_*.parquet"))
            if not cands:
                raise SystemExit("no research parquet found in research_data/")
            events = max(cands, key=os.path.getsize)
    log.info("filing_calendar: reading issuer_cik from %s", events)

    ciks = pd.read_parquet(events, columns=["issuer_cik"])["issuer_cik"]
    records = fetch_many(ciks, refresh=a.refresh)
    if not records:
        raise SystemExit("filing_calendar: nothing fetched")

    df = records_to_frame(records)
    if df.empty:
        raise SystemExit("filing_calendar: no kept filings across all issuers")

    out_path = os.path.join(REPO_ROOT, a.out) if not os.path.isabs(a.out) else a.out
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    df.to_parquet(out_path, index=False)
    log.info("filing_calendar: wrote %s (%d filing rows across %d issuers)",
              out_path, len(df), df["cik"].nunique())

    print_coverage_summary(records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
