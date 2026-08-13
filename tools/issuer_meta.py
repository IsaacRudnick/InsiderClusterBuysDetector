"""Issuer reference data from SEC's free submissions API, keyed on issuer_cik.

WHY this exists. The research dataset has 59 features and not one of them
describes WHAT THE COMPANY IS. No industry, no listing venue, no size. The
consequence shows up in the label, not the features: adj_h is
`fwd_h - spy_h`, so an insider cluster in a biotech during a biotech
drawdown is recorded as a bad decision by that insider. It was not -- the
sector fell. Every model trained on that label is asked to learn insider
skill through sector noise it has no way to see or subtract.

SEC serves the missing piece for free at
https://data.sec.gov/submissions/CIK##########.json, keyed on exactly the
issuer_cik the events parquet already carries. This module fetches it once
per CIK, caches to disk, and returns a tidy frame.

What comes back and why each field is kept:
  sic / sic_description -- 4-digit Standard Industrial Classification. The
      point of the exercise: lets a label be computed relative to an
      industry peer group instead of relative to SPY.
  sic_major             -- first 2 digits. SIC's 4-digit codes are far too
      granular for peer groups at this sample size (6,787 issuers spread
      over ~400 codes leaves single-digit cells). The 2-digit major group
      is the coarser cut peer-relative labels actually want.
  exchange              -- NYSE/Nasdaq/OTC/none. OTC and no-exchange names
      behave differently enough from listed ones that this is worth having
      as its own feature, and it is a partial, imperfect proxy for the
      delisting risk the price cache cannot see (see RESEARCH_NOTES.md's
      survivorship section).
  state_of_incorporation, fiscal_year_end -- cheap, already in the payload.
      fiscal_year_end in particular allows an earnings-proximity feature
      later without another fetch.

IMPORTANT -- this data is NOT point-in-time. The submissions endpoint
returns the issuer's CURRENT classification, not what it was on the event
day. A company that reclassified its SIC, moved from OTC to Nasdaq, or
changed its name is reported here only in its present form. For SIC that is
a mild concern: industry reassignment is rare, and the peer-grouping use is
coarse (2-digit). For `exchange` it is a REAL one: uplisting and delisting
are exactly the events that move a stock, so a model given today's exchange
for a 2019 event is being told something about the future. Treat `exchange`
as unsafe for features and use it only for cohort diagnostics, unless and
until a point-in-time source replaces it. `_UNSAFE_POINT_IN_TIME` names the
fields this applies to so a caller cannot forget.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import insider_cluster_buys as ics  # noqa: E402  (reuses its limiter, session, UA)

log = logging.getLogger("issuer_meta")

CACHE_DIR = os.path.join(REPO_ROOT, "issuer_meta_cache")
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
MAX_WORKERS = 8

# Fields whose value is "as of today", not "as of the event". See the module
# docstring. Anything listed here must not become an x_ feature.
_UNSAFE_POINT_IN_TIME: tuple[str, ...] = ("exchange", "issuer_name_current")


def _cache_path(cik10: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{cik10}.json")


def normalize_cik(cik) -> str | None:
    """SEC's endpoint wants a zero-padded 10-digit CIK. The events parquet
    stores it that way already, but a caller passing an int or a stripped
    string should not silently produce a 404."""
    if cik is None or (isinstance(cik, float) and cik != cik):
        return None
    s = str(cik).strip()
    if not s:
        return None
    s = s.split(".")[0]  # an int that round-tripped through float
    if not s.isdigit():
        return None
    return s.zfill(10)


def fetch_one(cik10: str, *, refresh: bool = False) -> dict | None:
    """Fetch (or read from cache) one issuer's metadata.

    A 404 is cached as an explicit empty record rather than left absent, so
    a rerun does not re-request CIKs SEC does not know about. Roughly 1-2%
    of issuer CIKs in the events file are pre-2001 or otherwise absent from
    the submissions endpoint.
    """
    path = _cache_path(cik10)
    if not refresh and os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError):
            log.warning("issuer_meta: unreadable cache %s, refetching", path)

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
        log.warning("issuer_meta: %s failed (%s)", cik10, exc)
        return None

    exchanges = payload.get("exchanges") or []
    sic = (payload.get("sic") or "").strip()
    rec = {
        "cik": cik10,
        "found": True,
        "issuer_name_current": payload.get("name"),
        "sic": sic or None,
        "sic_description": payload.get("sicDescription") or None,
        # 2-digit major group; see the docstring for why not the full code.
        "sic_major": sic[:2] if len(sic) >= 2 else None,
        "exchange": exchanges[0] if exchanges else None,
        "state_of_incorporation": payload.get("stateOfIncorporation") or None,
        "fiscal_year_end": payload.get("fiscalYearEnd") or None,
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rec, fh)
    return rec


def fetch_many(ciks, *, refresh: bool = False, workers: int = MAX_WORKERS) -> pd.DataFrame:
    norm = []
    seen = set()
    for c in ciks:
        n = normalize_cik(c)
        if n and n not in seen:
            seen.add(n)
            norm.append(n)

    cached = [c for c in norm if os.path.exists(_cache_path(c))] if not refresh else []
    todo = [c for c in norm if c not in set(cached)]
    log.info("issuer_meta: %d CIK(s) -- %d cached, %d to fetch @ %.0f req/s",
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
                if done % 500 == 0:
                    log.info("issuer_meta: %d/%d fetched", done, len(todo))

    df = pd.DataFrame(records)
    if df.empty:
        return df
    found = int(df.get("found", pd.Series(dtype=bool)).sum())
    log.info("issuer_meta: %d record(s), %d found, %d missing from SEC",
             len(df), found, len(df) - found)
    return df


def load_cached() -> pd.DataFrame:
    """Everything already on disk, without touching the network."""
    if not os.path.isdir(CACHE_DIR):
        return pd.DataFrame()
    recs = []
    for name in os.listdir(CACHE_DIR):
        if not name.endswith(".json"):
            continue
        try:
            with open(os.path.join(CACHE_DIR, name), encoding="utf-8") as fh:
                recs.append(json.load(fh))
        except (json.JSONDecodeError, OSError):
            continue
    return pd.DataFrame(recs)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--events", default="", help="Events parquet to take issuer_cik from. Default: widest in clusters_history/.")
    p.add_argument("--out", default="research_data/issuer_meta.parquet")
    p.add_argument("--refresh", action="store_true", help="Ignore the disk cache and refetch.")
    p.add_argument("--verbose", action="store_true")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    events = a.events
    if not events:
        import glob
        cands = glob.glob(os.path.join(REPO_ROOT, "clusters_history", "*.parquet"))
        if not cands:
            raise SystemExit("no events parquet found in clusters_history/")
        events = max(cands, key=os.path.getsize)
    log.info("issuer_meta: reading issuer_cik from %s", events)

    ciks = pd.read_parquet(events, columns=["issuer_cik"])["issuer_cik"]
    df = fetch_many(ciks, refresh=a.refresh)
    if df.empty:
        raise SystemExit("issuer_meta: nothing fetched")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    df.to_parquet(a.out, index=False)
    log.info("issuer_meta: wrote %s (%d rows)", a.out, len(df))

    ok = df[df.get("found", False) == True]  # noqa: E712
    if not ok.empty:
        log.info("issuer_meta: top SIC major groups --\n%s",
                 ok["sic_major"].value_counts().head(12).to_string())
        log.info("issuer_meta: exchanges --\n%s",
                 ok["exchange"].fillna("(none)").value_counts().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
