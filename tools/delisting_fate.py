"""What happened to the companies that vanished, and what that does to every
number this project has produced.

THE PROBLEM, STATED PRECISELY
=============================
33.2% of the tickers in the events file (2,455 of 7,390) have no price history,
because the data provider deletes a symbol when it stops trading. Those rows
are not zeroed, they are DROPPED -- and measurably so: of the 10,861 rows in
the research dataset, exactly ZERO are on a dead ticker, while dead tickers
account for 24.9% of raw insider transactions and 27.1% of insider dollars.

The research dataset is therefore a survivors-only sample. Every return,
Sharpe and win rate in RESEARCH_NOTES.md is conditioned on the company still
existing today, which is not a condition an investor in 2019 could have
selected on. The bias runs one way: companies disappear mostly by failing.

WHY YOU DO NOT NEED PRICES TO FIX THIS
--------------------------------------
Recovering delisted price history from a free source turned out to be a dead
end -- Stooq, the obvious candidate, returned nothing for 0 of 29 tested dead
tickers. But prices are not what is missing. What is missing is each dead
company's FATE, and that is in EDGAR, which this project already scrapes:

  Form 25 / 25-NSE   the exchange removed the security, with a date
  Form 15-12B/12G/15D  the company deregistered, with a date
  8-K Item 1.03      bankruptcy
  still filing today  the company is ALIVE -- usually under a new ticker,
                     which the submissions JSON reports directly

That last case is the happy one: a live company with a new symbol has real
price history that can simply be fetched under the new name, converting a
dropped row into a real measurement rather than an assumption.

WHAT THIS MODULE PRODUCES, AND WHAT IT REFUSES TO PRETEND
---------------------------------------------------------
For the genuinely dead, no free source gives the final trade. So this does NOT
invent one. It produces a BOUND: re-run the headline results three times, with
the dead assigned a total loss, a half loss, and no excess loss at all. The
truth lies between the first and the last, and a range that is honestly
measured beats a point estimate that is quietly wrong.

An acquisition is the one case where dropping the row biases DOWNWARD -- deals
close at a premium -- so acquisitions are separated out rather than lumped in
with the failures.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import insider_cluster_buys as icb  # noqa: E402  (for _load_dotenv / SEC_USER_AGENT)

#: Deliberately NOT tools/filing_calendar.py's `filing_cache/`. That module
#: caches its own reduced shape (a list of the periodic filings it cares
#: about) under the same CIK-named files, so reading it here would silently
#: hand this module a list where it expects the raw submissions object. Two
#: caches of the same endpoint is the lesser evil against one cache with two
#: incompatible schemas in it.
CACHE_DIR = os.path.join(REPO_ROOT, "delist_cache")

#: Three requests a second, not the 8 the screener uses. Several tools may be
#: hitting SEC concurrently and the published ceiling is 10/s for the whole
#: user agent, not per process.
RATE_LIMIT = 3.0

#: Delisting-related form types. 25 and 25-NSE are the exchange's notice that a
#: security is being removed; the 15 family is the issuer deregistering, which
#: usually follows.
DELIST_FORMS = {"25", "25-NSE"}
DEREG_FORMS = {"15-12B", "15-12G", "15-15D", "15F-12B", "15F-12G", "15F-15D"}

#: 8-K item 1.03 is "Bankruptcy or Receivership". Item 2.01 is "Completion of
#: Acquisition or Disposition of Assets", which is how a completed merger shows
#: up on the target's own filing history.
ITEM_BANKRUPTCY = "1.03"
ITEM_ACQUISITION = "2.01"


@dataclass
class Fate:
    cik: str
    event_ticker: str
    found: bool = False
    current_tickers: str = ""       # comma list from the submissions JSON
    last_filing_date: str = ""
    first_filing_date: str = ""
    delist_date: str = ""           # Form 25 / 25-NSE
    dereg_date: str = ""            # Form 15 family
    bankruptcy_date: str = ""       # 8-K item 1.03
    acquisition_date: str = ""      # 8-K item 2.01
    n_filings: int = 0
    classification: str = "unknown"


def _headers() -> dict:
    icb._load_dotenv()
    ua = os.environ.get("SEC_USER_AGENT", "").strip()
    if not ua:
        raise SystemExit(
            "SEC_USER_AGENT is not set. SEC fair-access rules require a real "
            "name and email; without one they throttle or block."
        )
    # No Accept-Encoding header on purpose. urllib does NOT transparently
    # decompress a gzip response, so advertising gzip earns compressed bytes
    # that json.loads then chokes on -- and the failure surfaces as a generic
    # exception per CIK, which looks exactly like "SEC has no record of this
    # company" rather than like a bug. Ask for plain text and pay the extra
    # bandwidth.
    return {"User-Agent": ua}


def fetch_submissions(cik: str, headers: dict) -> dict | None:
    """Submissions JSON for one CIK, cached on disk.

    Cached even on a 404: a CIK that SEC does not know about will still not be
    known about tomorrow, and re-asking every run wastes the rate limit that
    the real fetches need.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{cik}.json")
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError):
            pass  # corrupt cache entry: fall through and refetch
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        data = {"_error": exc.code}
    except Exception as exc:  # noqa: BLE001 - network is allowed to fail
        return {"_error": str(exc)}
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh)
    os.replace(tmp, path)
    return data


def classify(cik: str, ticker: str, sub: dict | None) -> Fate:
    """Read one submissions payload into a Fate.

    Note `items` is a parallel array to `form` in `filings.recent`, holding the
    comma-separated 8-K item numbers for that filing. It is the cheap way to
    spot a bankruptcy without fetching the 8-K document itself.
    """
    f = Fate(cik=cik, event_ticker=ticker)
    # Defensive: only a dict-shaped payload is a submissions object. A cache
    # written by a different tool, or a truncated download, must degrade to
    # "unknown" rather than crash a 2,000-CIK run near the end of it.
    if not isinstance(sub, dict) or "_error" in sub or "filings" not in sub:
        return f
    f.found = True
    f.current_tickers = ",".join(sub.get("tickers") or [])
    recent = (sub.get("filings") or {}).get("recent") or {}
    forms = list(recent.get("form", []))
    dates = list(recent.get("filingDate", []))
    items = list(recent.get("items", [])) or [""] * len(forms)
    f.n_filings = len(forms)
    if dates:
        f.last_filing_date = max(dates)
        f.first_filing_date = min(dates)

    for form, date, item in zip(forms, dates, items):
        base = (form or "").upper()
        if base in DELIST_FORMS and (not f.delist_date or date < f.delist_date):
            f.delist_date = date
        if base in DEREG_FORMS and (not f.dereg_date or date < f.dereg_date):
            f.dereg_date = date
        if base.startswith("8-K") and item:
            if ITEM_BANKRUPTCY in item and (
                not f.bankruptcy_date or date < f.bankruptcy_date
            ):
                f.bankruptcy_date = date
            if ITEM_ACQUISITION in item and (
                not f.acquisition_date or date < f.acquisition_date
            ):
                f.acquisition_date = date

    f.classification = _classify_one(f, ticker)
    return f


#: A company filing this recently is treated as alive. Chosen because a live
#: registrant files at minimum an annual report plus quarterly reports; going
#: 18 months in silence means it has stopped reporting whatever its formal
#: status.
ALIVE_CUTOFF = "2025-02-01"


def _classify_one(f: Fate, ticker: str) -> str:
    if not f.found:
        return "not_on_edgar"
    if f.bankruptcy_date:
        return "bankruptcy"
    still_filing = f.last_filing_date >= ALIVE_CUTOFF
    tickers = [t for t in f.current_tickers.split(",") if t]
    if still_filing and tickers and ticker not in tickers:
        # The single most valuable case: the company is alive under a new
        # symbol, so its real price history exists and can simply be fetched.
        return "renamed_alive"
    if still_filing:
        return "alive_no_ticker"
    if f.acquisition_date and not f.delist_date:
        return "acquired"
    if f.delist_date or f.dereg_date:
        # Delisting alone does not say why. An acquisition 8-K near the
        # delisting is the usual tell that this was a deal rather than a
        # failure, and deals close at a premium -- so they must not be lumped
        # in with the failures, which would bias the correction the wrong way.
        if f.acquisition_date:
            return "acquired_delisted"
        return "delisted"
    return "unknown"


def resolve(pairs: list[tuple[str, str]], *, workers: int = 6) -> pd.DataFrame:
    headers = _headers()
    delay = 1.0 / RATE_LIMIT
    out: list[Fate] = []
    done = 0

    def one(pair):
        nonlocal done
        cik, ticker = pair
        sub = fetch_submissions(cik, headers)
        done += 1
        if done % 200 == 0:
            print(f"    {done}/{len(pairs)}", flush=True)
        return classify(cik, ticker, sub)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = []
        for p in pairs:
            futures.append(ex.submit(one, p))
            time.sleep(delay)
        for fut in futures:
            out.append(fut.result())
    return pd.DataFrame([asdict(f) for f in out])


def dead_ticker_pairs(events: str | None = None) -> list[tuple[str, str]]:
    """(cik, ticker) for every event ticker with no price file.

    `events` defaults to the NEWEST events_*.parquet in clusters_history/
    rather than a pinned filename. It was hardcoded to
    events_20180813_20260813.parquet, which meant a rebuilt events file was
    silently ignored and the dead-ticker set was whatever it had been in
    August -- including tickers that a later scrape prices fine.
    """
    if events is None:
        d = os.path.join(REPO_ROOT, "clusters_history")
        cands = sorted(f for f in os.listdir(d)
                       if f.startswith("events_") and f.endswith(".parquet"))
        if not cands:
            raise SystemExit(f"no events_*.parquet in {d}")
        events = os.path.join(d, cands[-1])
    print(f"events file: {events}", flush=True)
    ev = pd.read_parquet(events)
    price_dir = os.path.join(REPO_ROOT, "price_cache")
    have = {f[:-8] for f in os.listdir(price_dir) if f.endswith(".parquet")}
    cik_of = ev.drop_duplicates("ticker").set_index("ticker")["issuer_cik"]
    pairs = []
    for t in sorted(set(ev["ticker"].dropna())):
        if t in have:
            continue
        c = str(cik_of.get(t, "")).strip()
        if c and c.strip("0"):
            pairs.append((c.zfill(10), t))
    return pairs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=os.path.join(
        REPO_ROOT, "research_data", "delisting_fate.parquet"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--events", default="")
    args = ap.parse_args(argv)

    pairs = dead_ticker_pairs(args.events or None)
    if args.limit:
        pairs = pairs[: args.limit]
    print(f"resolving the fate of {len(pairs)} dead tickers "
          f"({len({c for c, _ in pairs})} distinct CIKs)", flush=True)

    df = resolve(pairs)
    df.to_parquet(args.out, index=False)

    print("\n===== WHAT HAPPENED TO THEM =====")
    counts = df["classification"].value_counts()
    for k, v in counts.items():
        print(f"  {k:18s} {v:5d}  {v / len(df):6.1%}")
    print(f"\n  on EDGAR at all: {int(df['found'].sum())}/{len(df)}")
    renamed = df[df["classification"] == "renamed_alive"]
    print(f"  alive under a NEW ticker: {len(renamed)} "
          f"-- these have real price history to recover")
    if len(renamed):
        print(renamed[["event_ticker", "current_tickers", "last_filing_date"]]
              .head(15).to_string(index=False))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
