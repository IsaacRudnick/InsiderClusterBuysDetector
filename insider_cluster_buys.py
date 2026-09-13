"""
Insider Cluster-Buy Detector (SEC EDGAR)

Scans SEC EDGAR for insider acquisitions on Form 4 / 4/A, parses the
structured ownership XML, and flags "cluster buys" - at least MIN_INSIDERS
distinct reporting owners making qualifying purchases of the same issuer
within a rolling WINDOW_DAYS window.

A "qualifying" purchase defaults to transaction code "P" (open-market
purchase). Grants/awards (code "A" in the transaction code column) also
produce an "A" in the acquired/disposed column but are compensation, not
conviction buys - they are excluded by default. Widen QUALIFYING_CODES if
you want a looser definition.

Outputs (in out/, overwritten each run):
  - insider_cluster_buys.xlsx   (Flagged Clusters / All Tx / Errors)
  - dashboard.html              (self-contained dashboard with embedded JSON payload)

Env vars:
    ICB_LOOKBACK (default 30), ICB_MIN_INSIDERS (default 2),
    ICB_WINDOW_DAYS (default 14) -- override the three interactive
    _prompt_inputs() prompts (lookback days / min distinct insiders / cluster
    window days) so a backgrounded, piped, or scheduled run never blocks on
    a bare input() with no stdin (see backtest.py's BT_MONTHS et al. for the
    same pattern).
    Also reads SEC_USER_AGENT, LIVE_SCORE_FETCH_PRICES (0|1),
    LIVE_SCORE_MODEL_PATH, LIVE_SCORE_HISTORY_PATH (pre-existing, unchanged).

This is informational tooling, not financial advice.
"""

import glob
import json
import os
import re
import time
import threading
import logging
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from typing import Optional
from xml.etree import ElementTree as ET

import pandas as pd
import requests
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

import ipo_lookup
from build_html import factor_favorable, fmt_factor_value, model_sort_key, render_html, verdict_label
import findings


# ---------------------------------------------------------------------------
# Load .env
# ---------------------------------------------------------------------------
def _load_dotenv() -> None:
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path) as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


_load_dotenv()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# SEC fair-access REQUIRES a descriptive User-Agent identifying you and a
# contact email. Replace the placeholder with your real name and address
# (or set SEC_USER_AGENT in .env).
USER_AGENT = os.environ.get(
    "SEC_USER_AGENT",
    "Your Name your.email@example.com",
)

# Transaction codes that count as a "buy" for cluster detection.
# "P" = open-market or private purchase (the conviction-buy signal).
# Other common codes (DO NOT include by default):
#   "A" = grant/award (compensation)         "M" = option exercise
#   "S" = sale                               "G" = gift
#   "F" = tax withholding                    "J" = other
# Widen this set ONLY if you understand what you are letting in.
QUALIFYING_CODES = {"P"}

# SEC fair-access policy caps at 10 req/s; stay comfortably under.
RATE_LIMIT = 8.0
MAX_WORKERS = 10

PARSE_CACHE_DIR = "parse_cache"   # one JSON per accession - skipped on rerun
OUTPUT_DIR = "out"                # xlsx / json artifacts (overwritten each run)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Runtime inputs
# ---------------------------------------------------------------------------
# Same env var + prompt pattern backtest.py uses for its BT_* family (see
# backtest.py's _prompt_with_default / module docstring) -- lets a
# backgrounded, piped, or scheduled run configure these without a TTY
# instead of dying on a bare input() with EOFError.
ICB_LOOKBACK_ENV = "ICB_LOOKBACK"
ICB_MIN_INSIDERS_ENV = "ICB_MIN_INSIDERS"
ICB_WINDOW_DAYS_ENV = "ICB_WINDOW_DAYS"


def _prompt_with_default(prompt: str, default: str, env_var: str | None = None) -> str:
    """Mirrors backtest.py's helper of the same name: an env var wins if
    set (and non-empty) and is logged so a piped/backgrounded run's log
    shows exactly what configured it; otherwise falls back to the usual
    interactive input() prompt, unchanged."""
    if env_var and os.environ.get(env_var):
        val = os.environ[env_var].strip()
        log.info("Using %s=%s from env", env_var, val)
        return val
    raw = input(f"{prompt} [default {default}]: ").strip()
    return raw or default


def _parse_int_setting(raw: str, env_var: str) -> int:
    """int() with a clear, variable-naming SystemExit instead of a bare
    ValueError traceback -- a garbage env value (or typo'd prompt answer)
    should say which setting is wrong, not just crash."""
    try:
        return int(raw)
    except ValueError:
        raise SystemExit(f"Invalid {env_var} {raw!r}; expected an integer") from None


def _prompt_inputs() -> tuple[int, int, int]:
    print("Insider Cluster-Buy Detector")
    print("Re-runs load from cache and only process new accessions.")
    print()
    lookback = _parse_int_setting(
        _prompt_with_default("Lookback period in days", "30", ICB_LOOKBACK_ENV),
        ICB_LOOKBACK_ENV,
    )
    min_insiders = _parse_int_setting(
        _prompt_with_default(
            "Minimum distinct insiders for a cluster", "2", ICB_MIN_INSIDERS_ENV,
        ),
        ICB_MIN_INSIDERS_ENV,
    )
    window_days = _parse_int_setting(
        _prompt_with_default("Cluster window in days", "14", ICB_WINDOW_DAYS_ENV),
        ICB_WINDOW_DAYS_ENV,
    )
    print()
    log.info(
        "Settings: lookback=%dd, min_insiders=%d, window=%dd, codes=%s",
        lookback, min_insiders, window_days, sorted(QUALIFYING_CODES),
    )
    return lookback, min_insiders, window_days


# ---------------------------------------------------------------------------
# Rate-limited HTTP session
# ---------------------------------------------------------------------------
class _RateLimiter:
    """Token-bucket-style limiter: each acquire() reserves a slot at the
    configured rate. Threads block in time.sleep outside the lock so we
    don't serialize the whole pool."""

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
SESSION.headers.update({
    "User-Agent": USER_AGENT,
    "Accept-Encoding": "gzip",
})


def _get(url: str, params: dict | None = None, accept_html: bool = False) -> requests.Response:
    headers = {"Accept": "text/html,application/xhtml+xml,application/xml,text/xml,*/*"} \
        if accept_html else {"Accept": "application/json"}
    last_exc: Optional[requests.HTTPError] = None
    for attempt in range(5):
        _LIMITER.acquire()
        resp = SESSION.get(url, params=params, headers=headers, timeout=30)
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


# ---------------------------------------------------------------------------
# Per-filing parse cache  (parse_cache/{adsh}.json)
# ---------------------------------------------------------------------------
def _parse_cache_path(adsh: str) -> str:
    os.makedirs(PARSE_CACHE_DIR, exist_ok=True)
    return os.path.join(PARSE_CACHE_DIR, f"{adsh}.json")


def _load_parse_cache(adsh: str) -> Optional[dict]:
    path = _parse_cache_path(adsh)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("Corrupt parse cache %s: %s - ignoring", path, exc)
        return None


def _save_parse_cache(adsh: str, data: dict) -> None:
    with open(_parse_cache_path(adsh), "w") as fh:
        json.dump(data, fh, indent=2, default=str)


# ---------------------------------------------------------------------------
# EDGAR daily-index discovery
# ---------------------------------------------------------------------------
# Form 3 is the positional-context-only initial statement; we never produce
# qualifying rows from it, so don't bother parsing.
INTERESTING_FORMS = {"4", "4/A"}


def fetch_daily_index(day: date) -> list[dict]:
    """Return Form 4 filings from the SEC daily index for `day` (empty on weekends/holidays)."""
    q = (day.month - 1) // 3 + 1
    url = (
        f"https://www.sec.gov/Archives/edgar/daily-index/{day.year}/"
        f"QTR{q}/form.{day.strftime('%Y%m%d')}.idx"
    )
    try:
        resp = _get(url, accept_html=True)
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code in (403, 404):
            return []
        raise

    lines = resp.text.splitlines()
    start = 0
    for i, line in enumerate(lines):
        if line.startswith("---"):
            start = i + 1
            break

    rows: list[dict] = []
    for line in lines[start:]:
        if not line.strip():
            continue
        form_type = line[0:12].strip()
        if form_type not in INTERESTING_FORMS:
            continue
        # Nominal column widths in the daily index are not strict: long
        # company names overflow the company-name column and shift CIK /
        # date / filename. The last three whitespace-separated tokens are
        # well-defined (file name, date filed, CIK), so right-split.
        rest = line[12:]
        parts = rest.rsplit(None, 3)
        if len(parts) < 4:
            continue
        company_name, cik, date_filed, file_name = parts
        rows.append({
            "form_type": form_type,
            "company_name": company_name.strip(),
            "cik": cik.strip(),
            "date_filed": date_filed.strip(),
            "file_name": file_name.strip(),
        })
    return rows


def discover_filings(lookback_days: int) -> list[dict]:
    """Walk daily indexes for the last `lookback_days` and collect Form 4 filings."""
    today = date.today()
    all_rows: list[dict] = []
    for offset in range(lookback_days + 1):
        day = today - timedelta(days=offset)
        if day.weekday() >= 5:
            continue
        log.info("Daily index %s", day.isoformat())
        try:
            all_rows.extend(fetch_daily_index(day))
        except Exception as exc:
            log.error("Failed to fetch daily index for %s: %s", day, exc)
    # Deduplicate by ACCESSION, not by file_name. The daily index lists one
    # row per CIK involved in a filing -- once under the issuer and once under
    # each reporting owner -- and each row carries a different
    # edgar/data/<cik>/<accession>.txt path for the same document. Deduping on
    # the path therefore never fires: a typical Form 4 was parsed twice, and
    # one with several reporting owners up to ten times, so every transaction
    # was counted that many times and every shares/value total was inflated to
    # match. A single 2026-09-04 index has 934 rows and 458 accessions, every
    # one of them appearing more than once.
    #
    # Keeping the first row is fine: the CIK only feeds _filing_base_url, and
    # SEC serves a filing under any CIK associated with it.
    seen = set()
    deduped = []
    for r in all_rows:
        try:
            key = _accession_from_filename(r["file_name"])
        except ValueError:
            # Unparseable path: fall back to the path itself rather than
            # dropping the filing entirely.
            key = r["file_name"]
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)
    log.info("Discovered %d Form 4 filings in last %d days (%d index rows)",
             len(deduped), lookback_days, len(all_rows))
    return deduped


# ---------------------------------------------------------------------------
# Filing parse (XML)
# ---------------------------------------------------------------------------
ADSH_RE = re.compile(r"/(\d{10}-\d{2}-\d{6})\.txt$")


def _accession_from_filename(file_name: str) -> str:
    m = ADSH_RE.search(file_name)
    if not m:
        raise ValueError(f"Could not extract accession from {file_name}")
    return m.group(1)


def _filing_base_url(cik: str, adsh: str) -> str:
    stripped = adsh.replace("-", "")
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{stripped}/"


def _filing_html_url(cik: str, adsh: str) -> str:
    stripped = adsh.replace("-", "")
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{stripped}/{adsh}-index.htm"


def _pick_ownership_xml(items: list[dict]) -> Optional[str]:
    """Choose the ownership XML file from the index.json item list."""
    candidates: list[str] = []
    for item in items:
        name = item.get("name", "")
        nl = name.lower()
        if not nl.endswith(".xml"):
            continue
        if nl in {"filingsummary.xml", "metadata.xml"}:
            continue
        if re.match(r"^r\d+\.xml$", nl):
            continue
        candidates.append(name)

    for prefer in ("primary_doc.xml", "form4.xml", "form3.xml", "form5.xml"):
        for c in candidates:
            if c.lower() == prefer:
                return c

    for c in candidates:
        if re.search(r"(?i)form[345]", c) or re.match(r"(?i)wf-form", c):
            return c

    return candidates[0] if candidates else None


def _text(el: Optional[ET.Element], path: str, default: str = "") -> str:
    if el is None:
        return default
    found = el.find(path)
    if found is None or found.text is None:
        return default
    return found.text.strip()


def _bool_flag(el: Optional[ET.Element], path: str) -> bool:
    raw = _text(el, path, "0")
    return raw in ("1", "true", "True")


def _num(el: Optional[ET.Element], path: str) -> Optional[float]:
    raw = _text(el, path, "")
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _collect_footnote_ids(elem: ET.Element) -> list[str]:
    """Return all footnoteId/@id values referenced anywhere under `elem`."""
    ids: list[str] = []
    for sub in elem.iter():
        if sub.tag == "footnoteId":
            fid = sub.get("id")
            if fid:
                ids.append(fid)
    return ids


_TEN_B5_1_RE = re.compile(r"10b5[-–—]?1|rule\s*10b5", re.IGNORECASE)


def parse_ownership_xml(xml_bytes: bytes, filing: dict, adsh: str) -> dict:
    """Parse a Form 4/4A ownership XML into our normalized dict."""
    root = ET.fromstring(xml_bytes)

    issuer_el = root.find("issuer")
    issuer = {
        "cik": _text(issuer_el, "issuerCik"),
        "name": _text(issuer_el, "issuerName"),
        "ticker": _text(issuer_el, "issuerTradingSymbol"),
    }

    # Footnote table: {id -> text}. Used downstream for ESPP / 10b5-1 hints.
    footnotes: dict[str, str] = {}
    for fn in root.findall("footnotes/footnote"):
        fid = fn.get("id", "")
        if fid:
            footnotes[fid] = (fn.text or "").strip()

    owners: list[dict] = []
    for owner_el in root.findall("reportingOwner"):
        rel = owner_el.find("reportingOwnerRelationship")
        owners.append({
            "cik": _text(owner_el, "reportingOwnerId/rptOwnerCik"),
            "name": _text(owner_el, "reportingOwnerId/rptOwnerName"),
            "is_director": _bool_flag(rel, "isDirector"),
            "is_officer": _bool_flag(rel, "isOfficer"),
            "is_ten_percent_owner": _bool_flag(rel, "isTenPercentOwner"),
            "is_other": _bool_flag(rel, "isOther"),
            "officer_title": _text(rel, "officerTitle"),
        })

    transactions: list[dict] = []
    for tx in root.findall("nonDerivativeTable/nonDerivativeTransaction"):
        coding = tx.find("transactionCoding")
        amounts = tx.find("transactionAmounts")
        post = tx.find("postTransactionAmounts")
        fn_ids = _collect_footnote_ids(tx)
        fn_text = " | ".join(footnotes.get(fid, "") for fid in fn_ids if footnotes.get(fid))
        transactions.append({
            "date": _text(tx, "transactionDate/value"),
            "code": _text(coding, "transactionCode"),
            "acquired_disposed": _text(amounts, "transactionAcquiredDisposedCode/value"),
            "shares": _num(amounts, "transactionShares/value"),
            "price_per_share": _num(amounts, "transactionPricePerShare/value"),
            "shares_owned_after": _num(post, "sharesOwnedFollowingTransaction/value"),
            "footnote_text": fn_text,
            "is_10b5_1": bool(_TEN_B5_1_RE.search(fn_text)) if fn_text else False,
        })

    holdings: list[dict] = []
    for h in root.findall("nonDerivativeTable/nonDerivativeHolding"):
        holdings.append({
            "shares_owned": _num(h, "postTransactionAmounts/sharesOwnedFollowingTransaction/value"),
        })

    return {
        "adsh": adsh,
        "form_type": filing["form_type"],
        "filing_date": filing["date_filed"],
        "issuer": issuer,
        "owners": owners,
        "transactions": transactions,
        "holdings": holdings,
        "filing_url": _filing_html_url(filing["cik"], adsh),
        "source": "xml",
    }


def parse_filing(filing: dict) -> dict:
    """Fetch + parse a single filing. Uses parse cache; tries primary_doc.xml
    first to avoid the index.json round-trip on the common case."""
    adsh = _accession_from_filename(filing["file_name"])
    cached = _load_parse_cache(adsh)
    if cached is not None:
        return cached

    base = _filing_base_url(filing["cik"], adsh)

    # Fast path: SEC's standard name covers the large majority of Form 4 filings.
    try:
        xml_resp = _get(base + "primary_doc.xml", accept_html=True)
        parsed = parse_ownership_xml(xml_resp.content, filing, adsh)
    except (requests.HTTPError, ET.ParseError):
        # Fallback: list directory and find ownership XML
        idx_resp = _get(base + "index.json")
        idx = idx_resp.json()
        items = idx.get("directory", {}).get("item", [])
        xml_name = _pick_ownership_xml(items)
        if xml_name is None:
            raise FileNotFoundError(f"No ownership XML in {base}")
        xml_resp = _get(base + xml_name, accept_html=True)
        parsed = parse_ownership_xml(xml_resp.content, filing, adsh)

    _save_parse_cache(adsh, parsed)
    return parsed


# ---------------------------------------------------------------------------
# Qualifying-transaction extraction + cluster detection
# ---------------------------------------------------------------------------
def _owner_role_str(owner: dict) -> str:
    roles = []
    if owner.get("is_director"):
        roles.append("Director")
    if owner.get("is_officer"):
        title = (owner.get("officer_title") or "").strip()
        roles.append(f"Officer ({title})" if title else "Officer")
    if owner.get("is_ten_percent_owner"):
        roles.append("10% Owner")
    if owner.get("is_other"):
        roles.append("Other")
    return ", ".join(roles) or "Unknown"


def extract_qualifying_rows(parsed_filings: list[dict]) -> list[dict]:
    """Flatten parsed filings into one row per qualifying (Form 4) acquisition."""
    rows: list[dict] = []
    for f in parsed_filings:
        if not f["form_type"].startswith("4"):
            continue
        issuer = f.get("issuer", {})
        owners = f.get("owners", []) or [{}]
        for tx in f.get("transactions", []):
            if (tx.get("acquired_disposed") or "").upper() != "A":
                continue
            if (tx.get("code") or "").upper() not in QUALIFYING_CODES:
                continue
            shares = tx.get("shares") or 0
            price = tx.get("price_per_share") or 0
            value = (shares or 0) * (price or 0)
            # Stake delta: % increase relative to pre-trade holdings.
            # None when prior stake unknown (no post amount) or zero (initial buy).
            owned_after = tx.get("shares_owned_after")
            pct_of_prior_stake: Optional[float] = None
            if owned_after is not None and shares:
                prior = owned_after - shares
                if prior > 0:
                    pct_of_prior_stake = (shares / prior) * 100.0
            owner = owners[0]
            rows.append({
                "adsh": f["adsh"],
                "form_type": f["form_type"],
                "filing_date": f["filing_date"],
                "filing_url": f["filing_url"],
                "issuer_cik": issuer.get("cik", ""),
                "issuer_name": issuer.get("name", ""),
                "ticker": issuer.get("ticker", ""),
                "owner_cik": owner.get("cik", ""),
                "owner_name": owner.get("name", ""),
                "owner_roles": _owner_role_str(owner),
                "is_director": bool(owner.get("is_director")),
                "is_officer": bool(owner.get("is_officer")),
                "is_ten_percent_owner": bool(owner.get("is_ten_percent_owner")),
                "transaction_date": tx.get("date", ""),
                "transaction_code": tx.get("code", ""),
                "acquired_disposed": tx.get("acquired_disposed", ""),
                "shares": shares,
                "price_per_share": price,
                "value": value,
                "shares_owned_after": owned_after,
                "pct_of_prior_stake": pct_of_prior_stake,
                "footnote_text": tx.get("footnote_text", ""),
                "is_10b5_1": bool(tx.get("is_10b5_1")),
            })
    return rows


def detect_clusters(rows: list[dict], min_insiders: int, window_days: int) -> list[dict]:
    """Group qualifying rows by issuer; emit one cluster per maximal qualifying window."""
    by_issuer: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["transaction_date"]:
            by_issuer[r["issuer_cik"]].append(r)

    clusters: list[dict] = []
    for issuer_cik, txs in by_issuer.items():
        txs_sorted = sorted(txs, key=lambda t: t["transaction_date"])
        dates = []
        for t in txs_sorted:
            try:
                dates.append(datetime.strptime(t["transaction_date"], "%Y-%m-%d").date())
            except ValueError:
                dates.append(None)

        n = len(txs_sorted)
        i = 0
        while i < n:
            if dates[i] is None:
                i += 1
                continue
            j = i
            while j + 1 < n and dates[j + 1] is not None and \
                    (dates[j + 1] - dates[i]).days <= window_days:
                j += 1
            window = txs_sorted[i:j + 1]
            distinct = {t["owner_cik"] for t in window if t["owner_cik"]}
            if len(distinct) >= min_insiders:
                clusters.append(_build_cluster(window))
                i = j + 1
            else:
                i += 1
    return clusters


_ROUTINE_FOOTNOTE_RE = re.compile(
    r"ESPP"
    r"|Employee\s+Stock\s+Purchase"
    r"|401\s*\(?\s*k\s*\)?"
    r"|Dividend\s+Reinvest"
    r"|\bDRIP\b"
    r"|payroll\s+deduction"
    r"|automatic\s+(purchase|investment|acquisition)",
    re.IGNORECASE,
)


# Hand-tuned point value for each scoring component. Keys match the "key"
# field _component_flags() returns for each fired condition; _score_cluster
# maps fired flags -> {"delta": weights[key], ...} and drops zero-delta
# entries, so a component with weight 0 is inert (never scored, never shown).
DEFAULT_WEIGHTS: dict[str, int] = {
    "median_value_high": 3,
    "median_value_low": -2,
    "identical_prices": -3,
    "tight_price_band": -2,
    "multi_date_spread": 1,
    "single_day_burst": -2,
    "fractional_shares": -2,
    "ten_percent_owner": 3,
    "directors_only": 2,
    "routine_footnote": -5,
    "plan_10b5_1": -1,
    "recent_ipo": -3,
    "big_stake_add": 2,
    "filed_promptly": 1,
    "filed_late": -1,
    "director_heavy": 2,
    # Candidate components - zero by default so they are inert until a
    # signal_weights.json (typically fit by backtest/signal_fit.py) assigns
    # them a non-zero weight.
    "three_plus_insiders": 0,
    "five_plus_insiders": 0,
    "large_total_value": 0,
    "officer_and_director": 0,
    "huge_stake_add": 0,
    "ten_pct_owner_multi": 0,
}

_ACTIVE_WEIGHTS: tuple[dict[str, int], str] | None = None


def _coerce_date(v) -> Optional[date]:
    """Normalize a filing_date/transaction_date value to a date object.

    Callers disagree on shape: the live scanner passes filing_date as a
    YYYYMMDD string and transaction_date as YYYY-MM-DD, while the backtest
    passes datetime.date objects for both. Returns None if unparseable.
    """
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    if isinstance(v, str):
        for fmt in ("%Y-%m-%d", "%Y%m%d"):
            try:
                return datetime.strptime(v, fmt).date()
            except ValueError:
                continue
    return None


def load_signal_weights(path: str | None = None) -> tuple[dict[str, int], str]:
    """Load signal weights, merged over DEFAULT_WEIGHTS.

    Precedence: `path` arg -> SIGNAL_WEIGHTS_PATH env var -> signal_weights.json
    in the current working directory -> DEFAULT_WEIGHTS.

    The resolved file may be the full fit payload written by
    backtest/signal_fit.py ({"version": 1, "generated_at": ..., "fit": {...},
    "weights": {...}}) or a bare flat {key: int} dict for hand-written
    overrides. Unknown keys are warned about and ignored. A missing or
    corrupt file falls back to DEFAULT_WEIGHTS with source "defaults".

    Returns (weights, source) where source is "defaults" or the resolved path.
    """
    candidate = path or os.environ.get("SIGNAL_WEIGHTS_PATH") or "signal_weights.json"
    if not os.path.exists(candidate):
        return dict(DEFAULT_WEIGHTS), "defaults"

    try:
        with open(candidate, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, ValueError) as exc:
        log.warning("Could not read signal weights from %s (%s) - using defaults", candidate, exc)
        return dict(DEFAULT_WEIGHTS), "defaults"

    raw_weights = payload.get("weights", payload) if isinstance(payload, dict) else None
    if not isinstance(raw_weights, dict):
        log.warning("Signal weights file %s has no usable weights dict - using defaults", candidate)
        return dict(DEFAULT_WEIGHTS), "defaults"

    merged = dict(DEFAULT_WEIGHTS)
    for key, value in raw_weights.items():
        if key not in DEFAULT_WEIGHTS:
            log.warning("Unknown signal weight key %r in %s - ignoring", key, candidate)
            continue
        try:
            merged[key] = int(value)
        except (TypeError, ValueError):
            log.warning("Non-integer weight %r for %r in %s - ignoring", value, key, candidate)
            continue
    return merged, candidate


def get_active_weights() -> dict[str, int]:
    """Lazily load + cache the process-wide active signal weights."""
    global _ACTIVE_WEIGHTS
    if _ACTIVE_WEIGHTS is None:
        _ACTIVE_WEIGHTS = load_signal_weights()
    return _ACTIVE_WEIGHTS[0]


def _component_flags(window: list[dict], cluster: dict) -> list[dict]:
    """Evaluate every scoring condition and return the ones that fired.

    Returns [{"key": str, "text": str}, ...]. `_score_cluster` maps these
    onto weighted deltas; keeping the fired-condition list separate from the
    weights lets the backtest score the same window under multiple weight
    sets without re-deriving the conditions.
    """
    flags: list[dict] = []

    # Per-insider median dollar value.
    values = [i["value"] for i in cluster["insiders"] if i["value"]]
    if values:
        median = sorted(values)[len(values) // 2]
        if median >= 100_000:
            if median >= 1_000_000_000:
                median_str = f"${median/1_000_000_000:.2f}B"
            elif median >= 1_000_000:
                median_str = f"${median/1_000_000:.2f}M"
            else:
                median_str = f"${median/1_000:.0f}K"
            flags.append({"key": "median_value_high", "text": f"Median {median_str} per insider"})
        elif median <= 5_000:
            flags.append({
                "key": "median_value_low", "text": f"Median ${median:,.0f} per insider (small)"
            })

    # Price uniformity across the window.
    prices = [t["price_per_share"] for t in window if t.get("price_per_share")]
    if len(prices) >= 3:
        unique = {round(p, 4) for p in prices}
        if len(unique) == 1:
            flags.append({
                "key": "identical_prices",
                "text": f"All {len(prices)} txs at identical price ${prices[0]:.4f}",
            })
        else:
            spread = (max(prices) - min(prices)) / max(prices)
            if spread < 0.005:
                flags.append({
                    "key": "tight_price_band", "text": f"All txs within {spread*100:.2f}% price band"
                })

    # Date spread.
    dates = sorted({t["transaction_date"] for t in window if t["transaction_date"]})
    if len(dates) >= 3:
        flags.append({"key": "multi_date_spread", "text": f"Spread across {len(dates)} dates"})
    elif len(dates) == 1 and len(window) >= 4:
        flags.append({
            "key": "single_day_burst", "text": f"All {len(window)} txs on one date"
        })

    # Fractional share counts (ESPP payroll math).
    frac = sum(
        1 for t in window
        if t.get("shares") and abs(t["shares"] - round(t["shares"])) > 1e-4
    )
    if frac >= max(2, len(window) // 2):
        flags.append({
            "key": "fractional_shares", "text": f"{frac}/{len(window)} txs have fractional shares"
        })

    # Role mix.
    if cluster.get("includes_ten_percent_owner"):
        flags.append({"key": "ten_percent_owner", "text": "Includes 10% owner"})
    if cluster.get("includes_director") and not cluster.get("includes_officer"):
        flags.append({"key": "directors_only", "text": "Directors only (no officers)"})
    if cluster.get("includes_officer") and cluster.get("includes_director"):
        flags.append({
            "key": "officer_and_director", "text": "Includes both an officer and a director"
        })

    # Footnote tiebreakers.
    fn_blob = " | ".join(t.get("footnote_text", "") for t in window if t.get("footnote_text"))
    if fn_blob and _ROUTINE_FOOTNOTE_RE.search(fn_blob):
        flags.append({
            "key": "routine_footnote", "text": "Footnote mentions ESPP / 401(k) / DRIP / payroll"
        })
    if any(t.get("is_10b5_1") for t in window):
        flags.append({
            "key": "plan_10b5_1", "text": "Includes Rule 10b5-1 plan transaction(s)"
        })

    # Recent IPO — lockup expiries and post-listing director top-ups masquerade
    # as conviction clusters.
    if cluster.get("is_recent_ipo"):
        flags.append({
            "key": "recent_ipo",
            "text": f"Recent IPO (<{ipo_lookup.RECENT_IPO_DAYS}d since first trade)",
        })

    # Buy-size vs prior stake. Large average % increase = conviction; a huge
    # single-insider add (>=100%, i.e. more than doubling their stake) is
    # its own (currently inert) signal.
    pcts = [t["pct_of_prior_stake"] for t in window
            if t.get("pct_of_prior_stake") is not None]
    if pcts:
        avg_pct = sum(pcts) / len(pcts)
        if avg_pct >= 25:
            flags.append({
                "key": "big_stake_add",
                "text": f"Large position add (avg {avg_pct:.0f}% of prior stake)",
            })
    max_pct = cluster.get("max_pct_of_prior_stake")
    if max_pct is None and pcts:
        max_pct = max(pcts)
    if max_pct is not None and max_pct >= 100:
        flags.append({
            "key": "huge_stake_add",
            "text": f"Single insider more than doubled their stake (max {max_pct:.0f}%)",
        })

    # Filing promptness. SEC rule is 2 business days; sweep-up admin filings
    # land much later.
    delays: list[int] = []
    for t in window:
        fd = _coerce_date(t.get("filing_date"))
        td = _coerce_date(t.get("transaction_date"))
        if fd is None or td is None:
            continue
        delays.append((fd - td).days)
    if delays:
        delays.sort()
        median_delay = delays[len(delays) // 2]
        if median_delay <= 2:
            flags.append({
                "key": "filed_promptly", "text": f"Filed promptly (median {median_delay}d)"
            })
        elif median_delay >= 10:
            flags.append({
                "key": "filed_late", "text": f"Filed late (median {median_delay}d)"
            })

    # Independent-director-heavy. Pure directors (not also officers) carry
    # more signal than execs whose buys can be exercise-and-hold flavored.
    pure_directors = {
        (t["owner_cik"] or t["owner_name"])
        for t in window
        if t.get("is_director") and not t.get("is_officer")
    }
    n_insiders = cluster.get("num_insiders", 0)
    if n_insiders >= 2 and len(pure_directors) * 2 > n_insiders:
        flags.append({
            "key": "director_heavy",
            "text": f"Independent-director-heavy ({len(pure_directors)}/{n_insiders})",
        })

    # Insider-count breadth.
    if n_insiders >= 3:
        flags.append({"key": "three_plus_insiders", "text": f"{n_insiders} insiders buying"})
    if n_insiders >= 5:
        flags.append({
            "key": "five_plus_insiders", "text": f"{n_insiders} insiders buying (large cluster)"
        })

    # Total cluster dollar value.
    total_value = cluster.get("total_value") or 0
    if total_value >= 1_000_000:
        flags.append({
            "key": "large_total_value",
            "text": f"Cluster total value ${total_value/1_000_000:.2f}M+",
        })

    # 10% owner buying alongside a broader cluster (vs. a lone 10% owner).
    if cluster.get("includes_ten_percent_owner") and n_insiders >= 3:
        flags.append({
            "key": "ten_pct_owner_multi",
            "text": "10% owner buying alongside 3+ other insiders",
        })

    return flags


def _score_cluster(
    window: list[dict], cluster: dict, weights: dict[str, int] | None = None
) -> list[dict]:
    """Return a list of contributions: [{"key": str, "delta": int, "text": str}, ...].

    Score is the sum of deltas. The dashboard renders each contribution
    inline so the user can see exactly how the score was assembled.
    `weights` defaults to get_active_weights(); components whose weight is 0
    are dropped, so the default configuration renders identically to the
    pre-parameterized scorer.
    """
    if weights is None:
        weights = get_active_weights()
    contributions: list[dict] = []
    for flag in _component_flags(window, cluster):
        delta = weights.get(flag["key"], 0)
        if delta == 0:
            continue
        contributions.append({"key": flag["key"], "delta": delta, "text": flag["text"]})
    return contributions


def _label_for_score(score: int) -> str:
    if score >= 3:
        return "conviction"
    if score <= -3:
        return "routine"
    return "mixed"


def _build_cluster(window: list[dict]) -> dict:
    issuer_name = window[0]["issuer_name"]
    ticker = window[0]["ticker"]
    issuer_cik = window[0]["issuer_cik"]
    insiders: dict[str, dict] = {}
    for tx in window:
        key = tx["owner_cik"] or tx["owner_name"]
        slot = insiders.setdefault(key, {
            "name": tx["owner_name"],
            "roles": tx["owner_roles"],
            "shares": 0.0,
            "value": 0.0,
        })
        slot["shares"] += tx["shares"] or 0
        slot["value"] += tx["value"] or 0
    dates = sorted(t["transaction_date"] for t in window if t["transaction_date"])
    edgar_issuer_url = (
        f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={issuer_cik}"
        f"&type=4&dateb=&owner=include&count=40"
    )
    pct_values: list[float] = []
    for t in window:
        p = t.get("pct_of_prior_stake")
        if p is not None:
            pct_values.append(p)
    max_pct_of_prior_stake = max(pct_values) if pct_values else None
    first_trade = ipo_lookup.get_first_trade_date(ticker) if ticker else None
    cluster = {
        "issuer_cik": issuer_cik,
        "issuer_name": issuer_name,
        "ticker": ticker,
        "cluster_start": dates[0],
        "cluster_end": dates[-1],
        "num_insiders": len(insiders),
        "insiders": [
            {"name": v["name"], "roles": v["roles"],
             "shares": v["shares"], "value": v["value"]}
            for v in insiders.values()
        ],
        "total_shares": sum(v["shares"] for v in insiders.values()),
        "total_value": sum(v["value"] for v in insiders.values()),
        "max_pct_of_prior_stake": max_pct_of_prior_stake,
        "includes_ten_percent_owner": any(t["is_ten_percent_owner"] for t in window),
        "includes_director": any(t["is_director"] for t in window),
        "includes_officer": any(t["is_officer"] for t in window),
        "num_transactions": len(window),
        "transactions": window,
        "edgar_url": edgar_issuer_url,
        "first_trade_date": first_trade.isoformat() if first_trade else None,
        "days_since_ipo": ipo_lookup.days_since_ipo(first_trade),
        "is_recent_ipo": ipo_lookup.is_recent_ipo(first_trade),
    }
    contributions = _score_cluster(window, cluster)
    contributions.sort(key=lambda c: abs(c["delta"]), reverse=True)
    score = sum(c["delta"] for c in contributions)
    cluster["conviction_score"] = score
    cluster["conviction_label"] = _label_for_score(score)
    cluster["conviction_contributions"] = contributions
    return cluster


# ---------------------------------------------------------------------------
# Model-based live scoring (research/live_score.py) -- replaces
# conviction_score in the screener's presentation and sort. See that
# module's own docstring for the evidence behind the banded verdict, and
# this module's PART 1/2/3 task notes for why: conviction_score has been
# measured at zero risk-adjusted edge (0/5 folds, excess -0.24pp vs a
# volatility-matched benchmark). DEFAULT_WEIGHTS / _score_cluster /
# _component_flags above are left completely alone -- backtest/state.py
# depends on them for the experimental CONTROL that demonstrates the model
# beats the old approach, and _build_cluster still populates
# conviction_score/conviction_label/conviction_contributions on every
# cluster for that reason. This section only decides what the SCREENER
# shows and how it sorts; it never touches those fields.
# ---------------------------------------------------------------------------
RESEARCH_DATA_DIR = "research_data"
# Preferred first: the screening ensemble (research/screen_model.py) is the
# score research.live_score's four verdict bands (elevated_risk / middle /
# top_band / above_band) were measured against. Loading the other bundle
# instead falls back to that score's own retired two-state banding, which is
# the correct behaviour but a materially different reading of the same
# percentile -- 95 is the best badge under one and an explicitly-not-better
# badge under the other. Hence the warning when the fallback is what runs.
SCREEN_MODEL_GLOB = "screen_model_*.joblib"
PRODUCTION_MODEL_GLOB = "production_model_*.joblib"
RESEARCH_HISTORY_GLOB = "research_*rows_*.parquet"

# Live price fetch is a network cost proportional to the number of distinct
# tickers in a scrape. Opt out with LIVE_SCORE_FETCH_PRICES=0 (env or .env,
# see _load_dotenv) or --no-prices on the command line; a disabled or
# failed fetch degrades to the reduced (31/50 or 35/50) feature set rather
# than failing the run -- see attach_model_scores.
LIVE_SCORE_FETCH_PRICES_ENV = "LIVE_SCORE_FETCH_PRICES"

# research.live_score's price-context features look back up to 252 trading
# days (x_mom_252_skip5, x_drawdown_252) plus a 200-day SMA. 400 calendar
# days comfortably covers that including weekends/holidays -- the same
# cushion backtest.py's own Phase 3 price fetch uses.
PRICE_LOOKBACK_CALENDAR_DAYS = 400


def _resolve_latest_artifact(pattern: str, out_dir: str = RESEARCH_DATA_DIR) -> Optional[str]:
    """Newest file matching `pattern` under `out_dir` by mtime, or None.
    Mirrors run_research.py's own _resolve_latest -- same convention, so a
    freshly-produced production_model_*.joblib / research_*.parquet is
    always picked up here without a code change."""
    matches = glob.glob(os.path.join(out_dir, pattern))
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


def _price_fetch_enabled() -> bool:
    if "--no-prices" in sys.argv:
        return False
    raw = os.environ.get(LIVE_SCORE_FETCH_PRICES_ENV, "1").strip().lower()
    return raw not in ("0", "false", "no")


def build_live_price_universe(clusters: list[dict]):
    """Fetch price history for every distinct ticker in `clusters` and
    return a backtest.prices.PriceUniverse, or None if there is nothing to
    fetch or the fetch failed outright. Always logs elapsed time. Returning
    None (rather than raising) is deliberate: a screener run must never
    fail wholesale because prices were unavailable -- see attach_model_scores,
    which falls back to the reduced feature set and says so in the output."""
    tickers = sorted({c["ticker"] for c in clusters if c.get("ticker")})
    if not tickers:
        return None

    earliest = min(
        (_coerce_date(c.get("cluster_end")) or date.today()) for c in clusters
    )
    start = earliest - timedelta(days=PRICE_LOOKBACK_CALENDAR_DAYS)
    end = date.today()

    from backtest.prices import PriceUniverse
    t0 = time.monotonic()
    pu = PriceUniverse()
    try:
        pu.ensure(tickers, start, end)
        pu.finalize()
    except Exception as exc:
        elapsed = time.monotonic() - t0
        log.warning(
            "Live price fetch failed after %.1fs for %d ticker(s) (%s) - "
            "falling back to the reduced feature set.",
            elapsed, len(tickers), exc,
        )
        return None
    elapsed = time.monotonic() - t0
    log.info("Live price fetch: %d ticker(s) requested, %d loaded, %.1fs",
              len(tickers), len(pu.frames), elapsed)
    return pu


def score_clusters_with_model(
    clusters: list[dict], *, bundle=None, prices=None, issuer_history=None,
) -> None:
    """Score every cluster with the trained ranking model
    (research.live_score.score_live_cluster) and attach the result in
    place as cluster["model_score"]. If `bundle` is None this sets every
    cluster's model_score to None (build_html/write_excel render that as
    "not scored", never as a crash). A per-cluster scoring failure is
    caught and logged the same way -- one bad cluster degrades to
    unscored, it does not take the whole run down.
    """
    if bundle is None:
        for c in clusters:
            c["model_score"] = None
        return

    from research import live_score as ls

    # Counted per verdict rather than as one "good" tally: the current score
    # emits four bands and the retired one emits two, and which set appears
    # follows the bundle that loaded. A hardcoded counter for either set would
    # silently read zero under the other -- which is exactly what happened to
    # this log line when the score was replaced.
    from collections import Counter

    verdict_counts: "Counter[str]" = Counter()
    n_ok = 0
    for c in clusters:
        window = c.get("transactions") or []
        if not window:
            c["model_score"] = None
            continue
        try:
            result = ls.score_live_cluster(
                c, window, bundle, prices=prices, issuer_history=issuer_history,
            )
        except Exception as exc:
            log.warning("Model scoring failed for %s (%s): %s",
                        c.get("ticker"), c.get("issuer_name"), exc)
            c["model_score"] = None
            continue

        c["model_score"] = {
            "raw_score": result.raw_score,
            "percentile": result.percentile,
            "verdict": result.verdict.value,
            "n_features_available": len(result.availability.computed),
            "n_features_total": result.availability.n_total,
            "missing_features": list(result.availability.missing),
            "n_training_scores": int(len(bundle.training_scores)),
            "factors": [
                {
                    "feature": f.feature, "value": f.value, "ic": f.ic,
                    "favorable_direction": f.favorable_direction,
                    "description": f.description, "available": f.available,
                }
                for f in result.factors
            ],
        }
        n_ok += 1
        verdict_counts[result.verdict.value] += 1

    log.info(
        "Model-scored %d/%d cluster(s): %s",
        n_ok, len(clusters),
        ", ".join(f"{n} {v}" for v, n in sorted(verdict_counts.items()))
        or "none",
    )


def attach_model_scores(clusters: list[dict]) -> dict:
    """Load the production model bundle + issuer-history frame, optionally
    fetch live prices, score every cluster in place, and return a summary
    dict describing what was actually available this run -- consumed by
    build_payload/render_html/write_excel to render an honest coverage
    note (Part 3's "surface feature availability" requirement). Never
    raises: every failure mode here (no bundle on disk, no price data, no
    issuer-history frame, a network hiccup) degrades to a smaller feature
    set or to model_score=None, never to a crashed run (Part 1's "degrade
    gracefully" requirement).
    """
    from research import model as rm
    from research import screen_model

    # SCREEN_MODEL_GLOB is tried before PRODUCTION_MODEL_GLOB -- see the
    # comment on that constant above. LIVE_SCORE_MODEL_PATH is an explicit
    # override and outranks both, same as it always has.
    bundle_path = (
        os.environ.get("LIVE_SCORE_MODEL_PATH")
        or _resolve_latest_artifact(SCREEN_MODEL_GLOB)
        or _resolve_latest_artifact(PRODUCTION_MODEL_GLOB)
    )
    bundle = None
    if bundle_path and os.path.exists(bundle_path):
        try:
            bundle = rm.load_production_bundle(bundle_path)
            if screen_model.is_screen_bundle(bundle):
                log.info(
                    "Loaded screening ensemble bundle %s (%d training scores, %d features)",
                    bundle_path, len(bundle.training_scores), len(bundle.feature_cols),
                )
            else:
                log.warning(
                    "Loaded the retired single-classifier bundle %s (%d training scores, %d "
                    "features) - this score was measured to rank crash risk UPWARD, not down. "
                    "Run `python run_research.py --fit-screen` to produce the current score.",
                    bundle_path, len(bundle.training_scores), len(bundle.feature_cols),
                )
        except Exception as exc:
            log.warning(
                "Could not load model bundle %s (%s) - clusters will not be model-scored.",
                bundle_path, exc,
            )
            bundle = None
    else:
        log.warning(
            "No model bundle found (%s/%s or %s) - clusters will not be model-scored.",
            RESEARCH_DATA_DIR, SCREEN_MODEL_GLOB, PRODUCTION_MODEL_GLOB,
        )

    issuer_history = None
    issuer_history_path = None
    if bundle is not None:
        issuer_history_path = os.environ.get("LIVE_SCORE_HISTORY_PATH") or _resolve_latest_artifact(RESEARCH_HISTORY_GLOB)
        if issuer_history_path and os.path.exists(issuer_history_path):
            try:
                issuer_history = rm.load_research_dataset(issuer_history_path)
                log.info("Loaded issuer-history reference frame %s (%d rows)",
                          issuer_history_path, len(issuer_history))
            except Exception as exc:
                log.warning(
                    "Could not load issuer-history frame %s (%s) - issuer-history features unavailable.",
                    issuer_history_path, exc,
                )
                issuer_history = None
        else:
            log.warning(
                "No research history parquet found (%s/%s) - issuer-history features unavailable.",
                RESEARCH_DATA_DIR, RESEARCH_HISTORY_GLOB,
            )

    price_fetch_enabled = _price_fetch_enabled()
    prices_universe = None
    price_elapsed = 0.0
    if bundle is not None and clusters and price_fetch_enabled:
        t0 = time.monotonic()
        prices_universe = build_live_price_universe(clusters)
        price_elapsed = time.monotonic() - t0
    elif bundle is not None and clusters and not price_fetch_enabled:
        log.info(
            "Live price fetch disabled (%s=0 or --no-prices) - clusters will be "
            "scored on the reduced (no price-context) feature set.",
            LIVE_SCORE_FETCH_PRICES_ENV,
        )

    score_clusters_with_model(clusters, bundle=bundle, prices=prices_universe, issuer_history=issuer_history)

    return {
        "model_available": bundle is not None,
        "bundle_path": bundle_path if bundle is not None else None,
        "n_training_scores": int(len(bundle.training_scores)) if bundle is not None else 0,
        "n_model_features": len(bundle.feature_cols) if bundle is not None else 0,
        "price_fetch_enabled": price_fetch_enabled,
        "prices_loaded": prices_universe is not None,
        "n_tickers_priced": len(prices_universe.frames) if prices_universe is not None else 0,
        "price_fetch_elapsed_sec": round(price_elapsed, 1),
        "issuer_history_available": issuer_history is not None,
        "issuer_history_path": issuer_history_path if issuer_history is not None else None,
        "n_issuer_history_rows": int(len(issuer_history)) if issuer_history is not None else 0,
    }


# ---------------------------------------------------------------------------
# Output: xlsx + clusters JSON
# ---------------------------------------------------------------------------
HEADER_FILL = PatternFill(start_color="BDD7EE", end_color="BDD7EE", fill_type="solid")
HEADER_FONT = Font(bold=True)
HYPERLINK_FONT = Font(color="0563C1", underline="single")
TEN_PCT_FILL = PatternFill(start_color="FFF2CC", end_color="FFF2CC", fill_type="solid")


def _style_header(ws, columns: list[str]) -> None:
    for col_idx, name in enumerate(columns, start=1):
        cell = ws.cell(row=1, column=col_idx, value=name)
        cell.font = HEADER_FONT
        cell.fill = HEADER_FILL
        cell.alignment = Alignment(horizontal="center", vertical="center")
    ws.auto_filter.ref = f"A1:{get_column_letter(len(columns))}1"
    ws.freeze_panes = "A2"


def _autosize(ws, columns: list[str], data: list[list]) -> None:
    for col_idx, name in enumerate(columns, start=1):
        max_len = len(str(name))
        for row in data:
            v = row[col_idx - 1] if col_idx - 1 < len(row) else ""
            if v is None:
                continue
            ln = len(str(v))
            if ln > max_len:
                max_len = ln
        ws.column_dimensions[get_column_letter(col_idx)].width = min(max(max_len + 2, 10), 60)


def write_excel(clusters: list[dict], all_rows: list[dict],
                errors: list[dict], path: str, model_info: dict | None = None) -> None:
    wb = Workbook()

    # --- Sheet 1: Flagged Clusters ---
    ws1 = wb.active
    ws1.title = "Flagged Clusters"
    # Signal Score / Signal Drivers (the old conviction_score presentation)
    # are gone from this sheet -- replaced by the model's percentile,
    # verdict, feature coverage, and per-factor reasons, matching the HTML
    # dashboard (build_html.render_html). conviction_score/_contributions
    # are still computed on every cluster (see attach_model_scores' own
    # docstring / the module-level note above write_excel) but are no
    # longer part of this project's user-facing output.
    cluster_cols = [
        "Issuer Name", "Ticker", "Issuer CIK",
        "Cluster Start", "Cluster End", "# Insiders",
        "Insiders (names + roles)", "Total Shares Acquired", "Total $ Value",
        "Max % of Prior Stake",
        "Model Percentile", "Model Verdict", "Feature Coverage", "Model Reasons",
        "Includes 10% Owner", "Includes Director", "Includes Officer",
        "# Transactions", "EDGAR Issuer Filings URL",
    ]
    _style_header(ws1, cluster_cols)

    sorted_clusters = sorted(clusters, key=model_sort_key)
    cluster_rows = []
    for c in sorted_clusters:
        insider_str = "; ".join(
            f"{i['name']} [{i['roles']}] ({int(i['shares']):,} sh)"
            for i in c["insiders"]
        )
        ms = c.get("model_score")
        percentile = ms.get("percentile") if ms else None
        percentile_val = percentile if percentile is not None and percentile == percentile else ""
        verdict_str = verdict_label(ms.get("verdict")) if ms else verdict_label("not_scored")
        n_avail = ms.get("n_features_available") if ms else None
        n_total = ms.get("n_features_total") if ms else None
        coverage_str = f"{n_avail}/{n_total}" if ms else ""
        reasons_bits = []
        for f in (ms.get("factors") or []) if ms else []:
            if not f.get("available"):
                continue
            fav = factor_favorable(f["feature"], f["value"], f["favorable_direction"])
            fav_str = "FAVORABLE" if fav is True else "unfavorable" if fav is False else "n/a"
            value_str = fmt_factor_value(f["feature"], f["value"])
            reasons_bits.append(f"{fav_str}: {f['description']} (value={value_str})")
        reasons_str = "; ".join(reasons_bits)
        cluster_rows.append([
            c["issuer_name"], c["ticker"], c["issuer_cik"],
            c["cluster_start"], c["cluster_end"], c["num_insiders"],
            insider_str, c["total_shares"], c["total_value"],
            c.get("max_pct_of_prior_stake"),
            percentile_val, verdict_str, coverage_str, reasons_str,
            "YES" if c["includes_ten_percent_owner"] else "",
            "YES" if c["includes_director"] else "",
            "YES" if c["includes_officer"] else "",
            c["num_transactions"], c["edgar_url"],
        ])
    for row_idx, row in enumerate(cluster_rows, start=2):
        for col_idx, val in enumerate(row, start=1):
            cell = ws1.cell(row=row_idx, column=col_idx, value=val)
            if col_idx == len(cluster_cols):  # URL column
                cell.hyperlink = val
                cell.font = HYPERLINK_FONT
            if col_idx in (8, 9):
                cell.number_format = "#,##0"
            if col_idx == 10:
                cell.number_format = '0.0"%"'
            if col_idx == 11:
                cell.number_format = '0.0'
        if sorted_clusters[row_idx - 2]["includes_ten_percent_owner"]:
            for col_idx in range(1, len(cluster_cols) + 1):
                if col_idx == len(cluster_cols):
                    continue
                ws1.cell(row=row_idx, column=col_idx).fill = TEN_PCT_FILL
    _autosize(ws1, cluster_cols, cluster_rows)

    # --- Sheet 2: All Qualifying Transactions ---
    ws2 = wb.create_sheet("All Qualifying Transactions")
    tx_cols = [
        "Issuer", "Ticker", "Insider Name", "Role(s)",
        "Form Type", "Transaction Date", "Transaction Code", "A/D",
        "Shares", "Price", "$ Value", "Shares Owned After",
        "% of Prior Stake",
        "Filing Date", "Accession", "Filing URL",
    ]
    _style_header(ws2, tx_cols)
    tx_rows = []
    for r in sorted(all_rows, key=lambda r: r["transaction_date"], reverse=True):
        tx_rows.append([
            r["issuer_name"], r["ticker"], r["owner_name"], r["owner_roles"],
            r["form_type"], r["transaction_date"], r["transaction_code"],
            r["acquired_disposed"], r["shares"], r["price_per_share"],
            r["value"], r["shares_owned_after"],
            r.get("pct_of_prior_stake"),
            r["filing_date"], r["adsh"], r["filing_url"],
        ])
    for row_idx, row in enumerate(tx_rows, start=2):
        for col_idx, val in enumerate(row, start=1):
            cell = ws2.cell(row=row_idx, column=col_idx, value=val)
            if col_idx == len(tx_cols):
                cell.hyperlink = val
                cell.font = HYPERLINK_FONT
            if col_idx in (9, 11, 12):
                cell.number_format = "#,##0"
            if col_idx == 10:
                cell.number_format = "#,##0.0000"
            if col_idx == 13:
                cell.number_format = '0.0"%"'
    _autosize(ws2, tx_cols, tx_rows)

    # --- Sheet 3: Errors ---
    ws3 = wb.create_sheet("Errors")
    err_cols = ["Accession", "Form Type", "Filing URL", "Error"]
    _style_header(ws3, err_cols)
    err_rows = [[e["adsh"], e["form_type"], e["filing_url"], e["error"]] for e in errors]
    for row_idx, row in enumerate(err_rows, start=2):
        for col_idx, val in enumerate(row, start=1):
            cell = ws3.cell(row=row_idx, column=col_idx, value=val)
            if col_idx == 3:
                cell.hyperlink = val
                cell.font = HYPERLINK_FONT
    _autosize(ws3, err_cols, err_rows)

    # --- Sheet 4: Model Coverage -- mirrors the HTML dashboard's always-
    # visible "What this ranking means" banner (build_html.render_html), so
    # the Excel artifact carries the same honesty-about-feature-coverage
    # disclosure even for a reader who never opens the HTML.
    ws4 = wb.create_sheet("Model Coverage")
    mi = model_info or {}
    coverage_rows = [
        ["Model available", "YES" if mi.get("model_available") else "NO"],
        ["Model bundle path", mi.get("bundle_path") or ""],
        ["Training scores (historical reference distribution)", mi.get("n_training_scores", 0)],
        ["Model features", mi.get("n_model_features", 0)],
        ["Live price fetch enabled", "YES" if mi.get("price_fetch_enabled") else "NO"],
        ["Live prices loaded", "YES" if mi.get("prices_loaded") else "NO"],
        ["Tickers priced", mi.get("n_tickers_priced", 0)],
        ["Price fetch time (s)", mi.get("price_fetch_elapsed_sec", 0)],
        ["Issuer-history reference frame available", "YES" if mi.get("issuer_history_available") else "NO"],
        ["Issuer-history reference frame path", mi.get("issuer_history_path") or ""],
        ["Issuer-history reference rows", mi.get("n_issuer_history_rows", 0)],
        ["", ""],
        [findings.headline(), ""],
    ]
    ws4.append(["Key", "Value"])
    for cell in ws4[1]:
        cell.font = HEADER_FONT
        cell.fill = HEADER_FILL
    for row in coverage_rows:
        ws4.append(row)
    ws4.column_dimensions["A"].width = 70
    ws4.column_dimensions["B"].width = 30

    # --- Sheet 5: What To Expect -- the measured base rates, so the Excel
    # artifact carries the same honesty the dashboard's "What to expect"
    # panel does. Numbers come from findings.py, never inline here.
    ws5 = wb.create_sheet("What To Expect")
    exp_cols = ["If you hold for", "vs S&P 500", "vs small-cap index",
                "vs micro-cap index", "Chance of a gain"]
    _style_header(ws5, exp_cols)
    exp_rows = [list(t) for t in findings.expectation_rows()]
    for row_idx, row in enumerate(exp_rows, start=2):
        for col_idx, val in enumerate(row, start=1):
            ws5.cell(row=row_idx, column=col_idx, value=val)
    _autosize(ws5, exp_cols, exp_rows)

    r = len(exp_rows) + 3
    ws5.cell(row=r, column=1, value="Refinements tested that did NOT help "
                                    "(median 3-week result by quartile, worst to best)").font = HEADER_FONT
    for label, quartiles in findings.FLAT_REFINEMENTS:
        r += 1
        ws5.cell(row=r, column=1, value=label)
        ws5.cell(row=r, column=2, value="  ".join(f"{v * 100:+.2f}%" for v in quartiles))
    r += 2
    ws5.cell(row=r, column=1, value="Before you act on any of this").font = HEADER_FONT
    for point in findings.key_points():
        r += 1
        ws5.cell(row=r, column=1, value=point)
    r += 2
    ws5.cell(row=r, column=1, value=f"Source: {findings.PROVENANCE}. Measured "
                                    f"{findings.MEASURED_ON}. Round-trip cost assumed "
                                    f"{findings.ROUND_TRIP_COST * 1e4:.0f}bps. "
                                    f"Full record in RESEARCH_NOTES.md.")

    wb.save(path)


def build_payload(clusters: list[dict],
                  scanned_range: tuple[date, date],
                  model_info: dict | None = None) -> dict:
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "scanned_from": scanned_range[0].isoformat(),
        "scanned_to": scanned_range[1].isoformat(),
        "qualifying_codes": sorted(QUALIFYING_CODES),
        "clusters": clusters,
        "model_info": model_info or {},
    }


# ---------------------------------------------------------------------------
# Concurrent parsing
# ---------------------------------------------------------------------------
def _parse_one(filing: dict) -> dict:
    """Wrap parse_filing so worker threads return either a parsed dict or an
    error sentinel - never raise."""
    try:
        adsh = _accession_from_filename(filing["file_name"])
    except ValueError as exc:
        return {"_error": True, "adsh": "?", "form_type": filing["form_type"],
                "filing_url": "", "error": str(exc)}
    try:
        return parse_filing(filing)
    except Exception as exc:
        return {
            "_error": True,
            "adsh": adsh,
            "form_type": filing["form_type"],
            "filing_url": _filing_html_url(filing["cik"], adsh),
            "error": f"{type(exc).__name__}: {exc}",
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    lookback, min_insiders, window_days = _prompt_inputs()

    weights, weights_source = load_signal_weights()
    log.info("Signal weights source: %s (%d active component(s))",
             weights_source, sum(1 for v in weights.values() if v))

    if USER_AGENT.startswith("Your Name"):
        log.warning(
            "USER_AGENT is the placeholder - SEC may rate-limit or block you. "
            "Set SEC_USER_AGENT in .env to a real name + email."
        )

    filings = discover_filings(lookback)
    parsed: list[dict] = []
    errors: list[dict] = []

    total = len(filings)
    log.info("Parsing %d filings with %d workers @ %.0f req/s",
             total, MAX_WORKERS, RATE_LIMIT)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(_parse_one, f) for f in filings]
        for i, fut in enumerate(as_completed(futures), start=1):
            result = fut.result()
            if result.get("_error"):
                result.pop("_error", None)
                errors.append(result)
            else:
                parsed.append(result)
            if i % 100 == 0 or i == total:
                log.info("Parsed %d/%d filings", i, total)

    log.info("Parsed %d filings (%d errors)", len(parsed), len(errors))

    qualifying = extract_qualifying_rows(parsed)
    log.info("Qualifying acquisition rows: %d (codes=%s)",
             len(qualifying), sorted(QUALIFYING_CODES))

    clusters = detect_clusters(qualifying, min_insiders, window_days)
    log.info("Detected %d cluster(s)", len(clusters))

    # Model-based live scoring (Part 1+2): loads the production bundle +
    # issuer-history frame, optionally fetches live prices, and scores
    # every cluster in place. Degrades gracefully -- see attach_model_scores.
    model_info = attach_model_scores(clusters)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    xlsx_path = os.path.join(OUTPUT_DIR, "insider_cluster_buys.xlsx")
    html_path = os.path.join(OUTPUT_DIR, "dashboard.html")

    write_excel(clusters, qualifying, errors, xlsx_path, model_info=model_info)
    payload = build_payload(
        clusters,
        (date.today() - timedelta(days=lookback), date.today()),
        model_info=model_info,
    )
    with open(html_path, "w", encoding="utf-8") as fh:
        fh.write(render_html(payload))

    log.info("Wrote %s, %s", xlsx_path, html_path)
    print()
    print(f"Done. {len(clusters)} flagged cluster(s) across {len(qualifying)} qualifying tx.")
    if model_info.get("model_available"):
        n_scored = sum(1 for c in clusters if c.get("model_score"))
        verdicts = [
            (c.get("model_score") or {}).get("verdict") for c in clusters
        ]
        print(f"  Model:     {n_scored}/{len(clusters)} scored")

        # Which summary to print follows the verdicts that were actually
        # emitted, not a constant in findings.py. A retired bundle bands into
        # top_decile/no_edge and never produces a top_band row, so keying off
        # findings (which always defines the current bands) would print zeros
        # for a run that scored perfectly well under the older score.
        current = {"top_band", "above_band", "middle", "elevated_risk"}
        if current & set(v for v in verdicts if v):
            top = findings.BANDS_BY_VERDICT["top_band"]
            risk = findings.BANDS_BY_VERDICT["elevated_risk"]
            n_top = sum(1 for v in verdicts if v == "top_band")
            n_risk = sum(1 for v in verdicts if v == "elevated_risk")
            # The count to look at, then the count to avoid, in that order.
            # The avoid count is the one with durable evidence behind it.
            print(f"  Top band:  {n_top} in the 70th-90th percentile -- the "
                  f"best-measured band ({top.p_loses_30pct * 100:.1f}% "
                  f"historical 30%-loss rate over 21 days)")
            print(f"  Avoid:     {n_risk} in the bottom 30% "
                  f"({risk.p_loses_30pct * 100:.1f}% historical 30%-loss rate, "
                  f"and 7.3-9.2% in every year measured)")
        else:
            n_old = sum(1 for v in verdicts if v == "top_decile")
            crash_hi = findings.SHIPPED_MODEL_CRASH_RATE_BY_DECILE[-1] * 100
            print(f"  Retired score in use -- {n_old} in its top decile "
                  f"({crash_hi:.0f}% historical 30%-loss rate; highest risk, "
                  f"not best).")
            print("  Run `python run_research.py --fit-screen` for the "
                  "current score.")
    else:
        print("  Model:     no production model bundle found -- clusters are unscored")
    print(f"  Excel:     {xlsx_path}")
    print(f"  Dashboard: {html_path}")

    # The one thing a user most needs to carry away, printed every run rather
    # than buried in the HTML. See findings.py for the measurements.
    best = findings.HORIZON_EXPECTATIONS[0]
    print()
    print("  " + "-" * 68)
    print(f"  {findings.band_headline()}")
    print(f"  Held ~{best.trading_days} trading days, the average flagged cluster has "
          f"historically returned")
    print(f"  {best.vs_iwm * 100:+.1f}%/yr vs a small-cap index fund and "
          f"{best.vs_spy * 100:+.1f}%/yr vs the S&P 500.")
    print(f"  Holding longer has been worse, not better ({findings.HORIZON_EXPECTATIONS[2].trading_days}d: "
          f"{findings.HORIZON_EXPECTATIONS[2].vs_iwm * 100:+.1f}%/yr vs small caps).")
    print(f"  See the 'What to expect' panel in {html_path}, or the")
    print(f"  'What To Expect' sheet in the workbook, for the full picture.")
    print("  " + "-" * 68)


if __name__ == "__main__":
    main()
