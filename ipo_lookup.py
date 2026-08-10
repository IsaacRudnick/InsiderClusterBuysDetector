"""
Cached lookup of an issuer's first-trade date via yfinance, used to flag
recent IPOs whose insider activity is dominated by lockup expiries,
selling-shareholder follow-ons, and director top-ups rather than conviction
buys.

One JSON cache file per ticker in CACHE_DIR. IPO dates are immutable, so
positive hits are kept forever. Negative results (yfinance can't resolve the
symbol) are kept for NEGATIVE_TTL_DAYS before retrying.
"""

import json
import logging
import os
import time
from datetime import date, datetime, timezone
from typing import Optional

try:
    import yfinance as yf
except Exception:
    yf = None

log = logging.getLogger(__name__)

CACHE_DIR = "ipo_cache"
NEGATIVE_TTL_DAYS = 7
RECENT_IPO_DAYS = 180


def _cache_path(ticker: str) -> str:
    safe = "".join(c for c in ticker.upper() if c.isalnum() or c in "-._")
    return os.path.join(CACHE_DIR, f"{safe}.json")


def _read_cache(ticker: str) -> Optional[dict]:
    path = _cache_path(ticker)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def _write_cache(ticker: str, first_trade_date: Optional[str]) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    payload = {
        "ticker": ticker.upper(),
        "first_trade_date": first_trade_date,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    try:
        with open(_cache_path(ticker), "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
    except OSError as e:
        log.warning("Failed writing IPO cache for %s: %s", ticker, e)


def _negative_cache_expired(entry: dict) -> bool:
    try:
        fetched = datetime.fromisoformat(entry["fetched_at"])
    except (KeyError, ValueError):
        return True
    age_days = (datetime.now(timezone.utc) - fetched).days
    return age_days >= NEGATIVE_TTL_DAYS


def _epoch_to_date(epoch_seconds) -> Optional[date]:
    try:
        return datetime.fromtimestamp(int(epoch_seconds), tz=timezone.utc).date()
    except (TypeError, ValueError, OSError):
        return None


def _fetch_first_trade_date(ticker: str) -> Optional[date]:
    if yf is None:
        return None
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:
        return None
    for key in ("firstTradeDateEpochUtc", "firstTradeDateMilliseconds"):
        raw = info.get(key)
        if raw is None:
            continue
        if key == "firstTradeDateMilliseconds":
            raw = raw / 1000
        d = _epoch_to_date(raw)
        if d:
            return d
    inception = info.get("fundInceptionDate")
    if inception:
        return _epoch_to_date(inception)
    return None


def get_first_trade_date(ticker: str) -> Optional[date]:
    """Return the issuer's first-trade (or fund-inception) date, or None.

    Cache-first: positive hits are permanent, negative hits expire after
    NEGATIVE_TTL_DAYS. Returns None if the ticker is blank, yfinance is not
    installed, or the symbol can't be resolved.
    """
    if not ticker:
        return None
    ticker = ticker.strip().upper()
    if not ticker:
        return None

    cached = _read_cache(ticker)
    if cached is not None:
        cached_date = cached.get("first_trade_date")
        if cached_date:
            try:
                return date.fromisoformat(cached_date)
            except ValueError:
                pass
        elif not _negative_cache_expired(cached):
            return None

    fetched = _fetch_first_trade_date(ticker)
    _write_cache(ticker, fetched.isoformat() if fetched else None)
    return fetched


def is_recent_ipo(first_trade: Optional[date], today: Optional[date] = None,
                  threshold_days: int = RECENT_IPO_DAYS) -> bool:
    if first_trade is None:
        return False
    if today is None:
        today = date.today()
    return (today - first_trade).days < threshold_days


def days_since_ipo(first_trade: Optional[date], today: Optional[date] = None) -> Optional[int]:
    if first_trade is None:
        return None
    if today is None:
        today = date.today()
    return (today - first_trade).days
