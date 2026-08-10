"""yfinance-backed price data layer with parquet cache.

Per-ticker parquet under price_cache/{TICKER}.parquet; meta json tracks the
date range we've already fetched so reruns only top up forward.

Negative-lookup cache distinguishes two reasons:
  - "missing"       : yfinance can't resolve / no price data. TTL'd (7 days).
  - "rate_limited"  : transient throttle. ALWAYS retried on the next run.

yfinance's batch downloader logs per-ticker failures via the Python logging
system (e.g. "YFRateLimitError: Too Many Requests"). We attach a context-
managed handler around each batch call, parse those messages to classify
each failure, and adapt the inter-batch throttle accordingly.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from bisect import bisect_left
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from typing import Iterable, Iterator, Optional

import pandas as pd

try:
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None

log = logging.getLogger(__name__)

CACHE_DIR = "price_cache"
META_DIR = os.path.join(CACHE_DIR, "_meta")
MISSING_PATH = os.path.join(CACHE_DIR, "_missing.json")
NEEDS_REFETCH_PATH = os.path.join(CACHE_DIR, "_needs_refetch.json")
TZ_CACHE_DIR = os.path.join(CACHE_DIR, "_yf_tz")

# Cache-miss marker for median_dollar_volume, whose valid results include None.
_MDV_UNSET = object()

# Cache-miss marker for median_share_volume, whose valid results include None.
_MSV_UNSET = object()

# yfinance keeps a SQLite timezone/cookie cache. The default location under
# %LOCALAPPDATA% is shared by every yfinance process on the machine, and
# concurrent batch downloads make it raise "database is locked". A locked
# cache makes yf.download return an EMPTY frame with no error message, so the
# ticker looks missing. The 2026-07-18 run lost 2,618 of 7,594 tickers this
# way. Point the cache at a project-local directory to isolate it.
if yf is not None:
    try:
        os.makedirs(TZ_CACHE_DIR, exist_ok=True)
        yf.set_tz_cache_location(TZ_CACHE_DIR)
    except Exception as exc:  # pragma: no cover - best effort
        log.warning("Could not relocate yfinance tz cache: %s", exc)
NEGATIVE_TTL_DAYS = 7
DOWNLOAD_BATCH = 25            # smaller batches reduce rate-limit blast radius
MAX_BATCH_RETRIES = 3          # internal yf.download retries on hard exceptions
MAX_PASSES = 5                 # how many times to retry rate-limited tickers

# Ordinary dividend-adjustment drift between two auto_adjust=True fetches is
# a small fraction of a percent to low-single-digits at most. A real split
# is never subtle: it is a halving/doubling or an integer-ish factor (or its
# reciprocal). A 2% band cleanly separates the two, so anything past it is
# treated as a basis change rather than noise.
_BASIS_CHANGE_TOLERANCE = 0.02


def _safe_ticker(t: str) -> str:
    return "".join(c for c in t.upper() if c.isalnum() or c in "-._")


def _cache_path(ticker: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{_safe_ticker(ticker)}.parquet")


def _meta_path(ticker: str) -> str:
    os.makedirs(META_DIR, exist_ok=True)
    return os.path.join(META_DIR, f"{_safe_ticker(ticker)}.json")


def _load_missing() -> dict:
    if not os.path.exists(MISSING_PATH):
        return {}
    try:
        with open(MISSING_PATH, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}


def _save_missing(d: dict) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(MISSING_PATH, "w", encoding="utf-8") as fh:
        json.dump(d, fh, indent=2)


def _load_needs_refetch() -> dict:
    """Registry of tickers whose cache needs a full, single-basis refetch.

    A ticker lands here when a merge finds no overlapping date between the
    cached frame and a fresh fetch, so no adjustment-basis ratio can be
    inferred between them. See `PriceUniverse._merge_price_frames`.
    """
    if not os.path.exists(NEEDS_REFETCH_PATH):
        return {}
    try:
        with open(NEEDS_REFETCH_PATH, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}


def _save_needs_refetch(d: dict) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(NEEDS_REFETCH_PATH, "w", encoding="utf-8") as fh:
        json.dump(d, fh, indent=2)


def _missing_is_fresh(entry: dict) -> bool:
    """Rate-limited entries always retry. Genuinely-missing entries TTL out."""
    reason = entry.get("reason", "missing")
    if reason == "rate_limited":
        return False
    try:
        fetched = datetime.fromisoformat(entry["last_attempt"])
    except (KeyError, ValueError):
        return False
    if fetched.tzinfo is None:
        fetched = fetched.replace(tzinfo=timezone.utc)
    age = (datetime.now(timezone.utc) - fetched).days
    return age < NEGATIVE_TTL_DAYS


def _load_meta(ticker: str) -> dict:
    p = _meta_path(ticker)
    if not os.path.exists(p):
        return {}
    try:
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}


def _save_meta(ticker: str, first_date: date, last_date: date) -> None:
    with open(_meta_path(ticker), "w", encoding="utf-8") as fh:
        json.dump({
            "ticker": ticker.upper(),
            "first_date": first_date.isoformat(),
            "last_date": last_date.isoformat(),
            "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }, fh)


def _normalize_history(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance returns a tz-aware DatetimeIndex with Title-case columns. Flatten."""
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    idx = df.index
    if hasattr(idx, "tz") and idx.tz is not None:
        idx = idx.tz_localize(None)
    df.index = pd.DatetimeIndex(idx).normalize()
    df.index.name = "date"

    # yfinance returns MultiIndex columns whenever group_by is set, including
    # for a SINGLE ticker, where the caller has no ticker level to select
    # away first. Without this, str() of a ('AAGR', 'Open') tuple matches no
    # OHLCV name, `keep` comes out empty, and the frame silently normalizes
    # to zero rows. That made every single-ticker fetch look like a delisting.
    # Pick the level that actually holds the field names rather than assuming
    # an order, because group_by='ticker' and group_by='column' swap them.
    if isinstance(df.columns, pd.MultiIndex):
        fields = {"open", "high", "low", "close", "volume", "adj close"}
        level = next(
            (
                lv
                for lv in range(df.columns.nlevels)
                if {str(v).lower() for v in df.columns.get_level_values(lv)} & fields
            ),
            df.columns.nlevels - 1,
        )
        df.columns = df.columns.get_level_values(level)

    df.columns = [str(c).lower() for c in df.columns]
    # A single-ticker frame can repeat a field if the source returned one
    # column per ticker level. Keep the first, so `keep` selects a Series.
    df = df.loc[:, ~df.columns.duplicated()]
    keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    df = df[keep].dropna(how="all")
    df["dollar_volume"] = df.get("close", 0) * df.get("volume", 0)
    return df


# ---------------------------------------------------------------------------
# Failure classification from yfinance log output
# ---------------------------------------------------------------------------
_BRACKET_RE = re.compile(r"\[([^\]]+)\]")
_RATE_LIMIT_RE = re.compile(r"rate.?limit|too\s+many\s+requests|429", re.IGNORECASE)
_MISSING_RE = re.compile(
    r"delisted|no\s+timezone\s+found|no\s+price\s+data\s+found|symbol\s+may\s+be\s+delisted",
    re.IGNORECASE,
)


def _extract_tickers_from_log(msg: str) -> list[str]:
    """Pull '['ABP', 'BRK.B', ...]' style ticker lists from a log message."""
    out: list[str] = []
    for m in _BRACKET_RE.finditer(msg):
        for tok in m.group(1).split(","):
            tok = tok.strip().strip("'\"").strip()
            if tok and not tok.startswith("$"):
                out.append(tok)
    # Also catch '$XXX:' or '$XXX OTC:' prefixes (single-ticker errors)
    if not out:
        m = re.match(r"\s*\$([A-Z0-9.\-/:_ ]+):", msg)
        if m:
            out.append(m.group(1).strip())
    return out


class _YFLogCapture(logging.Handler):
    """Captures yfinance log records and any 'Failed download' messages from root."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
        except Exception:
            return
        if (
            record.name.startswith("yfinance")
            or "Failed download" in msg
            or "delisted" in msg.lower()
            or "RateLimit" in msg
            or "Too Many Requests" in msg
        ):
            self.messages.append(msg)


@contextmanager
def _capture_yf_errors() -> Iterator[_YFLogCapture]:
    handler = _YFLogCapture()
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        yield handler
    finally:
        root.removeHandler(handler)


def _classify_failures(messages: list[str], requested: list[str]
                       ) -> tuple[set[str], set[str]]:
    """Return (rate_limited, truly_missing) sets restricted to requested tickers."""
    req_upper = {t.upper() for t in requested}
    rate_limited: set[str] = set()
    missing: set[str] = set()
    for msg in messages:
        tickers = [t.upper() for t in _extract_tickers_from_log(msg)]
        # Tickers only count if they're actually in our batch
        tickers = [t for t in tickers if t in req_upper]
        if not tickers:
            continue
        if _RATE_LIMIT_RE.search(msg):
            rate_limited.update(tickers)
        elif _MISSING_RE.search(msg):
            missing.update(tickers)
        else:
            # Unknown failure mode — treat as transient (retry) rather than missing
            rate_limited.update(tickers)
    return rate_limited, missing


# ---------------------------------------------------------------------------
# Adaptive throttle
# ---------------------------------------------------------------------------
class _AdaptiveThrottle:
    """Inter-batch sleep that grows when rate limits hit, decays when clear."""

    def __init__(self) -> None:
        self.delay = 1.0
        self.min_delay = 0.5
        self.max_delay = 60.0
        self.consec_clean = 0
        self.consec_hot = 0

    def observe(self, rate_limited: int, batch_size: int) -> None:
        if batch_size <= 0:
            return
        ratio = rate_limited / batch_size
        if rate_limited == 0:
            self.consec_clean += 1
            self.consec_hot = 0
            if self.consec_clean >= 3:
                self.delay = max(self.min_delay, self.delay * 0.8)
                self.consec_clean = 0
        else:
            self.consec_clean = 0
            self.consec_hot += 1
            if ratio > 0.7:
                self.delay = min(self.max_delay, self.delay * 3 + 5)
            elif ratio > 0.3:
                self.delay = min(self.max_delay, self.delay * 2 + 2)
            else:
                self.delay = min(self.max_delay, self.delay + 1)

    def sleep(self) -> None:
        if self.delay > 0:
            time.sleep(self.delay)

    def cooldown(self, factor: float = 4.0) -> None:
        """Bigger sleep between full retry passes."""
        wait = min(self.max_delay * 2, max(5.0, self.delay * factor))
        log.info("Cooldown %.1fs before retrying rate-limited tickers", wait)
        time.sleep(wait)


# ---------------------------------------------------------------------------
# Batch download
# ---------------------------------------------------------------------------
def _download_batch(tickers: list[str], start: date, end: date
                    ) -> dict[str, pd.DataFrame]:
    """Single yf.download call for a batch of tickers. Returns ticker -> df."""
    if yf is None or not tickers:
        return {}
    out: dict[str, pd.DataFrame] = {}
    end_excl = end + timedelta(days=1)
    raw = None
    for attempt in range(MAX_BATCH_RETRIES):
        try:
            raw = yf.download(
                tickers=tickers,
                start=start.isoformat(),
                end=end_excl.isoformat(),
                auto_adjust=True,
                progress=False,
                threads=True,
                group_by="ticker",
                actions=False,
            )
            break
        except Exception as exc:
            log.warning("yf.download raised (attempt %d/%d): %s",
                        attempt + 1, MAX_BATCH_RETRIES, exc)
            time.sleep(3 * (attempt + 1))
    if raw is None or raw.empty:
        return {}
    if len(tickers) == 1:
        out[tickers[0]] = _normalize_history(raw)
        return out
    for t in tickers:
        try:
            sub = raw[t]
        except KeyError:
            continue
        norm = _normalize_history(sub)
        if not norm.empty:
            out[t] = norm
    return out


def _download_serial(tickers: list[str], start: date, end: date
                     ) -> dict[str, pd.DataFrame]:
    """Refetch tickers one at a time with threading off.

    A threaded batch download can drop a ticker for reasons that have nothing
    to do with the ticker: a locked timezone cache returns an empty frame and
    logs no error. Retrying serially separates a real delisting from batch
    noise, so `_fetch_and_cache` only records a ticker as missing after it
    fails on its own.
    """
    if yf is None or not tickers:
        return {}
    out: dict[str, pd.DataFrame] = {}
    end_excl = end + timedelta(days=1)
    for t in tickers:
        try:
            raw = yf.download(
                tickers=[t],
                start=start.isoformat(),
                end=end_excl.isoformat(),
                auto_adjust=True,
                progress=False,
                threads=False,
                group_by="ticker",
                actions=False,
            )
        except Exception as exc:
            log.debug("serial refetch of %s raised: %s", t, exc)
            continue
        if raw is None or raw.empty:
            continue
        norm = _normalize_history(raw)
        if not norm.empty:
            out[t] = norm
    return out


# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------
class PriceUniverse:
    """Holds price data for all tickers needed by the backtest."""

    def __init__(self) -> None:
        self.frames: dict[str, pd.DataFrame] = {}
        self.missing: dict = _load_missing()
        self.throttle = _AdaptiveThrottle()
        self.open_by_ticker: dict[str, dict[date, float]] = {}
        self.close_by_ticker: dict[str, dict[date, float]] = {}
        self.dv_by_ticker: dict[str, dict[date, float]] = {}
        self.volume_by_ticker: dict[str, dict[date, float]] = {}
        self.dates_by_ticker: dict[str, list[date]] = {}
        self._signal_cache: dict[tuple[str, date], dict] = {}
        # None is a real answer here (too little history), so the miss marker
        # has to be a distinct sentinel rather than None.
        self._mdv_cache: dict[tuple[str, date, int], Optional[float]] = {}
        self._msv_cache: dict[tuple[str, date, int], Optional[float]] = {}

    # ---- public API -------------------------------------------------------
    def ensure(self, tickers: Iterable[str], start: date, end: date) -> None:
        wanted = sorted({t.strip().upper() for t in tickers if t and t.strip()})
        to_fetch: list[str] = []
        for t in wanted:
            entry = self.missing.get(t)
            if entry and _missing_is_fresh(entry):
                continue  # cached as genuinely-missing within TTL → skip
            cached = self._load_cached(t)
            if cached is None:
                to_fetch.append(t)
                continue
            meta = _load_meta(t)
            try:
                cached_last = date.fromisoformat(meta.get("last_date", ""))
            except ValueError:
                cached_last = cached.index.max().date()
            if cached_last < end:
                to_fetch.append(t)

        if to_fetch:
            n_rl_retry = sum(
                1 for t in to_fetch
                if self.missing.get(t, {}).get("reason") == "rate_limited"
            )
            log.info(
                "Fetching prices for %d tickers (%d cached, %d retry-rate-limited) over %s..%s",
                len(to_fetch), len(wanted) - len(to_fetch), n_rl_retry, start, end,
            )
            self._fetch_and_cache(to_fetch, start, end)

        for t in wanted:
            if t in self.frames:
                continue
            cached = self._load_cached(t)
            if cached is not None and not cached.empty:
                self.frames[t] = cached

    # ---- cache I/O --------------------------------------------------------
    def _load_cached(self, ticker: str) -> Optional[pd.DataFrame]:
        path = _cache_path(ticker)
        if not os.path.exists(path):
            return None
        try:
            df = pd.read_parquet(path)
            df.index = pd.DatetimeIndex(df.index).normalize()
            # Re-apply any confirmed manual split correction for this ticker
            # (see price_overrides.py). Idempotent: a frame already on the
            # corrected basis is left alone, so re-reading an
            # already-adjusted cache file every call never double-adjusts it.
            from price_overrides import apply_price_overrides
            df = apply_price_overrides(ticker, df)
            return df
        except Exception as exc:
            log.warning("Corrupt price cache %s: %s", path, exc)
            return None

    def _save(self, ticker: str, df: pd.DataFrame) -> None:
        if df is None or df.empty:
            return
        df.to_parquet(_cache_path(ticker))
        _save_meta(ticker, df.index.min().date(), df.index.max().date())

    def _record_missing(self, ticker: str, reason: str) -> None:
        self.missing[ticker] = {
            "reason": reason,
            "last_attempt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

    # ---- merge --------------------------------------------------------------
    def _merge_price_frames(
        self, ticker: str, existing: Optional[pd.DataFrame], new: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge a fresh fetch into the cached frame, basis-aware.

        yfinance's auto_adjust=True adjustment basis shifts whenever a split
        happens after the last fetch. If we blindly weld old rows onto new
        rows, an overlapping date range on two different bases fabricates a
        price jump at the seam. This checks the overlap first and rescales
        the stale segment when a split is detected.
        """
        if existing is None:
            return new

        overlap = existing.index.intersection(new.index)
        if len(overlap) == 0:
            # No shared date means there is no way to infer an adjustment
            # ratio between the two frames. Concatenating them blindly could
            # silently splice two different bases together, so do not do
            # that. Record the ticker for a full, single-fetch refetch and
            # use `new` alone for now: it is internally self-consistent and
            # is the freshest data for the requested range. The stale
            # `existing` segment gets reconciled later by
            # repair_price_cache.py, which does a full refetch on one basis.
            log.warning(
                "%s: merge range [%s, %s] is disjoint from cached range "
                "[%s, %s]; needs a full refetch",
                ticker, new.index.min().date(), new.index.max().date(),
                existing.index.min().date(), existing.index.max().date(),
            )
            registry = _load_needs_refetch()
            now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
            entry = registry.get(ticker, {})
            entry["reason"] = "disjoint_merge_range"
            entry.setdefault("first_seen", now_iso)
            entry["last_attempt"] = now_iso
            registry[ticker] = entry
            _save_needs_refetch(registry)
            return new

        existing_close = existing.loc[overlap, "close"]
        new_close = new.loc[overlap, "close"]
        usable = existing_close.notna() & (existing_close > 0) & new_close.notna() & (new_close > 0)
        if not usable.any():
            # No usable pair to compute a ratio from. Fall back to the old
            # behavior: existing wins, only genuinely new dates get appended.
            return pd.concat(
                [existing, new[~new.index.isin(existing.index)]]
            ).sort_index()

        ratio = float((new_close[usable] / existing_close[usable]).median())

        if abs(ratio - 1.0) <= _BASIS_CHANGE_TOLERANCE:
            # Ordinary dividend-adjustment drift, not a split. Merge exactly
            # as before: keep existing rows, append only genuinely new dates.
            return pd.concat(
                [existing, new[~new.index.isin(existing.index)]]
            ).sort_index()

        # A split happened between fetches. Rows that only exist in
        # `existing` are on the stale basis and must be rescaled to `new`'s
        # basis before they can sit next to it. Overlapping dates come from
        # `new` outright; their stale `existing` counterparts are discarded.
        existing_only = existing[~existing.index.isin(new.index)].copy()
        if not existing_only.empty:
            for col in ("open", "high", "low", "close"):
                if col in existing_only.columns:
                    existing_only[col] = existing_only[col] * ratio
            if "volume" in existing_only.columns:
                existing_only["volume"] = existing_only["volume"] / ratio
            existing_only["dollar_volume"] = (
                existing_only.get("close", 0) * existing_only.get("volume", 0)
            )
            boundary = existing_only.index.max()
        else:
            boundary = new.index.min()

        log.warning(
            "%s: basis change detected across merge (ratio=%.4f); rescaling "
            "stale rows up to boundary %s",
            ticker, ratio, boundary.date(),
        )

        return pd.concat([existing_only, new]).sort_index()

    # ---- fetch ------------------------------------------------------------
    def _fetch_and_cache(self, tickers: list[str], start: date, end: date) -> None:
        pending = list(tickers)
        pass_n = 0
        while pending and pass_n < MAX_PASSES:
            pass_n += 1
            log.info(
                "Pass %d/%d: %d tickers to fetch (throttle delay %.1fs)",
                pass_n, MAX_PASSES, len(pending), self.throttle.delay,
            )
            still_rate_limited: list[str] = []

            for i in range(0, len(pending), DOWNLOAD_BATCH):
                batch = pending[i : i + DOWNLOAD_BATCH]
                with _capture_yf_errors() as cap:
                    fetched = _download_batch(batch, start, end)
                rate_limited, miss = _classify_failures(cap.messages, batch)

                # Anything the batch returned nothing for, and that was not
                # explicitly rate-limited, gets one serial retry before it can
                # be recorded as missing. This is what keeps a locked tz cache
                # from silently deleting a third of the universe.
                blanks = [
                    t for t in batch
                    if (fetched.get(t) is None or fetched[t].empty)
                    and t not in rate_limited
                ]
                n_rescued = 0
                if blanks:
                    with _capture_yf_errors() as cap2:
                        recovered = _download_serial(blanks, start, end)
                    fetched.update(recovered)
                    n_rescued = len(recovered)
                    # Re-classify the leftovers from the serial attempt.
                    still_blank = [t for t in blanks if t not in recovered]
                    rl2, miss2 = _classify_failures(cap2.messages, still_blank)
                    rate_limited |= rl2
                    miss = (miss | miss2) - set(recovered)

                fetched_ok = 0
                for t in batch:
                    new = fetched.get(t)
                    if new is None or new.empty:
                        if t in miss:
                            self._record_missing(t, "missing")
                        elif t in rate_limited:
                            still_rate_limited.append(t)
                            self._record_missing(t, "rate_limited")
                        else:
                            # No data and no error message: be conservative,
                            # treat as transient and retry.
                            still_rate_limited.append(t)
                            self._record_missing(t, "rate_limited")
                        continue
                    # Re-apply any confirmed manual split correction to the
                    # freshly fetched frame BEFORE it is merged against the
                    # cached one. `existing` (via _load_cached) already gets
                    # the same treatment; without doing it here too, a
                    # corrected `existing` merged against a still-broken raw
                    # `new` reads the correction itself as a spurious basis
                    # change (see _merge_price_frames) and discards it -- the
                    # exact way the 2026-08-07 DKI corruption happened.
                    from price_overrides import apply_price_overrides
                    new = apply_price_overrides(t, new)
                    existing = self._load_cached(t)
                    merged = self._merge_price_frames(t, existing, new)
                    self._save(t, merged)
                    self.missing.pop(t, None)
                    fetched_ok += 1

                self.throttle.observe(len(rate_limited), len(batch))
                log.info(
                    "  batch %3d/%d (%d tickers) ok=%d  rescued=%d  rl=%d  miss=%d  next_sleep=%.1fs",
                    i // DOWNLOAD_BATCH + 1,
                    (len(pending) + DOWNLOAD_BATCH - 1) // DOWNLOAD_BATCH,
                    len(batch), fetched_ok, n_rescued, len(rate_limited), len(miss),
                    self.throttle.delay,
                )
                _save_missing(self.missing)
                self.throttle.sleep()

            pending = still_rate_limited
            if pending and pass_n < MAX_PASSES:
                log.warning("Retrying %d rate-limited tickers on next pass", len(pending))
                self.throttle.cooldown()

        if pending:
            log.warning(
                "%d tickers remained rate-limited after %d passes; cached as "
                "'rate_limited' (will retry on next run)",
                len(pending), MAX_PASSES,
            )
        _save_missing(self.missing)

    # ---- finalize for fast lookup ----------------------------------------
    def finalize(self) -> None:
        for t, df in self.frames.items():
            d_index = [ts.date() for ts in df.index]
            self.dates_by_ticker[t] = d_index
            self.open_by_ticker[t] = dict(zip(d_index, df["open"].to_list()))
            self.close_by_ticker[t] = dict(zip(d_index, df["close"].to_list()))
            self.dv_by_ticker[t] = dict(zip(d_index, df["dollar_volume"].to_list()))
            self.volume_by_ticker[t] = dict(zip(d_index, df["volume"].to_list()))
        n_missing = sum(1 for e in self.missing.values()
                        if e.get("reason", "missing") == "missing")
        n_rl = sum(1 for e in self.missing.values()
                   if e.get("reason") == "rate_limited")
        log.info("Price universe ready: %d tickers loaded, %d missing, %d rate-limited",
                 len(self.frames), n_missing, n_rl)

    def trading_calendar(self, start: date, end: date, ref: str = "SPY") -> list[date]:
        if ref not in self.frames:
            raise RuntimeError(f"Reference ticker {ref!r} not loaded into universe")
        return [d for d in self.dates_by_ticker[ref] if start <= d <= end]

    def open(self, ticker: str, dt: date) -> Optional[float]:
        d = self.open_by_ticker.get(ticker)
        if d is None:
            return None
        v = d.get(dt)
        if v is None or v != v:
            return None
        return float(v)

    def close(self, ticker: str, dt: date) -> Optional[float]:
        d = self.close_by_ticker.get(ticker)
        if d is None:
            return None
        v = d.get(dt)
        if v is None or v != v:
            return None
        return float(v)

    def last_close_on_or_before(self, ticker: str, dt: date) -> Optional[float]:
        dates = self.dates_by_ticker.get(ticker)
        if not dates:
            return None
        for d in reversed(dates):
            if d <= dt:
                v = self.close_by_ticker[ticker].get(d)
                if v is not None and v == v:
                    return float(v)
        return None

    def next_trading_day(self, ticker: str, dt: date) -> Optional[date]:
        dates = self.dates_by_ticker.get(ticker)
        if not dates:
            return None
        for d in dates:
            if d > dt:
                return d
        return None

    def last_available_date(self, ticker: str) -> Optional[date]:
        dates = self.dates_by_ticker.get(ticker)
        if not dates:
            return None
        return dates[-1]

    def is_past_last_bar(self, ticker: str, dt: date) -> bool:
        last = self.last_available_date(ticker)
        return last is not None and dt > last

    def median_dollar_volume(self, ticker: str, dt: date, window: int = 20) -> Optional[float]:
        """Median dollar volume over the `window` bars strictly before `dt`.

        Memoized, and it binary-searches the date list instead of scanning it.
        The engine calls this per candidate per day per run, so the original
        linear scan over a ticker's full 2,000-bar history dominated the
        simulation loop.
        """
        key = (ticker, dt, window)
        cached = self._mdv_cache.get(key, _MDV_UNSET)
        if cached is not _MDV_UNSET:
            return cached

        result: Optional[float] = None
        dates = self.dates_by_ticker.get(ticker)
        if dates:
            # dates is ascending, so everything left of the insertion point of
            # `dt` is strictly earlier.
            cut = bisect_left(dates, dt)
            prior = dates[max(0, cut - window):cut]
            if prior:
                dv = self.dv_by_ticker[ticker]
                vals = sorted(
                    v for v in (dv.get(d, 0.0) for d in prior) if v == v
                )
                if vals:
                    result = float(vals[len(vals) // 2])

        self._mdv_cache[key] = result
        return result

    def median_share_volume(self, ticker: str, dt: date, window: int = 20) -> Optional[float]:
        """Median SHARE volume over the `window` bars strictly before `dt`.

        Same memoization, binary-search, and strictly-before-`dt` convention
        as median_dollar_volume above -- deliberately kept identical so the
        two accessors behave the same way under a split or a data gap.

        This exists because dollar volume alone cannot catch a collapsing
        low-priced name: SMFL cleared LIQUIDITY_FLOOR's 20-day median
        $-volume screen on 2024-09-20 (2024-09-12 alone traded $108M) while
        the stock itself changed hands only 316,699 times that day. The
        engine's participation cap (backtest/engine.py, MAX_PARTICIPATION_PCT)
        needs the share count, not the dollar figure, to catch that case.
        """
        key = (ticker, dt, window)
        cached = self._msv_cache.get(key, _MSV_UNSET)
        if cached is not _MSV_UNSET:
            return cached

        result: Optional[float] = None
        dates = self.dates_by_ticker.get(ticker)
        if dates:
            # dates is ascending, so everything left of the insertion point of
            # `dt` is strictly earlier.
            cut = bisect_left(dates, dt)
            prior = dates[max(0, cut - window):cut]
            if prior:
                vol = self.volume_by_ticker[ticker]
                vals = sorted(
                    v for v in (vol.get(d, 0.0) for d in prior) if v == v
                )
                if vals:
                    result = float(vals[len(vals) // 2])

        self._msv_cache[key] = result
        return result

    def price_signals(self, ticker: str, dt: date) -> dict:
        """Memoized momentum/vol/distance-from-90d-high snapshot for (ticker, dt).

        All values use close prices strictly before dt (no lookahead). Returns
        a dict with possibly-None fields when history is too short:
          - momentum_20d: trailing-20-trading-day total return (close-to-close).
          - vol_30d: annualized stdev of trailing-30-day daily returns.
          - dist_from_high_90d: (last_close / max_close_in_90d) - 1, ≤ 0.
        """
        key = (ticker, dt)
        cached = self._signal_cache.get(key)
        if cached is not None:
            return cached

        out = {
            "momentum_20d": None,
            "vol_30d": None,
            "dist_from_high_90d": None,
        }
        dates = self.dates_by_ticker.get(ticker)
        closes = self.close_by_ticker.get(ticker)
        if not dates or not closes:
            self._signal_cache[key] = out
            return out

        prior = [d for d in dates if d < dt]
        if not prior:
            self._signal_cache[key] = out
            return out
        last_d = prior[-1]
        last_px = closes.get(last_d)
        if last_px is None or last_px != last_px or last_px <= 0:
            self._signal_cache[key] = out
            return out

        # Momentum: close 20 trading days ago vs last close.
        if len(prior) >= 21:
            ref_px = closes.get(prior[-21])
            if ref_px is not None and ref_px == ref_px and ref_px > 0:
                out["momentum_20d"] = float(last_px / ref_px - 1.0)

        # Vol: stdev of last 30 daily returns, annualized.
        tail = prior[-31:] if len(prior) >= 31 else prior
        if len(tail) >= 11:
            tail_px = [closes.get(d) for d in tail]
            tail_px = [p for p in tail_px if p is not None and p == p and p > 0]
            if len(tail_px) >= 11:
                rets = [tail_px[i] / tail_px[i - 1] - 1.0 for i in range(1, len(tail_px))]
                if len(rets) >= 2:
                    mean = sum(rets) / len(rets)
                    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
                    out["vol_30d"] = float((var ** 0.5) * (252 ** 0.5))

        # Distance from 90-day high.
        win = prior[-90:]
        if win:
            highs = [closes.get(d) for d in win]
            highs = [p for p in highs if p is not None and p == p and p > 0]
            if highs:
                hi = max(highs)
                if hi > 0:
                    out["dist_from_high_90d"] = float(last_px / hi - 1.0)

        self._signal_cache[key] = out
        return out
