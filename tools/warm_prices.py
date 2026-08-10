"""Fill price_cache/ for every ticker in a scraped events parquet.

Run this once before a research session. The backtest and the research
dataset both need daily bars for about 7,500 tickers. A cold fetch of that
universe takes hours because of yfinance rate limits. A warm cache makes the
price phase almost free, so model iteration costs seconds instead of a day.

Usage:
    python tools/warm_prices.py                      # widest file in clusters_history/
    python tools/warm_prices.py --events <path>      # a specific parquet
    python tools/warm_prices.py --limit 200          # first N tickers, for a smoke test

The cache is incremental. Stop this script and start it again at any time.
Tickers that are already cached over the requested range are skipped.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date, timedelta

import pandas as pd

# Run directly (`python tools/warm_prices.py ...`), the interpreter puts
# this file's own directory (tools/) on sys.path[0], not the repo root -- so
# `from backtest import ...` below would fail without this. Harmless no-op
# when this module is instead imported normally, since the repo root is
# already on sys.path in that case.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from backtest import history, prices, tickers as tickers_module

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("warm_prices")

# Forward cushion so a 365-trading-day lot opened near the end of the window
# still has an exit bar.
FORWARD_CUSHION_DAYS = 400
BACKWARD_CUSHION_DAYS = 400


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--events", default="latest",
                    help="Events parquet path, or 'latest' (default)")
    ap.add_argument("--limit", type=int, default=0,
                    help="Only warm the first N tickers (0 = all)")
    ap.add_argument("--as-of", default="",
                    help="End date YYYY-MM-DD (default: today)")
    args = ap.parse_args()

    as_of = date.today()
    if args.as_of:
        as_of = date.fromisoformat(args.as_of)

    path = history.resolve_events_path(args.events)
    df = pd.read_parquet(path, columns=["ticker", "transaction_date"])
    # Use the same normalizer the backtest uses, so the warmed cache is keyed
    # exactly like the symbols the run will ask for. A raw pass would both
    # waste requests on junk like '[N/A]' and miss the recovered form of a
    # malformed symbol, for example 'NYSE: KRC' -> 'KRC'.
    # Test for str, not truthiness. Unusable symbols come back as None or as
    # NaN once pandas touches the column, and NaN is truthy.
    normalized = tickers_module.normalize_ticker_series(df["ticker"])
    tickers = sorted({t for t in normalized if isinstance(t, str) and t})
    if args.limit:
        tickers = tickers[: args.limit]
    tickers.append("SPY")

    # Reach back before the first transaction so momentum and volatility
    # features have history at the first event.
    start = df["transaction_date"].min() - timedelta(days=BACKWARD_CUSHION_DAYS)
    end = as_of + timedelta(days=FORWARD_CUSHION_DAYS)

    log.info("Warming %d tickers over %s .. %s", len(tickers), start, end)
    pu = prices.PriceUniverse()
    pu.ensure(tickers, start, end)
    pu.finalize()

    loaded = len(pu.frames)
    n_missing = sum(1 for e in pu.missing.values()
                    if e.get("reason", "missing") == "missing")
    n_rl = sum(1 for e in pu.missing.values() if e.get("reason") == "rate_limited")
    log.info("Done. %d/%d tickers cached, %d missing, %d rate-limited",
             loaded, len(tickers), n_missing, n_rl)
    if n_rl:
        log.warning("Run this script again to retry the %d rate-limited tickers.", n_rl)
    return 0


if __name__ == "__main__":
    sys.exit(main())
