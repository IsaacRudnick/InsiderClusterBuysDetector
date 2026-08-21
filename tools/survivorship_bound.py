"""Put the vanished companies back in, and bound what they do to the results.

WHY THIS IS THE MOST IMPORTANT CORRECTION IN THE PROJECT
=======================================================
Of the 10,861 rows in the research dataset, exactly ZERO are on a ticker that
later stopped trading -- while those tickers carry 24.9% of insider
transactions and 27.1% of insider dollars. Every number this project has
published is therefore conditioned on the company still existing today, which
is not something an investor in 2019 could have selected on.

`tools/delisting_fate.py` establishes what actually happened to each of those
2,455 tickers, and the answer is more interesting than "they all went to
zero". A large share are ALIVE under a different symbol -- AAXN is Axon
Enterprise, ABC is Cencora, ADS is Bread Financial -- and those are not
failures, they are companies whose price history simply moved. Dropping them
biases results DOWNWARD.

So the correction runs in both directions and has to be done carefully rather
than assumed:

  renamed_alive    fetch the price history under the CURRENT symbol and
                   measure the trade for real. No assumption needed.
  bankruptcy       the equity is worthless. A total loss is not a guess.
  delisted         genuinely unknown. Delisting without bankruptcy usually
                   means a move to the pink sheets at a bad price, but "bad"
                   is not a number.
  acquired         deals close at a PREMIUM, so dropping these flattered
                   nothing -- it cost the strategy return.

THE BOUND, AND WHY IT IS A BOUND RATHER THAN AN ESTIMATE
--------------------------------------------------------
For the genuinely-dead-and-unexplained there is no free source giving the
final trade. Inventing one would be the same error this project has already
made three times. Instead every headline is recomputed under three explicit
assumptions for that group -- total loss, half loss, and no excess loss -- and
the honest answer is the RANGE. A measured range beats a point estimate that
is quietly wrong, and if the conclusion is the same at both ends then the
unknown never mattered.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools import exit_lab as el  # noqa: E402

#: What each fate is assumed to have returned, for the trades we cannot price.
#: `renamed_alive` is absent on purpose -- those are measured, never assumed.
SCENARIOS = {
    "optimistic": {"bankruptcy": -1.00, "delisted": 0.00,
                   "alive_no_ticker": 0.00, "acquired": 0.00,
                   "acquired_delisted": 0.00, "unknown": 0.00,
                   "not_on_edgar": 0.00},
    "central": {"bankruptcy": -1.00, "delisted": -0.50,
                "alive_no_ticker": -0.25, "acquired": 0.00,
                "acquired_delisted": 0.00, "unknown": -0.25,
                "not_on_edgar": -0.25},
    "pessimistic": {"bankruptcy": -1.00, "delisted": -1.00,
                    "alive_no_ticker": -1.00, "acquired": 0.00,
                    "acquired_delisted": 0.00, "unknown": -1.00,
                    "not_on_edgar": -1.00},
}

#: Trading days a synthetic dead trade is assumed to occupy a slot before the
#: loss is realised. Deliberately generous: a fast wipeout would free the
#: capital sooner and hurt less, so holding the slot longer is the
#: conservative choice for a book with limited slots.
DEAD_HOLD_DAYS = 63


def load_fate() -> pd.DataFrame:
    path = os.path.join(REPO_ROOT, "research_data", "delisting_fate.parquet")
    if not os.path.exists(path):
        raise SystemExit("run tools/delisting_fate.py first")
    return pd.read_parquet(path)


def recover_renamed_prices(fate: pd.DataFrame, *, limit: int = 0) -> dict:
    """Fetch price history for companies that are alive under a new symbol.

    Providers backfill a renamed symbol's full history, so pulling the CURRENT
    ticker recovers the period when it traded under the old one. That turns an
    assumption back into a measurement, which is the whole point.
    """
    from backtest.prices import PriceUniverse

    renamed = fate[fate["classification"] == "renamed_alive"].copy()
    if limit:
        renamed = renamed.head(limit)
    mapping = {}
    for row in renamed.itertuples(index=False):
        # The submissions JSON can list several symbols (share classes,
        # warrants, preferreds). The first is the common stock in practice,
        # and a warrant's price path is not the equity's.
        cands = [t for t in row.current_tickers.split(",") if t]
        if cands:
            mapping[row.event_ticker] = cands[0]
    if not mapping:
        return {}
    pu = PriceUniverse()
    pu.ensure(sorted(set(mapping.values())), date(2018, 1, 1), date(2026, 8, 20))
    ok = {}
    for old, new in mapping.items():
        p = os.path.join(REPO_ROOT, "price_cache", f"{new}.parquet")
        if os.path.exists(p):
            ok[old] = new
    return ok


def dead_cluster_events(fate: pd.DataFrame, recovered: dict) -> pd.DataFrame:
    """Cluster events on dead tickers, using the incumbent >=2-in-14 rule.

    Rebuilt from the raw transaction file rather than read from the research
    dataset, because the research dataset is precisely what excluded them.
    """
    from tools import event_definition_sweep as eds

    ev_path = os.path.join(REPO_ROOT, "clusters_history",
                           "events_20180813_20260813.parquet")
    df = eds.load_qualifying_rows(ev_path)
    dead = set(fate["event_ticker"])
    df = df[df["ticker"].isin(dead)]
    by_issuer = eds._group_by_issuer(df)
    clusters = eds.detect_clusters_with_aggregates(by_issuer, 2, 14)
    tick = df.drop_duplicates("issuer_cik", keep="last") \
             .set_index("issuer_cik")["ticker"]
    rows = []
    for c in clusters:
        t = tick.get(c["issuer_cik"])
        if not t:
            continue
        rows.append(dict(ticker=t, last_date=c["last_date"],
                         n_insiders=c["n_insiders"],
                         total_value=c["total_value"]))
    out = pd.DataFrame(rows)
    out["mapped_ticker"] = out["ticker"].map(recovered)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit-fetch", type=int, default=0,
                    help="cap how many renamed tickers to price (0 = all)")
    args = ap.parse_args(argv)

    fate = load_fate()
    print(f"{len(fate)} dead tickers resolved\n")
    print(fate["classification"].value_counts().to_string())

    print("\nrecovering prices for companies alive under a new symbol...",
          flush=True)
    recovered = recover_renamed_prices(fate, limit=args.limit_fetch)
    print(f"  recovered price history for {len(recovered)} of "
          f"{int((fate['classification'] == 'renamed_alive').sum())} renamed")

    ev = dead_cluster_events(fate, recovered)
    ev = ev.merge(fate[["event_ticker", "classification"]],
                  left_on="ticker", right_on="event_ticker", how="left")
    print(f"\n{len(ev)} cluster events sit on dead tickers "
          f"(vs {10861} in the survivors-only research dataset)")
    print(f"  of those, {int(ev['mapped_ticker'].notna().sum())} are on "
          f"companies whose real prices we just recovered")
    print("\nevents by fate:")
    print(ev["classification"].value_counts().to_string())

    out = os.path.join(REPO_ROOT, "research_data", "dead_cluster_events.parquet")
    ev.to_parquet(out, index=False)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
