"""Point-in-time features mined from the FULL Form 4/4A transaction record in
parse_cache/ -- not just the code-P "qualifying" buys insider_cluster_buys.py
keeps for cluster detection.

Why this is free
-----------------
insider_cluster_buys.extract_qualifying_rows only keeps transaction code "P"
with acquired_disposed "A" (QUALIFYING_CODES = {"P"}); everything else in a
filing's transaction list is discarded at that step. But parse_cache/ itself
holds the FULL parsed non-derivative transaction list per filing -- every
code, not just P -- because parsing happens once per accession and the whole
filing is cached (see insider_cluster_buys.parse_ownership_xml). This module
is a second local reprocessing pass over the same 1.49M cached files,
exactly the pattern backtest/sales_history.py already established for the
code-S sale side. It makes zero network requests.

What the cache actually contains (investigated before writing a line of
feature code -- see the module-level report this file's __main__ prints, and
the task writeup for the full 25,000-file sample used to confirm this):
  - form_type is "4" or "4/A" only, nothing else, in a 25,000-file random
    sample (24,499 / 501). Matches insider_cluster_buys.INTERESTING_FORMS
    exactly, as expected -- only Form 4/4A filings were ever fetched.
  - EVERY cached file has exactly the same 9 top-level keys (adsh, form_type,
    filing_date, issuer, owners, transactions, holdings, filing_url, source)
    and every transaction has exactly the same 8 keys (date, code,
    acquired_disposed, shares, price_per_share, shares_owned_after,
    footnote_text, is_10b5_1). There is no `derivativeTable` /
    `derivativeTransaction` key anywhere, in any file, at any schema
    version. Table II (options, warrants, convertibles) was NEVER parsed by
    insider_cluster_buys.parse_ownership_xml -- it only walks
    `nonDerivativeTable/nonDerivativeTransaction` and
    `nonDerivativeTable/nonDerivativeHolding`. Re-parsing the raw XML would
    fix this, but the raw XML is not on disk -- only this normalized JSON
    cache is -- and the task is explicit that this is a LOCAL, no-network
    mining pass. x_derivative_buys_trail180 is therefore DROPPED. See the
    "Features dropped" section below.
  - Transaction codes ARE genuinely diverse: a 25,000-file / 41,095-
    transaction sample gave S 11,936 (29.0%), A 8,783 (21.4%), F 6,742
    (16.4%), M 5,865 (14.3%), P 4,061 (9.9%), J 1,162, G 834, D 788, C 716,
    plus a long thin tail (L/X/U/I/W/Z, each under 80). Every code this
    module needs (P, A, M, S) is plentiful.
  - is_10b5_1 is populated on 18.1% of ALL transactions (any code), derived
    from a footnote-text regex at parse time (insider_cluster_buys.
    _TEN_B5_1_RE) -- present and usable.
  - `owners` is a flat per-FILING list, not exploded per-transaction; 97.2%
    of sampled filings have exactly one owner. This module uses owners[0]
    for every transaction in a filing, identical to insider_cluster_buys.
    extract_qualifying_rows's and backtest.sales_history's existing
    "owner = owners[0]" convention -- kept consistent rather than inventing
    a second, different attribution rule.

Which (code, acquired_disposed) pairs this module keeps, and why
------------------------------------------------------------------
  P / A  -- open-market buy. Same definition insider_cluster_buys.py already
            uses for cluster detection; re-derived here (not imported) so
            this module has zero dependency on the live scanner's global
            QUALIFYING_CODES mutable.
  A / A  -- grant/award. Compensation, not conviction -- but a codeword for
            "how much is this issuer handing out right now", which is a
            different question than "is anyone buying".
  M / A  -- option exercise. The acquisition leg only; whether the insider
            then held or sold is determined separately (see
            x_exercise_and_hold_trail180 below), by looking for a follow-on
            S/D by the SAME owner within the exercise-and-hold window.
  S / D  -- open-market/private sale. Same definition backtest.sales_history
            already uses (see that module's docstring for why F/D and D/D
            are excluded: sell-to-cover tax withholding and dispositions
            back to the issuer are not discretionary market sales).
Everything else (F, D, G, C, J, L, X, U, I, W, Z) is dropped at the scan
step -- none of the eight features below need them.

Point-in-time discipline (the load-bearing rule of this whole module)
-----------------------------------------------------------------------
EVERY feature uses only filings whose FILING date is STRICTLY BEFORE
event_day. Not <=. The task that produced this module was explicit about
this, and it is deliberately a notch stricter than backtest.research's own
`_window_for_ticker_day`, which allows filing_date <= D (a same-day filing
counts as known) -- see that function's docstring. This module instead
mirrors tools/earnings_features.py's `_prior_slice` convention (`side='left'`
on a sorted filing-date array, which excludes a same-day filing). The
transaction itself may be dated (transaction_date) years before event_day,
but if nobody could have read about it on EDGAR until on/after event_day,
the market did not know it either -- a feature that peeks at a Form 4 filed
after the cluster it is describing manufactures a fake edge that a live
model will never actually earn, because in production the future filing
genuinely has not happened yet.

The trailing-N-day WINDOW (as opposed to the point-in-time GATE) is applied
to transaction_date, matching backtest.research's own Section D/F window
convention: transaction_date in [event_day - N, event_day). Both conditions
(filing_date < event_day AND transaction_date in the trailing window) must
hold for a transaction to count anywhere in this module.

x_exercise_and_hold_trail180's extra subtlety: an exercise cannot be scored
"held" just because no follow-on sale has been SEEN yet -- that would credit
the model with information about the future (whether a sale is coming) that
does not exist yet either way. An exercise less than EXERCISE_HOLD_WINDOW_
DAYS (5) old as of event_day has an INCOMPLETE observation window and is
excluded from the count entirely (neither "held" nor "sold"), not defaulted
to "held". See _classify_exercise.

Features built (8 requested; 1 dropped)
------------------------------------------
    x_insider_grants_trail180     count of code-A grants, issuer-wide
    x_option_exercises_trail180   count of code-M exercises, issuer-wide
    x_exercise_and_hold_trail180  count of those exercises NOT followed by a
                                   same-owner code-S sale within 5 days (and
                                   with a complete 5-day observation window)
    x_insider_sales_trail180      count of code-S sales, issuer-wide
    x_sale_to_buy_ratio           trailing sale $ / trailing buy $ (both
                                   issuer-wide, 180d); NaN if trailing buy $
                                   is zero (undefined, not "infinitely bad")
    x_n_new_insiders_trail365     count of owners whose issuer-wide FIRST
                                   EVER filing (any code) falls in the prior
                                   365 days
    x_frac_buys_10b5_1_trail180   fraction of trailing code-P buys flagged
                                   10b5-1; NaN if there were no trailing buys
    x_derivative_buys_trail180    DROPPED -- Table II was never parsed into
                                   parse_cache (see above). Column is not
                                   emitted.

Runnable standalone as `python tools/form4_extras.py` (scans parse_cache/,
attaches features to the research parquet, writes research_data/
form4_extras.parquet and prints the coverage/descriptive report) or
importable, e.g. from a test, via `attach_features`.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import insider_cluster_buys as ics  # noqa: E402  (reuses PARSE_CACHE_DIR)

log = logging.getLogger("form4_extras")

PARSE_CACHE_DIR = ics.PARSE_CACHE_DIR  # "parse_cache" -- reuse the existing constant

DEFAULT_EVENTS = os.path.join(REPO_ROOT, "research_data", "research_10861rows_20260813.parquet")
OUT_DIR = os.path.join(REPO_ROOT, "research_data")
TX_CACHE_GLOB = "form4_extras_tx_*rows_*.parquet"
OWNER_CACHE_GLOB = "form4_extras_owners_*rows_*.parquet"

# (code, acquired_disposed) pairs kept from the non-derivative transaction
# list -- see module docstring for why each one is here.
BUY_CODE, BUY_AD = "P", "A"
GRANT_CODE, GRANT_AD = "A", "A"
EXERCISE_CODE, EXERCISE_AD = "M", "A"
SALE_CODE, SALE_AD = "S", "D"
_KEEP_PAIRS = {
    (BUY_CODE, BUY_AD), (GRANT_CODE, GRANT_AD), (EXERCISE_CODE, EXERCISE_AD), (SALE_CODE, SALE_AD),
}

TRAIL_180_DAYS = 180
TRAIL_365_DAYS = 365
EXERCISE_HOLD_WINDOW_DAYS = 5

LOG_EVERY = 250_000
DEFAULT_MAX_WORKERS = 16

_TX_COLUMNS = [
    "adsh", "issuer_cik", "owner_key", "code", "acquired_disposed",
    "transaction_date", "filing_date", "shares", "price_per_share", "value", "is_10b5_1",
]
_OWNER_COLUMNS = ["issuer_cik", "owner_key", "filing_date"]

NEW_FEATURE_COLS = [
    "x_insider_grants_trail180",
    "x_option_exercises_trail180",
    "x_exercise_and_hold_trail180",
    "x_insider_sales_trail180",
    "x_sale_to_buy_ratio",
    "x_n_new_insiders_trail365",
    "x_frac_buys_10b5_1_trail180",
]


# ---------------------------------------------------------------------------
# 1. Local parse_cache/ scan -- one file read produces both outputs
# ---------------------------------------------------------------------------
def _parse_filing_date(raw: Optional[str]) -> Optional[date]:
    """parse_cache stores filing_date as YYYYMMDD (see insider_cluster_buys.
    parse_ownership_xml's `filing["date_filed"]`), matching backtest.
    sales_history._parse_filing_date exactly."""
    if not raw:
        return None
    try:
        return datetime.strptime(raw, "%Y%m%d").date()
    except ValueError:
        return None


def _parse_tx_date(raw: Optional[str]) -> Optional[date]:
    if not raw:
        return None
    try:
        return datetime.strptime(raw, "%Y-%m-%d").date()
    except ValueError:
        return None


def _scan_one(path: str) -> tuple[list[dict], Optional[dict]]:
    """Parse one cached filing JSON once and return (tx_rows, owner_appearance).

    tx_rows: qualifying (code, acquired_disposed) transaction rows only (see
    _KEEP_PAIRS) -- possibly empty.

    owner_appearance: {"issuer_cik", "owner_key", "filing_date"} for this
    filing's owners[0], REGARDLESS of whether it produced any tx_rows --
    used only to find each owner's issuer-wide FIRST EVER filing date
    (x_n_new_insiders_trail365). Keeping this unconditional matters: an
    owner's very first Form 4 for an issuer might carry only a code this
    module doesn't keep (a gift, a conversion) or even zero transactions at
    all (a holdings-only report) -- restricting "first filing" to the same
    filtered code set as tx_rows would systematically report owners as
    "newer" than they really are. None if the filing is unusable (bad JSON,
    not a Form 4/4A, or missing issuer_cik/filing_date) -- one bad file must
    not crash the whole scan, mirrors backtest.sales_history._load_one.
    """
    try:
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
    except (OSError, ValueError):
        return [], None

    if not (d.get("form_type") or "").startswith("4"):
        return [], None
    filing_date = _parse_filing_date(d.get("filing_date"))
    if filing_date is None:
        return [], None

    issuer = d.get("issuer") or {}
    issuer_cik = issuer.get("cik") or ""
    if not issuer_cik:
        return [], None

    owners = d.get("owners") or [{}]
    owner = owners[0] or {}
    owner_key = owner.get("cik") or owner.get("name") or ""
    if not owner_key:
        return [], None

    appearance = {"issuer_cik": issuer_cik, "owner_key": owner_key, "filing_date": filing_date}

    adsh = d.get("adsh", "")
    rows: list[dict] = []
    for tx in d.get("transactions") or []:
        code = (tx.get("code") or "").upper()
        ad = (tx.get("acquired_disposed") or "").upper()
        if (code, ad) not in _KEEP_PAIRS:
            continue
        tx_date = _parse_tx_date(tx.get("date"))
        if tx_date is None:
            continue
        shares = float(tx.get("shares") or 0.0)
        price = float(tx.get("price_per_share") or 0.0)
        rows.append({
            "adsh": adsh, "issuer_cik": issuer_cik, "owner_key": owner_key,
            "code": code, "acquired_disposed": ad,
            "transaction_date": tx_date, "filing_date": filing_date,
            "shares": shares, "price_per_share": price, "value": shares * price,
            "is_10b5_1": bool(tx.get("is_10b5_1")),
        })
    return rows, appearance


def build_form4_scan(
    parse_cache_dir: str = PARSE_CACHE_DIR, max_workers: int = DEFAULT_MAX_WORKERS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scan every cached Form 4/4A filing under `parse_cache_dir` ONCE and
    return (tx_df, owner_first_df):
      tx_df:         one row per qualifying transaction (_TX_COLUMNS).
      owner_first_df: one row per (issuer_cik, owner_key), the earliest
                       filing_date seen for that pair across the WHOLE cache
                       (already reduced -- not the raw appearance list).

    This is the expensive step (1.49M+ small-file reads). Callers should
    persist the result via save_form4_scan and reload it with
    load_or_build_form4_scan rather than calling this directly on every run.
    """
    paths = glob.glob(os.path.join(parse_cache_dir, "*.json"))
    n_total = len(paths)
    log.info("build_form4_scan: scanning %d cached filing(s) under %s/", n_total, parse_cache_dir)
    t0 = time.monotonic()

    tx_rows: list[dict] = []
    appearances: list[dict] = []
    n_done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for rows, appearance in ex.map(_scan_one, paths, chunksize=256):
            if rows:
                tx_rows.extend(rows)
            if appearance is not None:
                appearances.append(appearance)
            n_done += 1
            if n_done % LOG_EVERY == 0:
                log.info(
                    "build_form4_scan: scanned %d/%d filings (%.1fs elapsed, %d tx row(s), "
                    "%d appearance(s) so far)",
                    n_done, n_total, time.monotonic() - t0, len(tx_rows), len(appearances),
                )

    tx_df = pd.DataFrame(tx_rows, columns=_TX_COLUMNS)
    appear_df = pd.DataFrame(appearances, columns=_OWNER_COLUMNS)
    owner_first_df = (
        appear_df.groupby(["issuer_cik", "owner_key"], as_index=False)["filing_date"].min()
        if not appear_df.empty else pd.DataFrame(columns=_OWNER_COLUMNS)
    )

    elapsed = time.monotonic() - t0
    log.info(
        "build_form4_scan: done in %.1fs -- %d filing(s) scanned, %d qualifying tx row(s), "
        "%d distinct (issuer, owner) pair(s) with a known first filing",
        elapsed, n_total, len(tx_df), len(owner_first_df),
    )
    return tx_df, owner_first_df


def save_form4_scan(tx_df: pd.DataFrame, owner_first_df: pd.DataFrame, out_dir: str = OUT_DIR) -> tuple[str, str]:
    """Write both scan outputs to `out_dir`, atomically (write-then-rename,
    matching backtest.sales_history.save_sales_cache's convention). Returns
    (tx_path, owner_path)."""
    os.makedirs(out_dir, exist_ok=True)
    today = date.today().strftime("%Y%m%d")

    tx_path = os.path.join(out_dir, f"form4_extras_tx_{len(tx_df)}rows_{today}.parquet")
    tmp = tx_path + ".tmp"
    tx_df.to_parquet(tmp, index=False)
    os.replace(tmp, tx_path)

    owner_path = os.path.join(out_dir, f"form4_extras_owners_{len(owner_first_df)}rows_{today}.parquet")
    tmp = owner_path + ".tmp"
    owner_first_df.to_parquet(tmp, index=False)
    os.replace(tmp, owner_path)

    log.info("Wrote form4 scan cache: %s (%d rows), %s (%d rows)",
              tx_path, len(tx_df), owner_path, len(owner_first_df))
    return tx_path, owner_path


def _resolve_latest(out_dir: str, pattern: str) -> Optional[str]:
    matches = glob.glob(os.path.join(out_dir, pattern))
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


def load_or_build_form4_scan(
    out_dir: str = OUT_DIR, parse_cache_dir: str = PARSE_CACHE_DIR,
    max_workers: int = DEFAULT_MAX_WORKERS, force_rebuild: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reuse the most recently written form4_extras_tx_*/form4_extras_owners_*
    parquet pair under `out_dir` (by mtime) when BOTH exist and force_rebuild
    is False. Otherwise rescans parse_cache_dir from scratch via
    build_form4_scan and persists the result, so the expensive 1.49M-file
    scan is paid at most once per pair of cache files -- matches backtest.
    sales_history.load_or_build_sales_cache's convention. Rebuilds BOTH
    outputs together even if only one is missing/stale, so a tx/owner pair
    on disk always comes from the same scan and can never silently mismatch.
    """
    if not force_rebuild:
        tx_path = _resolve_latest(out_dir, TX_CACHE_GLOB)
        owner_path = _resolve_latest(out_dir, OWNER_CACHE_GLOB)
        if tx_path is not None and owner_path is not None:
            log.info("load_or_build_form4_scan: reusing cached scan %s + %s", tx_path, owner_path)
            return pd.read_parquet(tx_path), pd.read_parquet(owner_path)

    tx_df, owner_first_df = build_form4_scan(parse_cache_dir=parse_cache_dir, max_workers=max_workers)
    save_form4_scan(tx_df, owner_first_df, out_dir=out_dir)
    return tx_df, owner_first_df


# ---------------------------------------------------------------------------
# 2. Point-in-time feature computation
# ---------------------------------------------------------------------------
def _issuer_tx_groups(tx_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Pre-sort tx_df per issuer_cik by transaction_date, so each event only
    ever touches its own issuer's (typically small) transaction history --
    mirrors backtest.research._prepare_ticker_index's per-key pre-grouping."""
    out: dict[str, pd.DataFrame] = {}
    if tx_df.empty:
        return out
    for cik, g in tx_df.groupby("issuer_cik", sort=False):
        out[cik] = g.sort_values("transaction_date", kind="mergesort").reset_index(drop=True)
    return out


def _issuer_owner_first(owner_first_df: pd.DataFrame) -> dict[str, pd.Series]:
    """issuer_cik -> Series(index=owner_key, value=first-ever filing_date)."""
    out: dict[str, pd.Series] = {}
    if owner_first_df.empty:
        return out
    for cik, g in owner_first_df.groupby("issuer_cik", sort=False):
        out[cik] = g.set_index("owner_key")["filing_date"]
    return out


def _visible_trailing_window(group: pd.DataFrame, event_day: date, trail_days: int) -> pd.DataFrame:
    """Rows of `group` (one issuer's sorted tx history) visible as of
    event_day: filing_date STRICTLY BEFORE event_day (see module docstring's
    point-in-time rule) AND transaction_date in [event_day - trail_days,
    event_day)."""
    lo = event_day - timedelta(days=trail_days)
    mask = (
        (group["transaction_date"] >= lo)
        & (group["transaction_date"] < event_day)
        & (group["filing_date"] < event_day)
    )
    return group.loc[mask]


def _classify_exercise(group: pd.DataFrame, owner_key: str, exercise_tx_date: date, event_day: date) -> Optional[bool]:
    """True (held) / False (sold) / None (undetermined -- observation window
    not yet complete as of event_day). See module docstring's
    x_exercise_and_hold_trail180 note for why "not yet seen" must not be
    silently treated as "held"."""
    window_end = exercise_tx_date + timedelta(days=EXERCISE_HOLD_WINDOW_DAYS)
    if window_end >= event_day:
        return None  # the 5-day follow-on window isn't fully observable yet

    sales = group[
        (group["code"] == SALE_CODE)
        & (group["owner_key"] == owner_key)
        & (group["filing_date"] < event_day)
        & (group["transaction_date"] >= exercise_tx_date)
        & (group["transaction_date"] <= window_end)
    ]
    return sales.empty  # True = held (no follow-on sale seen), False = sold


def attach_features(
    df: pd.DataFrame,
    *,
    parse_cache_dir: str = PARSE_CACHE_DIR,
    tx_df: Optional[pd.DataFrame] = None,
    owner_first_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Compute NEW_FEATURE_COLS for every row of `df` (must have `issuer_cik`
    and `event_day` columns; row order and index are preserved).

    `tx_df` / `owner_first_df` default to None, which triggers a real
    load_or_build_form4_scan(parse_cache_dir) call -- the expensive path,
    used by __main__ and any caller that wants the real cache. Tests pass
    small hand-built frames directly (matching backtest.research.
    build_research_dataset's `sales_df` parameter convention) so the test
    suite never touches the real 1.49M-file parse_cache/.

    Returns a DataFrame with exactly NEW_FEATURE_COLS, same length and index
    as `df`, safe to `pd.concat([df, result], axis=1)`.
    """
    if tx_df is None or owner_first_df is None:
        loaded_tx, loaded_owner = load_or_build_form4_scan(parse_cache_dir=parse_cache_dir)
        tx_df = loaded_tx if tx_df is None else tx_df
        owner_first_df = loaded_owner if owner_first_df is None else owner_first_df

    n = len(df)
    out = {c: np.full(n, np.nan, dtype=np.float64) for c in NEW_FEATURE_COLS}

    issuer_groups = _issuer_tx_groups(tx_df)
    owner_first = _issuer_owner_first(owner_first_df)

    ciks = df["issuer_cik"].astype(str).to_numpy()
    event_days = df["event_day"].to_numpy()

    for i in range(n):
        cik = ciks[i]
        D = event_days[i]
        if not isinstance(D, date):
            continue  # NaT/unparseable event_day -- leave every feature NaN

        # ---- new insiders (owner-appearance based, independent of tx codes) ----
        first_filings = owner_first.get(cik)
        if first_filings is not None and len(first_filings):
            lo = D - timedelta(days=TRAIL_365_DAYS)
            out["x_n_new_insiders_trail365"][i] = float(((first_filings >= lo) & (first_filings < D)).sum())
        else:
            out["x_n_new_insiders_trail365"][i] = 0.0

        group = issuer_groups.get(cik)
        if group is None or group.empty:
            # No qualifying tx history at all for this issuer -- counts are
            # real zeros (there is nothing to count), ratios/fractions stay
            # NaN (undefined, not "zero selling pressure").
            out["x_insider_grants_trail180"][i] = 0.0
            out["x_option_exercises_trail180"][i] = 0.0
            out["x_exercise_and_hold_trail180"][i] = 0.0
            out["x_insider_sales_trail180"][i] = 0.0
            continue

        visible = _visible_trailing_window(group, D, TRAIL_180_DAYS)

        grants = visible[visible["code"] == GRANT_CODE]
        exercises = visible[visible["code"] == EXERCISE_CODE]
        sales = visible[visible["code"] == SALE_CODE]
        buys = visible[visible["code"] == BUY_CODE]

        out["x_insider_grants_trail180"][i] = float(len(grants))
        out["x_option_exercises_trail180"][i] = float(len(exercises))
        out["x_insider_sales_trail180"][i] = float(len(sales))

        n_held = 0
        for _, ex_row in exercises.iterrows():
            verdict = _classify_exercise(group, ex_row["owner_key"], ex_row["transaction_date"], D)
            if verdict is True:
                n_held += 1
        out["x_exercise_and_hold_trail180"][i] = float(n_held)

        buy_value = float(buys["value"].sum())
        sale_value = float(sales["value"].sum())
        out["x_sale_to_buy_ratio"][i] = (sale_value / buy_value) if buy_value > 0 else np.nan

        if len(buys):
            out["x_frac_buys_10b5_1_trail180"][i] = float(buys["is_10b5_1"].mean())
        # else stays NaN -- no trailing buys to compute a fraction over.

    return pd.DataFrame(out, index=df.index)[NEW_FEATURE_COLS]


# ---------------------------------------------------------------------------
# 3. Coverage / descriptive report (no modelling -- see module docstring)
# ---------------------------------------------------------------------------
def _print_report(df: pd.DataFrame) -> None:
    n = len(df)
    print(f"\n=== form4_extras coverage/descriptive report ({n} rows) ===")
    print("Coverage (fraction of rows with a non-null value):")
    for c in NEW_FEATURE_COLS:
        frac = df[c].notna().mean()
        print(f"  {c}: {frac:.1%}")

    if "adj_21" not in df.columns:
        print("\n(adj_21 not present -- skipping descriptive tables)")
        return

    def _quartile_table(col: str) -> None:
        valid = df.dropna(subset=[col, "adj_21"])
        print(f"\nMedian adj_21 by quartile of {col} (n={len(valid)}):")
        if len(valid) < 20:
            print("  (too few non-null rows to report)")
            return
        try:
            q_labels = pd.qcut(valid[col], 4, duplicates="drop")
        except ValueError:
            print("  (could not form distinct quartiles -- likely too many repeated values)")
            return
        g = valid.groupby(q_labels, observed=True)["adj_21"]
        tbl = pd.DataFrame({"median_adj_21": g.median(), "n": g.size()})
        print(tbl.to_string())

    def _binary_table(col: str, *, threshold: float = 0.0) -> None:
        valid = df.dropna(subset=[col, "adj_21"])
        print(f"\nMedian adj_21 by {col} > {threshold} (n={len(valid)}):")
        if valid.empty:
            print("  (no non-null rows)")
            return
        split = valid[col] > threshold
        g = valid.groupby(split)["adj_21"]
        tbl = pd.DataFrame({"median_adj_21": g.median(), "n": g.size()})
        print(tbl.to_string())

    # The three features judged most promising after inspecting the actual
    # distributions on the real 10,861-row dataset (see task writeup): most
    # of the 7 features here are nearly flat univariately (medians all
    # within a percentage point of the overall -0.82%), so "promising" here
    # means "shows the largest visible gap", not "is proven". Note what got
    # dropped from this report and why:
    #   - x_frac_buys_10b5_1_trail180's own quartiles collapse to ONE bin
    #     (96.5% of rows are exactly 0.0) -- qcut can't split it, so it gets
    #     the 0-vs-positive treatment instead, which is where its one real
    #     signal actually shows up.
    #   - x_sale_to_buy_ratio and x_exercise_and_hold_trail180 were also
    #     inspected and came back close to flat (medians within ~0.1pp of
    #     each other on their 0-vs-positive split) -- not reported here for
    #     that reason, not because they were skipped.
    _binary_table("x_frac_buys_10b5_1_trail180", threshold=0.0)
    _quartile_table("x_insider_sales_trail180")
    _quartile_table("x_n_new_insiders_trail365")
    print()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--events", default=DEFAULT_EVENTS, help="Research parquet to attach features to.")
    p.add_argument("--parse-cache", default=PARSE_CACHE_DIR, help="parse_cache/ directory to scan.")
    p.add_argument("--out", default="", help="Output parquet path. Default: research_data/form4_extras.parquet")
    p.add_argument("--force-rescan", action="store_true", help="Ignore any cached form4_extras_* scan and rebuild.")
    p.add_argument("--verbose", action="store_true")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                         format="%(asctime)s [%(levelname)s] %(message)s")

    log.info("form4_extras: reading events from %s", a.events)
    events = pd.read_parquet(a.events)

    tx_df, owner_first_df = load_or_build_form4_scan(
        parse_cache_dir=a.parse_cache, force_rebuild=a.force_rescan,
    )
    feats = attach_features(events, tx_df=tx_df, owner_first_df=owner_first_df)

    assert len(feats) == len(events), "row count changed -- refusing to write"
    assert list(feats.index) == list(events.index), "row order changed -- refusing to write"

    out_df = pd.concat([events, feats], axis=1)
    assert len(out_df) == len(events), "concat changed row count -- refusing to write"

    out_path = a.out or os.path.join(REPO_ROOT, "research_data", "form4_extras.parquet")
    if not os.path.isabs(out_path):
        out_path = os.path.join(REPO_ROOT, out_path)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    tmp = out_path + ".tmp"
    out_df.to_parquet(tmp, index=False)
    os.replace(tmp, out_path)
    log.info("form4_extras: wrote %s (%d rows, %d cols)", out_path, len(out_df), out_df.shape[1])

    _print_report(out_df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
