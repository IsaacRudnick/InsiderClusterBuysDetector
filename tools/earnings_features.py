"""Point-in-time earnings/filing-proximity features, attached to a COPY of
the research dataset.

EVERY feature in this module must be point-in-time: for a given event, only
filings whose `filing_date` is STRICTLY BEFORE that event's `event_day` may
be used to compute it. Getting this wrong -- letting a feature see a filing
that had not happened yet -- is classic lookahead bias. A model trained on
such a feature is not learning anything real; it is being handed a peek at
the future and rewarded for noticing it. The resulting "edge" evaporates the
moment the model is run live, where the future filing genuinely is unknown,
and until then it silently inflates every backtest number that touches it.

Enforcement mechanism. For each issuer, filing dates are sorted ascending
into a plain numpy array. For a given `event_day`, `numpy.searchsorted(...,
event_day, side='left')` returns the index of the first filing whose date is
`>= event_day` -- i.e. the boundary between "known before this event" and
"not yet known". Every feature below is computed using ONLY entries at
indices strictly less than that boundary. `side='left'` matters here: it
puts a same-day filing (filing_date == event_day) on the "not yet known"
side, which is correct -- a filing dated the same calendar day as the event
was not necessarily public before the event happened intraday, and treating
it as known would be exactly the kind of leak this module exists to avoid.

Runnable standalone as `python tools/earnings_features.py` (reads the
research parquet and filing_calendar.parquet from disk, writes the augmented
parquet) or importable, e.g. from a test, via `compute_earnings_features`.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

log = logging.getLogger("earnings_features")

DEFAULT_EVENTS = os.path.join(REPO_ROOT, "research_data", "research_10861rows_20260813.parquet")
DEFAULT_CALENDAR = os.path.join(REPO_ROOT, "research_data", "filing_calendar.parquet")

PERIODIC_FORMS = ("10-Q", "10-K")
EIGHT_K_FORM = "8-K"

# Fallback median gap (calendar days) between periodic filings when an
# issuer has fewer than 2 periodic filings strictly before the event to
# measure a real gap from. 91 days ~= one fiscal quarter.
_FALLBACK_PERIODIC_GAP_DAYS = 91.0

# x_earnings_inside_horizon window: roughly a 21-trading-day hold expressed
# in calendar days.
_HORIZON_LO_DAYS = 0
_HORIZON_HI_DAYS = 31

NEW_FEATURE_COLS = [
    "x_days_since_last_periodic",
    "x_days_since_last_8k",
    "x_n_8k_trail30",
    "x_days_to_expected_periodic",
    "x_earnings_inside_horizon",
]


def _to_day_ordinal(series: pd.Series) -> np.ndarray:
    """Convert a column of dates/timestamps/python `date` objects to int64
    ordinal days (matplotlib/proleptic-Gregorian ordinal), so all date
    arithmetic in this module is done on plain integers -- avoids any
    surprises from mixing `datetime.date` (what event_day actually holds in
    the research parquet) with pandas Timestamps (what filing_calendar.parquet
    holds after being read back from parquet)."""
    dt = pd.to_datetime(series)
    # dt.values is datetime64[ns]; convert to date ordinals via .dt accessor
    # through a Series so NaT survives as NaN rather than raising.
    return dt.map(lambda ts: ts.toordinal() if pd.notna(ts) else np.nan).to_numpy()


def _issuer_filing_index(calendar: pd.DataFrame) -> dict:
    """Pre-build, per issuer CIK, ascending-sorted filing-date arrays (as day
    ordinals) split by form, for O(log n) point-in-time lookups per event via
    numpy.searchsorted. Built once and reused across all events for that
    issuer rather than re-sorting per event.

    Returns {cik: {"periodic": (ordinals, is_periodic_mask_unused), ...}}
    Concretely: {cik: {"periodic_ord": np.ndarray, "8k_ord": np.ndarray}}
    both sorted ascending.
    """
    idx: dict[str, dict[str, np.ndarray]] = {}
    if calendar.empty:
        return idx

    cal = calendar.copy()
    cal["filing_ord"] = _to_day_ordinal(cal["filing_date"])
    cal = cal.dropna(subset=["filing_ord"])
    cal["filing_ord"] = cal["filing_ord"].astype(np.int64)

    is_periodic = cal["form"].isin(PERIODIC_FORMS)
    is_8k = cal["form"] == EIGHT_K_FORM

    for cik, grp in cal[is_periodic].groupby("cik"):
        idx.setdefault(cik, {})["periodic_ord"] = np.sort(grp["filing_ord"].to_numpy())
    for cik, grp in cal[is_8k].groupby("cik"):
        idx.setdefault(cik, {})["8k_ord"] = np.sort(grp["filing_ord"].to_numpy())

    return idx


def _prior_slice(sorted_ord: np.ndarray, event_ord: float) -> np.ndarray:
    """All entries in `sorted_ord` (ascending) strictly before `event_ord`.

    `side='left'` gives the index of the first entry >= event_ord, so
    everything before that index -- and only that -- is strictly earlier.
    A filing dated exactly on event_day is therefore excluded, matching the
    module docstring's point-in-time rule.
    """
    if sorted_ord.size == 0 or event_ord != event_ord:  # NaN event_ord check
        return sorted_ord[:0]
    boundary = np.searchsorted(sorted_ord, event_ord, side="left")
    return sorted_ord[:boundary]


def compute_earnings_features(events_df: pd.DataFrame, filing_calendar_df: pd.DataFrame) -> pd.DataFrame:
    """Compute the point-in-time earnings-proximity feature columns.

    Parameters
    ----------
    events_df : must have `issuer_cik` and `event_day` columns. Row order and
        index are preserved in the output.
    filing_calendar_df : must have `cik`, `form`, `filing_date` columns
        (`report_date` is ignored here -- only filing_date is point-in-time
        meaningful; report_date describes a period the filing covers, not
        when the market learned about it).

    Returns
    -------
    A DataFrame with exactly `NEW_FEATURE_COLS`, same length and index as
    events_df, safe to `pd.concat([events_df, result], axis=1)`.
    """
    n = len(events_df)
    out = {c: np.full(n, np.nan, dtype=np.float64) for c in NEW_FEATURE_COLS}

    fidx = _issuer_filing_index(filing_calendar_df)
    event_ciks = events_df["issuer_cik"].astype(str).to_numpy()
    event_ords = _to_day_ordinal(events_df["event_day"])

    for i in range(n):
        cik = event_ciks[i]
        eo = event_ords[i]
        entry = fidx.get(cik)
        if entry is None or eo != eo:  # no filings at all for this issuer, or NaT event_day
            continue

        periodic_ord = entry.get("periodic_ord", np.empty(0, dtype=np.int64))
        eightk_ord = entry.get("8k_ord", np.empty(0, dtype=np.int64))

        prior_periodic = _prior_slice(periodic_ord, eo)
        prior_8k = _prior_slice(eightk_ord, eo)

        if prior_periodic.size:
            last_periodic = prior_periodic[-1]
            out["x_days_since_last_periodic"][i] = eo - last_periodic
        if prior_8k.size:
            last_8k = prior_8k[-1]
            out["x_days_since_last_8k"][i] = eo - last_8k
            trail30_lo = eo - 30
            out["x_n_8k_trail30"][i] = int(np.sum(prior_8k >= trail30_lo))
        else:
            out["x_n_8k_trail30"][i] = 0.0

        if prior_periodic.size:
            last_periodic = prior_periodic[-1]
            if prior_periodic.size >= 2:
                gaps = np.diff(prior_periodic)
                median_gap = float(np.median(gaps))
            else:
                # Fewer than 2 periodic filings before this event -- no real
                # gap to measure, so fall back to a typical fiscal-quarter
                # cadence rather than leaving the estimate unanchored.
                median_gap = _FALLBACK_PERIODIC_GAP_DAYS
            # This is a SAME-ISSUER HISTORICAL estimate -- the next filing
            # date this issuer's own past cadence implies -- never the
            # actual future filing date (which would be lookahead). It can
            # be, and often is, wrong; that is the point: the model gets a
            # forecast built only from what was knowable at event_day, not
            # an oracle.
            expected = last_periodic + median_gap
            days_to_expected = expected - eo
            out["x_days_to_expected_periodic"][i] = days_to_expected
            if _HORIZON_LO_DAYS <= days_to_expected <= _HORIZON_HI_DAYS:
                out["x_earnings_inside_horizon"][i] = 1.0
            else:
                out["x_earnings_inside_horizon"][i] = 0.0
        # else: no periodic filing before this event at all -> both stay NaN.

    return pd.DataFrame(out, index=events_df.index)[NEW_FEATURE_COLS]


def _print_sanity_report(df: pd.DataFrame) -> None:
    n = len(df)
    print(f"\n=== earnings_features sanity report ({n} rows) ===")
    print("Coverage (fraction of rows with a non-null value):")
    for c in NEW_FEATURE_COLS:
        frac = df[c].notna().mean()
        print(f"  {c}: {frac:.1%}")

    dsp = df["x_days_since_last_periodic"].dropna()
    print("\nx_days_since_last_periodic quartiles (calendar days, n=%d):" % len(dsp))
    if len(dsp):
        q = dsp.quantile([0.25, 0.5, 0.75])
        print(f"  25th: {q.loc[0.25]:.1f}   50th (median): {q.loc[0.5]:.1f}   75th: {q.loc[0.75]:.1f}")
    else:
        print("  (no non-null values)")

    eih = df["x_earnings_inside_horizon"].dropna()
    frac_inside = (eih == 1.0).mean() if len(eih) else float("nan")
    print(f"\nFraction of events with x_earnings_inside_horizon == 1: {frac_inside:.1%}"
          f" (of {len(eih)} events with a defined value)")

    print("\nMEDIAN adj_21 by quartile of x_days_since_last_periodic (medians, not means -- "
          "adj_21 has a large right tail that would make a mean misleading):")
    if "adj_21" in df.columns and len(dsp):
        valid = df.loc[dsp.index]
        try:
            q_labels = pd.qcut(valid["x_days_since_last_periodic"], 4, duplicates="drop")
        except ValueError:
            q_labels = None
        if q_labels is not None:
            tbl = valid.groupby(q_labels, observed=True)["adj_21"].median()
            print(tbl.to_string())
        else:
            print("  (could not form 4 distinct quartiles)")
    else:
        print("  (adj_21 not present or no data)")

    print("\nMEDIAN adj_21 by x_earnings_inside_horizon (0 vs 1):")
    if "adj_21" in df.columns and len(eih):
        valid = df.loc[eih.index]
        tbl = valid.groupby("x_earnings_inside_horizon")["adj_21"].median()
        print(tbl.to_string())
    else:
        print("  (adj_21 not present or no data)")
    print()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--events", default=DEFAULT_EVENTS,
                    help="Research parquet to attach features to.")
    p.add_argument("--calendar", default=DEFAULT_CALENDAR,
                    help="filing_calendar.parquet produced by tools/filing_calendar.py.")
    p.add_argument("--out", default="",
                    help="Output parquet path. Default: "
                         "research_data/research_earnings_<nrows>rows_<YYYYMMDD>.parquet")
    p.add_argument("--verbose", action="store_true")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                         format="%(asctime)s [%(levelname)s] %(message)s")

    log.info("earnings_features: reading events from %s", a.events)
    events = pd.read_parquet(a.events)
    log.info("earnings_features: reading filing calendar from %s", a.calendar)
    calendar = pd.read_parquet(a.calendar)

    feats = compute_earnings_features(events, calendar)

    # Guard against the one thing this module must never do: change the
    # underlying research dataset's rows. Same length, same row order
    # (checked via the index, which pd.concat below relies on lining up).
    assert len(feats) == len(events), "row count changed -- refusing to write"
    assert list(feats.index) == list(events.index), "row order changed -- refusing to write"

    out_df = pd.concat([events, feats], axis=1)
    assert len(out_df) == len(events), "concat changed row count -- refusing to write"

    out_path = a.out
    if not out_path:
        nrows = len(out_df)
        today = date.today().strftime("%Y%m%d")
        out_path = os.path.join(REPO_ROOT, "research_data", f"research_earnings_{nrows}rows_{today}.parquet")
    elif not os.path.isabs(out_path):
        out_path = os.path.join(REPO_ROOT, out_path)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    out_df.to_parquet(out_path, index=False)
    log.info("earnings_features: wrote %s (%d rows, %d cols)", out_path, len(out_df), out_df.shape[1])

    _print_sanity_report(out_df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
