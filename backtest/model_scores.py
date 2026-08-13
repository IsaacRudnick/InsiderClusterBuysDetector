"""Loader for a ranking model's out-of-fold (OOF) scores, turning the
research/model.py output parquet into the (ticker, event_day) -> float
mapping DailyStateBuilder.set_model_scores (backtest/state.py) expects.

Kept as its own module rather than folded into backtest/state.py: state.py's
job is building per-day cluster state from the SEC filing history (it
already depends on ipo_lookup and insider_cluster_buys), and has no other
reason to know a specific OOF parquet's column names or on-disk layout.
This module's only job is that one schema, so it can be read, tested, and
changed (e.g. when research/model.py's output columns change) independently
of the state-building code that consumes its result.
"""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd

log = logging.getLogger(__name__)

# oof_tail_classifier: the tail ("moonshot") classifier's OOF probability.
# Chosen as the default over oof_regressor/oof_classifier because the
# strategies this loader feeds (backtest/strategies.py's model_ranked_*)
# want a ranking, not a point-estimate return forecast, and the tail
# classifier is what the offline validation actually cleared -- see the
# job's own report for the comparison. Callers can still pass any other
# score column (e.g. oof_regressor) explicitly.
DEFAULT_SCORE_COL = "oof_tail_classifier"

# Where 'latest' looks. Same directory research/model.py writes its dataset
# to, so the scores sit beside the data they were fit on.
SCORES_DIR = "research_data"
SCORES_GLOB = "oof_scores*.parquet"

# How close two candidates' mtimes may be before 'latest' refuses to pick.
#
# This is not a hypothetical. out/backtest_20260812_234954 ran with
# model_scores='latest' against four oof_scores_objsweep_thresh_*.parquet
# files written 40 MILLISECONDS apart by one sweep script. max(mtime) picked
# one of the four essentially at random -- it landed on thresh_0.20, the
# variant that sweep had already found worse than thresh_0.05 -- and the
# config recorded only the string 'latest', so nothing in the output
# directory showed which model the 64-strategy grid had actually used.
#
# One second is far longer than the sub-millisecond spread a batch writer
# produces and far shorter than the gap between two deliberate fit runs, so
# it separates "the same job wrote these" from "I fit a new model" without
# needing either process to cooperate.
AMBIGUOUS_MTIME_WINDOW_S = 1.0


def resolve_model_scores_path(spec: str) -> str | None:
    """Turn a prompt answer into a path, or None for 'do not attach'.

    Accepts an explicit path, 'latest' to pick the most recently modified
    file matching SCORES_GLOB in SCORES_DIR, or blank for None. This mirrors
    BT_EVENTS_FROM's 'latest' convention (see backtest/history.py) so the
    two prompts behave the same way.

    'latest' with no matching file returns None rather than raising. A run
    that selected no model-ranked strategy should not fail just because no
    scores have been fit yet. The caller (backtest.py) raises separately if
    a strategy actually needs scores and none were found, which keeps the
    "you asked for model ranking but have no model" error where it can name
    the offending strategies.

    'latest' RAISES, however, when the newest candidate is within
    AMBIGUOUS_MTIME_WINDOW_S of another one. A tie there means several files
    were written by the same batch job and max(mtime) is choosing between
    them arbitrarily -- silently deciding which model an entire backtest grid
    ranks on. That is worth a hard stop with the candidate list in the
    message, not a warning that scrolls past. Pass an explicit path (or
    `run_research.py --oof-path-out`, which hands over the file it just fit)
    to resolve it.
    """
    spec = (spec or "").strip()
    if not spec:
        return None
    if spec != "latest":
        return spec

    import glob
    import os

    matches = glob.glob(os.path.join(SCORES_DIR, SCORES_GLOB))
    if not matches:
        log.warning(
            "model scores 'latest' found no %s in %s/. No scores attached.",
            SCORES_GLOB, SCORES_DIR,
        )
        return None

    by_mtime = sorted(matches, key=os.path.getmtime, reverse=True)
    best = by_mtime[0]
    best_mtime = os.path.getmtime(best)
    tied = [
        p for p in by_mtime[1:]
        if abs(best_mtime - os.path.getmtime(p)) <= AMBIGUOUS_MTIME_WINDOW_S
    ]
    if tied:
        raise SystemExit(
            f"model scores 'latest' is ambiguous: {len(tied) + 1} files in "
            f"{SCORES_DIR}/ share a modification time within "
            f"{AMBIGUOUS_MTIME_WINDOW_S}s, so picking the newest would choose "
            f"between them arbitrarily and silently decide which model this run "
            f"ranks on.\n"
            f"  Tied candidates: {', '.join(os.path.basename(p) for p in [best] + tied)}\n"
            f"Set BT_MODEL_SCORES to the exact parquet you mean, or run "
            f"`python run_research.py --fit-model --oof-path-out <file>` and pass "
            f"the path it writes."
        )

    log.info(
        "model scores 'latest' resolved to %s (newest of %d candidate(s): %s)",
        best, len(matches), ", ".join(os.path.basename(p) for p in by_mtime),
    )
    return best


def load_model_scores(
    path: str, score_col: str = DEFAULT_SCORE_COL,
) -> dict[tuple[str, date], float]:
    """Read an OOF-scores parquet and return {(ticker, event_day): score}.

    event_day is coerced to plain datetime.date, because that is exactly
    the type DailyStateBuilder.state_for_day(D) passes as `as_of`, and the
    dict lookup in _build_state is a plain `==`/bisect comparison, not a
    tolerant one. A silent mismatch (e.g. pandas.Timestamp keys instead of
    date) would make every lookup miss and every state's model_score come
    back None -- indistinguishable from "the model has no edge" unless
    this is checked explicitly, so we assert the key type and log the
    match rate rather than trusting the parquet's stored dtype.
    """
    df = pd.read_parquet(path, columns=["ticker", "event_day", score_col])
    n_rows = len(df)
    if df.empty:
        log.warning("load_model_scores: %s (column=%s) produced zero rows", path, score_col)
        return {}

    # .dt.date guarantees plain datetime.date objects regardless of how
    # event_day round-tripped through parquet (pandas Timestamp, python
    # date, and object dtype have all been observed from different writers).
    event_days = pd.to_datetime(df["event_day"]).dt.date
    tickers = df["ticker"].astype(str)
    scores = df[score_col].astype(float)

    scores_by_key: dict[tuple[str, date], float] = {}
    n_nan = 0
    for ticker, event_day, score in zip(tickers, event_days, scores):
        if score != score:  # NaN check (NaN never equals itself); no usable score.
            n_nan += 1
            continue
        scores_by_key[(ticker, event_day)] = float(score)

    n_valid_rows = n_rows - n_nan
    n_collisions = n_valid_rows - len(scores_by_key)
    if n_collisions > 0:
        # (ticker, event_day) should be unique per research.py's one-row-
        # per-distinct-cluster-event construction; a collision here means
        # two rows mapped to the same key and one silently overwrote the
        # other, so it is worth knowing about even though we don't raise.
        log.warning(
            "load_model_scores: %d duplicate (ticker, event_day) rows collapsed "
            "in %s (column=%s); last value wins per duplicate",
            n_collisions, path, score_col,
        )

    # Assert the key type actually is datetime.date, not e.g.
    # pandas.Timestamp -- .dt.date should already guarantee this, but this
    # whole function exists to not silently trust that.
    if scores_by_key:
        sample_key = next(iter(scores_by_key))
        assert type(sample_key[1]) is date, (
            f"load_model_scores: event_day key came back as "
            f"{type(sample_key[1])!r}, not datetime.date -- "
            f"DailyStateBuilder.state_for_day passes datetime.date as "
            f"`as_of`, so a type mismatch here would make every lookup "
            f"silently miss and every model_score come back None"
        )

    log.info(
        "load_model_scores: %s column=%s -> %d scored keys from %d rows "
        "(%d NaN dropped, %d duplicate collisions, match rate %.1f%%)",
        path, score_col, len(scores_by_key), n_rows, n_nan, n_collisions,
        100.0 * len(scores_by_key) / n_rows if n_rows else 0.0,
    )
    return scores_by_key
