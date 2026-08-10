"""Tests for backtest.history: the backtest's own event-loading path.

Focus of this file is the ticker-reuse guard's wiring into _clean_events_df
/ load_events_df / build_history (see ticker_reuse.py and backtest/
research.py's "Ticker-reuse wiring" note, which this mirrors on the backtest
side). Follows this repo's pytest conventions: plain pytest, class-grouped
tests, synthetic frames only, no network access (see tests/test_ticker_reuse.py
and tests/test_research.py).
"""

from __future__ import annotations

import importlib.util
import os
import sys
from datetime import date

import pandas as pd
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import ticker_reuse  # noqa: E402
from backtest import history  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------
def _row(*, ticker: str, issuer_cik: str, issuer_name: str, filing_date: date,
         transaction_date: date | None = None) -> dict:
    """Minimal row shape _clean_events_df / ticker_reuse actually read
    (ticker, issuer_cik, issuer_name, transaction_date, filing_date). Other
    real events_df columns (owner_cik, value, shares, ...) are irrelevant to
    this seam and omitted, matching tests/test_ticker_reuse.py's make_row."""
    return {
        "ticker": ticker,
        "issuer_cik": issuer_cik,
        "issuer_name": issuer_name,
        "transaction_date": transaction_date or filing_date,
        "filing_date": filing_date,
    }


def _span_rows(ticker: str, cik: str, name: str, dates: list[date]) -> list[dict]:
    return [_row(ticker=ticker, issuer_cik=cik, issuer_name=name, filing_date=d) for d in dates]


def _write_events_parquet(rows: list[dict], path) -> str:
    pd.DataFrame(rows).to_parquet(path, index=False)
    return str(path)


# ---------------------------------------------------------------------------
# load_events_df: ticker-reuse guard wiring
# ---------------------------------------------------------------------------
class TestLoadEventsDfTickerReuseGuard:
    def test_drops_unsafe_predecessor_keeps_successor(self, tmp_path):
        """Same REUSE shape as the real AI (Arlington Asset Investment ->
        C3.ai) transition: an old, delisted, unrelated company's rows must
        be dropped from the events the BACKTEST loads, not just the ones
        the research pipeline loads."""
        rows = (
            _span_rows("REUSD", "OLDCO", "Arlington Asset Investment Corp.",
                       [date(2018, 10, 3), date(2020, 9, 28)])
            + _span_rows("REUSD", "NEWCO", "C3.ai, Inc.",
                         [date(2021, 6, 11), date(2022, 1, 1)])
        )
        path = _write_events_parquet(rows, tmp_path / "events_test.parquet")

        df, window_start, window_end, n_dropped = history.load_events_df(path)

        assert n_dropped == 2
        assert not (df["issuer_cik"] == "OLDCO").any()
        assert (df["issuer_cik"] == "NEWCO").sum() == 2

    def test_rename_ticker_drops_nothing(self, tmp_path):
        """Same RENAME shape as the real APA (Apache Corp -> APA Corp)
        transition: near-identical name, short gap -- both eras must be
        kept."""
        rows = (
            _span_rows("APAT", "CIK_A", "APACHE CORP", [date(2018, 8, 29), date(2020, 4, 7)])
            + _span_rows("APAT", "CIK_B", "APA Corp", [date(2021, 3, 12), date(2021, 6, 1)])
        )
        path = _write_events_parquet(rows, tmp_path / "events_rename.parquet")

        df, _, _, n_dropped = history.load_events_df(path)

        assert n_dropped == 0
        assert len(df) == len(rows)

    def test_drop_ticker_reuse_false_keeps_every_row(self, tmp_path):
        """drop_ticker_reuse=False (what BT_DROP_TICKER_REUSE=0 threads
        through to, see backtest.py) must keep every row, including the
        ones the guard would otherwise drop -- explicit and controllable,
        not hardwired on, matching build_research_dataset's own kwarg."""
        rows = (
            _span_rows("REUSD2", "OLDCO", "Arlington Asset Investment Corp.",
                       [date(2018, 10, 3), date(2020, 9, 28)])
            + _span_rows("REUSD2", "NEWCO", "C3.ai, Inc.",
                         [date(2021, 6, 11), date(2022, 1, 1)])
        )
        path = _write_events_parquet(rows, tmp_path / "events_disabled.parquet")

        df, _, _, n_dropped = history.load_events_df(path, drop_ticker_reuse=False)

        assert n_dropped == 0
        assert len(df) == len(rows)
        assert (df["issuer_cik"] == "OLDCO").sum() == 2


# ---------------------------------------------------------------------------
# build_history: ordering (classify on the FULL history, before BT_AS_OF
# trims by filing_date)
# ---------------------------------------------------------------------------
class TestBuildHistoryOrdering:
    def test_classification_uses_full_history_before_as_of_trim(self, tmp_path):
        """ORDERING IS LOAD-BEARING (see build_history's / _clean_events_df's
        docstrings). CIK_A is an old, unrelated occupant of ticker ZORP,
        filing only BEFORE the as_of pin used below. CIK_B is ZORP's real,
        current occupant -- also unrelated to CIK_A -- but it only filed
        AFTER that as_of pin.

        If the BT_AS_OF filing-date trim ran BEFORE ticker-reuse
        classification, CIK_B's rows would vanish first, ZORP would look
        like a single-CIK ticker to the classifier, and CIK_A's rows (which
        pass the as_of trim on their own filing dates) would wrongly
        survive -- exactly the mispricing bug this guard exists to catch,
        since backtest/prices.py fetches TODAY's real ZORP price history
        (CIK_B's, not CIK_A's) regardless of any as_of replay pin.

        With the correct order (classify on the full file, trim after),
        CIK_A's rows must be dropped even though they individually predate
        as_of.
        """
        as_of = date(2019, 1, 1)
        rows = (
            _span_rows("ZORP", "CIK_A", "Arlington Asset Investment Corp.",
                       [date(2018, 1, 1), date(2018, 6, 1)])  # before as_of
            + _span_rows("ZORP", "CIK_B", "C3.ai, Inc.",
                         [date(2022, 1, 1), date(2022, 6, 1)])  # after as_of
        )
        path = _write_events_parquet(rows, tmp_path / "events_ordering.parquet")
        raw_df = pd.DataFrame(rows)

        # Sanity: classifying the FULL, untrimmed file really is REUSE, so
        # the fixture actually exercises the guard.
        _filtered_full, n_dropped_full, transitions_full = ticker_reuse.filter_unsafe_ticker_reuse(raw_df)
        assert n_dropped_full == 2
        assert transitions_full.iloc[0]["classification"] == "REUSE"

        # Correct order, exercised through the real build_history path.
        df, _win_start, _win_end, _errors, n_dropped = history.build_history(
            months_back=999, as_of=as_of, events_from=path,
        )
        assert not (df["issuer_cik"] == "CIK_A").any()
        assert n_dropped == 2
        # CIK_B's rows were trimmed by as_of (they postdate it) -- this
        # fixture's df is empty after both operations, which is expected;
        # what matters is that CIK_A never sneaks through the trim.
        assert df.empty

        # Demonstrate the WRONG order would have produced a different,
        # unsafe result: trim by as_of FIRST, classify SECOND.
        wrong_order_df = raw_df[raw_df["filing_date"] <= as_of].copy()
        assert len(wrong_order_df) == 2  # only CIK_A's rows survive the trim
        _filtered_wrong, n_dropped_wrong, transitions_wrong = ticker_reuse.filter_unsafe_ticker_reuse(
            wrong_order_df
        )
        assert n_dropped_wrong == 0  # CIK_B is invisible -> no transition -> nothing flagged unsafe
        assert transitions_wrong.empty


# ---------------------------------------------------------------------------
# build_history: fresh-scrape branch persists RAW (basic-cleaned only)
# events, never the ticker-reuse-filtered frame -- see history.py's
# build_history docstring and _basic_clean_events_df / _apply_ticker_reuse_
# filter split. A full rescrape costs about a day per year of history, so
# baking the ticker-reuse drop into the saved parquet would make those rows
# unrecoverable even by a later run with BT_DROP_TICKER_REUSE=0.
# ---------------------------------------------------------------------------
class TestBuildHistoryScrapeBranchPersistsRawEvents:
    def test_saves_unfiltered_returns_filtered(self, tmp_path, monkeypatch):
        """Stub scrape_filings/build_events_df (no network) to hand
        build_history a REUSE-shaped frame, same shape as the other tests in
        this file. The in-memory frame build_history returns must have
        OLDCO's rows dropped (drop_ticker_reuse=True, the default) exactly
        like today, but the parquet it writes to clusters_history/ must
        still contain OLDCO's rows -- proving the ticker-reuse drop never
        reaches the cache."""
        rows = (
            _span_rows("REUSD3", "OLDCO", "Arlington Asset Investment Corp.",
                       [date(2018, 10, 3), date(2020, 9, 28)])
            + _span_rows("REUSD3", "NEWCO", "C3.ai, Inc.",
                         [date(2021, 6, 11), date(2022, 1, 1)])
        )
        raw_df = pd.DataFrame(rows)

        monkeypatch.setattr(history, "scrape_filings", lambda months_back: ([], []))
        monkeypatch.setattr(history, "build_events_df", lambda parsed: raw_df.copy())
        monkeypatch.chdir(tmp_path)

        as_of = date(2022, 1, 1)
        df, _win_start, _win_end, errors, n_dropped = history.build_history(
            months_back=999, as_of=as_of, events_from=None,
        )

        # In-memory frame: filtered, same as today's (unchanged) behavior.
        assert errors == []
        assert n_dropped == 2
        assert not (df["issuer_cik"] == "OLDCO").any()
        assert (df["issuer_cik"] == "NEWCO").sum() == 2

        # Persisted parquet: UNFILTERED -- OLDCO's rows must survive on disk.
        cache_dir = os.path.join(str(tmp_path), history.EVENTS_CACHE_DIR)
        saved = [f for f in os.listdir(cache_dir) if f.startswith("events_") and f.endswith(".parquet")]
        assert len(saved) == 1
        saved_df = pd.read_parquet(os.path.join(cache_dir, saved[0]))
        assert len(saved_df) == len(rows)
        assert (saved_df["issuer_cik"] == "OLDCO").sum() == 2
        assert (saved_df["issuer_cik"] == "NEWCO").sum() == 2

    def test_drop_ticker_reuse_false_matches_saved_file(self, tmp_path, monkeypatch):
        """With the guard disabled, the in-memory frame must equal the
        persisted file row-for-row (nothing to reconcile)."""
        rows = (
            _span_rows("REUSD4", "OLDCO", "Arlington Asset Investment Corp.",
                       [date(2018, 10, 3), date(2020, 9, 28)])
            + _span_rows("REUSD4", "NEWCO", "C3.ai, Inc.",
                         [date(2021, 6, 11), date(2022, 1, 1)])
        )
        raw_df = pd.DataFrame(rows)

        monkeypatch.setattr(history, "scrape_filings", lambda months_back: ([], []))
        monkeypatch.setattr(history, "build_events_df", lambda parsed: raw_df.copy())
        monkeypatch.chdir(tmp_path)

        as_of = date(2022, 1, 1)
        df, _win_start, _win_end, _errors, n_dropped = history.build_history(
            months_back=999, as_of=as_of, events_from=None, drop_ticker_reuse=False,
        )

        assert n_dropped == 0
        assert len(df) == len(rows)

        cache_dir = os.path.join(str(tmp_path), history.EVENTS_CACHE_DIR)
        saved = [f for f in os.listdir(cache_dir) if f.startswith("events_") and f.endswith(".parquet")]
        saved_df = pd.read_parquet(os.path.join(cache_dir, saved[0]))
        assert len(saved_df) == len(df)


# ---------------------------------------------------------------------------
# BT_DROP_TICKER_REUSE env var (backtest.py's _prompt_with_default wiring)
# ---------------------------------------------------------------------------
def _load_backtest_entrypoint():
    """Import root-level backtest.py (the CLI entry point) under an alias.

    `import backtest` resolves to the backtest/ PACKAGE (backtest/__init__.py),
    not this file, because Python prefers a package over a same-named module
    on the same path entry -- confirmed by `python -c "import backtest;
    print(backtest.__file__)"` resolving to backtest/__init__.py. Load the
    script directly by path instead, under a distinct module name, so
    _prompt_with_default can be exercised without colliding with the
    backtest/ package that every other test file imports from.
    """
    path = os.path.join(REPO_ROOT, "backtest.py")
    spec = importlib.util.spec_from_file_location("backtest_cli_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def backtest_cli():
    return _load_backtest_entrypoint()


class TestDropTickerReuseEnvVar:
    def test_env_var_absent_uses_default_on(self, backtest_cli, monkeypatch):
        monkeypatch.delenv("BT_DROP_TICKER_REUSE", raising=False)
        # _prompt_with_default only falls back to input() when the env var
        # is unset; passing a default directly here (as backtest.py itself
        # does for BT_DROP_TICKER_REUSE) still requires stdin if the env var
        # is absent, so simulate "user accepted the default" via monkeypatched
        # input rather than truly blocking on stdin.
        monkeypatch.setattr("builtins.input", lambda *_a, **_kw: "")
        val = backtest_cli._prompt_with_default(
            "Drop unsafe ticker-reuse events? [1/0]", "1", "BT_DROP_TICKER_REUSE",
        )
        assert val == "1"

    def test_env_var_set_to_0_disables_without_touching_stdin(self, backtest_cli, monkeypatch):
        """The whole point of routing this through _prompt_with_default's
        env-var branch is that a backgrounded run with BT_DROP_TICKER_REUSE
        set never blocks on stdin. Fail the test if input() is ever called."""
        monkeypatch.setenv("BT_DROP_TICKER_REUSE", "0")

        def _explode(*_a, **_kw):
            raise AssertionError("_prompt_with_default touched stdin despite the env var being set")

        monkeypatch.setattr("builtins.input", _explode)
        val = backtest_cli._prompt_with_default(
            "Drop unsafe ticker-reuse events? [1/0]", "1", "BT_DROP_TICKER_REUSE",
        )
        assert val == "0"

    def test_env_var_documented_in_module_docstring(self, backtest_cli):
        assert "BT_DROP_TICKER_REUSE" in backtest_cli.__doc__
