"""Tests for insider_cluster_buys.py's _prompt_inputs / _prompt_with_default /
_parse_int_setting -- the env-var fallback for the three interactive
settings (lookback days, min distinct insiders, cluster window days).

Context: _prompt_inputs used bare input() with no env-var escape hatch, so
any backgrounded, piped, or scheduled run (stdin closed / redirected from
nothing) died immediately with EOFError. backtest.py already solved this
class of bug for its own BT_* family via _prompt_with_default (env var wins
if set and logs "Using %s=%s from env", else falls back to input()); this
mirrors that exact mechanism for ICB_LOOKBACK / ICB_MIN_INSIDERS /
ICB_WINDOW_DAYS. See tests/test_history.py's TestDropTickerReuseEnvVar for
the analogous backtest.py coverage this file follows the shape of.

No network, no SEC calls, no full screener run -- _prompt_inputs is pure
input/env plumbing and is exercised directly.
"""

from __future__ import annotations

import pytest

import insider_cluster_buys as ics


# ---------------------------------------------------------------------------
# _prompt_with_default -- shared helper, mirrors backtest.py's
# ---------------------------------------------------------------------------
class TestPromptWithDefault:
    def test_env_var_used_when_set(self, monkeypatch, caplog):
        monkeypatch.setenv(ics.ICB_LOOKBACK_ENV, "45")

        def _explode(*_a, **_kw):
            raise AssertionError("_prompt_with_default touched stdin despite the env var being set")

        monkeypatch.setattr("builtins.input", _explode)
        with caplog.at_level("INFO"):
            val = ics._prompt_with_default("Lookback period in days", "30", ics.ICB_LOOKBACK_ENV)
        assert val == "45"
        assert "Using ICB_LOOKBACK=45 from env" in caplog.text

    def test_prompt_used_when_env_unset(self, monkeypatch):
        monkeypatch.delenv(ics.ICB_LOOKBACK_ENV, raising=False)
        seen_prompts = []

        def _fake_input(prompt=""):
            seen_prompts.append(prompt)
            return ""

        monkeypatch.setattr("builtins.input", _fake_input)
        val = ics._prompt_with_default("Lookback period in days", "30", ics.ICB_LOOKBACK_ENV)
        assert val == "30"
        assert seen_prompts == ["Lookback period in days [default 30]: "]

    def test_empty_env_var_falls_back_to_prompt(self, monkeypatch):
        # os.environ.get(env_var) on an empty string is falsy, matching
        # backtest.py's own `if env_var and os.environ.get(env_var)` check.
        monkeypatch.setenv(ics.ICB_LOOKBACK_ENV, "")
        monkeypatch.setattr("builtins.input", lambda *_a, **_kw: "")
        val = ics._prompt_with_default("Lookback period in days", "30", ics.ICB_LOOKBACK_ENV)
        assert val == "30"


# ---------------------------------------------------------------------------
# _parse_int_setting -- clear, variable-naming error on garbage input
# ---------------------------------------------------------------------------
class TestParseIntSetting:
    def test_valid_int_passes_through(self):
        assert ics._parse_int_setting("7", ics.ICB_LOOKBACK_ENV) == 7

    def test_invalid_value_raises_systemexit_naming_the_variable(self):
        with pytest.raises(SystemExit) as exc_info:
            ics._parse_int_setting("not-a-number", ics.ICB_LOOKBACK_ENV)
        msg = str(exc_info.value)
        assert "ICB_LOOKBACK" in msg
        assert "not-a-number" in msg


# ---------------------------------------------------------------------------
# _prompt_inputs -- end to end, no network
# ---------------------------------------------------------------------------
class TestPromptInputs:
    def test_all_env_vars_set_never_touches_stdin(self, monkeypatch):
        monkeypatch.setenv(ics.ICB_LOOKBACK_ENV, "7")
        monkeypatch.setenv(ics.ICB_MIN_INSIDERS_ENV, "3")
        monkeypatch.setenv(ics.ICB_WINDOW_DAYS_ENV, "21")

        def _explode(*_a, **_kw):
            raise AssertionError("_prompt_inputs touched stdin despite all env vars being set")

        monkeypatch.setattr("builtins.input", _explode)
        lookback, min_insiders, window_days = ics._prompt_inputs()
        assert (lookback, min_insiders, window_days) == (7, 3, 21)

    def test_env_unset_falls_back_to_interactive_prompts(self, monkeypatch):
        monkeypatch.delenv(ics.ICB_LOOKBACK_ENV, raising=False)
        monkeypatch.delenv(ics.ICB_MIN_INSIDERS_ENV, raising=False)
        monkeypatch.delenv(ics.ICB_WINDOW_DAYS_ENV, raising=False)
        answers = iter(["7", "2", "14"])
        monkeypatch.setattr("builtins.input", lambda *_a, **_kw: next(answers))
        lookback, min_insiders, window_days = ics._prompt_inputs()
        assert (lookback, min_insiders, window_days) == (7, 2, 14)

    def test_piped_stdin_still_works(self, monkeypatch, capsys):
        """Regression check for `printf '7\\n2\\n14\\n' | python
        insider_cluster_buys.py`: with no env vars set, input() reads
        sequential lines exactly as it did before this change -- blank
        answers fall back to the documented defaults."""
        monkeypatch.delenv(ics.ICB_LOOKBACK_ENV, raising=False)
        monkeypatch.delenv(ics.ICB_MIN_INSIDERS_ENV, raising=False)
        monkeypatch.delenv(ics.ICB_WINDOW_DAYS_ENV, raising=False)
        piped_lines = iter(["7", "2", ""])  # blank -> default window of 14
        monkeypatch.setattr("builtins.input", lambda *_a, **_kw: next(piped_lines))
        lookback, min_insiders, window_days = ics._prompt_inputs()
        assert (lookback, min_insiders, window_days) == (7, 2, 14)

    def test_invalid_env_value_fails_clearly_not_bare_traceback(self, monkeypatch):
        # Set all three so only ICB_MIN_INSIDERS's bad value is under test --
        # otherwise the earlier ICB_LOOKBACK prompt would hit stdin first.
        monkeypatch.setenv(ics.ICB_LOOKBACK_ENV, "30")
        monkeypatch.setenv(ics.ICB_MIN_INSIDERS_ENV, "two")
        monkeypatch.setenv(ics.ICB_WINDOW_DAYS_ENV, "14")

        def _explode(*_a, **_kw):
            raise AssertionError("should not prompt once the bad env var is read")

        monkeypatch.setattr("builtins.input", _explode)
        with pytest.raises(SystemExit) as exc_info:
            ics._prompt_inputs()
        assert "ICB_MIN_INSIDERS" in str(exc_info.value)
        assert "two" in str(exc_info.value)

    def test_env_vars_documented_in_module_docstring(self):
        assert "ICB_LOOKBACK" in ics.__doc__
        assert "ICB_MIN_INSIDERS" in ics.__doc__
        assert "ICB_WINDOW_DAYS" in ics.__doc__


# ---------------------------------------------------------------------------
# Regression: unrelated env-var opt-out (_price_fetch_enabled) is unaffected
# ---------------------------------------------------------------------------
class TestPriceFetchEnabledUnaffected:
    """_prompt_inputs and _price_fetch_enabled are independent code paths
    (different env vars, different call sites), but both got touched by the
    same PR review pass -- confirm the pre-existing opt-out still behaves
    exactly as tests/test_insider_cluster_buys_model_scoring.py expects."""

    def test_no_prices_flag_still_disables_fetch(self, monkeypatch):
        monkeypatch.delenv(ics.LIVE_SCORE_FETCH_PRICES_ENV, raising=False)
        monkeypatch.setattr("sys.argv", ["insider_cluster_buys.py", "--no-prices"])
        assert ics._price_fetch_enabled() is False

    def test_live_score_fetch_prices_0_still_disables_fetch(self, monkeypatch):
        monkeypatch.setenv(ics.LIVE_SCORE_FETCH_PRICES_ENV, "0")
        monkeypatch.setattr("sys.argv", ["insider_cluster_buys.py"])
        assert ics._price_fetch_enabled() is False

    def test_default_still_enabled(self, monkeypatch):
        monkeypatch.delenv(ics.LIVE_SCORE_FETCH_PRICES_ENV, raising=False)
        monkeypatch.setattr("sys.argv", ["insider_cluster_buys.py"])
        assert ics._price_fetch_enabled() is True
