"""Tests for ticker normalization and validation."""

from __future__ import annotations

import pytest
import pandas as pd

from backtest.tickers import normalize_ticker, normalize_ticker_series


class TestNormalizeTickerBasicPassthrough:
    """Test cases where tickers pass through unchanged."""

    def test_uppercase_ticker(self) -> None:
        assert normalize_ticker("AAPL") == "AAPL"

    def test_whitespace_stripping(self) -> None:
        assert normalize_ticker(" msft ") == "MSFT"

    def test_lowercase_uppercasing(self) -> None:
        assert normalize_ticker("aapl") == "AAPL"

    def test_ticker_with_dot(self) -> None:
        assert normalize_ticker("BRK.B") == "BRK.B"

    def test_ticker_with_dash(self) -> None:
        assert normalize_ticker("BRK-B") == "BRK-B"

    def test_whitespace_and_case_combination(self) -> None:
        assert normalize_ticker("  GooGL  ") == "GOOGL"


class TestNormalizeTickerQuotedAndWrapped:
    """Test cases with wrapping punctuation."""

    def test_double_quoted_ticker(self) -> None:
        assert normalize_ticker('"OMEX"') == "OMEX"

    def test_single_quoted_ticker(self) -> None:
        assert normalize_ticker("'OSH'") == "OSH"

    def test_parenthesized_ticker(self) -> None:
        assert normalize_ticker("(CALX)") == "CALX"

    def test_square_bracketed_ticker(self) -> None:
        assert normalize_ticker("[LUMO]") == "LUMO"

    def test_question_mark_wrapped(self) -> None:
        assert normalize_ticker("?NHF?") == "NHF"

    def test_unbalanced_leading_paren(self) -> None:
        assert normalize_ticker("(NUGN") == "NUGN"

    def test_unbalanced_trailing_bracket(self) -> None:
        assert normalize_ticker("AREN]") == "AREN"

    def test_mixed_wrapping_chars(self) -> None:
        assert normalize_ticker('([TARA])') == "TARA"


class TestNormalizeTickerExchangePrefix:
    """Test cases with exchange prefixes."""

    def test_nyse_prefix(self) -> None:
        assert normalize_ticker("NYSE:FBC") == "FBC"

    def test_asx_prefix(self) -> None:
        assert normalize_ticker("ASX:CRN") == "CRN"

    def test_exchange_prefix_with_wrapping(self) -> None:
        assert normalize_ticker("(NYSE:FBC)") == "FBC"


class TestNormalizeTickerMultipleTickers:
    """Test cases with multiple tickers separated by various delimiters."""

    def test_comma_separated(self) -> None:
        assert normalize_ticker("AMC,APE") == "AMC"

    def test_comma_with_space(self) -> None:
        assert normalize_ticker("BFA, BFB") == "BFA"

    def test_semicolon_separated(self) -> None:
        assert normalize_ticker("BCDA;BCDAW") == "BCDA"

    def test_slash_separated(self) -> None:
        assert normalize_ticker("BBX/BBXTB") == "BBX"

    def test_slash_separated_with_alt_form(self) -> None:
        assert normalize_ticker("ASAQ/U") == "ASAQ"

    def test_slash_in_compound_ticker(self) -> None:
        assert normalize_ticker("BBXIA/B") == "BBXIA"

    def test_multiple_delimiters_comma_wins(self) -> None:
        # Should split on comma first based on order in regex
        assert normalize_ticker("ABC,DEF;GHI") == "ABC"


class TestNormalizeTickerPlaceholders:
    """Test cases for placeholder rejection."""

    def test_double_dash(self) -> None:
        assert normalize_ticker("--") is None

    def test_single_dash(self) -> None:
        assert normalize_ticker("-") is None

    def test_na_string(self) -> None:
        assert normalize_ticker("NA") is None

    def test_na_slash(self) -> None:
        # N/A as a complete string is a placeholder, checked before splitting
        assert normalize_ticker("N/A") is None

    def test_none_uppercase(self) -> None:
        assert normalize_ticker("NONE") is None

    def test_nan_uppercase(self) -> None:
        assert normalize_ticker("NAN") is None

    def test_null_uppercase(self) -> None:
        assert normalize_ticker("NULL") is None


class TestNormalizeTickerInvalid:
    """Test cases for invalid tickers that should be rejected."""

    def test_double_dot_ticker(self) -> None:
        # ACRG.A.U has two dots, fails validation
        assert normalize_ticker("ACRG.A.U") is None

    def test_starts_with_asterisk(self) -> None:
        assert normalize_ticker("*H6ZMFDX") is None

    def test_starts_with_digit(self) -> None:
        assert normalize_ticker("9QGNC@RY") is None

    def test_too_many_dots_invalid(self) -> None:
        # More complex validation failure case
        assert normalize_ticker("A.B.C.D") is None

    def test_contains_at_sign(self) -> None:
        assert normalize_ticker("9QGNC@RY") is None

    def test_too_long(self) -> None:
        # More than 7 alphanumeric characters before optional dot/dash
        assert normalize_ticker("VERYLONGNAME") is None

    def test_dot_at_end(self) -> None:
        # Dot needs to be followed by 1-4 characters
        assert normalize_ticker("TICKER.") is None

    def test_empty_after_dot(self) -> None:
        assert normalize_ticker("BRK.") is None


class TestNormalizeTickerNoneAndNaN:
    """Test cases for None and NaN inputs."""

    def test_none_input(self) -> None:
        assert normalize_ticker(None) is None

    def test_nan_float_input(self) -> None:
        assert normalize_ticker(float("nan")) is None

    def test_empty_string(self) -> None:
        assert normalize_ticker("") is None

    def test_whitespace_only(self) -> None:
        assert normalize_ticker("   ") is None


class TestNormalizeTickerEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_letter_ticker(self) -> None:
        # Minimum valid ticker: must have at least 1 char after first letter
        # Actually, the pattern is ^[A-Z][A-Z0-9]{0,6}, so "A" alone is valid
        assert normalize_ticker("A") == "A"

    def test_two_letter_ticker(self) -> None:
        assert normalize_ticker("AT") == "AT"

    def test_eight_letter_ticker_too_long(self) -> None:
        # Pattern allows max 7 alphanumeric after first letter (8 total including first)
        assert normalize_ticker("VERYLNG") == "VERYLNG"  # 7 letters, valid
        assert normalize_ticker("VERYLONGS") is None  # 8 letters, invalid

    def test_dot_with_one_char(self) -> None:
        # [.-][A-Z0-9]{1,4} means 1-4 chars after dot
        assert normalize_ticker("BRK.A") == "BRK.A"

    def test_dot_with_four_chars(self) -> None:
        assert normalize_ticker("BRK.ABCD") == "BRK.ABCD"

    def test_dot_with_five_chars_too_long(self) -> None:
        # After dot/dash, only 1-4 alphanumeric allowed
        assert normalize_ticker("BRK.ABCDE") is None

    def test_dash_variant(self) -> None:
        assert normalize_ticker("BRK-A") == "BRK-A"

    def test_with_uppercase_conversion(self) -> None:
        assert normalize_ticker("brk.b") == "BRK.B"


class TestNormalizeTickerRealMalformedExamples:
    """Test with the real malformed examples from the task."""

    def test_quoted_omex(self) -> None:
        assert normalize_ticker('"OMEX"') == "OMEX"

    def test_quoted_osh(self) -> None:
        assert normalize_ticker('"OSH"') == "OSH"

    def test_quoted_tara(self) -> None:
        assert normalize_ticker('"TARA"') == "TARA"

    def test_paren_calx(self) -> None:
        assert normalize_ticker("(CALX)") == "CALX"

    def test_paren_lumo(self) -> None:
        assert normalize_ticker("(LUMO)") == "LUMO"

    def test_unbalanced_paren_nugn(self) -> None:
        assert normalize_ticker("(NUGN") == "NUGN"

    def test_paren_nyse_fbc(self) -> None:
        assert normalize_ticker("(NYSE:FBC)") == "FBC"

    def test_paren_siri(self) -> None:
        assert normalize_ticker("(SIRI)") == "SIRI"

    def test_asterisk_h6zmfdx(self) -> None:
        assert normalize_ticker("*H6ZMFDX") is None

    def test_digit_start_9qgnc_ry(self) -> None:
        assert normalize_ticker("9QGNC@RY") is None

    def test_question_nhf(self) -> None:
        # Question marks are stripped as wrapping punctuation
        assert normalize_ticker("?NHF?") == "NHF"

    def test_question_osh(self) -> None:
        # Question marks are stripped as wrapping punctuation
        assert normalize_ticker("?OSH?") == "OSH"

    def test_dot_acrg_a_u(self) -> None:
        assert normalize_ticker("ACRG.A.U") is None

    def test_comma_amc_ape(self) -> None:
        assert normalize_ticker("AMC,APE") == "AMC"

    def test_bracket_aren(self) -> None:
        assert normalize_ticker("AREN]") == "AREN"

    def test_slash_asaq_u(self) -> None:
        assert normalize_ticker("ASAQ/U") == "ASAQ"

    def test_exchange_asx_crn(self) -> None:
        assert normalize_ticker("ASX:CRN") == "CRN"

    def test_slash_bbx_bbxtb(self) -> None:
        assert normalize_ticker("BBX/BBXTB") == "BBX"

    def test_comma_space_bfa_bfb(self) -> None:
        assert normalize_ticker("BFA, BFB") == "BFA"

    def test_semicolon_bcda_bcdaw(self) -> None:
        assert normalize_ticker("BCDA;BCDAW") == "BCDA"

    def test_slash_bbxia_b(self) -> None:
        assert normalize_ticker("BBXIA/B") == "BBXIA"

    def test_acrg_variant(self) -> None:
        # Just to ensure single-dot variants work
        assert normalize_ticker("ACRG.A") == "ACRG.A"


class TestNormalizeTickerSeries:
    """Test the Series vectorization helper."""

    def test_series_basic(self) -> None:
        s = pd.Series(["AAPL", "MSFT", "GOOGL"])
        result = normalize_ticker_series(s)
        expected = pd.Series(["AAPL", "MSFT", "GOOGL"])
        pd.testing.assert_series_equal(result, expected)

    def test_series_mixed(self) -> None:
        s = pd.Series(['AAPL', '"OMEX"', '--', 'NYSE:FBC', 'AMC,APE'])
        result = normalize_ticker_series(s)
        expected = pd.Series(["AAPL", "OMEX", None, "FBC", "AMC"])
        pd.testing.assert_series_equal(result, expected)

    def test_series_with_nan(self) -> None:
        s = pd.Series(['AAPL', float('nan'), 'MSFT'])
        result = normalize_ticker_series(s)
        expected = pd.Series(['AAPL', None, 'MSFT'])
        pd.testing.assert_series_equal(result, expected)

    def test_series_with_none(self) -> None:
        s = pd.Series(['AAPL', None, 'MSFT'])
        result = normalize_ticker_series(s)
        expected = pd.Series(['AAPL', None, 'MSFT'])
        pd.testing.assert_series_equal(result, expected)

    def test_series_empty(self) -> None:
        s = pd.Series([], dtype=object)
        result = normalize_ticker_series(s)
        expected = pd.Series([], dtype=object)
        pd.testing.assert_series_equal(result, expected)
