"""Ticker symbol normalization and validation."""

from __future__ import annotations

import logging
import re

import pandas as pd

log = logging.getLogger(__name__)


def normalize_ticker(raw: str) -> str | None:
    """Return a clean uppercase ticker, or None if unusable.

    Rules applied in order:
    1. None/NaN/empty -> None
    2. Uppercase and strip whitespace
    3. Strip wrapping punctuation (quotes, parens, brackets, question marks)
    4. Strip exchange prefix (e.g. 'NYSE:FBC' -> 'FBC')
    5. Reject placeholder strings (--,  -, N/A, NA, NONE, NAN, NULL)
    6. If multiple tickers separated by comma/semicolon/slash/whitespace, take first
    7. Final validation: must match ^[A-Z][A-Z0-9]{0,6}([.-][A-Z0-9]{1,4})?$
    """
    # 1. None/NaN/empty -> None
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None

    s = str(raw).strip()
    if not s:
        return None

    # 2. Uppercase and strip whitespace
    s = s.upper().strip()

    # 3. Strip wrapping punctuation (handle unbalanced cases)
    wrapping_chars = {'"', "'", '(', '[', '?', ']', ')'}
    while s and s[0] in wrapping_chars:
        s = s[1:].strip()
    while s and s[-1] in wrapping_chars:
        s = s[:-1].strip()

    if not s:
        return None

    # 4. Strip exchange prefix: anything matching ^[A-Z]+:
    # Strip again afterwards. Real filings write 'NYSE: KRC' with a space, and
    # a leading space would otherwise make the token split below return an
    # empty first token and discard a valid ticker.
    s = re.sub(r'^[A-Z]+:', '', s).strip()

    if not s:
        return None

    # 5. Reject placeholder strings (checked before splitting so 'N/A' is caught)
    if s in ('--', '-', 'N/A', 'NA', 'NONE', 'NAN', 'NULL'):
        return None

    # 6. Take the first token when several symbols share one field. '(' is a
    # separator too, because filings append venue markers like 'NWIN(OB)'.
    # Skip empty tokens so stray leading separators do not discard the symbol.
    tokens = [tok for tok in re.split(r'[,;/\s(]+', s) if tok]
    s = tokens[0].strip() if tokens else ''

    if not s:
        return None

    # 7. Final validation: must match ^[A-Z][A-Z0-9]{0,6}([.-][A-Z0-9]{1,4})?$
    # Pattern: starts with uppercase letter, then 0-6 alphanumeric, optionally
    # followed by . or - and 1-4 alphanumeric (e.g. BRK.B, BRK-B)
    if not re.match(r'^[A-Z][A-Z0-9]{0,6}([.-][A-Z0-9]{1,4})?$', s):
        return None

    return s


def normalize_ticker_series(s: pd.Series) -> pd.Series:
    """Vectorized-ish helper returning normalized values (None where unusable)."""
    return s.apply(normalize_ticker)
