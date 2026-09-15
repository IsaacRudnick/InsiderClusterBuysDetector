"""Where the current research dataset and events file are.

Tools used to pin research_10861rows_20260813.parquet or
events_20180813_20260813.parquet as their default input. Those files still
exist after the 2026-09-14 rebuild -- they are the double-counted data -- so
a tool run with defaults silently measured the wrong thing. Resolve the
newest file instead.

"Newest" is decided by the date in the filename, never by name order or
mtime: name order puts events_20200804_... after events_20180914_...
although it is the older scrape, and copying a file resets its mtime.
"""

from __future__ import annotations

import os
import re

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_DATASET_RE = re.compile(r"research_(\d+)rows_(\d{8})\.parquet")
_EVENTS_RE = re.compile(r"events_(\d{8})_(\d{8})\.parquet")

# Returned when nothing matches, so importing a tool never fails in a
# checkout without data. Reading it then fails loudly on the missing file.
_FALLBACK_DATASET = "research_10861rows_20260813.parquet"
_FALLBACK_EVENTS = "events_20180813_20260813.parquet"


def _pick(dirname: str, rx: re.Pattern, key, fallback: str, required: bool) -> str:
    d = os.path.join(REPO_ROOT, dirname)
    try:
        names = [n for n in os.listdir(d) if rx.fullmatch(n)]
    except FileNotFoundError:
        names = []
    if not names:
        if required:
            raise SystemExit(f"no file matching {rx.pattern} in {d}")
        return os.path.join(d, fallback)
    return os.path.join(d, max(names, key=lambda n: key(rx.fullmatch(n))))


def latest_research_dataset(required: bool = False) -> str:
    """research_<N>rows_<YYYYMMDD>.parquet with the latest date, most rows on
    a tie. Tagged variants (research_groupE_...) are experiments and never
    the default."""
    return _pick("research_data", _DATASET_RE,
                 lambda m: (m.group(2), int(m.group(1))),
                 _FALLBACK_DATASET, required)


def latest_events_file(required: bool = False) -> str:
    """events_<start>_<end>.parquet with the latest end date, widest window
    on a tie."""
    return _pick("clusters_history", _EVENTS_RE,
                 lambda m: (m.group(2), -int(m.group(1))),
                 _FALLBACK_EVENTS, required)
