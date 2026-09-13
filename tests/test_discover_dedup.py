"""discover_filings must deduplicate by accession, not by index path.

The SEC daily index lists one row per CIK involved in a filing: once under
the issuer and once under each reporting owner, each with its own
edgar/data/<cik>/<accession>.txt path for the same document. Deduping on the
path therefore never fires, every filing is parsed once per CIK, and every
share and dollar total comes out inflated by that multiple.
"""
import unittest
from unittest import mock

import insider_cluster_buys as ics


def _row(cik: str, adsh: str) -> dict:
    return {"file_name": f"edgar/data/{cik}/{adsh}.txt", "cik": cik}


class TestDiscoverFilingsDedup(unittest.TestCase):
    def _run(self, rows):
        with mock.patch.object(ics, "fetch_daily_index", return_value=rows):
            # lookback 0 so exactly one weekday index is consulted.
            with mock.patch.object(ics, "date") as d:
                import datetime as _dt
                d.today.return_value = _dt.date(2026, 9, 4)  # a Friday
                d.side_effect = lambda *a, **k: _dt.date(*a, **k)
                return ics.discover_filings(0)

    def test_same_accession_under_two_ciks_collapses(self):
        """The common case: issuer CIK and one reporting owner."""
        adsh = "0001193125-26-382783"
        out = self._run([_row("1821468", adsh), _row("1843196", adsh)])
        self.assertEqual(len(out), 1, "one filing indexed twice is one filing")

    def test_same_accession_under_many_ciks_collapses(self):
        """Filings with several reporting owners were inflated up to 10x."""
        adsh = "0001214659-26-011255"
        out = self._run([_row(str(1000 + i), adsh) for i in range(10)])
        self.assertEqual(len(out), 1)

    def test_distinct_accessions_are_kept(self):
        """Dedup must not collapse genuinely different filings."""
        out = self._run([_row("1821468", "0001193125-26-382783"),
                         _row("1821468", "0001193125-26-382784")])
        self.assertEqual(len(out), 2)

    def test_unparseable_path_is_not_dropped(self):
        """A path we cannot read an accession out of is kept, not discarded."""
        out = self._run([{"file_name": "edgar/data/123/not-an-accession.txt",
                          "cik": "123"}])
        self.assertEqual(len(out), 1)


if __name__ == "__main__":
    unittest.main()
