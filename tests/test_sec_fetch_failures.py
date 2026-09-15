"""A day of filings must not go missing without failing the run.

A read timeout on the SEC daily index used to escape _get without a retry
(only 5xx responses were retried), and discover_filings then logged it and
carried on. The 8-year rebuild lost 7 days that way, and a weekly report
could have published run_status: ok while short a day.
"""
import datetime as _dt
import unittest
from unittest import mock

import requests

import insider_cluster_buys as ics


def _ok():
    r = mock.Mock(status_code=200)
    r.raise_for_status.return_value = None
    return r


class TestGetRetriesTransientFailures(unittest.TestCase):
    def _get(self, side_effect):
        with mock.patch.object(ics.SESSION, "get", side_effect=side_effect) as g, \
             mock.patch.object(ics._LIMITER, "acquire"), \
             mock.patch.object(ics.time, "sleep"):
            try:
                return ics._get("https://www.sec.gov/x", accept_html=True), g.call_count
            except Exception as exc:  # noqa: BLE001
                return exc, g.call_count

    def test_timeout_is_retried(self):
        out, calls = self._get([requests.ReadTimeout("slow"), _ok()])
        self.assertEqual(out.status_code, 200)
        self.assertEqual(calls, 2)

    def test_connection_error_is_retried(self):
        out, calls = self._get([requests.ConnectionError("reset"), _ok()])
        self.assertEqual(out.status_code, 200)
        self.assertEqual(calls, 2)

    def test_gives_up_after_five_attempts(self):
        out, calls = self._get([requests.ReadTimeout("slow")] * 5)
        self.assertIsInstance(out, requests.ReadTimeout)
        self.assertEqual(calls, 5)


class TestDiscoverFailsOnAMissingDay(unittest.TestCase):
    def _discover(self, **fetch):
        with mock.patch.object(ics, "fetch_daily_index", **fetch), \
             mock.patch.object(ics, "date") as d:
            d.today.return_value = _dt.date(2026, 9, 4)  # a Friday
            d.side_effect = lambda *a, **k: _dt.date(*a, **k)
            return ics.discover_filings(0)

    def test_unfetchable_weekday_fails_the_run(self):
        with self.assertRaisesRegex(RuntimeError, "2026-09-04"):
            self._discover(side_effect=requests.ReadTimeout("slow"))

    def test_every_weekday_empty_is_a_block_not_holidays(self):
        """SEC answers a blocked client with 403, which reads as a holiday.
        A whole week of "holidays" must fail rather than publish empty."""
        with mock.patch.object(ics, "fetch_daily_index", return_value=[]), \
             mock.patch.object(ics, "date") as d:
            d.today.return_value = _dt.date(2026, 9, 4)  # Fri; 4 days back = 5 weekdays
            d.side_effect = lambda *a, **k: _dt.date(*a, **k)
            with self.assertRaisesRegex(RuntimeError, "block, not holidays"):
                ics.discover_filings(4)

    def test_holiday_with_no_index_is_not_a_failure(self):
        """fetch_daily_index returns [] for a 403/404 (holidays); that is a
        day with no filings, not a missing day."""
        self.assertEqual(self._discover(return_value=[]), [])


if __name__ == "__main__":
    unittest.main()
