"""Storage: job-slot idempotency, round trips, and an honest fallback."""

import shutil
import tempfile
import unittest
from pathlib import Path

from fusion.store import LocalStore, Store, StoreError, SupabaseStore


class TestLocalStore(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.store = Store(None, LocalStore(self.directory))

    def tearDown(self):
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_write_then_read(self):
        self.store.save_candles([{"timestamp": "2026-01-01T00:00:00Z", "pair": "EUR/USD", "close": 1.1}])
        rows = self.store.load_candles("EUR/USD")
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(float(rows[0]["close"]), 1.1)

    def test_a_job_slot_is_claimed_once(self):
        self.assertTrue(self.store.claim_job("signal", "2026-01-01T00:00Z"))
        self.assertFalse(self.store.claim_job("signal", "2026-01-01T00:00Z"))
        # A different bucket is a different job.
        self.assertTrue(self.store.claim_job("signal", "2026-01-01T00:01Z"))

    def test_signal_delivery_state_can_be_updated(self):
        payload = {"signal_id": "abc", "pair": "EUR/USD", "settled": False}
        self.store.save_signal(payload)
        self.store.update_signal("abc", {"delivery_state": "SENT", "message_id": "42"})
        rows = self.store._read("signals")
        self.assertEqual(rows[0]["delivery_state"], "SENT")
        self.assertEqual(rows[0]["message_id"], "42")

    def test_settlement_marks_the_signal_settled(self):
        self.store.save_signal({"signal_id": "abc", "pair": "EUR/USD", "settled": False})
        self.store.save_settlement({"signal_id": "abc", "outcome": "WIN"})
        self.assertEqual(self.store.pending_signals("EUR/USD"), [])

    def test_oldest_candle_reports_the_earliest_bar(self):
        """Backfill starts here so it buys new bars, not ones we already have."""
        self.assertIsNone(self.store.oldest_candle("EUR/USD"))
        for stamp in ("2026-01-03T00:00:00Z", "2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z"):
            self.store.save_candles([{"timestamp": stamp, "pair": "EUR/USD", "close": 1.1}])
        self.assertEqual(self.store.oldest_candle("EUR/USD"), "2026-01-01T00:00:00Z")

    def test_candle_count_lets_backfill_stop_spending(self):
        self.assertEqual(self.store.count_candles("EUR/USD"), 0)
        rows = [
            {"timestamp": f"2026-01-01T00:{index:02d}:00Z", "pair": "EUR/USD", "close": 1.1}
            for index in range(12)
        ]
        self.store.save_candles(rows)
        self.assertEqual(self.store.count_candles("EUR/USD"), 12)

    def test_model_state_is_a_single_row(self):
        self.store.save_model_state({"model": {"weights": [1, 2, 3]}})
        self.store.save_model_state({"model": {"weights": [4, 5, 6]}})
        rows = self.store._read("model_state")
        self.assertEqual(len(rows), 1)
        self.assertEqual(self.store.load_model_state()["model"]["weights"], [4, 5, 6])

    def test_outbox_pending_and_resolved(self):
        self.store.outbox_append({"message_key": "m1", "text": "hello"})
        self.assertEqual(len(self.store.outbox_pending()), 1)
        self.store.outbox_update("m1", {"state": "SENT"})
        self.assertEqual(self.store.outbox_pending(), [])

    def test_alerts_are_recorded(self):
        self.store.alert("stale_feed", "feed is 400s old", dedupe_key="stale|2026-01-01")
        rows = self.store.recent_alerts()
        self.assertEqual(rows[0]["dedupe_key"], "stale|2026-01-01")

    def test_an_alert_key_can_be_recognised_as_already_seen(self):
        """The dedupe key has to be readable back, not just written.

        The bot restarts every cycle, so 'have we told the human?' cannot live
        in memory; it is answered by looking the key up here.
        """
        self.assertFalse(self.store.alert_seen("stale|2026-01-01"))
        self.store.alert("stale_feed", "feed is 400s old", dedupe_key="stale|2026-01-01")
        self.assertTrue(self.store.alert_seen("stale|2026-01-01"))
        self.assertFalse(self.store.alert_seen("stale|2026-01-02"))

    def test_an_empty_key_is_never_treated_as_seen(self):
        self.store.alert("stale_feed", "no key here")
        self.assertFalse(self.store.alert_seen(""))

    def test_latest_heartbeat_is_the_newest(self):
        self.store.heartbeat({"runner": "github-actions", "created_at": "2026-01-01T00:00:00Z"})
        self.store.heartbeat({"runner": "local", "created_at": "2026-01-01T00:01:00Z"})
        self.assertEqual(self.store.latest_heartbeat(runner="github-actions")["created_at"], "2026-01-01T00:00:00Z")

    def test_unknown_tables_do_not_crash(self):
        self.assertEqual(self.store._read("no_such_table"), [])

    def test_every_documented_table_is_reachable(self):
        from fusion.store import TABLES

        self.assertEqual(len(TABLES), 13)
        for table in TABLES:
            self.store._write(table, [{"probe": table}])
            self.assertEqual(self.store._read(table)[-1]["probe"], table)


class TestSupabaseFailure(unittest.TestCase):
    def test_write_errors_degrade_to_the_local_mirror(self):
        directory = Path(tempfile.mkdtemp())

        class Broken(SupabaseStore):
            def _call(self, method, path, *, body=None, headers=None):
                raise StoreError("boom")

        try:
            store = Store(Broken("https://example.supabase.co", "key", fallback=LocalStore(directory)),
                          LocalStore(directory))
            store.save_candles([{"timestamp": "2026-01-01T00:00:00Z", "pair": "EUR/USD"}])
            self.assertTrue(store.errors)
            # The local mirror still holds the row, so nothing is lost silently.
            self.assertEqual(len(store.local.read("candles_m1")), 1)
        finally:
            shutil.rmtree(directory, ignore_errors=True)


class TestPostgrestPaginationAndCounts(unittest.TestCase):
    """PostgREST caps each GET at 1,000 rows; large reads must page."""

    def test_large_candle_reads_page_past_the_thousand_row_ceiling(self):
        directory = Path(tempfile.mkdtemp())
        dataset = [
            {"timestamp": f"2026-01-{(i // 1440) + 1:02d}T{(i // 60) % 24:02d}:{i % 60:02d}:00Z", "pair": "EUR/USD", "close": 1.1}
            for i in range(2_450)
        ]
        # Reverse order matches order=timestamp.desc on the server.
        server_rows = list(reversed(dataset))
        calls: list[str] = []

        class PagedServer(SupabaseStore):
            def _call(self, method, path, *, body=None, headers=None):
                calls.append(path)
                import urllib.parse

                _, _, query = path.partition("?")
                params = urllib.parse.parse_qs(query)
                req_limit = int(params.get("limit", ["1000"])[0])
                offset = int(params.get("offset", ["0"])[0])
                # Enforce Supabase's server-side db-max-rows = 1000 cap.
                capped = min(req_limit, 1000)
                return server_rows[offset : offset + capped]

        try:
            store = Store(PagedServer("https://example.supabase.co", "key"), LocalStore(directory))
            loaded = store.load_candles("EUR/USD", limit=2_300)
            self.assertEqual(len(loaded), 2_300)
            self.assertGreaterEqual(len(calls), 3)
            # Chronologically ascending on return, newest 2,300 preserved.
            self.assertEqual(loaded[-1]["timestamp"], dataset[-1]["timestamp"])
            self.assertEqual(loaded[0]["timestamp"], dataset[-2_300]["timestamp"])
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    def test_count_candles_reads_exact_content_range_header(self):
        directory = Path(tempfile.mkdtemp())
        seen_headers: list[dict] = []

        class HeaderCountingServer(SupabaseStore):
            def _call(self, method, path, *, body=None, headers=None):
                seen_headers.append(dict(headers or {}))
                self.last_response_headers = {"content-range": "0-0/101730"}
                return [{"timestamp": "2026-01-01T00:00:00Z"}]

        try:
            store = Store(HeaderCountingServer("https://example.supabase.co", "key"), LocalStore(directory))
            self.assertEqual(store.count_candles("EUR/USD"), 101_730)
            self.assertEqual(seen_headers[0].get("Prefer"), "count=exact")
        finally:
            shutil.rmtree(directory, ignore_errors=True)


class TestPerMarketStateIsolation(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.store = Store(None, LocalStore(self.directory))

    def tearDown(self):
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_btc_and_eur_model_states_do_not_clobber_each_other(self):
        self.store.save_model_state({"model": {"weights": [1.0]}}, pair="EUR/USD")
        self.store.save_model_state({"model": {"weights": [9.0]}}, pair="BTC/USD")
        self.assertEqual(self.store.load_model_state(pair="EUR/USD")["model"]["weights"], [1.0])
        self.assertEqual(self.store.load_model_state(pair="BTC/USD")["model"]["weights"], [9.0])
        # Default call remains EUR/USD ("champion").
        self.assertEqual(self.store.load_model_state()["id"], "champion")

    def test_btc_and_eur_council_weights_do_not_clobber_each_other(self):
        self.store.save_council_weights({"weights": {"breakout": 0.12}}, pair="EUR/USD")
        self.store.save_council_weights({"weights": {"breakout": 0.28}}, pair="BTC/USD")
        self.assertEqual(self.store.load_council_weights(pair="EUR/USD")["weights"]["breakout"], 0.12)
        self.assertEqual(self.store.load_council_weights(pair="BTC/USD")["weights"]["breakout"], 0.28)


if __name__ == "__main__":
    unittest.main()
