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


if __name__ == "__main__":
    unittest.main()
