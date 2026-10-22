"""Heartbeat-only liveness, and the fallback that depends on it."""

import shutil
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

from fusion import charter, heartbeat
from fusion.config import Config
from fusion.store import LocalStore, Store
from fusion.timeutil import format_datetime, now


class TestHeartbeat(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.store = Store(None, LocalStore(self.directory))

    def tearDown(self):
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_a_fresh_heartbeat_is_not_stale(self):
        row = heartbeat.build_heartbeat(runner="termux", phase="minute")
        self.assertFalse(heartbeat.is_stale(row))

    def test_an_old_heartbeat_is_stale(self):
        row = {"runner": "termux", "created_at": format_datetime(now() - timedelta(minutes=5))}
        self.assertTrue(heartbeat.is_stale(row))

    def test_a_missing_heartbeat_is_stale(self):
        self.assertTrue(heartbeat.is_stale(None))

    def test_fallback_runs_only_when_the_primary_goes_quiet(self):
        self.assertTrue(heartbeat.fallback_should_run(self.store))
        self.store.heartbeat(heartbeat.build_heartbeat(runner="termux", phase="minute"))
        self.assertFalse(heartbeat.fallback_should_run(self.store))
        # The primary goes quiet: only a stale heartbeat is left behind.
        quiet = Store(None, LocalStore(self.directory / "quiet"))
        quiet.heartbeat(
            {
                "runner": "termux",
                "phase": "minute",
                "created_at": format_datetime(now() - timedelta(seconds=charter.HEARTBEAT_STALE_SECONDS + 30)),
            }
        )
        self.assertTrue(heartbeat.fallback_should_run(quiet))

    def test_the_fallback_tags_itself(self):
        config = Config(runner="actions-fallback")
        self.assertEqual(config.runner, "actions-fallback")

    def test_age_is_measured_in_seconds(self):
        row = {"created_at": format_datetime(now() - timedelta(seconds=42))}
        age = heartbeat.heartbeat_age_seconds(row)
        self.assertIsNotNone(age)
        self.assertAlmostEqual(age, 42.0, delta=2.0)


if __name__ == "__main__":
    unittest.main()
