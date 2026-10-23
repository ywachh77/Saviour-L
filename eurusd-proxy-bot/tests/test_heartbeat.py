"""Heartbeat-only liveness.

There is no primary/fallback split any more, so these tests cover the part that
survived: a heartbeat is written each cycle, and its age is the only honest
answer to "is the bot alive?".
"""

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
        row = heartbeat.build_heartbeat(runner="github-actions", phase="minute")
        self.assertFalse(heartbeat.is_stale(row))

    def test_an_old_heartbeat_is_stale(self):
        row = {"runner": "github-actions", "created_at": format_datetime(now() - timedelta(minutes=5))}
        self.assertTrue(heartbeat.is_stale(row))

    def test_a_missing_heartbeat_is_stale(self):
        self.assertTrue(heartbeat.is_stale(None))

    def test_the_stored_heartbeat_decides_whether_the_bot_looks_alive(self):
        # Nothing written yet: the bot has never completed a cycle.
        self.assertTrue(heartbeat.is_stale(self.store.latest_heartbeat(runner="github-actions")))

        self.store.heartbeat(heartbeat.build_heartbeat(runner="github-actions", phase="minute"))
        self.assertFalse(heartbeat.is_stale(self.store.latest_heartbeat(runner="github-actions")))

        # It stops writing: only an ageing heartbeat is left behind.
        quiet = Store(None, LocalStore(self.directory / "quiet"))
        quiet.heartbeat(
            {
                "runner": "github-actions",
                "phase": "minute",
                "created_at": format_datetime(now() - timedelta(seconds=charter.HEARTBEAT_STALE_SECONDS + 30)),
            }
        )
        self.assertTrue(heartbeat.is_stale(quiet.latest_heartbeat(runner="github-actions")))

    def test_the_runner_identifies_itself_as_actions(self):
        self.assertEqual(Config().runner, "github-actions")
        self.assertEqual(Config(runner="github-actions").runner, "github-actions")

    def test_never_having_run_counts_as_stale(self):
        """A fresh container starts with no heartbeat. That must read as dead,
        not as alive-by-default, or a bot that never started looks healthy."""
        self.assertIsNone(heartbeat.stale_seconds_since(None))
        self.assertTrue(heartbeat.is_stale(None))

    def test_age_is_measured_in_seconds(self):
        row = {"created_at": format_datetime(now() - timedelta(seconds=42))}
        age = heartbeat.heartbeat_age_seconds(row)
        self.assertIsNotNone(age)
        self.assertAlmostEqual(age, 42.0, delta=2.0)


if __name__ == "__main__":
    unittest.main()
