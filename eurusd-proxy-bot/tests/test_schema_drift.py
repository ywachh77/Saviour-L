"""Guard against the bot writing a column Supabase does not have.

This exists because of a real bug. Migration 0002_add_missing_columns.sql
was written to add five columns the runtime writes that 0001_init.sql had
omitted. The local JSON-lines mirror accepts any key you hand it, so the
entire test suite passed happily while the live database would have answered

    400 Could not find the 'entry_price' column of 'signals' in the schema cache

and dropped the whole insert. Nothing local can catch that, because nothing
local is strict. Hence this file: a snapshot of the real schema, checked
against what the pipeline actually writes.

The snapshot below was read from the live project. If you add a column to the
bot, add it to the migration AND to EXPECTED here -- that is the point. If
it disagrees with production, production wins and the test should be updated
to match, deliberately.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from fusion.simulate import run_simulation, synthetic_series

# Read from the live Supabase project (information_schema.columns).
# Keyed by table; values are the columns that exist in production.
EXPECTED: dict[str, list[str]] = {
    "alerts": ["id", "kind", "message", "dedupe_key", "created_at"],
    "api_keys": ["id", "slots", "cursor", "updated_at"],
    "candles_m1": ["timestamp", "pair", "open", "high", "low", "close", "volume"],
    "council_weights": ["id", "net", "seen", "weights", "version"],
    "heartbeats": [
        "id", "runner", "phase", "paused", "cycles", "seal_state", "reason",
        "created_at", "bucket", "regime", "signal_id",
    ],
    "job_runs": ["id", "job", "bucket", "runner", "created_at"],
    "key_usage": [
        "id", "key_index", "label", "day_bucket", "minute_used", "day_used",
        "created_at",
    ],
    "learning_events": [
        "id", "kind", "signal_id", "applied", "reason", "predicted", "raw",
        "label", "squared_error", "created_at", "outcome",
    ],
    "model_state": [
        "id", "model", "ensemble", "runtime", "calibration_points", "mistakes", "updated_at",
    ],
    "model_versions": ["version_id", "report", "created_at"],
    "outbox": [
        "message_key", "text", "kind", "attempts", "state", "message_id",
        "last_error", "created_at",
    ],
    "settlements": [
        "signal_id", "outcome", "entry_price", "exit_price", "gross_pips",
        "move_pips", "net_pips", "spread_pips", "settled_at", "source",
    ],
    "signals": [
        "signal_id", "pair", "side", "entry_time", "expiry_time",
        "generated_at", "horizon_minutes", "strategy", "reason", "source",
        "probability_up", "probability_raw", "calibration_state",
        "uncertainty", "regime", "council", "features", "spread_pips",
        "seal_state", "delivery_state", "message_id", "settled", "bucket",
        "entry_price",
    ],
}


def _written_columns(directory: Path) -> dict[str, set[str]]:
    """Every column the pipeline touched, per table, from the local mirror."""
    found: dict[str, set[str]] = {}
    for path in sorted(directory.glob("*.jsonl")):
        keys: set[str] = set()
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                keys |= set(json.loads(line).keys())
        found[path.stem] = keys
    return found


class TestSchemaDrift(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp()
        self.directory = Path(self._tmp)

    def tearDown(self) -> None:
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_every_column_the_pipeline_writes_exists_in_production(self) -> None:
        """A column missing here is a 400 and a silent data loss in production.

        The local mirror cannot catch it: LocalStore writes whatever keys it is
        given. Only comparing against the real schema does.
        """
        from fusion.config import Config
        from fusion.runner import FusionBot
        from fusion.store import LocalStore, Store

        bot = FusionBot(
            config=Config(local_state_dir=str(self.directory), runner="github-actions"),
            store=Store(None, LocalStore(self.directory)),
            provider=None,
            telegram=None,
        )
        run_simulation(bot, synthetic_series(600, seed=20260101))

        written = _written_columns(self.directory)
        self.assertTrue(written, "the pipeline wrote nothing, so this proves nothing")

        problems: list[str] = []
        for table, columns in sorted(written.items()):
            if table not in EXPECTED:
                problems.append(f"{table}: table does not exist in production")
                continue
            missing = sorted(columns - set(EXPECTED[table]))
            if missing:
                problems.append(f"{table}: writes {missing}, not in production")

        self.assertFalse(
            problems,
            "Columns the bot writes that Supabase would reject:\n  "
            + "\n  ".join(problems)
            + "\n\nAdd them to supabase/migrations/ and update EXPECTED here.",
        )

    def test_the_snapshot_has_not_lost_a_table(self) -> None:
        """Guards the guard: an empty or trimmed snapshot would pass vacuously."""
        self.assertEqual(
            len(EXPECTED), 13, "the live project has 13 tables; this snapshot has drifted"
        )


if __name__ == "__main__":
    unittest.main()
