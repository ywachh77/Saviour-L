"""Guard the contract between the bot's write payloads and the database schema.

The local JSONL mirror stores any key you hand it, so every other test in this
suite is blind to a column the bot writes but the database does not have. Live,
PostgREST answers that insert with

    400 Could not find the 'entry_price' column of 'signals' in the schema cache

and the row is silently dropped. That is how signals.entry_price,
learning_events.outcome and three heartbeat columns shipped missing.

Two tests hold the contract shut, in opposite directions:

  * the replay test replays every write path and checks each payload key
    against EXPECTED_COLUMNS;
  * the migration test parses supabase/migrations/*.sql and checks
    EXPECTED_COLUMNS is really declared there.

Adding a field to a payload without a migration now fails here instead of
losing data at 3am.
"""

from __future__ import annotations

import contextlib
import io
import re
import tempfile
import unittest
from pathlib import Path

from fusion import charter
from fusion.config import Config
from fusion.keypool import KeyPool
from fusion.runner import FusionBot
from fusion.simulate import SyntheticFeed, run_simulation, synthetic_series
from fusion.store import LocalStore, Store
from fusion.timeutil import now

ROOT = Path(__file__).resolve().parents[1]
MIGRATIONS = ROOT / "supabase" / "migrations"

# The columns each table may receive, mirroring supabase/migrations/.
EXPECTED_COLUMNS: dict[str, set[str]] = {
    "candles_m1": {
        "timestamp", "pair", "open", "high", "low", "close", "volume",
    },
    "signals": {
        "signal_id", "pair", "side", "entry_time", "expiry_time", "generated_at",
        "horizon_minutes", "strategy", "reason", "source", "probability_up",
        "probability_raw", "calibration_state", "uncertainty", "regime", "council",
        "features", "spread_pips", "seal_state", "delivery_state", "message_id",
        "settled", "bucket", "entry_price",
    },
    "settlements": {
        "signal_id", "outcome", "entry_price", "exit_price", "gross_pips",
        "move_pips", "net_pips", "spread_pips", "settled_at", "source",
    },
    "model_state": {
        "id", "model", "ensemble", "runtime", "calibration_points", "mistakes", "updated_at",
    },
    "model_versions": {"version_id", "report", "created_at"},
    "council_weights": {"id", "net", "seen", "weights", "version"},
    "learning_events": {
        "id", "kind", "signal_id", "applied", "reason", "predicted", "raw",
        "label", "squared_error", "created_at", "outcome",
    },
    "job_runs": {"id", "job", "bucket", "runner", "created_at"},
    "heartbeats": {
        "id", "runner", "phase", "paused", "cycles", "seal_state", "reason",
        "created_at", "bucket", "regime", "signal_id",
    },
    "api_keys": {"id", "slots", "cursor", "updated_at"},
    "key_usage": {
        "id", "key_index", "label", "day_bucket", "minute_used", "day_used",
        "created_at",
    },
    "outbox": {
        "message_key", "text", "kind", "attempts", "state", "message_id",
        "last_error", "created_at",
    },
    "alerts": {"id", "kind", "message", "dedupe_key", "created_at"},
}


def _declared_columns() -> dict[str, set[str]]:
    """Parse the column names the migration files actually declare."""
    sql = "\n".join(path.read_text() for path in sorted(MIGRATIONS.glob("*.sql")))
    tables: dict[str, set[str]] = {}

    # create table if not exists public.<name> ( ... \n);
    # Anchored to the line start: an unanchored pattern also matches text inside
    # a "--" comment, which is exactly what a commented-out migration looks like.
    for match in re.finditer(
        r"^create table if not exists public\.(\w+)\s*\((.*?)\n\);", sql, re.S | re.M
    ):
        name, body = match.group(1), match.group(2)
        columns = tables.setdefault(name, set())
        for line in body.splitlines():
            line = line.strip().rstrip(",")
            if not line or line.startswith("--"):
                continue
            token = line.split()[0]
            if token in ("primary", "unique", "foreign", "constraint", "check"):
                continue
            columns.add(token)

    # alter table public.<name> add column if not exists <column>
    for match in re.finditer(
        r"^\s*alter table public\.(\w+)\s+add column if not exists (\w+)", sql, re.M
    ):
        tables.setdefault(match.group(1), set()).add(match.group(2))

    return tables


_CACHE: dict[str, set[str]] | None = None


def _replay_all_write_paths() -> dict[str, set[str]]:
    """Run the whole pipeline and record the columns written to each table."""
    global _CACHE
    if _CACHE is not None:
        return _CACHE

    import fusion.store as store_mod

    recorded: dict[str, set[str]] = {}
    real_write = store_mod.Store._write
    real_upsert = store_mod.SupabaseStore.upsert
    real_local_write = store_mod.LocalStore.write
    real_local_replace = store_mod.LocalStore.replace

    def record(table: str, rows) -> None:
        for row in rows:
            recorded.setdefault(table, set()).update(row.keys())

    def patched_write(self, table, rows, *, on_conflict=None):
        record(table, rows)
        return real_write(self, table, rows, on_conflict=on_conflict)

    def patched_upsert(self, table, rows, on_conflict="id"):
        record(table, rows)
        return real_upsert(self, table, rows, on_conflict)

    def patched_local_write(self, table, rows):
        record(table, list(rows))
        return real_local_write(self, table, rows)

    def patched_local_replace(self, table, rows):
        record(table, list(rows))
        return real_local_replace(self, table, rows)

    store_mod.Store._write = patched_write
    store_mod.SupabaseStore.upsert = patched_upsert
    store_mod.LocalStore.write = patched_local_write
    store_mod.LocalStore.replace = patched_local_replace
    try:
        with tempfile.TemporaryDirectory(prefix="fusion-schema-") as tmp:
            config = Config(
                pair=charter.ALLOWED_PAIR,
                timeframe=charter.ALLOWED_TIMEFRAME,
                horizon_minutes=3,
                local_state_dir=tmp,
                runner="schema-contract",
            )
            store = Store(None, LocalStore(Path(tmp)))
            bot = FusionBot(config=config, store=store, provider=None, telegram=None)
            candles = synthetic_series(900, seed=20260101)

            with contextlib.redirect_stdout(io.StringIO()):
                run_simulation(bot, candles, feed=SyntheticFeed(candles))
                bot.nightly_job(now())
                # Paths the simulation cannot reach on its own.
                store.save_key_pool(KeyPool(["dummy-key"]).to_dict())
                store.log_key_usage(
                    [
                        {
                            "key_index": 0,
                            "label": "slot-1",
                            "day_bucket": "2026-10-22",
                            "minute_used": 1,
                            "day_used": 1,
                            "created_at": "2026-10-22T00:00:00Z",
                        }
                    ]
                )
                store.alert("selftest", "schema contract check")
                store.claim_job("selftest", "bucket-x", runner="schema-contract")
    finally:
        store_mod.Store._write = real_write
        store_mod.SupabaseStore.upsert = real_upsert
        store_mod.LocalStore.write = real_local_write
        store_mod.LocalStore.replace = real_local_replace

    _CACHE = recorded
    return recorded


class SchemaContractTests(unittest.TestCase):
    """The bot's payloads and the database schema must agree."""

    def test_the_replay_touches_every_table(self) -> None:
        written = _replay_all_write_paths()
        self.assertEqual(set(EXPECTED_COLUMNS), set(written))

    def test_every_column_the_bot_writes_is_declared_for_that_table(self) -> None:
        written = _replay_all_write_paths()
        for table, columns in sorted(written.items()):
            with self.subTest(table=table):
                self.assertIn(table, EXPECTED_COLUMNS)
                self.assertEqual(
                    set(),
                    columns - EXPECTED_COLUMNS[table],
                    f"the bot writes columns {sorted(columns - EXPECTED_COLUMNS[table])} "
                    f"to {table} that the schema does not declare",
                )

    def test_migrations_declare_every_expected_column(self) -> None:
        declared = _declared_columns()
        self.assertEqual(set(EXPECTED_COLUMNS), set(declared))
        for table, columns in sorted(EXPECTED_COLUMNS.items()):
            with self.subTest(table=table):
                self.assertEqual(
                    set(),
                    columns - declared[table],
                    f"{table} columns {sorted(columns - declared[table])} are declared "
                    f"in this test but missing from supabase/migrations/",
                )


if __name__ == "__main__":
    unittest.main()
