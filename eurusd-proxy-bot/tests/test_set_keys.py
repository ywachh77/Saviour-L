"""The key-import helper must be safe to run against a real .env.

It edits the file that holds your Supabase service_role key, so the rewrite has
to be surgical: replace the Twelve Data block, touch nothing else, and never
print a key in full.
"""

from __future__ import annotations

import importlib.util
import os
import stat
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from fusion.keypool import MAX_SLOTS, mask_key

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "set_keys.py"


def _load():
    spec = importlib.util.spec_from_file_location("set_keys", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


set_keys = _load()


def key(number: int) -> str:
    """A plausible-looking 32 character key."""
    return f"{number:032x}"


SAMPLE_ENV = [
    "# FUSION BOT configuration",
    "",
    "# ---------------------------------------------------------------- Telegram",
    "TELEGRAM_BOT_TOKEN=",
    "TELEGRAM_OWNER_CHAT_ID=",
    "",
    "# ---------------------------------------------------------------- Supabase",
    "SUPABASE_URL=https://example.supabase.co",
    "SUPABASE_SERVICE_ROLE_KEY=abc123",
    "",
    "# ------------------------------------------------------------ Twelve Data",
    "TWELVE_DATA_API_KEY=",
    "# TWELVE_DATA_KEY_1=",
    "",
    "# ----------------------------------------------------------------- market",
    "PAIR=EUR/USD",
]


class ParseTests(unittest.TestCase):
    """Reading keys out of pasted or filed input."""

    def test_skips_blanks_and_comments(self) -> None:
        keys, problems = set_keys.normalise_keys(["", "   ", "# a comment", key(1)])
        self.assertEqual([key(1)], keys)
        self.assertEqual([], problems)

    def test_strips_whitespace_and_quotes(self) -> None:
        keys, _ = set_keys.normalise_keys([f'  "{key(1)}"  ', f"'{key(2)}'"])
        self.assertEqual([key(1), key(2)], keys)

    def test_drops_duplicates_keeping_first_order(self) -> None:
        keys, problems = set_keys.normalise_keys([key(2), key(1), key(2)])
        self.assertEqual([key(2), key(1)], keys)
        self.assertTrue(any("duplicate" in p for p in problems))

    def test_rejects_values_with_internal_whitespace(self) -> None:
        keys, problems = set_keys.normalise_keys(["abc def ghijklmnop"])
        self.assertEqual([], keys)
        self.assertTrue(any("whitespace" in p for p in problems))

    def test_rejects_values_that_are_too_short(self) -> None:
        keys, problems = set_keys.normalise_keys(["abc"])
        self.assertEqual([], keys)
        self.assertTrue(any("shorter" in p for p in problems))

    def test_warns_but_keeps_an_unusual_length(self) -> None:
        keys, problems = set_keys.normalise_keys(["a" * 20])
        self.assertEqual(1, len(keys))
        self.assertTrue(any("not the usual" in p for p in problems))

    def test_caps_at_the_pool_maximum(self) -> None:
        keys, problems = set_keys.normalise_keys([key(i) for i in range(MAX_SLOTS + 7)])
        self.assertEqual(MAX_SLOTS, len(keys))
        self.assertTrue(any("first 30" in p for p in problems))


class RewriteTests(unittest.TestCase):
    """Rewriting the block must not disturb anything around it."""

    def test_everything_outside_the_key_block_survives(self) -> None:
        result = set_keys.rewrite_key_block(SAMPLE_ENV, [key(1), key(2)])
        joined = "\n".join(result)
        self.assertIn("SUPABASE_SERVICE_ROLE_KEY=abc123", joined)
        self.assertIn("TELEGRAM_BOT_TOKEN=", joined)
        self.assertIn("PAIR=EUR/USD", joined)

    def test_multiple_keys_go_to_numbered_slots(self) -> None:
        result = set_keys.rewrite_key_block(SAMPLE_ENV, [key(1), key(2), key(3)])
        lines = "\n".join(result)
        self.assertIn(f"TWELVE_DATA_KEY_1={key(1)}", lines)
        self.assertIn(f"TWELVE_DATA_KEY_3={key(3)}", lines)
        self.assertNotIn("TWELVE_DATA_KEY_4", lines)

    def test_a_single_key_uses_the_plain_variable(self) -> None:
        result = set_keys.rewrite_key_block(SAMPLE_ENV, [key(9)])
        self.assertIn(f"TWELVE_DATA_API_KEY={key(9)}", result)

    def test_rewriting_twice_does_not_stack(self) -> None:
        once = set_keys.rewrite_key_block(SAMPLE_ENV, [key(1), key(2)])
        twice = set_keys.rewrite_key_block(once, [key(1), key(2)])
        self.assertEqual(once, twice)

    def test_replacing_fewer_keys_removes_the_older_ones(self) -> None:
        once = set_keys.rewrite_key_block(SAMPLE_ENV, [key(1), key(2), key(3)])
        fewer = set_keys.rewrite_key_block(once, [key(5)])
        self.assertNotIn(f"TWELVE_DATA_KEY_3={key(3)}", fewer)
        self.assertNotIn("TWELVE_DATA_KEY_2", "\n".join(fewer))

    def test_clearing_leaves_an_empty_variable(self) -> None:
        once = set_keys.rewrite_key_block(SAMPLE_ENV, [key(1), key(2)])
        cleared = set_keys.rewrite_key_block(once, [])
        self.assertIn("TWELVE_DATA_API_KEY=", cleared)
        self.assertNotIn(key(1), "\n".join(cleared))

    def test_a_file_with_no_key_block_gets_one_appended(self) -> None:
        result = set_keys.rewrite_key_block(["PAIR=EUR/USD", ""], [key(1)])
        self.assertEqual("PAIR=EUR/USD", result[0])
        self.assertIn(f"TWELVE_DATA_API_KEY={key(1)}", result)


class RoundTripTests(unittest.TestCase):
    """What is written must be exactly what is read back."""

    def _round_trip(self, keys, tmp_path: Path) -> None:
        env = tmp_path / ".env"
        env.write_text("\n".join(SAMPLE_ENV) + "\n", encoding="utf-8")
        set_keys.write_env(env, set_keys.rewrite_key_block(SAMPLE_ENV, keys))
        self.assertEqual(keys, set_keys.read_keys_from_env(env))

    def test_one_key(self) -> None:
        with TemporaryDirectory() as tmp:
            self._round_trip([key(1)], Path(tmp))

    def test_thirty_keys(self) -> None:
        with TemporaryDirectory() as tmp:
            self._round_trip([key(i) for i in range(30)], Path(tmp))

    def test_no_keys(self) -> None:
        with TemporaryDirectory() as tmp:
            self._round_trip([], Path(tmp))

    def test_permissions_are_locked_to_the_owner(self) -> None:
        with TemporaryDirectory() as tmp:
            env = Path(tmp) / ".env"
            env.write_text("PAIR=EUR/USD\n", encoding="utf-8")
            set_keys.write_env(env, ["PAIR=EUR/USD"])
            mode = stat.S_IMODE(os.stat(env).st_mode)
            self.assertEqual(0o600, mode)

    def test_masking_never_leaks_the_whole_key(self) -> None:
        secret = key(42)
        shown = mask_key(secret)
        self.assertNotIn(secret, shown)
        self.assertTrue(shown.startswith(secret[:4]))
        self.assertIn(str(len(secret)), shown)


if __name__ == "__main__":
    unittest.main()
