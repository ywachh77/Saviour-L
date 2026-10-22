"""The .env writer must be surgical.

That file holds the Supabase service_role key alongside everything else, so a
rewrite that drops or reorders unrelated lines is not a cosmetic bug.
"""

from __future__ import annotations

import os
import stat
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from fusion import envfile

SAMPLE = [
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
    "# ----------------------------------------------------------------- market",
    "PAIR=EUR/USD",
    "SPREAD_PIPS=1.2",
]


class GetValueTests(unittest.TestCase):
    def test_reads_a_live_value(self) -> None:
        self.assertEqual("1.2", envfile.get_value(SAMPLE, "SPREAD_PIPS"))

    def test_returns_empty_when_unset(self) -> None:
        self.assertEqual("", envfile.get_value(SAMPLE, "NOT_THERE"))

    def test_ignores_a_commented_placeholder(self) -> None:
        lines = ["# PAIR=EUR/GBP"]
        self.assertEqual("", envfile.get_value(lines, "PAIR"))

    def test_ignores_a_key_that_is_only_a_substring(self) -> None:
        lines = ["PAIR_EXTRA=EUR/JPY"]
        self.assertEqual("", envfile.get_value(lines, "PAIR"))

    def test_strips_quotes(self) -> None:
        for raw in ('PAIR="EUR/USD"', "PAIR='EUR/USD'", "PAIR = EUR/USD "):
            with self.subTest(raw=raw):
                self.assertEqual("EUR/USD", envfile.get_value([raw], "PAIR"))


class SetValueTests(unittest.TestCase):
    def test_replaces_a_live_value_in_place(self) -> None:
        result = envfile.set_value(SAMPLE, "PAIR", "EUR/GBP")
        self.assertIn("PAIR=EUR/GBP", result)
        self.assertEqual(envfile.get_value(result, "PAIR"), "EUR/GBP")

    def test_does_not_move_the_line(self) -> None:
        result = envfile.set_value(SAMPLE, "PAIR", "EUR/GBP")
        self.assertEqual(result.index("PAIR=EUR/GBP"), SAMPLE.index("PAIR=EUR/USD"))

    def test_everything_else_survives_untouched(self) -> None:
        result = envfile.set_value(SAMPLE, "TELEGRAM_BOT_TOKEN", "123:secret")
        for line in ("# FUSION BOT configuration",
                     "SUPABASE_SERVICE_ROLE_KEY=abc123",
                     "SPREAD_PIPS=1.2"):
            self.assertIn(line, result)

    def test_an_empty_placeholder_is_filled(self) -> None:
        result = envfile.set_value(SAMPLE, "TELEGRAM_OWNER_CHAT_ID", "42")
        self.assertEqual("42", envfile.get_value(result, "TELEGRAM_OWNER_CHAT_ID"))

    def test_promotes_a_commented_placeholder(self) -> None:
        lines = ["# ---------------- Twelve Data", "# TWELVE_DATA_KEY_1=", "PAIR=EUR/USD"]
        result = envfile.set_value(lines, "TWELVE_DATA_KEY_1", "abc")
        self.assertEqual("abc", envfile.get_value(result, "TWELVE_DATA_KEY_1"))

    def test_drops_duplicate_commented_placeholders(self) -> None:
        lines = ["# TWELVE_DATA_KEY_1=", "# TWELVE_DATA_KEY_2=", "# TWELVE_DATA_KEY_3="]
        result = envfile.set_value(lines, "TWELVE_DATA_KEY_2", "abc")
        self.assertEqual(1, sum("TWELVE_DATA_KEY_2" in line for line in result))

    def test_a_second_live_copy_is_dropped_not_duplicated(self) -> None:
        lines = ["PAIR=EUR/USD", "PAIR=EUR/GBP"]
        result = envfile.set_value(lines, "PAIR", "EUR/JPY")
        self.assertEqual(1, sum(line.startswith("PAIR=") for line in result))
        self.assertEqual("EUR/JPY", envfile.get_value(result, "PAIR"))

    def test_appends_under_a_comment_when_absent(self) -> None:
        result = envfile.set_value(SAMPLE, "NEW_KEY", "v", comment="a new section")
        self.assertIn("# a new section", result)
        self.assertEqual("v", envfile.get_value(result, "NEW_KEY"))
        self.assertEqual("NEW_KEY=v", result[-1])

    def test_setting_twice_leaves_one_line(self) -> None:
        once = envfile.set_value(SAMPLE, "TELEGRAM_BOT_TOKEN", "first")
        twice = envfile.set_value(once, "TELEGRAM_BOT_TOKEN", "second")
        self.assertEqual(1, sum(line.startswith("TELEGRAM_BOT_TOKEN=") for line in twice))
        self.assertEqual("second", envfile.get_value(twice, "TELEGRAM_BOT_TOKEN"))


class RemoveKeyTests(unittest.TestCase):
    def test_removes_live_and_commented_copies(self) -> None:
        lines = ["PAIR=EUR/USD", "# PAIR=EUR/GBP", "SPREAD_PIPS=1.2"]
        result = envfile.remove_key(lines, "PAIR")
        self.assertNotIn("PAIR=EUR/USD", result)
        self.assertNotIn("# PAIR=EUR/GBP", result)
        self.assertIn("SPREAD_PIPS=1.2", result)

    def test_removing_an_absent_key_changes_nothing(self) -> None:
        self.assertEqual(SAMPLE, envfile.remove_key(SAMPLE, "NOT_THERE"))


class WriteTests(unittest.TestCase):
    def test_the_file_is_written_and_locked_down(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            envfile.write(path, ["PAIR=EUR/USD"])
            self.assertEqual("PAIR=EUR/USD\n", path.read_text(encoding="utf-8"))
            self.assertEqual(0o600, stat.S_IMODE(os.stat(path).st_mode))

    def test_a_rewrite_leaves_no_temp_file_behind(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            envfile.write(path, ["A=1"])
            envfile.write(path, ["A=2"])
            self.assertEqual(["A=2"], path.read_text(encoding="utf-8").splitlines())
            self.assertEqual([".env"], sorted(item.name for item in Path(tmp).iterdir()))

    def test_round_trip_preserves_the_whole_file(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            envfile.write(path, SAMPLE)
            reread = path.read_text(encoding="utf-8").splitlines()
            updated = envfile.set_value(reread, "TELEGRAM_BOT_TOKEN", "123:secret")
            envfile.write(path, updated)
            final = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual("123:secret", envfile.get_value(final, "TELEGRAM_BOT_TOKEN"))
            self.assertEqual("abc123", envfile.get_value(final, "SUPABASE_SERVICE_ROLE_KEY"))
            self.assertEqual("1.2", envfile.get_value(final, "SPREAD_PIPS"))


if __name__ == "__main__":
    unittest.main()
