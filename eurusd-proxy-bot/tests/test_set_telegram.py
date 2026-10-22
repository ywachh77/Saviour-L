"""The Telegram credential helper: validate before storing, never echo in full."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from fusion import envfile
from fusion.keypool import mask_key

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "set_telegram.py"


def _load():
    spec = importlib.util.spec_from_file_location("set_telegram", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


set_telegram = _load()

# A realistic-looking bot token: numeric id, colon, 35-character secret.
GOOD_TOKEN = "123456789:AAF" + "k" * 32


class TokenValidationTests(unittest.TestCase):
    def test_accepts_a_well_formed_token(self) -> None:
        ok, note = set_telegram.validate_token(GOOD_TOKEN)
        self.assertTrue(ok)
        self.assertEqual("", note)

    def test_rejects_an_empty_value(self) -> None:
        ok, _ = set_telegram.validate_token("")
        self.assertFalse(ok)

    def test_rejects_a_value_with_no_colon(self) -> None:
        ok, note = set_telegram.validate_token("123456789AAFabcdef")
        self.assertFalse(ok)
        self.assertIn("colon", note)

    def test_rejects_a_non_numeric_bot_id(self) -> None:
        ok, _ = set_telegram.validate_token("bot123:AAFabcdefghijklmnop")
        self.assertFalse(ok)

    def test_rejects_whitespace_inside_the_token(self) -> None:
        ok, note = set_telegram.validate_token("123456789:AAF abcdef")
        self.assertFalse(ok)
        self.assertIn("whitespace", note)

    def test_warns_but_keeps_a_short_secret(self) -> None:
        ok, note = set_telegram.validate_token("123456789:AAFshort")
        self.assertTrue(ok)
        self.assertIn("truncated", note)

    def test_strips_surrounding_whitespace_and_quotes(self) -> None:
        ok, _ = set_telegram.validate_token(f'  "{GOOD_TOKEN}"  ')
        self.assertTrue(ok)


class ChatIdValidationTests(unittest.TestCase):
    def test_accepts_a_plain_numeric_id(self) -> None:
        ok, note = set_telegram.validate_chat_id("987654321")
        self.assertTrue(ok)
        self.assertEqual("", note)

    def test_rejects_an_empty_value(self) -> None:
        ok, _ = set_telegram.validate_chat_id("")
        self.assertFalse(ok)

    def test_rejects_a_non_numeric_value(self) -> None:
        ok, note = set_telegram.validate_chat_id("my-chat")
        self.assertFalse(ok)
        self.assertIn("number", note)

    def test_rejects_a_username_instead_of_an_id(self) -> None:
        ok, _ = set_telegram.validate_chat_id("@myhandle")
        self.assertFalse(ok)

    def test_warns_on_a_negative_group_id(self) -> None:
        ok, note = set_telegram.validate_chat_id("-1001234567890")
        self.assertTrue(ok)
        self.assertIn("group", note)


class StorageTests(unittest.TestCase):
    """What is written must load back, and must not be printed in full."""

    def _env(self, tmp: str) -> Path:
        path = Path(tmp) / ".env"
        path.write_text("TELEGRAM_BOT_TOKEN=\nTELEGRAM_OWNER_CHAT_ID=\nPAIR=EUR/USD\n", encoding="utf-8")
        return path

    def test_both_values_round_trip(self) -> None:
        with TemporaryDirectory() as tmp:
            path = self._env(tmp)
            lines = path.read_text(encoding="utf-8").splitlines()
            updated = envfile.set_value(lines, set_telegram.TOKEN_KEY, GOOD_TOKEN)
            updated = envfile.set_value(updated, set_telegram.CHAT_KEY, "987654321")
            envfile.write(path, updated)

            final = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(GOOD_TOKEN, envfile.get_value(final, set_telegram.TOKEN_KEY))
            self.assertEqual("987654321", envfile.get_value(final, set_telegram.CHAT_KEY))
            # The unrelated line survives.
            self.assertEqual("EUR/USD", envfile.get_value(final, "PAIR"))

    def test_writing_twice_does_not_stack(self) -> None:
        with TemporaryDirectory() as tmp:
            path = self._env(tmp)
            for token in (GOOD_TOKEN, "222222222:BBB" + "x" * 32):
                lines = path.read_text(encoding="utf-8").splitlines()
                envfile.write(path, envfile.set_value(lines, set_telegram.TOKEN_KEY, token))
            final = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(1, sum(line.startswith("TELEGRAM_BOT_TOKEN=") for line in final))

    def test_the_masked_form_never_contains_the_token(self) -> None:
        shown = mask_key(GOOD_TOKEN)
        self.assertNotIn(GOOD_TOKEN, shown)
        self.assertNotIn(GOOD_TOKEN.split(":", 1)[1], shown)

    def test_clear_removes_both(self) -> None:
        with TemporaryDirectory() as tmp:
            path = self._env(tmp)
            lines = path.read_text(encoding="utf-8").splitlines()
            lines = envfile.set_value(lines, set_telegram.TOKEN_KEY, GOOD_TOKEN)
            lines = envfile.set_value(lines, set_telegram.CHAT_KEY, "42")
            cleared = envfile.remove_key(lines, set_telegram.TOKEN_KEY)
            cleared = envfile.remove_key(cleared, set_telegram.CHAT_KEY)
            self.assertNotIn(GOOD_TOKEN, "\n".join(cleared))
            self.assertEqual("", envfile.get_value(cleared, set_telegram.CHAT_KEY))


if __name__ == "__main__":
    unittest.main()
