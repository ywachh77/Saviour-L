"""The doctor must be right, because the operator will trust it completely."""

import shutil
import tempfile
import unittest
from pathlib import Path

from fusion import doctor
from fusion.config import Config
from fusion.doctor import FAIL, PASS, SKIP, check_env, check_provider, check_storage, check_supabase, check_telegram
from fusion.store import TABLES


class TestEnvCheck(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_missing_env_fails_with_the_copy_command(self):
        check = check_env(Config(), self.directory / ".env")
        self.assertEqual(check.status, FAIL)
        self.assertIn("cp .env.example", check.action)

    def test_present_env_passes(self):
        path = self.directory / ".env"
        path.write_text("PAIR=EUR/USD\n", encoding="utf-8")
        self.assertEqual(check_env(Config(), path).status, PASS)


class TestProviderCheck(unittest.TestCase):
    def test_no_key_is_a_failure_with_a_where_to_get_one(self):
        check = check_provider(Config())
        self.assertEqual(check.status, FAIL)
        self.assertIn("twelvedata.com", check.action)

    def test_offline_mode_skips_the_paid_call(self):
        config = Config(twelve_data_keys=["abcdefghijklmnop"])
        check = check_provider(config, offline=True)
        self.assertEqual(check.status, SKIP)
        # The key itself must never appear in the output.
        self.assertNotIn("abcdefghijklmnop", check.detail)


class TestSupabaseCheck(unittest.TestCase):
    def test_missing_credentials_fail(self):
        check = check_supabase(Config())
        self.assertEqual(check.status, FAIL)
        self.assertIn("service_role", check.action)

    def test_a_missing_table_points_at_the_migration(self):
        class FakeStore:
            def __init__(self, *args, **kwargs):
                pass

            def probe(self, table):
                if table == "signals":
                    return 'relation "public.signals" does not exist'
                return None

        original = doctor.SupabaseStore
        doctor.SupabaseStore = FakeStore
        try:
            check = check_supabase(Config(supabase_url="https://x.supabase.co", supabase_service_key="k"))
        finally:
            doctor.SupabaseStore = original
        self.assertEqual(check.status, FAIL)
        self.assertIn("signals", check.detail)
        self.assertIn("0001_init.sql", check.action)

    def test_all_tables_present_passes(self):
        class FakeStore:
            def __init__(self, *args, **kwargs):
                self.touched = []

            def probe(self, table):
                self.touched.append(table)
                return None

        holder = {}

        def factory(*args, **kwargs):
            holder["store"] = FakeStore(*args, **kwargs)
            return holder["store"]

        original = doctor.SupabaseStore
        doctor.SupabaseStore = factory
        try:
            check = check_supabase(Config(supabase_url="https://x.supabase.co", supabase_service_key="k"))
        finally:
            doctor.SupabaseStore = original
        self.assertEqual(check.status, PASS)
        self.assertEqual(len(holder["store"].touched), len(TABLES))

    def test_an_unreachable_project_is_not_reported_as_missing_tables(self):
        class FakeStore:
            def __init__(self, *args, **kwargs):
                pass

            def probe(self, table):
                return "GET failed: timed out"

        original = doctor.SupabaseStore
        doctor.SupabaseStore = FakeStore
        try:
            check = check_supabase(Config(supabase_url="https://x.supabase.co", supabase_service_key="k"))
        finally:
            doctor.SupabaseStore = original
        self.assertEqual(check.status, FAIL)
        self.assertIn("could not reach", check.detail)
        self.assertIn("service_role", check.action)


class TestTelegramCheck(unittest.TestCase):
    def test_missing_credentials_fail(self):
        check = check_telegram(Config())
        self.assertEqual(check.status, FAIL)
        self.assertIn("BotFather", check.action)

    def test_a_reachable_bot_passes(self):
        from fusion.telegram import TelegramClient

        original = TelegramClient._call
        TelegramClient._call = lambda self, method, payload: ({"ok": True, "result": {"username": "fusion_bot"}}, "")
        try:
            check = check_telegram(Config(telegram_token="t", telegram_owner_chat_id="1"))
        finally:
            TelegramClient._call = original
        self.assertEqual(check.status, PASS)
        self.assertIn("fusion_bot", check.detail)

    def test_a_failed_getme_names_the_token(self):
        from fusion.telegram import TelegramClient

        original = TelegramClient._call
        TelegramClient._call = lambda self, method, payload: (None, "HTTP 401: unauthorized")
        try:
            check = check_telegram(Config(telegram_token="t", telegram_owner_chat_id="1"))
        finally:
            TelegramClient._call = original
        self.assertEqual(check.status, FAIL)
        self.assertIn("token", check.action)

    def test_a_network_failure_does_not_blame_the_token(self):
        """No route to host is not a revoked token, and must not be reported as one."""
        from fusion.telegram import TelegramClient

        original = TelegramClient._call
        TelegramClient._call = lambda self, method, payload: (None, "network: <urlopen error timed out>")
        try:
            check = check_telegram(Config(telegram_token="t", telegram_owner_chat_id="1"))
        finally:
            TelegramClient._call = original
        self.assertEqual(check.status, FAIL)
        self.assertIn("connection", check.action)
        self.assertNotIn("revoked", check.action)
        self.assertNotIn("wrong", check.action)

    def test_an_unexpected_getme_error_blames_neither(self):
        from fusion.telegram import TelegramClient

        original = TelegramClient._call
        TelegramClient._call = lambda self, method, payload: (None, "bad json: unexpected token")
        try:
            check = check_telegram(Config(telegram_token="t", telegram_owner_chat_id="1"))
        finally:
            TelegramClient._call = original
        self.assertEqual(check.status, FAIL)
        self.assertNotIn("revoked", check.action)

    def test_an_undelivered_test_message_explains_the_first_message_rule(self):
        from fusion import charter
        from fusion.telegram import SendResult, TelegramClient

        original = TelegramClient._call
        TelegramClient._call = lambda self, method, payload: ({"ok": True, "result": {"username": "fusion_bot"}}, "")
        original_send = TelegramClient.send
        TelegramClient.send = lambda self, text, chat_id=None: SendResult(
            False, charter.DELIVERY_FAILED, error="chat not found"
        )
        try:
            check = check_telegram(Config(telegram_token="t", telegram_owner_chat_id="1"), send_test=True)
        finally:
            TelegramClient._call = original
            TelegramClient.send = original_send
        self.assertEqual(check.status, FAIL)
        self.assertIn("/start", check.action)


class TestStorageAndSummary(unittest.TestCase):
    def test_storage_reports_its_real_path(self):
        directory = Path(tempfile.mkdtemp())
        try:
            check = check_storage(Config(local_state_dir=str(directory)))
            self.assertEqual(check.status, PASS)
            self.assertIn("writable", check.detail)
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    def test_summary_counts(self):
        from fusion.doctor import Check, summarise

        checks = [
            Check("a", PASS, ""),
            Check("b", FAIL, ""),
            Check("c", SKIP, ""),
        ]
        self.assertEqual(summarise(checks), (1, 1))


if __name__ == "__main__":
    unittest.main()
