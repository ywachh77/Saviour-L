"""Operator Console HTTP & Runtime tests: signals, controls, telegram, health, strategies, and credentials."""

from __future__ import annotations

from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import unittest
import urllib.error
import urllib.request

from scripts.web_dashboard import OperatorRuntime, make_handler


class TestOperatorDashboardAPI(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.env_path = root / ".env"
        self.state_dir = root / "var"
        self.runtime = OperatorRuntime(env_path=self.env_path, state_dir=self.state_dir)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.runtime))
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self._tmp.cleanup()

    def _get(self, path: str) -> tuple[int, dict]:
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}")
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status, json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as err:
            return err.code, json.loads(err.read().decode("utf-8"))

    def _post(self, path: str, body: dict) -> tuple[int, dict]:
        raw = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=raw,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                return resp.status, json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as err:
            return err.code, json.loads(err.read().decode("utf-8"))

    def test_status_exposes_three_markets_and_runtime_gates(self):
        code, data = self._get("/api/status")
        self.assertEqual(code, 200)
        self.assertEqual(data["engine"], "SAVIOUR")
        self.assertEqual(data["active_pairs"], ["EUR/USD", "BTC/USD", "ETH/USD"])
        self.assertEqual(len(data["markets"]), 3)
        self.assertTrue(data["runtime_config"]["mtf_filter"])

    def test_generate_signal_works_standalone_without_telegram(self):
        # Scan mode walks paper bars through unmodified engine gates to return a real sealed signal.
        code, data = self._post(
            "/api/signals",
            {"pair": "BTC/USD", "mode": "scan", "dispatch_telegram": False, "balance_usd": 10000, "risk_pct": 1.0},
        )
        self.assertEqual(code, 200)
        self.assertTrue(data["ok"])
        self.assertFalse(data["dispatch_telegram"])
        self.assertEqual(len(data["evaluations"]), 1)
        card = data["evaluations"][0]
        self.assertEqual(card["pair"], "BTC/USD")
        self.assertEqual(card["feed_source"], "PAPER_SIMULATION")
        self.assertIn("risk_plan", card)
        self.assertIn("stop_price", card["risk_plan"])
        self.assertGreaterEqual(len(card["votes"]), 1)

    def test_pause_and_resume_controls_persist_state(self):
        code, paused = self._post("/api/control", {"action": "pause"})
        self.assertEqual(code, 200)
        self.assertTrue(paused["paused"])

        _, status = self._get("/api/status")
        self.assertTrue(status["paused"])

        code, resumed = self._post("/api/control", {"action": "resume"})
        self.assertEqual(code, 200)
        self.assertFalse(resumed["paused"])

    def test_telegram_web_command_executes_locally_without_chat_dispatch(self):
        code, res = self._post("/api/telegram/command", {"command": "/status", "send_to_chat": False})
        self.assertEqual(code, 200)
        self.assertTrue(res["ok"])
        self.assertFalse(res["sent_to_chat"])
        self.assertIn("FUSION BOT status", res["reply"])
        self.assertIn("pairs EUR/USD, BTC/USD, ETH/USD", res["reply"])

    def test_strategies_and_runtime_config_toggle(self):
        code, strat = self._get("/api/strategies")
        self.assertEqual(code, 200)
        self.assertEqual(len(strat["members"]), 13)

        code, updated = self._post(
            "/api/config",
            {"mtf_minutes": 5, "pairs": ["BTC/USD", "ETH/USD"], "news_filter": False},
        )
        self.assertEqual(code, 200)
        rc = updated["strategies"]["runtime_config"]
        self.assertEqual(rc["mtf_minutes"], 5)
        self.assertEqual(rc["pairs"], ["BTC/USD", "ETH/USD"])
        self.assertFalse(rc["news_filter"])

    def test_credentials_vault_masks_secrets_and_never_echoes_plaintext(self):
        secret_td = "abcdef0123456789abcdef0123456789"
        secret_tg = "123456789:AAH_super_secret_bot_token_value"
        secret_sb = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9_service_role_secret"

        code, saved = self._post(
            "/api/credentials",
            {
                "twelvedata_keys": secret_td,
                "telegram_token": secret_tg,
                "telegram_owner_chat_id": "987654321",
                "supabase_url": "https://idudlcxkihqnvecwqqoj.supabase.co",
                "supabase_service_key": secret_sb,
            },
        )
        self.assertEqual(code, 200)
        self.assertTrue(saved["ok"])
        raw_dump = json.dumps(saved)
        self.assertNotIn(secret_td, raw_dump)
        self.assertNotIn(secret_tg, raw_dump)
        self.assertNotIn(secret_sb, raw_dump)
        self.assertNotIn("987654321", raw_dump)
        self.assertEqual(saved["credentials"]["twelvedata"]["slot_count"], 1)
        self.assertTrue(saved["credentials"]["telegram"]["configured"])
        self.assertTrue(saved["credentials"]["supabase"]["configured"])

    def test_health_and_live_lookahead_audit(self):
        code, data = self._post("/api/audit", {})
        self.assertEqual(code, 200)
        self.assertTrue(data["lookahead"]["ok"])
        self.assertEqual(len(data["lookahead"]["checks"]), 5)


if __name__ == "__main__":
    unittest.main()
