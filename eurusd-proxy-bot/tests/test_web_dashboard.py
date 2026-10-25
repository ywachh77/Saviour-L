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

    def test_cloud_synced_credentials_and_persisted_candles(self):
        from datetime import datetime, timedelta, timezone
        from fusion.store import LocalStore
        from fusion.timeutil import format_datetime

        local = LocalStore(str(self.state_dir))
        local.replace(
            "api_keys",
            [{"id": "keypool", "slots": [{"index": 0, "label": "f6df...fb(32)", "enabled": True, "day_used": 2, "day_limit": 800}]}],
        )
        local.replace(
            "model_state",
            [{"id": "champion", "runtime": {"paused": False, "telegram_offset": 309256988}}],
        )
        t0 = datetime(2026, 10, 25, 6, 0, tzinfo=timezone.utc)
        bars = [
            {
                "timestamp": format_datetime(t0 + timedelta(minutes=i)),
                "pair": "BTC/USD",
                "open": 80200.0 + i,
                "high": 80215.0 + i,
                "low": 80190.0 + i,
                "close": 80210.0 + i,
                "volume": 0.0,
            }
            for i in range(90)
        ]
        local.replace("candles_m1", bars)

        self.runtime.reload()
        _, status = self._get("/api/status")
        self.assertEqual(status["feed_mode"], "SUPABASE_SYNC")
        btc_mkt = [m for m in status["markets"] if m["symbol"] == "BTC/USD"][0]
        self.assertAlmostEqual(btc_mkt["latest_price"], 80299.0, places=2)

        _, creds = self._get("/api/credentials")
        self.assertTrue(creds["twelvedata"]["configured"])
        self.assertEqual(creds["twelvedata"]["source"], "CLOUD_SYNCED")
        self.assertEqual(creds["twelvedata"]["slots_masked"][0]["masked"], "f6df...fb(32)")
        self.assertTrue(creds["telegram"]["configured"])
        self.assertEqual(creds["telegram"]["source"], "CLOUD_SYNCED")

    def test_open_gate_emoji_and_six_distinct_page_animations(self):
        repo_root = Path(__file__).resolve().parents[2]
        index_html = (repo_root / "docs" / "index.html").read_text(encoding="utf-8")
        console_html = (repo_root / "docs" / "console.html").read_text(encoding="utf-8")
        style_css = (repo_root / "docs" / "assets" / "style.css").read_text(encoding="utf-8")
        main_js = (repo_root / "docs" / "assets" / "main.js").read_text(encoding="utf-8")

        # Open Gate emoji present on Home Gate and Console Gate controls
        self.assertIn("&#x26E9;&#xFE0F;", index_html)
        self.assertIn("OPEN GATE", index_html)
        self.assertIn("&#x26E9;&#xFE0F;", console_html)
        self.assertIn("Open Gate", console_html)

        # Six distinct page transition FX & keyframes present
        for fx_cls in ("fx-signals", "fx-telegram", "fx-health", "fx-strategies", "fx-credentials", "fx-risk"):
            self.assertIn(fx_cls, style_css)
            self.assertIn(fx_cls, main_js)

        for kf in (
            "pageSignalsCleave",
            "pageTelegramUnfurl",
            "pageHealthDiag",
            "flankFromLeft",
            "pageCredentialsIris",
            "pageRiskMonolith",
        ):
            self.assertIn(kf, style_css)

    def test_telegram_format_studio_and_protection_config_endpoints(self):
        code, initial = self._get("/api/telegram/format")
        self.assertEqual(code, 200)
        self.assertEqual(initial["format"]["signal_style"], "standard")
        self.assertIn("scheduled_signal", initial["preview"])
        self.assertGreaterEqual(len(initial["signal_placeholders"]), 10)

        code, updated = self._post(
            "/api/telegram/format",
            {
                "signal_style": "custom",
                "signal_template": "DESK {pair} {side} | {horizon}m | #{short_id}",
                "result_style": "compact",
                "apply_to_signals_cmd": True,
            },
        )
        self.assertEqual(code, 200)
        self.assertTrue(updated["saved"])
        self.assertEqual(updated["format"]["signal_style"], "custom")
        self.assertIn("DESK EUR/USD BUY | 30m", updated["preview"]["scheduled_signal"])
        self.assertIn("RESULT EUR/USD BUY", updated["preview"]["settlement_result"])

        code, bad = self._post("/api/telegram/format", {"signal_style": "nonexistent_style"})
        self.assertEqual(code, 400)
        self.assertFalse(bad["ok"])

        code, reset_res = self._post("/api/telegram/format", {"reset": True})
        self.assertEqual(code, 200)
        self.assertEqual(reset_res["format"]["signal_style"], "standard")

        code, cfg_res = self._post(
            "/api/config",
            {
                "pairs": ["EUR/USD", "BTC/USD", "ETH/USD"],
                "mtf_min_gap_bps": 2.0,
                "mtf_macro_filter": True,
                "signal_cooldown_minutes": 30,
                "loss_guard_streak": 3,
                "loss_guard_shadow_recovery": True,
                "fx_rollover_guard": True,
            },
        )
        self.assertEqual(code, 200)
        rc = cfg_res["strategies"]["runtime_config"]
        self.assertAlmostEqual(rc["mtf_min_gap_bps"], 2.0, places=3)
        self.assertTrue(rc["mtf_macro_filter"])
        self.assertEqual(rc["signal_cooldown_minutes"], 30)
        self.assertEqual(rc["loss_guard_streak"], 3)
        self.assertTrue(rc["loss_guard_shadow_recovery"])
        self.assertTrue(rc["fx_rollover_guard"])


if __name__ == "__main__":
    unittest.main()
