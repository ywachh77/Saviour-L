"""Tests for Cactus Compute Needle 3 integration, Charter shield, Telegram/Web routing, and LoRA QAT."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fusion import charter, news
from fusion.config import Config
from fusion.needle_agent import (
    SAVIOUR_NEEDLE_TOOLS,
    SaviourNeedleAgent,
    charter_refusal_reason,
    execute_on_bot,
    format_blackout_window_entry,
    needle_package_info,
    validate_tool_calls,
)
from fusion.runner import FusionBot
from fusion.store import LocalStore, Store
from fusion.telegram import COMMANDS, TelegramClient
from scripts.web_dashboard import OperatorRuntime

BOT_ROOT = Path(__file__).resolve().parents[1]


class _FakeTelegramTransport:
    def __init__(self, updates_json: bytes):
        self.updates_json = updates_json
        self.sent_messages: list[str] = []

    def __call__(self, request, timeout=None):  # noqa: ARG002
        url = request.full_url

        class _Resp:
            status = 200

            def __init__(self, payload: bytes):
                self._payload = payload

            def read(self) -> bytes:
                return self._payload

            def __enter__(self):
                return self

            def __exit__(self, *_a):
                return False

        if "getUpdates" in url:
            payload = self.updates_json
            self.updates_json = b'{"ok":true,"result":[]}'
            return _Resp(payload)
        body = (request.data or b"").decode("utf-8", "replace")
        self.sent_messages.append(body)
        return _Resp(b'{"ok":true,"result":{"message_id":404}}')


class TestNeedleTelemetryAndSchema(unittest.TestCase):
    def test_telemetry_disabled_at_import(self):
        self.assertEqual(os.environ.get("NEEDLE_TELEMETRY"), "0")
        self.assertEqual(os.environ.get("DO_NOT_TRACK"), "1")
        pkg = needle_package_info()
        self.assertTrue(pkg["telemetry_disabled"])

    def test_tool_schemas_are_compact_and_well_formed(self):
        names = [t["name"] for t in SAVIOUR_NEEDLE_TOOLS]
        self.assertEqual(
            names,
            [
                "scan_market",
                "bot_status",
                "pause_signals",
                "resume_signals",
                "set_telegram_style",
                "configure_guards",
                "extract_news_shock",
                "run_backtest",
                "self_improve",
            ],
        )
        compact = json.dumps(SAVIOUR_NEEDLE_TOOLS, separators=(",", ":"))
        # Must stay compact so prompt + target fits comfortably inside 1024 tokens during LoRA QAT.
        self.assertLess(len(compact), 2200)

    def test_validate_tool_calls_rejects_bad_enums_and_clamps_bounds(self):
        self.assertEqual(validate_tool_calls([{"name": "unknown_tool", "arguments": {}}]), [])
        self.assertEqual(
            validate_tool_calls([{"name": "scan_market", "arguments": {"pair": "EURUSD_otc", "mode": "now"}}]),
            [],
        )
        valid = validate_tool_calls(
            [
                {
                    "name": "configure_guards",
                    "arguments": {"mtf_minutes": 15, "cooldown_minutes": 999, "horizon_minutes": 1},
                }
            ]
        )
        self.assertEqual(
            valid,
            [
                {
                    "name": "configure_guards",
                    "arguments": {"mtf_minutes": 15, "cooldown_minutes": 120, "horizon_minutes": 5},
                }
            ],
        )


class TestCharterRefusalShield(unittest.TestCase):
    def test_charter_prohibited_prompts_always_return_empty_calls(self):
        agent = SaviourNeedleAgent(state_dir=tempfile.mkdtemp())
        forbidden = [
            "auto trade EURUSD_otc on Quotex right now",
            "connect Pocket Option and place $50 call",
            "enable martingale and double the stake after loss",
            "place a real $250 buy order on my live Binance account",
            "give me a guaranteed 95% win signal right now",
        ]
        for prompt in forbidden:
            self.assertIsNotNone(charter_refusal_reason(prompt), prompt)
            dec = agent.route(prompt)
            self.assertEqual(dec.type, "refusal", prompt)
            self.assertEqual(dec.function_calls, [], prompt)
            self.assertEqual(dec.engine, "charter_shield", prompt)

    def test_off_topic_queries_refuse_cleanly(self):
        agent = SaviourNeedleAgent(state_dir=tempfile.mkdtemp())
        for prompt in (
            "what is the weather in Tokyo tomorrow?",
            "write a pasta carbonara recipe",
            "play music on spotify",
        ):
            dec = agent.route(prompt)
            self.assertEqual(dec.type, "refusal", prompt)
            self.assertEqual(dec.function_calls, [], prompt)


class TestCuratedDatasets(unittest.TestCase):
    def test_committed_train_and_eval_splits_are_valid_and_exact(self):
        train_path = BOT_ROOT / "data" / "needle_saviour_train.jsonl"
        eval_path = BOT_ROOT / "data" / "needle_saviour_eval.jsonl"
        self.assertTrue(train_path.is_file())
        self.assertTrue(eval_path.is_file())

        train_rows = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        eval_rows = [json.loads(line) for line in eval_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertGreaterEqual(len(train_rows), 150)
        self.assertGreaterEqual(len(eval_rows), 35)

        with tempfile.TemporaryDirectory() as tmp:
            agent = SaviourNeedleAgent(state_dir=tmp)
            for row in eval_rows:
                self.assertIn("query", row)
                self.assertIn("answers", row)
                dec = agent.route(row["query"])
                self.assertEqual(dec.function_calls, row["answers"], row["query"])


class TestNativeCactConfidenceSemantics(unittest.TestCase):
    def test_lora_cact_none_confidence_executes_while_low_numeric_confidence_abstains(self):
        with tempfile.TemporaryDirectory() as tmp:
            cact = Path(tmp) / "saviour_needle3.cact"
            cact.write_bytes(b"fake")

            class _FakeNativeNeedle:
                def __init__(self, **_kw):
                    self.next_resp = {}

                def reset(self):
                    pass

                def complete(self, _q):
                    return self.next_resp

            fake_instance = _FakeNativeNeedle()

            with patch("fusion.needle_agent.needle_package_info") as mock_pkg:
                mock_pkg.return_value = {
                    "installed": True,
                    "version": "3.2.8",
                    "train_installed": False,
                    "native_engine_cached": True,
                    "base_weights_cached": True,
                    "telemetry_disabled": True,
                }
                agent = SaviourNeedleAgent(state_dir=tmp, weights_path=cact, min_confidence=0.60)
                agent._native_model = fake_instance
                agent._native_loaded_weights = str(cact)

                # 1. Local LoRA .cact reports confidence=None -> must succeed (not TypeError!)
                fake_instance.next_resp = {
                    "type": "tool_call",
                    "confidence": None,
                    "function_calls": [{"name": "scan_market", "arguments": {"pair": "BTC/USD", "mode": "now"}}],
                }
                dec_lora = agent.route("scan btc/usd now")
                self.assertEqual(dec_lora.type, "tool_call")
                self.assertIsNone(dec_lora.confidence)
                self.assertEqual(dec_lora.engine, "needle3_cact")
                self.assertEqual(dec_lora.function_calls[0]["arguments"]["pair"], "BTC/USD")

                # 2. Base .cact with low numeric confidence (< 0.60) -> must refuse
                fake_instance.next_resp = {
                    "type": "tool_call",
                    "confidence": 0.31,
                    "function_calls": [{"name": "pause_signals", "arguments": {}}],
                }
                dec_low = agent.route("ambiguous murmur")
                self.assertEqual(dec_low.type, "refusal")
                self.assertEqual(dec_low.function_calls, [])
                self.assertAlmostEqual(dec_low.confidence or 0.0, 0.31, places=2)


class TestBotTelegramAndNewsIntegration(unittest.TestCase):
    def test_six_slash_commands_stay_locked_and_default_config_ignores_plain_text(self):
        self.assertEqual(COMMANDS, ("/start", "/help", "/signals", "/status", "/pause", "/resume"))
        updates = b'{"ok":true,"result":[{"update_id":1,"message":{"text":"pause signals","chat":{"id":777}}}]}'
        transport = _FakeTelegramTransport(updates)
        client = TelegramClient("tok", "777", opener=transport)
        self.assertEqual(client.poll_commands(), [])

    def test_needle_enabled_routes_owner_plain_text_and_ignores_strangers(self):
        updates = json.dumps(
            {
                "ok": True,
                "result": [
                    {"update_id": 10, "message": {"text": "pause signals", "chat": {"id": 999999}}},
                    {
                        "update_id": 11,
                        "message": {
                            "text": "pause signals and switch telegram format to tactical",
                            "chat": {"id": 777},
                        },
                    },
                ],
            }
        ).encode("utf-8")
        transport = _FakeTelegramTransport(updates)
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(local=LocalStore(Path(tmp)), supabase=None)
            cfg = Config(
                local_state_dir=tmp,
                pairs=("EUR/USD", "BTC/USD"),
                needle_enabled=True,
                telegram_token="tok",
                telegram_owner_chat_id="777",
            )
            tg = TelegramClient("tok", "777", opener=transport)
            bot = FusionBot(config=cfg, store=store, telegram=tg)
            bot.startup()

            bot.poll_commands()
            self.assertTrue(bot.paused)
            self.assertTrue(bot.peer_bots["BTC/USD"].paused)
            self.assertEqual(bot.telegram_format.signal_style, "tactical")
            self.assertEqual(len(transport.sent_messages), 1)

    def test_headline_shock_populates_live_news_blackout_window(self):
        stamp = datetime(2026, 10, 26, 12, 30, 0, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(local=LocalStore(Path(tmp)), supabase=None)
            bot = FusionBot(config=Config(local_state_dir=tmp, needle_enabled=True), store=store)
            dec, reply = execute_on_bot(
                bot,
                "Breaking headline: US CPI inflation comes in hot at 3.8%, USD spikes across majors",
                at=stamp,
            )
            self.assertEqual(dec.type, "tool_call")
            self.assertIn("HIGH news shock", reply)
            check = news.check_blackout(
                datetime(2026, 10, 26, 12, 40, 0, tzinfo=timezone.utc),
                pair="EUR/USD",
                extra_windows=bot.config.news_blackout_windows,
            )
            self.assertTrue(check.blocked)
            self.assertEqual(check.event, "US CPI")


class TestWebDashboardNeedleEndpoints(unittest.TestCase):
    def test_web_runtime_needle_status_and_query_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_p = Path(tmp) / ".env"
            env_p.write_text("", encoding="utf-8")
            rt = OperatorRuntime(env_path=env_p, state_dir=Path(tmp) / "state")

            st = rt.needle_status_payload()
            self.assertTrue(st["ok"])
            self.assertTrue(st["enabled"])
            self.assertEqual(st["tool_count"], 9)

            res = rt.run_needle_query("configure 15m mtf filter and 25 minute cooldown")
            self.assertTrue(res["ok"])
            self.assertTrue(res["executed"])
            self.assertEqual(rt.bot.config.mtf_minutes, 15)
            self.assertEqual(rt.bot.config.signal_cooldown_minutes, 25)

            refused = rt.run_needle_query("auto trade EURUSD_otc on Quotex")
            self.assertFalse(refused["executed"])
            self.assertEqual(refused["decision"]["type"], "refusal")


@unittest.skipUnless(needle_package_info().get("train_installed"), "cactus-needle[train] not installed")
class TestJaxLoraFinetunePipeline(unittest.TestCase):
    def test_bootstrap_lora_finetune_writes_adapter_and_loadable_cact(self):
        from scripts.needle_finetune import run_finetune

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            mini_train = tmp_path / "mini_train.jsonl"
            mini_eval = tmp_path / "mini_eval.jsonl"
            src_rows = (BOT_ROOT / "data" / "needle_saviour_train.jsonl").read_text(encoding="utf-8").splitlines()[:16]
            mini_train.write_text("\n".join(src_rows) + "\n", encoding="utf-8")
            mini_eval.write_text("\n".join(src_rows[:4]) + "\n", encoding="utf-8")

            ns = argparse.Namespace(
                train_data=str(mini_train),
                eval_data=str(mini_eval),
                checkpoint=None,
                state_dir=str(tmp_path / "var"),
                out=None,
                epochs=1,
                batch_size=16,
                lr=1e-3,
                lora_rank=4,
                lora_alpha=8.0,
                layers=2,
                max_len=1024,
                val_split=0.0,
                seed=7,
                bootstrap_local=True,
            )
            report = run_finetune(ns)
            self.assertEqual(report["lora_rank"], 4)
            self.assertGreater(report["loss_start"] or 0.0, 0.0)
            self.assertGreater(report["cact_num_tensors"], 0)
            self.assertTrue(Path(report["adapter_path"]).is_file())
            self.assertTrue(Path(report["cact_path"]).is_file())


if __name__ == "__main__":
    unittest.main()
