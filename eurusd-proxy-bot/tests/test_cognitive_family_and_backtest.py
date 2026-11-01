"""Verification for 6-family fusion, every-trade learning, auto-backtesting,
SAVIOUR × Needle 3 human cognitive partner, shadow-recovery schema safety,
and Bleach manga/anime UI assets.
"""

from __future__ import annotations

import json
import re
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fusion import charter, cognitive, strategies
from fusion.config import Config
from fusion.domain import Candle, ModelRead, Signal
from fusion.family_fusion import FamilyFusionEngine
from fusion.model import BootstrapEnsemble, OnlineLogisticModel
from fusion.runner import FusionBot
from fusion.store import LocalStore, Store
from scripts.web_dashboard import OperatorRuntime

BOT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BOT_ROOT.parent
DOCS_DIR = REPO_ROOT / "docs"


class _RealHistoricalProvider:
    """Deterministic non-synthetic provider for testing real-bar learning & auto-backtest."""

    name = "real_test_feed"
    synthetic = False

    def __init__(self, candles: list[Candle]) -> None:
        self._candles = list(candles)

    def latest_candles(self, pair: str, count: int) -> list[Candle]:
        return list(self._candles[-count:])

    def current_spread_pips(self, pair: str) -> tuple[float, str]:
        return 1.2, "LIVE"


def _make_m1_series(count: int = 340, start_price: float = 1.0840) -> list[Candle]:
    base = datetime(2026, 10, 22, 9, 0, tzinfo=timezone.utc)
    bars: list[Candle] = []
    price = start_price
    for idx in range(count):
        wave = 0.00024 if (idx % 18) < 11 else -0.00017
        o = price
        c = round(o + wave, 5)
        h = round(max(o, c) + 0.00018, 5)
        l = round(min(o, c) - 0.00015, 5)
        bars.append(
            Candle(
                timestamp=base + timedelta(minutes=idx),
                open=o,
                high=h,
                low=l,
                close=c,
                volume=120.0 + (idx % 17),
            )
        )
        price = c
    return bars


class TestMergedStrategicFamilies(unittest.TestCase):
    def test_six_families_and_synergy_learning_and_veto(self) -> None:
        ff = FamilyFusionEngine()
        self.assertEqual(len(ff.family_weights), 6)
        self.assertGreaterEqual(len(ff.synergy_weights), 10)

        votes_aligned = {
            "orbix_confluence": "BUY",
            "smc_fvg_displacement": "BUY",
            "smc_order_block": "BUY",
            "ttm_keltner_squeeze": "BUY",
            "supertrend_flip": "BUY",
            "session_box_breakout": "BUY",
            "trend_pullback": "BUY",
        }
        # Cold-start Charter rule: family weights start at 0.0 until real outcomes are observed
        cold_verdict = ff.evaluate_votes("BUY", votes_aligned, regime="TREND", session="LONDON")
        self.assertFalse(cold_verdict.vetoed)
        self.assertEqual(cold_verdict.net_score, 0.0)

        # Observe winning outcomes -> supporting families & synergies gain weight
        for _ in range(10):
            ff.observe_outcome(
                votes_aligned,
                "BUY",
                regime="TREND",
                session="LONDON",
                signal_side="BUY",
                was_loss=False,
            )
        self.assertGreater(ff.family_weights["SMC_ICT"], 0.0)
        self.assertGreater(ff.family_weights["VIRAL_MOMENTUM"], 0.0)
        self.assertGreater(ff.family_weights["CLASSIC_COUNCIL"], 0.0)

        verdict = ff.evaluate_votes("BUY", votes_aligned, regime="TREND", session="LONDON")
        self.assertFalse(verdict.vetoed)
        self.assertGreater(verdict.net_score, 0.10)
        self.assertTrue(len(verdict.synergy_pairs) >= 2)

        # Refine model read shifts probability gently upward when families align with BUY
        read = ModelRead(
            raw=0.52,
            probability=0.52,
            calibrated=False,
            calibration_state="RAW",
        )
        refined = ff.refine_model_read(read, verdict)
        self.assertIsNotNone(refined)
        assert refined is not None
        self.assertGreater(refined.probability, 0.52)

        # Multi-family opposition veto when >= 2 opposing families carry >= 0.30 mass
        votes_opposed = {
            "orbix_confluence": "BUY",
            "smc_liquidity_sweep": "SELL",
            "smc_fvg_displacement": "SELL",
            "ttm_keltner_squeeze": "SELL",
            "supertrend_flip": "SELL",
            "trend_pullback": "SELL",
        }
        opp_verdict = ff.evaluate_votes("BUY", votes_opposed, regime="TREND", session="LONDON")
        self.assertTrue(opp_verdict.vetoed)
        self.assertIn("family veto", opp_verdict.reason.lower())

        # Round-trip serialization
        restored = FamilyFusionEngine.from_dict(ff.to_dict())
        self.assertEqual(restored.trades_observed, ff.trades_observed)
        self.assertAlmostEqual(restored.family_weights["SMC_ICT"], ff.family_weights["SMC_ICT"], places=4)


class TestEveryTradeLearningAndShadowSchema(unittest.TestCase):
    def test_every_settled_trade_updates_ensemble_families_journal_and_backtest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            bars = _make_m1_series(340)
            cfg = Config(
                pair="EUR/USD",
                pairs=("EUR/USD",),
                local_state_dir=str(state_dir),
                horizon_minutes=15,
                loss_guard_streak=2,
                loss_guard_shadow_recovery=True,
            )
            store = Store(local=LocalStore(state_dir), supabase=None)
            prov = _RealHistoricalProvider(bars)
            bot = FusionBot(config=cfg, store=store, provider=prov)
            bot.candles = list(bars)
            bot._ensure_baseline_veto_model()

            # Build a realistic feature vector via OnlineLogisticModel feature order
            feats = {name: 0.15 for name in (bot.model.feature_names if bot.model else ())}
            entry_bar = bars[200]
            sig = Signal.create(
                pair="EUR/USD",
                side="BUY",
                generated_at=entry_bar.timestamp - timedelta(seconds=2),
                entry_time=entry_bar.timestamp,
                horizon_minutes=15,
                probability_up=0.58,
                probability_raw=0.57,
                calibration_state="RAW",
                regime="TREND",
                strategy="orbix_confluence",
                council=(
                    {"strategy": "orbix_confluence", "direction": "BUY"},
                    {"strategy": "smc_liquidity_sweep", "direction": "BUY"},
                    {"strategy": "ttm_keltner_squeeze", "direction": "BUY"},
                ),
                features=feats,
                reason="Core 5/6 + SMC + Viral Momentum",
                source="LIVE",
                spread_pips=1.2,
            )
            store.save_signal(dict(sig.to_dict(), seal_state=charter.SEAL_SIGNAL, settled=False))

            settled_count = bot.settle_due(at=bars[220].timestamp)
            self.assertEqual(settled_count, 1)
            self.assertGreaterEqual(bot.family_fusion.trades_observed, 1)
            self.assertEqual(len(bot.learning_journal), 1)
            self.assertGreaterEqual(len(bot.auto_backtest_history), 1)
            self.assertTrue(bot.auto_backtest_history[-1]["ok"])

            # Verify shadow_recovery log_event rows strictly obey learning_events schema columns
            allowed_cols = {
                "id",
                "kind",
                "signal_id",
                "applied",
                "reason",
                "predicted",
                "raw",
                "label",
                "squared_error",
                "created_at",
                "outcome",
            }
            store.log_event(
                {
                    "kind": "shadow_recovery_queued",
                    "pair": "EUR/USD",
                    "side": "BUY",
                    "net_pips": 2.4,
                    "reason": "queued",
                }
            )
            for ev in store.local.read("learning_events"):
                extra = set(ev.keys()) - allowed_cols
                self.assertEqual(extra, set(), f"Unexpected columns in learning_events: {extra}")


class TestSaviourCognitivePartnerAndWebEndpoints(unittest.TestCase):
    def test_human_dialogue_self_heal_and_backtest_endpoints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            env_file = tmp_path / ".env"
            env_file.write_text("PAIR=EUR/USD\nNEEDLE_ENABLED=1\n", encoding="utf-8")
            rt = OperatorRuntime(env_path=env_file, state_dir=tmp_path / "var")
            try:
                # 1. Human conversational query -> natural human strategist response
                human_res = rt.run_needle_query(
                    "Hey SAVIOUR, how are our trades looking and what did you learn recently?"
                )
                self.assertTrue(human_res["ok"])
                self.assertTrue(human_res["executed"])
                self.assertIn("EUR/USD", human_res["reply"])
                self.assertIn("cognitive", human_res)

                # 2. Autonomous self-heal & error rectification
                heal_res = rt.self_heal_action()
                self.assertTrue(heal_res["ok"])
                self.assertTrue(heal_res["heal"]["causality_ok"])

                # 3. Walk-forward auto-backtest action across all markets (EUR/USD only)
                bt_res = rt.run_backtest_action("ALL")
                self.assertTrue(bt_res["ok"])
                self.assertEqual(len(bt_res["runs"]), 1)
                self.assertTrue(all(r["ok"] for r in bt_res["runs"]))

                # 4. Learning payload includes 6-family weights, synergies, backtests & cognitive state
                learn = rt.learning_payload()
                self.assertTrue(learn["ok"])
                self.assertIn("EUR/USD", learn["markets"])
                self.assertEqual(
                    len(learn["markets"]["EUR/USD"]["family_fusion"]["family_weights"]),
                    6,
                )

                # 5. Needle 9-tool routing for run_backtest & self_improve
                bt_route = rt.run_needle_query("run walk-forward backtest on eur/usd")
                self.assertTrue(bt_route["executed"])
                self.assertEqual(bt_route["decision"]["function_calls"][0]["name"], "run_backtest")

                imp_route = rt.run_needle_query("find any errors or flaws and fix them automatically")
                self.assertTrue(imp_route["executed"])
                self.assertEqual(imp_route["decision"]["function_calls"][0]["name"], "self_improve")
            finally:
                rt.close()


class TestBleachMangaAssetsAndPaletteGuard(unittest.TestCase):
    FORBIDDEN_AI_COLOURS = re.compile(
        r"\b(purple|violet|indigo|magenta|cyan|aqua|neon|fuchsia|lavender|teal)\b",
        re.IGNORECASE,
    )

    def test_curated_bleach_plates_exist_and_docs_stay_ai_colour_free(self) -> None:
        manga_dir = DOCS_DIR / "assets" / "manga"
        expected = (
            "zangetsu-cleave.jpg",
            "ichigo-yhwach-tybw.jpg",
            "yhwach-sovereign.jpg",
            "vasto-ulquiorra.jpg",
            "gotei-captains.webp",
            "bankai-clash.jpg",
            "shunsui-bankai.jpg",
        )
        for fname in expected:
            fpath = manga_dir / fname
            self.assertTrue(fpath.is_file(), f"Missing Bleach manga asset: {fpath}")
            self.assertGreater(fpath.stat().st_size, 10_000, f"Asset suspiciously small: {fpath}")

        index_html = (DOCS_DIR / "index.html").read_text(encoding="utf-8")
        console_html = (DOCS_DIR / "console.html").read_text(encoding="utf-8")
        self.assertIn("22 COUNCIL GATES", index_html)
        self.assertNotIn("13 COUNCIL GATES", index_html)
        self.assertIn("reiatsu-canvas", index_html)
        self.assertIn("reiatsu-canvas", console_html)
        self.assertIn("family-fusion-panel", console_html)
        self.assertIn("auto-backtest-panel", console_html)

        for rel in ("index.html", "console.html", "assets/style.css", "assets/main.js"):
            txt = (DOCS_DIR / rel).read_text(encoding="utf-8")
            m = self.FORBIDDEN_AI_COLOURS.search(txt)
            self.assertIsNone(m, f"Forbidden AI colour word {m.group(0) if m else ''} found in docs/{rel}")


if __name__ == "__main__":
    unittest.main()
