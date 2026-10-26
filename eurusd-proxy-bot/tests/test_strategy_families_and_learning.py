"""Tests for multi-family strategies, automatic settlement, every-signal learning,
and whole-system bug fixes (zero-bias paper series, cloud feature preservation,
council backward compatibility, and newest-first ledger ordering).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import re
import tempfile
import unittest

from fusion import charter, council, features as feat_mod, methods, regime, session, strategies
from fusion.config import Config
from fusion.council import CouncilWeights, run_council
from fusion.domain import Candle, Decision, Signal
from fusion.mistakes import MistakeMemory
from fusion.model import MODE_VETO
from fusion.runner import SAVE_STATE_EVERY, FusionBot
from fusion.store import LocalStore, Store
from scripts.web_dashboard import DOCS_DIR, OperatorRuntime, _build_scaled_paper_series

from .helpers import make_candles


def _base_feature_row() -> tuple[Candle, float, dict[str, float]]:
    candles = make_candles(120, seed=23, volatility=0.00035, drift=0.0)
    feats = feat_mod.compute_features(candles)
    row = dict(feats[-1] or {})
    row["atr_pips"] = 2.8
    row["atr"] = 0.00028
    row["atr_pct"] = 0.00026
    return candles[-1], candles[-2].close, row


class TestMultiFamilyStrategies(unittest.TestCase):
    def test_all_nine_new_methods_registered_in_council_regime_and_session(self) -> None:
        self.assertEqual(len(methods.METHOD_PROTOTYPES), 9)
        self.assertEqual(len(strategies.MEMBERS), 22)
        self.assertEqual(len(strategies.PROTOTYPES), 21)

        for name in methods.METHOD_PROTOTYPES:
            self.assertIn(name, strategies.PROTOTYPES)
            self.assertIn(name, strategies.MEMBERS)
            self.assertIn(name, strategies.REGISTRY)
            self.assertIn(
                strategies.strategy_family(name),
                {"SMC_ICT", "VIRAL_MOMENTUM", "QUANT_SCALP"},
            )
            allowed_regimes = [
                r for r in ("TREND", "RANGE", "VOLATILE") if regime.regime_allows(r, name)
            ]
            self.assertTrue(allowed_regimes, f"{name} must be allowed in at least one regime")
            allowed_sessions = [
                s for s in session.ALL_SESSIONS if session.session_allows(s, name)
            ]
            self.assertTrue(allowed_sessions, f"{name} must be allowed in at least one FX session")
            self.assertTrue(session.session_allows(session.ROUND_THE_CLOCK, name))

    def test_all_nine_methods_refuse_none_features_and_sub_spread_atr(self) -> None:
        bar, prev_close, f = _base_feature_row()
        f_quiet = dict(f, atr_pips=0.5)
        for name in methods.METHOD_PROTOTYPES:
            fn = strategies.REGISTRY[name]
            d_none = fn(bar, prev_close, None, 0.55)
            self.assertEqual(d_none.side, "NO SIGNAL")
            d_quiet = fn(bar, prev_close, f_quiet, 0.55, spread_pips=1.2)
            self.assertEqual(d_quiet.side, "NO SIGNAL")
            self.assertIn("spread-coverage", d_quiet.reason)

    def test_smc_family_buy_and_sell_symmetry(self) -> None:
        ts = datetime(2026, 10, 26, 12, 0, tzinfo=timezone.utc)
        _, prev, base_f = _base_feature_row()

        # 1. smc_liquidity_sweep
        bull_bar = Candle(ts, open=1.0848, high=1.0854, low=1.0835, close=1.0852, volume=100.0)
        f_bull_sweep = dict(base_f, prior_low_15=1.0842, prior_high_15=1.0865, close_position=0.89, rsi=46.0)
        self.assertEqual(methods.smc_liquidity_sweep(bull_bar, prev, f_bull_sweep).side, "BUY")

        bear_bar = Candle(ts, open=1.0852, high=1.0865, low=1.0846, close=1.0848, volume=100.0)
        f_bear_sweep = dict(base_f, prior_low_15=1.0835, prior_high_15=1.0858, close_position=0.11, rsi=54.0)
        self.assertEqual(methods.smc_liquidity_sweep(bear_bar, prev, f_bear_sweep).side, "SELL")

        # 2. smc_fvg_displacement
        f_bull_fvg = dict(base_f, fvg_bull_active=1.0, fvg_bear_active=0.0, ema_fast=1.0855, ema_slow=1.0850, rsi=55.0)
        self.assertEqual(methods.smc_fvg_displacement(bull_bar, prev, f_bull_fvg).side, "BUY")

        f_bear_fvg = dict(base_f, fvg_bull_active=0.0, fvg_bear_active=1.0, ema_fast=1.0845, ema_slow=1.0850, rsi=45.0)
        self.assertEqual(methods.smc_fvg_displacement(bear_bar, prev, f_bear_fvg).side, "SELL")

        # 3. smc_order_block
        f_bull_ob = dict(base_f, ob_bull_tap=1.0, ob_bear_tap=0.0, close_position=0.72, rsi=48.0)
        self.assertEqual(methods.smc_order_block(bull_bar, prev, f_bull_ob).side, "BUY")

        f_bear_ob = dict(base_f, ob_bull_tap=0.0, ob_bear_tap=1.0, close_position=0.28, rsi=52.0)
        self.assertEqual(methods.smc_order_block(bear_bar, prev, f_bear_ob).side, "SELL")

    def test_viral_and_quant_families_buy_and_sell_symmetry(self) -> None:
        ts = datetime(2026, 10, 26, 13, 0, tzinfo=timezone.utc)
        _, _, base_f = _base_feature_row()
        bull_bar = Candle(ts, open=1.0850, high=1.0858, low=1.0847, close=1.0857, volume=120.0)
        bear_bar = Candle(ts, open=1.0850, high=1.0853, low=1.0842, close=1.0843, volume=120.0)

        # 4. ttm_keltner_squeeze
        f_sq_buy = dict(base_f, ttm_squeeze_recent=1.0, ttm_squeeze_now=0.0, linreg_mom_12=0.45, linreg_mom_previous=0.10, bb_mid=1.0850)
        self.assertEqual(methods.ttm_keltner_squeeze(bull_bar, 1.0849, f_sq_buy).side, "BUY")
        f_sq_sell = dict(base_f, ttm_squeeze_recent=1.0, ttm_squeeze_now=0.0, linreg_mom_12=-0.45, linreg_mom_previous=-0.10, bb_mid=1.0850)
        self.assertEqual(methods.ttm_keltner_squeeze(bear_bar, 1.0851, f_sq_sell).side, "SELL")

        # 5. supertrend_flip
        f_st_buy = dict(base_f, supertrend_dir=1.0, supertrend_flipped=1.0, adx=25.0, rsi=56.0, body_atr_ratio=0.55)
        self.assertEqual(methods.supertrend_flip(bull_bar, 1.0849, f_st_buy).side, "BUY")
        f_st_sell = dict(base_f, supertrend_dir=-1.0, supertrend_flipped=1.0, adx=25.0, rsi=44.0, body_atr_ratio=0.55)
        self.assertEqual(methods.supertrend_flip(bear_bar, 1.0851, f_st_sell).side, "SELL")

        # 6. ema_ribbon_pullback
        f_rib_buy = dict(base_f, ema_8=1.0854, ema_13=1.0851, ema_21=1.0849, ema_34=1.0845, close_position=0.85, rsi=54.0)
        self.assertEqual(methods.ema_ribbon_pullback(bull_bar, 1.0850, f_rib_buy).side, "BUY")
        f_rib_sell = dict(base_f, ema_8=1.0846, ema_13=1.0849, ema_21=1.0851, ema_34=1.0855, close_position=0.15, rsi=46.0)
        self.assertEqual(methods.ema_ribbon_pullback(bear_bar, 1.0850, f_rib_sell).side, "SELL")

        # 7. connors_rsi2_reversion
        f_r2_buy = dict(base_f, rsi_2=5.0, rsi_2_previous=4.0, ema_50=1.0840, adx=20.0)
        self.assertEqual(methods.connors_rsi2_reversion(bull_bar, 1.0849, f_r2_buy).side, "BUY")
        f_r2_sell = dict(base_f, rsi_2=95.0, rsi_2_previous=96.0, ema_50=1.0860, adx=20.0)
        self.assertEqual(methods.connors_rsi2_reversion(bear_bar, 1.0851, f_r2_sell).side, "SELL")

        # 8. vwap_sigma_reversion
        f_vw_buy = dict(base_f, vwap_z=-2.25, adx=18.0, return_1=0.0003, close_position=0.80)
        self.assertEqual(methods.vwap_sigma_reversion(bull_bar, 1.0849, f_vw_buy).side, "BUY")
        f_vw_sell = dict(base_f, vwap_z=2.25, adx=18.0, return_1=-0.0003, close_position=0.20)
        self.assertEqual(methods.vwap_sigma_reversion(bear_bar, 1.0851, f_vw_sell).side, "SELL")

        # 9. session_box_breakout
        f_box_buy = dict(base_f, prior_box_30_atr=2.8, prior_high_30=1.0853, prior_low_30=1.0844, body_atr_ratio=0.75, close_position=0.85)
        self.assertEqual(methods.session_box_breakout(bull_bar, 1.0851, f_box_buy).side, "BUY")
        f_box_sell = dict(base_f, prior_box_30_atr=2.8, prior_high_30=1.0856, prior_low_30=1.0847, body_atr_ratio=0.75, close_position=0.15)
        self.assertEqual(methods.session_box_breakout(bear_bar, 1.0849, f_box_sell).side, "SELL")

    def test_council_weights_from_legacy_payload_preserves_new_prototypes(self) -> None:
        legacy = {"net": {"trend_pullback": 4.0}, "seen": {"trend_pullback": 10}, "weights": {}, "version": 3}
        restored = CouncilWeights.from_dict(legacy)
        for name in methods.METHOD_PROTOTYPES:
            self.assertIn(name, restored.net)
            self.assertIn(name, restored.seen)
        for _ in range(council.MIN_SEEN_PER_PROTOTYPE + 5):
            restored.observe({"smc_liquidity_sweep": "SELL"}, "SELL")
        self.assertGreater(restored.weights["smc_liquidity_sweep"], 0.0)


class TestImmediateLossLearningAndPersistence(unittest.TestCase):
    def test_save_state_every_is_one_and_first_real_loss_updates_and_persists_all_learners(self) -> None:
        self.assertEqual(SAVE_STATE_EVERY, 1)
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(local=LocalStore(tmp), supabase=None)
            cfg = Config(local_state_dir=tmp, horizon_minutes=5)
            candles = make_candles(140, seed=31, volatility=0.0003, drift=-0.00008)
            store.save_candles([dict(c.to_dict(), pair="EUR/USD") for c in candles])

            bot = FusionBot(config=cfg, store=store)
            bot.startup()
            self.assertIsNone(bot.model)

            feat_row = feat_mod.compute_features(candles[:120])[-1]
            assert feat_row is not None
            entry_t = candles[120].timestamp
            # Create a LIVE BUY signal on a downward series so it settles as a LOSS
            sig = Signal.create(
                pair="EUR/USD",
                side="BUY",
                entry_time=entry_t,
                strategy="fusion_council",
                reason="Core 5/6 test",
                source="LIVE",
                generated_at=entry_t,
                horizon_minutes=5,
                regime="TREND",
                council=(
                    {"name": "smc_liquidity_sweep", "side": "BUY", "weight": 0.0, "heard": False, "reason": "test"},
                    {"name": "supertrend_flip", "side": "SELL", "weight": 0.0, "heard": False, "reason": "test"},
                ),
                features=dict(feat_row),
                spread_pips=1.2,
            )
            store.save_signal(dict(sig.to_dict(), seal_state=charter.SEAL_SIGNAL, settled=False, bucket="test"))

            settled_n = bot.settle_due(at=candles[-1].timestamp + timedelta(minutes=1))
            self.assertEqual(settled_n, 1)

            # Reload a brand-new container from disk to prove immediate persistence!
            reloaded = FusionBot(config=cfg, store=Store(local=LocalStore(tmp), supabase=None))
            reloaded.startup()
            self.assertIsNotNone(reloaded.model)
            assert reloaded.model is not None
            self.assertTrue(reloaded.model.trained)
            self.assertEqual(reloaded.model.mode, MODE_VETO)
            self.assertEqual(reloaded.model.online_updates, 1)
            self.assertEqual(reloaded.mistakes.total_records(), 1)
            self.assertEqual(reloaded.weights.seen["smc_liquidity_sweep"], 1)
            self.assertEqual(reloaded.weights.seen["supertrend_flip"], 1)

            # Reconciling again must be 100% idempotent (0 duplicate replays)
            self.assertEqual(reloaded.reconcile_settled_signals(), 0)
            self.assertEqual(reloaded.mistakes.total_records(), 1)

    def test_two_consecutive_losses_tighten_confluence_to_six_of_six_until_win_resets(self) -> None:
        candles = make_candles(120, seed=19)
        feats = feat_mod.compute_features(candles)[-1]
        mem = MistakeMemory()
        key = MistakeMemory.fingerprint("RANGE", "LONDON", "BUY")
        mem.record(charter.LOSS, key)
        mem.record(charter.LOSS, key)
        self.assertFalse(mem.refuses(key)[0])
        self.assertEqual(mem.extra_confluence_checks(key), 1)

        original = dict(strategies.REGISTRY)
        try:
            strategies.REGISTRY[strategies.CORE] = lambda *a, **k: Decision("BUY", "5/6 marginal", None, 5)
            blocked = run_council(
                candles[-1],
                candles[-2].close,
                feats,
                0.55,
                weights=CouncilWeights(),
                regime="RANGE",
                session="LONDON",
                mistakes=mem,
            )
            self.assertFalse(blocked.actionable)
            self.assertIn("Post-mortem tightening", blocked.reason)

            strategies.REGISTRY[strategies.CORE] = lambda *a, **k: Decision("BUY", "6/6 full", None, 6)
            allowed = run_council(
                candles[-1],
                candles[-2].close,
                feats,
                0.55,
                weights=CouncilWeights(),
                regime="RANGE",
                session="LONDON",
                mistakes=mem,
            )
            self.assertTrue(allowed.actionable)

            # A win resets the consecutive loss streak
            mem.record(charter.WIN, key)
            self.assertEqual(mem.loss_streak(key), 0)
            self.assertEqual(mem.extra_confluence_checks(key), 0)
        finally:
            strategies.REGISTRY.clear()
            strategies.REGISTRY.update(original)


class TestAutoSettleZeroBiasAndWholeSystemFixes(unittest.TestCase):
    def test_scaled_paper_series_is_symmetric_not_bull_only(self) -> None:
        end_min = datetime(2026, 10, 26, 12, 0, tzinfo=timezone.utc)
        net_moves = []
        for step in range(10):
            bars = _build_scaled_paper_series("EUR/USD", 200, end_min, seed_offset=step)
            net_moves.append(bars[-1].close - bars[0].open)
        ups = sum(1 for m in net_moves if m > 0)
        downs = sum(1 for m in net_moves if m < 0)
        self.assertGreater(ups, 0)
        self.assertGreater(downs, 0)

    def test_web_runtime_auto_settles_and_cloud_sync_preserves_features_and_newest_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text("PAIR=EUR/USD\n", encoding="utf-8")
            rt = OperatorRuntime(env_path=env_path, state_dir=Path(tmp) / "var", auto_settle=True, auto_settle_interval_s=10.0)
            try:
                st = rt.status_payload()
                self.assertTrue(st["ledger"]["auto_settle_active"])
                self.assertEqual(st["council_members"], 22)

                candles = make_candles(140, seed=41, volatility=0.0003, drift=0.00006)
                feat_row = feat_mod.compute_features(candles[:120])[-1]
                assert feat_row is not None
                entry_t = candles[120].timestamp
                sig = Signal.create(
                    pair="EUR/USD",
                    side="BUY",
                    entry_time=entry_t,
                    strategy="fusion_council",
                    reason="Core 5/6",
                    source="LIVE",
                    generated_at=entry_t,
                    horizon_minutes=5,
                    regime="TREND",
                    council=({"name": "trend_pullback", "side": "BUY", "weight": 0.0, "heard": False, "reason": "ok"},),
                    features=dict(feat_row),
                    spread_pips=1.2,
                )
                rt.store.save_signal(dict(sig.to_dict(), seal_state="SIGNAL", settled=False, bucket="b1"))

                # Cloud sync sends stripped features={} for the same signal PLUS candles through expiry
                cloud_sig = dict(sig.to_dict(), features={}, settled=False, seal_state="SIGNAL")
                sync_res = rt.ingest_cloud_snapshot(
                    {
                        "project_ref": "idudlcxkihqnvecwqqoj",
                        "signals": [cloud_sig],
                        "candles_m1": [dict(c.to_dict(), pair="EUR/USD") for c in candles],
                    }
                )
                self.assertEqual(sync_res["auto_settled"], 1)
                after = rt.status_payload()
                self.assertEqual(after["ledger"]["pending_signals"], 0)
                self.assertGreaterEqual(after["model"]["online_updates"], 1)
                self.assertEqual(after["ledger"]["recent_signals"][0]["signal_id"], sig.signal_id)
                self.assertIn(after["ledger"]["recent_signals"][0]["outcome"], ("WIN", "LOSS"))

                # Needle scan_market summary uses evaluations cleanly
                needle_res = rt.run_needle_query("scan btc/usd right now", execute=True)
                self.assertTrue(needle_res["ok"])
                if needle_res["executed"]:
                    self.assertIn("BTC/USD:", needle_res["reply"])
            finally:
                rt.close()

    def test_frontend_charter_zero_ai_palette_and_utc_ist_placeholders(self) -> None:
        banned = re.compile(
            r"\b(purple|violet|indigo|magenta|cyan|aqua|neon|fuchsia|lavender|teal)\b",
            re.IGNORECASE,
        )
        for rel in ("index.html", "console.html", "assets/style.css", "assets/main.js"):
            text = (DOCS_DIR / rel).read_text(encoding="utf-8")
            match = banned.search(text)
            self.assertIsNone(match, f"Forbidden AI palette term {match.group(0) if match else ''} in {rel}")

        html = (DOCS_DIR / "console.html").read_text(encoding="utf-8")
        self.assertNotIn("{entry_hm}Z", html)
        self.assertIn("UTC+05:30", html)

    def test_ingest_all_performs_single_catchup_poll_when_cdn_lags_previous_bar(self) -> None:
        from unittest.mock import patch
        from fusion.simulate import synthetic_series

        base_end = datetime(2026, 10, 26, 9, 39, 0, tzinfo=timezone.utc)
        lagged = synthetic_series(220, seed=11, start=base_end - timedelta(minutes=219))
        m_minus_1 = Candle(
            timestamp=datetime(2026, 10, 26, 9, 40, 0, tzinfo=timezone.utc),
            open=lagged[-1].close,
            high=lagged[-1].close + 0.0004,
            low=lagged[-1].close - 0.0001,
            close=lagged[-1].close + 0.0003,
            volume=120.0,
        )

        class _LaggedLiveProvider:
            configured = True
            synthetic = False
            pool = object()

            def __init__(self) -> None:
                self.calls: list[int] = []

            def time_series(self, symbol: str, interval: str = "1min", outputsize: int = 320):
                self.calls.append(int(outputsize))
                if len(self.calls) == 1:
                    return list(lagged)
                return list(lagged[-5:]) + [m_minus_1]

        with tempfile.TemporaryDirectory() as tmp:
            prov = _LaggedLiveProvider()
            bot = FusionBot(
                config=Config(local_state_dir=tmp, candle_history=320),
                store=Store(local=LocalStore(Path(tmp)), supabase=None),
                provider=prov,
            )
            woke_at = datetime(2026, 10, 26, 9, 41, 5, 500_000, tzinfo=timezone.utc)
            slept: list[float] = []
            with patch("fusion.runner.now", return_value=woke_at):
                bot.ingest_all(at=woke_at, sleep_fn=slept.append)

            self.assertEqual(prov.calls, [320, 6])
            self.assertEqual(len(slept), 1)
            self.assertLessEqual(slept[0], 1.6)
            self.assertEqual(bot.candles[-1].timestamp, m_minus_1.timestamp)


if __name__ == "__main__":
    unittest.main()
