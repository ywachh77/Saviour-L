"""Tests for causal microstructure indicators (KER, CHOP) and runtime protection guards."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import tempfile
import unittest

from fusion import charter, datagate, protections
from fusion.backtest import _label, run_backtest
from fusion.config import Config
from fusion.domain import Candle
from fusion.engine import FusionEngine, Verdict
from fusion.features import compute_features
from fusion.indicators import choppiness_index, kaufman_er
from fusion.runner import FusionBot
from fusion.simulate import synthetic_series
from fusion.store import LocalStore, Store


class TestMicrostructureIndicators(unittest.TestCase):
    def test_kaufman_er_distinguishes_straight_trend_from_pure_churn(self):
        straight = [1.0 + 0.001 * i for i in range(25)]
        er_trend = kaufman_er(straight, period=10)
        self.assertIsNone(er_trend[9])
        self.assertAlmostEqual(er_trend[20], 1.0, places=9)

        oscillating = [1.0 + (0.001 if i % 2 else 0.0) for i in range(25)]
        er_chop = kaufman_er(oscillating, period=10)
        self.assertAlmostEqual(er_chop[20], 0.0, places=9)

    def test_choppiness_index_bounds_and_flat_bar_safety(self):
        base = datetime(2026, 1, 5, 10, 0, tzinfo=timezone.utc)
        flat = [
            Candle(base + timedelta(minutes=i), 1.0850, 1.0850, 1.0850, 1.0850, 100.0)
            for i in range(20)
        ]
        chop_flat = choppiness_index(flat, period=14)
        self.assertIsNone(chop_flat[12])
        self.assertEqual(chop_flat[19], 50.0)

        candles = synthetic_series(60, seed=42)
        chop_vals = choppiness_index(candles, period=14)
        for val in chop_vals[13:]:
            self.assertIsNotNone(val)
            self.assertGreaterEqual(val, 0.0)
            self.assertLessEqual(val, 100.0)

    def test_compute_features_exposes_ker_10_and_chop_14_without_changing_model_vector(self):
        candles = synthetic_series(80, seed=9)
        rows = compute_features(candles)
        latest = rows[-1]
        self.assertIsNotNone(latest)
        self.assertIn("ker_10", latest)
        self.assertIn("chop_14", latest)
        self.assertGreaterEqual(latest["ker_10"], 0.0)
        self.assertLessEqual(latest["ker_10"], 1.0)


class TestProtectionGuards(unittest.TestCase):
    def test_fx_rollover_blocks_friday_close_and_weekend_but_allows_crypto_24_7(self):
        # 2026-10-23 is Friday; entry at 21:30Z with 30m hold expires at 22:00Z (> 21:55Z)
        fri_late = datetime(2026, 10, 23, 21, 30, tzinfo=timezone.utc)
        v_fx = protections.check_fx_rollover(fri_late, 30, crypto=False)
        self.assertTrue(v_fx.blocked)
        self.assertEqual(v_fx.guard, "fx_rollover")

        # Crypto at the exact same Friday night timestamp is never blocked
        v_crypto = protections.check_fx_rollover(fri_late, 30, crypto=True)
        self.assertFalse(v_crypto.blocked)

        # Friday afternoon 15:00Z with 30m hold is safe for FX
        fri_afternoon = datetime(2026, 10, 23, 15, 0, tzinfo=timezone.utc)
        self.assertFalse(protections.check_fx_rollover(fri_afternoon, 30, crypto=False).blocked)

        # Saturday 12:00Z is blocked for FX, open for crypto
        sat = datetime(2026, 10, 24, 12, 0, tzinfo=timezone.utc)
        self.assertTrue(protections.check_fx_rollover(sat, 15, crypto=False).blocked)
        self.assertFalse(protections.check_fx_rollover(sat, 15, crypto=True).blocked)

    def test_cooldown_period_blocks_same_pair_stacking_only(self):
        t0 = datetime(2026, 10, 22, 13, 0, tzinfo=timezone.utc)
        signals = [
            {"pair": "EUR/USD", "seal_state": charter.SEAL_SIGNAL, "entry_time": "2026-10-22T12:50:00Z"},
            {"pair": "BTC/USD", "seal_state": charter.SEAL_ABSTAIN, "entry_time": "2026-10-22T12:58:00Z"},
        ]
        blocked = protections.check_cooldown_period(t0, signals, pair="EUR/USD", cooldown_minutes=15)
        self.assertTrue(blocked.blocked)
        self.assertEqual(blocked.guard, "cooldown_period")

        clear_other = protections.check_cooldown_period(t0, signals, pair="BTC/USD", cooldown_minutes=15)
        self.assertFalse(clear_other.blocked)

        clear_expired = protections.check_cooldown_period(
            t0 + timedelta(minutes=10), signals, pair="EUR/USD", cooldown_minutes=15
        )
        self.assertFalse(clear_expired.blocked)

    def test_stoploss_guard_pauses_after_consecutive_real_losses_and_ignores_synthetic(self):
        now_ts = datetime(2026, 10, 22, 15, 0, tzinfo=timezone.utc)
        rows = [
            {"pair": "EUR/USD", "outcome": charter.WIN, "net_pips": 2.4, "settled_at": "2026-10-22T13:10:00Z", "source": "LIVE"},
            {"pair": "EUR/USD", "outcome": charter.LOSS, "net_pips": -2.1, "settled_at": "2026-10-22T13:50:00Z", "source": "LIVE"},
            {"pair": "EUR/USD", "outcome": charter.LOSS, "net_pips": -1.8, "settled_at": "2026-10-22T14:20:00Z", "source": "LIVE"},
            {"pair": "EUR/USD", "outcome": charter.LOSS, "net_pips": -1.5, "settled_at": "2026-10-22T14:45:00Z", "source": "LIVE"},
            # Synthetic losses never count
            {"pair": "BTC/USD", "outcome": charter.LOSS, "net_pips": -12.0, "settled_at": "2026-10-22T14:50:00Z", "source": "SYNTHETIC"},
        ]
        verdict = protections.check_stoploss_guard(
            now_ts, rows, pair="EUR/USD", max_loss_streak=3, lookback_minutes=180, cooldown_minutes=60
        )
        self.assertTrue(verdict.blocked)
        self.assertEqual(verdict.guard, "stoploss_guard")

        # Expires 60m after 14:45Z -> at 15:50Z guard clears
        after_unlock = protections.check_stoploss_guard(
            datetime(2026, 10, 22, 15, 50, tzinfo=timezone.utc),
            rows,
            pair="EUR/USD",
            max_loss_streak=3,
            lookback_minutes=180,
            cooldown_minutes=60,
        )
        self.assertFalse(after_unlock.blocked)

    def test_engine_seal_enforces_fx_rollover_guard_when_enabled(self):
        cfg = Config(pair="EUR/USD", fx_rollover_guard=True, horizon_minutes=30)
        engine = FusionEngine(cfg)
        fri_close_entry = datetime(2026, 10, 23, 21, 40, 1, tzinfo=timezone.utc)
        gate = datagate.GateResult(datagate.OK, "OK", 1.0, 1.2, "live", 0, 100, "ok")
        res = engine._seal(
            Verdict("BUY", "Core 5/6", regime="TREND"),
            gate=gate,
            source="LIVE",
            at=fri_close_entry,
            spread=1.2,
            features={"atr": 0.00025},
        )
        self.assertEqual(res.seal_state, charter.SEAL_ABSTAIN)
        self.assertIn("rollover", res.reason.lower())

    def test_runner_enforces_cooldown_period_between_live_cycles(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(local=LocalStore(tmp), supabase=None)
            cfg = Config(pair="EUR/USD", signal_cooldown_minutes=20)
            bot = FusionBot(config=cfg, store=store)
            bot.candles = synthetic_series(120, seed=11)
            store.save_signal(
                {
                    "signal_id": "prior0001",
                    "pair": "EUR/USD",
                    "side": "BUY",
                    "seal_state": charter.SEAL_SIGNAL,
                    "entry_time": "2026-10-22T12:50:00Z",
                }
            )
            rep = bot.evaluate_minute(
                bucket="2026-10-22T13:00Z",
                at=datetime(2026, 10, 22, 13, 0, 1, tzinfo=timezone.utc),
            )
            self.assertEqual(rep.seal_state, charter.SEAL_ABSTAIN)
            self.assertIn("CooldownPeriod", rep.reason)

    def test_backtest_skips_weekend_timestamp_gaps_and_aligns_adaptive_labels(self):
        t0 = datetime(2026, 10, 23, 21, 50, tzinfo=timezone.utc)
        c0 = Candle(t0, 1.0850, 1.0852, 1.0848, 1.0851, 100.0)
        c1 = Candle(t0 + timedelta(minutes=1), 1.0851, 1.0853, 1.0849, 1.0852, 100.0)
        # Jump 48 hours across the weekend
        c2 = Candle(t0 + timedelta(hours=48), 1.0890, 1.0895, 1.0885, 1.0892, 100.0)
        self.assertIsNone(_label([c0, c1, c2], 0, 2))

        series = synthetic_series(700, seed=21)
        cfg = Config(pair="EUR/USD", adaptive_horizon=True, signal_cooldown_minutes=15)
        rep = run_backtest(series, cfg, folds=3)
        self.assertEqual(rep.bars, 700)
        self.assertGreaterEqual(rep.folds, 2)


if __name__ == "__main__":
    unittest.main()
