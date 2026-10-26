"""Regression guards for live entry-timing causality and anti-chasing signal gates."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from fusion import charter
from fusion.config import Config
from fusion.domain import Candle, Decision, Signal
from fusion.engine import FusionEngine
from fusion.features import compute_features
from fusion.keypool import KeyPool
from fusion.learning import run_nightly
from fusion.model import MODE_VETO
from fusion.runner import FusionBot
from fusion.simulate import synthetic_series
from fusion.store import LocalStore, Store
from fusion.strategies import orbix_confluence


class _FormingBarProvider:
    """Fake live provider that includes a still-forming candle for the current minute."""

    configured = True
    synthetic = False

    def __init__(self, closed_candles: list[Candle], forming_candle: Candle, delay_s: float = 0.0):
        self._closed = list(closed_candles)
        self._forming = forming_candle
        self._delay_s = delay_s
        self.calls = 0

    def time_series(self, symbol: str, interval: str = "1min", outputsize: int = 320) -> list[Candle]:
        self.calls += 1
        if self._delay_s > 0:
            time.sleep(self._delay_s)
        return list(self._closed) + [self._forming]


class TestEntryTimingCausality(unittest.TestCase):
    def test_wait_for_seal_window_aligns_mid_minute_wakeup_to_next_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(local=LocalStore(Path(tmp)), supabase=None)
            bot = FusionBot(config=Config(local_state_dir=tmp), store=store)

            slept: list[float] = []
            # Woke at 17:57:18.250Z -> must sleep until 17:58:01.500Z (~43.25s)
            t_woke = datetime(2026, 10, 25, 17, 57, 18, 250_000, tzinfo=timezone.utc)
            t_landed = datetime(2026, 10, 25, 17, 58, 1, 500_000, tzinfo=timezone.utc)
            clocks = [t_woke, t_woke, t_woke, t_landed]

            def fake_now():
                return clocks.pop(0) if len(clocks) > 1 else clocks[0]

            with patch("fusion.runner.now", side_effect=fake_now):
                landed = bot.wait_for_seal_window(sleep_fn=slept.append)

            self.assertEqual(len(slept), 1)
            self.assertAlmostEqual(slept[0], 43.25, places=2)
            self.assertEqual(landed, t_landed)

    def test_wait_for_seal_window_runs_immediately_inside_early_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(local=LocalStore(Path(tmp)), supabase=None)
            bot = FusionBot(config=Config(local_state_dir=tmp), store=store)
            slept: list[float] = []
            t_prompt = datetime(2026, 10, 25, 20, 42, 2, 400_000, tzinfo=timezone.utc)
            with patch("fusion.runner.now", return_value=t_prompt):
                landed = bot.wait_for_seal_window(sleep_fn=slept.append)
            self.assertEqual(slept, [])
            self.assertEqual(landed, t_prompt)

    def test_on_demand_signal_strips_unfinished_forming_candle_when_rolled_forward(self):
        base_end = datetime(2026, 1, 5, 13, 4, 0, tzinfo=timezone.utc)
        closed = synthetic_series(
            220,
            seed=19,
            start=base_end - timedelta(minutes=219),
        )
        forming_ts = datetime(2026, 1, 5, 13, 5, 0, tzinfo=timezone.utc)
        real_wall = datetime(2026, 1, 5, 13, 5, 37, tzinfo=timezone.utc)

        forming_up = Candle(forming_ts, 1.1000, 1.1350, 1.0999, 1.1345, 5000.0)
        forming_dn = Candle(forming_ts, 1.1000, 1.1001, 1.0650, 1.0655, 5000.0)

        captured_closes: list[float] = []

        def spy_core(candle, prev_close, features, *args, **kwargs):
            captured_closes.append(candle.close)
            return Decision("NO SIGNAL", "Core abstains: spy", None, 0)

        with tempfile.TemporaryDirectory() as tmp1, tempfile.TemporaryDirectory() as tmp2:
            bot_up = FusionBot(
                config=Config(local_state_dir=tmp1),
                store=Store(local=LocalStore(Path(tmp1)), supabase=None),
                provider=_FormingBarProvider(closed, forming_up),
            )
            bot_dn = FusionBot(
                config=Config(local_state_dir=tmp2),
                store=Store(local=LocalStore(Path(tmp2)), supabase=None),
                provider=_FormingBarProvider(closed, forming_dn),
            )
            from fusion import strategies

            orig = strategies.REGISTRY[strategies.CORE]
            strategies.REGISTRY[strategies.CORE] = spy_core
            try:
                with patch("fusion.runner.now", return_value=real_wall):
                    bot_up.on_demand_signal()
                    bot_dn.on_demand_signal()
            finally:
                strategies.REGISTRY[strategies.CORE] = orig

        # Both runs MUST see the last CLOSED bar (13:04), never the forming 13:05 bar!
        self.assertEqual(len(captured_closes), 2)
        self.assertEqual(captured_closes[0], closed[-1].close)
        self.assertEqual(captured_closes[1], closed[-1].close)

    def test_boundary_seal_refuses_missing_previous_minute_bar(self):
        # Woke at 14:43:02Z, but provider only has bars up to 14:41Z (14:42Z unpublished!)
        stale_prev_end = datetime(2026, 1, 5, 14, 41, 0, tzinfo=timezone.utc)
        closed = synthetic_series(220, seed=29, start=stale_prev_end - timedelta(minutes=219))
        forming = Candle(datetime(2026, 1, 5, 14, 43, 0, tzinfo=timezone.utc), 1.1, 1.1002, 1.0998, 1.1001, 100.0)
        seal_now = datetime(2026, 1, 5, 14, 43, 2, 100_000, tzinfo=timezone.utc)

        with tempfile.TemporaryDirectory() as tmp:
            bot = FusionBot(
                config=Config(local_state_dir=tmp, mtf_filter=False, news_filter=False),
                store=Store(local=LocalStore(Path(tmp)), supabase=None),
                provider=_FormingBarProvider(closed, forming),
            )
            with patch("fusion.runner.now", return_value=seal_now):
                report = bot.minute_cycle()
            self.assertEqual(report.seal_state, charter.SEAL_ABSTAIN)
            self.assertIn("not published yet", report.reason)

    def test_slow_ingest_crossing_seal_limit_is_discarded_late_not_masked_by_pre_ingest_stamp(self):
        base_end = datetime(2026, 1, 5, 14, 42, 0, tzinfo=timezone.utc)
        closed = synthetic_series(220, seed=23, start=base_end - timedelta(minutes=219))
        forming = Candle(datetime(2026, 1, 5, 14, 43, 0, tzinfo=timezone.utc), 1.1, 1.1002, 1.0998, 1.1001, 100.0)

        woke_at = datetime(2026, 1, 5, 14, 43, 1, 200_000, tzinfo=timezone.utc)
        post_ingest = datetime(2026, 1, 5, 14, 43, 11, 600_000, tzinfo=timezone.utc)  # 11.6s > 10s!
        clock_seq = [woke_at, post_ingest, post_ingest, post_ingest]

        def fake_now():
            return clock_seq.pop(0) if len(clock_seq) > 1 else clock_seq[0]

        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(local_state_dir=tmp, max_seal_latency_seconds=10, mtf_filter=False, news_filter=False)
            bot = FusionBot(
                config=cfg,
                store=Store(local=LocalStore(Path(tmp)), supabase=None),
                provider=_FormingBarProvider(closed, forming),
            )
            from fusion import strategies

            orig = strategies.REGISTRY[strategies.CORE]
            strategies.REGISTRY[strategies.CORE] = lambda *a, **k: Decision(
                "BUY", "Core 6/6: forced. Rule score is a count, not a probability.", None, 6
            )
            try:
                with patch("fusion.runner.now", side_effect=fake_now):
                    report = bot.minute_cycle()
            finally:
                strategies.REGISTRY[strategies.CORE] = orig

            self.assertEqual(report.seal_state, charter.SEAL_DISCARDED_LATE)
            self.assertEqual(bot.store.load_signals(limit=10), [])

    def test_concurrent_multi_pair_ingest_and_keypool_thread_safety(self):
        pool = KeyPool(["key_slot_alpha_123456", "key_slot_beta_123456"], minute_limit=50, day_limit=500)

        def worker(_i: int) -> bool:
            res = pool.reserve(1)
            if res.ok:
                pool.report_success(res)
            return res.ok

        with ThreadPoolExecutor(max_workers=12) as ex:
            results = list(ex.map(worker, range(40)))
        self.assertEqual(sum(results), 40)
        summary = pool.summary()
        total_used = sum(s["day_used"] for s in summary["slots"])
        self.assertEqual(total_used, 40)

        # Also verify 3-pair ingest_all runs concurrently in ~0.12s instead of ~0.36s
        base_end = datetime(2026, 1, 5, 14, 0, 0, tzinfo=timezone.utc)
        closed = synthetic_series(90, seed=7, start=base_end - timedelta(minutes=89))
        forming = Candle(base_end + timedelta(minutes=1), 1.1, 1.1001, 1.0999, 1.1, 100.0)
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(local=LocalStore(Path(tmp)), supabase=None)
            cfg = Config(local_state_dir=tmp, pairs=("EUR/USD", "BTC/USD", "ETH/USD"))
            bot = FusionBot(config=cfg, store=store, provider=_FormingBarProvider(closed, forming, delay_s=0.12))
            for peer in bot.peer_bots.values():
                peer.provider = _FormingBarProvider(closed, forming, delay_s=0.12)
            t0 = time.monotonic()
            bot.ingest_all(at=base_end + timedelta(minutes=1, seconds=2))
            elapsed = time.monotonic() - t0
            self.assertLess(elapsed, 0.30, f"Expected parallel ingest <0.30s, took {elapsed:.3f}s")

    def test_settle_due_refreshes_provider_only_when_pending_signal_crossed_expiry(self):
        t_entry = datetime(2026, 1, 5, 14, 10, 0, tzinfo=timezone.utc)
        closed_initial = synthetic_series(100, seed=13, start=t_entry - timedelta(minutes=100))
        closed_after = synthetic_series(110, seed=13, start=t_entry - timedelta(minutes=100))

        class _StepProvider:
            configured = True
            synthetic = False

            def __init__(self):
                self.calls = 0

            def time_series(self, symbol: str, interval: str = "1min", outputsize: int = 320):
                self.calls += 1
                return list(closed_after)

        with tempfile.TemporaryDirectory() as tmp:
            store = Store(local=LocalStore(Path(tmp)), supabase=None)
            prov = _StepProvider()
            bot = FusionBot(config=Config(local_state_dir=tmp), store=store, provider=prov)
            bot.candles = list(closed_initial)

            # No pending signals -> 0 provider calls
            bot.settle_due(at=t_entry + timedelta(minutes=10))
            self.assertEqual(prov.calls, 0)

            # Save an unsettled signal expiring at t_entry + 5m (exit bar = t_entry + 4m, closes at t_entry + 5m)
            sig = Signal(
                signal_id="sig_due_01",
                pair="EUR/USD",
                side="BUY",
                entry_time=t_entry,
                expiry_time=t_entry + timedelta(minutes=5),
                generated_at=t_entry + timedelta(seconds=2),
                horizon_minutes=5,
                strategy="orbix_confluence",
                reason="test",
                source="LIVE",
                probability_up=None,
                probability_raw=None,
                calibration_state="RAW",
                uncertainty=None,
                regime="TREND",
                council={},
                features={},
                spread_pips=1.2,
            )
            store.save_signal(dict(sig.to_dict(), seal_state=charter.SEAL_SIGNAL, settled=False, bucket="2026-01-05T14:10Z"))

            # Before expiry -> still 0 provider calls
            bot.settle_due(at=t_entry + timedelta(minutes=3))
            self.assertEqual(prov.calls, 0)

            # Past expiry + 2s buffer -> refreshes once and settles immediately!
            settled_n = bot.settle_due(at=t_entry + timedelta(minutes=5, seconds=5))
            self.assertEqual(prov.calls, 1)
            self.assertEqual(settled_n, 1)


class TestLosingSignalRootCauseGuards(unittest.TestCase):
    def test_climax_breakout_top_9549b5e1_is_rejected_by_core_gate(self):
        base = synthetic_series(140, seed=41)
        f = dict(compute_features(base)[-1])
        # Exact indicator profile of live loss #9549b5e1 (BTC/USD BUY -160.01p)
        f.update(
            {
                "atr": 120.0,
                "atr_pct": 0.0015,
                "ema_fast": 80710.0,
                "ema_slow": 80600.0,
                "return_1": 180.0 / 80680.0,
                "return_3": 290.0 / 80570.0,
                "rsi": 67.94,
                "rsi_previous": 63.10,
                "stoch_k": 96.24,
                "cci": 151.1,
                "adx": 27.5,
                "plus_di": 34.0,
                "minus_di": 12.0,
                "body_atr_ratio": 1.607,
                "close_position": 1.0,
                "prior_high_20": 80810.0,
                "prior_low_20": 80420.0,
            }
        )
        climax_bar = Candle(datetime(2026, 10, 25, 17, 56, tzinfo=timezone.utc), 80667.0, 80860.01, 80660.0, 80860.01, 100.0)
        decision = orbix_confluence(climax_bar, 80667.0, f)
        self.assertEqual(decision.side, "NO SIGNAL")

    def test_opposing_5m_mtf_cascade_blocks_range_exhaustion_b1157db2(self):
        base = synthetic_series(140, seed=43)
        f = dict(compute_features(base)[-1])
        # Strong SELL setup on 1m + 15m, but 5m MTF is strongly rallying (+11.26 bps, slope +2.50 bps) like #b1157db2
        f.update(
            {
                "atr": 2.8,
                "atr_pct": 0.0012,
                "ema_fast": 2328.0,
                "ema_slow": 2331.0,
                "return_1": -0.0008,
                "return_3": 0.0009,  # pullback reset up before turn down
                "rsi": 48.0,
                "rsi_previous": 53.0,
                "stoch_k": 45.0,
                "cci": -40.0,
                "adx": 24.0,
                "plus_di": 14.0,
                "minus_di": 28.0,
                "body_atr_ratio": 0.55,
                "close_position": 0.20,
                "prior_high_20": 2340.0,
                "prior_low_20": 2318.0,
                "mtf_ready_15": 1.0,
                "mtf_bias_15": -1.0,
                "mtf_gap_15": -4.63,
                "mtf_slope_15": -0.20,
                "mtf_ready_5": 1.0,
                "mtf_bias_5": 1.0,
                "mtf_gap_5": 11.26,
                "mtf_slope_5": 2.50,
            }
        )
        bar = Candle(datetime(2026, 10, 26, 0, 40, tzinfo=timezone.utc), 2329.2, 2329.4, 2326.5, 2326.8, 100.0)
        cfg = Config(pair="ETH/USD", mtf_filter=True, mtf_minutes=15, mtf_min_gap_bps=2.0, session_filter=False, news_filter=False)
        verdict = FusionEngine(cfg).decide(bar, 2329.2, f, spread_pips=2.4)
        self.assertEqual(verdict.side, "NO SIGNAL")
        self.assertIn("5m cascade", verdict.reason)

    def test_run_nightly_seeds_defensive_veto_baseline_when_advisory_gate_withholds(self):
        candles = synthetic_series(700, seed=77)
        cfg = Config(pair="EUR/USD", horizon_minutes=5, nightly_min_samples=200)
        model, ensemble, report = run_nightly(candles, [], config=cfg, champion=None)
        if not report.promoted:
            self.assertIsNotNone(model)
            self.assertTrue(model.trained)
            self.assertEqual(model.mode, MODE_VETO)
            self.assertTrue(ensemble.ready)


if __name__ == "__main__":
    unittest.main()
