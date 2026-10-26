"""Tests for system-wide UTC+05:30 display timing, strict future next-candle
signal generation on 'Generate Signal' / on-demand requests, and ATR-scaled
multi-asset signal accuracy guards."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

from types import SimpleNamespace

from fusion import charter, digest, news, protections, settle, strategies
from fusion.config import Config
from fusion.domain import Candle, Signal
from fusion.engine import FusionEngine
from fusion.features import compute_features
from fusion.runner import FusionBot
from fusion.simulate import SyntheticFeed, synthetic_series
from fusion.store import LocalStore, Store
from fusion.telegram import (
    RESULT_STYLES,
    SIGNAL_STYLES,
    TelegramFormatConfig,
    render_result_message,
    render_signal_message,
)
from fusion.timeutil import (
    DISPLAY_TZ_LABEL,
    floor_minute,
    format_display_datetime,
    format_display_hm,
    format_display_iso,
    next_future_candle_open,
    parse_datetime,
)


class _NoopTransport:
    def __init__(self):
        self.sent: list[str] = []

    def send_message(self, chat_id: str, text: str) -> int:
        self.sent.append(text)
        return len(self.sent)

    def get_updates(self, offset=None, timeout: int = 0):
        return []


class TestUtcPlus0530SystemWideDisplay(unittest.TestCase):
    def test_display_helpers_convert_utc_to_utc_plus_0530_across_day_boundary(self):
        t_morning = datetime(2026, 10, 26, 10, 42, 15, tzinfo=timezone.utc)
        self.assertEqual(format_display_hm(t_morning), "16:12 UTC+05:30")
        self.assertEqual(format_display_datetime(t_morning, seconds=False), "2026-10-26 16:12 UTC+05:30")
        self.assertEqual(
            format_display_datetime(t_morning, seconds=True),
            "2026-10-26 16:12:15 UTC+05:30",
        )
        self.assertEqual(format_display_iso(t_morning), "2026-10-26T16:12:15+05:30")

        # 20:45 UTC crosses midnight into next calendar date 02:15 UTC+05:30
        t_late = datetime(2026, 10, 26, 20, 45, 0, tzinfo=timezone.utc)
        self.assertEqual(format_display_datetime(t_late, seconds=False), "2026-10-27 02:15 UTC+05:30")
        self.assertEqual(Config().timezone_name, DISPLAY_TZ_LABEL)

    def test_all_signal_and_result_styles_render_strictly_in_utc_plus_0530(self):
        sig = Signal.create(
            pair="EUR/USD",
            side="BUY",
            entry_time=datetime(2026, 10, 26, 10, 42, tzinfo=timezone.utc),
            strategy="fusion_council",
            reason="Core 6/6 (pullback rebound)",
            source="LIVE",
            generated_at=datetime(2026, 10, 26, 10, 41, 15, tzinfo=timezone.utc),
            horizon_minutes=30,
            spread_pips=1.2,
            regime="TREND",
        )
        for style in SIGNAL_STYLES:
            fmt = TelegramFormatConfig(
                signal_style=style,
                signal_template="{pair} {side} @ {entry_hm} {tz} -> {expiry_hm} {tz}",
            )
            rendered = render_signal_message(sig, fmt)
            self.assertIn("16:12", rendered, f"style={style} missing 16:12")
            self.assertIn("16:42", rendered, f"style={style} missing 16:42")
            self.assertIn("UTC+05:30", rendered, f"style={style} missing UTC+05:30")
            self.assertNotIn("10:42 UTC", rendered)

        settlement = SimpleNamespace(
            signal_id=sig.signal_id,
            entry_price=1.08500,
            exit_price=1.08532,
            gross_pips=3.2,
            spread_pips=1.2,
            net_pips=2.0,
            outcome=charter.WIN,
        )
        for rstyle in RESULT_STYLES:
            fmt = TelegramFormatConfig(
                result_style=rstyle,
                result_template="{pair} {outcome} {entry_price}->{exit_price}",
            )
            rendered_res = render_result_message(sig, settlement, fmt)
            self.assertIn("EUR/USD", rendered_res)

    def test_digest_status_news_and_rollover_messages_use_utc_plus_0530(self):
        cfg = Config()
        stats = settle.summarise([])
        d_text = digest.build_digest(
            day="2026-10-26",
            signals=[
                {
                    "pair": "EUR/USD",
                    "side": "BUY",
                    "entry_time": "2026-10-26T10:42:00Z",
                    "horizon_minutes": 30,
                }
            ],
            stats=stats,
            unresolved=0,
            calibration_state="RAW",
            calibration_samples=0,
            model_mode="VETO",
            council_weights={},
            learning_events=[],
            key_pool={"configured": 1, "exhausted": 0, "total_used": 4, "total_limit": 800},
            runner="github-actions",
            alerts=[],
            nightly=None,
        )
        self.assertIn("UTC+05:30", d_text)

        s_text = digest.build_status(
            config=cfg,
            paused=False,
            runner="github-actions",
            heartbeat_row={"timestamp": "2026-10-26T12:04:00Z", "note": "ok"},
            stats=stats,
            unresolved=0,
            model_mode="VETO",
            calibration_state="RAW",
            calibration_samples=0,
            key_pool={"configured": 1, "exhausted": 0, "total_used": 4, "total_limit": 800},
            last_cycle={"at": "2026-10-26T12:04:05Z", "note": "ok"},
            store_errors=[],
        )
        self.assertIn("UTC+05:30", s_text)
        self.assertIn("clock:", s_text)

        # Friday 21:56 UTC -> Saturday 03:26 UTC+05:30 rollover guard
        fri_rollover = datetime(2026, 10, 23, 21, 56, tzinfo=timezone.utc)
        ro = protections.check_fx_rollover(fri_rollover, 30, crypto=False)
        self.assertTrue(ro.blocked)
        self.assertIn("UTC+05:30", ro.reason)

        # Custom news blackout window displays UTC+05:30 range while gating exact UTC instant
        blk = news.check_blackout(
            datetime(2026, 10, 26, 13, 30, tzinfo=timezone.utc),
            pair="EUR/USD",
            extra_windows="2026-10-26T13:15Z/2026-10-26T13:45Z",
        )
        self.assertTrue(blk.blocked)
        self.assertIn("UTC+05:30", blk.reason)
        self.assertIn("18:45", blk.reason)
        self.assertIn("19:15", blk.reason)


class TestFutureNextCandleGeneration(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="fusion_future_sig_")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_next_future_candle_open_is_strictly_greater_than_now_at_every_second(self):
        for sec, micro in ((0, 0), (0, 500_000), (3, 0), (9, 900_000), (30, 0), (59, 999_000)):
            now_t = datetime(2026, 10, 26, 13, 14, sec, micro, tzinfo=timezone.utc)
            nxt = next_future_candle_open(now_t)
            self.assertEqual(nxt, datetime(2026, 10, 26, 13, 15, 0, tzinfo=timezone.utc))
            self.assertGreater(nxt, now_t)

    def test_engine_evaluate_with_observed_at_seals_future_bar_without_forming_bar_leakage(self):
        # Build history up through 13:13 UTC (closed) plus forming 13:14 UTC bar
        start = datetime(2026, 10, 26, 9, 0, tzinfo=timezone.utc)
        candles = synthetic_series(255, seed=19, start=start)
        self.assertEqual(candles[-1].timestamp, datetime(2026, 10, 26, 13, 14, tzinfo=timezone.utc))

        observed = datetime(2026, 10, 26, 13, 14, 3, tzinfo=timezone.utc)  # 3s into 13:14
        future_entry = next_future_candle_open(observed)  # 13:15:00 UTC

        poisoned = list(candles[:-1]) + [
            Candle(
                timestamp=candles[-1].timestamp,
                open=candles[-1].open,
                high=9.9999,
                low=0.0001,
                close=9.9999,
                volume=999_999.0,
            )
        ]

        cfg = Config(pair="EUR/USD", session_filter=False, news_filter=False)
        engine = FusionEngine(cfg)
        clean_res = engine.evaluate(candles[:-1], at=future_entry, observed_at=observed, source="LIVE")
        poison_res = engine.evaluate(poisoned, at=future_entry, observed_at=observed, source="LIVE")

        # Forming 13:14 bar was causally stripped in both calls
        self.assertEqual(clean_res.reason, poison_res.reason)
        self.assertEqual(clean_res.gate.state, "OK")
        self.assertEqual(poison_res.gate.state, "OK")

    def test_on_demand_signal_early_in_minute_targets_future_next_candle_not_present_candle(self):
        # At 13:14:03 UTC (inside the first 10s of 13:14), _sealable_stamp would pick 13:14,
        # which has ALREADY opened 3s ago. on_demand_signal MUST target 13:15 (> real_now).
        real_now = datetime(2026, 10, 26, 13, 14, 3, tzinfo=timezone.utc)
        start = real_now.replace(second=0, microsecond=0) - timedelta(minutes=260)
        candles = synthetic_series(261, seed=20260101, start=start)
        feed = SyntheticFeed(candles)
        feed.advance_to(real_now + timedelta(minutes=1))

        cfg = Config(
            pair="EUR/USD",
            pairs=("EUR/USD",),
            local_state_dir=self.tmpdir,
            session_filter=False,
            news_filter=False,
        )
        store = Store(supabase=None, local=LocalStore(self.tmpdir))
        bot = FusionBot(cfg, store=store, provider=feed, telegram=None)

        # Force an actionable BUY verdict so we can inspect the sealed Signal's entry_time
        from fusion.engine import Verdict
        from fusion.council import CouncilResult

        forced_verdict = Verdict(
            "BUY",
            "Core 6/6 (pullback rebound)",
            "TREND",
            council=CouncilResult("BUY", "Core 6/6 (pullback rebound)", 6, (), 1.0, 0.0, False),
        )
        with mock.patch("fusion.runner.now", return_value=real_now), mock.patch.object(
            FusionEngine, "decide", return_value=forced_verdict
        ):
            reply = bot.on_demand_signal()

        self.assertIn("18:45 UTC+05:30", reply)  # 13:15 UTC == 18:45 UTC+05:30
        saved = store.load_signals(limit=5)
        self.assertEqual(len(saved), 1)
        sealed_entry = parse_datetime(saved[0]["entry_time"])
        self.assertEqual(sealed_entry, datetime(2026, 10, 26, 13, 15, 0, tzinfo=timezone.utc))
        self.assertGreater(sealed_entry, real_now)

    def test_web_dashboard_generate_signals_emits_future_next_candle_in_utc_plus_0530(self):
        import sys
        from pathlib import Path

        scripts_dir = str(Path(__file__).resolve().parents[1] / "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        import web_dashboard

        env_p = Path(self.tmpdir) / ".env"
        env_p.write_text("", encoding="utf-8")
        runtime = web_dashboard.OperatorRuntime(
            env_path=env_p,
            state_dir=Path(self.tmpdir) / "var",
        )
        # Pin clock to a weekday London/NY overlap minute so FX & crypto are all open
        fake_now = datetime(2026, 10, 26, 13, 22, 18, tzinfo=timezone.utc)
        expected_entry = datetime(2026, 10, 26, 13, 23, 0, tzinfo=timezone.utc)

        with mock.patch.object(web_dashboard, "utcnow", return_value=fake_now):
            status = runtime.status_payload()
            self.assertEqual(status["timezone"], "UTC+05:30")
            self.assertIn("UTC+05:30", status["server_time"])
            self.assertEqual(status["next_candle_hm"], "18:53 UTC+05:30")

            res_scan = runtime.generate_signals(pair="ALL", mode="scan")
            self.assertTrue(res_scan["ok"])
            self.assertEqual(res_scan["timezone"], "UTC+05:30")
            self.assertEqual(res_scan["target_entry_hm"], "18:53 UTC+05:30")
            self.assertGreater(res_scan["seconds_until_entry"], 0)
            self.assertGreater(len(res_scan["evaluations"]), 0)
            for ev in res_scan["evaluations"]:
                self.assertTrue(ev["is_future_entry"])
                entry_dt = parse_datetime(ev["target_entry_at"])
                self.assertEqual(entry_dt, expected_entry)
                self.assertGreater(entry_dt, fake_now)
                self.assertEqual(ev["target_entry_hm"], "18:53 UTC+05:30")
                if ev.get("signal"):
                    sig_entry = parse_datetime(ev["signal"]["entry_time"])
                    self.assertEqual(sig_entry, expected_entry)
                    self.assertEqual(ev["signal"]["entry_hm"], "18:53 UTC+05:30")
                    self.assertIn("UTC+05:30", ev["signal"]["expiry_hm"])


class TestSignalAccuracyUpgrades(unittest.TestCase):
    def test_mode2_captures_orderly_range_absorption_and_blocks_macro_waterfall(self):
        candles = synthetic_series(120, seed=44)
        base_f = dict(compute_features(candles)[-1])
        bar = Candle(
            datetime(2026, 10, 26, 13, 0, tzinfo=timezone.utc),
            open=1.08520,
            high=1.08525,
            low=1.08460,
            close=1.08505,  # red bar with long lower rejection wick -> absorption
            volume=900.0,
        )
        absorb_f = dict(
            base_f,
            atr_pct=0.00025,
            bb_lower=1.08510,
            bb_upper=1.08650,  # close < bb_lower -> bb_pos <= 0.0
            vwap_dev=-2.15,
            rsi=27.5,
            rsi_previous=29.0,
            stoch_k=14.0,
            cci=-145.0,
            body_atr_ratio=0.45,
            close_position=0.69,
            wick_lower=0.65,
            body_ratio=0.23,
            mtf_gap_15=-2.0,
            mtf_slope_15=-0.2,
            mtf_gap_60=-1.8,
            mtf_slope_60=-0.1,
        )
        d_ok = strategies.orbix_confluence(bar, 1.08520, absorb_f, regime="RANGE")
        self.assertEqual(d_ok.side, "BUY")
        self.assertIn("exhaustion reversal", d_ok.reason)

        # Same oversold bar inside a violent -14 bps / -2.2 bps/bar 60m crash -> blocked!
        waterfall_f = dict(absorb_f, mtf_gap_60=-14.0, mtf_slope_60=-2.2)
        d_crash = strategies.orbix_confluence(bar, 1.08520, waterfall_f, regime="RANGE")
        self.assertEqual(d_crash.side, "NO SIGNAL")

    def test_atr_scaled_macro_filter_rejects_flat_noise_on_crypto_but_keeps_fx_scale(self):
        candles = synthetic_series(120, seed=31)
        f_row = dict(compute_features(candles)[-1])
        bar = Candle(
            datetime(2026, 10, 26, 13, 0, tzinfo=timezone.utc),
            95000.0,
            95120.0,
            94980.0,
            95100.0,
            500.0,
        )
        crypto_buy = dict(
            f_row,
            ema_fast=95080.0,
            ema_slow=94950.0,
            ema_fast_previous=95040.0,
            ema_slow_previous=94940.0,
            bb_lower=94700.0,
            bb_upper=95400.0,
            prior_high_20=95350.0,
            prior_low_20=94600.0,
            return_1=0.0008,
            return_3=0.0018,
            rsi=58.0,
            rsi_previous=54.0,
            stoch_k=62.0,
            cci=75.0,
            chop_14=48.0,
            adx=27.0,
            plus_di=31.0,
            minus_di=14.0,
            body_atr_ratio=0.55,
            close_position=0.80,
            atr=85.0,
            atr_pct=0.00090,  # 9.0 bps M1 ATR (typical BTC/USD) -> requires >= 9.9 bps 1h gap
            mtf_ready_15=1.0,
            mtf_bias_15=1.0,
            mtf_gap_15=6.0,
            mtf_ready_60=1.0,
            mtf_gap_60=2.5,  # 2.5 bps is flat noise on BTC/USD (< 1.1 * 9.0 = 9.9 bps)
            mtf_slope_60=0.5,
        )
        cfg = Config(
            pair="BTC/USD",
            mtf_filter=True,
            mtf_minutes=15,
            mtf_min_gap_bps=2.0,
            mtf_macro_filter=True,
            mtf_macro_min_gap_bps=1.5,
            session_filter=False,
            news_filter=False,
        )
        engine = FusionEngine(cfg)
        v_flat = engine.decide(bar, 95020.0, crypto_buy)
        self.assertEqual(v_flat.side, "NO SIGNAL")
        self.assertIn("Macro 60m HTF filter", v_flat.reason)

        # Genuine 12.5 bps 1h macro drift (>= 9.9 bps) confirms BTC/USD BUY
        crypto_trend = dict(crypto_buy, mtf_gap_60=12.5, mtf_slope_60=1.8)
        v_trend = engine.decide(bar, 95020.0, crypto_trend)
        self.assertEqual(v_trend.side, "BUY")


if __name__ == "__main__":
    unittest.main()
