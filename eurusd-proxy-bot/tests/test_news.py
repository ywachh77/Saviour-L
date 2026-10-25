"""Deterministic macro-release blackout tests (NFP + explicit windows)."""

from datetime import datetime, timedelta, timezone
import unittest

from fusion import news
from fusion.config import Config
from fusion.engine import FusionEngine
from fusion.features import compute_features

from .helpers import make_candles

UTC = timezone.utc


class TestNewsBlackout(unittest.TestCase):
    def test_winter_nfp_first_friday_0830_est_is_1330_utc(self):
        # 2026-01-02 is the first Friday of January 2026 (EST = UTC-5).
        at_print = datetime(2026, 1, 2, 13, 30, tzinfo=UTC)
        self.assertTrue(news.is_nfp_window(at_print))
        self.assertTrue(news.check_blackout(at_print, pair="EUR/USD").blocked)
        self.assertTrue(news.check_blackout(at_print, pair="BTC/USD").blocked)
        # 20 minutes after release is clear again (default +15m buffer).
        self.assertFalse(news.is_nfp_window(datetime(2026, 1, 2, 13, 50, tzinfo=UTC)))

    def test_summer_nfp_first_friday_0830_edt_is_1230_utc(self):
        # 2026-08-07 is the first Friday of August 2026 (EDT = UTC-4).
        at_print = datetime(2026, 8, 7, 12, 30, tzinfo=UTC)
        self.assertTrue(news.is_nfp_window(at_print))
        # 13:30 UTC in summer is an hour AFTER the print, so it must NOT block.
        self.assertFalse(news.is_nfp_window(datetime(2026, 8, 7, 13, 30, tzinfo=UTC)))

    def test_second_friday_is_never_nfp(self):
        second_friday = datetime(2026, 1, 9, 13, 30, tzinfo=UTC)
        self.assertFalse(news.is_nfp_window(second_friday))

    def test_explicit_operator_windows_parse_named_and_ranged_entries(self):
        spec = "FOMC@2026-11-04T19:00Z, 2026-11-12T13:25Z/2026-11-12T13:45Z, malformed"
        hit_fomc = news.check_blackout(datetime(2026, 11, 4, 19, 5, tzinfo=UTC), extra_windows=spec)
        self.assertTrue(hit_fomc.blocked)
        self.assertEqual(hit_fomc.event, "FOMC")

        hit_range = news.check_blackout(datetime(2026, 11, 12, 13, 30, tzinfo=UTC), extra_windows=spec)
        self.assertTrue(hit_range.blocked)

        clear = news.check_blackout(datetime(2026, 11, 4, 20, 0, tzinfo=UTC), extra_windows=spec)
        self.assertFalse(clear.blocked)

    def test_engine_abstains_during_nfp_when_news_filter_enabled(self):
        # Decision after 13:29Z candle targets the 13:30Z NFP entry bar.
        nfp_entry = datetime(2026, 1, 2, 13, 30, tzinfo=UTC)
        candles = make_candles(160, seed=11, start=nfp_entry - timedelta(minutes=160))
        features = compute_features(candles)[-1]

        blocked_engine = FusionEngine(Config(pair="EUR/USD", news_filter=True))
        verdict = blocked_engine.decide(candles[-1], candles[-2].close, features)
        self.assertEqual(verdict.side, "NO SIGNAL")
        self.assertIn("News blackout", verdict.reason)


if __name__ == "__main__":
    unittest.main()
