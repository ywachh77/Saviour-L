"""The council's rule sets: readable, bounded, and unwilling to guess."""

import unittest

from fusion.domain import Candle
from fusion.features import compute_features
from fusion.strategies import CORE, PROTOTYPES, REGISTRY, orbix_confluence

from .helpers import make_candles

# ATR band the core gate was written for.
ATR_MIN, ATR_MAX = 0.0, 0.004


def last_bar(count: int = 160, **kwargs):
    candles = make_candles(count, **kwargs)
    features = compute_features(candles)[-1]
    return candles, candles[-1], candles[-2].close, features


class TestCoreGate(unittest.TestCase):
    def test_missing_features_abstains(self):
        candles = make_candles(120)
        decision = orbix_confluence(candles[-1], candles[-2].close, None)
        self.assertEqual(decision.side, "NO SIGNAL")
        self.assertIn("Not enough completed candles", decision.reason)

    def test_abstains_outside_its_volatility_band(self):
        candles, candle, previous_close, features = last_bar(160, volatility=0.02)
        decision = orbix_confluence(candle, previous_close, features)
        self.assertEqual(decision.side, "NO SIGNAL")
        self.assertIn("volatility", decision.reason)

    def test_a_signal_names_which_checks_passed(self):
        candles, candle, previous_close, features = last_bar(400, volatility=0.00025, drift=0.00004, seed=19)
        decision = orbix_confluence(candle, previous_close, features)
        if decision.actionable:
            self.assertIn("Core", decision.reason)
            self.assertGreaterEqual(decision.score, 5)
        else:
            self.assertIn("abstains", decision.reason)

    def test_mandatory_checks_cannot_be_skipped(self):
        """A high score with a flat market is still a NO SIGNAL."""
        candles, candle, previous_close, features = last_bar(200, volatility=0.0)
        decision = orbix_confluence(candle, previous_close, features)
        self.assertEqual(decision.side, "NO SIGNAL")

    def test_score_is_a_count_not_a_probability(self):
        candles, candle, previous_close, features = last_bar(200)
        decision = orbix_confluence(candle, previous_close, features)
        if decision.actionable:
            self.assertLessEqual(decision.score, 6)
            self.assertIn("not a probability", decision.reason)


class TestPrototypes(unittest.TestCase):
    def test_every_member_is_registered_and_callable(self):
        candles, candle, previous_close, features = last_bar(200)
        for name in (CORE,) + tuple(PROTOTYPES):
            self.assertIn(name, REGISTRY)
            decision = REGISTRY[name](candle, previous_close, features, 0.55, calibrated=False)
            self.assertIn(decision.side, ("BUY", "SELL", "NO SIGNAL"))
            self.assertTrue(decision.reason)

    def test_ml_reads_need_a_model(self):
        candles, candle, previous_close, features = last_bar(200)
        for name in ("ml_consensus", "orbix_cadence"):
            decision = REGISTRY[name](candle, previous_close, features, None)
            self.assertEqual(decision.side, "NO SIGNAL")
            self.assertIn("not ready", decision.reason)

    def test_cadence_refuses_a_coin_flip(self):
        candles, candle, previous_close, features = last_bar(200)
        decision = REGISTRY["orbix_cadence"](candle, previous_close, features, 0.5, calibrated=False)
        self.assertEqual(decision.side, "NO SIGNAL")
        self.assertIn("below the", decision.reason)

    def test_breakout_needs_a_close_beyond_the_range(self):
        candles, candle, previous_close, features = last_bar(200)
        inside = Candle(
            timestamp=candle.timestamp,
            open=features["prior_high_20"] - 0.0002,
            high=features["prior_high_20"] - 0.0001,
            low=features["prior_low_20"] + 0.0001,
            close=features["prior_low_20"] + 0.0002,
            volume=1000.0,
        )
        decision = REGISTRY["breakout"](inside, previous_close, features)
        self.assertEqual(decision.side, "NO SIGNAL")
        self.assertIn("inside", decision.reason)


if __name__ == "__main__":
    unittest.main()
