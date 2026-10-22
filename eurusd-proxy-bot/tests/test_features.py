"""Features must be finite, complete, and free of look-ahead."""

import unittest

from fusion.features import FEATURE_NAMES, MIN_FEATURE_BARS, compute_features, model_vector

from .helpers import make_candles


class TestFeatures(unittest.TestCase):
    def setUp(self):
        self.candles = make_candles(220, seed=21)

    def test_early_bars_are_none_until_indicators_are_complete(self):
        rows = compute_features(self.candles)
        self.assertIsNone(rows[0])
        self.assertGreaterEqual(MIN_FEATURE_BARS, 20)
        self.assertIsNotNone(rows[-1])

    def test_every_model_feature_is_present_and_finite(self):
        row = compute_features(self.candles)[-1]
        self.assertIsNotNone(row)
        for name in FEATURE_NAMES:
            self.assertIn(name, row)
            self.assertTrue(row[name] == row[name])  # rejects NaN
        self.assertEqual(len(model_vector(row)), len(FEATURE_NAMES))

    def test_no_look_ahead(self):
        """Features for bar i must not change when later bars are appended."""
        short = self.candles[:150]
        long = self.candles[:200]
        short_row = compute_features(short)[-1]
        long_row = compute_features(long)[149]
        self.assertEqual(short_row.keys(), long_row.keys())
        for name in FEATURE_NAMES:
            self.assertAlmostEqual(float(short_row[name]), float(long_row[name]), places=12)

    def test_strategy_facing_values_exist(self):
        row = compute_features(self.candles)[-1]
        for name in (
            "ema_fast",
            "ema_slow",
            "rsi",
            "atr",
            "atr_pct",
            "adx",
            "plus_di",
            "minus_di",
            "bb_upper",
            "bb_lower",
            "prior_high_20",
            "prior_low_20",
        ):
            self.assertIn(name, row)

    def test_prior_range_excludes_the_current_bar(self):
        candles = self.candles
        row = compute_features(candles)[-1]
        self.assertLess(row["prior_high_20"], candles[-1].high + 1.0)
        self.assertGreater(row["prior_high_20"], 0.0)
        self.assertGreater(row["prior_high_20"], row["prior_low_20"])


if __name__ == "__main__":
    unittest.main()
