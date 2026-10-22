"""The data gate is the first refusal point, and it refuses a lot."""

import unittest
from datetime import timedelta

from fusion import datagate
from fusion.datagate import DEGRADED, OK, count_gaps, evaluate, modelled_spread_pips

from .helpers import make_candles


class TestModelledSpread(unittest.TestCase):
    def test_spread_is_wider_outside_the_overlap(self):
        from datetime import datetime

        from fusion.timeutil import UTC

        overlap = datetime(2026, 1, 5, 14, 30, tzinfo=UTC)
        asian = datetime(2026, 1, 5, 2, 30, tzinfo=UTC)
        self.assertLess(modelled_spread_pips(overlap), modelled_spread_pips(asian))


class TestGaps(unittest.TestCase):
    def test_a_continuous_series_has_no_gaps(self):
        self.assertEqual(count_gaps(make_candles(60)), 0)

    def test_a_missing_bar_is_counted(self):
        candles = make_candles(60)
        holed = candles[:30] + candles[31:]
        self.assertEqual(count_gaps(holed), 1)


class TestEvaluate(unittest.TestCase):
    def setUp(self):
        self.candles = make_candles(200, seed=51)
        self.now = self.candles[-1].timestamp + timedelta(minutes=1)

    def test_healthy_data_passes(self):
        gate = evaluate(self.candles, now=self.now)
        self.assertTrue(gate.usable)
        self.assertEqual(gate.state, OK)

    def test_stale_data_is_blocked(self):
        stamp = self.candles[-1].timestamp + timedelta(minutes=30)
        gate = evaluate(self.candles, now=stamp)
        self.assertFalse(gate.usable)
        self.assertEqual(gate.quality, "STALE")
        self.assertIn("freshness limit", gate.reason)

    def test_a_wide_spread_is_blocked(self):
        gate = evaluate(
            self.candles, now=self.now, max_spread_pips=0.1, quote={"spread_pips": 4.0, "source": "provider"}
        )
        self.assertFalse(gate.usable)
        self.assertEqual(gate.quality, "WIDE_SPREAD")
        self.assertEqual(gate.spread_source, "provider")

    def test_modelled_spread_is_labelled_as_modelled(self):
        gate = evaluate(self.candles, now=self.now)
        self.assertEqual(gate.spread_source, "modelled")

    def test_too_few_candles_is_blocked(self):
        gate = evaluate(self.candles[:20], now=self.candles[19].timestamp + timedelta(minutes=1))
        self.assertFalse(gate.usable)
        self.assertEqual(gate.quality, "INSUFFICIENT")

    def test_a_gappy_series_is_degraded_not_silently_accepted(self):
        holed = self.candles[:40] + self.candles[50:90] + self.candles[100:140] + self.candles[155:]
        gate = evaluate(holed, now=holed[-1].timestamp + timedelta(minutes=1), max_gaps=2)
        self.assertEqual(gate.state, DEGRADED)
        self.assertIn("gaps", gate.reason)

    def test_gate_result_is_serialisable(self):
        payload = evaluate(self.candles, now=self.now).to_dict()
        self.assertIn("quality", payload)
        self.assertIn("spread_source", payload)


if __name__ == "__main__":
    unittest.main()
