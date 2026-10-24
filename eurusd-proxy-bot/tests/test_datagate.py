"""The data gate is the first refusal point, and it refuses a lot."""

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fusion import charter, datagate
from fusion.datagate import DEGRADED, OK, count_gaps, evaluate, modelled_spread_pips

from .helpers import make_candles


class TestModelledSpread(unittest.TestCase):
    def test_the_model_honours_the_configured_spread(self):
        """It used to hardcode 1.2, so SPREAD_PIPS had no effect on the gate."""
        stamp = datetime(2026, 1, 5, 3, 0, tzinfo=timezone.utc)  # off-hours
        at_default = datagate.modelled_spread_pips(stamp, base=charter.DEFAULT_SPREAD_PIPS)
        at_three = datagate.modelled_spread_pips(stamp, base=3.0)
        self.assertAlmostEqual(at_three / at_default, 2.5, places=1)

    def test_off_hours_is_refused_by_a_real_margin_not_a_coincidence(self):
        """1.2 x 2.1 = 2.52 against a 2.50 cap cleared by 0.02 pips.

        That is not a decision, it is an accident: nudge SPREAD_PIPS and the
        behaviour inverts with nothing saying why. The cap is now a multiple of
        the assumed spread, so the two move together.
        """
        stamp = datetime(2026, 1, 5, 3, 0, tzinfo=timezone.utc)
        spread = datagate.modelled_spread_pips(stamp, base=charter.DEFAULT_SPREAD_PIPS)
        cap = charter.DEFAULT_SPREAD_PIPS * charter.MAX_SPREAD_MULTIPLE
        self.assertGreater(spread, cap)
        self.assertGreater(spread - cap, 0.1, "margin is too thin to be intentional")

        # Active sessions must still clear it, with room to spare.
        for hour in (9, 14, 19):
            active = datagate.modelled_spread_pips(
                datetime(2026, 1, 5, hour, 0, tzinfo=timezone.utc),
                base=charter.DEFAULT_SPREAD_PIPS,
            )
            self.assertLess(active, cap, f"hour {hour} should be tradeable")

    def test_the_cap_follows_the_assumed_spread(self):
        from fusion.config import load_config

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text("SPREAD_PIPS=2.0\n", encoding="utf-8")
            config = load_config(path)
            self.assertAlmostEqual(config.spread_pips, 2.0)
            self.assertAlmostEqual(
                config.max_spread_pips, 2.0 * charter.MAX_SPREAD_MULTIPLE, places=6
            )

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
