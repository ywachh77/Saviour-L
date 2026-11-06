from datetime import datetime, timedelta, timezone
import unittest

from fusion.domain import Candle
from fusion.research_pattern import evaluate_trend_pullback


class TrendPullbackResearchPatternTests(unittest.TestCase):
    def candles(self, count=80):
        out = []
        price = 1.10000
        for i in range(count):
            # Positive drift with regular counter-moves keeps the trend while
            # avoiding the unrealistic RSI=100 produced by one-way toy data.
            change = (0.000020, 0.000010, -0.000020)[i % 3]
            op = price
            close = price + change
            out.append(Candle(
                datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=i),
                op, max(op, close) + 0.00004,
                min(op, close) - 0.00004, close,
            ))
            price = close
        alpha = 2.0 / 21.0
        fast = out[0].close
        for c in out[1:-1]:
            fast = alpha * c.close + (1.0 - alpha) * fast
        t = out[-1].timestamp
        out[-1] = Candle(t, fast - 0.00001, fast + 0.00005,
                         fast - 0.00004, fast + 0.00002)
        return out

    def test_insufficient_bars_abstains(self):
        result = evaluate_trend_pullback(self.candles(54), spread_pips=0.8)
        self.assertFalse(result.actionable_candidate)
        self.assertIn("55", result.reason)

    def test_spread_veto_is_explicit(self):
        result = evaluate_trend_pullback(self.candles(), spread_pips=1.21)
        self.assertFalse(result.actionable_candidate)
        self.assertIn("Spread", result.reason)

    def test_cost_veto_rejects_quiet_conditions(self):
        candles = self.candles()
        # Compress all price movement while maintaining valid positive prices.
        compressed = [Candle(c.timestamp, 1.1 + (c.open - 1.1) * 0.01,
                             1.1 + (c.high - 1.1) * 0.01,
                             1.1 + (c.low - 1.1) * 0.01,
                             1.1 + (c.close - 1.1) * 0.01) for c in candles]
        result = evaluate_trend_pullback(compressed, spread_pips=1.0)
        self.assertFalse(result.actionable_candidate)
        self.assertIn("Volatility", result.reason)

    def test_candidate_reports_cost_adjusted_break_even_not_accuracy(self):
        candles = self.candles()
        result = evaluate_trend_pullback(candles, spread_pips=0.2)
        self.assertTrue(result.actionable_candidate, result.reason)
        self.assertEqual(result.side, "BUY")
        self.assertGreater(result.stop_pips, 0)
        self.assertAlmostEqual(result.target_pips, result.stop_pips * 1.5)
        self.assertGreater(result.break_even_win_rate, 0)
        self.assertLess(result.break_even_win_rate, 100)

    def test_short_setup_is_the_directional_mirror(self):
        mirror = [Candle(c.timestamp, 2.2 - c.open, 2.2 - c.low,
                         2.2 - c.high, 2.2 - c.close) for c in self.candles()]
        result = evaluate_trend_pullback(mirror, spread_pips=0.2)
        self.assertTrue(result.actionable_candidate, result.reason)
        self.assertEqual(result.side, "SELL")

    def test_never_claims_a_win_probability(self):
        result = evaluate_trend_pullback(self.candles(), spread_pips=0.2)
        self.assertTrue(result.actionable_candidate)
        self.assertFalse(hasattr(result, "win_probability"))
        self.assertIn("candidate", result.reason.lower())


if __name__ == "__main__":
    unittest.main()
