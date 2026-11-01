"""Paper risk management tests: stops, targets, 1-2% sizing, circuit breaker."""

import unittest

from fusion import risk


class TestRiskPlan(unittest.TestCase):
    def test_buy_and_sell_levels_sit_on_the_correct_sides_of_entry(self):
        buy = risk.build_risk_plan(
            symbol="EUR/USD", side="BUY", entry_price=1.08500, atr_pips=4.0, balance_usd=10_000.0
        )
        self.assertLess(buy.stop_price, buy.entry_price)
        self.assertGreater(buy.target_price, buy.entry_price)
        self.assertAlmostEqual(buy.risk_amount_usd, 100.0)

        sell = risk.build_risk_plan(
            symbol="EUR/USD", side="SELL", entry_price=1.08500, atr_pips=4.0, balance_usd=10_000.0
        )
        self.assertGreater(sell.stop_price, sell.entry_price)
        self.assertLess(sell.target_price, sell.entry_price)

    def test_risk_fraction_caps_at_two_percent_by_default(self):
        capped = risk.build_risk_plan(
            symbol="EUR/USD", side="BUY", entry_price=1.08500, atr_pips=60.0,
            balance_usd=5_000.0, risk_fraction=0.10
        )
        self.assertAlmostEqual(capped.risk_fraction, 0.02)
        self.assertAlmostEqual(capped.risk_amount_usd, 100.0)
        # 60 * 1.5 = 90 pip stop + 1.2 pip spread = 91.2 pip loss per 1 EUR
        # loss_per_unit = 91.2 * 0.0001 = $0.00912
        # units = $100 / $0.00912 = 10964.91 EUR
        self.assertAlmostEqual(capped.position_units, 100.0 / (91.2 * 0.0001), places=2)

    def test_net_reward_to_risk_increases_with_atr(self):
        low_atr = risk.build_risk_plan(symbol="EUR/USD", side="BUY", entry_price=1.0850, atr_pips=1.95)
        high_atr = risk.build_risk_plan(symbol="EUR/USD", side="BUY", entry_price=1.0850, atr_pips=6.0)
        self.assertGreater(high_atr.net_reward_to_risk, low_atr.net_reward_to_risk)


class TestCircuitBreaker(unittest.TestCase):
    def test_halts_on_consecutive_losses(self):
        rows = [{"outcome": "LOSS", "net_pips": -2.0} for _ in range(4)]
        state = risk.evaluate_circuit_breaker(rows, max_loss_streak=4)
        self.assertTrue(state.halted)
        self.assertEqual(state.consecutive_losses, 4)

    def test_a_win_resets_the_leading_loss_streak(self):
        rows = [
            {"outcome": "LOSS", "net_pips": -2.0},
            {"outcome": "WIN", "net_pips": 4.0},
            {"outcome": "LOSS", "net_pips": -2.0},
            {"outcome": "LOSS", "net_pips": -2.0},
            {"outcome": "LOSS", "net_pips": -2.0},
        ]
        state = risk.evaluate_circuit_breaker(rows, max_loss_streak=4, stop_pips_per_r=10.0)
        self.assertFalse(state.halted)
        self.assertEqual(state.consecutive_losses, 1)

    def test_halts_on_daily_drawdown_limit(self):
        rows = [
            {"outcome": "LOSS", "net_pips": -16.0},
            {"outcome": "WIN", "net_pips": 2.0},
            {"outcome": "LOSS", "net_pips": -18.0},
        ]
        state = risk.evaluate_circuit_breaker(rows, stop_pips_per_r=10.0, max_daily_loss_r=3.0)
        self.assertTrue(state.halted)
        self.assertIn("breached", state.reason)


if __name__ == "__main__":
    unittest.main()
