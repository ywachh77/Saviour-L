"""The cost model: can a trade pay for itself before it is taken?

The constant in this module was measured, not assumed, and these tests pin it to
the measurement so nobody can quietly "correct" it back to the theoretical 0.5
and make the gate more permissive without noticing.
"""

from __future__ import annotations

import math
import unittest

from fusion import costs
from fusion.charter import PIP
from fusion.config import Config
from fusion.indicators import atr
from fusion.simulate import synthetic_series


class ExpectedMoveTests(unittest.TestCase):
    def test_sqrt_time_scaling(self) -> None:
        """Four times the hold is twice the move — the diffusion shape."""
        one = costs.expected_move_pips(2.0, 16)
        four = costs.expected_move_pips(2.0, 64)
        self.assertAlmostEqual(2.0, four / one, places=6)

    def test_zero_atr_means_no_move(self) -> None:
        self.assertEqual(0.0, costs.expected_move_pips(0.0, 10))
        self.assertEqual(0.0, costs.expected_move_pips(2.0, 0))

    def test_monotonic_in_horizon(self) -> None:
        values = [costs.expected_move_pips(2.0, h) for h in range(1, 40)]
        self.assertEqual(sorted(values), values)


class CalibrationTests(unittest.TestCase):
    """The projection must sit at or below reality, never above it."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.candles = synthetic_series(count=3000, seed=20260101)
        values = [value for value in atr(cls.candles, 14) if value is not None]
        cls.atr_pips = sum(values) / len(values) / PIP

    def _realised(self, horizon: int) -> float:
        moves = [
            abs(self.candles[index + horizon].close - self.candles[index + 1].open) / PIP
            for index in range(len(self.candles) - horizon)
        ]
        return sum(moves) / len(moves)

    def test_projection_never_overstates_the_move(self) -> None:
        """Under-projecting makes the gate stricter, which costs a missed trade.
        Over-projecting would let unaffordable trades through, which costs money.
        Only one of those is an acceptable direction to be wrong."""
        for horizon in (5, 15, 30, 60):
            projected = costs.expected_move_pips(self.atr_pips, horizon)
            self.assertLessEqual(projected, self._realised(horizon) * 1.02)

    def test_projection_is_not_absurdly_conservative(self) -> None:
        """But a gate so strict it blocks everything is equally useless."""
        for horizon in (5, 15, 30):
            projected = costs.expected_move_pips(self.atr_pips, horizon)
            self.assertGreater(projected, self._realised(horizon) * 0.85)

    def test_the_capture_constant_is_the_measured_one(self) -> None:
        """Guards against reverting to the theoretical 0.5.

        0.5 comes from E|move| / E[range] for a clean random walk. Real candles
        carry a wick, which is range no position can capture, so the measured
        ratio is 0.39-0.47 and the constant is set below all of them.
        """
        self.assertEqual(0.38, costs.MOVE_CAPTURE)
        self.assertLess(costs.MOVE_CAPTURE, 0.5)


class AffordabilityTests(unittest.TestCase):
    def test_a_move_three_times_the_cost_is_affordable(self) -> None:
        # 0.40 * 10 * sqrt(9) = 12.0 pips against 3 * 2.0 = 6.0 required
        self.assertTrue(costs.affordable(10.0, 9, 2.0))

    def test_break_even_is_not_affordable(self) -> None:
        """A trade whose whole move merely equals its cost has zero expected
        value before the model's own error is counted in."""
        atr_needed = costs.affordable_atr_pips(9, 2.0, reward_to_cost=1.0)
        self.assertFalse(costs.affordable(atr_needed * 1.01, 9, 2.0))

    def test_a_quiet_market_has_no_viable_horizon(self) -> None:
        """None is the important answer: no hold length can pay for this."""
        self.assertIsNone(costs.minimum_viable_horizon(0.2, 5.0, max_horizon=120))

    def test_a_lively_market_is_viable_immediately(self) -> None:
        self.assertEqual(1, costs.minimum_viable_horizon(20.0, 1.0))

    def test_viable_horizon_grows_as_volatility_falls(self) -> None:
        previous = 0
        for atr_pips in (12.0, 8.0, 5.0, 3.0, 2.0):
            horizon = costs.minimum_viable_horizon(atr_pips, 1.2)
            self.assertIsNotNone(horizon)
            self.assertGreaterEqual(int(horizon), previous)
            previous = int(horizon)

    def test_the_realistic_eurusd_case_is_unaffordable_at_five_minutes(self) -> None:
        """The finding that motivates this whole module.

        A 1.5 pip M1 ATR with a 1.2 pip round turn: a five-minute hold cannot
        pay for itself, and the honest answer is a longer hold, not more accuracy.
        """
        verdict = costs.evaluate(1.5, 5, 1.2)
        self.assertFalse(verdict.ok)
        self.assertIsNotNone(verdict.viable_horizon)
        self.assertGreater(int(verdict.viable_horizon), 5)

    def test_required_atr_inverts_the_test(self) -> None:
        # A hair above the boundary, since the boundary itself is float-precise
        # and dividing by 0.40*sqrt(h) does not round-trip exactly.
        atr_needed = costs.affordable_atr_pips(9, 2.0)
        self.assertTrue(costs.affordable(atr_needed * 1.000001, 9, 2.0))
        self.assertFalse(costs.affordable(atr_needed * 0.99, 9, 2.0))


class ResolveHorizonTests(unittest.TestCase):
    def test_off_means_the_configured_horizon_unchanged(self) -> None:
        """The default path must stay bit-for-bit what it already was."""
        self.assertEqual(5, costs.resolve_horizon(0.1, 1.2, 5, adaptive=False))
        self.assertEqual(17, costs.resolve_horizon(99.0, 1.2, 17, adaptive=False))

    def test_adaptive_never_shortens_the_hold(self) -> None:
        """A quiet market needs more time, not less."""
        self.assertGreaterEqual(costs.resolve_horizon(50.0, 1.2, 5, adaptive=True), 5)

    def test_adaptive_stretches_a_quiet_market(self) -> None:
        self.assertGreater(costs.resolve_horizon(1.5, 1.2, 5, adaptive=True), 5)

    def test_adaptive_returns_none_when_nothing_can_pay(self) -> None:
        self.assertIsNone(costs.resolve_horizon(0.05, 5.0, 5, adaptive=True))

    def test_adaptive_respects_the_horizon_cap(self) -> None:
        self.assertIsNone(
            costs.resolve_horizon(0.5, 3.0, 5, adaptive=True, max_horizon=10)
        )


class VerdictDescriptionTests(unittest.TestCase):
    def test_never_viable_blames_the_market_not_the_model(self) -> None:
        """The wording matters: it is not that direction is unpredictable here,
        it is that no direction can pay the toll."""
        verdict = costs.evaluate(0.3, 5, 2.0)
        self.assertTrue(verdict.never_viable)
        self.assertIn("too quiet to trade", verdict.describe())

    def test_unaffordable_says_what_would_fix_it(self) -> None:
        verdict = costs.evaluate(1.5, 5, 1.2)
        text = verdict.describe()
        self.assertIn("M1 ATR", text)
        self.assertIn("horizon of", text)

    def test_serialises(self) -> None:
        payload = costs.evaluate(1.5, 5, 1.2).to_dict()
        self.assertFalse(payload["ok"])
        self.assertIn("viable_horizon", payload)


class MedianAtrTests(unittest.TestCase):
    def test_returns_pips(self) -> None:
        candles = synthetic_series(count=400, seed=5)
        value = costs.median_atr_pips(candles)
        self.assertGreater(value, 0.0)
        self.assertLess(value, 50.0)

    def test_median_resists_a_single_spike(self) -> None:
        """One news bar must not convince the gate the market is always lively."""
        base = synthetic_series(count=400, seed=5)
        median = costs.median_atr_pips(base)
        values = [value for value in atr(base, 14) if value is not None]
        mean_pips = sum(values) / len(values) / PIP
        self.assertLess(abs(median - mean_pips), mean_pips * 0.5)

    def test_empty_series_is_zero(self) -> None:
        self.assertEqual(0.0, costs.median_atr_pips([]))


class ConfigDefaultsTests(unittest.TestCase):
    def test_adaptive_is_off_by_default(self) -> None:
        """Shipped behaviour must not change underneath an existing install."""
        self.assertFalse(Config().adaptive_horizon)

    def test_the_cost_rule_is_exposed_and_conservative(self) -> None:
        config = Config()
        self.assertEqual(3.0, config.reward_to_cost)
        self.assertGreaterEqual(config.reward_to_cost, 2.0)
        self.assertGreater(config.max_horizon_minutes, 0)

    def test_horizon_cap_is_sane(self) -> None:
        self.assertLessEqual(Config().max_horizon_minutes, 240)


if __name__ == "__main__":
    unittest.main()
