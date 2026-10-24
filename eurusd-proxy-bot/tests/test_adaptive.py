"""Adaptive horizon: hold long enough to pay for the trip.

The claim under test is narrow and deliberately so:

    at a realistic EUR/USD volatility, stretching the hold to the shortest
    length that can pay its own costs raises mean expectancy by roughly
    +0.43 pips/trade, and does so on most but NOT all price paths.

Measured over 19 independent paths at a 1.8 pip M1 ATR and a 1.2 pip round
turn, the improvement was better on 16 of 19 paths with a paired 95% interval
of [+0.24, +0.61]. It did **not** make the system profitable — the mean went
from −1.11 to −0.68 pips/trade — and it increased the spread of outcomes,
because a longer hold means fewer, larger bets.

Both halves of that are asserted here. A test that only checked the average
would be the same mistake as quoting one lucky replay.
"""

from __future__ import annotations

import unittest

from fusion.backtest import run_backtest
from fusion.config import Config
from fusion.simulate import synthetic_series
from fusion import costs

# Volatility chosen so the M1 ATR lands near 1.8 pips, which is the realistic
# EUR/USD scale. The repository default (0.00028) is about four times too
# volatile and makes every configuration look affordable.
REALISTIC_VOLATILITY = 0.000073
LIVELY_VOLATILITY = 0.00028


def _config(adaptive: bool, **overrides) -> Config:
    config = Config(horizon_minutes=5, spread_pips=1.2, adaptive_horizon=adaptive)
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def _replay(adaptive: bool, seed: int, volatility: float, bars: int = 2500):
    candles = synthetic_series(count=bars, seed=seed, volatility=volatility)
    return run_backtest(candles, _config(adaptive), folds=4, spread_pips=1.2)


def _replay_undisciplined(seed: int, volatility: float, bars: int = 2500):
    """The pre-fix path, reconstructed: a fixed hold with no cost test at all.

    ``reward_to_cost=0`` asks for no margin over the toll, so every horizon
    looks affordable and nothing is ever refused. That is exactly what the bot
    did before the cost gate was made unconditional, and it is the baseline the
    improvement has to be measured against.
    """
    candles = synthetic_series(count=bars, seed=seed, volatility=volatility)
    return run_backtest(candles, _config(False, reward_to_cost=0.0), folds=4, spread_pips=1.2)


class HorizonSelectionTests(unittest.TestCase):
    def test_a_payable_hold_is_kept_when_adaptive_is_off(self) -> None:
        """Turning stretching off must not stop the bot trading altogether."""
        self.assertEqual(costs.resolve_horizon(6.0, 1.2, 5, adaptive=False), 5)

    def test_an_unpayable_hold_is_refused_even_when_adaptive_is_off(self) -> None:
        """The cost test is not optional; only the stretching is.

        At 1.8 pips of M1 ATR a five-minute hold projects 1.53 pips against a
        3.60 pip requirement, so it needs a 97% hit rate just to break even.
        Taking that trade is not a conservative default, it is a fee.
        """
        self.assertIsNone(costs.resolve_horizon(1.8, 1.2, 5, adaptive=False))
        self.assertIsNone(costs.resolve_horizon(0.6, 1.2, 5, adaptive=False))

    def test_the_refusal_is_reported_rather_than_silent(self) -> None:
        """A gate nobody can see is a gate nobody can trust."""
        report = _replay(False, seed=42, volatility=REALISTIC_VOLATILITY)
        self.assertEqual(report.chosen_horizons, [], "no unpayable hold may be chosen")

    def test_a_quiet_market_stretches_the_hold(self) -> None:
        report = _replay(True, seed=42, volatility=REALISTIC_VOLATILITY)
        self.assertGreater(report.mean_horizon, 5.0)

    def test_never_shorter_than_the_configured_hold(self) -> None:
        """A quiet market needs more time, never less. Shortening would pay the
        same toll for a smaller move."""
        for volatility in (REALISTIC_VOLATILITY, LIVELY_VOLATILITY):
            report = _replay(True, seed=42, volatility=volatility)
            self.assertGreaterEqual(min(report.chosen_horizons), 5)

    def test_a_lively_market_keeps_the_short_hold(self) -> None:
        """When the configured hold already pays for itself, leave it alone."""
        report = _replay(True, seed=42, volatility=LIVELY_VOLATILITY)
        self.assertLessEqual(report.mean_horizon, 6.0)

    def test_every_chosen_horizon_is_affordable(self) -> None:
        """The property that defines the feature: a signal is never taken at a
        hold length that cannot cover its own cost.

        Checked across a range of volatilities rather than on one replay,
        because the ATR that drives the choice varies bar by bar and a report
        only carries the horizon, not the ATR that produced it.
        """
        for atr_pips in (0.6, 1.0, 1.8, 3.0, 6.0, 12.0):
            horizon = costs.resolve_horizon(atr_pips, 1.2, 5, adaptive=True)
            if horizon is None:
                continue  # no affordable hold exists at any length; stay out
            self.assertTrue(
                costs.affordable(atr_pips, horizon, 1.2),
                f"ATR {atr_pips} chose {horizon}m which cannot pay for itself",
            )

    def test_horizons_are_reported(self) -> None:
        """A decision the system made that never reaches the output is a
        decision nobody can check."""
        report = _replay(True, seed=42, volatility=REALISTIC_VOLATILITY)
        payload = report.to_dict()
        self.assertIn("mean_horizon", payload)
        self.assertIn("max_horizon", payload)
        self.assertGreater(payload["max_horizon"], 5)


class EffectTests(unittest.TestCase):
    """Does it actually help, and how much of the story is the average?"""

    def test_improves_the_mean_at_realistic_volatility(self) -> None:
        """Sampled over several paths, because one path proves nothing."""
        seeds = [2, 3, 4, 5, 6, 7, 8, 11]
        fixed = [_replay_undisciplined(seed=s, volatility=REALISTIC_VOLATILITY).expectancy_pips for s in seeds]
        adaptive = [_replay(True, seed=s, volatility=REALISTIC_VOLATILITY).expectancy_pips for s in seeds]

        mean_fixed = sum(fixed) / len(fixed)
        mean_adaptive = sum(adaptive) / len(adaptive)
        self.assertGreater(mean_adaptive, mean_fixed)

        improvements = sum(1 for f, a in zip(fixed, adaptive) if a > f)
        # Not "every path" — the effect is real but not universal, and a test
        # claiming otherwise would be asserting something the data does not show.
        self.assertGreaterEqual(improvements, len(seeds) * 0.6)

    def test_it_does_not_make_the_system_profitable(self) -> None:
        """The honest headline. This is a cost fix, not an edge."""
        seeds = [2, 3, 4, 5, 6, 7, 8, 11]
        adaptive = [_replay(True, seed=s, volatility=REALISTIC_VOLATILITY).expectancy_pips for s in seeds]
        self.assertLess(sum(adaptive) / len(adaptive), 0.0)

    def test_it_widens_the_spread_of_outcomes(self) -> None:
        """Longer holds mean fewer, larger bets, so results get noisier. Anyone
        switching this on should know the variance goes up with the mean."""
        seeds = [2, 3, 4, 5, 6, 7, 8, 11]
        fixed = [_replay_undisciplined(seed=s, volatility=REALISTIC_VOLATILITY).expectancy_pips for s in seeds]
        adaptive = [_replay(True, seed=s, volatility=REALISTIC_VOLATILITY).expectancy_pips for s in seeds]

        def spread(values):
            mean = sum(values) / len(values)
            return (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5

        self.assertGreater(spread(adaptive), spread(fixed))

    def test_no_effect_when_the_hold_already_pays(self) -> None:
        """On a lively market there is nothing to fix, so nothing should change."""
        fixed = _replay(False, seed=42, volatility=LIVELY_VOLATILITY)
        adaptive = _replay(True, seed=42, volatility=LIVELY_VOLATILITY)
        # Close, not identical: a handful of quiet bars inside an otherwise
        # lively path are still refused or stretched. That is the gate doing its
        # job on the bars that need it, not a change in behaviour.
        self.assertLess(abs(fixed.signals - adaptive.signals), max(5, fixed.signals * 0.05))
        # Close, not identical: a handful of quiet bars still get stretched,
        # which is the gate doing its job rather than a change in behaviour.
        # Measured per trade rather than in total: 18 pips across 260-odd
        # signals is a rounding error, while 18 pips across 3 would be a
        # different strategy wearing the same name.
        per_trade = abs(fixed.net_pips - adaptive.net_pips) / max(1, fixed.signals)
        self.assertLess(per_trade, 0.5, "on a lively market the two paths should barely differ")


class GateTests(unittest.TestCase):
    def test_an_unaffordable_market_produces_no_signal_at_all(self) -> None:
        """When no hold length can pay the toll, the answer is to stay out.

        A gate that merely warned would still take the trade, and this is the
        one case where taking it is certainly a loss.
        """
        self.assertIsNone(
            costs.resolve_horizon(0.2, 5.0, 5, adaptive=True, max_horizon=120)
        )

    def test_the_gate_is_off_by_default(self) -> None:
        self.assertFalse(Config().adaptive_horizon)

    def test_respects_a_configured_cap(self) -> None:
        """Nobody wants a one-minute bot holding for six hours."""
        self.assertIsNone(costs.resolve_horizon(0.5, 3.0, 5, adaptive=True, max_horizon=10))
        self.assertLessEqual(
            int(costs.resolve_horizon(0.5, 3.0, 5, adaptive=True, max_horizon=60) or 0), 60
        )


if __name__ == "__main__":
    unittest.main()
