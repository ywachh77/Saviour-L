"""Council power balance: the core decides, learning may veto, never create."""

import unittest
from contextlib import contextmanager

from fusion.council import (
    MIN_SEEN_PER_PROTOTYPE,
    PER_PROTOTYPE_WEIGHT_CAP,
    TOTAL_PROTOTYPE_MASS_CAP,
    VETO_MASS,
    CouncilWeights,
    run_council,
)
from fusion.features import compute_features
from fusion.strategies import PROTOTYPES

from .helpers import make_candles


def bar(count: int = 200, **kwargs):
    candles = make_candles(count, **kwargs)
    return candles, candles[-1], candles[-2].close, compute_features(candles)[-1]


@contextmanager
def forced_votes(*, core: str, prototype: str):
    """Pin every council member to a side so the vote arithmetic can be tested."""
    from fusion import strategies
    from fusion.domain import Decision

    original = dict(strategies.REGISTRY)
    strategies.REGISTRY[strategies.CORE] = lambda *a, **k: Decision(core, "forced core read", None, 6)
    for name in strategies.PROTOTYPES:
        strategies.REGISTRY[name] = lambda *a, _s=prototype, **k: Decision(_s, "forced prototype read", None, 2)
    try:
        yield
    finally:
        strategies.REGISTRY.clear()
        strategies.REGISTRY.update(original)


class TestCouncilWeights(unittest.TestCase):
    def test_prototypes_start_silent(self):
        weights = CouncilWeights()
        self.assertEqual(weights.active(), {})

    def test_learning_below_the_gate_grants_no_weight(self):
        weights = CouncilWeights()
        for _ in range(MIN_SEEN_PER_PROTOTYPE - 1):
            weights.observe({"trend_pullback": "BUY"}, "BUY")
        self.assertEqual(weights.weights["trend_pullback"], 0.0)

    def test_a_winning_record_earns_a_voice(self):
        weights = CouncilWeights()
        for _ in range(120):
            weights.observe({"trend_pullback": "BUY"}, "BUY")
        self.assertGreater(weights.weights["trend_pullback"], 0.0)
        self.assertLessEqual(weights.weights["trend_pullback"], PER_PROTOTYPE_WEIGHT_CAP)

    def test_a_losing_record_stays_silent(self):
        weights = CouncilWeights()
        for _ in range(120):
            weights.observe({"trend_pullback": "BUY"}, "SELL")
        self.assertEqual(weights.weights["trend_pullback"], 0.0)

    def test_total_mass_is_capped_below_the_core(self):
        weights = CouncilWeights()
        for _ in range(400):
            weights.observe({name: "BUY" for name in PROTOTYPES}, "BUY")
        self.assertLessEqual(sum(weights.weights.values()), TOTAL_PROTOTYPE_MASS_CAP + 1e-9)

    def test_unresolved_outcomes_teach_nothing(self):
        weights = CouncilWeights()
        for _ in range(200):
            weights.observe({"trend_pullback": "BUY"}, None)
        self.assertEqual(weights.weights["trend_pullback"], 0.0)
        self.assertEqual(weights.seen["trend_pullback"], 0)

    def test_round_trip(self):
        weights = CouncilWeights()
        for _ in range(80):
            weights.observe({"breakout": "SELL"}, "SELL")
        restored = CouncilWeights.from_dict(weights.to_dict())
        self.assertEqual(restored.weights, weights.weights)


class TestRunCouncil(unittest.TestCase):
    def test_core_decides_alone_when_nothing_is_learned(self):
        candles, candle, previous_close, features = bar(220, volatility=0.00025, drift=0.00005, seed=31)
        result = run_council(candle, previous_close, features, 0.55, weights=CouncilWeights(), regime="TREND")
        if result.actionable:
            self.assertIn("No learned prototype is active yet.", result.reason)
        for vote in result.votes:
            if vote.name != "orbix_confluence":
                self.assertFalse(vote.heard)

    def test_strong_learned_opposition_vetoes_the_core(self):
        candles, candle, previous_close, features = bar(220, volatility=0.00025, drift=0.00005, seed=31)
        weights = CouncilWeights()
        # Teach every prototype that the opposite side wins.
        for _ in range(400):
            weights.observe({name: "SELL" for name in PROTOTYPES}, "SELL")

        with forced_votes(core="BUY", prototype="SELL"):
            result = run_council(candle, previous_close, features, 0.55, weights=weights, regime="TREND")

        self.assertFalse(result.actionable)
        self.assertTrue(result.vetoed)
        self.assertIn("Council veto", result.reason)
        # The veto is a block, never a reversal: the core's side is not replaced.
        self.assertNotEqual(result.side, "SELL")

    def test_a_single_learned_prototype_cannot_veto_alone(self):
        self.assertGreater(VETO_MASS, PER_PROTOTYPE_WEIGHT_CAP)

    def test_regime_silences_out_of_family_prototypes(self):
        candles, candle, previous_close, features = bar(200)
        weights = CouncilWeights()
        for _ in range(200):
            weights.observe({name: "BUY" for name in PROTOTYPES}, "BUY")
        volatile = run_council(candle, previous_close, features, 0.55, weights=weights, regime="VOLATILE")
        heard = {vote.name for vote in volatile.votes if vote.heard}
        self.assertNotIn("range_reversal", heard)


if __name__ == "__main__":
    unittest.main()
