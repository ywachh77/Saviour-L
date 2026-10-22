"""The scalping members: cost-aware, regime-gated, and silent until proven.

Scalping is the family most likely to look brilliant on paper and lose money
live, because its whole edge can be smaller than the spread. These tests hold
the three guards in place: a scalp that cannot pay for itself stays quiet, a
scalp only speaks in the regime it was written for, and no scalp can create a
signal until learning has seen enough real outcomes to give it a voice.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timezone

from fusion import charter, scalping, strategies
from fusion.council import MIN_SEEN_PER_PROTOTYPE, CouncilWeights, run_council
from fusion.domain import Candle, Decision
from fusion.features import FEATURE_NAMES, FEATURE_VERSION, compute_features
from fusion.model import OnlineLogisticModel
from fusion.regime import RANGE, TREND, UNKNOWN, VOLATILE, regime_allows
from fusion.simulate import synthetic_series

SCALPS = scalping.SCALPING_PROTOTYPES


def _features(**overrides):
    """A genuine feature row with chosen values overridden."""
    rows = compute_features(synthetic_series(200, seed=4242))
    row = dict(next(item for item in reversed(rows) if item is not None))
    row.update(overrides)
    return row


def _candle(**overrides) -> Candle:
    base = {
        "timestamp": datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc),
        "open": 1.1000,
        "high": 1.1008,
        "low": 1.0992,
        "close": 1.1005,
        "volume": 1000.0,
    }
    base.update(overrides)
    return Candle(**base)


def _run(name: str, candle: Candle, features: dict, spread: float = charter.DEFAULT_SPREAD_PIPS) -> Decision:
    return strategies.REGISTRY[name](candle, candle.open, features, None, spread_pips=spread)


class ScalpCostGuardTests(unittest.TestCase):
    """A scalp that cannot pay for its own entry is not a scalp."""

    def test_a_quiet_bar_suppresses_the_whole_family(self) -> None:
        quiet = _features(atr_pips=1.0)
        for name in SCALPS:
            with self.subTest(member=name):
                decision = _run(name, _candle(), quiet)
                self.assertEqual("NO SIGNAL", decision.side)
                self.assertIn("cost", decision.reason.lower())

    def test_the_guard_threshold_is_where_the_constants_say_it_is(self) -> None:
        needed = charter.SCALP_MIN_REWARD_TO_COST * charter.DEFAULT_SPREAD_PIPS
        boundary = needed / charter.SCALP_ATR_CAPTURE
        self.assertFalse(charter.scalp_cost_ok(boundary - 0.01))
        self.assertTrue(charter.scalp_cost_ok(boundary + 0.01))

    def test_a_wide_spread_silences_every_scalp(self) -> None:
        live = _features(atr_pips=8.0)
        for name in SCALPS:
            with self.subTest(member=name):
                decision = _run(name, _candle(), live, spread=6.0)
                self.assertEqual("NO SIGNAL", decision.side)

    def test_a_normal_spread_lets_the_patterns_decide(self) -> None:
        """When the bar can pay for itself the guard steps aside.

        Whether anything fires is then up to the pattern, not the cost filter —
        so the invariant is that no member gives cost as its reason, not that
        some member fires on this particular bar.
        """
        live = _features(atr_pips=8.0)
        for name in SCALPS:
            with self.subTest(member=name):
                self.assertNotIn("cost", _run(name, _candle(), live).reason.lower())


class ScalpPatternTests(unittest.TestCase):
    """Each member fires on its own pattern and stays quiet otherwise."""

    def test_impulse_needs_a_body_that_closes_on_its_extreme(self) -> None:
        bull = _features(impulse=0.8, body_ratio=0.70, close_position=0.90, atr_pips=8.0)
        self.assertEqual("BUY", _run("scalp_impulse", _candle(), bull).side)

        bear = _features(impulse=-0.8, body_ratio=0.70, close_position=0.10, atr_pips=8.0)
        self.assertEqual("SELL", _run("scalp_impulse", _candle(), bear).side)

        # A long body that closes back in the middle is the end of a move.
        faded = _features(impulse=0.8, body_ratio=0.70, close_position=0.50, atr_pips=8.0)
        self.assertEqual("NO SIGNAL", _run("scalp_impulse", _candle(), faded).side)

    def test_reversion_waits_for_the_extreme_to_turn(self) -> None:
        oversold = _features(cci=-150.0, stoch_k=15.0, stoch_d=10.0, atr_pips=8.0)
        self.assertEqual("BUY", _run("scalp_reversion", _candle(), oversold).side)

        overbought = _features(cci=150.0, stoch_k=85.0, stoch_d=90.0, atr_pips=8.0)
        self.assertEqual("SELL", _run("scalp_reversion", _candle(), overbought).side)

        # Stretched but still falling: not a reversal yet.
        still_falling = _features(cci=-150.0, stoch_k=15.0, stoch_d=25.0, atr_pips=8.0)
        self.assertEqual("NO SIGNAL", _run("scalp_reversion", _candle(), still_falling).side)

    def test_micro_breakout_needs_a_tight_box(self) -> None:
        base = dict(range_compression=0.40, impulse=0.50, atr_pips=8.0)
        up = _features(prior_high_5=1.0995, **base)
        self.assertEqual("BUY", _run("scalp_micro_breakout", _candle(close=1.1005), up).side)

        down = _features(prior_low_5=1.1015, impulse=-0.50, range_compression=0.40, atr_pips=8.0)
        self.assertEqual("SELL", _run("scalp_micro_breakout", _candle(close=1.1005), down).side)

        # Same breakout out of a loose range is just noise.
        loose = _features(prior_high_5=1.0995, **dict(base, range_compression=0.90))
        self.assertEqual("NO SIGNAL", _run("scalp_micro_breakout", _candle(close=1.1005), loose).side)

    def test_stoch_cross_must_agree_with_the_ema_trend(self) -> None:
        with_trend = _features(
            stoch_k=60.0, stoch_d=50.0, stoch_k_previous=45.0, stoch_d_previous=50.0,
            ema_fast=1.1001, ema_slow=1.1000, atr_pips=8.0,
        )
        self.assertEqual("BUY", _run("scalp_stoch_cross", _candle(), with_trend).side)

        against = _features(
            stoch_k=60.0, stoch_d=50.0, stoch_k_previous=45.0, stoch_d_previous=50.0,
            ema_fast=1.0999, ema_slow=1.1000, atr_pips=8.0,
        )
        self.assertEqual("NO SIGNAL", _run("scalp_stoch_cross", _candle(), against).side)

    def test_vwap_reversion_waits_for_price_to_close_back(self) -> None:
        below = _features(vwap_dev=-0.30, atr_pips=8.0)
        self.assertEqual("BUY", _run("scalp_vwap_reversion", _candle(open=1.1000, close=1.1005), below).side)

        # Stretched but still closing away from value: the knife is still falling.
        still_away = _features(vwap_dev=-0.30, atr_pips=8.0)
        self.assertEqual(
            "NO SIGNAL", _run("scalp_vwap_reversion", _candle(open=1.1005, close=1.1000), still_away).side
        )

        above = _features(vwap_dev=0.30, atr_pips=8.0)
        self.assertEqual("SELL", _run("scalp_vwap_reversion", _candle(open=1.1005, close=1.1000), above).side)

    def test_pivot_fade_needs_the_close_back_inside(self) -> None:
        swept_low = _features(prior_low_5=1.0995, close_position=0.90, atr_pips=8.0)
        candle = _candle(open=1.1000, high=1.1008, low=1.0990, close=1.1005)
        self.assertEqual("BUY", _run("scalp_pivot_fade", candle, swept_low).side)

        swept_high = _features(prior_high_5=1.1010, close_position=0.10, atr_pips=8.0)
        candle = _candle(open=1.1005, high=1.1015, low=1.0994, close=1.1000)
        self.assertEqual("SELL", _run("scalp_pivot_fade", candle, swept_high).side)

        # Broke through and stayed through: that is a breakout, not a fade.
        accepted = _features(prior_high_5=1.0995, close_position=0.90, atr_pips=8.0)
        candle = _candle(open=1.1000, high=1.1012, low=1.0998, close=1.1010)
        self.assertEqual("NO SIGNAL", _run("scalp_pivot_fade", candle, accepted).side)


class ScalpRegimeTests(unittest.TestCase):
    """Continuation scalps need a trend; reversion scalps need a range."""

    def test_continuation_scalps_sit_out_a_range(self) -> None:
        for name in ("scalp_impulse", "scalp_micro_breakout"):
            with self.subTest(member=name):
                self.assertFalse(regime_allows(RANGE, name))

    def test_reversion_scalps_sit_out_a_trend_and_volatility(self) -> None:
        for name in ("scalp_reversion", "scalp_pivot_fade"):
            with self.subTest(member=name):
                self.assertFalse(regime_allows(TREND, name))
                self.assertFalse(regime_allows(VOLATILE, name))

    def test_no_scalp_speaks_before_the_regime_is_known(self) -> None:
        for name in SCALPS:
            with self.subTest(member=name):
                self.assertFalse(regime_allows(UNKNOWN, name))

    def test_each_family_is_heard_somewhere(self) -> None:
        for name in SCALPS:
            with self.subTest(member=name):
                heard = [regime for regime in (TREND, RANGE, VOLATILE) if regime_allows(regime, name)]
                self.assertTrue(heard, f"{name} is gated out of every regime and can never speak")

    def test_volatility_allows_riding_but_not_fading(self) -> None:
        self.assertTrue(regime_allows(VOLATILE, "scalp_impulse"))
        self.assertFalse(regime_allows(VOLATILE, "scalp_vwap_reversion"))


class ScalpAuthorityTests(unittest.TestCase):
    """New scalps are prototypes, so they can veto but never create."""

    def test_scalps_are_prototypes_and_not_the_core(self) -> None:
        self.assertNotIn(strategies.CORE, SCALPS)
        for name in SCALPS:
            with self.subTest(member=name):
                self.assertIn(name, strategies.PROTOTYPES)
                self.assertIn(name, strategies.REGISTRY)
                self.assertIn(name, strategies.MEMBERS)

    def test_every_scalp_starts_silent(self) -> None:
        weights = CouncilWeights()
        for name in SCALPS:
            with self.subTest(member=name):
                self.assertEqual(0.0, weights.weights.get(name, 0.0))

    def test_a_scalp_earns_no_voice_before_enough_real_outcomes(self) -> None:
        weights = CouncilWeights()
        for _ in range(MIN_SEEN_PER_PROTOTYPE - 1):
            weights.observe({"scalp_impulse": "BUY"}, "BUY")
        self.assertEqual(0.0, weights.weights["scalp_impulse"])

        weights.observe({"scalp_impulse": "BUY"}, "BUY")
        self.assertGreater(weights.weights["scalp_impulse"], 0.0)

    def test_a_scalp_that_keeps_losing_earns_no_weight(self) -> None:
        weights = CouncilWeights()
        for _ in range(120):
            weights.observe({"scalp_impulse": "BUY"}, "SELL")
        self.assertEqual(0.0, weights.weights["scalp_impulse"])

    def test_learned_scalps_can_veto_the_core(self) -> None:
        weights = CouncilWeights()
        for name in ("scalp_impulse", "scalp_micro_breakout"):
            weights.seen[name] = 100
            weights.net[name] = 42.0
        weights.recompute()
        self.assertGreater(weights.weights["scalp_impulse"], 0.0)
        self.assertGreater(weights.weights["scalp_micro_breakout"], 0.0)

        original = dict(strategies.REGISTRY)
        try:
            strategies.REGISTRY[strategies.CORE] = lambda *a, **k: Decision("BUY", "stub core", None, 6)
            for name in ("scalp_impulse", "scalp_micro_breakout"):
                strategies.REGISTRY[name] = lambda *a, _n=name, **k: Decision("SELL", f"stub {_n}", None, 2)
            result = run_council(
                _candle(), 1.0999, _features(atr_pips=8.0), None,
                weights=weights, regime=TREND, min_checks=1,
            )
        finally:
            strategies.REGISTRY.clear()
            strategies.REGISTRY.update(original)

        self.assertTrue(result.vetoed, "two learned scalps opposing the core should veto it")
        self.assertEqual("NO SIGNAL", result.side)

    def test_an_unlearned_scalp_cannot_veto(self) -> None:
        original = dict(strategies.REGISTRY)
        try:
            strategies.REGISTRY[strategies.CORE] = lambda *a, **k: Decision("BUY", "stub core", None, 6)
            for name in ("scalp_impulse", "scalp_micro_breakout"):
                strategies.REGISTRY[name] = lambda *a, _n=name, **k: Decision("SELL", f"stub {_n}", None, 2)
            result = run_council(
                _candle(), 1.0999, _features(atr_pips=8.0), None,
                weights=CouncilWeights(), regime=TREND, min_checks=1,
            )
        finally:
            strategies.REGISTRY.clear()
            strategies.REGISTRY.update(original)

        self.assertFalse(result.vetoed)
        self.assertEqual("BUY", result.side)


class ScalpFeatureTests(unittest.TestCase):
    """The model can only see a scalp if the inputs describe one."""

    def test_scalping_inputs_are_in_the_model_vector(self) -> None:
        expected = {
            "micro_range_5", "range_compression", "body_ratio", "impulse",
            "stoch_k_scaled", "stoch_d_scaled", "cci_scaled", "vwap_dev",
        }
        self.assertTrue(expected.issubset(set(FEATURE_NAMES)))

    def test_every_scalping_input_is_finite_on_a_real_series(self) -> None:
        rows = compute_features(synthetic_series(400, seed=99))
        usable = [row for row in rows if row is not None]
        self.assertTrue(usable)
        for row in usable:
            for name in ("micro_range_5", "range_compression", "body_ratio", "impulse",
                         "stoch_k_scaled", "stoch_d_scaled", "cci_scaled", "vwap_dev"):
                self.assertTrue(row[name] == row[name], f"{name} is NaN")

    def test_scalping_inputs_are_bounded(self) -> None:
        rows = [row for row in compute_features(synthetic_series(600, seed=7)) if row is not None]
        for row in rows:
            self.assertGreaterEqual(row["body_ratio"], 0.0)
            self.assertLessEqual(row["body_ratio"], 1.0)
            self.assertGreaterEqual(row["cci_scaled"], -1.0)
            self.assertLessEqual(row["cci_scaled"], 1.0)
            self.assertGreaterEqual(row["stoch_k_scaled"], -1.0)
            self.assertLessEqual(row["stoch_k_scaled"], 1.0)

    def test_micro_pivot_excludes_the_current_bar(self) -> None:
        """A breakout must be measured against a level that already existed."""
        candles = synthetic_series(200, seed=31)
        rows = compute_features(candles)
        for index, row in enumerate(rows):
            if row is None or index < 1:
                continue
            self.assertGreaterEqual(row["prior_high_5"], min(c.high for c in candles[index - 5 : index]))
            self.assertLessEqual(row["prior_low_5"], max(c.low for c in candles[index - 5 : index]))

    def test_the_version_bump_invalidates_models_trained_on_the_old_vector(self) -> None:
        self.assertTrue(OnlineLogisticModel().compatible)
        stale = OnlineLogisticModel(
            feature_names=tuple(name for name in FEATURE_NAMES if name not in {"impulse", "vwap_dev"}),
            feature_version=3,
        )
        self.assertFalse(stale.compatible)
        self.assertEqual(4, FEATURE_VERSION)


class ScalpIntegrationTests(unittest.TestCase):
    """On a real series every member must fire sometimes, and never always."""

    def setUp(self) -> None:
        self.candles = synthetic_series(1500, seed=20260101)
        self.rows = compute_features(self.candles)

    def _tally(self, spread: float = charter.DEFAULT_SPREAD_PIPS) -> dict[str, int]:
        counts = {name: 0 for name in SCALPS}
        for index in range(30, len(self.candles)):
            features = self.rows[index]
            if features is None:
                continue
            for name in SCALPS:
                decision = _run(name, self.candles[index], dict(features), spread=spread)
                if decision.actionable:
                    counts[name] += 1
        return counts

    def test_every_scalp_fires_and_none_fires_on_every_bar(self) -> None:
        counts = self._tally()
        bars = len(self.candles) - 30
        for name, fired in counts.items():
            with self.subTest(member=name):
                self.assertGreater(fired, 0, f"{name} never fired on {bars} bars")
                self.assertLess(fired / bars, 0.60, f"{name} fires on nearly every bar; it is not selective")

    def test_a_wide_spread_stops_all_of_them(self) -> None:
        counts = self._tally(spread=6.0)
        self.assertEqual(0, sum(counts.values()), f"scalps fired through a 6-pip spread: {counts}")


if __name__ == "__main__":
    unittest.main()
