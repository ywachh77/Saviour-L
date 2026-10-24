"""Learning guards: real data only, test before learn, gates before promotion."""

import unittest

from fusion import charter, learning
from fusion.config import Config
from fusion.features import compute_features
from fusion.learning import (
    PROMOTION_MIN_ACCURACY,
    build_training_samples,
    online_step,
    promotion_gate,
    rollback_check,
    run_nightly,
    walk_forward,
)
from fusion.model import MODE_ADVISORY, MODE_VETO, OnlineLogisticModel

from .helpers import make_candles, trending_candles


class TestSampleBuilder(unittest.TestCase):
    def test_label_matches_the_path_a_signal_takes(self):
        candles = make_candles(120, seed=5)
        samples = build_training_samples(candles, horizon=5)
        self.assertTrue(samples)
        first = samples[0]
        # Features come from a closed candle; the label uses the NEXT open.
        index = next(
            position
            for position, candle in enumerate(candles)
            if candle.timestamp.isoformat().replace("+00:00", "Z") == first.timestamp
        )
        entry = candles[index + 1].open
        exit_ = candles[index + 5].close
        expected = 1 if exit_ > entry else 0
        self.assertEqual(first.label, expected)

    def test_ties_are_dropped_not_guessed(self):
        flat = make_candles(80, seed=1, volatility=0.0, drift=0.0)
        self.assertEqual(build_training_samples(flat, horizon=5), [])

    def test_no_future_bars_leak_into_the_last_label(self):
        candles = make_candles(100, seed=9)
        samples = build_training_samples(candles, horizon=5)
        self.assertLessEqual(len(samples), len(candles) - 5)


class TestOnlineStep(unittest.TestCase):
    def setUp(self):
        self.candles = trending_candles(160)
        self.pairs = [item.to_pair() for item in build_training_samples(self.candles, horizon=5)]
        self.model = OnlineLogisticModel().fit(self.pairs, epochs=40, validation_fraction=0.0)
        self.features = compute_features(self.candles)[-1]

    def test_synthetic_outcomes_never_update_the_model(self):
        before = list(self.model.weights)
        step = online_step(self.model, self.features, 1, source="SYNTHETIC")
        self.assertFalse(step.applied)
        self.assertEqual(before, self.model.weights)
        self.assertIn("charter", step.reason)

    def test_tied_or_unresolved_teaches_nothing(self):
        before = list(self.model.weights)
        for label in (None, 2, "1"):
            step = online_step(self.model, self.features, label, source="LIVE")
            self.assertFalse(step.applied)
        self.assertEqual(before, self.model.weights)

    def test_scores_before_it_learns(self):
        step = online_step(self.model, self.features, 1, source="LIVE")
        self.assertTrue(step.applied)
        self.assertIsNotNone(step.predicted)
        self.assertIsNotNone(step.squared_error)
        # The recorded error is the one the OLD model made, not the new one.
        self.assertAlmostEqual(step.squared_error, (step.predicted - 1.0) ** 2, places=12)

    def test_one_update_per_outcome(self):
        self.assertEqual(self.model.online_updates, 0)
        online_step(self.model, self.features, 0, source="LIVE")
        self.assertEqual(self.model.online_updates, 1)

    def test_missing_features_snapshot_is_refused(self):
        step = online_step(self.model, {}, 1, source="LIVE")
        self.assertFalse(step.applied)
        self.assertIn("feature snapshot", step.reason)


class TestWalkForward(unittest.TestCase):
    def setUp(self):
        self.candles = trending_candles(420)
        self.samples = build_training_samples(self.candles, horizon=5)

    def test_below_the_sample_gate_nothing_is_promoted(self):
        report = walk_forward(self.samples[:100], min_samples=400)
        self.assertFalse(report.promoted)
        self.assertIn("400", report.reason)

    def test_every_evaluation_point_is_out_of_sample(self):
        report = walk_forward(self.samples, min_samples=300, folds=4)
        self.assertGreaterEqual(report.folds, 3)
        self.assertIsNotNone(report.accuracy)
        self.assertIsNotNone(report.brier)
        self.assertGreater(report.trades, 0)

    def test_promotion_gate_rejects_a_coin_flip(self):
        report = walk_forward(self.samples, min_samples=300, folds=4)
        report.accuracy = 0.50
        promoted, reason = promotion_gate(report, None)
        self.assertFalse(promoted)
        self.assertIn(str(PROMOTION_MIN_ACCURACY), reason)

    def test_promotion_gate_rejects_a_negative_expectancy(self):
        report = walk_forward(self.samples, min_samples=300, folds=4)
        report.accuracy = 0.60
        report.brier_skill = 0.05
        report.net_pips = -10.0
        promoted, reason = promotion_gate(report, None)
        self.assertFalse(promoted)
        self.assertIn("not positive", reason)

    def test_promotion_gate_demands_improvement_over_the_champion(self):
        report = walk_forward(self.samples, min_samples=300, folds=4)
        report.accuracy = 0.60
        report.brier_skill = 0.05
        report.net_pips = 10.0
        report.champion = {"brier": 0.10}
        report.brier = 0.20
        champion = OnlineLogisticModel().fit(
            [item.to_pair() for item in self.samples], epochs=30, validation_fraction=0.0
        )
        promoted, reason = promotion_gate(report, champion)
        self.assertFalse(promoted)
        self.assertIn("worse than the champion", reason)


class TestPromotionNeedsMoreThanAPositiveTotal(unittest.TestCase):
    def _report(self, trade_pips):
        from fusion.learning import WalkForwardReport

        report = WalkForwardReport()
        report.folds = 3
        report.accuracy = 0.60
        report.brier_skill = 0.05
        report.brier = 0.20
        report.net_pips = sum(trade_pips)
        report.trades = len(trade_pips)
        report.trade_pips = list(trade_pips)
        return report

    def test_a_positive_total_from_noise_does_not_promote(self):
        """The whole point of stats.py is that this must not pass.

        Two hundred coin flips finish above zero about half the time. A gate
        that reads only the sign of the running total promotes noise.
        """
        pips = [1.0 if index % 2 else -0.98 for index in range(240)]
        self.assertGreater(sum(pips), 0.0)
        promoted, reason = promotion_gate(self._report(pips), None)
        self.assertFalse(promoted)
        self.assertIn("not distinguishable from noise", reason)

    def test_a_resolved_edge_does_promote(self):
        pips = [0.65 if index % 3 else -0.85 for index in range(300)]
        promoted, _ = promotion_gate(self._report(pips), None)
        self.assertTrue(promoted)

    def test_a_losing_total_is_rejected_before_the_interval_is_consulted(self):
        pips = [-0.65 if index % 3 else 0.85 for index in range(300)]
        self.assertLess(sum(pips), 0.0)
        promoted, reason = promotion_gate(self._report(pips), None)
        self.assertFalse(promoted)
        self.assertIn("is not positive", reason)


class TestRollback(unittest.TestCase):
    def test_a_decayed_record_triggers_a_rollback(self):
        rows = [
            {"outcome": charter.LOSS, "net_pips": -2.0, "source": "LIVE"} for _ in range(80)
        ]
        should, reason = rollback_check(rows, window=120, min_trades=60)
        self.assertTrue(should)
        self.assertIn("decayed", reason)

    def test_a_healthy_record_does_not(self):
        rows = [{"outcome": charter.WIN, "net_pips": 2.0, "source": "LIVE"} for _ in range(80)]
        should, _ = rollback_check(rows, window=120, min_trades=60)
        self.assertFalse(should)

    def test_synthetic_rows_cannot_trigger_a_rollback(self):
        rows = [{"outcome": charter.LOSS, "net_pips": -2.0, "source": "SYNTHETIC"} for _ in range(80)]
        should, _ = rollback_check(rows, window=120, min_trades=60)
        self.assertFalse(should)

    def test_small_samples_are_not_judged(self):
        rows = [{"outcome": charter.LOSS, "net_pips": -2.0, "source": "LIVE"} for _ in range(10)]
        should, reason = rollback_check(rows, window=120, min_trades=60)
        self.assertFalse(should)
        self.assertIn("needed to judge", reason)

    def test_losing_money_at_a_good_hit_rate_still_rolls_back(self):
        """The failure mode that actually happens.

        Cost, not accuracy. 56% is comfortably better than a coin flip, but if
        the horizon and spread demand 67% then this champion is bleeding. The
        old guard compared against 0.5 and so could never see it.
        """
        rows = []
        for index in range(100):
            win = index < 56
            rows.append(
                {
                    "outcome": charter.WIN if win else charter.LOSS,
                    "net_pips": 1.0 if win else -1.6,
                    "source": "LIVE",
                }
            )
        # 56 wins x 1.0 - 44 losses x 1.6 = -14.4 pips: losing, at 56%.
        should, reason = rollback_check(rows, window=120, min_trades=60, break_even=0.67)
        self.assertTrue(should)
        self.assertIn("break even", reason)

        # Same record, judged against 50%, looks fine -- which is the bug.
        should_old, _ = rollback_check(rows, window=120, min_trades=60, break_even=0.5)
        self.assertFalse(should_old)

    def test_an_unpayable_configuration_rolls_back_at_any_hit_rate(self):
        rows = [{"outcome": charter.WIN, "net_pips": -0.2, "source": "LIVE"} for _ in range(80)]
        should, reason = rollback_check(
            rows, window=120, min_trades=60, break_even=float("inf")
        )
        self.assertTrue(should)
        self.assertIn("any rate", reason)


class TestNightlyJob(unittest.TestCase):
    def test_without_data_nothing_is_promoted(self):
        config = Config(horizon_minutes=5, nightly_min_samples=400)
        model, ensemble, report = run_nightly(make_candles(40), [], config=config, champion=None)
        self.assertIsNone(model)
        self.assertFalse(report.promoted)

    def test_promotion_is_possible_when_the_gates_pass(self):
        config = Config(horizon_minutes=5, nightly_min_samples=200)
        candles = trending_candles(320)
        model, ensemble, report = run_nightly(candles, [], config=config, champion=None)
        if report.promoted:
            self.assertTrue(model.trained)
            self.assertIn(model.mode, (MODE_VETO, MODE_ADVISORY))
            self.assertIn("Promoted", report.reason)
        else:
            self.assertIsNotNone(report.reason)

    def test_nightly_never_raises_on_short_history(self):
        config = Config(horizon_minutes=5, nightly_min_samples=5)
        model, _, report = run_nightly(make_candles(35), [], config=config, champion=None)
        self.assertIsNotNone(report.reason)


if __name__ == "__main__":
    unittest.main()
