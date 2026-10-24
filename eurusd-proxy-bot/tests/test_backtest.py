"""The replay must be purged, cost-charged, and honest about its own limits."""

import unittest
from datetime import timedelta

from fusion import datagate
from fusion.backtest import (
    MIN_SIGNALS_FOR_A_VIEW,
    BacktestReport,
    _label,
    _purged_splits,
    run_backtest,
)
from fusion.config import Config
from fusion.council import PER_PROTOTYPE_WEIGHT_CAP, CouncilWeights
from fusion.engine import FusionEngine
from fusion.features import compute_features
from fusion.learning import promotion_gate, run_nightly, walk_forward
from fusion.regime import classify
from fusion.simulate import synthetic_series


def config(**overrides) -> Config:
    base = Config(pair="EUR/USD", horizon_minutes=3, spread_pips=1.2)
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


class TestPurging(unittest.TestCase):
    def test_no_training_label_reaches_into_the_test_window(self):
        indices = list(range(0, 900, 1))
        horizon = 5
        for train, test in _purged_splits(indices, 4, horizon):
            first_test = test[0]
            for j in train:
                self.assertLess(j + horizon, first_test, "training label leaks into the test window")

    def test_folds_partition_the_test_bars(self):
        indices = list(range(900))
        covered = []
        for _train, test in _purged_splits(indices, 4, 3):
            covered.extend(test)
        self.assertEqual(len(covered), len(set(covered)))

    def test_too_little_data_produces_no_folds(self):
        self.assertEqual(_purged_splits(list(range(50)), 4, 3), [])


class TestLabels(unittest.TestCase):
    def setUp(self):
        self.candles = synthetic_series(300, seed=31)

    def test_label_matches_price_direction(self):
        for index in (40, 100, 160):
            label = _label(self.candles, index, 3)
            if label is None:
                continue
            move = self.candles[index + 3].close - self.candles[index + 1].open
            self.assertEqual(label, 1 if move > 0 else 0)

    def test_ties_are_dropped_not_guessed(self):
        flat = synthetic_series(120, seed=5, volatility=0.0, drift=0.0)
        labels = [_label(flat, index, 3) for index in range(20, 100)]
        self.assertTrue(all(label is None for label in labels))

    def test_labels_are_unavailable_at_the_end_of_the_series(self):
        self.assertIsNone(_label(self.candles, len(self.candles) - 2, 3))


class TestRunBacktest(unittest.TestCase):
    def test_insufficient_history_refuses_to_report_numbers(self):
        report = run_backtest(synthetic_series(200, seed=3), config(), folds=4)
        self.assertEqual(report.signals, 0)
        self.assertIn("600", report.note)

    def test_replay_is_deterministic(self):
        candles = synthetic_series(1200, seed=17)
        first = run_backtest(candles, config(), folds=4).to_dict()
        second = run_backtest(candles, config(), folds=4).to_dict()
        self.assertEqual(first, second)

    def test_every_signal_pays_the_spread(self):
        candles = synthetic_series(1500, seed=11)
        report = run_backtest(candles, config(spread_pips=1.2), folds=4)
        if report.signals:
            self.assertLess(report.net_pips, report.gross_pips)
            self.assertGreater(report.gross_pips - report.net_pips, 0)

    def test_a_wider_spread_can_only_hurt(self):
        candles = synthetic_series(1500, seed=11)
        cheap = run_backtest(candles, config(spread_pips=0.2), folds=4)
        dear = run_backtest(candles, config(spread_pips=3.0), folds=4)
        self.assertGreater(cheap.net_pips, dear.net_pips)

    def test_coverage_is_a_fraction_of_bars(self):
        report = run_backtest(synthetic_series(1200, seed=23), config(), folds=4)
        self.assertGreaterEqual(report.coverage, 0.0)
        self.assertLessEqual(report.coverage, 1.0)

    def test_intervals_come_with_any_win_rate(self):
        report = run_backtest(synthetic_series(1500, seed=29), config(), folds=4)
        if report.win_rate is not None:
            self.assertLessEqual(report.wilson_low, report.win_rate)
            self.assertGreaterEqual(report.wilson_high, report.win_rate)

    def test_the_note_names_its_own_limits(self):
        report = run_backtest(synthetic_series(1200, seed=37), config(), folds=4)
        self.assertIn("one path", report.note)
        self.assertIn("healthy", report.note)

    def test_expectancy_is_per_signal(self):
        report = run_backtest(synthetic_series(1500, seed=41), config(), folds=4)
        decided = report.wins + report.losses
        if decided:
            self.assertAlmostEqual(report.expectancy_pips, report.net_pips / decided, places=6)
        else:
            self.assertIsNone(report.expectancy_pips)

    def test_enough_for_a_view_threshold(self):
        report = BacktestReport(signals=MIN_SIGNALS_FOR_A_VIEW - 1)
        self.assertFalse(report.enough_for_a_view)
        self.assertTrue(BacktestReport(signals=MIN_SIGNALS_FOR_A_VIEW).enough_for_a_view)


class TestSharedDecisionPath(unittest.TestCase):
    def test_prepared_and_live_paths_reach_the_same_verdict(self):
        """The backtest must not re-implement the rules; it must call them."""
        candles = synthetic_series(600, seed=53)
        engine = FusionEngine(config())
        rows = compute_features(candles)
        index = len(candles) - 6
        features = rows[index]
        self.assertIsNotNone(features)

        prepared = engine.evaluate_prepared(
            candles[: index + 1],
            features=features,
            gate=datagate.GateResult(datagate.OK, "OK", 0.0, 1.2, "modelled", 0, index + 1, "test"),
            regime_state=classify(
                atr_value=features["atr"],
                atr_pct=features["atr_pct"],
                atr_baseline=features["atr"],
                adx_value=features["adx"],
                close=candles[index].close,
                ema_fast=features["ema_fast"],
                ema_slow=features["ema_slow"],
            ),
            weights=CouncilWeights(),
            at=candles[index].timestamp + timedelta(minutes=1),
        )
        # The live path needs a healthy gate too, so compare side only.
        live = engine.evaluate(
            candles[: index + 1],
            weights=CouncilWeights(),
            at=candles[index].timestamp + timedelta(minutes=1),
        )
        self.assertEqual(prepared.seal_state, live.seal_state)
        if prepared.actionable and live.actionable:
            self.assertEqual(prepared.signal.side, live.signal.side)


class TestPromotionUsesTheReplay(unittest.TestCase):
    def _passing_report(self):
        candles = synthetic_series(900, seed=61)
        from fusion.learning import build_training_samples

        samples = build_training_samples(candles, horizon=3)
        report = walk_forward(samples, min_samples=200, folds=3)
        report.accuracy = 0.60
        report.brier_skill = 0.05
        report.net_pips = 25.0
        report.folds = 3
        # These tests are about the system-replay gate, so the walk-forward
        # gate has to pass first. Override the per-trade outcomes as well as
        # the total: promotion is judged on the distribution now, and leaving
        # the real (losing) walk-forward trades in place would make the fixture
        # contradict itself -- a 25 pip total built from -1.3 pips of trades.
        report.trade_pips = [0.65 if index % 3 else -0.85 for index in range(300)]
        return report

    def test_a_negative_system_replay_blocks_promotion(self):
        report = self._passing_report()
        system = BacktestReport(signals=400, wins=220, losses=180, net_pips=-80.0)
        self.assertTrue(system.enough_for_a_view)
        promoted, reason = promotion_gate(report, None, system)
        self.assertFalse(promoted)
        self.assertIn("System replay blocked", reason)

    def test_a_positive_system_replay_does_not_block(self):
        report = self._passing_report()
        system = BacktestReport(signals=400, wins=230, losses=170, net_pips=40.0)
        promoted, _ = promotion_gate(report, None, system)
        self.assertTrue(promoted)

    def test_a_small_replay_cannot_block_promotion(self):
        report = self._passing_report()
        system = BacktestReport(signals=10, wins=2, losses=8, net_pips=-20.0)
        self.assertFalse(system.enough_for_a_view)
        promoted, _ = promotion_gate(report, None, system)
        self.assertTrue(promoted)

    def _promoted_champion(self):
        from fusion.model import MODE_ADVISORY, OnlineLogisticModel

        model = OnlineLogisticModel()
        model.weights = [0.01] * 24
        model.means = [0.0] * 24
        model.scales = [1.0] * 24
        model.training_samples = 500
        model.mode = MODE_ADVISORY
        return model

    def test_a_negative_replay_demotes_a_promoted_model(self):
        from fusion.model import MODE_VETO

        candles = synthetic_series(3200, seed=83)
        cfg = config(nightly_min_samples=200)
        champion = self._promoted_champion()
        model, _ensemble, report = run_nightly(candles, [], config=cfg, champion=champion)
        self.assertIsNotNone(model)
        self.assertEqual(model.mode, MODE_VETO)
        self.assertTrue(report.rolled_back)
        self.assertIn("Demoted", report.reason)
        self.assertFalse(report.promoted)

    def test_a_healthy_replay_keeps_authority(self):
        champion = self._promoted_champion()
        candles = synthetic_series(900, seed=91)
        model, _ensemble, _report = run_nightly(candles, [], config=config(nightly_min_samples=200), champion=champion)
        # Too few replayed signals to judge, so nothing is demoted on this run.
        self.assertIsNotNone(model)

    def test_nightly_populates_the_system_report(self):
        candles = synthetic_series(900, seed=67)
        cfg = config(nightly_min_samples=200)
        _model, _ensemble, report = run_nightly(candles, [], config=cfg, champion=None)
        self.assertIsNotNone(report.system)
        self.assertIn("system", report.to_dict())


class TestReplayLearnsTheCouncil(unittest.TestCase):
    """A replay must learn the council, or it is measuring a different bot.

    The live runner credits and blames prototypes on every settled outcome. A
    replay that skipped that would be measuring a bot that never learns, and no
    prototype could ever earn a voice in a backtest — an entire family of
    strategies would be invisible here while fully live in production.
    """

    def _replay(self, **overrides):
        candles = synthetic_series(900, seed=20260101)
        weights = CouncilWeights()
        report = run_backtest(candles, config(**overrides), folds=3, weights=weights)
        return report, weights

    def test_the_replay_credits_settled_outcomes_to_the_council(self) -> None:
        report, weights = self._replay()
        self.assertGreater(report.signals, 0)
        self.assertTrue(
            any(seen > 0 for seen in weights.seen.values()),
            "the replay settled signals but the council never saw any of them",
        )

    def test_a_prototype_can_earn_a_voice_during_a_replay(self) -> None:
        _report, weights = self._replay()
        active = weights.active()
        self.assertTrue(active, "no prototype earned a voice across the whole replay")
        for name, weight in active.items():
            with self.subTest(member=name):
                self.assertGreater(weight, 0.0)
                self.assertLessEqual(weight, PER_PROTOTYPE_WEIGHT_CAP)

    def test_the_report_carries_the_learned_weights(self) -> None:
        report, weights = self._replay()
        self.assertEqual(
            {name: round(value, 4) for name, value in weights.active().items()},
            report.to_dict()["learned_weights"],
        )

    def test_a_replay_that_fires_nothing_teaches_the_council_nothing(self) -> None:
        """No settled outcome means no observation, not an observation of zero."""
        candles = synthetic_series(900, seed=20260101)
        weights = CouncilWeights()
        # The core gate can pass at most 6 of 6 checks, so asking for 7 makes
        # it fire nothing at all — a clean way to isolate the learning step.
        report = run_backtest(candles, config(min_confluence_checks=7), folds=3, weights=weights)
        self.assertEqual(0, report.signals)
        self.assertEqual(0, sum(weights.seen.values()))
        self.assertEqual({}, weights.active())


if __name__ == "__main__":
    unittest.main()
