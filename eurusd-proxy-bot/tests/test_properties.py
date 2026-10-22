"""Property tests: invariants that must hold for ANY candle series.

Hand-picked cases catch the bugs you expect. These catch the ones you do not:
random regimes, gaps, spikes, dead-flat tapes, and everything in between, fed
through the whole pipeline while asserting the charter invariants.
"""

import math
import random
import shutil
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

from fusion import charter, settle
from fusion.config import Config
from fusion.engine import FusionEngine
from fusion.features import FEATURE_NAMES, compute_features
from fusion.learning import build_training_samples
from fusion.model import OnlineLogisticModel
from fusion.runner import FusionBot
from fusion.simulate import run_simulation, synthetic_series
from fusion.store import LocalStore, Store
from fusion.timeutil import floor_minute

SEEDS = range(1, 41)


def make_config(directory: Path, **overrides) -> Config:
    config = Config(
        pair="EUR/USD",
        horizon_minutes=3,
        local_state_dir=str(directory),
        runner="properties",
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def wild_series(seed: int, count: int = 260) -> list:
    """Candle series with random regimes, gaps and spikes."""
    from fusion.domain import Candle
    from fusion.timeutil import UTC
    from datetime import datetime

    rng = random.Random(seed)
    stamps = []
    stamp = datetime(2026, 4, 1, 0, 0, tzinfo=UTC)
    while len(stamps) < count:
        stamps.append(stamp)
        # 6% of minutes are missing from the feed entirely.
        stamp += timedelta(minutes=2 if rng.random() < 0.06 else 1)

    candles = []
    price = 1.1000
    for index, moment in enumerate(stamps):
        regime = index // 40
        volatility = [0.00005, 0.0004, 0.00002, 0.0015, 0.0002][regime % 5]
        if rng.random() < 0.02:  # spike bar
            volatility *= 12
        open_price = price
        close_price = max(0.5, open_price + rng.gauss(0.0, volatility))
        wick = abs(rng.gauss(0.0, volatility * 0.8))
        candles.append(
            Candle(
                timestamp=moment,
                open=round(open_price, 5),
                high=round(max(open_price, close_price) + wick, 5),
                low=round(min(open_price, close_price) - wick, 5),
                close=round(close_price, 5),
                volume=float(rng.randrange(0, 5000)),
            )
        )
        price = close_price
    return candles


class TestFeatureInvariants(unittest.TestCase):
    def test_features_are_finite_or_none_across_wild_series(self):
        for seed in SEEDS:
            candles = wild_series(seed)
            rows = compute_features(candles)
            self.assertEqual(len(rows), len(candles))
            for row in rows:
                if row is None:
                    continue
                for name in FEATURE_NAMES:
                    self.assertTrue(math.isfinite(float(row[name])), f"seed {seed}, feature {name}")

    def test_features_never_see_the_future(self):
        for seed in (3, 17, 29):
            candles = wild_series(seed, count=200)
            full = compute_features(candles)
            truncated = compute_features(candles[:150])
            for index in range(0, 150):
                if full[index] is None or truncated[index] is None:
                    self.assertIs(full[index], None)
                    self.assertIs(truncated[index], None)
                    continue
                for name in FEATURE_NAMES:
                    self.assertAlmostEqual(
                        float(full[index][name]), float(truncated[index][name]), places=12,
                        msg=f"feature {name} changed at bar {index}",
                    )

    def test_training_labels_are_always_zero_or_one(self):
        for seed in SEEDS:
            samples = build_training_samples(wild_series(seed), horizon=3)
            for sample in samples:
                self.assertIn(sample.label, (0, 1))
                self.assertTrue(math.isfinite(sample.gross_pips))
                self.assertFalse(math.isclose(sample.gross_pips, 0.0, abs_tol=1e-12))


class TestEngineInvariants(unittest.TestCase):
    def test_the_engine_never_crashes_and_never_invents_a_price(self):
        engine = FusionEngine(make_config(Path(tempfile.mkdtemp())))
        for seed in SEEDS:
            candles = wild_series(seed)
            for offset in (0, 1, 2, 7, 25):
                stamp = candles[-1].timestamp + timedelta(minutes=1 + offset)
                result = engine.evaluate(candles, at=stamp)
                self.assertIn(result.seal_state, ("SIGNAL", "ABSTAIN", "DISCARDED_LATE"))
                self.assertTrue(result.reason)
                if result.actionable:
                    signal = result.signal
                    self.assertIsNone(signal.entry_price)
                    self.assertEqual(signal.entry_time, floor_minute(stamp))
                    self.assertIn(signal.side, ("BUY", "SELL"))
                    self.assertGreater(signal.spread_pips, 0)

    def test_a_signal_is_never_sealed_for_a_past_minute(self):
        engine = FusionEngine(make_config(Path(tempfile.mkdtemp()), max_seal_latency_seconds=10))
        for seed in SEEDS:
            candles = wild_series(seed)
            for seconds in (0, 3, 9, 11, 45):
                stamp = candles[-1].timestamp + timedelta(minutes=1, seconds=seconds)
                result = engine.evaluate(candles, at=stamp)
                if seconds > 10:
                    self.assertNotEqual(result.seal_state, "SIGNAL")


class TestSettlementInvariants(unittest.TestCase):
    def test_costs_are_charged_and_direction_is_consistent(self):
        for seed in SEEDS:
            candles = wild_series(seed)
            for side in ("BUY", "SELL"):
                for index in range(40, len(candles) - 8):
                    entry = candles[index].timestamp
                    signal = _signal(entry, side, horizon=3)
                    settlement = settle.settle(signal, candles)
                    if settlement.outcome == charter.UNRESOLVED:
                        self.assertIsNone(settlement.net_pips)
                        continue
                    if settlement.outcome == charter.TIED:
                        self.assertAlmostEqual(settlement.gross_pips, 0.0, places=9)
                        continue
                    # Net is always gross minus the spread, never gross alone.
                    self.assertAlmostEqual(
                        settlement.net_pips,
                        settlement.gross_pips - settlement.spread_pips,
                        places=6,
                    )
                    # Direction and the raw move agree.
                    expected = "BUY" if settlement.move_pips > 0 else "SELL"
                    self.assertEqual(settle.winning_side(settlement), expected)


class TestFullLoopInvariants(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_the_loop_holds_the_charter_on_wild_data(self):
        for seed in (2, 5, 9, 14, 23):
            directory = self.directory / f"seed{seed}"
            directory.mkdir(parents=True, exist_ok=True)
            bot = FusionBot(config=make_config(directory), store=Store(None, LocalStore(directory)), provider=None)
            candles = wild_series(seed, count=300)
            run_simulation(bot, candles)

            rows = bot.store.load_settlements()
            for row in rows:
                if row["outcome"] == charter.UNRESOLVED:
                    self.assertIsNone(row["net_pips"])
                elif row["outcome"] in (charter.WIN, charter.LOSS):
                    self.assertIsNotNone(row["net_pips"])
                    self.assertLess(row["net_pips"], row["gross_pips"])
            # Synthetic data must not train anything, however wild the tape.
            self.assertFalse(bot.model and bot.model.trained)
            self.assertEqual(bot.weights.active(), {})
            self.assertEqual(bot.calibration_points, [])

    def test_a_paused_bot_never_sends_anything(self):
        directory = self.directory / "paused"
        bot = FusionBot(config=make_config(directory), store=Store(None, LocalStore(directory)), provider=None)
        bot.paused = True
        candles = synthetic_series(200, seed=77)
        run_simulation(bot, candles, minutes=80)
        rows = bot.store._read("signals", limit=500)
        for row in rows:
            self.assertIn(
                row.get("delivery_state"),
                (charter.DELIVERY_PENDING, charter.DELIVERY_SUPPRESSED, charter.DELIVERY_FAILED),
            )
            self.assertNotEqual(row.get("delivery_state"), charter.DELIVERY_SENT)


class TestModelInvariants(unittest.TestCase):
    def test_predictions_stay_in_bounds_on_wild_features(self):
        candles = wild_series(11, count=300)
        pairs = [item.to_pair() for item in build_training_samples(candles, horizon=3)]
        if len(pairs) < 40:
            self.skipTest("not enough labelled bars in this series")
        model = OnlineLogisticModel().fit(pairs, epochs=40, validation_fraction=0.0)
        for features, _ in pairs:
            probability = model.predict_up(features)
            self.assertGreaterEqual(probability, 0.0)
            self.assertLessEqual(probability, 1.0)
            self.assertTrue(math.isfinite(probability))


def _signal(entry, side, horizon=3):
    from fusion.domain import Signal

    return Signal.create(
        pair="EUR/USD",
        side=side,
        entry_time=entry,
        strategy="fusion_council",
        reason="property test",
        source="LIVE",
        generated_at=entry,
        horizon_minutes=horizon,
        spread_pips=1.2,
    )


if __name__ == "__main__":
    unittest.main()
