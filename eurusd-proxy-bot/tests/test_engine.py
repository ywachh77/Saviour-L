"""The pipeline's hard refusals: late, stale, uncalibrated, unstable."""

import unittest
from datetime import timedelta

from fusion import charter
from fusion.config import Config
from fusion.domain import ModelRead
from fusion.engine import FLAT, FusionEngine, build_read, model_verdict
from fusion.features import compute_features
from fusion.model import MODE_ADVISORY, MODE_VETO, OnlineLogisticModel

from .helpers import make_candles


def engine(**overrides) -> FusionEngine:
    config = Config(pair="EUR/USD", horizon_minutes=5, candle_history=200)
    for key, value in overrides.items():
        setattr(config, key, value)
    return FusionEngine(config)


class TestBuildRead(unittest.TestCase):
    def setUp(self):
        self.candles = make_candles(220, seed=13)
        self.features = compute_features(self.candles)[-1]

    def test_untrained_model_gives_no_read(self):
        self.assertIsNone(build_read(None, None, self.features))
        self.assertIsNone(build_read(OnlineLogisticModel(), None, self.features))

    def test_raw_state_below_the_calibration_threshold(self):
        model = OnlineLogisticModel()
        model.weights = [0.01] * 24
        model.training_samples = 10
        read = build_read(model, None, self.features)
        self.assertEqual(read.calibration_state, "RAW")
        self.assertFalse(read.calibrated)

    def test_flat_calibration_is_refused(self):
        model = OnlineLogisticModel()
        model.weights = [0.01] * 24
        model.training_samples = 100
        model.calibrator.thresholds = [0.5]
        model.calibrator.values = [0.5]
        model.calibrator.samples = 100
        read = build_read(model, None, self.features)
        self.assertEqual(read.calibration_state, FLAT)
        self.assertFalse(read.usable)
        self.assertIn("no usable rank signal", read.abstain_reason)


class TestModelVerdict(unittest.TestCase):
    def _read(self, probability: float, state: str = "CALIBRATED") -> ModelRead:
        return ModelRead(
            raw=probability,
            probability=probability,
            calibrated=state == "CALIBRATED",
            calibration_state=state,
        )

    def test_no_model_never_blocks(self):
        allowed, _ = model_verdict("BUY", None)
        self.assertTrue(allowed)

    def test_veto_only_blocks_confident_disagreement(self):
        allowed, _ = model_verdict("BUY", self._read(0.40), mode=MODE_VETO)
        self.assertFalse(allowed)
        allowed, _ = model_verdict("BUY", self._read(0.49), mode=MODE_VETO)
        self.assertTrue(allowed)

    def test_promoted_mode_requires_agreement(self):
        allowed, reason = model_verdict("BUY", self._read(0.49), mode=MODE_ADVISORY)
        self.assertFalse(allowed)
        self.assertIn("Promoted model", reason)
        allowed, _ = model_verdict("BUY", self._read(0.53), mode=MODE_ADVISORY)
        self.assertTrue(allowed)

    def test_flat_read_blocks_every_side(self):
        allowed, _ = model_verdict("BUY", self._read(0.60, state=FLAT), mode=MODE_VETO)
        self.assertFalse(allowed)


class TestEngineRefusals(unittest.TestCase):
    def test_stale_data_is_blocked_before_anything_else(self):
        candles = make_candles(200, seed=17)
        stamp = candles[-1].timestamp + timedelta(minutes=30)
        result = engine().evaluate(candles, at=stamp)
        self.assertEqual(result.seal_state, charter.SEAL_ABSTAIN)
        self.assertIn("freshness limit", result.reason)
        self.assertEqual(result.gate.quality, "STALE")

    def test_a_late_seal_is_discarded_not_backdated(self):
        candles = make_candles(260, volatility=0.00025, drift=0.00006, seed=23)
        stamp = candles[-1].timestamp + timedelta(minutes=1, seconds=25)
        result = engine(max_seal_latency_seconds=10).evaluate(candles, at=stamp)
        # Either there was no signal, or the late one was dropped. Never sent.
        self.assertNotEqual(result.seal_state, charter.SEAL_SIGNAL)
        if result.seal_state == charter.SEAL_DISCARDED_LATE:
            self.assertIn("not backdated", result.reason)

    def test_signal_entry_is_the_open_of_the_forming_candle(self):
        candles = make_candles(300, volatility=0.00022, drift=0.00008, seed=29)
        stamp = candles[-1].timestamp + timedelta(minutes=1)
        result = engine().evaluate(candles, at=stamp)
        if result.actionable:
            signal = result.signal
            self.assertEqual(signal.entry_time, candles[-1].timestamp + timedelta(minutes=1))
            self.assertEqual(signal.expiry_time, signal.entry_time + timedelta(minutes=5))
            self.assertIsNone(signal.entry_price)  # filled from the candle's open later

    def test_uncertainty_above_the_limit_abstains(self):
        from fusion.learning import build_training_samples

        candles = make_candles(240, seed=37)
        stamp = candles[-1].timestamp + timedelta(minutes=1)
        pairs = [item.to_pair() for item in build_training_samples(candles, horizon=5)]
        model = OnlineLogisticModel().fit(pairs, epochs=40, validation_fraction=0.0)

        class LoudEnsemble:
            ready = True
            members = [1, 2, 3]

            def predict(self, features):
                return 0.5, 0.9

        result = engine(max_uncertainty=0.10).evaluate(candles, model=model, ensemble=LoudEnsemble(), at=stamp)
        self.assertEqual(result.seal_state, charter.SEAL_ABSTAIN)
        self.assertIn("stability limit", result.reason)

    def test_too_few_candles_never_produce_a_signal(self):
        candles = make_candles(20, seed=41)
        result = engine().evaluate(candles, at=candles[-1].timestamp + timedelta(minutes=1))
        self.assertEqual(result.seal_state, charter.SEAL_ABSTAIN)


if __name__ == "__main__":
    unittest.main()
