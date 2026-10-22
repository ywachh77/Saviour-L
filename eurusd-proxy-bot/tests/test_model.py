"""The meta-model: trainable, explainable, and hostile to stale parameters."""

import unittest

from fusion.features import FEATURE_NAMES, FEATURE_VERSION, compute_features
from fusion.learning import build_training_samples
from fusion.model import MODE_VETO, BootstrapEnsemble, OnlineLogisticModel

from .helpers import make_candles, trending_candles


def training_pairs(count: int = 200, *, trending: bool = True):
    candles = trending_candles(160) if trending else make_candles(160, seed=42)
    samples = build_training_samples(candles, horizon=5)
    return [item.to_pair() for item in samples][:count]


class TestOnlineLogisticModel(unittest.TestCase):
    def setUp(self):
        self.pairs = training_pairs(200)
        self.model = OnlineLogisticModel().fit(self.pairs, epochs=60, validation_fraction=0.0)

    def test_training_produces_compatible_parameters(self):
        self.assertTrue(self.model.trained)
        self.assertTrue(self.model.compatible)
        self.assertEqual(len(self.model.weights), len(FEATURE_NAMES))
        self.assertEqual(self.model.feature_version, FEATURE_VERSION)
        self.assertEqual(self.model.mode, MODE_VETO)

    def test_predictions_are_probabilities(self):
        features, _ = self.pairs[10]
        probability = self.model.predict_up(features)
        self.assertGreaterEqual(probability, 0.0)
        self.assertLessEqual(probability, 1.0)

    def test_learning_one_changes_the_weights(self):
        features, label = self.pairs[0]
        before = list(self.model.weights)
        self.model.learn_one(features, label)
        self.assertNotEqual(before, self.model.weights)
        self.assertEqual(self.model.online_updates, 1)

    def test_learning_requires_a_trained_model(self):
        with self.assertRaises(ValueError):
            OnlineLogisticModel().learn_one(self.pairs[0][0], 1)

    def test_learning_rejects_a_bad_label(self):
        with self.assertRaises(ValueError):
            self.model.learn_one(self.pairs[0][0], 7)

    def test_too_few_samples_refuses_to_train(self):
        with self.assertRaises(ValueError):
            OnlineLogisticModel().fit(self.pairs[:10])

    def test_contributions_explain_the_score(self):
        features, _ = self.pairs[5]
        rows = self.model.contributions(features, top=4)
        self.assertEqual(len(rows), 4)
        self.assertTrue(all("contribution" in row and "label" in row for row in rows))
        magnitudes = [abs(row["contribution"]) for row in rows]
        self.assertEqual(magnitudes, sorted(magnitudes, reverse=True))

    def test_incompatible_version_is_rejected_on_load(self):
        payload = self.model.to_dict()
        payload["feature_version"] = FEATURE_VERSION + 99
        self.assertIsNone(OnlineLogisticModel.from_dict(payload))

    def test_round_trip_through_dict(self):
        restored = OnlineLogisticModel.from_dict(self.model.to_dict())
        self.assertIsNotNone(restored)
        features, _ = self.pairs[3]
        self.assertAlmostEqual(restored.predict_up(features), self.model.predict_up(features), places=12)

    def test_feature_order_is_stable(self):
        row = compute_features(trending_candles(120))[-1]
        vector = [row[name] for name in FEATURE_NAMES]
        from fusion.features import model_vector

        self.assertEqual(model_vector(row), vector)


class TestBootstrapEnsemble(unittest.TestCase):
    def test_ensemble_measures_disagreement(self):
        pairs = training_pairs(160)
        ensemble = BootstrapEnsemble.fit(pairs, member_count=3, epochs=30)
        self.assertTrue(ensemble.ready)
        mean, spread = ensemble.predict(pairs[0][0])
        self.assertGreaterEqual(spread, 0.0)
        self.assertGreaterEqual(mean, 0.0)

    def test_ensemble_needs_data(self):
        self.assertFalse(BootstrapEnsemble().ready)
        with self.assertRaises(ValueError):
            BootstrapEnsemble().predict({})

    def test_ensemble_round_trip(self):
        pairs = training_pairs(160)
        ensemble = BootstrapEnsemble.fit(pairs, member_count=3, epochs=30)
        restored = BootstrapEnsemble.from_dict(ensemble.to_dict())
        self.assertTrue(restored.ready)


if __name__ == "__main__":
    unittest.main()
