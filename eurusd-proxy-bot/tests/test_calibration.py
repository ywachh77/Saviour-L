"""Isotonic calibration: monotone, honest, and never a fake ranking."""

import unittest

from fusion.calibration import IsotonicCalibrator, brier_score, expected_calibration_error, fit_validated


def separable_points(count: int = 200) -> list[tuple[float, int]]:
    """A clearly monotone relationship: high score means up."""
    points = []
    for index in range(count):
        score = 0.30 + 0.40 * (index / count)
        points.append((score, 1 if index > count * 0.55 else 0))
    return points


def noise_points(count: int = 200, seed: int = 5) -> list[tuple[float, int]]:
    import random

    rng = random.Random(seed)
    return [(rng.uniform(0.4, 0.6), rng.randint(0, 1)) for _ in range(count)]


class TestIsotonic(unittest.TestCase):
    def test_below_minimum_points_there_is_no_fit(self):
        calibrator = IsotonicCalibrator.fit(separable_points(30))
        self.assertFalse(calibrator.fitted)
        self.assertEqual(calibrator.transform(0.9), 0.9)

    def test_fit_is_monotone(self):
        calibrator = IsotonicCalibrator.fit(separable_points(400))
        self.assertTrue(calibrator.fitted)
        self.assertGreaterEqual(calibrator.levels, 2)
        values = calibrator.values
        self.assertEqual(values, sorted(values))

    def test_transform_stays_in_bounds_and_monotone(self):
        calibrator = IsotonicCalibrator.fit(separable_points(400))
        outputs = [calibrator.transform(score / 100.0) for score in range(1, 100)]
        self.assertTrue(all(0.0 <= value <= 1.0 for value in outputs))
        self.assertEqual(outputs, sorted(outputs))

    def test_flat_map_is_labelled_not_trusted(self):
        calibrator = IsotonicCalibrator.fit(noise_points(400))
        # Whether PAVA collapses or not, a single level must never pass as a fit.
        if calibrator.levels < 2:
            self.assertTrue(calibrator.flat)
        self.assertEqual(calibrator.flat, calibrator.fitted and calibrator.levels < 2)

    def test_round_trip_through_dict(self):
        calibrator = IsotonicCalibrator.fit(separable_points(400))
        restored = IsotonicCalibrator.from_dict(calibrator.to_dict())
        self.assertEqual(calibrator.thresholds, restored.thresholds)
        self.assertEqual(calibrator.values, restored.values)
        self.assertAlmostEqual(calibrator.transform(0.55), restored.transform(0.55), places=12)

    def test_rejects_mismatched_payloads(self):
        restored = IsotonicCalibrator.from_dict({"thresholds": [0.1, 0.2], "values": [0.3]})
        self.assertFalse(restored.fitted)

    def test_fit_validated_accepts_a_real_improvement(self):
        calibrator, report = fit_validated(separable_points(400))
        self.assertTrue(calibrator.fitted)
        self.assertGreater(int(report["validated_on"]), 0)
        self.assertLessEqual(float(report["brier_after"]), float(report["brier_before"]) + 1e-9)
        self.assertLessEqual(float(report["ece_after"]), float(report["ece_before"]) + 1e-9)

    def test_fit_validated_reports_unvalidated_below_holdout_size(self):
        calibrator, report = fit_validated(separable_points(100))
        self.assertEqual(int(report["validated_on"]), 0)
        self.assertIn("not split", report["reason"])

    def test_scores_behave(self):
        perfect = [(0.9, 1), (0.1, 0)]
        self.assertAlmostEqual(brier_score(perfect), 0.01, places=9)
        self.assertAlmostEqual(expected_calibration_error(perfect), 0.1, places=6)
        self.assertIsNone(brier_score([]))


if __name__ == "__main__":
    unittest.main()
