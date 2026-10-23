"""The inference layer: when may a number be called an edge?

These tests exist because a reporting bug is worse than a modelling bug. A
model that is wrong loses money slowly and visibly; a report that calls noise
an edge gets believed, and then acted on.

The central property under test is the one that was violated in practice:
**a small sample with a flattering average must not be able to claim an edge**,
no matter how confident its interval looks.
"""

from __future__ import annotations

import math
import random
import statistics
import unittest

from fusion import stats


def _sample(mean: float, sd: float, n: int, seed: int = 11) -> list[float]:
    rng = random.Random(seed)
    return [rng.gauss(mean, sd) for _ in range(n)]


class ConfidenceIntervalTests(unittest.TestCase):
    def test_recovers_a_known_mean(self) -> None:
        mean, low, high = stats.confidence_interval(_sample(1.0, 5.0, 600))
        self.assertLess(low, mean)
        self.assertGreater(high, mean)
        self.assertLess(abs(mean - 1.0), 0.6)

    def test_interval_contains_the_mean(self) -> None:
        data = _sample(-0.8, 4.0, 500)
        mean, low, high = stats.confidence_interval(data)
        self.assertLessEqual(low, mean)
        self.assertGreaterEqual(high, mean)

    def test_is_deterministic_for_a_given_seed(self) -> None:
        data = _sample(0.5, 3.0, 200)
        first = stats.confidence_interval(data, seed=99)
        second = stats.confidence_interval(data, seed=99)
        self.assertEqual(first, second)

    def test_empty_input_is_not_an_error(self) -> None:
        self.assertEqual((0.0, 0.0, 0.0), stats.confidence_interval([]))

    def test_two_points_get_no_fake_precision(self) -> None:
        """Pretending to measure the spread of two numbers is worse than not trying."""
        mean, low, high = stats.confidence_interval([1.0, 3.0])
        self.assertEqual(mean, low)
        self.assertEqual(mean, high)

    def test_non_finite_values_are_dropped(self) -> None:
        data = _sample(1.0, 2.0, 100) + [float("nan"), float("inf")]
        mean, low, high = stats.confidence_interval(data)
        self.assertTrue(all(value == value and abs(value) != float("inf") for value in (mean, low, high)))

    def test_autocorrelation_widens_the_interval(self) -> None:
        """The reason for a block bootstrap rather than a plain t-interval.

        Two series with the same spread, one independent and one with runs. The
        runny one carries less information per observation, so its interval must
        be wider — a per-trade shuffle would miss this and invent significance.
        """
        rng = random.Random(5)
        independent = [rng.gauss(0.0, 3.0) for _ in range(400)]

        rng2 = random.Random(5)
        runny: list[float] = []
        level = 0.0
        for _ in range(400 // 20):
            level = rng2.gauss(0.0, 3.0)
            runny.extend(rng2.gauss(level, 0.5) for _ in range(20))

        _, iid_low, iid_high = stats.confidence_interval(independent, block_size=1)
        _, run_low, run_high = stats.confidence_interval(runny, block_size=20)

        self.assertGreater(run_high - run_low, iid_high - iid_low)


class VerdictTests(unittest.TestCase):
    def test_a_clear_positive_edge_is_demonstrated(self) -> None:
        verdict = stats.edge_verdict(_sample(1.5, 4.0, 600))
        self.assertTrue(verdict.demonstrated)
        self.assertEqual(stats.EDGE_DEMONSTRATED, verdict.status)

    def test_a_clear_negative_edge_is_refuted(self) -> None:
        verdict = stats.edge_verdict(_sample(-1.5, 4.0, 600))
        self.assertTrue(verdict.refuted)
        self.assertEqual(stats.EDGE_REFUTED, verdict.status)

    def test_no_edge_on_a_big_sample_is_inconclusive(self) -> None:
        verdict = stats.edge_verdict(_sample(0.0, 4.0, 900))
        self.assertTrue(verdict.inconclusive)

    def test_a_hot_streak_on_few_trades_claims_nothing(self) -> None:
        """The property that was violated.

        Thirty trades averaging +3.5 pips produce an interval entirely above
        zero. Under a naive reading that is a proven edge. It is not: it is an
        under-powered sample, and the module must refuse it.
        """
        verdict = stats.edge_verdict(_sample(3.5, 5.0, 30))
        self.assertTrue(verdict.inconclusive)
        self.assertLess(verdict.n, verdict.min_samples)

    def test_the_minimum_sample_gate_is_enforced(self) -> None:
        """Even a gigantic, clean, entirely-positive interval cannot pass under-powered."""
        data = [5.0] * 150  # absurdly consistent, interval is a point above zero
        verdict = stats.edge_verdict(data)
        self.assertTrue(verdict.inconclusive)
        self.assertFalse(verdict.demonstrated)

    def test_inconclusive_never_claims_there_is_no_edge(self) -> None:
        """Absence of evidence is not evidence of absence.

        This is the sentence most likely to be misread by someone skimming the
        output, so it is asserted directly rather than left to the wording.
        """
        verdict = stats.edge_verdict(_sample(0.0, 4.0, 300))
        self.assertTrue(verdict.inconclusive)
        self.assertFalse(verdict.refuted)

    def test_refuted_requires_the_whole_interval_below_zero(self) -> None:
        """Calling something a loser is a strong claim and needs the same bar as winning."""
        verdict = stats.edge_verdict(_sample(-0.05, 4.0, 250))
        self.assertTrue(verdict.inconclusive)

    def test_empty_history_is_inconclusive(self) -> None:
        verdict = stats.edge_verdict([])
        self.assertEqual(0, verdict.n)
        self.assertTrue(verdict.inconclusive)
        self.assertIn("no settled trades", verdict.describe())

    def test_statuses_are_mutually_exclusive(self) -> None:
        for mean in (-2.0, -0.2, 0.0, 0.2, 2.0):
            verdict = stats.edge_verdict(_sample(mean, 4.0, 500, seed=int(mean * 10) + 50))
            flags = [verdict.demonstrated, verdict.refuted, verdict.inconclusive]
            self.assertEqual(1, sum(1 for flag in flags if flag))

    def test_describe_always_states_the_sample_size(self) -> None:
        """A verdict quoted without its evidence is how a number becomes a promise."""
        for data in (_sample(2.0, 3.0, 600), _sample(-2.0, 3.0, 600), _sample(0.0, 3.0, 600), _sample(1.0, 3.0, 20)):
            text = stats.edge_verdict(data).describe()
            self.assertRegex(text, r"\d+ (trades|paths)")


class RequiredSamplesTests(unittest.TestCase):
    def test_more_precision_needs_more_data(self) -> None:
        coarse = stats.required_samples(6.0, 1.0)
        fine = stats.required_samples(6.0, 0.25)
        self.assertGreater(fine, coarse)

    def test_lower_confidence_needs_less_data(self) -> None:
        strict = stats.required_samples(6.0, 0.5, confidence=0.99)
        loose = stats.required_samples(6.0, 0.5, confidence=0.90)
        self.assertGreater(strict, loose)

    def test_an_unreachable_target_has_no_answer(self) -> None:
        self.assertEqual(-1, stats.required_samples(6.0, 0.0))
        self.assertEqual(-1, stats.required_samples(6.0, -1.0))

    def test_matches_the_normal_approximation(self) -> None:
        # n = (z * sd / target)^2 = (1.96 * 6 / 2)^2 = 34.57 -> 35
        self.assertEqual(35, stats.required_samples(6.0, 2.0))

    def test_required_samples_sets_the_interval_width(self) -> None:
        """The contract of the formula: at n trades the CI half-width = target.

        Deliberately NOT a claim that n trades guarantee a verdict. Sizing on
        width is a ~50%-power rule: at exactly n, whether the interval clears
        zero is close to a coin toss, and a sample that happens to draw a low
        mean stays inconclusive. That is correct, and it is why the verdict
        carries a floor as well as this estimate.
        """
        sd, target = 4.0, 1.0
        needed = stats.required_samples(sd, target)
        # block_size=1 because the normal formula assumes independence. The
        # production default resamples in blocks and is deliberately wider, so
        # comparing it to an iid formula would be testing the wrong thing.
        verdict = stats.edge_verdict(_sample(target, sd, needed, seed=11), block_size=1)
        half_width = (verdict.ci_high - verdict.ci_low) / 2.0
        self.assertLess(abs(half_width - target), 0.25)

    def test_the_interval_is_never_tighter_than_the_naive_estimate(self) -> None:
        """The safety property, stated as a test.

        Measured on iid data the block bootstrap runs slightly narrow at larger
        block sizes, and a narrow interval invents significance. So the reported
        interval is floored at the independence-assuming estimate. Every error
        this can make is in the cautious direction: a false negative costs more
        data, a false positive costs trading a system that does not work.
        """
        data = _sample(1.0, 4.0, 120, seed=11)
        z = stats._z_for(stats.DEFAULT_CONFIDENCE)
        naive_half = z * stats.stdev(data) / math.sqrt(len(data))
        for block in (1, 5, 10, 20):
            _, low, high = stats.confidence_interval(data, block_size=block)
            self.assertGreaterEqual((high - low) / 2.0 + 1e-9, naive_half)

    def test_samples_needed_is_the_stricter_of_the_two_thresholds(self) -> None:
        """A power estimate below the floor must not be quoted on its own."""
        weak = stats.edge_verdict(_sample(0.1, 4.0, 50), target_edge_pips=1.0)
        self.assertEqual(stats.MIN_SAMPLES_FOR_EDGE, weak.samples_needed)
        # A hard target pushes the power estimate above the floor.
        hard = stats.edge_verdict(_sample(0.1, 6.0, 50), target_edge_pips=0.1)
        self.assertGreater(hard.samples_needed, stats.MIN_SAMPLES_FOR_EDGE)


class MultiPathTests(unittest.TestCase):
    def test_paths_that_disagree_on_sign_are_flagged(self) -> None:
        summary = stats.summarise_paths([1.0, -1.0, 2.0, -0.5])
        self.assertTrue(summary.single_path_is_misleading)
        self.assertEqual(2, summary.profitable_paths)
        self.assertEqual(4, summary.n_paths)

    def test_paths_that_agree_are_not_flagged(self) -> None:
        self.assertFalse(stats.summarise_paths([1.0, 2.0, 0.5]).single_path_is_misleading)
        self.assertFalse(stats.summarise_paths([-1.0, -2.0, -0.5]).single_path_is_misleading)

    def test_the_unit_is_paths_not_trades(self) -> None:
        """Eight paths is eight observations, not eight trades. Saying otherwise
        would understate the evidence by orders of magnitude."""
        self.assertEqual("paths", stats.summarise_paths([1.0] * 8).verdict.unit)
        self.assertEqual("trades", stats.edge_verdict(_sample(1.0, 1.0, 300)).unit)

    def test_per_path_mean_is_reported_with_spread(self) -> None:
        summary = stats.summarise_paths([1.0, 2.0, 3.0])
        self.assertAlmostEqual(2.0, summary.mean_expectancy, places=6)
        self.assertAlmostEqual(1.0, summary.stdev_expectancy, places=6)
        self.assertEqual(3.0, summary.best)
        self.assertEqual(1.0, summary.worst)

    def test_empty_paths_are_handled(self) -> None:
        summary = stats.summarise_paths([])
        self.assertEqual(0, summary.n_paths)
        self.assertFalse(summary.single_path_is_misleading)

    def test_describe_mentions_the_sign_disagreement(self) -> None:
        text = stats.summarise_paths([1.0, -1.0, 0.5, -0.5]).describe()
        self.assertIn("disagree on the sign", text)

    def test_serialises(self) -> None:
        payload = stats.summarise_paths([1.0, -1.0]).to_dict()
        self.assertEqual(2, payload["paths"])
        self.assertIn("verdict", payload)


class BootstrapTests(unittest.TestCase):
    def test_returns_one_mean_per_iteration(self) -> None:
        means = stats.block_bootstrap_means(_sample(1.0, 2.0, 100), iterations=50)
        self.assertEqual(50, len(means))

    def test_output_is_sorted(self) -> None:
        means = stats.block_bootstrap_means(_sample(1.0, 2.0, 100), iterations=50)
        self.assertEqual(sorted(means), means)

    def test_bootstrap_means_hug_the_sample_mean(self) -> None:
        data = _sample(2.0, 3.0, 300)
        means = stats.block_bootstrap_means(data, iterations=500)
        self.assertLess(abs(statistics.mean(means) - statistics.mean(data)), 0.5)

    def test_percentile_edges(self) -> None:
        self.assertEqual(0.0, stats._percentile([0.0, 1.0, 2.0], 0.0))
        self.assertEqual(2.0, stats._percentile([0.0, 1.0, 2.0], 1.0))

    def test_single_value_percentile(self) -> None:
        self.assertEqual(7.0, stats._percentile([7.0], 0.5))

    def test_stdev_helpers(self) -> None:
        self.assertEqual(0.0, stats.stdev([1.0]))
        self.assertEqual(0.0, stats.stdev([]))
        self.assertAlmostEqual(1.0, stats.stdev([1.0, 2.0, 3.0]), places=6)


if __name__ == "__main__":
    unittest.main()
