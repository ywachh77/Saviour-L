"""The charter values, asserted so they cannot drift quietly."""

import unittest

from fusion import charter


class TestCharter(unittest.TestCase):
    def test_synthetic_sources_are_recognised(self):
        self.assertTrue(charter.is_synthetic("SYNTHETIC"))
        self.assertTrue(charter.is_synthetic("synthetic demo"))
        self.assertTrue(charter.is_synthetic("SMOKE"))
        self.assertFalse(charter.is_synthetic("LIVE"))
        self.assertFalse(charter.is_synthetic(None))

    def test_only_real_non_tied_outcomes_are_learnable(self):
        self.assertTrue(charter.real_outcome_label(charter.WIN))
        self.assertTrue(charter.real_outcome_label(charter.LOSS))
        self.assertFalse(charter.real_outcome_label(charter.TIED))
        self.assertFalse(charter.real_outcome_label(charter.UNRESOLVED))

    def test_costs_are_charged_on_every_settlement(self):
        # A three-pip gross winner at a 1.2 pip spread nets 1.8 pips, not 3.
        self.assertAlmostEqual(charter.net_pips(3.0, 1.2), 1.8, places=9)
        # A one-pip gross winner is a net loser once the spread is paid.
        self.assertLess(charter.net_pips(1.0, 1.2), 0.0)

    def test_wilson_interval_punishes_small_samples(self):
        low, high = charter.wilson_interval(3, 3)
        # Three wins in three trades is not evidence of a 100% win rate.
        self.assertLess(low, 0.5)
        self.assertLessEqual(high, 1.0)
        wide = high - low
        narrow_low, narrow_high = charter.wilson_interval(300, 500)
        self.assertLess(narrow_high - narrow_low, wide)

    def test_wilson_handles_empty_sample(self):
        self.assertEqual(charter.wilson_interval(0, 0), (0.0, 0.0))

    def test_late_signal_limit_is_ten_seconds(self):
        self.assertEqual(charter.MAX_SEAL_LATENCY_SECONDS, 10)

    def test_heartbeat_stale_threshold_is_ninety_seconds(self):
        self.assertEqual(charter.HEARTBEAT_STALE_SECONDS, 90)

    def test_unresolved_is_never_a_win_or_a_loss(self):
        self.assertNotIn(charter.UNRESOLVED, (charter.WIN, charter.LOSS, charter.TIED))

    def test_forbidden_topics_are_documented(self):
        joined = " ".join(charter.FORBIDDEN_TOPICS).lower()
        for word in ("broker", "martingale", "guaranteed", "otc"):
            self.assertIn(word, joined)


if __name__ == "__main__":
    unittest.main()
