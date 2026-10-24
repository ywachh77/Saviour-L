"""Post-mortem memory: the bot must not pay to relearn the same lesson.

The property under test is not "it blocks things". A memory that blocks
eagerly is just a second opinion with worse manners. What matters is that it
stays quiet until the evidence is real, then refuses a setup it can
demonstrate is losing, and lets go once that stops being true.
"""

from __future__ import annotations

import unittest

from fusion import charter
from fusion.mistakes import MAX_SAMPLES, MIN_SAMPLES, MistakeMemory
from fusion.council import CouncilWeights, run_council
from fusion.features import compute_features

from .helpers import make_candles

KEY = "RANGE|LONDON|BUY"


def memory_with(wins: int, losses: int, key: str = KEY) -> MistakeMemory:
    memory = MistakeMemory()
    for _ in range(wins):
        memory.observe(key, True)
    for _ in range(losses):
        memory.observe(key, False)
    return memory


class TestItStaysQuietWithoutEvidence(unittest.TestCase):
    def test_a_bad_run_is_not_yet_a_lesson(self) -> None:
        """Twelve losses in a row feels conclusive and is not."""
        memory = memory_with(0, 12)
        self.assertFalse(memory.refuses(KEY)[0])

    def test_below_the_sample_floor_nothing_is_banned(self) -> None:
        """Checked with the worst possible record, not a middling one."""
        for losses in range(1, MIN_SAMPLES):
            memory = memory_with(0, losses)
            self.assertFalse(memory.refuses(KEY)[0], f"banned on {losses} straight losses")

    def test_an_empty_memory_bans_nothing(self) -> None:
        self.assertFalse(MistakeMemory().refuses(KEY)[0])
        self.assertEqual(MistakeMemory().banned(), [])


class TestItRefusesWhatItHasPaidToLearn(unittest.TestCase):
    def test_the_floor_is_where_evidence_starts_not_where_it_ends(self) -> None:
        """Once the sample floor is reached, an extreme record is acted on.

        Fifteen straight losses in one cell is not noise. Refusing it is the
        point of the feature; the floor exists to stop six losses doing the
        same job.
        """
        self.assertTrue(memory_with(0, MIN_SAMPLES).refuses(KEY)[0])

    def test_a_confidently_losing_setup_is_refused(self) -> None:
        memory = memory_with(10, 30)
        refused, why = memory.refuses(KEY)
        self.assertTrue(refused)
        self.assertIn("below even money", why)

    def test_the_reason_quotas_the_record(self) -> None:
        """A refusal the human cannot audit is a refusal they cannot trust."""
        memory = memory_with(10, 30)
        _refused, why = memory.refuses(KEY)
        self.assertIn("10 of 40", why)
        self.assertIn("25%", why)

    def test_a_winning_setup_is_never_refused(self) -> None:
        self.assertFalse(memory_with(30, 10).refuses(KEY)[0])

    def test_a_coin_flip_is_left_alone(self) -> None:
        """Even money is not a mistake; it is the cost gate's problem."""
        self.assertFalse(memory_with(20, 20).refuses(KEY)[0])

    def test_banned_lists_only_what_is_actually_refused(self) -> None:
        memory = memory_with(10, 30)
        memory.observe("TREND|ASIA|SELL", True)
        self.assertEqual(memory.banned(), [KEY])


class TestItOnlyLearnsFromDecidedOutcomes(unittest.TestCase):
    def test_ties_and_unresolved_teach_nothing(self) -> None:
        memory = MistakeMemory()
        for outcome in (charter.TIED, charter.UNRESOLVED):
            memory.record(outcome, KEY)
        self.assertEqual(memory.total_records(), 0)

    def test_wins_and_losses_are_counted(self) -> None:
        memory = MistakeMemory()
        memory.record(charter.WIN, KEY)
        memory.record(charter.LOSS, KEY)
        memory.record(charter.LOSS, KEY)
        self.assertEqual(memory.sample_size(KEY), 3)
        self.assertAlmostEqual(memory.win_rate(KEY), 1 / 3)


class TestItCanForgive(unittest.TestCase):
    def test_a_banned_setup_can_earn_its_way_back(self) -> None:
        memory = memory_with(10, 30)
        self.assertTrue(memory.refuses(KEY)[0])
        for _ in range(60):
            memory.observe(KEY, True)
        self.assertFalse(memory.refuses(KEY)[0], "a reformed setup stayed banned")

    def test_counts_decay_rather_than_freeze(self) -> None:
        """A hard clamp would make every ban permanent."""
        memory = MistakeMemory()
        for _ in range(MAX_SAMPLES + 20):
            memory.observe(KEY, False)
        self.assertLessEqual(memory.sample_size(KEY), MAX_SAMPLES)
        self.assertGreater(memory.sample_size(KEY), MAX_SAMPLES // 2)


class TestPersistence(unittest.TestCase):
    def test_round_trip_keeps_the_record(self) -> None:
        memory = memory_with(10, 30)
        restored = MistakeMemory.from_dict(memory.to_dict())
        self.assertEqual(restored.wins, memory.wins)
        self.assertEqual(restored.losses, memory.losses)
        self.assertTrue(restored.refuses(KEY)[0])

    def test_a_missing_payload_is_an_empty_memory(self) -> None:
        self.assertEqual(MistakeMemory.from_dict(None).total_records(), 0)


class TestTheCouncilObeysIt(unittest.TestCase):
    """The memory is only real if it can stop a signal that would otherwise fire."""

    def _fire(self, memory: MistakeMemory | None, regime: str = "RANGE", session: str = "LONDON"):
        from fusion import strategies
        from fusion.domain import Decision

        candles = make_candles(200, seed=11)
        features = compute_features(candles)[-1]
        original = dict(strategies.REGISTRY)
        strategies.REGISTRY[strategies.CORE] = lambda *a, **k: Decision("BUY", "forced", None, 6)
        try:
            return run_council(
                candles[-1],
                candles[-2].close,
                features,
                0.55,
                weights=CouncilWeights(),
                regime=regime,
                session=session,
                mistakes=memory,
            )
        finally:
            strategies.REGISTRY.clear()
            strategies.REGISTRY.update(original)

    def test_the_same_signal_fires_without_the_memory(self) -> None:
        self.assertTrue(self._fire(None).actionable)

    def test_a_banned_fingerprint_stops_it(self) -> None:
        memory = memory_with(10, 30, key="RANGE|LONDON|BUY")
        result = self._fire(memory)
        self.assertFalse(result.actionable)
        self.assertIn("Post-mortem", result.reason)

    def test_only_the_banned_fingerprint_is_stopped(self) -> None:
        """A coarse memory that bans everything is just a kill switch."""
        memory = memory_with(10, 30, key="TREND|ASIA|SELL")
        self.assertTrue(self._fire(memory).actionable)

    def test_a_different_session_still_fires(self) -> None:
        memory = memory_with(10, 30, key="RANGE|LONDON|BUY")
        self.assertTrue(self._fire(memory, session="ASIA").actionable)


if __name__ == "__main__":
    unittest.main()
