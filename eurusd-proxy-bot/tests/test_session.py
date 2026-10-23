"""Session classification and the strategy mapping that depends on it.

The claim under test is narrow: the mapping says continuation rules suit London
and the overlap, and reversion rules suit Asia. ``scripts/validate_edge.py``
tests whether that mapping actually separates the families on data built to
contain the effect; these tests pin the mapping itself so it cannot drift
quietly.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timezone

from fusion import session


def _at(hour: int) -> datetime:
    return datetime(2026, 3, 2, hour, 0, tzinfo=timezone.utc)


class ClassificationTests(unittest.TestCase):
    def test_every_hour_has_exactly_one_session(self) -> None:
        """A gap here would silently drop bars out of the gate."""
        for hour in range(24):
            self.assertIn(session.session_of(_at(hour)), session.ALL_SESSIONS)

    def test_windows_cover_all_hours_without_overlap(self) -> None:
        covered = []
        for name in session.ALL_SESSIONS:
            start, end = session.window_of(name)
            covered.extend(range(start, end))
        self.assertEqual(sorted(range(24)), sorted(covered))

    def test_boundary_hours(self) -> None:
        self.assertEqual(session.ASIA, session.session_of(_at(0)))
        self.assertEqual(session.ASIA, session.session_of(_at(6)))
        self.assertEqual(session.LONDON, session.session_of(_at(7)))
        self.assertEqual(session.LONDON, session.session_of(_at(12)))
        self.assertEqual(session.OVERLAP, session.session_of(_at(13)))
        self.assertEqual(session.OVERLAP, session.session_of(_at(16)))
        self.assertEqual(session.NEW_YORK, session.session_of(_at(17)))
        self.assertEqual(session.NEW_YORK, session.session_of(_at(20)))
        self.assertEqual(session.OFF_HOURS, session.session_of(_at(21)))
        self.assertEqual(session.OFF_HOURS, session.session_of(_at(23)))

    def test_minutes_do_not_move_the_session(self) -> None:
        """Only the hour matters; :59 is still the same session as :00."""
        stamp = datetime(2026, 3, 2, 12, 59, tzinfo=timezone.utc)
        self.assertEqual(session.LONDON, session.session_of(stamp))


class MappingTests(unittest.TestCase):
    def test_continuation_rules_speak_in_london_and_the_overlap(self) -> None:
        for name in ("trend_pullback", "breakout", "scalp_impulse", "scalp_micro_breakout"):
            self.assertTrue(session.session_allows(session.LONDON, name), name)
            self.assertTrue(session.session_allows(session.OVERLAP, name), name)

    def test_reversion_rules_speak_in_asia_and_off_hours(self) -> None:
        for name in ("range_reversal", "scalp_reversion", "scalp_pivot_fade"):
            self.assertTrue(session.session_allows(session.ASIA, name), name)
            self.assertTrue(session.session_allows(session.OFF_HOURS, name), name)

    def test_continuation_is_silenced_in_asia(self) -> None:
        """The whole point: the same rule is not trusted in the wrong hour."""
        for name in ("trend_pullback", "breakout", "scalp_impulse"):
            self.assertFalse(session.session_allows(session.ASIA, name), name)

    def test_reversion_is_silenced_in_london(self) -> None:
        for name in ("range_reversal", "scalp_reversion", "scalp_pivot_fade"):
            self.assertFalse(session.session_allows(session.LONDON, name), name)

    def test_the_core_gate_is_never_silenced(self) -> None:
        """It is the only member that can originate a signal. Muting it by hour
        would mute the bot for hours rather than change what it believes."""
        for name in session.ALL_SESSIONS:
            self.assertTrue(session.session_allows(name, "orbix_confluence"), name)

    def test_model_backed_members_speak_everywhere(self) -> None:
        for name in session.ALL_SESSIONS:
            self.assertTrue(session.session_allows(name, "ml_consensus"), name)
            self.assertTrue(session.session_allows(name, "orbix_cadence"), name)

    def test_new_york_allows_both_families(self) -> None:
        """It trends while London is open and ranges once London has closed."""
        self.assertTrue(session.session_allows(session.NEW_YORK, "trend_pullback"))
        self.assertTrue(session.session_allows(session.NEW_YORK, "range_reversal"))

    def test_an_unmapped_strategy_is_heard_not_muted(self) -> None:
        """A prototype nobody added to the table should be judged on its record,
        not silenced by an omission in a lookup."""
        self.assertTrue(session.session_allows(session.ASIA, "some_future_rule"))

    def test_an_unknown_session_does_not_silence_either(self) -> None:
        self.assertTrue(session.session_allows("MARS", "trend_pullback"))

    def test_every_registered_member_is_in_at_least_one_session(self) -> None:
        from fusion.strategies import MEMBERS

        for name in MEMBERS:
            heard = [s for s in session.ALL_SESSIONS if session.session_allows(s, name)]
            self.assertTrue(heard, f"{name} is silenced in every session")


class DescriptionTests(unittest.TestCase):
    def test_every_session_has_a_description(self) -> None:
        for name in session.ALL_SESSIONS:
            self.assertIsInstance(session.describe(name), str)
            self.assertTrue(session.describe(name))

    def test_unknown_session_falls_back_to_its_name(self) -> None:
        self.assertEqual("MARS", session.describe("MARS"))

    def test_unknown_session_window_defaults_to_the_day(self) -> None:
        self.assertEqual((0, 24), session.window_of("MARS"))


if __name__ == "__main__":
    unittest.main()
