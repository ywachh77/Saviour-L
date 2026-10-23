"""The digest reports the bad news too, with intervals attached."""

import unittest

from fusion import digest
from fusion.settle import LedgerStats


def stats(wins: int = 7, losses: int = 5, tied: int = 1, unresolved: int = 2, net: float = 4.0) -> LedgerStats:
    return LedgerStats(
        settled=wins + losses + tied + unresolved,
        wins=wins,
        losses=losses,
        tied=tied,
        unresolved=unresolved,
        net_pips=net,
        win_rate=wins / (wins + losses) if wins + losses else None,
        wilson_low=0.35,
        wilson_high=0.80,
    )


class TestDigest(unittest.TestCase):
    def test_unresolved_is_reported_and_never_counted_as_a_win(self):
        text = digest.build_digest(
            day="2026-01-05",
            signals=[],
            stats=stats(),
            unresolved=2,
            calibration_state="RAW",
            calibration_samples=12,
            model_mode="VETO",
            council_weights={},
            learning_events=[],
            key_pool={},
            runner="github-actions",
            alerts=[],
        )
        self.assertIn("unresolved 2", text)
        self.assertIn("wins 7", text)

    def test_win_rate_always_carries_its_interval(self):
        text = digest.build_digest(
            day="2026-01-05",
            signals=[],
            stats=stats(),
            unresolved=0,
            calibration_state="CALIBRATED",
            calibration_samples=90,
            model_mode="ADVISORY",
            council_weights={"breakout": 0.12},
            learning_events=[{"applied": True}],
            key_pool={"slots": [{"enabled": True, "day_used": 10, "day_limit": 800}]},
            runner="github-actions",
            alerts=[],
        )
        self.assertIn("Wilson", text)
        self.assertIn("breakout", text)

    def test_discarded_late_signals_are_disclosed(self):
        text = digest.build_digest(
            day="2026-01-05",
            signals=[
                {"delivery_state": "SENT"},
                {"delivery_state": "SUPPRESSED"},
                {"seal_state": "DISCARDED_LATE"},
            ],
            stats=stats(),
            unresolved=0,
            calibration_state="RAW",
            calibration_samples=0,
            model_mode="VETO",
            council_weights={},
            learning_events=[],
            key_pool={},
            runner="github-actions",
            alerts=[],
        )
        self.assertIn("dropped as late", text)

    def test_alerts_are_summarised(self):
        text = digest.build_digest(
            day="2026-01-05",
            signals=[],
            stats=stats(),
            unresolved=0,
            calibration_state="RAW",
            calibration_samples=0,
            model_mode="VETO",
            council_weights={},
            learning_events=[],
            key_pool={},
            runner="github-actions",
            alerts=[{"kind": "stale_feed", "message": "feed 400s old"}],
        )
        self.assertIn("stale_feed", text)

    def test_the_boundary_travels_with_every_message(self):
        text = digest.build_digest(
            day="2026-01-05",
            signals=[],
            stats=stats(),
            unresolved=0,
            calibration_state="RAW",
            calibration_samples=0,
            model_mode="VETO",
            council_weights={},
            learning_events=[],
            key_pool={},
            runner="github-actions",
            alerts=[],
        )
        self.assertIn("No broker order", text)


class TestStatus(unittest.TestCase):
    def test_status_shows_pause_state_and_intervals(self):
        from fusion.config import Config

        text = digest.build_status(
            config=Config(),
            paused=True,
            runner="github-actions",
            heartbeat_row=None,
            stats=stats(),
            unresolved=2,
            model_mode="VETO",
            calibration_state="RAW",
            calibration_samples=0,
            key_pool={},
            last_cycle=None,
            store_errors=[],
        )
        self.assertIn("PAUSED", text)
        self.assertIn("Wilson", text)
        self.assertIn("heartbeat: missing", text)


class TestAlerts(unittest.TestCase):
    def test_every_human_action_kind_has_a_readable_message(self):
        known = {
            digest.ALERT_STALE_FEED,
            digest.ALERT_DELIVERY_UNKNOWN,
            digest.ALERT_KEY_POOL,
            digest.ALERT_NEGATIVE_EXPECTANCY,
            digest.ALERT_ROLLBACK,
            digest.ALERT_STORE_ERROR,
        }
        self.assertEqual(set(digest.HUMAN_ACTION_ALERTS), known)
        for kind in known:
            self.assertNotEqual(digest.alert_message(kind, "detail"), "Attention needed: detail")

    def test_alert_text_names_the_problem(self):
        text = digest.alert_message(digest.ALERT_DELIVERY_UNKNOWN, "signal abc may not have landed")
        self.assertIn("Delivery state unknown", text)
        self.assertIn("signal abc", text)


if __name__ == "__main__":
    unittest.main()
