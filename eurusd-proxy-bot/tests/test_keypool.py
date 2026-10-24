"""Key pool: reserve before the call, refund, rotate, cool down, admit defeat."""

import unittest

from fusion.keypool import MAX_SLOTS, KeyPool, mask_key


class FakeClock:
    def __init__(self, start: float = 1_700_000_000.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class TestMasking(unittest.TestCase):
    def test_keys_are_never_stored_or_logged_in_full(self):
        masked = mask_key("abcdefghijklmnop")
        self.assertNotIn("abcdefghijklmnop", masked)
        self.assertIn("abcd", masked)
        self.assertEqual(mask_key(""), "")


class TestKeyPool(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.pool = KeyPool(["key-one", "key-two", "key-three"], clock=self.clock, minute_limit=2, day_limit=5)

    def test_reserve_rotates_across_slots(self):
        first = self.pool.reserve()
        second = self.pool.reserve()
        self.assertTrue(first.ok)
        self.assertNotEqual(first.key_index, second.key_index)

    def test_quota_is_reserved_before_the_call_and_refunded_on_failure(self):
        reservation = self.pool.reserve()
        self.assertEqual(self.pool.slots[reservation.key_index].minute_used, 1)
        self.pool.refund(reservation)
        self.assertEqual(self.pool.slots[reservation.key_index].minute_used, 0)

    def test_minute_quota_is_enforced_then_rotates(self):
        # minute_limit is 2 per key, so three calls must not land on one key.
        used: set[int] = set()
        for _ in range(3):
            reservation = self.pool.reserve()
            self.assertTrue(reservation.ok)
            used.add(reservation.key_index)
        self.assertGreater(len(used), 1)

    def test_exhaustion_is_reported_honestly(self):
        pool = KeyPool(["only-one"], clock=self.clock, minute_limit=1, day_limit=1)
        self.assertTrue(pool.reserve().ok)
        second = pool.reserve()
        self.assertFalse(second.ok)
        self.assertIn("exhausted", second.reason)
        self.assertEqual(pool.exhausted_events, 1)

    def test_no_keys_means_no_calls(self):
        pool = KeyPool([], clock=self.clock)
        self.assertFalse(pool.reserve().ok)
        self.assertFalse(pool.configured)

    def test_rate_limit_puts_the_key_in_cooldown(self):
        reservation = self.pool.reserve()
        self.pool.report_failure(reservation, 429)
        slot = self.pool.slots[reservation.key_index]
        self.assertGreater(slot.cooldown_until, self.clock())
        # It stays out of rotation until the cooldown passes.
        self.clock.advance(5)
        indexes = {self.pool.reserve().key_index for _ in range(3)}
        self.assertNotIn(reservation.key_index, indexes)

    def test_auth_failure_disables_the_key_permanently(self):
        reservation = self.pool.reserve()
        self.pool.report_failure(reservation, 401)
        slot = self.pool.slots[reservation.key_index]
        self.assertFalse(slot.enabled)
        self.assertIn("rejected", slot.disabled_reason)

    def test_a_local_network_fault_does_not_punish_the_key(self):
        """The call never reached the provider, so the key is not at fault.

        Credits are refunded on this path precisely because nothing was sent.
        Cooling the slot as well would charge the key for our own network and,
        since the fault is shared, would empty the pool one slot at a time.
        """
        reservation = self.pool.reserve(1)
        self.pool.report_network_error(reservation)
        self.pool.refund(reservation)
        slot = self.pool.slots[reservation.key_index]
        self.assertLessEqual(slot.cooldown_until, self.clock())
        self.assertEqual(slot.consecutive_failures, 0)
        self.assertEqual(slot.total_failures, 1)
        self.assertEqual(slot.minute_used, 0)

    def test_network_failure_gets_a_short_cooldown(self):
        reservation = self.pool.reserve()
        self.pool.report_failure(reservation, None)
        slot = self.pool.slots[reservation.key_index]
        self.assertGreater(slot.cooldown_until, self.clock())
        self.assertTrue(slot.enabled)

    def test_success_clears_the_failure_streak(self):
        reservation = self.pool.reserve()
        self.pool.report_failure(reservation, 500)
        self.assertEqual(self.pool.slots[reservation.key_index].consecutive_failures, 1)
        self.pool.report_success(reservation)
        self.assertEqual(self.pool.slots[reservation.key_index].consecutive_failures, 0)

    def test_daily_quota_rolls_over(self):
        reservation = self.pool.reserve()
        self.pool.slots[reservation.key_index].day_used = 5
        self.clock.advance(86400)
        self.assertTrue(self.pool.usable(self.pool.slots[reservation.key_index], self.clock(), 1)[0])

    def test_state_round_trip_preserves_usage(self):
        reservation = self.pool.reserve()
        state = self.pool.to_dict()
        restored = KeyPool.from_state(["key-one", "key-two", "key-three"], state, minute_limit=2, day_limit=5)
        self.assertEqual(restored.slots[reservation.key_index].minute_used, 1)

    def test_slot_count_is_capped(self):
        pool = KeyPool([f"key-{index}" for index in range(MAX_SLOTS + 12)], clock=self.clock)
        self.assertEqual(len(pool), MAX_SLOTS)

    def test_summary_never_exposes_a_key(self):
        summary = self.pool.summary()
        for row in summary["slots"]:
            self.assertNotIn("key-one", str(row))


if __name__ == "__main__":
    unittest.main()
