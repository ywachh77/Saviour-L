"""Regressions for storage bugs that would have silently disabled learning.

Each of these was a real defect found after the first build:

* ``settled=eq.false`` never matched Python's ``False`` (str "False" vs "false"),
  so ``pending_signals`` came back empty and **no signal was ever settled or
  learned from** — the bot would have run for months and trained nothing.
* reads sliced "the newest N" off an unordered page, so the freshest candles
  could be missing from the window the indicators were computed on.
* ``order=`` fell into ``**filters`` and matched a non-existent column.
"""

import shutil
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

from fusion import charter
from fusion.config import Config
from fusion.runner import FusionBot
from fusion.store import LocalStore, Store, _apply_order, _filter_value, _matches


class TestPostgRESTFilterMatching(unittest.TestCase):
    def test_boolean_filters_match_python_booleans(self):
        self.assertTrue(_matches({"settled": False}, "settled", "eq.false"))
        self.assertTrue(_matches({"settled": True}, "settled", "eq.true"))
        self.assertFalse(_matches({"settled": True}, "settled", "eq.false"))
        self.assertTrue(_matches({"settled": "false"}, "settled", "eq.false"))

    def test_prefixes_are_stripped(self):
        self.assertEqual(_filter_value("eq.EUR/USD"), "EUR/USD")
        self.assertEqual(_filter_value("gte.5"), "5")
        self.assertEqual(_filter_value("plain"), "plain")

    def test_pending_signals_actually_returns_pending_rows(self):
        directory = Path(tempfile.mkdtemp())
        try:
            store = Store(None, LocalStore(directory))
            store.save_signal({"signal_id": "a", "pair": "EUR/USD", "settled": False})
            store.save_signal({"signal_id": "b", "pair": "EUR/USD", "settled": True})
            pending = store.pending_signals("EUR/USD")
            self.assertEqual([row["signal_id"] for row in pending], ["a"])
        finally:
            shutil.rmtree(directory, ignore_errors=True)


class TestOrdering(unittest.TestCase):
    def test_descending_order_puts_newest_first(self):
        rows = [
            {"timestamp": "2026-01-01T00:00:00Z"},
            {"timestamp": "2026-01-01T00:02:00Z"},
            {"timestamp": "2026-01-01T00:01:00Z"},
        ]
        ordered = _apply_order(list(rows), "timestamp.desc")
        self.assertEqual(ordered[0]["timestamp"], "2026-01-01T00:02:00Z")

    def test_ascending_order_puts_oldest_first(self):
        rows = [{"t": "b"}, {"t": "a"}]
        self.assertEqual(_apply_order(list(rows), "t.asc")[0]["t"], "a")

    def test_read_honours_order_before_slicing(self):
        directory = Path(tempfile.mkdtemp())
        try:
            store = Store(None, LocalStore(directory))
            store.save_candles(
                [
                    {"timestamp": "2026-01-01T00:00:00Z", "pair": "EUR/USD", "close": 1.0},
                    {"timestamp": "2026-01-01T00:01:00Z", "pair": "EUR/USD", "close": 2.0},
                    {"timestamp": "2026-01-01T00:02:00Z", "pair": "EUR/USD", "close": 3.0},
                ]
            )
            newest = store.load_candles("EUR/USD", limit=1)
            self.assertEqual(len(newest), 1)
            self.assertAlmostEqual(float(newest[0]["close"]), 3.0)
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    def test_order_is_not_mistaken_for_a_filter(self):
        directory = Path(tempfile.mkdtemp())
        try:
            store = Store(None, LocalStore(directory))
            store.save_candles([{"timestamp": "2026-01-01T00:00:00Z", "pair": "EUR/USD"}])
            rows = store._read("candles_m1", order="timestamp.desc", **{"pair": "eq.EUR/USD"})
            self.assertEqual(len(rows), 1)
        finally:
            shutil.rmtree(directory, ignore_errors=True)


class TestUnresolvedGrace(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.directory, ignore_errors=True)

    def _bot(self) -> FusionBot:
        config = Config(
            pair="EUR/USD",
            horizon_minutes=3,
            local_state_dir=str(self.directory / "bot"),
            runner="regression",
        )
        return FusionBot(config=config, store=Store(None, LocalStore(self.directory / "bot")), provider=None)

    def test_a_signal_with_no_prices_stays_pending_through_the_grace_period(self):
        from fusion.domain import Signal
        from fusion.timeutil import now

        bot = self._bot()
        stamp = now().replace(second=0, microsecond=0) - timedelta(minutes=1)
        signal = Signal.create(
            pair="EUR/USD",
            side="BUY",
            entry_time=stamp,
            strategy="fusion_council",
            reason="regression",
            source="LIVE",
            generated_at=stamp,
            horizon_minutes=3,
        )
        bot.store.save_signal(dict(signal.to_dict(), settled=False, seal_state="SIGNAL"))

        inside = stamp + timedelta(minutes=3 + charter.UNRESOLVE_AFTER_MINUTES - 10)
        self.assertEqual(bot.settle_due(at=inside), 0)
        self.assertEqual(len(bot.store.load_settlements()), 0)

    def test_after_the_grace_period_it_is_recorded_as_unresolved(self):
        from fusion.domain import Signal
        from fusion.timeutil import now

        bot = self._bot()
        stamp = now().replace(second=0, microsecond=0) - timedelta(minutes=1)
        signal = Signal.create(
            pair="EUR/USD",
            side="BUY",
            entry_time=stamp,
            strategy="fusion_council",
            reason="regression",
            source="LIVE",
            generated_at=stamp,
            horizon_minutes=3,
        )
        bot.store.save_signal(dict(signal.to_dict(), settled=False, seal_state="SIGNAL"))

        past = stamp + timedelta(minutes=3 + charter.UNRESOLVE_AFTER_MINUTES + 5)
        self.assertEqual(bot.settle_due(at=past), 1)
        rows = bot.store.load_settlements()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["outcome"], charter.UNRESOLVED)
        self.assertIsNone(rows[0]["net_pips"])


if __name__ == "__main__":
    unittest.main()
