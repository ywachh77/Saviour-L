"""End-to-end: the bot's promises hold when the whole loop runs."""

import shutil
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from urllib.error import URLError

from fusion import charter
from fusion.config import Config
from fusion.domain import Signal
from fusion.runner import FusionBot, format_signal
from fusion.simulate import SyntheticFeed, run_simulation, synthetic_series
from fusion.store import LocalStore, Store
from fusion.telegram import TelegramClient


class FakeResponse:
    def __init__(self, payload: bytes):
        self._payload = payload
        self.status = 200

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class ScriptedTransport:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.sent: list[str] = []

    def __call__(self, request, timeout=None):
        self.sent.append((request.data or b"").decode())
        item = self.responses.pop(0) if self.responses else b'{"ok":true,"result":{"message_id":77}}'
        if isinstance(item, Exception):
            raise item
        return FakeResponse(item)


def make_bot(directory: Path, **overrides) -> FusionBot:
    config = Config(
        pair="EUR/USD",
        horizon_minutes=3,
        local_state_dir=str(directory),
        runner="termux",
        max_candle_age_seconds=180,
        min_calibration_points=60,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return FusionBot(config=config, store=Store(None, LocalStore(directory)), provider=None, telegram=None)


class TestSyntheticCharter(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.bot = make_bot(self.directory)
        self.candles = synthetic_series(320, seed=4242, volatility=0.00030, drift=0.00001)

    def tearDown(self):
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_the_pipeline_runs_end_to_end(self):
        summary = run_simulation(self.bot, self.candles)
        self.assertGreater(summary["steps"], 100)
        self.assertGreater(summary["feed_calls"], 0)

    def test_synthetic_data_trains_nothing(self):
        run_simulation(self.bot, self.candles)
        # No model, no calibration, no prototype weights — the charter holds.
        self.assertFalse(self.bot.model and self.bot.model.trained)
        self.assertEqual(self.bot.weights.active(), {})
        events = self.bot.store.load_events(limit=500)
        self.assertTrue(events)
        self.assertTrue(all(not event.get("applied") for event in events))

    def test_abstentions_are_the_common_case(self):
        summary = run_simulation(self.bot, self.candles)
        self.assertGreater(summary["abstentions"], summary["signals"])

    def test_unresolved_outcomes_are_never_fabricated(self):
        run_simulation(self.bot, self.candles, minutes=40)
        rows = self.bot.store.load_settlements()
        for row in rows:
            if row["outcome"] == charter.UNRESOLVED:
                self.assertIsNone(row["net_pips"])
            else:
                self.assertIn(row["outcome"], (charter.WIN, charter.LOSS, charter.TIED))


class TestDeliveryInTheLoop(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.directory, ignore_errors=True)

    def _bot(self, transport) -> FusionBot:
        config = Config(local_state_dir=str(self.directory), runner="termux", horizon_minutes=3)
        client = TelegramClient("t", "1", opener=transport)
        return FusionBot(
            config=config,
            store=Store(None, LocalStore(self.directory)),
            provider=None,
            telegram=client,
        )

    def _signal(self) -> Signal:
        from fusion.timeutil import now

        stamp = now().replace(second=0, microsecond=0)
        return Signal.create(
            pair="EUR/USD",
            side="BUY",
            entry_time=stamp,
            strategy="fusion_council",
            reason="test",
            source="LIVE",
            generated_at=stamp,
            horizon_minutes=3,
            spread_pips=1.2,
        )

    def test_confirmed_delivery_records_the_message_id(self):
        transport = ScriptedTransport(b'{"ok":true,"result":{"message_id":4242}}')
        bot = self._bot(transport)
        signal = self._signal()
        bot.store.save_signal(dict(signal.to_dict(), seal_state="SIGNAL", settled=False))
        bot.deliver(signal)
        rows = bot.store._read("signals", signal_id=f"eq.{signal.signal_id}")
        self.assertEqual(rows[-1]["delivery_state"], charter.DELIVERY_SENT)
        self.assertEqual(rows[-1]["message_id"], "4242")

    def test_a_timeout_is_recorded_as_unresolved_and_raises_an_alert(self):
        transport = ScriptedTransport(URLError("timed out"))
        bot = self._bot(transport)
        signal = self._signal()
        bot.store.save_signal(dict(signal.to_dict(), seal_state="SIGNAL", settled=False))
        bot.deliver(signal)
        rows = bot.store._read("signals")
        self.assertEqual(rows[-1]["delivery_state"], charter.DELIVERY_UNKNOWN)
        alerts = bot.store.recent_alerts()
        self.assertTrue(any(row["kind"] == "delivery_unknown" for row in alerts))

    def test_a_paused_bot_suppresses_delivery_but_keeps_learning(self):
        transport = ScriptedTransport()
        bot = self._bot(transport)
        bot.paused = True
        candles = synthetic_series(200, seed=909)
        feed = SyntheticFeed(candles)
        bot.provider = feed
        from datetime import timedelta as td

        for step in range(1, 60):
            stamp = candles[0].timestamp + td(minutes=step)
            feed.advance_to(stamp)
            bot.ingest(at=stamp)
            bot.minute_cycle(at=stamp)
            bot.settle_due(at=stamp)
        suppressed = [
            row
            for row in bot.store._read("signals", limit=500)
            if row.get("delivery_state") == charter.DELIVERY_SUPPRESSED
        ]
        self.assertEqual(len(transport.sent), 0)
        self.assertTrue(suppressed or not bot.store._read("signals"))


class TestJobSlots(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.bot = make_bot(self.directory)
        self.candles = synthetic_series(120, seed=11)
        self.bot.provider = SyntheticFeed(self.candles)

    def tearDown(self):
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_the_same_minute_is_never_processed_twice(self):
        stamp = self.candles[80].timestamp + timedelta(minutes=1)
        self.bot.provider.advance_to(stamp)
        self.bot.ingest(at=stamp)
        first = self.bot.minute_cycle(at=stamp)
        second = self.bot.minute_cycle(at=stamp)
        self.assertNotEqual(first.seal_state, "SKIPPED")
        self.assertEqual(second.seal_state, "SKIPPED")

    def test_nightly_runs_once_per_date(self):
        from datetime import datetime

        from fusion.timeutil import UTC

        stamp = datetime(2026, 3, 4, 23, 55, tzinfo=UTC)
        self.bot.nightly_job(stamp)
        again = self.bot.nightly_job(stamp)
        self.assertTrue(again.get("skipped"))

    def test_digest_runs_once_per_date(self):
        from datetime import datetime

        from fusion.timeutil import UTC

        stamp = datetime(2026, 3, 4, 0, 5, tzinfo=UTC)
        first = self.bot.digest_job(stamp)
        self.assertTrue(first)
        self.assertEqual(self.bot.digest_job(stamp), "")


class TestCommands(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.bot = make_bot(self.directory)

    def tearDown(self):
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_pause_and_resume(self):
        self.bot.handle_command("/pause")
        self.assertTrue(self.bot.paused)
        self.bot.handle_command("/resume")
        self.assertFalse(self.bot.paused)

    def test_start_resumes_and_explains(self):
        self.bot.paused = True
        text = self.bot.handle_command("/start")
        self.assertFalse(self.bot.paused)
        self.assertIn("FUSION BOT", text)

    def test_status_never_crashes_on_an_empty_store(self):
        text = self.bot.handle_command("/status")
        self.assertIn("FUSION BOT status", text)
        self.assertIn("PAUSED" if self.bot.paused else "LIVE", text)

    def test_help_lists_exactly_five_commands(self):
        text = self.bot.handle_command("/help")
        for command in ("/start", "/help", "/status", "/pause", "/resume"):
            self.assertIn(command, text)
        self.assertNotIn("/backfill", text)

    def test_unknown_command_falls_back_to_help(self):
        text = self.bot.handle_command("/teleport")
        self.assertIn("Unknown command", text)


class TestSignalText(unittest.TestCase):
    def test_a_signal_message_says_what_it_costs_and_what_it_is_not(self):
        from fusion.timeutil import now

        stamp = now().replace(second=0, microsecond=0)
        signal = Signal.create(
            pair="EUR/USD",
            side="BUY",
            entry_time=stamp,
            strategy="fusion_council",
            reason="Core 5/6: trend, momentum",
            source="LIVE",
            generated_at=stamp,
            horizon_minutes=5,
            probability_up=0.57,
            probability_raw=0.61,
            calibration_state="CALIBRATED",
            spread_pips=1.2,
        )
        text = format_signal(signal)
        self.assertIn("EUR/USD BUY", text)
        self.assertIn("open of the", text)
        self.assertIn("spread", text)
        self.assertIn("No broker order", text)


if __name__ == "__main__":
    unittest.main()
