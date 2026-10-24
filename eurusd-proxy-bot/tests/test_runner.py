"""End-to-end: the bot's promises hold when the whole loop runs."""

import shutil
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import URLError

from fusion import charter
from fusion.config import Config
from fusion.domain import Candle, Settlement, Signal
from fusion.runner import FusionBot, format_signal
from fusion.simulate import SyntheticFeed, run_simulation, synthetic_series
from fusion.store import LocalStore, Store
from fusion.telegram import TelegramClient
from fusion.timeutil import parse_datetime
from tests.helpers import make_candles


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
        runner="github-actions",
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
        config = Config(local_state_dir=str(self.directory), runner="github-actions", horizon_minutes=3)
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


class TestNightlyTrainsOnHistory(unittest.TestCase):
    """Training must see the stored bars, not the few hundred the cycle keeps."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.directory = Path(self._tmp)
        self.bot = make_bot(self.directory)

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _store_bars(self, count):
        rows = []
        for index in range(count):
            stamp = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=index)
            rows.append({
                "timestamp": stamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "pair": "EUR/USD", "open": 1.10, "high": 1.11,
                "low": 1.09, "close": 1.10, "volume": 700.0,
            })
        self.bot.store.save_candles(rows)

    def test_training_sees_stored_history_not_the_live_window(self):
        """The live buffer is CANDLE_HISTORY bars wide. Training on it silently
        starved the nightly job, which needs far more to run at all."""
        self._store_bars(1500)
        self.bot.candles = []  # the in-memory window is empty this cycle

        candles = self.bot._training_candles()
        self.assertEqual(len(candles), 1500)
        self.assertTrue(all(isinstance(item, Candle) for item in candles))

    def test_training_history_is_capped_by_config(self):
        self._store_bars(900)
        self.bot.config.training_history = 120
        self.assertEqual(len(self.bot._training_candles()), 120)

    def test_it_falls_back_to_the_live_window_when_the_store_is_empty(self):
        """No history yet: behave exactly as before rather than training on nothing."""
        self.bot.candles = list(synthetic_series(50, seed=1))
        self.assertEqual(len(self.bot._training_candles()), 50)


class TestSchedulingSurvivesACron(unittest.TestCase):
    """A cron cannot land on an arbitrary minute, so neither can the bot rely on one."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.directory = Path(self._tmp)
        self.bot = make_bot(self.directory)

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _at(self, hour, minute, day=6):
        return datetime(2026, 1, day, hour, minute, tzinfo=timezone.utc)

    def test_the_exact_minute_still_fires(self):
        self.assertTrue(self.bot.due_nightly(self._at(23, 55)))

    def test_a_half_hourly_cron_catches_the_nightly(self):
        """23:55 is not on a :00/:30 boundary. Without slack it never runs."""
        self.assertTrue(self.bot.due_nightly(self._at(0, 0, day=7)))
        self.assertTrue(self.bot.due_nightly(self._at(0, 30, day=7)))

    def test_a_half_hourly_cron_catches_the_digest(self):
        self.assertTrue(self.bot.due_digest(self._at(0, 30, day=7)))

    def test_it_does_not_fire_early(self):
        self.assertFalse(self.bot.due_nightly(self._at(23, 0)))

    def test_the_grace_window_expires(self):
        """Otherwise a run at noon would think the nightly was still due."""
        self.assertFalse(self.bot.due_nightly(self._at(6, 0, day=7)))
        self.assertFalse(self.bot.due_digest(self._at(6, 0, day=7)))

    def test_the_previous_days_job_does_not_leak(self):
        """At 23:00 the due job is yesterday's, long past its window."""
        self.assertFalse(self.bot.due_nightly(self._at(23, 0)))
        self.assertTrue(self.bot.due_nightly(self._at(23, 56)))


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

    def test_help_lists_every_command_and_nothing_else(self):
        text = self.bot.handle_command("/help")
        for command in ("/start", "/help", "/signals", "/status", "/pause", "/resume"):
            self.assertIn(command, text)
        self.assertNotIn("/backfill", text)
        self.assertNotIn("/train", text)

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


class TestShortLivedContainerDoesNotRepeatItself(unittest.TestCase):
    """The bot runs in containers that die after a few minutes.

    Any 'already done this' flag held only in memory is wiped between cycles,
    so the human gets the same alert and the same command reply over and over.
    These tests restart the bot against the same store to prove it does not.
    """

    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.directory, ignore_errors=True)

    def _store(self) -> Store:
        return Store(None, LocalStore(self.directory))

    def _bot(self, store: Store, transport=None) -> FusionBot:
        config = Config(local_state_dir=str(self.directory), runner="github-actions", horizon_minutes=3)
        client = TelegramClient("t", "1", opener=transport) if transport else None
        return FusionBot(config=config, store=store, provider=None, telegram=client)

    def test_an_alert_is_not_re_sent_after_the_container_restarts(self):
        store = self._store()
        transport = ScriptedTransport()
        first = self._bot(store, transport)
        first._alert_once("stale_feed", "feed is old", key="stale_feed|2026-10-24")
        self.assertEqual(len(transport.sent), 1, "the first container tells the human once")

        second_transport = ScriptedTransport()
        second = self._bot(store, second_transport)
        second._alert_once("stale_feed", "feed is old", key="stale_feed|2026-10-24")
        self.assertEqual(
            second_transport.sent, [], "a restarted container must not repeat an alert already sent"
        )

    def test_a_different_key_still_raises_its_own_alert(self):
        store = self._store()
        transport = ScriptedTransport()
        bot = self._bot(store, transport)
        bot._alert_once("stale_feed", "feed is old", key="stale_feed|2026-10-24")
        bot._alert_once("stale_feed", "feed is old", key="stale_feed|2026-10-25")
        self.assertEqual(len(transport.sent), 2, "a new day is a new alert, not a duplicate of yesterday")

    def test_the_stale_feed_streak_survives_a_restart(self):
        """Five stale minutes is two and a half hours on a half-hourly cron.

        One cycle evaluates one minute, so a counter that lives only in memory
        tops out at one and the alert is unreachable. It has to be written down.
        """
        store = self._store()
        candles = make_candles(200, seed=17)
        stale_at = candles[-1].timestamp + timedelta(minutes=30)
        bucket = "2026-10-24T00:00"

        early = ScriptedTransport()
        for _ in range(4):
            bot = self._bot(store, early)
            bot.startup()
            bot.candles = candles
            bot.evaluate_minute(bucket=bucket, at=stale_at)
        self.assertEqual(early.sent, [], "four stale minutes is below the threshold of five")

        late = ScriptedTransport()
        final = self._bot(store, late)
        final.startup()
        self.assertEqual(final.stale_feed_minutes, 4, "the streak arrived with the new container")
        final.candles = candles
        final.evaluate_minute(bucket=bucket, at=stale_at)
        self.assertEqual(len(late.sent), 1, "the fifth stale minute finally raises the alert")

    def test_a_healthy_minute_clears_the_streak(self):
        store = self._store()
        candles = make_candles(200, seed=17)
        bot = self._bot(store, ScriptedTransport())
        bot.startup()
        bot.candles = candles
        bot.evaluate_minute(bucket="2026-10-24T00:00", at=candles[-1].timestamp + timedelta(minutes=30))
        self.assertEqual(bot.stale_feed_minutes, 1)
        bot.evaluate_minute(bucket="2026-10-24T00:01", at=candles[-1].timestamp + timedelta(minutes=1))
        self.assertEqual(bot.stale_feed_minutes, 0, "a fresh minute resets the count")

    def test_the_telegram_cursor_is_carried_into_the_next_container(self):
        updates = b'{"ok":true,"result":[{"update_id":100,"message":{"text":"/pause","chat":{"id":"1"}}}]}'
        empty = b'{"ok":true,"result":[]}'

        store = self._store()
        first_transport = ScriptedTransport(updates)
        first = self._bot(store, first_transport)
        first.poll_commands()
        self.assertTrue(first.paused, "the owner's /pause was applied")

        second_transport = ScriptedTransport(empty)
        second = self._bot(store, second_transport)
        second.startup()
        second.poll_commands()

        self.assertTrue(second.paused, "the paused state itself survives, independently of the cursor")
        get_updates = second_transport.sent[0]
        self.assertIn(
            "offset=101",
            get_updates,
            "the next container must acknowledge the update it already answered",
        )
        self.assertEqual(
            len(second_transport.sent),
            1,
            "no command reply should be sent: /pause was handled before this container existed",
        )


class TestOnDemandSignals(unittest.TestCase):
    """/signals must read the market, not manufacture an answer.

    These use candles anchored to the current minute on purpose. The shared
    helper starts history in January 2026, so a bot handed those bars is
    always looking at data eight months old and the freshness gate refuses it
    -- which makes every assertion below pass for the wrong reason.
    """

    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.directory, ignore_errors=True)

    @staticmethod
    def _fresh_candles(count: int = 200, seed: int = 11):
        end = datetime.now(timezone.utc).replace(second=0, microsecond=0) - timedelta(minutes=1)
        return make_candles(count, seed=seed, start=end - timedelta(minutes=count - 1))

    def _bot(self, transport=None, candles=None) -> FusionBot:
        # adaptive_horizon matches the deployed workflow: a hold too short to
        # pay is stretched, not taken.
        config = Config(local_state_dir=str(self.directory), runner="github-actions",
                        horizon_minutes=3, adaptive_horizon=True, max_horizon_minutes=120)
        client = TelegramClient("t", "1", opener=transport) if transport else None
        bot = FusionBot(config=config, store=Store(None, LocalStore(self.directory)),
                        provider=None, telegram=client)
        if candles is not None:
            bot.candles = candles
        return bot

    def test_the_brief_format_carries_only_the_four_trade_facts(self):
        from fusion.runner import format_brief_signal
        signal = Signal.create(
            pair="EUR/USD", side="BUY",
            entry_time=datetime(2026, 10, 24, 10, 42, tzinfo=timezone.utc),
            strategy="fusion_council", reason="test", source="LIVE",
            generated_at=datetime(2026, 10, 24, 10, 42, tzinfo=timezone.utc),
            horizon_minutes=40, spread_pips=1.2,
        )
        text = format_brief_signal(signal)
        self.assertIn("EUR/USD", text)
        self.assertIn("BUY", text)
        self.assertIn("10:42", text)
        self.assertIn("11:22", text)
        for hidden in ("Model:", "Regime:", "Why:"):
            self.assertNotIn(hidden, text)

    def test_no_setup_is_reported_not_invented(self):
        """The core point: a request must never become a prediction."""
        transport = ScriptedTransport()
        bot = self._bot(transport, candles=self._fresh_candles(seed=5))
        reply = bot.on_demand_signal()
        self.assertEqual(transport.sent, [], "nothing may be sent when there is no setup")
        self.assertIn("No setup", reply)
        self.assertNotIn("EUR/USD BUY", reply)

    def test_a_real_setup_is_delivered_briefly(self):
        """When the gates do agree, the phone gets the short version."""
        from fusion import strategies
        from fusion.domain import Decision

        transport = ScriptedTransport()
        bot = self._bot(transport, candles=self._fresh_candles(seed=11))
        original = dict(strategies.REGISTRY)
        strategies.REGISTRY[strategies.CORE] = lambda *a, **k: Decision("BUY", "forced", None, 6)
        try:
            reply = bot.on_demand_signal()
        finally:
            strategies.REGISTRY.clear()
            strategies.REGISTRY.update(original)

        self.assertIn("EUR/USD", reply)
        self.assertIn("BUY", reply)
        self.assertIn("Entry", reply)
        self.assertIn("Expiry", reply)
        self.assertEqual(len(transport.sent), 1)
        self.assertIn("Entry", transport.sent[0])

    def test_the_prediction_is_about_a_future_minute(self):
        """Asked mid-minute, the call must be for a bar that has not opened.

        Entry prices that already happened are backdated, and the charter
        discards those. Targeting the next minute is what makes /signals
        usable at any second rather than only in the first ten.
        """
        from fusion import strategies
        from fusion.domain import Decision

        bot = self._bot(ScriptedTransport(), candles=self._fresh_candles(seed=11))
        original = dict(strategies.REGISTRY)
        strategies.REGISTRY[strategies.CORE] = lambda *a, **k: Decision("BUY", "forced", None, 6)
        try:
            reply = bot.on_demand_signal()
        finally:
            strategies.REGISTRY.clear()
            strategies.REGISTRY.update(original)
        if "No setup" in reply:
            self.skipTest("gates declined on this synthetic path")
        for row in self._load_signals():
            entry = parse_datetime(row["entry_time"])
            self.assertGreater(entry, datetime.now(timezone.utc) - timedelta(seconds=5),
                               "entry must be now or in the future, never in the past")

    def _load_signals(self):
        return Store(None, LocalStore(self.directory))._read("signals")

    def test_it_refuses_while_paused(self):
        bot = self._bot(ScriptedTransport(), candles=self._fresh_candles(seed=5))
        bot.paused = True
        self.assertIn("paused", bot.on_demand_signal())

    def test_it_says_so_when_there_is_no_data(self):
        bot = self._bot()
        self.assertIn("No price data", bot.on_demand_signal())


class TestResultReporting(unittest.TestCase):
    """Every prediction must report what it did, including when it did nothing."""

    def _signal(self):
        stamp = datetime(2026, 10, 24, 10, 42, tzinfo=timezone.utc)
        return Signal.create(
            pair="EUR/USD", side="BUY", entry_time=stamp, strategy="fusion_council",
            reason="test", source="LIVE", generated_at=stamp, horizon_minutes=5, spread_pips=1.2,
        )

    def _settlement(self, outcome, gross, net, spread=1.2):
        return Settlement(
            signal_id="s1", outcome=outcome, entry_price=1.10000, exit_price=1.10000 + gross / 10000,
            gross_pips=gross, move_pips=gross, net_pips=net, spread_pips=spread,
            settled_at=datetime(2026, 10, 24, 10, 47, tzinfo=timezone.utc), source="LIVE",
        )

    def test_a_win_reports_net_not_gross(self):
        from fusion.runner import format_result
        text = format_result(self._signal(), self._settlement(charter.WIN, 2.9, 1.7))
        self.assertIn("WIN", text)
        self.assertIn("+1.7", text)
        self.assertIn("cost 1.2", text)

    def test_a_loss_is_still_reported(self):
        """A bot that only reports its wins is marketing, not research."""
        from fusion.runner import format_result
        text = format_result(self._signal(), self._settlement(charter.LOSS, -1.0, -2.2))
        self.assertIn("LOSS", text)

    def test_unresolved_claims_no_number(self):
        from fusion.runner import format_result
        text = format_result(self._signal(), self._settlement(charter.UNRESOLVED, 0.0, 0.0))
        self.assertIn("UNRESOLVED", text)
        self.assertNotIn("0.0", text.split("\n")[-1])

    def test_a_tie_is_a_tie(self):
        from fusion.runner import format_result
        text = format_result(self._signal(), self._settlement(charter.TIED, 0.0, -1.2))
        self.assertIn("TIED", text)

    def test_the_result_is_sent_when_a_real_signal_settles(self):
        directory = Path(tempfile.mkdtemp())
        try:
            transport = ScriptedTransport()
            config = Config(local_state_dir=str(directory), runner="github-actions", horizon_minutes=3)
            bot = FusionBot(config=config, store=Store(None, LocalStore(directory)),
                            provider=None, telegram=TelegramClient("t", "1", opener=transport))
            signal = self._signal()
            bot.deliver_result(signal, self._settlement(charter.WIN, 2.9, 1.7))
            self.assertEqual(len(transport.sent), 1)
            self.assertIn("RESULT", transport.sent[0])
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    def test_a_result_key_is_not_mistaken_for_a_signal_id(self):
        self.assertIsNone(FusionBot._signal_id_from_key("result|abc123"))
        self.assertEqual(FusionBot._signal_id_from_key("signal|abc123"), "abc123")


if __name__ == "__main__":
    unittest.main()
