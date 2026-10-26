"""Zero-touch runtime: one loop, minute-aligned, no command to remember.

The bot owns its own schedule:

* just after each minute boundary → ingest the candle that just closed, run the
  pipeline, seal or abstain, deliver or drop;
* continuously → heartbeat, Telegram commands, outbox flush, settlement;
* 23:55 UTC → nightly walk-forward with promote/rollback gates;
* 00:05 UTC → daily digest.

Job slots make every scheduled step idempotent, so a restart, a double trigger
or an Actions overlap cannot run the same minute twice.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import signal
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Sequence

from . import charter, instruments, digest, heartbeat, learning, needle_agent, protections, settle, stats
from .config import Config
from .council import CouncilWeights, Vote
from .domain import Candle, Signal, closed_before
from .engine import FusionEngine
from .keypool import KeyPool
from .model import MODE_VETO, BootstrapEnsemble, OnlineLogisticModel
from .mistakes import MistakeMemory
from .provider import TwelveDataProvider
from .session import session_of
from .store import Store, build_store
from .telegram import (
    HELP_TEXT,
    TelegramClient,
    TelegramFormatConfig,
    render_result_message,
    render_signal_message,
)
from .timeutil import (
    floor_minute,
    format_display_hm,
    minute_bucket,
    next_future_candle_open,
    now,
    parse_datetime,
    seconds_until_next_minute,
)

# How far past the minute boundary the pipeline is allowed to seal before the
# signal is considered late and dropped.
SEAL_OFFSET_SECONDS = 0.5
COMMAND_POLL_SECONDS = 2.0
MAX_STORED_CANDLES = 750
SAVE_STATE_EVERY = 1


def format_signal(signal: Signal, fmt: TelegramFormatConfig | None = None) -> str:
    return render_signal_message(signal, fmt, brief=False)


def format_brief_signal(signal: Signal, fmt: TelegramFormatConfig | None = None) -> str:
    """Pair, direction, entry, expiry (unless custom format overrides /signals)."""
    return render_signal_message(signal, fmt, brief=True)


def format_result(signal: Signal, settlement, fmt: TelegramFormatConfig | None = None) -> str:
    """What the prediction actually did, sent the moment it is known."""
    return render_result_message(signal, settlement, fmt)


@dataclass
class CycleReport:
    bucket: str
    seal_state: str
    reason: str
    regime: str = "UNKNOWN"
    signal_id: str | None = None

    def to_dict(self) -> dict:
        return {
            "bucket": self.bucket,
            "seal_state": self.seal_state,
            "reason": self.reason,
            "regime": self.regime,
            "signal_id": self.signal_id,
        }


@dataclass
class FusionBot:
    config: Config
    store: Store = None  # type: ignore[assignment]
    provider: Any = None
    telegram: TelegramClient | None = None

    candles: list[Candle] = field(default_factory=list)
    model: OnlineLogisticModel | None = None
    ensemble: BootstrapEnsemble = field(default_factory=BootstrapEnsemble)
    weights: CouncilWeights = field(default_factory=CouncilWeights)

    paused: bool = False
    running: bool = False
    cycles: int = 0
    settled_since_save: int = 0
    last_cycle: CycleReport | None = None
    abstentions: int = 0
    stale_feed_minutes: int = 0
    alerted_today: set[str] = field(default_factory=set)
    nightly_report: dict | None = None
    calibration_points: list[tuple[float, int]] = field(default_factory=list)
    mistakes: MistakeMemory = field(default_factory=MistakeMemory)
    telegram_format: TelegramFormatConfig = field(default_factory=TelegramFormatConfig)
    shadow_recovery: dict[str, Any] = field(default_factory=lambda: {"pending": None, "cleared_at": None})
    _saved_telegram_offset: int | None = None
    peer_bots: dict[str, "FusionBot"] = field(default_factory=dict)

    # ------------------------------------------------------------------ setup
    def __post_init__(self) -> None:
        if self.store is None:
            self.store = build_store(self.config)
        if self.provider is None:
            self.provider = TwelveDataProvider(self.config.twelve_data_keys)
        if self.telegram is None and self.config.telegram_ready:
            self.telegram = TelegramClient(self.config.telegram_token, self.config.telegram_owner_chat_id)
        self.telegram_format = TelegramFormatConfig(
            signal_style=getattr(self.config, "telegram_signal_style", "standard"),
            signal_template=getattr(self.config, "telegram_signal_template", ""),
            result_style=getattr(self.config, "telegram_result_style", "standard"),
            result_template=getattr(self.config, "telegram_result_template", ""),
            apply_to_signals_cmd=bool(getattr(self.config, "telegram_apply_to_signals_cmd", False)),
        )
        self.engine = FusionEngine(self.config)
        for symbol in self.config.active_pairs[1:]:
            self.peer_bots[symbol] = FusionBot(
                config=self.config.for_pair(symbol),
                store=self.store,
                provider=self.provider,
                telegram=self.telegram,
            )

    @classmethod
    def from_config(cls, config: Config, **kwargs) -> "FusionBot":
        return cls(config=config, **kwargs)

    def startup(self, *, _as_peer: bool = False) -> None:
        """Load persisted state. Safe to call twice."""
        self._load_state()
        if not _as_peer:
            self._hydrate_key_pool()
        self._load_candles()
        self.running = True
        for peer in self.peer_bots.values():
            peer.startup(_as_peer=True)
            peer.paused = self.paused
            peer.telegram_format = self.telegram_format

    def set_telegram_format(self, fmt: TelegramFormatConfig) -> None:
        """Update and persist the Telegram message format across all active markets."""
        self.telegram_format = fmt
        for peer in self.peer_bots.values():
            peer.telegram_format = fmt
        self._save_state(force=True)

    def _hydrate_key_pool(self) -> None:
        if not isinstance(self.provider, TwelveDataProvider) or not self.provider.configured:
            return
        saved = self.store.load_key_pool()
        if not saved:
            return
        first = self.provider.pool.slots[0]
        self.provider.pool = KeyPool.from_state(
            self.provider.keys,
            saved,
            minute_limit=first.minute_limit,
            day_limit=first.day_limit,
            clock=self.provider.pool._clock,
        )

    def flush_key_pool(self) -> None:
        if (
            self.provider is not None
            and hasattr(self.provider, "pool")
            and getattr(self.provider.pool, "configured", False)
        ):
            self.store.save_key_pool(self.provider.pool.to_dict())

    def _load_state(self) -> None:
        row = self.store.load_model_state(pair=self.config.pair) or {}
        self.model = OnlineLogisticModel.from_dict(row.get("model"))
        self.ensemble = BootstrapEnsemble.from_dict(row.get("ensemble"))
        self.weights = CouncilWeights.from_dict(self.store.load_council_weights(pair=self.config.pair))
        runtime = row.get("runtime")
        own_runtime = runtime if isinstance(runtime, dict) else {}
        if runtime is None and self.config.pair != "EUR/USD":
            primary = self.store.load_model_state(pair="EUR/USD") or {}
            runtime = primary.get("runtime")
        runtime = runtime or {}
        self.paused = bool(runtime.get("paused", False))
        offset = runtime.get("telegram_offset")
        if isinstance(offset, int) and offset > 0:
            self._saved_telegram_offset = offset
            if self.telegram is not None:
                self.telegram.offset = offset
        self.stale_feed_minutes = int(runtime.get("stale_feed_minutes") or 0)
        saved_fmt = runtime.get("telegram_format")
        if isinstance(saved_fmt, dict):
            self.telegram_format = TelegramFormatConfig.from_dict(saved_fmt)
        saved_shadow = own_runtime.get("shadow_recovery")
        if isinstance(saved_shadow, dict):
            self.shadow_recovery = {
                "pending": saved_shadow.get("pending") if isinstance(saved_shadow.get("pending"), dict) else None,
                "cleared_at": saved_shadow.get("cleared_at") if saved_shadow.get("cleared_at") else None,
            }
        else:
            self.shadow_recovery = {"pending": None, "cleared_at": None}
        self.mistakes = MistakeMemory.from_dict(row.get("mistakes"))
        points = row.get("calibration_points") or []
        self.calibration_points = [
            (float(item[0]), int(item[1])) for item in points if len(item) == 2
        ][-4000:]

    def _load_candles(self) -> None:
        cap = max(MAX_STORED_CANDLES, int(getattr(self.config, "candle_history", 0) or 0))
        rows = self.store.load_candles(self.config.pair, limit=cap)
        candles: list[Candle] = []
        for row in rows:
            try:
                candles.append(Candle.from_mapping(row))
            except (ValueError, KeyError, TypeError):
                continue
        candles.sort(key=lambda candle: candle.timestamp)
        self.candles = candles[-cap:]

    # ------------------------------------------------------------------- loop
    def run_forever(self) -> None:  # pragma: no cover - long running
        self.startup()
        self._install_signal_handlers()
        while self.running:
            try:
                self.housekeeping()
                self._wait_for_minute()
                self.minute_cycle()
            except KeyboardInterrupt:
                self.running = False
                break
            except Exception:  # noqa: BLE001 - a 24/7 loop must not die on one bar
                self.store.alert("cycle_error", traceback.format_exc(limit=3)[:900])
                time.sleep(min(30.0, float(self.config.tick_seconds)))
        self.shutdown()

    def run_once(self, at: datetime | None = None) -> CycleReport:
        """Run a single minute cycle. Used by tests, smoke runs and Actions."""
        stamp = at or now()
        self.startup()
        self.housekeeping(at=stamp)
        return self.minute_cycle(at=stamp)

    def _install_signal_handlers(self) -> None:  # pragma: no cover
        def _stop(_signum, _frame):
            self.running = False

        for name in ("SIGTERM", "SIGINT"):
            if hasattr(signal, name):
                try:
                    signal.signal(getattr(signal, name), _stop)
                except (ValueError, OSError):
                    continue

    def shutdown(self) -> None:
        self.running = False
        self._save_state(force=True)

    def _wait_for_minute(self) -> None:
        """Sleep to just after the next minute boundary, staying responsive.

        The wait is punctuated so /pause and /status still land, but at
        TICK_SECONDS rather than once a second. Waking every second costs about
        86,000 Telegram reads a day to service a handful of commands, and buys
        nothing on a bot whose cycle is a minute long anyway.
        """
        interval = max(1.0, float(self.config.tick_seconds))
        deadline = time.time() + seconds_until_next_minute() + SEAL_OFFSET_SECONDS
        while self.running and time.time() < deadline:
            try:
                self.poll_commands()
                self.flush_outbox()
            except Exception:  # noqa: BLE001
                pass
            time.sleep(min(interval, max(0.05, deadline - time.time())))

    # ---------------------------------------------------------- housekeeping
    def listen(self, seconds: float, *, poll_timeout: float = 20.0) -> int:
        """Stay on the line, answering commands the moment they arrive.

        A scheduled runner is only alive for the length of its job, so a
        command sent while it is down waits for the next wake-up. Long polling
        closes most of that gap: while this runs, Telegram holds the request
        open and returns the instant the owner sends something, so /signals is
        answered in about a second rather than whenever the cron next fires.

        Settlements and outbox retries are drained on every pass too, so a
        result is delivered during the same window instead of half an hour
        later.
        """
        if self.telegram is None or not self.telegram.configured:
            return 0
        deadline = time.monotonic() + max(0.0, float(seconds))
        handled = 0
        failures = 0
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            timeout = int(max(1, min(poll_timeout, remaining)))
            poll_error = False
            try:
                before = self.telegram.offset
                for command in self.telegram.poll_commands(
                    timeout_seconds=timeout,
                    include_plain_text=self.config.needle_enabled,
                ):
                    if not command.from_owner:
                        # Silent by design: the bot does not talk to strangers.
                        continue
                    self.handle_command(
                        command.text if command.command == "__needle__" else command.command
                    )
                    handled += 1
                poll_error = bool(self.telegram.last_poll_error)
                self._report_telegram_visibility()
                if self.telegram.offset != before:
                    # The cursor only survives if it is written down. Without
                    # this the next container starts from the same update and
                    # answers the owner's command a second time.
                    self._save_state(force=True)
            except Exception as error:  # noqa: BLE001 - listening must not die
                self.store.alert("listen_error", str(error)[:300])
                poll_error = True
            if poll_error:
                # A transport that fails instantly would otherwise spin the
                # loop thousands of times inside one window. Telegram answers
                # that with a rate limit rather than a fix, so back off.
                failures += 1
                delay = min(15.0, 0.5 * (2 ** min(failures, 5)))
                time.sleep(min(delay, max(0.0, deadline - time.monotonic())))
            else:
                failures = 0
            try:
                self.settle_due()
                self.flush_outbox()
            except Exception as error:  # noqa: BLE001
                self.store.alert("listen_error", str(error)[:300])
        return handled

    def housekeeping(self, at: datetime | None = None) -> None:
        stamp = at or now()
        self.store.heartbeat(
            heartbeat.build_heartbeat(
                runner=self.config.runner,
                phase="tick",
                extra={"paused": self.paused, "cycles": self.cycles},
            )
        )
        self.poll_commands()
        self.settle_due(at=stamp)
        self.flush_outbox()

    def _report_telegram_visibility(self) -> None:
        """Put transport problems in the database instead of nowhere.

        A bot that cannot hear its owner and a bot with nothing to say look
        identical from the outside: no reply arrives. These alerts separate
        them, which is the only way to diagnose a silent bot without asking
        the owner to read a log. The hour bucket keeps a permanently failing
        poll from writing thousands of identical rows a day.
        """
        client = self.telegram
        if client is None:
            return
        if client.last_poll_error:
            self._alert_once(
                "telegram_poll_error",
                client.last_poll_error[:300],
                key="telegram_poll_error|" + now().strftime("%Y%m%d%H"),
                notify=False,
            )
            client.last_poll_error = None
        for ignored in client.ignored_senders:
            chat_id = str(ignored.get("chat_id", ""))
            self._alert_once(
                "telegram_ignored_sender",
                (
                    f"Ignored {ignored.get('command', '')} from chat {chat_id}; "
                    f"owner id is {self._masked_owner_id()}."
                ),
                key=f"telegram_ignored_sender|{chat_id}",
                notify=False,
            )
        client.ignored_senders = []

    def _masked_owner_id(self) -> str:
        """The configured owner id, enough to compare against, not to use."""
        owner = str(self.telegram.owner_chat_id or "") if self.telegram else ""
        if len(owner) <= 4:
            return "*" * len(owner)
        return f"{owner[:2]}{'*' * (len(owner) - 4)}{owner[-2:]}"

    def poll_commands(self, *, timeout_seconds: int = 0) -> None:
        if self.telegram is None or not self.telegram.configured:
            return
        before = self.telegram.offset
        for command in self.telegram.poll_commands(
            timeout_seconds=timeout_seconds,
            include_plain_text=self.config.needle_enabled,
        ):
            if not command.from_owner:
                # Silent by design: the bot does not talk to strangers.
                continue
            self.handle_command(
                command.text if command.command == "__needle__" else command.command
            )
        self._report_telegram_visibility()
        if self.telegram.offset != before:
            # Acknowledge the updates. This container exits in a few minutes, so
            # unless the cursor is written down the next one starts from zero and
            # re-answers every command from the last 24 hours, every cycle.
            self._save_state(force=True)

    def _set_paused(self, value: bool) -> None:
        self.paused = value
        for peer in self.peer_bots.values():
            peer.paused = value

    def _get_needle_agent(self) -> needle_agent.SaviourNeedleAgent:
        agent = getattr(self, "_needle_agent", None)
        if agent is None:
            agent = needle_agent.SaviourNeedleAgent(
                state_dir=self.config.local_state_dir,
                weights_path=(self.config.needle_weights_path or None),
                min_confidence=self.config.needle_min_confidence,
            )
            self._needle_agent = agent
        return agent

    def handle_natural_language(self, query: str) -> str:
        """Route a plain-English owner message via Cactus Compute Needle 3."""
        _decision, reply = needle_agent.execute_on_bot(
            self,
            query,
            agent=self._get_needle_agent(),
        )
        if self.telegram is not None:
            self._report_command_reply("needle", self.telegram.send(reply))
        return reply

    def handle_command(self, command: str) -> str:
        clean = (command or "").strip()
        if clean == "/start":
            self._set_paused(False)
            self._save_state(force=True)
            markets = ", ".join(self.config.active_pairs)
            text = f"FUSION BOT is live for {markets} M1 research signals.\n\n" + HELP_TEXT
        elif clean == "/help":
            text = HELP_TEXT
        elif clean == "/status":
            text = self.status_text()
        elif clean == "/pause":
            self._set_paused(True)
            self._save_state(force=True)
            text = "Signals paused. Settlement and learning continue."
        elif clean == "/resume":
            self._set_paused(False)
            self._save_state(force=True)
            text = "Signals resumed."
        elif clean == "/signals":
            text = self.on_demand_signal()
        elif self.config.needle_enabled and clean and not clean.startswith("/"):
            return self.handle_natural_language(clean)
        else:
            text = "Unknown command.\n\n" + HELP_TEXT
        if self.telegram is not None:
            self._report_command_reply(clean or command, self.telegram.send(text))
        return text

    def _report_command_reply(self, command: str, result) -> None:
        """Write down what happened to a reply.

        Command replies go straight to Telegram rather than through the
        outbox, so until now a reply that Telegram rejected left no trace
        anywhere: the owner saw silence and the database stayed clean, which
        is the one thing guaranteed to make a silent bot undiagnosable. A
        success is proof the round trip works; a failure says why not.
        """
        hour = now().strftime("%Y%m%d%H")
        if result.confirmed:
            self._alert_once(
                "telegram_reply_sent",
                f"Answered {command} (message {result.message_id}).",
                key=f"telegram_reply_sent|{command}|{hour}",
                notify=False,
            )
            return
        self._alert_once(
            "telegram_reply_failed",
            f"Reply to {command} not delivered: {result.state} {result.error}"[:300],
            key=f"telegram_reply_failed|{command}|{hour}",
            notify=False,
        )

    def on_demand_signal(self, at: datetime | None = None) -> str:
        """Answer /signals: read the live market now and say what it says.

        This deliberately runs the same engine and the same gates as the
        scheduled cycle. It does not relax a single threshold to produce an
        answer, because a signal manufactured to satisfy a request is not a
        prediction, it is a comfort -- and the whole point of this bot is that it
        only ever says what it measured.

        When the gates decline, the honest reply is that there is no setup.
        Silence would be easier to mistake for a working system than a
        refusal, which is exactly why the refusal is spelled out.
        """
        raw_now = at or now()
        if self.paused:
            return "Signals are paused. Send /resume to start again."

        if not self.peer_bots:
            if self.provider is not None and getattr(self.provider, "configured", False):
                self.ingest(at=raw_now)
                self.flush_key_pool()
            real_now = at or now()
            stamp = next_future_candle_open(real_now) if at is None else at
            obs_now = real_now if at is None else None
            causal_candles = closed_before(self.candles, floor_minute(real_now) if at is None else real_now)
            if not causal_candles:
                return "No price data available. The provider may be unreachable; try again shortly."

            guard = self._check_runtime_protections(stamp)
            if guard.blocked:
                return (
                    f"No setup right now (next candle {format_display_hm(stamp)}).\n"
                    f"{guard.reason}\n\n"
                    "The bot only reports what its gates actually agree on. "
                    "Asking again will not change the answer until the market does."
                )

            result = self.engine.evaluate(
                causal_candles,
                model=self.model,
                ensemble=self.ensemble,
                weights=self.weights,
                source="LIVE",
                at=stamp,
                observed_at=obs_now,
                mistakes=self.mistakes,
            )
            if not result.actionable or result.signal is None:
                return (
                    f"No setup right now (next candle {format_display_hm(stamp)}).\n"
                    f"{result.reason}\n\n"
                    "The bot only reports what its gates actually agree on. "
                    "Asking again will not change the answer until the market does."
                )

            signal = result.signal
            self.store.save_signal(
                dict(signal.to_dict(), seal_state=charter.SEAL_SIGNAL, settled=False,
                     bucket=minute_bucket(stamp))
            )
            self.deliver(signal, brief=True)
            self.settle_due(at=real_now)
            return format_brief_signal(signal, self.telegram_format)

        # Multi-market evaluation across every configured instrument (concurrent ingest).
        self.ingest_all(at=raw_now)
        self.flush_key_pool()
        real_now = at or now()
        stamp = next_future_candle_open(real_now) if at is None else at
        obs_now = real_now if at is None else None

        emitted: list[str] = []
        refusals: list[str] = []
        any_candles = False
        for member in (self, *self.peer_bots.values()):
            causal_candles = closed_before(member.candles, floor_minute(real_now) if at is None else real_now)
            if not causal_candles:
                refusals.append(f"{member.config.pair}: no price data available.")
                continue
            any_candles = True
            guard = member._check_runtime_protections(stamp)
            if guard.blocked:
                refusals.append(f"{member.config.pair}: {guard.reason}")
                continue
            result = member.engine.evaluate(
                causal_candles,
                model=member.model,
                ensemble=member.ensemble,
                weights=member.weights,
                source="LIVE",
                at=stamp,
                observed_at=obs_now,
                mistakes=member.mistakes,
            )
            if not result.actionable or result.signal is None:
                refusals.append(f"{member.config.pair}: {result.reason}")
                continue
            signal = result.signal
            member.store.save_signal(
                dict(signal.to_dict(), seal_state=charter.SEAL_SIGNAL, settled=False,
                     bucket=minute_bucket(stamp))
            )
            member.deliver(signal, brief=True)
            emitted.append(format_brief_signal(signal, self.telegram_format))

        self.settle_due(at=real_now)
        if emitted:
            return "\n\n".join(emitted)
        if not any_candles:
            return "No price data available. The provider may be unreachable; try again shortly."
        return (
            f"No setup right now (next candle {format_display_hm(stamp)}).\n"
            + "\n".join(refusals)
            + "\n\nThe bot only reports what its gates actually agree on. "
            "Asking again will not change the answer until the market does."
        )

    def _latest_system_report(self) -> dict | None:
        versions = self.store.load_model_versions(limit=1)
        if not versions:
            return None
        report = versions[-1].get("report") or {}
        return report.get("system")

    def status_text(self) -> str:
        settlements = [
            row for row in self.store.load_settlements(limit=2000) if not charter.is_synthetic(row.get("source"))
        ]
        stats = settle.summarise([_settlement_from_row(row) for row in settlements])
        calibrator = self.model.calibrator if self.model else None
        return digest.build_status(
            config=self.config,
            paused=self.paused,
            runner=self.config.runner,
            heartbeat_row=self.store.latest_heartbeat(runner=self.config.runner),
            stats=stats,
            unresolved=stats.unresolved,
            model_mode=self.model.mode if self.model else MODE_VETO,
            calibration_state=(
                "CALIBRATED" if calibrator and calibrator.fitted and not calibrator.flat
                else "FLAT" if calibrator and calibrator.flat
                else "RAW"
            ),
            calibration_samples=(calibrator.samples if calibrator else 0),
            key_pool=self.store.load_key_pool() or {},
            last_cycle=self.last_cycle.to_dict() if self.last_cycle else None,
            store_errors=self.store.errors[-5:],
            system=self._latest_system_report(),
        )

    # ---------------------------------------------------------- minute cycle
    def _sealable_stamp(self, stamp: datetime) -> datetime:
        """Roll to the next minute when this one can no longer be sealed.

        GitHub's cron does not land on minute boundaries. Measured across a
        day of live runs, the wake-up fell anywhere from :03 to :58 past the
        minute, and the charter discards anything sealed more than ten seconds
        after a minute opens because backdating an entry is a lie. A scheduled
        cycle that evaluates the minute it woke into is therefore discarded
        for being late far more often than for being wrong.

        Measured on the live table: of the first twenty-one cycles, two found
        a real setup and both were thrown away -- sealed 14.2s and 46.6s
        after the minute opened. Zero signals in eleven hours.

        Predicting the minute that has not opened yet is not backdated: its
        entry price does not exist yet. It is the same thing /signals does,
        and it is the only kind of call a scheduled runner can honestly make.
        """
        opened = floor_minute(stamp)
        if (stamp - opened).total_seconds() <= self.config.max_seal_latency_seconds:
            return stamp
        return opened + timedelta(minutes=1, seconds=1)

    def wait_for_seal_window(
        self,
        *,
        min_offset_seconds: float = 1.5,
        max_entry_seconds: float = 4.5,
        sleep_fn=time.sleep,
    ) -> datetime:
        """Align a live scheduled runner to the opening seal window of a real minute.

        Waking mid-minute (e.g. :24s or :48s) and fabricating ``next_minute + 1s``
        without waiting caused two live timing defects:
        1. Twelve Data's unfinished forming bar for the current minute was
           accepted by ``closed_before(..., next_minute)`` as if it had closed;
        2. Signals were dispatched up to 36s BEFORE ``entry_time`` opened.

        Waiting at most ~57s for ``next_minute + 1.5s`` guarantees the previous
        M1 candle has genuinely closed on the exchange AND ``entry_time`` has
        already opened on the real wall clock, with ~8.5s of seal headroom left.
        """
        current = now()
        opened = floor_minute(current)
        elapsed = (current - opened).total_seconds()
        if elapsed < min_offset_seconds:
            sleep_fn(max(0.0, min_offset_seconds - elapsed))
            return now()
        if elapsed <= max_entry_seconds:
            return current

        # Free time before the next minute opens: settle any already-expired
        # signals and flush pending outbox deliveries first.
        self.settle_due()
        self.flush_outbox()

        current = now()
        opened = floor_minute(current)
        elapsed = (current - opened).total_seconds()
        if min_offset_seconds <= elapsed <= max_entry_seconds:
            return current
        target = (
            opened + timedelta(seconds=min_offset_seconds)
            if elapsed < min_offset_seconds
            else opened + timedelta(minutes=1, seconds=min_offset_seconds)
        )
        wait_seconds = (target - current).total_seconds()
        if wait_seconds > 0:
            sleep_fn(wait_seconds)
        return now()

    def ingest_all(self, at: datetime | None = None, *, sleep_fn=time.sleep) -> int:
        """Fetch latest candles across all configured instruments concurrently.

        Twelve Data's free CDN occasionally publishes the just-closed ``M-1``
        minute bar ~6-7s after ``:00``. When ``at`` falls inside the opening
        seal window and a live ``TwelveDataProvider`` has not yet indexed
        ``M-1`` on the first pass, perform at most one lightweight
        ``outputsize=6`` catch-up poll around ``:07.1s`` while staying well
        inside both the 8/min key quota and the 10s charter seal ceiling.
        """
        members = [
            m
            for m in (self, *self.peer_bots.values())
            if m.provider is not None and getattr(m.provider, "configured", False)
        ]
        if not members:
            return 0

        def _fetch_batch(targets: list[FusionBot], size: int | None = None) -> int:
            if len(targets) == 1:
                return int(targets[0].ingest(at=at, outputsize=size) or 0)
            with ThreadPoolExecutor(max_workers=len(targets)) as pool:
                futs = [pool.submit(m.ingest, at=at, outputsize=size) for m in targets]
                return sum(int(f.result() or 0) for f in futs)

        added = _fetch_batch(members)

        if at is not None:
            opened = floor_minute(at)
            expected_prev = opened - timedelta(minutes=1)
            lagging = [
                m
                for m in members
                if getattr(m.provider, "pool", None) is not None
                and not getattr(m.provider, "synthetic", False)
                and (
                    not closed_before(m.candles, opened)
                    or closed_before(m.candles, opened)[-1].timestamp < expected_prev
                )
            ]
            if lagging:
                elapsed = (now() - opened).total_seconds()
                if 0.0 <= elapsed <= 6.8:
                    wait_s = min(1.5, max(0.25, 7.1 - elapsed))
                    sleep_fn(wait_s)
                    added += _fetch_batch(lagging, size=6)

        return added

    def minute_cycle(
        self,
        at: datetime | None = None,
        *,
        _as_peer: bool = False,
        _already_ingested: bool = False,
        _causal_cutoff: datetime | None = None,
    ) -> CycleReport:
        real_clock_mode = at is None
        woke_at = now() if real_clock_mode else at
        assert woke_at is not None
        planned_stamp = self._sealable_stamp(woke_at)
        planned_minute = floor_minute(planned_stamp)
        bucket = minute_bucket(planned_stamp)
        job_bucket = bucket if self.config.pair == "EUR/USD" else f"{self.config.pair}|{bucket}"
        self.cycles += 1

        if not self.store.claim_job("minute", job_bucket, runner=self.config.runner):
            # Another runner already owns this minute. Nothing to do, no harm done.
            return CycleReport(bucket, "SKIPPED", "Another runner already claimed this minute.")

        if not _already_ingested:
            if not _as_peer and (self.peer_bots or real_clock_mode):
                self.ingest_all(at=woke_at if real_clock_mode else None)
            else:
                self.ingest(at=woke_at)

        if real_clock_mode:
            post_ingest_now = now()
            causal_cutoff = post_ingest_now
            # If we woke inside planned_minute's seal window, measure seal latency
            # against the TRUE post-ingest wall clock so a slow HTTP fetch (>10s)
            # is honestly caught as DISCARDED_LATE instead of masked by pre-ingest stamp.
            eval_stamp = (
                post_ingest_now
                if floor_minute(woke_at) == planned_minute
                else planned_stamp
            )
        else:
            causal_cutoff = _causal_cutoff or woke_at
            eval_stamp = planned_stamp

        report = self.evaluate_minute(bucket=bucket, at=eval_stamp, causal_cutoff=causal_cutoff)
        if self.peer_bots:
            per_pair: list[tuple[str, CycleReport]] = [(self.config.pair, report)]
            for symbol, peer in self.peer_bots.items():
                if real_clock_mode:
                    peer_now = now()
                    peer_stamp = peer_now if floor_minute(woke_at) == planned_minute else planned_stamp
                    peer_report = peer._run_peer_minute_cycle(
                        bucket=bucket,
                        eval_stamp=peer_stamp,
                        causal_cutoff=peer_now,
                    )
                else:
                    peer_report = peer.minute_cycle(
                        at=planned_stamp,
                        _as_peer=True,
                        _already_ingested=True,
                        _causal_cutoff=causal_cutoff,
                    )
                per_pair.append((symbol, peer_report))
            report = _summarise_market_reports(bucket, per_pair)

        self.last_cycle = report
        hb_extra = report.to_dict()
        if _as_peer or (not self.peer_bots and self.config.pair != "EUR/USD"):
            hb_extra["bucket"] = job_bucket
            if not str(hb_extra.get("reason", "")).startswith(f"{self.config.pair}:"):
                hb_extra["reason"] = f"{self.config.pair}: {hb_extra.get('reason', '')}"
        self.store.heartbeat(
            heartbeat.build_heartbeat(
                runner=self.config.runner,
                phase="minute",
                extra=hb_extra,
            )
        )
        if not _as_peer:
            self.flush_key_pool()
        return report

    def _run_peer_minute_cycle(
        self,
        *,
        bucket: str,
        eval_stamp: datetime,
        causal_cutoff: datetime,
    ) -> CycleReport:
        """Execute a peer instrument's already-ingested minute evaluation with real post-ingest clock."""
        job_bucket = f"{self.config.pair}|{bucket}"
        self.cycles += 1
        if not self.store.claim_job("minute", job_bucket, runner=self.config.runner):
            return CycleReport(bucket, "SKIPPED", "Another runner already claimed this minute.")
        report = self.evaluate_minute(bucket=bucket, at=eval_stamp, causal_cutoff=causal_cutoff)
        self.last_cycle = report
        hb_extra = report.to_dict()
        hb_extra["bucket"] = job_bucket
        if not str(hb_extra.get("reason", "")).startswith(f"{self.config.pair}:"):
            hb_extra["reason"] = f"{self.config.pair}: {hb_extra.get('reason', '')}"
        self.store.heartbeat(
            heartbeat.build_heartbeat(
                runner=self.config.runner,
                phase="minute",
                extra=hb_extra,
            )
        )
        return report

    def ingest(self, at: datetime | None = None, *, outputsize: int | None = None) -> int:
        """Fetch the newest candles and merge them into memory and the store."""
        if self.provider is None or not getattr(self.provider, "configured", False):
            return 0
        if outputsize is not None:
            req_size = int(outputsize)
        elif (
            at is not None
            and len(self.candles) >= 120
            and getattr(self.provider, "pool", None) is not None
            and not getattr(self.provider, "synthetic", False)
            and 0.0 <= (at - self.candles[-1].timestamp).total_seconds() <= 1800.0
        ):
            # Warm live cycle: fetch incremental tail instead of 750-bar history
            # so Twelve Data's CDN responds faster with fresher tail bars and
            # all 3 markets finish well inside the 10s Charter seal ceiling.
            req_size = min(int(self.config.candle_history), 45)
        else:
            req_size = int(self.config.candle_history)
        try:
            fresh = self.provider.time_series(self.config.pair, self.config.timeframe, req_size)
        except Exception as error:  # noqa: BLE001 - provider failures must not kill the loop
            self.store.alert("provider_error", str(error)[:300])
            return 0
        if not fresh:
            return 0

        known = {candle.timestamp: candle for candle in self.candles}
        added = 0
        revised = 0
        rows = []
        for candle in fresh:
            existing = known.get(candle.timestamp)
            if existing is not None and existing == candle:
                continue
            # A timestamp we have seen before is not automatically news we have
            # heard before. The newest bar is still forming: its high, low and
            # close keep moving until the minute closes. Skipping it froze the
            # forming bar at whatever it looked like the first time we asked,
            # so every feature for the current minute was computed on a stale
            # snapshot. Re-uptake it when it actually changed.
            known[candle.timestamp] = candle
            rows.append(dict(candle.to_dict(), pair=self.config.pair))
            if existing is None:
                added += 1
            else:
                revised += 1
        if rows:
            cap = max(MAX_STORED_CANDLES, int(getattr(self.config, "candle_history", 0) or 0))
            self.candles = sorted(known.values(), key=lambda item: item.timestamp)[-cap:]
            self.store.save_candles(rows)
        return added

    def _check_runtime_protections(self, at: datetime) -> protections.ProtectionVerdict:
        """Evaluate CooldownPeriod and StoplossGuard against persisted ledger rows."""
        cd_mins = int(getattr(self.config, "signal_cooldown_minutes", 0) or 0)
        if cd_mins > 0:
            verdict = protections.check_cooldown_period(
                at,
                self.store.load_signals(limit=120),
                pair=self.config.pair,
                cooldown_minutes=cd_mins,
            )
            if verdict.blocked:
                return verdict

        streak_cap = int(getattr(self.config, "loss_guard_streak", 0) or 0)
        if streak_cap > 0:
            verdict = protections.check_stoploss_guard(
                at,
                self.store.load_settlements(limit=200),
                pair=self.config.pair,
                max_loss_streak=streak_cap,
                lookback_minutes=int(getattr(self.config, "loss_guard_lookback_minutes", 180) or 180),
                cooldown_minutes=int(getattr(self.config, "loss_guard_cooldown_minutes", 60) or 60),
                require_shadow_win=bool(getattr(self.config, "loss_guard_shadow_recovery", False)),
                shadow_cleared_at=self.shadow_recovery.get("cleared_at"),
            )
            if verdict.blocked:
                return verdict

        return protections.CLEAR

    def _resolve_shadow_recovery(self, at: datetime | None = None) -> bool:
        """Settle an in-flight StoplossGuard shadow candidate without touching the live ledger."""
        pending = self.shadow_recovery.get("pending")
        if not isinstance(pending, dict):
            return False
        stamp = at or now()
        try:
            sig = Signal.from_dict(pending)
        except (KeyError, ValueError, TypeError):
            self.shadow_recovery["pending"] = None
            self._save_state(force=True)
            return False

        expiry = parse_datetime(sig.expiry_time)
        if expiry > stamp:
            return False
        prices_ready = self._has_candle(sig.entry_time) and self._has_candle(_exit_stamp(sig))
        if not prices_ready and (stamp - expiry) < timedelta(minutes=charter.UNRESOLVE_AFTER_MINUTES):
            return False

        settlement = settle.settle(sig, self.candles)
        self.shadow_recovery["pending"] = None
        net_val = float(settlement.net_pips or 0.0)
        won = settlement.outcome == charter.WIN and net_val > 0.0
        if won:
            cleared_stamp = settlement.settled_at or stamp
            self.shadow_recovery["cleared_at"] = (
                cleared_stamp.strftime("%Y-%m-%dT%H:%M:%SZ")
                if hasattr(cleared_stamp, "strftime")
                else str(cleared_stamp)
            )
            self.store.log_event(
                {
                    "kind": "shadow_recovery_win",
                    "pair": self.config.pair,
                    "signal_id": sig.signal_id,
                    "net_pips": round(net_val, 3),
                }
            )
        else:
            self.store.log_event(
                {
                    "kind": "shadow_recovery_miss",
                    "pair": self.config.pair,
                    "signal_id": sig.signal_id,
                    "outcome": settlement.outcome,
                    "net_pips": round(net_val, 3),
                }
            )
        self._save_state(force=True)
        return won

    def evaluate_minute(
        self,
        *,
        bucket: str,
        at: datetime | None = None,
        causal_cutoff: datetime | None = None,
    ) -> CycleReport:
        stamp = at or now()
        eval_candles = (
            closed_before(self.candles, causal_cutoff)
            if causal_cutoff is not None
            else self.candles
        )
        if not eval_candles:
            return CycleReport(bucket, charter.SEAL_ABSTAIN, "No candles available yet.")

        # On a real-clock boundary seal (causal_cutoff inside the same minute as
        # stamp), require the immediately preceding M-1 bar to be present so a
        # delayed provider cannot silently trade off a 2-minute-old M-2 bar.
        if (
            causal_cutoff is not None
            and self.provider is not None
            and getattr(self.provider, "configured", False)
            and not getattr(self.provider, "synthetic", False)
            and floor_minute(causal_cutoff) == floor_minute(stamp)
        ):
            expected_prev = floor_minute(stamp) - timedelta(minutes=1)
            if eval_candles[-1].timestamp < expected_prev:
                age_s = (stamp - eval_candles[-1].timestamp).total_seconds()
                return CycleReport(
                    bucket,
                    charter.SEAL_ABSTAIN,
                    f"Data gate: previous minute bar ({expected_prev.strftime('%H:%M')}Z) not published yet "
                    f"(latest {eval_candles[-1].timestamp.strftime('%H:%M')}Z, {age_s:.0f}s old).",
                )

        self._resolve_shadow_recovery(stamp)
        guard = self._check_runtime_protections(stamp)
        if guard.blocked:
            if (
                bool(getattr(self.config, "loss_guard_shadow_recovery", False))
                and guard.guard in ("stoploss_guard", "stoploss_shadow_recovery")
                and self.shadow_recovery.get("pending") is None
                and not getattr(self.provider, "synthetic", False)
            ):
                shadow_res = self.engine.evaluate(
                    eval_candles,
                    model=self.model,
                    ensemble=self.ensemble,
                    weights=self.weights,
                    source="LIVE",
                    at=stamp,
                    mistakes=self.mistakes,
                )
                if shadow_res.actionable and shadow_res.signal is not None:
                    self.shadow_recovery["pending"] = shadow_res.signal.to_dict()
                    self.store.log_event(
                        {
                            "kind": "shadow_recovery_open",
                            "pair": self.config.pair,
                            "signal_id": shadow_res.signal.signal_id,
                            "side": shadow_res.signal.side,
                            "horizon_minutes": shadow_res.signal.horizon_minutes,
                        }
                    )
                    self._save_state(force=True)
            self.abstentions += 1
            return CycleReport(bucket, charter.SEAL_ABSTAIN, guard.reason)

        result = self.engine.evaluate(
            eval_candles,
            model=self.model,
            ensemble=self.ensemble,
            weights=self.weights,
            source="LIVE" if not getattr(self.provider, "synthetic", False) else "SYNTHETIC",
            at=stamp,
            mistakes=self.mistakes,
        )

        if result.gate is not None and result.gate.quality == "GAPPY":
            self._alert_once(
                "data_gaps",
                f"{self.config.pair}: {result.gate.gaps} gaps in recent M1 series.",
                key=f"data_gaps|{self.config.pair}|{bucket[:10]}",
                notify=False,
            )

        if result.gate is not None and result.gate.quality == "STALE":
            self.stale_feed_minutes += 1
            if self.stale_feed_minutes >= 5:
                self._alert_once(
                    digest.ALERT_STALE_FEED,
                    f"{result.gate.age_seconds:.0f}s for {self.stale_feed_minutes} minutes.",
                    key=f"stale_feed|{bucket[:10]}",
                )
            # The streak has to be written down, not just counted: this
            # container exits within minutes and one cycle evaluates a single
            # minute, so an in-memory counter restarts at zero forever and the
            # alert can never reach five.
            self._save_state(force=True)
        elif self.stale_feed_minutes:
            self.stale_feed_minutes = 0
            self._save_state(force=True)

        if result.seal_state == charter.SEAL_DISCARDED_LATE:
            return CycleReport(bucket, result.seal_state, result.reason, regime=result.regime)

        if not result.actionable or result.signal is None:
            self.abstentions += 1
            return CycleReport(bucket, charter.SEAL_ABSTAIN, result.reason, regime=result.regime)

        signal = result.signal
        self.store.save_signal(dict(signal.to_dict(), seal_state=charter.SEAL_SIGNAL, settled=False, bucket=bucket))

        if self.paused:
            self.store.update_signal(signal.signal_id, {"delivery_state": charter.DELIVERY_SUPPRESSED})
            return CycleReport(
                bucket, charter.SEAL_SIGNAL, "Signals are paused; settled and learned from but not sent.",
                regime=signal.regime, signal_id=signal.signal_id,
            )

        self.deliver(signal)
        return CycleReport(
            bucket, charter.SEAL_SIGNAL, signal.reason, regime=signal.regime, signal_id=signal.signal_id
        )

    # -------------------------------------------------------------- delivery
    def deliver(self, signal: Signal, *, brief: bool = False) -> None:
        """Queue and send. SENT is recorded only on a confirmed message id.

        ``brief`` is what goes to the phone for an on-demand /signals request:
        pair, direction, entry, expiry. The outbox still stores the full text
        with the model read and the reasoning, so the short version is a
        display choice and not a loss of the audit trail.
        """
        key = f"signal|{signal.signal_id}"
        self.store.outbox_append(
            {
                "message_key": key,
                "text": format_signal(signal, self.telegram_format),
                "kind": "signal",
                "state": charter.DELIVERY_PENDING,
            }
        )
        outgoing = (
            format_brief_signal(signal, self.telegram_format)
            if brief
            else format_signal(signal, self.telegram_format)
        )
        if self.telegram is None or not self.telegram.configured:
            self.store.outbox_update(key, {"state": charter.DELIVERY_FAILED, "last_error": "Telegram not configured"})
            self.store.update_signal(signal.signal_id, {"delivery_state": charter.DELIVERY_FAILED})
            return

        result = self.telegram.send(outgoing)
        if result.confirmed:
            self.store.outbox_update(
                key, {"state": charter.DELIVERY_SENT, "message_id": result.message_id, "attempts": 1}
            )
            self.store.update_signal(
                signal.signal_id, {"delivery_state": charter.DELIVERY_SENT, "message_id": result.message_id}
            )
            return

        # Not confirmed. Say so, and make a human look at it when the state is
        # genuinely unknown rather than clearly rejected.
        self.store.outbox_update(key, {"state": result.state, "last_error": result.error[:300], "attempts": 1})
        self.store.update_signal(signal.signal_id, {"delivery_state": result.state})
        if result.state == charter.DELIVERY_UNKNOWN:
            self._alert_once(
                digest.ALERT_DELIVERY_UNKNOWN,
                f"Signal {signal.signal_id[:8]} may or may not have been delivered. {result.error[:200]}",
                key=f"delivery|{signal.signal_id}",
            )

    def deliver_result(self, signal: Signal, settlement) -> None:
        """Send what the prediction did, as soon as the exit price exists.

        Follows the same delivery discipline as the signal itself: SENT is
        recorded only on a confirmed message id, and a failure is queued for
        retry rather than assumed delivered.
        """
        key = f"result|{signal.signal_id}"
        text = format_result(signal, settlement, self.telegram_format)
        if self.telegram is None or not self.telegram.configured:
            return
        self.store.outbox_append(
            {"message_key": key, "text": text, "kind": "result", "state": charter.DELIVERY_PENDING}
        )
        result = self.telegram.send(text)
        if result.confirmed:
            self.store.outbox_update(
                key, {"state": charter.DELIVERY_SENT, "message_id": result.message_id, "attempts": 1}
            )
        else:
            self.store.outbox_update(
                key, {"state": result.state, "last_error": result.error[:300], "attempts": 1}
            )

    def flush_outbox(self) -> None:
        """Retry anything still pending. Telegram has no 'did I send?' query.

        A retry that lands has to clear the *signal's* delivery_state too, not
        just the outbox row. Leaving it alone means the signal stays marked
        FAILED in the signals table while the message is sitting in the chat --
        the record and reality disagree, and every digest downstream counts a
        delivered signal as lost.
        """
        if self.telegram is None or not self.telegram.configured:
            return
        for row in self.store.outbox_pending():
            attempts = int(row.get("attempts", 0) or 0)
            if attempts >= 3:
                continue
            result = self.telegram.send(str(row.get("text", "")))
            patch = {
                "attempts": attempts + 1,
                "state": charter.DELIVERY_SENT if result.confirmed else result.state,
                "last_error": (result.error or "")[:300],
            }
            signal_id = self._signal_id_from_key(row.get("message_key"))
            if result.confirmed:
                patch["message_id"] = result.message_id or ""
                if signal_id:
                    self.store.update_signal(
                        signal_id,
                        {"delivery_state": charter.DELIVERY_SENT, "message_id": patch["message_id"]},
                    )
            else:
                # Still failing. Say which state, and on the last attempt admit
                # we have given up rather than leaving it looking pending.
                state = charter.DELIVERY_FAILED if attempts + 1 >= 3 else result.state
                patch["state"] = state
                if signal_id:
                    self.store.update_signal(signal_id, {"delivery_state": state})
            self.store.outbox_update(str(row.get("message_key")), patch)

    @staticmethod
    def _signal_id_from_key(message_key: object) -> str | None:
        """`signal|<id>` -> `<id>`. Returns None for anything else.

        Result rows are keyed `result|<id>`, and treating that whole string as
        a signal id would silently miss the signal and then write a SENT
        delivery state against a row that does not exist.
        """
        key = str(message_key or "")
        return key.split("|", 1)[-1] if key.startswith("signal|") else None

    # ------------------------------------------------------------ settlement
    def _needs_settlement_ingest(self, stamp: datetime) -> bool:
        """True only when a due signal's exit bar has closed on the clock and is missing in memory."""
        if self.provider is None or not getattr(self.provider, "configured", False):
            return False
        last_refresh = float(getattr(self, "_last_settle_refresh_mono", 0.0) or 0.0)
        if last_refresh > 0.0 and (time.monotonic() - last_refresh) < 30.0:
            return False
        candidates: list[dict] = list(self.store.pending_signals(self.config.pair))
        pending_shadow = self.shadow_recovery.get("pending")
        if isinstance(pending_shadow, dict):
            candidates.append(pending_shadow)
        for row in candidates:
            try:
                sig = Signal.from_dict(row)
            except (KeyError, ValueError, TypeError):
                continue
            exit_bar = _exit_stamp(sig)
            # An M1 bar stamped exit_bar closes at exit_bar + 1 minute (+2s publish buffer).
            if stamp >= exit_bar + timedelta(minutes=1, seconds=2):
                if not (self._has_candle(sig.entry_time) and self._has_candle(exit_bar)):
                    return True
        return False

    def settle_due(self, at: datetime | None = None, *, _as_peer: bool = False) -> int:
        stamp = at or now()
        if not _as_peer:
            due_members = [
                m for m in (self, *self.peer_bots.values()) if m._needs_settlement_ingest(stamp)
            ]
            if due_members:
                now_mono = time.monotonic()
                for m in due_members:
                    m._last_settle_refresh_mono = now_mono
                if len(due_members) == 1:
                    due_members[0].ingest(at=stamp)
                else:
                    with ThreadPoolExecutor(max_workers=len(due_members)) as pool:
                        list(pool.map(lambda m: m.ingest(at=stamp), due_members))
                self.flush_key_pool()

        self._resolve_shadow_recovery(stamp)
        count = 0
        for row in self.store.pending_signals(self.config.pair):
            try:
                signal = Signal.from_dict(row)
            except (KeyError, ValueError, TypeError) as exc:
                # A signal we cannot read must never just disappear. This is
                # what a non-numeric value in the feature row did once: every
                # sealed signal failed to parse, was skipped here without a
                # trace, and stayed "pending" for ever while the bot carried
                # on reporting that it was working. Record it and move on
                # rather than pretending the queue is clean.
                self.store.log_event(
                    {
                        "kind": "signal_unreadable",
                        "signal_id": str(row.get("signal_id", "unknown")),
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            expiry = parse_datetime(signal.expiry_time)
            if expiry > stamp:
                continue
            prices_ready = self._has_candle(signal.entry_time) and self._has_candle(_exit_stamp(signal))
            if not prices_ready:
                grace = timedelta(minutes=charter.UNRESOLVE_AFTER_MINUTES)
                if stamp - expiry < grace:
                    # Prices may still land (late-arriving bars, a restart that
                    # skipped an ingest). Keep waiting rather than guessing.
                    continue
                # Grace expired. settle() records UNRESOLVED rather than a number.
            self._settle_one(signal)
            count += 1
        for peer in self.peer_bots.values():
            count += peer.settle_due(at=stamp, _as_peer=True)
        return count

    def _settle_one(self, signal: Signal) -> None:
        settlement = settle.settle(signal, self.candles)
        self.store.save_settlement(settlement.to_dict())
        label = settle.learning_label(signal, settlement)
        winning = settle.winning_side(settlement)

        # Report the outcome the moment it exists. Synthetic replays are
        # plumbing, not predictions, so they settle silently.
        if not charter.is_synthetic(signal.source):
            self.deliver_result(signal, settlement)

        if charter.is_synthetic(signal.source):
            # Charter: synthetic data trains nothing. The settlement is still
            # recorded so the pipeline can be observed end to end.
            self.store.log_event(
                {"kind": "settlement_synthetic", "signal_id": signal.signal_id, "outcome": settlement.outcome}
            )
            return

        if settlement.outcome == charter.UNRESOLVED:
            self.store.log_event(
                {"kind": "unresolved", "signal_id": signal.signal_id, "reason": "price data never arrived"}
            )
            return

        # Council learning: only prototypes that actually spoke get credit.
        if winning is not None:
            votes = _votes_from_signal(signal)
            self.weights.observe(votes, winning)

        # Post-mortem: remember what this setup cost, so a pattern that keeps
        # losing can be refused instead of re-tried. Rebuilt from what the
        # signal stored, so the same bar always lands in the same cell.
        self.mistakes.mark_reconciled(signal.signal_id)
        self.mistakes.record(
            settlement.outcome,
            MistakeMemory.fingerprint(
                signal.regime,
                session_of(signal.entry_time, instruments.for_symbol(self.config.pair))
                if self.config.session_filter
                else None,
                signal.side,
            ),
        )

        # Ensure a causal MODE_VETO baseline model exists (fit strictly on bars
        # closed before signal.entry_time, or cold-start neutral VETO) so every
        # confirmed real outcome updates online weights from trade #1.
        self._ensure_baseline_veto_model(before=signal.entry_time)

        # Loss-focused learning rate: mistakes carry 1.35x gradient focus (capped
        # at 0.05) so repeating losing setups are down-weighted promptly.
        effective_lr = float(self.config.online_learning_rate)
        if settlement.outcome == charter.LOSS:
            effective_lr = min(0.05, effective_lr * 1.35)

        # Model learning: test before learn, once per confirmed non-tied outcome.
        step = learning.online_step(
            self.model,
            signal.features,
            label,
            source=signal.source,
            learning_rate=effective_lr,
            allow_synthetic=self.config.allow_synthetic_learning,
        )
        self.store.log_event(dict(step.to_dict(), kind="online_step", signal_id=signal.signal_id))

        if step.raw is not None and label is not None:
            self.calibration_points.append((float(step.raw), int(label)))
            self.calibration_points = self.calibration_points[-4000:]
            self._refit_calibration()

        self.settled_since_save += 1
        self._save_state(force=True)

        self._check_expectancy()

    def _refit_calibration(self) -> None:
        if self.model is None or not self.model.trained:
            return
        if len(self.calibration_points) < self.config.min_calibration_points:
            return
        learning.refit_calibration(self.model, self.calibration_points, self.config.min_calibration_points)

    def _check_expectancy(self) -> None:
        """Warn only when a losing record is statistically real.

        This used to fire whenever the running total was negative across 60
        trades, while stats refuses to judge an edge below 200. So it rang on
        ordinary variance, and an alert that cries wolf trains you to ignore
        the one time it is telling the truth.

        Waits for EDGE_REFUTED: the confidence interval sits entirely below
        zero, meaning the loss is resolved rather than sampled from noise.
        """
        rows = [
            _settlement_from_row(row)
            for row in self.store.load_settlements(limit=400)
            if not charter.is_synthetic(row.get("source"))
        ]
        rows = [item for item in rows if item.outcome in (charter.WIN, charter.LOSS)]
        if len(rows) < 3:
            return
        verdict = stats.edge_verdict([float(item.net_pips or 0.0) for item in rows])
        if not verdict.refuted:
            return
        self._alert_once(
            digest.ALERT_NEGATIVE_EXPECTANCY,
            f"Last {verdict.n} settled real trades net {verdict.mean_net_pips:.1f} pips "
            f"after costs (95% CI {verdict.ci_low:.1f} to {verdict.ci_high:.1f}). "
            "Resolved as a loss, not noise.",
            key=f"expectancy|{minute_bucket()[:10]}",
        )

    def _has_candle(self, stamp: datetime) -> bool:
        target = parse_datetime(stamp)
        return any(candle.timestamp == target for candle in self.candles)

    # -------------------------------------------------------- scheduled jobs
    # How late a scheduled job may start and still be considered due.
    SCHEDULE_GRACE_MINUTES = 90

    def due_nightly(self, at: datetime | None = None) -> bool:
        return self._due_at(at, self.config.nightly_hour_utc, self.config.nightly_minute_utc)

    def due_digest(self, at: datetime | None = None) -> bool:
        return self._due_at(at, self.config.digest_hour_utc, self.config.digest_minute_utc)

    def _due_at(self, at: datetime | None, hour: int, minute: int) -> bool:
        """True once the scheduled minute has passed, and for a while after.

        An exact minute match only works if something wakes the bot at exactly
        23:55. Nothing can promise that here: a cron fires on its own schedule,
        and GitHub queues jobs and routinely starts them minutes late. Matching
        exactly meant a half-hourly cron -- which never lands on :55 -- would
        skip the nightly and the digest every single day, silently, while every
        other cycle looked healthy.

        Job slots make the slack safe. claim_job("nightly", date) and the digest
        equivalent mean a late run cannot fire twice in one day.
        """
        stamp = at or now()
        scheduled = stamp.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if stamp < scheduled:
            # The 23:55 job is due when the clock reads 00:30 the next day.
            scheduled -= timedelta(days=1)
        elapsed = (stamp - scheduled).total_seconds() / 60.0
        return 0.0 <= elapsed <= self.SCHEDULE_GRACE_MINUTES

    def _training_candles(self) -> list:
        """Every bar we have, not the live window.

        self.candles is a rolling CANDLE_HISTORY-sized buffer -- a few hundred
        bars, enough to generate one signal and deliberately small. Training on
        it meant the nightly job saw five hours of market no matter how much
        history the database held, which is below NIGHTLY_MIN_SAMPLES and so
        silently refused to train forever.

        Falls back to the live buffer when the store is empty or unreadable, so
        a first run with no backfill still behaves as it always did.
        """
        limit = int(getattr(self.config, "training_history", 200_000) or 200_000)
        try:
            rows = self.store.load_candles(self.config.pair, limit=limit)
        except Exception:  # noqa: BLE001 - a training run must not kill the cycle
            return list(self.candles)
        by_ts: dict[datetime, Candle] = {}
        for row in rows:
            try:
                candle = Candle.from_mapping(row)
                by_ts[candle.timestamp] = candle
            except (ValueError, KeyError, TypeError):
                continue
        if not by_ts:
            return list(self.candles)
        return sorted(by_ts.values(), key=lambda item: item.timestamp)

    def nightly_job(self, at: datetime | None = None) -> dict:
        stamp = at or now()
        bucket = stamp.strftime("%Y-%m-%d")
        job_bucket = bucket if self.config.pair == "EUR/USD" else f"{self.config.pair}:{bucket}"
        if not self.store.claim_job("nightly", job_bucket, runner=self.config.runner):
            return {"skipped": True, "reason": "Nightly already ran for this date."}

        settlements = self.store.load_settlements(limit=4000)
        model, ensemble, report = learning.run_nightly(
            self._training_candles(),
            settlements,
            config=self.config,
            champion=self.model,
        )
        if model is not None:
            self.model = model
        if ensemble.ready:
            self.ensemble = ensemble
        if (
            not getattr(self.provider, "synthetic", False)
            and report.system is not None
            and getattr(report.system, "council_state", None)
            and not self.weights.active()
        ):
            self.weights = CouncilWeights.from_dict(report.system.council_state)
        self.nightly_report = report.to_dict()
        self.store.save_model_version(
            {
                "version_id": job_bucket,
                "report": report.to_dict(),
                "created_at": stamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
        )
        self._save_state(force=True)
        if report.rolled_back:
            self._alert_once(digest.ALERT_ROLLBACK, report.reason, key=f"rollback|{job_bucket}")
        for peer in self.peer_bots.values():
            peer.nightly_job(at=stamp)
        return report.to_dict()

    def digest_job(self, at: datetime | None = None) -> str:
        stamp = at or now()
        bucket = stamp.strftime("%Y-%m-%d")
        if not self.store.claim_job("digest", bucket, runner=self.config.runner):
            return ""

        today = stamp.strftime("%Y-%m-%d")
        signals = [
            row
            for row in self.store.load_signals(limit=2000)
            if str(row.get("generated_at", "")).startswith(today)
        ]
        settlement_rows = self.store.load_settlements(limit=2000)
        real = [row for row in settlement_rows if not charter.is_synthetic(row.get("source"))]
        stats = settle.summarise([_settlement_from_row(row) for row in real])
        calibrator = self.model.calibrator if self.model else None

        text = digest.build_digest(
            day=bucket,
            signals=signals,
            stats=stats,
            unresolved=stats.unresolved,
            calibration_state=(
                "CALIBRATED" if calibrator and calibrator.fitted and not calibrator.flat
                else "FLAT" if calibrator and calibrator.flat
                else "RAW"
            ),
            calibration_samples=(calibrator.samples if calibrator else 0),
            model_mode=self.model.mode if self.model else MODE_VETO,
            council_weights=self.weights.weights,
            learning_events=[row for row in self.store.load_events(limit=400)],
            key_pool=self.store.load_key_pool() or {},
            runner=self.config.runner,
            alerts=self.store.recent_alerts(limit=50),
            nightly=self.nightly_report,
        )
        if self.telegram is not None and self.telegram.configured:
            self.telegram.send(text)
        return text

    def run_scheduled(self, at: datetime | None = None) -> None:
        """Automation entry point: fire whatever is due right now."""
        stamp = at or now()
        if self.due_nightly(stamp):
            self.nightly_job(stamp)
        if self.due_digest(stamp):
            self.digest_job(stamp)

    # ---------------------------------------------------------------- alerts
    def _alert_once(self, kind: str, detail: str, *, key: str, notify: bool = True) -> None:
        """Record an alert at most once per key, for the life of the key.

        ``store.alert`` inserts unconditionally; the dedupe_key column only
        does anything when the caller checks it, so without this a poll that
        fails on every pass writes a row per pass -- thousands a run.
        ``notify`` is switched off for transport diagnostics: reporting over
        Telegram that Telegram is broken cannot work, and would only spam.
        """
        if key in self.alerted_today:
            return
        if self.store.alert_seen(key):
            # Already told the human in an earlier run. This container just
            # booted, so the in-memory set is empty and would otherwise let the
            # same alert through again on every wake-up.
            self.alerted_today.add(key)
            return
        self.alerted_today.add(key)
        self.store.alert(kind, detail, dedupe_key=key)
        if notify and self.telegram is not None and self.telegram.configured:
            self.telegram.send(digest.alert_message(kind, detail))

    def _ensure_baseline_veto_model(self, *, before: datetime | None = None) -> None:
        """Fit a causal MODE_VETO baseline on real prior bars (or cold-start neutral VETO)."""
        if self.model is not None and self.model.trained:
            return
        if getattr(self.provider, "synthetic", False) and not self.config.allow_synthetic_learning:
            return

        history = self._training_candles()
        if before is not None:
            cutoff = parse_datetime(before)
            rep_h = learning._representative_horizon(history[-600:] if history else [], self.config)
            # Purge by rep_h minutes so no training label's exit crosses ``before``.
            safe_cutoff = cutoff - timedelta(minutes=max(1, rep_h))
            history = closed_before(history, safe_cutoff)
        history = history[-6000:]
        if len(history) >= 60:
            inst = instruments.for_symbol(self.config.pair)
            rep_horizon = learning._representative_horizon(history, self.config)
            samples = learning.build_training_samples(history, horizon=rep_horizon, instrument=inst)
            if len(samples) >= 40:
                try:
                    fitted = OnlineLogisticModel(mode=MODE_VETO).fit(
                        [item.to_pair() for item in samples], epochs=80
                    )
                    fitted.mode = MODE_VETO
                    self.model = fitted
                    return
                except ValueError:
                    pass
        self.model = OnlineLogisticModel.cold_start_veto()

    def reconcile_settled_signals(self, *, _as_peer: bool = False) -> int:
        """Idempotently replay historical settled real outcomes not yet recorded in memory."""
        replayed = 0
        if not getattr(self.provider, "synthetic", False):
            signals_by_id = {
                str(r.get("signal_id")): r
                for r in self.store.load_signals(limit=500)
                if r.get("signal_id") and str(r.get("pair") or "EUR/USD").upper() == self.config.pair
            }
            stl_rows = [
                r
                for r in self.store.load_settlements(limit=500)
                if r.get("signal_id") in signals_by_id and not charter.is_synthetic(r.get("source"))
            ]
            stl_rows.sort(key=lambda r: str(r.get("settled_at", "")))
            for s_row in stl_rows:
                sid = str(s_row["signal_id"])
                if not self.mistakes.mark_reconciled(sid):
                    continue
                sig_raw = signals_by_id.get(sid)
                if not sig_raw:
                    continue
                try:
                    sig = Signal.from_dict(sig_raw)
                    stl = _settlement_from_row(s_row)
                except (KeyError, ValueError, TypeError):
                    continue
                if stl.outcome in (charter.TIED, charter.UNRESOLVED):
                    continue
                winning = settle.winning_side(stl)
                if winning is not None:
                    self.weights.observe(_votes_from_signal(sig), winning)
                self.mistakes.record(
                    stl.outcome,
                    MistakeMemory.fingerprint(
                        sig.regime,
                        session_of(sig.entry_time, instruments.for_symbol(self.config.pair))
                        if self.config.session_filter
                        else None,
                        sig.side,
                    ),
                )
                label = settle.learning_label(sig, stl)
                if label in (0, 1) and sig.features:
                    self._ensure_baseline_veto_model(before=sig.entry_time)
                    eff_lr = float(self.config.online_learning_rate)
                    if stl.outcome == charter.LOSS:
                        eff_lr = min(0.05, eff_lr * 1.35)
                    step = learning.online_step(
                        self.model,
                        sig.features,
                        label,
                        source=sig.source,
                        learning_rate=eff_lr,
                    )
                    if step.raw is not None:
                        self.calibration_points.append((float(step.raw), int(label)))
                replayed += 1
            if replayed > 0:
                self.calibration_points = self.calibration_points[-4000:]
                self._refit_calibration()
                self._save_state(force=True)

        if not _as_peer:
            for peer in self.peer_bots.values():
                replayed += peer.reconcile_settled_signals(_as_peer=True)
        return replayed

    def save_all_states(self, *, force: bool = True) -> None:
        """Persist primary and peer market states immediately."""
        self._save_state(force=force)
        for peer in self.peer_bots.values():
            peer._save_state(force=force)

    # ------------------------------------------------------------ persistence
    def _save_state(self, *, force: bool = False) -> None:
        if not force and self.settled_since_save < SAVE_STATE_EVERY:
            return
        payload: dict[str, Any] = {
            "model": self.model.to_dict() if self.model else None,
            "ensemble": self.ensemble.to_dict(),
            "runtime": {
                "paused": self.paused,
                "telegram_offset": (
                    self.telegram.offset if self.telegram is not None else self._saved_telegram_offset
                ),
                "stale_feed_minutes": self.stale_feed_minutes,
                "telegram_format": self.telegram_format.to_dict(),
                "shadow_recovery": dict(self.shadow_recovery),
            },
            "calibration_points": [[point[0], point[1]] for point in self.calibration_points[-2000:]],
            "mistakes": self.mistakes.to_dict(),
        }
        self.store.save_model_state(payload, pair=self.config.pair)
        self.store.save_council_weights(self.weights.to_dict(), pair=self.config.pair)
        self.settled_since_save = 0
        if self.provider is not None and hasattr(self.provider, "pool"):
            self.store.save_key_pool(self.provider.pool.to_dict())

    def snapshot(self) -> dict:
        calibrator = self.model.calibrator if self.model else None
        return {
            "cycles": self.cycles,
            "abstentions": self.abstentions,
            "paused": self.paused,
            "candles": len(self.candles),
            "model_trained": bool(self.model and self.model.trained),
            "model_mode": self.model.mode if self.model else MODE_VETO,
            "calibration": (
                "FLAT" if calibrator and calibrator.flat
                else "CALIBRATED" if calibrator and calibrator.fitted
                else "RAW"
            ),
            "council_weights": self.weights.weights,
            "last_cycle": self.last_cycle.to_dict() if self.last_cycle else None,
            "nightly": self.nightly_report,
        }

    def to_json(self) -> str:
        return json.dumps(self.snapshot(), indent=2, default=str)


# --------------------------------------------------------------------- helpers
def _exit_stamp(signal: Signal) -> datetime:
    from datetime import timedelta

    return parse_datetime(signal.expiry_time) - timedelta(minutes=1)


def _votes_from_signal(signal: Signal) -> dict[str, str]:
    votes: dict[str, str] = {}
    for row in signal.council:
        side = str(row.get("side", ""))
        if side in ("BUY", "SELL"):
            votes[str(row.get("name"))] = side
    return votes


def _settlement_from_row(row: dict):
    from .domain import Settlement

    return Settlement.from_dict(row)


def _summarise_market_reports(bucket: str, per_pair: Sequence[tuple[str, CycleReport]]) -> CycleReport:
    """Build a single top-level CycleReport across all evaluated markets.

    Without this, an off-hours FX data-gate abstention overwrote the summary
    heartbeat last, hiding active 24/7 crypto evaluations from /status and
    store.latest_heartbeat().
    """
    emitted = [(sym, rep) for sym, rep in per_pair if rep.seal_state == charter.SEAL_SIGNAL]
    if emitted:
        first_sym, first_rep = emitted[0]
        summary_reason = " | ".join(f"{sym}: {rep.reason}" for sym, rep in emitted)
        return CycleReport(
            bucket=bucket,
            seal_state=charter.SEAL_SIGNAL,
            reason=summary_reason,
            regime=first_rep.regime,
            signal_id=first_rep.signal_id,
        )

    late = [rep for _, rep in per_pair if rep.seal_state == charter.SEAL_DISCARDED_LATE]
    seal_state = (
        charter.SEAL_DISCARDED_LATE
        if late
        else charter.SEAL_ABSTAIN
        if any(rep.seal_state == charter.SEAL_ABSTAIN for _, rep in per_pair)
        else per_pair[0][1].seal_state
    )
    active_regime = next(
        (rep.regime for _, rep in per_pair if rep.regime and rep.regime != "UNKNOWN"),
        "UNKNOWN",
    )
    parts: list[str] = []
    for sym, rep in per_pair:
        short = str(rep.reason or "").split(";")[0].strip()
        if rep.regime and rep.regime != "UNKNOWN":
            parts.append(f"{sym} ({rep.regime}): {short}")
        else:
            parts.append(f"{sym}: {short}")
    return CycleReport(
        bucket=bucket,
        seal_state=seal_state,
        reason=" | ".join(parts),
        regime=active_regime,
    )


def votes_to_dicts(votes: Sequence[Vote]) -> list[dict]:
    return [vote.to_dict() for vote in votes]
