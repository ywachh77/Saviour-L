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

import json
import signal
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Sequence

from . import charter, digest, heartbeat, learning, settle, stats
from .config import Config
from .council import CouncilWeights, Vote
from .domain import Candle, Signal
from .engine import FusionEngine
from .model import MODE_VETO, BootstrapEnsemble, OnlineLogisticModel
from .provider import TwelveDataProvider
from .store import Store, build_store
from .telegram import HELP_TEXT, TelegramClient
from .timeutil import minute_bucket, now, parse_datetime, seconds_until_next_minute

# How far past the minute boundary the pipeline is allowed to seal before the
# signal is considered late and dropped.
SEAL_OFFSET_SECONDS = 0.5
COMMAND_POLL_SECONDS = 2.0
MAX_STORED_CANDLES = 600
SAVE_STATE_EVERY = 10


def format_signal(signal: Signal) -> str:
    probability = "n/a" if signal.probability_up is None else f"{signal.probability_up:.3f}"
    raw = "n/a" if signal.probability_raw is None else f"{signal.probability_raw:.3f}"
    state = {
        "RAW": "raw (not enough outcomes to calibrate yet)",
        "CALIBRATED": "calibrated",
        "FLAT": "flat — abstained, never sent",
    }[signal.calibration_state]
    lines = [
        f"{signal.pair} {signal.side}",
        f"Entry: open of the {signal.entry_time.strftime('%H:%M')}Z candle",
        f"Expires: {signal.expiry_time.strftime('%H:%M')}Z ({signal.horizon_minutes} min)",
        f"Regime: {signal.regime}",
        f"Model: {probability} — {state} (raw {raw})",
        f"Why: {signal.reason}",
        f"Cost: {signal.spread_pips:.2f} pip spread charged on settlement.",
        "Paper research signal. No broker order is placed.",
    ]
    return "\n".join(lines)


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

    # ------------------------------------------------------------------ setup
    def __post_init__(self) -> None:
        if self.store is None:
            self.store = build_store(self.config)
        if self.provider is None:
            self.provider = TwelveDataProvider(self.config.twelve_data_keys)
        if self.telegram is None and self.config.telegram_ready:
            self.telegram = TelegramClient(self.config.telegram_token, self.config.telegram_owner_chat_id)
        self.engine = FusionEngine(self.config)

    @classmethod
    def from_config(cls, config: Config, **kwargs) -> "FusionBot":
        return cls(config=config, **kwargs)

    def startup(self) -> None:
        """Load persisted state. Safe to call twice."""
        self._load_state()
        self._load_candles()
        self.running = True

    def _load_state(self) -> None:
        row = self.store.load_model_state() or {}
        self.model = OnlineLogisticModel.from_dict(row.get("model"))
        self.ensemble = BootstrapEnsemble.from_dict(row.get("ensemble"))
        self.weights = CouncilWeights.from_dict(self.store.load_council_weights())
        runtime = row.get("runtime") or {}
        self.paused = bool(runtime.get("paused", False))
        offset = runtime.get("telegram_offset")
        if self.telegram is not None and isinstance(offset, int) and offset > 0:
            self.telegram.offset = offset
        self.stale_feed_minutes = int(runtime.get("stale_feed_minutes") or 0)
        points = row.get("calibration_points") or []
        self.calibration_points = [
            (float(item[0]), int(item[1])) for item in points if len(item) == 2
        ][-4000:]

    def _load_candles(self) -> None:
        rows = self.store.load_candles(self.config.pair, limit=MAX_STORED_CANDLES)
        candles: list[Candle] = []
        for row in rows:
            try:
                candles.append(Candle.from_mapping(row))
            except (ValueError, KeyError, TypeError):
                continue
        candles.sort(key=lambda candle: candle.timestamp)
        self.candles = candles[-MAX_STORED_CANDLES:]

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

    def poll_commands(self) -> None:
        if self.telegram is None or not self.telegram.configured:
            return
        before = self.telegram.offset
        for command in self.telegram.poll_commands(timeout_seconds=0):
            if not command.from_owner:
                # Silent by design: the bot does not talk to strangers.
                continue
            self.handle_command(command.command)
        if self.telegram.offset != before:
            # Acknowledge the updates. This container exits in a few minutes, so
            # unless the cursor is written down the next one starts from zero and
            # re-answers every command from the last 24 hours, every cycle.
            self._save_state(force=True)

    def handle_command(self, command: str) -> str:
        if command == "/start":
            self.paused = False
            self._save_state(force=True)
            text = "FUSION BOT is live for EUR/USD M1 research signals.\n\n" + HELP_TEXT
        elif command == "/help":
            text = HELP_TEXT
        elif command == "/status":
            text = self.status_text()
        elif command == "/pause":
            self.paused = True
            self._save_state(force=True)
            text = "Signals paused. Settlement and learning continue."
        elif command == "/resume":
            self.paused = False
            self._save_state(force=True)
            text = "Signals resumed."
        else:
            text = "Unknown command.\n\n" + HELP_TEXT
        if self.telegram is not None:
            self.telegram.send(text)
        return text

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
    def minute_cycle(self, at: datetime | None = None) -> CycleReport:
        stamp = at or now()
        bucket = minute_bucket(stamp)
        self.cycles += 1

        if not self.store.claim_job("minute", bucket, runner=self.config.runner):
            # Another runner already owns this minute. Nothing to do, no harm done.
            return CycleReport(bucket, "SKIPPED", "Another runner already claimed this minute.")

        self.ingest(at=stamp)
        report = self.evaluate_minute(bucket=bucket, at=stamp)
        self.last_cycle = report
        self.store.heartbeat(
            heartbeat.build_heartbeat(
                runner=self.config.runner,
                phase="minute",
                extra=report.to_dict(),
            )
        )
        return report

    def ingest(self, at: datetime | None = None) -> int:
        """Fetch the newest candles and merge them into memory and the store."""
        if self.provider is None or not getattr(self.provider, "configured", False):
            return 0
        try:
            fresh = self.provider.time_series(self.config.pair, self.config.timeframe, self.config.candle_history)
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
            self.candles = sorted(known.values(), key=lambda item: item.timestamp)[-MAX_STORED_CANDLES:]
            self.store.save_candles(rows)
        return added

    def evaluate_minute(self, *, bucket: str, at: datetime | None = None) -> CycleReport:
        stamp = at or now()
        if not self.candles:
            return CycleReport(bucket, charter.SEAL_ABSTAIN, "No candles available yet.")

        result = self.engine.evaluate(
            self.candles,
            model=self.model,
            ensemble=self.ensemble,
            weights=self.weights,
            source="LIVE" if not getattr(self.provider, "synthetic", False) else "SYNTHETIC",
            at=stamp,
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
    def deliver(self, signal: Signal) -> None:
        """Queue and send. SENT is recorded only on a confirmed message id."""
        key = f"signal|{signal.signal_id}"
        self.store.outbox_append(
            {"message_key": key, "text": format_signal(signal), "kind": "signal", "state": charter.DELIVERY_PENDING}
        )
        if self.telegram is None or not self.telegram.configured:
            self.store.outbox_update(key, {"state": charter.DELIVERY_FAILED, "last_error": "Telegram not configured"})
            self.store.update_signal(signal.signal_id, {"delivery_state": charter.DELIVERY_FAILED})
            return

        result = self.telegram.send(format_signal(signal))
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
                self.store.update_signal(
                    signal_id,
                    {"delivery_state": charter.DELIVERY_SENT, "message_id": patch["message_id"]},
                )
            else:
                # Still failing. Say which state, and on the last attempt admit
                # we have given up rather than leaving it looking pending.
                state = charter.DELIVERY_FAILED if attempts + 1 >= 3 else result.state
                patch["state"] = state
                self.store.update_signal(signal_id, {"delivery_state": state})
            self.store.outbox_update(str(row.get("message_key")), patch)

    @staticmethod
    def _signal_id_from_key(message_key: object) -> str:
        """`signal|<id>` -> `<id>`. Outbox rows are keyed by what they carry."""
        key = str(message_key or "")
        return key.split("|", 1)[-1] if key.startswith("signal|") else key

    # ------------------------------------------------------------ settlement
    def settle_due(self, at: datetime | None = None) -> int:
        stamp = at or now()
        count = 0
        for row in self.store.pending_signals(self.config.pair):
            try:
                signal = Signal.from_dict(row)
            except (KeyError, ValueError, TypeError):
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
        return count

    def _settle_one(self, signal: Signal) -> None:
        settlement = settle.settle(signal, self.candles)
        self.store.save_settlement(settlement.to_dict())
        label = settle.learning_label(signal, settlement)
        winning = settle.winning_side(settlement)

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

        # Model learning: test before learn, once per confirmed non-tied outcome.
        step = learning.online_step(
            self.model,
            signal.features,
            label,
            source=signal.source,
            learning_rate=self.config.online_learning_rate,
            allow_synthetic=self.config.allow_synthetic_learning,
        )
        self.store.log_event(dict(step.to_dict(), kind="online_step", signal_id=signal.signal_id))

        if step.raw is not None and label is not None:
            self.calibration_points.append((float(step.raw), int(label)))
            self.calibration_points = self.calibration_points[-4000:]
            self._refit_calibration()

        self.settled_since_save += 1
        if self.settled_since_save >= SAVE_STATE_EVERY:
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
        candles = []
        for row in rows:
            try:
                candles.append(Candle.from_mapping(row))
            except (ValueError, KeyError, TypeError):
                continue
        return candles or list(self.candles)

    def nightly_job(self, at: datetime | None = None) -> dict:
        stamp = at or now()
        bucket = stamp.strftime("%Y-%m-%d")
        if not self.store.claim_job("nightly", bucket, runner=self.config.runner):
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
        self.nightly_report = report.to_dict()
        self.store.save_model_version(
            {
                "version_id": bucket,
                "report": report.to_dict(),
                "created_at": stamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
        )
        self._save_state(force=True)
        if report.rolled_back:
            self._alert_once(digest.ALERT_ROLLBACK, report.reason, key=f"rollback|{bucket}")
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
    def _alert_once(self, kind: str, detail: str, *, key: str) -> None:
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
        if self.telegram is not None and self.telegram.configured:
            self.telegram.send(digest.alert_message(kind, detail))

    # ------------------------------------------------------------ persistence
    def _save_state(self, *, force: bool = False) -> None:
        if not force and self.settled_since_save < SAVE_STATE_EVERY:
            return
        payload: dict[str, Any] = {
            "model": self.model.to_dict() if self.model else None,
            "ensemble": self.ensemble.to_dict(),
            "runtime": {
                "paused": self.paused,
                "telegram_offset": self.telegram.offset if self.telegram else None,
                "stale_feed_minutes": self.stale_feed_minutes,
            },
            "calibration_points": [[point[0], point[1]] for point in self.calibration_points[-2000:]],
        }
        self.store.save_model_state(payload)
        self.store.save_council_weights(self.weights.to_dict())
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


def votes_to_dicts(votes: Sequence[Vote]) -> list[dict]:
    return [vote.to_dict() for vote in votes]
