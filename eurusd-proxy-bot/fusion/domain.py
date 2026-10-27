"""Core value types: candles, decisions, sealed signals, outcomes."""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .timeutil import add_minutes, floor_minute, format_datetime, parse_datetime


@dataclass(frozen=True, slots=True)
class Candle:
    """One OHLCV candle. ``timestamp`` is the candle OPEN time, always UTC."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", parse_datetime(self.timestamp))
        numbers = (self.open, self.high, self.low, self.close, self.volume)
        if not all(math.isfinite(float(number)) for number in numbers):
            raise ValueError("Candle prices must be finite numbers")
        if min(self.open, self.high, self.low, self.close) <= 0:
            raise ValueError("Candle prices must be greater than zero")
        if self.volume < 0:
            raise ValueError("Candle volume cannot be negative")
        if self.low > min(self.open, self.close):
            raise ValueError("Candle low must not be above its open or close")
        if self.high < max(self.open, self.close):
            raise ValueError("Candle high must not be below its open or close")

    @property
    def body(self) -> float:
        return self.close - self.open

    @property
    def range(self) -> float:
        return self.high - self.low

    @classmethod
    def from_mapping(cls, row: dict[str, Any]) -> "Candle":
        timestamp = row.get("timestamp", row.get("datetime", row.get("time", row.get("date"))))
        if timestamp is None:
            raise ValueError("Each candle needs a timestamp")
        return cls(
            timestamp=parse_datetime(timestamp),
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row.get("volume", 0) or 0),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": format_datetime(self.timestamp),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }


@dataclass(frozen=True, slots=True)
class Decision:
    """One strategy's read on one bar. ``side`` is BUY, SELL or NO SIGNAL."""

    side: str
    reason: str
    probability_up: float | None = None
    score: int = 0

    @property
    def actionable(self) -> bool:
        return self.side in ("BUY", "SELL")


@dataclass(frozen=True, slots=True)
class ModelRead:
    """The meta-model's probability read for one bar."""

    raw: float
    probability: float
    calibrated: bool
    calibration_state: str  # RAW | CALIBRATED | FLAT
    uncertainty: float | None = None
    members: int = 0
    contributions: tuple[dict, ...] = ()
    abstain_reason: str | None = None

    @property
    def usable(self) -> bool:
        """A flat map carries no ranking, so the gate must not trade on it."""
        return self.calibration_state != "FLAT"

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw": round(self.raw, 6),
            "probability": round(self.probability, 6),
            "calibrated": self.calibrated,
            "calibration_state": self.calibration_state,
            "uncertainty": None if self.uncertainty is None else round(self.uncertainty, 6),
            "members": self.members,
            "contributions": [dict(item) for item in self.contributions],
            "abstain_reason": self.abstain_reason,
        }


@dataclass(frozen=True, slots=True)
class Signal:
    """A sealed, immutable research signal.

    ``entry_time`` is the open of the candle the prediction applies to. The
    price is filled from that candle's open once it exists — never invented
    before it does.
    """

    signal_id: str
    pair: str
    side: str
    entry_time: datetime
    expiry_time: datetime
    generated_at: datetime
    strategy: str
    reason: str
    source: str
    probability_up: float | None = None
    probability_raw: float | None = None
    calibration_state: str = "RAW"
    uncertainty: float | None = None
    regime: str = "UNKNOWN"
    council: tuple[dict, ...] = ()
    features: dict[str, float] = field(default_factory=dict)
    horizon_minutes: int = 5
    spread_pips: float = 1.2
    delivery_state: str = "PENDING"
    message_id: str | None = None
    entry_price: float | None = None
    sureshot: bool = False
    present_conditions: dict = field(default_factory=dict)

    @property
    def horizon(self) -> int:
        return self.horizon_minutes

    @classmethod
    def create(
        cls,
        *,
        pair: str,
        side: str,
        entry_time: datetime,
        strategy: str,
        reason: str,
        source: str,
        generated_at: datetime,
        horizon_minutes: int,
        probability_up: float | None = None,
        probability_raw: float | None = None,
        calibration_state: str = "RAW",
        uncertainty: float | None = None,
        regime: str = "UNKNOWN",
        council: tuple[dict, ...] = (),
        features: dict[str, float] | None = None,
        spread_pips: float = 1.2,
        sureshot: bool = False,
        present_conditions: dict | None = None,
    ) -> "Signal":
        entry = parse_datetime(entry_time)
        return cls(
            signal_id=uuid.uuid4().hex,
            pair=pair,
            side=side,
            entry_time=entry,
            expiry_time=add_minutes(entry, horizon_minutes),
            generated_at=parse_datetime(generated_at),
            strategy=strategy,
            reason=reason,
            source=source,
            probability_up=probability_up,
            probability_raw=probability_raw,
            calibration_state=calibration_state,
            uncertainty=uncertainty,
            regime=regime,
            council=council,
            features=dict(features or {}),
            horizon_minutes=int(horizon_minutes),
            spread_pips=float(spread_pips),
            sureshot=bool(sureshot),
            present_conditions=dict(present_conditions or {}),
        )

    def with_entry_price(self, price: float) -> "Signal":
        from dataclasses import replace

        return replace(self, entry_price=float(price))

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal_id": self.signal_id,
            "pair": self.pair,
            "side": self.side,
            "entry_time": format_datetime(self.entry_time),
            "expiry_time": format_datetime(self.expiry_time),
            "generated_at": format_datetime(self.generated_at),
            "horizon_minutes": self.horizon_minutes,
            "strategy": self.strategy,
            "reason": self.reason,
            "source": self.source,
            "probability_up": self.probability_up,
            "probability_raw": self.probability_raw,
            "calibration_state": self.calibration_state,
            "uncertainty": self.uncertainty,
            "regime": self.regime,
            "council": [dict(item) for item in self.council],
            "features": dict(self.features),
            "spread_pips": self.spread_pips,
            "delivery_state": self.delivery_state,
            "message_id": self.message_id,
            "entry_price": self.entry_price,
        }

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "Signal":
        return cls(
            signal_id=str(row["signal_id"]),
            pair=str(row.get("pair", "EUR/USD")),
            side=str(row["side"]),
            entry_time=parse_datetime(row["entry_time"]),
            expiry_time=parse_datetime(row["expiry_time"]),
            generated_at=parse_datetime(row.get("generated_at", row["entry_time"])),
            strategy=str(row.get("strategy", "fusion")),
            reason=str(row.get("reason", "")),
            source=str(row.get("source", "LIVE")),
            probability_up=(None if row.get("probability_up") is None else float(row["probability_up"])),
            probability_raw=(None if row.get("probability_raw") is None else float(row["probability_raw"])),
            calibration_state=str(row.get("calibration_state", "RAW")),
            uncertainty=(None if row.get("uncertainty") is None else float(row["uncertainty"])),
            regime=str(row.get("regime", "UNKNOWN")),
            council=tuple(dict(item) for item in row.get("council", []) or []),
            features={str(k): float(v) for k, v in (row.get("features") or {}).items()},
            horizon_minutes=int(row.get("horizon_minutes", 5)),
            spread_pips=float(row.get("spread_pips", 1.2)),
            delivery_state=str(row.get("delivery_state", "PENDING")),
            message_id=(None if row.get("message_id") is None else str(row["message_id"])),
            entry_price=(None if row.get("entry_price") is None else float(row["entry_price"])),
        )


@dataclass(frozen=True, slots=True)
class Settlement:
    """The cost-charged result of one sealed signal."""

    signal_id: str
    outcome: str  # WIN | LOSS | TIED | UNRESOLVED
    entry_price: float | None
    exit_price: float | None
    gross_pips: float | None
    move_pips: float | None
    net_pips: float | None
    spread_pips: float
    settled_at: datetime
    source: str

    @property
    def resolved(self) -> bool:
        return self.outcome in ("WIN", "LOSS")

    @property
    def tradable(self) -> bool:
        return self.outcome in ("WIN", "LOSS", "TIED")

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal_id": self.signal_id,
            "outcome": self.outcome,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "gross_pips": self.gross_pips,
            "move_pips": self.move_pips,
            "net_pips": self.net_pips,
            "spread_pips": self.spread_pips,
            "settled_at": format_datetime(self.settled_at),
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "Settlement":
        return cls(
            signal_id=str(row["signal_id"]),
            outcome=str(row["outcome"]),
            entry_price=(None if row.get("entry_price") is None else float(row["entry_price"])),
            exit_price=(None if row.get("exit_price") is None else float(row["exit_price"])),
            gross_pips=(None if row.get("gross_pips") is None else float(row["gross_pips"])),
            move_pips=(None if row.get("move_pips") is None else float(row["move_pips"])),
            net_pips=(None if row.get("net_pips") is None else float(row["net_pips"])),
            spread_pips=float(row.get("spread_pips", 1.2)),
            settled_at=parse_datetime(row["settled_at"]),
            source=str(row.get("source", "LIVE")),
        )


NO_SIGNAL = Decision("NO SIGNAL", "No actionable read on this bar.")


def closed_before(candles, stamp):
    """Bars that had already closed when the minute ``stamp`` falls in began.

    A bar timestamped 10:43 covers 10:43:00-10:44:00, so its high, low and
    close do not exist until the clock passes 10:44. A decision to enter at
    the 10:43 open can therefore only use bars that closed before 10:43.

    Letting the entry bar into the feature set is look-ahead, and it is not a
    small effect. Measured on the real indicators: a single 100-pip move in
    the entry bar swings RSI from 100 to 39 and drags the fast EMA 30 pips.
    The model is effectively shown part of the answer before it is asked.

    The walk-forward backtest has always decided on one bar's close and
    entered on the next bar's open. Live did not -- it decided on whatever
    the provider happened to return, which on some polls includes the
    still-forming entry bar. This is what makes the two agree.
    """
    cutoff = floor_minute(stamp)
    return [candle for candle in candles if candle.timestamp < cutoff]
