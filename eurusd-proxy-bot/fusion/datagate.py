"""The data gate: decide whether this minute's data is trustworthy at all.

A signal built on a stale, gappy or absurdly wide feed is worse than no signal,
because it looks identical to a good one. The gate refuses to let that happen
and says why.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

from .domain import Candle

BLOCKED = "BLOCKED"
DEGRADED = "DEGRADED"
OK = "OK"


@dataclass(frozen=True, slots=True)
class GateResult:
    state: str
    quality: str
    age_seconds: float
    spread_pips: float
    spread_source: str
    gaps: int
    candles: int
    reason: str

    @property
    def usable(self) -> bool:
        return self.state != BLOCKED

    def to_dict(self) -> dict:
        return {
            "state": self.state,
            "quality": self.quality,
            "age_seconds": round(self.age_seconds, 1),
            "spread_pips": round(self.spread_pips, 3),
            "spread_source": self.spread_source,
            "gaps": self.gaps,
            "candles": self.candles,
            "reason": self.reason,
        }


def modelled_spread_pips(stamp: datetime | None = None, base: float = 1.2) -> float:
    """Modelled EUR/USD spread when the provider gives no quote.

    Wider outside the London/New York overlap. This is a model, not a broker
    quote, and it is always labelled as such.
    """
    hour = (stamp.hour if stamp is not None else 12) % 24
    if 12 <= hour < 17:
        return round(base, 2)
    if 7 <= hour < 12 or 17 <= hour < 21:
        return round(base * 1.35, 2)
    return round(base * 2.1, 2)


def count_gaps(candles: Sequence[Candle], expected_seconds: float = 60.0, tolerance: float = 5.0) -> int:
    gaps = 0
    for index in range(1, len(candles)):
        delta = (candles[index].timestamp - candles[index - 1].timestamp).total_seconds()
        if abs(delta - expected_seconds) > tolerance:
            gaps += 1
    return gaps


def evaluate(
    candles: Sequence[Candle],
    *,
    now: datetime,
    max_age_seconds: int = 180,
    max_spread_pips: float = 2.5,
    quote: dict | None = None,
    min_candles: int = 60,
    max_gaps: int = 3,
) -> GateResult:
    candles = list(candles)
    if len(candles) < min_candles:
        return GateResult(
            BLOCKED,
            "INSUFFICIENT",
            0.0,
            0.0,
            "none",
            0,
            len(candles),
            f"Only {len(candles)} candles stored; at least {min_candles} are needed for indicators.",
        )

    last = candles[-1]
    age = (now - last.timestamp).total_seconds()
    if quote and quote.get("spread_pips") is not None:
        spread = float(quote["spread_pips"])
        spread_source = "provider"
    else:
        spread = modelled_spread_pips(last.timestamp)
        spread_source = "modelled"

    gaps = count_gaps(candles)

    if age > max_age_seconds:
        return GateResult(
            BLOCKED,
            "STALE",
            age,
            spread,
            spread_source,
            gaps,
            len(candles),
            f"Latest candle is {age:.0f}s old, beyond the {max_age_seconds}s freshness limit.",
        )
    if spread > max_spread_pips:
        return GateResult(
            BLOCKED,
            "WIDE_SPREAD",
            age,
            spread,
            spread_source,
            gaps,
            len(candles),
            f"Spread is {spread:.2f} pips, above the {max_spread_pips:.2f} pip limit.",
        )
    if gaps > max_gaps:
        return GateResult(
            DEGRADED,
            "GAPPY",
            age,
            spread,
            spread_source,
            gaps,
            len(candles),
            f"{gaps} gaps in the recent series; indicators are computed on an incomplete tape.",
        )
    return GateResult(OK, "OK", age, spread, spread_source, gaps, len(candles), "Data is fresh, tight and continuous.")
