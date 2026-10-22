"""Market regime classification.

The regime is context for the council and the digest. It never generates a
signal on its own: it only tells the rule sets which of them are operating in
conditions they were written for.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .domain import Candle
from .indicators import atr, mean

TREND = "TREND"
RANGE = "RANGE"
VOLATILE = "VOLATILE"
UNKNOWN = "UNKNOWN"

# thresholds kept as named constants so the digest can quote them
ADX_TREND_MIN = 22.0
ATR_PCT_VOLATILE = 0.0012
EMA_SEPARATION_MIN = 0.00012  # 1.2 pips of EMA gap on EUR/USD


@dataclass(frozen=True, slots=True)
class RegimeState:
    name: str
    adx: float
    atr_pct: float
    ema_separation: float
    confidence: float
    note: str

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "adx": round(self.adx, 4),
            "atr_pct": round(self.atr_pct, 8),
            "ema_separation": round(self.ema_separation, 8),
            "confidence": round(self.confidence, 4),
            "note": self.note,
        }


def detect_regime(candles: Sequence[Candle], features: dict[str, float] | None) -> RegimeState:
    """Classify the current bar from completed candles only."""
    candles = list(candles)
    if features is None or len(candles) < 30:
        return RegimeState(UNKNOWN, 0.0, 0.0, 0.0, 0.0, "Not enough completed bars to read the regime.")

    adx_value = float(features["adx"])
    atr_pct = float(features["atr_pct"])
    close = candles[-1].close
    ema_separation = abs(features["ema_fast"] - features["ema_slow"]) / close if close else 0.0

    # Volatility wins: a spike invalidates both the trend and range plays.
    atr_values = atr(candles, 14)
    recent = [value for value in atr_values[-30:] if value is not None]
    baseline = mean(recent) if recent else 0.0
    spike = baseline > 0 and float(features["atr"]) > 1.8 * baseline

    if spike or atr_pct > ATR_PCT_VOLATILE:
        return RegimeState(
            VOLATILE,
            adx_value,
            atr_pct,
            ema_separation,
            0.7,
            "One-minute volatility is elevated; continuation and range rules are both unreliable here.",
        )
    if adx_value >= ADX_TREND_MIN and ema_separation >= EMA_SEPARATION_MIN:
        return RegimeState(
            TREND,
            adx_value,
            atr_pct,
            ema_separation,
            min(1.0, (adx_value - ADX_TREND_MIN) / 20.0 + 0.5),
            "ADX and EMA separation agree that directional pressure is present.",
        )
    return RegimeState(
        RANGE,
        adx_value,
        atr_pct,
        ema_separation,
        0.5,
        "Directional pressure is weak; the tape is closer to a range than a trend.",
    )


def regime_allows(regime: str, strategy_id: str) -> bool:
    """Which prototype families are allowed to speak in which regime."""
    if regime == UNKNOWN:
        return strategy_id in ("orbix_confluence", "ml_consensus")
    if regime == VOLATILE:
        return strategy_id in ("orbix_confluence", "breakout")
    if regime == TREND:
        return strategy_id in ("orbix_confluence", "ml_consensus", "trend_pullback", "breakout", "orbix_cadence")
    return strategy_id in ("orbix_confluence", "ml_consensus", "range_reversal", "orbix_cadence")
