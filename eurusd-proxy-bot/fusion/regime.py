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


def classify(
    *,
    atr_value: float,
    atr_pct: float,
    atr_baseline: float,
    adx_value: float,
    close: float,
    ema_fast: float,
    ema_slow: float,
) -> RegimeState:
    """Pure classification from already-computed values.

    Split out from {@link detect_regime} so a backtest can call it per bar
    without recomputing an ATR series each time, which would make a replay
    quadratic. The live path and the backtest therefore share one rule.
    """
    ema_separation = abs(ema_fast - ema_slow) / close if close else 0.0
    spike = atr_baseline > 0 and atr_value > 1.8 * atr_baseline

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


# Continuation scalps need something to continue. Reversal scalps need
# something to revert from. Letting each family speak only in the conditions it
# was written for is most of the value of having a regime at all.
CONTINUATION_SCALPS = frozenset({"scalp_impulse", "scalp_micro_breakout", "scalp_expansion"})
REVERSION_SCALPS = frozenset({"scalp_reversion", "scalp_vwap_reversion", "scalp_pivot_fade"})
TREND_SCALPS = frozenset({"scalp_stoch_cross"})
CONTINUATION_METHODS = frozenset(
    {
        "smc_fvg_displacement",
        "ttm_keltner_squeeze",
        "supertrend_flip",
        "ema_ribbon_pullback",
        "session_box_breakout",
    }
)
REVERSION_METHODS = frozenset(
    {
        "smc_liquidity_sweep",
        "smc_order_block",
        "connors_rsi2_reversion",
        "vwap_sigma_reversion",
    }
)

ALLOWED = {
    # Too early to know anything: only the model-backed reads may speak.
    UNKNOWN: frozenset({"orbix_confluence", "ml_consensus"}),
    # Elevated volatility: breakouts, impulses, squeezes and stop-sweeps can ride it,
    # while passive mean-reversion scalps sit out.
    VOLATILE: frozenset({"orbix_confluence", "breakout", "smc_liquidity_sweep"})
    | CONTINUATION_SCALPS
    | frozenset({"smc_fvg_displacement", "ttm_keltner_squeeze", "supertrend_flip", "session_box_breakout"}),
    # Directional pressure present: continuation plus pullback-to-value & order blocks.
    TREND: frozenset(
        {"orbix_confluence", "ml_consensus", "trend_pullback", "breakout", "orbix_cadence"}
    )
    | CONTINUATION_SCALPS
    | TREND_SCALPS
    | CONTINUATION_METHODS
    | frozenset({"scalp_vwap_reversion", "smc_order_block", "connors_rsi2_reversion"}),
    # No directional pressure: this is where fading the edges & liquidity sweeps belong.
    RANGE: frozenset({"orbix_confluence", "ml_consensus", "range_reversal", "orbix_cadence"})
    | REVERSION_SCALPS
    | TREND_SCALPS
    | REVERSION_METHODS,
}


def regime_allows(regime: str, strategy_id: str) -> bool:
    """Which prototype families are allowed to speak in which regime."""
    return strategy_id in ALLOWED.get(regime, ALLOWED[UNKNOWN])


def detect_regime(candles: Sequence[Candle], features: dict[str, float] | None) -> RegimeState:
    """Classify the current bar from completed candles only."""
    candles = list(candles)
    if features is None or len(candles) < 30:
        return RegimeState(UNKNOWN, 0.0, 0.0, 0.0, 0.0, "Not enough completed bars to read the regime.")

    atr_values = atr(candles, 14)
    recent = [value for value in atr_values[-30:] if value is not None]
    baseline = mean(recent) if recent else 0.0
    close = candles[-1].close
    return classify(
        atr_value=float(features["atr"]),
        atr_pct=float(features["atr_pct"]),
        atr_baseline=baseline,
        adx_value=float(features["adx"]),
        close=close,
        ema_fast=float(features["ema_fast"]),
        ema_slow=float(features["ema_slow"]),
    )
