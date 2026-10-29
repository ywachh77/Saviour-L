#!/usr/bin/env python3
"""
Expiry-Aware Scalping Methods — Original Indicators Not Based on Existing TA
=========================================================================
Creates scalping methods that work across all expiry periods (1-min, 3-min, 30-min).
Each method uses original mathematical constructs, not RSI/MACD/Bollinger/etc.

Key principles:
  - Each method adapts thresholds based on expiry (tighter for 1-min, wider for 30-min)
  - Uses custom indicators from custom_indicators.py (VPIN proxy, Smart Money Index, etc.)
  - Cost-aware: refuses to trade when ATR cannot cover spread
  - Symmetric across BUY/SELL

Methods:
  1. flow_imbalance_scalp — Trade order flow imbalance from body pressure
  2. vpin_breakout_scalp — Trade VPIN expansion breakouts
  3. smart_money_scalp — Follow smart money accumulation/distribution
  4. liquidity_gap_scalp — Trade liquidity gap continuations
  5. kyle_impact_scalp — Trade when Kyle's lambda signals strong market
  6. session_velocity_scalp — Trade session velocity shifts
  7. tick_imbalance_scalp — Trade tick volume imbalance extremes
"""

from __future__ import annotations

from typing import Callable

from . import charter
from .domain import Candle, Decision

# ─── Expiry-aware thresholds ─────────────────────────────────────────
# Each expiry has different optimal thresholds. 1-min = microstructure,
# 3-min = moderate, 30-min = swing scalp.

EXPIRY_CONFIGS = {
    1: {
        "min_atr_pips": 1.0,       # Tight: need at least 1 pip ATR
        "spread_cover": 1.5,       # 1.5x spread minimum
        "confidence_min": 0.15,    # Lower bar for fast signals
        "body_frac_min": 0.30,     # Body must fill 30% of range
        "imbalance_threshold": 0.15,
        "vpin_threshold": 0.35,
        "gap_ratio_min": 1.5,
        "lambda_threshold": 0.15,
        "velocity_shift_min": 0.30,
        "tick_imbalance_min": 0.15,
    },
    3: {
        "min_atr_pips": 1.5,       # Moderate
        "spread_cover": 1.3,
        "confidence_min": 0.20,
        "body_frac_min": 0.35,
        "imbalance_threshold": 0.20,
        "vpin_threshold": 0.40,
        "gap_ratio_min": 1.8,
        "lambda_threshold": 0.20,
        "velocity_shift_min": 0.40,
        "tick_imbalance_min": 0.20,
    },
    30: {
        "min_atr_pips": 3.0,       # Wider: need more movement
        "spread_cover": 1.0,
        "confidence_min": 0.25,
        "body_frac_min": 0.40,
        "imbalance_threshold": 0.25,
        "vpin_threshold": 0.50,
        "gap_ratio_min": 2.0,
        "lambda_threshold": 0.25,
        "velocity_shift_min": 0.50,
        "tick_imbalance_min": 0.25,
    },
}

SCALPING_EXPIRY_PROTOTYPES = (
    "flow_imbalance_scalp",
    "vpin_breakout_scalp",
    "smart_money_scalp",
    "liquidity_gap_scalp",
    "kyle_impact_scalp",
    "session_velocity_scalp",
    "tick_imbalance_scalp",
)

SCALPING_EXPIRY_MEMBERS = {
    "flow_imbalance_scalp": "Expiry-aware: trade order flow imbalance from body pressure",
    "vpin_breakout_scalp": "Expiry-aware: trade VPIN expansion breakouts",
    "smart_money_scalp": "Expiry-aware: follow smart money accumulation/distribution",
    "liquidity_gap_scalp": "Expiry-aware: trade liquidity gap continuations",
    "kyle_impact_scalp": "Expiry-aware: trade when Kyle's lambda signals strong market",
    "session_velocity_scalp": "Expiry-aware: trade session velocity shifts",
    "tick_imbalance_scalp": "Expiry-aware: trade tick volume imbalance extremes",
}

NOT_ENOUGH_BARS = "Not enough completed candles to calculate indicators."


def _expiry_config(expiry_minutes: int) -> dict:
    """Get threshold config for an expiry period."""
    return EXPIRY_CONFIGS.get(expiry_minutes, EXPIRY_CONFIGS[3])


def _cost_guard(
    features: dict[str, float] | None,
    spread_pips: float,
    expiry_minutes: int,
    probability_up: float | None = None,
) -> Decision | None:
    """Return an abstaining Decision when the bar cannot pay for itself."""
    if features is None:
        return Decision("NO SIGNAL", NOT_ENOUGH_BARS, probability_up, 0)
    cfg = _expiry_config(expiry_minutes)
    atr_pips = float(features.get("atr_pips", 0.0) or 0.0)
    needed = float(spread_pips) * cfg["spread_cover"]
    if atr_pips >= needed and atr_pips >= cfg["min_atr_pips"]:
        return None
    projected = charter.SCALP_ATR_CAPTURE * atr_pips
    return Decision(
        "NO SIGNAL",
        f"Scalp suppressed by cost: ATR {atr_pips:.2f}p projects {projected:.1f}p "
        f"against {needed:.2f}p cost floor for {expiry_minutes}-min expiry.",
        probability_up,
        0,
    )


def _abstain(reason: str, probability_up: float | None = None) -> Decision:
    return Decision("NO SIGNAL", reason, probability_up, 0)


# ─── 1. Flow Imbalance Scalp ─────────────────────────────────────────
def flow_imbalance_scalp(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    expiry_minutes: int = 1,
    **_ignored,
) -> Decision:
    """Trade order flow imbalance from body pressure.

    Uses the custom `body_pressure_imbalance` indicator — a proxy for
    buy/sell order flow derived from candle geometry. When the imbalance
    exceeds the threshold AND the current candle confirms direction,
    enter with the flow.

    Original concept: uses body pressure as a flow proxy, not standard
    volume or RSI.
    """
    guard = _cost_guard(features, spread_pips, expiry_minutes, probability_up)
    if guard is not None:
        return guard

    cfg = _expiry_config(expiry_minutes)
    imbalance = float(features.get("body_pressure_imbalance", 0.0) or 0.0)
    body_frac = abs(candle.close - candle.open) / max(candle.high - candle.low, 1e-12)
    close_pos = float(features.get("close_position", 0.5) or 0.5)

    if imbalance >= cfg["imbalance_threshold"] and body_frac >= cfg["body_frac_min"]:
        if candle.close > candle.open and close_pos >= 0.60:
            return Decision(
                "BUY",
                f"Flow imbalance: body pressure {imbalance:+.3f} with {body_frac:.0%} body, "
                f"closing top of range ({expiry_minutes}-min expiry).",
                probability_up,
                2,
            )
    if imbalance <= -cfg["imbalance_threshold"] and body_frac >= cfg["body_frac_min"]:
        if candle.close < candle.open and close_pos <= 0.40:
            return Decision(
                "SELL",
                f"Flow imbalance: body pressure {imbalance:+.3f} with {body_frac:.0%} body, "
                f"closing bottom of range ({expiry_minutes}-min expiry).",
                probability_up,
                2,
            )
    return _abstain("No strong order flow imbalance with candle confirmation.", probability_up)


# ─── 2. VPIN Breakout Scalp ──────────────────────────────────────────
def vpin_breakout_scalp(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    expiry_minutes: int = 1,
    **_ignored,
) -> Decision:
    """Trade VPIN expansion breakouts.

    When VPIN proxy exceeds the threshold AND price breaks a recent
    range with trend alignment, enter. High VPIN = informed trading
    active = potential for sharp directional moves.
    """
    guard = _cost_guard(features, spread_pips, expiry_minutes, probability_up)
    if guard is not None:
        return guard

    cfg = _expiry_config(expiry_minutes)
    vpin = float(features.get("vpin_proxy", 0.0) or 0.0)
    body_atr = float(features.get("body_atr_ratio", 0.0) or 0.0)
    close_pos = float(features.get("close_position", 0.5) or 0.5)
    trend_up = float(features.get("ema_fast", 0) or 0) >= float(features.get("ema_slow", 0) or 0)
    trend_down = float(features.get("ema_fast", 0) or 0) <= float(features.get("ema_slow", 0) or 0)

    if vpin >= cfg["vpin_threshold"] and body_atr >= 0.50:
        if candle.close > candle.open and close_pos >= 0.65 and trend_up:
            return Decision(
                "BUY",
                f"VPIN breakout: informed flow active (VPIN {vpin:.2f}) with "
                f"{body_atr:.2f} ATR body in uptrend ({expiry_minutes}-min expiry).",
                probability_up,
                2,
            )
        if candle.close < candle.open and close_pos <= 0.35 and trend_down:
            return Decision(
                "SELL",
                f"VPIN breakout: informed flow active (VPIN {vpin:.2f}) with "
                f"{body_atr:.2f} ATR body in downtrend ({expiry_minutes}-min expiry).",
                probability_up,
                2,
            )
    return _abstain("No VPIN expansion breakout with trend alignment.", probability_up)


# ─── 3. Smart Money Scalp ────────────────────────────────────────────
def smart_money_scalp(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    expiry_minutes: int = 1,
    **_ignored,
) -> Decision:
    """Follow smart money accumulation/distribution.

    When the Smart Money Index shows late-session buying (positive) or
    selling (negative) AND the candle confirms with a strong body,
    enter with the smart money direction.
    """
    guard = _cost_guard(features, spread_pips, expiry_minutes, probability_up)
    if guard is not None:
        return guard

    cfg = _expiry_config(expiry_minutes)
    smi = float(features.get("smart_money_index", 0.0) or 0.0)
    body_frac = abs(candle.close - candle.open) / max(candle.high - candle.low, 1e-12)
    close_pos = float(features.get("close_position", 0.5) or 0.5)

    if smi >= cfg["velocity_shift_min"] and body_frac >= cfg["body_frac_min"]:
        if candle.close > candle.open and close_pos >= 0.58:
            return Decision(
                "BUY",
                f"Smart money: late-session accumulation detected (SMI {smi:+.3f}) "
                f"with confirming candle ({expiry_minutes}-min expiry).",
                probability_up,
                2,
            )
    if smi <= -cfg["velocity_shift_min"] and body_frac >= cfg["body_frac_min"]:
        if candle.close < candle.open and close_pos <= 0.42:
            return Decision(
                "SELL",
                f"Smart money: late-session distribution detected (SMI {smi:+.3f}) "
                f"with confirming candle ({expiry_minutes}-min expiry).",
                probability_up,
                2,
            )
    return _abstain("No smart money accumulation/distribution signal.", probability_up)


# ─── 4. Liquidity Gap Scalp ─────────────────────────────────────────
def liquidity_gap_scalp(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    expiry_minutes: int = 1,
    **_ignored,
) -> Decision:
    """Trade liquidity gap continuations.

    When a liquidity gap is detected (gap > threshold) AND it aligns
    with the recent trend, enter in the gap direction. Large gaps in
    the trend direction signal strong continuation.
    """
    guard = _cost_guard(features, spread_pips, expiry_minutes, probability_up)
    if guard is not None:
        return guard

    cfg = _expiry_config(expiry_minutes)
    gap_signal = float(features.get("liquidity_gap_detector", 0.0) or 0.0)
    body_frac = abs(candle.close - candle.open) / max(candle.high - candle.low, 1e-12)
    close_pos = float(features.get("close_position", 0.5) or 0.5)

    gap_ratio = abs(gap_signal)
    if gap_ratio >= cfg["gap_ratio_min"] and body_frac >= cfg["body_frac_min"]:
        if gap_signal > 0 and candle.close > candle.open and close_pos >= 0.60:
            return Decision(
                "BUY",
                f"Liquidity gap: gap signal {gap_signal:+.2f} (ratio {gap_ratio:.2f}) "
                f"in bullish direction with confirming candle ({expiry_minutes}-min expiry).",
                probability_up,
                2,
            )
        if gap_signal < 0 and candle.close < candle.open and close_pos <= 0.40:
            return Decision(
                "SELL",
                f"Liquidity gap: gap signal {gap_signal:+.2f} (ratio {gap_ratio:.2f}) "
                f"in bearish direction with confirming candle ({expiry_minutes}-min expiry).",
                probability_up,
                2,
            )
    return _abstain("No qualifying liquidity gap continuation.", probability_up)


# ─── 5. Kyle Impact Scalp ───────────────────────────────────────────
def kyle_impact_scalp(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    expiry_minutes: int = 1,
    **_ignored,
) -> Decision:
    """Trade when Kyle's lambda signals a strong market.

    When the Kyle lambda proxy exceeds the threshold AND the candle
    has a strong body in the direction of net flow, enter. High lambda
    = strong price impact = informed trading = potential continuation.
    """
    guard = _cost_guard(features, spread_pips, expiry_minutes, probability_up)
    if guard is not None:
        return guard

    cfg = _expiry_config(expiry_minutes)
    lambda_val = float(features.get("kyle_lambda_proxy", 0.0) or 0.0)
    flow_dir = float(features.get("body_pressure_imbalance", 0.0) or 0.0)
    body_frac = abs(candle.close - candle.open) / max(candle.high - candle.low, 1e-12)
    close_pos = float(features.get("close_position", 0.5) or 0.5)

    if lambda_val >= cfg["lambda_threshold"] and body_frac >= cfg["body_frac_min"]:
        if flow_dir > 0 and candle.close > candle.open and close_pos >= 0.60:
            return Decision(
                "BUY",
                f"Kyle impact: price impact {lambda_val:.3f} with positive flow "
                f"({expiry_minutes}-min expiry).",
                probability_up,
                2,
            )
        if flow_dir < 0 and candle.close < candle.open and close_pos <= 0.40:
            return Decision(
                "SELL",
                f"Kyle impact: price impact {lambda_val:.3f} with negative flow "
                f"({expiry_minutes}-min expiry).",
                probability_up,
                2,
            )
    return _abstain("No qualifying Kyle impact signal.", probability_up)


# ─── 6. Session Velocity Scalp ──────────────────────────────────────
def session_velocity_scalp(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    expiry_minutes: int = 1,
    **_ignored,
) -> Decision:
    """Trade session velocity shifts.

    When the session velocity shift indicator shows a significant
    change AND the candle confirms direction, enter. This captures
    momentum shifts at session boundaries (London open, NY open, etc.).
    """
    guard = _cost_guard(features, spread_pips, expiry_minutes, probability_up)
    if guard is not None:
        return guard

    cfg = _expiry_config(expiry_minutes)
    velocity = float(features.get("session_velocity_shift", 0.0) or 0.0)
    body_frac = abs(candle.close - candle.open) / max(candle.high - candle.low, 1e-12)
    close_pos = float(features.get("close_position", 0.5) or 0.5)

    if velocity >= cfg["velocity_shift_min"] and body_frac >= cfg["body_frac_min"]:
        if candle.close > candle.open and close_pos >= 0.58:
            return Decision(
                "BUY",
                f"Session velocity: momentum shift {velocity:+.3f} with "
                f"confirming candle ({expiry_minutes}-min expiry).",
                probability_up,
                2,
            )
    if velocity <= -cfg["velocity_shift_min"] and body_frac >= cfg["body_frac_min"]:
        if candle.close < candle.open and close_pos <= 0.42:
            return Decision(
                "SELL",
                f"Session velocity: momentum shift {velocity:+.3f} with "
                f"confirming candle ({expiry_minutes}-min expiry).",
                probability_up,
                2,
            )
    return _abstain("No qualifying session velocity shift.", probability_up)


# ─── 7. Tick Imbalance Scalp ─────────────────────────────────────────
def tick_imbalance_scalp(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    expiry_minutes: int = 1,
    **_ignored,
) -> Decision:
    """Trade tick volume imbalance extremes.

    When the tick volume imbalance (body-weighted volume proxy) exceeds
    the threshold AND the candle confirms, enter. This is an original
    adaptation — uses body-to-range ratio as a volume proxy.
    """
    guard = _cost_guard(features, spread_pips, expiry_minutes, probability_up)
    if guard is not None:
        return guard

    cfg = _expiry_config(expiry_minutes)
    imbalance = float(features.get("tick_volume_imbalance", 0.0) or 0.0)
    body_frac = abs(candle.close - candle.open) / max(candle.high - candle.low, 1e-12)
    close_pos = float(features.get("close_position", 0.5) or 0.5)

    if imbalance >= cfg["tick_imbalance_min"] and body_frac >= cfg["body_frac_min"]:
        if candle.close > candle.open and close_pos >= 0.60:
            return Decision(
                "BUY",
                f"Tick imbalance: volume imbalance {imbalance:+.3f} with "
                f"confirming candle ({expiry_minutes}-min expiry).",
                probability_up,
                2,
            )
    if imbalance <= -cfg["tick_imbalance_min"] and body_frac >= cfg["body_frac_min"]:
        if candle.close < candle.open and close_pos <= 0.40:
            return Decision(
                "SELL",
                f"Tick imbalance: volume imbalance {imbalance:+.3f} with "
                f"confirming candle ({expiry_minutes}-min expiry).",
                probability_up,
                2,
            )
    return _abstain("No qualifying tick volume imbalance.", probability_up)


SCALPING_EXPIRY_REGISTRY: dict[str, Callable[..., Decision]] = {
    "flow_imbalance_scalp": flow_imbalance_scalp,
    "vpin_breakout_scalp": vpin_breakout_scalp,
    "smart_money_scalp": smart_money_scalp,
    "liquidity_gap_scalp": liquidity_gap_scalp,
    "kyle_impact_scalp": kyle_impact_scalp,
    "session_velocity_scalp": session_velocity_scalp,
    "tick_imbalance_scalp": tick_imbalance_scalp,
}

assert set(SCALPING_EXPIRY_REGISTRY) == set(SCALPING_EXPIRY_PROTOTYPES) == set(SCALPING_EXPIRY_MEMBERS)
