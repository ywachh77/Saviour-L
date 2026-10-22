"""The strategy council's members.

Each member is a transparent rule set that returns BUY, SELL or NO SIGNAL with
a reason a human can read. None of them is a black box, and none of them is
evidence of profitability on its own: passing rules is not a probability.
"""

from __future__ import annotations

from typing import Callable

from . import scalping
from .domain import Candle, Decision

CORE = "orbix_confluence"

# The scalping family is appended last and stays last: it is the newest, the
# least proven, and the most exposed to spread. Order here is documentation of
# that, not a ranking the code depends on.
PROTOTYPES = (
    "trend_pullback",
    "range_reversal",
    "breakout",
    "ml_consensus",
    "orbix_cadence",
) + scalping.SCALPING_PROTOTYPES

MEMBERS = {
    CORE: "Orbix Confluence Gate (6 checks, 5-of-6 with mandatory trend + ADX)",
    "trend_pullback": "Trend pullback reclaims the fast EMA",
    "range_reversal": "Bollinger stretch plus an RSI turn",
    "breakout": "Close beyond the prior 20-bar range",
    "ml_consensus": "Model and short-term trend must agree",
    "orbix_cadence": "Calibrated model read with EMA context",
}
MEMBERS.update(scalping.SCALPING_MEMBERS)

# Volatility band the core gate was written for. Outside it the gate abstains
# rather than pretending a rule set tuned on calm bars still applies.
ATR_PCT_MIN = 0.0
ATR_PCT_MAX = 0.004


def side_of(value: float, *, bullish: bool) -> bool:
    return value > 0 if bullish else value < 0


def orbix_confluence(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    min_checks: int = 5,
    **_ignored,
) -> Decision:
    """The 6-check core gate.

    Requires at least ``min_checks`` of six checks AND the two mandatory ones
    (trend, directional ADX). A rule score is a count, not a probability.
    """
    if features is None:
        return Decision("NO SIGNAL", "Not enough completed candles to calculate indicators.", None, 0)

    atr_pct = features["atr_pct"]
    if atr_pct <= ATR_PCT_MIN or atr_pct > ATR_PCT_MAX:
        return Decision(
            "NO SIGNAL",
            "Core gate abstains: one-minute volatility is outside its operating band.",
            probability_up,
            0,
        )

    close = candle.close
    rsi = features["rsi"]
    checks = {
        "trend": {
            "BUY": features["ema_fast"] > features["ema_slow"] and close > features["ema_fast"],
            "SELL": features["ema_fast"] < features["ema_slow"] and close < features["ema_fast"],
        },
        "momentum": {
            "BUY": features["return_1"] > 0 and features["return_3"] > 0,
            "SELL": features["return_1"] < 0 and features["return_3"] < 0,
        },
        "RSI band": {
            "BUY": 50.0 <= rsi <= 68.0,
            "SELL": 32.0 <= rsi <= 50.0,
        },
        "directional ADX": {
            "BUY": features["adx"] >= 18.0 and features["plus_di"] > features["minus_di"],
            "SELL": features["adx"] >= 18.0 and features["minus_di"] > features["plus_di"],
        },
        "candle quality": {
            "BUY": 0.12 <= features["body_atr_ratio"] <= 1.5 and features["close_position"] >= 0.65,
            "SELL": 0.12 <= features["body_atr_ratio"] <= 1.5 and features["close_position"] <= 0.35,
        },
        "20-bar breakout": {
            "BUY": close > features["prior_high_20"],
            "SELL": close < features["prior_low_20"],
        },
    }

    scores = {side: sum(bool(criteria[side]) for criteria in checks.values()) for side in ("BUY", "SELL")}
    side = max(scores, key=lambda key: scores[key])
    score = scores[side]
    aligned = ", ".join(name for name, criteria in checks.items() if criteria[side])
    mandatory_ok = checks["trend"][side] and checks["directional ADX"][side]

    if score >= min_checks and mandatory_ok:
        return Decision(
            side,
            f"Core {score}/6: {aligned}. Rule score is a count, not a probability.",
            probability_up,
            score,
        )
    if score >= min_checks and not mandatory_ok:
        missing = [name for name in ("trend", "directional ADX") if not checks[name][side]]
        return Decision(
            "NO SIGNAL",
            f"Core abstains: {side} passed {score}/6 but the mandatory checks failed ({', '.join(missing)}).",
            probability_up,
            score,
        )
    return Decision(
        "NO SIGNAL",
        f"Core abstains: strongest side passed {score}/6; "
        f"{min_checks}/6 plus trend and ADX alignment are required.",
        probability_up,
        score,
    )


def trend_pullback(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    **_ignored,
) -> Decision:
    if features is None:
        return Decision("NO SIGNAL", "Not enough completed candles to calculate indicators.", None, 0)
    close = candle.close
    rsi = features["rsi"]
    bullish = previous_close <= features["ema_fast_previous"] and close > features["ema_fast"]
    bearish = previous_close >= features["ema_fast_previous"] and close < features["ema_fast"]
    if features["ema_fast"] > features["ema_slow"] and bullish and 45.0 <= rsi <= 68.0:
        return Decision("BUY", "Uptrend pullback reclaimed the fast EMA with RSI confirmation.", probability_up, 2)
    if features["ema_fast"] < features["ema_slow"] and bearish and 32.0 <= rsi <= 55.0:
        return Decision("SELL", "Downtrend pullback rejected the fast EMA with RSI confirmation.", probability_up, 2)
    return Decision("NO SIGNAL", "No confirmed trend pullback and EMA reclaim is present.", probability_up, 0)


def range_reversal(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    **_ignored,
) -> Decision:
    if features is None:
        return Decision("NO SIGNAL", "Not enough completed candles to calculate indicators.", None, 0)
    close = candle.close
    rsi = features["rsi"]
    rsi_previous = features["rsi_previous"]
    if close <= features["bb_lower"] and rsi < 35.0 and rsi > rsi_previous:
        return Decision("BUY", "Price stretched below its recent range while RSI started turning up.", probability_up, 2)
    if close >= features["bb_upper"] and rsi > 65.0 and rsi < rsi_previous:
        return Decision("SELL", "Price stretched above its recent range while RSI started turning down.", probability_up, 2)
    return Decision("NO SIGNAL", "No confirmed range-edge reversal is present.", probability_up, 0)


def breakout(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    **_ignored,
) -> Decision:
    if features is None:
        return Decision("NO SIGNAL", "Not enough completed candles to calculate indicators.", None, 0)
    close = candle.close
    if close > features["prior_high_20"]:
        return Decision("BUY", "Close broke above the previous 20-candle high.", probability_up, 1)
    if close < features["prior_low_20"]:
        return Decision("SELL", "Close broke below the previous 20-candle low.", probability_up, 1)
    return Decision("NO SIGNAL", "Price is still inside the recent 20-candle range.", probability_up, 0)


def ml_consensus(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    buy_threshold: float = 0.57,
    sell_threshold: float = 0.43,
    **_ignored,
) -> Decision:
    if features is None or probability_up is None:
        return Decision("NO SIGNAL", "The model is not ready; wait for more data.", probability_up, 0)
    if probability_up >= buy_threshold and features["ema_fast"] > features["ema_slow"] and features["return_1"] > 0:
        return Decision("BUY", f"Model leans up ({probability_up:.2f}) and the short-term EMA trend agrees.", probability_up, 2)
    if probability_up <= sell_threshold and features["ema_fast"] < features["ema_slow"] and features["return_1"] < 0:
        return Decision("SELL", f"Model leans down ({probability_up:.2f}) and the short-term EMA trend agrees.", probability_up, 2)
    return Decision("NO SIGNAL", "Model and trend filters do not agree strongly enough.", probability_up, 0)


def orbix_cadence(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    calibrated: bool = False,
    **_ignored,
) -> Decision:
    """A deliberately less selective read, for a steadier collection rhythm.

    Frequency is not accuracy: this member exists to gather outcomes, and it
    says so in its own reason string.
    """
    if features is None or probability_up is None:
        return Decision("NO SIGNAL", "The model is not ready; the cadence read waits.", probability_up, 0)
    minimum = 0.005 if calibrated else 0.010
    edge = float(probability_up) - 0.5
    if abs(edge) < minimum:
        return Decision(
            "NO SIGNAL",
            f"Cadence read abstains: model edge {edge:+.3f} is below the {minimum:.3f} minimum.",
            probability_up,
            0,
        )
    side = "BUY" if edge > 0 else "SELL"
    with_trend = features["ema_fast"] > features["ema_slow"] if side == "BUY" else features["ema_fast"] < features["ema_slow"]
    return Decision(
        side,
        f"Cadence read: model {probability_up:.2f} ({edge:+.2f} from 0.50), "
        f"{'with' if with_trend else 'against'} the short-term EMA trend. "
        "High-frequency collection mode: the estimate is not a promised win rate.",
        probability_up,
        1,
    )


REGISTRY: dict[str, Callable[..., Decision]] = {
    CORE: orbix_confluence,
    "trend_pullback": trend_pullback,
    "range_reversal": range_reversal,
    "breakout": breakout,
    "ml_consensus": ml_consensus,
    "orbix_cadence": orbix_cadence,
}
REGISTRY.update(scalping.SCALPING_REGISTRY)

# Every member must be either the core gate or a prototype, and every prototype
# must have a rule and a human-readable description. A member registered in one
# place but not the others would be invisible to learning, or unvotable.
assert set(REGISTRY) == set(MEMBERS), "a member is missing its description"
assert set(PROTOTYPES) == set(REGISTRY) - {CORE}, "a prototype is not registered"
