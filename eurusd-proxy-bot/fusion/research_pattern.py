"""One transparent, paper-only EUR/USD trend-pullback research pattern.

This module is deliberately NOT registered with the live strategy council and
cannot send orders or signals. It creates a candidate after a completed bar;
any evaluation must enter no earlier than the next bar and charge real costs.
The thresholds are fixed hypotheses, not fitted probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .domain import Candle

PAIR = "EUR/USD"
PIP = 0.0001
EMA_FAST = 20
EMA_SLOW = 50
EMA_SLOPE_LOOKBACK = 5
ATR_PERIOD = 14
RSI_PERIOD = 14
STOP_ATR = 1.0
TARGET_ATR = 1.5
MAX_HOLD_MINUTES = 120
MIN_ATR_TO_SPREAD = 2.0


@dataclass(frozen=True, slots=True)
class PatternCandidate:
    """A research candidate, not an instruction to place a trade."""

    side: str | None
    reason: str
    atr_pips: float | None = None
    entry_reference: float | None = None  # next bar open; unknown at signal time
    stop_pips: float | None = None
    target_pips: float | None = None
    break_even_win_rate: float | None = None

    @property
    def actionable_candidate(self) -> bool:
        return self.side in ("BUY", "SELL")


def _ema(values: Sequence[float], period: int) -> list[float]:
    alpha = 2.0 / (period + 1)
    result = [float(values[0])]
    for value in values[1:]:
        result.append(alpha * float(value) + (1.0 - alpha) * result[-1])
    return result


def _atr(candles: Sequence[Candle], period: int) -> float:
    true_ranges = []
    for i, candle in enumerate(candles):
        previous_close = candles[i - 1].close if i else candle.close
        true_ranges.append(max(
            candle.high - candle.low,
            abs(candle.high - previous_close),
            abs(candle.low - previous_close),
        ))
    return sum(true_ranges[-period:]) / period


def _rsi(closes: Sequence[float], period: int) -> float:
    changes = [closes[i] - closes[i - 1] for i in range(len(closes) - period, len(closes))]
    gains = sum(max(0.0, change) for change in changes) / period
    losses = sum(max(0.0, -change) for change in changes) / period
    if losses == 0.0:
        return 100.0 if gains else 50.0
    return 100.0 - 100.0 / (1.0 + gains / losses)


def evaluate_trend_pullback(
    candles: Sequence[Candle],
    *,
    spread_pips: float,
    max_spread_pips: float = 1.2,
) -> PatternCandidate:
    """Evaluate one completed M1 candle; never reads a future bar.

    Long: EMA20 > EMA50 and EMA20 rising over five bars; current low tags the
    EMA20 and candle closes bullish above it; RSI14 is 45–65.
    Short: exact directional mirror with RSI14 35–55.

    Cost veto requires ATR14 >= 2 * observed spread and observed spread no
    greater than the caller's cap. Hypothetical exits are 1 ATR stop / 1.5 ATR
    target from the next-bar entry. If both barriers touch within one bar,
    evaluation must resolve stop-first. No same-candle entry or re-entry.
    """
    if len(candles) < EMA_SLOW + EMA_SLOPE_LOOKBACK:
        return PatternCandidate(None, "Need at least 55 completed M1 candles.")
    if spread_pips < 0.0 or max_spread_pips < 0.0:
        return PatternCandidate(None, "Spread inputs must be non-negative.")
    if spread_pips > max_spread_pips:
        return PatternCandidate(None, "Spread exceeds the configured research cap.")

    closes = [c.close for c in candles]
    fast = _ema(closes, EMA_FAST)
    slow = _ema(closes, EMA_SLOW)
    atr_pips = _atr(candles, ATR_PERIOD) / PIP
    if atr_pips <= 0.0 or atr_pips < MIN_ATR_TO_SPREAD * spread_pips:
        return PatternCandidate(None, "Volatility is too small relative to spread.", atr_pips)

    candle = candles[-1]
    fast_now = fast[-1]
    fast_then = fast[-1 - EMA_SLOPE_LOOKBACK]
    rsi = _rsi(closes, RSI_PERIOD)

    long_ok = (
        fast[-1] > slow[-1]
        and fast_now > fast_then
        and candle.low <= fast_now <= candle.close
        and candle.close > candle.open
        and 45.0 <= rsi <= 65.0
    )
    short_ok = (
        fast[-1] < slow[-1]
        and fast_now < fast_then
        and candle.high >= fast_now >= candle.close
        and candle.close < candle.open
        and 35.0 <= rsi <= 55.0
    )
    if long_ok == short_ok:  # neither, or conflicting data: abstain
        return PatternCandidate(None, "No unambiguous trend-pullback reclaim.", atr_pips)

    side = "BUY" if long_ok else "SELL"
    stop_pips = STOP_ATR * atr_pips
    target_pips = TARGET_ATR * atr_pips
    # Assumes spread_pips is the complete modeled round-turn cost. A deployment
    # would additionally need commission, slippage, gaps, and financing.
    be = 100.0 * (stop_pips + spread_pips) / (stop_pips + target_pips)
    return PatternCandidate(
        side=side,
        reason=(f"{side} trend-pullback reclaim candidate; EMA20/50 trend, "
                f"EMA20 slope, candle reclaim, RSI and cost gate agree. "
                "Paper/research only; entry is next bar open."),
        atr_pips=atr_pips,
        stop_pips=stop_pips,
        target_pips=target_pips,
        break_even_win_rate=be,
    )
