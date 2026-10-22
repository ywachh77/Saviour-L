"""Transparent technical indicators, plain Python, no look-ahead.

Every function returns a list aligned to the input, with ``None`` where the
indicator is not yet defined. Nothing here peeks at future bars.
"""

from __future__ import annotations

import math
from typing import Sequence

from .domain import Candle


def mean(values: Sequence[float]) -> float:
    items = list(values)
    if not items:
        return 0.0
    return sum(items) / len(items)


def pstdev(values: Sequence[float]) -> float:
    items = list(values)
    if len(items) < 2:
        return 0.0
    centre = mean(items)
    return math.sqrt(sum((value - centre) ** 2 for value in items) / len(items))


def median(values: Sequence[float]) -> float:
    items = sorted(values)
    if not items:
        return 0.0
    middle = len(items) // 2
    return items[middle] if len(items) % 2 else (items[middle - 1] + items[middle]) / 2.0


def ema(values: Sequence[float], period: int) -> list[float | None]:
    """Exponential moving average seeded with the simple mean of the first window."""
    result: list[float | None] = [None] * len(values)
    if len(values) < period or period <= 0:
        return result
    current = mean(values[:period])
    result[period - 1] = current
    alpha = 2.0 / (period + 1.0)
    for index in range(period, len(values)):
        current = alpha * values[index] + (1.0 - alpha) * current
        result[index] = current
    return result


def rsi(values: Sequence[float], period: int = 14) -> list[float | None]:
    """Wilder-style RSI."""
    result: list[float | None] = [None] * len(values)
    if len(values) <= period:
        return result
    changes = [values[index] - values[index - 1] for index in range(1, len(values))]
    average_gain = sum(max(change, 0.0) for change in changes[:period]) / period
    average_loss = sum(max(-change, 0.0) for change in changes[:period]) / period

    def score(gain: float, loss: float) -> float:
        if loss == 0:
            return 100.0 if gain > 0 else 50.0
        return 100.0 - (100.0 / (1.0 + gain / loss))

    result[period] = score(average_gain, average_loss)
    for index in range(period + 1, len(values)):
        change = changes[index - 1]
        average_gain = ((average_gain * (period - 1)) + max(change, 0.0)) / period
        average_loss = ((average_loss * (period - 1)) + max(-change, 0.0)) / period
        result[index] = score(average_gain, average_loss)
    return result


def true_ranges(candles: Sequence[Candle]) -> list[float]:
    ranges: list[float] = []
    for index, candle in enumerate(candles):
        previous_close = candles[index - 1].close if index else candle.close
        ranges.append(
            max(
                candle.high - candle.low,
                abs(candle.high - previous_close),
                abs(candle.low - previous_close),
            )
        )
    return ranges


def atr(candles: Sequence[Candle], period: int = 14) -> list[float | None]:
    ranges = true_ranges(candles)
    result: list[float | None] = [None] * len(candles)
    for index in range(period - 1, len(candles)):
        result[index] = mean(ranges[index - period + 1 : index + 1])
    return result


def adx(
    candles: Sequence[Candle], period: int = 14
) -> tuple[list[float | None], list[float | None], list[float | None]]:
    """Return (adx, +DI, -DI) with the standard Wilder smoothing of DX."""
    size = len(candles)
    ranges: list[float] = []
    positive_moves: list[float] = []
    negative_moves: list[float] = []
    for index, candle in enumerate(candles):
        previous = candles[index - 1] if index else candle
        upward = candle.high - previous.high
        downward = previous.low - candle.low
        positive_moves.append(upward if upward > downward and upward > 0 else 0.0)
        negative_moves.append(downward if downward > upward and downward > 0 else 0.0)
        ranges.append(
            max(
                candle.high - candle.low,
                abs(candle.high - previous.close),
                abs(candle.low - previous.close),
            )
        )

    adx_values: list[float | None] = [None] * size
    plus_di: list[float | None] = [None] * size
    minus_di: list[float | None] = [None] * size
    dx: list[float | None] = [None] * size
    for index in range(period - 1, size):
        start = index - period + 1
        window_range = sum(ranges[start : index + 1])
        if window_range <= 0:
            continue
        positive = 100.0 * sum(positive_moves[start : index + 1]) / window_range
        negative = 100.0 * sum(negative_moves[start : index + 1]) / window_range
        plus_di[index] = positive
        minus_di[index] = negative
        denominator = positive + negative
        dx[index] = 100.0 * abs(positive - negative) / denominator if denominator else 0.0
        dx_start = index - period + 1
        window = dx[dx_start : index + 1]
        if dx_start >= period - 1 and all(value is not None for value in window):
            adx_values[index] = mean(value for value in window if value is not None)
    return adx_values, plus_di, minus_di


def rolling_std(values: Sequence[float], end: int, period: int) -> float | None:
    start = end - period + 1
    if start < 0:
        return None
    window = values[start : end + 1]
    if len(window) != period:
        return None
    return pstdev(window)


def autocorrelation(values: Sequence[float]) -> float:
    """Lag-1 autocorrelation; 0.0 when the series has no variance."""
    items = list(values)
    if len(items) < 10:
        return 0.0
    left, right = items[:-1], items[1:]
    left_mean, right_mean = mean(left), mean(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    denominator = math.sqrt(
        sum((a - left_mean) ** 2 for a in left) * sum((b - right_mean) ** 2 for b in right)
    )
    if denominator <= 1e-15:
        return 0.0
    return max(-1.0, min(1.0, numerator / denominator))


def streak(closes: Sequence[float], index: int) -> int:
    """Signed count of consecutive up or down closes ending at ``index``."""
    count = 0
    cursor = index
    while cursor >= 1:
        up = closes[cursor] > closes[cursor - 1]
        down = closes[cursor] < closes[cursor - 1]
        if not up and not down:
            break
        if count == 0:
            count = 1 if up else -1
        elif (up and count > 0) or (down and count < 0):
            count += 1 if up else -1
        else:
            break
        cursor -= 1
    return count


def bollinger(closes: Sequence[float], period: int = 20, multiplier: float = 2.0) -> tuple[float, float, float]:
    """Return (lower, middle, upper) for the most recent ``period`` closes."""
    window = list(closes)[-period:]
    if len(window) < 2:
        centre = window[0] if window else 0.0
        return centre, centre, centre
    middle = mean(window)
    width = multiplier * pstdev(window)
    return middle - width, middle, middle + width


def typical_prices(candles: Sequence[Candle]) -> list[float]:
    return [(candle.high + candle.low + candle.close) / 3.0 for candle in candles]


def stochastic(
    candles: Sequence[Candle], period: int = 14, smooth: int = 3
) -> tuple[list[float | None], list[float | None]]:
    """Return (fast %K, %D) where %K is smoothed over ``smooth`` bars.

    The classic scalping oscillator: where the close sits inside the recent
    high-low envelope. 50.0 is returned when the envelope has collapsed to
    nothing rather than dividing by zero.
    """
    size = len(candles)
    raw: list[float | None] = [None] * size
    for index in range(period - 1, size):
        window = candles[index - period + 1 : index + 1]
        highest = max(candle.high for candle in window)
        lowest = min(candle.low for candle in window)
        span = highest - lowest
        raw[index] = 50.0 if span <= 1e-15 else 100.0 * (candles[index].close - lowest) / span

    k_values: list[float | None] = [None] * size
    d_values: list[float | None] = [None] * size
    for index in range(size):
        window = [value for value in raw[index - smooth + 1 : index + 1] if value is not None]
        if len(window) == smooth:
            k_values[index] = mean(window)
    for index in range(size):
        window = [value for value in k_values[index - smooth + 1 : index + 1] if value is not None]
        if len(window) == smooth:
            d_values[index] = mean(window)
    return k_values, d_values


def cci(candles: Sequence[Candle], period: int = 20) -> list[float | None]:
    """Commodity Channel Index: distance from the mean, in mean deviations."""
    typical = typical_prices(candles)
    size = len(candles)
    result: list[float | None] = [None] * size
    for index in range(period - 1, size):
        window = typical[index - period + 1 : index + 1]
        centre = mean(window)
        mean_deviation = mean([abs(value - centre) for value in window])
        result[index] = 0.0 if mean_deviation <= 1e-15 else (typical[index] - centre) / (0.015 * mean_deviation)
    return result


def vwap(candles: Sequence[Candle], period: int = 30) -> list[float | None]:
    """Rolling volume-weighted average of the typical price.

    Falls back to the plain mean when the window carries no volume, which is
    what a synthetic feed looks like. That fallback is honest: with zero
    volume every weight is equal, so the mean *is* the volume-weighted mean.
    """
    typical = typical_prices(candles)
    size = len(candles)
    result: list[float | None] = [None] * size
    for index in range(period - 1, size):
        start = index - period + 1
        weights = [max(float(candle.volume), 0.0) for candle in candles[start : index + 1]]
        total = sum(weights)
        window = typical[start : index + 1]
        if total <= 0.0:
            result[index] = mean(window)
        else:
            result[index] = sum(price * weight for price, weight in zip(window, weights)) / total
    return result
