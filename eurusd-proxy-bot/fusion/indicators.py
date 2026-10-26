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


def kaufman_er(values: Sequence[float], period: int = 10) -> list[float | None]:
    """Perry Kaufman's Efficiency Ratio (1995): directional displacement over path noise.

    Returns values in [0.0, 1.0]: 1.0 is a straight-line trend with zero
    retracement; 0.0 is pure whipsaw churn ending back where it started.
    Computed in O(N) via cumulative absolute step sums with zero look-ahead.
    """
    items = list(values)
    size = len(items)
    result: list[float | None] = [None] * size
    if size <= period or period <= 0:
        return result
    prefix = [0.0] * (size + 1)
    for index in range(1, size):
        prefix[index + 1] = prefix[index] + abs(items[index] - items[index - 1])
    for index in range(period, size):
        path = prefix[index + 1] - prefix[index - period + 1]
        net = abs(items[index] - items[index - period])
        result[index] = 0.0 if path <= 1e-15 else max(0.0, min(1.0, net / path))
    return result


def choppiness_index(candles: Sequence[Candle], period: int = 14) -> list[float | None]:
    """E.W. Dreiss Choppiness Index (1993) on [0.0, 100.0].

    Values above 61.8 indicate range-bound consolidation chop; values below
    38.2 indicate sustained directional expansion. Returns 50.0 when a flat
    envelope has zero high-low span rather than dividing by zero.
    """
    items = list(candles)
    size = len(items)
    result: list[float | None] = [None] * size
    if size < period or period <= 1:
        return result
    trs = true_ranges(items)
    prefix_tr = [0.0] * (size + 1)
    for index in range(size):
        prefix_tr[index + 1] = prefix_tr[index] + trs[index]
    denom = math.log10(float(period))
    for index in range(period - 1, size):
        start = index - period + 1
        window = items[start : index + 1]
        span = max(c.high for c in window) - min(c.low for c in window)
        sum_tr = prefix_tr[index + 1] - prefix_tr[start]
        if span <= 1e-15 or sum_tr <= 1e-15:
            result[index] = 50.0
        else:
            raw = 100.0 * math.log10(max(1.0, sum_tr / span)) / denom
            result[index] = max(0.0, min(100.0, raw))
    return result


def vwap_zscore(candles: Sequence[Candle], period: int = 30) -> list[float | None]:
    """Rolling causal VWAP Z-score: signed standard-deviation distance from VWAP."""
    items = list(candles)
    size = len(items)
    result: list[float | None] = [None] * size
    if size < period or period < 2:
        return result
    vwap_vals = vwap(items, period=period)
    typical = typical_prices(items)
    for index in range(period - 1, size):
        anchor = vwap_vals[index]
        if anchor is None:
            continue
        window = typical[index - period + 1 : index + 1]
        sigma = pstdev(window)
        if sigma <= 1e-12:
            result[index] = 0.0
        else:
            z = (items[index].close - anchor) / sigma
            result[index] = max(-8.0, min(8.0, z))
    return result


def keltner_squeeze(
    candles: Sequence[Candle],
    *,
    bb_period: int = 20,
    bb_mult: float = 2.0,
    kc_period: int = 20,
    kc_mult: float = 1.5,
    atr_period: int = 14,
    mom_period: int = 12,
) -> tuple[list[bool], list[float | None]]:
    """Causal John Carter TTM Bollinger-inside-Keltner Squeeze + normalized momentum.

    Returns ``(squeeze_on, linreg_mom_atr)`` aligned to ``candles``. ``squeeze_on[i]``
    is True when the 20-bar 2.0-sigma Bollinger envelope sits strictly inside the
    20-EMA +/- 1.5x ATR(14) Keltner Channel. ``linreg_mom_atr[i]`` is the 12-bar
    linear-regression midline oscillator scaled by ATR so thresholds hold identically
    across FX and spot crypto.
    """
    items = list(candles)
    size = len(items)
    squeeze_on: list[bool] = [False] * size
    momentum: list[float | None] = [None] * size
    if size < max(bb_period, kc_period, atr_period, mom_period):
        return squeeze_on, momentum

    closes = [c.close for c in items]
    kc_mid = ema(closes, kc_period)
    atr_vals = atr(items, atr_period)

    # Precompute constants for least-squares slope/endpoint over mom_period points (x = 0..m-1)
    m = mom_period
    mean_x = (m - 1) / 2.0
    denom_x = sum((x - mean_x) ** 2 for x in range(m)) or 1.0

    start_idx = max(bb_period, kc_period, atr_period, mom_period) - 1
    for index in range(start_idx, size):
        mid_kc = kc_mid[index]
        atr_now = atr_vals[index]
        if mid_kc is None or atr_now is None:
            continue
        bb_win = closes[index - bb_period + 1 : index + 1]
        bb_mid = mean(bb_win)
        bb_half = bb_mult * pstdev(bb_win)
        kc_half = kc_mult * atr_now
        if atr_now > 1e-15:
            squeeze_on[index] = (bb_mid - bb_half) > (mid_kc - kc_half) and (bb_mid + bb_half) < (mid_kc + kc_half)

        # Midline delta oscillator smoothed via OLS endpoint at x = m - 1
        deltas: list[float] = []
        for cursor in range(index - m + 1, index + 1):
            sub = items[cursor - m + 1 : cursor + 1]
            if len(sub) < m:
                deltas.append(0.0)
                continue
            donchian_mid = 0.5 * (max(c.high for c in sub) + min(c.low for c in sub))
            sma_mid = mean([c.close for c in sub])
            deltas.append(items[cursor].close - 0.5 * (donchian_mid + sma_mid))
        mean_y = mean(deltas)
        slope = sum((x - mean_x) * (deltas[x] - mean_y) for x in range(m)) / denom_x
        fitted_end = mean_y + slope * (m - 1 - mean_x)
        scale = atr_now if atr_now > 1e-12 else 1.0
        momentum[index] = max(-10.0, min(10.0, fitted_end / scale))

    return squeeze_on, momentum


def supertrend_window(
    candles: Sequence[Candle],
    *,
    period: int = 10,
    multiplier: float = 3.0,
    window: int = 52,
) -> tuple[list[float], list[float]]:
    """Causal fixed-lookback ATR Supertrend direction (+1.0 / -1.0) and flip flag (1.0 / 0.0).

    Evaluating over a bounded causal window ``candles[i - window + 1 : i + 1]``
    guarantees both zero future leakage and exact invariance between 240-bar and
    320-bar warmup histories.
    """
    items = list(candles)
    size = len(items)
    directions: list[float] = [0.0] * size
    flips: list[float] = [0.0] * size
    if size < period + 2:
        return directions, flips

    trs = true_ranges(items)
    local_atr: list[float | None] = [None] * size
    prefix_tr = [0.0] * (size + 1)
    for idx in range(size):
        prefix_tr[idx + 1] = prefix_tr[idx] + trs[idx]
    for idx in range(period - 1, size):
        local_atr[idx] = (prefix_tr[idx + 1] - prefix_tr[idx - period + 1]) / period

    for index in range(period + 1, size):
        start = max(period - 1, index - window + 1)
        prev_upper = 0.0
        prev_lower = 0.0
        prev_dir = 1.0 if items[start].close >= items[start].open else -1.0
        curr_dir = prev_dir
        prev_step_dir = prev_dir

        for k in range(start, index + 1):
            a_val = local_atr[k]
            if a_val is None:
                continue
            hl2 = 0.5 * (items[k].high + items[k].low)
            basic_upper = hl2 + multiplier * a_val
            basic_lower = hl2 - multiplier * a_val
            if k == start:
                final_upper = basic_upper
                final_lower = basic_lower
                curr_dir = 1.0 if items[k].close >= hl2 else -1.0
            else:
                prev_close = items[k - 1].close
                final_upper = (
                    basic_upper
                    if (basic_upper < prev_upper or prev_close > prev_upper)
                    else prev_upper
                )
                final_lower = (
                    basic_lower
                    if (basic_lower > prev_lower or prev_close < prev_lower)
                    else prev_lower
                )
                if prev_dir < 0 and items[k].close > final_upper:
                    curr_dir = 1.0
                elif prev_dir > 0 and items[k].close < final_lower:
                    curr_dir = -1.0
                else:
                    curr_dir = prev_dir
            prev_step_dir = prev_dir
            prev_dir = curr_dir
            prev_upper = final_upper
            prev_lower = final_lower

        directions[index] = curr_dir
        flips[index] = 1.0 if (curr_dir != prev_step_dir and index > start) else 0.0

    return directions, flips


def smc_structure(
    candles: Sequence[Candle],
    atr_values: Sequence[float | None],
    *,
    fvg_lookback: int = 5,
    ob_lookback: int = 24,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Causal ICT Fair Value Gap (FVG) & Institutional Order Block (OB) mitigation detector.

    Returns ``(fvg_bull, fvg_bear, ob_bull_tap, ob_bear_tap)`` as floats in ``{0.0, 1.0}``
    aligned to ``candles`` using strictly completed bars ``<= index``.
    """
    items = list(candles)
    size = len(items)
    fvg_bull = [0.0] * size
    fvg_bear = [0.0] * size
    ob_bull_tap = [0.0] * size
    ob_bear_tap = [0.0] * size
    if size < 15:
        return fvg_bull, fvg_bear, ob_bull_tap, ob_bear_tap

    for index in range(14, size):
        atr_now = atr_values[index]
        if atr_now is None or atr_now <= 1e-12:
            continue
        bar = items[index]

        # 1. ICT 3-Candle Displacement Fair Value Gap (fresh formation or unmitigated retest)
        start_fvg = max(2, index - fvg_lookback)
        for k in range(index, start_fvg - 1, -1):
            c0 = items[k - 2]
            c1 = items[k - 1]
            c2 = items[k]
            mid_body = abs(c1.close - c1.open)
            if mid_body < 0.70 * atr_now:
                continue
            # Bullish FVG: gap between candle k-2 high and candle k low
            if c1.close > c1.open and c2.low > c0.high + 0.08 * atr_now:
                gap_low, gap_high = c0.high, c2.low
                invalidated = any(items[m].close < gap_low for m in range(k + 1, index + 1))
                if not invalidated and (k == index or (bar.low <= gap_high and bar.close >= gap_low and bar.close > bar.open)):
                    fvg_bull[index] = 1.0
                    break
            # Bearish FVG: gap between candle k-2 low and candle k high
            if c1.close < c1.open and c2.high < c0.low - 0.08 * atr_now:
                gap_low, gap_high = c2.high, c0.low
                invalidated = any(items[m].close > gap_high for m in range(k + 1, index + 1))
                if not invalidated and (k == index or (bar.high >= gap_low and bar.close <= gap_high and bar.close < bar.open)):
                    fvg_bear[index] = 1.0
                    break

        # 2. SMC Institutional Order Block Mitigation Tap (last opposite candle before BOS impulse)
        start_ob = max(6, index - ob_lookback)
        for k in range(index - 2, start_ob - 1, -1):
            origin = items[k]
            imp1 = items[k + 1]
            prior_struct_high = max(c.high for c in items[max(0, k - 6) : k])
            prior_struct_low = min(c.low for c in items[max(0, k - 6) : k])

            # Bullish OB: bearish origin candle followed by displacement break above prior 6-bar high
            if (
                origin.close < origin.open
                and imp1.close > imp1.open
                and (imp1.close - origin.low) >= 1.15 * atr_now
                and imp1.close > prior_struct_high
            ):
                zone_low, zone_high = origin.low, max(origin.open, origin.close)
                broken = any(items[m].close < zone_low for m in range(k + 2, index))
                if not broken and bar.low <= zone_high and bar.close > zone_low and bar.close > bar.open:
                    ob_bull_tap[index] = 1.0
                    break

            # Bearish OB: bullish origin candle followed by displacement break below prior 6-bar low
            if (
                origin.close > origin.open
                and imp1.close < imp1.open
                and (origin.high - imp1.close) >= 1.15 * atr_now
                and imp1.close < prior_struct_low
            ):
                zone_low, zone_high = min(origin.open, origin.close), origin.high
                broken = any(items[m].close > zone_high for m in range(k + 2, index))
                if not broken and bar.high >= zone_low and bar.close < zone_high and bar.close < bar.open:
                    ob_bear_tap[index] = 1.0
                    break

    return fvg_bull, fvg_bear, ob_bull_tap, ob_bear_tap


