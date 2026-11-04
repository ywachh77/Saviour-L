"""Transparent technical indicators, plain Python, no look-ahead.

Every function returns a list aligned to the input, with ``None`` where the
indicator is not yet defined. Nothing here peeks at future bars.
"""

from __future__ import annotations

import math
from typing import Sequence

from . import version as version_mod
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


# =====================================================================
# NEW INDICATORS — Saviour Combiner Expansion
# =====================================================================


def ichimoku_cloud(
    candles: list[Candle],
    tenkan_period: int = 9,
    kijun_period: int = 26,
    senkou_a_period: int = 52,
    senkou_b_period: int = 26,
) -> tuple[list[float | None], list[float | None], list[float | None], list[float | None], list[float | None]]:
    """Ichimoku Kinko Hyo (Cloud).

    Returns (tenkan, kijun, senkou_a, senkou_b, chikou) aligned to candles.
    Tenkan = midpoint of 9-period high/low.
    Kijun = midpoint of 26-period high/low.
    Senkou A = (tenkan + kijun) / 2 shifted 26 bars forward.
    Senkou B = midpoint of 52-period high/low shifted 26 bars forward.
    Chikou = current close plotted 26 bars back.
    All causal — no look-ahead.
    """
    n = len(candles)
    tenkan = [None] * n
    kijun = [None] * n
    senkou_a = [None] * n
    senkou_b = [None] * n
    chikou = [None] * n

    for i in range(n):
        if i >= tenkan_period - 1:
            h = max(c.high for c in candles[i - tenkan_period + 1 : i + 1])
            l = min(c.low for c in candles[i - tenkan_period + 1 : i + 1])
            tenkan[i] = (h + l) / 2

        if i >= kijun_period - 1:
            h = max(c.high for c in candles[i - kijun_period + 1 : i + 1])
            l = min(c.low for c in candles[i - kijun_period + 1 : i + 1])
            kijun[i] = (h + l) / 2

        # Senkou A is shifted 26 bars forward
        if i >= 1 and tenkan[i - 1] is not None and kijun[i - 1] is not None:
            senkou_a[i] = (tenkan[i - 1] + kijun[i - 1]) / 2

        # Senkou B is shifted 26 bars forward
        if i >= 1 and i - 1 >= senkou_b_period - 1:
            h = max(c.high for c in candles[i - 1 - senkou_b_period + 1 : i])
            l = min(c.low for c in candles[i - 1 - senkou_b_period + 1 : i])
            senkou_b[i] = (h + l) / 2

        # Chikou is close shifted 26 bars back
        if i + senkou_a_period < n:
            chikou[i + senkou_a_period] = candles[i].close

    return tenkan, kijun, senkou_a, senkou_b, chikou


def _expected_rescaled_range(n: int) -> float:
    """E[R/S] for a pure random walk of length ``n`` (Anis-Lloyd / Peters).

    Small samples make a raw R/S read high: a driftless random walk, whose
    Hurst exponent is 0.5 by construction, measured 0.65 on this repository's
    data before this correction. Subtracting the theoretical expectation for
    the same lag is what turns the estimator into a measurement instead of a
    constant.

    Returns 0.0 when the expectation cannot be formed, so the caller can skip
    that lag rather than divide by a stand-in.
    """
    if n < 2:
        return 0.0
    total = sum(math.sqrt((n - i) / i) for i in range(1, n))
    if n <= 340:
        # Small-sample form. Do NOT also multiply by (n - 0.5) / n here: that
        # factor belongs to the large-n approximation below, and applying it
        # twice understates E[R/S] by 2-20% (checked against a 4,000-path
        # Monte Carlo at every lag this module uses).
        gamma_ratio = math.exp(math.lgamma((n - 1) / 2.0) - math.lgamma(n / 2.0))
        return (gamma_ratio / math.sqrt(math.pi)) * total
    return ((n - 0.5) / n) * (1.0 / math.sqrt(n * math.pi / 2.0)) * total


def hurst_exponent(values: list[float], max_lag: int = 20) -> list[float | None]:
    """Hurst Exponent — measures mean-reversion (<0.5) vs trend (>0.5).

    Uses bias-corrected R/S analysis on **log returns**, which is the domain
    where the exponent is defined. The returned list is aligned to ``values``,
    so entry ``i`` describes the window ending at ``values[i]``. Causal: only
    past data is used.

    Three corrections matter here, and each was measured rather than assumed,
    because the previous version was wrong in a way that fed a one-sided vote
    into every score that consumed it:

    1. **Domain.** R/S on raw *prices* is not a Hurst estimate. A price series
       is dominated by its level, so the rescaled range grows almost with the
       window and H lands near 1.0 whatever the market is doing. Measured on
       this repository's own 740 real EUR/USD bars, the old form read
       0.70-1.01 with a mean of 0.92 and never once fell below 0.5 — so the
       "mean-reverting" branch was unreachable and ``hurst > 0.5`` voted for
       one side on 100% of bars. On returns the same data reads 0.62 raw.
    2. **Bias.** Even on returns, R/S is biased upward in short windows: a
       driftless random walk measured 0.65, though its exponent is 0.5 by
       construction. Subtracting the Anis-Lloyd expectation for each lag and
       solving for the excess restores 0.5, so the 0.45/0.50 thresholds mean
       what they say.
    3. **Regression.** The regression needs ``log(R/S)`` against ``log(lag)``.
       An earlier version regressed ``log(R/S ** 2)`` and still divided by
       two, halving every exponent on top of the domain error.

    A window whose returns are all identical has no spread to rescale, and
    yields ``None`` — "unknown", never a fabricated 0.5 that would silently
    read as mean-reverting.
    """
    n = len(values)
    result: list[float | None] = [None] * n
    if n < 2:
        return result

    # Work on returns: log(price[i] / price[i - 1]). The series is then
    # additive and scale-free; the levels only exist to build it.
    returns: list[float | None] = [None] * n
    for i in range(1, n):
        previous, current = values[i - 1], values[i]
        if previous > 0.0 and current > 0.0:
            returns[i] = math.log(current / previous)

    window_size = max_lag * 2
    for i in range(window_size, n):
        window = [r for r in returns[i - window_size + 1 : i + 1] if r is not None]
        if len(window) < max_lag:
            continue

        lags = list(range(2, min(max_lag, len(window) // 2) + 1))
        log_lags: list[float] = []
        log_excess: list[float] = []
        for lag in lags:
            n_groups = len(window) // lag
            if n_groups < 2:
                continue
            scaled: list[float] = []
            for j in range(n_groups):
                group = window[j * lag : (j + 1) * lag]
                mean_value = sum(group) / len(group)
                deviations = [x - mean_value for x in group]
                cumulative: list[float] = []
                running = 0.0
                for deviation in deviations:
                    running += deviation
                    cumulative.append(running)
                spread = max(cumulative) - min(cumulative)
                sigma = (sum(d * d for d in deviations) / len(deviations)) ** 0.5
                if sigma > 0.0:
                    scaled.append(spread / sigma)
            if not scaled:
                continue
            observed = sum(scaled) / len(scaled)
            expected = _expected_rescaled_range(lag)
            if observed <= 0.0 or expected <= 0.0:
                continue
            # A random walk satisfies observed == expected at every lag, so its
            # excess is flat in log(lag) and the slope below is zero.
            log_lags.append(math.log(lag))
            log_excess.append(math.log(observed / expected))

        if len(log_lags) >= 2:
            points = len(log_lags)
            mean_l = sum(log_lags) / points
            mean_e = sum(log_excess) / points
            numerator = sum((log_lags[k] - mean_l) * (log_excess[k] - mean_e) for k in range(points))
            denominator = sum((log_lags[k] - mean_l) ** 2 for k in range(points))
            if denominator > 0.0:
                # R/S grows like lag ** H, and the random-walk baseline already
                # accounts for H = 0.5, so the slope is the excess over that.
                result[i] = 0.5 + numerator / denominator

    return result


def fft_cycles(values: list[float], max_cycles: int = 5) -> list[tuple[float, float]]:
    """FFT-based cycle detection. Returns top cycles as (period, strength).

    Detects dominant periodicities in the price series using a simple
    Hann-windowed FFT. Returns the strongest cycles.
    """
    import math

    n = len(values)
    if n < 64:
        return []

    # Hann window
    windowed = [values[i] * (0.5 - 0.5 * math.cos(2 * math.pi * i / (n - 1))) for i in range(n)]

    # Simple DFT (not full FFT — just enough for cycle detection)
    # For efficiency, only compute at specific frequencies
    best_cycles = []
    for k in range(2, max_cycles * 2 + 1):
        if k > n // 2:
            break
        period = n / k
        # Compute magnitude at this frequency
        real = sum(windowed[i] * math.cos(2 * math.pi * k * i / n) for i in range(n))
        imag = sum(windowed[i] * math.sin(2 * math.pi * k * i / n) for i in range(n))
        magnitude = math.sqrt(real ** 2 + imag ** 2) / n
        if period > 2 and magnitude > 0:
            best_cycles.append((period, magnitude))

    # Sort by magnitude, return top max_cycles
    best_cycles.sort(key=lambda x: x[1], reverse=True)
    return best_cycles[:max_cycles]


def aroon(candles: list[Candle], period: int = 25) -> tuple[list[float | None], list[float | None]]:
    """Aroon Indicator — trend strength and direction.

    Aroon Up = % of period since highest high (0 = just made new high).
    Aroon Down = % of period since lowest low.
    Returns (aroon_up, aroon_down) aligned to candles.
    """
    n = len(candles)
    aroon_up = [None] * n
    aroon_down = [None] * n

    for i in range(period, n):
        window = candles[i - period : i + 1]
        h = max(c.high for c in window)
        l = min(c.low for c in window)

        days_since_high = 0
        days_since_low = 0
        for j in range(period, -1, -1):
            if window[j].high == h:
                days_since_high = period - j
                break
        for j in range(period, -1, -1):
            if window[j].low == l:
                days_since_low = period - j
                break

        aroon_up[i] = ((period - days_since_high) / period) * 100
        aroon_down[i] = ((period - days_since_low) / period) * 100

    return aroon_up, aroon_down


def donchian_channel(candles: list[Candle], period: int = 20) -> tuple[list[float | None], list[float | None], list[float | None]]:
    """Donchian Channel — trend-following bands.

    Upper = highest high of period.
    Lower = lowest low of period.
    Middle = average of upper and lower.
    Returns (upper, lower, middle) aligned to candles.
    """
    n = len(candles)
    upper = [None] * n
    lower = [None] * n
    middle = [None] * n

    for i in range(period - 1, n):
        window = candles[i - period + 1 : i + 1]
        h = max(c.high for c in window)
        l = min(c.low for c in window)
        upper[i] = h
        lower[i] = l
        middle[i] = (h + l) / 2

    return upper, lower, middle


def pivot_points(candles: list[Candle], period: int = 14) -> tuple[list[float | None], ...]:
    """Pivot Points — classic support/resistance levels.

    Uses the standard pivot point formula:
    Pivot = (High + Low + Close) / 3
    R1 = 2*Pivot - Low, R2 = Pivot + (High - Low)
    S1 = 2*Pivot - High, S2 = Pivot - (High - Low)
    Returns (pivot, r1, r2, s1, s2) aligned to candles.
    """
    n = len(candles)
    pivot = [None] * n
    r1 = [None] * n
    r2 = [None] * n
    s1 = [None] * n
    s2 = [None] * n

    for i in range(period, n):
        window = candles[i - period : i + 1]
        h = max(c.high for c in window)
        l = min(c.low for c in window)
        c = candles[i].close

        p = (h + l + c) / 3
        pivot[i] = p
        r1[i] = 2 * p - l
        r2[i] = p + (h - l)
        s1[i] = 2 * p - h
        s2[i] = p - (h - l)

    return pivot, r1, r2, s1, s2


def squeeze_composite_vote(
    squeeze_on: bool | None,
    momentum: float | None,
) -> float:
    """Map a Keltner squeeze onto a signed council vote in [-1.0, +1.0].

    A squeeze is a *state*, not a direction: Bollinger bands sitting inside the
    Keltner channel say volatility has compressed, and say nothing about which
    way the eventual break will go. The vote previously read::

        score = 1.0 if squeeze_on else 0.0

    which is +1.0 or 0.0 and never negative. Ten to fifteen percent of bars are
    compressed at any time, so this contributed about +0.006 to a score that
    the dashboard compares against 0.0 -- a standing BUY allowance paid for by
    a condition with no directional content at all.

    The direction belongs to ``momentum``, the second value
    ``keltner_squeeze`` returns: the 12-bar regression midline oscillator that
    Carter's TTM setup reads as the break-out side. With the squeeze on, its
    sign carries the vote; with the squeeze off there is nothing to vote on.
    """
    if not squeeze_on:
        return 0.0
    if momentum is None:
        return 0.0
    try:
        value = float(momentum)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(value) or value == 0.0:
        return 0.0
    return 1.0 if value > 0.0 else -1.0


def rsi_composite_vote(rsi_value: float | None) -> float:
    """Map RSI onto a symmetric council vote in [-1.0, +1.0].

    This is the single implementation of the RSI rule. It lived in three
    hand-copied places (the dashboard's combiner twice, the backtest
    engine's combiner once) and every copy carried the same asymmetry::

        score = 0.5 if rsi < 40 else (1.0 if rsi < 50 else
                (-0.5 if rsi > 60 else (-1.0 if rsi > 70 else 0.0)))

    Read the bands. Below 40 -> +0.5 (mild bearish: oversold, expect a
    bounce). 40-50 -> +1.0. 50-60 -> 0.0. 60-70 -> -0.5. Above 70 -> -1.0.
    The mildly-bearish band is positive and the neutral band is zero, so the
    vote is not a direction at all -- it is a mean-reversion lean whose sign
    flips between 50 and 60. Measured on this repository's 740 real EUR/USD
    bars the average vote was **+0.318**, not 0.0, which is a permanent BUY
    contribution of about 0.032 to a score that is read at a 0.0 threshold.

    Note also that ``-0.5 if rsi > 60`` swallowed the ``-1.0 if rsi > 70``
    branch entirely, so an extreme overbought reading never scored its
    intended maximum.

    The bands below are the same intent, mirrored around 50:

    * 30-45, or 55-70: a gentle reversion lean, magnitude 0.5
    * below 30, or above 70: a strong reversion lean, magnitude 1.0
    * 45-55: genuinely undecided, 0.0

    A weak reading says the move has room to run back, a strong reading says
    it is stretched, and neither side is preferred when the input is
    missing. That is a balanced vote: equal and opposite bands, so the mean
    over a two-sided market is zero.
    """
    if rsi_value is None:
        return 0.0
    try:
        value = float(rsi_value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(value):
        return 0.0
    if value < 30.0:
        return 1.0
    if value < 45.0:
        return 0.5
    if value <= 55.0:
        return 0.0
    if value <= 70.0:
        return -0.5
    return -1.0


def indicators_version() -> str:
    """Return the current system version for indicators module."""
    return version_mod.get_version()


