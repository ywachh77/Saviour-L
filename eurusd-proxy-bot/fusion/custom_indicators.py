#!/usr/bin/env python3
"""
Custom Novel Indicators — NOT existing market indicators.

These are original mathematical constructs that capture market microstructure
patterns that RSI, MACD, Bollinger, etc. do not cover. Each indicator is
designed to extract signal from 1-minute candle data in ways that standard TA
cannot.

Categories:
  1. Information Theory — Shannon entropy, Kolmogorov complexity proxy
  2. Order Flow Proxies — buy/sell pressure from candle geometry
  3. Microstructure — price clustering, gap patterns, body acceleration
  4. Pattern Recognition — multi-candle sequence encoding
  5. Adaptive Systems — volatility phase, trend break detection
"""

from __future__ import annotations

import math
from typing import Sequence

from .domain import Candle


# ─── 1. Information Theory ─────────────────────────────────────────────

def price_entropy(candles: Sequence[Candle], window: int = 20) -> list[float | None]:
    """Shannon entropy of recent price changes.

    High entropy = chaotic, unpredictable (range/random walk).
    Low entropy = ordered, directional (trend).

    Measures the information content of the price series itself — how much
    "surprise" each bar contains. A trending market has low entropy (small
    consistent moves); a choppy market has high entropy (large random moves).
    """
    n = len(candles)
    result: list[float | None] = [None] * n
    for i in range(window, n):
        changes = []
        for j in range(i - window + 1, i + 1):
            if j > 0:
                ch = candles[j].close - candles[j - 1].close
                changes.append(ch)
        if not changes:
            continue
        # Bin the changes into directional bins
        changes = [c for c in changes if c != 0]
        if not changes:
            result[i] = 0.0
            continue
        # Use sign-based bins with magnitude normalization
        avg_abs = sum(abs(c) for c in changes) / len(changes)
        if avg_abs < 1e-15:
            result[i] = 0.0
            continue
        bins = [-2, -1, 0, 1, 2]  # -2x, -1x, 0x, +1x, +2x of avg_abs
        bin_counts = [0] * len(bins)
        for c in changes:
            ratio = c / avg_abs
            for k in range(len(bins) - 1):
                if bins[k] <= ratio < bins[k + 1]:
                    bin_counts[k] += 1
                    break
            else:
                bin_counts[-1] += 1
        total = sum(bin_counts)
        if total == 0:
            result[i] = 0.0
            continue
        # Shannon entropy
        entropy = 0.0
        for count in bin_counts:
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        # Normalize to [0, 1] (max entropy = log2(5) ≈ 2.32)
        result[i] = entropy / math.log2(len(bins))
    return result


def directional_info_ratio(candles: Sequence[Candle], window: int = 10) -> list[float | None]:
    """Ratio of directional information to total information.

    DIF = |sum(up_moves) - sum(down_moves)| / sum(|all_moves|)

    1.0 = perfectly directional (all moves in one direction)
    0.0 = perfectly balanced (equal up and down moves)

    Unlike ADX, this captures the MAGNITUDE of directional bias, not just
    the consistency of direction.
    """
    n = len(candles)
    result: list[float | None] = [None] * n
    for i in range(window, n):
        up_sum = 0.0
        down_sum = 0.0
        for j in range(i - window + 1, i + 1):
            if j > 0:
                ch = candles[j].close - candles[j - 1].close
                if ch > 0:
                    up_sum += ch
                else:
                    down_sum += abs(ch)
        total = up_sum + down_sum
        if total < 1e-15:
            result[i] = 0.0
        else:
            result[i] = abs(up_sum - down_sum) / total
    return result


# ─── 2. Order Flow Proxies ────────────────────────────────────────────

def body_pressure_imbalance(candles: Sequence[Candle], window: int = 10) -> list[float | None]:
    """Buy/sell pressure from candle body geometry.

    For each candle, the body (close - open) represents the net pressure
    during that period. The upper wick represents rejected buying (sell
    pressure at the top), and the lower wick represents rejected selling
    (buy pressure at the bottom).

    pressure = body_direction * body_size + lower_wick * 0.5 - upper_wick * 0.5

    This approximates order flow without having actual order flow data.
    """
    n = len(candles)
    result: list[float | None] = [None] * n
    for i in range(window, n):
        pressure_sum = 0.0
        total_range = 0.0
        for j in range(i - window + 1, i + 1):
            c = candles[j]
            body = c.close - c.open
            rng = c.high - c.low
            if rng < 1e-15:
                continue
            upper_wick = c.high - max(c.open, c.close)
            lower_wick = min(c.open, c.close) - c.low
            body_frac = body / rng
            pressure = body_frac * 2.0 + (lower_wick - upper_wick) / rng
            pressure_sum += pressure
            total_range += rng
        if total_range < 1e-15:
            result[i] = 0.0
        else:
            result[i] = max(-1.0, min(1.0, pressure_sum / window))
    return result


def wick_reversal_score(candles: Sequence[Candle], window: int = 5) -> list[float | None]:
    """Detects reversal pressure from wick dominance.

    A long upper wick means sellers rejected higher prices (bullish reversal
    potential). A long lower wick means buyers rejected lower prices (bearish
    reversal potential).

    score = (avg_lower_wick - avg_upper_wick) / avg_range

    Positive = bullish reversal pressure (lower wicks dominant)
    Negative = bearish reversal pressure (upper wicks dominant)
    """
    n = len(candles)
    result: list[float | None] = [None] * n
    for i in range(window, n):
        upper_sum = 0.0
        lower_sum = 0.0
        range_sum = 0.0
        for j in range(i - window + 1, i + 1):
            c = candles[j]
            rng = c.high - c.low
            if rng < 1e-15:
                continue
            upper_sum += c.high - max(c.open, c.close)
            lower_sum += min(c.open, c.close) - c.low
            range_sum += rng
        if range_sum < 1e-15:
            result[i] = 0.0
        else:
            result[i] = (lower_sum - upper_sum) / range_sum
    return result


def volume_directional_flow(candles: Sequence[Candle], window: int = 10) -> list[float | None]:
    """Net volume-weighted directional flow.

    sum(up_volume * body_frac) - sum(down_volume * body_frac)

    Volume on up candles = buying pressure. Volume on down candles = selling
    pressure. The body fraction weights by how decisively the candle moved.
    """
    n = len(candles)
    result: list[float | None] = [None] * n
    for i in range(window, n):
        up_vol = 0.0
        down_vol = 0.0
        for j in range(i - window + 1, i + 1):
            c = candles[j]
            rng = c.high - c.low
            body = c.close - c.open
            if rng < 1e-15:
                continue
            body_frac = body / rng
            if body > 0:
                up_vol += c.volume * abs(body_frac)
            else:
                down_vol += c.volume * abs(body_frac)
        total = up_vol + down_vol
        if total < 1e-15:
            result[i] = 0.0
        else:
            result[i] = (up_vol - down_vol) / total
    return result


# ─── 3. Microstructure ─────────────────────────────────────────────────

def price_clustering_density(candles: Sequence[Candle], window: int = 20, bins: int = 10) -> list[float | None]:
    """How concentrated recent prices are around specific levels.

    High density = price spending lots of time near certain levels (support
    or resistance). Low density = price spread out (transitioning, breaking
    out, or trending).

    Uses a histogram of closes over the window. If most closes fall in one
    bin, density is high. If they spread across many bins, density is low.
    """
    n = len(candles)
    result: list[float | None] = [None] * n
    for i in range(window, n):
        closes = [candles[j].close for j in range(i - window + 1, i + 1)]
        price_min = min(closes)
        price_max = max(closes)
        if price_max - price_min < 1e-15:
            result[i] = 1.0  # Perfect clustering (flat market)
            continue
        # Histogram
        hist = [0] * bins
        for price in closes:
            idx = int((price - price_min) / (price_max - price_min) * (bins - 1))
            hist[idx] += 1
        # Normalize to probability distribution
        total = sum(hist)
        if total == 0:
            result[i] = 0.0
            continue
        # Compute concentration: sum of squared probabilities
        # (Schunck-Fano-like measure)
        conc = sum((h / total) ** 2 for h in hist)
        # Normalize: 1/bins = uniform (0 concentration), 1 = all in one bin
        result[i] = (conc - 1.0 / bins) / (1.0 - 1.0 / bins)
    return result


def body_acceleration(candles: Sequence[Candle], window: int = 5) -> list[float | None]:
    """Second derivative of candle body size — momentum acceleration.

    Positive acceleration = momentum increasing (trend strengthening).
    Negative acceleration = momentum decreasing (trend weakening, reversal
    risk).

    This is fundamentally different from RSI or MACD momentum — it measures
    the RATE OF CHANGE of body size, not the direction of price.
    """
    n = len(candles)
    result: list[float | None] = [None] * n
    for i in range(2 * window + 1, n):
        bodies = []
        for j in range(i - 2 * window, i + 1):
            c = candles[j]
            bodies.append(c.close - c.open)
        # First derivative: differences of body sizes
        first_deriv = [bodies[k + 1] - bodies[k] for k in range(len(bodies) - 1)]
        # Second derivative: differences of first derivative
        second_deriv = [first_deriv[k + 1] - first_deriv[k] for k in range(len(first_deriv) - 1)]
        if not second_deriv:
            continue
        avg_body = sum(abs(b) for b in bodies) / len(bodies)
        if avg_body < 1e-15:
            result[i] = 0.0
        else:
            result[i] = sum(second_deriv) / len(second_deriv) / avg_body
    return result


def gap_pressure(candles: Sequence[Candle], window: int = 5) -> list[float | None]:
    """Detects gap pressure between consecutive candles.

    A gap between close[i-1] and open[i] indicates overnight or session-
    transition pressure. Large gaps in one direction followed by continuation
    suggest strong sentiment. Large gaps followed by reversal suggest exhaustion.

    gap_pressure = sum(gap_direction * gap_size) / sum(gap_size)

    Positive = bullish gap pressure (gaps favor upside)
    Negative = bearish gap pressure (gaps favor downside)
    """
    n = len(candles)
    result: list[float | None] = [None] * n
    for i in range(window, n):
        gap_pressure_sum = 0.0
        total_gap = 0.0
        for j in range(i - window + 1, i + 1):
            if j > 0:
                gap = candles[j].open - candles[j - 1].close
                total_gap += abs(gap)
                gap_pressure_sum += gap
        if total_gap < 1e-15:
            result[i] = 0.0
        else:
            result[i] = max(-1.0, min(1.0, gap_pressure_sum / total_gap))
    return result


# ─── 4. Pattern Recognition ────────────────────────────────────────────

def candle_sequence_encoding(candles: Sequence[Candle], window: int = 5) -> list[float | None]:
    """Encodes recent candle patterns as a directional sequence score.

    Each candle is classified as:
      0 = bearish engulfing (small body, close < open, large range)
      1 = bearish (close < open)
      2 = doji (small body relative to range)
      3 = bullish (close > open)
      4 = bullish engulfing (small body, close > open, large range)

    The sequence is then scored:
      - Consistent direction (all 3s or all 1s) = strong trend signal
      - Alternating pattern (3,1,3,1) = choppy, no signal
      - Engulfing patterns = reversal signal
    """
    n = len(candles)
    result: list[float | None] = [None] * n
    for i in range(window, n):
        classifications = []
        for j in range(i - window + 1, i + 1):
            c = candles[j]
            rng = c.high - c.low
            body = abs(c.close - c.open)
            body_frac = body / rng if rng > 1e-15 else 0.0
            if body_frac < 0.15:
                classifications.append(2)  # doji
            elif c.close > c.open:
                classifications.append(3)  # bullish
            else:
                classifications.append(1)  # bearish

        # Score: directional consistency
        up_count = sum(1 for c in classifications if c >= 3)
        down_count = sum(1 for c in classifications if c <= 1)
        doji_count = sum(1 for c in classifications if c == 2)
        total = len(classifications)
        if total == 0:
            result[i] = 0.0
        else:
            # Net directional bias, penalized by doji count
            bias = (up_count - down_count) / total
            doji_penalty = doji_count / total * 0.5
            result[i] = max(-1.0, min(1.0, bias * (1.0 - doji_penalty)))
    return result


def micro_trend_break(candles: Sequence[Candle], short: int = 3, long: int = 5) -> list[float | None]:
    """Detects when a short-term trend breaks.

    Compares a short-term linear regression slope against a longer-term one.
    When the short slope reverses the long slope, a micro-trend break occurs.

    score = short_slope * long_slope
    Negative = trend break (short and long disagree)
    Positive = trend confirmed (short and long agree)
    """
    n = len(candles)
    result: list[float | None] = [None] * n

    def linreg_slope(prices: list[float]) -> float:
        m = len(prices)
        if m < 2:
            return 0.0
        x_mean = (m - 1) / 2.0
        y_mean = sum(prices) / m
        num = sum((x - x_mean) * (y - y_mean) for x, y in enumerate(prices))
        den = sum((x - x_mean) ** 2 for x in range(m))
        if den < 1e-15:
            return 0.0
        return num / den

    for i in range(long, n):
        closes = [c.close for c in candles]
        short_window = closes[i - short + 1: i + 1]
        long_window = closes[i - long + 1: i + 1]
        short_slope = linreg_slope(short_window)
        long_slope = linreg_slope(long_window)
        avg_price = closes[i]
        if avg_price < 1e-15:
            result[i] = 0.0
        else:
            result[i] = (short_slope * long_slope) / (avg_price ** 2)
    return result


# ─── 5. Adaptive Systems ───────────────────────────────────────────────

def volatility_phase(candles: Sequence[Candle], window: int = 10) -> list[float | None]:
    """Classifies the current volatility phase: expanding, contracting, or stable.

    Phase = (current_vol - avg_vol) / (avg_vol + epsilon)

    > 0.2 = expanding (volatility increasing — potential breakout)
    -0.2 to 0.2 = stable (range-bound, low information)
    < -0.2 = contracting (volatility decreasing — potential squeeze)

    This is different from VIX or ATR — it measures the RATE OF CHANGE of
    volatility itself, predicting vol expansion/contraction.
    """
    n = len(candles)
    result: list[float | None] = [None] * n
    for i in range(2 * window, n):
        returns = []
        for j in range(i - 2 * window + 1, i + 1):
            if j > 0:
                returns.append(abs(candles[j].close - candles[j - 1].close))
        if len(returns) < window:
            continue
        current_vol = sum(returns[-window:]) / window
        prev_vol = sum(returns[:-window]) / window
        if prev_vol < 1e-15:
            result[i] = 0.0
        else:
            result[i] = (current_vol - prev_vol) / prev_vol
    return result


def efficiency_ratio_custom(candles: Sequence[Candle], window: int = 10) -> list[float | None]:
    """Custom efficiency ratio — net displacement vs total path.

    ER = |close[i] - close[i-window]| / sum(|close[j] - close[j-1]| for j in range)

    1.0 = perfectly efficient (straight line, strong trend)
    0.0 = perfectly inefficient (random walk, no trend)

    Unlike Kaufman's ER, this uses absolute changes (not squared) and is
    normalized differently. It captures trend efficiency more directly.
    """
    n = len(candles)
    result: list[float | None] = [None] * n
    for i in range(window, n):
        net = abs(candles[i].close - candles[i - window].close)
        path = sum(abs(candles[j].close - candles[j - 1].close) for j in range(i - window + 1, i + 1))
        if path < 1e-15:
            result[i] = 0.0
        else:
            result[i] = net / path
    return result


def session_momentum_shift(candles: Sequence[Candle]) -> list[float | None]:
    """Detects momentum shifts at session boundaries (London open, NY open).

    For each candle, computes the ratio of pre-session to post-session momentum
    if a session boundary occurs within the window.

    Value > 0 = momentum increasing after session transition
    Value < 0 = momentum decreasing after session transition
    """
    n = len(candles)
    result: list[float | None] = [None] * n

    # Session boundaries: London opens at 07:00 UTC, NY opens at 13:00 UTC
    london_open = 7
    ny_open = 13

    for i in range(10, n):
        c = candles[i]
        hour = c.timestamp.hour
        pre_momentum = 0.0
        post_momentum = 0.0

        # Check if we're near a session boundary
        is_london_boundary = hour == london_open
        is_ny_boundary = hour == ny_open

        if is_london_boundary or is_ny_boundary:
            # Pre-session: last 5 candles before boundary
            for j in range(i - 5, i):
                if j > 0:
                    pre_momentum += candles[j].close - candles[j - 1].close
            # Post-session: 5 candles after boundary
            for j in range(i + 1, min(i + 6, n)):
                if j > 0:
                    post_momentum += candles[j].close - candles[j - 1].close

            total = abs(pre_momentum) + abs(post_momentum)
            if total < 1e-15:
                result[i] = 0.0
            else:
                result[i] = (post_momentum - pre_momentum) / total
        else:
            result[i] = 0.0
    return result


# ─── 6. Market Microstructure (VPIN, Kyle's Lambda, Smart Money) ──────

def vpin_proxy(candles: Sequence[Candle], window: int = 20, buckets: int = 5) -> list[float | None]:
    """Volume-Synchronized Probability of Informed Trading (VPIN) proxy.

    Adapted from Easley, Lopez de Prado & O'Hara (2012). Since we don't
    have tick-level order flow, we approximate VPIN using candle geometry:
    each candle is a "bucket" and we measure the directional imbalance
    of body size vs range. High VPIN = informed trading active (potential
    for sharp moves); low VPIN = noise trading (range-bound).

    This is an ORIGINAL adaptation — we use OHLC geometry instead of
    actual buy/sell volume data.
    """
    n = len(candles)
    result: list[float | None] = [None] * n
    for i in range(buckets, n):
        vpin_sum = 0.0
        for j in range(i - buckets + 1, i + 1):
            c = candles[j]
            rng = c.high - c.low
            if rng < 1e-15:
                continue
            body = c.close - c.open
            body_frac = abs(body) / rng
            # Directional imbalance: how decisively did price move?
            # High body_frac = strong directional move (informed)
            # Low body_frac = noise (uninformed)
            vpin_sum += body_frac * body_frac  # Squared for penalty
        result[i] = min(1.0, vpin_sum / buckets)
    return result


def smart_money_index(candles: Sequence[Candle], early: int = 5, late: int = 5) -> list[float | None]:
    """Smart Money Index — late vs early session flow differential.

    Compares price movement in the first `early` bars of a window to
    the last `late` bars. Positive = smart money buying after initial
    noise selling (accumulation). Negative = smart money selling after
    initial noise buying (distribution).

    Inspired by order flow research but implemented purely from OHLC.
    """
    n = len(candles)
    result: list[float | None] = [None] * n
    total_window = early + late
    for i in range(total_window, n):
        early_move = 0.0
        for j in range(i - total_window + 1, i - total_window + 1 + early):
            if j > 0:
                early_move += candles[j].close - candles[j - 1].close
        late_move = 0.0
        for j in range(i - late + 1, i + 1):
            if j > 0:
                late_move += candles[j].close - candles[j - 1].close
        total_range = 0.0
        for j in range(i - total_window + 1, i + 1):
            total_range += abs(candles[j].close - candles[j - 1].close) if j > 0 else 0
        if total_range < 1e-15:
            result[i] = 0.0
        else:
            result[i] = (late_move - early_move) / total_range
    return result


def flow_momentum(candles: Sequence[Candle], short: int = 3, long: int = 10) -> list[float | None]:
    """Order flow momentum — short-window vs long-window flow differential.

    Captures whether directional pressure is building (short > long) or
    fading (short < long). Positive = accelerating flow; negative =
    mean-reversion of flow.

    Measured as the ratio of short-window net body direction to
    long-window net body direction.
    """
    n = len(candles)
    result: list[float | None] = [None] * n
    for i in range(long, n):
        short_net = 0.0
        for j in range(i - short + 1, i + 1):
            c = candles[j]
            short_net += (c.close - c.open)
        long_net = 0.0
        for j in range(i - long + 1, i + 1):
            c = candles[j]
            long_net += (c.close - c.open)
        long_abs = sum(abs(candles[j].close - candles[j].open) for j in range(i - long + 1, i + 1))
        if long_abs < 1e-15:
            result[i] = 0.0
        else:
            # Normalize by long-window total body magnitude
            result[i] = (short_net / short) / (long_net / long + 1e-15)
            # Clamp to reasonable range
            result[i] = max(-5.0, min(5.0, result[i]))
    return result


def liquidity_gap_detector(candles: Sequence[Candle], window: int = 10) -> list[float | None]:
    """Detects liquidity gaps — abrupt price jumps between candles.

    A gap between close[i-1] and open[i] that exceeds the average
    candle range signals a liquidity void. Large gaps in the trend
    direction = strong continuation. Large gaps against the trend =
    potential exhaustion.

    Returns the gap magnitude normalized by the average range.
    """
    n = len(candles)
    result: list[float | None] = [None] * n
    for i in range(window, n):
        # Compute average range
        avg_range = 0.0
        for j in range(i - window + 1, i + 1):
            avg_range += candles[j].high - candles[j].low
        avg_range /= window
        if avg_range < 1e-15:
            result[i] = 0.0
            continue
        # Check gap at this candle
        gap = candles[i].open - candles[i - 1].close
        gap_ratio = abs(gap) / avg_range
        # Also check the gap direction relative to recent trend
        recent_trend = 0.0
        for j in range(i - 3, i):
            if j > 0:
                recent_trend += candles[j].close - candles[j - 1].close
        if gap_ratio > 1.5 and recent_trend != 0:
            # Gap is large and there's a recent trend
            # Sign: positive gap in uptrend = continuation; positive gap in downtrend = exhaustion
            result[i] = gap_ratio * (1.0 if (gap > 0) == (recent_trend > 0) else -1.0)
        elif gap_ratio > 1.5:
            result[i] = gap_ratio * (1.0 if gap > 0 else -1.0)
        else:
            result[i] = 0.0
    return result


def tick_volume_imbalance(candles: Sequence[Candle], window: int = 10) -> list[float | None]:
    """Volume imbalance proxy using candle geometry.

    Since we don't have true tick volume, we approximate order flow
    imbalance using the ratio of body size to range across the window.
    A high positive value = buyers dominate (up candles with large bodies);
    a high negative value = sellers dominate.

    This is different from standard volume indicators because it uses
    the body-to-range ratio as a proxy for trade aggressiveness.
    """
    n = len(candles)
    result: list[float | None] = [None] * n
    for i in range(window, n):
        up_body = 0.0
        down_body = 0.0
        total_body = 0.0
        for j in range(i - window + 1, i + 1):
            c = candles[j]
            rng = c.high - c.low
            if rng < 1e-15:
                continue
            body = abs(c.close - c.open)
            body_frac = body / rng
            if c.close > c.open:
                up_body += body_frac * c.volume
            elif c.close < c.open:
                down_body += body_frac * c.volume
            total_body += body_frac * c.volume
        if total_body < 1e-15:
            result[i] = 0.0
        else:
            result[i] = (up_body - down_body) / total_body
    return result


def session_velocity_shift(candles: Sequence[Candle], window: int = 5) -> list[float | None]:
    """Detects velocity shifts at session boundaries.

    Computes the ratio of momentum in the last `window` bars to the
    momentum in the previous `window` bars. A significant shift indicates
    a change in market direction or tempo.

    Useful for detecting London/NY session transitions and their impact
    on momentum.
    """
    n = len(candles)
    result: list[float | None] = [None] * n
    total = 2 * window
    for i in range(total, n):
        recent_mom = 0.0
        for j in range(i - window + 1, i + 1):
            if j > 0:
                recent_mom += candles[j].close - candles[j - 1].close
        prev_mom = 0.0
        for j in range(i - total + 1, i - window + 1):
            if j > 0:
                prev_mom += candles[j].close - candles[j - 1].close
        prev_abs = 0.0
        for j in range(i - total + 1, i - window + 1):
            if j > 0:
                prev_abs += abs(candles[j].close - candles[j - 1].close)
        if prev_abs < 1e-15:
            result[i] = 0.0
        else:
            result[i] = recent_mom / (prev_abs + 1e-15)
    return result


def kyle_lambda_proxy(candles: Sequence[Candle], window: int = 20) -> list[float | None]:
    """Kyle's Lambda proxy — price impact per unit of flow.

    Original adaptation: measures how much price moves per unit of
    directional body pressure. High lambda = high price impact (strong
    market, informed trading); low lambda = low impact (weak market,
    noise trading).

    lambda = |net_body_direction| / total_range
    """
    n = len(candles)
    result: list[float | None] = [None] * n
    for i in range(window, n):
        net_flow = 0.0
        total_range = 0.0
        for j in range(i - window + 1, i + 1):
            c = candles[j]
            net_flow += (c.close - c.open)
            total_range += c.high - c.low
        if total_range < 1e-15:
            result[i] = 0.0
        else:
            result[i] = abs(net_flow) / total_range
    return result


# ─── Combined Export ───────────────────────────────────────────────────

CUSTOM_INDICATOR_NAMES = (
    "price_entropy",
    "directional_info_ratio",
    "body_pressure_imbalance",
    "wick_reversal_score",
    "volume_directional_flow",
    "price_clustering_density",
    "body_acceleration",
    "gap_pressure",
    "candle_sequence_encoding",
    "micro_trend_break",
    "volatility_phase",
    "efficiency_ratio_custom",
    "session_momentum_shift",
    "vpin_proxy",
    "smart_money_index",
    "flow_momentum",
    "liquidity_gap_detector",
    "tick_volume_imbalance",
    "session_velocity_shift",
    "kyle_lambda_proxy",
)


def compute_custom_indicators(candles: Sequence[Candle]) -> list[dict[str, float] | None]:
    """Compute all custom indicators for each candle.

    Returns a list of dicts (one per candle), where each dict contains
    all custom indicator values. None entries mean the indicator is not
    yet computable (warmup period).
    """
    n = len(candles)
    results: list[dict[str, float] | None] = [None] * n

    # Compute each indicator series
    series = {
        "price_entropy": price_entropy(candles, 20),
        "directional_info_ratio": directional_info_ratio(candles, 10),
        "body_pressure_imbalance": body_pressure_imbalance(candles, 10),
        "wick_reversal_score": wick_reversal_score(candles, 5),
        "volume_directional_flow": volume_directional_flow(candles, 10),
        "price_clustering_density": price_clustering_density(candles, 20, 10),
        "body_acceleration": body_acceleration(candles, 5),
        "gap_pressure": gap_pressure(candles, 5),
        "candle_sequence_encoding": candle_sequence_encoding(candles, 5),
        "micro_trend_break": micro_trend_break(candles, 3, 5),
        "volatility_phase": volatility_phase(candles, 10),
        "efficiency_ratio_custom": efficiency_ratio_custom(candles, 10),
        "session_momentum_shift": session_momentum_shift(candles),
        "vpin_proxy": vpin_proxy(candles, 20, 5),
        "smart_money_index": smart_money_index(candles, 5, 5),
        "flow_momentum": flow_momentum(candles, 3, 10),
        "liquidity_gap_detector": liquidity_gap_detector(candles, 10),
        "tick_volume_imbalance": tick_volume_imbalance(candles, 10),
        "session_velocity_shift": session_velocity_shift(candles, 5),
        "kyle_lambda_proxy": kyle_lambda_proxy(candles, 20),
    }

    for i in range(n):
        row: dict[str, float] = {}
        all_valid = True
        for name, series_data in series.items():
            val = series_data[i] if i < len(series_data) else None
            if val is None or not math.isfinite(val):
                all_valid = False
                break
            row[name] = float(val)
        if all_valid:
            results[i] = row
        else:
            results[i] = None
    return results
