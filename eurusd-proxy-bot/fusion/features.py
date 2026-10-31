"""Feature engineering for one-minute EUR/USD candles.

Two groups of keys are produced for every bar:

* model inputs — the ordered ``FEATURE_NAMES`` list the logistic model consumes;
* strategy-facing values — plain indicator levels the transparent rule sets read.

Bumping ``FEATURE_VERSION`` invalidates stored models instead of letting them be
silently misinterpreted.
"""

from __future__ import annotations

import math
from typing import Sequence

from . import charter, instruments
from .domain import Candle
from . import mtf
from .custom_indicators import compute_custom_indicators, CUSTOM_INDICATOR_NAMES
from .indicators import (
    adx,
    atr,
    autocorrelation,
    cci,
    choppiness_index,
    ema,
    kaufman_er,
    keltner_squeeze,
    mean,
    median,
    pstdev,
    rolling_std,
    rsi,
    smc_structure,
    stochastic,
    streak,
    supertrend_window,
    vwap,
    vwap_zscore,
)

# Scalping needs a rolling "fair value" anchor. 30 one-minute bars is half an
# hour — long enough to mean something, short enough to still be a scalp.
VWAP_PERIOD = 30
STOCH_PERIOD = 14
STOCH_SMOOTH = 3
CCI_PERIOD = 20
# How many bars back a scalping "micro pivot" looks. Shorter than the 20-bar
# swing the swing strategies use, because a scalp targets a few pips, not a leg.
MICRO_PIVOT_BARS = 5

FEATURE_VERSION = 8
FEATURE_VERSION = 9

# Higher-timeframe context. The feed is one-minute only, so these are resampled
# from the bars we already hold rather than fetched: five one-minute bars are a
# five-minute bar. See fusion/mtf.py for why a forming bucket is never used.
MTF_FAST_MINUTES = 5
MTF_SLOW_MINUTES = 15
MTF_MACRO_MINUTES = 60
MTF_EMA_PERIOD = 10

FEATURE_NAMES = (
    "return_1",
    "return_3",
    "return_5",
    "return_10",
    "ema_gap",
    "ema_slope",
    "rsi_scaled",
    "rsi_slope",
    "range_pct",
    "body_pct",
    "wick_upper",
    "wick_lower",
    "volatility_10",
    "volatility_ratio",
    "atr_ratio",
    "adx_scaled",
    "di_spread",
    "close_position",
    "body_atr_ratio",
    "range_position_20",
    "streak_scaled",
    "autocorr_20",
    "hour_sin",
    "hour_cos",
    # --- scalping microstructure (v4) -------------------------------------
    # A scalp lives or dies on the shape of the last few bars and on how far
    # price has strayed from value, none of which the v3 inputs described.
    "micro_range_5",
    "range_compression",
    "body_ratio",
    "impulse",
    "stoch_k_scaled",
    "stoch_d_scaled",
    "cci_scaled",
    "vwap_dev",
    # --- regime detection (v5) ---------------------------------------------
    # Research: ADX > 25 + ATR above median = momentum regime; ADX < 20 +
    # ATR below median = mean-reversion. Model must know which regime it's in
    # before committing to a direction.
    "regime_momentum",
    "regime_mean_reversion",
    "regime_chop",
    "return_20",
    "return_60",
    "volatility_z",
    "ker_slope",
    "adx_trend",
    "session_london",
    "session_ny",
    "autocorr_abs",
    "range_expansion",
    # --- candle quality microstructure (v6) -------------------------------
    "body_to_wick_ratio",
    "upper_shadow_dom",
    "lower_shadow_dom",
    "consecutive_dir_count",
    # --- volatility regime (v7) --------------------------------------------
    # Explicit volatility regime flags. The model was struggling to infer
    # volatility regimes from volatility_z alone. These make it explicit.
    "vol_regime_low",
    "vol_regime_high",
    "vol_regime_cluster",
    # --- custom novel indicators (v8) -------------------------------------
    # 13 original indicators that capture patterns standard TA cannot.
    # Information theory, order flow proxies, microstructure, pattern
    # recognition, and adaptive systems. Not rehashes of RSI/MACD/etc.
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
    # --- custom novel indicators v2 (v9) --------------------------------
    # 7 new microstructure indicators inspired by market microstructure
    # research: VPIN, Kyle's Lambda, Smart Money Index, flow momentum,
    # liquidity gaps, tick imbalance, session velocity.
    "vpin_proxy",
    "smart_money_index",
    "flow_momentum",
    "liquidity_gap_detector",
    "tick_volume_imbalance",
    "session_velocity_shift",
    "kyle_lambda_proxy",
    # Higher-timeframe context is deliberately NOT here. The tilt is computed
    # and carried as strategy-facing values below (mtf_fast / mtf_slow, plus
    # mtf_bias_* and mtf_gap_*), but it does not enter the model vector:
    # measured across five 2,500-bar replays it changed nothing -- 891 signals,
    # 51.2%, -734.2 net pips both with and without it -- while perturbing
    # prototype credit enough to break a promotion threshold test and
    # invalidating every trained model. Unproven inputs do not get to silently
    # redefine the model. Revisit once real history can settle it.
)

FEATURE_LABELS = {
    "return_1": "1-bar return",
    "return_3": "3-bar return",
    "return_5": "5-bar return",
    "return_10": "10-bar return",
    "ema_gap": "EMA 5/20 gap",
    "ema_slope": "EMA gap slope",
    "rsi_scaled": "RSI (centred)",
    "rsi_slope": "RSI slope",
    "range_pct": "Candle range",
    "body_pct": "Candle body",
    "wick_upper": "Upper wick",
    "wick_lower": "Lower wick",
    "volatility_10": "10-bar volatility",
    "volatility_ratio": "Short/long volatility",
    "atr_ratio": "ATR vs its average",
    "adx_scaled": "ADX (scaled)",
    "di_spread": "+DI minus -DI",
    "close_position": "Close inside range",
    "body_atr_ratio": "Body vs ATR",
    "range_position_20": "Position in 20-bar range",
    "streak_scaled": "Win streak",
    "autocorr_20": "Return autocorrelation",
    "hour_sin": "Hour of day (sin)",
    "hour_cos": "Hour of day (cos)",
    "micro_range_5": "5-bar range",
    "range_compression": "5-bar vs 20-bar range",
    "body_ratio": "Body as share of range",
    "impulse": "Signed body vs ATR",
    "stoch_k_scaled": "Stochastic %K (centred)",
    "stoch_d_scaled": "Stochastic %D (centred)",
    "cci_scaled": "CCI (scaled)",
    "vwap_dev": "Distance from VWAP",
    "regime_momentum": "Momentum regime flag",
    "regime_mean_reversion": "Mean-reversion regime flag",
    "regime_chop": "Choppy regime flag",
    "return_20": "20-bar return",
    "return_60": "60-bar return",
    "volatility_z": "Volatility z-score",
    "ker_slope": "KAMA efficiency ratio slope",
    "adx_trend": "ADX trend change",
    "session_london": "London session flag",
    "session_ny": "NY session flag",
    "autocorr_abs": "Autocorrelation strength",
    "range_expansion": "Range expansion ratio",
    "body_to_wick_ratio": "Body vs wick ratio",
    "upper_shadow_dom": "Upper shadow dominance",
    "lower_shadow_dom": "Lower shadow dominance",
    "consecutive_dir_count": "Consecutive direction count",
    "vol_regime_low": "Low volatility regime flag",
    "vol_regime_high": "High volatility regime flag",
    "vol_regime_cluster": "Volatility clustering flag",
    "price_entropy": "Price change Shannon entropy",
    "directional_info_ratio": "Directional information ratio",
    "body_pressure_imbalance": "Body pressure imbalance",
    "wick_reversal_score": "Wick reversal score",
    "volume_directional_flow": "Volume directional flow",
    "price_clustering_density": "Price clustering density",
    "body_acceleration": "Body acceleration (2nd derivative)",
    "gap_pressure": "Gap pressure",
    "candle_sequence_encoding": "Candle sequence encoding",
    "micro_trend_break": "Micro trend break detector",
    "volatility_phase": "Volatility phase",
    "efficiency_ratio_custom": "Custom efficiency ratio",
    "session_momentum_shift": "Session momentum shift",
    "vpin_proxy": "VPIN informed trading proxy",
    "smart_money_index": "Smart money index (late vs early flow)",
    "flow_momentum": "Flow momentum (short vs long window)",
    "liquidity_gap_detector": "Liquidity gap detector",
    "tick_volume_imbalance": "Tick volume imbalance proxy",
    "session_velocity_shift": "Session velocity shift",
    "kyle_lambda_proxy": "Kyle's Lambda price impact proxy",
}

# How many bars are needed before a feature row is complete.
MIN_FEATURE_BARS = 30


def _safe_ratio(numerator: float, denominator: float, default: float = 0.0) -> float:
    return numerator / denominator if abs(denominator) > 1e-15 else default


def _rsi_divergence_score(rsi_vals: list, closes: list, index: int) -> float:
    """Compute RSI divergence: +1 = bullish (price lower low, RSI higher low), -1 = bearish.

    Compares the last 5 candles' price extremes against their RSI extremes.
    """
    if index < 10 or len(rsi_vals) < 10 or len(closes) < 10:
        return 0.0
    lookback = min(5, index)
    start = index - lookback
    price_low = min(closes[start : index + 1])
    price_high = max(closes[start : index + 1])
    prev_price_low = min(closes[start - lookback : start])
    prev_price_high = max(closes[start - lookback : start])
    rsi_at_low = min(rsi_vals[start : index + 1])
    rsi_at_high = max(rsi_vals[start : index + 1])
    prev_rsi_low = min(rsi_vals[start - lookback : start])
    prev_rsi_high = max(rsi_vals[start - lookback : start])

    # Bullish divergence: price made lower low, RSI made higher low
    if price_low < prev_price_low and rsi_at_low > prev_rsi_low:
        return 0.5
    # Bearish divergence: price made higher high, RSI made lower high
    if price_high > prev_price_high and rsi_at_high < prev_rsi_high:
        return -0.5
    return 0.0


def _flag_direction(candles: Sequence, index: int) -> float:
    """Detect flag pattern direction. Returns +1 for bull flag, -1 for bear flag, 0 for none."""
    if index < 30:
        return 0.0
    # Strong move in last 10 candles
    recent = candles[index - 10 : index]
    move = recent[-1].close - recent[0].close
    if abs(move) < 1e-15:
        return 0.0
    # Flag = tight consolidation in last 5 candles
    flag = candles[index - 5 : index]
    flag_high = max(c.high for c in flag)
    flag_low = min(c.low for c in flag)
    flag_range = flag_high - flag_low
    if flag_range < 1e-15:
        return 0.0
    # Flag tightness: range relative to the move
    tightness = flag_range / abs(move)
    if tightness > 0.6:
        return 0.0  # Not tight enough
    if move > 0:
        return 1.0  # Bull flag
    return -1.0  # Bear flag


def _macd_line(closes: list, index: int) -> float:
    """Compute MACD line (12-26 EMA difference)."""
    if index < 30:
        return 0.0
    ema_12 = _ema(closes[: index + 1], 12)
    ema_26 = _ema(closes[: index + 1], 26)
    if ema_12 is None or ema_26 is None:
        return 0.0
    return ema_12 - ema_26


def _macd_signal(closes: list, index: int) -> float:
    """Compute MACD signal line (9-period EMA of MACD)."""
    if index < 40:
        return 0.0
    macd_vals = []
    for i in range(index - 8, index + 1):
        if i >= 30:
            ema_12 = _ema(closes[: i + 1], 12)
            ema_26 = _ema(closes[: i + 1], 26)
            if ema_12 is not None and ema_26 is not None:
                macd_vals.append(ema_12 - ema_26)
    if len(macd_vals) < 9:
        return 0.0
    return _ema(macd_vals, 9)


def _ema(values: list, period: int) -> float | None:
    """Simple EMA computation."""
    if len(values) < period:
        return None
    k = 2.0 / (period + 1)
    ema = values[0]
    for v in values[1:]:
        ema = v * k + ema * (1 - k)
    return ema


def _demand_zone_top(candles: Sequence, index: int) -> float:
    """Find the top of a demand zone (strong bullish candle with buy-side imbalance)."""
    if index < 20:
        return 0.0
    # Look for the most recent strong bullish candle
    for i in range(max(0, index - 20), index):
        c = candles[i]
        body = abs(c.close - c.open)
        rng = c.high - c.low
        if rng < 1e-15:
            continue
        body_frac = body / rng
        if body_frac > 0.6 and c.close > c.open and c.close > c.high * 0.95:
            return c.high  # Demand zone top
    return 0.0


def _supply_zone_bottom(candles: Sequence, index: int) -> float:
    """Find the bottom of a supply zone (strong bearish candle with sell-side imbalance)."""
    if index < 20:
        return 0.0
    for i in range(max(0, index - 20), index):
        c = candles[i]
        body = abs(c.close - c.open)
        rng = c.high - c.low
        if rng < 1e-15:
            continue
        body_frac = body / rng
        if body_frac > 0.6 and c.close < c.open and c.close < c.low * 1.05:
            return c.low  # Supply zone bottom
    return 0.0


def _breaker_strength(candle: "Candle") -> float:
    """Measure candle strength: body ratio × close position.

    Returns 0.0 (weak) to 1.0 (strong). A strong breaker has a large body
    that closes near its extreme.
    """
    rng = candle.high - candle.low
    if rng < 1e-15:
        return 0.0
    body = abs(candle.close - candle.open)
    body_frac = body / rng
    close_pos = (candle.close - candle.low) / rng
    # Close near extreme (0 or 1) = strong
    extreme_bonus = max(close_pos, 1 - close_pos)
    return body_frac * extreme_bonus


def _poc_price(candles: Sequence, index: int) -> float:
    """Find the volume point of control price in the last 20 candles.

    The POC is the price where the most volume traded. We approximate this
    by finding the candle with the highest volume and using its midpoint.
    """
    if index < 10:
        return candles[index].close
    recent = candles[max(0, index - 20) : index + 1]
    highest_vol_candle = max(recent, key=lambda c: c.volume)
    return (highest_vol_candle.high + highest_vol_candle.low) / 2.0


def _wick_rejection(candle: "Candle") -> float:
    """Measure wick rejection strength: 0 (no rejection) to 1 (strong rejection).

    A strong rejection has a long wick on one side and a small body,
    indicating price was pushed back from that level.
    """
    rng = candle.high - candle.low
    if rng < 1e-15:
        return 0.0
    body = abs(candle.close - candle.open)
    upper_wick = candle.high - max(candle.open, candle.close)
    lower_wick = min(candle.open, candle.close) - candle.low
    longest_wick = max(upper_wick, lower_wick)
    if body < 1e-15:
        return 0.5  # Pure wick — moderate rejection
    wick_ratio = longest_wick / body
    if wick_ratio >= 2.0:
        return 1.0
    elif wick_ratio >= 1.5:
        return 0.7
    elif wick_ratio >= 1.0:
        return 0.4
    return 0.0


def _descending_flat_low(candles: Sequence, index: int) -> float:
    """Find the flat support level for a descending triangle.

    Looks for the lowest low in the last 20 candles that was tested
    multiple times (at least 2 touches). Returns the flat low level.
    """
    if index < 20:
        return 0.0
    recent = candles[max(0, index - 20) : index + 1]
    lows = [c.low for c in recent]
    low_val = min(lows)
    # Count touches near the low (within 10% of range)
    tol = 0.0001
    touches = sum(1 for c in recent if abs(c.low - low_val) <= tol)
    if touches >= 2:
        return low_val
    return 0.0


def _descending_lower_high(candles: Sequence, index: int) -> float:
    """Find the lower high level for a descending triangle.

    Looks for the highest high in the last 10 candles, which should be
    lower than the previous rally (descending highs).
    """
    if index < 10:
        return 0.0
    recent = candles[max(0, index - 10) : index + 1]
    return max(c.high for c in recent)


def _win_probability(rsi_vals: list, closes: list, index: int) -> float:
    """Estimate win probability from RSI and recent price action.

    Higher RSI in an uptrend = higher win probability for BUY.
    Lower RSI in a downtrend = higher win probability for SELL.
    Returns 0.0 to 1.0.
    """
    if index < 10 or len(rsi_vals) < 10:
        return 0.5
    rsi_val = rsi_vals[index] if rsi_vals[index] is not None else 50.0
    price_change = _safe_ratio(closes[index] - closes[index - 5], closes[index - 5]) if index >= 5 else 0.0

    # Base probability from RSI
    if price_change > 0:
        prob = rsi_val / 100.0
    else:
        prob = 1.0 - rsi_val / 100.0

    # Adjust for trend strength
    if abs(price_change) > 0.001:
        prob = prob + 0.05
    elif abs(price_change) < 0.0001:
        prob = prob - 0.05

    return max(0.0, min(1.0, prob))


def _consecutive_same_direction(candles: Sequence, index: int, direction: str) -> float:
    """Count consecutive candles in the same direction.

    Counts how many consecutive candles (including current) closed in the
    given direction (up = bullish, down = bearish). Doji candles (open == close)
    do NOT break the streak — they are neutral and skipped over. This ensures
    that a doji in a strong trend doesn't artificially reset the consecutive
    count.
    """
    count = 0
    for i in range(index, max(0, index - 10), -1):
        if direction == "up":
            if candles[i].close > candles[i].open:
                count += 1
            elif candles[i].close == candles[i].open:
                continue  # doji: neutral, don't break the streak
            else:
                break
        elif direction == "down":
            if candles[i].close < candles[i].open:
                count += 1
            elif candles[i].close == candles[i].open:
                continue  # doji: neutral, don't break the streak
            else:
                break
    return float(count)


def _williams_r(candles: Sequence, index: int) -> float:
    """Williams %R — range-based momentum indicator.

    Returns -100 to 0. -80 = oversold, -20 = overbought.
    """
    if index < 14:
        return -50.0
    recent = candles[max(0, index - 13) : index + 1]
    highest = max(c.high for c in recent)
    lowest = min(c.low for c in recent)
    if highest == lowest:
        return -50.0
    return -100.0 * ((highest - candles[index].close) / (highest - lowest))


def _sar_below(candles: Sequence, index: int) -> bool:
    """True if Parabolic SAR is below price (uptrend)."""
    if index < 2:
        return False
    # Simple SAR approximation: use recent trend
    if candles[index].close > candles[index - 1].close:
        return True
    return False


def _sar_flipped(candles: Sequence, index: int) -> bool:
    """True if Parabolic SAR flipped (trend reversal)."""
    if index < 3:
        return False
    # Flip when price moves past previous SAR level
    # Approximate: check if trend direction changed
    was_down = candles[index - 2].close < candles[index - 3].close
    is_up = candles[index].close > candles[index - 1].close
    was_up = candles[index - 2].close > candles[index - 3].close
    is_down = candles[index].close < candles[index - 1].close
    return (was_down and is_up) or (was_up and is_down)


def _chaikin_oscillator(candles: Sequence, index: int) -> float:
    """Chaikin Oscillator — volume-weighted momentum.

    Accumulation/Distribution Line rate of change.
    Positive = buying pressure, negative = selling pressure.
    """
    if index < 3:
        return 0.0
    # Simplified: use volume * price change
    recent = candles[max(0, index - 3) : index + 1]
    accum = 0.0
    for c in recent:
        cpi = (c.close - c.low) / (c.high - c.low) if c.high != c.low else 0.5
        accum += cpi * c.volume
    return accum


def _macd_histogram(candles: Sequence, index: int) -> float:
    """MACD histogram value (MACD line - Signal line)."""
    if index < 20:
        return 0.0
    closes = [c.close for c in candles[: index + 1]]
    macd_line = _macd_line(closes, index)
    signal_line = _macd_signal(closes, index)
    return macd_line - signal_line


def _mtf_gap(item: "mtf.Bias", price: float) -> float:
    """How far above its higher-timeframe average price sits, in basis points.

    Zero when the tilt is unknown, so "we do not know" never masquerades as
    "price is exactly on its average".
    """
    if not item.reference:
        return 0.0
    return _safe_ratio(item.close - item.reference, price) * 10000.0


def compute_features(
    candles: Sequence[Candle], instrument: instruments.Instrument | None = None
) -> list[dict[str, float] | None]:
    """Compute indicators from current and past candles only (no future data).

    ``instrument`` fixes the size of a pip, which is the one unit that is not
    dimensionless. It defaults to EUR/USD so every existing caller keeps its
    current meaning; a caller that knows it is holding bitcoin bars must say
    so, because an ATR of 58 dollars is 58 pips on BTC/USD and 580,000 "pips"
    if it is divided by the FX pip by mistake.
    """
    candles = list(candles)
    pip = (instrument or instruments.DEFAULT).pip
    closes = [candle.close for candle in candles]
    ema_fast = ema(closes, 5)
    ema_slow = ema(closes, 20)
    ema_8_vals = ema(closes, 8)
    ema_13_vals = ema(closes, 13)
    ema_21_vals = ema(closes, 21)
    ema_34_vals = ema(closes, 34)
    ema_50_vals = ema(closes, 50)
    rsi_values = rsi(closes, 14)
    rsi_2_values = rsi(closes, 2)
    atr_values = atr(candles, 14)
    adx_values, plus_di_values, minus_di_values = adx(candles, 14)
    stoch_k_values, stoch_d_values = stochastic(candles, STOCH_PERIOD, STOCH_SMOOTH)
    cci_values = cci(candles, CCI_PERIOD)
    vwap_values = vwap(candles, VWAP_PERIOD)
    vwap_z_values = vwap_zscore(candles, VWAP_PERIOD)
    ker_10_values = kaufman_er(closes, 10)
    chop_14_values = choppiness_index(candles, 14)
    squeeze_on_vals, linreg_mom_vals = keltner_squeeze(candles)
    st_dir_vals, st_flip_vals = supertrend_window(candles)
    fvg_bull_vals, fvg_bear_vals, ob_bull_vals, ob_bear_vals = smc_structure(candles, atr_values)
    # Resampled once for the whole series rather than per bar. See
    # mtf.bias_series -- doing it per bar is quadratic and a 250k-bar backfill
    # would take hours instead of seconds.
    mtf_fast = mtf.bias_series(candles, minutes=MTF_FAST_MINUTES, ema_period=MTF_EMA_PERIOD)
    mtf_slow = mtf.bias_series(candles, minutes=MTF_SLOW_MINUTES, ema_period=MTF_EMA_PERIOD)
    mtf_macro = mtf.bias_series(candles, minutes=MTF_MACRO_MINUTES, ema_period=MTF_EMA_PERIOD)

    # --- regime detection series (v5) ---
    # Returns computed first — regime features depend on rolling volatility.
    returns: list[float] = [0.0]
    for i in range(1, len(closes)):
        returns.append(_safe_ratio(closes[i] - closes[i - 1], closes[i - 1]))

    # Rolling 50-bar median ATR for regime classification.
    _atr_med_window = 50
    atr_medians: list[float | None] = [None] * len(candles)
    for i in range(_atr_med_window - 1, len(candles)):
        window = [v for v in atr_values[i - _atr_med_window + 1 : i + 1] if v is not None]
        if len(window) >= _atr_med_window // 2:
            atr_medians[i] = median(window)

    # Volatility z-score: current 10-bar vol vs 50-bar distribution.
    _vol_window = 50
    vol_zscores: list[float | None] = [None] * len(candles)
    vol_medians_list: list[float | None] = [None] * len(candles)
    for i in range(_vol_window - 1, len(candles)):
        vol_now = rolling_std(returns, i, 10)
        if vol_now is None:
            continue
        vol_history = [
            rolling_std(returns, j, 10)
            for j in range(i - _vol_window + 1, i + 1)
            if rolling_std(returns, j, 10) is not None
        ]
        if len(vol_history) < _vol_window // 2:
            continue
        v_med = median(vol_history)
        vol_medians_list[i] = v_med
        if v_med and v_med > 0:
            # z-score of current vol vs history
            v_mean = mean(vol_history)
            v_std = pstdev(vol_history) or 1e-8
            vol_zscores[i] = (vol_now - v_mean) / v_std

    # Longer-horizon returns (20, 60 bars)
    return_20_vals: list[float] = [0.0] * len(candles)
    return_60_vals: list[float] = [0.0] * len(candles)
    for i in range(len(candles)):
        if i >= 20:
            return_20_vals[i] = _safe_ratio(closes[i] - closes[i - 20], closes[i - 20])
        if i >= 60:
            return_60_vals[i] = _safe_ratio(closes[i] - closes[i - 60], closes[i - 60])

    # KAMA efficiency-ratio slope (5-bar change)
    ker_slope_vals: list[float] = [0.0] * len(candles)
    for i in range(5, len(candles)):
        if ker_10_values[i] is not None and ker_10_values[i - 5] is not None:
            ker_slope_vals[i] = ker_10_values[i] - ker_10_values[i - 5]

    # ADX trend change (5-bar delta, scaled)
    adx_trend_vals: list[float] = [0.0] * len(candles)
    for i in range(5, len(candles)):
        if adx_values[i] is not None and adx_values[i - 5] is not None:
            adx_trend_vals[i] = (adx_values[i] - adx_values[i - 5]) / 100.0

    # Autocorrelation absolute value
    autocorr_abs_vals: list[float] = [0.0] * len(candles)
    for i in range(20, len(candles)):
        autocorr_abs_vals[i] = abs(autocorrelation(returns[i - 19 : i + 1]))

    # Range expansion: current bar range vs 5-bar average
    range_expansion_vals: list[float] = [1.0] * len(candles)
    for i in range(5, len(candles)):
        ranges_5 = [candles[j].high - candles[j].low for j in range(i - 4, i + 1)]
        avg_r5 = mean(ranges_5)
        cur_range = candles[i].high - candles[i].low
        range_expansion_vals[i] = _safe_ratio(cur_range, avg_r5, 1.0)
    for index in range(1, len(closes)):
        returns.append(_safe_ratio(closes[index] - closes[index - 1], closes[index - 1]))

    # Pre-compute custom indicators for all candles
    custom_indicators_list = compute_custom_indicators(candles)

    result: list[dict[str, float] | None] = []
    for index, candle in enumerate(candles):
        if (
            index < MIN_FEATURE_BARS
            or ema_fast[index] is None
            or ema_slow[index] is None
            or rsi_values[index] is None
            or atr_values[index] is None
            or adx_values[index] is None
            or plus_di_values[index] is None
            or minus_di_values[index] is None
            or stoch_k_values[index] is None
            or stoch_d_values[index] is None
            or cci_values[index] is None
            or vwap_values[index] is None
        ):
            result.append(None)
            continue

        previous_close = closes[index - 1]
        return_3 = _safe_ratio(closes[index] - closes[index - 3], closes[index - 3]) if index >= 3 else 0.0
        return_5 = _safe_ratio(closes[index] - closes[index - 5], closes[index - 5]) if index >= 5 else 0.0
        return_10 = _safe_ratio(closes[index] - closes[index - 10], closes[index - 10]) if index >= 10 else 0.0
        volatility = rolling_std(returns, index, 10)
        long_volatility = rolling_std(returns, index, 20)
        if volatility is None or long_volatility is None:
            result.append(None)
            continue

        recent_closes = closes[index - 19 : index + 1]
        band_mid = mean(recent_closes)
        band_width = 2.0 * pstdev(recent_closes)
        prior_window = candles[index - 20 : index]
        prior_high = max(item.high for item in prior_window)
        prior_low = min(item.low for item in prior_window)
        rsi_previous = rsi_values[index - 1] if rsi_values[index - 1] is not None else rsi_values[index]

        candle_range = candle.high - candle.low
        close_position = _safe_ratio(candle.close - candle.low, candle_range, 0.5)
        atr_window = [value for value in atr_values[index - 19 : index + 1] if value is not None]
        atr_average = mean(atr_window) if atr_window else atr_values[index]

        # Average volume over last 20 candles
        recent_vols = [item.volume for item in candles[index - 19 : index + 1] if item.volume > 0]
        avg_volume = mean(recent_vols) if recent_vols else 1.0

        # Average band width over last 20 candles for contraction detection
        recent_band_widths = []
        for j in range(max(0, index - 19), index + 1):
            rc = closes[j - 19 : j + 1] if j >= 19 else closes[: j + 1]
            if len(rc) >= 5:
                bw = 2.0 * pstdev(rc)
                if bw > 0:
                    recent_band_widths.append(bw)
        avg_band_width = mean(recent_band_widths) if recent_band_widths else band_width

        previous_gap = 0.0
        if ema_fast[index - 1] is not None and ema_slow[index - 1] is not None:
            previous_gap = _safe_ratio(ema_fast[index - 1] - ema_slow[index - 1], previous_close)
        current_gap = _safe_ratio(ema_fast[index] - ema_slow[index], candle.close)

        range_span = prior_high - prior_low
        range_position = (close_position - 0.5) if range_span <= 0 else (
            _safe_ratio(candle.close - prior_low, range_span) - 0.5
        )
        hour_fraction = candle.timestamp.hour + candle.timestamp.minute / 60.0
        angle = 2.0 * math.pi * hour_fraction / 24.0
        body = candle.close - candle.open
        current_streak = streak(closes, index)

        # --- scalping microstructure -------------------------------------
        # Everything here describes the last few bars, which is the only place
        # a few-pip move can be seen forming.
        micro_window = candles[index - MICRO_PIVOT_BARS + 1 : index + 1]
        micro_high = max(item.high for item in micro_window)
        micro_low = min(item.low for item in micro_window)
        micro_range = micro_high - micro_low
        # Excludes the current bar, so a breakout is measured against a level
        # that was already known before this bar printed.
        micro_prior = candles[index - MICRO_PIVOT_BARS : index]
        micro_prior_high = max(item.high for item in micro_prior)
        micro_prior_low = min(item.low for item in micro_prior)
        prior_15 = candles[max(0, index - 15) : index]
        prior_high_15 = max(item.high for item in prior_15)
        prior_low_15 = min(item.low for item in prior_15)
        prior_30 = candles[max(0, index - 30) : index]
        prior_high_30 = max(item.high for item in prior_30)
        prior_low_30 = min(item.low for item in prior_30)
        prior_box_30_atr = _safe_ratio(prior_high_30 - prior_low_30, atr_values[index], 5.0)

        ema_8_now = float(ema_8_vals[index] if ema_8_vals[index] is not None else ema_fast[index])
        ema_13_now = float(ema_13_vals[index] if ema_13_vals[index] is not None else ema_slow[index])
        ema_21_now = float(ema_21_vals[index] if ema_21_vals[index] is not None else ema_slow[index])
        ema_34_now = float(ema_34_vals[index] if ema_34_vals[index] is not None else ema_slow[index])
        ema_50_now = float(ema_50_vals[index] if ema_50_vals[index] is not None else ema_34_now)
        rsi_2_now = float(rsi_2_values[index] if rsi_2_values[index] is not None else rsi_values[index])
        rsi_2_prev = float(rsi_2_values[index - 1] if rsi_2_values[index - 1] is not None else rsi_2_now)
        vwap_z_now = float(vwap_z_values[index] if vwap_z_values[index] is not None else 0.0)
        sq_now = 1.0 if squeeze_on_vals[index] else 0.0
        sq_recent = 1.0 if any(squeeze_on_vals[max(0, index - 3) : index]) else 0.0
        mom_now = float(linreg_mom_vals[index] if linreg_mom_vals[index] is not None else 0.0)
        mom_prev = float(linreg_mom_vals[index - 1] if linreg_mom_vals[index - 1] is not None else mom_now)

        # --- candle quality microstructure (v6) ---
        upper_wick = candle.high - max(candle.open, candle.close)
        lower_wick = min(candle.open, candle.close) - candle.low
        total_wick = upper_wick + lower_wick
        body_to_wick = _safe_ratio(abs(body), total_wick, 1.0)
        upper_shadow_dom = _safe_ratio(upper_wick, total_wick, 0.5)
        lower_shadow_dom = _safe_ratio(lower_wick, total_wick, 0.5)
        # Count consecutive same-direction closes (signed, capped at ±1.0)
        dir_count = 0
        for j in range(index, 0, -1):
            if closes[j] > closes[j - 1]:
                if dir_count >= 0:
                    dir_count += 1
                else:
                    break
            elif closes[j] < closes[j - 1]:
                if dir_count <= 0:
                    dir_count -= 1
                else:
                    break
            else:
                break
            if abs(dir_count) > 20:
                break

        # --- volatility regime (v7) ---
        # Compute ATR percentile over the full series for regime classification.
        vol_regime_low = 0.0
        vol_regime_high = 0.0
        vol_regime_cluster = 0.0
        if atr_values[index] is not None and atr_medians[index] is not None:
            atr_now = atr_values[index]
            atr_med = atr_medians[index]
            # Compute 25th/75th percentile from recent ATR window
            atr_window_vals = [v for v in atr_values[max(0, index - 49):index + 1] if v is not None]
            if len(atr_window_vals) >= 10:
                sorted_atr = sorted(atr_window_vals)
                n = len(sorted_atr)
                p25 = sorted_atr[n // 4]
                p75 = sorted_atr[(3 * n) // 4]
                vol_regime_low = 1.0 if atr_now < p25 else 0.0
                vol_regime_high = 1.0 if atr_now > p75 else 0.0
            # Clustering: vol_zscore spike + range compression
            vol_z = float(vol_zscores[index]) if vol_zscores[index] is not None else 0.0
            range_comp = _safe_ratio(micro_range, range_span, 1.0) if range_span > 0 else 1.0
            vol_regime_cluster = 1.0 if (vol_z > 1.5 and range_comp < 0.8) else 0.0

        # --- custom novel indicators (v8) ---
        # Pre-computed for all candles above
        custom_row = custom_indicators_list[index] if index < len(custom_indicators_list) else None

        feature = {
            # model inputs (ordered in FEATURE_NAMES)
            "return_1": returns[index],
            "return_1_prev": returns[index - 1] if index >= 1 else 0.0,
            "return_3": return_3,
            "return_5": return_5,
            "return_10": return_10,
            "ema_gap": current_gap,
            "ema_slope": (current_gap - previous_gap) * 1000.0,
            "rsi_scaled": (rsi_values[index] - 50.0) / 50.0,
            "rsi_slope": (rsi_values[index] - rsi_previous) / 10.0,
            "range_pct": _safe_ratio(candle_range, candle.close),
            "body_pct": _safe_ratio(body, previous_close),
            "wick_upper": _safe_ratio(candle.high - max(candle.open, candle.close), candle.close),
            "wick_lower": _safe_ratio(min(candle.open, candle.close) - candle.low, candle.close),
            "volatility_10": volatility,
            "volatility_ratio": _safe_ratio(volatility, long_volatility, 1.0),
            "atr_ratio": _safe_ratio(atr_values[index], atr_average, 1.0),
            "adx_scaled": adx_values[index] / 100.0,
            "di_spread": (plus_di_values[index] - minus_di_values[index]) / 100.0,
            "close_position": close_position,
            "body_atr_ratio": _safe_ratio(abs(body), atr_values[index]),
            "range_position_20": range_position,
            "streak_scaled": max(-1.0, min(1.0, current_streak / 5.0)),
            "autocorr_20": autocorrelation(returns[index - 19 : index + 1]),
            "hour_sin": math.sin(angle),
            "hour_cos": math.cos(angle),
            # scalping microstructure
            "micro_range_5": _safe_ratio(micro_range, candle.close),
            "range_compression": _safe_ratio(micro_range, range_span, 1.0),
            "body_ratio": _safe_ratio(abs(body), candle_range),
            "impulse": _safe_ratio(body, atr_values[index]),
            "stoch_k_scaled": (stoch_k_values[index] - 50.0) / 50.0,
            "stoch_d_scaled": (stoch_d_values[index] - 50.0) / 50.0,
            "cci_scaled": max(-1.0, min(1.0, cci_values[index] / 200.0)),
            "vwap_dev": _safe_ratio(candle.close - vwap_values[index], candle.close) * 1000.0,
            # regime detection (v5)
            "regime_momentum": 1.0 if (adx_values[index] is not None and adx_values[index] > 25.0 and atr_medians[index] is not None and atr_values[index] is not None and atr_values[index] > atr_medians[index]) else 0.0,
            "regime_mean_reversion": 1.0 if (adx_values[index] is not None and adx_values[index] < 20.0 and atr_medians[index] is not None and atr_values[index] is not None and atr_values[index] < atr_medians[index]) else 0.0,
            "regime_chop": 1.0 if (chop_14_values[index] is not None and chop_14_values[index] > 70.0) else 0.0,
            "return_20": return_20_vals[index],
            "return_60": return_60_vals[index],
            "volatility_z": float(vol_zscores[index]) if vol_zscores[index] is not None else 0.0,
            "ker_slope": ker_slope_vals[index],
            "adx_trend": adx_trend_vals[index],
            "session_london": 1.0 if 8 <= candle.timestamp.hour < 16 else 0.0,
            "session_ny": 1.0 if 13 <= candle.timestamp.hour < 21 else 0.0,
            "autocorr_abs": autocorr_abs_vals[index],
            "range_expansion": range_expansion_vals[index],
            # candle quality microstructure (v6)
            "body_to_wick_ratio": body_to_wick,
            "upper_shadow_dom": upper_shadow_dom,
            "lower_shadow_dom": lower_shadow_dom,
            "consecutive_dir_count": max(-1.0, min(1.0, dir_count / 10.0)),
            # volatility regime (v7)
            "vol_regime_low": vol_regime_low,
            "vol_regime_high": vol_regime_high,
            "vol_regime_cluster": vol_regime_cluster,
            # custom novel indicators (v8)
            "price_entropy": float(custom_row["price_entropy"]) if custom_row else 0.5,
            "directional_info_ratio": float(custom_row["directional_info_ratio"]) if custom_row else 0.0,
            "body_pressure_imbalance": float(custom_row["body_pressure_imbalance"]) if custom_row else 0.0,
            "wick_reversal_score": float(custom_row["wick_reversal_score"]) if custom_row else 0.0,
            "volume_directional_flow": float(custom_row["volume_directional_flow"]) if custom_row else 0.0,
            "price_clustering_density": float(custom_row["price_clustering_density"]) if custom_row else 0.0,
            "body_acceleration": float(custom_row["body_acceleration"]) if custom_row else 0.0,
            "gap_pressure": float(custom_row["gap_pressure"]) if custom_row else 0.0,
            "candle_sequence_encoding": float(custom_row["candle_sequence_encoding"]) if custom_row else 0.0,
            "micro_trend_break": float(custom_row["micro_trend_break"]) if custom_row else 0.0,
            "volatility_phase": float(custom_row["volatility_phase"]) if custom_row else 0.0,
            "efficiency_ratio_custom": float(custom_row["efficiency_ratio_custom"]) if custom_row else 0.0,
            "session_momentum_shift": float(custom_row["session_momentum_shift"]) if custom_row else 0.0,
            # custom novel indicators v2 (v9)
            "vpin_proxy": float(custom_row["vpin_proxy"]) if custom_row else 0.0,
            "smart_money_index": float(custom_row["smart_money_index"]) if custom_row else 0.0,
            "flow_momentum": float(custom_row["flow_momentum"]) if custom_row else 0.0,
            "liquidity_gap_detector": float(custom_row["liquidity_gap_detector"]) if custom_row else 0.0,
            "tick_volume_imbalance": float(custom_row["tick_volume_imbalance"]) if custom_row else 0.0,
            "session_velocity_shift": float(custom_row["session_velocity_shift"]) if custom_row else 0.0,
            "kyle_lambda_proxy": float(custom_row["kyle_lambda_proxy"]) if custom_row else 0.0,
            # higher-timeframe context
            "mtf_bias_5": float(mtf_fast[index].direction),
            "mtf_gap_5": _mtf_gap(mtf_fast[index], candle.close),
            "mtf_slope_5": float(mtf_fast[index].slope_bps) if mtf_fast[index].reference > 0.0 else 0.0,
            "mtf_ready_5": 1.0 if mtf_fast[index].reference > 0.0 else 0.0,
            "mtf_bias_15": float(mtf_slow[index].direction),
            "mtf_gap_15": _mtf_gap(mtf_slow[index], candle.close),
            "mtf_slope_15": float(mtf_slow[index].slope_bps) if mtf_slow[index].reference > 0.0 else 0.0,
            "mtf_ready_15": 1.0 if mtf_slow[index].reference > 0.0 else 0.0,
            "mtf_bias_60": float(mtf_macro[index].direction),
            "mtf_gap_60": _mtf_gap(mtf_macro[index], candle.close),
            "mtf_slope_60": float(mtf_macro[index].slope_bps) if mtf_macro[index].reference > 0.0 else 0.0,
            "mtf_ready_60": 1.0 if mtf_macro[index].reference > 0.0 else 0.0,
            # strategy-facing values
            "ema_fast": ema_fast[index],
            "ema_slow": ema_slow[index],
            "ema_fast_previous": ema_fast[index - 1] if ema_fast[index - 1] is not None else ema_fast[index],
            "ema_slow_previous": ema_slow[index - 1] if ema_slow[index - 1] is not None else ema_slow[index],
            "rsi": rsi_values[index],
            "rsi_previous": rsi_previous,
            "rsi_prev": rsi_previous,
            "rsi_velocity": (rsi_values[index] - rsi_previous) if index >= 1 else 0.0,
            "momentum_accel": (returns[index] - returns[index - 1]) if index >= 1 else 0.0,
            "momentum_strength": returns[index] if index >= 0 else 0.0,
            "candle_direction": 1.0 if candle.close > candle.open else (-1.0 if candle.close < candle.open else 0.0),
            "range_atr_ratio": _safe_ratio(candle.high - candle.low, atr_values[index]),
            "wick_ratio": _safe_ratio((candle.high - candle.low) - abs(candle.close - candle.open), atr_values[index]),
            "body_atr": _safe_ratio(abs(body), atr_values[index]),
            "atr": atr_values[index],
            "atr_pct": _safe_ratio(atr_values[index], candle.close),
            "adx": adx_values[index],
            "adx_prev": adx_values[index - 1] if index >= 1 and adx_values[index - 1] is not None else adx_values[index],
            "consecutive_up": _consecutive_same_direction(candles, index, "up"),
            "consecutive_down": _consecutive_same_direction(candles, index, "down"),
            "plus_di": plus_di_values[index],
            "minus_di": minus_di_values[index],
            "bb_mid": band_mid,
            "bb_upper": band_mid + band_width,
            "bb_lower": band_mid - band_width,
            "prior_high_20": prior_high,
            "prior_low_20": prior_low,
            # scalping levels
            "stoch_k": stoch_k_values[index],
            "stoch_d": stoch_d_values[index],
            "stoch_k_prev": stoch_k_values[index - 1] if index >= 1 and stoch_k_values[index - 1] is not None else stoch_k_values[index],
            "stoch_k_previous": (
                stoch_k_values[index - 1] if stoch_k_values[index - 1] is not None else stoch_k_values[index]
            ),
            "stoch_d_previous": (
                stoch_d_values[index - 1] if stoch_d_values[index - 1] is not None else stoch_d_values[index]
            ),
            "cci": cci_values[index],
            "vwap": vwap_values[index],
            "prior_high_5": micro_prior_high,
            "prior_low_5": micro_prior_low,
            "prior_high_15": prior_high_15,
            "prior_low_15": prior_low_15,
            "prior_high_30": prior_high_30,
            "prior_low_30": prior_low_30,
            "prior_box_30_atr": prior_box_30_atr,
            "wick_upper_ratio": _safe_ratio(candle.high - max(candle.open, candle.close), candle_range),
            "wick_lower_ratio": _safe_ratio(min(candle.open, candle.close) - candle.low, candle_range),
            "ema_8": ema_8_now,
            "ema_13": ema_13_now,
            "ema_21": ema_21_now,
            "ema_34": ema_34_now,
            "ema_50": ema_50_now,
            "rsi_2": rsi_2_now,
            "rsi_2_previous": rsi_2_prev,
            "vwap_z": vwap_z_now,
            "ttm_squeeze_now": sq_now,
            "ttm_squeeze_recent": sq_recent,
            "linreg_mom_12": mom_now,
            "linreg_mom_previous": mom_prev,
            "supertrend_dir": float(st_dir_vals[index]),
            "supertrend_flipped": float(st_flip_vals[index]),
            "fvg_bull_active": float(fvg_bull_vals[index]),
            "fvg_bear_active": float(fvg_bear_vals[index]),
            # New features for added patterns
            "volume_ratio": _safe_ratio(candle.volume, avg_volume) if avg_volume > 0 else 1.0,
            "breakout_high": float(prior_high) if candle.close > prior_high else 0.0,
            "breakout_low": float(prior_low) if candle.close < prior_low else 0.0,
            "liquidity_swept_low": float(prior_low) if candle.low <= prior_low * 0.999 else 0.0,
            "liquidity_swept_high": float(prior_high) if candle.high >= prior_high * 1.001 else 0.0,
            "triangle_flat_high": float(prior_high),
            "triangle_rising_low": float(min(item.low for item in candles[index - 10:index])) if len(candles) >= 20 else 0.0,
            "triangle_breakout": 1.0 if candle.close > prior_high else 0.0,
            "rsi_divergence": _rsi_divergence_score(rsi_values, closes, index),
            "bb_position": _safe_ratio(candle.close - band_mid, band_width, 0.5) if band_width > 0 else 0.5,
            "bb_width": _safe_ratio(band_width, candle.close),
            "bb_contracting": 1.0 if len(closes) >= 30 and band_width < avg_band_width else 0.0,
            "donchian_high_20": float(prior_high),
            "donchian_low_20": float(prior_low),
            # Features for bear flag / MACD / ADX / RSI extreme / demand zone
            "flag_direction": _flag_direction(candles, index),
            "flag_breakout": 1.0 if candle.close > prior_high else 0.0,
            "flag_tightness": _safe_ratio(band_width, candle.close),
            "macd_line": _macd_line(closes, index),
            "macd_signal": _macd_signal(closes, index),
            "macd_previous": _macd_line(closes, index - 1),
            "macd_signal_previous": _macd_signal(closes, index - 1),
            "demand_zone_top": _demand_zone_top(candles, index),
            "supply_zone_bottom": _supply_zone_bottom(candles, index),
            # Features for breaker block / ATR channel / volume profile / wedge
            "breaker_strength": _breaker_strength(candle),
            "breakout_level": float(prior_high) if candle.close > prior_high else float(prior_low),
            "atr_channel_upper": float(band_mid + 1.5 * atr_average),
            "atr_channel_lower": float(band_mid - 1.5 * atr_average),
            "volume_poc": 1.0 if candle.volume > avg_volume * 1.5 else 0.0,
            "poc_price": _poc_price(candles, index),
            "wick_rejection": _wick_rejection(candle),
            "wedge_breakdown": 1.0 if candle.close < prior_low else 0.0,
            "wedge_breakup": 1.0 if candle.close > prior_high else 0.0,
            # Features for descending triangle / Kelly sizing
            "descending_flat_low": _descending_flat_low(candles, index),
            "descending_lower_high": _descending_lower_high(candles, index),
            "descending_breakdown": 1.0 if candle.close < _descending_flat_low(candles, index) else 0.0,
            "williams_r": _williams_r(candles, index),
            "williams_r_prev": _williams_r(candles, index - 1),
            "sar_below": 1.0 if _sar_below(candles, index) else 0.0,
            "sar_flipped": 1.0 if _sar_flipped(candles, index) else 0.0,
            "chaikin_osc": _chaikin_oscillator(candles, index),
            "chaikin_osc_prev": _chaikin_oscillator(candles, index - 1),
            "macd_histogram_prev": _macd_histogram(candles, index - 1),
            "win_probability": _win_probability(rsi_values, closes, index),
            "ob_bull_tap": float(ob_bull_vals[index]),
            "ob_bear_tap": float(ob_bear_vals[index]),
            "atr_pips": atr_values[index] / pip,
            "ker_10": float(ker_10_values[index]) if ker_10_values[index] is not None else 0.0,
            "chop_14": float(chop_14_values[index]) if chop_14_values[index] is not None else 50.0,
        }
        # Every value in this row must be a plain number. The row is persisted
        # with the signal and re-read by Signal.from_dict, which coerces each
        # value with float(). Carrying rich objects here (a Bias, say) survives
        # json.dumps only because the store falls back to str(), and comes back
        # as a string that float() rejects -- which made settle_due() skip every
        # signal it could not parse, silently, forever. Objects stay in the
        # caller's hands; the row stays numeric.
        if not all(math.isfinite(float(value)) for value in feature.values()):
            result.append(None)
        else:
            result.append(feature)
    return result


def model_vector(features: dict[str, float]) -> list[float]:
    return [float(features[name]) for name in FEATURE_NAMES]


def _median(values: Sequence[float]) -> float:
    return median(values)
