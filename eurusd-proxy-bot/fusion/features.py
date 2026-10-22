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

from .domain import Candle
from .indicators import adx, atr, autocorrelation, ema, mean, median, pstdev, rolling_std, rsi, streak

FEATURE_VERSION = 3

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
}

# How many bars are needed before a feature row is complete.
MIN_FEATURE_BARS = 30


def _safe_ratio(numerator: float, denominator: float, default: float = 0.0) -> float:
    return numerator / denominator if abs(denominator) > 1e-15 else default


def compute_features(candles: Sequence[Candle]) -> list[dict[str, float] | None]:
    """Compute indicators from current and past candles only (no future data)."""
    candles = list(candles)
    closes = [candle.close for candle in candles]
    ema_fast = ema(closes, 5)
    ema_slow = ema(closes, 20)
    rsi_values = rsi(closes, 14)
    atr_values = atr(candles, 14)
    adx_values, plus_di_values, minus_di_values = adx(candles, 14)

    returns: list[float] = [0.0]
    for index in range(1, len(closes)):
        returns.append(_safe_ratio(closes[index] - closes[index - 1], closes[index - 1]))

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

        feature = {
            # model inputs (ordered in FEATURE_NAMES)
            "return_1": returns[index],
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
            # strategy-facing values
            "ema_fast": ema_fast[index],
            "ema_slow": ema_slow[index],
            "ema_fast_previous": ema_fast[index - 1] if ema_fast[index - 1] is not None else ema_fast[index],
            "ema_slow_previous": ema_slow[index - 1] if ema_slow[index - 1] is not None else ema_slow[index],
            "rsi": rsi_values[index],
            "rsi_previous": rsi_previous,
            "atr": atr_values[index],
            "atr_pct": _safe_ratio(atr_values[index], candle.close),
            "adx": adx_values[index],
            "plus_di": plus_di_values[index],
            "minus_di": minus_di_values[index],
            "bb_mid": band_mid,
            "bb_upper": band_mid + band_width,
            "bb_lower": band_mid - band_width,
            "prior_high_20": prior_high,
            "prior_low_20": prior_low,
        }
        if not all(math.isfinite(float(value)) for value in feature.values()):
            result.append(None)
        else:
            result.append(feature)
    return result


def model_vector(features: dict[str, float]) -> list[float]:
    return [float(features[name]) for name in FEATURE_NAMES]


def _median(values: Sequence[float]) -> float:
    return median(values)
