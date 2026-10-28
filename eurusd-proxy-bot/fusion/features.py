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

FEATURE_VERSION = 5

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
}

# How many bars are needed before a feature row is complete.
MIN_FEATURE_BARS = 30


def _safe_ratio(numerator: float, denominator: float, default: float = 0.0) -> float:
    return numerator / denominator if abs(denominator) > 1e-15 else default


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
            # scalping levels
            "stoch_k": stoch_k_values[index],
            "stoch_d": stoch_d_values[index],
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
