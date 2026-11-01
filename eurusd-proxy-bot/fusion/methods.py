"""Multi-family institutional, viral momentum, and quantified scalping prototypes.

Adds three transparent, strictly causal strategy families to the council:

1. **SMC / ICT Institutional Structure (``SMC_ICT``)**
   * ``smc_liquidity_sweep`` — stop-run sweep beyond the prior 15-bar liquidity
     pool followed by an immediate rejection reclaim inside structure.
   * ``smc_fvg_displacement`` — 3-candle institutional displacement Fair Value
     Gap (FVG) impulse or unmitigated FVG retest aligned with trend structure.
   * ``smc_order_block`` — tap and rejection from an unmitigated institutional
     Order Block origin zone preceding a structural break.

2. **Viral Volatility & Momentum (``VIRAL_MOMENTUM``)**
   * ``ttm_keltner_squeeze`` — John Carter Bollinger-inside-Keltner volatility
     squeeze release with linear-regression momentum expansion.
   * ``supertrend_flip`` — causal ATR(10, 3.0) Supertrend structural flip or
     trend-locked continuation with ADX confirmation.
   * ``ema_ribbon_pullback`` — Fibonacci 8/13/21/34 EMA ribbon fan pullback into
     the fast value pocket followed by trend reclaim.

3. **Quantified Precision Scalping & Session (``QUANT_SCALP``)**
   * ``connors_rsi2_reversion`` — Larry Connors 2-period Wilder RSI extreme
     snapback aligned with the 50-bar structural anchor or deep range edge.
   * ``vwap_sigma_reversion`` — institutional rolling VWAP +/-2.0 sigma Z-score
     stretch rejection in bounded non-runaway regimes.
   * ``session_box_breakout`` — tight 30-bar pre-session compression box
     Opening Range Breakout (ORB) with full-body displacement.

Every member is symmetric across BUY and SELL, refuses compressed sub-spread
ticks, and earns council weight strictly from out-of-sample real outcomes.
"""

from __future__ import annotations

from typing import Callable

from . import charter
from .domain import Candle, Decision

FAMILY_SMC_ICT = "SMC_ICT"
FAMILY_VIRAL_MOMENTUM = "VIRAL_MOMENTUM"
FAMILY_QUANT_SCALP = "QUANT_SCALP"

SMC_PROTOTYPES = (
    "smc_liquidity_sweep",
    "smc_fvg_displacement",
    "smc_order_block",
    "smc_liquidity_grab",
    "smc_ascending_triangle",
    "smc_bear_flag",
    "smc_bull_flag",
    "smc_demand_zone",
    "smc_breaker_block",
    "smc_ascending_wedge",
    "smc_descending_triangle",
    "smc_short_trend",
    "smc_micro_fvg",
    "smc_wick_continuation",
    "smc_order_flow",
    "smc_trend_micro",
    "smc_sweep_reversal",
    "smc_trend_follow",
    "smc_candle_direction",
    "smc_trend_streak",
    "smc_order_flow_imbalance",
    "smc_micro_trend",
    "smc_candle_cluster_trend",
    "smc_close_extreme",
    "smc_doji_trend",
    "smc_vwap_position",
    "smc_trend_pullback_micro",
    "smc_trend_resilience",
    "smc_order_block_micro",
    "smc_trend_override",
    "smc_wick_direction",
    "smc_moderate_trend",
    "smc_moderate_candle",
    "smc_low_atr_trend",
    "smc_low_atr_candle",
)

VIRAL_PROTOTYPES = (
    "ttm_keltner_squeeze",
    "supertrend_flip",
    "ema_ribbon_pullback",
    "viral_rsi_divergence",
    "viral_macd_cross",
    "viral_adx_trend",
    "viral_rsi_trend",
    "viral_macd_histogram",
    "viral_volume_profile",
    "viral_ema_cross",
    "viral_stochastic_trend",
    "viral_williams_r",
    "viral_momentum_sprint",
    "viral_stoch_momentum",
    "viral_rsi_velocity",
    "viral_momentum_convergence",
    "viral_stoch_velocity",
    "viral_close_momentum",
    "viral_momentum_burst",
    "viral_price_action",
    "viral_rsi_extreme_follow",
    "viral_price_velocity",
    "viral_ema_slope_momentum",
    "viral_donchian_trend",
    "viral_momentum_continuation",
    "viral_range_position",
    "viral_rsi_zone",
    "viral_candle_velocity",
    "viral_momentum_persistence",
    "viral_close_position_momentum",
    "viral_trend_channel",
    "viral_volume_position",
    "viral_moderate_momentum",
    "viral_moderate_confluence",
    "viral_low_atr_momentum",
    "viral_low_atr_confluence",
)

QUANT_SCALP_PROTOTYPES = (
    "connors_rsi2_reversion",
    "vwap_sigma_reversion",
    "session_box_breakout",
    "quant_bollinger_bounce",
    "quant_donchian_breakout",
    "quant_rsi_extreme",
    "quant_atr_channel",
    "quant_vwap_trend",
    "quant_kelly_sizing",
    "quant_parabolic_sar",
    "quant_chaikin_oscillator",
    "quant_ema_velocity",
    "quant_range_breakout",
    "quant_ema_slope",
    "quant_donchian_slope",
    "quant_breakout_volume",
    "quant_rsi_momentum",
    "quant_trend_strength",
    "quant_price_action_swing",
    "quant_ema_separation",
    "quant_adx_threshold",
    "quant_moderate_range",
    "quant_low_atr_range",
)

METHOD_PROTOTYPES = SMC_PROTOTYPES + VIRAL_PROTOTYPES + QUANT_SCALP_PROTOTYPES

METHOD_FAMILIES: dict[str, str] = {
    **{name: FAMILY_SMC_ICT for name in SMC_PROTOTYPES},
    **{name: FAMILY_VIRAL_MOMENTUM for name in VIRAL_PROTOTYPES},
    **{name: FAMILY_QUANT_SCALP for name in QUANT_SCALP_PROTOTYPES},
}

METHOD_MEMBERS: dict[str, str] = {
    "smc_liquidity_sweep": "SMC/ICT stop-hunt sweep of 15-bar liquidity with wick reclaim",
    "smc_fvg_displacement": "SMC/ICT 3-candle Fair Value Gap displacement or unmitigated retest",
    "smc_order_block": "SMC institutional Order Block mitigation tap with rejection close",
    "smc_liquidity_grab": "SMC/ICT liquidity grab — stop hunt reversal at swept level",
    "smc_ascending_triangle": "SMC/ICT ascending triangle breakout — flat resistance, rising lows",
    "smc_bear_flag": "SMC/ICT flag pattern — strong move + tight consolidation breakout",
    "smc_demand_zone": "SMC/ICT demand/supply zone — buy-side/sell-side imbalance revisit",
    "smc_breaker_block": "SMC/ICT breaker block — strong candle breaks level and holds",
    "smc_ascending_wedge": "SMC/ICT ascending wedge — rising support + resistance converging",
    "smc_descending_triangle": "SMC/ICT descending triangle — flat support + lower highs, bearish breakout",
    "ttm_keltner_squeeze": "TTM Bollinger-inside-Keltner volatility squeeze release",
    "supertrend_flip": "ATR(10, 3.0) Supertrend structural flip with ADX confirmation",
    "ema_ribbon_pullback": "Fibonacci 8/13/21/34 EMA ribbon value-pocket pullback reclaim",
    "viral_rsi_divergence": "RSI momentum divergence — price extreme with RSI divergence",
    "viral_macd_cross": "MACD crossover — momentum line cross with trend filter",
    "viral_adx_trend": "ADX trend strength — enter strong trends only, avoid chop",
    "viral_rsi_trend": "RSI trend — ride RSI in trend direction, fade against trend",
    "viral_volume_profile": "Volume profile — enter at volume point of control with rejection",
    "viral_ema_cross": "EMA cross — fast EMA crosses slow EMA (trend change)",
    "viral_stochastic_trend": "Stochastic trend — stochastic in trend direction confirms move",
    "connors_rsi2_reversion": "Connors 2-period Wilder RSI extreme snapback with 50-EMA filter",
    "vwap_sigma_reversion": "Institutional rolling VWAP +/-2.0 sigma Z-score stretch hook",
    "session_box_breakout": "Tight 30-bar compression box Opening Range Breakout (ORB)",
    "quant_bollinger_bounce": "Bollinger band bounce — fade extreme band touch in contracting range",
    "quant_donchian_breakout": "Donchian channel breakout — Turtle trading 20-bar high/low break",
    "quant_rsi_extreme": "RSI extreme reversal — fade oversold/overbought with RSI turn",
    "quant_atr_channel": "ATR channel breakout — break of ATR-based volatility channel",
    "quant_vwap_trend": "VWAP trend — price above VWAP in uptrend, below in downtrend",
    "quant_kelly_sizing": "Kelly criterion — enter when edge exceeds cost threshold",
    "smc_bull_flag": "SMC bull flag: strong rally, tight consolidation, breakout continuation",
    "viral_macd_histogram": "VIRAL MACD histogram momentum cross: zero-line or rising histogram",
    "quant_parabolic_sar": "QUANT parabolic SAR flip: trend reversal on SAR dot switch",
    "viral_williams_r": "VIRAL Williams %R extreme cross: -80/-20 threshold breakthrough",
    "quant_chaikin_oscillator": "QUANT Chaikin oscillator: volume-weighted momentum cross",
    "smc_short_trend": "SMC short trend: ADX + DI + EMA alignment for short expiry",
    "viral_momentum_sprint": "VIRAL momentum sprint: return acceleration with RSI confirmation",
    "quant_ema_velocity": "QUANT EMA velocity: expanding EMA separation for short expiry",
    "smc_micro_fvg": "SMC micro FVG: small gap above/below 5-bar range for short expiry",
    "viral_stoch_momentum": "VIRAL stochastic momentum: oscillator cross with momentum",
    "smc_wick_continuation": "SMC wick continuation: rejection wick + trend continuation for short expiry",
    "smc_order_flow": "SMC order flow: accumulation/distribution with large body and small wick",
    "viral_rsi_velocity": "VIRAL RSI velocity: RSI acceleration in trend direction for short expiry",
    "viral_momentum_convergence": "VIRAL momentum convergence: multiple momentum signals align for short expiry",
    "quant_range_breakout": "QUANT range breakout: break of tight range with strong candle",
    "smc_trend_micro": "SMC trend micro: micro-trend with EMA alignment and consecutive candles",
    "viral_stoch_velocity": "VIRAL stoch velocity: stochastic cross with velocity confirmation",
    "quant_ema_slope": "QUANT EMA slope: fast EMA slope for short-term momentum",
    "smc_sweep_reversal": "SMC sweep reversal: liquidity sweep followed by immediate reversal",
    "viral_close_momentum": "VIRAL close momentum: close vs previous close for short-term",
    "smc_trend_follow": "SMC trend follow: pure EMA + price action, no RSI filter for extreme conditions",
    "viral_momentum_burst": "VIRAL momentum burst: momentum acceleration burst for short expiry",
    "quant_donchian_slope": "QUANT Donchian slope: direction of Donchian channel for short-term trend",
    "smc_candle_direction": "SMC candle direction: pure candle direction with EMA confirmation",
    "viral_price_action": "VIRAL price action: consecutive candles with EMA confirmation, no oscillator filter",
    "smc_trend_streak": "SMC trend streak: 4+ consecutive candles with EMA alignment",
    "viral_rsi_extreme_follow": "VIRAL RSI extreme follow: trade when RSI is extreme but trend continues",
    "quant_breakout_volume": "QUANT breakout volume: breakout with VWAP and POC confirmation",
    "smc_order_flow_imbalance": "SMC order flow imbalance: close position in extreme of range",
    "viral_price_velocity": "VIRAL price velocity: velocity of price change for short expiry",
    "smc_micro_trend": "SMC micro trend: last 3-candle micro-trend with EMA confirmation",
    "viral_ema_slope_momentum": "VIRAL EMA slope momentum: EMA slope with momentum confirmation",
    "quant_rsi_momentum": "QUANT RSI momentum: RSI direction aligns with price momentum",
    "smc_candle_cluster_trend": "SMC candle cluster trend: multiple candles cluster in one direction",
    "viral_donchian_trend": "VIRAL Donchian trend: price near Donchian channel edge with trend",
    "smc_close_extreme": "SMC close extreme: close at extreme of range regardless of body",
    "viral_momentum_continuation": "VIRAL momentum continuation: trade based on prior momentum continuation",
    "quant_trend_strength": "QUANT trend strength: trade when trend strength is high and price near trend",
    "smc_doji_trend": "SMC doji trend: trade doji candles in direction of prevailing trend",
    "viral_range_position": "VIRAL range position: trade based on close position within recent range",
    "smc_vwap_position": "SMC VWAP position: trade based on price position relative to VWAP",
    "viral_rsi_zone": "VIRAL RSI zone: trade when RSI is in middle zone with momentum",
    "quant_price_action_swing": "QUANT price action swing: trade swing highs and lows in trend direction",
    "smc_trend_pullback_micro": "SMC trend pullback micro: trade pullback to EMA in strong trend",
    "viral_candle_velocity": "VIRAL candle velocity: trade velocity of candle movement for short expiry",
    "smc_trend_resilience": "SMC trend resilience: trade when trend shows resilience against extreme RSI",
    "viral_momentum_persistence": "VIRAL momentum persistence: trade when momentum persists across multiple candles",
    "quant_ema_separation": "QUANT EMA separation: trade when EMA separation is expanding for trend continuation",
    "smc_order_block_micro": "SMC order block micro: trade when price revisits a recent order block",
    "viral_close_position_momentum": "VIRAL close position momentum: trade when close position aligns with momentum",
    "smc_trend_override": "SMC trend override: enter with trend regardless of individual candle direction",
    "viral_trend_channel": "VIRAL trend channel: trade when price is within the EMA trend channel",
    "quant_adx_threshold": "QUANT ADX threshold: trade when ADX crosses above strong trend threshold",
    "smc_wick_direction": "SMC wick direction: trade the direction of the larger wick for reversal",
    "viral_volume_position": "VIRAL volume position: trade when price is at VWAP with trend confirmation",
    "smc_moderate_trend": "SMC moderate trend: trend continuation with relaxed ATR requirements",
    "viral_moderate_momentum": "VIRAL moderate momentum: momentum with relaxed thresholds for quiet markets",
    "quant_moderate_range": "QUANT moderate range: range position with relaxed thresholds",
    "smc_moderate_candle": "SMC moderate candle: candle direction with relaxed body requirements",
    "viral_moderate_confluence": "VIRAL moderate confluence: multiple signals align with relaxed thresholds",
    "smc_low_atr_trend": "SMC low ATR trend: trade trend in very quiet markets with minimal body",
    "viral_low_atr_momentum": "VIRAL low ATR momentum: trade momentum in very quiet markets",
    "quant_low_atr_range": "QUANT low ATR range: trade range position in very quiet markets",
    "smc_low_atr_candle": "SMC low ATR candle: trade candle direction in very quiet markets",
    "viral_low_atr_confluence": "VIRAL low ATR confluence: trade multiple aligned signals in quiet markets",
}

MIN_ATR_SPREAD_COVER = 1.15


def _cost_gate(features: dict[str, float] | None, spread_pips: float) -> str | None:
    if features is None:
        return "Not enough completed candles to calculate indicators."
    atr_pips = float(features.get("atr_pips", 0.0) or 0.0)
    needed = max(spread_pips, 0.0) * MIN_ATR_SPREAD_COVER
    if atr_pips < needed:
        return (
            f"Method abstains: 1m ATR ({atr_pips:.2f}p) is below the "
            f"{needed:.2f}p spread-coverage floor."
        )
    return None


def _wick_ratios(candle: Candle, features: dict[str, float]) -> tuple[float, float]:
    span = candle.high - candle.low
    if span > 1e-12:
        upper = (candle.high - max(candle.open, candle.close)) / span
        lower = (min(candle.open, candle.close) - candle.low) / span
        return upper, lower
    return (
        float(features.get("wick_upper_ratio", 0.0) or 0.0),
        float(features.get("wick_lower_ratio", 0.0) or 0.0),
    )


# ---------------------------------------------------------------------------
# Family 1: SMC / ICT Institutional Structure
# ---------------------------------------------------------------------------
def smc_liquidity_sweep(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """Stop-run sweep outside 15-bar liquidity followed by immediate reclaim."""
    blocked = _cost_gate(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked, None if features is None else probability_up, 0)
    assert features is not None

    prior_high = float(features.get("prior_high_15", features["prior_high_20"]))
    prior_low = float(features.get("prior_low_15", features["prior_low_20"]))
    wick_hi, wick_lo = _wick_ratios(candle, features)
    close_pos = float(features["close_position"])
    rsi = float(features["rsi"])

    swept_low = (
        candle.low < prior_low
        and candle.close > prior_low
        and candle.close > candle.open
        and wick_lo >= 0.30
        and close_pos >= 0.55
        and rsi <= 62.0
    )
    swept_high = (
        candle.high > prior_high
        and candle.close < prior_high
        and candle.close < candle.open
        and wick_hi >= 0.30
        and close_pos <= 0.45
        and rsi >= 38.0
    )
    if swept_low and not swept_high:
        return Decision(
            "BUY",
            "SMC liquidity sweep: sell-side stops below 15-bar low were raided and reclaimed.",
            probability_up,
            2,
        )
    if swept_high and not swept_low:
        return Decision(
            "SELL",
            "SMC liquidity sweep: buy-side stops above 15-bar high were raided and rejected.",
            probability_up,
            2,
        )
    return Decision("NO SIGNAL", "No 15-bar liquidity stop-sweep and reclaim is present.", probability_up, 0)


def smc_fvg_displacement(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """ICT 3-candle displacement Fair Value Gap or unmitigated FVG retest."""
    blocked = _cost_gate(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked, None if features is None else probability_up, 0)
    assert features is not None

    bull_fvg = float(features.get("fvg_bull_active", 0.0) or 0.0) > 0.5
    bear_fvg = float(features.get("fvg_bear_active", 0.0) or 0.0) > 0.5
    rsi = float(features["rsi"])

    bull_ok = (
        bull_fvg
        and features["ema_fast"] >= features["ema_slow"] * 0.9995
        and 35.0 <= rsi <= 70.0
        and candle.close > candle.open
    )
    bear_ok = (
        bear_fvg
        and features["ema_fast"] <= features["ema_slow"] * 1.0005
        and 30.0 <= rsi <= 65.0
        and candle.close < candle.open
    )
    if bull_ok and not bear_ok:
        return Decision(
            "BUY",
            "SMC Fair Value Gap: bullish 3-candle displacement imbalance active with EMA support.",
            probability_up,
            2,
        )
    if bear_ok and not bull_ok:
        return Decision(
            "SELL",
            "SMC Fair Value Gap: bearish 3-candle displacement imbalance active with EMA pressure.",
            probability_up,
            2,
        )
    return Decision("NO SIGNAL", "No aligned SMC Fair Value Gap displacement setup.", probability_up, 0)


def smc_order_block(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """Institutional Order Block mitigation tap with rejection close."""
    blocked = _cost_gate(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked, None if features is None else probability_up, 0)
    assert features is not None

    bull_ob = float(features.get("ob_bull_tap", 0.0) or 0.0) > 0.5
    bear_ob = float(features.get("ob_bear_tap", 0.0) or 0.0) > 0.5
    close_pos = float(features["close_position"])
    rsi = float(features["rsi"])

    if bull_ob and not bear_ob and close_pos >= 0.58 and rsi <= 66.0:
        return Decision(
            "BUY",
            "SMC Order Block: price tapped an unmitigated bullish demand block and rejected higher.",
            probability_up,
            2,
        )
    if bear_ob and not bull_ob and close_pos <= 0.42 and rsi >= 34.0:
        return Decision(
            "SELL",
            "SMC Order Block: price tapped an unmitigated bearish supply block and rejected lower.",
            probability_up,
            2,
        )
    return Decision("NO SIGNAL", "No unmitigated institutional Order Block tap.", probability_up, 0)


# ---------------------------------------------------------------------------
# Family 2: Viral Volatility & Momentum
# ---------------------------------------------------------------------------
def ttm_keltner_squeeze(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """John Carter TTM Squeeze: Bollinger compressed inside Keltner now releasing."""
    blocked = _cost_gate(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked, None if features is None else probability_up, 0)
    assert features is not None

    sq_now = float(features.get("ttm_squeeze_now", 0.0) or 0.0) > 0.5
    sq_recent = float(features.get("ttm_squeeze_recent", 0.0) or 0.0) > 0.5
    mom = float(features.get("linreg_mom_12", 0.0) or 0.0)
    mom_prev = float(features.get("linreg_mom_previous", mom) or 0.0)
    bb_mid = float(features.get("bb_mid", candle.close))

    if not sq_recent or sq_now:
        return Decision("NO SIGNAL", "No TTM Bollinger-inside-Keltner squeeze firing on this bar.", probability_up, 0)

    if mom > 0.15 and mom > mom_prev and candle.close > bb_mid and candle.close > candle.open:
        return Decision(
            "BUY",
            f"TTM Squeeze fired bullish: Keltner compression released with +{mom:.2f}x ATR momentum.",
            probability_up,
            2,
        )
    if mom < -0.15 and mom < mom_prev and candle.close < bb_mid and candle.close < candle.open:
        return Decision(
            "SELL",
            f"TTM Squeeze fired bearish: Keltner compression released with {mom:.2f}x ATR momentum.",
            probability_up,
            2,
        )
    return Decision("NO SIGNAL", "TTM Squeeze released without clean directional momentum.", probability_up, 0)


def supertrend_flip(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """ATR(10, 3.0) Supertrend structural flip with ADX confirmation."""
    blocked = _cost_gate(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked, None if features is None else probability_up, 0)
    assert features is not None

    st_dir = float(features.get("supertrend_dir", 0.0) or 0.0)
    st_flip = float(features.get("supertrend_flipped", 0.0) or 0.0) > 0.5
    adx_val = float(features["adx"])
    rsi = float(features["rsi"])
    body_atr = float(features["body_atr_ratio"])

    bull_imp = st_dir > 0 and (
        st_flip or (adx_val >= 22.0 and features["plus_di"] > features["minus_di"] and previous_close <= features["ema_fast_previous"] and candle.close > features["ema_fast"])
    )
    bear_imp = st_dir < 0 and (
        st_flip or (adx_val >= 22.0 and features["minus_di"] > features["plus_di"] and previous_close >= features["ema_fast_previous"] and candle.close < features["ema_fast"])
    )

    if bull_imp and 45.0 <= rsi <= 70.0 and body_atr >= 0.25 and candle.close > candle.open:
        tag = "structural flip" if st_flip else "trend continuation"
        return Decision("BUY", f"Supertrend(10,3) bullish {tag} confirmed by ADX {adx_val:.1f}.", probability_up, 2)
    if bear_imp and 30.0 <= rsi <= 55.0 and body_atr >= 0.25 and candle.close < candle.open:
        tag = "structural flip" if st_flip else "trend continuation"
        return Decision("SELL", f"Supertrend(10,3) bearish {tag} confirmed by ADX {adx_val:.1f}.", probability_up, 2)
    return Decision("NO SIGNAL", "Supertrend shows no fresh flip or confirmed impulse.", probability_up, 0)


def ema_ribbon_pullback(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """Fibonacci 8/13/21/34 EMA ribbon fan pullback and reclaim."""
    blocked = _cost_gate(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked, None if features is None else probability_up, 0)
    assert features is not None

    e8 = float(features.get("ema_8", features["ema_fast"]))
    e13 = float(features.get("ema_13", features["ema_slow"]))
    e21 = float(features.get("ema_21", features["ema_slow"]))
    e34 = float(features.get("ema_34", features["ema_slow"]))
    rsi = float(features["rsi"])
    close_pos = float(features["close_position"])

    bull_stack = e8 > e13 > e21 > e34
    bear_stack = e8 < e13 < e21 < e34

    if (
        bull_stack
        and candle.low <= e13
        and candle.low >= e34
        and candle.close > e8
        and candle.close > candle.open
        and close_pos >= 0.60
        and 42.0 <= rsi <= 66.0
    ):
        return Decision(
            "BUY",
            "EMA 8/13/21/34 ribbon bullish stack: pullback into 13-EMA value pocket reclaimed 8-EMA.",
            probability_up,
            2,
        )
    if (
        bear_stack
        and candle.high >= e13
        and candle.high <= e34
        and candle.close < e8
        and candle.close < candle.open
        and close_pos <= 0.40
        and 34.0 <= rsi <= 58.0
    ):
        return Decision(
            "SELL",
            "EMA 8/13/21/34 ribbon bearish stack: rally into 13-EMA value pocket rejected below 8-EMA.",
            probability_up,
            2,
        )
    return Decision("NO SIGNAL", "Fibonacci EMA ribbon is not stacked with a value-pocket pullback.", probability_up, 0)


# ---------------------------------------------------------------------------
# Family 3: Quantified Precision Scalping & Session
# ---------------------------------------------------------------------------
def connors_rsi2_reversion(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """Larry Connors 2-period Wilder RSI extreme snapback with 50-EMA anchor."""
    blocked = _cost_gate(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked, None if features is None else probability_up, 0)
    assert features is not None

    rsi2 = float(features.get("rsi_2", features["rsi"]))
    rsi2_prev = float(features.get("rsi_2_previous", rsi2))
    ema50 = float(features.get("ema_50", features["ema_slow"]))
    adx_val = float(features["adx"])

    # Dip in structural uptrend OR deep non-trending 2-period capitulation hook
    bull_dip = (
        (rsi2 <= 10.0 or (rsi2_prev <= 8.0 and rsi2 > rsi2_prev))
        and candle.close > candle.open
        and (candle.close > ema50 or (adx_val < 25.0 and rsi2_prev <= 5.0))
    )
    bear_rip = (
        (rsi2 >= 90.0 or (rsi2_prev >= 92.0 and rsi2 < rsi2_prev))
        and candle.close < candle.open
        and (candle.close < ema50 or (adx_val < 25.0 and rsi2_prev >= 95.0))
    )
    if bull_dip and not bear_rip:
        return Decision(
            "BUY",
            f"Connors RSI(2) snapback: 2-period RSI washed out ({min(rsi2, rsi2_prev):.1f}) and hooked up.",
            probability_up,
            2,
        )
    if bear_rip and not bull_dip:
        return Decision(
            "SELL",
            f"Connors RSI(2) snapback: 2-period RSI spiked ({max(rsi2, rsi2_prev):.1f}) and hooked down.",
            probability_up,
            2,
        )
    return Decision("NO SIGNAL", "Connors 2-period RSI is not at an aligned snapback extreme.", probability_up, 0)


def vwap_sigma_reversion(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """Institutional rolling VWAP +/-2.0 sigma Z-score mean reversion."""
    blocked = _cost_gate(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked, None if features is None else probability_up, 0)
    assert features is not None

    vwap_z = float(features.get("vwap_z", 0.0) or 0.0)
    adx_val = float(features["adx"])
    close_pos = float(features["close_position"])

    if adx_val >= 32.0:
        return Decision("NO SIGNAL", "VWAP sigma reversion stands aside in strong runaway ADX.", probability_up, 0)

    if vwap_z <= -1.95 and candle.close > candle.open and features["return_1"] > 0 and close_pos >= 0.55:
        return Decision(
            "BUY",
            f"VWAP sigma reversion: price stretched {vwap_z:.2f} sigma below VWAP and turned up.",
            probability_up,
            2,
        )
    if vwap_z >= 1.95 and candle.close < candle.open and features["return_1"] < 0 and close_pos <= 0.45:
        return Decision(
            "SELL",
            f"VWAP sigma reversion: price stretched +{vwap_z:.2f} sigma above VWAP and turned down.",
            probability_up,
            2,
        )
    return Decision("NO SIGNAL", "Price is within +/-1.95 sigma of VWAP or shows no turn.", probability_up, 0)


def session_box_breakout(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """Tight 30-bar compression box Opening Range Breakout (ORB)."""
    blocked = _cost_gate(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked, None if features is None else probability_up, 0)
    assert features is not None

    box_atr = float(features.get("prior_box_30_atr", 6.0) or 6.0)
    prior_high = float(features.get("prior_high_30", features["prior_high_20"]))
    prior_low = float(features.get("prior_low_30", features["prior_low_20"]))
    body_atr = float(features["body_atr_ratio"])
    close_pos = float(features["close_position"])

    if box_atr > 4.5:
        return Decision("NO SIGNAL", "Prior 30-bar box is already expanded; no tight ORB coil.", probability_up, 0)

    if (
        candle.close > prior_high
        and previous_close <= prior_high
        and 0.40 <= body_atr <= 1.80
        and close_pos >= 0.68
    ):
        return Decision(
            "BUY",
            f"Session box breakout: closed above coiled 30-bar range ({box_atr:.1f}x ATR box).",
            probability_up,
            2,
        )
    if (
        candle.close < prior_low
        and previous_close >= prior_low
        and 0.40 <= body_atr <= 1.80
        and close_pos <= 0.32
    ):
        return Decision(
            "SELL",
            f"Session box breakout: closed below coiled 30-bar range ({box_atr:.1f}x ATR box).",
            probability_up,
            2,
        )
    return Decision("NO SIGNAL", "Price has not broken out of a coiled 30-bar session box.", probability_up, 0)


def smc_liquidity_grab(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """Liquidity Grab — market sweeps a level then reverses into it.

    Price takes out a recent high/low (stop hunt), then snaps back.
    The snap back is the entry — BUY after a swept low, SELL after a swept high.
    Classic Wyckoff / Smart Money pattern.
    """
    blocked = _cost_gate(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked, None if features is None else probability_up, 0)
    assert features is not None

    swept_low = float(features.get("liquidity_swept_low", 0.0) or 0.0)
    swept_high = float(features.get("liquidity_swept_high", 0.0) or 0.0)
    price = candle.close

    # Bullish: swept a low and closed above it
    if swept_low > 0 and candle.low <= swept_low and candle.close > swept_low and candle.close > candle.open:
        return Decision(
            "BUY",
            f"Liquidity grab: price swept {swept_low:.5f} and closed above it — stop hunt reversed.",
            probability_up, 2,
        )
    # Bearish: swept a high and closed below it
    if swept_high > 0 and candle.high >= swept_high and candle.close < swept_high and candle.close < candle.open:
        return Decision(
            "SELL",
            f"Liquidity grab: price swept {swept_high:.5f} and closed below it — stop hunt reversed.",
            probability_up, 2,
        )
    return Decision("NO SIGNAL", "No liquidity grab — no swept level with reversal.", probability_up, 0)


def smc_ascending_triangle(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """Ascending Triangle Breakout — flat resistance, rising lows, then breakout.

    Classic bullish continuation pattern. Requires: flat highs (resistance),
    rising lows (higher lows), and a breakout above the resistance line.
    """
    blocked = _cost_gate(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked, None if features is None else probability_up, 0)
    assert features is not None

    flat_high = float(features.get("triangle_flat_high", 0.0) or 0.0)
    rising_low = float(features.get("triangle_rising_low", 0.0) or 0.0)
    breakout_confirmed = float(features.get("triangle_breakout", 0.0) or 0.0) > 0.5

    if flat_high > 0 and rising_low > 0 and breakout_confirmed:
        if candle.close > flat_high:
            return Decision(
                "BUY",
                f"Ascending triangle breakout: flat high at {flat_high:.5f} broken with rising lows.",
                probability_up, 3,
            )
    return Decision("NO SIGNAL", "No ascending triangle breakout pattern.", probability_up, 0)


def viral_rsi_divergence(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """RSI Divergence — price makes a new extreme but RSI does not.

    Bullish divergence: price makes a lower low, RSI makes a higher low.
    Bearish divergence: price makes a higher high, RSI makes a lower high.
    Classic momentum divergence signal — one of the most reliable reversal indicators.
    """
    blocked = _cost_gate(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked, None if features is None else probability_up, 0)
    assert features is not None

    rsi_val = float(features.get("rsi_14", 50.0))
    price_change = float(features.get("price_change_pct", 0.0))
    rsi_divergence = float(features.get("rsi_divergence", 0.0) or 0.0)

    # Bullish divergence: price down, RSI up, RSI not oversold
    if rsi_divergence > 0.3 and price_change < 0 and rsi_val > 30.0 and candle.close > candle.open:
        return Decision(
            "BUY",
            f"RSI bullish divergence: price lower low but RSI higher low ({rsi_val:.1f}).",
            probability_up, 2,
        )
    # Bearish divergence: price up, RSI down, RSI not overbought
    if rsi_divergence < -0.3 and price_change > 0 and rsi_val < 70.0 and candle.close < candle.open:
        return Decision(
            "SELL",
            f"RSI bearish divergence: price higher high but RSI lower high ({rsi_val:.1f}).",
            probability_up, 2,
        )
    return Decision("NO SIGNAL", "No RSI divergence detected.", probability_up, 0)


def quant_bollinger_bounce(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """Bollinger Band Bounce — fade the extreme band touch in a range.

    When price touches the outer Bollinger band and the band is contracting
    (low volatility), price typically reverts to the mean. BUY on a lower
    band touch in a contracting range, SELL on an upper band touch.
    """
    blocked = _cost_gate(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked, None if features is None else probability_up, 0)
    assert features is not None

    bb_pos = float(features.get("bb_position", 0.5))
    bb_width = float(features.get("bb_width", 0.0))
    bb_contracting = float(features.get("bb_contracting", 0.0) or 0.0) > 0.5

    if not bb_contracting:
        return Decision("NO SIGNAL", "Bollinger bands not contracting — no mean reversion edge.", probability_up, 0)

    if bb_pos < 0.1 and candle.low <= float(features.get("bb_lower", 0.0)):
        return Decision(
            "BUY",
            f"Bollinger bounce: price touched lower band ({bb_pos:.2f}) in contracting range.",
            probability_up, 2,
        )
    if bb_pos > 0.9 and candle.high >= float(features.get("bb_upper", 0.0)):
        return Decision(
            "SELL",
            f"Bollinger bounce: price touched upper band ({bb_pos:.2f}) in contracting range.",
            probability_up, 2,
        )
    return Decision("NO SIGNAL", "No Bollinger band bounce — price not at extreme band.", probability_up, 0)


def quant_donchian_breakout(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """Donchian Channel Breakout — enter on a break of the N-bar high/low.

    Classic trend-following breakout. When price closes above the 20-bar
    high, enter long. When price closes below the 20-bar low, enter short.
    The Donchian channel is the basis of the original Turtle Trading system.
    """
    blocked = _cost_gate(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked, None if features is None else probability_up, 0)
    assert features is not None

    donchian_high = float(features.get("donchian_high_20", 0.0) or 0.0)
    donchian_low = float(features.get("donchian_low_20", 0.0) or 0.0)
    price = candle.close

    if donchian_high > 0 and price > donchian_high and candle.close > candle.open:
        return Decision(
            "BUY",
            f"Donchian breakout: price closed above 20-bar high at {donchian_high:.5f}.",
            probability_up, 3,
        )
    if donchian_low > 0 and price < donchian_low and candle.close < candle.open:
        return Decision(
            "SELL",
            f"Donchian breakout: price closed below 20-bar low at {donchian_low:.5f}.",
            probability_up, 3,
        )
    return Decision("NO SIGNAL", "No Donchian channel breakout.", probability_up, 0)


def smc_bear_flag(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """Bear/Bull Flag — strong move followed by a tight counter-trend channel.

    Classic continuation pattern. After a strong directional move, price
    consolidates in a tight channel (flag) against the move. Breakout of
    the flag continues the original direction. BUY on a bull flag breakout,
    SELL on a bear flag breakout.
    """
    blocked = _cost_gate(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked, None if features is None else probability_up, 0)
    assert features is not None

    flag_direction = float(features.get("flag_direction", 0.0) or 0.0)
    flag_breakout = float(features.get("flag_breakout", 0.0) or 0.0) > 0.5
    flag_tight = float(features.get("flag_tightness", 0.0) or 0.0) < 0.5

    if not flag_breakout or not flag_tight:
        return Decision("NO SIGNAL", "No flag pattern — no tight consolidation or breakout.", probability_up, 0)

    if flag_direction > 0 and candle.close > candle.open:
        return Decision(
            "BUY",
            f"Bull flag breakout: {flag_direction:.1f} strength, tight consolidation broken.",
            probability_up, 3,
        )
    if flag_direction < 0 and candle.close < candle.open:
        return Decision(
            "SELL",
            f"Bear flag breakout: {flag_direction:.1f} strength, tight consolidation broken.",
            probability_up, 3,
        )
    return Decision("NO SIGNAL", "Flag not aligned with candle direction.", probability_up, 0)


def viral_macd_cross(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """MACD Cross — classic momentum crossover signal.

    Bullish: MACD line crosses above signal line.
    Bearish: MACD line crosses below signal line.
    One of the most widely followed momentum indicators.
    """
    blocked = _cost_gate(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked, None if features is None else probability_up, 0)
    assert features is not None

    macd_line = float(features.get("macd_line", 0.0))
    signal_line = float(features.get("macd_signal", 0.0))
    macd_prev = float(features.get("macd_previous", 0.0))
    signal_prev = float(features.get("macd_signal_previous", 0.0))

    # Bullish cross
    if macd_prev <= signal_prev and macd_line > signal_line:
        return Decision(
            "BUY",
            f"MACD bullish cross: MACD ({macd_line:.4f}) crossed above signal ({signal_line:.4f}).",
            probability_up, 2,
        )
    # Bearish cross
    if macd_prev >= signal_prev and macd_line < signal_line:
        return Decision(
            "SELL",
            f"MACD bearish cross: MACD ({macd_line:.4f}) crossed below signal ({signal_line:.4f}).",
            probability_up, 2,
        )
    return Decision("NO SIGNAL", "No MACD crossover — lines not crossing.", probability_up, 0)


def viral_adx_trend(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """ADX Trend Strength — enter strong trends, avoid chop.

    ADX > 25 indicates a strong trend. +DI > -DI confirms bullish trend,
    -DI > +DI confirms bearish trend. This enters strong trends only,
    avoiding the range regime where most indicators fail.
    """
    blocked = _cost_gate(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked, None if features is None else probability_up, 0)
    assert features is not None

    adx_val = float(features.get("adx", 0.0))
    plus_di = float(features.get("plus_di", 0.0))
    minus_di = float(features.get("minus_di", 0.0))
    supertrend_dir = float(features.get("supertrend_dir", 0.0))

    if adx_val < 25.0:
        return Decision("NO SIGNAL", f"ADX {adx_val:.1f} — not a strong trend.", probability_up, 0)

    if plus_di > minus_di and supertrend_dir > 0 and candle.close > candle.open:
        return Decision(
            "BUY",
            f"Strong uptrend: ADX {adx_val:.1f}, +DI ({plus_di:.1f}) > -DI ({minus_di:.1f}).",
            probability_up, 3,
        )
    if minus_di > plus_di and supertrend_dir < 0 and candle.close < candle.open:
        return Decision(
            "SELL",
            f"Strong downtrend: ADX {adx_val:.1f}, -DI ({minus_di:.1f}) > +DI ({plus_di:.1f}).",
            probability_up, 3,
        )
    return Decision("NO SIGNAL", "ADX strong but direction not confirmed.", probability_up, 0)


def quant_rsi_extreme(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """RSI Extreme Reversal — fade oversold/overbought extremes.

    When RSI hits extreme levels (>70 or <30) and shows reversal signs,
    fade the extreme. BUY when RSI < 30 and candle turns bullish,
    SELL when RSI > 70 and candle turns bearish.
    """
    blocked = _cost_gate(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked, None if features is None else probability_up, 0)
    assert features is not None

    rsi_val = float(features.get("rsi", 50.0))
    rsi_prev = float(features.get("rsi_previous", rsi_val))

    # Oversold reversal
    if rsi_val < 30.0 and rsi_val > rsi_prev and candle.close > candle.open:
        return Decision(
            "BUY",
            f"RSI extreme reversal: RSI {rsi_val:.1f} rising from oversold.",
            probability_up, 2,
        )
    # Overbought reversal
    if rsi_val > 70.0 and rsi_val < rsi_prev and candle.close < candle.open:
        return Decision(
            "SELL",
            f"RSI extreme reversal: RSI {rsi_val:.1f} falling from overbought.",
            probability_up, 2,
        )
    return Decision("NO SIGNAL", f"RSI {rsi_val:.1f} — not at extreme.", probability_up, 0)


def smc_demand_zone(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """Demand/Supply Zone — buy at demand, sell at supply.

    SMC (Smart Money Concepts) demand zones are areas where buyers previously
    stepped in strongly (buy-side imbalance). When price revisits a demand
    zone, buyers are likely to step in again. Supply zones are the opposite.
    """
    blocked = _cost_gate(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked, None if features is None else probability_up, 0)
    assert features is not None

    demand_zone_top = float(features.get("demand_zone_top", 0.0) or 0.0)
    supply_zone_bottom = float(features.get("supply_zone_bottom", 0.0) or 0.0)
    price = candle.close

    # At demand zone
    if demand_zone_top > 0 and candle.low <= demand_zone_top and candle.close > candle.open and candle.close > demand_zone_top * 0.999:
        return Decision(
            "BUY",
            f"Demand zone: price revisited buy-side imbalance at {demand_zone_top:.5f}.",
            probability_up, 3,
        )
    # At supply zone
    if supply_zone_bottom > 0 and candle.high >= supply_zone_bottom and candle.close < candle.open and candle.close < supply_zone_bottom * 1.001:
        return Decision(
            "SELL",
            f"Supply zone: price revisited sell-side imbalance at {supply_zone_bottom:.5f}.",
            probability_up, 3,
        )
    return Decision("NO SIGNAL", "Not at a demand or supply zone.", probability_up, 0)


def smc_breaker_block(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """Breaker Block — a strong candle that breaks a level and holds.

    SMC concept: a strong bullish/bearish candle that breaks a previous
    swing high/low and closes near its extreme. The close confirms the
    break. BUY on a bullish breaker block, SELL on a bearish one.
    """
    blocked = _cost_gate(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked, None if features is None else probability_up, 0)
    assert features is not None

    breaker_strength = float(features.get("breaker_strength", 0.0) or 0.0)
    breakout_level = float(features.get("breakout_level", 0.0) or 0.0)

    if breaker_strength < 0.6 or breakout_level <= 0:
        return Decision("NO SIGNAL", "No breaker block — weak candle or no breakout level.", probability_up, 0)

    if candle.close > candle.open and candle.close > breakout_level:
        return Decision(
            "BUY",
            f"Breaker block: bullish candle broke {breakout_level:.5f} with {breaker_strength:.1f} strength.",
            probability_up, 3,
        )
    if candle.close < candle.open and candle.close < breakout_level:
        return Decision(
            "SELL",
            f"Breaker block: bearish candle broke {breakout_level:.5f} with {breaker_strength:.1f} strength.",
            probability_up, 3,
        )
    return Decision("NO SIGNAL", "Candle not aligned with breakout level.", probability_up, 0)


def viral_rsi_trend(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """RSI Trend — ride RSI in trend direction, fade RSI against trend.

    In an uptrend (EMA slope > 0), buy when RSI > 50 (momentum confirms trend).
    In a downtrend (EMA slope < 0), sell when RSI < 50 (momentum confirms trend).
    """
    blocked = _cost_gate(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked, None if features is None else probability_up, 0)
    assert features is not None

    rsi_val = float(features.get("rsi", 50.0))
    ema_slope = float(features.get("ema_slope", 0.0))
    supertrend_dir = float(features.get("supertrend_dir", 0.0))

    if ema_slope > 0.00005 and rsi_val > 50.0 and supertrend_dir > 0:
        return Decision(
            "BUY",
            f"RSI trend: uptrend confirmed, RSI {rsi_val:.1f} above 50.",
            probability_up, 2,
        )
    if ema_slope < -0.00005 and rsi_val < 50.0 and supertrend_dir < 0:
        return Decision(
            "SELL",
            f"RSI trend: downtrend confirmed, RSI {rsi_val:.1f} below 50.",
            probability_up, 2,
        )
    return Decision("NO SIGNAL", "RSI not confirming trend direction.", probability_up, 0)


def quant_atr_channel(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """ATR Channel Breakout — break of an ATR-based channel.

    Price channel built from ATR multiples around a rolling mean. Breakout
    above the upper channel = bullish, breakout below lower channel = bearish.
    Channel width adapts to volatility.
    """
    blocked = _cost_gate(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked, None if features is None else probability_up, 0)
    assert features is not None

    channel_upper = float(features.get("atr_channel_upper", 0.0) or 0.0)
    channel_lower = float(features.get("atr_channel_lower", 0.0) or 0.0)
    price = candle.close

    if channel_upper > 0 and price > channel_upper and candle.close > candle.open:
        return Decision(
            "BUY",
            f"ATR channel breakout: price broke upper channel at {channel_upper:.5f}.",
            probability_up, 3,
        )
    if channel_lower > 0 and price < channel_lower and candle.close < candle.open:
        return Decision(
            "SELL",
            f"ATR channel breakout: price broke lower channel at {channel_lower:.5f}.",
            probability_up, 3,
        )
    return Decision("NO SIGNAL", "No ATR channel breakout.", probability_up, 0)


def viral_volume_profile(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """Volume Profile — enter where volume is highest (point of control).

    When price is at a volume point of control (high volume node) and shows
    rejection, enter in the direction of the rejection. High volume nodes
    act as magnets — price tends to revisit them.
    """
    blocked = _cost_gate(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked, None if features is None else probability_up, 0)
    assert features is not None

    at_poc = float(features.get("volume_poc", 0.0) or 0.0) > 0.5
    poc_price = float(features.get("poc_price", 0.0) or 0.0)
    rejection = float(features.get("wick_rejection", 0.0) or 0.0)

    if not at_poc or poc_price <= 0 or rejection < 0.5:
        return Decision("NO SIGNAL", "Not at volume point of control or no rejection.", probability_up, 0)

    if candle.close > candle.open:
        return Decision(
            "BUY",
            f"Volume profile: price at POC {poc_price:.5f} with rejection, bullish close.",
            probability_up, 2,
        )
    if candle.close < candle.open:
        return Decision(
            "SELL",
            f"Volume profile: price at POC {poc_price:.5f} with rejection, bearish close.",
            probability_up, 2,
        )
    return Decision("NO SIGNAL", "No directional close at POC.", probability_up, 0)


def smc_ascending_wedge(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """Ascending Wedge — rising support + rising resistance converging.

    Typically a bearish reversal pattern. When the wedge breaks downward
    from the lower support line, enter SELL. When it breaks upward from
    the upper resistance, enter BUY (less common but valid).
    """
    blocked = _cost_gate(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked, None if features is None else probability_up, 0)
    assert features is not None

    wedge_breakdown = float(features.get("wedge_breakdown", 0.0) or 0.0) > 0.5
    wedge_breakup = float(features.get("wedge_breakup", 0.0) or 0.0) > 0.5

    if wedge_breakdown and candle.close < candle.open:
        return Decision(
            "SELL",
            "Ascending wedge breakdown: support line broken, bearish reversal.",
            probability_up, 3,
        )
    if wedge_breakup and candle.close > candle.open:
        return Decision(
            "BUY",
            "Ascending wedge breakup: resistance line broken, bullish continuation.",
            probability_up, 3,
        )
    return Decision("NO SIGNAL", "No wedge breakout.", probability_up, 0)


def smc_descending_triangle(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """Descending Triangle — flat support + lower highs, bearish breakout.

    Classic bearish continuation pattern. Flat support (demand zone) with
    decreasing rallies (lower highs). Break below support = SELL.
    """
    blocked = _cost_gate(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked, None if features is None else probability_up, 0)
    assert features is not None

    flat_low = float(features.get("descending_flat_low", 0.0) or 0.0)
    lower_high = float(features.get("descending_lower_high", 0.0) or 0.0)
    breakdown = float(features.get("descending_breakdown", 0.0) or 0.0) > 0.5

    if flat_low > 0 and lower_high > 0 and breakdown and candle.close < flat_low:
        return Decision(
            "SELL",
            f"Descending triangle breakdown: flat support at {flat_low:.5f} broken.",
            probability_up, 3,
        )
    return Decision("NO SIGNAL", "No descending triangle breakdown.", probability_up, 0)


def viral_ema_cross(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """EMA Cross — fast EMA crosses slow EMA.

    Classic trend change signal. Bullish: fast EMA crosses above slow EMA.
    Bearish: fast EMA crosses below slow EMA.
    """
    blocked = _cost_gate(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked, None if features is None else probability_up, 0)
    assert features is not None

    ema_fast = float(features.get("ema_fast", 0.0))
    ema_slow = float(features.get("ema_slow", 0.0))
    ema_fast_prev = float(features.get("ema_fast_previous", ema_fast))
    ema_slow_prev = float(features.get("ema_slow_previous", ema_slow))

    if ema_fast_prev <= ema_slow_prev and ema_fast > ema_slow:
        return Decision(
            "BUY",
            f"EMA cross: fast ({ema_fast:.5f}) crossed above slow ({ema_slow:.5f}).",
            probability_up, 2,
        )
    if ema_fast_prev >= ema_slow_prev and ema_fast < ema_slow:
        return Decision(
            "SELL",
            f"EMA cross: fast ({ema_fast:.5f}) crossed below slow ({ema_slow:.5f}).",
            probability_up, 2,
        )
    return Decision("NO SIGNAL", "No EMA crossover.", probability_up, 0)


def viral_stochastic_trend(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """Stochastic Trend — stochastic in trend direction confirms the move.

    In an uptrend (EMA slope > 0), stochastic rising from oversold = buy.
    In a downtrend (EMA slope < 0), stochastic falling from overbought = sell.
    """
    blocked = _cost_gate(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked, None if features is None else probability_up, 0)
    assert features is not None

    stoch_k = float(features.get("stoch_k", 50.0))
    stoch_k_prev = float(features.get("stoch_k_previous", stoch_k))
    ema_slope = float(features.get("ema_slope", 0.0))

    if ema_slope > 0.00005 and stoch_k > stoch_k_prev and stoch_k < 60.0:
        return Decision(
            "BUY",
            f"Stochastic trend: rising from {stoch_k:.1f} in uptrend.",
            probability_up, 2,
        )
    if ema_slope < -0.00005 and stoch_k < stoch_k_prev and stoch_k > 40.0:
        return Decision(
            "SELL",
            f"Stochastic trend: falling from {stoch_k:.1f} in downtrend.",
            probability_up, 2,
        )
    return Decision("NO SIGNAL", "Stochastic not confirming trend.", probability_up, 0)


def quant_vwap_trend(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """VWAP Trend — price above VWAP in uptrend, below in downtrend.

    Institutional order flow benchmark. In an uptrend, buy when price
    pulls back to VWAP. In a downtrend, sell when price rallies to VWAP.
    """
    blocked = _cost_gate(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked, None if features is None else probability_up, 0)
    assert features is not None

    vwap_val = float(features.get("vwap", 0.0))
    ema_slope = float(features.get("ema_slope", 0.0))
    price = candle.close
    dist_pips = abs(price - vwap_val) / 0.0001

    if ema_slope > 0.00005 and dist_pips < 5.0 and candle.close > candle.open:
        return Decision(
            "BUY",
            f"VWAP trend: price reclaimed VWAP ({vwap_val:.5f}) in uptrend.",
            probability_up, 2,
        )
    if ema_slope < -0.00005 and dist_pips < 5.0 and candle.close < candle.open:
        return Decision(
            "SELL",
            f"VWAP trend: price rejected at VWAP ({vwap_val:.5f}) in downtrend.",
            probability_up, 2,
        )
    return Decision("NO SIGNAL", "Price not at VWAP for trend entry.", probability_up, 0)


def quant_kelly_sizing(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """Kelly Criterion Sizing — enter when edge exceeds cost threshold.

    Uses the Kelly formula: f* = (b*p - q) / b, where b = win/loss ratio,
    p = win probability, q = 1 - p. Only enter when Kelly fraction > 0,
    indicating a positive expected value trade.
    """
    blocked = _cost_gate(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked, None if features is None else probability_up, 0)
    assert features is not None

    win_prob = float(features.get("win_probability", 0.0))
    atr_pips = float(features.get("atr_pips", 0.0))
    cost_pips = float(spread_pips)

    if win_prob <= 0.5:
        return Decision("NO SIGNAL", f"Win probability {win_prob:.2f} too low for Kelly.", probability_up, 0)

    # Kelly: b = ATR/cost (reward to risk ratio)
    b = atr_pips / cost_pips if cost_pips > 0 else 1.0
    q = 1.0 - win_prob
    kelly_f = (b * win_prob - q) / b if b > 0 else 0.0

    if kelly_f > 0.05 and candle.close > candle.open:
        return Decision(
            "BUY",
            f"Kelly sizing: edge = {kelly_f:.1%} (win prob {win_prob:.0%}, ratio {b:.1f}).",
            probability_up, 2,
        )
    if kelly_f > 0.05 and candle.close < candle.open:
        return Decision(
            "SELL",
            f"Kelly sizing: edge = {kelly_f:.1%} (win prob {win_prob:.0%}, ratio {b:.1f}).",
            probability_up, 2,
        )
    return Decision("NO SIGNAL", f"No Kelly edge (f*={kelly_f:.3f}).", probability_up, 0)



def smc_bull_flag(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """Bull Flag — strong rally followed by a tight counter-trend channel.

    Classic continuation pattern. After a strong rally, price consolidates
    in a tight channel (flag) against the move. Breakout of the flag
    continues the rally. BUY on the breakout.
    """
    blocked = _cost_gate(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked, None if features is None else probability_up, 0)
    assert features is not None

    flag_direction = float(features.get("flag_direction", 0.0) or 0.0)
    flag_breakout = float(features.get("flag_breakout", 0.0) or 0.0) > 0.5
    flag_tight = float(features.get("flag_tightness", 0.0) or 0.0) < 0.5

    if not flag_breakout or not flag_tight:
        return Decision("NO SIGNAL", "No flag pattern — no tight consolidation or breakout.", probability_up, 0)

    if flag_direction > 0 and candle.close > candle.open:
        return Decision(
            "BUY",
            f"Bull flag breakout: {flag_direction:.1f} strength, tight consolidation broken.",
            probability_up, 3,
        )
    return Decision("NO SIGNAL", "Flag not aligned with candle direction.", probability_up, 0)


def viral_macd_histogram(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """MACD Histogram — momentum acceleration/deceleration.

    Bullish: histogram rising from negative or crossing above zero.
    Bearish: histogram falling from positive or crossing below zero.
    """
    blocked = _cost_gate(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked, None if features is None else probability_up, 0)
    assert features is not None

    macd_line = float(features.get("macd_line", 0.0))
    signal_line = float(features.get("macd_signal", 0.0))
    hist = macd_line - signal_line
    hist_prev = float(features.get("macd_histogram_prev", 0.0))

    if hist_prev < 0 and hist > 0:
        return Decision("BUY", f"MACD histogram crossed above zero: {hist:.4f}.", probability_up, 2)
    if hist_prev > 0 and hist < 0:
        return Decision("SELL", f"MACD histogram crossed below zero: {hist:.4f}.", probability_up, 2)
    if hist > 0 and hist > hist_prev * 1.1 and candle.close > candle.open:
        return Decision("BUY", f"MACD histogram rising: {hist:.4f}.", probability_up, 2)
    if hist < 0 and hist < hist_prev * 1.1 and candle.close < candle.open:
        return Decision("SELL", f"MACD histogram falling: {hist:.4f}.", probability_up, 2)
    return Decision("NO SIGNAL", "No MACD histogram signal.", probability_up, 0)


def quant_parabolic_sar(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """Parabolic SAR — trend reversal indicator.

    Bullish: SAR dots below price (uptrend). Bearish: SAR dots above price.
    Enter on SAR flip — when the dots switch sides.
    """
    blocked = _cost_gate(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked, None if features is None else probability_up, 0)
    assert features is not None

    sar_below = float(features.get("sar_below", 0.0) or 0.0) > 0.5
    sar_flipped = float(features.get("sar_flipped", 0.0) or 0.0) > 0.5

    if sar_flipped and sar_below and candle.close > candle.open:
        return Decision("BUY", "Parabolic SAR flip: dots moved below price, bullish.", probability_up, 3)
    if sar_flipped and not sar_below and candle.close < candle.open:
        return Decision("SELL", "Parabolic SAR flip: dots moved above price, bearish.", probability_up, 3)
    return Decision("NO SIGNAL", "No SAR flip — trend not reversing.", probability_up, 0)


def viral_williams_r(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """Williams %R — oversold/overbought momentum indicator.

    -80 = oversold (bullish), -20 = overbought (bearish).
    BUY when Williams R rises above -80, SELL when it falls below -20.
    """
    blocked = _cost_gate(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked, None if features is None else probability_up, 0)
    assert features is not None

    wr_val = float(features.get("williams_r", -50.0))
    wr_prev = float(features.get("williams_r_prev", wr_val))

    if wr_prev < -80.0 and wr_val > -80.0 and candle.close > candle.open:
        return Decision("BUY", f"Williams R rose above -80: {wr_val:.1f}.", probability_up, 2)
    if wr_prev > -20.0 and wr_val < -20.0 and candle.close < candle.open:
        return Decision("SELL", f"Williams R fell below -20: {wr_val:.1f}.", probability_up, 2)
    return Decision("NO SIGNAL", f"Williams R {wr_val:.1f} — not at extreme.", probability_up, 0)


def quant_chaikin_oscillator(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """Chaikin Oscillator — volume-weighted momentum.

    Bullish: oscillator rising from negative or crossing above zero.
    Bearish: oscillator falling from positive or crossing below zero.
    """
    blocked = _cost_gate(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked, None if features is None else probability_up, 0)
    assert features is not None

    chaikin_val = float(features.get("chaikin_osc", 0.0))
    chaikin_prev = float(features.get("chaikin_osc_prev", 0.0))

    if chaikin_prev < 0 and chaikin_val > 0 and candle.close > candle.open:
        return Decision("BUY", f"Chaikin oscillator crossed above zero: {chaikin_val:.4f}.", probability_up, 2)
    if chaikin_prev > 0 and chaikin_val < 0 and candle.close < candle.open:
        return Decision("SELL", f"Chaikin oscillator crossed below zero: {chaikin_val:.4f}.", probability_up, 2)
    if chaikin_val > 0 and chaikin_val > chaikin_prev and candle.close > candle.open:
        return Decision("BUY", f"Chaikin oscillator rising: {chaikin_val:.4f}.", probability_up, 2)
    if chaikin_val < 0 and chaikin_val < chaikin_prev and candle.close < candle.open:
        return Decision("SELL", f"Chaikin oscillator falling: {chaikin_val:.4f}.", probability_up, 2)
    return Decision("NO SIGNAL", "No Chaikin oscillator signal.", probability_up, 0)



def smc_short_trend(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """Short Trend — follow the immediate trend with ADX confirmation.

    For 1-3 minute expiry: when ADX > 20 and the dominant DI aligns with
    the candle direction and EMA trend, enter with the trend. The shorter
    the expiry, the more important the immediate momentum alignment.
    """
    blocked = _cost_gate(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked, None if features is None else probability_up, 0)
    assert features is not None

    adx = float(features.get("adx", 0.0))
    adx_prev = float(features.get("adx_prev", 0.0))
    plus_di = float(features.get("plus_di", 0.0))
    minus_di = float(features.get("minus_di", 0.0))
    ema_fast = float(features.get("ema_fast", 0.0))
    ema_slow = float(features.get("ema_slow", 0.0))
    close_pos = float(features.get("close_position", 0.5))

    if adx > 20.0 and plus_di > minus_di and ema_fast > ema_slow and close_pos >= 0.6:
        return Decision(
            "BUY",
            f"Short trend: ADX {adx:.1f}, +DI > -DI, EMA bullish, close at {close_pos:.0%} of range.",
            probability_up, 3,
        )
    if adx > 20.0 and minus_di > plus_di and ema_fast < ema_slow and close_pos <= 0.4:
        return Decision(
            "SELL",
            f"Short trend: ADX {adx:.1f}, -DI > +DI, EMA bearish, close at {close_pos:.0%} of range.",
            probability_up, 3,
        )
    return Decision("NO SIGNAL", "No short trend alignment.", probability_up, 0)


def viral_momentum_sprint(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """Momentum Sprint — trade momentum acceleration with RSI confirmation.

    For 1-3 minute expiry: when 1-bar return is accelerating (current > previous)
    and RSI is in the favorable zone (50-70 for BUY, 30-50 for SELL), enter.
    This captures the initial thrust of a short-term move.
    """
    blocked = _cost_gate(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked, None if features is None else probability_up, 0)
    assert features is not None

    momentum = float(features.get("return_1", 0.0))
    momentum_prev = float(features.get("return_1_prev", 0.0))
    rsi = float(features.get("rsi", 50.0))
    close = candle.close
    body_atr = float(features.get("body_atr", 0.0))

    if (
        momentum > 0
        and momentum > momentum_prev * 1.2
        and 50.0 < rsi < 70.0
        and close > candle.open
        and body_atr >= 0.12
    ):
        return Decision(
            "BUY",
            f"Momentum sprint: return accelerating, RSI {rsi:.1f}, body {body_atr:.2f} ATR.",
            probability_up, 2,
        )
    if (
        momentum < 0
        and momentum < momentum_prev * 1.2
        and 30.0 < rsi < 50.0
        and close < candle.open
        and body_atr >= 0.12
    ):
        return Decision(
            "SELL",
            f"Momentum sprint: return accelerating, RSI {rsi:.1f}, body {body_atr:.2f} ATR.",
            probability_up, 2,
        )
    return Decision("NO SIGNAL", "No momentum sprint.", probability_up, 0)


def quant_ema_velocity(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """EMA Velocity — trade the rate of change of EMA separation.

    For 1-3 minute expiry: when the EMA fast/slow separation is increasing
    (expanding) and the candle closes in the direction of expansion, enter.
    This captures the acceleration of trend, not just its presence.
    """
    blocked = _cost_gate(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked, None if features is None else probability_up, 0)
    assert features is not None

    ema_fast = float(features.get("ema_fast", 0.0))
    ema_slow = float(features.get("ema_slow", 0.0))
    ema_fast_prev = float(features.get("ema_fast_previous", ema_fast))
    ema_slow_prev = float(features.get("ema_slow_previous", ema_slow))
    close = candle.close
    rsi = float(features.get("rsi", 50.0))

    sep_now = ema_fast - ema_slow
    sep_prev = ema_fast_prev - ema_slow_prev

    if sep_now > 0 and sep_now > sep_prev and close > candle.open and 45.0 < rsi < 70.0:
        return Decision(
            "BUY",
            f"EMA velocity: separation expanding ({sep_prev:.5f} → {sep_now:.5f}), bullish candle.",
            probability_up, 2,
        )
    if sep_now < 0 and sep_now < sep_prev and close < candle.open and 30.0 < rsi < 55.0:
        return Decision(
            "SELL",
            f"EMA velocity: separation expanding ({sep_prev:.5f} → {sep_now:.5f}), bearish candle.",
            probability_up, 2,
        )
    return Decision("NO SIGNAL", "No EMA velocity expansion.", probability_up, 0)


def smc_micro_fvg(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """Micro FVG — trade a small Fair Value Gap on short timeframes.

    For 1-3 minute expiry: when the current candle gaps above the previous
    candle's high (bullish FVG) or below its low (bearish FVG) and closes
    in the direction of the gap, enter. Small FVGs on M1 are powerful
    short-term continuation signals.
    """
    blocked = _cost_gate(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked, None if features is None else probability_up, 0)
    assert features is not None

    close = candle.close
    candle_open = candle.open
    body_atr = float(features.get("body_atr", 0.0))
    prior_high = float(features.get("prior_high_5", 0.0))
    prior_low = float(features.get("prior_low_5", 0.0))
    close_pos = float(features.get("close_position", 0.5))

    # Bullish FVG: gap above previous high, close bullish
    if (
        candle_open > prior_high
        and close > candle_open
        and close_pos >= 0.6
        and body_atr >= 0.1
    ):
        return Decision(
            "BUY",
            f"Micro FVG: bullish gap above 5-bar high, close at {close_pos:.0%} of range.",
            probability_up, 2,
        )

    # Bearish FVG: gap below previous low, close bearish
    if (
        candle_open < prior_low
        and close < candle_open
        and close_pos <= 0.4
        and body_atr >= 0.1
    ):
        return Decision(
            "SELL",
            f"Micro FVG: bearish gap below 5-bar low, close at {close_pos:.0%} of range.",
            probability_up, 2,
        )
    return Decision("NO SIGNAL", "No micro FVG pattern.", probability_up, 0)


def viral_stoch_momentum(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """Stoch Momentum — trade stochastic cross with momentum confirmation.

    For 1-3 minute expiry: when stochastic K crosses above/below signal
    (or crosses 20/80) and the 1-bar return confirms the direction, enter.
    This combines oscillator timing with momentum confirmation.
    """
    blocked = _cost_gate(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked, None if features is None else probability_up, 0)
    assert features is not None

    stoch_k = float(features.get("stoch_k", 50.0))
    stoch_k_prev = float(features.get("stoch_k_prev", stoch_k))
    momentum = float(features.get("return_1", 0.0))
    close = candle.close
    rsi = float(features.get("rsi", 50.0))

    # Stochastic rising from oversold with bullish momentum
    if (
        stoch_k_prev < 25.0
        and stoch_k >= 25.0
        and momentum > 0
        and close > candle.open
        and 45.0 < rsi < 65.0
    ):
        return Decision(
            "BUY",
            f"Stoch momentum: K rising from oversold ({stoch_k_prev:.1f} → {stoch_k:.1f}), momentum confirmed.",
            probability_up, 2,
        )

    # Stochastic falling from overbought with bearish momentum
    if (
        stoch_k_prev > 75.0
        and stoch_k <= 75.0
        and momentum < 0
        and close < candle.open
        and 35.0 < rsi < 55.0
    ):
        return Decision(
            "SELL",
            f"Stoch momentum: K falling from overbought ({stoch_k_prev:.1f} → {stoch_k:.1f}), momentum confirmed.",
            probability_up, 2,
        )
    return Decision("NO SIGNAL", "No stochastic momentum cross.", probability_up, 0)



def smc_wick_continuation(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """Wick Continuation — trade wick rejection with trend continuation.

    For 1-3 minute expiry: when the candle has a long wick in the opposite
    direction of the trend (rejection wick) and closes in the trend direction,
    enter with the trend. This is a high-probability continuation setup.
    """
    blocked = _cost_gate(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked, None if features is None else probability_up, 0)
    assert features is not None

    candle_direction = float(features.get("candle_direction", 0.0))
    wick_ratio = float(features.get("wick_ratio", 0.0))
    body_atr = float(features.get("body_atr", 0.0))
    ema_fast = float(features.get("ema_fast", 0.0))
    ema_slow = float(features.get("ema_slow", 0.0))
    close_pos = float(features.get("close_position", 0.5))
    range_atr = float(features.get("range_atr_ratio", 0.0))

    # Bullish: downtrend rejected, bullish candle in uptrend
    if (
        candle_direction > 0
        and ema_fast > ema_slow
        and close_pos >= 0.65
        and wick_ratio >= 0.2
        and body_atr >= 0.15
        and range_atr >= 0.3
    ):
        return Decision(
            "BUY",
            f"Wick continuation: rejection wick + bullish close, EMA bullish, "
            f"wick {wick_ratio:.2f} ATR, body {body_atr:.2f} ATR.",
            probability_up, 3,
        )

    # Bearish: uptrend rejected, bearish candle in downtrend
    if (
        candle_direction < 0
        and ema_fast < ema_slow
        and close_pos <= 0.35
        and wick_ratio >= 0.2
        and body_atr >= 0.15
        and range_atr >= 0.3
    ):
        return Decision(
            "SELL",
            f"Wick continuation: rejection wick + bearish close, EMA bearish, "
            f"wick {wick_ratio:.2f} ATR, body {body_atr:.2f} ATR.",
            probability_up, 3,
        )
    return Decision("NO SIGNAL", "No wick continuation pattern.", probability_up, 0)


def viral_rsi_velocity(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """RSI Velocity — trade RSI acceleration in trend direction.

    For 1-3 minute expiry: when RSI is accelerating (velocity positive for BUY,
    negative for SELL) and in the favorable zone, enter with the trend.
    This captures the momentum of momentum — RSI speeding up.
    """
    blocked = _cost_gate(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked, None if features is None else probability_up, 0)
    assert features is not None

    rsi = float(features.get("rsi", 50.0))
    rsi_velocity = float(features.get("rsi_velocity", 0.0))
    ema_fast = float(features.get("ema_fast", 0.0))
    ema_slow = float(features.get("ema_slow", 0.0))
    close = candle.close
    candle_direction = float(features.get("candle_direction", 0.0))

    # RSI accelerating upward in uptrend
    if (
        rsi_velocity > 0.5
        and 50.0 < rsi < 70.0
        and ema_fast > ema_slow
        and candle_direction > 0
    ):
        return Decision(
            "BUY",
            f"RSI velocity: RSI rising at {rsi_velocity:.1f} pts/bar, "
            f"now {rsi:.1f}, uptrend confirmed.",
            probability_up, 2,
        )

    # RSI accelerating downward in downtrend
    if (
        rsi_velocity < -0.5
        and 30.0 < rsi < 50.0
        and ema_fast < ema_slow
        and candle_direction < 0
    ):
        return Decision(
            "SELL",
            f"RSI velocity: RSI falling at {rsi_velocity:.1f} pts/bar, "
            f"now {rsi:.1f}, downtrend confirmed.",
            probability_up, 2,
        )
    return Decision("NO SIGNAL", "No RSI velocity signal.", probability_up, 0)


def quant_range_breakout(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """Range Breakout — trade break of recent range with volume confirmation.

    For 1-3 minute expiry: when price breaks above the 10-bar high (or below
    the 10-bar low) with a strong candle and the range was tight, enter with
    the break. Tight ranges that break are high-probability continuation setups.
    """
    blocked = _cost_gate(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked, None if features is None else probability_up, 0)
    assert features is not None

    close = candle.close
    body_atr = float(features.get("body_atr", 0.0))
    close_pos = float(features.get("close_position", 0.5))
    range_atr = float(features.get("range_atr_ratio", 0.0))
    donchian_high = float(features.get("donchian_high_20", 0.0))
    donchian_low = float(features.get("donchian_low_20", 0.0))
    range_width = float(features.get("donchian_high_20", 0.0)) - float(features.get("donchian_low_20", 0.0))

    # Breakout above 20-bar high with tight range
    if (
        close > donchian_high
        and close_pos >= 0.65
        and body_atr >= 0.2
        and range_width > 0
        and (candle.high - candle.low) / range_width < 0.5
    ):
        return Decision(
            "BUY",
            f"Range breakout: closed above 20-bar high {donchian_high:.5f}, "
            f"tight range, strong candle.",
            probability_up, 2,
        )

    # Breakout below 20-bar low with tight range
    if (
        close < donchian_low
        and close_pos <= 0.35
        and body_atr >= 0.2
        and range_width > 0
        and (candle.high - candle.low) / range_width < 0.5
    ):
        return Decision(
            "SELL",
            f"Range breakout: closed below 20-bar low {donchian_low:.5f}, "
            f"tight range, strong candle.",
            probability_up, 2,
        )
    return Decision("NO SIGNAL", "No range breakout pattern.", probability_up, 0)


def smc_order_flow(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """Order Flow — trade when price action shows accumulation/distribution.

    For 1-3 minute expiry: when the candle closes near its high with a large
    body (accumulation) in an uptrend, or closes near its low with a large
    body (distribution) in a downtrend, enter with the flow.
    """
    blocked = _cost_gate(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked, None if features is None else probability_up, 0)
    assert features is not None

    close_pos = float(features.get("close_position", 0.5))
    body_atr = float(features.get("body_atr", 0.0))
    wick_ratio = float(features.get("wick_ratio", 0.0))
    ema_fast = float(features.get("ema_fast", 0.0))
    ema_slow = float(features.get("ema_slow", 0.0))
    range_atr = float(features.get("range_atr_ratio", 0.0))

    # Accumulation: close near high, large body, small upper wick, uptrend
    if (
        close_pos >= 0.75
        and body_atr >= 0.25
        and wick_ratio <= 0.3
        and range_atr >= 0.3
        and ema_fast > ema_slow
    ):
        return Decision(
            "BUY",
            f"Order flow: accumulation pattern, close at {close_pos:.0%} of range, "
            f"body {body_atr:.2f} ATR, uptrend.",
            probability_up, 3,
        )

    # Distribution: close near low, large body, small lower wick, downtrend
    if (
        close_pos <= 0.25
        and body_atr >= 0.25
        and wick_ratio <= 0.3
        and range_atr >= 0.3
        and ema_fast < ema_slow
    ):
        return Decision(
            "SELL",
            f"Order flow: distribution pattern, close at {close_pos:.0%} of range, "
            f"body {body_atr:.2f} ATR, downtrend.",
            probability_up, 3,
        )
    return Decision("NO SIGNAL", "No order flow pattern.", probability_up, 0)


def viral_momentum_convergence(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """Momentum Convergence — trade when multiple momentum signals align.

    For 1-3 minute expiry: when RSI, stochastic, and 1-bar return all
    point in the same direction with the EMA trend, enter. This is the
    highest-probability short-expiry setup — multiple confirmations.
    """
    blocked = _cost_gate(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked, None if features is None else probability_up, 0)
    assert features is not None

    rsi = float(features.get("rsi", 50.0))
    stoch_k = float(features.get("stoch_k", 50.0))
    momentum = float(features.get("return_1", 0.0))
    momentum_accel = float(features.get("momentum_accel", 0.0))
    ema_fast = float(features.get("ema_fast", 0.0))
    ema_slow = float(features.get("ema_slow", 0.0))
    close = candle.close
    body_atr = float(features.get("body_atr", 0.0))

    # Bullish convergence: RSI > 50, stoch > 50, momentum > 0, accel > 0, EMA bullish
    if (
        rsi > 52.0
        and stoch_k > 52.0
        and momentum > 0
        and momentum_accel > 0
        and ema_fast > ema_slow
        and close > candle.open
        and body_atr >= 0.15
    ):
        return Decision(
            "BUY",
            f"Momentum convergence: RSI {rsi:.1f}, Stoch {stoch_k:.1f}, "
            f"momentum +{momentum:.5f}, accel +{momentum_accel:.5f}, all bullish.",
            probability_up, 3,
        )

    # Bearish convergence: RSI < 48, stoch < 48, momentum < 0, accel < 0, EMA bearish
    if (
        rsi < 48.0
        and stoch_k < 48.0
        and momentum < 0
        and momentum_accel < 0
        and ema_fast < ema_slow
        and close < candle.open
        and body_atr >= 0.15
    ):
        return Decision(
            "SELL",
            f"Momentum convergence: RSI {rsi:.1f}, Stoch {stoch_k:.1f}, "
            f"momentum {momentum:.5f}, accel {momentum_accel:.5f}, all bearish.",
            probability_up, 3,
        )
    return Decision("NO SIGNAL", "No momentum convergence.", probability_up, 0)



def smc_trend_micro(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """Trend Micro — trade micro-trend with EMA alignment and strong candle.

    For 1-3 minute expiry: when the last 3 candles show consistent direction
    and EMA alignment is bullish/bearish, enter with the trend. This is a
    high-frequency micro-trend capture for short expiry.
    """
    blocked = _cost_gate(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked, None if features is None else probability_up, 0)
    assert features is not None

    consecutive_up = float(features.get("consecutive_up", 0.0))
    consecutive_down = float(features.get("consecutive_down", 0.0))
    ema_fast = float(features.get("ema_fast", 0.0))
    ema_slow = float(features.get("ema_slow", 0.0))
    body_atr = float(features.get("body_atr", 0.0))
    close_pos = float(features.get("close_position", 0.5))
    adx = float(features.get("adx", 0.0))

    # Bullish micro-trend
    if (
        consecutive_up >= 2
        and ema_fast > ema_slow
        and body_atr >= 0.15
        and close_pos >= 0.55
        and adx >= 20
    ):
        return Decision(
            "BUY",
            f"Trend micro: {consecutive_up:.0f} consecutive up candles, "
            f"EMA bullish, body {body_atr:.2f} ATR, ADX {adx:.1f}.",
            probability_up, 3,
        )

    # Bearish micro-trend
    if (
        consecutive_down >= 2
        and ema_fast < ema_slow
        and body_atr >= 0.15
        and close_pos <= 0.45
        and adx >= 20
    ):
        return Decision(
            "SELL",
            f"Trend micro: {consecutive_down:.0f} consecutive down candles, "
            f"EMA bearish, body {body_atr:.2f} ATR, ADX {adx:.1f}.",
            probability_up, 3,
        )
    return Decision("NO SIGNAL", "No micro-trend pattern.", probability_up, 0)


def viral_stoch_velocity(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """Stoch Velocity — trade stochastic cross with velocity confirmation.

    For 1-3 minute expiry: when stochastic K crosses above D (bullish) or
    below D (bearish) and K is moving fast, enter with the cross. The
    velocity filter ensures we only trade strong crosses.
    """
    blocked = _cost_gate(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked, None if features is None else probability_up, 0)
    assert features is not None

    stoch_k = float(features.get("stoch_k", 50.0))
    stoch_d = float(features.get("stoch_d", 50.0))
    stoch_k_prev = float(features.get("stoch_k_prev", 50.0))
    ema_fast = float(features.get("ema_fast", 0.0))
    ema_slow = float(features.get("ema_slow", 0.0))
    close = candle.close
    body_atr = float(features.get("body_atr", 0.0))

    # Bullish stochastic cross with velocity
    if (
        stoch_k > stoch_d
        and stoch_k_prev <= stoch_d
        and stoch_k > 45.0
        and ema_fast >= ema_slow * 0.999
        and body_atr >= 0.1
        and close >= candle.open
    ):
        return Decision(
            "BUY",
            f"Stoch velocity: K crossed above D, K now {stoch_k:.1f}, "
            f"moving up from {stoch_k_prev:.1f}.",
            probability_up, 2,
        )

    # Bearish stochastic cross with velocity
    if (
        stoch_k < stoch_d
        and stoch_k_prev >= stoch_d
        and stoch_k < 55.0
        and ema_fast <= ema_slow * 1.001
        and body_atr >= 0.1
        and close <= candle.open
    ):
        return Decision(
            "SELL",
            f"Stoch velocity: K crossed below D, K now {stoch_k:.1f}, "
            f"moving down from {stoch_k_prev:.1f}.",
            probability_up, 2,
        )
    return Decision("NO SIGNAL", "No stochastic velocity signal.", probability_up, 0)


def quant_ema_slope(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """EMA Slope — trade the slope of the fast EMA for short-term momentum.

    For 1-3 minute expiry: when the fast EMA is sloping up (or down) and
    the candle closes in the direction of the slope, enter. This captures
    short-term momentum in the EMA direction.
    """
    blocked = _cost_gate(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked, None if features is None else probability_up, 0)
    assert features is not None

    ema_fast = float(features.get("ema_fast", 0.0))
    ema_fast_prev = float(features.get("ema_fast_prev", 0.0))
    ema_slow = float(features.get("ema_slow", 0.0))
    close = candle.close
    body_atr = float(features.get("body_atr", 0.0))
    close_pos = float(features.get("close_position", 0.5))
    atr_pips = float(features.get("atr_pips", 1.0))

    # EMA slope in pips
    slope = (ema_fast - ema_fast_prev) * 10000

    # Bullish EMA slope
    if (
        slope > 0
        and ema_fast > ema_slow
        and close_pos >= 0.55
        and body_atr >= 0.1
        and slope >= atr_pips * 0.05
    ):
        return Decision(
            "BUY",
            f"EMA slope: fast EMA rising {slope:.1f} pips/bar, "
            f"above slow EMA, close at {close_pos:.0%} of range.",
            probability_up, 2,
        )

    # Bearish EMA slope
    if (
        slope < 0
        and ema_fast < ema_slow
        and close_pos <= 0.45
        and body_atr >= 0.1
        and slope <= -atr_pips * 0.05
    ):
        return Decision(
            "SELL",
            f"EMA slope: fast EMA falling {slope:.1f} pips/bar, "
            f"below slow EMA, close at {close_pos:.0%} of range.",
            probability_up, 2,
        )
    return Decision("NO SIGNAL", "No EMA slope signal.", probability_up, 0)


def smc_sweep_reversal(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """Sweep Reversal — trade liquidity sweep followed by immediate reversal.

    For 1-3 minute expiry: when the candle sweeps a previous low (for bullish)
    or high (for bearish) but closes back inside the range, enter with the
    reversal. This is a high-probability mean-reversion setup after a sweep.
    """
    blocked = _cost_gate(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked, None if features is None else probability_up, 0)
    assert features is not None

    close_pos = float(features.get("close_position", 0.5))
    body_atr = float(features.get("body_atr", 0.0))
    wick_ratio = float(features.get("wick_ratio", 0.0))
    range_atr = float(features.get("range_atr_ratio", 0.0))
    donchian_low = float(features.get("donchian_low_20", 0.0))
    donchian_high = float(features.get("donchian_high_20", 0.0))
    close = candle.close
    candle_direction = float(features.get("candle_direction", 0.0))

    # Bullish sweep reversal: low below donchian low, close back inside
    if (
        candle.low < donchian_low
        and close > donchian_low
        and candle_direction > 0
        and close_pos >= 0.6
        and wick_ratio >= 0.25
        and range_atr >= 0.3
    ):
        return Decision(
            "BUY",
            f"Sweep reversal: swept below 20-bar low {donchian_low:.5f}, "
            f"closed back inside at {close:.5f}, bullish reversal.",
            probability_up, 3,
        )

    # Bearish sweep reversal: high above donchian high, close back inside
    if (
        candle.high > donchian_high
        and close < donchian_high
        and candle_direction < 0
        and close_pos <= 0.4
        and wick_ratio >= 0.25
        and range_atr >= 0.3
    ):
        return Decision(
            "SELL",
            f"Sweep reversal: swept above 20-bar high {donchian_high:.5f}, "
            f"closed back inside at {close:.5f}, bearish reversal.",
            probability_up, 3,
        )
    return Decision("NO SIGNAL", "No sweep reversal pattern.", probability_up, 0)


def viral_close_momentum(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """Close Momentum — trade close price vs previous close for short-term.

    For 1-3 minute expiry: when the close is significantly above (for BUY)
    or below (for SELL) the previous close and the candle has a strong body,
    enter with the momentum. This is a simple but effective short-expiry
    momentum capture.
    """
    blocked = _cost_gate(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked, None if features is None else probability_up, 0)
    assert features is not None

    close = candle.close
    previous_close_val = previous_close
    body_atr = float(features.get("body_atr", 0.0))
    close_pos = float(features.get("close_position", 0.5))
    atr_pips = float(features.get("atr_pips", 1.0))
    momentum = float(features.get("return_1", 0.0))
    momentum_accel = float(features.get("momentum_accel", 0.0))

    # Calculate close difference in pips
    close_diff = (close - previous_close_val) * 10000

    # Bullish close momentum
    if (
        close_diff > 0
        and momentum > 0
        and momentum_accel >= 0
        and body_atr >= 0.2
        and close_pos >= 0.6
        and close_diff >= atr_pips * 0.1
    ):
        return Decision(
            "BUY",
            f"Close momentum: closed +{close_diff:.1f} pips above prev close, "
            f"body {body_atr:.2f} ATR, close at {close_pos:.0%} of range.",
            probability_up, 2,
        )

    # Bearish close momentum
    if (
        close_diff < 0
        and momentum < 0
        and momentum_accel <= 0
        and body_atr >= 0.2
        and close_pos <= 0.4
        and close_diff <= -atr_pips * 0.1
    ):
        return Decision(
            "SELL",
            f"Close momentum: closed {close_diff:.1f} pips below prev close, "
            f"body {body_atr:.2f} ATR, close at {close_pos:.0%} of range.",
            probability_up, 2,
        )
    return Decision("NO SIGNAL", "No close momentum signal.", probability_up, 0)



def smc_trend_follow(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """Trend Follow — pure EMA + price action, no RSI filter for extreme conditions.

    For 1-3 minute expiry: when EMA alignment is strong and the candle closes
    in the trend direction, enter. This works even in extreme RSI conditions
    where most methods abstain. Designed for high-momentum trending markets.
    """
    blocked = _cost_gate(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked, None if features is None else probability_up, 0)
    assert features is not None

    ema_fast = float(features.get("ema_fast", 0.0))
    ema_slow = float(features.get("ema_slow", 0.0))
    ema_fast_prev = float(features.get("ema_fast_prev", 0.0))
    body_atr = float(features.get("body_atr", 0.0))
    close_pos = float(features.get("close_position", 0.5))
    adx = float(features.get("adx", 0.0))
    close = candle.close
    momentum = float(features.get("return_1", 0.0))
    momentum_accel = float(features.get("momentum_accel", 0.0))

    # Calculate EMA separation in pips
    sep = (ema_fast - ema_slow) * 10000

    # Bullish trend follow: EMA separation expanding, strong ADX, bullish candle
    if (
        ema_fast > ema_slow
        and ema_fast > ema_fast_prev
        and sep >= 1.0
        and adx >= 25
        and close > candle.open
        and close_pos >= 0.6
        and body_atr >= 0.15
        and momentum > 0
    ):
        return Decision(
            "BUY",
            f"Trend follow: EMA fast above slow by {sep:.1f} pips and rising, "
            f"ADX {adx:.1f}, bullish close at {close_pos:.0%} of range.",
            probability_up, 3,
        )

    # Bearish trend follow: EMA separation expanding, strong ADX, bearish candle
    if (
        ema_fast < ema_slow
        and ema_fast < ema_fast_prev
        and sep <= -1.0
        and adx >= 25
        and close < candle.open
        and close_pos <= 0.4
        and body_atr >= 0.15
        and momentum < 0
    ):
        return Decision(
            "SELL",
            f"Trend follow: EMA fast below slow by {abs(sep):.1f} pips and falling, "
            f"ADX {adx:.1f}, bearish close at {close_pos:.0%} of range.",
            probability_up, 3,
        )
    return Decision("NO SIGNAL", "No trend follow signal.", probability_up, 0)


def viral_momentum_burst(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """Momentum Burst — trade momentum acceleration burst for short expiry.

    For 1-3 minute expiry: when the 1-bar return accelerates sharply (increase
    in magnitude) and the candle has a strong body, enter with the momentum.
    This captures explosive short-term moves regardless of RSI level.
    """
    blocked = _cost_gate(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked, None if features is None else probability_up, 0)
    assert features is not None

    momentum = float(features.get("return_1", 0.0))
    momentum_prev = float(features.get("return_1_prev", 0.0))
    momentum_accel = float(features.get("momentum_accel", 0.0))
    body_atr = float(features.get("body_atr", 0.0))
    close_pos = float(features.get("close_position", 0.5))
    close = candle.close
    atr_pips = float(features.get("atr_pips", 1.0))

    # Convert momentum to pips
    mom_pips = momentum * 10000
    mom_prev_pips = momentum_prev * 10000
    accel_pips = momentum_accel * 10000

    # Bullish momentum burst: strong positive momentum, accelerating
    if (
        mom_pips > 2.0
        and mom_prev_pips < mom_pips
        and accel_pips > 1.0
        and close > candle.open
        and close_pos >= 0.65
        and body_atr >= 0.2
    ):
        return Decision(
            "BUY",
            f"Momentum burst: +{mom_pips:.1f} pips (prev {mom_prev_pips:.1f}), "
            f"accel +{accel_pips:.1f} pips, strong candle.",
            probability_up, 2,
        )

    # Bearish momentum burst: strong negative momentum, accelerating
    if (
        mom_pips < -2.0
        and mom_prev_pips > mom_pips
        and accel_pips < -1.0
        and close < candle.open
        and close_pos <= 0.35
        and body_atr >= 0.2
    ):
        return Decision(
            "SELL",
            f"Momentum burst: {mom_pips:.1f} pips (prev {mom_prev_pips:.1f}), "
            f"accel {accel_pips:.1f} pips, strong candle.",
            probability_up, 2,
        )
    return Decision("NO SIGNAL", "No momentum burst.", probability_up, 0)


def quant_donchian_slope(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """Donchian Slope — trade direction of Donchian channel for short-term trend.

    For 1-3 minute expiry: when the Donchian channel is rising (or falling)
    and the candle closes near the channel edge, enter with the trend. This
    is a pure price-action trend following method that works in any RSI level.
    """
    blocked = _cost_gate(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked, None if features is None else probability_up, 0)
    assert features is not None

    donchian_high = float(features.get("donchian_high_20", 0.0))
    donchian_low = float(features.get("donchian_low_20", 0.0))
    body_atr = float(features.get("body_atr", 0.0))
    close_pos = float(features.get("close_position", 0.5))
    close = candle.close
    ema_fast = float(features.get("ema_fast", 0.0))
    ema_slow = float(features.get("ema_slow", 0.0))
    range_atr = float(features.get("range_atr_ratio", 0.0))

    # Bullish: close near 20-bar high, EMA bullish, strong body
    if (
        close >= donchian_high * 0.999
        and ema_fast > ema_slow
        and close_pos >= 0.65
        and body_atr >= 0.15
        and range_atr >= 0.2
    ):
        return Decision(
            "BUY",
            f"Donchian slope: closed near 20-bar high {donchian_high:.5f}, "
            f"EMA bullish, body {body_atr:.2f} ATR.",
            probability_up, 2,
        )

    # Bearish: close near 20-bar low, EMA bearish, strong body
    if (
        close <= donchian_low * 1.001
        and ema_fast < ema_slow
        and close_pos <= 0.35
        and body_atr >= 0.15
        and range_atr >= 0.2
    ):
        return Decision(
            "SELL",
            f"Donchian slope: closed near 20-bar low {donchian_low:.5f}, "
            f"EMA bearish, body {body_atr:.2f} ATR.",
            probability_up, 2,
        )
    return Decision("NO SIGNAL", "No Donchian slope signal.", probability_up, 0)


def smc_candle_direction(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """Candle Direction — trade pure candle direction with EMA confirmation.

    For 1-3 minute expiry: when the candle is strongly directional (large body,
    closes at extreme of range) and EMA confirms the direction, enter. This is
    the simplest and most effective short-expiry signal — no oscillator filter.
    """
    blocked = _cost_gate(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked, None if features is None else probability_up, 0)
    assert features is not None

    candle_direction = float(features.get("candle_direction", 0.0))
    body_atr = float(features.get("body_atr", 0.0))
    close_pos = float(features.get("close_position", 0.5))
    ema_fast = float(features.get("ema_fast", 0.0))
    ema_slow = float(features.get("ema_slow", 0.0))
    close = candle.close
    prev_close_val = previous_close
    close_diff = (close - prev_close_val) * 10000

    # Bullish: strong bullish candle, EMA bullish, closed above prev close
    if (
        candle_direction > 0
        and body_atr >= 0.2
        and close_pos >= 0.7
        and ema_fast > ema_slow
        and close_diff > 0
    ):
        return Decision(
            "BUY",
            f"Candle direction: bullish candle, body {body_atr:.2f} ATR, "
            f"close at {close_pos:.0%} of range, EMA bullish.",
            probability_up, 2,
        )

    # Bearish: strong bearish candle, EMA bearish, closed below prev close
    if (
        candle_direction < 0
        and body_atr >= 0.2
        and close_pos <= 0.3
        and ema_fast < ema_slow
        and close_diff < 0
    ):
        return Decision(
            "SELL",
            f"Candle direction: bearish candle, body {body_atr:.2f} ATR, "
            f"close at {close_pos:.0%} of range, EMA bearish.",
            probability_up, 2,
        )
    return Decision("NO SIGNAL", "No candle direction signal.", probability_up, 0)


def viral_price_action(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """Price Action — trade pure price action with consecutive candle confirmation.

    For 1-3 minute expiry: when multiple consecutive candles move in the same
    direction with strong bodies, enter with the trend. This is a high-probability
    short-expiry method that works in any market condition — no oscillator filter.
    """
    blocked = _cost_gate(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked, None if features is None else probability_up, 0)
    assert features is not None

    consecutive_up = float(features.get("consecutive_up", 0.0))
    consecutive_down = float(features.get("consecutive_down", 0.0))
    body_atr = float(features.get("body_atr", 0.0))
    close_pos = float(features.get("close_position", 0.5))
    ema_fast = float(features.get("ema_fast", 0.0))
    ema_slow = float(features.get("ema_slow", 0.0))
    close = candle.close
    momentum = float(features.get("return_1", 0.0))

    # Bullish: 3+ consecutive up candles, EMA bullish, strong body
    if (
        consecutive_up >= 3
        and ema_fast > ema_slow
        and body_atr >= 0.15
        and close_pos >= 0.55
        and close > candle.open
        and momentum > 0
    ):
        return Decision(
            "BUY",
            f"Price action: {consecutive_up:.0f} consecutive up candles, "
            f"EMA bullish, body {body_atr:.2f} ATR, momentum positive.",
            probability_up, 3,
        )

    # Bearish: 3+ consecutive down candles, EMA bearish, strong body
    if (
        consecutive_down >= 3
        and ema_fast < ema_slow
        and body_atr >= 0.15
        and close_pos <= 0.45
        and close < candle.open
        and momentum < 0
    ):
        return Decision(
            "SELL",
            f"Price action: {consecutive_down:.0f} consecutive down candles, "
            f"EMA bearish, body {body_atr:.2f} ATR, momentum negative.",
            probability_up, 3,
        )
    return Decision("NO SIGNAL", "No price action signal.", probability_up, 0)



def smc_trend_streak(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """Trend Streak — trade consecutive candle streak with EMA alignment.

    For 1-3 minute expiry: when 4+ consecutive candles move in the same
    direction and EMA confirms the trend, enter. This is a pure momentum
    method that works in extreme conditions where oscillators are saturated.
    """
    blocked = _cost_gate(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked, None if features is None else probability_up, 0)
    assert features is not None

    consecutive_up = float(features.get("consecutive_up", 0.0))
    consecutive_down = float(features.get("consecutive_down", 0.0))
    ema_fast = float(features.get("ema_fast", 0.0))
    ema_slow = float(features.get("ema_slow", 0.0))
    body_atr = float(features.get("body_atr", 0.0))
    close = candle.close
    momentum = float(features.get("return_1", 0.0))
    close_pos = float(features.get("close_position", 0.5))

    # Bullish streak: 4+ consecutive up, EMA bullish
    if (
        consecutive_up >= 4
        and ema_fast > ema_slow
        and body_atr >= 0.1
        and close > candle.open
        and momentum > 0
    ):
        return Decision(
            "BUY",
            f"Trend streak: {consecutive_up:.0f} consecutive up candles, "
            f"EMA bullish, body {body_atr:.2f} ATR, momentum positive.",
            probability_up, 3,
        )

    # Bearish streak: 4+ consecutive down, EMA bearish
    if (
        consecutive_down >= 4
        and ema_fast < ema_slow
        and body_atr >= 0.1
        and close < candle.open
        and momentum < 0
    ):
        return Decision(
            "SELL",
            f"Trend streak: {consecutive_down:.0f} consecutive down candles, "
            f"EMA bearish, body {body_atr:.2f} ATR, momentum negative.",
            probability_up, 3,
        )
    return Decision("NO SIGNAL", "No trend streak signal.", probability_up, 0)


def viral_rsi_extreme_follow(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """RSI Extreme Follow — trade when RSI is at extreme but trend continues.

    For 1-3 minute expiry: when RSI is extremely overbought (>85) or oversold
    (<15) but the price is still moving in the trend direction with strong
    momentum, enter with the trend. In strong trends, RSI can stay at extreme
    levels for extended periods — this method captures those continuations.
    """
    blocked = _cost_gate(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked, None if features is None else probability_up, 0)
    assert features is not None

    rsi = float(features.get("rsi", 50.0))
    rsi_velocity = float(features.get("rsi_velocity", 0.0))
    ema_fast = float(features.get("ema_fast", 0.0))
    ema_slow = float(features.get("ema_slow", 0.0))
    body_atr = float(features.get("body_atr", 0.0))
    close = candle.close
    close_pos = float(features.get("close_position", 0.5))
    momentum = float(features.get("return_1", 0.0))
    adx = float(features.get("adx", 0.0))

    # Extreme overbought, but still bullish momentum with strong trend
    if (
        rsi > 85.0
        and rsi_velocity >= -2.0
        and ema_fast > ema_slow
        and close > candle.open
        and close_pos >= 0.6
        and body_atr >= 0.1
        and momentum > 0
        and adx >= 25
    ):
        return Decision(
            "BUY",
            f"RSI extreme follow: RSI at {rsi:.1f} but momentum still bullish, "
            f"ADX {adx:.1f}, body {body_atr:.2f} ATR.",
            probability_up, 3,
        )

    # Extreme oversold, but still bearish momentum with strong trend
    if (
        rsi < 15.0
        and rsi_velocity <= 2.0
        and ema_fast < ema_slow
        and close < candle.open
        and close_pos <= 0.4
        and body_atr >= 0.1
        and momentum < 0
        and adx >= 25
    ):
        return Decision(
            "SELL",
            f"RSI extreme follow: RSI at {rsi:.1f} but momentum still bearish, "
            f"ADX {adx:.1f}, body {body_atr:.2f} ATR.",
            probability_up, 3,
        )
    return Decision("NO SIGNAL", "No RSI extreme follow signal.", probability_up, 0)


def quant_breakout_volume(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """Breakout Volume — trade breakout with volume point of control confirmation.

    For 1-3 minute expiry: when price breaks above the VWAP or Donchian range
    and is above the volume POC, enter with the break. Volume POC acts as
    resistance that becomes support once broken.
    """
    blocked = _cost_gate(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked, None if features is None else probability_up, 0)
    assert features is not None

    vwap = float(features.get("vwap", 0.0))
    close = candle.close
    body_atr = float(features.get("body_atr", 0.0))
    close_pos = float(features.get("close_position", 0.5))
    ema_fast = float(features.get("ema_fast", 0.0))
    ema_slow = float(features.get("ema_slow", 0.0))
    donchian_high = float(features.get("donchian_high_20", 0.0))
    range_atr = float(features.get("range_atr_ratio", 0.0))

    # Break above VWAP and Donchian high, above POC
    if (
        close > vwap
        and close > donchian_high
        and ema_fast > ema_slow
        and close_pos >= 0.65
        and body_atr >= 0.15
        and range_atr >= 0.2
    ):
        return Decision(
            "BUY",
            f"Breakout volume: closed above VWAP and 20-bar high, "
            f"EMA bullish, body {body_atr:.2f} ATR.",
            probability_up, 2,
        )

    # Break below VWAP and Donchian low
    if (
        close < vwap
        and close < float(features.get("donchian_low_20", 0.0))
        and ema_fast < ema_slow
        and close_pos <= 0.35
        and body_atr >= 0.15
        and range_atr >= 0.2
    ):
        return Decision(
            "SELL",
            f"Breakout volume: closed below VWAP and 20-bar low, "
            f"EMA bearish, body {body_atr:.2f} ATR.",
            probability_up, 2,
        )
    return Decision("NO SIGNAL", "No breakout volume signal.", probability_up, 0)


def smc_order_flow_imbalance(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """Order Flow Imbalance — trade when close position shows imbalance.

    For 1-3 minute expiry: when the close is in the extreme of the range
    (top 15% for bullish, bottom 15% for bearish) with a strong body and
    EMA confirmation, enter with the flow. The close position relative to
    the range is a powerful short-term indicator of order flow.
    """
    blocked = _cost_gate(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked, None if features is None else probability_up, 0)
    assert features is not None

    close_pos = float(features.get("close_position", 0.5))
    body_atr = float(features.get("body_atr", 0.0))
    ema_fast = float(features.get("ema_fast", 0.0))
    ema_slow = float(features.get("ema_slow", 0.0))
    close = candle.close
    wick_ratio = float(features.get("wick_ratio", 0.0))
    range_atr = float(features.get("range_atr_ratio", 0.0))
    consecutive_up = float(features.get("consecutive_up", 0.0))

    # Bullish order flow: close in top 15% of range, EMA bullish, strong body
    if (
        close_pos >= 0.85
        and ema_fast > ema_slow
        and body_atr >= 0.15
        and range_atr >= 0.2
        and close > candle.open
    ):
        return Decision(
            "BUY",
            f"Order flow imbalance: closed at {close_pos:.0%} of range, "
            f"EMA bullish, body {body_atr:.2f} ATR, {consecutive_up:.0f} consecutive up.",
            probability_up, 3,
        )

    # Bearish order flow: close in bottom 15% of range, EMA bearish, strong body
    if (
        close_pos <= 0.15
        and ema_fast < ema_slow
        and body_atr >= 0.15
        and range_atr >= 0.2
        and close < candle.open
    ):
        return Decision(
            "SELL",
            f"Order flow imbalance: closed at {close_pos:.0%} of range, "
            f"EMA bearish, body {body_atr:.2f} ATR.",
            probability_up, 3,
        )
    return Decision("NO SIGNAL", "No order flow imbalance signal.", probability_up, 0)


def viral_price_velocity(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """Price Velocity — trade the velocity of price change for short expiry.

    For 1-3 minute expiry: when the price velocity (momentum) is high and
    accelerating in the trend direction, enter. This captures explosive
    short-term moves that are ideal for 1-3 minute expiry.
    """
    blocked = _cost_gate(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked, None if features is None else probability_up, 0)
    assert features is not None

    momentum = float(features.get("return_1", 0.0))
    momentum_prev = float(features.get("return_1_prev", 0.0))
    momentum_accel = float(features.get("momentum_accel", 0.0))
    ema_fast = float(features.get("ema_fast", 0.0))
    ema_slow = float(features.get("ema_slow", 0.0))
    body_atr = float(features.get("body_atr", 0.0))
    close = candle.close
    close_pos = float(features.get("close_position", 0.5))
    atr_pips = float(features.get("atr_pips", 1.0))

    # Convert to pips
    mom_pips = momentum * 10000
    accel_pips = momentum_accel * 10000

    # Bullish velocity: high positive momentum, accelerating, EMA bullish
    if (
        mom_pips > atr_pips * 0.2
        and accel_pips > 0
        and ema_fast > ema_slow
        and close > candle.open
        and close_pos >= 0.6
        and body_atr >= 0.15
    ):
        return Decision(
            "BUY",
            f"Price velocity: +{mom_pips:.1f} pips ({mom_pips/atr_pips:.1f}x ATR), "
            f"accelerating, EMA bullish.",
            probability_up, 2,
        )

    # Bearish velocity: high negative momentum, accelerating, EMA bearish
    if (
        mom_pips < -atr_pips * 0.2
        and accel_pips < 0
        and ema_fast < ema_slow
        and close < candle.open
        and close_pos <= 0.4
        and body_atr >= 0.15
    ):
        return Decision(
            "SELL",
            f"Price velocity: {mom_pips:.1f} pips ({abs(mom_pips)/atr_pips:.1f}x ATR), "
            f"accelerating, EMA bearish.",
            probability_up, 2,
        )
    return Decision("NO SIGNAL", "No price velocity signal.", probability_up, 0)



def smc_micro_trend(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """Micro Trend — trade the last 3-candle micro-trend with EMA confirmation.

    For 1-3 minute expiry: when the last 3 candles show consistent direction
    (all bullish or all bearish) and EMA confirms, enter with the micro-trend.
    This is a very short-term trend capture that fires frequently.
    """
    blocked = _cost_gate(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked, None if features is None else probability_up, 0)
    assert features is not None

    consecutive_up = float(features.get("consecutive_up", 0.0))
    consecutive_down = float(features.get("consecutive_down", 0.0))
    ema_fast = float(features.get("ema_fast", 0.0))
    ema_slow = float(features.get("ema_slow", 0.0))
    body_atr = float(features.get("body_atr", 0.0))
    close = candle.close
    close_pos = float(features.get("close_position", 0.5))
    momentum = float(features.get("return_1", 0.0))

    # Bullish micro-trend: 2+ consecutive up, EMA bullish
    if (
        consecutive_up >= 2
        and ema_fast > ema_slow
        and body_atr >= 0.08
        and close > candle.open
        and close_pos >= 0.55
        and momentum > 0
    ):
        return Decision(
            "BUY",
            f"Micro trend: {consecutive_up:.0f} consecutive up candles, "
            f"EMA bullish, body {body_atr:.2f} ATR.",
            probability_up, 2,
        )

    # Bearish micro-trend: 2+ consecutive down, EMA bearish
    if (
        consecutive_down >= 2
        and ema_fast < ema_slow
        and body_atr >= 0.08
        and close < candle.open
        and close_pos <= 0.45
        and momentum < 0
    ):
        return Decision(
            "SELL",
            f"Micro trend: {consecutive_down:.0f} consecutive down candles, "
            f"EMA bearish, body {body_atr:.2f} ATR.",
            probability_up, 2,
        )
    return Decision("NO SIGNAL", "No micro trend signal.", probability_up, 0)


def viral_ema_slope_momentum(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """EMA Slope Momentum — trade EMA slope with momentum confirmation.

    For 1-3 minute expiry: when the fast EMA is sloping up (or down) and
    the momentum is also in the same direction, enter. This combines the
    trend indicator (EMA slope) with momentum (return) for a double confirmation.
    """
    blocked = _cost_gate(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked, None if features is None else probability_up, 0)
    assert features is not None

    ema_fast = float(features.get("ema_fast", 0.0))
    ema_fast_prev = float(features.get("ema_fast_prev", 0.0))
    ema_slow = float(features.get("ema_slow", 0.0))
    momentum = float(features.get("return_1", 0.0))
    body_atr = float(features.get("body_atr", 0.0))
    close = candle.close
    close_pos = float(features.get("close_position", 0.5))
    atr_pips = float(features.get("atr_pips", 1.0))

    slope_pips = (ema_fast - ema_fast_prev) * 10000
    mom_pips = momentum * 10000

    # Bullish: EMA rising, momentum positive, EMA above slow
    if (
        slope_pips > 0
        and mom_pips > 0
        and ema_fast > ema_slow
        and close > candle.open
        and close_pos >= 0.55
        and body_atr >= 0.08
    ):
        return Decision(
            "BUY",
            f"EMA slope momentum: EMA rising {slope_pips:.2f} pips, "
            f"momentum +{mom_pips:.2f} pips, double confirmation.",
            probability_up, 2,
        )

    # Bearish: EMA falling, momentum negative, EMA below slow
    if (
        slope_pips < 0
        and mom_pips < 0
        and ema_fast < ema_slow
        and close < candle.open
        and close_pos <= 0.45
        and body_atr >= 0.08
    ):
        return Decision(
            "SELL",
            f"EMA slope momentum: EMA falling {slope_pips:.2f} pips, "
            f"momentum {mom_pips:.2f} pips, double confirmation.",
            probability_up, 2,
        )
    return Decision("NO SIGNAL", "No EMA slope momentum signal.", probability_up, 0)


def quant_rsi_momentum(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """RSI Momentum — trade when RSI direction aligns with price momentum.

    For 1-3 minute expiry: when RSI is rising (or falling) and the price
    momentum is also in the same direction, enter. This is a simple but
    effective method that combines the RSI direction with price momentum.
    """
    blocked = _cost_gate(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked, None if features is None else probability_up, 0)
    assert features is not None

    rsi = float(features.get("rsi", 50.0))
    rsi_velocity = float(features.get("rsi_velocity", 0.0))
    momentum = float(features.get("return_1", 0.0))
    body_atr = float(features.get("body_atr", 0.0))
    close = candle.close
    close_pos = float(features.get("close_position", 0.5))
    ema_fast = float(features.get("ema_fast", 0.0))
    ema_slow = float(features.get("ema_slow", 0.0))

    # Bullish: RSI rising, momentum positive
    if (
        rsi_velocity > 0
        and momentum > 0
        and close > candle.open
        and close_pos >= 0.55
        and body_atr >= 0.08
        and ema_fast >= ema_slow * 0.998
    ):
        return Decision(
            "BUY",
            f"RSI momentum: RSI rising at {rsi_velocity:.1f} pts/bar (now {rsi:.1f}), "
            f"momentum positive, bullish close.",
            probability_up, 2,
        )

    # Bearish: RSI falling, momentum negative
    if (
        rsi_velocity < 0
        and momentum < 0
        and close < candle.open
        and close_pos <= 0.45
        and body_atr >= 0.08
        and ema_fast <= ema_slow * 1.002
    ):
        return Decision(
            "SELL",
            f"RSI momentum: RSI falling at {rsi_velocity:.1f} pts/bar (now {rsi:.1f}), "
            f"momentum negative, bearish close.",
            probability_up, 2,
        )
    return Decision("NO SIGNAL", "No RSI momentum signal.", probability_up, 0)


def smc_candle_cluster_trend(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """Candle Cluster Trend — trade when multiple candles cluster in one direction.

    For 1-3 minute expiry: when the last 3+ candles all close in the upper
    (bullish) or lower (bearish) half of their ranges, enter with the cluster
    direction. This is a powerful short-term indicator of order flow imbalance.
    """
    blocked = _cost_gate(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked, None if features is None else probability_up, 0)
    assert features is not None

    close_pos = float(features.get("close_position", 0.5))
    consecutive_up = float(features.get("consecutive_up", 0.0))
    consecutive_down = float(features.get("consecutive_down", 0.0))
    ema_fast = float(features.get("ema_fast", 0.0))
    ema_slow = float(features.get("ema_slow", 0.0))
    body_atr = float(features.get("body_atr", 0.0))
    close = candle.close
    momentum = float(features.get("return_1", 0.0))

    # Bullish cluster: multiple up candles, closes in upper half, EMA bullish
    if (
        consecutive_up >= 2
        and close_pos >= 0.6
        and ema_fast > ema_slow
        and body_atr >= 0.08
        and close > candle.open
        and momentum > 0
    ):
        return Decision(
            "BUY",
            f"Candle cluster: {consecutive_up:.0f} consecutive up, "
            f"close at {close_pos:.0%} of range, EMA bullish.",
            probability_up, 2,
        )

    # Bearish cluster: multiple down candles, closes in lower half, EMA bearish
    if (
        consecutive_down >= 2
        and close_pos <= 0.4
        and ema_fast < ema_slow
        and body_atr >= 0.08
        and close < candle.open
        and momentum < 0
    ):
        return Decision(
            "SELL",
            f"Candle cluster: {consecutive_down:.0f} consecutive down, "
            f"close at {close_pos:.0%} of range, EMA bearish.",
            probability_up, 2,
        )
    return Decision("NO SIGNAL", "No candle cluster signal.", probability_up, 0)


def viral_donchian_trend(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """Donchian Trend — trade when price is near Donchian channel edge with trend.

    For 1-3 minute expiry: when the close is near the 20-bar high (or low)
    and EMA confirms the trend direction, enter with the trend. This is a
    classic trend-following method optimized for short expiry.
    """
    blocked = _cost_gate(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked, None if features is None else probability_up, 0)
    assert features is not None

    donchian_high = float(features.get("donchian_high_20", 0.0))
    donchian_low = float(features.get("donchian_low_20", 0.0))
    close = candle.close
    ema_fast = float(features.get("ema_fast", 0.0))
    ema_slow = float(features.get("ema_slow", 0.0))
    body_atr = float(features.get("body_atr", 0.0))
    close_pos = float(features.get("close_position", 0.5))
    momentum = float(features.get("return_1", 0.0))

    # Bullish: close near 20-bar high, EMA bullish, positive momentum
    if (
        close >= donchian_high * 0.998
        and ema_fast > ema_slow
        and momentum > 0
        and close_pos >= 0.6
        and body_atr >= 0.08
    ):
        return Decision(
            "BUY",
            f"Donchian trend: closed near 20-bar high {donchian_high:.5f}, "
            f"EMA bullish, momentum positive.",
            probability_up, 2,
        )

    # Bearish: close near 20-bar low, EMA bearish, negative momentum
    if (
        close <= donchian_low * 1.002
        and ema_fast < ema_slow
        and momentum < 0
        and close_pos <= 0.4
        and body_atr >= 0.08
    ):
        return Decision(
            "SELL",
            f"Donchian trend: closed near 20-bar low {donchian_low:.5f}, "
            f"EMA bearish, momentum negative.",
            probability_up, 2,
        )
    return Decision("NO SIGNAL", "No Donchian trend signal.", probability_up, 0)



def smc_close_extreme(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """Close Extreme — trade when close is at extreme of range regardless of body.

    For 1-3 minute expiry: when the close is at the very top (or bottom) of
    the candle's range, enter with the direction. This fires even on doji
    candles where the body is zero but the close is at an extreme. The close
    position is a strong short-term indicator of order flow.
    """
    blocked = _cost_gate(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked, None if features is None else probability_up, 0)
    assert features is not None

    close_pos = float(features.get("close_position", 0.5))
    ema_fast = float(features.get("ema_fast", 0.0))
    ema_slow = float(features.get("ema_slow", 0.0))
    consecutive_up = float(features.get("consecutive_up", 0.0))
    close = candle.close
    momentum = float(features.get("return_1", 0.0))
    atr_pips = float(features.get("atr_pips", 1.0))

    # Bullish: close at very top of range, EMA bullish, or prior up trend
    if (
        close_pos >= 0.95
        and (ema_fast > ema_slow or consecutive_up >= 2)
        and close >= candle.open
    ):
        return Decision(
            "BUY",
            f"Close extreme: closed at {close_pos:.0%} of range, "
            f"EMA fast above slow, prior {consecutive_up:.0f} consecutive up candles.",
            probability_up, 2,
        )

    # Bearish: close at very bottom of range, EMA bearish, or prior down trend
    if (
        close_pos <= 0.05
        and (ema_fast < ema_slow or float(features.get("consecutive_down", 0.0)) >= 2)
        and close <= candle.open
    ):
        return Decision(
            "SELL",
            f"Close extreme: closed at {close_pos:.0%} of range, "
            f"EMA fast below slow, prior down trend.",
            probability_up, 2,
        )
    return Decision("NO SIGNAL", "No close extreme signal.", probability_up, 0)


def viral_momentum_continuation(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """Momentum Continuation — trade based on prior momentum continuation.

    For 1-3 minute expiry: when the prior candles showed strong momentum in
    one direction and the current candle doesn't reverse it, enter with the
    momentum. This fires even on doji candles by looking at the momentum of
    the preceding candles.
    """
    blocked = _cost_gate(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked, None if features is None else probability_up, 0)
    assert features is not None

    consecutive_up = float(features.get("consecutive_up", 0.0))
    consecutive_down = float(features.get("consecutive_down", 0.0))
    ema_fast = float(features.get("ema_fast", 0.0))
    ema_slow = float(features.get("ema_slow", 0.0))
    close = candle.close
    close_pos = float(features.get("close_position", 0.5))
    momentum = float(features.get("return_1", 0.0))
    momentum_prev = float(features.get("return_1_prev", 0.0))
    atr_pips = float(features.get("atr_pips", 1.0))

    # Bullish: prior momentum was positive, EMA bullish, close not below open
    mom_prev_pips = momentum_prev * 10000
    if (
        consecutive_up >= 3
        and ema_fast > ema_slow
        and close >= candle.open
        and close_pos >= 0.5
        and mom_prev_pips > 0
    ):
        return Decision(
            "BUY",
            f"Momentum continuation: {consecutive_up:.0f} consecutive up candles, "
            f"prior momentum +{mom_prev_pips:.1f} pips, EMA bullish.",
            probability_up, 2,
        )

    # Bearish: prior momentum was negative, EMA bearish, close not above open
    if (
        consecutive_down >= 3
        and ema_fast < ema_slow
        and close <= candle.open
        and close_pos <= 0.5
        and mom_prev_pips < 0
    ):
        return Decision(
            "SELL",
            f"Momentum continuation: {consecutive_down:.0f} consecutive down candles, "
            f"prior momentum {mom_prev_pips:.1f} pips, EMA bearish.",
            probability_up, 2,
        )
    return Decision("NO SIGNAL", "No momentum continuation signal.", probability_up, 0)


def quant_trend_strength(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """Trend Strength — trade when trend strength is high and price is near trend direction.

    For 1-3 minute expiry: when ADX is high (strong trend) and the price is
    above VWAP (bullish trend) or below VWAP (bearish trend), enter with the
    trend. This works even on doji candles because it looks at the broader
    trend context, not the individual candle.
    """
    blocked = _cost_gate(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked, None if features is None else probability_up, 0)
    assert features is not None

    adx = float(features.get("adx", 0.0))
    vwap = float(features.get("vwap", 0.0))
    ema_fast = float(features.get("ema_fast", 0.0))
    ema_slow = float(features.get("ema_slow", 0.0))
    close = candle.close
    close_pos = float(features.get("close_position", 0.5))
    atr_pips = float(features.get("atr_pips", 1.0))

    # Bullish: strong trend (ADX high), price above VWAP, EMA bullish
    if (
        adx >= 30
        and close > vwap
        and ema_fast > ema_slow
        and close >= candle.open
        and close_pos >= 0.5
    ):
        return Decision(
            "BUY",
            f"Trend strength: ADX {adx:.1f}, price above VWAP, "
            f"EMA bullish, close at {close_pos:.0%} of range.",
            probability_up, 2,
        )

    # Bearish: strong trend (ADX high), price below VWAP, EMA bearish
    if (
        adx >= 30
        and close < vwap
        and ema_fast < ema_slow
        and close <= candle.open
        and close_pos <= 0.5
    ):
        return Decision(
            "SELL",
            f"Trend strength: ADX {adx:.1f}, price below VWAP, "
            f"EMA bearish, close at {close_pos:.0%} of range.",
            probability_up, 2,
        )
    return Decision("NO SIGNAL", "No trend strength signal.", probability_up, 0)


def smc_doji_trend(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """Doji Trend — trade doji candles in the direction of the prevailing trend.

    For 1-3 minute expiry: when the candle is a doji (small body, close near
    open) but the prevailing trend is strong (EMA alignment, high ADX), enter
    with the trend. A doji in a strong trend is a pause before continuation,
    not a reversal.
    """
    blocked = _cost_gate(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked, None if features is None else probability_up, 0)
    assert features is not None

    body_atr = float(features.get("body_atr", 0.0))
    close = candle.close
    open_price = candle.open
    ema_fast = float(features.get("ema_fast", 0.0))
    ema_slow = float(features.get("ema_slow", 0.0))
    adx = float(features.get("adx", 0.0))
    close_pos = float(features.get("close_position", 0.5))
    consecutive_up = float(features.get("consecutive_up", 0.0))
    consecutive_down = float(features.get("consecutive_down", 0.0))

    # Doji: small body relative to range
    is_doji = body_atr < 0.1 and abs(close - open_price) < 1e-10

    # Bullish doji in uptrend
    if (
        is_doji
        and ema_fast > ema_slow
        and adx >= 25
        and close_pos >= 0.7
        and consecutive_up >= 2
    ):
        return Decision(
            "BUY",
            f"Doji trend: doji in uptrend (ADX {adx:.1f}, {consecutive_up:.0f} consecutive up), "
            f"pause before continuation.",
            probability_up, 2,
        )

    # Bearish doji in downtrend
    if (
        is_doji
        and ema_fast < ema_slow
        and adx >= 25
        and close_pos <= 0.3
        and consecutive_down >= 2
    ):
        return Decision(
            "SELL",
            f"Doji trend: doji in downtrend (ADX {adx:.1f}, {consecutive_down:.0f} consecutive down), "
            f"pause before continuation.",
            probability_up, 2,
        )
    return Decision("NO SIGNAL", "No doji trend signal.", probability_up, 0)


def viral_range_position(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """Range Position — trade based on where the close is within the recent range.

    For 1-3 minute expiry: when the close is in the upper (or lower) portion
    of the 20-bar Donchian range and EMA confirms the direction, enter. This
    works on any candle type including dojis because it looks at the position
    within the broader range, not the individual candle direction.
    """
    blocked = _cost_gate(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked, None if features is None else probability_up, 0)
    assert features is not None

    close = candle.close
    donchian_high = float(features.get("donchian_high_20", 0.0))
    donchian_low = float(features.get("donchian_low_20", 0.0))
    ema_fast = float(features.get("ema_fast", 0.0))
    ema_slow = float(features.get("ema_slow", 0.0))
    close_pos = float(features.get("close_position", 0.5))
    range_width = donchian_high - donchian_low

    if range_width <= 0:
        return Decision("NO SIGNAL", "Zero range width.", probability_up, 0)

    # Position within range (0 = low, 1 = high)
    range_pos = (close - donchian_low) / range_width

    # Bullish: close in upper 30% of range, EMA bullish
    if (
        range_pos >= 0.7
        and ema_fast > ema_slow
        and close_pos >= 0.5
    ):
        return Decision(
            "BUY",
            f"Range position: close at {range_pos:.0%} of 20-bar range, "
            f"EMA bullish, close at {close_pos:.0%} of candle range.",
            probability_up, 2,
        )

    # Bearish: close in lower 30% of range, EMA bearish
    if (
        range_pos <= 0.3
        and ema_fast < ema_slow
        and close_pos <= 0.5
    ):
        return Decision(
            "SELL",
            f"Range position: close at {range_pos:.0%} of 20-bar range, "
            f"EMA bearish, close at {close_pos:.0%} of candle range.",
            probability_up, 2,
        )
    return Decision("NO SIGNAL", "No range position signal.", probability_up, 0)



def smc_vwap_position(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """VWAP Position — trade based on price position relative to VWAP.

    For 1-3 minute expiry: when the close is above VWAP (bullish) or below
    VWAP (bearish) and the close is at an extreme of the candle range, enter.
    This works on doji candles because it uses VWAP as the reference, not
    the candle body.
    """
    blocked = _cost_gate(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked, None if features is None else probability_up, 0)
    assert features is not None

    close = candle.close
    vwap = float(features.get("vwap", 0.0))
    close_pos = float(features.get("close_position", 0.5))
    adx = float(features.get("adx", 0.0))
    atr_pips = float(features.get("atr_pips", 1.0))

    # Bullish: close above VWAP and at top of range
    if (
        close > vwap
        and close_pos >= 0.8
        and adx >= 20
    ):
        return Decision(
            "BUY",
            f"VWAP position: closed above VWAP at {close:.5f} (VWAP {vwap:.5f}), "
            f"close at {close_pos:.0%} of range, ADX {adx:.1f}.",
            probability_up, 2,
        )

    # Bearish: close below VWAP and at bottom of range
    if (
        close < vwap
        and close_pos <= 0.2
        and adx >= 20
    ):
        return Decision(
            "SELL",
            f"VWAP position: closed below VWAP at {close:.5f} (VWAP {vwap:.5f}), "
            f"close at {close_pos:.0%} of range, ADX {adx:.1f}.",
            probability_up, 2,
        )
    return Decision("NO SIGNAL", "No VWAP position signal.", probability_up, 0)


def viral_rsi_zone(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIP_S if False else charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """RSI Zone — trade when RSI is in the middle zone with momentum.

    For 1-3 minute expiry: when RSI is in the 40-60 zone (neutral) and the
    momentum is in the trend direction, enter. This catches transition
    points where RSI is neutral but momentum is starting to shift.
    """
    blocked = _cost_gate(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked, None if features is None else probability_up, 0)
    assert features is not None

    rsi = float(features.get("rsi", 50.0))
    rsi_velocity = float(features.get("rsi_velocity", 0.0))
    momentum = float(features.get("return_1", 0.0))
    ema_fast = float(features.get("ema_fast", 0.0))
    ema_slow = float(features.get("ema_slow", 0.0))
    close = candle.close
    close_pos = float(features.get("close_position", 0.5))
    body_atr = float(features.get("body_atr", 0.0))

    # Bullish: RSI in 40-55 zone, rising, momentum positive, EMA bullish
    if (
        40.0 < rsi < 55.0
        and rsi_velocity > 0
        and momentum > 0
        and ema_fast > ema_slow
        and close > candle.open
        and close_pos >= 0.55
        and body_atr >= 0.08
    ):
        return Decision(
            "BUY",
            f"RSI zone: RSI at {rsi:.1f} rising (zone 40-55), "
            f"momentum positive, EMA bullish.",
            probability_up, 2,
        )

    # Bearish: RSI in 45-60 zone, falling, momentum negative, EMA bearish
    if (
        45.0 < rsi < 60.0
        and rsi_velocity < 0
        and momentum < 0
        and ema_fast < ema_slow
        and close < candle.open
        and close_pos <= 0.45
        and body_atr >= 0.08
    ):
        return Decision(
            "SELL",
            f"RSI zone: RSI at {rsi:.1f} falling (zone 45-60), "
            f"momentum negative, EMA bearish.",
            probability_up, 2,
        )
    return Decision("NO SIGNAL", "No RSI zone signal.", probability_up, 0)


def quant_price_action_swing(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """Price Action Swing — trade swing highs and lows in the trend direction.

    For 1-3 minute expiry: when the candle makes a swing high (or low) that
    is above (or below) the previous swing and EMA confirms the trend, enter.
    This is a pure price action method that works in any RSI condition.
    """
    blocked = _cost_gate(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked, None if features is None else probability_up, 0)
    assert features is not None

    close = candle.close
    high = candle.high
    low = candle.low
    ema_fast = float(features.get("ema_fast", 0.0))
    ema_slow = float(features.get("ema_slow", 0.0))
    body_atr = float(features.get("body_atr", 0.0))
    close_pos = float(features.get("close_position", 0.5))
    donchian_high = float(features.get("donchian_high_20", 0.0))
    donchian_low = float(features.get("donchian_low_20", 0.0))

    # Bullish: candle high above Donchian high, close near high, EMA bullish
    if (
        high >= donchian_high * 1.001
        and close > candle.open
        and ema_fast > ema_slow
        and close_pos >= 0.6
        and body_atr >= 0.08
    ):
        return Decision(
            "BUY",
            f"Price action swing: high {high:.5f} broke 20-bar high {donchian_high:.5f}, "
            f"EMA bullish, close at {close_pos:.0%} of range.",
            probability_up, 2,
        )

    # Bearish: candle low below Donchian low, close near low, EMA bearish
    if (
        low <= donchian_low * 0.999
        and close < candle.open
        and ema_fast < ema_slow
        and close_pos <= 0.4
        and body_atr >= 0.08
    ):
        return Decision(
            "SELL",
            f"Price action swing: low {low:.5f} broke 20-bar low {donchian_low:.5f}, "
            f"EMA bearish, close at {close_pos:.0%} of range.",
            probability_up, 2,
        )
    return Decision("NO SIGNAL", "No price action swing signal.", probability_up, 0)


def smc_trend_pullback_micro(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """Trend Pullback Micro — trade pullback to EMA in strong trend.

    For 1-3 minute expiry: when the price pulls back to the fast EMA in a
    strong trend (high ADX) and the candle shows rejection from the EMA,
    enter with the trend. This is a high-probability pullback entry for
    short expiry.
    """
    blocked = _cost_gate(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked, None if features is None else probability_up, 0)
    assert features is not None

    close = candle.close
    low = candle.low
    high = candle.high
    ema_fast = float(features.get("ema_fast", 0.0))
    ema_slow = float(features.get("ema_slow", 0.0))
    adx = float(features.get("adx", 0.0))
    body_atr = float(features.get("body_atr", 0.0))
    close_pos = float(features.get("close_position", 0.5))
    wick_ratio = float(features.get("wick_ratio", 0.0))

    # Bullish pullback: low touches EMA fast, closes back above, EMA bullish
    if (
        low <= ema_fast * 1.001
        and close > ema_fast
        and ema_fast > ema_slow
        and adx >= 25
        and close > candle.open
        and close_pos >= 0.6
        and body_atr >= 0.1
    ):
        return Decision(
            "BUY",
            f"Trend pullback: low touched EMA fast {ema_fast:.5f}, "
            f"closed above at {close:.5f}, ADX {adx:.1f}, bullish rejection.",
            probability_up, 3,
        )

    # Bearish pullback: high touches EMA fast, closes back below, EMA bearish
    if (
        high >= ema_fast * 0.999
        and close < ema_fast
        and ema_fast < ema_slow
        and adx >= 25
        and close < candle.open
        and close_pos <= 0.4
        and body_atr >= 0.1
    ):
        return Decision(
            "SELL",
            f"Trend pullback: high touched EMA fast {ema_fast:.5f}, "
            f"closed below at {close:.5f}, ADX {adx:.1f}, bearish rejection.",
            probability_up, 3,
        )
    return Decision("NO SIGNAL", "No trend pullback signal.", probability_up, 0)


def viral_candle_velocity(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """Candle Velocity — trade the velocity of candle movement for short expiry.

    For 1-3 minute expiry: when the candle range is significantly larger than
    the average range (velocity spike) and the close is in the direction of
    the trend, enter. This captures explosive short-term moves.
    """
    blocked = _cost_gate(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked, None if features is None else probability_up, 0)
    assert features is not None

    close = candle.close
    high = candle.high
    low = candle.low
    range_size = high - low
    atr_pips = float(features.get("atr_pips", 1.0))
    ema_fast = float(features.get("ema_fast", 0.0))
    ema_slow = float(features.get("ema_slow", 0.0))
    close_pos = float(features.get("close_position", 0.5))
    body_atr = float(features.get("body_atr", 0.0))

    # Convert range to pips
    range_pips = range_size * 10000

    # Velocity: range is 1.5x ATR or more
    velocity_high = range_pips >= atr_pips * 1.5

    # Bullish velocity: high range, close near high, EMA bullish
    if (
        velocity_high
        and close > candle.open
        and close_pos >= 0.65
        and ema_fast > ema_slow
        and body_atr >= 0.1
    ):
        return Decision(
            "BUY",
            f"Candle velocity: range {range_pips:.1f} pips ({range_pips/atr_pips:.1f}x ATR), "
            f"close at {close_pos:.0%} of range, EMA bullish.",
            probability_up, 2,
        )

    # Bearish velocity: high range, close near low, EMA bearish
    if (
        velocity_high
        and close < candle.open
        and close_pos <= 0.35
        and ema_fast < ema_slow
        and body_atr >= 0.1
    ):
        return Decision(
            "SELL",
            f"Candle velocity: range {range_pips:.1f} pips ({range_pips/atr_pips:.1f}x ATR), "
            f"close at {close_pos:.0%} of range, EMA bearish.",
            probability_up, 2,
        )
    return Decision("NO SIGNAL", "No candle velocity signal.", probability_up, 0)



def smc_trend_resilience(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """Trend Resilience — trade when trend shows resilience against extreme RSI.

    For 1-3 minute expiry: when RSI is extremely overbought (>85) or oversold
    (<15) but the EMA trend is strong and the price is still above VWAP (or
    below VWAP), enter with the trend. In strong trends, RSI can stay at
    extreme levels — this method captures the trend resilience.
    """
    blocked = _cost_gate(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked, None if features is None else probability_up, 0)
    assert features is not None

    rsi = float(features.get("rsi", 50.0))
    ema_fast = float(features.get("ema_fast", 0.0))
    ema_slow = float(features.get("ema_slow", 0.0))
    vwap = float(features.get("vwap", 0.0))
    close = candle.close
    adx = float(features.get("adx", 0.0))
    atr_pips = float(features.get("atr_pips", 1.0))

    # Extreme overbought but trend resilient
    if (
        rsi > 85.0
        and ema_fast > ema_slow
        and close > vwap
        and adx >= 30
        and close >= candle.open
    ):
        return Decision(
            "BUY",
            f"Trend resilience: RSI at {rsi:.1f} but EMA bullish, "
            f"price above VWAP, ADX {adx:.1f} — trend resilient.",
            probability_up, 3,
        )

    # Extreme oversold but trend resilient
    if (
        rsi < 15.0
        and ema_fast < ema_slow
        and close < vwap
        and adx >= 30
        and close <= candle.open
    ):
        return Decision(
            "SELL",
            f"Trend resilience: RSI at {rsi:.1f} but EMA bearish, "
            f"price below VWAP, ADX {adx:.1f} — trend resilient.",
            probability_up, 3,
        )
    return Decision("NO SIGNAL", "No trend resilience signal.", probability_up, 0)


def viral_momentum_persistence(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """Momentum Persistence — trade when momentum persists across multiple candles.

    For 1-3 minute expiry: when the momentum has been positive (or negative)
    for multiple consecutive candles and the EMA trend confirms, enter with
    the persistent momentum. This is a high-probability short-expiry method
    that works even when individual candles are small or mixed.
    """
    blocked = _cost_gate(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked, None if features is None else probability_up, 0)
    assert features is not None

    consecutive_up = float(features.get("consecutive_up", 0.0))
    consecutive_down = float(features.get("consecutive_down", 0.0))
    ema_fast = float(features.get("ema_fast", 0.0))
    ema_slow = float(features.get("ema_slow", 0.0))
    close = candle.close
    vwap = float(features.get("vwap", 0.0))
    adx = float(features.get("adx", 0.0))

    # Bullish: 3+ consecutive up, EMA bullish, price above VWAP
    if (
        consecutive_up >= 3
        and ema_fast > ema_slow
        and close > vwap
        and adx >= 20
    ):
        return Decision(
            "BUY",
            f"Momentum persistence: {consecutive_up:.0f} consecutive up candles, "
            f"EMA bullish, price above VWAP, ADX {adx:.1f}.",
            probability_up, 3,
        )

    # Bearish: 3+ consecutive down, EMA bearish, price below VWAP
    if (
        consecutive_down >= 3
        and ema_fast < ema_slow
        and close < vwap
        and adx >= 20
    ):
        return Decision(
            "SELL",
            f"Momentum persistence: {consecutive_down:.0f} consecutive down candles, "
            f"EMA bearish, price below VWAP, ADX {adx:.1f}.",
            probability_up, 3,
        )
    return Decision("NO SIGNAL", "No momentum persistence signal.", probability_up, 0)


def quant_ema_separation(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """EMA Separation — trade when EMA separation is expanding for trend continuation.

    For 1-3 minute expiry: when the fast EMA is separating from the slow EMA
    (expanding gap) and the price is moving in the direction of the separation,
    enter with the trend. Expanding EMA separation indicates strong trend
    momentum that is ideal for short-expiry trades.
    """
    blocked = _cost_gate(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked, None if features is None else probability_up, 0)
    assert features is not None

    ema_fast = float(features.get("ema_fast", 0.0))
    ema_fast_prev = float(features.get("ema_fast_prev", 0.0))
    ema_slow = float(features.get("ema_slow", 0.0))
    ema_slow_prev = float(features.get("ema_slow_prev", 0.0))
    close = candle.close
    close_pos = float(features.get("close_position", 0.5))
    body_atr = float(features.get("body_atr", 0.0))

    # Current and previous separation in pips
    sep = (ema_fast - ema_slow) * 10000
    sep_prev = (ema_fast_prev - ema_slow_prev) * 10000

    # Bullish: separation expanding and positive
    if (
        sep > sep_prev
        and sep > 1.0
        and ema_fast > ema_slow
        and close > candle.open
        and close_pos >= 0.55
        and body_atr >= 0.08
    ):
        return Decision(
            "BUY",
            f"EMA separation: expanding from {sep_prev:.1f} to {sep:.1f} pips, "
            f"EMA bullish, close at {close_pos:.0%} of range.",
            probability_up, 2,
        )

    # Bearish: separation expanding and negative
    if (
        sep < sep_prev
        and sep < -1.0
        and ema_fast < ema_slow
        and close < candle.open
        and close_pos <= 0.45
        and body_atr >= 0.08
    ):
        return Decision(
            "SELL",
            f"EMA separation: expanding from {sep_prev:.1f} to {sep:.1f} pips, "
            f"EMA bearish, close at {close_pos:.0%} of range.",
            probability_up, 2,
        )
    return Decision("NO SIGNAL", "No EMA separation signal.", probability_up, 0)


def smc_order_block_micro(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """Order Block Micro — trade when price revisits a recent order block.

    For 1-3 minute expiry: when the price pulls back to a recent swing high
    (or low) and bounces back in the direction of the trend, enter with the
    trend. This is a micro order block that acts as support/resistance for
    short-expiry trades.
    """
    blocked = _cost_gate(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked, None if features is None else probability_up, 0)
    assert features is not None

    close = candle.close
    low = candle.low
    high = candle.high
    ema_fast = float(features.get("ema_fast", 0.0))
    ema_slow = float(features.get("ema_slow", 0.0))
    adx = float(features.get("adx", 0.0))
    body_atr = float(features.get("body_atr", 0.0))
    close_pos = float(features.get("close_position", 0.5))
    donchian_low = float(features.get("donchian_low_20", 0.0))
    donchian_high = float(features.get("donchian_high_20", 0.0))

    # Bullish: price bounced from near Donchian low, EMA bullish
    if (
        low <= donchian_low * 1.005
        and close > donchian_low
        and close > candle.open
        and ema_fast > ema_slow
        and close_pos >= 0.6
        and body_atr >= 0.1
        and adx >= 20
    ):
        return Decision(
            "BUY",
            f"Order block micro: bounced from 20-bar low {donchian_low:.5f}, "
            f"EMA bullish, ADX {adx:.1f}, close at {close_pos:.0%} of range.",
            probability_up, 2,
        )

    # Bearish: price rejected from near Donchian high, EMA bearish
    if (
        high >= donchian_high * 0.995
        and close < donchian_high
        and close < candle.open
        and ema_fast < ema_slow
        and close_pos <= 0.4
        and body_atr >= 0.1
        and adx >= 20
    ):
        return Decision(
            "SELL",
            f"Order block micro: rejected from 20-bar high {donchian_high:.5f}, "
            f"EMA bearish, ADX {adx:.1f}, close at {close_pos:.0%} of range.",
            probability_up, 2,
        )
    return Decision("NO SIGNAL", "No order block micro signal.", probability_up, 0)


def viral_close_position_momentum(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """Close Position Momentum — trade when close position aligns with momentum.

    For 1-3 minute expiry: when the close is in the upper (or lower) half of
    the range AND the momentum is positive (or negative), enter with the
    alignment. This is a simple but effective method that combines the
    candle's close position with the momentum direction.
    """
    blocked = _cost_gate(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked, None if features is None else probability_up, 0)
    assert features is not None

    close_pos = float(features.get("close_position", 0.5))
    momentum = float(features.get("return_1", 0.0))
    close = candle.close
    ema_fast = float(features.get("ema_fast", 0.0))
    ema_slow = float(features.get("ema_slow", 0.0))
    body_atr = float(features.get("body_atr", 0.0))
    atr_pips = float(features.get("atr_pips", 1.0))

    # Convert momentum to pips
    mom_pips = momentum * 10000

    # Bullish: close in upper half, positive momentum, EMA bullish
    if (
        close_pos >= 0.6
        and mom_pips > 0
        and ema_fast >= ema_slow * 0.998
        and close >= candle.open
        and body_atr >= 0.08
    ):
        return Decision(
            "BUY",
            f"Close position momentum: close at {close_pos:.0%} of range, "
            f"momentum +{mom_pips:.1f} pips, EMA bullish.",
            probability_up, 2,
        )

    # Bearish: close in lower half, negative momentum, EMA bearish
    if (
        close_pos <= 0.4
        and mom_pips < 0
        and ema_fast <= ema_slow * 1.002
        and close <= candle.open
        and body_atr >= 0.08
    ):
        return Decision(
            "SELL",
            f"Close position momentum: close at {close_pos:.0%} of range, "
            f"momentum {mom_pips:.1f} pips, EMA bearish.",
            probability_up, 2,
        )
    return Decision("NO SIGNAL", "No close position momentum signal.", probability_up, 0)



def smc_trend_override(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """Trend Override — enter with trend regardless of individual candle direction.

    For 1-3 minute expiry: when the EMA trend is strong (high ADX) and the
    price is above VWAP (bullish) or below VWAP (bearish), enter with the
    trend regardless of whether the individual candle is bullish or bearish.
    In strong trends, individual candles can be counter-trend — this method
    overrides the candle signal with the trend signal.
    """
    blocked = _cost_gate(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked, None if features is None else probability_up, 0)
    assert features is not None

    ema_fast = float(features.get("ema_fast", 0.0))
    ema_slow = float(features.get("ema_slow", 0.0))
    vwap = float(features.get("vwap", 0.0))
    close = candle.close
    adx = float(features.get("adx", 0.0))
    atr_pips = float(features.get("atr_pips", 1.0))
    consecutive_up = float(features.get("consecutive_up", 0.0))

    # Bullish override: strong uptrend, price above VWAP, regardless of candle
    if (
        ema_fast > ema_slow
        and close > vwap
        and adx >= 35
        and (consecutive_up >= 1 or close >= candle.open)
    ):
        return Decision(
            "BUY",
            f"Trend override: EMA bullish, price above VWAP, ADX {adx:.1f}, "
            f"entering with trend regardless of candle direction.",
            probability_up, 3,
        )

    # Bearish override: strong downtrend, price below VWAP, regardless of candle
    if (
        ema_fast < ema_slow
        and close < vwap
        and adx >= 35
        and (float(features.get("consecutive_down", 0.0)) >= 1 or close <= candle.open)
    ):
        return Decision(
            "SELL",
            f"Trend override: EMA bearish, price below VWAP, ADX {adx:.1f}, "
            f"entering with trend regardless of candle direction.",
            probability_up, 3,
        )
    return Decision("NO SIGNAL", "No trend override signal.", probability_up, 0)


def viral_trend_channel(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """Trend Channel — trade when price is within the EMA trend channel.

    For 1-3 minute expiry: when the price is between the fast and slow EMA
    (in the trend channel) and the ADX is high, enter with the trend direction.
    The EMA channel provides natural support/resistance for short-expiry trades.
    """
    blocked = _cost_gate(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked, None if features is None else probability_up, 0)
    assert features is not None

    ema_fast = float(features.get("ema_fast", 0.0))
    ema_slow = float(features.get("ema_slow", 0.0))
    close = candle.close
    adx = float(features.get("adx", 0.0))
    close_pos = float(features.get("close_position", 0.5))
    body_atr = float(features.get("body_atr", 0.0))

    # Bullish: price above slow EMA, fast EMA above slow, high ADX
    if (
        close > ema_slow
        and ema_fast > ema_slow
        and adx >= 30
        and close_pos >= 0.5
        and body_atr >= 0.08
    ):
        return Decision(
            "BUY",
            f"Trend channel: price above slow EMA {ema_slow:.5f}, "
            f"EMA bullish, ADX {adx:.1f}, close at {close_pos:.0%} of range.",
            probability_up, 2,
        )

    # Bearish: price below slow EMA, fast EMA below slow, high ADX
    if (
        close < ema_slow
        and ema_fast < ema_slow
        and adx >= 30
        and close_pos <= 0.5
        and body_atr >= 0.08
    ):
        return Decision(
            "SELL",
            f"Trend channel: price below slow EMA {ema_slow:.5f}, "
            f"EMA bearish, ADX {adx:.1f}, close at {close_pos:.0%} of range.",
            probability_up, 2,
        )
    return Decision("NO SIGNAL", "No trend channel signal.", probability_up, 0)


def quant_adx_threshold(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """ADX Threshold — trade when ADX crosses above a strong trend threshold.

    For 1-3 minute expiry: when ADX rises above 50 (very strong trend) and
    the price is in the direction of the trend, enter. ADX above 50 indicates
    an extremely strong trend that is ideal for short-expiry trend following.
    """
    blocked = _cost_gate(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked, None if features is None else probability_up, 0)
    assert features is not None

    adx = float(features.get("adx", 0.0))
    adx_prev = float(features.get("adx_prev", 0.0))
    ema_fast = float(features.get("ema_fast", 0.0))
    ema_slow = float(features.get("ema_slow", 0.0))
    close = candle.close
    close_pos = float(features.get("close_position", 0.5))
    vwap = float(features.get("vwap", 0.0))

    # Bullish: ADX very high, EMA bullish, price above VWAP
    if (
        adx >= 50
        and ema_fast > ema_slow
        and close > vwap
        and close >= candle.open
    ):
        return Decision(
            "BUY",
            f"ADX threshold: ADX at {adx:.1f} (prev {adx_prev:.1f}), "
            f"EMA bullish, price above VWAP.",
            probability_up, 3,
        )

    # Bearish: ADX very high, EMA bearish, price below VWAP
    if (
        adx >= 50
        and ema_fast < ema_slow
        and close < vwap
        and close <= candle.open
    ):
        return Decision(
            "SELL",
            f"ADX threshold: ADX at {adx:.1f} (prev {adx_prev:.1f}), "
            f"EMA bearish, price below VWAP.",
            probability_up, 3,
        )
    return Decision("NO SIGNAL", "No ADX threshold signal.", probability_up, 0)


def smc_wick_direction(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """Wick Direction — trade the direction of the larger wick for reversal.

    For 1-3 minute expiry: when the candle has a dominant wick in one direction
    and the close is on the opposite side of the body, enter with the wick
    rejection direction. This is a mean-reversion setup that works even on
    doji and small-body candles.
    """
    blocked = _cost_gate(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked, None if features is None else probability_up, 0)
    assert features is not None

    close = candle.close
    high = candle.high
    low = candle.low
    open_price = candle.open
    body = abs(close - open_price)
    range_size = high - low
    upper_wick = high - max(close, open_price)
    lower_wick = min(close, open_price) - low
    ema_fast = float(features.get("ema_fast", 0.0))
    ema_slow = float(features.get("ema_slow", 0.0))
    close_pos = float(features.get("close_position", 0.5))

    if range_size < 1e-15:
        return Decision("NO SIGNAL", "No wick direction (zero range).", probability_up, 0)

    # Bearish: dominant upper wick, close below midpoint
    if (
        upper_wick > lower_wick * 2
        and upper_wick > range_size * 0.3
        and close_pos <= 0.4
        and ema_fast <= ema_slow * 1.002
    ):
        return Decision(
            "SELL",
            f"Wick direction: upper wick {upper_wick:.5f} dominant, "
            f"close at {close_pos:.0%} of range, rejection at highs.",
            probability_up, 2,
        )

    # Bullish: dominant lower wick, close above midpoint
    if (
        lower_wick > upper_wick * 2
        and lower_wick > range_size * 0.3
        and close_pos >= 0.6
        and ema_fast >= ema_slow * 0.998
    ):
        return Decision(
            "BUY",
            f"Wick direction: lower wick {lower_wick:.5f} dominant, "
            f"close at {close_pos:.0%} of range, rejection at lows.",
            probability_up, 2,
        )
    return Decision("NO SIGNAL", "No wick direction signal.", probability_up, 0)


def viral_volume_position(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """Volume Position — trade when price is at VWAP with trend confirmation.

    For 1-3 minute expiry: when the price is near VWAP (mean reversion zone)
    and the EMA trend is strong, enter with the trend. VWAP acts as a magnet
    in short timeframes — price tends to return to VWAP, and trading with
    the trend at VWAP provides a high-probability entry.
    """
    blocked = _cost_gate(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked, None if features is None else probability_up, 0)
    assert features is not None

    close = candle.close
    vwap = float(features.get("vwap", 0.0))
    ema_fast = float(features.get("ema_fast", 0.0))
    ema_slow = float(features.get("ema_slow", 0.0))
    close_pos = float(features.get("close_position", 0.5))
    body_atr = float(features.get("body_atr", 0.0))
    atr_pips = float(features.get("atr_pips", 1.0))

    # Distance from VWAP in pips
    vwap_dist = (close - vwap) * 10000

    # Bullish: price slightly above VWAP, EMA bullish, close in upper half
    if (
        0 < vwap_dist <= atr_pips * 0.5
        and ema_fast > ema_slow
        and close_pos >= 0.55
        and close >= candle.open
        and body_atr >= 0.08
    ):
        return Decision(
            "BUY",
            f"Volume position: price {vwap_dist:.1f} pips above VWAP, "
            f"EMA bullish, close at {close_pos:.0%} of range.",
            probability_up, 2,
        )

    # Bearish: price slightly below VWAP, EMA bearish, close in lower half
    if (
        -atr_pips * 0.5 <= vwap_dist < 0
        and ema_fast < ema_slow
        and close_pos <= 0.45
        and close <= candle.open
        and body_atr >= 0.08
    ):
        return Decision(
            "SELL",
            f"Volume position: price {abs(vwap_dist):.1f} pips below VWAP, "
            f"EMA bearish, close at {close_pos:.0%} of range.",
            probability_up, 2,
        )
    return Decision("NO SIGNAL", "No volume position signal.", probability_up, 0)



def smc_moderate_trend(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """Moderate Trend — trade trend continuation with relaxed ATR requirements.

    For 1-3 minute expiry in moderate volatility (ATR 1.5-3 pips): when the
    EMA trend is aligned and the candle shows directional bias, enter with
    the trend. This method uses lower body_atr thresholds than the main
    trend methods, making it fire in quieter markets.
    """
    blocked = _cost_gate(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked, None if features is None else probability_up, 0)
    assert features is not None

    ema_fast = float(features.get("ema_fast", 0.0))
    ema_slow = float(features.get("ema_slow", 0.0))
    body_atr = float(features.get("body_atr", 0.0))
    close = candle.close
    close_pos = float(features.get("close_position", 0.5))
    momentum = float(features.get("return_1", 0.0))
    consecutive_up = float(features.get("consecutive_up", 0.0))

    # Bullish: EMA bullish, close above open, moderate body
    if (
        ema_fast > ema_slow
        and close > candle.open
        and close_pos >= 0.55
        and body_atr >= 0.05
        and momentum > 0
    ):
        return Decision(
            "BUY",
            f"Moderate trend: EMA bullish, body {body_atr:.2f} ATR, "
            f"close at {close_pos:.0%} of range, {consecutive_up:.0f} consecutive up.",
            probability_up, 2,
        )

    # Bearish: EMA bearish, close below open, moderate body
    if (
        ema_fast < ema_slow
        and close < candle.open
        and close_pos <= 0.45
        and body_atr >= 0.05
        and momentum < 0
    ):
        return Decision(
            "SELL",
            f"Moderate trend: EMA bearish, body {body_atr:.2f} ATR, "
            f"close at {close_pos:.0%} of range.",
            probability_up, 2,
        )
    return Decision("NO SIGNAL", "No moderate trend signal.", probability_up, 0)


def viral_moderate_momentum(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """Moderate Momentum — trade momentum with relaxed thresholds for quiet markets.

    For 1-3 minute expiry in moderate volatility: when the 1-bar return is
    positive (or negative) and the RSI is moving in the same direction,
    enter with the momentum. Uses lower RSI velocity thresholds than the
    main momentum methods.
    """
    blocked = _cost_gate(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked, None if features is None else probability_up, 0)
    assert features is not None

    momentum = float(features.get("return_1", 0.0))
    rsi = float(features.get("rsi", 50.0))
    rsi_velocity = float(features.get("rsi_velocity", 0.0))
    ema_fast = float(features.get("ema_fast", 0.0))
    ema_slow = float(features.get("ema_slow", 0.0))
    close = candle.close
    close_pos = float(features.get("close_position", 0.5))
    body_atr = float(features.get("body_atr", 0.0))
    atr_pips = float(features.get("atr_pips", 1.0))

    mom_pips = momentum * 10000

    # Bullish: moderate momentum, RSI rising, EMA bullish
    if (
        mom_pips > 0.5
        and rsi_velocity >= -1.0
        and ema_fast >= ema_slow * 0.998
        and close > candle.open
        and close_pos >= 0.55
        and body_atr >= 0.05
    ):
        return Decision(
            "BUY",
            f"Moderate momentum: +{mom_pips:.1f} pips, RSI {rsi:.1f} (velocity {rsi_velocity:.1f}), "
            f"EMA bullish.",
            probability_up, 2,
        )

    # Bearish: moderate momentum, RSI falling, EMA bearish
    if (
        mom_pips < -0.5
        and rsi_velocity <= 1.0
        and ema_fast <= ema_slow * 1.002
        and close < candle.open
        and close_pos <= 0.45
        and body_atr >= 0.05
    ):
        return Decision(
            "SELL",
            f"Moderate momentum: {mom_pips:.1f} pips, RSI {rsi:.1f} (velocity {rsi_velocity:.1f}), "
            f"EMA bearish.",
            probability_up, 2,
        )
    return Decision("NO SIGNAL", "No moderate momentum signal.", probability_up, 0)


def quant_moderate_range(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """Moderate Range — trade range position with relaxed thresholds.

    For 1-3 minute expiry in moderate volatility: when the close is in the
    upper (or lower) half of the 20-bar range and the EMA trend confirms,
    enter with the range position. Uses lower close_pos thresholds than
    the main range methods.
    """
    blocked = _cost_gate(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked, None if features is None else probability_up, 0)
    assert features is not None

    close = candle.close
    donchian_high = float(features.get("donchian_high_20", 0.0))
    donchian_low = float(features.get("donchian_low_20", 0.0))
    ema_fast = float(features.get("ema_fast", 0.0))
    ema_slow = float(features.get("ema_slow", 0.0))
    close_pos = float(features.get("close_position", 0.5))
    body_atr = float(features.get("body_atr", 0.0))
    range_width = donchian_high - donchian_low

    if range_width <= 0:
        return Decision("NO SIGNAL", "Zero range width.", probability_up, 0)

    range_pos = (close - donchian_low) / range_width

    # Bullish: close in upper 40% of range, EMA bullish
    if (
        range_pos >= 0.6
        and ema_fast > ema_slow
        and close_pos >= 0.5
        and body_atr >= 0.05
    ):
        return Decision(
            "BUY",
            f"Moderate range: close at {range_pos:.0%} of 20-bar range, "
            f"EMA bullish, body {body_atr:.2f} ATR.",
            probability_up, 2,
        )

    # Bearish: close in lower 40% of range, EMA bearish
    if (
        range_pos <= 0.4
        and ema_fast < ema_slow
        and close_pos <= 0.5
        and body_atr >= 0.05
    ):
        return Decision(
            "SELL",
            f"Moderate range: close at {range_pos:.0%} of 20-bar range, "
            f"EMA bearish, body {body_atr:.2f} ATR.",
            probability_up, 2,
        )
    return Decision("NO SIGNAL", "No moderate range signal.", probability_up, 0)


def smc_moderate_candle(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """Moderate Candle — trade candle direction with relaxed body requirements.

    For 1-3 minute expiry in moderate volatility: when the candle shows
    directional bias (close above/below open) and the EMA trend confirms,
    enter with the candle direction. Uses lower body_atr thresholds than
    the main candle methods.
    """
    blocked = _cost_gate(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked, None if features is None else probability_up, 0)
    assert features is not None

    candle_direction = float(features.get("candle_direction", 0.0))
    body_atr = float(features.get("body_atr", 0.0))
    ema_fast = float(features.get("ema_fast", 0.0))
    ema_slow = float(features.get("ema_slow", 0.0))
    close = candle.close
    close_pos = float(features.get("close_position", 0.5))
    momentum = float(features.get("return_1", 0.0))

    # Bullish: bullish candle, EMA bullish, moderate body
    if (
        candle_direction > 0
        and ema_fast > ema_slow
        and body_atr >= 0.05
        and close_pos >= 0.5
        and momentum > 0
    ):
        return Decision(
            "BUY",
            f"Moderate candle: bullish candle, body {body_atr:.2f} ATR, "
            f"EMA bullish, close at {close_pos:.0%} of range.",
            probability_up, 2,
        )

    # Bearish: bearish candle, EMA bearish, moderate body
    if (
        candle_direction < 0
        and ema_fast < ema_slow
        and body_atr >= 0.05
        and close_pos <= 0.5
        and momentum < 0
    ):
        return Decision(
            "SELL",
            f"Moderate candle: bearish candle, body {body_atr:.2f} ATR, "
            f"EMA bearish, close at {close_pos:.0%} of range.",
            probability_up, 2,
        )
    return Decision("NO SIGNAL", "No moderate candle signal.", probability_up, 0)


def viral_moderate_confluence(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """Moderate Confluence — trade when multiple signals align with relaxed thresholds.

    For 1-3 minute expiry in moderate volatility: when the RSI, stochastic,
    and momentum all point in the same direction with the EMA trend, enter.
    Uses lower thresholds than the main confluence methods, making it fire
    more frequently in quieter markets.
    """
    blocked = _cost_gate(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked, None if features is None else probability_up, 0)
    assert features is not None

    rsi = float(features.get("rsi", 50.0))
    stoch_k = float(features.get("stoch_k", 50.0))
    momentum = float(features.get("return_1", 0.0))
    ema_fast = float(features.get("ema_fast", 0.0))
    ema_slow = float(features.get("ema_slow", 0.0))
    close = candle.close
    body_atr = float(features.get("body_atr", 0.0))
    close_pos = float(features.get("close_position", 0.5))

    # Bullish: RSI > 48, Stoch > 48, momentum > 0, EMA bullish
    if (
        rsi > 48.0
        and stoch_k > 48.0
        and momentum > 0
        and ema_fast > ema_slow
        and close > candle.open
        and body_atr >= 0.05
    ):
        return Decision(
            "BUY",
            f"Moderate confluence: RSI {rsi:.1f}, Stoch {stoch_k:.1f}, "
            f"momentum positive, EMA bullish.",
            probability_up, 3,
        )

    # Bearish: RSI < 52, Stoch < 52, momentum < 0, EMA bearish
    if (
        rsi < 52.0
        and stoch_k < 52.0
        and momentum < 0
        and ema_fast < ema_slow
        and close < candle.open
        and body_atr >= 0.05
    ):
        return Decision(
            "SELL",
            f"Moderate confluence: RSI {rsi:.1f}, Stoch {stoch_k:.1f}, "
            f"momentum negative, EMA bearish.",
            probability_up, 3,
        )
    return Decision("NO SIGNAL", "No moderate confluence signal.", probability_up, 0)



def smc_low_atr_trend(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """Low ATR Trend — trade trend in very quiet markets with minimal body.

    For 1-3 minute expiry when ATR is between 0.3-1.38 pips: when the EMA
    trend is aligned and the candle closes in the trend direction, enter
    with the trend. Uses minimal body requirements (body_atr >= 0.03) to
    fire in quiet conditions. The cost guard still applies — this method
    only fires when the strategic cost gate passes.
    """
    blocked = _cost_gate(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked, None if features is None else probability_up, 0)
    assert features is not None

    ema_fast = float(features.get("ema_fast", 0.0))
    ema_slow = float(features.get("ema_slow", 0.0))
    body_atr = float(features.get("body_atr", 0.0))
    close = candle.close
    close_pos = float(features.get("close_position", 0.5))
    momentum = float(features.get("return_1", 0.0))
    consecutive_up = float(features.get("consecutive_up", 0.0))

    # Bullish: EMA bullish, close above open, minimal body
    if (
        ema_fast > ema_slow
        and close >= candle.open
        and close_pos >= 0.5
        and body_atr >= 0.03
        and momentum >= 0
    ):
        return Decision(
            "BUY",
            f"Low ATR trend: EMA bullish, body {body_atr:.2f} ATR, "
            f"close at {close_pos:.0%} of range, {consecutive_up:.0f} consecutive up.",
            probability_up, 2,
        )

    # Bearish: EMA bearish, close below open, minimal body
    if (
        ema_fast < ema_slow
        and close <= candle.open
        and close_pos <= 0.5
        and body_atr >= 0.03
        and momentum <= 0
    ):
        return Decision(
            "SELL",
            f"Low ATR trend: EMA bearish, body {body_atr:.2f} ATR, "
            f"close at {close_pos:.0%} of range.",
            probability_up, 2,
        )
    return Decision("NO SIGNAL", "No low ATR trend signal.", probability_up, 0)


def viral_low_atr_momentum(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """Low ATR Momentum — trade momentum in very quiet markets.

    For 1-3 minute expiry when ATR is low: when the 1-bar return is
    positive (or negative) and the EMA trend confirms, enter with the
    momentum. Uses very low momentum thresholds (0.2 pips) to fire in
    quiet conditions.
    """
    blocked = _cost_gate(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked, None if features is None else probability_up, 0)
    assert features is not None

    momentum = float(features.get("return_1", 0.0))
    ema_fast = float(features.get("ema_fast", 0.0))
    ema_slow = float(features.get("ema_slow", 0.0))
    close = candle.close
    body_atr = float(features.get("body_atr", 0.0))
    close_pos = float(features.get("close_position", 0.5))

    mom_pips = momentum * 10000

    # Bullish: small positive momentum, EMA bullish
    if (
        mom_pips > 0.2
        and ema_fast >= ema_slow * 0.998
        and close >= candle.open
        and body_atr >= 0.03
    ):
        return Decision(
            "BUY",
            f"Low ATR momentum: +{mom_pips:.1f} pips, EMA bullish, body {body_atr:.2f} ATR.",
            probability_up, 2,
        )

    # Bearish: small negative momentum, EMA bearish
    if (
        mom_pips < -0.2
        and ema_fast <= ema_slow * 1.002
        and close <= candle.open
        and body_atr >= 0.03
    ):
        return Decision(
            "SELL",
            f"Low ATR momentum: {mom_pips:.1f} pips, EMA bearish, body {body_atr:.2f} ATR.",
            probability_up, 2,
        )
    return Decision("NO SIGNAL", "No low ATR momentum signal.", probability_up, 0)


def quant_low_atr_range(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """Low ATR Range — trade range position in very quiet markets.

    For 1-3 minute expiry when ATR is low: when the close is in the upper
    (or lower) portion of the 20-bar range and the EMA trend confirms,
    enter with the range position. Uses very low body requirements.
    """
    blocked = _cost_gate(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked, None if features is None else probability_up, 0)
    assert features is not None

    close = candle.close
    donchian_high = float(features.get("donchian_high_20", 0.0))
    donchian_low = float(features.get("donchian_low_20", 0.0))
    ema_fast = float(features.get("ema_fast", 0.0))
    ema_slow = float(features.get("ema_slow", 0.0))
    close_pos = float(features.get("close_position", 0.5))
    body_atr = float(features.get("body_atr", 0.0))
    range_width = donchian_high - donchian_low

    if range_width <= 0:
        return Decision("NO SIGNAL", "Zero range width.", probability_up, 0)

    range_pos = (close - donchian_low) / range_width

    # Bullish: close in upper 50% of range, EMA bullish
    if (
        range_pos >= 0.5
        and ema_fast > ema_slow
        and body_atr >= 0.03
    ):
        return Decision(
            "BUY",
            f"Low ATR range: close at {range_pos:.0%} of 20-bar range, "
            f"EMA bullish, body {body_atr:.2f} ATR.",
            probability_up, 2,
        )

    # Bearish: close in lower 50% of range, EMA bearish
    if (
        range_pos <= 0.5
        and ema_fast < ema_slow
        and body_atr >= 0.03
    ):
        return Decision(
            "SELL",
            f"Low ATR range: close at {range_pos:.0%} of 20-bar range, "
            f"EMA bearish, body {body_atr:.2f} ATR.",
            probability_up, 2,
        )
    return Decision("NO SIGNAL", "No low ATR range signal.", probability_up, 0)


def smc_low_atr_candle(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """Low ATR Candle — trade candle direction in very quiet markets.

    For 1-3 minute expiry when ATR is low: when the candle shows
    directional bias and the EMA trend confirms, enter with the candle
    direction. Uses very low body requirements.
    """
    blocked = _cost_gate(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked, None if features is None else probability_up, 0)
    assert features is not None

    candle_direction = float(features.get("candle_direction", 0.0))
    body_atr = float(features.get("body_atr", 0.0))
    ema_fast = float(features.get("ema_fast", 0.0))
    ema_slow = float(features.get("ema_slow", 0.0))
    close = candle.close
    close_pos = float(features.get("close_position", 0.5))

    # Bullish: bullish candle, EMA bullish
    if (
        candle_direction > 0
        and ema_fast > ema_slow
        and body_atr >= 0.03
        and close_pos >= 0.5
    ):
        return Decision(
            "BUY",
            f"Low ATR candle: bullish candle, body {body_atr:.2f} ATR, "
            f"EMA bullish, close at {close_pos:.0%} of range.",
            probability_up, 2,
        )

    # Bearish: bearish candle, EMA bearish
    if (
        candle_direction < 0
        and ema_fast < ema_slow
        and body_atr >= 0.03
        and close_pos <= 0.5
    ):
        return Decision(
            "SELL",
            f"Low ATR candle: bearish candle, body {body_atr:.2f} ATR, "
            f"EMA bearish, close at {close_pos:.0%} of range.",
            probability_up, 2,
        )
    return Decision("NO SIGNAL", "No low ATR candle signal.", probability_up, 0)


def viral_low_atr_confluence(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """Low ATR Confluence — trade multiple aligned signals in quiet markets.

    For 1-3 minute expiry when ATR is low: when the RSI, stochastic, and
    momentum all point in the same direction with the EMA trend, enter.
    Uses very relaxed thresholds to fire in quiet conditions.
    """
    blocked = _cost_gate(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked, None if features is None else probability_up, 0)
    assert features is not None

    rsi = float(features.get("rsi", 50.0))
    stoch_k = float(features.get("stoch_k", 50.0))
    momentum = float(features.get("return_1", 0.0))
    ema_fast = float(features.get("ema_fast", 0.0))
    ema_slow = float(features.get("ema_slow", 0.0))
    close = candle.close
    body_atr = float(features.get("body_atr", 0.0))

    # Bullish: RSI > 45, Stoch > 45, momentum >= 0, EMA bullish
    if (
        rsi > 45.0
        and stoch_k > 45.0
        and momentum >= 0
        and ema_fast > ema_slow
        and close >= candle.open
        and body_atr >= 0.03
    ):
        return Decision(
            "BUY",
            f"Low ATR confluence: RSI {rsi:.1f}, Stoch {stoch_k:.1f}, "
            f"momentum non-negative, EMA bullish.",
            probability_up, 3,
        )

    # Bearish: RSI < 55, Stoch < 55, momentum <= 0, EMA bearish
    if (
        rsi < 55.0
        and stoch_k < 55.0
        and momentum <= 0
        and ema_fast < ema_slow
        and close <= candle.open
        and body_atr >= 0.03
    ):
        return Decision(
            "SELL",
            f"Low ATR confluence: RSI {rsi:.1f}, Stoch {stoch_k:.1f}, "
            f"momentum non-positive, EMA bearish.",
            probability_up, 3,
        )
    return Decision("NO SIGNAL", "No low ATR confluence signal.", probability_up, 0)

METHOD_REGISTRY: dict[str, Callable[..., Decision]] = {
    "smc_liquidity_sweep": smc_liquidity_sweep,
    "smc_fvg_displacement": smc_fvg_displacement,
    "smc_order_block": smc_order_block,
    "smc_liquidity_grab": smc_liquidity_grab,
    "smc_ascending_triangle": smc_ascending_triangle,
    "ttm_keltner_squeeze": ttm_keltner_squeeze,
    "supertrend_flip": supertrend_flip,
    "ema_ribbon_pullback": ema_ribbon_pullback,
    "viral_rsi_divergence": viral_rsi_divergence,
    "connors_rsi2_reversion": connors_rsi2_reversion,
    "vwap_sigma_reversion": vwap_sigma_reversion,
    "session_box_breakout": session_box_breakout,
    "quant_bollinger_bounce": quant_bollinger_bounce,
    "quant_donchian_breakout": quant_donchian_breakout,
    "smc_bear_flag": smc_bear_flag,
    "viral_macd_cross": viral_macd_cross,
    "viral_adx_trend": viral_adx_trend,
    "quant_rsi_extreme": quant_rsi_extreme,
    "smc_demand_zone": smc_demand_zone,
    "smc_breaker_block": smc_breaker_block,
    "viral_rsi_trend": viral_rsi_trend,
    "quant_atr_channel": quant_atr_channel,
    "viral_volume_profile": viral_volume_profile,
    "smc_ascending_wedge": smc_ascending_wedge,
    "smc_descending_triangle": smc_descending_triangle,
    "viral_ema_cross": viral_ema_cross,
    "viral_stochastic_trend": viral_stochastic_trend,
    "quant_vwap_trend": quant_vwap_trend,
    "quant_kelly_sizing": quant_kelly_sizing,
    "smc_bull_flag": smc_bull_flag,
    "viral_macd_histogram": viral_macd_histogram,
    "quant_parabolic_sar": quant_parabolic_sar,
    "viral_williams_r": viral_williams_r,
    "quant_chaikin_oscillator": quant_chaikin_oscillator,
    "smc_short_trend": smc_short_trend,
    "viral_momentum_sprint": viral_momentum_sprint,
    "quant_ema_velocity": quant_ema_velocity,
    "smc_micro_fvg": smc_micro_fvg,
    "viral_stoch_momentum": viral_stoch_momentum,
    "smc_wick_continuation": smc_wick_continuation,
    "viral_rsi_velocity": viral_rsi_velocity,
    "quant_range_breakout": quant_range_breakout,
    "smc_order_flow": smc_order_flow,
    "viral_momentum_convergence": viral_momentum_convergence,
    "smc_trend_micro": smc_trend_micro,
    "viral_stoch_velocity": viral_stoch_velocity,
    "quant_ema_slope": quant_ema_slope,
    "smc_sweep_reversal": smc_sweep_reversal,
    "viral_close_momentum": viral_close_momentum,
    "smc_trend_follow": smc_trend_follow,
    "viral_momentum_burst": viral_momentum_burst,
    "quant_donchian_slope": quant_donchian_slope,
    "smc_candle_direction": smc_candle_direction,
    "viral_price_action": viral_price_action,
    "smc_trend_streak": smc_trend_streak,
    "viral_rsi_extreme_follow": viral_rsi_extreme_follow,
    "quant_breakout_volume": quant_breakout_volume,
    "smc_order_flow_imbalance": smc_order_flow_imbalance,
    "viral_price_velocity": viral_price_velocity,
    "smc_micro_trend": smc_micro_trend,
    "viral_ema_slope_momentum": viral_ema_slope_momentum,
    "quant_rsi_momentum": quant_rsi_momentum,
    "smc_candle_cluster_trend": smc_candle_cluster_trend,
    "viral_donchian_trend": viral_donchian_trend,
    "smc_close_extreme": smc_close_extreme,
    "viral_momentum_continuation": viral_momentum_continuation,
    "quant_trend_strength": quant_trend_strength,
    "smc_doji_trend": smc_doji_trend,
    "viral_range_position": viral_range_position,
    "smc_vwap_position": smc_vwap_position,
    "viral_rsi_zone": viral_rsi_zone,
    "quant_price_action_swing": quant_price_action_swing,
    "smc_trend_pullback_micro": smc_trend_pullback_micro,
    "viral_candle_velocity": viral_candle_velocity,
    "smc_trend_resilience": smc_trend_resilience,
    "viral_momentum_persistence": viral_momentum_persistence,
    "quant_ema_separation": quant_ema_separation,
    "smc_order_block_micro": smc_order_block_micro,
    "viral_close_position_momentum": viral_close_position_momentum,
    "smc_trend_override": smc_trend_override,
    "viral_trend_channel": viral_trend_channel,
    "quant_adx_threshold": quant_adx_threshold,
    "smc_wick_direction": smc_wick_direction,
    "viral_volume_position": viral_volume_position,
    "smc_moderate_trend": smc_moderate_trend,
    "viral_moderate_momentum": viral_moderate_momentum,
    "quant_moderate_range": quant_moderate_range,
    "smc_moderate_candle": smc_moderate_candle,
    "viral_moderate_confluence": viral_moderate_confluence,
    "smc_low_atr_trend": smc_low_atr_trend,
    "viral_low_atr_momentum": viral_low_atr_momentum,
    "quant_low_atr_range": quant_low_atr_range,
    "smc_low_atr_candle": smc_low_atr_candle,
    "viral_low_atr_confluence": viral_low_atr_confluence,
}

assert set(METHOD_REGISTRY) == set(METHOD_MEMBERS) == set(METHOD_PROTOTYPES)
