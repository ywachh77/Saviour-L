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
)

VIRAL_PROTOTYPES = (
    "ttm_keltner_squeeze",
    "supertrend_flip",
    "ema_ribbon_pullback",
)

QUANT_SCALP_PROTOTYPES = (
    "connors_rsi2_reversion",
    "vwap_sigma_reversion",
    "session_box_breakout",
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
    "ttm_keltner_squeeze": "TTM Bollinger-inside-Keltner volatility squeeze release",
    "supertrend_flip": "ATR(10, 3.0) Supertrend structural flip with ADX confirmation",
    "ema_ribbon_pullback": "Fibonacci 8/13/21/34 EMA ribbon value-pocket pullback reclaim",
    "connors_rsi2_reversion": "Connors 2-period Wilder RSI extreme snapback with 50-EMA filter",
    "vwap_sigma_reversion": "Institutional rolling VWAP +/-2.0 sigma Z-score stretch hook",
    "session_box_breakout": "Tight 30-bar compression box Opening Range Breakout (ORB)",
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


METHOD_REGISTRY: dict[str, Callable[..., Decision]] = {
    "smc_liquidity_sweep": smc_liquidity_sweep,
    "smc_fvg_displacement": smc_fvg_displacement,
    "smc_order_block": smc_order_block,
    "ttm_keltner_squeeze": ttm_keltner_squeeze,
    "supertrend_flip": supertrend_flip,
    "ema_ribbon_pullback": ema_ribbon_pullback,
    "connors_rsi2_reversion": connors_rsi2_reversion,
    "vwap_sigma_reversion": vwap_sigma_reversion,
    "session_box_breakout": session_box_breakout,
}

assert set(METHOD_REGISTRY) == set(METHOD_MEMBERS) == set(METHOD_PROTOTYPES)
