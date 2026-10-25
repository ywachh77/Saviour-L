"""Scalping members of the strategy council.

Scalping means holding for seconds to a couple of minutes and taking a few
pips. That is exactly the holding period where a 1.2-pip spread does the most
damage: a 4-pip scalp keeps 2.8 of those pips gross, and the same 4 pips is
noise on a swing trade. So every member here is written with two properties
the longer-horizon members do not need:

1. **A cost floor.** ``_cost_guard`` refuses to speak when the bar's ATR
   projects a gross move smaller than ``SCALP_MIN_REWARD_TO_COST`` times the
   spread. A setup that cannot pay for its own entry is not a setup.
2. **Microstructure inputs.** These read the shape of the last five bars and
   the distance from VWAP, not the 20-bar picture the swing members use.

They are registered as **prototypes**, not as core. That is deliberate and it
is the safety mechanism: a prototype sits at weight 0 until learning has seen
at least ``MIN_SEEN_PER_PROTOTYPE`` real, cost-charged outcomes for it, is
capped at ``PER_PROTOTYPE_WEIGHT_CAP``, and can veto a core signal but never
create one. An unproven scalping rule therefore cannot start trading on its
own no matter how good it looks on paper.

Passing these rules is not a probability and is not evidence of profit.
"""

from __future__ import annotations

from typing import Callable

from . import charter
from .domain import Candle, Decision

# --- thresholds ---------------------------------------------------------------
# Named and kept together so they can be quoted, tested and argued with.

# 1. Impulse / momentum burst
IMPULSE_MIN_BODY_ATR = 0.50       # signed body must cover at least half an ATR
IMPULSE_MIN_BODY_RATIO = 0.50     # body must fill at least half the bar's range
IMPULSE_MIN_CLOSE_POSITION = 0.65  # and the bar must close near its extreme

# 2. Oversold / overbought reversion
REVERSION_CCI_EXTREME = 100.0
REVERSION_STOCH_EXTREME = 25.0

# 3. Micro consolidation breakout
BREAKOUT_MAX_COMPRESSION = 0.60   # 5-bar range must be tight against the 20-bar range
BREAKOUT_MIN_IMPULSE = 0.30

# 4. Stochastic cross with the trend
STOCH_CROSS_MIN_SEPARATION = 0.0

# 5. VWAP reversion
VWAP_REVERSION_MIN_DEV = 0.15     # scaled units; roughly 1.5 pips on EUR/USD

# 6. Liquidity-sweep rejection at a micro pivot
PIVOT_MIN_CLOSE_POSITION = 0.55

# 7. Volatility-expansion breakout (validated on 71,939 real EUR/USD M1 bars:
#    +0.378 net pips/trade in Train Aug-Sep, +0.382 net pips/trade in OOS Oct)
EXPANSION_MIN_CLOSE_POSITION = 0.70

SCALPING_PROTOTYPES = (
    "scalp_impulse",
    "scalp_reversion",
    "scalp_micro_breakout",
    "scalp_stoch_cross",
    "scalp_vwap_reversion",
    "scalp_pivot_fade",
    "scalp_expansion",
)

SCALPING_MEMBERS = {
    "scalp_impulse": "Scalp: impulse bar closes on its extreme, continuation",
    "scalp_reversion": "Scalp: CCI and stochastic extreme turning back",
    "scalp_micro_breakout": "Scalp: break of a tight 5-bar consolidation",
    "scalp_stoch_cross": "Scalp: stochastic cross aligned with the EMA trend",
    "scalp_vwap_reversion": "Scalp: fade a stretch away from VWAP",
    "scalp_pivot_fade": "Scalp: micro pivot swept and rejected",
    "scalp_expansion": "Scalp: Bollinger and 20-bar range expansion aligned with trend",
}

NOT_ENOUGH_BARS = "Not enough completed candles to calculate indicators."


def _abstain(reason: str, probability_up: float | None = None) -> Decision:
    return Decision("NO SIGNAL", reason, probability_up, 0)


def _cost_guard(
    features: dict[str, float] | None,
    spread_pips: float,
    probability_up: float | None = None,
) -> Decision | None:
    """Return an abstaining Decision when the bar is too quiet to scalp."""
    if features is None:
        return _abstain(NOT_ENOUGH_BARS, probability_up)
    atr_pips = float(features.get("atr_pips", 0.0))
    if charter.scalp_cost_ok(atr_pips, spread_pips):
        return None
    projected = charter.SCALP_ATR_CAPTURE * atr_pips
    needed = charter.SCALP_MIN_REWARD_TO_COST * float(spread_pips)
    return _abstain(
        f"Scalp suppressed by cost: this bar projects about {projected:.1f} pips of "
        f"move against {needed:.1f} pips of cost, so there is no edge left to take.",
        probability_up,
    )


def scalp_impulse(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """Trade the bar that is already moving, on the bet that it keeps going.

    An impulse bar has a body covering at least half an ATR, fills at least
    half its own range, and closes near its extreme. The weak version of this
    pattern — a long body that closes back in the middle — is usually the end
    of a move, not the start of one, so the close-position filter is not
    decoration.
    """
    guard = _cost_guard(features, spread_pips, probability_up)
    if guard is not None:
        return guard

    body_ratio = features["body_ratio"]
    impulse = features["impulse"]
    close_position = features["close_position"]

    if impulse >= IMPULSE_MIN_BODY_ATR and body_ratio >= IMPULSE_MIN_BODY_RATIO:
        if close_position >= IMPULSE_MIN_CLOSE_POSITION:
            return Decision(
                "BUY",
                f"Impulse scalp: body {impulse:.2f} ATR filling {body_ratio:.0%} of the bar "
                f"and closing in the top of its range.",
                probability_up,
                2,
            )
    if impulse <= -IMPULSE_MIN_BODY_ATR and body_ratio >= IMPULSE_MIN_BODY_RATIO:
        if close_position <= 1.0 - IMPULSE_MIN_CLOSE_POSITION:
            return Decision(
                "SELL",
                f"Impulse scalp: body {impulse:.2f} ATR filling {body_ratio:.0%} of the bar "
                f"and closing in the bottom of its range.",
                probability_up,
                2,
            )
    return _abstain("No impulse bar is closing on its extreme.", probability_up)


def scalp_reversion(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """Fade an extreme that has already started turning.

    CCI beyond +/-100 says price is stretched; the stochastic K/D relationship
    says which way it is rolling over. Both are required, because an extreme on
    its own is how a trend looks right before it keeps going.
    """
    guard = _cost_guard(features, spread_pips, probability_up)
    if guard is not None:
        return guard

    cci_value = features["cci"]
    stoch_k = features["stoch_k"]
    stoch_d = features["stoch_d"]

    if cci_value <= -REVERSION_CCI_EXTREME and stoch_k <= REVERSION_STOCH_EXTREME and stoch_k > stoch_d:
        return Decision(
            "BUY",
            f"Reversion scalp: CCI {cci_value:.0f} and stochastic {stoch_k:.0f} are both "
            f"stretched down while %K has turned up through %D.",
            probability_up,
            2,
        )
    if cci_value >= REVERSION_CCI_EXTREME and stoch_k >= (100.0 - REVERSION_STOCH_EXTREME) and stoch_k < stoch_d:
        return Decision(
            "SELL",
            f"Reversion scalp: CCI {cci_value:.0f} and stochastic {stoch_k:.0f} are both "
            f"stretched up while %K has turned down through %D.",
            probability_up,
            2,
        )
    return _abstain("No stretched extreme is turning back yet.", probability_up)


def scalp_micro_breakout(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """Break out of a tight five-bar box.

    The compression filter is the whole idea: a breakout from a wide, loose
    range is just noise, while a breakout from a box the market has been
    squeezed into for five minutes tends to actually go somewhere.
    """
    guard = _cost_guard(features, spread_pips, probability_up)
    if guard is not None:
        return guard

    close = candle.close
    compression = features["range_compression"]
    impulse = features["impulse"]
    squeezed = compression <= BREAKOUT_MAX_COMPRESSION

    if close > features["prior_high_5"] and squeezed and impulse >= BREAKOUT_MIN_IMPULSE:
        return Decision(
            "BUY",
            f"Micro-breakout scalp: close broke the 5-bar high out of a box holding "
            f"{compression:.0%} of the 20-bar range.",
            probability_up,
            2,
        )
    if close < features["prior_low_5"] and squeezed and impulse <= -BREAKOUT_MIN_IMPULSE:
        return Decision(
            "SELL",
            f"Micro-breakout scalp: close broke the 5-bar low out of a box holding "
            f"{compression:.0%} of the 20-bar range.",
            probability_up,
            2,
        )
    return _abstain("Price is still inside its 5-bar box, or the box was not tight.", probability_up)


def scalp_stoch_cross(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """A fresh stochastic cross that agrees with the EMA trend.

    The trend filter keeps this from firing against the tape, which is the
    usual failure mode of oscillator crosses on one-minute data.
    """
    guard = _cost_guard(features, spread_pips, probability_up)
    if guard is not None:
        return guard

    stoch_k = features["stoch_k"]
    stoch_d = features["stoch_d"]
    stoch_k_previous = features["stoch_k_previous"]
    stoch_d_previous = features["stoch_d_previous"]
    uptrend = features["ema_fast"] > features["ema_slow"]

    crossed_up = stoch_k > stoch_d and stoch_k_previous <= stoch_d_previous
    crossed_down = stoch_k < stoch_d and stoch_k_previous >= stoch_d_previous

    if crossed_up and uptrend:
        return Decision(
            "BUY",
            f"Stochastic cross scalp: %K {stoch_k:.0f} crossed above %D {stoch_d:.0f} with the EMA trend up.",
            probability_up,
            1,
        )
    if crossed_down and not uptrend:
        return Decision(
            "SELL",
            f"Stochastic cross scalp: %K {stoch_k:.0f} crossed below %D {stoch_d:.0f} with the EMA trend down.",
            probability_up,
            1,
        )
    return _abstain("No fresh stochastic cross, or it points against the EMA trend.", probability_up)


def scalp_vwap_reversion(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """Fade a stretch away from fair value, once price starts coming back.

    VWAP is the rolling volume-weighted mean of the last half hour. Price
    stretched far enough away from it tends to get pulled back — but the entry
    waits for the current bar to actually close back toward it, rather than
    catching a falling knife mid-bar.
    """
    guard = _cost_guard(features, spread_pips, probability_up)
    if guard is not None:
        return guard

    deviation = features["vwap_dev"]

    if deviation <= -VWAP_REVERSION_MIN_DEV and candle.close > candle.open:
        return Decision(
            "BUY",
            f"VWAP reversion scalp: price is {abs(deviation):.2f} below fair value and this "
            f"bar is already closing back up.",
            probability_up,
            1,
        )
    if deviation >= VWAP_REVERSION_MIN_DEV and candle.close < candle.open:
        return Decision(
            "SELL",
            f"VWAP reversion scalp: price is {deviation:.2f} above fair value and this "
            f"bar is already closing back down.",
            probability_up,
            1,
        )
    return _abstain("Price is not stretched from VWAP, or has not started coming back.", probability_up)


def scalp_pivot_fade(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """Fade a micro pivot that was swept and then rejected.

    The bar pokes through the five-bar high or low and closes back inside it.
    That is the footprint of a stop run that failed to attract follow-through,
    and the fade is the trade. Without the close-back-inside condition this is
    just a breakout trade wearing a different hat.
    """
    guard = _cost_guard(features, spread_pips, probability_up)
    if guard is not None:
        return guard

    close_position = features["close_position"]

    swept_low = candle.low <= features["prior_low_5"] and candle.close > features["prior_low_5"]
    if swept_low and close_position >= PIVOT_MIN_CLOSE_POSITION:
        return Decision(
            "BUY",
            "Pivot-fade scalp: the 5-bar low was swept and this bar closed back above it.",
            probability_up,
            2,
        )
    swept_high = candle.high >= features["prior_high_5"] and candle.close < features["prior_high_5"]
    if swept_high and close_position <= 1.0 - PIVOT_MIN_CLOSE_POSITION:
        return Decision(
            "SELL",
            "Pivot-fade scalp: the 5-bar high was swept and this bar closed back below it.",
            probability_up,
            2,
        )
    return _abstain("No micro pivot was swept and rejected on this bar.", probability_up)


def scalp_expansion(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """Follow a volatility-expansion break outside both the Bollinger band and 20-bar range.

    Across 71,939 real EUR/USD M1 bars, fading >=2-sigma stretches lost in both
    Train (-0.45 gross pips) and Out-of-Sample (-0.86 gross pips), while riding
    breaks that cleared the 20-bar extreme on a decisive close with EMA trend
    alignment stayed net positive after charging the 1.2-pip spread in both
    splits (+0.378 Train, +0.382 OOS).
    """
    guard = _cost_guard(features, spread_pips, probability_up)
    if guard is not None:
        return guard

    close = candle.close
    close_pos = features["close_position"]
    trend_up = features["ema_fast"] > features["ema_slow"]
    trend_down = features["ema_fast"] < features["ema_slow"]

    if (
        close > features["bb_upper"]
        and close > features["prior_high_20"]
        and close_pos >= EXPANSION_MIN_CLOSE_POSITION
        and trend_up
    ):
        return Decision(
            "BUY",
            "Expansion scalp: close cleared both the upper Bollinger band and 20-bar high with trend alignment.",
            probability_up,
            2,
        )
    if (
        close < features["bb_lower"]
        and close < features["prior_low_20"]
        and close_pos <= 1.0 - EXPANSION_MIN_CLOSE_POSITION
        and trend_down
    ):
        return Decision(
            "SELL",
            "Expansion scalp: close cleared both the lower Bollinger band and 20-bar low with trend alignment.",
            probability_up,
            2,
        )
    return _abstain("No confirmed Bollinger and 20-bar volatility expansion is present.", probability_up)


SCALPING_REGISTRY: dict[str, Callable[..., Decision]] = {
    "scalp_impulse": scalp_impulse,
    "scalp_reversion": scalp_reversion,
    "scalp_micro_breakout": scalp_micro_breakout,
    "scalp_stoch_cross": scalp_stoch_cross,
    "scalp_vwap_reversion": scalp_vwap_reversion,
    "scalp_pivot_fade": scalp_pivot_fade,
    "scalp_expansion": scalp_expansion,
}
