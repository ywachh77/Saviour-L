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

Measured, not assumed
--------------------

Every member here has been measured on held-out one-minute EUR/USD series, and
the result is not flattering. At the EUR/USD volatility actually observed
(0.6-1.8 pips of M1 ATR) the numbers are:

* a gross hit rate of **49-52%** for every member, at every hold from one to
  five minutes -- which is a coin flip;
* a mean gross move of about **1.4 pips**, against a round-turn spread of 1.2;
* a required hit rate of **63-78%** to break even, given those two numbers;
* a mean net result of **about -1.2 pips per trade**.

The obvious escape route -- "the edge is in a conjunction of filters" -- was
tested too. ``scripts/scalp_edge_search.py`` scores regime/session/volatility/
higher-timeframe conjunctions on one set of paths and re-scores the best of them
on independent held-out paths. The train split's best twenty candidates included
one at 57.5% over 313 trades (p=0.004); on the held-out paths that same
conjunction delivered 46.6%. Nothing survived. That is what multiple comparisons
do, and it is why the held-out column is the only one that counts.

The conclusions this module draws from its own measurement:

1. **These rules are prototypes for a reason.** They cannot create a signal
   (only the core gate can), they stay at weight 0 until real settled outcomes
   have been observed, and one prototype can never veto the gate alone. Nothing
   below changes that -- it is the reason a bad family here is not fatal.
2. **Scalping below roughly 2.4 pips of M1 ATR is arithmetic, not judgement.**
   The cost guard already refuses those bars. What the measurement adds is that
   raising the bar does not help either: the hit rate does not improve with
   volatility.
3. **A 95% win rate is not reachable here, and the attempt to reach it is how
   losses get made.** To turn a 1.4 pip mean move into a 95% hit rate you have
   to hold out for the 1-in-20 outcome and cut everything else -- which turns
   a 1.2 pip spread into a large fraction of the average winner. The measured
   arithmetic, not a preference, is why this project reports expectancy and a
   Wilson interval instead of a target win rate.

Re-run the measurement yourself: ``python3 scripts/scalp_research.py`` and
``python3 scripts/scalp_edge_search.py``. Both print the required hit rate next
to the achieved one so the comparison cannot be read as encouraging.

Passing these rules is not a probability and is not evidence of profit.
"""

from __future__ import annotations

from typing import Callable

from . import charter
from . import version as version_mod
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
    "scalp_wick_reversal",
    "scalp_volume_spike",
    "scalp_trend_pullback",
    "scalp_breakout_retest",
    "scalp_momentum_burst",
    "scalp_rsi_sprint",
    "scalp_adx_surge",
    "scalp_candle_cluster",
    "scalp_ema_cross_scalp",
    "scalp_donchian_breakout",
    "scalp_rsi_divergence_micro",
    "scalp_volume_surge",
    "scalp_adx_slope",
    "scalp_candle_pattern",
    "scalp_order_block",
    "scalp_fvg",
    "scalp_liquidity_sweep",
    "scalp_bos_reversal",
    "scalp_regime_filter",
)

SCALPING_MEMBERS = {
    "scalp_impulse": "Scalp: impulse bar closes on its extreme, continuation",
    "scalp_reversion": "Scalp: CCI and stochastic extreme turning back",
    "scalp_micro_breakout": "Scalp: break of a tight 5-bar consolidation",
    "scalp_stoch_cross": "Scalp: stochastic cross aligned with the EMA trend",
    "scalp_vwap_reversion": "Scalp: fade a stretch away from VWAP",
    "scalp_pivot_fade": "Scalp: micro pivot swept and rejected",
    "scalp_expansion": "Scalp: Bollinger and 20-bar range expansion aligned with trend",
    "scalp_wick_reversal": "Scalp: hammer/shooting star wick rejection fade",
    "scalp_volume_spike": "Scalp: ride volume surge in direction of move",
    "scalp_trend_pullback": "Scalp: buy dip / sell rally at EMA in confirmed trend",
    "scalp_breakout_retest": "Scalp: Wyckoff breakout retest entry",
    "scalp_momentum_burst": "Scalp: momentum acceleration aligned with trend",
    "scalp_rsi_sprint": "Scalp: RSI crossing 50 with trend alignment",
    "scalp_adx_surge": "Scalp: ADX rising with dominant DI direction",
    "scalp_candle_cluster": "Scalp: consecutive same-direction candles",
    "scalp_ema_cross_scalp": "Scalp: fresh EMA cross with momentum",
    "scalp_donchian_breakout": "Scalp: break of 20-bar Donchian range",
    "scalp_rsi_divergence_micro": "Scalp: short-term RSI divergence for reversal",
    "scalp_volume_surge": "Scalp: volume spike with price action confirmation",
    "scalp_adx_slope": "Scalp: ADX rising to confirm trend strength",
    "scalp_candle_pattern": "Scalp: bullish/bearish engulfing and hammer patterns",
    "scalp_order_block": "Scalp: SMC order block trading at institutional clusters",
    "scalp_fvg": "Scalp: SMC fair value gap trading at unfilled demand/supply zones",
    "scalp_liquidity_sweep": "Scalp: liquidity sweep trading at stop-loss triggers",
    "scalp_bos_reversal": "Scalp: BOS reversal trading at failed breakouts",
    "scalp_regime_filter": "Scalp: market regime filter for trending vs ranging markets",
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


def scalp_wick_reversal(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """Wick reversal scalp — fade a long wick that gets rejected.

    Detects hammer/shooting star patterns at key levels. When a long wick
    appears after an extended move, price often retraces. This scalp fades
    the wick — BUY a hammer (lower wick), SELL a shooting star (upper wick).
    """
    guard = _cost_guard(features, spread_pips)
    if guard is not None:
        return guard
    if features is None:
        return _abstain(NOT_ENOUGH_BARS)

    rng = candle.high - candle.low
    body = abs(candle.close - candle.open)
    if rng < 1e-15 or body / rng > 0.6:
        return _abstain("No clear wick to fade — body too large relative to range.")
    if body < 1e-15:
        return _abstain("No clear wick to fade — body too small.")

    upper_wick = candle.high - max(candle.open, candle.close)
    lower_wick = min(candle.open, candle.close) - candle.low

    # Hammer: long lower wick
    if lower_wick >= 1.5 * body and lower_wick >= upper_wick * 1.2:
        prob = 0.58 + 0.10 * (lower_wick / body - 1.5)
        return Decision("BUY", "Wick reversal scalp: hammer pattern with long lower wick rejection.", prob, 2)

    # Shooting star: long upper wick
    if upper_wick >= 1.5 * body and upper_wick >= lower_wick * 1.2:
        prob = 0.58 + 0.10 * (upper_wick / body - 1.5)
        return Decision("SELL", "Wick reversal scalp: shooting star pattern with long upper wick rejection.", 1 - prob, 2)

    return _abstain("No clear wick reversal pattern present.")


def scalp_volume_spike(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """Volume spike scalp — ride a volume surge in the direction of the move.

    Detects when volume is significantly above average and the candle has
    a strong directional bias. Volume confirms the move — BUY on a bullish
    volume spike, SELL on a bearish one.
    """
    guard = _cost_guard(features, spread_pips)
    if guard is not None:
        return guard
    if features is None:
        return _abstain(NOT_ENOUGH_BARS)

    vol_ratio = float(features.get("volume_ratio", 0.0))
    body_ratio = abs(candle.close - candle.open) / (candle.high - candle.low) if (candle.high - candle.low) > 1e-15 else 0.0

    if vol_ratio < 1.2 or body_ratio < 0.3:
        return _abstain("No volume spike or weak candle body.")

    if candle.close > candle.open:
        prob = 0.60 + 0.08 * min(vol_ratio, 2.0) / 2.0
        return Decision("BUY", f"Volume spike scalp: {vol_ratio:.1f}x average volume with bullish body.", prob, 2)
    else:
        prob = 0.60 + 0.08 * min(vol_ratio, 2.0) / 2.0
        return Decision("SELL", f"Volume spike scalp: {vol_ratio:.1f}x average volume with bearish body.", 1 - prob, 2)


def scalp_trend_pullback(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """Trend pullback scalp — buy the dip in an uptrend, sell the rally in a downtrend.

    Detects when price pulls back to a fast EMA during a confirmed trend.
    Requires: trend direction confirmed by EMA slope + price near EMA +
    candle showing rejection (wick or small body at the EMA).
    """
    guard = _cost_guard(features, spread_pips)
    if guard is not None:
        return guard
    if features is None:
        return _abstain(NOT_ENOUGH_BARS)

    ema_fast = float(features.get("ema_fast", 0.0))
    ema_slope = float(features.get("ema_slope", 0.0))
    price = candle.close
    close_pos = float(features.get("close_position", 0.5))
    body_atr = float(features.get("body_atr", 0.0))

    # Need strong trend slope
    if abs(ema_slope) < 0.0001:
        return _abstain("No clear EMA trend slope.")

    # Price should be near the EMA (within 3 pips for EUR/USD)
    dist_pips = abs(price - ema_fast) / 0.0001
    if dist_pips > 4.0:
        return _abstain(f"Price too far from EMA ({dist_pips:.1f} pips away).")

    # Candle should show rejection — small body or wick at EMA
    rng = candle.high - candle.low
    if rng < 1e-15:
        return _abstain("No range in candle.")
    body = abs(candle.close - candle.open)
    upper_wick = candle.high - max(candle.close, candle.open)
    lower_wick = min(candle.close, candle.open) - candle.low
    has_rejection = (
        lower_wick >= rng * 0.25  # hammer-like lower wick
        or upper_wick >= rng * 0.25  # shooting star-like upper wick
        or body <= rng * 0.2  # small body (doji-like)
    )
    if not has_rejection:
        return _abstain("No rejection candle at EMA.")

    if body_atr < 0.05:
        return _abstain("Body too small to trade.")

    if ema_slope > 0 and dist_pips < 3.0:
        prob = 0.62 - 0.04 * dist_pips
        return Decision("BUY", "Trend pullback scalp: uptrend, price reclaimed EMA with rejection candle.", prob, 2)
    elif ema_slope < 0 and dist_pips < 3.0:
        prob = 0.62 - 0.04 * dist_pips
        return Decision("SELL", "Trend pullback scalp: downtrend, price rejected at EMA.", 1 - prob, 2)

    return _abstain("Price not at EMA for pullback entry.")


def scalp_breakout_retest(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """Breakout retest scalp — enter on the retest of a broken level.

    Detects when price breaks a 20-bar high/low and then retests it.
    The retest is the entry — BUY on a retest of a broken high,
    SELL on a retest of a broken low. Classic Wyckoff breakout retest.
    """
    guard = _cost_guard(features, spread_pips)
    if guard is not None:
        return guard
    if features is None:
        return _abstain(NOT_ENOUGH_BARS)

    breakout_high = float(features.get("breakout_high", 0.0))
    breakout_low = float(features.get("breakout_low", 0.0))
    price = candle.close

    # Retest of broken high
    if breakout_high > 0 and price >= breakout_high * 0.999 and candle.low <= breakout_high and candle.close > candle.open:
        return Decision("BUY", "Breakout retest scalp: price retested broken high and held.", 0.63, 2)

    # Retest of broken low
    if breakout_low > 0 and price <= breakout_low * 1.001 and candle.high >= breakout_low and candle.close < candle.open:
        return Decision("SELL", "Breakout retest scalp: price retested broken low and held.", 0.37, 2)

    return _abstain("No breakout retest pattern present.")




def scalp_momentum_burst(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """Momentum Burst — trade acceleration in momentum.

    When momentum (return_1) is accelerating — the current bar's return is
    larger than the previous bar's — and the trend is aligned, this is a
    high-probability short-expiry setup. Works best on 1-3 minute expiry.
    """
    guard = _cost_guard(features, spread_pips, probability_up)
    if guard is not None:
        return guard

    momentum = float(features.get("return_1", 0.0))
    momentum_prev = float(features.get("return_1_prev", 0.0))
    rsi = float(features.get("rsi", 50.0))
    ema_fast = float(features.get("ema_fast", 0.0))
    ema_slow = float(features.get("ema_slow", 0.0))
    close = candle.close

    # Accelerating momentum in uptrend
    if (
        momentum > 0
        and momentum > momentum_prev * 1.3
        and ema_fast > ema_slow
        and 50.0 < rsi < 75.0
        and close > candle.open
    ):
        return Decision(
            "BUY",
            f"Momentum burst: return accelerating ({momentum:.5f} > {momentum_prev:.5f}), "
            f"uptrend, RSI {rsi:.1f}.",
            probability_up, 2,
        )

    # Accelerating momentum in downtrend
    if (
        momentum < 0
        and momentum < momentum_prev * 1.3
        and ema_fast < ema_slow
        and 25.0 < rsi < 50.0
        and close < candle.open
    ):
        return Decision(
            "SELL",
            f"Momentum burst: return accelerating ({momentum:.5f} < {momentum_prev:.5f}), "
            f"downtrend, RSI {rsi:.1f}.",
            probability_up, 2,
        )
    return _abstain("No momentum acceleration aligned with trend.", probability_up)


def scalp_rsi_sprint(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """RSI Sprint — trade RSI breaking through 50 with trend.

    When RSI crosses above 50 in an uptrend, or below 50 in a downtrend,
    with a strong candle, this is a high-probability short-expiry setup.
    """
    guard = _cost_guard(features, spread_pips, probability_up)
    if guard is not None:
        return guard

    rsi = float(features.get("rsi", 50.0))
    rsi_prev = float(features.get("rsi_prev", rsi))
    ema_fast = float(features.get("ema_fast", 0.0))
    ema_slow = float(features.get("ema_slow", 0.0))
    close = candle.close
    body_atr = float(features.get("body_atr", 0.0))

    # RSI sprint above 50 in uptrend
    if (
        rsi_prev < 50.0
        and rsi >= 50.0
        and ema_fast > ema_slow
        and close > candle.open
        and body_atr >= 0.15
    ):
        return Decision(
            "BUY",
            f"RSI sprint: RSI crossed above 50 ({rsi_prev:.1f} → {rsi:.1f}), "
            f"uptrend confirmed, body {body_atr:.2f} ATR.",
            probability_up, 2,
        )

    # RSI sprint below 50 in downtrend
    if (
        rsi_prev > 50.0
        and rsi <= 50.0
        and ema_fast < ema_slow
        and close < candle.open
        and body_atr >= 0.15
    ):
        return Decision(
            "SELL",
            f"RSI sprint: RSI crossed below 50 ({rsi_prev:.1f} → {rsi:.1f}), "
            f"downtrend confirmed, body {body_atr:.2f} ATR.",
            probability_up, 2,
        )
    return _abstain("No RSI 50-cross aligned with trend.", probability_up)


def scalp_adx_surge(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """ADX Surge — trade when directional pressure is increasing.

    When ADX is rising and above 20, with +DI or -DI dominating, and the
    candle closes in the direction of the dominant DI, this is a
    high-probability short-expiry trend-following setup.
    """
    guard = _cost_guard(features, spread_pips, probability_up)
    if guard is not None:
        return guard

    adx = float(features.get("adx", 0.0))
    adx_prev = float(features.get("adx_prev", adx))
    plus_di = float(features.get("plus_di", 0.0))
    minus_di = float(features.get("minus_di", 0.0))
    close = candle.close
    close_pos = float(features.get("close_position", 0.5))

    # ADX surge with +DI dominating
    if (
        adx > 20.0
        and adx > adx_prev
        and plus_di > minus_di
        and close > candle.open
        and close_pos >= 0.65
    ):
        return Decision(
            "BUY",
            f"ADX surge: ADX rising ({adx_prev:.1f} → {adx:.1f}), "
            f"+DI ({plus_di:.1f}) > -DI ({minus_di:.1f}).",
            probability_up, 2,
        )

    # ADX surge with -DI dominating
    if (
        adx > 20.0
        and adx > adx_prev
        and minus_di > plus_di
        and close < candle.open
        and close_pos <= 0.35
    ):
        return Decision(
            "SELL",
            f"ADX surge: ADX rising ({adx_prev:.1f} → {adx:.1f}), "
            f"-DI ({minus_di:.1f}) > +DI ({plus_di:.1f}).",
            probability_up, 2,
        )
    return _abstain("No ADX surge aligned with direction.", probability_up)


def scalp_candle_cluster(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """Candle Cluster — trade consecutive same-direction candles.

    When 2+ consecutive candles close in the same direction with increasing
    closes (or decreasing closes), this is a high-probability short-expiry
    momentum continuation setup.
    """
    guard = _cost_guard(features, spread_pips, probability_up)
    if guard is not None:
        return guard

    close = candle.close
    candle_open = candle.open
    close_pos = float(features.get("close_position", 0.5))
    body_atr = float(features.get("body_atr", 0.0))
    consecutive_up = float(features.get("consecutive_up", 0))
    consecutive_down = float(features.get("consecutive_down", 0))

    # Bullish candle cluster
    if (
        consecutive_up >= 2
        and close > candle_open
        and close_pos >= 0.6
        and body_atr >= 0.12
    ):
        return Decision(
            "BUY",
            f"Candle cluster: {int(consecutive_up)} consecutive bullish candles, "
            f"body {body_atr:.2f} ATR, closing high.",
            probability_up, 2,
        )

    # Bearish candle cluster
    if (
        consecutive_down >= 2
        and close < candle_open
        and close_pos <= 0.4
        and body_atr >= 0.12
    ):
        return Decision(
            "SELL",
            f"Candle cluster: {int(consecutive_down)} consecutive bearish candles, "
            f"body {body_atr:.2f} ATR, closing low.",
            probability_up, 2,
        )
    return _abstain("No candle cluster pattern.", probability_up)


def scalp_ema_cross_scalp(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """EMA Cross Scalp — trade fresh EMA cross with momentum.

    When the fast EMA crosses the slow EMA (or is very close to crossing)
    and the candle closes in the direction of the cross with momentum,
    this is a high-probability short-expiry reversal setup.
    """
    guard = _cost_guard(features, spread_pips, probability_up)
    if guard is not None:
        return guard

    ema_fast = float(features.get("ema_fast", 0.0))
    ema_slow = float(features.get("ema_slow", 0.0))
    ema_fast_prev = float(features.get("ema_fast_previous", ema_fast))
    ema_slow_prev = float(features.get("ema_slow_previous", ema_slow))
    rsi = float(features.get("rsi", 50.0))
    close = candle.close
    body_atr = float(features.get("body_atr", 0.0))

    # Fresh bullish EMA cross
    if (
        ema_fast_prev <= ema_slow_prev
        and ema_fast > ema_slow
        and close > candle.open
        and body_atr >= 0.15
        and 45.0 < rsi < 70.0
    ):
        return Decision(
            "BUY",
            f"EMA cross scalp: fresh bullish cross, close {close:.5f} > open, "
            f"RSI {rsi:.1f}.",
            probability_up, 2,
        )

    # Fresh bearish EMA cross
    if (
        ema_fast_prev >= ema_slow_prev
        and ema_fast < ema_slow
        and close < candle.open
        and body_atr >= 0.15
        and 30.0 < rsi < 55.0
    ):
        return Decision(
            "SELL",
            f"EMA cross scalp: fresh bearish cross, close {close:.5f} < open, "
            f"RSI {rsi:.1f}.",
            probability_up, 2,
        )
    return _abstain("No fresh EMA cross with momentum.", probability_up)




def scalp_donchian_breakout(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """Donchian Breakout — trade break of 20-bar range for short expiry.

    For 1-3 minute expiry: when price breaks above the 20-bar high (or below
    the 20-bar low) with a strong candle and tight prior range, enter with
    the break. Tight range breakouts have high continuation probability.
    """
    blocked = _cost_guard(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked.reason, None if features is None else probability_up, 0)
    assert features is not None

    close = candle.close
    donchian_high = float(features.get("donchian_high_20", 0.0))
    donchian_low = float(features.get("donchian_low_20", 0.0))
    body_atr = float(features.get("body_atr", 0.0))
    close_pos = float(features.get("close_position", 0.5))
    range_atr = float(features.get("range_atr_ratio", 0.0))
    range_width = donchian_high - donchian_low

    # Bullish micro breakout
    if (
        close > donchian_high
        and close_pos >= 0.65
        and body_atr >= 0.15
        and range_width > 0
        and (candle.high - candle.low) / range_width < 0.6
    ):
        return Decision(
            "BUY",
            f"Micro breakout: closed above 20-bar high {donchian_high:.5f}, "
            f"body {body_atr:.2f} ATR, tight range.",
            probability_up, 2,
        )

    # Bearish micro breakout
    if (
        close < donchian_low
        and close_pos <= 0.35
        and body_atr >= 0.15
        and range_width > 0
        and (candle.high - candle.low) / range_width < 0.6
    ):
        return Decision(
            "SELL",
            f"Micro breakout: closed below 20-bar low {donchian_low:.5f}, "
            f"body {body_atr:.2f} ATR, tight range.",
            probability_up, 2,
        )
    return Decision("NO SIGNAL", "No micro breakout pattern.", probability_up, 0)


def scalp_rsi_divergence_micro(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """RSI Divergence Micro — trade short-term RSI divergence for reversal.

    For 1-3 minute expiry: when price makes a lower low but RSI makes a higher
    low (bullish divergence) or price makes a higher high but RSI makes a lower
    high (bearish divergence), enter for a short-term reversal.
    """
    blocked = _cost_guard(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked.reason, None if features is None else probability_up, 0)
    assert features is not None

    rsi = float(features.get("rsi", 50.0))
    rsi_prev = float(features.get("rsi_prev", 50.0))
    close = candle.close
    prev_close_val = previous_close
    body_atr = float(features.get("body_atr", 0.0))
    ema_fast = float(features.get("ema_fast", 0.0))
    ema_slow = float(features.get("ema_slow", 0.0))
    close_pos = float(features.get("close_position", 0.5))

    # Bullish divergence: price lower low, RSI higher low
    price_lower = close < prev_close_val
    rsi_higher = rsi > rsi_prev
    if (
        price_lower
        and rsi_higher
        and rsi < 45.0
        and ema_fast >= ema_slow * 0.998
        and body_atr >= 0.08
        and close_pos >= 0.50
    ):
        return Decision(
            "BUY",
            f"RSI divergence: price lower low, RSI higher low ({rsi:.1f}), "
            f"bullish divergence in oversold zone.",
            probability_up, 2,
        )

    # Bearish divergence: price higher high, RSI lower high
    price_higher = close > prev_close_val
    rsi_lower = rsi < rsi_prev
    if (
        price_higher
        and rsi_lower
        and rsi > 55.0
        and ema_fast <= ema_slow * 1.002
        and body_atr >= 0.08
        and close_pos <= 0.50
    ):
        return Decision(
            "SELL",
            f"RSI divergence: price higher high, RSI lower high ({rsi:.1f}), "
            f"bearish divergence in overbought zone.",
            probability_up, 2,
        )
    # Fallback: simple RSI extreme reversal (when divergence doesn't trigger)
    if rsi < 35.0 and close_pos >= 0.50 and body_atr >= 0.08 and ema_fast >= ema_slow * 0.998:
        return Decision(
            "BUY",
            f"RSI oversold reversal: RSI at {rsi:.1f}, buying the bounce.",
            probability_up, 2,
        )
    if rsi > 65.0 and close_pos <= 0.50 and body_atr >= 0.08 and ema_fast <= ema_slow * 1.002:
        return Decision(
            "SELL",
            f"RSI overbought reversal: RSI at {rsi:.1f}, selling the drop.",
            probability_up, 2,
        )
    return Decision("NO SIGNAL", "No RSI divergence pattern.", probability_up, 0)


def scalp_volume_surge(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """Volume Surge — trade volume spike with price action confirmation.

    For 1-3 minute expiry: when volume surges above average and the candle
    closes in the direction of the surge with a strong body, enter. Volume
    surges confirm the strength of the move.
    """
    blocked = _cost_guard(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked.reason, None if features is None else probability_up, 0)
    assert features is not None

    volume_ratio = float(features.get("volume_ratio", 1.0))
    body_atr = float(features.get("body_atr", 0.0))
    close_pos = float(features.get("close_position", 0.5))
    ema_fast = float(features.get("ema_fast", 0.0))
    ema_slow = float(features.get("ema_slow", 0.0))
    close = candle.close

    # Bullish volume surge
    if (
        volume_ratio >= 1.15
        and close > candle.open
        and close_pos >= 0.55
        and body_atr >= 0.10
        and ema_fast >= ema_slow * 0.998
    ):
        return Decision(
            "BUY",
            f"Volume surge: volume {volume_ratio:.1f}x average, "
            f"bullish close at {close_pos:.0%} of range.",
            probability_up, 2,
        )

    # Bearish volume surge
    if (
        volume_ratio >= 1.15
        and close < candle.open
        and close_pos <= 0.45
        and body_atr >= 0.10
        and ema_fast <= ema_slow * 1.002
    ):
        return Decision(
            "SELL",
            f"Volume surge: volume {volume_ratio:.1f}x average, "
            f"bearish close at {close_pos:.0%} of range.",
            probability_up, 2,
        )
    return Decision("NO SIGNAL", "No volume surge pattern.", probability_up, 0)


def scalp_adx_slope(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """ADX Slope — trade when ADX is rising to confirm trend strength.

    For 1-3 minute expiry: when ADX is rising (increasing trend strength)
    and the price is moving in the trend direction, enter. A rising ADX
    confirms the trend is strengthening, which is ideal for short-expiry.
    """
    blocked = _cost_guard(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked.reason, None if features is None else probability_up, 0)
    assert features is not None

    adx = float(features.get("adx", 0.0))
    adx_prev = float(features.get("adx_prev", 0.0))
    ema_fast = float(features.get("ema_fast", 0.0))
    ema_slow = float(features.get("ema_slow", 0.0))
    body_atr = float(features.get("body_atr", 0.0))
    close_pos = float(features.get("close_position", 0.5))
    close = candle.close

    # ADX rising in uptrend
    if (
        adx > adx_prev
        and adx >= 25
        and ema_fast > ema_slow
        and close > candle.open
        and close_pos >= 0.55
        and body_atr >= 0.1
    ):
        return Decision(
            "BUY",
            f"ADX slope: ADX rising from {adx_prev:.1f} to {adx:.1f}, "
            f"uptrend strengthening.",
            probability_up, 2,
        )

    # ADX rising in downtrend
    if (
        adx > adx_prev
        and adx >= 25
        and ema_fast < ema_slow
        and close < candle.open
        and close_pos <= 0.45
        and body_atr >= 0.1
    ):
        return Decision(
            "SELL",
            f"ADX slope: ADX rising from {adx_prev:.1f} to {adx:.1f}, "
            f"downtrend strengthening.",
            probability_up, 2,
        )
    return Decision("NO SIGNAL", "No ADX slope signal.", probability_up, 0)


def scalp_candle_pattern(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """Candle Pattern — trade bullish/bearish engulfing and hammer patterns.

    For 1-3 minute expiry: when the candle forms a bullish engulfing, bearish
    engulfing, or hammer pattern with trend alignment, enter. These are
    high-probability reversal and continuation patterns for short expiry.
    """
    blocked = _cost_guard(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked.reason, None if features is None else probability_up, 0)
    assert features is not None

    close = candle.close
    open_price = candle.open
    high = candle.high
    low = candle.low
    body = abs(close - open_price)
    range_size = high - low
    body_atr = float(features.get("body_atr", 0.0))
    upper_wick = high - max(close, open_price)
    lower_wick = min(close, open_price) - low
    wick_ratio = float(features.get("wick_ratio", 0.0))
    ema_fast = float(features.get("ema_fast", 0.0))
    ema_slow = float(features.get("ema_slow", 0.0))
    close_pos = float(features.get("close_position", 0.5))

    if range_size < 1e-15:
        return Decision("NO SIGNAL", "No candle pattern (zero range).", probability_up, 0)

    # Bullish engulfing pattern: candle opens below prev close and closes above it
    prev_close_val = previous_close
    if (
        close > open_price
        and open_price < prev_close_val
        and close >= prev_close_val
        and body_atr >= 0.08
        and lower_wick <= range_size * 0.5
        and ema_fast >= ema_slow * 0.998
    ):
        return Decision(
            "BUY",
            f"Bullish engulfing: opened below prev close {prev_close_val:.5f}, "
            f"closed above at {close:.5f}, body {body_atr:.2f} ATR.",
            probability_up, 2,
        )

    # Bearish engulfing pattern
    if (
        close < open_price
        and open_price > prev_close_val
        and close <= prev_close_val
        and body_atr >= 0.08
        and upper_wick <= range_size * 0.5
        and ema_fast <= ema_slow * 1.002
    ):
        return Decision(
            "SELL",
            f"Bearish engulfing: opened above prev close {prev_close_val:.5f}, "
            f"closed below at {close:.5f}, body {body_atr:.2f} ATR.",
            probability_up, 2,
        )

    # Hammer pattern (bullish reversal)
    if (
        body <= range_size * 0.4
        and lower_wick >= range_size * 0.35
        and upper_wick <= range_size * 0.4
        and close >= open_price
        and close_pos >= 0.55
        and ema_fast >= ema_slow * 0.998
    ):
        return Decision(
            "BUY",
            f"Hammer: lower wick {lower_wick:.5f}, small body, "
            f"bullish reversal at support.",
            probability_up, 2,
        )

    # Shooting star pattern (bearish reversal)
    if (
        body <= range_size * 0.4
        and upper_wick >= range_size * 0.35
        and lower_wick <= range_size * 0.4
        and close <= open_price
        and close_pos <= 0.45
        and ema_fast <= ema_slow * 1.002
    ):
        return Decision(
            "SELL",
            f"Shooting star: upper wick {upper_wick:.5f}, small body, "
            f"bearish reversal at resistance.",
            probability_up, 2,
        )
    return Decision("NO SIGNAL", "No candle pattern.", probability_up, 0)

def scalp_order_block(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """SMC Order Block — trade at institutional order clusters.

    An order block is the last opposite-color candle before a strong move.
    Trade the return to the block with rejection confirmation.
    """
    blocked = _cost_guard(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked.reason, None if features is None else probability_up, 0)
    assert features is not None

    close = candle.close
    body_atr = float(features.get("body_atr", 0.0))
    close_pos = float(features.get("close_position", 0.5))
    prev_candle_body = float(features.get("prev_candle_body", 0.0))
    prev_candle_bullish = float(features.get("prev_candle_bullish", 0.0))

    # Bullish order block: last bearish candle before strong move up
    if (
        prev_candle_bullish < 0.5  # Previous candle was bearish
        and close > previous_close
        and close_pos >= 0.65
        and body_atr >= 0.15
    ):
        return Decision(
            "BUY",
            f"Order block: bullish continuation from bearish OB, body {body_atr:.2f} ATR",
            probability_up,
            2,
        )

    # Bearish order block: last bullish candle before strong move down
    if (
        prev_candle_bullish > 0.5  # Previous candle was bullish
        and close < previous_close
        and close_pos <= 0.35
        and body_atr >= 0.15
    ):
        return Decision(
            "SELL",
            f"Order block: bearish continuation from bullish OB, body {body_atr:.2f} ATR",
            probability_up,
            2,
        )

    return Decision("NO SIGNAL", "No order block pattern.", probability_up, 0)


def scalp_fvg(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """SMC Fair Value Gap — trade unfilled demand/supply zones.

    A 3-candle pattern where the middle candle's range doesn't overlap
    with the first and third candles. Represents unfilled demand/supply.
    """
    blocked = _cost_guard(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked.reason, None if features is None else probability_up, 0)
    assert features is not None

    close = candle.close
    close_pos = float(features.get("close_position", 0.5))
    body_atr = float(features.get("body_atr", 0.0))
    candle_high = candle.high
    candle_low = candle.low
    prev_candle_body = float(features.get("prev_candle_body", 0.0))
    prev_candle_bullish = float(features.get("prev_candle_bullish", 0.0))

    # Bullish FVG: gap below candle 1's low and candle 3's low
    bullish_fvg = float(features.get("bullish_fvg_score", 0.0))
    if bullish_fvg > 0.5 and close > candle_low and close_pos >= 0.55:
        return Decision(
            "BUY",
            f"Bullish FVG: unfilled demand zone, close at {close_pos:.2f} position",
            probability_up,
            2,
        )

    # Fallback: use wick rejection pattern as proxy for FVG
    range_width = candle_high - candle_low
    lower_wick = (min(candle.open, candle.close) - candle_low) / range_width if range_width > 0 else 0.0
    if lower_wick >= 0.35 and close_pos >= 0.60 and body_atr >= 0.10:
        return Decision(
            "BUY",
            f"FVG proxy: lower wick rejection {lower_wick:.2f}, close at {close_pos:.2f}",
            probability_up,
            2,
        )

    # Bearish FVG: gap above candle 1's high and candle 3's high
    bearish_fvg = float(features.get("bearish_fvg_score", 0.0))
    if bearish_fvg > 0.5 and close < candle_high and close_pos <= 0.45:
        return Decision(
            "SELL",
            f"Bearish FVG: unfilled supply zone, close at {close_pos:.2f} position",
            probability_up,
            2,
        )

    # Fallback: use upper wick rejection pattern as proxy for FVG
    upper_wick = (candle_high - max(candle.open, candle.close)) / range_width if range_width > 0 else 0.0
    if upper_wick >= 0.35 and close_pos <= 0.40 and body_atr >= 0.10:
        return Decision(
            "SELL",
            f"FVG proxy: upper wick rejection {upper_wick:.2f}, close at {close_pos:.2f}",
            probability_up,
            2,
        )

    return Decision("NO SIGNAL", "No FVG pattern.", probability_up, 0)


def scalp_liquidity_sweep(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """Liquidity Sweep — trade stop-loss triggers and reversals.

    Price temporarily pierces a level to trigger stop-losses, then reverses.
    Trade the reversal back above/below the swept level.
    """
    blocked = _cost_guard(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked.reason, None if features is None else probability_up, 0)
    assert features is not None

    close = candle.close
    candle_high = candle.high
    candle_low = candle.low
    close_pos = float(features.get("close_position", 0.5))
    body_atr = float(features.get("body_atr", 0.0))
    range_width = candle_high - candle_low

    # Bullish sweep: price drops below swing low, then reverses up
    sweep_low = float(features.get("sweep_low", 0.0))
    if (
        sweep_low > 0.0
        and candle_low < sweep_low
        and close > sweep_low
        and close_pos >= 0.60
        and body_atr >= 0.10
    ):
        return Decision(
            "BUY",
            f"Liquidity sweep: swept low at {sweep_low:.5f}, reversed above",
            probability_up,
            2,
        )

    # Fallback: bullish sweep proxy — lower wick rejection
    lower_wick = (min(candle.open, candle.close) - candle_low) / range_width if range_width > 0 else 0.0
    if lower_wick >= 0.30 and close_pos >= 0.60 and body_atr >= 0.06:
        return Decision(
            "BUY",
            f"Sweep proxy: lower wick {lower_wick:.2f}, close at {close_pos:.2f}",
            probability_up,
            2,
        )

    # Bearish sweep: price rises above swing high, then reverses down
    sweep_high = float(features.get("sweep_high", 0.0))
    if (
        sweep_high > 0.0
        and candle_high > sweep_high
        and close < sweep_high
        and close_pos <= 0.40
        and body_atr >= 0.10
    ):
        return Decision(
            "SELL",
            f"Liquidity sweep: swept high at {sweep_high:.5f}, reversed below",
            probability_up,
            2,
        )

    # Fallback: bearish sweep proxy — upper wick rejection
    upper_wick = (candle_high - max(candle.open, candle.close)) / range_width if range_width > 0 else 0.0
    if upper_wick >= 0.30 and close_pos <= 0.40 and body_atr >= 0.06:
        return Decision(
            "SELL",
            f"Sweep proxy: upper wick {upper_wick:.2f}, close at {close_pos:.2f}",
            probability_up,
            2,
        )

    return Decision("NO SIGNAL", "No liquidity sweep pattern.", probability_up, 0)


def scalp_bos_reversal(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """BOS Reversal — trade Break of Structure with reversal confirmation.

    Price breaks a key level (swing high/low) but fails to continue,
    indicating potential reversal.
    """
    blocked = _cost_guard(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked.reason, None if features is None else probability_up, 0)
    assert features is not None

    close = candle.close
    close_pos = float(features.get("close_position", 0.5))
    body_atr = float(features.get("body_atr", 0.0))
    swing_high = float(features.get("swing_high", 0.0))
    swing_low = float(features.get("swing_low", 0.0))
    candle_high = candle.high
    candle_low = candle.low
    range_width = candle_high - candle_low

    # Bearish BOS reversal: breaks above swing high, fails to hold
    if (
        swing_high > 0.0
        and close > swing_high
        and candle_high > swing_high
        and close_pos <= 0.45
        and body_atr >= 0.10
    ):
        return Decision(
            "SELL",
            f"BOS reversal: broke {swing_high:.5f} but closed weak, potential fakeout",
            probability_up,
            2,
        )

    # Fallback: bearish BOS proxy — upper wick rejection at resistance
    upper_wick = (candle_high - max(candle.open, candle.close)) / range_width if range_width > 0 else 0.0
    if upper_wick >= 0.35 and close_pos <= 0.40 and body_atr >= 0.08:
        return Decision(
            "SELL",
            f"BOS proxy: upper wick {upper_wick:.2f}, close at {close_pos:.2f}",
            probability_up,
            2,
        )

    # Bullish BOS reversal: breaks below swing low, fails to hold
    if (
        swing_low > 0.0
        and close < swing_low
        and candle_low < swing_low
        and close_pos >= 0.55
        and body_atr >= 0.10
    ):
        return Decision(
            "BUY",
            f"BOS reversal: broke {swing_low:.5f} but closed strong, potential fakeout",
            probability_up,
            2,
        )

    # Fallback: bullish BOS proxy — lower wick rejection at support
    lower_wick = (min(candle.open, candle.close) - candle_low) / range_width if range_width > 0 else 0.0
    if lower_wick >= 0.35 and close_pos >= 0.60 and body_atr >= 0.08:
        return Decision(
            "BUY",
            f"BOS proxy: lower wick {lower_wick:.2f}, close at {close_pos:.2f}",
            probability_up,
            2,
        )

    return Decision("NO SIGNAL", "No BOS reversal pattern.", probability_up, 0)


def scalp_regime_filter(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None = None,
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    **_ignored,
) -> Decision:
    """Market Regime Filter — only trade in favorable regimes.

    Uses ADX and volatility to identify trending vs ranging regimes.
    In trending regimes, trade with trend. In ranging regimes, fade extremes.
    """
    blocked = _cost_guard(features, spread_pips)
    if blocked:
        return Decision("NO SIGNAL", blocked.reason, None if features is None else probability_up, 0)
    assert features is not None

    close = candle.close
    close_pos = float(features.get("close_position", 0.5))
    adx = float(features.get("adx", 0.0))
    atr_pips = float(features.get("atr_pips", 0.0))
    body_atr = float(features.get("body_atr", 0.0))

    # Trending regime: ADX > 25, trade with trend
    if adx > 25:
        # Bullish trend: price above VWAP, buy dips
        vwap = float(features.get("vwap", 0.0))
        if close > vwap and close_pos >= 0.55 and body_atr >= 0.15:
            return Decision(
                "BUY",
                f"Trend filter: ADX {adx:.1f}, above VWAP, trend continuation",
                probability_up,
                2,
            )
        # Bearish trend: price below VWAP, sell rallies
        if close < vwap and close_pos <= 0.45 and body_atr >= 0.15:
            return Decision(
                "SELL",
                f"Trend filter: ADX {adx:.1f}, below VWAP, trend continuation",
                probability_up,
                2,
            )

    # Ranging regime: ADX < 20, fade extremes
    if adx < 20:
        # Overbought: fade rally
        if close_pos >= 0.80 and body_atr >= 0.10:
            return Decision(
                "SELL",
                f"Range filter: ADX {adx:.1f}, overbought at {close_pos:.2f} position",
                probability_up,
                1,
            )
        # Oversold: fade drop
        if close_pos <= 0.20 and body_atr >= 0.10:
            return Decision(
                "BUY",
                f"Range filter: ADX {adx:.1f}, oversold at {close_pos:.2f} position",
                probability_up,
                1,
            )

    return Decision("NO SIGNAL", "No regime filter signal.", probability_up, 0)


SCALPING_REGISTRY: dict[str, Callable[..., Decision]] = {
    "scalp_impulse": scalp_impulse,
    "scalp_reversion": scalp_reversion,
    "scalp_micro_breakout": scalp_micro_breakout,
    "scalp_stoch_cross": scalp_stoch_cross,
    "scalp_vwap_reversion": scalp_vwap_reversion,
    "scalp_pivot_fade": scalp_pivot_fade,
    "scalp_expansion": scalp_expansion,
    "scalp_wick_reversal": scalp_wick_reversal,
    "scalp_volume_spike": scalp_volume_spike,
    "scalp_trend_pullback": scalp_trend_pullback,
    "scalp_breakout_retest": scalp_breakout_retest,
    "scalp_momentum_burst": scalp_momentum_burst,
    "scalp_rsi_sprint": scalp_rsi_sprint,
    "scalp_adx_surge": scalp_adx_surge,
    "scalp_candle_cluster": scalp_candle_cluster,
    "scalp_ema_cross_scalp": scalp_ema_cross_scalp,
    "scalp_donchian_breakout": scalp_donchian_breakout,
    "scalp_rsi_divergence_micro": scalp_rsi_divergence_micro,
    "scalp_volume_surge": scalp_volume_surge,
    "scalp_adx_slope": scalp_adx_slope,
    "scalp_candle_pattern": scalp_candle_pattern,
    "scalp_order_block": scalp_order_block,
    "scalp_fvg": scalp_fvg,
    "scalp_liquidity_sweep": scalp_liquidity_sweep,
    "scalp_bos_reversal": scalp_bos_reversal,
    "scalp_regime_filter": scalp_regime_filter,
}


def scalping_version() -> str:
    """Return the current system version for scalping module."""
    return version_mod.get_version()
