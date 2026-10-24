"""What a trade has to earn before it is worth taking.

The binding constraint
----------------------

Every settled trade pays a round-turn spread. The bot's default is 1.2 pips on
EUR/USD, and the default hold is five minutes. Those two numbers are in conflict,
and no amount of model accuracy resolves it: the question is not *which way*
price goes but *whether the move is big enough to pay for the trip*.

This module answers that question before a signal is allowed to exist.

Calibrating the move projection
-------------------------------

Projecting the move a horizon can deliver needs a constant, and the obvious one
is wrong. For a driftless random walk, ``E|Δ| = σ·sqrt(2/π) ≈ 0.80σ`` while
``E[range] ≈ 1.60σ``, which suggests a capture factor of 0.5 — the value already
used by ``charter.SCALP_ATR_CAPTURE``.

Measured against this repository's own series, 0.5 is too generous. With a
14-period M1 ATR of 5.77 pips, the ratio of actual mean absolute move to
``ATR_1m·sqrt(h)`` was:

    h=5    0.39   h=30   0.41   h=120  0.47
    h=10   0.39   h=60   0.44

The gap is because ATR(14) carries the wick, and a wick is range that no
position can capture.

The constant is **0.38**, below every one of those ratios, and the reason it is
below the smallest rather than the average is directional. At short horizons a
larger constant lets the projection run slightly ABOVE reality — at 0.40 it
overstated the five-minute move by 3.2%. Overstating the move tells the gate a
trade is affordable when it is not, which is the expensive kind of mistake.
Understating it costs a missed trade. So the constant is set where the
projection never overstates at any horizon, and ``test_costs.py`` asserts that
property directly rather than trusting this paragraph.

The rule
--------

A signal is affordable when::

    expected_move(horizon) >= REWARD_TO_COST x spread

with ``REWARD_TO_COST`` defaulting to 3.0, not 1.0. A trade whose entire
expected move merely equals its cost has an expected value of zero before the
model's own error is counted in; three times the cost leaves room for a model
that is right only slightly more often than not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

# Fraction of ``ATR_1m * sqrt(h)`` that a position can realistically capture.
# See the module docstring: measured 0.39-0.47, theoretical 0.50, set at 0.38 so
# the projection never runs above reality at any horizon.
MOVE_CAPTURE = 0.38

# How many times the round-turn cost the projected move must be. One is break
# even before any model error, which is not a trade, it is a coin flip with a
# fee.
REWARD_TO_COST = 3.0

# Nobody is holding a signal from a one-minute bot for longer than two hours;
# beyond this the projection is answering a question the system is not asking.
MAX_HORIZON_MINUTES = 120

MIN_HORIZON_MINUTES = 1


def expected_move_pips(m1_atr_pips: float, horizon_minutes: int) -> float:
    """Projected mean absolute move over ``horizon_minutes``, in pips.

    Uses sqrt-time scaling, which is the right shape for a diffusion and is
    deliberately conservative in its constant.
    """
    atr = float(m1_atr_pips)
    horizon = int(horizon_minutes)
    if atr <= 0.0 or horizon <= 0:
        return 0.0
    return MOVE_CAPTURE * atr * math.sqrt(horizon)


def minimum_viable_horizon(
    m1_atr_pips: float,
    spread_pips: float,
    reward_to_cost: float = REWARD_TO_COST,
    max_horizon: int = MAX_HORIZON_MINUTES,
) -> int | None:
    """The shortest horizon whose projected move clears the cost, or ``None``.

    ``None`` is the important answer: it means **no horizon can pay for this
    trade**, and the correct action is to stay out rather than to hold longer
    and hope.
    """
    atr = float(m1_atr_pips)
    cost = float(spread_pips)
    if atr <= 0.0 or cost < 0.0:
        return None
    needed = float(reward_to_cost) * cost
    if needed <= 0.0:
        return MIN_HORIZON_MINUTES
    for horizon in range(MIN_HORIZON_MINUTES, int(max_horizon) + 1):
        if expected_move_pips(atr, horizon) >= needed:
            return horizon
    return None


def affordable(
    m1_atr_pips: float,
    horizon_minutes: int,
    spread_pips: float,
    reward_to_cost: float = REWARD_TO_COST,
) -> bool:
    """True when this horizon's projected move pays its own way with margin."""
    return expected_move_pips(m1_atr_pips, horizon_minutes) >= float(reward_to_cost) * float(spread_pips)


def affordable_atr_pips(
    horizon_minutes: int,
    spread_pips: float,
    reward_to_cost: float = REWARD_TO_COST,
) -> float:
    """The M1 ATR a market needs before ``horizon_minutes`` is affordable.

    The same test inverted, so a diagnostic can say "you need 8.7 pips of M1
    volatility and the market is offering 5.8" instead of just "no".
    """
    horizon = int(horizon_minutes)
    if horizon <= 0:
        return float("inf")
    return (float(reward_to_cost) * float(spread_pips)) / (MOVE_CAPTURE * math.sqrt(horizon))


@dataclass(frozen=True, slots=True)
class CostVerdict:
    """Whether the configured pair of (horizon, cost) can ever pay."""

    horizon_minutes: int
    spread_pips: float
    m1_atr_pips: float
    projected_move_pips: float
    required_move_pips: float
    viable_horizon: int | None
    required_atr_pips: float

    @property
    def ok(self) -> bool:
        return self.projected_move_pips >= self.required_move_pips

    @property
    def never_viable(self) -> bool:
        """True when no horizon up to the cap can pay for this spread."""
        return self.viable_horizon is None

    def describe(self) -> str:
        if self.never_viable:
            return (
                f"no horizon up to {MAX_HORIZON_MINUTES}m can pay a "
                f"{self.spread_pips:.2f} pip spread at {self.m1_atr_pips:.2f} pips of M1 "
                f"ATR — the market is too quiet to trade, not too hard to predict"
            )
        if self.ok:
            return (
                f"horizon {self.horizon_minutes}m projects {self.projected_move_pips:.2f} pips "
                f"against {self.required_move_pips:.2f} required — affordable"
            )
        return (
            f"horizon {self.horizon_minutes}m projects only {self.projected_move_pips:.2f} pips "
            f"but {self.required_move_pips:.2f} are needed to pay {self.spread_pips:.2f} pips of "
            f"cost — needs {self.required_atr_pips:.2f} pips of M1 ATR (market offers "
            f"{self.m1_atr_pips:.2f}) or a horizon of {self.viable_horizon}m"
        )

    def to_dict(self) -> dict:
        return {
            "horizon_minutes": self.horizon_minutes,
            "spread_pips": self.spread_pips,
            "m1_atr_pips": self.m1_atr_pips,
            "projected_move_pips": self.projected_move_pips,
            "required_move_pips": self.required_move_pips,
            "viable_horizon": self.viable_horizon,
            "required_atr_pips": self.required_atr_pips,
            "ok": self.ok,
        }


def evaluate(
    m1_atr_pips: float,
    horizon_minutes: int,
    spread_pips: float,
    reward_to_cost: float = REWARD_TO_COST,
) -> CostVerdict:
    """Full verdict for one (market, horizon, cost) triple."""
    atr = float(m1_atr_pips)
    horizon = int(horizon_minutes)
    cost = float(spread_pips)
    return CostVerdict(
        horizon_minutes=horizon,
        spread_pips=cost,
        m1_atr_pips=atr,
        projected_move_pips=expected_move_pips(atr, horizon),
        required_move_pips=float(reward_to_cost) * cost,
        viable_horizon=minimum_viable_horizon(atr, cost, reward_to_cost),
        required_atr_pips=affordable_atr_pips(horizon, cost, reward_to_cost),
    )


def median_atr_pips(candles: Sequence) -> float:
    """Median M1 ATR in pips over a candle series, for a robust scale read.

    Median rather than mean so one news spike does not convince the gate that
    the market is permanently volatile.
    """
    from .indicators import atr

    values = [value for value in atr(list(candles), 14) if value is not None]
    if not values:
        return 0.0
    values.sort()
    from .charter import PIP

    ordered = [value / PIP for value in values]
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def resolve_horizon(
    m1_atr_pips: float,
    spread_pips: float,
    base_horizon_minutes: int,
    *,
    adaptive: bool = False,
    reward_to_cost: float = REWARD_TO_COST,
    max_horizon: int = MAX_HORIZON_MINUTES,
) -> int | None:
    """The horizon this signal should use, or ``None`` meaning "do not trade".

    ``adaptive`` controls one thing only: whether a hold that is too short to
    pay is *stretched* to a length that can. It does not control whether the
    cost is checked, because a trade that cannot cover its own toll is not a
    trade — it is a fee with a direction attached.

    So both paths refuse an unpayable hold:

    * adaptive off — the configured horizon, but only if it can pay;
    * adaptive on  — stretched to the shortest horizon that can pay, and never
      shortened below the configured one, since a quiet market needs more time
      and not less.

    ``None`` means even the cap cannot cover the cost. The honest response is to
    stay out rather than to hold longer and hope.

    A previous version returned the configured horizon unchecked when adaptive
    was off, to preserve the behaviour of existing installs. That was a mistake:
    at the EUR/USD volatility actually measured from live candles (0.6-1.5 pips
    M1 ATR) a five-minute hold demands a 97% hit rate to break even against a
    1.2 pip round turn. Preserving that behaviour preserved a guaranteed loss.
    """
    base = max(MIN_HORIZON_MINUTES, int(base_horizon_minutes))
    if not adaptive:
        return base if affordable(m1_atr_pips, base, spread_pips, reward_to_cost) else None
    viable = minimum_viable_horizon(m1_atr_pips, spread_pips, reward_to_cost, max_horizon)
    if viable is None:
        return None
    return max(base, int(viable))


def break_even_win_rate(
    m1_atr_pips: float,
    horizon_minutes: int,
    spread_pips: float,
) -> float:
    """The directional hit rate at which this trade pays for itself.

    With a fixed horizon and no take-profit or stop, a win and a loss are the
    same size, so expected gross move is ``(2p - 1) x E|move|``. Setting that
    equal to the round-turn cost and solving for ``p``::

        p = 0.5 + cost / (2 x E|move|)

    Returns ``inf`` when the projected move is zero or smaller than the cost,
    because no hit rate can rescue a trade whose average move cannot cover the
    toll. That is a real answer, not a missing one: it means the configuration
    is broken rather than the model being inaccurate.

    This is the number to put next to any reported win rate. A 56% hit rate
    sounds like skill until you learn the horizon and the spread demand 67%.
    """
    move = expected_move_pips(m1_atr_pips, horizon_minutes)
    cost = float(spread_pips)
    if move <= 0.0:
        return float("inf")
    if cost <= 0.0:
        return 0.5
    required = 0.5 + cost / (2.0 * move)
    if required > 1.0:
        return float("inf")
    return required
