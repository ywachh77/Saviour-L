"""Which trading session a bar belongs to, and which strategies suit it.

Why this exists when ``hour_sin`` and ``hour_cos`` are already model features
----------------------------------------------------------------------------

Time of day is already available to the logistic model, but only as a smooth
circular encoding. A linear model given ``sin(hour)`` and ``cos(hour)`` can
express "the edge varies smoothly around the clock". It cannot express the thing
that is actually true about FX:

    London and the London/New York overlap are where directional moves happen.
    Asia is where ranges happen.

Those are not two points on a sine wave. They are different regimes with
different appropriate strategies, and the difference shows up in *which rule
should be trusted*, not only in how much to believe a single probability. This
module encodes that as a strategy-selection gate, which is a conditional effect
the model's linear term structurally cannot represent.

What is claimed here, and what is not
-------------------------------------

Claimed: these session windows are the standard FX sessions, and the broad
characterisation (London/overlap directional, Asia ranging) is well established.
That is domain knowledge, not something fitted to data in this repository.

NOT claimed: that this improves the bot's hit rate. That depends on whether real
EUR/USD still behaves this way and on whether these particular rules capture it,
and **this repository has no real market data to check that against**. On a
synthetic random walk there is no session structure at all, so this filter must
do nothing there — which is itself the control experiment, and
``scripts/validate_edge.py`` runs it.

Sessions (UTC)
--------------

    ASIA       00:00-07:00   Tokyo and Sydney, thinnest trend, ranges
    LONDON     07:00-13:00   London open, directional
    OVERLAP    13:00-17:00   London and New York together, peak volume
    NEW_YORK   17:00-21:00   New York afternoon, mixed
    OFF_HOURS  21:00-24:00   post-New York, thinnest book of the day
"""

from __future__ import annotations

from datetime import datetime

ASIA = "ASIA"
LONDON = "LONDON"
OVERLAP = "OVERLAP"
NEW_YORK = "NEW_YORK"
OFF_HOURS = "OFF_HOURS"

ALL_SESSIONS = (ASIA, LONDON, OVERLAP, NEW_YORK, OFF_HOURS)

# (start_hour_inclusive, end_hour_exclusive, name) in UTC. Contiguous and
# covering all 24 hours, so every bar has exactly one session.
_WINDOWS = (
    (0, 7, ASIA),
    (7, 13, LONDON),
    (13, 17, OVERLAP),
    (17, 21, NEW_YORK),
    (21, 24, OFF_HOURS),
)

# The two behavioural families.
CONTINUATION_SESSIONS = frozenset({LONDON, OVERLAP, NEW_YORK})
REVERSION_SESSIONS = frozenset({ASIA, OFF_HOURS, NEW_YORK})

# New York appears in both lists on purpose: the early part trends with London
# still open, the late part ranges once London has closed. Splitting further
# would need a liquidity feed this project does not have.

# Which strategies may speak in which sessions. The core confluence gate is
# deliberately allowed everywhere: it is the only member that can originate a
# signal, and silencing it would mute the bot for hours at a time rather than
# changing what it believes.
ALLOWED: dict[str, frozenset[str]] = {
    ASIA: frozenset(
        {
            "orbix_confluence",
            "ml_consensus",
            "orbix_cadence",
            "range_reversal",
            "scalp_reversion",
            "scalp_pivot_fade",
            "scalp_vwap_reversion",
            "scalp_stoch_cross",
        }
    ),
    LONDON: frozenset(
        {
            "orbix_confluence",
            "ml_consensus",
            "orbix_cadence",
            "trend_pullback",
            "breakout",
            "scalp_impulse",
            "scalp_micro_breakout",
            "scalp_vwap_reversion",
            "scalp_stoch_cross",
        }
    ),
    OVERLAP: frozenset(
        {
            "orbix_confluence",
            "ml_consensus",
            "orbix_cadence",
            "trend_pullback",
            "breakout",
            "scalp_impulse",
            "scalp_micro_breakout",
            "scalp_vwap_reversion",
            "scalp_stoch_cross",
        }
    ),
    NEW_YORK: frozenset(
        {
            "orbix_confluence",
            "ml_consensus",
            "orbix_cadence",
            "trend_pullback",
            "breakout",
            "range_reversal",
            "scalp_impulse",
            "scalp_reversion",
            "scalp_pivot_fade",
            "scalp_vwap_reversion",
            "scalp_stoch_cross",
        }
    ),
    OFF_HOURS: frozenset(
        {
            "orbix_confluence",
            "ml_consensus",
            "orbix_cadence",
            "range_reversal",
            "scalp_reversion",
            "scalp_pivot_fade",
            "scalp_vwap_reversion",
            "scalp_stoch_cross",
        }
    ),
}

_EVERYTHING = frozenset().union(*ALLOWED.values())


def session_of(stamp: datetime) -> str:
    """The FX session a UTC timestamp falls in."""
    hour = int(getattr(stamp, "hour", 0))
    for start, end, name in _WINDOWS:
        if start <= hour < end:
            return name
    return OFF_HOURS  # unreachable: windows cover 0-23


def session_allows(session: str, strategy_id: str) -> bool:
    """Whether this strategy may speak during this session.

    Unknown strategy ids are allowed through rather than silenced. A new
    prototype that nobody added to the map should still be heard and judged on
    its record, not muted by an omission in a lookup table.
    """
    allowed = ALLOWED.get(session)
    if allowed is None:
        return True
    if strategy_id not in _EVERYTHING:
        return True
    return strategy_id in allowed


def describe(session: str) -> str:
    """One line for logs and diagnostics."""
    return {
        ASIA: "Asia — thin, range-bound",
        LONDON: "London — directional open",
        OVERLAP: "London/New York overlap — peak volume",
        NEW_YORK: "New York afternoon — mixed",
        OFF_HOURS: "Off-hours — thinnest book",
    }.get(session, session)


def window_of(session: str) -> tuple[int, int]:
    """The UTC hour window ``[start, end)`` for a session."""
    for start, end, name in _WINDOWS:
        if name == session:
            return start, end
    return 0, 24
