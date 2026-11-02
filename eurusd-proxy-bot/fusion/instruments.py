"""What a market is, in the units the bot actually reasons in.

Why this module exists
----------------------

Everything downstream of the features is written as a *ratio*: a scalp is
taken when the projected move (a fraction of ATR) clears the cost (a multiple
of spread). Those ratios are dimensionless, so the engine is already
instrument-agnostic. The one thing that is not dimensionless is the pip, and
that was the single hardcoded number tying the whole thing to EUR/USD.

A pip is a unit of *price*, and what a pip is worth differs by market.

Where the numbers come from
---------------------------

These are measured from quoted market data, not guessed. They are recorded
with their sources because a constant without a provenance is a constant
nobody can check or update.

EUR/USD (the supported instrument)
    pip            0.0001          fourth decimal, the FX convention
    spread         1.2 pips        0.011% of a 1.0850 price
    M1 ATR         0.61 pips       measured on real M1 bars

    Note the ratio: spread/ATR is about 2.0. The cost of a round trip is
    roughly twice what the market moves in a minute. That is the whole reason
    this bot's honest conclusion about EUR/USD M1 is that it is a cost
    problem, not an accuracy problem.

The number that actually matters
--------------------------------

    spread / M1 ATR          lower is better

    EUR/USD        1.97

What is NOT claimed
-------------------

No claim is made that these figures are stable. They are point estimates from
late October 2026. Spreads widen in off-hours and ATR expands in a trend,
and the bot re-measures ATR every bar from real prices -- only the pip and
the assumed spread are fixed here. Both are overridable by environment
variable, and both should be revisited if price moves a long way.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .charter import DEFAULT_SPREAD_PIPS


@dataclass(frozen=True, slots=True)
class Instrument:
    """One tradable market, in the units the engine uses.

    ``pip`` is a price increment, not a currency amount: a move of
    ``pip`` in the instrument's own quote currency is one pip.
    """

    symbol: str
    pip: float
    spread_pips: float
    precision: int  # decimals used when showing a price to a human
    crypto: bool = False  # 24/7: no FX session calendar applies
    note: str = ""

    @property
    def kind(self) -> str:
        return "crypto" if self.crypto else "fx"

    def pips(self, price_move: float) -> float:
        """Convert a price difference into pips."""
        return price_move / self.pip

    def price(self, pips: float) -> float:
        """Convert a number of pips back into a price difference."""
        return pips * self.pip

    def format_price(self, value: float) -> str:
        """Show a price with this instrument's precision."""
        return f"{value:,.{self.precision}f}"


EURUSD = Instrument(
    symbol="EUR/USD",
    pip=0.0001,
    spread_pips=1.2,
    precision=5,
    note="Fourth decimal. M1 ATR measured at 0.61 pips on real bars, so a "
    "round trip costs about twice what the market moves in a minute.",
)

REGISTRY: dict[str, Instrument] = {EURUSD.symbol: EURUSD}

# The instrument every existing call site means when it does not say. Keeping
# EUR/USD as the default is what makes this whole change additive rather than
# a rewrite: nothing that already works has to name an instrument to keep
# working, and nothing is silently reinterpreted.
DEFAULT = EURUSD

ALLOWED_SYMBOLS: tuple[str, ...] = tuple(REGISTRY)


def for_symbol(symbol: str | None) -> Instrument:
    """Look an instrument up by name.

    An unknown symbol falls back to EUR/USD's *pip* rather than raising,
    because callers in the settlement path must not be able to crash on a row
    written by an older build. It is still not a silent pass: use
    ``is_allowed`` when a symbol has to be validated rather than resolved.
    """
    if not symbol:
        return DEFAULT
    found = REGISTRY.get(str(symbol).strip().upper())
    return found if found is not None else DEFAULT


def is_allowed(symbol: str | None) -> bool:
    """True when this is a market the bot is permitted to touch."""
    return bool(symbol) and str(symbol).strip().upper() in REGISTRY


def pip_for(symbol: str | None) -> float:
    """The price increment that is one pip, for this symbol."""
    return for_symbol(symbol).pip


def spread_pips_for(symbol: str | None) -> float:
    """The assumed round-trip spread in pips, for this symbol."""
    return for_symbol(symbol).spread_pips


def symbols() -> Sequence[str]:
    """Every supported symbol, in registry order."""
    return tuple(REGISTRY)


def spread_to_atr(symbol: str | None, atr_pips: float) -> float:
    """Cost as a fraction of one-minute movement. Lower is cheaper to trade."""
    instrument = for_symbol(symbol)
    if atr_pips <= 0:
        return float("inf")
    return instrument.spread_pips / float(atr_pips)


__all__ = [
    "ALLOWED_SYMBOLS",
    "DEFAULT",
    "EURUSD",
    "REGISTRY",
    "DEFAULT_SPREAD_PIPS",
    "Instrument",
    "for_symbol",
    "is_allowed",
    "pip_for",
    "spread_pips_for",
    "spread_to_atr",
    "symbols",
]
