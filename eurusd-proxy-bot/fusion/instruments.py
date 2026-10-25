"""What a market is, in the units the bot actually reasons in.

Why this module exists
----------------------

Everything downstream of the features is written as a *ratio*: a scalp is
taken when the projected move (a fraction of ATR) clears the cost (a multiple
of spread). Those ratios are dimensionless, so the engine is already
instrument-agnostic. The one thing that is not dimensionless is the pip, and
that was the single hardcoded number tying the whole thing to EUR/USD.

A pip is a unit of *price*, and what a pip is worth differs by market:

    EUR/USD   1 pip = 0.0001 of price
    BTC/USD   1 pip = 1 dollar
    ETH/USD   1 pip = 10 cents

Get that wrong and every number the bot reports is wrong by orders of
magnitude -- not "slightly mispriced", wrong. A one-dollar move on bitcoin
would have been recorded as a ten-thousand pip win.

Where the numbers come from
---------------------------

These are measured from quoted market data, not guessed. They are recorded
with their sources because a constant without a provenance is a constant
nobody can check or update.

EUR/USD (the original instrument)
    pip            0.0001          fourth decimal, the FX convention
    spread         1.2 pips        0.011% of a 1.0850 price
    M1 ATR         0.61 pips       measured on real M1 bars

    Note the ratio: spread/ATR is about 2.0. The cost of a round trip is
    roughly twice what the market moves in a minute. That is the whole reason
    this bot's honest conclusion about EUR/USD M1 is that it is a cost
    problem, not an accuracy problem.

BTC/USD (price ~ $80,500, late October 2026)
    pip            $1.00
    spread         0.01% of price  CoinGecko exchange table: Binance, Gate
                                   and Coinbase all quoting 0.01% on
                                   BTC/USD(T). 0.01% of 80,500 = $8.05.
    daily ATR      $2,188.61       ATR(14) daily, 23 Oct 2026
    M1 ATR         ~ 58 pips       daily / sqrt(1440), the usual
                                   square-root-of-time scaling. The 15-minute
                                   figure quoted the same day ($255.61) sits
                                   15% above what that rule predicts and the
                                   hourly ($406.36) 9% below, so the estimate
                                   is good to roughly +/-15%.

ETH/USD (price ~ $2,380, late October 2026)
    pip            $0.10
    spread         0.01% of price  CoinGecko: Binance, OKX, Bitget and Gate
                                   all quoting 0.01% on ETH/USDT.
                                   0.01% of 2,380 = $0.238, so ~2.4 pips.
    daily range    ~ 3.7%          30-day realised volatility of 44% a year
                                   (Admiral Markets, to 11 Oct 2026) is a
                                   2.30% daily sigma; a bar's high-low range
                                   averages about 1.6 sigma.
    M1 ATR         ~ 23 pips       daily range / sqrt(1440)

The number that actually matters
--------------------------------

    spread / M1 ATR          lower is better

    EUR/USD        1.97
    BTC/USD        0.14
    ETH/USD        0.10

On one-minute bars, bitcoin and ether are somewhere between fourteen and
twenty times cheaper to trade relative to how far they move. That is not a
claim that crypto is easier to predict -- it is not, and nothing here says
otherwise. It is a claim about cost, and cost is the binding constraint this
bot has already measured itself against.

This is why adding crypto is worth doing at all, and it is the honest form of
the argument: not "crypto wins more", but "the same edge, if it exists, is not
automatically eaten by the spread before it can be measured."

What is NOT claimed
-------------------

No claim is made that these figures are stable. They are point estimates from
late October 2026. Crypto spreads widen in a liquidation cascade and ATR
expands in a trend, and the bot re-measures ATR every bar from real prices --
only the pip and the assumed spread are fixed here. Both are overridable by
environment variable, and both should be revisited if price moves a long way,
since a $40,000 bitcoin pays half the dollar spread a $80,000 one does.
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

BTCUSD = Instrument(
    symbol="BTC/USD",
    pip=1.0,
    spread_pips=8.0,
    precision=2,
    crypto=True,
    note="One dollar. Spread 0.01% of an ~80,500 price; M1 ATR ~58 pips "
    "from a $2,188 daily ATR scaled by sqrt(1440).",
)

ETHUSD = Instrument(
    symbol="ETH/USD",
    pip=0.1,
    spread_pips=2.4,
    precision=2,
    crypto=True,
    note="Ten cents. Spread 0.01% of an ~2,380 price; M1 ATR ~23 pips from "
    "44% annualised realised volatility scaled to one minute.",
)

REGISTRY: dict[str, Instrument] = {item.symbol: item for item in (EURUSD, BTCUSD, ETHUSD)}

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
    "BTCUSD",
    "DEFAULT",
    "ETHUSD",
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
