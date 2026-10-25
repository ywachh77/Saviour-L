"""Cost-charged paper settlement.

Rules that cannot be bent:

* the entry price is the OPEN of the candle the signal named — never a price
  that existed before the signal was sent;
* if the price data for the entry or the exit never arrived, the outcome is
  UNRESOLVED and stays that way;
* every settled trade pays the spread, so the ledger shows what a real fill
  would have paid.

The model label is the realised DIRECTION (up/down). Costs are charged in the
ledger and in every reported expectancy, and a negative net expectancy raises
a human-action alert rather than being averaged away.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from . import charter, instruments
from .domain import Candle, Settlement, Signal
from .timeutil import add_minutes, format_datetime, now, parse_datetime


def find_candle(candles: Sequence[Candle], stamp: datetime) -> Candle | None:
    for candle in candles:
        if candle.timestamp == parse_datetime(stamp):
            return candle
    return None


def settle(signal: Signal, candles: Sequence[Candle], spread_pips: float | None = None) -> Settlement:
    """Settle one signal against completed candles."""
    candles = list(candles)
    instrument = instruments.for_symbol(signal.pair)
    cost = float(signal.spread_pips if spread_pips is None else spread_pips)
    settled_at = now()

    entry_candle = find_candle(candles, signal.entry_time)
    exit_stamp = add_minutes(signal.expiry_time, -1)
    exit_candle = find_candle(candles, exit_stamp)

    if entry_candle is None or exit_candle is None:
        return Settlement(
            signal_id=signal.signal_id,
            outcome=charter.UNRESOLVED,
            entry_price=None if entry_candle is None else entry_candle.open,
            exit_price=None if exit_candle is None else exit_candle.close,
            gross_pips=None,
            move_pips=None,
            net_pips=None,
            spread_pips=cost,
            settled_at=settled_at,
            source=signal.source,
        )

    entry_price = entry_candle.open
    exit_price = exit_candle.close
    direction = 1.0 if signal.side == "BUY" else -1.0
    gross = direction * (exit_price - entry_price) / instrument.pip

    if abs(gross) < 1e-9:
        outcome = charter.TIED
    else:
        outcome = charter.WIN if gross > 0 else charter.LOSS

    return Settlement(
        signal_id=signal.signal_id,
        outcome=outcome,
        entry_price=entry_price,
        exit_price=exit_price,
        gross_pips=round(gross, 4),
        move_pips=round((exit_price - entry_price) / instrument.pip, 4),
        net_pips=round(charter.net_pips(gross, cost), 4),
        spread_pips=cost,
        settled_at=settled_at,
        source=signal.source,
    )


def winning_side(settlement: Settlement) -> str | None:
    """The side that would have profited, or None when nothing can be learned.

    Based on the raw price move, not on the signal's direction, so a losing SELL
    on a rising bar correctly credits BUY.
    """
    if settlement.outcome not in (charter.WIN, charter.LOSS):
        return None
    move = settlement.move_pips
    if move is None:
        return None
    return "BUY" if move > 0 else "SELL"


def learning_label(signal: Signal, settlement: Settlement) -> int | None:
    """1 when price went up over the horizon, 0 when down, None when unknowable."""
    if settlement.outcome not in (charter.WIN, charter.LOSS):
        return None
    up = (settlement.exit_price or 0.0) > (settlement.entry_price or 0.0)
    return 1 if up else 0


@dataclass(frozen=True, slots=True)
class LedgerStats:
    settled: int
    wins: int
    losses: int
    tied: int
    unresolved: int
    net_pips: float
    win_rate: float | None
    wilson_low: float
    wilson_high: float

    def to_dict(self) -> dict:
        return {
            "settled": self.settled,
            "wins": self.wins,
            "losses": self.losses,
            "tied": self.tied,
            "unresolved": self.unresolved,
            "net_pips": round(self.net_pips, 2),
            "win_rate": None if self.win_rate is None else round(self.win_rate, 4),
            "wilson": [round(self.wilson_low, 4), round(self.wilson_high, 4)],
        }


def summarise(settlements: Sequence[Settlement]) -> LedgerStats:
    wins = sum(1 for item in settlements if item.outcome == charter.WIN)
    losses = sum(1 for item in settlements if item.outcome == charter.LOSS)
    tied = sum(1 for item in settlements if item.outcome == charter.TIED)
    unresolved = sum(1 for item in settlements if item.outcome == charter.UNRESOLVED)
    net = sum(float(item.net_pips or 0.0) for item in settlements if item.net_pips is not None)
    decided = wins + losses
    low, high = charter.wilson_interval(wins, decided)
    return LedgerStats(
        settled=len(settlements),
        wins=wins,
        losses=losses,
        tied=tied,
        unresolved=unresolved,
        net_pips=net,
        win_rate=(wins / decided) if decided else None,
        wilson_low=low,
        wilson_high=high,
    )


def format_settlement(signal: Signal, settlement: Settlement) -> str:
    if settlement.outcome == charter.UNRESOLVED:
        return (
            f"{signal.pair} {signal.side} — UNRESOLVED. The price data needed to judge this "
            "signal never arrived; no result is being claimed."
        )
    entry = f"{settlement.entry_price:.5f}" if settlement.entry_price else "?"
    exit_ = f"{settlement.exit_price:.5f}" if settlement.exit_price else "?"
    gross = f"{settlement.gross_pips:+.2f}" if settlement.gross_pips is not None else "?"
    net = f"{settlement.net_pips:+.2f}" if settlement.net_pips is not None else "?"
    verdict = {
        charter.WIN: "direction correct",
        charter.LOSS: "direction wrong",
        charter.TIED: "no move (tie, no cost charged)",
    }[settlement.outcome]
    return (
        f"{signal.pair} {signal.side} — {verdict}. "
        f"Entry {entry} ({format_datetime(signal.entry_time)}) → exit {exit_} "
        f"({format_datetime(signal.expiry_time)}). Gross {gross} pips, "
        f"after {settlement.spread_pips:.2f} pip spread: {net} pips."
    )
