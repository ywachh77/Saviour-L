"""Deterministic offline simulation of the live loop.

Used by ``scripts/smoke.py`` and by the integration tests. Everything it
produces is tagged SYNTHETIC, which is exactly what the charter test needs: the
whole pipeline can fire — abstentions, signals, cost-charged settlements — while
the model, the calibrator and the council weights stay untouched.
"""

from __future__ import annotations

import random
from typing import Sequence

from fusion.domain import Candle

from .timeutil import UTC, now


def synthetic_series(
    count: int = 420,
    *,
    seed: int = 20260101,
    base: float = 1.1000,
    volatility: float = 0.00028,
    drift: float = 0.000004,
    start=None,
) -> list[Candle]:
    """A reproducible random walk aligned to whole minutes, ending near 'now'."""
    from datetime import datetime, timedelta

    rng = random.Random(seed)
    first = start or (now().replace(second=0, microsecond=0) - timedelta(minutes=count))
    candles: list[Candle] = []
    price = base
    for index in range(count):
        open_price = price
        close_price = max(0.5, open_price + rng.gauss(drift, volatility))
        wick = abs(rng.gauss(0.0, volatility * 0.7)) + 0.00002
        candles.append(
            Candle(
                timestamp=first + timedelta(minutes=index),
                open=round(open_price, 5),
                high=round(max(open_price, close_price) + wick, 5),
                low=round(min(open_price, close_price) - wick, 5),
                close=round(close_price, 5),
                volume=float(700 + rng.randrange(400)),
            )
        )
        price = close_price
    return candles


class SyntheticFeed:
    """A provider that reveals candles as the clock reaches them."""

    synthetic = True

    def __init__(self, candles: Sequence[Candle]):
        self.candles = list(candles)
        self.cursor = 0
        self.calls = 0
        self.failures = 0

    @property
    def configured(self) -> bool:
        return bool(self.candles)

    def advance_to(self, stamp) -> int:
        """Reveal every candle that has closed at or before ``stamp``."""
        target = 0
        for index, candle in enumerate(self.candles):
            if candle.timestamp < stamp:
                target = index + 1
        self.cursor = max(self.cursor, min(target, len(self.candles)))
        return self.cursor

    def time_series(self, symbol: str, interval: str = "1min", outputsize: int = 60) -> list[Candle]:
        self.calls += 1
        return self.candles[: self.cursor]

    def quote(self, symbol: str) -> dict | None:
        return None


def run_simulation(bot, candles: Sequence[Candle], *, minutes: int | None = None, feed: SyntheticFeed | None = None):
    """Step the bot minute by minute over a synthetic series.

    Returns a summary dict. No network, no wall-clock sleeping.
    """
    from datetime import timedelta

    feed = feed or SyntheticFeed(candles)
    bot.provider = feed
    bot.candles = []

    steps = minutes if minutes is not None else len(candles) - 1
    first_stamp = candles[0].timestamp
    summary = {
        "steps": 0,
        "signals": 0,
        "abstentions": 0,
        "discarded_late": 0,
        "settled": 0,
        "unresolved": 0,
        "trained_model": False,
        "council_weights": {},
    }

    for step in range(1, min(steps, len(candles)) + 1):
        stamp = first_stamp + timedelta(minutes=step)
        feed.advance_to(stamp)
        bot.ingest(at=stamp)
        report = bot.minute_cycle(at=stamp)
        bot.settle_due(at=stamp)
        bot.run_scheduled(at=stamp)
        summary["steps"] += 1
        if report.seal_state == "SIGNAL":
            summary["signals"] += 1
        elif report.seal_state == "DISCARDED_LATE":
            summary["discarded_late"] += 1
        elif report.seal_state == "ABSTAIN":
            summary["abstentions"] += 1

    settlements = bot.store.load_settlements()
    summary["settled"] = len(settlements)
    summary["unresolved"] = sum(1 for row in settlements if row.get("outcome") == "UNRESOLVED")
    summary["trained_model"] = bool(bot.model and bot.model.trained)
    summary["council_weights"] = bot.weights.weights
    summary["feed_calls"] = feed.calls
    summary["utc"] = now().strftime("%Y-%m-%dT%H:%MZ")
    return summary
