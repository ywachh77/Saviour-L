"""Shared builders for the test-suite: deterministic synthetic candles.

Everything produced here is tagged SYNTHETIC. Tests rely on that tag to prove
the charter holds: synthetic data may run the pipeline but trains nothing.
"""

from __future__ import annotations

import math
import os
from contextlib import contextmanager
import random
from datetime import datetime, timedelta, timezone

from fusion.domain import Candle

UTC = timezone.utc
START = datetime(2026, 1, 5, 8, 0, tzinfo=UTC)

SYNTHETIC = "SYNTHETIC"
LIVE = "LIVE"


def make_candles(
    count: int = 200,
    *,
    start: datetime | None = None,
    seed: int = 7,
    base: float = 1.1000,
    volatility: float = 0.00035,
    drift: float = 0.0,
    volume: int = 1000,
) -> list[Candle]:
    """A reproducible random walk with plausible OHLC structure."""
    rng = random.Random(seed)
    stamp = start or START
    candles: list[Candle] = []
    price = base
    for index in range(count):
        open_price = price
        shock = rng.gauss(drift, volatility)
        close_price = max(0.5, open_price + shock)
        wick = abs(rng.gauss(0.0, volatility * 0.6))
        high = max(open_price, close_price) + wick
        low = min(open_price, close_price) - wick
        if low <= 0:
            low = min(open_price, close_price) * 0.9999
        candles.append(
            Candle(
                timestamp=stamp + timedelta(minutes=index),
                open=round(open_price, 5),
                high=round(high, 5),
                low=round(low, 5),
                close=round(close_price, 5),
                volume=float(volume + rng.randrange(200)),
            )
        )
        price = close_price
    return candles


def trending_candles(count: int = 200, *, seed: int = 11, step: float = 0.00022) -> list[Candle]:
    return make_candles(count, seed=seed, drift=step, volatility=0.00018)


def flat_candles(count: int = 200, *, seed: int = 3) -> list[Candle]:
    return make_candles(count, seed=seed, drift=0.0, volatility=0.0)


def candle_at(stamp: datetime, price: float = 1.1000) -> Candle:
    return Candle(
        timestamp=stamp,
        open=price,
        high=price + 0.0002,
        low=price - 0.0002,
        close=price + 0.0001,
        volume=1000.0,
    )


def pip(value: float) -> float:
    return value / 0.0001


def nearly(a: float, b: float, tolerance: float = 1e-9) -> bool:
    return math.isclose(a, b, rel_tol=tolerance, abs_tol=tolerance)


@contextmanager
def preserved_environ():
    """Undo any environment change a test makes, including indirect ones.

    Needed because ``load_config`` calls ``_load_dotenv``, which writes the
    file's keys straight into ``os.environ`` and never takes them out again.
    A test that loads a config from a temporary .env therefore leaks those
    values into every test that runs after it -- which surfaced as a config
    test reading another module's SPREAD_PIPS and failing only when the whole
    suite ran.

    Wrap any test that calls ``load_config`` in this.
    """
    snapshot = dict(os.environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(snapshot)
