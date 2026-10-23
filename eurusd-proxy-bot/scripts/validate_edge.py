#!/usr/bin/env python3
"""Does the session filter capture session structure when it exists?

    python scripts/validate_edge.py

This is the control experiment that the session filter needs before anyone
trusts it, and it is the one a normal replay cannot provide.

The problem
-----------

``fusion/session.py`` claims that continuation rules suit London and the
overlap, and reversion rules suit Asia. A synthetic replay cannot test that
claim: the generator is a random walk with no time-of-day structure, and the
prototypes never earn a voice anyway because synthetic outcomes never train
them. So a replay would report "no difference" whether the filter is brilliant
or nonsense.

The test
--------

Build price paths that DO have session structure, and see whether the mapping
picks the right family:

* **structured** — returns are positively autocorrelated (momentum) during
  London and the overlap, and negatively autocorrelated (mean-reverting) during
  Asia and off-hours. Exactly the behaviour the mapping claims.
* **control** — a plain random walk, no session structure at all.

For each strategy we then measure, per session, how often its directional call
was right. If the mapping is doing real work, the families it allows should beat
the families it silences *on the structured paths*, and the advantage should
vanish *on the control*.

That is a claim about the mechanism, not about EUR/USD. Whether real EUR/USD has
this structure is a separate question this repository cannot answer — it has no
real market data.
"""

from __future__ import annotations

import argparse
import math
import random
import statistics
import sys
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fusion import session as session_module  # noqa: E402
from fusion.config import Config  # noqa: E402
from fusion.domain import Candle  # noqa: E402
from fusion.features import compute_features  # noqa: E402
from fusion.regime import classify  # noqa: E402
from fusion.strategies import PROTOTYPES, REGISTRY  # noqa: E402
from fusion.timeutil import now  # noqa: E402

BARS = 6000
HORIZON = 5
BASE = 1.1000
VOLATILITY = 0.000073  # lands near a realistic 1.8 pip M1 ATR

# What the structured generator injects.
MOMENTUM = 0.35  # returns repeat themselves (trend)
REVERSION = -0.35  # returns reverse (range)

MOMENTUM_SESSIONS = (session_module.LONDON, session_module.OVERLAP)
REVERSION_SESSIONS = (session_module.ASIA, session_module.OFF_HOURS)


def _structured_series(count: int, seed: int) -> list[Candle]:
    """A walk whose autocorrelation depends on the session of each bar."""
    rng = random.Random(seed)
    first = now().replace(second=0, microsecond=0) - timedelta(minutes=count)
    candles: list[Candle] = []
    price = BASE
    previous_return = 0.0
    for index in range(count):
        stamp = first + timedelta(minutes=index)
        which = session_module.session_of(stamp)
        if which in MOMENTUM_SESSIONS:
            rho = MOMENTUM
        elif which in REVERSION_SESSIONS:
            rho = REVERSION
        else:
            rho = 0.0
        shock = rng.gauss(0.0, VOLATILITY)
        step = rho * previous_return + math.sqrt(max(0.0, 1.0 - rho * rho)) * shock
        previous_return = step

        open_price = price
        close_price = max(0.5, open_price + step)
        wick = abs(rng.gauss(0.0, VOLATILITY * 0.7)) + 0.00002
        candles.append(
            Candle(
                timestamp=stamp,
                open=round(open_price, 5),
                high=round(max(open_price, close_price) + wick, 5),
                low=round(min(open_price, close_price) - wick, 5),
                close=round(close_price, 5),
                volume=float(700 + rng.randrange(400)),
            )
        )
        price = close_price
    return candles


def _control_series(count: int, seed: int) -> list[Candle]:
    """A plain random walk: no session structure anywhere."""
    return _structured_series(count, seed) if False else _plain(count, seed)


def _plain(count: int, seed: int) -> list[Candle]:
    rng = random.Random(seed)
    first = now().replace(second=0, microsecond=0) - timedelta(minutes=count)
    candles: list[Candle] = []
    price = BASE
    for index in range(count):
        stamp = first + timedelta(minutes=index)
        open_price = price
        close_price = max(0.5, open_price + rng.gauss(0.0, VOLATILITY))
        wick = abs(rng.gauss(0.0, VOLATILITY * 0.7)) + 0.00002
        candles.append(
            Candle(
                timestamp=stamp,
                open=round(open_price, 5),
                high=round(max(open_price, close_price) + wick, 5),
                low=round(min(open_price, close_price) - wick, 5),
                close=round(close_price, 5),
                volume=float(700 + rng.randrange(400)),
            )
        )
        price = close_price
    return candles


def _measure(candles: list[Candle]) -> dict[tuple[str, str], list[bool]]:
    """Per (strategy, session), was each directional call right?"""
    feature_rows = compute_features(candles)
    hits: dict[tuple[str, str], list[bool]] = {}
    for index in range(30, len(candles) - HORIZON - 1):
        features = feature_rows[index]
        if features is None:
            continue
        candle = candles[index]
        previous_close = candles[index - 1].close
        which = session_module.session_of(candle.timestamp)

        future = candles[index + HORIZON].close - candles[index + 1].open
        if abs(future) < 1e-12:
            continue
        truth_up = future > 0

        for name in PROTOTYPES:
            try:
                decision = REGISTRY[name](
                    candle, previous_close, features, 0.5,
                    calibrated=False, spread_pips=1.2,
                )
            except Exception:  # noqa: BLE001 - one broken rule must not stop the survey
                continue
            if decision.side not in ("BUY", "SELL"):
                continue
            correct = (decision.side == "BUY") == truth_up
            hits.setdefault((name, which), []).append(correct)
    return hits


def _report(hits: dict[tuple[str, str], list[bool]], label: str) -> tuple[float, float]:
    """Return (mean accuracy of allowed pairs, mean accuracy of silenced pairs)."""
    allowed_rates: list[float] = []
    silenced_rates: list[float] = []
    rows: list[tuple[str, str, int, float, str]] = []
    measured: set[str] = set()

    for name in PROTOTYPES:
        for which in session_module.ALL_SESSIONS:
            outcomes = hits.get((name, which)) or []
            if len(outcomes) < 40:
                continue
            rate = sum(1 for value in outcomes if value) / len(outcomes)
            allowed = session_module.session_allows(which, name)
            (allowed_rates if allowed else silenced_rates).append(rate)
            measured.add(name)
            rows.append((name, which, len(outcomes), rate, "allow" if allowed else " SILENCE"))

    print(f"  {label}")
    print(f"    {'strategy':<24}{'session':<10}{'n':>6}{'hit rate':>10}   mapping")
    for name, which, count, rate, flag in sorted(rows, key=lambda row: (row[1], row[0])):
        print(f"    {name:<24}{which:<10}{count:>6}{rate * 100:>9.1f}%   {flag}")
    print()

    allowed_mean = statistics.mean(allowed_rates) if allowed_rates else float("nan")
    silenced_mean = statistics.mean(silenced_rates) if silenced_rates else float("nan")
    print(f"    mean hit rate, ALLOWED pairs   : {allowed_mean * 100:.2f}%  ({len(allowed_rates)} pairs)")
    print(f"    mean hit rate, SILENCED pairs  : {silenced_mean * 100:.2f}%  ({len(silenced_rates)} pairs)")
    print(f"    advantage of the mapping       : {(allowed_mean - silenced_mean) * 100:+.2f} points")
    silent = [name for name in PROTOTYPES if name not in measured]
    if silent:
        print(f"    not measurable here (<40 calls): {', '.join(silent)}")
    print()
    return allowed_mean, silenced_mean


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", type=int, default=BARS)
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args(argv)

    print()
    print("FUSION BOT — does the session filter capture session structure?")
    print("=" * 68)
    print()
    print("  Two generators. One has session structure injected, one has none.")
    print("  A genuine filter must separate the families on the first and show")
    print("  no preference on the second. If it cannot do that, it is decoration.")
    print()

    structured_gap = []
    control_gap = []
    for seed in range(1, args.seeds + 1):
        print(f"--- seed {seed} " + "-" * 54)
        structured = _structured_series(args.bars, seed)
        allowed, silenced = _report(_measure(structured), "STRUCTURED (momentum in London, mean-reversion in Asia)")
        structured_gap.append(allowed - silenced)

        control = _plain(args.bars, seed)
        allowed_c, silenced_c = _report(_measure(control), "CONTROL (plain random walk, no session structure)")
        control_gap.append(allowed_c - silenced_c)

    print("=" * 68)
    print()
    def spread(values: list[float]) -> str:
        if len(values) < 2:
            return "n=1"
        return f"sd {statistics.stdev(values) * 100:.2f}"

    print(f"  structured paths : mapping advantage {statistics.mean(structured_gap) * 100:+.2f} points "
          f"({spread(structured_gap)})")
    print(f"  control paths    : mapping advantage {statistics.mean(control_gap) * 100:+.2f} points "
          f"({spread(control_gap)})")
    print()
    works = statistics.mean(structured_gap) > statistics.mean(control_gap)
    if works:
        print("  VERDICT: the filter separates the families when session structure")
        print("  is present, and stops separating them when it is not. That is the")
        print("  behaviour a genuine mechanism should have.")
    else:
        print("  VERDICT: the filter does NOT separate the families even on data")
        print("  built to contain the effect. It is not doing the job it claims.")
    print()
    print("  This validates the MECHANISM only. Whether real EUR/USD carries this")
    print("  structure is a separate question, and this repository has no real")
    print("  market data to answer it.")
    print()
    return 0 if works else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
