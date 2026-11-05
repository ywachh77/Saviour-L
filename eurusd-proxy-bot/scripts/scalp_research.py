"""Measure what each scalping member actually does, gross and net of spread.

Why this exists
---------------

There are 26 scalping prototypes and, until now, no measurement of any of them
against a held-out series. They were written as patterns that look like they
should work, wired in as prototypes so learning could discover whether they do,
and then never checked. That is exactly the arrangement where a rule that loses
every time gets to keep firing, because its failures are absorbed by the
prototype weight cap rather than by anything noticing.

This script reports, per member:

* how often it speaks (a rule that fires on 4% of bars is not a scalping rule,
  it is a rare event, and its win rate will be measured on nothing);
* its gross win rate and mean gross pips, which is what the pattern does;
* its **net** win rate and mean net pips after the round-turn spread, which is
  what the account does.

What the comparison says, before any of it is run
-------------------------------------------------

A 1.2 pip spread on a scalp that captures ~1.5 pips of move loses money at any
win rate below 90%, because the spread is a constant subtraction on every trade.
Concretely, for a member whose gross wins and losses are the same size ``m``::

    net per trade = p * (m - s) - (1 - p) * (m + s) = m * (2p - 1) - s

so breaking even needs ``p = 0.5 + s / (2m)``. At m = 1.5 pips and s = 1.2 that
is 90%. The ``--required`` column below prints that number for every member, so
the achieved win rate can be read against the rate the spread demands rather
than against 50%.

Usage
-----

    python3 scripts/scalp_research.py                 # 5 paths, 4000 bars each
    python3 scripts/scalp_research.py --paths 9 --bars 6000
    python3 scripts/scalp_research.py --seeds 11,21,31
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fusion import charter, scalping  # noqa: E402
from fusion.features import compute_features  # noqa: E402
from fusion.instruments import for_symbol  # noqa: E402
from fusion.simulate import synthetic_series  # noqa: E402

# Hold lengths in minutes to test per member. A scalp holds for a couple of
# minutes at most; anything longer is a swing trade wearing a scalp's name.
HORIZONS = (1, 2, 3, 5)


class Tally:
    """Counts for one (member, horizon) cell."""

    __slots__ = ("spoken", "settled", "wins", "ties", "gross", "net", "by_regime")

    def __init__(self) -> None:
        self.spoken = 0
        self.settled = 0
        self.wins = 0
        self.ties = 0
        self.gross = 0.0
        self.net = 0.0
        self.by_regime: dict[str, list[float]] = defaultdict(list)

    def add(self, gross_pips: float, spread_pips: float, regime: str) -> None:
        self.settled += 1
        if abs(gross_pips) < 1e-12:
            self.ties += 1
            return
        if gross_pips > 0:
            self.wins += 1
        self.gross += gross_pips
        self.net += gross_pips - spread_pips
        self.by_regime[regime].append(gross_pips - spread_pips)

    @property
    def decided(self) -> int:
        return self.settled - self.ties

    @property
    def win_rate(self) -> float:
        return (self.wins / self.decided) if self.decided else 0.0

    @property
    def mean_gross(self) -> float:
        return (self.gross / self.decided) if self.decided else 0.0

    @property
    def mean_net(self) -> float:
        return (self.net / self.decided) if self.decided else 0.0


def required_win_rate(mean_abs_gross: float, spread_pips: float) -> float | None:
    """The hit rate a symmetric payoff needs to pay the spread, or None."""
    if mean_abs_gross <= 0.0:
        return None
    needed = 0.5 + spread_pips / (2.0 * mean_abs_gross)
    return needed if needed <= 1.0 else None


def _regime_bucket(features: dict[str, float]) -> str:
    """Coarse regime label from the ADX the features already carry."""
    adx = float(features.get("adx", 0.0) or 0.0)
    if adx >= 25.0:
        return "TREND"
    if adx >= 18.0:
        return "MIXED"
    return "RANGE"


def measure(
    member: str,
    candles,
    rows,
    *,
    spread_pips: float,
    pip: float,
    horizons=HORIZONS,
) -> dict[int, Tally]:
    """Run one member over one series and tally every horizon."""
    fn = scalping.SCALING_REGISTRY[member] if hasattr(scalping, "SCALING_REGISTRY") else scalping.SCALPING_REGISTRY[member]
    tallies = {h: Tally() for h in horizons}
    last_index = len(candles) - 1
    for index in range(1, last_index):
        features = rows[index]
        if features is None:
            continue
        decision = fn(candles[index], candles[index - 1].close, features, None, spread_pips=spread_pips)
        if decision.side not in ("BUY", "SELL"):
            continue
        for horizon in horizons:
            tallies[horizon].spoken += 1
            exit_index = index + horizon
            if exit_index > last_index:
                continue
            # Same settlement rule the backtest uses: enter at the OPEN of the
            # next bar, exit at the CLOSE of the bar ``horizon`` minutes later.
            # Anything else measures a fill nobody could have got.
            entry = candles[index + 1].open
            exit_price = candles[exit_index].close
            move = exit_price - entry
            gross = (move if decision.side == "BUY" else -move) / pip
            tallies[horizon].add(gross, spread_pips, _regime_bucket(features))
    return tallies


def run(seeds, bars: int, spread_pips: float, volatility: float) -> None:
    instrument = for_symbol("EUR/USD")
    pip = instrument.pip
    merged: dict[str, dict[int, Tally]] = {
        member: {h: Tally() for h in HORIZONS} for member in scalping.SCALPING_PROTOTYPES
    }
    total_bars = 0

    for seed in seeds:
        candles = synthetic_series(bars, seed=seed, volatility=volatility)
        rows = compute_features(candles, instrument=instrument)
        total_bars += bars
        for member in scalping.SCALPING_PROTOTYPES:
            for horizon, tally in measure(
                member, candles, rows, spread_pips=spread_pips, pip=pip
            ).items():
                target = merged[member][horizon]
                target.spoken += tally.spoken
                target.settled += tally.settled
                target.wins += tally.wins
                target.ties += tally.ties
                target.gross += tally.gross
                target.net += tally.net
                for regime, values in tally.by_regime.items():
                    target.by_regime[regime].extend(values)
        print(f"  path seed={seed} done ({total_bars} bars measured)", flush=True)

    print()
    print(
        f"EUR/USD, {total_bars} one-minute bars over {len(seeds)} independent paths, "
        f"round-turn spread {spread_pips:.2f} pips."
    )
    print("Net is what the account keeps. 'needed' is the hit rate the spread")
    print("demands for a payoff of that size; anything below it loses money.")
    print()
    header = (
        f"{'member':26s} {'h':>2s} {'spoke':>7s} {'rate%':>6s} {'win%':>6s} "
        f"{'needed%':>8s} {'gross':>7s} {'net':>7s}"
    )
    print(header)
    print("-" * len(header))

    ranking: list[tuple[float, str, int, Tally]] = []
    for member in scalping.SCALPING_PROTOTYPES:
        for horizon in HORIZONS:
            tally = merged[member][horizon]
            if not tally.decided:
                continue
            mean_abs = sum(abs(value) for value in tally.by_regime.get("ALL", []))
            # mean absolute gross move, for the required-rate column
            abs_sum = 0.0
            abs_count = 0
            for values in tally.by_regime.values():
                for value in values:
                    abs_sum += abs(value)
                    abs_count += 1
            mean_abs = (abs_sum / abs_count) if abs_count else abs(tally.mean_gross)
            needed = required_win_rate(mean_abs, spread_pips)
            rate = 100.0 * tally.spoken / max(1, total_bars)
            needed_txt = f"{needed * 100:8.1f}" if needed is not None else "     n/a"
            print(
                f"{member:26s} {horizon:2d} {tally.spoken:7d} {rate:6.2f} "
                f"{tally.win_rate * 100:6.1f} {needed_txt} "
                f"{tally.mean_gross:7.2f} {tally.mean_net:7.2f}"
            )
            ranking.append((tally.mean_net, member, horizon, tally))

    print()
    print("Worst members by mean net pips per settled trade (all horizons pooled):")
    pooled: dict[str, tuple[float, int]] = {}
    for member in scalping.SCALPING_PROTOTYPES:
        net = 0.0
        decided = 0
        for horizon in HORIZONS:
            tally = merged[member][horizon]
            net += tally.net
            decided += tally.decided
        if decided:
            pooled[member] = (net / decided, decided)
    for member, (mean_net, decided) in sorted(pooled.items(), key=lambda item: item[1][0])[:26]:
        verdict = "loses" if mean_net < 0 else "pays"
        print(f"  {member:26s} {mean_net:+7.3f} pips/trade over {decided:6d} settled  {verdict}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=int, default=5, help="number of independent synthetic paths")
    parser.add_argument("--bars", type=int, default=4000, help="bars per path")
    parser.add_argument("--seeds", default="", help="comma-separated seeds (overrides --paths)")
    parser.add_argument("--spread", type=float, default=charter.DEFAULT_SPREAD_PIPS)
    parser.add_argument(
        "--volatility",
        type=float,
        default=0.000073,
        help="per-bar synthetic volatility; 0.000073 puts the M1 ATR near 1.8 pips, "
        "which is the measured EUR/USD scale",
    )
    args = parser.parse_args(argv)

    if args.seeds.strip():
        seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    else:
        seeds = [101 + 13 * index for index in range(max(1, args.paths))]

    print(f"Measuring {len(scalping.SCALPING_PROTOTYPES)} scalping members over {len(seeds)} paths...")
    print(
        f"cost guard: a scalp needs ATR >= {charter.SCALP_MIN_REWARD_TO_COST * args.spread / charter.SCALP_ATR_CAPTURE:.2f} "
        f"pips to speak at a {args.spread:.2f} pip spread"
    )
    run(seeds, args.bars, args.spread, args.volatility)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
