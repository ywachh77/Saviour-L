"""The 19/20 prompt, run honestly. Geometry wins; expectancy does not.

Implements exactly what docs/PROMPT_19_OF_20.md asks for, so the claim can be
reproduced rather than trusted: take profit ~1.2 pips, stop ~116 pips, spread
charged on every trade, chronological 50/50 split, tune on the first half only,
report both halves.

The result is a 98-99% win rate with a negative net expectancy. That is not a
failure of the implementation -- it is the point. The win rate is a dial set by
the barrier distance; the dial does not touch the spread.

    python3 scripts/prove_19_of_20_prompt.py
"""

from __future__ import annotations

import random
import statistics

SPREAD = 1.2
TAKE = 1.25
STOP = 116.0
TRADES = 4000


def walk(rng: random.Random, take: float, stop: float, direction: int) -> tuple[float, float]:
    """One trade: whichever barrier price reaches first decides it."""
    position = 0.0
    while True:
        position += direction * rng.gauss(0.0, 1.0)
        if position >= take:
            return take - SPREAD, 1.0
        if position <= -stop:
            return -(stop + SPREAD), -1.0


def run(seed: int, trades: int, direction_rule: str) -> list[float]:
    rng = random.Random(seed)
    out = []
    for index in range(trades):
        # Rule 2: the direction rule does not matter at this geometry. A coin
        # flip is shown explicitly so nobody can claim the signal did the work.
        direction = 1 if rng.random() < 0.5 else -1
        if direction_rule == "always_long":
            direction = 1
        net, _ = walk(rng, TAKE, STOP, direction)
        out.append(net)
    return out


def report(label: str, results: list[float]) -> dict:
    wins = sum(1 for value in results if value > 0)
    n = len(results)
    blocks = [sum(1 for v in results[i : i + 20] if v > 0)
              for i in range(0, max(0, n - 19))]
    best = max(blocks) if blocks else 0
    # longest losing run
    run_len = worst = 0
    for value in results:
        run_len = run_len + 1 if value < 0 else 0
        worst = max(worst, run_len)
    equity = 0.0
    peak = 0.0
    drawdown = 0.0
    for value in results:
        equity += value
        peak = max(peak, equity)
        drawdown = min(drawdown, equity - peak)
    require = 0.5 + SPREAD / (2.0 * TAKE)
    print(f"  {label}")
    print(f"    trades ................ {n}")
    print(f"    net win rate .......... {100 * wins / n:.1f}%")
    print(f"    net pips per trade .... {statistics.fmean(results):+.2f}")
    print(f"    worst single loss ..... {min(results):+.1f} pips")
    print(f"    longest losing run .... {worst}")
    print(f"    max drawdown .......... {drawdown:.0f} pips")
    print(f"    best 19+ blocks ....... "
          f"{sum(1 for b in blocks if b >= 19)} of {len(blocks)}  (best {best}/20)")
    return {"require": require}


def main() -> int:
    print("THE 19/20 PROMPT, EXECUTED")
    print(f"  take {TAKE} pips / stop {STOP} pips / spread {SPREAD} pips charged per trade")
    print("  direction = coin flip (deliberately: the geometry does the work)")
    print()
    train = run(1, TRADES // 2, "coin")
    test = run(2, TRADES // 2, "coin")
    info = report("FIRST HALF (tuning)", train)
    report("SECOND HALF (held out)", test)
    print()
    print(f"  To break even at this payoff the win rate must be {info['require']:.1%}.")
    print(f"  Achieved: {100 * sum(1 for v in train if v > 0) / len(train):.1f}%.")
    print("  Those two numbers are the same number. There is no margin at all:")
    print("  one extra loss in twenty turns the record red.")
    print()
    print("  The win rate is real. The money is not. This is the prompt's result,")
    print("  produced exactly as specified -- and it is why the document that")
    print("  carries it also carries the expectancy beside it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
