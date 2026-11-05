"""What it would actually take to reach a 95% win rate.

The scalping measurement established the problem: a ~1.4 pip mean move against
a 1.2 pip round turn needs 92.9% to break even, and the rules deliver 50%.
This script answers the follow-up question -- "so how do we achieve it?" --
by working out, quantitatively, which levers move that required rate.

Four things are measured, in the order they matter:

1. **How the payoff grows with hold.** A fixed-expiry trade (which is what
   ``fusion.settle`` settles) earns the mean absolute move over its horizon.
   That grows roughly with sqrt(time), so a longer hold widens the payoff
   without needing a better forecast. This is the one lever the system already
   has and the reason the adaptive-horizon work moves the needle.

2. **The required hit rate as a function of hold and spread.** p = 0.5 + s/(2m).
   This is the table that says 95% needs either a big payoff or a small spread,
   and it says by how much.

3. **The stop/target trap.** The tempting way to reach a high win rate is a
   small target and a wide stop. On a random walk the hit rate at that geometry
   is 1/(1+k) where k is reward-to-risk -- *independent of the forecast*. A 95%
   hit rate therefore implies a 0.053:1 payoff, and once the spread is charged
   the trade loses. This is measured, not asserted.

4. **Why a 19-of-20 search on 20 trades proves nothing.** If you tune any
   threshold until it produces 19 wins in 20 trades, you will find one. This
   simulates that search on a pure random walk with *no* edge and reports how
   often a 19/20 appears and what it does on fresh data. That is precisely the
   mechanism that would be used to "achieve" 95%, and this shows its real
   content.

Run:

    python3 scripts/scalp_geometry_study.py
    python3 scripts/scalp_geometry_study.py --paths 40 --bars 4000
"""

from __future__ import annotations

import argparse
import math
import random
import statistics
import sys
from pathlib import Path

BOT_ROOT = Path(__file__).resolve().parents[1]
if str(BOT_ROOT) not in sys.path:
    sys.path.insert(0, str(BOT_ROOT))

from fusion import charter  # noqa: E402
from fusion.costs import break_even_win_rate_from_move  # noqa: E402
from fusion.instruments import for_symbol  # noqa: E402
from fusion.simulate import synthetic_series  # noqa: E402

PIP = for_symbol("EURUSD").pip


# --------------------------------------------------------------------------
# 1. How the payoff grows with hold
# --------------------------------------------------------------------------


def mean_move_by_hold(paths: list[list[float]], horizons: list[int]) -> dict[int, float]:
    """Mean absolute move in pips over each hold, measured on the paths."""
    out: dict[int, float] = {}
    for horizon in horizons:
        moves: list[float] = []
        for prices in paths:
            for index in range(len(prices) - horizon):
                moves.append(abs(prices[index + horizon] - prices[index]) / PIP)
        if moves:
            out[horizon] = statistics.fmean(moves)
    return out


def build_paths(count: int, bars: int, volatility: float, seed0: int) -> list[list[float]]:
    paths: list[list[float]] = []
    for index in range(count):
        candles = synthetic_series(bars, seed=seed0 + index * 7, volatility=volatility)
        paths.append([candle.close for candle in candles])
    return paths


# --------------------------------------------------------------------------
# 2. The required hit rate, as a table
# --------------------------------------------------------------------------


def required_table(moves: dict[int, float], spreads: list[float], horizons: list[int]) -> list[str]:
    header = f"{'hold':>6} | {'mean |move|':>11} |" + "".join(
        f"  p needed @ {spread:.2f} pip" for spread in spreads
    )
    lines = [header, "-" * len(header)]
    for horizon in horizons:
        move = moves.get(horizon)
        if not move:
            continue
        cells = []
        for spread in spreads:
            needed = break_even_win_rate_from_move(move, spread)
            cells.append(f"{'never':>20}" if math.isinf(needed) else f"{needed:>20.1%}")
        lines.append(f"{horizon:>4}m | {move:>10.2f}p |" + "".join(cells))
    return lines


# --------------------------------------------------------------------------
# 3. The stop/target trap: hit rate is set by geometry, not by the forecast
# --------------------------------------------------------------------------


def first_passage_hit_rate(
    paths: list[list[float]],
    *,
    reward_atr: float,
    risk_atr: float,
    atr_pips: float,
    spread_pips: float,
    cap: int = 600,
    stride: int = 5,
) -> dict[str, float]:
    """Hit rate and expectancy for a fixed target / fixed stop, on real paths.

    Entries are taken every ``stride`` bars in the direction of a coin flip: the
    point is to show that *geometry* sets the hit rate, so the entry signal is
    deliberately neutral. Whichever of target or stop the price reaches first
    decides the trade.

    ``cap`` is deliberately generous (600 bars). A short cap is a trap: trades
    that would eventually have been stopped are cut off at the close instead,
    which flatters wide-stop geometries and makes a losing system look like it
    breaks even. That artefact was present in an earlier version of this study
    and is the reason the cap is stated here rather than left implicit.
    """
    target = reward_atr * atr_pips
    # A market order pays the spread on entry, so the price must move
    # target + spread to bank the target, and only stop - spread to lose the
    # stop. The spread is charged against the trade on both sides.
    bankable = target + spread_pips
    losable = max(0.1, risk_atr * atr_pips - spread_pips)
    wins = losses = unsettled = 0
    net_pips = 0.0

    for prices in paths:
        for start in range(0, len(prices) - cap - 1, stride):
            entry = prices[start]
            direction = 1.0 if (start // stride) % 2 == 0 else -1.0
            result: float | None = None
            hanging = 0.0
            for step in range(1, cap + 1):
                moved = direction * (prices[start + step] - entry) / PIP
                hanging = moved
                if moved >= bankable:
                    result = target
                    break
                if moved <= -losable:
                    result = -(risk_atr * atr_pips)
                    break
            if result is None:
                unsettled += 1
                result = hanging - spread_pips
            net_pips += result
            if result > 0:
                wins += 1
            elif result < 0:
                losses += 1

    decided = wins + losses
    return {
        "wins": float(wins),
        "losses": float(losses),
        "unsettled": float(unsettled),
        "hit_rate": (wins / decided) if decided else 0.0,
        "net_per_trade": net_pips / decided if decided else 0.0,
    }


def payoff_needed_for_95(spread_pips: float) -> float:
    """Reward-to-risk a 95% hit rate implies, once spread is charged.

    At 95% you are wrong 1 time in 20. For the book to be flat:

        0.95 * (m - s) = 0.05 * (m + s)   =>   m = s * (1.05 / 0.90) = 1.167 s

    i.e. a 1.2 pip spread implies a 1.4 pip average move -- and that is only
    break-even, before any forecast error. To actually earn a sensible amount
    per trade the move must be several times that, which drags the required
    hit rate back down. The "95%" and "worth doing" conditions pull against
    each other, and this function is the arithmetic of that conflict.
    """
    return spread_pips * (1.05 / 0.90)


# --------------------------------------------------------------------------
# 4. Why a 19-of-20 search on twenty trades proves nothing
# --------------------------------------------------------------------------


def search_for_19_of_20(
    *,
    paths: list[list[float]],
    hold: int,
    atr_pips: float,
    rng: random.Random,
) -> dict[str, float]:
    """The "tune until 19 of 20 win" search, run on a random walk with no edge.

    Honesty requirement: count only trades actually taken. An earlier draft of
    this experiment counted *skipping* a trade as a win -- exactly the
    accounting trick this section exists to expose, since it makes any hard
    filter look like a 95% system in both splits.

    So the tunable parameter is a **selectivity threshold** on the previous
    move's size (a real thing a scalper tunes), and a configuration trades only
    on bars clearing it. Direction is a coin flip, because on a random walk
    past range says nothing about the direction of the next move.

    A block is twenty consecutive taken trades and "wins" if nineteen or more
    of them were profitable gross. On a true coin flip that is p = 2.0e-5 per
    block. The question is how many such blocks a *search* manufactures, and
    whether they replicate on bars the search never saw.
    """
    thresholds = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]

    def taken_moves(series: list[float], threshold: float) -> list[float]:
        """Gross pips of every trade this threshold would have taken."""
        moves: list[float] = []
        index = hold
        while index < len(series) - hold:
            prev = abs(series[index] - series[index - hold]) / PIP
            if prev >= threshold:
                direction = 1.0 if rng.random() < 0.5 else -1.0
                moves.append(direction * (series[index + hold] - series[index]) / PIP)
                index += hold
            else:
                index += 1
        return moves

    def blocks(moves: list[float], stride: int = 1) -> list[int]:
        return [
            sum(1 for move in moves[offset:offset + 20] if move > 0)
            for offset in range(0, max(0, len(moves) - 19), stride)
        ]

    # 1. In-sample: every threshold against every even (training) path.
    tested = hits = best = 0
    hit_thresholds: list[float] = []
    train_paths = [series for index, series in enumerate(paths) if index % 2 == 0]
    for threshold in thresholds:
        for series in train_paths:
            found = blocks(taken_moves(series, threshold))
            if not found:
                continue
            tested += len(found)
            best = max(best, max(found))
            for wins in found:
                if wins >= 19:
                    hits += 1
                    hit_thresholds.append(threshold)

    # 2. Replication: same threshold, fresh bars, no peeking -- one block each.
    checked = replications = 0
    out_of_sample: list[int] = []
    odd_paths = [series for index, series in enumerate(paths) if index % 2]
    for threshold in dict.fromkeys(hit_thresholds):
        for series in odd_paths:
            found = blocks(taken_moves(series, threshold), stride=20)
            if not found:
                continue
            out_of_sample.append(found[0])  # the first fresh block, unscored
            checked += 1
            if found[0] >= 19:
                replications += 1

    return {
        "configs_tested": float(tested),
        "in_sample_19": float(hits),
        "in_sample_rate": hits / tested if tested else 0.0,
        "best_in_sample": float(best),
        "held_out_checked": float(checked),
        "held_out_19": float(replications),
        "held_out_mean": statistics.fmean(out_of_sample) if out_of_sample else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=int, default=12)
    parser.add_argument("--bars", type=int, default=3000)
    parser.add_argument("--volatility", type=float, default=0.00024)
    parser.add_argument("--seed", type=int, default=900)
    parser.add_argument("--trials", type=int, default=20000)
    args = parser.parse_args()

    horizons = [1, 2, 3, 5, 10, 15, 30, 60]
    spreads = [1.2, 0.7, 0.3, 0.1]
    atr_pips = args.volatility / PIP

    print("SCALPING GEOMETRY STUDY")
    print(f"  {args.paths} paths x {args.bars} bars, one-minute, ATR ~ {atr_pips:.2f} pips")
    print(f"  contract: EUR/USD, 1 pip = {PIP}, one-minute settlement")
    print()

    paths = build_paths(args.paths, args.bars, args.volatility, args.seed)
    moves = mean_move_by_hold(paths, horizons)

    print("1. HOW THE PAYOFF GROWS WITH HOLD")
    print("   The system settles at a fixed expiry, so the mean absolute move over")
    print("   that expiry is the payoff. Holding longer widens it without a better")
    print("   forecast -- sqrt(time), so it is slow but it is free and real.")
    print()
    print(f"   {'hold':>6} | {'mean |move|':>11} | {'cost share of move':>19}")
    print("   " + "-" * 44)
    for horizon in horizons:
        move = moves.get(horizon)
        if not move:
            continue
        print(
            f"   {horizon:>4}m | {move:>10.2f}p | "
            f"{charter.DEFAULT_SPREAD_PIPS / move:>18.1%}"
        )
    print()

    print("2. THE REQUIRED HIT RATE, BY HOLD AND BY SPREAD")
    print("   p_break_even = 0.5 + spread / (2 x mean move).")
    print("   Read across a row: the same forecast gets easier as the spread falls.")
    print("   Read down a column: the same forecast gets easier as the hold grows.")
    print()
    for line in required_table(moves, spreads, horizons):
        print("   " + line)
    print()

    print("3. THE STOP/TARGET TRAP -- A HIGH WIN RATE IS GEOMETRY, NOT SKILL")
    print("   A small target and a wide stop produce a high hit rate on a random")
    print("   walk, with no forecast at all. Measured here on the same paths,")
    print("   entries taken in the direction of a coin flip. 'open' counts")
    print("   trades that touched neither level before the 600-bar cap:")
    print()
    print(f"   {'target':>7} {'stop':>7} | {'hit rate':>9} {'1/(1+k)':>8} {'net/trade':>10} {'open':>5}")
    print("   " + "-" * 55)
    for reward, risk in ((2.0, 0.5), (1.5, 0.5), (1.0, 1.0), (0.5, 2.0), (0.25, 5.0)):
        stats = first_passage_hit_rate(
            paths,
            reward_atr=reward,
            risk_atr=risk,
            atr_pips=atr_pips,
            spread_pips=charter.DEFAULT_SPREAD_PIPS,
        )
        theoretical = 1.0 / (1.0 + reward / risk)
        print(
            f"   {reward:>7.2f} {risk:>7.2f} | {stats['hit_rate']:>8.1%} "
            f"{theoretical:>8.1%} {stats['net_per_trade']:>+9.2f}p "
            f"{int(stats['unsettled']):>5}"
        )
    print()
    print("   The hit rate tracks 1/(1+k) in every row: it is set by where you put")
    print("   the target, not by the forecast. A 95% hit rate means a 0.053:1")
    print("   payoff, and the spread is charged on every one of those trades.")
    print()

    print("4. WHAT A 95% HIT RATE IMPLIES, ARITHMETICALLY")
    spread = charter.DEFAULT_SPREAD_PIPS
    break_even_move = payoff_needed_for_95(spread)
    print(f"   At 95% you are wrong once in twenty, so a flat book needs")
    print(f"      0.95 x (m - s) = 0.05 x (m + s)  ->  m = {(1.05 / 0.90):.3f} x s")
    print(f"   At a {spread:.2f} pip spread that is a {break_even_move:.2f} pip mean move -- and")
    print(f"   that is break-even, with zero forecast error.")
    print()
    print("   To merely reach the project's 3:1 reward-to-cost floor at 95%:")
    target_move = 3.0 * spread
    needed = break_even_win_rate_from_move(target_move, spread)
    print(f"      a {target_move:.2f} pip mean move needs only {needed:.1%} -- so the win rate")
    print(f"      requirement COLLAPSES as the payoff grows. That is the lever.")
    print()

    print("5. WHY A 19-OF-20 BLOCK IS NOT EVIDENCE")
    print("   Tuning a threshold until 20 trades contain 19 wins, on a random")
    print("   walk with no edge at all (a fair process gives 19/20 about 2.0e-5")
    print("   of the time):")
    print()
    rng = random.Random(4242)
    result = search_for_19_of_20(
        paths=paths,
        hold=5,
        atr_pips=atr_pips,
        rng=rng,
    )
    print(f"      {int(result['configs_tested']):,} threshold/window configurations tested in-sample")
    print(f"      {int(result['in_sample_19']):,} of them produced 19 or 20 wins out of 20")
    print(f"      ({result['in_sample_rate']:.1%} of configs -- on data with no edge)")
    print(f"      best in-sample block: {int(result['best_in_sample'])}/20")
    print()
    print(f"   Those 'discoveries', re-scored on bars they were not tuned on")
    print(f"   ({int(result['held_out_checked'])} blocks):")
    print(f"      {int(result['held_out_19'])} still produced 19/20")
    print(f"      mean block out of sample: {result['held_out_mean']:.1f}/20")
    print()
    print("   The in-sample column is what a 19/20 target gets you: a search")
    print("   that finds one. The out-of-sample column is what the system")
    print("   actually does: revert to chance. A 19/20 block is what a search")
    print("   PRODUCES, not what a system IS. This is why every candidate in")
    print("   the edge search is judged on its held-out score, and why that")
    print("   score killed all twenty of them.")
    print()

    print("6. SO WHAT WOULD ACTUALLY ACHIEVE A HIGH WIN RATE")
    print("   Three routes, in order of how much of the arithmetic they move:")
    print()
    for hold in (5, 15, 30, 60):
        move = moves.get(hold) or 0.0
        if not move:
            continue
        for label, spread_now in (("current broker", 1.2), ("institutional", 0.3)):
            needed = break_even_win_rate_from_move(move, spread_now)
            shown = "never" if math.isinf(needed) else f"{needed:.1%}"
            print(f"      hold {hold:>2}m, {label:<15} -> needs {shown:>6} to break even")
        print()
    print("   Route 1 (free, already in the code): hold longer. m grows with")
    print("   sqrt(hold), so the required rate falls with it. 60 minutes needs")
    print(f"   {break_even_win_rate_from_move(moves.get(60, 1.0), spread):.1%} where one minute needs 92.9%.")
    print("   That is not scalping any more, and that is the point.")
    print()
    print("   Route 2 (also free): pay less spread. At 0.3 pips the one-minute")
    print(f"   trade needs {break_even_win_rate_from_move(moves.get(1, 1.0), 0.3):.1%} instead of 92.9%.")
    print("   A better book moves the arithmetic more than any filter tested.")
    print()
    print("   Route 3 (the only one that scales): a forecast with real skill,")
    print("   demonstrated OUT OF SAMPLE, at a hold long enough that the skill")
    print("   clears the required rate in this table. That is what the walk-")
    print("   forward machinery in fusion/learning.py exists to prove, and it")
    print("   has to clear the held-out bar, not the training one.")
    print()
    print("   None of these is 'tune until 19 of 20 win'. That route is measured")
    print("   in section 5 and it is noise.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
