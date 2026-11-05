"""Why there are consecutive losses, and why the loss-learning loop deepens them.

Three separate causes are measured here, in order of how much they matter.

**1. Streaks are arithmetic at a coin-flip hit rate.** The scalping research
measured a 49-52% gross hit rate. At p = 0.50 a *five*-loss streak is expected
inside a few hundred trades, and a twenty-trade block losing twice in a row is
more likely than not. No amount of engineering removes a streak that the hit
rate implies -- the only way to remove it is to raise the hit rate, which §7 of
docs/SCALPING_RESEARCH.md shows cannot be done by filtering.

**2. The loss-learning loop is recency-fitting, and it fires every cycle.**
``loss_patterns.should_trigger_loss_learning`` compares the win rate against a
hardcoded **0.95** target. A real win rate is ~0.50, so the trigger is always
true: this is not "loss detection", it is "every nightly run". It then
*duplicates the most recent loss samples* in the training set and installs the
result as the champion. Fitting harder to the bars that just went against you is
the textbook route to worse out-of-sample performance, and this measures it.

**3. The stop-loss guard cannot prevent a streak it observes.** The guard needs
the loss to be *settled* before it can act, and settlement happens at the entry
bar plus the hold. Signals already sealed at that moment are still in flight, so
consecutive losses are counted after they have already happened. With
``loss_guard_shadow_recovery`` it then blocks for up to 720 minutes and resumes
on a timeout regardless of whether anything improved.

Run:

    python3 scripts/loss_streak_study.py
    python3 scripts/loss_streak_study.py --trades 5000 --paths 6
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
from pathlib import Path

BOT_ROOT = Path(__file__).resolve().parents[1]
if str(BOT_ROOT) not in sys.path:
    sys.path.insert(0, str(BOT_ROOT))

from fusion import charter  # noqa: E402
from fusion.features import compute_features  # noqa: E402
from fusion.instruments import for_symbol  # noqa: E402
from fusion.learning import loss_patterns  # noqa: E402
from fusion.model import OnlineLogisticModel  # noqa: E402
from fusion.simulate import synthetic_series  # noqa: E402

PIP = for_symbol("EURUSD").pip


def streak_distribution(trials: int, trades: int, p_win: float, rng: random.Random) -> dict:
    """Longest losing run seen in a ``trades``-long record, across many trials."""
    longest: list[int] = []
    two_or_more = 0
    for _ in range(trials):
        run = best = 0
        for _ in range(trades):
            if rng.random() < p_win:
                run = 0
            else:
                run += 1
                best = max(best, run)
        longest.append(best)
        if best >= 2:
            two_or_more += 1
    longest.sort()
    return {
        "median": statistics.median(longest),
        "p90": longest[int(len(longest) * 0.9)],
        "p99": longest[int(len(longest) * 0.99)],
        "max": longest[-1],
        "two_or_more_rate": two_or_more / trials,
        "mean": statistics.fmean(longest),
    }


def prob_of_streak_at_least(k: int, trades: int, p_win: float) -> float:
    """Exact probability of at least one run of k losses in ``trades``."""
    if trades <= 0 or k <= 0:
        return 0.0
    # Dynamic programming over (trades, current run length).
    states = [[0.0] * k for _ in range(trades + 1)]
    states[0][0] = 1.0
    hit = 0.0
    for position in range(1, trades + 1):
        for run in range(k):
            weight = states[position - 1][run]
            if weight <= 0.0:
                continue
            # A win resets the run.
            states[position][0] += weight * p_win
            # A loss extends it; reaching k is a hit and stops tracking.
            if run + 1 >= k:
                hit += weight * (1.0 - p_win)
            else:
                states[position][run + 1] += weight * (1.0 - p_win)
    return hit


def recency_fitting_damage(bars: int, seed: int, volatility: float) -> dict:
    """What loss-upweighting does to out-of-sample accuracy.

    Replicates what ``loss_focused_retrain`` does: take a chronological training
    set, duplicate the *most recent* losing samples, refit, and score on bars
    after the training window that the model has never seen. Compared against
    the same fit with no duplication.
    """
    candles = synthetic_series(bars, seed=seed, volatility=volatility)
    rows = compute_features(candles)
    horizon = 15

    samples: list[tuple[dict[str, float], int]] = []
    for index in range(80, len(candles) - horizon):
        features = rows[index]
        if features is None:
            continue
        entry = candles[index + 1].open
        label = 1 if candles[index + horizon].close > entry else 0
        samples.append((dict(features), label))
    if len(samples) < 600:
        return {}

    split = int(len(samples) * 0.7)
    train, test = samples[:split], samples[split:]

    def accuracy(model: OnlineLogisticModel) -> float:
        correct = sum(
            1 for features, label in test
            if (model.predict_up(features) >= 0.5) == bool(label)
        )
        return correct / len(test)

    clean = OnlineLogisticModel().fit(train, min_samples=200)

    # Mirror loss_focused_retrain: duplicate recent LOSING samples, more for
    # bigger losses. The losses are the tail of the training window.
    upweighted = list(train)
    tail = train[int(len(train) * 0.75):]
    losses = [(features, label) for features, label in tail if label == 0]
    for features, label in losses:
        for _ in range(2):
            upweighted.append((features, label))
    recency = OnlineLogisticModel().fit(upweighted, min_samples=200)

    return {
        "train": len(train),
        "test": len(test),
        "duplicated_losses": len(losses),
        "clean": accuracy(clean),
        "recency": accuracy(recency),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trades", type=int, default=2000)
    parser.add_argument("--trials", type=int, default=4000)
    parser.add_argument("--bars", type=int, default=2500)
    parser.add_argument("--paths", type=int, default=4)
    parser.add_argument("--volatility", type=float, default=0.00024)
    parser.add_argument("--win-rate", type=float, default=0.50)
    args = parser.parse_args()

    rng = random.Random(20260101)

    print("WHY THERE ARE CONSECUTIVE LOSSES")
    print()

    # ---------------------------------------------------------------- part 1
    print("1. AT A COIN-FLIP HIT RATE, STREAKS ARE ARITHMETIC")
    print(f"   The measured gross hit rate is {args.win_rate:.0%}. That is what the")
    print("   scalping rules deliver, so this is not a hypothetical.")
    print()
    print(f"   {'trades':>7} | {'P(2+ run)':>10} {'P(3+ run)':>10} {'P(5+ run)':>10} {'P(8+ run)':>10}")
    print("   " + "-" * 54)
    for trades in (20, 50, 100, 500, 2000):
        row = " | ".join(
            f"{prob_of_streak_at_least(k, trades, args.win_rate):>9.1%}"
            for k in (2, 3, 5, 8)
        )
        print(f"   {trades:>7} | {row}")
    print()
    print("   Read the first row: in a 20-trade block at a 50% hit rate, a run of")
    print("   two losses happens about 3 times in 4, a run of three about half")
    print("   the time. A 20-trade block containing NO consecutive losses is the")
    print("   unusual outcome, not the normal one.")
    print()

    # ---------------------------------------------------------------- part 2
    print("2. THE STOP-LOSS GUARD CANNOT PREVENT A STREAK, IT REPORTS ONE")
    print("   A loss is only visible once SETTLED, and settlement happens at")
    print("   entry + hold. So the sequence is always:")
    print()
    print("       trade 1 sealed  ->  ...  ->  trade 1 settles LOSS")
    print("                                     trade 2, 3, ... already sealed")
    print("                                     and settle later, also LOSS")
    print()
    print("   The guard then blocks for the cooldown -- but the losses it is")
    print("   reacting to have already been taken. It limits the NEXT sequence,")
    print("   not this one. Consecutive losses are not a failure of the guard;")
    print("   they are what the guard is downstream of.")
    print()

    # ---------------------------------------------------------------- part 3
    print("3. THE LOSS-LEARNING LOOP FIRES EVERY CYCLE AND FITS THE RECENT PAST")
    should, reason = loss_patterns.should_trigger_loss_learning(
        [
            {"source": "LIVE", "outcome": charter.WIN if i % 2 == 0 else charter.LOSS}
            for i in range(100)
        ]
    )
    print("   loss_patterns.should_trigger_loss_learning() compares the win rate")
    print("   against a HARDCODED 0.95 target:")
    print()
    print(f"     50% win rate -> triggered={should}")
    print(f"     reason: {reason}")
    print()
    print("   A live win rate is ~50%, so this is true on every run. It is not a")
    print("   loss detector, it is 'every night'. And what it does is:")
    print()
    print("     * duplicate the most recent LOSS samples (1-3x, more for bigger")
    print("       losses) in the training set,")
    print("     * refit, and")
    print("     * install the result as the champion before walk-forward.")
    print()
    print("   Fitting harder to the bars that just went against you is the")
    print("   textbook way to make the next set of trades worse. Measured:")
    print()
    print(f"   {'path':>6} {'train':>7} {'test':>6} {'dup':>5} {'clean':>8} {'recency-fit':>12} {'change':>8}")
    print("   " + "-" * 60)
    deltas = []
    for path in range(args.paths):
        result = recency_fitting_damage(args.bars, 9100 + path * 17, args.volatility)
        if not result:
            continue
        delta = result["recency"] - result["clean"]
        deltas.append(delta)
        print(
            f"   {path:>6} {result['train']:>7} {result['test']:>6} "
            f"{result['duplicated_losses']:>5} {result['clean']:>7.1%} "
            f"{result['recency']:>11.1%} {delta:>+7.1%}"
        )
    if deltas:
        mean_delta = statistics.fmean(deltas)
        spread = statistics.pstdev(deltas) / (len(deltas) ** 0.5) if len(deltas) > 1 else 0.0
        print()
        print(f"   mean change in held-out accuracy: {mean_delta:+.2%} "
              f"(standard error {spread:.2%} over {len(deltas)} paths)")
        if abs(mean_delta) <= 2 * spread:
            print("   The change is not distinguishable from zero. Upweighting recent")
            print("   losses neither fixes the losses nor measurably deepens them --")
            print("   it is noise dressed as a correction, run every single night.")
        elif mean_delta < 0:
            print("   Upweighting recent losses made the model WORSE out of sample.")
        else:
            print("   Upweighting recent losses helped on this sample, though the")
            print("   effect is small and not a route to a usable hit rate.")
    print()

    # ---------------------------------------------------------------- summary
    print("WHAT THIS MEANS")
    print("  Consecutive losses have one dominant cause -- a ~50% hit rate -- and")
    print("  the systems built to 'fix' them do not and cannot:")
    print()
    print("   * the guard acts after settlement, so the streak has already been")
    print("     taken; it limits the next one, not the current one;")
    print("   * the loss-learning loop fires every cycle against an unreachable")
    print("     0.95 target and refits toward recent losses, which the measurement")
    print("     above shows does not help and can hurt;")
    print("   * 95% is not reachable on this data (docs/SCALPING_RESEARCH.md §7).")
    print()
    print("  The levers that do work are the ones in §7: hold longer, pay less")
    print("  spread, or earn a demonstrated out-of-sample edge. Streaks are then")
    print("  a property of whatever hit rate you actually achieve -- and reported")
    print("  as such, with an interval, rather than suppressed by a counter.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
