"""Does one engine beat the committee? Measured, out of sample.

Builds the dataset the single engine needs -- every feature, plus the stance of
all 132 council members on the same bar -- then fits the engine on one half of
the paths and scores it on the other, against the two baselines that matter:

* **the council's majority vote**, which is the thing being replaced;
* **always long**, so a rising sample cannot flatter a long-only model.

Reported for every candidate: accuracy on the next move, the hit rate the
spread demands at that hold, net pips per trade, and Wilson intervals. Nothing
is claimed that is not in the table, and if the engine loses, the table says so.

    python3 scripts/oracle_study.py
    python3 scripts/oracle_study.py --bars 5000 --paths 10 --horizon 15
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

BOT_ROOT = Path(__file__).resolve().parents[1]
if str(BOT_ROOT) not in sys.path:
    sys.path.insert(0, str(BOT_ROOT))

from fusion import charter  # noqa: E402
from fusion.features import compute_features  # noqa: E402
from fusion.instruments import for_symbol  # noqa: E402
from fusion.oracle import (  # noqa: E402
    OracleEngine,
    expected_net_pips,
    majority_vote,
    required_probability,
)
from fusion.simulate import synthetic_series  # noqa: E402
from fusion.strategies import REGISTRY  # noqa: E402

PIP = for_symbol("EURUSD").pip
WARMUP = 80


def member_stances(candle, previous_close, features, spread) -> dict[str, float]:
    """Every member's stance on one bar: +1 buy, -1 sell, absent if silent."""
    out: dict[str, float] = {}
    for name, rule in REGISTRY.items():
        try:
            decision = rule(candle, previous_close, dict(features), None, spread_pips=spread)
        except Exception:
            continue
        if decision is None or not getattr(decision, "actionable", False):
            continue
        side = (decision.side or "").upper()
        if side == "BUY":
            out[name] = 1.0
        elif side == "SELL":
            out[name] = -1.0
    return out


def build_rows(paths, bars, horizon, spread, volatility, seed, drift=0.0):
    rows = []
    for path in range(paths):
        # drift=0 by default. A drifting series makes "always long" look skilful
        # and would flatter any model that leans long -- a baseline artefact,
        # not a market feature.
        candles = synthetic_series(
            bars, seed=seed + path * 13, volatility=volatility, drift=drift
        )
        features = compute_features(candles)
        for index in range(WARMUP, len(candles) - horizon):
            here = features[index]
            if here is None:
                continue
            entry = candles[index + 1].open
            exit_ = candles[index + horizon].close
            move = (exit_ - entry) / PIP
            if abs(move) < 1e-9:
                continue
            stances = member_stances(candles[index], candles[index - 1].close, here, spread)
            merged = dict(here)
            from fusion.oracle import panel_vector

            merged.update(panel_vector(stances))
            label = 1 if move > 0 else 0
            rows.append((path, merged, label, move, stances))
    return rows


def score(rows, side_of, spread, move_mean):
    """Accuracy, net pips and the interval for any side-picking function."""
    wins = losses = 0
    net = 0.0
    buys = sells = 0
    for _, features, label, move, stances in rows:
        side = side_of(features, stances)
        if side == "NO SIGNAL":
            continue
        gross = move if side == "BUY" else -move
        net += charter.net_pips(gross, spread)
        if gross > 0:
            wins += 1
        else:
            losses += 1
        buys += side == "BUY"
        sells += side == "SELL"
    decided = wins + losses
    return {
        "n": decided,
        "accuracy": wins / decided if decided else 0.0,
        "interval": charter.wilson_interval(wins, decided),
        "net_per_trade": net / decided if decided else 0.0,
        "buy_share": buys / decided if decided else 0.0,
        "needed": required_probability(move_mean, spread) if move_mean > spread else float("inf"),
    }


def show(name, stats, move_mean, spread):
    low, high = stats["interval"]
    needed = required_probability(move_mean, spread)
    verdict = "clears cost" if stats["net_per_trade"] > 0 else "loses to spread"
    print(
        f"  {name:<26} n={stats['n']:>6,}  acc={stats['accuracy']:>5.1%} "
        f"[{low:.1%}-{high:.1%}]  net={stats['net_per_trade']:>+6.2f}p  "
        f"buy={stats['buy_share']:>4.0%}  needed={needed:>5.1%}  {verdict}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bars", type=int, default=3000)
    parser.add_argument("--paths", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--volatility", type=float, default=0.00024)
    parser.add_argument("--seed", type=int, default=7700)
    parser.add_argument("--shuffle-labels", action="store_true",
                        help="control: scramble the TRAINING labels. A model must then "
                             "score ~50% on held-out data. If it does not, the dataset "
                             "leaks the future and no result from it means anything.")
    parser.add_argument("--drift", type=float, default=0.0,
                        help="per-bar drift; kept at 0 so 'always long' is a fair baseline")
    args = parser.parse_args()

    spread = charter.DEFAULT_SPREAD_PIPS
    print("SINGLE-ENGINE STUDY")
    print(f"  {args.paths} paths x {args.bars} bars, hold {args.horizon}m, "
          f"spread {spread:.2f} pips")
    print("  building features and 132-member stances per bar (slow on purpose:")
    print("  every member really is evaluated)")
    rows = build_rows(
        args.paths, args.bars, args.horizon, spread, args.volatility, args.seed, args.drift
    )
    if not rows:
        print("  no rows built")
        return 1

    train = [row for row in rows if row[0] % 2 == 0]
    test = [row for row in rows if row[0] % 2 == 1]
    moves = [row[3] for row in rows]
    move_mean = statistics.fmean(abs(move) for move in moves)
    print(f"  {len(rows):,} bars: {len(train):,} train, {len(test):,} held out")
    print(f"  mean |move| over {args.horizon}m: {move_mean:.2f} pips")
    print(f"  hit rate the spread demands:   {required_probability(move_mean, spread):.1%}")
    print()

    train_samples = [(features, label) for _, features, label, _, _ in train]
    if args.shuffle_labels:
        import random as _random

        shuffle_rng = _random.Random(1234)
        labels = [label for _, label in train_samples]
        shuffle_rng.shuffle(labels)
        train_samples = [(features, label) for (features, _), label in zip(train_samples, labels)]
        print("  CONTROL RUN: training labels have been scrambled on purpose.")
        print("  Held-out accuracy must fall to ~50%. Anything higher is leakage.")
        print()

    engine = OracleEngine(horizon=args.horizon)
    engine.fit(train_samples, min_samples=200)
    engine.move_pips = move_mean
    for note in engine.notes:
        print(f"  {note}")
    print()

    print("HELD-OUT RESULTS (bars the model never trained on)")
    show("always long", score(test, lambda f, s: "BUY", spread, move_mean), move_mean, spread)
    show(
        "council majority vote",
        score(test, lambda f, s: majority_vote(s), spread, move_mean),
        move_mean,
        spread,
    )

    def engine_side(features, stances):
        read = engine.read(features, stances, spread_pips=spread, min_net_pips=0.0)
        return read.side

    def engine_raw(features, stances):
        # The engine's DIRECTION, ignoring the cost floor. This is the honest
        # skill measure: scoring only the trades it chose to take would let it
        # hide every bar it was unsure about, and would make the label-shuffle
        # control unable to say anything (it abstains on all of them).
        return "BUY" if engine.read(features, stances, spread_pips=spread).prob_up >= 0.5 else "SELL"

    show("ONE ENGINE (raw call)", score(test, engine_raw, spread, move_mean), move_mean, spread)
    show("ONE ENGINE (cost floor)", score(test, engine_side, spread, move_mean), move_mean, spread)
    print()

    reads = [engine.read(f, s, spread_pips=spread) for _, f, _, _, s in test]
    taken = [read for read in reads if read.take]
    print(f"  engine abstention rate at the cost floor: {1 - len(taken) / len(reads):.1%}")
    if taken:
        print(f"  when it spoke: buy {sum(1 for r in taken if r.side == 'BUY')}, "
              f"sell {sum(1 for r in taken if r.side == 'SELL')}")
        print(f"  mean expected net it demanded: "
              f"{statistics.fmean(r.expected_net_pips for r in taken):+.2f} pips")
    print()

    print("READ THIS BEFORE BELIEVING ANY NUMBER ABOVE")
    print("  * accuracy is measured on the NEXT", args.horizon, "minutes, not on the")
    print("    direction of the past -- that distinction is the whole point.")
    print("  * 'needed' is the hit rate the spread demands at this payoff. A")
    print("    candidate whose accuracy sits below it loses money no matter how")
    print("    impressive its accuracy looks.")
    print("  * the interval is the honest width of the estimate. If it straddles")
    print("    'needed', the result is not yet evidence of anything.")
    print("  * the council vote is shown because it is being replaced. If the")
    print("    engine does not beat it, the engine is not an improvement.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
