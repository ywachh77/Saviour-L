#!/usr/bin/env python3
"""Replay the whole system over your stored candles and settle it honestly.

    python scripts/backtest.py            # replay everything in Supabase
    python scripts/backtest.py 5000       # use the most recent 5000 bars
    python scripts/backtest.py --folds 6  # more walk-forward folds

This is NOT the model's score on every bar. It is the record of the bars that
survive the data gate, the regime filter, the six-check council and the
meta-model veto — with the spread charged on every single one.

Read the note it prints. If it says fewer than 200 signals, the number is
entertainment, not evidence.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fusion.backtest import MIN_SIGNALS_FOR_A_VIEW, run_backtest  # noqa: E402
from fusion.config import Config, load_config  # noqa: E402
from fusion.council import CouncilWeights  # noqa: E402
from fusion.domain import Candle  # noqa: E402
from fusion.store import build_store  # noqa: E402


def main(argv: list[str]) -> int:
    limit = 20000
    folds = 4
    rest = [arg for arg in argv if not arg.startswith("--")]
    if "--folds" in argv:
        try:
            folds = int(argv[argv.index("--folds") + 1])
        except (ValueError, IndexError):
            folds = 4
    if rest:
        try:
            limit = max(600, int(rest[0]))
        except ValueError:
            pass

    config: Config = load_config(ROOT / ".env")
    store = build_store(config)
    rows = store.load_candles(config.pair, limit=limit)

    candles: list[Candle] = []
    for row in rows:
        try:
            candles.append(Candle.from_mapping(row))
        except (ValueError, KeyError, TypeError):
            continue

    print()
    print("FUSION BOT — purged walk-forward system replay")
    print("=" * 62)
    if len(candles) < 600:
        print(f"  only {len(candles)} candles stored; 600 are needed for a replay.")
        print()
        print("  Collect more history first:")
        print("    BACKFILL=1 BACKFILL_MINUTES=20000 python scripts/backfill.py")
        return 1

    weights = CouncilWeights.from_dict(store.load_council_weights())
    report = run_backtest(candles, config, folds=folds, weights=weights)
    payload = report.to_dict()

    print(f"  bars replayed            : {payload['bars']}")
    print(f"  purged folds             : {payload['folds']}")
    print(f"  signals the gate fired   : {payload['signals']} "
          f"({payload['coverage'] * 100:.2f}% of bars)")
    print(f"  wins / losses / ties     : {payload['wins']} / {payload['losses']} / {payload['ties']}")
    if payload["win_rate"] is not None:
        print(f"  win rate                 : {payload['win_rate'] * 100:.1f}%"
              f"  [95% Wilson {payload['wilson'][0] * 100:.1f}%–{payload['wilson'][1] * 100:.1f}%]")
    print(f"  gross pips               : {payload['gross_pips']:+.1f}")
    print(f"  net pips after costs     : {payload['net_pips']:+.1f}")
    if payload["expectancy_pips"] is not None:
        print(f"  expectancy per signal    : {payload['expectancy_pips']:+.3f} pips")
    if payload["model_accuracy"] is not None:
        print(f"  model accuracy, all bars : {payload['model_accuracy'] * 100:.1f}% "
              f"(on {payload['model_points']} bars — different population)")

    print()
    print("  per fold")
    for fold in payload["per_fold"]:
        print(f"    fold {fold['fold']}: {fold['signals']:>4} signals, "
              f"{fold['wins']}W/{fold['losses']}L, net {fold['net_pips']:+.1f} pips")

    print()
    if payload["learned_weights"]:
        print("  prototypes that earned a voice")
        for name, weight in sorted(payload["learned_weights"].items(), key=lambda item: -item[1]):
            print(f"    {name:24s} {weight:.3f}")
    else:
        print("  no prototype earned a voice — the core gate decided alone throughout.")

    print()
    print("  " + payload["note"])
    print()
    if payload["signals"] < MIN_SIGNALS_FOR_A_VIEW:
        print(f"  VERDICT: too few signals to judge (need {MIN_SIGNALS_FOR_A_VIEW}).")
    elif payload["net_pips"] > 0:
        print("  VERDICT: positive after costs on this history. Still one path, not a promise.")
    else:
        print("  VERDICT: not profitable after costs on this history.")
    print()
    print("  Research only. Past replay is not forward performance.")
    print()
    if "--json" in argv:
        print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
