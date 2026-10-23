#!/usr/bin/env python3
"""Replay the whole system over your stored candles and settle it honestly.

    python scripts/backtest.py              # replay everything in Supabase
    python scripts/backtest.py 5000         # use the most recent 5000 bars
    python scripts/backtest.py --folds 6    # more walk-forward folds
    python scripts/backtest.py --paths 8    # replay 8 synthetic price paths

This is NOT the model's score on every bar. It is the record of the bars that
survive the data gate, the regime filter, the six-check council and the
meta-model veto — with the spread charged on every single one.

Read the note it prints. If it says fewer than 200 signals, the number is
entertainment, not evidence.

**Use --paths before believing any single replay.** One history is one draw
from a distribution. --paths replays the same system over several independent
price paths and reports the spread between them, which is the number a single
run hides. A system whose per-path expectancy swings from -3.0 to +0.1 does
not have an edge of -1.3; it has no measurable edge and a lot of noise.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fusion.backtest import MIN_SIGNALS_FOR_A_VIEW, run_backtest, run_multi_path  # noqa: E402
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

    # --paths needs no stored history: it generates its own price paths, so it
    # works on a fresh clone before a single real candle has been collected.
    if "--paths" in argv:
        try:
            paths = int(argv[argv.index("--paths") + 1])
        except (ValueError, IndexError):
            paths = 8
        # Drop the flag AND its value before reading a positional bar count,
        # otherwise "--paths 8" is silently read as "8 bars".
        positional = [
            arg
            for index, arg in enumerate(argv)
            if not arg.startswith("--")
            and not (index > 0 and argv[index - 1] == "--paths")
        ]
        bars = 3000
        if positional:
            try:
                bars = max(600, int(positional[0]))
            except ValueError:
                pass
        config: Config = load_config(ROOT / ".env")
        print()
        print("FUSION BOT — multi-path replay (synthetic price paths)")
        print("=" * 62)
        print(f"  {paths} independent random-walk paths, {bars} bars each")
        print(f"  horizon {config.horizon_minutes}m · spread {config.spread_pips:.2f} pips "
              f"· {folds} purged folds")
        print()
        print("  A random walk has no predictable direction, so the true edge on these")
        print("  paths is zero by construction. What this measures is the NOISE: how far")
        print("  a single replay can stray from zero and still look like a finding.")
        print()
        report = run_multi_path(config, paths=paths, bars=bars, folds=folds)
        print("  per path")
        for index, item in enumerate(report.reports, start=1):
            exp = item.expectancy_pips
            text = "n/a" if exp is None else f"{exp:+.3f}"
            print(f"    path {index:>2}: {item.signals:>4} signals, "
                  f"{item.wins}W/{item.losses}L, expectancy {text} pips/trade")
        print()
        summary = report.summary
        if summary is not None:
            print(f"  mean expectancy   : {summary.mean_expectancy:+.3f} pips/trade "
                  f"(sd {summary.stdev_expectancy:.3f})")
            print(f"  best / worst path : {summary.best:+.3f} / {summary.worst:+.3f}")
            print(f"  profitable paths  : {summary.profitable_paths} of {summary.n_paths}")
            print()
            if summary.single_path_is_misleading:
                print("  !! The paths disagree on the SIGN. Any one of them, quoted alone,")
                print("     would have looked like a result. That is the trap this exists to")
                print("     close: a single replay cannot distinguish skill from a lucky draw.")
                print()
            print(f"  VERDICT: {summary.verdict.describe()}")
        print()
        return 0

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
        print(f"  expectancy per signal    : {payload['expectancy_pips']:+.3f} pips"
              f"  [95% CI {payload['expectancy_low']:+.3f} to {payload['expectancy_high']:+.3f}]")
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
    status = payload.get("edge_status", "")
    print(f"  EDGE: {status}")
    if status == "EDGE_DEMONSTRATED":
        print("    The interval sits above zero on this history. Replay --paths before")
        print("    treating it as a property of the system rather than of this path.")
    elif status == "EDGE_REFUTED":
        print("    Confidently negative after costs. Do not trade this configuration.")
    else:
        print("    Not proven either way. This is NOT evidence of no edge — it means the")
        print("    sample cannot resolve one. Absence of evidence is not evidence of absence.")
        if payload.get("samples_needed", -1) > 0:
            print(f"    Detecting a 0.25 pip/trade edge here needs about "
                  f"{payload['samples_needed']} settled trades.")
    if payload["signals"] < MIN_SIGNALS_FOR_A_VIEW:
        print(f"    Only {payload['signals']} signals; {MIN_SIGNALS_FOR_A_VIEW} are needed "
              f"before the interval means anything.")
    print()
    print("  Research only. Past replay is not forward performance.")
    print()
    if "--json" in argv:
        print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
