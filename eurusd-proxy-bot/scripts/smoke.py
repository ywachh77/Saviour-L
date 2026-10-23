#!/usr/bin/env python3
"""Run the whole pipeline offline on synthetic data and print what happened.

    python scripts/smoke.py 420

Nothing here touches the network, and nothing here trains the model: synthetic
data is for watching the machinery fire, so the charter refuses to learn from
it. If the report shows a trained model or non-zero prototype weights, the
charter has been broken.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tempfile  # noqa: E402

from fusion import charter
from fusion import stats as stats_module  # noqa: E402
from fusion.config import Config  # noqa: E402
from fusion.runner import FusionBot  # noqa: E402
from fusion.settle import summarise  # noqa: E402
from fusion.simulate import SyntheticFeed, run_simulation, synthetic_series  # noqa: E402
from fusion.store import LocalStore, Store  # noqa: E402


def main(argv: list[str]) -> int:
    try:
        bars = int(argv[1]) if len(argv) > 1 else 420
    except ValueError:
        print("Usage: python scripts/smoke.py [number-of-bars]")
        return 2
    if bars < 60:
        print("Give me at least 60 bars so the indicators have something to chew on.")
        return 2

    directory = Path(tempfile.mkdtemp(prefix="fusion-smoke-"))
    config = Config(
        pair=charter.ALLOWED_PAIR,
        timeframe=charter.ALLOWED_TIMEFRAME,
        horizon_minutes=3,
        local_state_dir=str(directory),
        runner="smoke",
    )
    bot = FusionBot(config=config, store=Store(None, LocalStore(directory)), provider=None, telegram=None)

    candles = synthetic_series(bars, seed=20260101)
    feed = SyntheticFeed(candles)
    summary = run_simulation(bot, candles, feed=feed)

    settlements = bot.store.load_settlements()
    stats = summarise([_settlement(row) for row in settlements])
    events = bot.store.load_events(limit=1000)
    learned = sum(1 for event in events if event.get("applied"))

    print()
    print("FUSION BOT — offline pipeline smoke run")
    print("=" * 62)
    print(f"  synthetic bars replayed      : {bars}")
    print(f"  minute cycles executed       : {summary['steps']}")
    print(f"  signals sealed               : {summary['signals']}")
    print(f"  abstentions (the norm)       : {summary['abstentions']}")
    print(f"  dropped as late              : {summary['discarded_late']}")
    print(f"  settlements recorded         : {stats.settled}"
          f"  (wins {stats.wins} / losses {stats.losses} / ties {stats.tied})")
    print(f"  unresolved (never claimed)   : {stats.unresolved}")
    if stats.win_rate is not None:
        print(f"  win rate                     : {stats.win_rate * 100:.1f}%"
              f"  [95% Wilson {stats.wilson_low * 100:.1f}%–{stats.wilson_high * 100:.1f}%]")
    print(f"  net pips after {config.spread_pips:.1f} pip cost : {stats.net_pips:+.1f}")

    # Expectancy, with its interval, sits next to the win rate on purpose. This
    # run routinely produces a 68% win rate alongside a negative net, because a
    # trade that wins 1 pip and loses 4 is a loser wearing a winner's number.
    # Showing the two together is the point.
    outcomes = [
        float(row["net_pips"])
        for row in settlements
        if row.get("outcome") in (charter.WIN, charter.LOSS) and row.get("net_pips") is not None
    ]
    if outcomes:
        verdict = stats_module.edge_verdict(outcomes)
        print(f"  expectancy per signal        : {verdict.mean_net_pips:+.3f} pips"
              f"  [95% CI {verdict.ci_low:+.3f} to {verdict.ci_high:+.3f}]")
        print(f"  edge verdict                 : {verdict.status}")
    print()
    print("Honesty checks")
    print("-" * 62)
    checks = [
        ("Synthetic data trained nothing", learned == 0),
        ("No model was fitted", not (bot.model and bot.model.trained)),
        ("No prototype earned a voice", bot.weights.active() == {}),
        (
            "Every unresolved outcome has no number attached",
            all(row.get("net_pips") is None for row in settlements if row.get("outcome") == charter.UNRESOLVED),
        ),
        ("Costs charged on every settled trade", all(row.get("net_pips") is not None for row in settlements
                                                     if row.get("outcome") in (charter.WIN, charter.LOSS))),
    ]
    failures = 0
    for label, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
        failures += 0 if passed else 1
    print()
    print(f"  state written to {directory} (delete it when you are done)")
    print()
    if failures:
        print("Charter violation. Do not ship this.")
        return 1
    print("Pipeline fires end to end and the charter holds. "
          "This is synthetic data: it says nothing about whether the strategy makes money.")
    return 0


def _settlement(row: dict):
    from fusion.domain import Settlement

    return Settlement.from_dict(row)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
