#!/usr/bin/env python3
"""One bot cycle, run from GitHub Actions.

This is the runner. Actions wakes up on a cron, this script does exactly one
minute cycle plus whatever scheduled job is due, and the container disappears.

There is no primary/fallback split any more. The bot used to expect an
always-on device (Termux) to be the primary and for Actions to stand in only
when that device went quiet. With one runner, gating on a heartbeat would mean
gating on *itself*: the previous run's heartbeat is always older than the
stale threshold on a 30-minute cron, so the check would be noise at best and a
permanent stand-down at worst. Every run now does its own work.

That is also the honest description of what this is: Actions cannot run
continuously. It wakes, evaluates the minute it finds itself in, and stops.
Signals are only ever emitted for "now" — the charter forbids backdating, so a
late wake-up does not go back and grade the minutes it slept through.

    python scripts/actions_cycle.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fusion.config import load_config  # noqa: E402
from fusion.runner import FusionBot  # noqa: E402
from fusion.store import build_store  # noqa: E402
from fusion.telegram import TelegramClient  # noqa: E402
from fusion.timeutil import now  # noqa: E402


def main() -> int:
    config = load_config(ROOT / ".env")
    config.runner = "github-actions"
    store = build_store(config)
    telegram = (
        TelegramClient(config.telegram_token, config.telegram_owner_chat_id)
        if config.telegram_ready
        else None
    )

    stamp = now()
    bot = FusionBot(config=config, store=store, telegram=telegram)
    bot.startup()
    report = bot.minute_cycle(at=stamp)
    bot.run_scheduled(at=stamp)

    print(report.to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
