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

It also stays on the line after the cycle, long polling Telegram for
LISTEN_SECONDS so commands are answered while the container is still alive
rather than at the next wake-up. That narrows the wait; it does not remove it,
because the container still only exists when the cron created it.

Signals are only ever emitted for "now" — the charter forbids backdating, so a
late wake-up does not go back and grade the minutes it slept through.

    python scripts/actions_cycle.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fusion.config import load_config  # noqa: E402
from fusion.runner import FusionBot  # noqa: E402
from fusion.store import build_store  # noqa: E402
from fusion.telegram import TelegramClient  # noqa: E402
from fusion.timeutil import now  # noqa: E402


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, ""))
    except ValueError:
        return default


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

    # Stay on the line for a while. The cron decides when this container
    # exists; long polling decides whether a command sent during that window is
    # answered at once or not at all. Without it every /signals waits for the
    # next scheduled wake-up.
    listen_seconds = _env_float("LISTEN_SECONDS", 180.0)
    if listen_seconds > 0:
        handled = bot.listen(listen_seconds)
        print(f"listened for {listen_seconds:.0f}s, handled {handled} command(s)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
