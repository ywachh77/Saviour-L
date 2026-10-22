#!/usr/bin/env python3
"""GitHub Actions fallback: work only when the primary runner has gone quiet.

The rule is heartbeat-only: if the Termux heartbeat is younger than 90 seconds,
this script exits immediately and does nothing. Otherwise it tags itself
'actions-fallback', runs one minute cycle, and runs whatever scheduled job is
due. Job slots make an overlap harmless.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fusion import heartbeat  # noqa: E402
from fusion.config import load_config  # noqa: E402
from fusion.runner import FusionBot  # noqa: E402
from fusion.store import build_store  # noqa: E402
from fusion.telegram import TelegramClient  # noqa: E402
from fusion.timeutil import now  # noqa: E402


def main() -> int:
    config = load_config(ROOT / ".env")
    config.runner = "actions-fallback"
    store = build_store(config)
    telegram = TelegramClient(config.telegram_token, config.telegram_owner_chat_id) if config.telegram_ready else None

    row = store.latest_heartbeat(runner="termux")
    age = heartbeat.heartbeat_age_seconds(row)
    if not heartbeat.is_stale(row, config.heartbeat_stale_seconds):
        print(f"Primary heartbeat is {age:.0f}s old — inside the "
              f"{config.heartbeat_stale_seconds}s window. Fallback standing down.")
        return 0

    print(f"Primary heartbeat is {'missing' if age is None else f'{age:.0f}s old'}. Fallback taking over.")
    bot = FusionBot(config=config, store=store, telegram=telegram)
    bot.startup()
    report = bot.minute_cycle(now())
    bot.run_scheduled(now())
    print(report.to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
