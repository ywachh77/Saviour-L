#!/usr/bin/env python3
"""Start the bot. That is the whole command.

    python scripts/run_bot.py

It reads .env next to this repository, then owns its own schedule: ingest and
signal at every minute boundary, settle whenever a horizon closes, learn online
after every confirmed real outcome, run the nightly walk-forward at 23:55 UTC
and send the daily digest at 00:05 UTC.

Only five Telegram commands exist: /start /help /status /pause /resume.
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


def main() -> int:
    config = load_config(ROOT / ".env")
    store = build_store(config)
    telegram = TelegramClient(config.telegram_token, config.telegram_owner_chat_id) if config.telegram_ready else None

    missing = []
    if not config.provider_ready:
        missing.append("TWELVE_DATA_API_KEY (or TWELVE_DATA_KEY_1..30)")
    if not config.telegram_ready:
        missing.append("TELEGRAM_BOT_TOKEN / TELEGRAM_OWNER_CHAT_ID")
    if missing:
        print("FUSION BOT — not configured yet. Missing from your .env:")
        for item in missing:
            print(f"  · {item}")
        print()
        print("Copy .env.example to .env, fill those in, and run this again.")
        print("Without a provider key the bot cannot ingest; without Telegram it cannot deliver.")
        return 2

    bot = FusionBot(config=config, store=store, telegram=telegram)
    print(f"FUSION BOT starting — {config.pair} M1, {config.horizon_minutes}-minute horizon, "
          f"{len(config.twelve_data_keys)} key slot(s), Supabase "
          f"{'on' if config.supabase_ready else 'off (local state only)'}.")
    print("Signals are paused with /pause and resumed with /resume. Ctrl+C to stop.")
    bot.run_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
