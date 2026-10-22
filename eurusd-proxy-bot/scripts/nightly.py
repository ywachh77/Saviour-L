#!/usr/bin/env python3
"""Nightly job: walk-forward, promote or roll back, then report.

Runs automatically at 23:55 UTC from scripts/run_bot.py and, if the primary
runner is quiet, from GitHub Actions (see scripts/fallback.py).

    python scripts/nightly.py
"""

from __future__ import annotations

import json
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
    config.runner = config.runner or "nightly"
    store = build_store(config)
    telegram = TelegramClient(config.telegram_token, config.telegram_owner_chat_id) if config.telegram_ready else None

    bot = FusionBot(config=config, store=store, telegram=telegram)
    bot.startup()
    report = bot.nightly_job(now())
    print(json.dumps(report, indent=2, default=str))

    if report.get("skipped"):
        return 0
    if telegram is not None and (report.get("promoted") or report.get("rolled_back")):
        telegram.send(
            "FUSION BOT nightly\n"
            f"{'Promoted' if report.get('promoted') else 'Rolled back'}: {report.get('reason', '')}\n"
            f"Accuracy {report.get('accuracy')}, Brier skill {report.get('brier_skill')}, "
            f"{report.get('trades')} cost-charged trades for {report.get('net_pips')} pips."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
