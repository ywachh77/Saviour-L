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
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fusion.config import load_config  # noqa: E402
from fusion.runner import FusionBot  # noqa: E402
from fusion.store import build_store  # noqa: E402
from fusion.telegram import TelegramClient  # noqa: E402
from fusion.timeutil import now  # noqa: E402

# Default 18-minute container budget inside fusion-cycle.yml's 20-minute job
# ceiling. An ordinary cycle takes ~4s so the full 720s Telegram listen window
# runs untouched; if a scheduled job (e.g. 23:55Z nightly) consumed part of the
# budget, LISTEN_SECONDS shrinks automatically instead of hitting SIGKILL.
DEFAULT_MAX_RUN_SECONDS = 1080.0


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, ""))
    except ValueError:
        return default


def resolve_listen_seconds(
    requested: float,
    elapsed: float,
    max_run_seconds: float = DEFAULT_MAX_RUN_SECONDS,
) -> float:
    """Cap Telegram long-polling so cycle + scheduled work + listen stays inside budget."""
    if requested <= 0:
        return 0.0
    if max_run_seconds <= 0:
        return float(requested)
    remaining = max(0.0, float(max_run_seconds) - max(0.0, float(elapsed)))
    return min(float(requested), remaining)


def main() -> int:
    started = time.monotonic()
    config = load_config(ROOT / ".env")
    config.runner = "github-actions"
    store = build_store(config)
    telegram = (
        TelegramClient(config.telegram_token, config.telegram_owner_chat_id)
        if config.telegram_ready
        else None
    )

    bot = FusionBot(config=config, store=store, telegram=telegram)
    bot.startup()
    try:
        # Align to the real minute seal window ([4.5s, 6.4s] past the boundary)
        # AFTER startup state hydration: warm cycles fetch an incremental 45-bar
        # tail first and, if Twelve Data's CDN has not published M-1 yet, run
        # one fast :07.1s outputsize=6 catch-up poll while keeping all 3 markets
        # comfortably inside the 10s Charter seal ceiling.
        bot.wait_for_seal_window(min_offset_seconds=4.5, max_entry_seconds=6.4)
        report = bot.minute_cycle()
        bot.settle_due()
        bot.reconcile_settled_signals()
        bot.flush_outbox()
        bot.run_scheduled()
        print(report.to_dict())

        # Stay on the line for a while, bounded by remaining container budget.
        requested_listen = _env_float("LISTEN_SECONDS", 180.0)
        max_run = _env_float("MAX_RUN_SECONDS", DEFAULT_MAX_RUN_SECONDS)
        listen_seconds = resolve_listen_seconds(
            requested_listen,
            elapsed=time.monotonic() - started,
            max_run_seconds=max_run,
        )
        if listen_seconds > 0:
            handled = bot.listen(listen_seconds)
            print(f"listened for {listen_seconds:.0f}s, handled {handled} command(s)")
    finally:
        bot.save_all_states(force=True)
        bot.flush_key_pool()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
