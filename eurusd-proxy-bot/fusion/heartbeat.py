"""Heartbeat-only liveness.

The runner writes a heartbeat at the end of every cycle. That row is the only
record of whether the bot is alive: a process that is up but wedged stops
writing heartbeats, and the bot is judged dead until one arrives.

This deliberately replaces "is the process running?" with "did it finish a
cycle?". A live process that is stuck in a retry loop looks dead here, because
to the outside world it is.

Heartbeats are for *observing*, not for gating. Every scheduled run does its
own work; nothing waits for another runner to go quiet first, because there is
only one runner.
"""

from __future__ import annotations

from typing import Any

from . import charter
from .timeutil import age_seconds, format_datetime, now, parse_datetime


def build_heartbeat(*, runner: str, phase: str, extra: dict[str, Any] | None = None) -> dict:
    row = {
        "runner": runner,
        "phase": phase,
        "created_at": format_datetime(now()),
    }
    if extra:
        row.update(extra)
    return row


def heartbeat_age_seconds(row: dict | None, reference=None) -> float | None:
    if not row:
        return None
    stamp = row.get("created_at") or row.get("timestamp")
    if not stamp:
        return None
    return age_seconds(parse_datetime(stamp), reference)


def is_stale(row: dict | None, stale_seconds: int = charter.HEARTBEAT_STALE_SECONDS) -> bool:
    age = heartbeat_age_seconds(row)
    if age is None:
        return True
    return age > float(stale_seconds)


def stale_seconds_since(row: dict | None) -> float | None:
    """How long since this heartbeat, or None when there has never been one."""
    return heartbeat_age_seconds(row)
