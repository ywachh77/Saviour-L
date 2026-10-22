"""Heartbeat-only liveness.

There is no second opinion about whether the bot is alive: the primary runner
writes a heartbeat each cycle, and the GitHub Actions fallback stands in only
when that heartbeat is older than the stale threshold. A process that is up but
wedged stops writing heartbeats, which is exactly the case the fallback exists
for.
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


def fallback_should_run(store, *, stale_seconds: int = charter.HEARTBEAT_STALE_SECONDS, runner: str = "termux") -> bool:
    """True only when the primary runner's heartbeat has gone quiet."""
    return is_stale(store.latest_heartbeat(runner=runner), stale_seconds)
