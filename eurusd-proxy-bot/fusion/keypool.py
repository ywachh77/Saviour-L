"""Twelve Data key pool: reserve before the call, refund, rotate, cool down.

Quota is reserved *before* the HTTP request so a burst of concurrent work can
never overspend the plan. A failed call refunds what it did not use. Repeated
rate-limit responses put a key in cooldown instead of hammering it.

Slots exist for keys a user has legitimately provisioned. This module does not
create accounts, rotate identities, or help anyone dodge a provider's rate
limit — see README, "Key slots".
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

MAX_SLOTS = 30

# Twelve Data free-plan defaults; override with env if your plan differs.
DEFAULT_MINUTE_LIMIT = 8
DEFAULT_DAY_LIMIT = 800


def mask_key(key: str) -> str:
    if not key:
        return ""
    head = key[:4]
    tail = key[-2:] if len(key) > 6 else ""
    return f"{head}...{tail}({len(key)})"


@dataclass
class KeySlot:
    index: int
    label: str
    enabled: bool = True
    cooldown_until: float = 0.0
    minute_bucket: str = ""
    minute_used: int = 0
    day_bucket: str = ""
    day_used: int = 0
    consecutive_failures: int = 0
    total_calls: int = 0
    total_failures: int = 0
    disabled_reason: str = ""
    minute_limit: int = DEFAULT_MINUTE_LIMIT
    day_limit: int = DEFAULT_DAY_LIMIT

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "label": self.label,
            "enabled": self.enabled,
            "cooldown_until": self.cooldown_until,
            "day_bucket": self.day_bucket,
            "day_used": self.day_used,
            "minute_bucket": self.minute_bucket,
            "minute_used": self.minute_used,
            "consecutive_failures": self.consecutive_failures,
            "total_calls": self.total_calls,
            "total_failures": self.total_failures,
            "disabled_reason": self.disabled_reason,
            "minute_limit": self.minute_limit,
            "day_limit": self.day_limit,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "KeySlot":
        slot = cls(index=int(payload.get("index", 0)), label=str(payload.get("label", "")))
        for name in (
            "enabled",
            "cooldown_until",
            "minute_bucket",
            "minute_used",
            "day_bucket",
            "day_used",
            "consecutive_failures",
            "total_calls",
            "total_failures",
            "disabled_reason",
            "minute_limit",
            "day_limit",
        ):
            if name in payload:
                setattr(slot, name, payload[name])
        return slot


@dataclass(frozen=True)
class Reservation:
    key_index: int
    cost: int
    ok: bool
    reason: str = ""


class KeyPool:
    """Round-robin pool with quota reservation, cooldown and honest exhaustion."""

    def __init__(
        self,
        keys: list[str],
        *,
        minute_limit: int = DEFAULT_MINUTE_LIMIT,
        day_limit: int = DEFAULT_DAY_LIMIT,
        clock=time.time,
    ):
        if len(keys) > MAX_SLOTS:
            keys = keys[:MAX_SLOTS]
        self.slots: list[KeySlot] = [
            KeySlot(
                index=index,
                label=mask_key(key),
                minute_limit=int(minute_limit),
                day_limit=int(day_limit),
            )
            for index, key in enumerate(keys)
        ]
        self.cursor = 0
        self._clock = clock
        self.exhausted_events = 0

    # ------------------------------------------------------------------ state
    def __len__(self) -> int:
        return len(self.slots)

    @property
    def configured(self) -> bool:
        return bool(self.slots)

    def _minute_key(self, now: float) -> str:
        return time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime(now))

    def _day_key(self, now: float) -> str:
        return time.strftime("%Y-%m-%dZ", time.gmtime(now))

    def _roll(self, slot: KeySlot, now: float) -> None:
        minute = self._minute_key(now)
        day = self._day_key(now)
        if slot.minute_bucket != minute:
            slot.minute_bucket = minute
            slot.minute_used = 0
        if slot.day_bucket != day:
            slot.day_bucket = day
            slot.day_used = 0

    def usable(self, slot: KeySlot, now: float, cost: int) -> tuple[bool, str]:
        if not slot.enabled:
            return False, f"disabled ({slot.disabled_reason or 'no reason recorded'})"
        if slot.cooldown_until > now:
            return False, f"cooling down for {slot.cooldown_until - now:.0f}s"
        self._roll(slot, now)
        if slot.minute_used + cost > slot.minute_limit:
            return False, "minute quota spent"
        if slot.day_used + cost > slot.day_limit:
            return False, "daily quota spent"
        return True, ""

    # -------------------------------------------------------------- operations
    def reserve(self, cost: int = 1) -> Reservation:
        """Reserve credits on the next usable key. Returns ok=False when spent."""
        if not self.slots:
            return Reservation(-1, cost, False, "no API keys configured")
        now = self._clock()
        for offset in range(len(self.slots)):
            index = (self.cursor + offset) % len(self.slots)
            slot = self.slots[index]
            allowed, reason = self.usable(slot, now, cost)
            if allowed:
                self._roll(slot, now)
                slot.minute_used += cost
                slot.day_used += cost
                slot.total_calls += 1
                self.cursor = (index + 1) % len(self.slots)
                return Reservation(index, cost, True)
        self.exhausted_events += 1
        return Reservation(-1, cost, False, "all keys exhausted (cooldown or quota)")

    def refund(self, reservation: Reservation) -> None:
        """Give back credits for a call that never reached the provider."""
        if not reservation.ok or reservation.key_index < 0:
            return
        slot = self.slots[reservation.key_index]
        slot.minute_used = max(0, slot.minute_used - reservation.cost)
        slot.day_used = max(0, slot.day_used - reservation.cost)
        slot.total_calls = max(0, slot.total_calls - 1)

    def report_success(self, reservation: Reservation) -> None:
        if not reservation.ok:
            return
        self.slots[reservation.key_index].consecutive_failures = 0

    def report_failure(self, reservation: Reservation, status: int | None = None, message: str = "") -> None:
        """Classify a provider failure and react: cooldown, rotate, or disable."""
        if not reservation.ok:
            return
        slot = self.slots[reservation.key_index]
        slot.total_failures += 1
        slot.consecutive_failures += 1
        now = self._clock()

        if status in (401, 403):
            slot.enabled = False
            slot.disabled_reason = f"provider rejected the key (HTTP {status})"
            return
        if status == 429:
            # Rate limited: back off hard, and grow with each repeat offence.
            penalty = min(300.0, 60.0 * (2 ** min(3, slot.consecutive_failures - 1)))
            slot.cooldown_until = now + penalty
            return
        if status is not None and 500 <= status < 600:
            slot.cooldown_until = now + min(120.0, 15.0 * slot.consecutive_failures)
            return
        if status is None:
            # Network-level failure (timeout, DNS). Short cooldown, then retry.
            slot.cooldown_until = now + min(60.0, 10.0 * slot.consecutive_failures)

    def key_for(self, index: int, keys: list[str]) -> str:
        return keys[index]

    def summary(self) -> dict[str, Any]:
        now = self._clock()
        rows = []
        for slot in self.slots:
            self._roll(slot, now)
            rows.append(
                {
                    "label": slot.label,
                    "enabled": slot.enabled,
                    "cooldown_s": round(max(0.0, slot.cooldown_until - now), 1),
                    "minute_used": slot.minute_used,
                    "minute_limit": slot.minute_limit,
                    "day_used": slot.day_used,
                    "day_limit": slot.day_limit,
                    "failures": slot.consecutive_failures,
                }
            )
        return {
            "slots": rows,
            "enabled": sum(1 for row in rows if row["enabled"]),
            "exhausted_events": self.exhausted_events,
        }

    def to_dict(self) -> dict:
        return {"slots": [slot.to_dict() for slot in self.slots], "cursor": self.cursor}

    @classmethod
    def from_state(cls, keys: list[str], payload: dict | None, **kwargs) -> "KeyPool":
        pool = cls(keys, **kwargs)
        if not payload:
            return pool
        saved = {int(row["index"]): row for row in payload.get("slots", [])}
        for slot in pool.slots:
            row = saved.get(slot.index)
            if row:
                slot.__dict__.update(KeySlot.from_dict(row).__dict__)
        pool.cursor = int(payload.get("cursor", 0))
        return pool
