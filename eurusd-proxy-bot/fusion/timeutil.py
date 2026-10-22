"""UTC clock helpers. Everything in the bot is UTC; nothing uses local time."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

UTC = timezone.utc


def now() -> datetime:
    return datetime.now(UTC)


def parse_datetime(value) -> datetime:
    """Parse an ISO-8601 timestamp, assuming UTC when no zone is given."""
    if isinstance(value, datetime):
        result = value
    else:
        text = str(value).strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        result = datetime.fromisoformat(text)
    if result.tzinfo is None:
        result = result.replace(tzinfo=UTC)
    return result.astimezone(UTC)


def format_datetime(value: datetime) -> str:
    return parse_datetime(value).astimezone(UTC).isoformat().replace("+00:00", "Z")


def floor_minute(value: datetime) -> datetime:
    """Truncate to the start of the minute."""
    return parse_datetime(value).replace(second=0, microsecond=0)


def minute_bucket(value: datetime | None = None) -> str:
    """Idempotency key for one wall-clock minute, e.g. '2026-10-22T12:34Z'."""
    stamp = floor_minute(value or now())
    return stamp.strftime("%Y-%m-%dT%H:%MZ")


def add_minutes(value: datetime, minutes: int) -> datetime:
    return parse_datetime(value) + timedelta(minutes=float(minutes))


def age_seconds(value: datetime, reference: datetime | None = None) -> float:
    return (parse_datetime(reference or now()) - parse_datetime(value)).total_seconds()


def seconds_until_next_minute(reference: datetime | None = None) -> float:
    """Seconds left before the next minute boundary (always > 0)."""
    stamp = parse_datetime(reference or now())
    next_minute = floor_minute(stamp) + timedelta(minutes=1)
    return max(0.001, (next_minute - stamp).total_seconds())
