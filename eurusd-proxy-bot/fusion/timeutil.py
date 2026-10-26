"""Clock helpers: internal UTC storage + system-wide UTC+05:30 display timing.

Internal candle keys, database ``timestamptz`` rows, and idempotency buckets
stay in UTC so settlement matching (``entry_time == candle.timestamp``) never
drifts across offset boundaries. Every operator-facing timestamp (entry,
expiry, status, digest, Web Console, and Telegram messages) renders strictly in
``UTC+05:30``.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

UTC = timezone.utc
DISPLAY_OFFSET = timedelta(hours=5, minutes=30)
DISPLAY_TZ_LABEL = "UTC+05:30"
DISPLAY_TZ = timezone(DISPLAY_OFFSET, name=DISPLAY_TZ_LABEL)


def now() -> datetime:
    return datetime.now(UTC)


def now_display() -> datetime:
    return datetime.now(DISPLAY_TZ)


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


def to_display_tz(value) -> datetime:
    """Convert any datetime or ISO string to the system display timezone (UTC+05:30)."""
    return parse_datetime(value).astimezone(DISPLAY_TZ)


def format_datetime(value: datetime) -> str:
    return parse_datetime(value).astimezone(UTC).isoformat().replace("+00:00", "Z")


def format_display_iso(value) -> str:
    """Format an instant as an ISO-8601 timestamp in UTC+05:30 (e.g. 2026-10-26T17:45:00+05:30)."""
    return to_display_tz(value).isoformat()


def format_display_hm(value, *, include_tz: bool = True) -> str:
    """Format an instant as HH:MM in UTC+05:30 (optionally suffixed with ' UTC+05:30')."""
    hm = to_display_tz(value).strftime("%H:%M")
    return f"{hm} {DISPLAY_TZ_LABEL}" if include_tz else hm


def format_display_datetime(value, *, seconds: bool = True) -> str:
    """Format an instant as 'YYYY-MM-DD HH:MM(:SS) UTC+05:30'."""
    fmt = "%Y-%m-%d %H:%M:%S" if seconds else "%Y-%m-%d %H:%M"
    return f"{to_display_tz(value).strftime(fmt)} {DISPLAY_TZ_LABEL}"


def floor_minute(value: datetime) -> datetime:
    """Truncate to the start of the minute."""
    return parse_datetime(value).replace(second=0, microsecond=0)


def next_future_candle_open(reference: datetime | None = None) -> datetime:
    """Return the open timestamp of the upcoming future M1 candle (> reference).

    At any moment inside minute ``T`` (whether ``T:00.001`` or ``T:59.999``),
    bar ``T`` is already ongoing, so an on-demand signal must target ``T + 1m``
    — never the present forming bar and never a past bar.
    """
    stamp = parse_datetime(reference or now())
    return floor_minute(stamp) + timedelta(minutes=1)


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
