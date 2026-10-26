"""Deterministic macro-release blackout filter.

Why a blackout filter belongs here
----------------------------------

Around major scheduled USD / EUR releases (above all US Non-Farm Payrolls at
08:30 America/New_York on the first Friday of the month, FOMC rate decisions,
and CPI prints), one-minute spreads blow out, order books thin, and price
whipsaws both sides inside seconds. A technical indicator computed on pre-news
calm bars has nothing useful to say about the release bar itself.

Because this engine uses zero paid third-party calendar feeds and zero external
dependencies, two honest mechanisms are provided:

1. **Deterministic recurring calendar**: US Non-Farm Payrolls (first Friday of
   every month, 08:30 America/New_York, DST-aware), blacked out from 5 minutes
   before to 15 minutes after the print on USD-quoted markets.
2. **Explicit operator windows** (``NEWS_BLACKOUT_WINDOWS``): comma-separated
   UTC intervals ``YYYY-MM-DDTHH:MM/YYYY-MM-DDTHH:MM`` (or a single timestamp
   ``YYYY-MM-DDTHH:MM`` which applies the standard ``[-5m, +15m]`` buffer) for
   movable releases such as FOMC, CPI, or ECB decisions.

Unscheduled headlines are NOT pretended to be predictable here; those are
handled after the fact by the volatility-band gate in ``strategies.py`` and the
spread gate in ``datagate.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .timeutil import format_display_hm, parse_datetime

PRE_RELEASE_MINUTES = 5
POST_RELEASE_MINUTES = 15


@dataclass(frozen=True, slots=True)
class BlackoutCheck:
    blocked: bool
    event: str = ""
    reason: str = ""


def _us_eastern_offset_hours(utc_stamp: datetime) -> int:
    """Fallback US Eastern offset (-4 EDT, -5 EST) when system tzdata is absent."""
    year = utc_stamp.year
    # Second Sunday in March at 07:00 UTC -> First Sunday in November at 06:00 UTC.
    march_first = datetime(year, 3, 1, tzinfo=timezone.utc)
    days_to_sunday = (6 - march_first.weekday()) % 7
    dst_start = datetime(year, 3, 1 + days_to_sunday + 7, 7, 0, tzinfo=timezone.utc)

    nov_first = datetime(year, 11, 1, tzinfo=timezone.utc)
    nov_sunday = (6 - nov_first.weekday()) % 7
    dst_end = datetime(year, 11, 1 + nov_sunday, 6, 0, tzinfo=timezone.utc)

    return -4 if dst_start <= utc_stamp < dst_end else -5


def _to_new_york(stamp: datetime) -> datetime:
    utc_stamp = parse_datetime(stamp).astimezone(timezone.utc)
    try:
        return utc_stamp.astimezone(ZoneInfo("America/New_York"))
    except ZoneInfoNotFoundError:
        offset = timezone(timedelta(hours=_us_eastern_offset_hours(utc_stamp)))
        return utc_stamp.astimezone(offset)


def is_nfp_window(
    stamp: datetime,
    *,
    pre_minutes: int = PRE_RELEASE_MINUTES,
    post_minutes: int = POST_RELEASE_MINUTES,
) -> bool:
    """True inside ``[08:30 - pre_minutes, 08:30 + post_minutes]`` ET on first Friday."""
    ny = _to_new_york(stamp)
    # weekday() == 4 is Friday; day 1..7 is the first Friday of the month.
    if ny.weekday() != 4 or ny.day > 7:
        return False
    release = ny.replace(hour=8, minute=30, second=0, microsecond=0)
    start = release - timedelta(minutes=max(0, int(pre_minutes)))
    end = release + timedelta(minutes=max(0, int(post_minutes)))
    return start <= ny <= end


def parse_windows(
    raw: str,
    *,
    pre_minutes: int = PRE_RELEASE_MINUTES,
    post_minutes: int = POST_RELEASE_MINUTES,
) -> list[tuple[datetime, datetime, str]]:
    """Parse ``NEWS_BLACKOUT_WINDOWS`` into ``(start_utc, end_utc, label)`` tuples.

    Accepted entries (comma-separated):
      * ``2026-11-04T19:00Z`` -> buffered ``[18:55Z, 19:15Z]``
      * ``2026-11-04T18:50Z/2026-11-04T19:30Z`` -> exact interval
      * ``FOMC@2026-11-04T19:00Z`` -> named buffered release
    Malformed entries are skipped rather than crashing the live loop.
    """
    windows: list[tuple[datetime, datetime, str]] = []
    if not raw or not raw.strip():
        return windows

    for token in raw.split(","):
        item = token.strip()
        if not item:
            continue
        label = "scheduled release"
        if "@" in item:
            prefix, _, rest = item.partition("@")
            if prefix.strip():
                label = prefix.strip()
            item = rest.strip()
        try:
            if "/" in item:
                left, _, right = item.partition("/")
                start = parse_datetime(left.strip()).astimezone(timezone.utc)
                end = parse_datetime(right.strip()).astimezone(timezone.utc)
            else:
                centre = parse_datetime(item).astimezone(timezone.utc)
                start = centre - timedelta(minutes=max(0, int(pre_minutes)))
                end = centre + timedelta(minutes=max(0, int(post_minutes)))
        except (ValueError, TypeError):
            continue
        if end >= start:
            windows.append((start, end, label))
    return windows


def check_blackout(
    stamp: datetime,
    *,
    pair: str = "EUR/USD",
    extra_windows: str = "",
    pre_minutes: int = PRE_RELEASE_MINUTES,
    post_minutes: int = POST_RELEASE_MINUTES,
) -> BlackoutCheck:
    """Return whether ``stamp`` sits inside a high-impact macro blackout window."""
    utc_stamp = parse_datetime(stamp).astimezone(timezone.utc)
    clean_pair = (pair or "").upper()

    for start, end, label in parse_windows(
        extra_windows, pre_minutes=pre_minutes, post_minutes=post_minutes
    ):
        if start <= utc_stamp <= end:
            return BlackoutCheck(
                True,
                label,
                f"News blackout: inside configured {label} window "
                f"({format_display_hm(start)}–{format_display_hm(end)}).",
            )

    if "USD" in clean_pair and is_nfp_window(
        utc_stamp, pre_minutes=pre_minutes, post_minutes=post_minutes
    ):
        return BlackoutCheck(
            True,
            "US NFP",
            "News blackout: US Non-Farm Payrolls release window (first Friday 08:30 ET).",
        )

    return BlackoutCheck(False)
