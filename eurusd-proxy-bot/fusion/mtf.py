"""Higher-timeframe context, built from the bars we already have.

The charter fixes the feed at one-minute bars, so there is no second timeframe
to fetch -- but there is no reason to be blind above one minute either. Five
one-minute bars *are* a five-minute bar. Resampling is arithmetic on data we
have already paid for and costs no API credits.

Why it matters: a one-minute chart is mostly noise. A move that looks decisive
on M1 is very often a pullback inside a five-minute move going the other way,
and the single most common advice in the scalping literature is to take M1
entries only in the direction of the higher-timeframe tilt. It is also the one
piece of the standard multi-timeframe playbook that is free here.

One rule is absolute and mirrors the look-ahead fix in domain.closed_before:
**a higher-timeframe bar that is still forming is not used.** A five-minute bar
stamped 10:45 is not complete until 10:50, and its close does not exist before
then. Partial buckets are dropped rather than rounded off, because a bias read
off a half-built bar is just look-ahead with extra steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Sequence

from .domain import Candle
from .indicators import ema

DEFAULT_MINUTES = 5
DEFAULT_EMA_PERIOD = 10

UP = 1
DOWN = -1
FLAT = 0


@dataclass(frozen=True, slots=True)
class Bias:
    """Which way the higher timeframe is leaning, and why."""

    minutes: int
    direction: int  # UP, DOWN or FLAT
    close: float
    reference: float  # the EMA the close is compared against
    bars: int
    reason: str
    slope_bps: float = 0.0  # 1-bar closed HTF EMA slope in basis points of price

    @property
    def known(self) -> bool:
        return self.direction != FLAT

    def agrees_with(self, side: str | None) -> bool:
        """True when a trade in ``side`` goes with the higher-timeframe tilt."""
        if not side or not self.known:
            return False
        wanted = UP if str(side).upper() == "BUY" else DOWN
        return self.direction == wanted

    def describe(self) -> str:
        word = {UP: "up", DOWN: "down", FLAT: "flat"}.get(self.direction, "unknown")
        return f"{self.minutes}m tilt is {word} ({self.reason})"


def resample(candles: Sequence[Candle], minutes: int = DEFAULT_MINUTES) -> list[Candle]:
    """Collapse one-minute bars into ``minutes``-minute bars.

    Buckets are aligned to wall-clock boundaries so the result matches what a
    real higher-timeframe feed would have produced, and any bucket that has not
    yet collected all of its bars is dropped: it is still forming.
    """
    if minutes <= 1:
        return list(candles)

    buckets: dict = {}
    for candle in candles:
        stamp = candle.timestamp
        floored = stamp.replace(
            minute=(stamp.minute // minutes) * minutes, second=0, microsecond=0
        )
        buckets.setdefault(floored, []).append(candle)

    out: list[Candle] = []
    for stamp in sorted(buckets):
        group = buckets[stamp]
        if len(group) < minutes:
            # Still forming. Using it would leak the current minute's prices
            # into a bar that pretends to be finished.
            continue
        out.append(
            Candle(
                timestamp=stamp,
                open=group[0].open,
                high=max(item.high for item in group),
                low=min(item.low for item in group),
                close=group[-1].close,
                volume=sum(item.volume for item in group),
            )
        )
    return out


def bias(
    candles: Sequence[Candle],
    minutes: int = DEFAULT_MINUTES,
    ema_period: int = DEFAULT_EMA_PERIOD,
    flat_band: float = 0.0,
    min_gap_bps: float = 0.0,
    require_slope: bool = False,
) -> Bias:
    """Which way the ``minutes``-minute timeframe is leaning.

    The tilt is the sign of (close - EMA) on the resampled series. ``flat_band``
    and ``min_gap_bps`` keep a market that is hugging a flat average from
    counting as a trend: inside the dead-zone the answer is "flat", which
    blocks rather than permits. When ``require_slope`` is True, a BUY tilt also
    requires a rising closed HTF EMA and a SELL tilt requires a falling one.
    """
    bars = resample(candles, minutes)
    needed = ema_period + 1
    if len(bars) < needed:
        return Bias(
            minutes,
            FLAT,
            0.0,
            0.0,
            len(bars),
            f"only {len(bars)} closed {minutes}m bars, {needed} needed to place the average",
            0.0,
        )

    closes = [bar.close for bar in bars]
    averages = ema(closes, ema_period)
    reference = averages[-1]
    prev_ref = averages[-2] if len(averages) >= 2 else None
    last = closes[-1]
    if reference is None:
        return Bias(minutes, FLAT, last, 0.0, len(bars), "the higher-timeframe average is undefined", 0.0)

    slope_bps = (
        ((float(reference) - float(prev_ref)) / last * 10000.0)
        if (prev_ref is not None and abs(last) > 1e-15)
        else 0.0
    )
    gap = last - reference
    gap_bps = (gap / last * 10000.0) if abs(last) > 1e-15 else 0.0
    if abs(gap) <= flat_band:
        return Bias(
            minutes,
            FLAT,
            last,
            float(reference),
            len(bars),
            f"close sits on its {minutes}m average (gap {gap:+.5f} inside the {flat_band:+.5f} band)",
            slope_bps,
        )
    if min_gap_bps > 0.0 and abs(gap_bps) < min_gap_bps:
        return Bias(
            minutes,
            FLAT,
            last,
            float(reference),
            len(bars),
            f"close hugs its {minutes}m average ({gap_bps:+.2f} bps inside {min_gap_bps:.2f} bps dead-zone)",
            slope_bps,
        )
    if require_slope and prev_ref is not None:
        if (gap > 0 and slope_bps <= 0.0) or (gap < 0 and slope_bps >= 0.0):
            return Bias(
                minutes,
                FLAT,
                last,
                float(reference),
                len(bars),
                f"{minutes}m EMA slope ({slope_bps:+.2f} bps) opposes price gap ({gap_bps:+.2f} bps)",
                slope_bps,
            )
    direction = UP if gap > 0 else DOWN
    return Bias(
        minutes,
        direction,
        last,
        float(reference),
        len(bars),
        f"close {last:.5f} vs {ema_period}-bar average {reference:.5f} on {minutes}m",
        slope_bps,
    )


def bias_series(
    candles: Sequence[Candle],
    minutes: int = DEFAULT_MINUTES,
    ema_period: int = DEFAULT_EMA_PERIOD,
    flat_band: float = 0.0,
    min_gap_bps: float = 0.0,
    require_slope: bool = False,
) -> list[Bias]:
    """The higher-timeframe tilt as it stood at each one-minute bar.

    This exists because calling ``bias()`` once per bar would resample the
    whole prefix every time: quadratic, and on a 250k-bar backfill that is
    measured in hours rather than seconds. The resampled series and its average
    are computed once and then looked up per bar.

    The lookup carries the same rule as everywhere else: a higher-timeframe bar
    only becomes usable once its window has closed. At 10:47 the five-minute
    bar covering 10:45-10:50 is still forming, so the freshest honest reading
    is the one that closed at 10:45.
    """
    from bisect import bisect_right

    candles = list(candles)
    bars = resample(candles, minutes)
    closes = [bar.close for bar in bars]
    averages = ema(closes, ema_period)
    ends = [bar.timestamp + timedelta(minutes=minutes) for bar in bars]

    series: list[Bias] = []
    for candle in candles:
        position = bisect_right(ends, candle.timestamp) - 1
        if position < 0 or position < ema_period - 1:
            series.append(
                Bias(
                    minutes,
                    FLAT,
                    candle.close,
                    0.0,
                    max(0, position + 1),
                    f"no {minutes}m average yet ({max(0, position + 1)} closed bars, {ema_period} needed)",
                    0.0,
                )
            )
            continue
        reference = averages[position]
        if reference is None:
            series.append(
                Bias(minutes, FLAT, candle.close, 0.0, position + 1, "the higher-timeframe average is undefined", 0.0)
            )
            continue
        prev_ref = averages[position - 1] if position >= 1 else None
        last = closes[position]
        slope_bps = (
            ((float(reference) - float(prev_ref)) / last * 10000.0)
            if (prev_ref is not None and abs(last) > 1e-15)
            else 0.0
        )
        gap = last - reference
        gap_bps = (gap / last * 10000.0) if abs(last) > 1e-15 else 0.0
        if abs(gap) <= flat_band:
            series.append(
                Bias(
                    minutes,
                    FLAT,
                    last,
                    float(reference),
                    position + 1,
                    f"close sits on its {minutes}m average (gap {gap:+.5f} inside the {flat_band:+.5f} band)",
                    slope_bps,
                )
            )
            continue
        if min_gap_bps > 0.0 and abs(gap_bps) < min_gap_bps:
            series.append(
                Bias(
                    minutes,
                    FLAT,
                    last,
                    float(reference),
                    position + 1,
                    f"close hugs its {minutes}m average ({gap_bps:+.2f} bps inside {min_gap_bps:.2f} bps dead-zone)",
                    slope_bps,
                )
            )
            continue
        if require_slope and prev_ref is not None:
            if (gap > 0 and slope_bps <= 0.0) or (gap < 0 and slope_bps >= 0.0):
                series.append(
                    Bias(
                        minutes,
                        FLAT,
                        last,
                        float(reference),
                        position + 1,
                        f"{minutes}m EMA slope ({slope_bps:+.2f} bps) opposes price gap ({gap_bps:+.2f} bps)",
                        slope_bps,
                    )
                )
                continue
        series.append(
            Bias(
                minutes,
                UP if gap > 0 else DOWN,
                last,
                float(reference),
                position + 1,
                f"close {last:.5f} vs {ema_period}-bar average {reference:.5f} on {minutes}m",
                slope_bps,
            )
        )
    return series
