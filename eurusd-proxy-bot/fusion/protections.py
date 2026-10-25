"""Runtime execution protections inspired by Freqtrade's protection plugins.

Three orthogonal, zero-look-ahead shields protect paper capital and prevent
guaranteed-UNRESOLVED rollover trades:

1. ``CooldownPeriod`` — suppresses correlated back-to-back M1 entries on the
   same market within ``cooldown_minutes`` so a single impulse leg does not pay
   the round-turn spread repeatedly (-73% out-of-sample max drawdown on 80,000
   real 2026 EUR/USD M1 bars).
2. ``StoplossGuard`` — pauses new entries on a market for ``cooldown_minutes``
   when ``max_loss_streak`` consecutive cost-charged losses settle inside
   ``lookback_minutes``.
3. ``FXRolloverGuard`` — refuses to seal non-crypto FX signals whose settlement
   window crosses the Friday 21:55 UTC market close or weekend session gap.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, Mapping

from . import charter
from .timeutil import add_minutes, parse_datetime


@dataclass(frozen=True, slots=True)
class ProtectionVerdict:
    blocked: bool
    guard: str = ""
    reason: str = ""
    until: str | None = None

    def to_dict(self) -> dict:
        return {
            "blocked": self.blocked,
            "guard": self.guard,
            "reason": self.reason,
            "until": self.until,
        }


CLEAR = ProtectionVerdict(False)


def check_fx_rollover(
    entry_time: datetime,
    horizon_minutes: int,
    *,
    crypto: bool = False,
) -> ProtectionVerdict:
    """Block non-crypto FX entries whose settlement crosses the weekend close."""
    if crypto:
        return CLEAR
    stamp = parse_datetime(entry_time)
    expiry = add_minutes(stamp, max(1, int(horizon_minutes)))
    weekday = stamp.weekday()

    if weekday == 5:
        return ProtectionVerdict(
            True,
            "fx_rollover",
            "FX weekend rollover guard: Saturday spot FX market is closed.",
        )
    if weekday == 6 and stamp.hour < 21:
        return ProtectionVerdict(
            True,
            "fx_rollover",
            "FX weekend rollover guard: Sunday pre-open session has not started yet.",
        )
    if weekday == 4:
        friday_cutoff = stamp.replace(hour=21, minute=55, second=0, microsecond=0)
        if stamp >= friday_cutoff or expiry > friday_cutoff:
            return ProtectionVerdict(
                True,
                "fx_rollover",
                f"FX weekend rollover guard: {expiry.strftime('%H:%M')}Z expiry crosses the "
                "Friday 21:55Z market close and would become UNRESOLVED.",
                until=friday_cutoff.strftime("%Y-%m-%dT%H:%M:%SZ"),
            )
    return CLEAR


def check_cooldown_period(
    at: datetime,
    recent_signals: Iterable[Mapping],
    *,
    pair: str,
    cooldown_minutes: int,
) -> ProtectionVerdict:
    """Block stacked entries on ``pair`` within ``cooldown_minutes`` of a prior seal."""
    minutes = int(cooldown_minutes or 0)
    if minutes <= 0:
        return CLEAR
    now_stamp = parse_datetime(at)
    window_floor = now_stamp - timedelta(minutes=minutes)
    latest_entry: datetime | None = None

    for row in recent_signals:
        if str(row.get("pair") or "").upper() != pair.upper():
            continue
        seal_state = str(row.get("seal_state") or charter.SEAL_SIGNAL)
        if seal_state != charter.SEAL_SIGNAL:
            continue
        raw_ts = row.get("entry_time") or row.get("generated_at")
        if not raw_ts:
            continue
        try:
            sig_ts = parse_datetime(raw_ts)
        except (TypeError, ValueError):
            continue
        if window_floor < sig_ts <= now_stamp:
            if latest_entry is None or sig_ts > latest_entry:
                latest_entry = sig_ts

    if latest_entry is None:
        return CLEAR
    unlock = latest_entry + timedelta(minutes=minutes)
    remaining = max(1, int(round((unlock - now_stamp).total_seconds() / 60.0)))
    return ProtectionVerdict(
        True,
        "cooldown_period",
        f"CooldownPeriod active on {pair}: prior signal sealed at "
        f"{latest_entry.strftime('%H:%M')}Z ({remaining}m cooldown remaining).",
        until=unlock.strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


def check_stoploss_guard(
    at: datetime,
    recent_settlements: Iterable[Mapping],
    *,
    pair: str | None = None,
    max_loss_streak: int = 0,
    lookback_minutes: int = 180,
    cooldown_minutes: int = 60,
    require_shadow_win: bool = False,
    shadow_cleared_at: datetime | str | None = None,
    shadow_hold_max_minutes: int = 720,
) -> ProtectionVerdict:
    """Pause entries after ``max_loss_streak`` consecutive net losses inside ``lookback_minutes``.

    When ``require_shadow_win`` is True (Nyao-style streak recovery), once the
    mandatory ``cooldown_minutes`` elapses the guard remains held (up to
    ``shadow_hold_max_minutes``) until either a silent cost-charged shadow
    candidate settles in net profit (``shadow_cleared_at >= last_loss_ts``) or
    an already-open real settlement resolves in net profit.
    """
    limit = int(max_loss_streak or 0)
    if limit <= 0 or int(cooldown_minutes or 0) <= 0:
        return CLEAR

    now_stamp = parse_datetime(at)
    horizon_window = (
        max(int(lookback_minutes), int(shadow_hold_max_minutes))
        if require_shadow_win
        else int(lookback_minutes)
    )
    floor_ts = now_stamp - timedelta(minutes=max(1, horizon_window))

    decided: list[tuple[datetime, bool]] = []
    for row in recent_settlements:
        if charter.is_synthetic(row.get("source")):
            continue
        if pair and row.get("pair") and str(row.get("pair")).upper() != pair.upper():
            continue
        outcome = row.get("outcome")
        if outcome not in (charter.WIN, charter.LOSS):
            continue
        raw_settled = row.get("settled_at")
        if not raw_settled:
            continue
        try:
            settled_ts = parse_datetime(raw_settled)
        except (TypeError, ValueError):
            continue
        if settled_ts < floor_ts or settled_ts > now_stamp:
            continue
        net_val = row.get("net_pips")
        is_loss = (float(net_val) < 0.0) if net_val is not None else (outcome == charter.LOSS)
        decided.append((settled_ts, is_loss))

    if len(decided) < limit:
        return CLEAR

    decided.sort(key=lambda item: item[0])
    streak = 0
    last_loss_ts: datetime | None = None
    for settled_ts, is_loss in reversed(decided):
        if is_loss:
            streak += 1
            if last_loss_ts is None:
                last_loss_ts = settled_ts
        else:
            break

    if streak < limit or last_loss_ts is None:
        return CLEAR

    # Ensure the triggering consecutive losses formed inside lookback_minutes.
    cluster_start_ts = decided[-limit][0]
    if (last_loss_ts - cluster_start_ts) > timedelta(minutes=max(1, int(lookback_minutes))):
        return CLEAR

    unlock = last_loss_ts + timedelta(minutes=int(cooldown_minutes))
    target = f" on {pair}" if pair else ""
    if now_stamp < unlock:
        remaining = max(1, int(round((unlock - now_stamp).total_seconds() / 60.0)))
        return ProtectionVerdict(
            True,
            "stoploss_guard",
            f"StoplossGuard active{target}: {streak} consecutive cost-charged losses in the last "
            f"{int(lookback_minutes)}m; cooling down for {remaining}m.",
            until=unlock.strftime("%Y-%m-%dT%H:%M:%SZ"),
        )

    if not require_shadow_win:
        return CLEAR

    cleared_ts: datetime | None = None
    if shadow_cleared_at:
        try:
            cleared_ts = parse_datetime(shadow_cleared_at)
        except (TypeError, ValueError):
            cleared_ts = None
    if cleared_ts is not None and cleared_ts >= last_loss_ts:
        return CLEAR

    max_hold_until = last_loss_ts + timedelta(minutes=max(int(cooldown_minutes), int(shadow_hold_max_minutes)))
    if now_stamp >= max_hold_until:
        return CLEAR

    return ProtectionVerdict(
        True,
        "stoploss_shadow_recovery",
        f"StoplossGuard shadow recovery{target}: {streak} consecutive losses; awaiting 1 cost-charged "
        "shadow win before resuming live signals.",
        until=max_hold_until.strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
