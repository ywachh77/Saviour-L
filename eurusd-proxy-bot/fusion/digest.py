"""The automatic daily digest (00:05 UTC), and the human-action alert policy.

The digest reports what happened, including the parts that look bad. It never
rounds an unresolved outcome into a result and always shows the Wilson interval
next to a win rate, because a small sample is not evidence.
"""

from __future__ import annotations

from typing import Sequence

from . import charter, heartbeat
from .settle import LedgerStats

ALERT_STALE_FEED = "stale_feed"
ALERT_DELIVERY_UNKNOWN = "delivery_unknown"
ALERT_KEY_POOL = "key_pool"
ALERT_NEGATIVE_EXPECTANCY = "negative_expectancy"
ALERT_ROLLBACK = "rollback"
ALERT_STORE_ERROR = "store_error"

# A human is only interrupted for these. Everything else waits for the digest.
HUMAN_ACTION_ALERTS = (
    ALERT_STALE_FEED,
    ALERT_DELIVERY_UNKNOWN,
    ALERT_KEY_POOL,
    ALERT_NEGATIVE_EXPECTANCY,
    ALERT_ROLLBACK,
    ALERT_STORE_ERROR,
)


def format_rate(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def build_digest(
    *,
    day: str,
    signals: Sequence[dict],
    stats: LedgerStats,
    unresolved: int,
    calibration_state: str,
    calibration_samples: int,
    model_mode: str,
    council_weights: dict[str, float],
    learning_events: Sequence[dict],
    key_pool: dict,
    runner: str,
    alerts: Sequence[dict],
    nightly: dict | None = None,
) -> str:
    lines = [f"FUSION BOT daily digest — {day} (UTC)", f"Runner: {runner}", ""]

    sent = sum(1 for row in signals if row.get("delivery_state") == charter.DELIVERY_SENT)
    suppressed = sum(1 for row in signals if row.get("delivery_state") == charter.DELIVERY_SUPPRESSED)
    discarded = sum(1 for row in signals if row.get("seal_state") == charter.SEAL_DISCARDED_LATE)
    lines.append(f"Signals: {len(signals)} sealed, {sent} delivered, {suppressed} while paused, {discarded} dropped as late.")
    lines.append("")

    lines.append("Ledger (paper, spread charged on every trade):")
    lines.append(f"  settled {stats.settled} · wins {stats.wins} · losses {stats.losses} · ties {stats.tied}")
    lines.append(f"  unresolved {unresolved} (never counted as a win or a loss)")
    low, high = stats.wilson_low, stats.wilson_high
    lines.append(
        f"  win rate {format_rate(stats.win_rate)} (95% Wilson {low * 100:.1f}%–{high * 100:.1f}%)"
    )
    lines.append(f"  net {stats.net_pips:+.1f} pips after costs")
    lines.append("")

    lines.append(f"Model: {model_mode} authority · calibration {calibration_state} on {calibration_samples} outcomes.")
    active = {name: value for name, value in council_weights.items() if value > 0}
    if active:
        spoken = ", ".join(f"{name} {value:.2f}" for name, value in sorted(active.items()))
        lines.append(f"Learned prototypes with a voice: {spoken}")
    else:
        lines.append("Learned prototypes: none yet (each needs 30 real settled outcomes).")
    lines.append("")

    if nightly:
        lines.append(
            f"Nightly: {'promoted' if nightly.get('promoted') else 'no promotion'} — {nightly.get('reason', '')}"
        )
        lines.append("")

    if key_pool:
        rows = key_pool.get("slots") or []
        enabled = sum(1 for row in rows if row.get("enabled"))
        day_used = sum(int(row.get("day_used", 0)) for row in rows)
        day_limit = sum(int(row.get("day_limit", 0)) for row in rows)
        lines.append(f"Key pool: {enabled}/{len(rows)} slots enabled · {day_used}/{day_limit} credits used today.")
        lines.append("")

    if learning_events:
        applied = sum(1 for event in learning_events if event.get("applied"))
        lines.append(f"Learning: {applied} online updates from confirmed real outcomes.")
        lines.append("")

    if alerts:
        lines.append(f"Alerts raised today: {len(alerts)}")
        for row in alerts[-5:]:
            lines.append(f"  · {row.get('kind')}: {row.get('message')}")
        lines.append("")

    lines.append("Research output only. No broker order is ever placed.")
    return "\n".join(lines)


def build_status(
    *,
    config,
    paused: bool,
    runner: str,
    heartbeat_row: dict | None,
    stats: LedgerStats,
    unresolved: int,
    model_mode: str,
    calibration_state: str,
    calibration_samples: int,
    key_pool: dict,
    last_cycle: dict | None,
    store_errors: Sequence[str],
) -> str:
    age = heartbeat.heartbeat_age_seconds(heartbeat_row)
    lines = [
        "FUSION BOT status",
        f"  pair {config.pair} · horizon {config.horizon_minutes}m · runner {runner}",
        f"  signals: {'PAUSED' if paused else 'LIVE'}",
        f"  heartbeat: {'missing' if age is None else f'{age:.0f}s ago'}",
        f"  model: {model_mode} · calibration {calibration_state} ({calibration_samples} outcomes)",
        f"  ledger: {stats.wins}W/{stats.losses}L/{stats.tied}T · {stats.unresolved} unresolved · "
        f"net {stats.net_pips:+.1f} pips",
    ]
    if stats.win_rate is not None:
        lines.append(
            f"  win rate {format_rate(stats.win_rate)} "
            f"(95% Wilson {stats.wilson_low * 100:.1f}%–{stats.wilson_high * 100:.1f}%)"
        )
    if key_pool:
        rows = key_pool.get("slots") or []
        enabled = sum(1 for row in rows if row.get("enabled"))
        lines.append(f"  key pool: {enabled}/{len(rows)} slots enabled")
    if last_cycle:
        lines.append(f"  last cycle: {last_cycle.get('seal_state', 'n/a')} — {last_cycle.get('reason', '')[:160]}")
    if store_errors:
        lines.append(f"  store errors: {len(store_errors)} (latest: {store_errors[-1][:120]})")
    return "\n".join(lines)


def alert_message(kind: str, detail: str) -> str:
    prefix = {
        ALERT_STALE_FEED: "Data feed is stale",
        ALERT_DELIVERY_UNKNOWN: "Delivery state unknown",
        ALERT_KEY_POOL: "Key pool needs attention",
        ALERT_NEGATIVE_EXPECTANCY: "Expectancy turned negative",
        ALERT_ROLLBACK: "Model rolled back",
        ALERT_STORE_ERROR: "Storage write failed",
    }.get(kind, "Attention needed")
    return f"{prefix}: {detail}"
