"""The honesty charter.

These are not comments about good intentions; they are enforced values. Tests
assert on them. Change a number here and the tests that protect it will fail,
which is the point.
"""

from __future__ import annotations

# --- outcome states -----------------------------------------------------------
# An outcome is UNRESOLVED when the price data needed to judge it never arrived.
# UNRESOLVED is never silently converted into a win, a loss, or a tie.
UNRESOLVED = "UNRESOLVED"
WIN = "WIN"
LOSS = "LOSS"
TIED = "TIED"

# --- delivery states ----------------------------------------------------------
# A signal is SENT only after Telegram confirms a message id. "Probably sent"
# does not exist as a state.
DELIVERY_PENDING = "PENDING"
DELIVERY_SENT = "SENT"
DELIVERY_FAILED = "FAILED"
DELIVERY_UNKNOWN = "UNRESOLVED"  # sent may or may not have happened
DELIVERY_SUPPRESSED = "SUPPRESSED"  # bot paused / not owner action needed

# --- signal lifecycle ---------------------------------------------------------
SEAL_DISCARDED_LATE = "DISCARDED_LATE"  # too slow to reach the entry, dropped
SEAL_ABSTAIN = "ABSTAIN"
SEAL_SIGNAL = "SIGNAL"

# A signal whose entry is already in the past is worthless and misleading.
# If sealing + delivery cannot happen inside this many seconds of the minute
# boundary, the signal is discarded instead of being backdated.
MAX_SEAL_LATENCY_SECONDS = 10

# A pending signal is declared UNRESOLVED once this many minutes have passed
# since its expiry. Waiting longer would leave rows pending forever; resolving
# earlier would risk calling a result unresolved while prices could still land.
UNRESOLVE_AFTER_MINUTES = 120

# --- learning guard -----------------------------------------------------------
# Synthetic/replay data is for watching the pipeline fire. It must never update
# a weight, fit a calibrator, or promote a model.
SYNTHETIC_SOURCES = frozenset({"SYNTHETIC", "SYNTHETIC DEMO", "REPLAY", "SMOKE"})

# --- calibration ladder -------------------------------------------------------
# Below this many confirmed real outcomes the raw model score is reported as
# raw. At or above it an isotonic map is fitted. If the map collapses to one
# level, the gate abstains instead of trading a constant probability.
MIN_CALIBRATION_POINTS = 60
MIN_CALIBRATION_BIN = 20

# --- liveness -----------------------------------------------------------------
# Heartbeats are the only liveness signal. GitHub Actions stands in only when
# the primary runner's heartbeat is older than this.
HEARTBEAT_STALE_SECONDS = 90

# --- costs --------------------------------------------------------------------
# Every settled paper trade pays the spread. A strategy that only wins gross is
# not winning.
DEFAULT_SPREAD_PIPS = 1.2
PIP = 0.0001  # EUR/USD fourth decimal

# --- boundaries ---------------------------------------------------------------
ALLOWED_PAIR = "EUR/USD"
ALLOWED_TIMEFRAME = "1min"

FORBIDDEN_TOPICS = (
    "broker orders",
    "real-money trading",
    "Quotex / Pocket Option / OTC binary options",
    "Martingale or position doubling",
    "guaranteed accuracy claims",
)


def is_synthetic(source: str | None) -> bool:
    """True when the row came from synthetic or replay data."""
    return str(source or "").strip().upper() in SYNTHETIC_SOURCES


def real_outcome_label(label: str | None) -> bool:
    """A label that learning may consume: a real, non-tied, non-unknown result."""
    return label in (WIN, LOSS)


def net_pips(gross_pips: float, spread_pips: float = DEFAULT_SPREAD_PIPS) -> float:
    """Gross move minus the round-turn spread cost, in pips."""
    return float(gross_pips) - float(spread_pips)


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a win rate.

    A 3-of-3 record is not 100%. The interval is reported with every win rate so
    a small sample can never be read as proof of an edge.
    """
    if total <= 0:
        return (0.0, 0.0)
    n = float(total)
    p = float(successes) / n
    denominator = 1.0 + (z * z) / n
    centre = p + (z * z) / (2.0 * n)
    spread = z * ((p * (1.0 - p) / n + (z * z) / (4.0 * n * n)) ** 0.5)
    lower = (centre - spread) / denominator
    upper = (centre + spread) / denominator
    return (max(0.0, lower), min(1.0, upper))
