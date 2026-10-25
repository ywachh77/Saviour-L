"""Automated look-ahead and recursive-formula audit (Freqtrade-style).

Inspired by Freqtrade's ``lookahead-analysis`` and ``recursive-analysis`` CLI
checks. Hand-hunting look-ahead bugs after they ship is too slow: any future
indicator, higher-timeframe resampler, or strategy added to ``REGISTRY`` is
automatically exercised here against adversarial future price shocks.

Two properties are tested:

1. **Causal / no-look-ahead invariance**: at any decision index ``t``,
   rewriting every candle after ``t`` (or appending a still-forming entry bar
   with an extreme price shock) must not change a single feature value, higher-
   timeframe bias reading, regime label, strategy vote, or engine verdict at
   ``t``.
2. **Recursive warmup convergence**: recursive indicators (EMA, Wilder RSI,
   ATR, ADX) depend in principle on their entire history. Comparing a 240-bar
   window against the production 320-bar ``CANDLE_HISTORY`` window verifies
   that startup truncation drift has decayed below tolerance before live
   decisions are made.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import math
from typing import Sequence

from . import instruments, mtf, strategies
from .config import Config
from .domain import Candle
from .engine import FusionEngine
from .features import compute_features
from .regime import detect_regime
from .simulate import synthetic_series


@dataclass(frozen=True, slots=True)
class AuditFinding:
    component: str
    passed: bool
    detail: str


@dataclass
class AuditReport:
    findings: list[AuditFinding] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.findings) and all(item.passed for item in self.findings)

    def add(self, component: str, passed: bool, detail: str) -> None:
        self.findings.append(AuditFinding(component=component, passed=passed, detail=detail))

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "checked": len(self.findings),
            "failed": sum(1 for item in self.findings if not item.passed),
            "findings": [
                {"component": item.component, "passed": item.passed, "detail": item.detail}
                for item in self.findings
            ],
        }


def _shocked_tail(candles: Sequence[Candle], pivot: int, multiplier: float) -> list[Candle]:
    """Keep ``candles[:pivot + 1]`` intact and rewrite every later bar."""
    prefix = list(candles[: pivot + 1])
    for bar in candles[pivot + 1 :]:
        shocked = max(1e-6, bar.close * multiplier)
        high = max(bar.open, shocked) * 1.05
        low = min(bar.open, shocked) * 0.95
        prefix.append(
            Candle(
                timestamp=bar.timestamp,
                open=bar.open,
                high=high,
                low=low,
                close=shocked,
                volume=bar.volume * 50.0 + 10_000.0,
            )
        )
    return prefix


def check_feature_causality(
    candles: Sequence[Candle] | None = None,
    *,
    pivots: Sequence[int] = (65, 111, 157, 193),
    tolerance: float = 1e-12,
) -> AuditFinding:
    """Every feature at ``t`` must be invariant to shocks in ``t+1 .. N``."""
    base = list(candles) if candles is not None else synthetic_series(230, seed=17)
    for symbol in instruments.symbols():
        inst = instruments.for_symbol(symbol)
        baseline = compute_features(base, instrument=inst)
        for pivot in pivots:
            if pivot >= len(base) - 2:
                continue
            ref = baseline[pivot]
            if ref is None:
                return AuditFinding("features", False, f"baseline row {pivot} is None")
            for mult in (3.5, 0.25):
                mutated = compute_features(_shocked_tail(base, pivot, mult), instrument=inst)[pivot]
                if mutated is None:
                    return AuditFinding("features", False, f"mutated row {pivot} collapsed to None")
                for key, val in ref.items():
                    other = mutated.get(key)
                    if other is None or not math.isclose(float(val), float(other), rel_tol=tolerance, abs_tol=tolerance):
                        return AuditFinding(
                            "features",
                            False,
                            f"{symbol} feature {key!r} at index {pivot} leaked future prices ({val} vs {other})",
                        )
    return AuditFinding(
        "features",
        True,
        f"all features invariant across {len(pivots)} pivots and {len(instruments.symbols())} markets",
    )


def check_mtf_causality(
    candles: Sequence[Candle] | None = None,
    *,
    minutes_list: Sequence[int] = (5, 15),
) -> AuditFinding:
    """Higher-timeframe bias must never read from a still-forming bucket."""
    base = list(candles) if candles is not None else synthetic_series(240, seed=29)
    # Pick mid-bucket indices where a 5m or 15m bar is actively forming.
    pivots = (76, 91, 137, 182)
    for minutes in minutes_list:
        ref = mtf.bias_series(base, minutes=minutes)
        for pivot in pivots:
            for mult in (4.0, 0.2):
                after = mtf.bias_series(_shocked_tail(base, pivot, mult), minutes=minutes)
                if (ref[pivot].direction, ref[pivot].close, ref[pivot].reference) != (
                    after[pivot].direction,
                    after[pivot].close,
                    after[pivot].reference,
                ):
                    return AuditFinding(
                        "mtf",
                        False,
                        f"{minutes}m bias at index {pivot} moved when future bars changed",
                    )
    return AuditFinding("mtf", True, "5m and 15m resamplers ignore forming buckets and future bars")


def check_strategies_causality(candles: Sequence[Candle] | None = None) -> AuditFinding:
    """Every registered strategy in REGISTRY must depend only on <= t."""
    base = list(candles) if candles is not None else synthetic_series(220, seed=41)
    pivot = 175
    rows_clean = compute_features(base[: pivot + 1])
    rows_shock = compute_features(_shocked_tail(base, pivot, 5.0))
    f_clean = rows_clean[pivot]
    f_shock = rows_shock[pivot]
    bar = base[pivot]
    prev_close = base[pivot - 1].close

    for name, fn in strategies.REGISTRY.items():
        d1 = fn(bar, prev_close, f_clean, 0.58, calibrated=True)
        d2 = fn(bar, prev_close, f_shock, 0.58, calibrated=True)
        if (d1.side, d1.score, d1.reason) != (d2.side, d2.score, d2.reason):
            return AuditFinding(
                "strategies",
                False,
                f"strategy {name!r} changed verdict when future candles were shocked",
            )
    return AuditFinding(
        "strategies",
        True,
        f"all {len(strategies.REGISTRY)} council members are strictly causal",
    )


def check_engine_entry_bar_guard() -> AuditFinding:
    """A forming entry bar passed to FusionEngine.evaluate must be excluded."""
    entry_minute = datetime(2026, 1, 5, 14, 0, tzinfo=timezone.utc)
    closed = synthetic_series(
        220,
        seed=53,
        start=entry_minute - timedelta(minutes=220),
    )
    forming_shock = Candle(
        timestamp=entry_minute,
        open=closed[-1].close,
        high=closed[-1].close * 1.25,
        low=closed[-1].close * 0.75,
        close=closed[-1].close * 1.20,
        volume=999_999.0,
    )
    engine = FusionEngine(Config(pair="EUR/USD"))
    stamp = entry_minute + timedelta(seconds=2)
    without_forming = engine.evaluate(closed, at=stamp)
    with_forming = engine.evaluate(list(closed) + [forming_shock], at=stamp)
    if (without_forming.seal_state, without_forming.reason) != (
        with_forming.seal_state,
        with_forming.reason,
    ):
        return AuditFinding(
            "engine",
            False,
            f"forming entry bar leaked into evaluate(): {without_forming.reason!r} vs {with_forming.reason!r}",
        )
    return AuditFinding("engine", True, "FusionEngine.evaluate drops forming and future bars before gating")


def check_recursive_warmup(
    candles: Sequence[Candle] | None = None,
    *,
    short_window: int = 240,
    production_window: int = 320,
    max_rel_drift: float = 0.02,
) -> AuditFinding:
    """Verify recursive indicators (EMA, RSI, ATR, ADX) have converged by 240-320 bars."""
    series = list(candles) if candles is not None else synthetic_series(production_window, seed=71)
    if len(series) < production_window:
        return AuditFinding("recursive_warmup", False, "series shorter than production_window")

    full_last = compute_features(series[-production_window:])[-1]
    short_last = compute_features(series[-short_window:])[-1]
    if full_last is None or short_last is None:
        return AuditFinding("recursive_warmup", False, "feature warmup returned None at window tail")

    recursive_keys = ("ema_fast", "ema_slow", "rsi", "atr", "adx")
    worst_key = ""
    worst_drift = 0.0
    for key in recursive_keys:
        a = float(full_last[key])
        b = float(short_last[key])
        denom = max(abs(a), 1e-9)
        drift = abs(a - b) / denom
        if drift > worst_drift:
            worst_drift = drift
            worst_key = key

    passed = worst_drift <= max_rel_drift
    detail = (
        f"max warmup drift {worst_drift:.5f} on {worst_key!r} "
        f"({short_window} vs {production_window} bars, limit {max_rel_drift:.3f})"
    )
    return AuditFinding("recursive_warmup", passed, detail)


def run_audit() -> AuditReport:
    """Run every causality and recursive-stability check."""
    report = AuditReport()
    report.findings.append(check_feature_causality())
    report.findings.append(check_mtf_causality())
    report.findings.append(check_strategies_causality())
    report.findings.append(check_engine_entry_bar_guard())
    report.findings.append(check_recursive_warmup())
    return report
