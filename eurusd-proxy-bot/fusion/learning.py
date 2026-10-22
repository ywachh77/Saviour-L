"""Learning: prequential online updates and gated nightly walk-forward.

Three rules guard every update:

1. **Real data only.** Synthetic or replay outcomes never touch a weight. The
   smoke test can fire the whole pipeline and still train nothing.
2. **Test before learn.** The model's prediction is scored *before* the update
   is applied, so the recorded error is genuinely out-of-sample.
3. **Gates, not vibes.** Nightly refits are promoted only when walk-forward
   beats the champion on calibration, direction and cost-charged expectancy.
   Otherwise the champion stays and the reason is recorded.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from . import charter
from .calibration import (
    IsotonicCalibrator,
    brier_score,
    expected_calibration_error,
    fit_validated,
)
from .domain import Candle
from .features import compute_features
from .model import (
    MODE_ADVISORY,
    MODE_VETO,
    BootstrapEnsemble,
    OnlineLogisticModel,
    evaluate,
)
from .timeutil import add_minutes, format_datetime, now

# A candidate must clear all of these to replace the champion.
MIN_WALK_FORWARD_FOLDS = 3
PROMOTION_MIN_ACCURACY = 0.505  # direction, barely better than a coin
PROMOTION_MIN_SKILL = 0.0  # Brier better than a 0.50 base rate


@dataclass(frozen=True, slots=True)
class Sample:
    timestamp: str
    features: dict  # typed as dict[str, float] at runtime
    label: int
    gross_pips: float

    def to_pair(self) -> tuple[dict, int]:
        return self.features, self.label


def build_training_samples(candles: Sequence[Candle], horizon: int = 5) -> list[Sample]:
    """Label a candle series the way a live signal would actually be settled.

    A signal sealed after candle ``i`` closes enters at the OPEN of candle
    ``i+1`` and exits at the CLOSE of candle ``i+horizon``. The label follows
    exactly that path, so training targets the same quantity the bot trades.
    """
    candles = list(candles)
    feature_rows = compute_features(candles)
    samples: list[Sample] = []
    for index, features in enumerate(feature_rows):
        if features is None:
            continue
        entry_index = index + 1
        exit_index = index + int(horizon)
        if exit_index >= len(candles):
            break
        entry_price = candles[entry_index].open
        exit_price = candles[exit_index].close
        gross_pips = (exit_price - entry_price) / charter.PIP
        samples.append(
            Sample(
                timestamp=format_datetime(candles[index].timestamp),
                features=features,
                label=1 if gross_pips > 0 else (0 if gross_pips < 0 else 1),
                gross_pips=gross_pips,
            )
        )
    # Ties are dropped rather than guessed: they carry no direction signal.
    return [item for item in samples if not math.isclose(item.gross_pips, 0.0, abs_tol=1e-12)]


# --------------------------------------------------------------------- online
@dataclass(frozen=True, slots=True)
class OnlineStep:
    applied: bool
    reason: str
    predicted: float | None = None
    raw: float | None = None
    label: int | None = None
    squared_error: float | None = None

    def to_dict(self) -> dict:
        return {
            "applied": self.applied,
            "reason": self.reason,
            "predicted": self.predicted,
            "raw": self.raw,
            "label": self.label,
            "squared_error": self.squared_error,
        }


def online_step(
    model: OnlineLogisticModel | None,
    features: dict[str, float] | None,
    label: int | None,
    *,
    source: str,
    learning_rate: float = 0.02,
    allow_synthetic: bool = False,
) -> OnlineStep:
    """Score first, then learn. One update per confirmed, non-tied real outcome."""
    if charter.is_synthetic(source) and not allow_synthetic:
        return OnlineStep(False, "Synthetic outcome: the charter forbids learning from it.")
    if label not in (0, 1):
        return OnlineStep(False, "Outcome is tied or unresolved: nothing can be learned.")
    if model is None or not model.trained:
        return OnlineStep(False, "No trained model yet; this outcome is recorded but not learned from.")
    if not features:
        return OnlineStep(False, "No feature snapshot stored with this signal; cannot learn from it.")

    raw = model.predict_up(features)
    probability = model.calibrator.transform(raw) if model.calibrator.fitted else raw
    squared_error = (probability - float(label)) ** 2

    model.learn_one(features, int(label), learning_rate=learning_rate)
    return OnlineStep(True, "Online update applied after scoring.", probability, raw, int(label), squared_error)


def refit_calibration(model: OnlineLogisticModel, points: Sequence[tuple[float, int]], minimum_points: int) -> dict:
    """Refit isotonic calibration on real, confirmed outcomes only."""
    calibrator, report = fit_validated(points)
    if calibrator.fitted and report.get("accepted"):
        model.calibrator = calibrator
    model.calibration_report = report
    return report


# --------------------------------------------------------------- walk-forward
@dataclass
class WalkForwardReport:
    samples: int = 0
    folds: int = 0
    brier: float | None = None
    brier_skill: float | None = None
    log_loss: float | None = None
    accuracy: float | None = None
    ece: float | None = None
    net_pips: float | None = None
    trades: int = 0
    promoted: bool = False
    rolled_back: bool = False
    mode: str = MODE_VETO
    reason: str = ""
    champion: dict = field(default_factory=dict)
    candidate: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "samples": self.samples,
            "folds": self.folds,
            "brier": self.brier,
            "brier_skill": self.brier_skill,
            "log_loss": self.log_loss,
            "accuracy": self.accuracy,
            "ece": self.ece,
            "net_pips": self.net_pips,
            "trades": self.trades,
            "promoted": self.promoted,
            "rolled_back": self.rolled_back,
            "mode": self.mode,
            "reason": self.reason,
            "champion": self.champion,
            "candidate": self.candidate,
        }


def _folds(samples: Sequence[Sample], folds: int) -> list[tuple[list[Sample], list[Sample]]]:
    size = len(samples)
    folds = max(MIN_WALK_FORWARD_FOLDS, min(folds, 6))
    if size < folds * 40:
        return []
    boundaries = [int(round(size * (index + 1) / (folds + 1))) for index in range(folds)]
    result = []
    start = 0
    for end in boundaries:
        result.append((list(samples[:end]), list(samples[end : min(size, end + (size // (folds + 1)))])))
        start = end
    return [(train, test) for train, test in result if train and test]


def walk_forward(
    samples: Sequence[Sample],
    *,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    folds: int = 5,
    champion: OnlineLogisticModel | None = None,
    min_samples: int = 400,
) -> WalkForwardReport:
    """Expanding-window walk-forward. Every evaluation point is out-of-sample."""
    report = WalkForwardReport(samples=len(samples))
    if len(samples) < min_samples:
        report.reason = (
            f"Only {len(samples)} labelled bars; {min_samples} are required before "
            "a refit can be considered."
        )
        return report

    splits = _folds(samples, folds)
    if not splits:
        report.reason = "Not enough labelled bars to build walk-forward folds."
        return report

    points: list[tuple[float, int]] = []
    gross_total = 0.0
    trades = 0
    champion_points: list[tuple[float, int]] = []

    for train, test in splits:
        try:
            model = OnlineLogisticModel().fit([item.to_pair() for item in train], epochs=90, validation_fraction=0.0)
        except ValueError:
            continue
        for sample in test:
            probability = model.predict_up(sample.features)
            points.append((probability, sample.label))
            # Only bars where the model actually took a side are traded.
            if abs(probability - 0.5) >= 0.02:
                direction = 1.0 if probability > 0.5 else -1.0
                gross_total += direction * sample.gross_pips - spread_pips
                trades += 1
            if champion is not None and champion.trained:
                champion_points.append((champion.predict_up(sample.features), sample.label))

    if not points:
        report.reason = "No fold produced a usable model."
        return report

    brier = brier_score(points)
    report.folds = len(splits)
    report.brier = brier
    report.brier_skill = (1.0 - brier / 0.25) if brier is not None else None
    report.log_loss = sum(
        -(
            math.log(max(1e-9, min(1 - 1e-9, p))) if y == 1 else math.log(max(1e-9, 1 - min(1 - 1e-9, p)))
        )
        for p, y in points
    ) / len(points)
    report.accuracy = sum(1 for p, y in points if (p >= 0.5) == bool(y)) / len(points)
    report.ece = expected_calibration_error(points)
    report.net_pips = gross_total
    report.trades = trades
    report.candidate = {
        "brier": report.brier,
        "brier_skill": report.brier_skill,
        "accuracy": report.accuracy,
        "ece": report.ece,
        "net_pips": report.net_pips,
        "trades": trades,
    }
    if champion_points:
        champion_brier = brier_score(champion_points)
        report.champion = {
            "brier": champion_brier,
            "brier_skill": (1.0 - champion_brier / 0.25) if champion_brier is not None else None,
            "accuracy": sum(1 for p, y in champion_points if (p >= 0.5) == bool(y)) / len(champion_points),
            "ece": expected_calibration_error(champion_points),
        }
    return report


def promotion_gate(report: WalkForwardReport, champion: OnlineLogisticModel | None) -> tuple[bool, str]:
    """All gates must pass. One failure and the champion stays."""
    if report.folds < MIN_WALK_FORWARD_FOLDS:
        return False, "Walk-forward produced too few folds to judge."
    if report.accuracy is None or report.accuracy < PROMOTION_MIN_ACCURACY:
        return False, (
            f"Directional accuracy {report.accuracy:.3f} is below the {PROMOTION_MIN_ACCURACY:.3f} gate."
            if report.accuracy is not None
            else "No accuracy was produced."
        )
    if report.brier_skill is None or report.brier_skill <= PROMOTION_MIN_SKILL:
        return False, "Brier skill is not better than a 0.50 base rate."
    if report.net_pips is None or report.net_pips <= 0:
        return False, (
            f"Walk-forward expectancy {report.net_pips:.1f} pips is not positive after "
            "charging the spread on every trade."
        )
    if champion is not None and champion.trained and report.champion:
        champion_brier = report.champion.get("brier")
        if champion_brier is not None and report.brier is not None and report.brier > champion_brier:
            return False, (
                f"Candidate Brier {report.brier:.4f} is worse than the champion's {champion_brier:.4f}."
            )
    return True, "All promotion gates passed."


def rollback_check(settlements: Sequence[dict], window: int = 120, min_trades: int = 60) -> tuple[bool, str]:
    """Has the champion's live record decayed enough to warrant a rollback?"""
    real = [row for row in settlements if not charter.is_synthetic(row.get("source"))]
    real = [row for row in real if row.get("outcome") in (charter.WIN, charter.LOSS)]
    window_rows = real[-window:]
    if len(window_rows) < min_trades:
        return False, f"Only {len(window_rows)} settled real trades in the window; {min_trades} needed to judge."
    net = sum(float(row.get("net_pips") or 0.0) for row in window_rows)
    wins = sum(1 for row in window_rows if row.get("outcome") == charter.WIN)
    win_rate = wins / len(window_rows)
    if net < 0 and win_rate < 0.5:
        return True, (
            f"Last {len(window_rows)} real trades: {wins} wins, {net:.1f} net pips. "
            "The champion has decayed; rolling back to the previous promoted version."
        )
    return False, f"Live record holds: {wins}/{len(window_rows)} wins, {net:.1f} net pips."


def run_nightly(
    candles: Sequence[Candle],
    settlements: Sequence[dict],
    *,
    config,
    champion: OnlineLogisticModel | None,
) -> tuple[OnlineLogisticModel | None, BootstrapEnsemble, WalkForwardReport]:
    """Full nightly job: walk-forward, promote or keep, refit calibration."""
    samples = build_training_samples(candles, horizon=config.horizon_minutes)
    report = walk_forward(
        samples,
        spread_pips=config.spread_pips,
        champion=champion,
        min_samples=config.nightly_min_samples,
    )

    should_rollback, rollback_reason = rollback_check(settlements)
    if should_rollback:
        report.rolled_back = True
        report.reason = rollback_reason
        if champion is not None:
            champion.mode = MODE_VETO
        return champion, BootstrapEnsemble([]), report

    promoted, gate_reason = promotion_gate(report, champion)
    if not promoted:
        report.promoted = False
        report.reason = gate_reason
        # The champion keeps its authority level; nothing is promoted.
        return champion, BootstrapEnsemble([]), report

    try:
        model = OnlineLogisticModel().fit([item.to_pair() for item in samples], epochs=120)
    except ValueError as error:
        report.reason = str(error)
        return champion, BootstrapEnsemble([]), report

    model.trained_at = format_datetime(now())
    # Calibration is fitted on prequential-style points from the walk-forward
    # path: real, confirmed, and produced before any promotion decision.
    calibration_points = _prequential_points(model, samples)
    model.calibrator = IsotonicCalibrator.fit(calibration_points)
    if model.calibrator.fitted and not model.calibrator.flat:
        model.mode = MODE_ADVISORY
    report.promoted = True
    report.mode = model.mode
    report.reason = (
        f"Promoted: accuracy {report.accuracy:.3f}, Brier skill {report.brier_skill:.3f}, "
        f"{report.trades} cost-charged trades for {report.net_pips:.1f} pips. "
        f"Authority is {model.mode}."
    )
    ensemble = BootstrapEnsemble.fit([item.to_pair() for item in samples], member_count=4, epochs=50)
    return model, ensemble, report


def _prequential_points(model: OnlineLogisticModel, samples: Sequence[Sample], cap: int = 1200) -> list[tuple[float, int]]:
    points: list[tuple[float, int]] = []
    for sample in samples[-cap:]:
        try:
            points.append((model.predict_up(sample.features), sample.label))
        except ValueError:
            continue
    return points


def outcome_of(gross_pips: float) -> str:
    if math.isclose(gross_pips, 0.0, abs_tol=1e-12):
        return charter.TIED
    return charter.WIN if gross_pips > 0 else charter.LOSS


def expiry_for(signal_stamp, horizon_minutes: int):
    return add_minutes(signal_stamp, horizon_minutes)
