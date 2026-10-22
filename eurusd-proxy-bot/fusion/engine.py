"""The per-minute pipeline: gate -> features -> regime -> council -> model -> seal.

Order matters and is fixed:

1. data gate  — refuse to reason about data that is stale, gappy or too wide;
2. features   — indicators from completed candles only;
3. regime     — context, never a signal on its own;
4. council    — the core 6-check gate decides; learned prototypes may veto;
5. meta-model — VETO by default, ADVISORY only after nightly gates promote it;
6. calibration — RAW below 60 real outcomes, CALIBRATED at or above, FLAT abstains;
7. seal       — entry is the open of the forming candle, or the signal is dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Sequence

from . import charter, datagate
from .calibration import IsotonicCalibrator
from .council import CouncilResult, CouncilWeights, run_council
from .domain import Candle, ModelRead, Signal
from .features import compute_features
from .model import MODE_ADVISORY, MODE_VETO, BootstrapEnsemble, OnlineLogisticModel
from .regime import detect_regime
from .timeutil import add_minutes, floor_minute, now as utcnow

# Meta-model margins. VETO only blocks a read it confidently disagrees with;
# ADVISORY (promoted) additionally requires agreement, with a tighter bar on a
# calibrated score than on a raw one.
VETO_MARGIN = 0.03
AGREE_MARGIN_CALIBRATED = 0.01
AGREE_MARGIN_RAW = 0.02

RAW = "RAW"
CALIBRATED = "CALIBRATED"
FLAT = "FLAT"


def build_read(
    model: OnlineLogisticModel | None,
    ensemble: BootstrapEnsemble | None,
    features: dict[str, float],
    *,
    max_uncertainty: float = 0.10,
) -> ModelRead | None:
    """Turn the model into an honest probability read, or None when untrained."""
    if model is None or not model.trained or features is None:
        return None
    raw = model.predict_up(features)
    uncertainty: float | None = None
    members = 0
    if ensemble is not None and ensemble.ready:
        _, spread = ensemble.predict(features)
        uncertainty = spread
        members = len(ensemble.members)

    calibrator: IsotonicCalibrator = model.calibrator
    if calibrator.fitted:
        probability = calibrator.transform(raw)
        state = FLAT if calibrator.flat else CALIBRATED
    else:
        probability = raw
        state = RAW

    reason = None
    if state == FLAT:
        reason = (
            "Calibration is flat: the score has no usable rank signal on recorded "
            "outcomes, so the gate abstains instead of trading a constant probability."
        )
    elif uncertainty is not None and uncertainty > max_uncertainty:
        reason = (
            f"Ensemble disagreement is {uncertainty:.3f}, wider than the {max_uncertainty:.2f} "
            "stability limit; abstaining instead of guessing."
        )
    return ModelRead(
        raw=raw,
        probability=probability,
        calibrated=state == CALIBRATED,
        calibration_state=state,
        uncertainty=uncertainty,
        members=members,
        contributions=tuple(model.contributions(features, top=6)),
        abstain_reason=reason,
    )


def model_verdict(side: str, read: ModelRead | None, *, mode: str = MODE_VETO) -> tuple[bool, str]:
    """Does the meta-model allow this side? Returns (allowed, reason)."""
    if read is None:
        return True, "No trained model yet; the rule gate decided alone."
    if not read.usable:
        return False, read.abstain_reason or "The model read is not usable on this bar."
    if read.uncertainty is not None and read.abstain_reason:
        return False, read.abstain_reason

    edge = read.probability - 0.5
    agrees = edge > 0 if side == "BUY" else edge < 0
    label = "calibrated" if read.calibrated else "raw"

    if mode == MODE_ADVISORY:
        margin = AGREE_MARGIN_CALIBRATED if read.calibrated else AGREE_MARGIN_RAW
        if not agrees:
            return False, (
                f"Promoted model veto: {label} read {read.probability:.3f} is on the other side of {side}."
            )
        if abs(edge) < margin:
            return False, (
                f"Promoted model abstains: {label} edge {edge:+.3f} is inside the {margin:.3f} agreement margin."
            )
        return True, f"Promoted model agrees ({label} {read.probability:.3f})."

    if not agrees and abs(edge) >= VETO_MARGIN:
        return False, (
            f"Model veto: {label} read {read.probability:.3f} disagrees with {side} "
            f"by more than {VETO_MARGIN:.2f}."
        )
    return True, f"Model does not object ({label} {read.probability:.3f})."


@dataclass(frozen=True, slots=True)
class EngineResult:
    seal_state: str
    signal: Signal | None
    reason: str
    gate: datagate.GateResult | None = None
    regime: str = "UNKNOWN"
    read: ModelRead | None = None
    council: CouncilResult | None = None
    model_reason: str = ""

    @property
    def actionable(self) -> bool:
        return self.seal_state == charter.SEAL_SIGNAL and self.signal is not None

    def to_dict(self) -> dict:
        return {
            "seal_state": self.seal_state,
            "reason": self.reason,
            "regime": self.regime,
            "gate": self.gate.to_dict() if self.gate else None,
            "read": self.read.to_dict() if self.read else None,
            "council": self.council.to_dict() if self.council else None,
            "model_reason": self.model_reason,
            "signal": self.signal.to_dict() if self.signal else None,
        }


class FusionEngine:
    def __init__(self, config):
        self.config = config

    def evaluate(
        self,
        candles: Sequence[Candle],
        *,
        model: OnlineLogisticModel | None = None,
        ensemble: BootstrapEnsemble | None = None,
        weights: CouncilWeights | None = None,
        source: str = "LIVE",
        quote: dict | None = None,
        at: datetime | None = None,
    ) -> EngineResult:
        config = self.config
        stamp = at or utcnow()
        candles = list(candles)

        gate = datagate.evaluate(
            candles,
            now=stamp,
            max_age_seconds=config.max_candle_age_seconds,
            max_spread_pips=config.max_spread_pips,
            quote=quote,
        )
        if not gate.usable:
            return EngineResult(
                charter.SEAL_ABSTAIN,
                None,
                f"Data gate: {gate.reason}",
                gate=gate,
            )

        feature_rows = compute_features(candles)
        features = feature_rows[-1] if feature_rows else None
        if features is None:
            return EngineResult(
                charter.SEAL_ABSTAIN,
                None,
                "Indicators are not complete for the latest bar.",
                gate=gate,
            )

        latest = candles[-1]
        previous_close = candles[-2].close if len(candles) > 1 else latest.close
        regime = detect_regime(candles, features)

        read = build_read(model, ensemble, features, max_uncertainty=config.max_uncertainty)
        if read is not None and read.abstain_reason:
            return EngineResult(
                charter.SEAL_ABSTAIN,
                None,
                read.abstain_reason,
                gate=gate,
                regime=regime.name,
                read=read,
            )

        council = run_council(
            latest,
            previous_close,
            features,
            read.probability if read else None,
            weights=weights,
            regime=regime.name,
            min_checks=config.min_confluence_checks,
            calibrated=bool(read and read.calibrated),
        )
        if not council.actionable:
            return EngineResult(
                charter.SEAL_ABSTAIN,
                None,
                council.reason,
                gate=gate,
                regime=regime.name,
                read=read,
                council=council,
            )

        mode = model.mode if model is not None else MODE_VETO
        allowed, model_reason = model_verdict(council.side, read, mode=mode)
        if not allowed:
            return EngineResult(
                charter.SEAL_ABSTAIN,
                None,
                model_reason,
                gate=gate,
                regime=regime.name,
                read=read,
                council=council,
                model_reason=model_reason,
            )

        entry_time = floor_minute(stamp)
        latency = (stamp - entry_time).total_seconds()
        if latency > config.max_seal_latency_seconds:
            # The entry is already in the past. A backdated signal would be a lie.
            return EngineResult(
                charter.SEAL_DISCARDED_LATE,
                None,
                f"Sealed {latency:.1f}s after the minute opened, past the "
                f"{config.max_seal_latency_seconds}s limit; discarded, not backdated.",
                gate=gate,
                regime=regime.name,
                read=read,
                council=council,
                model_reason=model_reason,
            )

        signal = Signal.create(
            pair=config.pair,
            side=council.side,
            entry_time=entry_time,
            strategy="fusion_council",
            reason=council.reason,
            source=source,
            generated_at=stamp,
            horizon_minutes=config.horizon_minutes,
            probability_up=read.probability if read else None,
            probability_raw=read.raw if read else None,
            calibration_state=read.calibration_state if read else RAW,
            uncertainty=read.uncertainty if read else None,
            regime=regime.name,
            council=tuple(vote.to_dict() for vote in council.votes),
            features=features,
            spread_pips=gate.spread_pips,
        )
        return EngineResult(
            charter.SEAL_SIGNAL,
            signal,
            council.reason,
            gate=gate,
            regime=regime.name,
            read=read,
            council=council,
            model_reason=model_reason,
        )


def expiry_of(signal: Signal) -> datetime:
    return add_minutes(signal.entry_time, signal.horizon_minutes)
