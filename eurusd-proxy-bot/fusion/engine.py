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

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

from . import charter, costs, datagate, instruments, news, protections
from .calibration import IsotonicCalibrator
from .council import CouncilResult, CouncilWeights, run_council
from .domain import Candle, ModelRead, Signal, closed_before
from .features import compute_features
from . import mtf
from .model import MODE_ADVISORY, MODE_VETO, BootstrapEnsemble, OnlineLogisticModel
from .mistakes import MistakeMemory
from .regime import RegimeState, detect_regime
from .session import session_of
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


def _mtf_tilt(features: dict | None, minutes: int) -> float | None:
    """The higher-timeframe tilt as -1, 0 or +1, or None when it is unknown.

    Read from the numeric row (``mtf_bias_5`` / ``mtf_bias_15``) rather than
    from a rich object: the row is persisted and re-read by Signal.from_dict,
    which floats every value, so anything non-numeric poisons the round-trip.

    Returns None when the tilt was never computed -- an absent filter must
    never be mistaken for a filter that agrees.
    """
    if not isinstance(features, dict):
        return None
    primary = "mtf_bias_15" if minutes >= 15 else "mtf_bias_5"
    fallback = "mtf_bias_5" if minutes >= 15 else "mtf_bias_15"
    for key in (primary, fallback):
        if key in features:
            try:
                value = float(features[key])
            except (TypeError, ValueError):
                return None
            return value if math.isfinite(value) else None
    return None


def _mtf_gap_bps(features: dict | None, minutes: int) -> tuple[float | None, bool]:
    """Return (gap_bps, ready) for the selected higher-timeframe EMA."""
    if not isinstance(features, dict):
        return None, False
    suffix = "60" if minutes >= 60 else ("15" if minutes >= 15 else "5")
    gap_key = f"mtf_gap_{suffix}"
    ready_key = f"mtf_ready_{suffix}"
    bias_key = f"mtf_bias_{suffix}"
    if gap_key not in features:
        return None, False
    try:
        gap = float(features[gap_key])
    except (TypeError, ValueError):
        return None, False
    if not math.isfinite(gap):
        return None, False
    if ready_key in features:
        ready = float(features.get(ready_key) or 0.0) > 0.5
    else:
        ready = abs(float(features.get(bias_key) or 0.0)) > 0.5
    return gap, ready


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
class Verdict:
    """The decision itself: council -> model -> allow or refuse.

    Split out from the seal step so the backtest can drive exactly this code.
    A backtest that re-implemented the rules would be measuring something the
    bot does not actually do.
    """

    side: str
    reason: str
    regime: str = "UNKNOWN"
    read: ModelRead | None = None
    council: CouncilResult | None = None
    model_reason: str = ""

    @property
    def actionable(self) -> bool:
        return self.side in ("BUY", "SELL")


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

    # --------------------------------------------------------------- decision
    def decide(
        self,
        candle: Candle,
        previous_close: float,
        features: dict[str, float] | None,
        *,
        model: OnlineLogisticModel | None = None,
        ensemble: BootstrapEnsemble | None = None,
        weights: CouncilWeights | None = None,
        regime_state: RegimeState | None = None,
        spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
        mistakes: MistakeMemory | None = None,
    ) -> Verdict:
        """Run regime, council and meta-model. No data gate, no sealing."""
        config = self.config
        if features is None:
            return Verdict("NO SIGNAL", "Not enough completed candles to calculate indicators.")

        regime = regime_state or RegimeState("UNKNOWN", 0.0, 0.0, 0.0, 0.0, "Regime not supplied.")

        if getattr(config, "news_filter", False):
            blackout = news.check_blackout(
                add_minutes(candle.timestamp, 1),
                pair=getattr(config, "pair", "EUR/USD"),
                extra_windows=getattr(config, "news_blackout_windows", ""),
            )
            if blackout.blocked:
                return Verdict("NO SIGNAL", blackout.reason, regime.name)

        read = build_read(model, ensemble, features, max_uncertainty=config.max_uncertainty)
        if read is not None and read.abstain_reason:
            return Verdict("NO SIGNAL", read.abstain_reason, regime.name, read=read)

        # The session is read from the bar being decided, not from the wall
        # clock, so a replay of history classifies each bar by the hour it
        # actually happened in.
        session = (
            session_of(candle.timestamp, instruments.for_symbol(self.config.pair))
            if config.session_filter
            else None
        )
        council = run_council(
            candle,
            previous_close,
            features,
            read.probability if read else None,
            weights=weights,
            regime=regime.name,
            min_checks=config.min_confluence_checks,
            calibrated=bool(read and read.calibrated),
            spread_pips=spread_pips,
            session=session,
            mistakes=mistakes,
        )
        if not council.actionable:
            return Verdict("NO SIGNAL", council.reason, regime.name, read=read, council=council)

        # Higher-timeframe agreement. A one-minute chart is mostly noise, and
        # the commonest way a scalp fails is by being a pullback inside a larger
        # move going the other way. The tilt was computed alongside the
        # features, so this costs nothing extra.
        if config.mtf_filter:
            tilt = _mtf_tilt(features, config.mtf_minutes)
            wanted = mtf.UP if council.side == "BUY" else mtf.DOWN
            if tilt is not None and tilt != mtf.FLAT and tilt != wanted:
                leaning = "up" if tilt == mtf.UP else "down"
                return Verdict(
                    "NO SIGNAL",
                    f"Higher-timeframe filter: {config.mtf_minutes}-minute bias is {leaning}"
                    f" and the trade is {council.side}.",
                    regime.name,
                    read=read,
                    council=council,
                )
            min_gap_bps = float(getattr(config, "mtf_min_gap_bps", 0.0) or 0.0)
            if min_gap_bps > 0.0:
                gap_bps, mtf_ready = _mtf_gap_bps(features, config.mtf_minutes)
                if mtf_ready and gap_bps is not None:
                    signed_gap = gap_bps if council.side == "BUY" else -gap_bps
                    if signed_gap < min_gap_bps:
                        return Verdict(
                            "NO SIGNAL",
                            f"Higher-timeframe dead-zone: {config.mtf_minutes}-minute EMA gap "
                            f"({gap_bps:+.2f} bps) is below the {min_gap_bps:.2f} bps clear-trend threshold for {council.side}.",
                            regime.name,
                            read=read,
                            council=council,
                        )

        if getattr(config, "mtf_macro_filter", False):
            gap_60, ready_60 = _mtf_gap_bps(features, 60)
            if ready_60 and gap_60 is not None:
                macro_min = max(0.0, float(getattr(config, "mtf_macro_min_gap_bps", 1.5) or 0.0))
                slope_60 = float(features.get("mtf_slope_60", 0.0) or 0.0) if isinstance(features, dict) else 0.0
                signed_gap_60 = gap_60 if council.side == "BUY" else -gap_60
                signed_slope_60 = slope_60 if council.side == "BUY" else -slope_60
                if signed_gap_60 < macro_min or signed_slope_60 <= 0.0:
                    return Verdict(
                        "NO SIGNAL",
                        f"Macro 60m HTF filter: gap {gap_60:+.2f} bps / slope {slope_60:+.2f} bps "
                        f"does not confirm {council.side} (min gap {macro_min:.2f} bps).",
                        regime.name,
                        read=read,
                        council=council,
                    )

        mode = model.mode if model is not None else MODE_VETO
        allowed, model_reason = model_verdict(council.side, read, mode=mode)
        if not allowed:
            return Verdict(
                "NO SIGNAL", model_reason, regime.name, read=read, council=council, model_reason=model_reason
            )
        return Verdict(
            council.side, council.reason, regime.name, read=read, council=council, model_reason=model_reason
        )

    # ----------------------------------------------------------------- minute
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
        mistakes: MistakeMemory | None = None,
    ) -> EngineResult:
        """Data gate -> features -> decision -> seal, for one live minute."""
        config = self.config
        stamp = at or utcnow()
        # Decide only on bars that had already closed when the entry minute
        # opened. The entry bar is still forming at this moment; letting its
        # high, low and close into the features is look-ahead, and the
        # walk-forward backtest never had it. See domain.closed_before.
        candles = closed_before(list(candles), stamp)
        instrument = instruments.for_symbol(getattr(config, "pair", None))

        gate = datagate.evaluate(
            candles,
            now=stamp,
            max_age_seconds=config.max_candle_age_seconds,
            max_spread_pips=config.max_spread_pips,
            quote=quote,
            base_spread_pips=config.spread_pips,
            crypto=instrument.crypto,
        )
        if not gate.usable:
            return EngineResult(charter.SEAL_ABSTAIN, None, f"Data gate: {gate.reason}", gate=gate)

        feature_rows = compute_features(candles, instrument=instrument)
        features = feature_rows[-1] if feature_rows else None

        latest = candles[-1]
        previous_close = candles[-2].close if len(candles) > 1 else latest.close
        regime_state = detect_regime(candles, features)
        verdict = self.decide(
            latest,
            previous_close,
            features,
            model=model,
            ensemble=ensemble,
            weights=weights,
            regime_state=regime_state,
            spread_pips=gate.spread_pips,
            mistakes=mistakes,
        )
        return self._seal(
            verdict, gate=gate, source=source, at=stamp, spread=gate.spread_pips, features=features
        )

    def evaluate_prepared(
        self,
        candles: Sequence[Candle],
        *,
        features: dict[str, float] | None,
        gate: datagate.GateResult,
        regime_state: RegimeState,
        model: OnlineLogisticModel | None = None,
        ensemble: BootstrapEnsemble | None = None,
        weights: CouncilWeights | None = None,
        source: str = "BACKTEST",
        at: datetime | None = None,
        mistakes: MistakeMemory | None = None,
    ) -> EngineResult:
        """Decision + seal with the gate, features and regime supplied.

        Used by the backtest, which precomputes those once for the whole series
        instead of per bar. The decision and seal code is identical to live.
        """
        if not isinstance(candles, (list, tuple)):
            candles = list(candles)
        latest = candles[-1]
        previous_close = candles[-2].close if len(candles) > 1 else latest.close
        verdict = self.decide(
            latest,
            previous_close,
            features,
            model=model,
            ensemble=ensemble,
            weights=weights,
            regime_state=regime_state,
            spread_pips=gate.spread_pips,
            mistakes=mistakes,
        )
        return self._seal(
            verdict,
            gate=gate,
            source=source,
            at=at or latest.timestamp,
            spread=gate.spread_pips,
            features=features,
        )

    def _seal(
        self,
        verdict: Verdict,
        *,
        gate: datagate.GateResult,
        source: str,
        at: datetime,
        spread: float,
        features: dict[str, float] | None = None,
    ) -> EngineResult:
        if not verdict.actionable:
            return EngineResult(
                charter.SEAL_ABSTAIN,
                None,
                verdict.reason,
                gate=gate,
                regime=verdict.regime,
                read=verdict.read,
                council=verdict.council,
                model_reason=verdict.model_reason,
            )

        entry_time = floor_minute(at)
        latency = (at - entry_time).total_seconds()
        if latency > self.config.max_seal_latency_seconds:
            # The entry is already in the past. A backdated signal would be a lie.
            return EngineResult(
                charter.SEAL_DISCARDED_LATE,
                None,
                f"Sealed {latency:.1f}s after the minute opened, past the "
                f"{self.config.max_seal_latency_seconds}s limit; discarded, not backdated.",
                gate=gate,
                regime=verdict.regime,
                read=verdict.read,
                council=verdict.council,
                model_reason=verdict.model_reason,
            )

        # Pick the hold length before the signal exists. When adaptive horizon
        # is on, a market too quiet to pay for the round turn produces None, and
        # the right response is to abstain rather than to take the trade anyway.
        atr_pips = 0.0
        if features:
            # The feature row already carries atr_pips, computed with this
            # instrument's own pip size. Dividing by charter.PIP here would
            # silently reinterpret a bitcoin ATR as an FX one.
            atr_pips = abs(float(features.get("atr", 0.0))) / instruments.pip_for(
                getattr(self.config, "pair", None)
            )
        horizon = costs.resolve_horizon(
            atr_pips,
            spread,
            self.config.horizon_minutes,
            adaptive=self.config.adaptive_horizon,
            reward_to_cost=self.config.reward_to_cost,
            max_horizon=self.config.max_horizon_minutes,
        )
        if horizon is None:
            verdict_obj = costs.evaluate(atr_pips, self.config.horizon_minutes, spread,
                                         self.config.reward_to_cost)
            return EngineResult(
                charter.SEAL_ABSTAIN,
                None,
                "No hold length can pay for this trade: " + verdict_obj.describe(),
                gate=gate,
                regime=verdict.regime,
                read=verdict.read,
                council=verdict.council,
                model_reason=verdict.model_reason,
            )

        if getattr(self.config, "fx_rollover_guard", False):
            inst = instruments.for_symbol(getattr(self.config, "pair", None))
            rollover = protections.check_fx_rollover(entry_time, horizon, crypto=inst.crypto)
            if rollover.blocked:
                return EngineResult(
                    charter.SEAL_ABSTAIN,
                    None,
                    rollover.reason,
                    gate=gate,
                    regime=verdict.regime,
                    read=verdict.read,
                    council=verdict.council,
                    model_reason=verdict.model_reason,
                )

        read = verdict.read
        signal = Signal.create(
            pair=self.config.pair,
            side=verdict.side,
            entry_time=entry_time,
            strategy="fusion_council",
            reason=verdict.reason,
            source=source,
            generated_at=at,
            horizon_minutes=horizon,
            probability_up=read.probability if read else None,
            probability_raw=read.raw if read else None,
            calibration_state=read.calibration_state if read else RAW,
            uncertainty=read.uncertainty if read else None,
            regime=verdict.regime,
            council=tuple(vote.to_dict() for vote in verdict.council.votes) if verdict.council else (),
            # Kept so the outcome can be learned from later; without it the
            # settlement cannot update the model and the bar is wasted.
            features=dict(features or {}),
            spread_pips=spread,
        )
        return EngineResult(
            charter.SEAL_SIGNAL,
            signal,
            verdict.reason,
            gate=gate,
            regime=verdict.regime,
            read=read,
            council=verdict.council,
            model_reason=verdict.model_reason,
        )


def expiry_of(signal: Signal) -> datetime:
    return add_minutes(signal.entry_time, signal.horizon_minutes)
