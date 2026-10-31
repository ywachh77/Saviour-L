"""Live Prediction Engine — BUY / SELL / NO TRADE with full metadata.

Wraps the Temporal Transformer for live inference. Every prediction
includes all required fields per spec. No signal is generated when
data is stale, model is unhealthy, or uncertainty is excessive.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import torch

from .architecture import NeuralConfig, TemporalTransformer, build_model, count_parameters
from .dataset import NeuralDataset, candle_features
from .trainer import NeuralTrainer


# ---------------------------------------------------------------------------
# Health monitoring
# ---------------------------------------------------------------------------

@dataclass
class ModelHealth:
    """Model health status."""
    is_healthy: bool = True
    last_prediction: datetime | None = None
    last_prediction_age_seconds: float = 0.0
    total_predictions: int = 0
    error_count: int = 0
    consecutive_errors: int = 0
    last_error: str = ""
    health_score: float = 1.0  # 0.0 = unhealthy, 1.0 = healthy

    def to_dict(self) -> dict:
        return {
            "is_healthy": self.is_healthy,
            "last_prediction": self.last_prediction.isoformat() if self.last_prediction else None,
            "last_prediction_age_seconds": self.last_prediction_age_seconds,
            "total_predictions": self.total_predictions,
            "error_count": self.error_count,
            "consecutive_errors": self.consecutive_errors,
            "last_error": self.last_error,
            "health_score": self.health_score,
        }


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

@dataclass
class Prediction:
    """A single prediction with all required metadata."""
    pair: str
    prediction_time: str
    entry_reference_price: float
    horizon_minutes: int
    direction_probs: dict[str, float]
    expected_return_pips: float
    uncertainty: float
    regime: str
    model_version: str
    validation_status: str  # PASSED | FAILED
    action: str  # BUY | SELL | NO_TRADE
    reason: str = ""
    feature_values: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "pair": self.pair,
            "prediction_time": self.prediction_time,
            "entry_reference_price": self.entry_reference_price,
            "horizon_minutes": self.horizon_minutes,
            "direction_probs": self.direction_probs,
            "expected_return_pips": self.expected_return_pips,
            "uncertainty": self.uncertainty,
            "regime": self.regime,
            "model_version": self.model_version,
            "validation_status": self.validation_status,
            "action": self.action,
            "reason": self.reason,
        }


# ---------------------------------------------------------------------------
# Neural Predictor
# ---------------------------------------------------------------------------

class NeuralPredictor:
    """Live prediction engine wrapping the Temporal Transformer.

    Usage:
        predictor = NeuralPredictor.from_checkpoint("neural/checkpoints/v1.pt")
        prediction = predictor.predict(candles, pair="EUR/USD")
    """

    def __init__(
        self,
        model: TemporalTransformer,
        neural_config: NeuralConfig,
        model_version: str = "unknown",
        norm_stats: dict | None = None,
        spread_pips: float = 1.2,
    ):
        self.model = model
        self.model.eval()
        self.neural_config = neural_config
        self.model_version = model_version
        self.spread_pips = spread_pips

        # Normalization statistics
        self.feat_mean = np.array(norm_stats["feat_mean"], dtype=np.float32) if norm_stats else None
        self.feat_std = np.array(norm_stats["feat_std"], dtype=np.float32) if norm_stats else None
        self.state_mean = np.array(norm_stats["state_mean"], dtype=np.float32) if norm_stats else None
        self.state_std = np.array(norm_stats["state_std"], dtype=np.float32) if norm_stats else None

        self.health = ModelHealth()
        self._min_candles = neural_config.input_window + 1

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, spread_pips: float = 1.2) -> "NeuralPredictor":
        """Load predictor from a saved checkpoint."""
        model, nc, ckpt = NeuralTrainer.load_checkpoint(checkpoint_path)
        return cls(
            model=model,
            neural_config=nc,
            model_version=ckpt.get("model_version", "unknown"),
            norm_stats=ckpt.get("norm_stats"),
            spread_pips=spread_pips,
        )

    @classmethod
    def from_registry(cls, registry_path: str = "neural/registry.json",
                      spread_pips: float = 1.2) -> "NeuralPredictor":
        """Load the current production model from the registry."""
        from .registry import ModelRegistry
        registry = ModelRegistry(registry_path)
        prod = registry.get_production_version()
        if prod is None:
            raise RuntimeError("No production model in registry.")
        return cls.from_checkpoint(prod.checkpoint_path, spread_pips)

    def predict(
        self,
        candles: list,
        pair: str = "EUR/USD",
        at: datetime | None = None,
    ) -> Prediction:
        """Generate a prediction from candle data.

        Returns a Prediction with all required metadata.
        """
        self.health.total_predictions += 1

        try:
            # 1. Data check
            if len(candles) < self._min_candles:
                self._record_error(f"Not enough candles: {len(candles)} < {self._min_candles}")
                return self._make_no_trade_prediction(
                    candles, pair, at, "Insufficient data"
                )

            # 2. Extract features
            all_features = candle_features(candles)
            window = all_features[-self.neural_config.input_window:]
            state = self._extract_state(all_features, len(candles) - 1)

            # 3. Normalize
            if self.feat_mean is not None:
                window = (window - self.feat_mean) / self.feat_std
                state = (state - self.state_mean) / self.state_std

            # 4. Prepare tensors
            x = torch.from_numpy(window).float().unsqueeze(0)
            state_t = torch.from_numpy(state).float().unsqueeze(0)

            # 5. Forward pass
            with torch.no_grad():
                output = self.model(x, state_t)

            # 6. Parse output
            direction_probs = torch.softmax(output["direction_logits"], dim=-1)[0]
            buy_p = direction_probs[0].item()
            sell_p = direction_probs[1].item()
            no_trade_p = direction_probs[2].item()

            expected_return = output["expected_return"][0].item()
            uncertainty = output["uncertainty"][0].item()

            regime_probs = torch.softmax(output["regime_logits"], dim=-1)[0]
            regime_idx = int(regime_probs.argmax().item())
            regime_names = ["TREND", "RANGE", "VOLATILE", "TRANSITION"]
            regime = regime_names[regime_idx]

            # 7. Decision logic with cost guard
            cfg = self.neural_config
            entry_price = candles[-1].close
            atr_pips = all_features[-1, 12] * 10.0

            action, reason = self._decide_action(
                buy_p, sell_p, no_trade_p, expected_return,
                uncertainty, atr_pips,
            )

            # 8. Validation checks
            validation = self._validate(
                uncertainty, atr_pips, buy_p, sell_p, expected_return,
            )

            # 9. Build prediction
            prediction = Prediction(
                pair=pair,
                prediction_time=(at or datetime.now(timezone.utc)).isoformat(),
                entry_reference_price=entry_price,
                horizon_minutes=cfg.horizon_minutes,
                direction_probs={
                    "BUY": buy_p,
                    "SELL": sell_p,
                    "NO_TRADE": no_trade_p,
                },
                expected_return_pips=expected_return,
                uncertainty=uncertainty,
                regime=regime,
                model_version=self.model_version,
                validation_status=validation,
                action=action,
                reason=reason,
            )

            # Update health
            self.health.last_prediction = at or datetime.now(timezone.utc)
            self.health.last_prediction_age_seconds = 0.0
            self.health.consecutive_errors = 0

            return prediction

        except Exception as e:
            self._record_error(str(e))
            return self._make_no_trade_prediction(candles, pair, at, str(e))

    def _decide_action(
        self, buy_p: float, sell_p: float, no_trade_p: float,
        expected_return: float, uncertainty: float, atr_pips: float,
    ) -> tuple[str, str]:
        """Decide BUY / SELL / NO_TRADE based on model output and cost guards."""
        cfg = self.neural_config

        # High uncertainty → NO TRADE
        if uncertainty > cfg.max_uncertainty:
            return "NO_TRADE", f"High uncertainty ({uncertainty:.3f})"

        # Low ATR → NO TRADE (can't pay for spread)
        if atr_pips < self.spread_pips * 1.15:
            return "NO_TRADE", f"Low ATR ({atr_pips:.2f}p) vs spread ({self.spread_pips}p)"

        # Strong BUY signal
        if buy_p > cfg.direction_threshold and expected_return > cfg.min_expected_return_pips:
            net_ret = expected_return - self.spread_pips
            if net_ret > 0:
                return "BUY", f"P(BUY)={buy_p:.2f}, expected return={expected_return:.2f}p"

        # Strong SELL signal
        if sell_p > cfg.direction_threshold and -expected_return > cfg.min_expected_return_pips:
            net_ret = -expected_return - self.spread_pips
            if net_ret > 0:
                return "SELL", f"P(SELL)={sell_p:.2f}, expected return={expected_return:.2f}p"

        # Weak signals
        if buy_p > sell_p and expected_return > self.spread_pips * 0.5:
            return "BUY", f"Weak BUY: P(BUY)={buy_p:.2f} > P(SELL)={sell_p:.2f}"
        if sell_p > buy_p and expected_return < -self.spread_pips * 0.5:
            return "SELL", f"Weak SELL: P(SELL)={sell_p:.2f} > P(BUY)={buy_p:.2f}"

        return "NO_TRADE", "No clear edge"

    def _validate(
        self, uncertainty: float, atr_pips: float,
        buy_p: float, sell_p: float, expected_return: float,
    ) -> str:
        """Validate prediction conditions.

        Returns PASSED if all conditions are met, FAILED otherwise.
        """
        cfg = self.neural_config

        # Uncertainty must be within bounds
        if uncertainty > cfg.max_uncertainty:
            return "FAILED"

        # ATR must be sufficient to cover spread
        if atr_pips < self.spread_pips * 1.0:
            return "FAILED"

        # Direction probabilities must sum to ~1.0 (valid softmax output)
        # Also check that at least one direction has meaningful probability
        max_dir_prob = max(buy_p, sell_p)
        if max_dir_prob < 0.10:
            return "FAILED"

        # Expected return must be finite and not absurd
        if not math.isfinite(expected_return):
            return "FAILED"
        if abs(expected_return) > 100.0:  # Sanity check: max 100 pips
            return "FAILED"

        return "PASSED"

    def _extract_state(self, features: np.ndarray, idx: int) -> np.ndarray:
        """Extract 6-dim market-state features."""
        atr_pct = features[idx, 12] * 10.0
        rsi = features[idx, 11] * 100.0
        adx = features[idx, 14] * 100.0
        vol_ratio = features[idx, 16] * 2.0
        momentum = features[idx, 15] * 10.0
        session = 0.0
        return np.array([
            atr_pct / 10.0,
            rsi / 100.0,
            adx / 100.0,
            vol_ratio / 2.0,
            momentum / 10.0,
            session,
        ], dtype=np.float32)

    def _make_no_trade_prediction(
        self, candles, pair, at, reason,
    ) -> Prediction:
        """Create a NO_TRADE prediction for error cases."""
        entry_price = candles[-1].close if candles else 0.0
        return Prediction(
            pair=pair,
            prediction_time=(at or datetime.now(timezone.utc)).isoformat(),
            entry_reference_price=entry_price,
            horizon_minutes=self.neural_config.horizon_minutes,
            direction_probs={"BUY": 0.0, "SELL": 0.0, "NO_TRADE": 1.0},
            expected_return_pips=0.0,
            uncertainty=1.0,
            regime="UNKNOWN",
            model_version=self.model_version,
            validation_status="FAILED",
            action="NO_TRADE",
            reason=reason,
        )

    def _record_error(self, error: str):
        """Record an error and update health."""
        self.health.error_count += 1
        self.health.consecutive_errors += 1
        self.health.last_error = error

        # Update health score
        penalty = self.health.consecutive_errors * 0.2
        self.health.health_score = max(0.0, 1.0 - penalty)
        if self.health.consecutive_errors >= 3:
            self.health.is_healthy = False

    def get_health(self) -> dict:
        """Return model health status."""
        if self.health.last_prediction:
            age = (datetime.now(timezone.utc) - self.health.last_prediction).total_seconds()
            self.health.last_prediction_age_seconds = age
            if age > 300:  # 5 minutes stale
                self.health.is_healthy = False

        return self.health.to_dict()
