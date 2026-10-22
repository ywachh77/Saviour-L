"""The meta-model: a small, explainable, prequentially-updated logistic model.

Why logistic and not a network: with a few thousand one-minute bars a larger
model memorises noise, and its output could not be explained bar by bar. Every
score here is a sum of `feature x weight`, so each signal can show its own
top contributors.

Pure standard library — Adam on plain Python lists, no numpy.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .calibration import (
    IsotonicCalibrator,
    accuracy,
    brier_score,
    expected_calibration_error,
    log_loss,
    skill_score,
)
from .features import FEATURE_LABELS, FEATURE_NAMES, FEATURE_VERSION, model_vector

_PROBABILITY_FLOOR = 1e-3
_STANDARDISED_LIMIT = 8.0

# Meta-model authority. VETO is the default: the model may block a council
# signal it disagrees with, but may never create one. Nightly walk-forward
# gates can promote it to ADVISORY, where agreement is also required.
MODE_VETO = "VETO"
MODE_ADVISORY = "ADVISORY"


def _sigmoid(value: float) -> float:
    probability = 1.0 / (1.0 + math.exp(-max(-35.0, min(35.0, value))))
    return max(_PROBABILITY_FLOOR, min(1.0 - _PROBABILITY_FLOOR, float(probability)))


@dataclass
class OnlineLogisticModel:
    feature_names: tuple[str, ...] = FEATURE_NAMES
    feature_version: int = FEATURE_VERSION
    means: list[float] = field(default_factory=list)
    scales: list[float] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)
    bias: float = 0.0
    training_samples: int = 0
    calibrator: IsotonicCalibrator = field(default_factory=IsotonicCalibrator)
    calibration_report: dict | None = None
    validation_log_loss: float | None = None
    trained_at: str | None = None
    mode: str = MODE_VETO
    online_updates: int = 0

    # ------------------------------------------------------------------ state
    @property
    def trained(self) -> bool:
        return bool(self.weights) and self.training_samples > 0

    @property
    def compatible(self) -> bool:
        return self.feature_version == FEATURE_VERSION and tuple(self.feature_names) == FEATURE_NAMES

    # ------------------------------------------------------------------ fitting
    def fit(
        self,
        samples: Iterable[tuple[dict[str, float], int]],
        epochs: int = 120,
        learning_rate: float = 0.05,
        l2: float = 0.002,
        validation_fraction: float = 0.15,
        min_samples: int = 40,
        patience: int = 20,
    ) -> "OnlineLogisticModel":
        rows = [
            (features, int(label))
            for features, label in samples
            if all(name in features for name in self.feature_names) and label in (0, 1)
        ]
        if len(rows) < min_samples:
            raise ValueError(f"At least {min_samples} labelled bars are needed to train (got {len(rows)})")

        vectors = [model_vector(features) for features, _ in rows]
        labels = [float(label) for _, label in rows]

        # Scalers come from the training portion only, so the validation tail
        # stays an honest check rather than being standardised with its own data.
        validation_count = int(len(rows) * validation_fraction) if len(rows) >= 200 else 0
        fit_rows = vectors[: len(rows) - validation_count] if validation_count else vectors
        dimension = len(self.feature_names)
        self.means = [mean_of_column(fit_rows, column) for column in range(dimension)]
        self.scales = [
            (std_of_column(fit_rows, column, self.means[column]) or 1e-8) for column in range(dimension)
        ]

        standardised = [self._standardise(vector) for vector in vectors]
        train_Z = standardised[: len(rows) - validation_count] if validation_count else standardised
        train_y = labels[: len(rows) - validation_count] if validation_count else labels
        valid_Z = standardised[len(rows) - validation_count :] if validation_count else None
        valid_y = labels[len(rows) - validation_count :] if validation_count else None

        positives = float(sum(train_y))
        negatives = float(len(train_y) - positives)
        weights = [0.0] * dimension
        bias = math.log((positives + 1.0) / (negatives + 1.0))

        m_weights = [0.0] * dimension
        v_weights = [0.0] * dimension
        m_bias = 0.0
        v_bias = 0.0
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        best_loss = float("inf")
        best_weights = list(weights)
        best_bias = bias
        best_validation: float | None = None
        stale = 0

        for epoch in range(1, max(1, epochs) + 1):
            gradient_weights = [0.0] * dimension
            gradient_bias = 0.0
            for vector, label in zip(train_Z, train_y):
                probability = _sigmoid(bias + dot(weights, vector))
                error = probability - label
                gradient_bias += error
                for column in range(dimension):
                    gradient_weights[column] += error * vector[column]
            count = float(len(train_y))
            gradient_bias /= count
            for column in range(dimension):
                gradient_weights[column] = gradient_weights[column] / count + l2 * weights[column]

            for column in range(dimension):
                m_weights[column] = beta1 * m_weights[column] + (1 - beta1) * gradient_weights[column]
                v_weights[column] = beta2 * v_weights[column] + (1 - beta2) * gradient_weights[column] ** 2
            m_bias = beta1 * m_bias + (1 - beta1) * gradient_bias
            v_bias = beta2 * v_bias + (1 - beta2) * gradient_bias ** 2
            correction1 = 1 - beta1 ** epoch
            correction2 = 1 - beta2 ** epoch

            for column in range(dimension):
                weights[column] -= (
                    learning_rate
                    * (m_weights[column] / correction1)
                    / (math.sqrt(v_weights[column] / correction2) + epsilon)
                )
            bias -= learning_rate * (m_bias / correction1) / (math.sqrt(v_bias / correction2) + epsilon)

            if epoch % 5 == 0 or epoch == 1:
                reference_Z = valid_Z if valid_Z is not None else train_Z
                reference_y = valid_y if valid_y is not None else train_y
                loss = binary_log_loss(
                    [_sigmoid(bias + dot(weights, vector)) for vector in reference_Z], reference_y
                )
                if loss < best_loss - 1e-6:
                    best_loss = loss
                    best_weights = list(weights)
                    best_bias = bias
                    best_validation = loss if valid_Z is not None else None
                    stale = 0
                else:
                    stale += 1
                    if stale >= patience:
                        break

        self.weights = [float(value) for value in best_weights]
        self.bias = float(best_bias)
        self.training_samples = len(rows)
        self.validation_log_loss = best_validation
        return self

    # ------------------------------------------------------------- inference
    def _standardise(self, values: Sequence[float]) -> list[float]:
        if not self.means or not self.scales:
            return [max(-_STANDARDISED_LIMIT, min(_STANDARDISED_LIMIT, float(value))) for value in values]
        result = []
        for value, centre, scale in zip(values, self.means, self.scales):
            scaled = (float(value) - centre) / (scale if scale else 1e-8)
            result.append(max(-_STANDARDISED_LIMIT, min(_STANDARDISED_LIMIT, scaled)))
        return result

    def _vector(self, features: dict[str, float]) -> list[float]:
        return self._standardise(model_vector(features))

    def predict_up(self, features: dict[str, float]) -> float:
        if not self.trained:
            raise ValueError("Model has not been trained")
        return _sigmoid(self.bias + dot(self.weights, self._vector(features)))

    def predict_calibrated(self, features: dict[str, float]) -> float:
        return self.calibrator.transform(self.predict_up(features))

    def contributions(self, features: dict[str, float], top: int = 6) -> list[dict]:
        if not self.trained:
            return []
        vector = self._vector(features)
        rows = []
        for name, weight, value in zip(self.feature_names, self.weights, vector):
            rows.append(
                {
                    "feature": name,
                    "label": FEATURE_LABELS.get(name, name),
                    "value": round(float(value), 4),
                    "weight": round(float(weight), 4),
                    "contribution": round(float(weight * value), 4),
                }
            )
        rows.sort(key=lambda item: abs(item["contribution"]), reverse=True)
        return rows[: max(1, top)]

    # --------------------------------------------------------------- learning
    def learn_one(
        self,
        features: dict[str, float],
        label: int,
        learning_rate: float = 0.02,
        l2: float = 0.001,
    ) -> None:
        """One guarded online update after a confirmed real outcome."""
        if not self.trained:
            raise ValueError("Train the model before applying an online update")
        if label not in (0, 1):
            raise ValueError("Model labels must be 0 (down) or 1 (up)")
        vector = self._vector(features)
        probability = _sigmoid(self.bias + dot(self.weights, vector))
        error = probability - float(label)
        self.bias -= learning_rate * error
        self.weights = [
            weight - learning_rate * (error * value + l2 * weight)
            for weight, value in zip(self.weights, vector)
        ]
        self.online_updates += 1

    # ------------------------------------------------------------ persistence
    def to_dict(self) -> dict:
        return {
            "feature_names": list(self.feature_names),
            "feature_version": self.feature_version,
            "means": self.means,
            "scales": self.scales,
            "weights": self.weights,
            "bias": self.bias,
            "training_samples": self.training_samples,
            "calibrator": self.calibrator.to_dict(),
            "calibration_report": self.calibration_report,
            "validation_log_loss": self.validation_log_loss,
            "trained_at": self.trained_at,
            "mode": self.mode,
            "online_updates": self.online_updates,
        }

    @classmethod
    def from_dict(cls, payload: dict | None) -> "OnlineLogisticModel | None":
        if not payload:
            return None
        try:
            model = cls(
                feature_names=tuple(payload.get("feature_names", FEATURE_NAMES)),
                feature_version=int(payload.get("feature_version", 0)),
                means=[float(value) for value in payload.get("means", [])],
                scales=[float(value) for value in payload.get("scales", [])],
                weights=[float(value) for value in payload.get("weights", [])],
                bias=float(payload.get("bias", 0.0)),
                training_samples=int(payload.get("training_samples", 0)),
                calibrator=IsotonicCalibrator.from_dict(payload.get("calibrator")),
                calibration_report=payload.get("calibration_report"),
                validation_log_loss=payload.get("validation_log_loss"),
                trained_at=payload.get("trained_at"),
                mode=str(payload.get("mode", MODE_VETO)),
                online_updates=int(payload.get("online_updates", 0)),
            )
        except (TypeError, ValueError):
            return None
        if not model.trained or not model.compatible:
            return None
        return model


class BootstrapEnsemble:
    """Bootstrap members used only to measure how unstable the champion is.

    The champion supplies the probability; the members supply the spread. The
    calibrator was fitted on champion scores, so a member's score must never be
    fed through it.
    """

    def __init__(self, members: list[OnlineLogisticModel] | None = None):
        self.members: list[OnlineLogisticModel] = members or []

    @property
    def ready(self) -> bool:
        return len(self.members) >= 2 and all(member.trained for member in self.members)

    @classmethod
    def fit(
        cls,
        samples: Sequence[tuple[dict[str, float], int]],
        member_count: int = 4,
        seed: int = 20240501,
        epochs: int = 60,
    ) -> "BootstrapEnsemble":
        rows = list(samples)
        if len(rows) < 80 or member_count < 2:
            return cls([])
        rng = random.Random(seed)
        members: list[OnlineLogisticModel] = []
        for _ in range(member_count):
            resampled = [rows[rng.randrange(len(rows))] for _ in range(len(rows))]
            try:
                members.append(OnlineLogisticModel().fit(resampled, epochs=epochs, validation_fraction=0.0))
            except ValueError:
                continue
        return cls(members)

    def predict(self, features: dict[str, float]) -> tuple[float, float]:
        if not self.ready:
            raise ValueError("Ensemble is not ready")
        scores = [member.predict_up(features) for member in self.members]
        return mean_of(scores), std_of(scores)

    def to_dict(self) -> dict:
        return {"members": [member.to_dict() for member in self.members]}

    @classmethod
    def from_dict(cls, payload: dict | None) -> "BootstrapEnsemble":
        if not payload:
            return cls([])
        members = []
        for item in payload.get("members", []) or []:
            model = OnlineLogisticModel.from_dict(item)
            if model is not None:
                members.append(model)
        return cls(members)


# ----------------------------------------------------------------- utilities
def dot(weights: Sequence[float], vector: Sequence[float]) -> float:
    return sum(weight * value for weight, value in zip(weights, vector))


def mean_of(values: Sequence[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def std_of(values: Sequence[float]) -> float:
    items = list(values)
    if len(items) < 2:
        return 0.0
    centre = mean_of(items)
    return math.sqrt(sum((value - centre) ** 2 for value in items) / len(items))


def mean_of_column(rows: Sequence[Sequence[float]], column: int) -> float:
    values = [row[column] for row in rows]
    return mean_of(values)


def std_of_column(rows: Sequence[Sequence[float]], column: int, centre: float) -> float:
    values = [row[column] for row in rows]
    if len(values) < 2:
        return 0.0
    return math.sqrt(sum((value - centre) ** 2 for value in values) / len(values))


def binary_log_loss(probabilities: Sequence[float], labels: Sequence[float]) -> float:
    total = 0.0
    for probability, label in zip(probabilities, labels):
        p = min(1.0 - 1e-9, max(1e-9, probability))
        total -= math.log(p) if label == 1 else math.log(1.0 - p)
    return total / len(labels)


def evaluate(model: OnlineLogisticModel, rows: Sequence[tuple[dict[str, float], int]]) -> dict:
    """Score a model on held-out rows: calibration and accuracy, honestly."""
    if not rows:
        return {}
    points = [(model.predict_up(features), label) for features, label in rows]
    brier = brier_score(points)
    return {
        "samples": len(rows),
        "brier": brier,
        "brier_skill": skill_score(brier),
        "log_loss": log_loss(points),
        "accuracy": accuracy(points),
        "ece": expected_calibration_error(points),
    }
