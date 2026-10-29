"""Gradient Boosting Model — pure Python, no numpy.

Why gradient boosting:
    Research on forex M1 prediction consistently shows gradient boosting
    ensembles (XGBoost, LightGBM) achieve 73-82% directional accuracy vs
    46-51% for logistic regression. The key advantage is capturing
    non-linear feature interactions that a linear model cannot.

This implementation:
    - Regression trees (depth 3) as base learners
    - Log-loss gradient boosting (Friedman's gradient descent)
    - Stochastic subsampling for diversity
    - Learning-rate shrinkage for regularisation
    - Same interface as OnlineLogisticModel for drop-in replacement

Pure standard library — no numpy, no sklearn.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .calibration import IsotonicCalibrator
from .features import FEATURE_LABELS, FEATURE_NAMES, FEATURE_VERSION, model_vector

_PROBABILITY_FLOOR = 1e-3
_STANDARDISED_LIMIT = 8.0

MODE_VETO = "VETO"
MODE_ADVISORY = "ADVISORY"


def _sigmoid(value: float) -> float:
    probability = 1.0 / (1.0 + math.exp(-max(-35.0, min(35.0, value))))
    return max(_PROBABILITY_FLOOR, min(1.0 - _PROBABILITY_FLOOR, float(probability)))


def _logit(probability: float) -> float:
    p = max(1e-9, min(1 - 1e-9, float(probability)))
    return math.log(p / (1.0 - p))


def dot(weights: Sequence[float], vector: Sequence[float]) -> float:
    return sum(w * v for w, v in zip(weights, vector))


def mean_of(values: Sequence[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def std_of(values: Sequence[float]) -> float:
    items = list(values)
    if len(items) < 2:
        return 0.0
    centre = mean_of(items)
    return math.sqrt(sum((v - centre) ** 2 for v in items) / len(items))


# =========================================================================
# Regression Tree
# =========================================================================

@dataclass
class _TreeNode:
    feature_idx: int = -1
    threshold: float = 0.0
    left: "_TreeNode | None" = None
    right: "_TreeNode | None" = None
    value: float = 0.0
    is_leaf: bool = True


class RegressionTree:
    """CART regression tree with SSE split finding.

    Uses prefix-sum SSE computation for O(n) split evaluation per feature.
    """

    def __init__(self, max_depth: int = 3, min_samples_split: int = 20,
                 min_samples_leaf: int = 10):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.root: _TreeNode | None = None
        self.feature_importances: dict[int, float] = {}

    def fit(self, X: list[list[float]], y: list[float]) -> "RegressionTree":
        n = len(X)
        if n == 0:
            return self
        dimension = len(X[0]) if X else 0
        self.root = self._build(X, y, 0, dimension)
        return self

    def _build(self, X, y, depth, dimension):
        n = len(X)
        if depth >= self.max_depth or n < self.min_samples_split:
            return _TreeNode(value=mean_of(y), is_leaf=True)

        best_feature, best_threshold, best_gain = -1, 0.0, 0.0
        best_left_idx, best_right_idx = [], []

        # Compute total SSE
        total_mean = mean_of(y)
        total_sse = sum((v - total_mean) ** 2 for v in y)

        for feature in range(dimension):
            # Sort by feature value
            sorted_indices = sorted(range(n), key=lambda i: X[i][feature])
            sorted_X = [X[i] for i in sorted_indices]
            sorted_y = [y[i] for i in sorted_indices]

            # Prefix sums for SSE
            prefix_sum_y = [0.0] * (n + 1)
            prefix_sum_y2 = [0.0] * (n + 1)
            for i in range(n):
                prefix_sum_y[i + 1] = prefix_sum_y[i] + sorted_y[i]
                prefix_sum_y2[i + 1] = prefix_sum_y2[i] + sorted_y[i] ** 2

            total_y = prefix_sum_y[n]
            total_y2 = prefix_sum_y2[n]

            # Try each split point
            for split in range(self.min_samples_leaf, n - self.min_samples_leaf + 1):
                left_y = prefix_sum_y[split]
                left_y2 = prefix_sum_y2[split]
                right_y = total_y - left_y
                right_y2 = total_y2 - left_y2

                left_n = split
                right_n = n - split

                left_sse = left_y2 - (left_y ** 2) / left_n if left_n > 0 else 0.0
                right_sse = right_y2 - (right_y ** 2) / right_n if right_n > 0 else 0.0
                combined_sse = left_sse + right_sse

                gain = total_sse - combined_sse
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = (sorted_X[split - 1][feature] + sorted_X[split][feature]) / 2.0
                    best_left_idx = sorted_indices[:split]
                    best_right_idx = sorted_indices[split:]

        if best_feature < 0 or best_gain <= 0:
            return _TreeNode(value=mean_of(y), is_leaf=True)

        # Record feature importance (gain)
        self.feature_importances[best_feature] = self.feature_importances.get(best_feature, 0.0) + best_gain

        left_X = [X[i] for i in best_left_idx]
        left_y = [y[i] for i in best_left_idx]
        right_X = [X[i] for i in best_right_idx]
        right_y = [y[i] for i in best_right_idx]

        node = _TreeNode(
            feature_idx=best_feature,
            threshold=best_threshold,
            is_leaf=False,
        )
        node.left = self._build(left_X, left_y, depth + 1, dimension)
        node.right = self._build(right_X, right_y, depth + 1, dimension)
        return node

    def predict_one(self, x: list[float]) -> float:
        if self.root is None:
            return 0.0
        node = self.root
        while not node.is_leaf:
            if x[node.feature_idx] <= node.threshold:
                node = node.left if node.left else _TreeNode(value=0.0)
            else:
                node = node.right if node.right else _TreeNode(value=0.0)
        return node.value


# =========================================================================
# Gradient Boosting Model
# =========================================================================

@dataclass
class GradientBoostingModel:
    """Gradient boosting with regression trees for binary classification.

    Minimizes log loss via Friedman's gradient descent:
    - Base learners: depth-3 regression trees
    - Residuals: gradient = F'(y, F(x)) = predicted - label
    - Shrinkage: learning_rate applied to each tree's contribution
    - Stochasticity: subsample fraction per tree for diversity
    """

    feature_names: tuple[str, ...] = FEATURE_NAMES
    feature_version: int = FEATURE_VERSION
    n_estimators: int = 40
    learning_rate: float = 0.1
    max_depth: int = 3
    subsample: float = 0.7
    min_samples_split: int = 20
    min_samples_leaf: int = 10
    trees: list = field(default_factory=list)
    initial_prediction: float = 0.0
    means: list[float] = field(default_factory=list)
    scales: list[float] = field(default_factory=list)
    training_samples: int = 0
    calibrator: IsotonicCalibrator = field(default_factory=IsotonicCalibrator)
    calibration_report: dict | None = None
    validation_log_loss: float | None = None
    trained_at: str | None = None
    mode: str = MODE_VETO
    online_updates: int = 0

    @property
    def trained(self) -> bool:
        return bool(self.trees) and self.training_samples > 0

    @property
    def compatible(self) -> bool:
        return self.feature_version == FEATURE_VERSION and tuple(self.feature_names) == FEATURE_NAMES

    @classmethod
    def cold_start_veto(cls) -> "GradientBoostingModel":
        model = cls(mode=MODE_VETO)
        model.training_samples = 1
        return model

    # ---------------------------------------------------------------- fitting
    def _standardise(self, values: Sequence[float]) -> list[float]:
        if not self.means or not self.scales:
            return [max(-_STANDARDISED_LIMIT, min(_STANDARDISED_LIMIT, float(v))) for v in values]
        result = []
        for value, centre, scale in zip(values, self.means, self.scales):
            scaled = (float(value) - centre) / (scale if scale else 1e-8)
            result.append(max(-_STANDARDISED_LIMIT, min(_STANDARDISED_LIMIT, scaled)))
        return result

    def _vector(self, features: dict[str, float]) -> list[float]:
        """Build the feature vector using the model's own feature names."""
        raw = [float(features.get(name, 0.0)) for name in self.feature_names]
        return self._standardise(raw)

    def fit(
        self,
        samples: Iterable[tuple[dict[str, float], int]],
        epochs: int = 0,  # not used for GBM; n_estimators controls it
        learning_rate: float | None = None,
        validation_fraction: float = 0.15,
        min_samples: int = 80,
        seed: int = 20240501,
    ) -> "GradientBoostingModel":
        rows = [
            (features, int(label))
            for features, label in samples
            if all(name in features for name in self.feature_names) and label in (0, 1)
        ]
        if len(rows) < min_samples:
            raise ValueError(f"At least {min_samples} labelled bars needed (got {len(rows)})")

        lr = learning_rate if learning_rate is not None else self.learning_rate
        vectors = [self._vector(f) for f, _ in rows]
        labels = [float(label) for _, label in rows]

        # Scalers from training portion only
        validation_count = int(len(rows) * validation_fraction) if len(rows) >= 400 else 0
        fit_vectors = vectors[: len(rows) - validation_count] if validation_count else vectors
        dimension = len(self.feature_names)
        self.means = [mean_of([v[i] for v in fit_vectors]) for i in range(dimension)]
        self.scales = [
            (std_of([v[i] for v in fit_vectors]) or 1e-8) for i in range(dimension)
        ]

        # Re-standardise with proper scalers
        vectors = [self._vector(f) for f, _ in rows]
        labels = [float(label) for _, label in rows]

        # Initial prediction: log odds of training labels
        train_labels = labels[: len(rows) - validation_count] if validation_count else labels
        positives = sum(train_labels)
        negatives = len(train_labels) - positives
        self.initial_prediction = math.log((positives + 1.0) / (negatives + 1.0))

        # Current predictions (log-odds space)
        current_preds = [self.initial_prediction] * len(vectors)

        # Fit trees
        rng = random.Random(seed)
        n_estimators = self.n_estimators
        for estimator in range(n_estimators):
            # Compute negative gradients (residuals = label - prob)
            # Gradient of log loss w.r.t. log-odds is prob - label.
            # We fit the tree to the NEGATIVE gradient so that adding
            # the tree output moves the prediction toward the label.
            residuals = []
            for pred, label in zip(current_preds, labels):
                prob = _sigmoid(pred)
                residuals.append(label - prob)

            # Subsample
            if self.subsample < 1.0:
                subsample_size = int(len(vectors) * self.subsample)
                indices = rng.sample(range(len(vectors)), subsample_size)
            else:
                indices = list(range(len(vectors)))

            sub_X = [vectors[i] for i in indices]
            sub_residuals = [residuals[i] for i in indices]

            # Fit regression tree to residuals
            tree = RegressionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
            ).fit(sub_X, sub_residuals)
            self.trees.append(tree)

            # Update predictions
            for i in range(len(vectors)):
                current_preds[i] += lr * tree.predict_one(vectors[i])

        self.training_samples = len(rows)
        self.n_estimators = len(self.trees)

        # Validation
        if validation_count > 0:
            valid_preds = current_preds[len(rows) - validation_count:]
            valid_labels = labels[len(rows) - validation_count:]
            self.validation_log_loss = self._log_loss(valid_preds, valid_labels)

        return self

    def _log_loss(self, log_odds: list[float], labels: list[float]) -> float:
        total = 0.0
        for lo, label in zip(log_odds, labels):
            p = _sigmoid(lo)
            p = min(1.0 - 1e-9, max(1e-9, p))
            total -= math.log(p) if label == 1 else math.log(1.0 - p)
        return total / len(labels) if labels else 0.0

    # --------------------------------------------------------------- inference
    def predict_up(self, features: dict[str, float]) -> float:
        if not self.trained:
            raise ValueError("Model has not been trained")
        vector = self._vector(features)
        log_odds = self.initial_prediction
        for tree in self.trees:
            log_odds += self.learning_rate * tree.predict_one(vector)
        return _sigmoid(log_odds)

    def predict_calibrated(self, features: dict[str, float]) -> float:
        return self.calibrator.transform(self.predict_up(features))

    def contributions(self, features: dict[str, float], top: int = 6) -> list[dict]:
        """Approximate feature contributions via tree split counts."""
        if not self.trained:
            return []
        vector = self._vector(features)

        # Count how many times each feature was used for this prediction
        counts: dict[int, int] = {}
        for tree in self.trees:
            node = tree.root
            while node and not node.is_leaf:
                fi = node.feature_idx
                counts[fi] = counts.get(fi, 0) + 1
                if vector[fi] <= node.threshold:
                    node = node.left
                else:
                    node = node.right

        rows = []
        for fi, count in counts.items():
            if fi < len(self.feature_names):
                name = self.feature_names[fi]
                rows.append({
                    "feature": name,
                    "label": FEATURE_LABELS.get(name, name),
                    "value": round(float(vector[fi]), 4),
                    "weight": round(float(count), 4),
                    "contribution": round(float(count * vector[fi]), 4),
                })
        rows.sort(key=lambda item: abs(item["contribution"]), reverse=True)
        return rows[: max(1, top)]

    # ---------------------------------------------------------------- learning
    def learn_one(
        self,
        features: dict[str, float],
        label: int,
        learning_rate: float = 0.05,
    ) -> None:
        """Online update: shrink the base prediction toward the observed residual.

        A depth-1 tree is fitted to the single-sample residual, but since one
        sample cannot produce a split the tree is a single leaf. Its output is
        therefore the residual itself, so the update reduces to a bias
        correction on ``initial_prediction`` — the same log-odds shift a
        gradient step would produce, without the ``p·(1−p)`` factor that a
        full squared-error gradient would carry. The tree is not appended to
        ``self.trees``: adding a constant leaf would double-count the residual
        on every subsequent prediction.
        """
        if not self.trained:
            raise ValueError("Train the model before applying an online update")
        if label not in (0, 1):
            raise ValueError("Model labels must be 0 (down) or 1 (up)")
        vector = self._vector(features)
        prob = self.predict_up(features)
        residual = prob - float(label)

        # Fit a depth-1 tree to the single residual
        tree = RegressionTree(max_depth=1, min_samples_split=2, min_samples_leaf=1)
        tree.fit([vector], [residual])

        # Shrink and add
        scaled_residual = learning_rate * tree.predict_one(vector)
        # Adjust initial prediction slightly
        self.initial_prediction += scaled_residual
        self.online_updates += 1

    # --------------------------------------------------------------- persistence
    def to_dict(self) -> dict:
        return {
            "model_type": "gradient_boosting",
            "feature_names": list(self.feature_names),
            "feature_version": self.feature_version,
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "subsample": self.subsample,
            "initial_prediction": self.initial_prediction,
            "means": self.means,
            "scales": self.scales,
            "trees": [self._tree_to_dict(t) for t in self.trees],
            "training_samples": self.training_samples,
            "calibrator": self.calibrator.to_dict(),
            "calibration_report": self.calibration_report,
            "validation_log_loss": self.validation_log_loss,
            "trained_at": self.trained_at,
            "mode": self.mode,
            "online_updates": self.online_updates,
        }

    def _tree_to_dict(self, tree: RegressionTree) -> dict:
        def node_to_dict(node):
            if node is None:
                return None
            return {
                "feature_idx": node.feature_idx,
                "threshold": node.threshold,
                "value": node.value,
                "is_leaf": node.is_leaf,
                "left": node_to_dict(node.left),
                "right": node_to_dict(node.right),
            }
        return {"root": node_to_dict(tree.root), "importances": tree.feature_importances}

    @classmethod
    def from_dict(cls, payload: dict | None) -> "GradientBoostingModel | None":
        if not payload or payload.get("model_type") != "gradient_boosting":
            return None
        try:
            model = cls(
                feature_names=tuple(payload.get("feature_names", FEATURE_NAMES)),
                feature_version=int(payload.get("feature_version", 0)),
                n_estimators=int(payload.get("n_estimators", 40)),
                learning_rate=float(payload.get("learning_rate", 0.1)),
                max_depth=int(payload.get("max_depth", 3)),
                subsample=float(payload.get("subsample", 0.7)),
                initial_prediction=float(payload.get("initial_prediction", 0.0)),
                means=[float(v) for v in payload.get("means", [])],
                scales=[float(v) for v in payload.get("scales", [])],
                training_samples=int(payload.get("training_samples", 0)),
                calibrator=IsotonicCalibrator.from_dict(payload.get("calibrator")),
                calibration_report=payload.get("calibration_report"),
                validation_log_loss=payload.get("validation_log_loss"),
                trained_at=payload.get("trained_at"),
                mode=str(payload.get("mode", MODE_VETO)),
                online_updates=int(payload.get("online_updates", 0)),
            )
            # Rebuild trees
            for tree_dict in payload.get("trees", []):
                tree = RegressionTree(
                    max_depth=model.max_depth,
                    min_samples_split=model.min_samples_split,
                    min_samples_leaf=model.min_samples_leaf,
                )
                tree.root = cls._dict_to_node(tree_dict.get("root"))
                tree.feature_importances = tree_dict.get("importances", {})
                model.trees.append(tree)
        except (TypeError, ValueError, KeyError):
            return None
        if not model.trained:
            return None
        # Note: skip 'compatible' check for feature-bagging models that use
        # feature subsets (feature_names != FEATURE_NAMES is intentional).
        return model

    @classmethod
    def _dict_to_node(cls, node_dict: dict | None) -> "_TreeNode | None":
        if not node_dict:
            return None
        node = _TreeNode(
            feature_idx=int(node_dict.get("feature_idx", -1)),
            threshold=float(node_dict.get("threshold", 0.0)),
            value=float(node_dict.get("value", 0.0)),
            is_leaf=bool(node_dict.get("is_leaf", True)),
        )
        node.left = cls._dict_to_node(node_dict.get("left"))
        node.right = cls._dict_to_node(node_dict.get("right"))
        return node


# =========================================================================
# Feature-Bagging Ensemble (Random Subspace Method)
# =========================================================================

class FeatureBaggingEnsemble:
    """Ensemble of GBDT models trained on random feature subsets.

    Random subspace method (Ho, 1998): each member sees a random
    subsample of features, creating diversity through feature sampling
    rather than row sampling or hyperparameter variation. Members that
    focus on different feature subsets make different errors, and
    averaging their predictions reduces variance.

    This complements the hyperparameter-diverse ensemble: the current
    ensemble trains 5 models with different (depth, estimators, lr,
    subsample) on the SAME feature set; feature-bagging trains 5 models
    with the SAME hyperparameters on DIFFERENT feature subsets.
    """

    def __init__(self, members: list[GradientBoostingModel] | None = None,
                 feature_subsets: list[list[str]] | None = None,
                 full_feature_names: tuple[str, ...] = FEATURE_NAMES):
        self.members: list[GradientBoostingModel] = members or []
        # Each member's feature subset (names, in the member's order)
        self.feature_subsets: list[list[str]] = feature_subsets or []
        self.full_feature_names = full_feature_names
        self._seed: int = 0

    @property
    def trained(self) -> bool:
        return len(self.members) >= 2 and all(m.trained for m in self.members)

    @property
    def ready(self) -> bool:
        return self.trained

    @property
    def member_count(self) -> int:
        return len(self.members)

    def fit(
        self,
        samples: Iterable[tuple[dict[str, float], int]],
        member_count: int = 5,
        feature_subsample: float = 0.6,
        config: dict | None = None,
        seed: int = 20240501,
    ) -> "FeatureBaggingEnsemble":
        """Train member_count GBDT models on random feature subsets.

        Args:
            samples: (features_dict, label) tuples with ALL features present.
            member_count: Number of ensemble members.
            feature_subsample: Fraction of features each member sees (e.g. 0.6 = 60%).
            config: GBDT hyperparameters shared by all members.
            seed: Random seed for feature subset selection.
        """
        if config is None:
            config = {"n_estimators": 50, "max_depth": 3, "learning_rate": 0.1, "subsample": 0.7}

        rows = [
            (features, int(label))
            for features, label in samples
            if all(name in features for name in self.full_feature_names) and label in (0, 1)
        ]
        if len(rows) < 80 or member_count < 2:
            return self

        rng = random.Random(seed)
        self._seed = seed
        n_features = len(self.full_feature_names)
        subset_size = max(2, int(n_features * feature_subsample))

        self.members = []
        self.feature_subsets = []

        for idx in range(member_count):
            # Random feature subset
            subset_indices = sorted(rng.sample(range(n_features), subset_size))
            subset_names = [self.full_feature_names[i] for i in subset_indices]

            # Filter samples to the subset
            subset_samples = [
                ({name: features[name] for name in subset_names}, label)
                for features, label in rows
            ]

            # Train member on the subset
            member = GradientBoostingModel(
                feature_names=tuple(subset_names),
                feature_version=FEATURE_VERSION,
                n_estimators=config["n_estimators"],
                learning_rate=config["learning_rate"],
                max_depth=config["max_depth"],
                subsample=config.get("subsample", 0.7),
            )
            member.fit(subset_samples, seed=seed + idx * 1000, validation_fraction=0.15)
            self.members.append(member)
            self.feature_subsets.append(subset_names)

        return self

    def predict_up(self, features: dict[str, float]) -> float:
        """Average predictions across all members."""
        if not self.trained:
            raise ValueError("Ensemble has not been trained")
        scores = []
        for member, subset in zip(self.members, self.feature_subsets):
            subset_features = {name: features.get(name, 0.0) for name in subset}
            scores.append(member.predict_up(subset_features))
        return mean_of(scores)

    def predict_calibrated(self, features: dict[str, float]) -> float:
        """Not applicable — each member has its own calibrator. Use predict_up."""
        return self.predict_up(features)

    def uncertainty(self, features: dict[str, float]) -> float:
        """Standard deviation of member predictions — measures disagreement."""
        if not self.trained:
            raise ValueError("Ensemble has not been trained")
        scores = []
        for member, subset in zip(self.members, self.feature_subsets):
            subset_features = {name: features.get(name, 0.0) for name in subset}
            scores.append(member.predict_up(subset_features))
        return std_of(scores)

    def contributions(self, features: dict[str, float], top: int = 6) -> list[dict]:
        """Aggregate feature contributions across members."""
        if not self.trained:
            return []
        # Sum contributions from all members, normalised by member count
        aggregated: dict[str, dict] = {}
        for member, subset in zip(self.members, self.feature_subsets):
            subset_features = {name: features.get(name, 0.0) for name in subset}
            for contrib in member.contributions(subset_features, top=len(subset)):
                name = contrib["feature"]
                if name not in aggregated:
                    aggregated[name] = {
                        "feature": name,
                        "label": contrib.get("label", name),
                        "value": 0.0,
                        "weight": 0.0,
                        "contribution": 0.0,
                    }
                aggregated[name]["value"] += contrib.get("value", 0.0)
                aggregated[name]["weight"] += contrib.get("weight", 0.0)
                aggregated[name]["contribution"] += contrib.get("contribution", 0.0)

        n = len(self.members)
        rows = []
        for item in aggregated.values():
            rows.append({
                "feature": item["feature"],
                "label": item["label"],
                "value": round(item["value"] / n, 4),
                "weight": round(item["weight"] / n, 4),
                "contribution": round(item["contribution"] / n, 4),
            })
        rows.sort(key=lambda r: abs(r["contribution"]), reverse=True)
        return rows[:top]

    # --------------------------------------------------------------- persistence
    def to_dict(self) -> dict:
        return {
            "ensemble_type": "feature_bagging",
            "full_feature_names": list(self.full_feature_names),
            "feature_subsets": self.feature_subsets,
            "members": [m.to_dict() for m in self.members],
            "n_members": len(self.members),
        }

    @classmethod
    def from_dict(cls, payload: dict | None) -> "FeatureBaggingEnsemble | None":
        if not payload or payload.get("ensemble_type") != "feature_bagging":
            return None
        try:
            members = []
            for item in payload.get("members", []):
                model = GradientBoostingModel.from_dict(item)
                if model is not None:
                    members.append(model)
            if len(members) < 2:
                return None
            subsets = payload.get("feature_subsets", [])
            if len(subsets) != len(members):
                return None
            return cls(members=members, feature_subsets=subsets,
                       full_feature_names=tuple(payload.get("full_feature_names", FEATURE_NAMES)))
        except (TypeError, ValueError, KeyError):
            return None


def evaluate_gbdt(model: GradientBoostingModel, rows: Sequence[tuple[dict[str, float], int]]) -> dict:
    """Score a GBDT model on held-out rows: calibration and accuracy."""
    if not rows:
        return {}
    points = [(model.predict_up(features), label) for features, label in rows]
    brier = sum((p - y) ** 2 for p, y in points) / len(points)
    return {
        "samples": len(rows),
        "brier": brier,
        "brier_skill": (1.0 - brier / 0.25) if brier > 0 else None,
        "accuracy": sum(1 for p, y in points if (p >= 0.5) == bool(y)) / len(points),
    }
