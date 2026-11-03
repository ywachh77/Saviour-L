"""Loss pattern extraction and analysis for self-learning.

Extracts common patterns from losing trades and uses them to:
1. Identify which feature combinations consistently lose
2. Adjust model weights to penalize similar setups
3. Create a feedback loop where losses trigger targeted learning
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Sequence

from . import charter


@dataclass(frozen=True, slots=True)
class LossPattern:
    """A common pattern found in losing trades."""
    feature_name: str
    typical_value: float
    occurrence_count: int
    total_loss_pips: float
    win_rate_when_present: float  # How often this pattern appears in wins vs losses

    def to_dict(self) -> dict:
        return {
            "feature_name": self.feature_name,
            "typical_value": self.typical_value,
            "occurrence_count": self.occurrence_count,
            "total_loss_pips": self.total_loss_pips,
            "win_rate_when_present": self.win_rate_when_present,
        }


@dataclass
class LossAnalysis:
    """Complete analysis of recent losses."""
    total_losses: int = 0
    total_loss_pips: float = 0.0
    patterns: list = field(default_factory=list)
    most_damaging_features: list = field(default_factory=list)
    regime_breakdown: dict = field(default_factory=dict)
    session_breakdown: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "total_losses": self.total_losses,
            "total_loss_pips": self.total_loss_pips,
            "patterns": [p.to_dict() for p in self.patterns],
            "most_damaging_features": self.most_damaging_features,
            "regime_breakdown": self.regime_breakdown,
            "session_breakdown": self.session_breakdown,
        }


def analyze_losses(settlements: Sequence[dict], features_by_signal: dict | None = None) -> LossAnalysis:
    """Analyze losing trades to extract common patterns.

    Args:
        settlements: List of settlement dictionaries
        features_by_signal: Optional mapping of signal_id to feature dict
    """
    analysis = LossAnalysis()
    losses = [s for s in settlements if s.get("outcome") == charter.LOSS]
    analysis.total_losses = len(losses)
    analysis.total_loss_pips = sum(float(s.get("net_pips") or 0.0) for s in losses)

    if not losses:
        return analysis

    # Extract feature patterns from losses
    feature_values: dict[str, list[float]] = defaultdict(list)
    feature_losses: dict[str, float] = defaultdict(float)
    feature_counts: dict[str, int] = defaultdict(int)
    win_feature_counts: dict[str, int] = defaultdict(int)

    wins = [s for s in settlements if s.get("outcome") == charter.WIN]
    total_settlements = len(settlements)

    for settlement in losses:
        signal_id = settlement.get("signal_id")
        features = (features_by_signal or {}).get(signal_id, {})
        net_pips = float(settlement.get("net_pips") or 0.0)

        for key, value in features.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                feature_values[key].append(float(value))
                feature_losses[key] += net_pips
                feature_counts[key] += 1

    for settlement in wins:
        signal_id = settlement.get("signal_id")
        features = (features_by_signal or {}).get(signal_id, {})
        for key in features:
            if isinstance(features[key], (int, float)) and not isinstance(features[key], bool):
                win_feature_counts[key] += 1

    # Build loss patterns
    for feature, values in feature_values.items():
        if not values:
            continue
        typical_value = sum(values) / len(values)
        total_loss = feature_losses[feature]
        loss_count = feature_counts[feature]
        win_count = win_feature_counts.get(feature, 0)

        pattern = LossPattern(
            feature_name=feature,
            typical_value=typical_value,
            occurrence_count=loss_count,
            total_loss_pips=total_loss,
            win_rate_when_present=win_count / (loss_count + win_count) if (loss_count + win_count) > 0 else 0.0,
        )
        analysis.patterns.append(pattern)

    # Sort by total loss (most damaging first)
    analysis.patterns.sort(key=lambda p: p.total_loss_pips)
    analysis.most_damaging_features = [
        {"feature": p.feature_name, "loss_pips": p.total_loss_pips}
        for p in analysis.patterns[:10]
    ]

    # Regime breakdown
    for settlement in losses:
        regime = settlement.get("regime") or "UNKNOWN"
        analysis.regime_breakdown[regime] = analysis.regime_breakdown.get(regime, 0) + 1

    # Session breakdown
    for settlement in losses:
        session = settlement.get("session") or "UNKNOWN"
        analysis.session_breakdown[session] = analysis.session_breakdown.get(session, 0) + 1

    return analysis


def extract_loss_features(losses: Sequence[dict], features_by_signal: dict) -> list[dict]:
    """Extract feature snapshots from losing trades for targeted retraining."""
    loss_features = []
    for loss in losses:
        signal_id = loss.get("signal_id")
        features = features_by_signal.get(signal_id, {})
        if features:
            loss_features.append({
                "features": features,
                "label": 0,  # Loss = label 0
                "gross_pips": float(loss.get("net_pips") or 0.0),
                "signal_id": signal_id,
            })
    return loss_features


def should_trigger_loss_learning(settlements: Sequence[dict], min_trades: int = 20, target_win_rate: float = 0.95) -> tuple[bool, str]:
    """Determine if losses warrant immediate learning intervention.

    Returns (should_learn, reason)
    """
    real = [s for s in settlements if not charter.is_synthetic(s.get("source"))]
    resolved = [s for s in real if s.get("outcome") in (charter.WIN, charter.LOSS)]

    if len(resolved) < min_trades:
        return False, f"Only {len(resolved)} resolved trades; {min_trades} needed to judge."

    wins = sum(1 for s in resolved if s.get("outcome") == charter.WIN)
    win_rate = wins / len(resolved)

    if win_rate < target_win_rate:
        return True, (
            f"Win rate {win_rate:.1%} is below target {target_win_rate:.1%} "
            f"({wins}/{len(resolved)} wins). Loss learning activated."
        )

    return False, (
        f"Win rate {win_rate:.1%} meets target {target_win_rate:.1%}. "
        f"No loss learning needed."
    )


def calculate_loss_learning_rate(base_rate: float, loss_count: int, win_rate: float) -> float:
    """Calculate an aggressive learning rate for loss-focused training.

    More losses = higher learning rate, capped to prevent overreaction.
    """
    if win_rate >= 0.95:
        return base_rate

    # Scale up based on how far below target we are
    deficit = 0.95 - win_rate
    multiplier = 1.0 + (deficit * 4.0)  # 4x amplification per unit deficit

    # More losses = higher rate
    loss_multiplier = min(3.0, 1.0 + (loss_count / 50.0))

    return min(0.12, base_rate * multiplier * loss_multiplier)
