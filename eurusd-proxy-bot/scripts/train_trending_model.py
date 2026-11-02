#!/usr/bin/env python3
"""
Train models on trending candle data for 95% win rate.

Uses only trending periods (ADX > 22) for higher accuracy.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fusion.domain import Candle
from fusion.features import compute_features, FEATURE_NAMES, FEATURE_VERSION
from fusion.gbdt import GradientBoostingModel, FeatureBaggingEnsemble

CANDLE_DIR = PROJECT_ROOT / "var" / "real_candles"
OUTPUT_DIR = CANDLE_DIR

HORIZON = 1  # 1-bar for quick scalps
MIN_BARS = 30
N_MODELS = 5


def load_trending_candles():
    """Load candles from batch_trending_*.txt files, sorted by timestamp."""
    candles = []
    for batch_file in sorted(CANDLE_DIR.glob("batch_trending_*.txt")):
        print(f"Loading {batch_file.name}...")
        content = batch_file.read_text().strip()
        for line in content.split(";"):
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 5:
                continue
            ts = int(float(parts[0]))
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            candles.append(Candle(
                timestamp=dt,
                open=float(parts[1]),
                high=float(parts[2]),
                low=float(parts[3]),
                close=float(parts[4]),
                volume=float(parts[5]) if len(parts) > 5 else 0.0,
            ))
    candles.sort(key=lambda c: c.timestamp)
    print(f"Total candles: {len(candles)}")
    return candles


def compute_features_all(candles):
    """Compute features for all candles."""
    return compute_features(candles)


def build_samples(candles, feat_dicts, horizon=None):
    """Build (features, label) samples."""
    if horizon is None:
        horizon = HORIZON

    samples = []
    for i in range(len(candles) - horizon):
        if i + horizon >= len(candles):
            break

        features = feat_dicts[i]
        if not features:
            continue

        # Label: 1 if up, 0 if down
        if candles[i + horizon].close > candles[i].close:
            label = 1.0
        else:
            label = 0.0

        samples.append((features, label))

    return samples


def train_ensemble(samples, n_models=5):
    """Train gradient boosting ensemble."""
    features = [s[0] for s in samples]
    labels = [s[1] for s in samples]

    print(f"\nTraining ensemble with {n_models} models on {len(samples)} samples...")

    models = []
    for i in range(n_models):
        # Random hyperparameters
        max_depth = 2 + (i % 3)  # 2-4
        n_estimators = 30 + (i * 10)  # 30-70
        lr = 0.15 - (i * 0.02)  # 0.15-0.07

        print(f"  Model {i+1}/{n_models}: depth={max_depth}, est={n_estimators}, lr={lr:.2f}")

        model = GradientBoostingModel(
            max_depth=max_depth,
            n_estimators=n_estimators,
            learning_rate=lr,
            feature_names=FEATURE_NAMES,
        )
        model.fit(features, labels)
        models.append(model)

    return models


def evaluate_ensemble(models, test_samples):
    """Evaluate ensemble on test set."""
    correct = 0
    total = 0
    probabilities = []

    for features, label in test_samples:
        # Get predictions from all models
        probs = []
        for model in models:
            prob_up = model.predict_up(features)
            probs.append(prob_up)

        # Ensemble prediction (majority vote)
        avg_prob = sum(probs) / len(probs)
        prediction = 1 if avg_prob >= 0.5 else 0

        if prediction == label:
            correct += 1
        total += 1
        probabilities.append(avg_prob)

    accuracy = correct / total if total > 0 else 0

    # Calculate confidence stats
    avg_confidence = sum(abs(p - 0.5) * 2 for p in probabilities) / len(probabilities) if probabilities else 0

    print(f"\nEvaluation:")
    print(f"  Accuracy: {accuracy*100:.1f}%")
    print(f"  Avg confidence: {avg_confidence*100:.1f}%")

    return accuracy, avg_confidence


def save_ensemble(models, output_path, feature_names):
    """Save ensemble to JSON."""
    data = {
        "version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "label": "direction",
        "horizon_bars": HORIZON,
        "feature_names": feature_names,
        "models": [m.to_dict() for m in models],
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\nSaved ensemble to: {output_path}")


def main():
    print("="*70)
    print("TRAINING ON TRENDING DATA")
    print("="*70)

    # Load trending candles
    candles = load_trending_candles()

    if len(candles) < MIN_BARS:
        print(f"Error: Too few candles for training ({len(candles)} < {MIN_BARS})")
        return

    # Compute features
    print("\nComputing features...")
    feat_dicts = compute_features_all(candles)

    # Build samples
    samples = build_samples(candles, feat_dicts, HORIZON)
    print(f"Total samples: {len(samples)}")

    if len(samples) < 100:
        print(f"Error: Too few samples for training ({len(samples)} < 100)")
        return

    # Split into train/test (80/20)
    split_idx = int(len(samples) * 0.8)
    train_samples = samples[:split_idx]
    test_samples = samples[split_idx:]

    print(f"Train: {len(train_samples)}, Test: {len(test_samples)}")

    # Train ensemble
    models = train_ensemble(train_samples, N_MODELS)

    # Evaluate
    accuracy, confidence = evaluate_ensemble(models, test_samples)

    # Save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = OUTPUT_DIR / f"ensemble_trending_{timestamp}.json"
    save_ensemble(models, output_file, FEATURE_NAMES)

    # Save report
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "total_candles": len(candles),
        "total_samples": len(samples),
        "train_samples": len(train_samples),
        "test_samples": len(test_samples),
        "accuracy": accuracy,
        "avg_confidence": confidence,
        "horizon_bars": HORIZON,
        "output_file": str(output_file),
    }

    report_file = OUTPUT_DIR / f"training_report_trending_{timestamp}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\nReport saved to: {report_file}")
    print(f"\n✓ Training complete with {accuracy*100:.1f}% accuracy")


if __name__ == "__main__":
    main()
