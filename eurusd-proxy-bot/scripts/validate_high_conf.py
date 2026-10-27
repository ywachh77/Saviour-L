#!/usr/bin/env python3
"""
Validate high-confidence signal accuracy — walk-forward OOS evaluation.

Key question: is the 93.4% accuracy at threshold ≥0.60 genuine (OOS) or in-sample overfitting?
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fusion.domain import Candle
from fusion.features import compute_features, FEATURE_NAMES
from fusion.gbdt import GradientBoostingModel

CANDLE_DIR = PROJECT_ROOT / "var" / "real_candles"
HORIZON = 5
MIN_BARS = 30

ENSEMBLE_CONFIGS = [
    {"n_estimators": 50, "learning_rate": 0.15, "max_depth": 3, "subsample": 0.7},
    {"n_estimators": 50, "learning_rate": 0.1, "max_depth": 4, "subsample": 0.8},
    {"n_estimators": 80, "learning_rate": 0.08, "max_depth": 3, "subsample": 0.6},
]


def load_candles() -> list[Candle]:
    candles = []
    for batch_file in sorted(CANDLE_DIR.glob("batch_*.txt")):
        for line in batch_file.read_text().strip().split(";"):
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 5:
                continue
            ts = int(parts[0])
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            candles.append(Candle(timestamp=dt, open=float(parts[1]), high=float(parts[2]),
                                  low=float(parts[3]), close=float(parts[4]), volume=0.0))
    candles.sort(key=lambda c: c.timestamp)
    return candles


def build_samples(candles):
    feature_dicts = compute_features(candles)
    samples = []
    for i in range(MIN_BARS, len(candles) - HORIZON):
        feat = feature_dicts[i]
        if feat is None:
            continue
        label = 1 if candles[i + HORIZON].close > candles[i].close else 0
        samples.append((feat, label))
    return samples


def purged_walk_forward_split(samples, n_folds=5, embargo_pct=0.05):
    total = len(samples)
    fold_size = total // (n_folds + 1)
    splits = []
    embargo_size = int(fold_size * embargo_pct)
    for fold in range(n_folds):
        test_start = fold * fold_size
        test_end = test_start + fold_size
        train_indices = []
        for i in range(test_start):
            if i + HORIZON < test_start - embargo_size:
                train_indices.append(i)
        test_indices = list(range(test_start, test_end))
        if train_indices and test_indices:
            splits.append((train_indices, test_indices))
    return splits


def main():
    candles = load_candles()
    print(f"Loaded {len(candles)} candles")
    samples = build_samples(candles)
    print(f"Built {len(samples)} samples")

    splits = purged_walk_forward_split(samples, n_folds=5)
    print(f"Walk-forward splits: {len(splits)}")

    # Collect all OOS predictions
    all_oos_preds = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        # Train ensemble on this fold
        models = []
        for config in ENSEMBLE_CONFIGS:
            model = GradientBoostingModel(
                n_estimators=config["n_estimators"],
                learning_rate=config["learning_rate"],
                max_depth=config["max_depth"],
                subsample=config["subsample"],
            )
            model.fit([(samples[i][0], samples[i][1]) for i in train_idx])
            models.append(model)

        # Predict on test set
        for idx in test_idx:
            probs = [m.predict_up(samples[idx][0]) for m in models]
            avg_prob = sum(probs) / len(probs)
            all_oos_preds.append((avg_prob, samples[idx][1]))

        if fold_idx % 2 == 0:
            print(f"  Fold {fold_idx}: trained {len(models)} models, "
                  f"{len(test_idx)} test samples")

    print(f"\nTotal OOS predictions: {len(all_oos_preds)}")

    # Evaluate at different confidence thresholds
    print(f"\n{'='*60}")
    print("OUT-OF-SAMPLE ACCURACY BY CONFIDENCE THRESHOLD")
    print(f"{'='*60}")
    print(f"  {'Threshold':<12} {'Trades':<8} {'Accuracy':<10} {'NetPips':<10} {'PF':<8}")
    print(f"  {'-'*48}")

    for threshold in [0.50, 0.52, 0.55, 0.58, 0.60, 0.62, 0.65, 0.68, 0.70, 0.75, 0.80]:
        filtered = [(p, l) for p, l in all_oos_preds if p >= threshold]
        if len(filtered) < 5:
            continue
        n_correct = sum(1 for p, l in filtered if (p > 0.5) == bool(l))
        accuracy = n_correct / len(filtered)
        gross_win = n_correct
        gross_loss = len(filtered) - n_correct
        net = (gross_win - gross_loss) * 5.0
        pf = (gross_win / gross_loss) if gross_loss > 0 else float("inf")
        print(f"  {threshold:.2f}       {len(filtered):<8} {accuracy:<10.4f} {net:<10.2f} {pf:<8.2f}")

    # Also evaluate at both extremes (very confident up or down)
    print(f"\n  {'VERY HIGH CONF':<12} {'Trades':<8} {'Accuracy':<10} {'NetPips':<10} {'PF':<8}")
    for threshold in [0.85, 0.90, 0.95]:
        filtered = [(p, l) for p, l in all_oos_preds if p >= threshold or p <= 1 - threshold]
        if len(filtered) < 5:
            continue
        n_correct = sum(1 for p, l in filtered if (p > 0.5) == bool(l))
        accuracy = n_correct / len(filtered)
        gross_win = n_correct
        gross_loss = len(filtered) - n_correct
        net = (gross_win - gross_loss) * 5.0
        pf = (gross_win / gross_loss) if gross_loss > 0 else float("inf")
        print(f"  {threshold:.2f}+/-{1-threshold:.2f}   {len(filtered):<8} {accuracy:<10.4f} {net:<10.2f} {pf:<8.2f}")

    # Check: is the 93.4% from the chart in-sample?
    print(f"\n{'='*60}")
    print("IN-SAMPLE vs OUT-OF-SAMPLE COMPARISON")
    print(f"{'='*60}")
    # In-sample: train on all data, predict on all data
    models_all = []
    for config in ENSEMBLE_CONFIGS:
        model = GradientBoostingModel(
            n_estimators=config["n_estimators"],
            learning_rate=config["learning_rate"],
            max_depth=config["max_depth"],
            subsample=config["subsample"],
        )
        model.fit(samples)
        models_all.append(model)

    is_preds = []
    for feat, label in samples:
        probs = [m.predict_up(feat) for m in models_all]
        avg_prob = sum(probs) / len(probs)
        is_preds.append((avg_prob, label))

    for threshold in [0.50, 0.55, 0.60, 0.65, 0.70]:
        is_filtered = [(p, l) for p, l in is_preds if p >= threshold]
        oos_filtered = [(p, l) for p, l in all_oos_preds if p >= threshold]
        if len(is_filtered) < 5 or len(oos_filtered) < 5:
            continue
        is_acc = sum(1 for p, l in is_filtered if (p > 0.5) == bool(l)) / len(is_filtered)
        oos_acc = sum(1 for p, l in oos_filtered if (p > 0.5) == bool(l)) / len(oos_filtered)
        print(f"  Threshold {threshold:.2f}: In-sample={is_acc:.4f} ({len(is_filtered)} trades) "
              f"vs OOS={oos_acc:.4f} ({len(oos_filtered)} trades)")


if __name__ == "__main__":
    main()
