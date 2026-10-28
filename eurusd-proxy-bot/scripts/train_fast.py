#!/usr/bin/env python3
"""
Fast Multi-Horizon GBDT Training — 1min, 3min, 30min expiry.
Uses a subset of candles for faster training in pure Python.
"""

from __future__ import annotations

import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fusion.domain import Candle
from fusion.features import compute_features, FEATURE_NAMES, FEATURE_LABELS, FEATURE_VERSION
from fusion.gbdt import GradientBoostingModel

# --- Config (optimized for speed) ---
SPREAD_PIPS = 1.2
MIN_BARS = 30
N_FOLDS = 3
SAMPLE_SIZE = 30000  # Use 30K samples for speed (was 225K)
ENSEMBLE_CONFIGS = [
    {"n_estimators": 30, "learning_rate": 0.15, "max_depth": 3, "subsample": 0.7},
    {"n_estimators": 30, "learning_rate": 0.10, "max_depth": 4, "subsample": 0.8},
    {"n_estimators": 50, "learning_rate": 0.08, "max_depth": 3, "subsample": 0.6},
]
HORIZONS = [1, 3, 30]


def load_candles(candle_dir: Path) -> list[Candle]:
    candles = []
    for batch_file in sorted(candle_dir.glob("batch_*.txt")):
        for line in batch_file.read_text().strip().split(";"):
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
    return candles


def build_samples(candles: list[Candle], horizon: int) -> list[tuple[dict, int, float]]:
    feature_dicts = compute_features(candles)
    samples = []
    for i in range(MIN_BARS, len(candles) - horizon):
        feat = feature_dicts[i]
        if feat is None:
            continue
        label = 1 if candles[i + horizon].close > candles[i].close else 0
        samples.append((feat, label, candles[i].timestamp.timestamp()))
    return samples


def purged_walk_forward_split(samples, horizon, n_folds=N_FOLDS):
    total = len(samples)
    if total < 100:
        return []
    fold_size = total // (n_folds + 1)
    embargo_bars = horizon
    splits = []
    for fold in range(n_folds):
        test_start = fold * fold_size + fold * embargo_bars
        test_end = test_start + fold_size
        train_end = max(0, test_start - horizon)
        train_indices = list(range(0, train_end))
        test_indices = list(range(test_start, min(test_end, total)))
        if train_indices and test_indices:
            splits.append((train_indices, test_indices))
    return splits


def train_ensemble(samples, config):
    model = GradientBoostingModel(
        feature_names=list(FEATURE_NAMES),
        feature_version=FEATURE_VERSION,
        **config,
    )
    fit_samples = [(feat, label) for feat, label, _ in samples]
    model.fit(fit_samples)
    return model


def predict_ensemble(models, samples):
    if not models:
        return [0.5] * len(samples)
    preds = [[m.predict_up(feat) for feat, _, _ in samples] for m in models]
    n = len(preds)
    return [sum(p[i] for p in preds) / n for i in range(len(samples))]


def evaluate_predictions(predictions, spread_pips=SPREAD_PIPS):
    if not predictions:
        return {"accuracy": 0.0, "win_rate": 0.0, "net_pips": 0.0,
                "n_total": 0, "n_correct": 0, "profit_factor": 0.0}
    n_correct = sum(1 for p, l in predictions if (p > 0.5) == bool(l))
    n_total = len(predictions)
    win_rate = n_correct / n_total
    net = (n_correct - (n_total - n_correct)) - n_total * spread_pips * 0.1
    pf = (n_correct / (n_total - n_correct)) if (n_total - n_correct) > 0 else float("inf")
    return {
        "accuracy": win_rate,
        "win_rate": win_rate,
        "net_pips": net,
        "n_total": n_total,
        "n_correct": n_correct,
        "profit_factor": pf,
    }


def train_horizon(candles, horizon):
    print(f"\n{'='*60}")
    print(f"HORIZON: {horizon} minutes")
    print(f"{'='*60}")

    samples = build_samples(candles, horizon)
    print(f"Total samples: {len(samples)}")

    # Use subset for speed
    if len(samples) > SAMPLE_SIZE:
        samples = samples[:SAMPLE_SIZE]
        print(f"Using subset: {len(samples)} samples")

    splits = purged_walk_forward_split(samples, horizon)
    print(f"Walk-forward folds: {len(splits)} (purge={horizon}, embargo={horizon})")

    oos_predictions = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        train_samples = [samples[i] for i in train_idx]
        test_samples = [samples[i] for i in test_idx]

        fold_models = []
        for config in ENSEMBLE_CONFIGS:
            fold_models.append(train_ensemble(train_samples, config))

        test_preds = predict_ensemble(fold_models, test_samples)
        test_labels = [label for _, label, _ in test_samples]

        oos_predictions.extend(zip(test_preds, test_labels))
        acc = sum(1 for p, l in zip(test_preds, test_labels) if (p > 0.5) == bool(l)) / len(test_labels)
        print(f"  Fold {fold_idx+1}/{len(splits)}: train={len(train_idx)} test={len(test_idx)} acc={acc:.3f}")

    # Final model
    print(f"\nTraining final model on all {len(samples)} samples...")
    final_models = []
    for config in ENSEMBLE_CONFIGS:
        final_models.append(train_ensemble(samples, config))

    # Evaluate OOS
    metrics = evaluate_predictions(oos_predictions)
    print(f"\nOOS Results ({horizon}min):")
    print(f"  Accuracy: {metrics['accuracy']:.3f}")
    print(f"  Win Rate: {metrics['win_rate']:.3f}")
    print(f"  Net Pips: {metrics['net_pips']:.1f}")
    print(f"  Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"  Samples: {metrics['n_total']}")

    # Save model
    model_dir = PROJECT_ROOT / "var" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_file = model_dir / f"horizon_{horizon}min.json"

    final_model = final_models[0]  # Save first model for simplicity
    model_data = {
        "horizon_minutes": horizon,
        "feature_version": FEATURE_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "feature_labels": list(FEATURE_LABELS),
        "n_estimators": final_model.n_estimators,
        "learning_rate": final_model.learning_rate,
        "max_depth": final_model.max_depth,
        "subsample": final_model.subsample,
        "trees": [
            {
                "root": {
                    "feature_idx": t.root.feature_idx if t.root else -1,
                    "threshold": t.root.threshold if t.root else 0.0,
                    "value": t.root.value if t.root else 0.0,
                } if t.root else None
            }
            for t in final_model.trees
        ],
        "initial_prediction": final_model.initial_prediction,
        "means": final_model.means,
        "scales": final_model.scales,
        "training_samples": final_model.training_samples,
        "oos_accuracy": metrics["accuracy"],
        "oos_win_rate": metrics["win_rate"],
        "oos_net_pips": metrics["net_pips"],
        "oos_profit_factor": metrics["profit_factor"],
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "data_source": "synthetic",
    }
    model_file.write_text(json.dumps(model_data, indent=2))
    print(f"\nSaved: {model_file}")

    return {"horizon": horizon, **metrics}


def main():
    print("=" * 60)
    print("MULTI-HORIZON GBDT TRAINING")
    print("=" * 60)

    candle_dir = PROJECT_ROOT / "var" / "real_candles"
    candles = load_candles(candle_dir)
    print(f"Loaded {len(candles)} candles")
    print(f"Range: {candles[0].timestamp} to {candles[-1].timestamp}")

    results = []
    for horizon in HORIZONS:
        result = train_horizon(candles, horizon)
        results.append(result)

    # Save summary
    summary_file = PROJECT_ROOT / "var" / "models" / "multi_horizon_summary.json"
    summary_file.write_text(json.dumps(results, indent=2))
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for r in results:
        print(f"{r['horizon']}min: acc={r['accuracy']:.3f} pf={r['profit_factor']:.2f}")
    print(f"\nSaved: {summary_file}")


if __name__ == "__main__":
    main()
