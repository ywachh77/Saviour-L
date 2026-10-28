#!/usr/bin/env python3
"""
Minimal Multi-Horizon GBDT Training — Fast version.
Uses minimal samples for quick completion.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fusion.domain import Candle
from fusion.features import compute_features, FEATURE_NAMES, FEATURE_LABELS, FEATURE_VERSION
from fusion.gbdt import GradientBoostingModel

# --- Minimal config ---
SPREAD_PIPS = 1.2
MIN_BARS = 30
SAMPLE_SIZE = 5000  # Minimal for speed
ENSEMBLE_CONFIGS = [
    {"n_estimators": 20, "learning_rate": 0.2, "max_depth": 2, "subsample": 0.8},
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


def simple_split(samples, horizon):
    """Simple train/test split."""
    split_idx = int(len(samples) * 0.7)
    train = samples[:split_idx]
    test = samples[split_idx:]
    return [train, test]


def train_model(samples, config):
    model = GradientBoostingModel(
        feature_names=list(FEATURE_NAMES),
        feature_version=FEATURE_VERSION,
        **config,
    )
    fit_samples = [(feat, label) for feat, label, _ in samples]
    model.fit(fit_samples)
    return model


def predict_model(model, samples):
    return [model.predict_up(feat) for feat, _, _ in samples]


def evaluate(predictions, labels, spread_pips=SPREAD_PIPS):
    if not predictions:
        return {"accuracy": 0.0, "win_rate": 0.0, "net_pips": 0.0, "profit_factor": 0.0}
    n_correct = sum(1 for p, l in zip(predictions, labels) if (p > 0.5) == bool(l))
    n_total = len(predictions)
    win_rate = n_correct / n_total
    net = (n_correct - (n_total - n_correct)) - n_total * spread_pips * 0.1
    pf = (n_correct / (n_total - n_correct)) if (n_total - n_correct) > 0 else float("inf")
    return {
        "accuracy": win_rate,
        "win_rate": win_rate,
        "net_pips": net,
        "profit_factor": pf,
        "n_total": n_total,
        "n_correct": n_correct,
    }


def train_horizon(candles, horizon):
    print(f"\n{'='*60}")
    print(f"HORIZON: {horizon} minutes")
    print(f"{'='*60}")

    samples = build_samples(candles, horizon)
    print(f"Total samples: {len(samples)}")

    if len(samples) > SAMPLE_SIZE:
        samples = samples[:SAMPLE_SIZE]
        print(f"Using subset: {len(samples)} samples")

    splits = simple_split(samples, horizon)
    train_samples, test_samples = splits[0], splits[1]
    print(f"Train: {len(train_samples)}, Test: {len(test_samples)}")

    # Train
    print("Training...")
    t0 = time.time()
    model = train_model(train_samples, ENSEMBLE_CONFIGS[0])
    print(f"Trained in {time.time() - t0:.1f}s")

    # Predict
    test_preds = predict_model(model, test_samples)
    test_labels = [label for _, label, _ in test_samples]

    # Evaluate
    metrics = evaluate(test_preds, test_labels)
    print(f"\nOOS Results ({horizon}min):")
    print(f"  Accuracy: {metrics['accuracy']:.3f}")
    print(f"  Win Rate: {metrics['win_rate']:.3f}")
    print(f"  Net Pips: {metrics['net_pips']:.1f}")
    print(f"  Profit Factor: {metrics['profit_factor']:.2f}")

    # Save model
    model_dir = PROJECT_ROOT / "var" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_file = model_dir / f"horizon_{horizon}min.json"

    model_data = {
        "horizon_minutes": horizon,
        "feature_version": FEATURE_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "feature_labels": list(FEATURE_LABELS),
        "n_estimators": model.n_estimators,
        "learning_rate": model.learning_rate,
        "max_depth": model.max_depth,
        "subsample": model.subsample,
        "trees": [
            {
                "root": {
                    "feature_idx": t.root.feature_idx if t.root else -1,
                    "threshold": t.root.threshold if t.root else 0.0,
                    "value": t.root.value if t.root else 0.0,
                } if t.root else None
            }
            for t in model.trees
        ],
        "initial_prediction": model.initial_prediction,
        "means": model.means,
        "scales": model.scales,
        "training_samples": model.training_samples,
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
    print("MINIMAL MULTI-HORIZON GBDT TRAINING")
    print("=" * 60)

    candle_dir = PROJECT_ROOT / "var" / "real_candles"
    candles = load_candles(candle_dir)
    print(f"Loaded {len(candles)} candles")

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
