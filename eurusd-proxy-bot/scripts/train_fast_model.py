#!/usr/bin/env python3
"""Train a fast GBDT ensemble on the last 5,000 candles only."""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fusion.domain import Candle
from fusion.features import compute_features
from fusion.gbdt import GradientBoostingModel

CANDLE_DIR = PROJECT_ROOT / "var" / "real_candles"
OUTPUT_JSON = CANDLE_DIR / "ensemble_model.json"
HORIZON = 5
MIN_BARS = 30
N_MODELS = 3
MAX_CANDLES = 5000


def load_candles():
    candles = []
    for batch_file in sorted(CANDLE_DIR.glob("batch_*.txt")):
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
    return candles[-MAX_CANDLES:]


def main():
    print(f"Loading candles (last {MAX_CANDLES} only)...")
    candles = load_candles()
    print(f"  Loaded {len(candles)} candles")

    print("Computing features...")
    feature_dicts = compute_features(candles)

    samples = []
    for i in range(MIN_BARS, len(candles) - HORIZON):
        feat = feature_dicts[i]
        if feat is None:
            continue
        future_close = candles[i + HORIZON].close
        current_close = candles[i].close
        label = 1 if future_close > current_close else 0
        samples.append((feat, label))

    print(f"  Samples: {len(samples)} (pos={sum(l for _,l in samples)}, neg={len(samples)-sum(l for _,l in samples)})")

    # Train N models with different seeds
    models = []
    for i in range(N_MODELS):
        print(f"Training model {i+1}/{N_MODELS}...")
        model = GradientBoostingModel(n_estimators=100, max_depth=3, learning_rate=0.1)
        model.fit(samples, seed=42 + i)
        models.append(model)

    # Evaluate
    correct = 0
    for feat, label in samples:
        pred = sum(m.predict_up(feat) for m in models) / N_MODELS
        if (pred > 0.5) == bool(label):
            correct += 1
    accuracy = correct / len(samples) if samples else 0
    print(f"  Accuracy: {accuracy:.1%}")

    # Save model
    model_data = {
        "n_features": len(models[0].feature_names) if models else 0,
        "feature_names": models[0].feature_names if models else [],
        "horizon": HORIZON,
        "accuracy": accuracy,
        "n_samples": len(samples),
        "models": [],
    }
    for m in models:
        model_data["models"].append(m.to_dict())

    OUTPUT_JSON.write_text(json.dumps(model_data, indent=2))
    print(f"Saved to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
