#!/usr/bin/env python3
"""Train a quick GBDT ensemble on local candles and save as ensemble_model.json."""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fusion.domain import Candle
from fusion.features import compute_features, FEATURE_NAMES, FEATURE_LABELS
from fusion.gbdt import GradientBoostingModel
import numpy as np

CANDLE_DIR = PROJECT_ROOT / "var" / "real_candles"
OUTPUT_JSON = CANDLE_DIR / "ensemble_model.json"
HORIZON = 5
MIN_BARS = 30
N_MODELS = 3


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
    return candles


def main():
    print("Loading candles...")
    candles = load_candles()
    print(f"  Loaded {len(candles)} candles")

    print("Computing features...")
    feature_dicts = compute_features(candles)

    samples = []
    for i in range(MIN_BARS, len(candles) - HORIZON):
        feat = feature_dicts[i]
        if feat is None:
            continue
        label = 1 if candles[i + HORIZON].close > candles[i].close else 0
        samples.append((feat, label))

    print(f"  Built {len(samples)} samples")

    split_idx = int(len(samples) * 0.7)
    train_samples = samples[:split_idx]
    test_samples = samples[split_idx:]

    # Extract test features and labels for evaluation
    test_feats = [s[0] for s in test_samples]
    test_labels = np.array([s[1] for s in test_samples])

    models = []
    for i in range(N_MODELS):
        print(f"  Training model {i+1}/{N_MODELS}...")
        model = GradientBoostingModel(n_estimators=100, max_depth=4, learning_rate=0.1)
        model.fit(train_samples)

        # Evaluate
        preds = np.array([model.predict_up(f) > 0.5 for f in test_feats])
        acc = np.mean(preds == test_labels)
        print(f"    Test accuracy: {acc:.1%}")

        models.append(model)

    # Save ensemble
    ensemble_data = {
        "ensemble": [{"model": m.to_dict()} for m in models],
        "horizon": HORIZON,
        "feature_names": FEATURE_NAMES,
        "feature_labels": FEATURE_LABELS,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candle_count": len(candles),
    }

    OUTPUT_JSON.write_text(json.dumps(ensemble_data))
    print(f"\nSaved ensemble to {OUTPUT_JSON} ({OUTPUT_JSON.stat().st_size} bytes)")

    # Ensemble accuracy
    probs = np.mean([m.predict_up(f) for m in models for f in test_feats], axis=0).reshape(-1, 1)
    # Actually compute per-sample
    ens_probs = []
    for f in test_feats:
        p = sum(m.predict_up(f) for m in models) / len(models)
        ens_probs.append(p)
    ens_probs = np.array(ens_probs)
    ens_preds = (ens_probs > 0.5).astype(int)
    ens_acc = np.mean(ens_preds == test_labels)
    print(f"Ensemble test accuracy: {ens_acc:.1%}")


if __name__ == "__main__":
    main()
