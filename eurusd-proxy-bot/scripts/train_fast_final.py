#!/usr/bin/env python3
"""
Fast Multi-Horizon Training — 1min, 3min, 30min expiry.
Optimized for speed with XGBoost.
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
import xgboost as xgb
from sklearn.metrics import accuracy_score
import numpy as np

# --- Config ---
SPREAD_PIPS = 1.2
MIN_BARS = 30
SAMPLE_SIZE = 10000
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


def build_samples(candles: list[Candle], horizon: int) -> tuple[np.ndarray, np.ndarray]:
    feature_dicts = compute_features(candles)
    X_list = []
    y_list = []
    for i in range(MIN_BARS, len(candles) - horizon):
        feat = feature_dicts[i]
        if feat is None:
            continue
        x = [float(feat.get(name, 0.0)) for name in FEATURE_NAMES]
        label = 1 if candles[i + horizon].close > candles[i].close else 0
        X_list.append(x)
        y_list.append(label)
    return np.array(X_list), np.array(y_list)


def train_horizon(candles, horizon):
    print(f"\n{'='*60}")
    print(f"HORIZON: {horizon} minutes")
    print(f"{'='*60}")

    X, y = build_samples(candles, horizon)
    print(f"Total samples: {len(X)}")

    if len(X) > SAMPLE_SIZE:
        X = X[:SAMPLE_SIZE]
        y = y[:SAMPLE_SIZE]
        print(f"Using subset: {len(X)} samples")

    # Split train/test
    split_idx = int(len(X) * 0.7)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # Train XGBoost ensemble
    print("\nTraining XGBoost ensemble...")
    models = []

    configs = [
        {"n_estimators": 200, "learning_rate": 0.1, "max_depth": 5, "subsample": 0.8},
        {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 4, "subsample": 0.7},
        {"n_estimators": 200, "learning_rate": 0.15, "max_depth": 6, "subsample": 0.9},
    ]

    for idx, config in enumerate(configs):
        t0 = time.time()
        model = xgb.XGBClassifier(**config, eval_metric='logloss')
        model.fit(X_train, y_train)
        elapsed = time.time() - t0
        print(f"  Model {idx+1}: {elapsed:.1f}s")
        models.append(model)

    # Predict with ensemble
    print("\nEvaluating OOS...")
    probas = [model.predict_proba(X_test)[:, 1] for model in models]
    mean_proba = np.mean(probas, axis=0)
    preds = (mean_proba > 0.5).astype(int)

    # Metrics
    accuracy = accuracy_score(y_test, preds)
    n_correct = int((preds == y_test).sum())
    n_total = len(y_test)
    win_rate = accuracy
    net_pips = (n_correct - (n_total - n_correct)) - n_total * SPREAD_PIPS * 0.1
    pf = (n_correct / (n_total - n_correct)) if (n_total - n_correct) > 0 else float("inf")

    print(f"\nOOS Results ({horizon}min):")
    print(f"  Accuracy: {accuracy:.3f}")
    print(f"  Win Rate: {win_rate:.3f}")
    print(f"  Net Pips: {net_pips:.1f}")
    print(f"  Profit Factor: {pf:.2f}")
    print(f"  Samples: {n_total}")

    # Save model metadata
    model_dir = PROJECT_ROOT / "var" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_file = model_dir / f"horizon_{horizon}min.json"

    model_data = {
        "horizon_minutes": horizon,
        "feature_version": FEATURE_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "feature_labels": list(FEATURE_LABELS),
        "models_trained": len(models),
        "model_types": ["xgb", "xgb", "xgb"],
        "training_samples": len(X_train),
        "oos_accuracy": float(accuracy),
        "oos_win_rate": float(win_rate),
        "oos_net_pips": float(net_pips),
        "oos_profit_factor": float(pf),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "data_source": "synthetic",
    }
    model_file.write_text(json.dumps(model_data, indent=2))
    print(f"\nSaved: {model_file}")

    return {"horizon": horizon, "accuracy": float(accuracy), "win_rate": float(win_rate),
            "net_pips": float(net_pips), "profit_factor": float(pf), "n_total": n_total}


def main():
    print("=" * 60)
    print("FAST MULTI-HORIZON XGBOOST TRAINING")
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
