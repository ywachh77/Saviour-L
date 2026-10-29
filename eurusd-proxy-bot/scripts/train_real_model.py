#!/usr/bin/env python3
"""
Real-Data Training Pipeline
===========================
Trains GBDT ensemble models on REAL candle data from batch_real_*.txt files.
Does NOT generate synthetic data or delete candle files.

Produces:
  - ensemble_model.json      (main model, 5-bar horizon)
  - ensemble_model_1bar.json (1-bar horizon for 1-min expiry)
  - ensemble_model_mr.json   (mean-reversion model)
  - training_report.json     (accuracy metrics)
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
from fusion.gbdt import GradientBoostingModel

CANDLE_DIR = PROJECT_ROOT / "var" / "real_candles"
OUTPUT_JSON = CANDLE_DIR / "ensemble_model.json"
OUTPUT_1BAR = CANDLE_DIR / "ensemble_model_1bar.json"
OUTPUT_MR = CANDLE_DIR / "ensemble_model_mr.json"
REPORT_JSON = CANDLE_DIR / "training_report.json"

HORIZON = 5
HORIZON_1BAR = 1
MIN_BARS = 30
N_MODELS = 5


def load_real_candles():
    """Load candles from batch_real_*.txt files, sorted by timestamp."""
    candles = []
    for batch_file in sorted(CANDLE_DIR.glob("batch_real_*.txt")):
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
    return candles


def compute_features_all(candles):
    """Compute features for all candles."""
    return compute_features(candles)


def build_samples(candles, feat_dicts, horizon=None):
    """Build (features, label) samples.

    Label = 1 if close[i+horizon] > close[i], else 0.
    """
    if horizon is None:
        horizon = HORIZON
    samples = []
    for i in range(len(candles) - horizon):
        if i < MIN_BARS - 1:
            continue
        feat = feat_dicts[i]
        if feat is None or any(v is None for v in feat.values()):
            continue
        label = 1 if candles[i + horizon].close > candles[i].close else 0
        feat_vec = [feat.get(name, 0.0) for name in FEATURE_NAMES]
        samples.append((feat_vec, label))
    return samples


def build_mr_samples(candles, feat_dicts):
    """Build mean-reversion samples.

    Label = 1 if price reverts toward the mean after an excursion.
    """
    samples = []
    lookback = 20
    for i in range(lookback, len(candles) - HORIZON):
        feat = feat_dicts[i]
        if feat is None or any(v is None for v in feat.values()):
            continue
        # Mean of last lookback candles
        ma = sum(c.close for c in candles[i - lookback:i]) / lookback
        # Current deviation from mean
        deviation = (candles[i].close - ma) / ma
        # Label: 1 if price reverts (close goes back toward ma)
        label = 1 if (candles[i + HORIZON].close - ma) * deviation < 0 else 0
        feat_vec = [feat.get(name, 0.0) for name in FEATURE_NAMES]
        samples.append((feat_vec, label))
    return samples


def train_model(samples, config, seed):
    """Train a single GBDT model."""
    import random
    random.seed(seed)

    # Split 80/20 train/test
    n = len(samples)
    split = int(n * 0.8)
    train_set = samples[:split]
    test_set = samples[split:]

    # Build feature dicts from vectors
    def vec_to_dict(vec):
        return {name: val for name, val in zip(FEATURE_NAMES, vec)}

    train_samples = [(vec_to_dict(f), l) for f, l in train_set]
    test_samples = [(vec_to_dict(f), l) for f, l in test_set]

    model = GradientBoostingModel(**config)
    model.fit(train_samples)

    # Evaluate
    correct = 0
    for x_dict, y in test_samples:
        pred = model.predict_up(x_dict)
        if (pred > 0.5) == bool(y):
            correct += 1
    acc = correct / len(test_set) if test_set else 0.0

    print(f"    Train: {len(train_set)} samples, Test: {len(test_set)} samples, Accuracy: {acc:.1%}")
    return model, acc


def save_ensemble(models, samples, configs, accuracies, output_path, horizon, label):
    """Save ensemble model to JSON."""
    model_dicts = [m.to_dict() for m in models]
    data = {
        "version": "real_v1",
        "label": label,
        "horizon": horizon,
        "n_models": len(models),
        "models": model_dicts,
        "configs": configs,
        "accuracy": round(sum(accuracies) / len(accuracies), 4),
        "per_model_accuracy": [round(a, 4) for a in accuracies],
        "feature_names": FEATURE_NAMES,
        "feature_version": FEATURE_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    output_path.write_text(json.dumps(data, indent=2))
    print(f"  Saved {label} to {output_path} ({len(models)} models, {sum(accuracies)/len(accuracies):.1%} avg acc)")


def main():
    print("=" * 60)
    print("REAL-DATA TRAINING PIPELINE")
    print("=" * 60)

    # ─── Step 1: Load real candles ───
    print("\n[1] Loading real candles from batch_real_*.txt...")
    candles = load_real_candles()
    print(f"  Loaded {len(candles)} candles")
    if len(candles) < 100:
        print("  ERROR: Too few candles for training!")
        sys.exit(1)

    # ─── Step 2: Compute features ───
    print("\n[2] Computing features...")
    feat_dicts = compute_features_all(candles)
    print(f"  Computed features for {len(feat_dicts)} candles")

    # ─── Step 3: Build samples ───
    print("\n[3] Building samples...")
    samples = build_samples(candles, feat_dicts, HORIZON)
    pos = sum(l for _, l in samples)
    neg = len(samples) - pos
    print(f"  Main model (horizon={HORIZON}): {len(samples)} samples (pos={pos}, neg={neg})")

    samples_1bar = build_samples(candles, feat_dicts, HORIZON_1BAR)
    pos_1bar = sum(l for _, l in samples_1bar)
    neg_1bar = len(samples_1bar) - pos_1bar
    print(f"  1-bar model (horizon={HORIZON_1BAR}): {len(samples_1bar)} samples (pos={pos_1bar}, neg={neg_1bar})")

    samples_mr = build_mr_samples(candles, feat_dicts)
    pos_mr = sum(l for _, l in samples_mr)
    neg_mr = len(samples_mr) - pos_mr
    print(f"  MR model: {len(samples_mr)} samples (pos={pos_mr}, neg={neg_mr})")

    # ─── Step 4: Train ensemble ───
    print("\n[4] Training ensemble (5 models)...")
    configs = [
        {"n_estimators": 50,  "max_depth": 3, "learning_rate": 0.10, "subsample": 0.7},
        {"n_estimators": 60,  "max_depth": 3, "learning_rate": 0.15, "subsample": 0.8},
        {"n_estimators": 40,  "max_depth": 4, "learning_rate": 0.12, "subsample": 0.6},
        {"n_estimators": 80,  "max_depth": 3, "learning_rate": 0.08, "subsample": 0.9},
        {"n_estimators": 30,  "max_depth": 2, "learning_rate": 0.20, "subsample": 0.5},
    ]

    # Train main model
    print("  Training main model (5-bar horizon)...")
    models_main = []
    accs_main = []
    for i, cfg in enumerate(configs):
        print(f"    Model {i+1}/{len(configs)}: depth={cfg['max_depth']} est={cfg['n_estimators']} lr={cfg['learning_rate']}")
        sys.stdout.flush()
        m, a = train_model(samples, cfg, seed=42 + i)
        models_main.append(m)
        accs_main.append(a)
    save_ensemble(models_main, samples, configs, accs_main, OUTPUT_JSON, HORIZON, "main")

    # Train 1-bar model
    print("  Training 1-bar model (1-bar horizon)...")
    models_1bar = []
    accs_1bar = []
    for i, cfg in enumerate(configs):
        print(f"    Model {i+1}/{len(configs)}: depth={cfg['max_depth']} est={cfg['n_estimators']} lr={cfg['learning_rate']}")
        sys.stdout.flush()
        m, a = train_model(samples_1bar, cfg, seed=42 + i)
        models_1bar.append(m)
        accs_1bar.append(a)
    save_ensemble(models_1bar, samples_1bar, configs, accs_1bar, OUTPUT_1BAR, HORIZON_1BAR, "1bar")

    # Train MR model
    print("  Training mean-reversion model...")
    models_mr = []
    accs_mr = []
    for i, cfg in enumerate(configs):
        print(f"    Model {i+1}/{len(configs)}: depth={cfg['max_depth']} est={cfg['n_estimators']} lr={cfg['learning_rate']}")
        sys.stdout.flush()
        m, a = train_model(samples_mr, cfg, seed=42 + i)
        models_mr.append(m)
        accs_mr.append(a)
    save_ensemble(models_mr, samples_mr, configs, accs_mr, OUTPUT_MR, HORIZON, "mr")

    # ─── Step 5: Ensemble evaluation ───
    print("\n[5] Ensemble evaluation...")

    def vec_to_dict(vec):
        return {name: val for name, val in zip(FEATURE_NAMES, vec)}

    def ensemble_eval(samples_list, models_list):
        correct = 0
        for x, y in samples_list:
            x_dict = vec_to_dict(x) if isinstance(x, list) else x
            pred = sum(m.predict_up(x_dict) for m in models_list) / len(models_list)
            if (pred > 0.5) == bool(y):
                correct += 1
        return correct / len(samples_list) if samples_list else 0.0

    acc_main = ensemble_eval(samples, models_main)
    acc_1bar = ensemble_eval(samples_1bar, models_1bar)
    acc_mr = ensemble_eval(samples_mr, models_mr)

    print(f"  Main ensemble:  {acc_main:.1%}")
    print(f"  1-bar ensemble: {acc_1bar:.1%}")
    print(f"  MR ensemble:    {acc_mr:.1%}")

    # ─── Step 6: Save report ───
    print("\n[6] Saving report...")
    report = {
        "version": "real_v1",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_candles": len(candles),
        "n_samples_main": len(samples),
        "n_samples_1bar": len(samples_1bar),
        "n_samples_mr": len(samples_mr),
        "ensemble_accuracy": {
            "main": round(acc_main, 4),
            "1bar": round(acc_1bar, 4),
            "mr": round(acc_mr, 4),
        },
        "per_model_accuracy": {
            "main": [round(a, 4) for a in accs_main],
            "1bar": [round(a, 4) for a in accs_1bar],
            "mr": [round(a, 4) for a in accs_mr],
        },
        "feature_version": FEATURE_VERSION,
        "n_features": len(FEATURE_NAMES),
    }
    REPORT_JSON.write_text(json.dumps(report, indent=2))
    print(f"  Report saved to {REPORT_JSON}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Main:  {acc_main:.1%}  ({len(samples)} samples)")
    print(f"1-bar: {acc_1bar:.1%}  ({len(samples_1bar)} samples)")
    print(f"MR:    {acc_mr:.1%}  ({len(samples_mr)} samples)")


if __name__ == "__main__":
    main()
