#!/usr/bin/env python3
"""
Multi-Horizon GBDT Training — 1min, 3min, 30min expiry.

Trains gradient boosting ensembles for all three horizons using walk-forward
CV with purging + embargo. Reports OOS metrics for each horizon. Saves
models to var/models/horizon_1min.json, horizon_3min.json, horizon_30min.json.

Usage:
    python3 scripts/train_multi_horizon.py
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

# --- Config ---
SPREAD_PIPS = 1.2  # Realistic EUR/USD spread
MIN_BARS = 30      # Minimum bars for feature computation
N_FOLDS = 4        # Walk-forward folds
ENSEMBLE_CONFIGS = [
    {"n_estimators": 50, "learning_rate": 0.15, "max_depth": 3, "subsample": 0.7},
    {"n_estimators": 50, "learning_rate": 0.10, "max_depth": 4, "subsample": 0.8},
    {"n_estimators": 80, "learning_rate": 0.08, "max_depth": 3, "subsample": 0.6},
]
HORIZONS = [1, 3, 30]  # Train for all three expiry options


def load_candles(candle_dir: Path) -> list[Candle]:
    """Load candles from var/real_candles/batch_*.txt files."""
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
    """Build (features, label, timestamp) samples for a given horizon."""
    feature_dicts = compute_features(candles)
    samples = []
    for i in range(MIN_BARS, len(candles) - horizon):
        feat = feature_dicts[i]
        if feat is None:
            continue
        label = 1 if candles[i + horizon].close > candles[i].close else 0
        samples.append((feat, label, candles[i].timestamp.timestamp()))
    return samples


def purged_walk_forward_split(
    samples: list[tuple], horizon: int, n_folds: int = N_FOLDS,
) -> list[tuple[list[int], list[int]]]:
    """Expanding-window walk-forward CV with purging + embargo.

    Purge: drop `horizon` bars from the end of training before each test fold
    (prevents label overlap). Embargo: leave a gap between consecutive test
    folds to prevent autocorrelation leakage.
    """
    total = len(samples)
    if total < n_folds * 50:
        return []
    fold_size = total // (n_folds + 1)
    embargo_bars = horizon  # Gap between test folds
    splits = []
    for fold in range(n_folds):
        test_start = fold * fold_size + fold * embargo_bars
        test_end = test_start + fold_size
        train_end = max(0, test_start - horizon)  # Purge
        train_indices = list(range(0, train_end))
        test_indices = list(range(test_start, min(test_end, total)))
        if train_indices and test_indices:
            splits.append((train_indices, test_indices))
    return splits


def train_ensemble(samples: list[tuple], config: dict) -> GradientBoostingModel:
    """Train a single GBDT model."""
    model = GradientBoostingModel(
        feature_names=list(FEATURE_NAMES),
        feature_version=FEATURE_VERSION,
        **config,
    )
    # fit expects (features_dict, label) pairs
    fit_samples = [(feat, label) for feat, label, _ in samples]
    model.fit(fit_samples)
    return model


def predict_ensemble(models: list[GradientBoostingModel], samples: list[tuple]) -> list[float]:
    """Average predictions from ensemble models."""
    if not models:
        return [0.5] * len(samples)
    preds = [[m.predict_up(feat) for feat, _, _ in samples] for m in models]
    n = len(preds)
    return [sum(p[i] for p in preds) / n for i in range(len(samples))]


def evaluate_predictions(predictions: list[tuple[float, int]], spread_pips: float = SPREAD_PIPS) -> dict:
    """Evaluate predictions: accuracy, win rate, net pips (after spread)."""
    if not predictions:
        return {"accuracy": 0.0, "win_rate": 0.0, "net_pips": 0.0,
                "n_total": 0, "n_correct": 0, "profit_factor": 0.0}
    n_correct = sum(1 for p, l in predictions if (p > 0.5) == bool(l))
    n_total = len(predictions)
    win_rate = n_correct / n_total
    # Net pips: each correct = +1, each wrong = -1, minus spread cost
    net = (n_correct - (n_total - n_correct)) - n_total * spread_pips * 0.1  # spread in pips
    pf = (n_correct / (n_total - n_correct)) if (n_total - n_correct) > 0 else float("inf")
    return {
        "accuracy": win_rate,
        "win_rate": win_rate,
        "net_pips": net,
        "n_total": n_total,
        "n_correct": n_correct,
        "profit_factor": pf,
    }


def train_horizon(candles: list[Candle], horizon: int) -> dict:
    """Train and evaluate a model for a specific horizon."""
    print(f"\n{'='*60}")
    print(f"HORIZON: {horizon} minutes")
    print(f"{'='*60}")

    samples = build_samples(candles, horizon)
    print(f"Total samples: {len(samples)}")
    if len(samples) < 100:
        print(f"  SKIP: only {len(samples)} samples, need at least 100")
        return {"horizon": horizon, "skipped": True, "reason": "insufficient samples"}

    splits = purged_walk_forward_split(samples, horizon)
    if not splits:
        print(f"  SKIP: not enough data for {N_FOLDS} folds")
        return {"horizon": horizon, "skipped": True, "reason": "insufficient folds"}

    print(f"Walk-forward folds: {len(splits)} (purge={horizon}, embargo={horizon})")

    # Train ensemble on all training data for final model
    print(f"\nTraining ensemble ({len(ENSEMBLE_CONFIGS)} models)...")
    t0 = time.time()
    all_train_X = []
    all_train_y = []
    oos_predictions = []
    oos_labels = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        train_samples = [samples[i] for i in train_idx]
        test_samples = [samples[i] for i in test_idx]

        # Train ensemble on this fold's training data
        fold_models = []
        for config in ENSEMBLE_CONFIGS:
            fold_models.append(train_ensemble(train_samples, config))

        # Predict on test fold
        test_preds = predict_ensemble(fold_models, test_samples)
        test_labels = [label for _, label, _ in test_samples]

        oos_predictions.extend(zip(test_preds, test_labels))
        print(f"  Fold {fold_idx+1}/{len(splits)}: train={len(train_idx)} test={len(test_idx)} "
              f"acc={sum(1 for p,l in zip(test_preds,test_labels) if (p>0.5)==bool(l))/len(test_labels):.3f}")

    # Final model: train on all data
    print(f"\nTraining final model on all {len(samples)} samples...")
    final_models = []
    for config in ENSEMBLE_CONFIGS:
        final_models.append(train_ensemble(samples, config))

    # Save final model
    model_dir = PROJECT_ROOT / "var" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"horizon_{horizon}min.json"
    final_models[0].save(str(model_path))
    print(f"  Saved: {model_path}")

    # Evaluate OOS
    metrics = evaluate_predictions(oos_predictions, SPREAD_PIPS)
    print(f"\nOUT-OF-SAMPLE RESULTS (horizon={horizon}min):")
    print(f"  Samples:     {metrics['n_total']}")
    print(f"  Accuracy:    {metrics['accuracy']:.4f} ({metrics['n_correct']}/{metrics['n_total']})")
    print(f"  Win rate:    {metrics['win_rate']:.4f}")
    print(f"  Net pips:    {metrics['net_pips']:.2f}")
    print(f"  Profit factor: {metrics['profit_factor']:.3f}")

    return {
        "horizon": horizon,
        "skipped": False,
        "n_samples": len(samples),
        "n_folds": len(splits),
        "oos_accuracy": metrics["accuracy"],
        "oos_win_rate": metrics["win_rate"],
        "oos_net_pips": metrics["net_pips"],
        "oos_profit_factor": metrics["profit_factor"],
        "spread_pips": SPREAD_PIPS,
        "model_path": str(model_path),
        "feature_version": FEATURE_VERSION,
    }


def main():
    candle_dir = PROJECT_ROOT / "var" / "real_candles"
    if not candle_dir.exists():
        print(f"ERROR: {candle_dir} does not exist")
        sys.exit(1)

    print("Loading candles...")
    candles = load_candles(candle_dir)
    print(f"Loaded {len(candles)} candles from {candle_dir}")
    if len(candles) < 100:
        print(f"ERROR: only {len(candles)} candles, need at least 100")
        sys.exit(1)

    if candles:
        print(f"Range: {candles[0].timestamp} to {candles[-1].timestamp}")

    results = []
    for horizon in HORIZONS:
        result = train_horizon(candles, horizon)
        results.append(result)

    # Save summary
    summary_path = PROJECT_ROOT / "var" / "models" / "multi_horizon_summary.json"
    summary_path.write_text(json.dumps(results, indent=2))
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for r in results:
        if r["skipped"]:
            print(f"  {r['horizon']}min: SKIPPED ({r['reason']})")
        else:
            print(f"  {r['horizon']}min: acc={r['oos_accuracy']:.4f} "
                  f"net={r['oos_net_pips']:.2f}pips "
                  f"PF={r['oos_profit_factor']:.3f} "
                  f"({r['n_samples']} samples, {r['n_folds']} folds)")
    print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    main()
