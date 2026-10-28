#!/usr/bin/env python3
"""
Ensemble GBDT Training v2 — Walk-Forward CV + Multi-Model Ensemble (FAST).

Optimized for pure Python speed:
- 50 estimators per model (was 100-200)
- 3 walk-forward folds (was 5)
- 3 ensemble configs (was 5)

Still delivers: walk-forward CV with purging, ensemble prediction,
feature importance, confidence threshold analysis.
"""

from __future__ import annotations

import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fusion.domain import Candle
from fusion.features import compute_features, FEATURE_NAMES, FEATURE_LABELS
from fusion.gbdt import GradientBoostingModel

SPREAD_PIPS = 0.5
HORIZON = 5
MIN_BARS = 30

ENSEMBLE_CONFIGS = [
    {"n_estimators": 50, "learning_rate": 0.15, "max_depth": 3, "subsample": 0.7},
    {"n_estimators": 50, "learning_rate": 0.1, "max_depth": 4, "subsample": 0.8},
    {"n_estimators": 80, "learning_rate": 0.08, "max_depth": 3, "subsample": 0.6},
]


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
            ts = int(parts[0])
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            candles.append(Candle(
                timestamp=dt,
                open=float(parts[1]),
                high=float(parts[2]),
                low=float(parts[3]),
                close=float(parts[4]),
                volume=0.0,
            ))
    candles.sort(key=lambda c: c.timestamp)
    return candles


def build_samples(candles: list[Candle], horizon: int = HORIZON) -> list[tuple]:
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
    samples: list[tuple], n_folds: int = 3, embargo_pct: float = 0.05,
) -> list[tuple[list[int], list[int]]]:
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


def evaluate_predictions(predictions: list[tuple[float, int]]) -> dict:
    if not predictions:
        return {"accuracy": 0.0, "net_pips": 0.0, "win_rate": 0.0,
                "profit_factor": 0.0, "n_correct": 0, "n_total": 0,
                "gross_win": 0.0, "gross_loss": 0.0}
    n_correct = sum(1 for p, l in predictions if (p > 0.5) == bool(l))
    n_total = len(predictions)
    gross_win = n_correct
    gross_loss = n_total - n_correct
    net = gross_win - gross_loss
    return {
        "accuracy": n_correct / n_total,
        "net_pips": net * 5.0,
        "win_rate": n_correct / n_total,
        "profit_factor": (gross_win / gross_loss) if gross_loss > 0 else float("inf"),
        "n_correct": n_correct,
        "n_total": n_total,
        "gross_win": gross_win,
        "gross_loss": gross_loss,
    }


def feature_importances(model: GradientBoostingModel) -> list[float]:
    counts = [0] * len(model.feature_names)
    for tree in model.trees:
        def count_splits(node):
            if node is None or node.is_leaf:
                return
            if 0 <= node.feature_idx < len(counts):
                counts[node.feature_idx] += 1
            count_splits(node.left)
            count_splits(node.right)
        count_splits(tree.root)
    total = sum(counts)
    if total == 0:
        return [0.0] * len(counts)
    return [c / total for c in counts]


def main():
    candle_dir = Path(__file__).parent.parent / "var" / "real_candles"

    print("Loading candles...")
    candles = load_candles(candle_dir)
    print(f"  Loaded {len(candles)} candles")

    print("Building samples...")
    samples = build_samples(candles, HORIZON)
    print(f"  Built {len(samples)} samples (horizon={HORIZON})")

    if len(samples) < 300:
        print(f"ERROR: Only {len(samples)} samples.")
        return

    splits = purged_walk_forward_split(samples, n_folds=3)
    print(f"\nWalk-forward splits: {len(splits)}")
    for i, (tr, te) in enumerate(splits):
        print(f"  Fold {i}: train={len(tr)} test={len(te)}")

    # Phase 1: Train each ensemble member on all data
    print(f"\n{'='*60}")
    print("PHASE 1: Training ensemble members")
    print(f"{'='*60}")

    ensemble = []
    for config_idx, config in enumerate(ENSEMBLE_CONFIGS):
        t0 = time.time()
        model = GradientBoostingModel(
            n_estimators=config["n_estimators"],
            learning_rate=config["learning_rate"],
            max_depth=config["max_depth"],
            subsample=config["subsample"],
        )
        model.fit([(s[0], s[1]) for s in samples])
        importances = feature_importances(model)
        elapsed = time.time() - t0
        print(f"  Model {config_idx+1}: {config['n_estimators']} trees, "
              f"depth={config['max_depth']}, {elapsed:.1f}s, "
              f"trained on {model.training_samples} samples")
        ensemble.append({"config": config, "model": model, "importances": importances})

    # Phase 2: Walk-forward evaluation with ensemble
    print(f"\n{'='*60}")
    print("PHASE 2: Walk-forward ensemble evaluation")
    print(f"{'='*60}")

    all_preds = []
    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        fold_models = []
        for config in ENSEMBLE_CONFIGS:
            model = GradientBoostingModel(
                n_estimators=config["n_estimators"],
                learning_rate=config["learning_rate"],
                max_depth=config["max_depth"],
                subsample=config["subsample"],
            )
            model.fit([(samples[i][0], samples[i][1]) for i in train_idx])
            fold_models.append(model)

        preds = []
        for idx in test_idx:
            probs = [m.predict_up(samples[idx][0]) for m in fold_models]
            avg_prob = sum(probs) / len(probs)
            preds.append((avg_prob, samples[idx][1]))
            all_preds.append((avg_prob, samples[idx][1]))

        metrics = evaluate_predictions(preds)
        fold_results.append((fold_idx, len(train_idx), len(test_idx), metrics))

        print(f"  Fold {fold_idx}: acc={metrics['accuracy']:.4f} "
              f"net={metrics['net_pips']:.2f} pips "
              f"({metrics['n_correct']}/{metrics['n_total']})")

    # Phase 3: Overall metrics + confidence analysis
    overall = evaluate_predictions(all_preds)

    print(f"\n{'='*60}")
    print("OVERALL ENSEMBLE RESULTS (all folds)")
    print(f"{'='*60}")
    print(f"  Accuracy: {overall['accuracy']:.4f}")
    print(f"  Net pips: {overall['net_pips']:.2f}")
    print(f"  Win rate: {overall['win_rate']:.4f}")
    print(f"  Profit factor: {overall['profit_factor']:.2f}")
    print(f"  Correct/Total: {overall['n_correct']}/{overall['n_total']}")

    print("\nConfidence threshold analysis:")
    for threshold in [0.50, 0.52, 0.55, 0.58, 0.60, 0.65]:
        filtered = [(p, l) for p, l in all_preds if p >= threshold]
        if len(filtered) < 10:
            continue
        m = evaluate_predictions(filtered)
        print(f"  Threshold {threshold:.2f}: acc={m['accuracy']:.4f} "
              f"net={m['net_pips']:.2f} pips "
              f"n={m['n_total']} pf={m['profit_factor']:.2f}")

    # Feature importance
    avg_importances = {}
    for name in FEATURE_NAMES:
        vals = []
        for e in ensemble:
            for fn, imp in zip(FEATURE_NAMES, e["importances"]):
                if fn == name:
                    vals.append(imp)
        avg_importances[name] = sum(vals) / len(vals) if vals else 0.0

    top_features = sorted(avg_importances.items(), key=lambda x: -x[1])[:15]
    print("\nTop 15 Features:")
    for name, imp in top_features:
        label = FEATURE_LABELS.get(name, name)
        print(f"  {name} ({label}): {imp:.4f}")

    # Save
    model_path = candle_dir / "ensemble_model.json"
    model_path.write_text(json.dumps({
        "ensemble": [
            {"config": e["config"], "model": e["model"].to_dict(),
             "importances": e["importances"]}
            for e in ensemble
        ],
        "feature_names": list(FEATURE_NAMES),
        "horizon": HORIZON,
        "spread_pips": SPREAD_PIPS,
    }))
    print(f"\nEnsemble saved to {model_path} ({model_path.stat().st_size} bytes)")

    summary_path = candle_dir / "ensemble_summary.json"
    summary_path.write_text(json.dumps({
        "n_candles": len(candles),
        "n_samples": len(samples),
        "ensemble_size": len(ensemble),
        "overall_accuracy": round(overall["accuracy"], 4),
        "overall_net_pips": round(overall["net_pips"], 2),
        "overall_profit_factor": round(overall["profit_factor"], 2),
        "top_features": [(n, round(v, 4)) for n, v in top_features],
    }, indent=2))
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
