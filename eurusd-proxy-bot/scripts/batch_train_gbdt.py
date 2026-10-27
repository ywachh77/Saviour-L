#!/usr/bin/env python3
"""Batch train GBDT model on real candles pulled from Supabase.

Reads candle batches from var/real_candles/batch_*.txt, computes features,
builds training samples, trains a GradientBoostingModel, evaluates on a
holdout set, and saves the model to the store.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Add the bot directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fusion.domain import Candle
from fusion.features import compute_features
from fusion.gbdt import GradientBoostingModel
from fusion.instruments import for_symbol
from fusion.timeutil import parse_datetime

BATCH_DIR = Path(__file__).parent.parent / "var" / "real_candles"


def load_candles_from_batches() -> list[Candle]:
    """Load all candle batches from files and sort chronologically.

    Supports two formats:
    - Original:  epoch:open:high:low:close  (float prices)
    - Compact:   epoch,open,high,low,close  (int prices * 100000)
    """
    candles: list[Candle] = []
    batch_files = sorted(BATCH_DIR.glob("batch_*.txt"))
    for batch_file in batch_files:
        data = batch_file.read_text().strip()
        if not data:
            continue
        for candle_str in data.split(";"):
            parts = candle_str.split(":")
            if len(parts) < 5:
                # Try compact format (comma-separated with float prices)
                parts = candle_str.split(",")
                if len(parts) < 5:
                    continue
                try:
                    epoch = int(parts[0])
                    open_p = float(parts[1])
                    high_p = float(parts[2])
                    low_p = float(parts[3])
                    close_p = float(parts[4])
                    ts = datetime.fromtimestamp(epoch, tz=timezone.utc)
                    candles.append(Candle(
                        timestamp=ts,
                        open=open_p,
                        high=high_p,
                        low=low_p,
                        close=close_p,
                        volume=0.0,
                    ))
                except (ValueError, IndexError):
                    continue
            else:
                try:
                    epoch = int(parts[0])
                    open_p = float(parts[1])
                    high_p = float(parts[2])
                    low_p = float(parts[3])
                    close_p = float(parts[4])
                    ts = datetime.fromtimestamp(epoch, tz=timezone.utc)
                    candles.append(Candle(
                        timestamp=ts,
                        open=open_p,
                        high=high_p,
                        low=low_p,
                        close=close_p,
                        volume=0.0,
                    ))
                except (ValueError, IndexError):
                    continue
    candles.sort(key=lambda c: c.timestamp)
    return candles


def build_samples(candles: list[Candle], horizon: int = 5) -> list[tuple[dict, int, float]]:
    """Build training samples with labels."""
    instrument = for_symbol("EUR/USD")
    pip = instrument.pip
    feature_rows = compute_features(candles, instrument=instrument)
    samples = []
    for index, features in enumerate(feature_rows):
        if features is None:
            continue
        entry_index = index + 1
        exit_index = index + horizon
        if exit_index >= len(candles):
            break
        entry_price = candles[entry_index].open
        exit_price = candles[exit_index].close
        gross_pips = (exit_price - entry_price) / pip
        if math.isclose(gross_pips, 0.0, abs_tol=1e-12):
            continue
        label = 1 if gross_pips > 0 else 0
        samples.append((features, label, gross_pips))
    return samples


def main():
    print("Loading candles from batches...")
    candles = load_candles_from_batches()
    print(f"Loaded {len(candles)} candles")
    if len(candles) < 100:
        print("Not enough candles to train")
        return

    print(f"Time range: {candles[0].timestamp} to {candles[-1].timestamp}")

    # Build training samples
    print("Building training samples...")
    samples = build_samples(candles, horizon=5)
    print(f"Built {len(samples)} training samples")
    if len(samples) < 80:
        print("Not enough samples to train")
        return

    # Split into train/validation (80/20, chronological)
    split = int(len(samples) * 0.8)
    train = samples[:split]
    valid = samples[split:]
    print(f"Train: {len(train)}, Validation: {len(valid)}")

    # Train GBDT model
    print("Training GradientBoostingModel...")
    t0 = time.time()
    model = GradientBoostingModel(
        n_estimators=30,
        learning_rate=0.1,
        max_depth=3,
        subsample=0.7,
    )
    model.fit([(f, l) for f, l, _ in train], min_samples=80)
    t1 = time.time()
    print(f"Training took {t1-t0:.1f}s, {len(model.trees)} trees")

    # Evaluate on validation set
    print("Evaluating on validation set...")
    correct = 0
    total = 0
    net_pips = 0.0
    spread = 0.5  # typical EUR/USD spread in pips
    for features, label, gross_pips in valid:
        prob = model.predict_up(features)
        pred = 1 if prob > 0.5 else 0
        if pred == label:
            correct += 1
        total += 1
        if abs(prob - 0.5) >= 0.02:
            direction = 1.0 if prob > 0.5 else -1.0
            net_pips += direction * gross_pips - spread

    accuracy = correct / total if total else 0
    print(f"Validation accuracy: {correct}/{total} = {accuracy:.3f}")
    print(f"Validation net pips: {net_pips:.2f}")

    # Evaluate on training set (for comparison)
    train_correct = 0
    for features, label, _ in train:
        prob = model.predict_up(features)
        pred = 1 if prob > 0.5 else 0
        if pred == label:
            train_correct += 1
    train_acc = train_correct / len(train) if train else 0
    print(f"Train accuracy: {train_acc:.3f}")

    # Save model
    model.trained_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    model_dict = model.to_dict()
    output_path = BATCH_DIR / "trained_model.json"
    output_path.write_text(json.dumps(model_dict, indent=2))
    print(f"Model saved to {output_path}")
    print(f"Model size: {output_path.stat().st_size} bytes")

    # Also save a summary
    summary = {
        "candles": len(candles),
        "samples": len(samples),
        "train_samples": len(train),
        "valid_samples": len(valid),
        "trees": len(model.trees),
        "train_accuracy": round(train_acc, 4),
        "valid_accuracy": round(accuracy, 4),
        "valid_net_pips": round(net_pips, 2),
        "trained_at": model.trained_at,
    }
    summary_path = BATCH_DIR / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Summary saved to {summary_path}")
    print(f"\nSUMMARY: {json.dumps(summary)}")


if __name__ == "__main__":
    main()
