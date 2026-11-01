#!/usr/bin/env python3
"""
Train neural model with aggressive settings for high win-rate trading.

Uses lower thresholds, trading-favorable class weights, and more epochs
to produce enough directional signals on real candles.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fusion.domain import Candle
from fusion.neural.trainer import NeuralTrainer, TrainingConfig
from fusion.neural.architecture import NeuralConfig, TemporalTransformer


def load_candles(path: str) -> list:
    with open(path) as f:
        rows = json.load(f)
    rows.reverse()  # Oldest-first
    candles = []
    for r in rows:
        ts = datetime.fromisoformat(r["ts"])
        candles.append(Candle(
            timestamp=ts, open=r["o"], high=r["h"],
            low=r["l"], close=r["c"], volume=r["v"],
        ))
    return candles


def main():
    candle_path = str(PROJECT_ROOT / "var" / "real_candles" / "eurusd_real_20261030_live.json")
    print(f"Loading candles from {candle_path}")
    candles = load_candles(candle_path)
    print(f"Loaded {len(candles)} candles")

    # Aggressive training config — favor trading over no_trade
    config = TrainingConfig(
        epochs=50,
        batch_size=32,
        learning_rate=0.0005,
        patience=10,
        direction_class_weights=[2.0, 2.0, 1.0],  # Favor trading
        w_direction=1.5,
        w_return=0.5,
        w_uncertainty=0.1,  # Low uncertainty penalty — trade more
        w_regime=0.1,
        shuffle_train=True,
        seed=42,
    )

    # Aggressive neural config — lower thresholds for more trades
    neural_config = NeuralConfig(
        direction_threshold=0.35,
        min_expected_return_pips=0.5,
        max_uncertainty=0.95,
        confirmation_required=0,
        block_regimes=(),  # Don't block any regimes
        kill_switch_threshold=3,
        kill_switch_cooldown_candles=5,
    )

    print(f"\nTraining config:")
    print(f"  Epochs: {config.epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  LR: {config.learning_rate}")
    print(f"  Class weights: {config.direction_class_weights}")
    print(f"  w_direction: {config.w_direction}, w_return: {config.w_return}")
    print(f"\nNeural config:")
    print(f"  Direction threshold: {neural_config.direction_threshold}")
    print(f"  Min expected return: {neural_config.min_expected_return_pips} pips")
    print(f"  Max uncertainty: {neural_config.max_uncertainty}")
    print(f"  Block regimes: {neural_config.block_regimes}")

    trainer = NeuralTrainer(config, neural_config)
    print("\nTraining...")
    report = trainer.train(candles)

    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Model version: {report.model_version}")
    print(f"Best epoch: {report.best_epoch}")
    print(f"Best val loss: {report.best_val_loss:.4f}")
    print(f"Best val direction acc: {report.best_val_direction_acc:.2%}")
    print(f"Test loss: {report.test_loss:.4f}")
    print(f"Test direction acc: {report.test_direction_acc:.2%}")
    print(f"Test return MAE: {report.test_return_mae:.4f}")
    print(f"\nCheckpoint: neural/checkpoints/{report.model_version}.pt")


if __name__ == "__main__":
    main()
