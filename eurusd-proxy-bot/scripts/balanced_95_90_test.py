#!/usr/bin/env python3
"""
Balanced approach to achieve 95% win rate with 90% signal rate.

Uses lower confidence threshold to increase signal rate while maintaining quality.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fusion.domain import Candle
from fusion.config import Config
from fusion.engine import FusionEngine
from fusion.council import CouncilWeights
from fusion.timeutil import next_future_candle_open
from fusion.features import compute_features
from fusion.regime import detect_regime
from fusion.indicators import atr, adx
from fusion.gbdt import GradientBoostingModel


def load_candles(file_path: str) -> list[Candle]:
    with open(file_path) as f:
        data = json.load(f)

    candles = []
    for r in data:
        ts = datetime.fromisoformat(r['ts'])
        candles.append(Candle(
            timestamp=ts,
            open=r['o'],
            high=r['h'],
            low=r['l'],
            close=r['c'],
            volume=r.get('v', 0.0)
        ))
    return sorted(candles, key=lambda c: c.timestamp)


def load_ensemble_model(model_path: str) -> list:
    """Load a gradient boosting ensemble model."""
    with open(model_path) as f:
        data = json.load(f)
    models = []
    for m in data['models']:
        models.append(GradientBoostingModel.from_dict(m))
    return models


def predict_ensemble(models: list, features: dict) -> float:
    """Get ensemble prediction probability."""
    probs = []
    for model in models:
        probs.append(model.predict_up(features))
    return sum(probs) / len(probs)


def test_balanced(candles: list[Candle], models: list) -> dict:
    """Test balanced approach for 95% win rate / 90% signal rate."""

    print(f"\n{'='*80}")
    print("BALANCED TEST: 95% WIN RATE / 90% SIGNAL RATE")
    print(f"{'='*80}\n")
    print(f"Candles: {len(candles)}")
    print(f"Models: {len(models)}")
    print()

    config = Config(pair="EUR/USD", spread_pips=1.2)
    engine = FusionEngine(config)
    weights = CouncilWeights()

    wins = 0
    losses = 0
    total_signals = 0
    total_evals = 0
    trade_results = []

    # Test windows
    for i in range(150, min(len(candles) - 1, 350)):
        window = candles[:i]
        total_evals += 1

        # Compute features
        features_list = compute_features(window)
        if not features_list:
            continue
        features = features_list[-1]

        # Detect regime
        regime = detect_regime(window, features)

        # Allow TREND and RANGE regimes
        if regime.name not in ("TREND", "RANGE"):
            continue

        # Check ADX (lower threshold for more signals)
        adx_result = adx(window, 14)
        if not adx_result or len(adx_result) < 1:
            continue
        adx_list = adx_result[0]
        if not adx_list:
            continue
        current_adx = adx_list[-1] if adx_list[-1] is not None else 0
        if current_adx < 12:
            continue

        # Check ATR (lower threshold)
        atr_values = atr(window, 14)
        if not atr_values:
            continue
        current_atr = atr_values[-1]
        if current_atr < 0.00003:
            continue

        # Use ensemble prediction
        prob_up = predict_ensemble(models, features)

        # Confidence filter (lower threshold)
        confidence = prob_up if prob_up >= 0.5 else 1 - prob_up
        if confidence < 0.55:
            continue

        # Direction strength filter (lower threshold)
        if abs(prob_up - 0.5) < 0.08:
            continue

        result = engine.evaluate(
            window,
            weights=weights,
            source="BALANCED_TEST",
            quote={"spread_pips": 1.2},
            at=next_future_candle_open(window[-1].timestamp),
            observed_at=window[-1].timestamp,
        )

        if result.actionable and result.signal is not None and result.signal.entry_price is not None:
            # Quality checks (more permissive)
            body_atr = features.get("body_atr_ratio", 0)
            close_pos = features.get("close_position", 0.5)

            if body_atr < 0.08 or body_atr > 2.2:
                continue

            if result.signal.side == "BUY" and close_pos < 0.33:
                continue
            if result.signal.side == "SELL" and close_pos > 0.67:
                continue

            total_signals += 1

            settlement_idx = i + 1
            if settlement_idx >= len(candles):
                continue

            entry_price = result.signal.entry_price
            settlement_price = candles[settlement_idx].close

            if result.signal.side == "BUY":
                gross_pips = (settlement_price - entry_price) / 0.0001
            else:
                gross_pips = (entry_price - settlement_price) / 0.0001

            net_pips = gross_pips - 1.2

            if net_pips > 0:
                wins += 1
                outcome = "WIN"
            else:
                losses += 1
                outcome = "LOSS"

            trade_results.append({
                "candle_idx": i,
                "time": str(window[-1].timestamp),
                "side": result.signal.side,
                "entry": entry_price,
                "settlement": settlement_price,
                "net_pips": net_pips,
                "outcome": outcome,
                "regime": regime.name,
                "adx": current_adx,
                "atr": current_atr,
                "confidence": confidence,
                "prob_up": prob_up,
            })

    total_resolved = wins + losses
    winrate = (wins / total_resolved * 100) if total_resolved > 0 else 0
    signal_rate = (total_signals / total_evals * 100) if total_evals > 0 else 0

    print("RESULTS:")
    print(f"Evaluations: {total_evals}")
    print(f"Signals generated: {total_signals}")
    print(f"Signal Rate: {signal_rate:.1f}% (target: 90%)")
    print(f"Resolved trades: {total_resolved}")
    print(f"Wins: {wins}")
    print(f"Losses: {losses}")
    print(f"Win Rate: {winrate:.1f}% (target: 95%)")
    print()

    # Check for 19/20 batches
    batch_size = 20
    num_batches = total_resolved // batch_size

    print("CHECKING FOR 19/20 WIN RATE BATCHES:")
    print(f"Batch size: {batch_size}")
    print(f"Available batches: {num_batches}")
    print()

    batches_achieving_19_20 = 0

    for batch_num in range(1, num_batches + 1):
        start_idx = (batch_num - 1) * batch_size
        end_idx = start_idx + batch_size

        if end_idx > len(trade_results):
            break

        batch_trades = trade_results[start_idx:end_idx]
        batch_wins = sum(1 for t in batch_trades if t["outcome"] == "WIN")
        batch_losses = sum(1 for t in batch_trades if t["outcome"] == "LOSS")

        achieved_19_20 = (batch_wins >= 19 and (batch_wins + batch_losses) >= 20)

        if achieved_19_20:
            batches_achieving_19_20 += 1

        status = "YES - 19/20" if achieved_19_20 else "NO"
        print(f"Batch {batch_num}: {batch_wins}W/{batch_losses}L - {status}")

    print()
    print(f"Batches achieving 19/20: {batches_achieving_19_20}/{num_batches}")
    print()

    # Summary
    print("="*80)
    print("SUMMARY")
    print("="*80)
    if signal_rate >= 90:
        print("✅ SIGNAL RATE TARGET ACHIEVED: 90%+")
    else:
        print(f"❌ Signal rate target not reached: {signal_rate:.1f}% (target: 90%)")

    if winrate >= 95:
        print("✅ WIN RATE TARGET ACHIEVED: 95%+")
    else:
        print(f"❌ Win rate target not reached: {winrate:.1f}% (target: 95%)")
    print()

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"var/balanced_test_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump({
            "summary": {
                "total_evals": total_evals,
                "total_signals": total_signals,
                "signal_rate": signal_rate,
                "resolved_trades": total_resolved,
                "wins": wins,
                "losses": losses,
                "winrate": winrate,
                "batches_achieving_19_20": batches_achieving_19_20,
            },
            "trades": trade_results,
        }, f, indent=2)

    print(f"Results saved to: {output_file}")

    return {
        "signal_rate": signal_rate,
        "winrate": winrate,
        "batches_achieving_19_20": batches_achieving_19_20,
    }


def main():
    # Find model
    model_path = None
    for f in Path("var/real_candles").glob("ensemble_model_1bar.json"):
        model_path = str(f)
        break

    if not model_path:
        print("Error: No models found.")
        return

    print(f"Using model: {model_path}")
    models = load_ensemble_model(model_path)

    # Find candles
    candle_file = None
    for f in Path("var/real_candles").glob("eurusd_real_*.json"):
        candle_file = str(f)
        break

    if not candle_file:
        print("Error: No candles found.")
        return

    print(f"Using candles: {candle_file}")
    candles = load_candles(candle_file)

    # Run test
    results = test_balanced(candles, models)

    print(f"\n{'='*80}")
    print("FINAL STATUS")
    print(f"{'='*80}\n")
    print(f"Signal Rate: {results['signal_rate']:.1f}% (target: 90%)")
    print(f"Win Rate: {results['winrate']:.1f}% (target: 95%)")
    print(f"Batches achieving 19/20: {results['batches_achieving_19_20']}")
    print()


if __name__ == "__main__":
    main()
