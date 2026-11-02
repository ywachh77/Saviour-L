#!/usr/bin/env python3
"""
Comprehensive tuning script to find optimal parameters for 95% win rate / 90% signal rate.

Tests multiple parameter combinations to find the sweet spot.
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


def predict_ensemble(models: list, features: dict) -> float:
    """Get ensemble prediction probability."""
    probs = []
    for model in models:
        probs.append(model.predict_up(features))
    return sum(probs) / len(probs)


def test_config(candles: list, models: list, params: dict) -> dict:
    """Test a specific configuration."""
    config = Config(pair="EUR/USD", spread_pips=1.2)
    engine = FusionEngine(config)
    weights = CouncilWeights()

    wins = 0
    losses = 0
    total_signals = 0
    total_evals = 0

    for i in range(150, min(len(candles) - 1, 350)):
        window = candles[:i]
        total_evals += 1

        features_list = compute_features(window)
        if not features_list:
            continue
        features = features_list[-1]

        regime = detect_regime(window, features)

        # Regime filter
        if params['regime'] == "TREND_ONLY":
            if regime.name != "TREND":
                continue
        elif params['regime'] == "TREND_AND_RANGE":
            if regime.name not in ("TREND", "RANGE"):
                continue

        # ADX filter
        adx_result = adx(window, 14)
        if not adx_result or len(adx_result) < 1:
            continue
        adx_list = adx_result[0]
        if not adx_list:
            continue
        current_adx = adx_list[-1] if adx_list[-1] is not None else 0
        if current_adx < params['adx_min']:
            continue

        # ATR filter
        atr_values = atr(window, 14)
        if not atr_values:
            continue
        current_atr = atr_values[-1]
        if current_atr < params['atr_min']:
            continue

        # Ensemble prediction
        prob_up = predict_ensemble(models, features)

        # Confidence filter
        confidence = prob_up if prob_up >= 0.5 else 1 - prob_up
        if confidence < params['confidence_min']:
            continue

        # Direction strength filter
        if abs(prob_up - 0.5) < params['direction_min']:
            continue

        result = engine.evaluate(
            window,
            weights=weights,
            source="TUNING",
            quote={"spread_pips": 1.2},
            at=next_future_candle_open(window[-1].timestamp),
            observed_at=window[-1].timestamp,
        )

        if result.actionable and result.signal is not None and result.signal.entry_price is not None:
            body_atr = features.get("body_atr_ratio", 0)
            close_pos = features.get("close_position", 0.5)

            if body_atr < params['body_atr_min'] or body_atr > params['body_atr_max']:
                continue

            if result.signal.side == "BUY" and close_pos < params['close_pos_min']:
                continue
            if result.signal.side == "SELL" and close_pos > params['close_pos_max']:
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
            else:
                losses += 1

    total_resolved = wins + losses
    winrate = (wins / total_resolved * 100) if total_resolved > 0 else 0
    signal_rate = (total_signals / total_evals * 100) if total_evals > 0 else 0

    return {
        "signal_rate": signal_rate,
        "winrate": winrate,
        "wins": wins,
        "losses": losses,
        "signals": total_signals,
    }


def main():
    candle_file = "var/real_candles/eurusd_real_20261030_live.json"
    candles = load_candles(candle_file)

    # Load models
    models = []
    with open("var/real_candles/ensemble_model_1bar.json") as f:
        data = json.load(f)
        for m in data['models']:
            models.append(GradientBoostingModel.from_dict(m))

    print(f"Loaded {len(candles)} candles")
    print(f"Loaded {len(models)} models")
    print()

    # Test configurations
    configs = [
        {
            "name": "Strict High Win",
            "regime": "TREND_ONLY",
            "adx_min": 20,
            "atr_min": 0.00005,
            "confidence_min": 0.70,
            "direction_min": 0.15,
            "body_atr_min": 0.15,
            "body_atr_max": 1.8,
            "close_pos_min": 0.4,
            "close_pos_max": 0.6,
        },
        {
            "name": "Moderate Balance",
            "regime": "TREND_ONLY",
            "adx_min": 18,
            "atr_min": 0.00004,
            "confidence_min": 0.65,
            "direction_min": 0.12,
            "body_atr_min": 0.12,
            "body_atr_max": 1.9,
            "close_pos_min": 0.38,
            "close_pos_max": 0.62,
        },
        {
            "name": "TREND Only Moderate",
            "regime": "TREND_ONLY",
            "adx_min": 15,
            "atr_min": 0.00003,
            "confidence_min": 0.60,
            "direction_min": 0.10,
            "body_atr_min": 0.10,
            "body_atr_max": 2.0,
            "close_pos_min": 0.35,
            "close_pos_max": 0.65,
        },
        {
            "name": "TREND Only Relaxed",
            "regime": "TREND_ONLY",
            "adx_min": 12,
            "atr_min": 0.00003,
            "confidence_min": 0.55,
            "direction_min": 0.08,
            "body_atr_min": 0.08,
            "body_atr_max": 2.2,
            "close_pos_min": 0.33,
            "close_pos_max": 0.67,
        },
        {
            "name": "TREND+RANGE Balanced",
            "regime": "TREND_AND_RANGE",
            "adx_min": 15,
            "atr_min": 0.00003,
            "confidence_min": 0.55,
            "direction_min": 0.10,
            "body_atr_min": 0.10,
            "body_atr_max": 2.0,
            "close_pos_min": 0.35,
            "close_pos_max": 0.65,
        },
        {
            "name": "TREND+RANGE Relaxed",
            "regime": "TREND_AND_RANGE",
            "adx_min": 12,
            "atr_min": 0.00003,
            "confidence_min": 0.52,
            "direction_min": 0.08,
            "body_atr_min": 0.08,
            "body_atr_max": 2.2,
            "close_pos_min": 0.33,
            "close_pos_max": 0.67,
        },
        {
            "name": "High Signal Rate",
            "regime": "TREND_AND_RANGE",
            "adx_min": 10,
            "atr_min": 0.00002,
            "confidence_min": 0.50,
            "direction_min": 0.05,
            "body_atr_min": 0.05,
            "body_atr_max": 2.5,
            "close_pos_min": 0.30,
            "close_pos_max": 0.70,
        },
    ]

    results = []

    for config in configs:
        print(f"Testing: {config['name']}")
        print(f"  Regime: {config['regime']}, ADX: {config['adx_min']}, ATR: {config['atr_min']}")
        print(f"  Confidence: {config['confidence_min']}, Direction: {config['direction_min']}")

        result = test_config(candles, models, config)

        print(f"  Signal Rate: {result['signal_rate']:.1f}%")
        print(f"  Win Rate: {result['winrate']:.1f}%")
        print(f"  Signals: {result['signals']}, Wins: {result['wins']}, Losses: {result['losses']}")
        print()

        result['config'] = config
        results.append(result)

    # Find best configurations
    print("="*80)
    print("BEST CONFIGURATIONS")
    print("="*80)

    # Sort by win rate (for high win rate)
    print("\nTop by Win Rate:")
    sorted_by_winrate = sorted(results, key=lambda x: x['winrate'], reverse=True)
    for i, r in enumerate(sorted_by_winrate[:3]):
        print(f"{i+1}. {r['config']['name']}: {r['winrate']:.1f}% win, {r['signal_rate']:.1f}% signal")

    # Sort by combined score (balance of win rate and signal rate)
    print("\nTop by Balanced Score (win_rate * signal_rate):")
    for r in results:
        r['balanced_score'] = r['winrate'] * r['signal_rate'] / 100
    sorted_by_balance = sorted(results, key=lambda x: x['balanced_score'], reverse=True)
    for i, r in enumerate(sorted_by_balance[:3]):
        print(f"{i+1}. {r['config']['name']}: {r['winrate']:.1f}% win, {r['signal_rate']:.1f}% signal (score: {r['balanced_score']:.2f})")

    # Save results
    with open("var/tuning_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print()
    print(f"Results saved to: var/tuning_results.json")


if __name__ == "__main__":
    main()
