#!/usr/bin/env python3
"""
Sweep threshold configurations to find the highest win rate on real candles.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fusion.domain import Candle
from fusion.neural.predictor import NeuralPredictor
from fusion.neural.architecture import NeuralConfig


def load_candles(path: str) -> list:
    with open(path) as f:
        rows = json.load(f)
    rows.reverse()
    candles = []
    for r in rows:
        ts = datetime.fromisoformat(r["ts"])
        candles.append(Candle(timestamp=ts, open=r["o"], high=r["h"],
            low=r["l"], close=r["c"], volume=r["v"]))
    return candles


def evaluate_config(predictor, candles, threshold, min_return, max_unc, horizon):
    """Evaluate a config and return win rate stats."""
    wins = 0
    losses = 0
    trades = []
    spread = predictor.spread_pips

    for i in range(121, len(candles) - horizon):
        window = candles[:i]
        p = predictor.predict(window, pair="EUR/USD", at=window[-1].timestamp)

        if p.action == "NO_TRADE":
            continue

        settlement = candles[i + horizon].close
        entry = p.entry_reference_price
        if p.action == "BUY":
            net = (settlement - entry) / 0.0001 - spread
        else:
            net = (entry - settlement) / 0.0001 - spread

        outcome = "WIN" if net > 0 else "LOSS"
        if outcome == "WIN":
            wins += 1
        else:
            losses += 1
        trades.append((p, net, outcome))

    total = wins + losses
    return {
        "threshold": threshold,
        "min_return": min_return,
        "max_unc": max_unc,
        "horizon": horizon,
        "trades": total,
        "wins": wins,
        "losses": losses,
        "win_rate": wins / total if total > 0 else 0,
        "total_pips": sum(t[1] for t in trades),
        "trade_details": trades,
    }


def main():
    candle_path = str(PROJECT_ROOT / "var" / "real_candles" / "eurusd_real_20261030_live.json")
    candles = load_candles(candle_path)
    print(f"Loaded {len(candles)} candles")

    # Load the aggressive model
    ckpt = str(PROJECT_ROOT / "neural" / "checkpoints" / "v1-20261031160956.pt")
    predictor = NeuralPredictor.from_checkpoint(ckpt, spread_pips=1.2)
    print(f"Model: {predictor.model_version}")

    # Sweep configurations
    results = []
    for threshold in [0.30, 0.35, 0.40, 0.45, 0.50]:
        for min_return in [0.5, 1.0, 1.5, 2.0]:
            for max_unc in [0.80, 0.85, 0.90, 0.95]:
                for horizon in [1, 2, 3]:
                    # Modify config temporarily
                    predictor.neural_config.direction_threshold = threshold
                    predictor.neural_config.min_expected_return_pips = min_return
                    predictor.neural_config.max_uncertainty = max_unc
                    r = evaluate_config(predictor, candles, threshold, min_return, max_unc, horizon)
                    results.append(r)

    # Sort by win rate, then by number of trades
    results.sort(key=lambda r: (-r["win_rate"], -r["trades"]))

    print(f"\n{'='*80}")
    print(f"TOP 20 CONFIGURATIONS BY WIN RATE")
    print(f"{'='*80}")
    for r in results[:20]:
        print(f"  thr={r['threshold']:.2f} ret={r['min_return']:.1f} unc={r['max_unc']:.2f} "
              f"hor={r['horizon']} | trades={r['trades']:3d} wins={r['wins']:3d} "
              f"win%={r['win_rate']:.1%} pips={r['total_pips']:.1f}")

    # Show configs with >=20 trades and highest win rate
    print(f"\n{'='*80}")
    print(f"CONFIGS WITH >=20 TRADES, SORTED BY WIN RATE")
    print(f"{'='*80}")
    valid = [r for r in results if r["trades"] >= 20]
    for r in valid[:10]:
        print(f"  thr={r['threshold']:.2f} ret={r['min_return']:.1f} unc={r['max_unc']:.2f} "
              f"hor={r['horizon']} | trades={r['trades']:3d} wins={r['wins']:3d} "
              f"win%={r['win_rate']:.1%} pips={r['total_pips']:.1f}")

    # Save full results
    out_path = PROJECT_ROOT / "var" / "threshold_sweep_results.json"
    with open(out_path, "w") as f:
        json.dump([{k: v for k, v in r.items() if k != "trade_details"} for r in results], f, indent=2, default=str)
    print(f"\nFull results: {out_path}")


if __name__ == "__main__":
    main()
