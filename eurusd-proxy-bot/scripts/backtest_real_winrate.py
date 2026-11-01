#!/usr/bin/env python3
"""
Walk-forward backtest on real EUR/USD M1 candles using the neural predictor.

For each candle (after warmup), generates a prediction and settles it against
the actual price at the prediction horizon. Counts wins/losses.

Usage: python3 scripts/backtest_real_winrate.py [--checkpoint neural/checkpoints/v1-*.pt] [--horizon 1]
"""

import argparse
import json
import os
import sys
import glob
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fusion.domain import Candle
from fusion.neural.predictor import NeuralPredictor


def load_candles(path: str) -> list[Candle]:
    """Load candles from JSON file (newest-first format)."""
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


def find_best_checkpoint() -> str:
    """Find the best checkpoint by reading report JSON files."""
    ckpt_dir = PROJECT_ROOT / "neural" / "checkpoints"
    best = None
    best_acc = 0.0
    for report_file in sorted(ckpt_dir.glob("v1-*_report.json")):
        try:
            report = json.loads(report_file.read_text())
            acc = report.get("test_metrics", {}).get("direction_acc", 0)
            if acc > best_acc:
                best_acc = acc
                best = str(ckpt_dir / report_file.name.replace("_report.json", ".pt"))
        except (json.JSONDecodeError, KeyError):
            continue
    if best is None:
        # Fallback: latest checkpoint
        ckpts = sorted(ckpt_dir.glob("v1-*.pt"))
        if ckpts:
            best = str(ckpts[-1])
        else:
            sys.exit("No checkpoints found")
    return best


def run_backtest(predictor: NeuralPredictor, candles: list[Candle], horizon: int) -> dict:
    """Walk-forward backtest: predict at each candle, settle at horizon."""
    spread_pips = predictor.spread_pips
    results = []
    wins = 0
    losses = 0
    no_trades = 0
    total_pips = 0.0
    trade_results = []

    min_candles = predictor._min_candles + horizon + 1

    for i in range(min_candles, len(candles)):
        window = candles[:i]  # Use all candles up to current

        # Generate prediction
        prediction = predictor.predict(window, pair="EUR/USD", at=window[-1].timestamp)

        if prediction.action == "NO_TRADE":
            no_trades += 1
            continue

        # Settle at horizon
        settlement_idx = i + horizon
        if settlement_idx >= len(candles):
            break

        settlement_price = candles[settlement_idx].close
        outcome = predictor.record_outcome(prediction, settlement_price)

        entry = prediction.entry_reference_price
        if prediction.action == "BUY":
            net_pips = (settlement_price - entry) / 0.0001 - spread_pips
        else:
            net_pips = (entry - settlement_price) / 0.0001 - spread_pips

        result = {
            "index": i,
            "time": window[-1].timestamp.isoformat(),
            "action": prediction.action,
            "entry": entry,
            "settlement": settlement_price,
            "net_pips": round(net_pips, 2),
            "outcome": outcome,
            "buy_p": round(prediction.direction_probs.get("buy", 0), 3),
            "sell_p": round(prediction.direction_probs.get("sell", 0), 3),
            "no_trade_p": round(prediction.direction_probs.get("no_trade", 0), 3),
            "expected_return": round(prediction.expected_return_pips, 2),
            "uncertainty": round(prediction.uncertainty, 3),
            "regime": prediction.regime,
        }
        trade_results.append(result)

        if outcome == "WIN":
            wins += 1
            total_pips += net_pips
        else:
            losses += 1
            total_pips += net_pips

    total_trades = wins + losses
    win_rate = wins / total_trades if total_trades > 0 else 0.0
    avg_pips = total_pips / total_trades if total_trades > 0 else 0.0

    return {
        "total_candles": len(candles),
        "min_candles": min_candles,
        "horizon_minutes": horizon,
        "spread_pips": spread_pips,
        "no_trades": no_trades,
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate, 4),
        "total_pips": round(total_pips, 2),
        "avg_pips_per_trade": round(avg_pips, 2),
        "trade_results": trade_results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candles", default="var/real_candles/eurusd_real_20261030_live.json")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--horizon", type=int, default=1)
    args = parser.parse_args()

    # Load candles
    candle_path = args.candles if os.path.isabs(args.candles) else str(PROJECT_ROOT / args.candles)
    print(f"Loading candles from {candle_path}")
    candles = load_candles(candle_path)
    print(f"Loaded {len(candles)} candles")
    print(f"  First: {candles[0].timestamp} close={candles[0].close}")
    print(f"  Last:  {candles[-1].timestamp} close={candles[-1].close}")

    # Find checkpoint
    ckpt_path = args.checkpoint or find_best_checkpoint()
    if not os.path.isabs(ckpt_path):
        ckpt_path = str(PROJECT_ROOT / ckpt_path)
    print(f"\nLoading checkpoint: {ckpt_path}")
    predictor = NeuralPredictor.from_checkpoint(ckpt_path, spread_pips=1.2)
    print(f"Model version: {predictor.model_version}")
    print(f"Min candles: {predictor._min_candles}")
    print(f"Input window: {predictor.neural_config.input_window}")

    # Run backtest
    print(f"\nRunning walk-forward backtest (horizon={args.horizon} min)...")
    results = run_backtest(predictor, candles, args.horizon)

    # Print summary
    print(f"\n{'='*60}")
    print(f"WALK-FORWARD BACKTEST RESULTS")
    print(f"{'='*60}")
    print(f"Candles: {results['total_candles']}")
    print(f"No-trades: {results['no_trades']}")
    print(f"Total trades: {results['total_trades']}")
    print(f"Wins: {results['wins']}")
    print(f"Losses: {results['losses']}")
    print(f"Win rate: {results['win_rate']:.1%}")
    print(f"Total pips: {results['total_pips']:.2f}")
    print(f"Avg pips/trade: {results['avg_pips_per_trade']:.2f}")

    if results["total_trades"] >= 20:
        target = 19
        actual = results["wins"]
        ratio = actual / results["total_trades"] if results["total_trades"] > 0 else 0
        print(f"\nTarget: 19/20 = 95% win rate")
        print(f"Actual: {actual}/{results['total_trades']} = {ratio:.1%} win rate")
        if ratio >= 0.95:
            print("✓ MEETS 19/20 TARGET")
        elif ratio >= 0.90:
            print("~ Close to 19/20 target (90-95%)")
        else:
            print("✗ Below 19/20 target (<90%)")

    print(f"\n{'='*60}")
    print(f"LAST 10 TRADES:")
    print(f"{'='*60}")
    for t in results["trade_results"][-10:]:
        print(f"  {t['time'][:19]} | {t['action']:4s} | "
              f"buy={t['buy_p']:.2f} sell={t['sell_p']:.2f} | "
              f"ret={t['expected_return']:.1f}p unc={t['uncertainty']:.2f} | "
              f"net={t['net_pips']:+.2f}p {t['outcome']}")

    # Write full results
    log_path = PROJECT_ROOT / "var" / "real_backtest_results.json"
    with open(log_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nFull results: {log_path}")


if __name__ == "__main__":
    main()
