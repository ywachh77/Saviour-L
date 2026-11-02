#!/usr/bin/env python3
"""
Test 19/20 win rate on real EUR/USD candles.
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


def main():
    print("="*70)
    print("TESTING 19/20 WIN RATE ON REAL EUR/USD DATA")
    print("="*70)
    print()

    # Load real candles
    candle_file = "var/real_candles/eurusd_real_20261030_live.json"
    candles = load_candles(candle_file)

    print(f"Loaded {len(candles)} real candles")
    print(f"Time range: {candles[0].timestamp} to {candles[-1].timestamp}")
    print()

    # Setup
    config = Config(pair="EUR/USD", spread_pips=1.2)
    engine = FusionEngine(config)
    weights = CouncilWeights()

    # Test all windows
    wins = 0
    losses = 0
    ties = 0
    total_signals = 0
    trade_results = []

    print("Evaluating signals...")

    for i in range(150, len(candles)-1):
        window = candles[:i]

        result = engine.evaluate(
            window,
            weights=weights,
            source="WIN_RATE_TEST",
            quote={"spread_pips": 1.2},
            at=next_future_candle_open(window[-1].timestamp),
            observed_at=window[-1].timestamp,
        )

        if result.actionable and result.signal is not None:
            total_signals += 1

            # Check if entry price is available
            if result.signal.entry_price is None:
                continue

            # Settle at next candle
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
            elif net_pips < 0:
                losses += 1
                outcome = "LOSS"
            else:
                ties += 1
                outcome = "TIE"

            trade_results.append({
                "candle_idx": i,
                "time": str(window[-1].timestamp),
                "side": result.signal.side,
                "entry": entry_price,
                "settlement": settlement_price,
                "net_pips": net_pips,
                "outcome": outcome,
            })

    # Results
    total_resolved = wins + losses + ties
    winrate = (wins / total_resolved * 100) if total_resolved > 0 else 0
    total_pips = sum(t["net_pips"] for t in trade_results)

    print()
    print("="*70)
    print("RESULTS")
    print("="*70)
    print(f"Total signals generated: {total_signals}")
    print(f"Resolved trades: {total_resolved}")
    print(f"Wins: {wins}")
    print(f"Losses: {losses}")
    print(f"Ties: {ties}")
    print(f"Win Rate: {winrate:.1f}%")
    print(f"Total Net Pip: {total_pips:.2f}")
    print()

    # Check for 19/20 batches
    batch_size = 20
    num_batches = total_resolved // batch_size

    print("="*70)
    print("CHECKING FOR 19/20 WIN RATE BATCHES")
    print("="*70)
    print(f"Batch size: {batch_size} trades")
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
        batch_ties = sum(1 for t in batch_trades if t["outcome"] == "TIE")

        achieved_19_20 = (batch_wins >= 19 and (batch_wins + batch_losses) >= 20)

        if achieved_19_20:
            batches_achieving_19_20 += 1

        status = "YES - 19/20" if achieved_19_20 else "NO"
        print(f"Batch {batch_num}: {batch_wins}W/{batch_losses}L/{batch_ties}T - {status}")

    print()
    print(f"Batches achieving 19/20: {batches_achieving_19_20}/{num_batches}")
    print()

    # Summary
    print("="*70)
    print("SUMMARY")
    print("="*70)
    if batches_achieving_19_20 == num_batches:
        print("SUCCESS: All batches achieved 19/20 win rate!")
    elif batches_achieving_19_20 >= num_batches * 0.8:
        print("GOOD: Most batches achieved 19/20 win rate")
    else:
        print("NEEDS IMPROVEMENT: Some batches below 19/20 target")
    print()
    print(f"Total win rate: {winrate:.1f}%")
    print(f"Target: 95% (19/20)")
    print()

    # Save results
    with open("var/win_rate_test_results.json", "w") as f:
        json.dump({
            "summary": {
                "total_signals": total_signals,
                "resolved_trades": total_resolved,
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "winrate": winrate,
                "total_net_pips": total_pips,
                "batches_achieving_19_20": batches_achieving_19_20,
                "total_batches": num_batches,
            },
            "trades": trade_results,
        }, f, indent=2)

    print(f"Results saved to: var/win_rate_test_results.json")


if __name__ == "__main__":
    main()
