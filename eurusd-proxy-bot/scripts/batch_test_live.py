#!/usr/bin/env python3
"""
Batch testing system for live EUR/USD data signals.
Tests the trading system on batches of candles to achieve 19/20 win rate.

Usage: python3 scripts/batch_test_live.py --batch-size 20 --num-batches 10
"""

import argparse
import json
import sys
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fusion.domain import Candle
from fusion.engine import FusionEngine
from fusion.council import CouncilWeights
from fusion.regime import detect_regime
from fusion.features import compute_features
from fusion import charter
from fusion.timeutil import next_future_candle_open, floor_minute
from fusion import instruments


def load_live_candles(candle_file: str = None) -> list[Candle]:
    """Load candles from file or fetch from Supabase."""
    if candle_file and Path(candle_file).exists():
        with open(candle_file) as f:
            data = json.load(f)
        # Handle different formats
        if isinstance(data, dict) and "rows" in data:
            rows = data["rows"]
        else:
            rows = data

        candles = []
        for r in rows:
            ts = datetime.fromisoformat(r["ts"])
            candles.append(Candle(
                timestamp=ts,
                open=r["o"],
                high=r["h"],
                low=r["l"],
                close=r["c"],
                volume=r.get("v", 1.0)
            ))
        return sorted(candles, key=lambda c: c.timestamp)
    else:
        # Try to fetch from Supabase
        try:
            from fusion.supabase_client import fetch_candles
            return fetch_candles(pair="EUR/USD", count=5000)
        except Exception as e:
            print(f"Warning: Could not fetch from Supabase: {e}")
            # Fallback to synthetic data
            return generate_synthetic_candles(5000)


def generate_synthetic_candles(count: int) -> list[Candle]:
    """Generate synthetic EUR/USD candles for testing."""
    from scripts.web_dashboard import _build_scaled_paper_series
    from datetime import datetime, timezone
    candles = _build_scaled_paper_series("EURUSD", count=count, end_minute=datetime.now(timezone.utc))
    return candles


def run_batch_test(candles: list[Candle], batch_size: int = 20, num_batches: int = 10,
                   spread_pips: float = 1.2) -> dict:
    """Run batch testing on candle data."""

    print(f"\n{'='*70}")
    print(f"BATCH TESTING - EUR/USD LIVE DATA")
    print(f"{'='*70}")
    print(f"Total candles: {len(candles)}")
    print(f"Batch size: {batch_size}")
    print(f"Number of batches: {num_batches}")
    print(f"Spread: {spread_pips} pips")
    print(f"\n{'='*70}\n")

    # Initialize engine and council
    engine = FusionEngine()
    weights = CouncilWeights()

    # Results tracking
    all_results = []
    batch_results = []
    total_wins = 0
    total_losses = 0

    # Process candles in batches
    warmup = 120  # Minimum candles for features
    start_idx = warmup

    for batch_num in range(1, num_batches + 1):
        batch_start = start_idx + (batch_num - 1) * batch_size
        batch_end = batch_start + batch_size

        if batch_end >= len(candles):
            break

        batch_candles = candles[batch_start:batch_end]
        batch_wins = 0
        batch_losses = 0
        batch_trades = []

        print(f"\n{'─'*70}")
        print(f"BATCH {batch_num}: Candles {batch_start} to {batch_end-1}")
        print(f"{'─'*70}")

        # Evaluate each candle in the batch
        for i in range(batch_size):
            candle_idx = batch_start + i
            if candle_idx >= len(candles) - 1:
                break

            # Get candles up to current point
            window = candles[:candle_idx]

            # Generate signal
            try:
                result = engine.evaluate(
                    window,
                    weights=weights,
                    source="BATCH_TEST",
                    quote={"spread_pips": spread_pips},
                    at=next_future_candle_open(window[-1].timestamp),
                    observed_at=window[-1].timestamp,
                )

                if result.actionable and result.signal is not None:
                    # Settle the signal
                    settlement_idx = candle_idx + 1
                    if settlement_idx >= len(candles):
                        continue

                    entry_price = result.signal.entry_price
                    settlement_price = candles[settlement_idx].close

                    # Calculate outcome
                    if result.signal.side == "BUY":
                        gross_pips = (settlement_price - entry_price) / 0.0001
                    else:
                        gross_pips = (entry_price - settlement_price) / 0.0001

                    net_pips = gross_pips - spread_pips

                    if net_pips > 0:
                        batch_wins += 1
                        total_wins += 1
                        outcome = "WIN"
                    else:
                        batch_losses += 1
                        total_losses += 1
                        outcome = "LOSS"

                    batch_trades.append({
                        "candle_idx": candle_idx,
                        "side": result.signal.side,
                        "entry": entry_price,
                        "settlement": settlement_price,
                        "net_pips": net_pips,
                        "outcome": outcome,
                    })
            except Exception as e:
                # Log errors but continue
                continue

        # Calculate batch statistics
        batch_total = batch_wins + batch_losses
        batch_winrate = (batch_wins / batch_total * 100) if batch_total > 0 else 0
        batch_net_pips = sum(t["net_pips"] for t in batch_trades)

        # Check if batch achieved 19/20
        achieved_19_20 = (batch_wins == 19 and batch_losses == 1) or \
                        (batch_wins == 20 and batch_losses == 0)

        status = "✅ 19/20 ACHIEVED" if achieved_19_20 else "❌ Below target"

        print(f"Trades: {batch_total}")
        print(f"Wins:   {batch_wins}")
        print(f"Losses: {batch_losses}")
        print(f"Win Rate: {batch_winrate:.1f}%")
        print(f"Net Pip:  {batch_net_pips:.2f}")
        print(f"Status:   {status}")

        batch_results.append({
            "batch": batch_num,
            "candles": f"{batch_start}-{batch_end-1}",
            "trades": batch_total,
            "wins": batch_wins,
            "losses": batch_losses,
            "winrate": batch_winrate,
            "net_pips": batch_net_pips,
            "achieved_19_20": achieved_19_20,
        })

        all_results.extend(batch_trades)

    # Summary statistics
    total_trades = total_wins + total_losses
    overall_winrate = (total_wins / total_trades * 100) if total_trades > 0 else 0
    total_net_pips = sum(t["net_pips"] for t in all_results)

    batches_achieving_19_20 = sum(1 for b in batch_results if b["achieved_19_20"])

    print(f"\n{'='*70}")
    print(f"OVERALL SUMMARY")
    print(f"{'='*70}")
    print(f"Total Batches: {len(batch_results)}")
    print(f"Batches Achieving 19/20: {batches_achieving_19_20}")
    print(f"Total Trades: {total_trades}")
    print(f"Total Wins: {total_wins}")
    print(f"Total Losses: {total_losses}")
    print(f"Overall Win Rate: {overall_winrate:.1f}%")
    print(f"Total Net Pip: {total_net_pips:.2f}")
    print(f"\n{'='*70}\n")

    return {
        "batches": batch_results,
        "summary": {
            "total_batches": len(batch_results),
            "batches_achieving_19_20": batches_achieving_19_20,
            "total_trades": total_trades,
            "total_wins": total_wins,
            "total_losses": total_losses,
            "overall_winrate": overall_winrate,
            "total_net_pips": total_net_pips,
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Batch testing for live EUR/USD data")
    parser.add_argument("--batch-size", type=int, default=20, help="Trades per batch")
    parser.add_argument("--num-batches", type=int, default=10, help="Number of batches to test")
    parser.add_argument("--candle-file", type=str, help="Path to candle JSON file")
    parser.add_argument("--spread", type=float, default=1.2, help="Spread in pips")
    parser.add_argument("--output", type=str, help="Output JSON file for results")

    args = parser.parse_args()

    # Load candles
    print("Loading candles...")
    candles = load_live_candles(args.candle_file)

    if len(candles) < args.batch_size + 120:
        print(f"Error: Need at least {args.batch_size + 120} candles")
        sys.exit(1)

    # Run batch test
    results = run_batch_test(
        candles,
        batch_size=args.batch_size,
        num_batches=args.num_batches,
        spread_pips=args.spread
    )

    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
