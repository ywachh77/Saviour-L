#!/usr/bin/env python3
"""
Batch testing with REAL EUR/USD data from Supabase.
Tests 19/20 win rate on actual market candles.

Requires:
- SUPABASE_URL
- SUPABASE_SERVICE_ROLE_KEY
- Real candle data in var/real_candles/EURUSD/
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fusion.config import Config
from fusion.engine import FusionEngine
from fusion.council import CouncilWeights
from fusion.domain import Candle
from fusion.timeutil import next_future_candle_open


def load_real_candles(candle_dir: str = "var/real_candles/EURUSD") -> list[Candle]:
    """Load real candles from local files."""
    candle_path = Path(candle_dir)
    
    # Look for JSON files
    for pattern in ["*.json", "candles_*.json", "eurusd_*.json"]:
        files = list(candle_path.glob(pattern))
        if files:
            for file in files:
                try:
                    with open(file) as f:
                        data = json.load(f)
                    
                    if isinstance(data, dict) and "rows" in data:
                        rows = data["rows"]
                    elif isinstance(data, list):
                        rows = data
                    else:
                        rows = [data]

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
                except Exception as e:
                    print(f"Warning: Could not load {file}: {e}")
                    continue
    
    return []


def fetch_from_supabase() -> list[Candle]:
    """Fetch real candles from Supabase."""
    try:
        import requests
        import os

        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        project_ref = os.environ.get("SUPABASE_PROJECT_REF", "idudlcxkihqnvecwqqoj")

        if not supabase_url or not supabase_key:
            print("Error: SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY required")
            return []

        url = f"{supabase_url}/rest/v1/candles_m1"
        headers = {
            "apikey": supabase_key,
            "Authorization": f"Bearer {supabase_key}",
            "Content-Type": "application/json"
        }
        params = {
            "select": "ts,o,h,l,c,v",
            "symbol": "EUR/USD",
            "order": "ts.asc",
            "limit": 5000
        }

        response = requests.get(url, headers=headers, params=params, timeout=30)
        if response.status_code != 200:
            print(f"Supabase error: {response.status_code}")
            return []

        rows = response.json()
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
        return candles

    except Exception as e:
        print(f"Supabase fetch failed: {e}")
        return []


def run_batch_test(candles: list[Candle], batch_size: int = 20, num_batches: int = 5) -> dict:
    """Run batch testing on real candles."""

    print(f"\n{'='*70}")
    print(f"BATCH TESTING - REAL EUR/USD DATA")
    print(f"{'='*70}")
    print(f"Total candles: {len(candles)}")
    print(f"Batch size: {batch_size} trades")
    print(f"Number of batches: {num_batches}")
    print(f"Target: 19/20 win rate per batch (95%)")
    print(f"\n{'='*70}\n")

    if len(candles) < batch_size + 120:
        print(f"Error: Need at least {batch_size + 120} candles")
        return {}

    # Setup
    config = Config(pair="EUR/USD", spread_pips=1.2)
    engine = FusionEngine(config)
    weights = CouncilWeights()

    # Results
    all_batch_results = []
    total_trades = 0
    total_wins = 0
    batches_achieving_19_20 = 0

    # Process in batches
    warmup = 120
    start_idx = warmup

    for batch_num in range(1, num_batches + 1):
        batch_candles = candles[start_idx:start_idx + batch_size]
        if len(batch_candles) < batch_size:
            break

        print(f"\n{'─'*70}")
        print(f"BATCH {batch_num} - Candles {start_idx} to {start_idx + batch_size - 1}")
        print(f"Time range: {batch_candles[0].timestamp} to {batch_candles[-1].timestamp}")
        print(f"{'─'*70}")

        batch_wins = 0
        batch_losses = 0
        batch_trades = []

        for i in range(len(batch_candles)):
            candle_idx = start_idx + i
            if candle_idx >= len(candles) - 1:
                break

            window = candles[:candle_idx]

            try:
                result = engine.evaluate(
                    window,
                    weights=weights,
                    source="REAL_DATA",
                    quote={"spread_pips": 1.2},
                    at=next_future_candle_open(window[-1].timestamp),
                    observed_at=window[-1].timestamp,
                )

                if result.actionable and result.signal is not None:
                    settlement_idx = candle_idx + 1
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
                        batch_wins += 1
                        outcome = "WIN"
                    else:
                        batch_losses += 1
                        outcome = "LOSS"

                    batch_trades.append({
                        "candle_idx": candle_idx,
                        "time": str(window[-1].timestamp),
                        "side": result.signal.side,
                        "entry": entry_price,
                        "settlement": settlement_price,
                        "net_pips": net_pips,
                        "outcome": outcome,
                    })
            except Exception as e:
                continue

        # Stats
        batch_total = batch_wins + batch_losses
        batch_winrate = (batch_wins / batch_total * 100) if batch_total > 0 else 0
        batch_net_pips = sum(t["net_pips"] for t in batch_trades)

        achieved_19_20 = batch_wins >= 19 and batch_total >= 20

        print(f"Trades:     {batch_total}")
        print(f"Wins:       {batch_wins}")
        print(f"Losses:     {batch_losses}")
        print(f"Win Rate:   {batch_winrate:.1f}%")
        print(f"Net Pip:    {batch_net_pips:.2f}")

        if achieved_19_20:
            print(f"Status:     ✅ 19/20 ACHIEVED")
            batches_achieving_19_20 += 1
        else:
            print(f"Status:     ❌ Below target")

        all_batch_results.append({
            "batch": batch_num,
            "time_range": f"{batch_candles[0].timestamp} to {batch_candles[-1].timestamp}",
            "trades": batch_total,
            "wins": batch_wins,
            "losses": batch_losses,
            "winrate": batch_winrate,
            "net_pips": batch_net_pips,
            "achieved_19_20": achieved_19_20,
        })

        total_trades += batch_total
        total_wins += batch_wins
        start_idx += batch_size

    # Summary
    overall_winrate = (total_wins / total_trades * 100) if total_trades > 0 else 0

    print(f"\n{'='*70}")
    print(f"OVERALL SUMMARY")
    print(f"{'='*70}")
    print(f"Total Batches:              {len(all_batch_results)}")
    print(f"Batches Achieving 19/20:   {batches_achieving_19_20}")
    print(f"Total Trades:               {total_trades}")
    print(f"Total Wins:                 {total_wins}")
    print(f"Overall Win Rate:           {overall_winrate:.1f}%")

    if batches_achieving_19_20 == len(all_batch_results):
        print(f"\n🎉 ALL BATCHES ACHIEVED 19/20 WIN RATE!")
    else:
        print(f"\n⚠️  Some batches below 19/20 target")
    print(f"\n{'='*70}\n")

    return {
        "batches": all_batch_results,
        "summary": {
            "total_batches": len(all_batch_results),
            "batches_achieving_19_20": batches_achieving_19_20,
            "total_trades": total_trades,
            "total_wins": total_wins,
            "overall_winrate": overall_winrate,
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Batch testing with real EUR/USD data")
    parser.add_argument("--batch-size", type=int, default=20, help="Trades per batch")
    parser.add_argument("--num-batches", type=int, default=5, help="Number of batches")
    parser.add_argument("--candle-dir", type=str, default="var/real_candles/EURUSD", help="Candle directory")
    parser.add_argument("--fetch", action="store_true", help="Fetch from Supabase first")
    parser.add_argument("--output", type=str, help="Output JSON file")

    args = parser.parse_args()

    # Load candles
    candles = []

    if args.fetch:
        print("Fetching candles from Supabase...")
        candles = fetch_from_supabase()
        if candles:
            # Save locally
            candle_dir = Path(args.candle_dir)
            candle_dir.mkdir(parents=True, exist_ok=True)
            with open(candle_dir / "candles.json", "w") as f:
                json.dump([
                    {
                        "ts": c.timestamp.isoformat(),
                        "o": c.open,
                        "h": c.high,
                        "l": c.low,
                        "c": c.close,
                        "v": c.volume
                    }
                    for c in candles
                ], f)
            print(f"Saved {len(candles)} candles to {args.candle_dir}")
    else:
        print("Loading candles from local files...")
        candles = load_real_candles(args.candle_dir)

    if not candles:
        print("\n❌ No candles available!")
        print("\nOptions:")
        print("1. Run with --fetch to fetch from Supabase (requires env vars)")
        print("2. Place candle JSON files in var/real_candles/EURUSD/")
        print("3. Use the GitHub Actions workflow to fetch candles")
        return

    # Run test
    results = run_batch_test(
        candles,
        batch_size=args.batch_size,
        num_batches=args.num_batches
    )

    if args.output and results:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
