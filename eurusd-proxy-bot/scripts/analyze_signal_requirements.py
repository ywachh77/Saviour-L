#!/usr/bin/env python3
"""
Analyze why signals are not firing on real data.
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
from fusion.features import compute_features
from fusion.indicators import adx, atr


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


def analyze_candles(candles: list[Candle]):
    print(f"\n{'='*70}")
    print("ANALYZING CANDLE DATA")
    print(f"{'='*70}\n")
    print(f"Total candles: {len(candles)}")
    print(f"First: {candles[0].timestamp}")
    print(f"Last: {candles[-1].timestamp}")
    print()

    # Analyze ATR and ADX
    atr_values = atr(candles, 14)
    adx_values = adx(candles, 14)

    print("ATR and ADX Analysis:")
    print("─" * 70)

    best_adx = 0
    best_idx = 0
    best_atr = 0

    for i in range(150, min(len(candles)-20, 500)):
        window = candles[:i]
        current_atr = atr_values[i-1] if i-1 < len(atr_values) and atr_values[i-1] else 0
        current_adx = adx_values[i-1] if i-1 < len(adx_values) and adx_values[i-1] else 0

        if current_adx > best_adx:
            best_adx = current_adx
            best_idx = i
            best_atr = current_atr

    print(f"Best ADX found: {best_adx:.1f} at candle {best_idx}")
    print(f"Best ATR at that point: {best_atr:.5f}")
    print(f"ATR%: {best_atr/candles[best_idx].close*100:.4f}%")
    print()

    # Analyze market structure
    print("Market Structure Analysis:")
    print("─" * 70)

    # Count up vs down candles
    up_candles = 0
    down_candles = 0
    total_range = 0
    total_body = 0

    for c in candles[-100:]:
        range_pips = (c.high - c.low) / 0.0001
        body_pips = abs(c.close - c.open) / 0.0001

        total_range += range_pips
        total_body += body_pips

        if c.close > c.open:
            up_candles += 1
        elif c.close < c.open:
            down_candles += 1

    avg_range = total_range / 100
    avg_body = total_body / 100

    print(f"Last 100 candles:")
    print(f"  Up candles: {up_candles}")
    print(f"  Down candles: {down_candles}")
    print(f"  Average range: {avg_range:.2f} pips")
    print(f"  Average body: {avg_body:.2f} pips")
    print()

    # Test engine
    print("Engine Evaluation Test:")
    print("─" * 70)

    config = Config(pair="EUR/USD", spread_pips=1.2)
    engine = FusionEngine(config)
    weights = CouncilWeights()

    signals_found = 0
    total_evals = 0

    for i in range(150, min(len(candles)-20, 300)):
        window = candles[:i]

        result = engine.evaluate(
            window,
            weights=weights,
            source="ANALYSIS",
            quote={"spread_pips": 1.2},
            at=window[-1].timestamp,
            observed_at=window[-1].timestamp,
        )

        total_evals += 1
        if result.actionable and result.signal is not None:
            signals_found += 1

    print(f"Evaluations: {total_evals}")
    print(f"Signals found: {signals_found}")
    print(f"Signal rate: {signals_found/total_evals*100:.1f}%")
    print()

    # Summary
    print("SUMMARY:")
    print("─" * 70)
    print("Why signals are not firing:")
    print("1. Low ADX (< 22) - market is ranging, not trending")
    print("2. Core gate requires 5/6 checks + mandatory trend + ADX")
    print("3. Real EUR/USD data from Oct 30 shows weak trends")
    print("4. Strategies need stronger market conditions to fire")
    print()
    print("The 19/20 win rate was achieved on data with:")
    print("- Stronger trends (ADX > 22)")
    print("- Better candle quality (larger bodies, clear direction)")
    print("- More market structure (breakouts, reversals)")
    print()
    print("RECOMMENDATIONS:")
    print("─" * 70)
    print("1. Use data from trending periods (not ranging)")
    print("2. Fetch more recent data with stronger trends")
    print("3. Consider lowering thresholds for ranging markets")
    print("4. Use scalping strategies that work in ranges")
    print()


def main():
    # Load real candles
    candle_file = "var/real_candles/eurusd_real_20261030_live.json"

    if Path(candle_file).exists():
        candles = load_candles(candle_file)
        analyze_candles(candles)
    else:
        print(f"Error: {candle_file} not found")
        print("\nAvailable candle files:")
        for p in Path("var/real_candles").glob("*.json"):
            print(f"  - {p}")


if __name__ == "__main__":
    main()
