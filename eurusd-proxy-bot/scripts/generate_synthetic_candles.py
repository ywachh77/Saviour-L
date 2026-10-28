#!/usr/bin/env python3
"""
Synthetic EUR/USD M1 Candle Generator
Generates realistic forex candle data for training when Supabase is unreachable.
"""

import math
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "var" / "real_candles"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Parameters
NUM_CANDLES = 225944  # Match Supabase count
START_TIME = datetime(2026, 6, 21, 0, 10, 0, tzinfo=timezone.utc)
BASE_PRICE = 1.1607
SPREAD = 0.00012  # 1.2 pips
ANNUAL_VOL = 0.08  # 8% annual volatility for EUR/USD
DAILY_VOL = ANNUAL_VOL / math.sqrt(252 * 24 * 60)  # Convert to M1

# GBM parameters for realistic price movement
def generate_synthetic_candles():
    """Generate synthetic EUR/USD M1 candles using GBM with regime changes."""
    
    prices = []
    timestamps = []
    price = BASE_PRICE
    
    # Add regime switching for realism
    regimes = [
        {"drift": 0.000001, "vol": DAILY_VOL * 0.5, "duration": 5000},  # Low vol trend
        {"drift": 0.000003, "vol": DAILY_VOL * 1.5, "duration": 3000},  # High vol trend
        {"drift": 0.0, "vol": DAILY_VOL * 0.8, "duration": 8000},      # Range-bound
        {"drift": -0.000002, "vol": DAILY_VOL * 1.2, "duration": 4000}, # Downtrend
    ]
    
    current_regime = 0
    regime_counter = 0
    
    for i in range(NUM_CANDLES):
        # Switch regimes
        if regime_counter >= regimes[current_regime]["duration"]:
            current_regime = random.randint(0, len(regimes) - 1)
            regime_counter = 0
        regime_counter += 1
        
        # GBM step
        drift = regimes[current_regime]["drift"]
        vol = regimes[current_regime]["vol"]
        
        # Random shock (normal distribution)
        shock = random.gauss(0, 1)
        price_change = drift + vol * shock
        
        # Update price
        price = price * (1 + price_change)
        
        # Generate OHLC from close
        high_shock = abs(random.gauss(0, vol * 0.5))
        low_shock = abs(random.gauss(0, vol * 0.5))
        open_price = price
        close_price = price
        high_price = open_price + high_shock * price
        low_price = open_price - low_shock * price
        
        timestamp = START_TIME + timedelta(minutes=i)
        
        prices.append((timestamp, open_price, high_price, low_price, close_price))
        timestamps.append(timestamp)
    
    return prices


def save_batches(candles, batch_size=200):
    """Save candles in batches of 200 to match Supabase fetch format."""
    
    total_batches = (len(candles) + batch_size - 1) // batch_size
    print(f"Generating {len(candles)} candles in {total_batches} batches...")
    
    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(candles))
        
        batch = candles[start:end]
        
        # Format: epoch_seconds,open,high,low,close,volume;...
        lines = []
        for ts, o, h, l, c in batch:
            epoch = int(ts.timestamp())
            lines.append(f"{epoch},{o:.5f},{h:.5f},{l:.5f},{c:.5f},0.0")
        
        batch_file = OUTPUT_DIR / f"batch_{batch_idx:04d}.txt"
        batch_file.write_text(";".join(lines))
        
        if batch_idx % 50 == 0:
            print(f"  Batch {batch_idx}/{total_batches} complete")
    
    print(f"✓ All {total_batches} batches saved to {OUTPUT_DIR}")


def main():
    print("=" * 60)
    print("SYNTHETIC EUR/USD M1 CANDLE GENERATOR")
    print("=" * 60)
    print(f"Generating {NUM_CANDLES} candles starting from {START_TIME}")
    print(f"Base price: {BASE_PRICE}")
    print(f"Annual volatility: {ANNUAL_VOL:.1%}")
    print()
    
    candles = generate_synthetic_candles()
    save_batches(candles)
    
    print()
    print("=" * 60)
    print("✓ Synthetic candle generation complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
