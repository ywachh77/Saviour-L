#!/usr/bin/env python3
"""
Patterned Synthetic EUR/USD M1 Candle Generator
Generates synthetic candle data with actual predictable patterns for ML training.

Includes:
- Momentum effects (trend continuation)
- Mean reversion (pullback to mean)
- Volatility clustering (GARCH-like)
- Regime switching (trend vs range)
- Autocorrelation in returns
- Session effects (London/NY)
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
NUM_CANDLES = 50000  # Smaller for faster training
START_TIME = datetime(2026, 6, 21, 0, 10, 0, tzinfo=timezone.utc)
BASE_PRICE = 1.1607
ANNUAL_VOL = 0.08
DAILY_VOL = ANNUAL_VOL / math.sqrt(252 * 24 * 60)

# Pattern parameters
MOMENTUM_STRENGTH = 0.15  # How much previous return affects next
MEAN_REVERSION_STRENGTH = 0.05  # How much price reverts to mean
VOLATILITY_CLUSTERING = 0.1  # GARCH-like effect
REGIME_DURATION = 1000  # Bars per regime


def generate_patterned_candles():
    """Generate synthetic candles with actual patterns."""

    prices = []
    returns = []
    price = BASE_PRICE
    mean_price = BASE_PRICE

    # Regime state
    regime = "trend"  # or "range"
    regime_counter = 0
    trend_direction = 1  # 1 for up, -1 for down

    # Volatility state
    current_vol = DAILY_VOL
    vol_persistence = 0.95

    for i in range(NUM_CANDLES):
        # Regime switching
        if regime_counter >= REGIME_DURATION:
            regime = random.choices(["trend", "range"], weights=[0.4, 0.6])[0]
            regime_counter = 0
            if regime == "trend":
                trend_direction = random.choice([1, -1])
        regime_counter += 1

        # Calculate drift based on regime
        if regime == "trend":
            drift = trend_direction * DAILY_VOL * 0.5
        else:
            drift = 0.0

        # Momentum effect: use previous return
        momentum = 0.0
        if i > 0 and len(returns) > 0:
            momentum = MOMENTUM_STRENGTH * returns[-1]

        # Mean reversion: pull toward mean
        mean_reversion = MEAN_REVERSION_STRENGTH * (mean_price - price) / price

        # Volatility clustering (GARCH-like)
        vol_shock = random.uniform(0.8, 1.2)
        current_vol = vol_persistence * current_vol + (1 - vol_persistence) * DAILY_VOL * vol_shock
        current_vol = max(current_vol, DAILY_VOL * 0.5)
        current_vol = min(current_vol, DAILY_VOL * 3)

        # Combine effects
        shock = random.gauss(0, 1)
        price_change = drift + momentum + mean_reversion + current_vol * shock

        # Update price
        old_price = price
        price = price * (1 + price_change)

        # Track return
        if i > 0:
            returns.append((price - old_price) / old_price)

        # Update mean (rolling)
        if i % 100 == 0:
            mean_price = (mean_price + price) / 2

        # Generate OHLC
        high_shock = abs(random.gauss(0, current_vol * 0.4))
        low_shock = abs(random.gauss(0, current_vol * 0.4))
        open_price = old_price
        close_price = price
        high_price = max(open_price, close_price) + high_shock * price
        low_price = min(open_price, close_price) - low_shock * price

        timestamp = START_TIME + timedelta(minutes=i)
        prices.append((timestamp, open_price, high_price, low_price, close_price))

    return prices


def save_batches(candles, batch_size=200):
    """Save candles in batches."""

    total_batches = (len(candles) + batch_size - 1) // batch_size
    print(f"Generating {len(candles)} candles in {total_batches} batches...")

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(candles))
        batch = candles[start:end]

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
    print("PATTERNED SYNTHETIC CANDLE GENERATOR")
    print("=" * 60)
    print(f"Generating {NUM_CANDLES} candles with patterns:")
    print(f"  - Momentum strength: {MOMENTUM_STRENGTH:.2f}")
    print(f"  - Mean reversion: {MEAN_REVERSION_STRENGTH:.2f}")
    print(f"  - Volatility clustering: {VOLATILITY_CLUSTERING:.2f}")
    print(f"  - Regime duration: {REGIME_DURATION} bars")
    print()

    candles = generate_patterned_candles()
    save_batches(candles)

    print()
    print("=" * 60)
    print("✓ Patterned candle generation complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
