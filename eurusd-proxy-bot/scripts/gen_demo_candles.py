#!/usr/bin/env python3
"""Generate realistic EUR/USD M1 candles for the combiner chart demo.

Uses a random walk with volatility clustering and session patterns
to produce realistic-looking FX data. Not for trading — research/demo only.
"""

import math
import random
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "var" / "real_candles"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Start from a realistic EUR/USD price 500 minutes ago
START_TIME = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0) - timedelta(minutes=500)
START_PRICE = 1.11200

random.seed(42)

candles = []
price = START_PRICE

# Volatility clustering: use a GARCH-like process
vol = 0.0003  # base volatility per minute

for i in range(500):
    ts = START_TIME + timedelta(minutes=i)
    ts_epoch = int(ts.timestamp())

    # GARCH-like volatility clustering
    vol_target = 0.0003 + 0.0002 * abs(math.sin(i / 30))  # periodic volatility
    vol = 0.9 * vol + 0.1 * vol_target + random.gauss(0, 0.00002)

    # Drift: slight mean reversion to START_PRICE
    drift = -0.02 * (price - START_PRICE) / START_PRICE

    # Random walk with drift
    change = drift + vol * random.gauss(0, 1)

    open_price = price
    close_price = max(0.0001, price + change)

    # High/low based on volatility
    high = max(open_price, close_price) + abs(random.gauss(0, vol * 0.5))
    low = min(open_price, close_price) - abs(random.gauss(0, vol * 0.5))
    low = max(0.0001, low)

    # Volume: higher during "London session" (roughly 07:00-16:00 UTC)
    hour = ts.hour
    if 7 <= hour <= 16:
        volume = round(random.uniform(10, 50), 1)
    else:
        volume = round(random.uniform(1, 10), 1)

    candles.append(f"{ts_epoch},{open_price:.5f},{high:.5f},{low:.5f},{close_price:.5f},{volume}")
    price = close_price

# Save in batches of 200 (oldest first)
BATCH_SIZE = 200
for batch_idx in range(0, len(candles), BATCH_SIZE):
    batch = candles[batch_idx:batch_idx + BATCH_SIZE]
    batch_file = OUTPUT_DIR / f"batch_{batch_idx // BATCH_SIZE:03d}.txt"
    batch_file.write_text(";".join(batch))
    print(f"  Saved {len(batch)} candles to {batch_file.name}")

print(f"\nTotal: {len(candles)} candles generated")
print(f"Price range: {min(float(c.split(',')[3]) for c in candles):.5f} - {max(float(c.split(',')[2]) for c in candles):.5f}")
print(f"Time range: {START_TIME.strftime('%Y-%m-%d %H:%M UTC')} to {(START_TIME + timedelta(minutes=499)).strftime('%Y-%m-%d %H:%M UTC')}")
