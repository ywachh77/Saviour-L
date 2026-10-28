#!/usr/bin/env python3
"""
Generate patterns specifically tuned for 1-minute horizon.
Creates clear short-term trends with realistic micro-pullbacks.
Each pattern: clear trend → brief pause → reversal signal
Output format: ts,open,high,low,close,volume; per line
"""
import os
import random
from datetime import datetime, timedelta

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "var", "real_candles")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def make_candle(ts, open_p, close_p, spread_frac=0.5):
    """Create a valid candle with proper OHLC relationships."""
    o, c = open_p, close_p
    body = abs(c - o)
    wick = body * random.uniform(0.3, 1.5) if body > 0 else random.uniform(2, 10) * 0.0001
    # Ensure high >= max(o,c) and low <= min(o,c)
    high = max(o, c) + wick * random.uniform(0.2, spread_frac)
    low = min(o, c) - wick * random.uniform(0.2, spread_frac)
    return (int(ts), f"{o:.6f}", f"{high:.6f}", f"{low:.6f}", f"{c:.6f}", str(random.randint(1000, 10000)))

def generate_micro_trend_pattern(base_price=1.1000, pip=0.0001):
    candles = []
    price = base_price
    start_time = datetime.utcnow().replace(minute=0, second=0, microsecond=0) - timedelta(hours=random.randint(1, 48))
    
    # Phase 1: Clear directional segment
    direction = random.choice([1, -1])
    trend_length = random.randint(5, 8)
    for i in range(trend_length):
        move = random.uniform(8, 20) * pip * direction
        if random.random() < 0.3:
            move *= random.uniform(0.5, 0.8)
        open_p = price
        price += move
        candles.append(make_candle(int((start_time + timedelta(minutes=i)).timestamp()), open_p, price))
    
    # Phase 2: Brief consolidation
    consolidation_len = random.randint(2, 4)
    consolidation_width = random.uniform(5, 15) * pip
    for i in range(consolidation_len):
        offset = random.uniform(-consolidation_width, consolidation_width)
        open_p = price
        price = price + offset * 0.5
        candles.append(make_candle(int((start_time + timedelta(minutes=trend_length + i)).timestamp()), open_p, price, spread_frac=1.5))
    
    # Phase 3: Signal candle (reversal)
    signal_direction = direction * -1
    signal_move = random.uniform(15, 35) * pip * signal_direction
    open_p = price
    price += signal_move
    candles.append(make_candle(int((start_time + timedelta(minutes=trend_length + consolidation_len)).timestamp()), open_p, price))
    
    # Phase 4: Result candles
    remaining = random.randint(10, 15)
    for i in range(remaining):
        move = signal_direction * random.uniform(5, 18) * pip
        if random.random() < 0.2:
            move *= random.uniform(0.4, 0.7)
        open_p = price
        price += move
        candles.append(make_candle(int((start_time + timedelta(minutes=trend_length + consolidation_len + 1 + i)).timestamp()), open_p, price))
    
    return candles

def generate_micro_trend_batch(batch_num):
    candles = []
    for _ in range(random.randint(50, 80)):
        candles.extend(generate_micro_trend_pattern())
    candles.sort(key=lambda c: c[0])
    
    # Format: ts,open,high,low,close,volume;ts,open,high,low,close,volume;...
    content = ";".join(",".join(str(x) for x in c) for c in candles) + "\n"
    path = os.path.join(OUTPUT_DIR, f"batch_{batch_num:04d}.txt")
    with open(path, "w") as f:
        f.write(content)
    return len(candles)

total_candles = 0
for i in range(500):
    total_candles += generate_micro_trend_batch(i)
    if (i + 1) % 100 == 0:
        print(f"Generated {i+1}/500 batches, {total_candles} candles")

print(f"Done! Generated {total_candles} candles across 500 batches")
print(f"Output: {OUTPUT_DIR}")
