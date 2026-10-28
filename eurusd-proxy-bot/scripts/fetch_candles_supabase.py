#!/usr/bin/env python3
"""Fetch candles from Supabase and save to var/real_candles/batch_*.txt"""

import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Read Supabase credentials from .env
def load_env(path):
    env = {}
    for line in path.read_text().strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        env[key.strip()] = val.strip()
    return env

env = load_env(PROJECT_ROOT / "eurusd-proxy-bot" / ".env")
SUPABASE_URL = env.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = env.get("SUPABASE_SERVICE_ROLE_KEY", "")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("ERROR: No Supabase credentials in .env")
    sys.exit(1)

import urllib.request

BATCH_SIZE = 300
NUM_BATCHES = 10  # Fetch 3000 candles total
OUTPUT_DIR = PROJECT_ROOT / "eurusd-proxy-bot" / "var" / "real_candles"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def fetch_batch(offset: int, limit: int) -> list[str]:
    """Fetch a batch of candles from Supabase."""
    url = f"{SUPABASE_URL}/rest/v1/candles_m1?pair=eq.EUR%2FUSD&select=timestamp,open,high,low,close,volume&order=timestamp.asc&offset={offset}&limit={limit}"
    req = urllib.request.Request(url, headers={
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
    })
    resp = urllib.request.urlopen(req, timeout=15)
    data = json.loads(resp.read())
    lines = []
    for row in data:
        ts = int(datetime.fromisoformat(row["timestamp"]).timestamp())
        line = f"{ts},{row['open']},{row['high']},{row['low']},{row['close']},{row['volume']}"
        lines.append(line)
    return lines

print(f"Fetching {NUM_BATCHES} batches of {BATCH_SIZE} candles from Supabase...")
print(f"Output: {OUTPUT_DIR}")

for i in range(NUM_BATCHES):
    offset = i * BATCH_SIZE
    print(f"  Batch {i+1}/{NUM_BATCHES} (offset={offset})...")
    try:
        lines = fetch_batch(offset, BATCH_SIZE)
        if not lines:
            print(f"    No more candles at offset {offset}")
            break
        batch_file = OUTPUT_DIR / f"batch_{i:03d}.txt"
        batch_file.write_text(";".join(lines))
        print(f"    Saved {len(lines)} candles to {batch_file.name}")
        time.sleep(0.5)  # Rate limit
    except Exception as e:
        print(f"    ERROR: {e}")
        # Try with smaller batch
        print(f"    Retrying with smaller batch (100)...")
        try:
            lines = fetch_batch(offset, 100)
            if lines:
                batch_file = OUTPUT_DIR / f"batch_{i:03d}.txt"
                batch_file.write_text(";".join(lines))
                print(f"    Saved {len(lines)} candles to {batch_file.name}")
        except Exception as e2:
            print(f"    FAILED: {e2}")

print("\nDone!")
# Count total candles
total = 0
for f in sorted(OUTPUT_DIR.glob("batch_*.txt")):
    count = len(f.read_text().strip().split(";"))
    total += count
    print(f"  {f.name}: {count} candles")
print(f"\nTotal: {total} candles")
