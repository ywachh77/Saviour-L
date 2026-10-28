#!/usr/bin/env python3
"""Fetch candles from Supabase REST API and save to var/real_candles/batch_*.txt"""

import json
import sys
import time
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "var" / "real_candles"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUPABASE_URL = "https://idudlcxkihqnvecwqqoj.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImlkdWRsY3hraWhxbnZlY3dxcW9qIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc5MjY3MjI1MywiZXhwIjoyMTA4MjQ4MjUzfQ.uMafh4b2NCSqcFe5796i18h9pLthNCVNKnpAtMSojYs"

BATCH_SIZE = 300
NUM_BATCHES = 5  # Fetch 1500 candles total


def fetch_batch(offset: int, limit: int) -> list[dict]:
    """Fetch a batch of EUR/USD candles from Supabase."""
    url = (
        f"{SUPABASE_URL}/rest/v1/candles_m1"
        f"?pair=eq.EUR%2FUSD"
        f"&select=timestamp,open,high,low,close,volume"
        f"&order=timestamp.desc"
        f"&offset={offset}"
        f"&limit={limit}"
    )
    req = urllib.request.Request(url, headers={
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
    })
    resp = urllib.request.urlopen(req, timeout=15)
    data = json.loads(resp.read())
    return data


print(f"Fetching {NUM_BATCHES} batches of {BATCH_SIZE} candles from Supabase...")
print(f"Output: {OUTPUT_DIR}")

for i in range(NUM_BATCHES):
    offset = i * BATCH_SIZE
    print(f"  Batch {i+1}/{NUM_BATCHES} (offset={offset})...")
    try:
        data = fetch_batch(offset, BATCH_SIZE)
        if not data:
            print(f"    No more candles at offset {offset}")
            break

        lines = []
        for row in data:
            ts = int(row["timestamp"].split("+")[0].replace(" ", "T"))
            # Actually, the timestamp is like "2026-10-28 08:43:00+00"
            # We need to convert to epoch
            from datetime import datetime, timezone
            dt = datetime.fromisoformat(row["timestamp"])
            ts = int(dt.timestamp())
            line = f"{ts},{row['open']},{row['high']},{row['low']},{row['close']},{row['volume']}"
            lines.append(line)

        # Save in reverse order (oldest first) for the chart
        lines = list(reversed(lines))
        batch_file = OUTPUT_DIR / f"batch_{i:03d}.txt"
        batch_file.write_text(";".join(lines))
        print(f"    Saved {len(lines)} candles to {batch_file.name}")
        time.sleep(0.3)
    except Exception as e:
        print(f"    ERROR: {e}")
        break

print("\nDone!")
# Count total
total = 0
for f in sorted(OUTPUT_DIR.glob("batch_*.txt")):
    lines = f.read_text().strip().split(";")
    total += len(lines)
    print(f"  {f.name}: {len(lines)} candles")
print(f"Total: {total} candles")
