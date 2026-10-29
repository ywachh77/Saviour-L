#!/usr/bin/env python3
"""
Save real candle data to batch files for backtest_engine.py.

Reads candle CSV from stdin, saves to var/real_candles/batch_real_*.txt
Format: ts,open,high,low,close,volume (one candle per line, separated by newlines)
"""

import sys
import os
import json
import re

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'var', 'real_candles')
os.makedirs(OUT_DIR, exist_ok=True)

def parse_candles_from_json(json_text):
    """Extract candle CSV from Supabase JSON response."""
    # Find the candles field
    match = re.search(r'"candles"\s*:\s*"([^"]+)"', json_text)
    if match:
        return match.group(1)
    return None

def save_batch(candles_csv, batch_num):
    """Save candles to batch file."""
    # The candles_csv is already in the correct format: ts,open,high,low,close,volume;...
    # backtest_engine.py splits on ';' so we keep the ';' separator
    # But it expects the file content to be the CSV string
    filename = f'batch_real_{batch_num:03d}.txt'
    filepath = os.path.join(OUT_DIR, filename)
    with open(filepath, 'w') as f:
        f.write(candles_csv)
    candle_count = candles_csv.count(';') + 1
    print(f"Saved {candle_count} candles to {filename}")
    return candle_count

def main():
    # Read all candle data from stdin
    # Format: each line is a JSON response from Supabase
    batches = []
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        candles_csv = parse_candles_from_json(line)
        if candles_csv:
            batches.append(candles_csv)
        else:
            # Maybe the line is already the raw CSV
            if ',' in line and ';' in line:
                batches.append(line)
    
    if not batches:
        print("ERROR: No candle data found in input", file=sys.stderr)
        sys.exit(1)
    
    total = 0
    for i, batch in enumerate(batches):
        count = save_batch(batch, i)
        total += count
    
    print(f"\nTotal candles saved: {total} across {len(batches)} batches")

if __name__ == '__main__':
    main()
