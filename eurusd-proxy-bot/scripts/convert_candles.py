#!/usr/bin/env python3
"""Convert batch_*.txt candle files to the JSON format the neural trainer expects.

The fetch script saves candles as:
    batch_0000.txt: ts,open,high,low,close,volume;ts,open,...

The neural trainer expects:
    [{"ts": "...", "o": ..., "h": ..., "l": ..., "c": ..., "v": ...}, ...]

This script merges all batch files, sorts by timestamp, and writes the JSON.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CANDLE_DIR = PROJECT_ROOT / "var" / "real_candles"


def convert_batches_to_json(
    input_dir: Path = CANDLE_DIR,
    output_path: Path = CANDLE_DIR / "eurusd_real_20261030.json",
    pair_slug: str = "EURUSD",
) -> dict:
    """Merge all batch_*.txt files into a single JSON file."""
    
    # Find all batch files
    batch_files = sorted(input_dir.glob(f"batch_*.txt"))
    if not batch_files:
        print(f"ERROR: No batch files found in {input_dir}")
        return {"status": "error", "error": "no batch files"}

    all_candles = []
    for batch_file in batch_files:
        content = batch_file.read_text().strip()
        if not content:
            continue
        for line in content.split(";"):
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 5:
                continue
            try:
                ts = int(float(parts[0]))
                open_p = float(parts[1])
                high = float(parts[2])
                low = float(parts[3])
                close = float(parts[4])
                volume = float(parts[5]) if len(parts) > 5 else 0.0
            except (ValueError, IndexError):
                continue
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            all_candles.append({
                "ts": dt.isoformat(),
                "o": open_p,
                "h": high,
                "l": low,
                "c": close,
                "v": volume,
            })

    if not all_candles:
        print(f"ERROR: No candles parsed from {len(batch_files)} batch files")
        return {"status": "error", "error": "no candles parsed"}

    # Sort by timestamp ascending (oldest first)
    all_candles.sort(key=lambda r: r["ts"])

    # Reverse so newest is first (matching existing format)
    all_candles.reverse()

    output_path.write_text(json.dumps(all_candles, indent=2))
    
    non_zero_vol = sum(1 for r in all_candles if r["v"] > 0)
    print(f"Converted {len(all_candles)} candles from {len(batch_files)} batch files")
    print(f"Non-zero volume: {non_zero_vol}/{len(all_candles)}")
    print(f"Output: {output_path}")
    
    return {
        "status": "ok",
        "candles": len(all_candles),
        "batches": len(batch_files),
        "non_zero_volume": non_zero_vol,
        "output": str(output_path),
    }


if __name__ == "__main__":
    result = convert_batches_to_json()
    sys.exit(0 if result.get("status") == "ok" else 1)
