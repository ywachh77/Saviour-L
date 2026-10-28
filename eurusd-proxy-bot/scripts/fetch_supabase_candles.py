#!/usr/bin/env python3
"""
Fetch real EUR/USD M1 candles from Supabase and save to var/real_candles/.

This script can be run locally or in any environment with internet access.
GitHub Actions runners have full internet access, so they can reach Supabase.

Usage:
    python3 scripts/fetch_supabase_candles.py

Environment variables (or .env file):
    SUPABASE_URL: Your Supabase project URL
    SUPABASE_SERVICE_ROLE_KEY: Your Supabase service role key
    SUPABASE_PROJECT_REF: Your Supabase project reference (optional)
"""

from __future__ import annotations

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, Optional

try:
    import requests
except ImportError:
    print("Error: 'requests' library not installed.")
    print("Install with: pip install requests")
    sys.exit(1)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CANDLE_DIR = PROJECT_ROOT / "var" / "real_candles"
CANDLE_DIR.mkdir(parents=True, exist_ok=True)


def load_env(env_path: Path) -> Dict[str, str]:
    """Load environment variables from .env file if it exists."""
    env_vars = {}
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    env_vars[key.strip()] = value.strip()
    return env_vars


def get_supabase_config() -> tuple[str, str, str]:
    """Get Supabase configuration from environment variables."""
    # Try environment variables first
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    project_ref = os.environ.get("SUPABASE_PROJECT_REF", "idudlcxkihqnvecwqqoj")

    # Fall back to .env file
    if not supabase_url or not supabase_key:
        env_vars = load_env(PROJECT_ROOT / ".env")
        supabase_url = supabase_url or env_vars.get("SUPABASE_URL")
        supabase_key = supabase_key or env_vars.get("SUPABASE_SERVICE_ROLE_KEY")

    if not supabase_url or not supabase_key:
        print("Error: SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required.")
        print("Set them as environment variables or in .env file.")
        sys.exit(1)

    return supabase_url, supabase_key, project_ref


def fetch_candles_batch(supabase_url: str, supabase_key: str, project_ref: str, offset: int, limit: int = 200) -> Optional[list]:
    """Fetch a batch of candles from Supabase."""
    # Use string_agg to avoid truncation
    query = f"""
    SELECT string_agg(row_data, ';') as candles
    FROM (
        SELECT extract(epoch from timestamp)::bigint::text || ',' || open || ',' || high || ',' || low || ',' || close || ',' || volume as row_data
        FROM candles_m1
        WHERE pair = 'EUR/USD'
        ORDER BY timestamp ASC
        LIMIT {limit}
        OFFSET {offset}
    ) t;
    """

    url = f"{supabase_url}/rest/v1/"
    headers = {
        "apikey": supabase_key,
        "Authorization": f"Bearer {supabase_key}",
        "Content-Type": "application/json",
        "Prefer": "return=representation"
    }

    try:
        response = requests.post(url, headers=headers, json={"query": query}, timeout=30)
        response.raise_for_status()

        data = response.json()
        if not data:
            return None

        candles_str = data[0]["candles"]
        if not candles_str:
            return None

        candles = []
        for candle_str in candles_str.split(";"):
            parts = candle_str.split(",")
            if len(parts) >= 5:
                candles.append({
                    "timestamp": int(float(parts[0])),
                    "open": float(parts[1]),
                    "high": float(parts[2]),
                    "low": float(parts[3]),
                    "close": float(parts[4]),
                    "volume": float(parts[5]) if len(parts) > 5 else 0.0,
                })
        return candles
    except Exception as e:
        print(f"Error fetching batch (offset={offset}): {e}")
        return None


def save_batch(candles: list, batch_idx: int, output_dir: Path) -> Path:
    """Save a batch of candles to a file."""
    lines = []
    for candle in candles:
        lines.append(f"{candle['timestamp']},{candle['open']:.5f},{candle['high']:.5f},{candle['low']:.5f},{candle['close']:.5f},{candle['volume']:.1f}")

    batch_file = output_dir / f"batch_{batch_idx:04d}.txt"
    batch_file.write_text(";".join(lines))
    return batch_file


def main():
    print("=" * 60)
    print("FETCHING REAL CANDLES FROM SUPABASE")
    print("=" * 60)

    supabase_url, supabase_key, project_ref = get_supabase_config()
    print(f"Supabase URL: {supabase_url}")
    print(f"Project Ref: {project_ref}")
    print(f"Output Directory: {CANDLE_DIR}")
    print()

    # First, get total count
    print("Checking candle count...")
    count_query = "SELECT count(*) as total FROM candles_m1 WHERE pair = 'EUR/USD';"
    url = f"{supabase_url}/rest/v1/"
    headers = {
        "apikey": supabase_key,
        "Authorization": f"Bearer {supabase_key}",
        "Content-Type": "application/json",
        "Prefer": "return=representation"
    }

    try:
        response = requests.post(url, headers=headers, json={"query": count_query}, timeout=30)
        response.raise_for_status()
        data = response.json()
        total_candles = int(data[0]["total"])
        print(f"Total candles in database: {total_candles}")
        print()
    except Exception as e:
        print(f"Error checking count: {e}")
        sys.exit(1)

    # Calculate number of batches
    batch_size = 200
    num_batches = (total_candles + batch_size - 1) // batch_size
    print(f"Will fetch {num_batches} batches ({total_candles} candles total)")
    print()

    # Fetch all batches
    start_time = time.time()
    successful_batches = 0

    for batch_idx in range(num_batches):
        offset = batch_idx * batch_size
        print(f"Fetching batch {batch_idx + 1}/{num_batches} (offset={offset})...", end=" ", flush=True)

        candles = fetch_candles_batch(supabase_url, supabase_key, project_ref, offset, batch_size)

        if candles:
            save_batch(candles, batch_idx, CANDLE_DIR)
            print(f"✓ {len(candles)} candles")
            successful_batches += 1
        else:
            print("✗ failed")

        # Rate limiting
        if batch_idx < num_batches - 1:
            time.sleep(0.5)

    elapsed = time.time() - start_time
    print()
    print("=" * 60)
    print("FETCH COMPLETE")
    print("=" * 60)
    print(f"Successful batches: {successful_batches}/{num_batches}")
    print(f"Total candles: {successful_batches * batch_size}")
    print(f"Time elapsed: {elapsed:.1f}s")
    print(f"Output: {CANDLE_DIR}")

    # Save summary
    summary = {
        "total_candles_in_db": total_candles,
        "batches_fetched": successful_batches,
        "candles_fetched": successful_batches * batch_size,
        "batch_size": batch_size,
        "fetched_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }
    summary_file = CANDLE_DIR / "fetch_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2))
    print(f"Summary: {summary_file}")


if __name__ == "__main__":
    main()
