#!/usr/bin/env python3
"""
Fetch real EUR/USD M1 candles from Supabase and save to var/real_candles/.

This script can be run locally or in any environment with internet access.
GitHub Actions runners have full internet access, so they can reach Supabase.

Usage:
    python3 scripts/fetch_supabase_candles.py

Environment variables (or .env file):
    SUPABASE_URL: Your Supabase project URL (e.g. https://abc.supabase.co)
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
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    project_ref = os.environ.get("SUPABASE_PROJECT_REF", "idudlcxkihqnvecwqqoj")

    if not supabase_url or not supabase_key:
        env_vars = load_env(PROJECT_ROOT / ".env")
        supabase_url = supabase_url or env_vars.get("SUPABASE_URL")
        supabase_key = supabase_key or env_vars.get("SUPABASE_SERVICE_ROLE_KEY")

    if not supabase_url or not supabase_key:
        print("Error: SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required.")
        print("Set them as environment variables or in .env file.")
        sys.exit(1)

    # Ensure URL doesn't have trailing slash
    supabase_url = supabase_url.rstrip("/")

    return supabase_url, supabase_key, project_ref


def _headers(supabase_key: str) -> dict:
    """Build auth headers for Supabase PostgREST."""
    return {
        "apikey": supabase_key,
        "Authorization": f"Bearer {supabase_key}",
        "Accept": "application/json",
    }


def get_total_count(supabase_url: str, supabase_key: str) -> int:
    """Get total candle count from Supabase using Content-Range header."""
    url = f"{supabase_url}/rest/v1/candles_m1"
    params = {
        "pair": "eq.EUR/USD",
        "select": "timestamp",
        "limit": "1",
    }
    try:
        response = requests.get(url, headers=_headers(supabase_key), params=params, timeout=30)
        response.raise_for_status()
        # Content-Range header looks like "0-0/685830"
        content_range = response.headers.get("Content-Range", "")
        if "/" in content_range:
            total = int(content_range.split("/")[-1])
            return total
        # Fallback: try counting returned rows
        data = response.json()
        return len(data)
    except Exception as e:
        print(f"Error checking count: {e}")
        sys.exit(1)


def fetch_candles_batch(
    supabase_url: str,
    supabase_key: str,
    offset: int,
    limit: int = 1000,
) -> Optional[list]:
    """Fetch a batch of candles from Supabase PostgREST API.

    Uses GET with URL parameters instead of POST with SQL — PostgREST
    does not accept raw SQL in POST bodies.
    """
    url = f"{supabase_url}/rest/v1/candles_m1"
    params = {
        "pair": "eq.EUR/USD",
        "select": "timestamp,open,high,low,close,volume",
        "order": "timestamp.asc",
        "limit": str(limit),
        "offset": str(offset),
    }

    try:
        response = requests.get(url, headers=_headers(supabase_key), params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        if not data:
            return None

        candles = []
        for row in data:
            candles.append({
                "timestamp": _to_epoch_seconds(row["timestamp"]),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row.get("volume", 0.0) or 0.0),
            })
        return candles
    except Exception as e:
        print(f"Error fetching batch (offset={offset}): {e}")
        return None


def _to_epoch_seconds(ts_value) -> int:
    """Convert a timestamp value to Unix epoch seconds.

    Handles both integer timestamps and ISO string timestamps
    (PostgREST returns timestamp columns as ISO strings by default).
    """
    if isinstance(ts_value, (int, float)):
        return int(ts_value)
    from datetime import datetime
    dt = datetime.fromisoformat(str(ts_value).replace("Z", "+00:00"))
    return int(dt.timestamp())


def save_batch(candles: list, batch_idx: int, output_dir: Path) -> Path:
    """Save a batch of candles to a file."""
    lines = []
    for candle in candles:
        lines.append(
            f"{candle['timestamp']},{candle['open']:.5f},{candle['high']:.5f},"
            f"{candle['low']:.5f},{candle['close']:.5f},{candle['volume']:.1f}"
        )
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
    total_candles = get_total_count(supabase_url, supabase_key)
    print(f"Total candles in database: {total_candles}")
    print()

    # Calculate number of batches
    batch_size = 1000
    num_batches = (total_candles + batch_size - 1) // batch_size
    print(f"Will fetch {num_batches} batches ({total_candles} candles total)")
    print()

    # Fetch all batches
    start_time = time.time()
    successful_batches = 0
    total_fetched = 0

    for batch_idx in range(num_batches):
        offset = batch_idx * batch_size
        print(f"Fetching batch {batch_idx + 1}/{num_batches} (offset={offset})...", end=" ", flush=True)

        candles = fetch_candles_batch(supabase_url, supabase_key, offset, batch_size)

        if candles:
            save_batch(candles, batch_idx, CANDLE_DIR)
            print(f"✓ {len(candles)} candles")
            successful_batches += 1
            total_fetched += len(candles)
        else:
            print("✗ failed")

        # Rate limiting
        if batch_idx < num_batches - 1:
            time.sleep(0.3)

    elapsed = time.time() - start_time
    print()
    print("=" * 60)
    print("FETCH COMPLETE")
    print("=" * 60)
    print(f"Successful batches: {successful_batches}/{num_batches}")
    print(f"Total candles: {total_fetched}")
    print(f"Time elapsed: {elapsed:.1f}s")
    print(f"Output: {CANDLE_DIR}")

    # Save summary
    summary = {
        "total_candles_in_db": total_candles,
        "batches_fetched": successful_batches,
        "candles_fetched": total_fetched,
        "batch_size": batch_size,
        "fetched_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }
    summary_file = CANDLE_DIR / "fetch_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2))
    print(f"Summary: {summary_file}")


if __name__ == "__main__":
    main()
