#!/usr/bin/env python3
"""
Automatically delete unwanted data from Supabase.

Purges crypto (BTC/USD, ETH/USD) candle and signal rows, since crypto
instruments are no longer supported by the trading system.

Can be run locally or from GitHub Actions:

    python3 scripts/cleanup_unwanted_data.py

Environment variables:
    SUPABASE_URL: Your Supabase project URL
    SUPABASE_SERVICE_ROLE_KEY: Your Supabase service role key
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.parse
from pathlib import Path

try:
    import requests
except ImportError:
    print("Error: 'requests' library not installed.")
    print("Install with: pip install requests")
    sys.exit(1)


ROOT = Path(__file__).resolve().parents[1]

# Pairs to purge — anything that is NOT in the allowed symbols list
UNWANTED_PAIRS = ["BTC/USD", "ETH/USD"]


def load_supabase_config():
    """Get Supabase configuration from environment variables."""
    supabase_url = os.environ.get("SUPABASE_URL", "").rstrip("/")
    supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")

    if not supabase_url or not supabase_key:
        # Fall back to .env file
        env_file = ROOT / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    key, value = key.strip(), value.strip()
                    if key == "SUPABASE_URL" and not supabase_url:
                        supabase_url = value.rstrip("/")
                    elif key == "SUPABASE_SERVICE_ROLE_KEY" and not supabase_key:
                        supabase_key = value

    if not supabase_url or not supabase_key:
        print("Error: SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required.")
        sys.exit(1)

    return supabase_url, supabase_key


def _headers(key: str) -> dict:
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Accept": "application/json",
    }


def count_rows(supabase_url: str, key: str, table: str, pair: str) -> int:
    """Count rows for a pair using PostgREST Content-Range."""
    url = f"{supabase_url}/rest/v1/{table}"
    params = {"pair": f"eq.{pair}", "select": "*", "limit": "1"}
    try:
        resp = requests.get(url, headers=_headers(key), params=params, timeout=30)
        if resp.status_code == 400:
            # Table might not have a 'pair' column (e.g. settlements)
            return 0
        resp.raise_for_status()
        cr = resp.headers.get("Content-Range", "")
        if "/" in cr:
            return int(cr.split("/")[-1])
        return len(resp.json())
    except Exception as e:
        print(f"  Count error for {table}/{pair}: {e}")
        return 0


def delete_by_pair(supabase_url: str, key: str, table: str, pair: str) -> int:
    """Delete all rows for a pair using PostgREST DELETE.

    Uses Prefer: return=minimal to avoid downloading deleted rows.
    Returns the number of rows deleted (from Content-Range header).
    """
    url = f"{supabase_url}/rest/v1/{table}"
    params = {"pair": f"eq.{pair}"}
    headers = {**_headers(key), "Prefer": "return=minimal"}
    try:
        resp = requests.delete(url, headers=headers, params=params, timeout=60)
        if resp.status_code == 400:
            # Table doesn't have a 'pair' column
            return 0
        resp.raise_for_status()
        # Content-Range header tells us how many were deleted
        cr = resp.headers.get("Content-Range", "")
        if "/" in cr:
            return int(cr.rsplit("/", 1)[-1])
        return 0
    except Exception as e:
        print(f"  Delete error for {table}/{pair}: {e}")
        return 0


def main() -> int:
    supabase_url, supabase_key = load_supabase_config()
    print(f"Supabase URL: {supabase_url}")
    print(f"Unwanted pairs to purge: {UNWANTED_PAIRS}")
    print()

    results = {}
    total_deleted = 0

    # Clean candles_m1
    print("Cleaning candles_m1 table...")
    candle_results = {}
    for pair in UNWANTED_PAIRS:
        before = count_rows(supabase_url, supabase_key, "candles_m1", pair)
        deleted = delete_by_pair(supabase_url, supabase_key, "candles_m1", pair)
        candle_results[pair] = {"before": before, "deleted": deleted}
        print(f"  {pair}: {before} rows -> deleted {deleted}")
        total_deleted += deleted
        time.sleep(0.5)
    print()

    # Clean signals
    print("Cleaning signals table...")
    signal_results = {}
    for pair in UNWANTED_PAIRS:
        before = count_rows(supabase_url, supabase_key, "signals", pair)
        deleted = delete_by_pair(supabase_url, supabase_key, "signals", pair)
        signal_results[pair] = {"before": before, "deleted": deleted}
        print(f"  {pair}: {before} rows -> deleted {deleted}")
        total_deleted += deleted
        time.sleep(0.5)
    print()

    # settlements table has no 'pair' column — skip it
    print("Cleaning settlements table...")
    print("  Skipped: settlements table has no 'pair' column")
    settlement_results = {}
    print()

    print("=" * 60)
    print("CLEANUP COMPLETE")
    print("=" * 60)
    print(f"Total rows deleted: {total_deleted}")
    print()

    # Save summary
    summary_file = ROOT / "var" / "cleanup_summary.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "cleaned_pairs": UNWANTED_PAIRS,
        "candles": candle_results,
        "signals": signal_results,
        "settlements": settlement_results,
        "total_deleted": total_deleted,
        "cleaned_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    summary_file.write_text(json.dumps(summary, indent=2))
    print(f"Summary saved to: {summary_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
