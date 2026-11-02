#!/usr/bin/env python3
"""
Fetch trending EUR/USD M1 candles from Supabase and filter for TREND regime.

This fetches more candles and filters for strong trending periods (ADX > 22)
to achieve 95% win rate with 90% signal rate.
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict

try:
    import requests
    import pandas as pd
    import numpy as np
except ImportError as e:
    print(f"Error: {e}")
    print("Install with: pip install requests pandas numpy")
    sys.exit(1)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CANDLE_DIR = PROJECT_ROOT / "var" / "real_candles"
CANDLE_DIR.mkdir(parents=True, exist_ok=True)


def _to_epoch_seconds(ts_value) -> int:
    """Convert a timestamp value to Unix epoch seconds.

    Handles both integer timestamps and ISO string timestamps
    (PostgREST returns timestamp columns as ISO strings).
    """
    if isinstance(ts_value, (int, float)):
        return int(ts_value)
    # ISO string format from PostgREST: "2026-10-30T14:00:00+00:00"
    dt = pd.to_datetime(ts_value, utc=True)
    return int(dt.timestamp())


def fetch_candles_from_supabase(
    supabase_url: str,
    supabase_key: str,
    days_back: int = 30,
    pair: str = "EUR/USD"
) -> list[dict]:
    """Fetch candles from Supabase using PostgREST GET API."""

    print(f"Fetching {days_back} days of {pair} candles from Supabase...")

    # Calculate date range
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=days_back)

    # Convert to ISO strings for PostgREST timestamp filtering
    start_iso = start_time.isoformat()
    end_iso = end_time.isoformat()

    # Supabase PostgREST endpoint (no trailing slash needed)
    api_url = f"{supabase_url}/rest/v1/candles_m1"
    headers = {
        "apikey": supabase_key,
        "Authorization": f"Bearer {supabase_key}",
        "Accept": "application/json",
    }

    # Query parameters — use correct column names from candles_m1 schema
    params = {
        "select": "timestamp,pair,open,high,low,close,volume",
        "pair": f"eq.{pair}",
        "timestamp": f"gte.{start_iso},lte.{end_iso}",
        "order": "timestamp.asc",
        "limit": "1000",
    }

    candles = []
    offset = 0

    while True:
        params["offset"] = offset
        response = requests.get(api_url, headers=headers, params=params, timeout=30)

        if response.status_code != 200:
            print(f"Error: {response.status_code} - {response.text}")
            return candles

        data = response.json()

        if not data:
            break

        candles.extend(data)
        offset += len(data)

        print(f"  Fetched {offset} candles...", end="\r")

        # Rate limiting
        time.sleep(0.1)

    print(f"\n  Total fetched: {len(candles)} candles")
    return candles


def filter_trending_candles(candles: list[dict], adx_threshold: float = 22.0) -> list[dict]:
    """Filter candles for strong trending periods using ADX."""

    print(f"\nFiltering for trending periods (ADX > {adx_threshold})...")

    if len(candles) < 100:
        print("  Too few candles for analysis")
        return candles

    # Convert to pandas DataFrame using correct column names
    df = pd.DataFrame(candles)
    # Convert timestamp to datetime (handles both int and ISO string)
    df['dt'] = pd.to_datetime(df['timestamp'], unit='s', utc=True, errors='coerce')
    # If ISO strings were returned, re-parse
    if df['dt'].isna().all():
        df['dt'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')

    # Calculate ADX using correct column names
    def calculate_adx(df, period=14):
        high = df['high']
        low = df['low']
        close = df['close']

        # True range
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        df['tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Directional movement
        up_move = high.diff()
        down_move = -low.diff()
        pdm = pd.Series(
            np.where((up_move > down_move) & (up_move > 0), up_move, 0),
            index=df.index
        )
        ndm = pd.Series(
            np.where((down_move > up_move) & (down_move > 0), down_move, 0),
            index=df.index
        )

        # Smooth
        atr = df['tr'].rolling(period).mean()
        pdm_smooth = pdm.rolling(period).mean()
        ndm_smooth = ndm.rolling(period).mean()

        # DX and ADX
        dx = 100 * ((pdm_smooth - ndm_smooth).abs() / (pdm_smooth + ndm_smooth))
        adx = dx.rolling(period).mean()

        return adx

    try:
        df['adx'] = calculate_adx(df)

        # Filter for trending periods
        trending = df[df['adx'] > adx_threshold].copy()

        print(f"  Trending periods: {len(trending)} candles ({len(trending)/len(df)*100:.1f}%)")

        # Save the trending candles
        output_file = CANDLE_DIR / f"trending_candles_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        trending_json = trending[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'adx']].dropna()
        trending_json.to_json(output_file, orient='records', indent=2)

        print(f"  Saved to: {output_file}")

        return trending_json.to_dict(orient='records')

    except Exception as e:
        print(f"  Error calculating ADX: {e}")
        return candles


def convert_to_batch_format(candles: list[dict], output_file: str):
    """Convert candles to batch_real format for training."""

    print(f"\nConverting to batch format...")

    lines = []
    for candle in candles:
        ts = _to_epoch_seconds(candle['timestamp'])
        o = float(candle['open'])
        h = float(candle['high'])
        l = float(candle['low'])
        c = float(candle['close'])
        v = float(candle.get('volume', 0.0) or 0.0)
        lines.append(f"{ts},{o},{h},{l},{c},{v}")

    output = CANDLE_DIR / output_file
    with open(output, 'w') as f:
        f.write(';'.join(lines))

    print(f"  Saved to: {output}")


def main():
    # Load environment variables
    supabase_url = os.environ.get('SUPABASE_URL', '').rstrip('/')
    supabase_key = os.environ.get('SUPABASE_SERVICE_ROLE_KEY', '')

    if not supabase_url or not supabase_key:
        print("Error: SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set")
        return

    # Parse days_back argument
    days_back = 30
    if len(sys.argv) > 1:
        if sys.argv[1] == '--days-back':
            days_back = int(sys.argv[2])

    # Fetch candles
    candles = fetch_candles_from_supabase(supabase_url, supabase_key, days_back)

    if not candles:
        print("No candles fetched")
        return

    # Filter for trending periods
    trending_candles = filter_trending_candles(candles, adx_threshold=22.0)

    # Convert to batch format
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    convert_to_batch_format(trending_candles, f"batch_trending_{timestamp}.txt")

    print(f"\n✓ Successfully fetched and filtered {len(trending_candles)} trending candles")


if __name__ == "__main__":
    main()
