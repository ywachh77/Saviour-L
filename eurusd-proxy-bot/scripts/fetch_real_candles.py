#!/usr/bin/env python3
"""
Fetch real candles from Supabase and run walk-forward backtest.

Downloads EUR/USD 1-minute candles from the Supabase candles_m1 table,
saves them locally, and runs the backtester to measure real-world accuracy.
"""

import json
import sys
import urllib.request
import urllib.error
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SUPABASE_URL = "https://idudlcxkihqnvecwqqoj.supabase.co"
ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImlkdWRsY3hraWhxbnZlY3dxcW9qIiwicm9sZSI6ImFub24iLCJpYXQiOjE3OTI2NzIyNTMsImV4cCI6MjEwODI0ODI1M30.yFL4mOOf_-tMtduP0hCuv2EPQmX8Hhfyn3tagnkleDE"
CANDLE_DIR = PROJECT_ROOT / "var" / "real_candles"
BATCH_SIZE = 1000
NUM_CANDLESTERS = 5000


def fetch_candles_batch(offset: int) -> list[dict]:
    """Fetch one batch of EUR/USD candles from Supabase REST API."""
    url = (
        f"{SUPABASE_URL}/rest/v1/candles_m1"
        f"?pair=eq.EUR%2FUSD"
        f"&select=timestamp,open,high,low,close,volume"
        f"&order=timestamp.desc"
        f"&limit={BATCH_SIZE}"
        f"&offset={offset}"
    )
    req = urllib.request.Request(url, headers={
        "apikey": ANON_KEY,
        "Authorization": f"Bearer {ANON_KEY}",
        "Content-Type": "application/json",
    })
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
            return data
    except urllib.error.URLError as e:
        print(f"  Error fetching batch at offset {offset}: {e}")
        return []


def save_candles(candles: list[dict], filename: str) -> None:
    """Save candles to a local batch file in the standard format."""
    lines = []
    for c in candles:
        ts = int(c["timestamp"])
        o = float(c["open"])
        h = float(c["high"])
        l = float(c["low"])
        cl = float(c["close"])
        v = float(c.get("volume", 0))
        lines.append(f"{ts};{o:.6f},{h:.6f},{l:.6f},{cl:.6f},{v:.2f}")
    content = ";".join(lines)
    out_file = CANDLE_DIR / filename
    out_file.write_text(content)
    print(f"  Saved {len(lines)} candles to {out_file}")


def main():
    print("=" * 60)
    print("FETCHING REAL EUR/USD CANDLES FROM SUPABASE")
    print("=" * 60)

    CANDLE_DIR.mkdir(parents=True, exist_ok=True)

    all_candles = []
    batch_num = 0

    # Fetch in batches, ordered DESC (newest first), so we need to reverse
    for offset in range(0, NUM_CANDLESTERS, BATCH_SIZE):
        print(f"\nFetching batch {batch_num + 1} (offset={offset})...")
        candles = fetch_candles_batch(offset)
        if not candles:
            print(f"  No more candles. Stopping.")
            break
        all_candles.extend(candles)
        print(f"  Got {len(candles)} candles (total: {len(all_candles)})")
        batch_num += 1

    if not all_candles:
        print("ERROR: No candles fetched.")
        sys.exit(1)

    # Reverse to chronological order (oldest first)
    all_candles.reverse()
    print(f"\nTotal candles fetched: {len(all_candles)}")

    # Save to a single batch file for the backtester
    # Use batch_900.txt to avoid conflict with synthetic training data
    save_candles(all_candles, "batch_900_real.txt")

    # Also save as a separate real candles file
    save_candles(all_candles, "batch_real_eunusd.txt")

    # Print price range
    closes = [float(c["close"]) for c in all_candles]
    print(f"\nPrice range: {min(closes):.5f} - {max(closes):.5f}")
    print(f"First candle: {all_candles[0]['timestamp']}")
    print(f"Last candle:  {all_candles[-1]['timestamp']}")

    # Now run the backtest
    print("\n" + "=" * 60)
    print("RUNNING WALK-FORWARD BACKTEST ON REAL CANDLES")
    print("=" * 60)

    from backtest_engine import run_backtest, load_candles

    # The backtester loads from batch_*.txt files. Let's use our real candles file.
    # We need to temporarily move the synthetic files out of the way.
    # Actually, the backtester loads ALL batch files. Let's just run it with
    # the real candles file only.

    # Load just the real candles
    real_candles = []
    real_file = CANDLE_DIR / "batch_real_eunusd.txt"
    for line in real_file.read_text().strip().split(";"):
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) < 5:
            continue
        ts = int(float(parts[0]))
        from datetime import datetime, timezone
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        from fusion.domain import Candle
        real_candles.append(Candle(
            timestamp=dt,
            open=float(parts[1]),
            high=float(parts[2]),
            low=float(parts[3]),
            close=float(parts[4]),
            volume=float(parts[5]) if len(parts) > 5 else 0.0,
        ))

    print(f"Loaded {len(real_candles)} real candles")

    # Run backtest for each expiry
    for expiry in [1, 3, 30]:
        print(f"\n{'=' * 40}")
        print(f"BACKTEST: {expiry}-MINUTE EXPIRY (REAL CANDLES)")
        print(f"{'=' * 40}")
        result = run_backtest(real_candles, expiry_minutes=expiry)
        if result:
            print(f"  Total trades:  {result['total_trades']}")
            print(f"  Wins:          {result['wins']}")
            print(f"  Losses:        {result['losses']}")
            print(f"  Win rate:      {result['win_rate']:.1%}")
            print(f"  Total pips:    {result['total_pips']:.2f}")
            print(f"  Avg pips/trade: {result['avg_pips_per_trade']:.2f}")
            print(f"  Sharpe ratio:  {result['sharpe_ratio']:.2f}")
            print(f"  Max drawdown:  {result['max_drawdown']:.2%}")
            print(f"  Final equity:  ${result['final_equity']:.2f}")

            print(f"\n  Per-regime:")
            for regime, data in result["per_regime"].items():
                if data["trades"] > 0:
                    print(f"    {regime:8s}: {data['trades']} trades, "
                          f"{data['win_rate']:.1%} win rate, "
                          f"{data['total_pips']:.2f} pips")

            print(f"\n  Sureshot analysis:")
            for threshold, data in result["sureshot_analysis"].items():
                if data["trades"] > 0:
                    print(f"    {threshold:12s}: {data['trades']} trades, "
                          f"{data['win_rate']:.1%} win rate, "
                          f"{data['total_pips']:.2f} pips")

    print(f"\n{'=' * 60}")
    print("REAL DATA BACKTEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
