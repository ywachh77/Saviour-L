#!/usr/bin/env python3
"""Sync all real candles from Supabase to local store, then run nightly training.

Run via start_process (needs Supabase network access).
"""
from __future__ import annotations

import json, os, sys, time, urllib.request, urllib.error
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Read .env
supabase_url = ""
supabase_key = ""
for line in (ROOT / ".env").read_text().splitlines():
    if line.startswith("SUPABASE_URL="):
        supabase_url = line.strip().split("=", 1)[1]
    if line.startswith("SUPABASE_PUBLISHABLE_KEY="):
        supabase_key = line.strip().split("=", 1)[1]

if not supabase_url or not supabase_key:
    print("ERROR: SUPABASE_URL or SUPABASE_PUBLISHABLE_KEY missing from .env")
    sys.exit(1)

print(f"Supabase: {supabase_url}")

# Test connection
try:
    url = f"{supabase_url}/rest/v1/candles_m1?select=timestamp,close&pair=eq.EUR%2FUSD&limit=1"
    req = urllib.request.Request(url, headers={
        "apikey": supabase_key,
        "Authorization": f"Bearer {supabase_key}",
        "Content-Type": "application/json"
    })
    with urllib.request.urlopen(req, timeout=15) as r:
        data = json.loads(r.read())
        print(f"Connection OK — got {len(data)} candle")
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)

# Clear local candles
from fusion.store import LocalStore
local = LocalStore(ROOT / "var")
local.purge("candles_m1")
local.purge("signals")
local.purge("settlements")
print("Local store cleared")

# Sync candles in batches
PAIRS = ["EUR/USD", "BTC/USD", "ETH/USD"]
BATCH_SIZE = 1000

for pair in PAIRS:
    total = 0
    offset = 0
    while True:
        pair_encoded = urllib.parse.quote(pair, safe="")
        url = f"{supabase_url}/rest/v1/candles_m1?select=timestamp,open,high,low,close,volume&pair=eq.{pair_encoded}&order=timestamp.asc&limit={BATCH_SIZE}&offset={offset}"
        req = urllib.request.Request(url, headers={
            "apikey": supabase_key,
            "Authorization": f"Bearer {supabase_key}",
            "Content-Type": "application/json"
        })
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                rows = json.loads(r.read())
        except Exception as e:
            print(f"  {pair} batch {offset}: ERROR {e}")
            break
        if not rows:
            break
        # Convert timestamp to ISO format
        for row in rows:
            ts = row.get("timestamp", "")
            if ts and "+" not in ts and "Z" not in ts:
                ts = ts.replace(" ", "T")
                if "." in ts:
                    ts = ts.split(".")[0] + "+00:00"
                else:
                    ts = ts + "+00:00"
            row["timestamp"] = ts
        local.write("candles_m1", rows)
        total += len(rows)
        offset += BATCH_SIZE
        print(f"  {pair}: synced {total} candles (batch {offset})", flush=True)
        time.sleep(0.1)
    print(f"{pair}: DONE — {total} candles", flush=True)

print("\n--- CANDLES SYNCED ---")

# Now run the nightly job on real candles
from fusion.config import load_config
from fusion.runner import FusionBot
from fusion.store import build_store

config = load_config(ROOT / ".env")
config.runner = "nightly_real"
config.candle_history = 750
store = build_store(config)

bot = FusionBot(config=config, store=store, telegram=None)
bot.startup()

# Check candles loaded
print(f"  {config.pair}: {len(bot.candles)} candles loaded")
for key, peer in (bot.peer_bots.items() if hasattr(bot, 'peer_bots') else []):
    print(f"  {peer.config.pair}: {len(peer.candles)} candles loaded")

# Run nightly job
now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
print(f"\nRunning nightly job at {now}...")
report = bot.nightly_job(now)

print("\n=== NIGHTLY REPORT ===")
print(json.dumps(report, indent=2, default=str)[:3000])
