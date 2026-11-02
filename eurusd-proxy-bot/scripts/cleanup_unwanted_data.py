#!/usr/bin/env python3
"""
Automatically delete unwanted data from Supabase.

Purges crypto (BTC/USD, ETH/USD) candle rows and any associated signals
from the candles_m1 table, since crypto instruments are no longer
supported by the trading system.

Can be run locally or from GitHub Actions:

    python3 scripts/cleanup_unwanted_data.py

Environment variables:
    SUPABASE_URL: Your Supabase project URL
    SUPABASE_SERVICE_ROLE_KEY: Your Supabase service role key
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fusion.config import load_config  # noqa: E402
from fusion.store import build_store  # noqa: E402
from fusion.instruments import symbols  # noqa: E402


# Pairs to purge — anything that is NOT in the allowed symbols list
UNWANTED_PAIRS = ["BTC/USD", "ETH/USD"]


def load_supabase_config():
    """Get Supabase configuration from environment variables."""
    supabase_url = os.environ.get("SUPABASE_URL", "").rstrip("/")
    supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")

    if not supabase_url or not supabase_key:
        # Fall back to .env file
        from fusion.store import SupabaseStore
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
        print("Set them as environment variables or in .env file.")
        sys.exit(1)

    return supabase_url, supabase_key


def cleanup_candles(store, supabase, unwanted_pairs: list[str]) -> dict:
    """Delete unwanted candle rows from candles_m1 table."""
    results = {}
    for pair in unwanted_pairs:
        # Count before delete
        count_before = supabase.count("candles_m1", filters={"pair": f"eq.{pair}"})
        if count_before is None:
            # Fallback: try paged select
            rows = supabase.select("candles_m1", filters={"pair": f"eq.{pair}"}, limit=1)
            count_before = len(rows) if rows else 0

        if count_before == 0:
            results[pair] = {"count_before": 0, "deleted": 0, "status": "already clean"}
            continue

        # Delete in batches to avoid PostgREST timeout on large tables
        batch_size = 1000
        total_deleted = 0
        remaining = count_before

        while remaining > 0:
            # Get one batch of IDs to delete
            batch = supabase.select(
                "candles_m1",
                filters={"pair": f"eq.{pair}"},
                order="timestamp.asc",
                limit=batch_size,
            )
            if not batch:
                break

            # Delete by timestamp range (first and last in batch)
            first_ts = batch[0].get("timestamp")
            last_ts = batch[-1].get("timestamp")

            deleted = supabase.delete(
                "candles_m1",
                {
                    "pair": f"eq.{pair}",
                    "timestamp": f"gte.{first_ts},lte.{last_ts}",
                }
            )
            total_deleted += deleted
            remaining -= len(batch)

            print(f"  Deleting batch: {len(batch)} rows (deleted={deleted})")

        results[pair] = {
            "count_before": count_before,
            "deleted": total_deleted,
            "status": "purged" if total_deleted > 0 else "already clean",
        }

    return results


def cleanup_signals(store, supabase, unwanted_pairs: list[str]) -> dict:
    """Delete unwanted signal rows from signals table."""
    results = {}
    for pair in unwanted_pairs:
        try:
            count_before = supabase.count("signals", filters={"pair": f"eq.{pair}"})
            if count_before is None:
                rows = supabase.select("signals", filters={"pair": f"eq.{pair}"}, limit=1)
                count_before = len(rows) if rows else 0

            if count_before == 0:
                results[pair] = {"count_before": 0, "deleted": 0, "status": "already clean"}
                continue

            deleted = supabase.delete("signals", {"pair": f"eq.{pair}"})
            results[pair] = {
                "count_before": count_before,
                "deleted": deleted,
                "status": "purged" if deleted > 0 else "already clean",
            }
        except Exception as e:
            # signals table might not have a pair column or might not exist
            results[pair] = {"count_before": 0, "deleted": 0, "status": f"skipped ({e})"}

    return results


def cleanup_settlements(store, supabase, unwanted_pairs: list[str]) -> dict:
    """Delete unwanted settlement rows from settlements table."""
    results = {}
    for pair in unwanted_pairs:
        try:
            count_before = supabase.count("settlements", filters={"pair": f"eq.{pair}"})
            if count_before is None:
                rows = supabase.select("settlements", filters={"pair": f"eq.{pair}"}, limit=1)
                count_before = len(rows) if rows else 0

            if count_before == 0:
                results[pair] = {"count_before": 0, "deleted": 0, "status": "already clean"}
                continue

            deleted = supabase.delete("settlements", {"pair": f"eq.{pair}"})
            results[pair] = {
                "count_before": count_before,
                "deleted": deleted,
                "status": "purged" if deleted > 0 else "already clean",
            }
        except Exception as e:
            results[pair] = {"count_before": 0, "deleted": 0, "status": f"skipped ({e})"}

    return results


def main() -> int:
    allowed = symbols()
    print(f"Allowed symbols: {allowed}")

    # Determine what's unwanted (not in allowed list)
    unwanted = [p for p in UNWANTED_PAIRS if p not in allowed]

    if not unwanted:
        print("No unwanted pairs to clean up.")
        return 0

    print(f"Unwanted pairs to purge: {unwanted}")
    print()

    supabase_url, supabase_key = load_supabase_config()
    print(f"Supabase URL: {supabase_url}")
    print()

    # Build store with Supabase
    from fusion.store import SupabaseStore

    supabase = SupabaseStore(supabase_url, supabase_key, timeout=30.0)
    store = build_store(load_config(ROOT / ".env"))

    # Use the Supabase store directly
    if store.supabase is not None:
        supabase = store.supabase

    print("=" * 60)
    print("PURGING CRYPTIC CANDLE DATA FROM SUPABASE")
    print("=" * 60)
    print()

    # Clean candles
    print("Cleaning candles_m1 table...")
    candle_results = cleanup_candles(store, supabase, unwanted)
    for pair, result in candle_results.items():
        print(f"  {pair}: {result['count_before']} rows → deleted {result['deleted']} ({result['status']})")
    print()

    # Clean signals
    print("Cleaning signals table...")
    signal_results = cleanup_signals(store, supabase, unwanted)
    for pair, result in signal_results.items():
        print(f"  {pair}: {result['count_before']} rows → deleted {result['deleted']} ({result['status']})")
    print()

    # Clean settlements
    print("Cleaning settlements table...")
    settlement_results = cleanup_settlements(store, supabase, unwanted)
    for pair, result in settlement_results.items():
        print(f"  {pair}: {result['count_before']} rows → deleted {result['deleted']} ({result['status']})")
    print()

    # Summary
    total_deleted = sum(r["deleted"] for r in candle_results.values()) + \
                     sum(r["deleted"] for r in signal_results.values()) + \
                     sum(r["deleted"] for r in settlement_results.values())

    print("=" * 60)
    print("CLEANUP COMPLETE")
    print("=" * 60)
    print(f"Total rows deleted: {total_deleted}")
    print()

    # Save summary
    import json
    summary_file = ROOT / "var" / "cleanup_summary.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "cleaned_pairs": unwanted,
        "candles": candle_results,
        "signals": signal_results,
        "settlements": settlement_results,
        "total_deleted": total_deleted,
        "cleaned_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
    }
    summary_file.write_text(json.dumps(summary, indent=2))
    print(f"Summary saved to: {summary_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
