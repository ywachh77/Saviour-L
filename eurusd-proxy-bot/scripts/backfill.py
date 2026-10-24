#!/usr/bin/env python3
"""Import historical candles so the model has something to learn from.

    BACKFILL=1 BACKFILL_MINUTES=20000 python scripts/backfill.py

Deliberately NOT a Telegram command: a backfill spends a lot of provider
credits, so it is an explicit, environment-flagged action you run on purpose.

Twelve Data returns history in pages; each page costs credits from your key
pool, and the pool stops honestly when the day's quota is gone.

Incremental and safe on a schedule. Each run extends history backwards from
the earliest bar already stored rather than re-fetching the recent window, and
BACKFILL_TARGET_BARS stops the whole thing once there is enough -- so a daily
cron spends a bounded number of credits and then goes quiet instead of paying
for the same bars every day forever.

    BACKFILL_MINUTES        how many further bars to fetch this run (max 200000)
    BACKFILL_TARGET_BARS    stop once the table holds this many (0 = no target)
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fusion.config import load_config  # noqa: E402
from fusion.domain import Candle  # noqa: E402
from fusion.provider import TwelveDataProvider  # noqa: E402
from fusion.store import build_store  # noqa: E402


def main() -> int:
    import os

    if os.environ.get("BACKFILL", "0").strip().lower() not in ("1", "true", "yes"):
        print("Refusing to run: set BACKFILL=1 to confirm you want to spend provider credits.")
        return 2

    config = load_config(ROOT / ".env")
    if not config.provider_ready:
        print("No Twelve Data key configured. Add one to .env first.")
        return 2

    try:
        minutes = int(os.environ.get("BACKFILL_MINUTES", "10000"))
    except ValueError:
        minutes = 10000
    minutes = max(60, min(minutes, 200_000))

    store = build_store(config)
    provider = TwelveDataProvider(config.twelve_data_keys, base_url=config.provider_base_url)
    target = int(os.environ.get("BACKFILL_TARGET_BARS") or os.environ.get("BACKFILL_TARGET") or 0)
    have = store.count_candles(config.pair)
    if target and have >= target:
        print(f"Already holding {have} bars for {config.pair}, target is {target}. "
              "Nothing to do, and no credits spent.")
        return 0
    if target:
        print(f"Holding {have} bars for {config.pair}; target is {target}.")

    print(f"Fetching up to {minutes} further one-minute candles for {config.pair}...")

    # Walk backwards from the earliest bar we already hold. Starting from "now"
    # every time would re-buy the same history on each scheduled run.
    oldest = store.oldest_candle(config.pair)
    if oldest:
        print(f"History already starts at {oldest}; extending further back.")
    else:
        print("No history yet; starting from now and walking back.")

    total = 0
    remaining = minutes
    end = oldest
    while remaining > 0:
        size = min(5000, remaining)
        params = {
            "symbol": config.pair,
            "interval": config.timeframe,
            "outputsize": str(size),
            "order": "ASC",
            "timezone": "UTC",
            "format": "JSON",
        }
        if end is not None:
            params["end_date"] = end
        try:
            payload = provider._request("time_series", params)
            values = payload.get("values") or []
        except Exception as error:  # noqa: BLE001 - a backfill must stop cleanly
            print(f"Stopped: {error}")
            break
        if not values:
            break
        rows = []
        for raw in values:
            try:
                candle = Candle.from_mapping(raw)
            except (ValueError, KeyError, TypeError):
                continue
            rows.append(dict(candle.to_dict(), pair=config.pair))
        if not rows:
            break
        store.save_candles(rows)
        total += len(rows)
        remaining -= size
        end = min(row["timestamp"] for row in rows)
        print(f"  stored {total} candles (walking back to {end})")
        if len(values) < size:
            break

    print(f"Done. {total} candles stored for {config.pair}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
