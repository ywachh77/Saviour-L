#!/usr/bin/env python3
"""Import historical candles so the model has something to learn from.

    BACKFILL=1 BACKFILL_MINUTES=20000 python scripts/backfill.py

Deliberately NOT a Telegram command: a backfill spends a lot of provider
credits, so it is an explicit, environment-flagged action you run on purpose.

Twelve Data returns history in pages; each page costs credits from your key
pool, and the pool stops honestly when the day's quota is gone.
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
    print(f"Backfilling up to {minutes} one-minute candles for {config.pair}...")

    total = 0
    remaining = minutes
    end = None
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
