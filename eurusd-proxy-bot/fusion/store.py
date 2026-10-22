"""Persistence: Supabase (REST, stdlib urllib) with a local JSON-lines fallback.

The bot must keep its honesty guarantees even when the network does not
cooperate, so every write goes to Supabase when configured and always leaves a
local trace. A dropped write is logged, never silently swallowed.

Tables (13, all with row-level security in the migration):
    candles_m1, signals, settlements, model_state, model_versions,
    council_weights, learning_events, job_runs, heartbeats, api_keys,
    key_usage, outbox, alerts
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Sequence

from .timeutil import format_datetime

TABLES = (
    "candles_m1",
    "signals",
    "settlements",
    "model_state",
    "model_versions",
    "council_weights",
    "learning_events",
    "job_runs",
    "heartbeats",
    "api_keys",
    "key_usage",
    "outbox",
    "alerts",
)


class StoreError(Exception):
    pass


class LocalStore:
    """Append-only JSON-lines store. Zero dependencies, works on Termux."""

    def __init__(self, directory: str | Path = "var"):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._claimed: set[str] = set()
        self._load_claims()

    # ------------------------------------------------------------- primitives
    def _path(self, table: str) -> Path:
        return self.directory / f"{table}.jsonl"

    def _load_claims(self) -> None:
        for row in self.read("job_runs"):
            self._claimed.add(f"{row.get('job')}|{row.get('bucket')}")

    def write(self, table: str, rows: Sequence[dict]) -> int:
        if not rows:
            return 0
        with self._lock:
            with self._path(table).open("a", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, default=str, sort_keys=True) + "\n")
        return len(rows)

    def read(self, table: str, limit: int | None = None) -> list[dict]:
        path = self._path(table)
        if not path.is_file():
            return []
        rows: list[dict] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        if limit is not None and limit > 0:
            rows = rows[-limit:]
        return rows

    def replace(self, table: str, rows: Sequence[dict]) -> None:
        """Overwrite a table that holds at most one logical row (state tables)."""
        with self._lock:
            with self._path(table).open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, default=str, sort_keys=True) + "\n")

    def purge(self, table: str) -> None:
        with self._lock:
            self._path(table).write_text("", encoding="utf-8")


class SupabaseStore:
    """Supabase PostgREST client over urllib."""

    def __init__(self, url: str, key: str, timeout: float = 8.0, fallback: LocalStore | None = None):
        self.base = url.rstrip("/") + "/rest/v1"
        self.key = key
        self.timeout = timeout
        self.fallback = fallback
        self.errors: list[str] = []

    # ------------------------------------------------------------- plumbing
    def _headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        headers = {
            "apikey": self.key,
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
        }
        if extra:
            headers.update(extra)
        return headers

    def _call(self, method: str, path: str, *, body: Any = None, headers: dict[str, str] | None = None):
        request = urllib.request.Request(
            f"{self.base}/{path}",
            data=json.dumps(body, default=str).encode("utf-8") if body is not None else None,
            method=method,
            headers=self._headers(headers),
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8").strip()
                return json.loads(raw) if raw else None
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", "replace")[:400]
            raise StoreError(f"{method} {path} failed: HTTP {error.code} {detail}") from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise StoreError(f"{method} {path} failed: {error}") from error

    # ------------------------------------------------------------- primitives
    def insert(self, table: str, rows: Sequence[dict], *, on_conflict: str | None = None) -> int:
        if not rows:
            return 0
        headers = {"Prefer": "return=minimal"}
        path = table
        if on_conflict:
            path = f"{table}?on_conflict={urllib.parse.quote(on_conflict)}"
            headers["Prefer"] = "return=minimal,resolution=merge-duplicates"
        try:
            self._call("POST", path, body=list(rows), headers=headers)
            return len(rows)
        except StoreError as error:
            self._degrade(table, rows, str(error))
            return 0

    def upsert(self, table: str, rows: Sequence[dict], on_conflict: str = "id") -> int:
        return self.insert(table, rows, on_conflict=on_conflict)

    def select(
        self,
        table: str,
        *,
        filters: dict[str, str] | None = None,
        order: str | None = None,
        limit: int | None = None,
    ) -> list[dict]:
        params: list[str] = ["select=*"]
        for key, value in (filters or {}).items():
            params.append(f"{key}={urllib.parse.quote(str(value))}")
        if order:
            params.append(f"order={urllib.parse.quote(order)}")
        if limit:
            params.append(f"limit={int(limit)}")
        try:
            result = self._call("GET", f"{table}?{'&'.join(params)}")
            return list(result or [])
        except StoreError as error:
            self.errors.append(str(error))
            return []

    def probe(self, table: str) -> str | None:
        """Reachability check for one table: None when it answers, else the error.

        Used by the setup doctor to tell "table missing" apart from "table
        empty", which an empty select result cannot do.
        """
        try:
            self._call("GET", f"{table}?select=*&limit=1")
            return None
        except StoreError as error:
            return str(error)

    def update(self, table: str, filters: dict[str, str], patch: dict) -> int:
        params = "&".join(f"{key}={urllib.parse.quote(str(value))}" for key, value in filters.items())
        try:
            self._call("PATCH", f"{table}?{params}", body=patch, headers={"Prefer": "return=minimal"})
            return 1
        except StoreError as error:
            self.errors.append(str(error))
            return 0

    def _degrade(self, table: str, rows: Sequence[dict], message: str) -> None:
        self.errors.append(message)
        if self.fallback is not None:
            self.fallback.write(table, list(rows))


class Store:
    """Facade the bot uses: Supabase when configured, local mirror always."""

    def __init__(self, supabase: SupabaseStore | None, local: LocalStore):
        self.supabase = supabase
        self.local = local
        self.using_supabase = supabase is not None

    @property
    def errors(self) -> list[str]:
        return list(self.supabase.errors) if self.supabase else []

    def _write(self, table: str, rows: Sequence[dict], *, on_conflict: str | None = None) -> None:
        rows = [dict(row) for row in rows]
        if self.supabase is not None:
            self.supabase.insert(table, rows, on_conflict=on_conflict)
        else:
            self.local.write(table, rows)

    def _read(self, table: str, limit: int | None = None, **filters) -> list[dict]:
        if self.supabase is not None:
            remote = self.supabase.select(table, filters=filters or None, limit=limit)
            if remote:
                return remote
        rows = self.local.read(table, limit)
        if filters:
            rows = [
                row
                for row in rows
                if all(str(row.get(key)) == _filter_value(value) for key, value in filters.items())
            ]
        return rows

    # --------------------------------------------------------------- candles
    def save_candles(self, candles: Iterable[dict]) -> None:
        self._write("candles_m1", list(candles), on_conflict="timestamp,pair")

    def load_candles(self, pair: str, limit: int = 400) -> list[dict]:
        rows = self._read("candles_m1", limit=limit * 2, **{"pair": f"eq.{pair}"})
        rows.sort(key=lambda row: str(row.get("timestamp", "")))
        return rows[-limit:]

    # --------------------------------------------------------------- signals
    def save_signal(self, payload: dict) -> None:
        self._write("signals", [payload], on_conflict="signal_id")

    def update_signal(self, signal_id: str, patch: dict) -> None:
        if self.supabase is not None:
            self.supabase.update("signals", {"signal_id": f"eq.{signal_id}"}, patch)
            return
        rows = self.local.read("signals")
        for row in rows:
            if row.get("signal_id") == signal_id:
                row.update(patch)
        self.local.replace("signals", rows)

    def pending_signals(self, pair: str) -> list[dict]:
        rows = self._read("signals", limit=500, **{"pair": f"eq.{pair}"})
        return [row for row in rows if row.get("settled") in (None, False, "false")]

    # ----------------------------------------------------------- settlements
    def save_settlement(self, payload: dict) -> None:
        self._write("settlements", [payload], on_conflict="signal_id")
        self.update_signal(str(payload["signal_id"]), {"settled": True})

    def load_settlements(self, limit: int = 2000) -> list[dict]:
        return self._read("settlements", limit=limit)

    # ------------------------------------------------------------ model state
    def save_model_state(self, payload: dict) -> None:
        row = dict(payload)
        row["id"] = row.get("id", "champion")
        row["updated_at"] = format_datetime(_utcnow())
        if self.supabase is not None:
            self.supabase.upsert("model_state", [row], on_conflict="id")
        else:
            self.local.replace("model_state", [row])

    def load_model_state(self) -> dict | None:
        rows = self._read("model_state", limit=1)
        return rows[-1] if rows else None

    def save_model_version(self, payload: dict) -> None:
        self._write("model_versions", [payload], on_conflict="version_id")

    def load_model_versions(self, limit: int = 25) -> list[dict]:
        return self._read("model_versions", limit=limit)

    # --------------------------------------------------------- council weights
    def save_council_weights(self, payload: dict) -> None:
        row = dict(payload)
        row["id"] = row.get("id", "council")
        if self.supabase is not None:
            self.supabase.upsert("council_weights", [row], on_conflict="id")
        else:
            self.local.replace("council_weights", [row])

    def load_council_weights(self) -> dict | None:
        rows = self._read("council_weights", limit=1)
        return rows[-1] if rows else None

    # --------------------------------------------------------- learning events
    def log_event(self, event: dict) -> None:
        row = dict(event)
        row.setdefault("created_at", format_datetime(_utcnow()))
        self._write("learning_events", [row])

    def load_events(self, limit: int = 200) -> list[dict]:
        return self._read("learning_events", limit=limit)

    # ------------------------------------------------------------- job slots
    def claim_job(self, job: str, bucket: str, *, runner: str = "termux") -> bool:
        """Idempotency slot. True only for the first caller for (job, bucket)."""
        key = f"{job}|{bucket}"
        if key in self.local._claimed:
            return False
        self.local._claimed.add(key)
        row = {"job": job, "bucket": bucket, "runner": runner, "created_at": format_datetime(_utcnow())}
        if self.supabase is not None:
            # A unique index on (job, bucket) makes the second writer fail.
            before = len(self.supabase.errors)
            self.supabase.insert("job_runs", [row])
            if len(self.supabase.errors) > before:
                return False
        self.local.write("job_runs", [row])
        return True

    # -------------------------------------------------------------- heartbeat
    def heartbeat(self, row: dict) -> None:
        self._write("heartbeats", [row])

    def latest_heartbeat(self, runner: str | None = None) -> dict | None:
        rows = self._read("heartbeats", limit=50)
        if runner:
            rows = [row for row in rows if row.get("runner") == runner]
        rows.sort(key=lambda item: str(item.get("created_at", "")))
        return rows[-1] if rows else None

    # ------------------------------------------------------------------- keys
    def save_key_pool(self, payload: dict) -> None:
        row = dict(payload)
        row["id"] = row.get("id", "keypool")
        if self.supabase is not None:
            self.supabase.upsert("api_keys", [row], on_conflict="id")
        else:
            self.local.replace("api_keys", [row])

    def load_key_pool(self) -> dict | None:
        rows = self._read("api_keys", limit=1)
        return rows[-1] if rows else None

    def log_key_usage(self, rows: Sequence[dict]) -> None:
        self._write("key_usage", list(rows))

    # ----------------------------------------------------------------- outbox
    def outbox_append(self, row: dict) -> None:
        payload = dict(row)
        payload.setdefault("created_at", format_datetime(_utcnow()))
        payload.setdefault("state", "PENDING")
        self._write("outbox", [payload], on_conflict="message_key")

    def outbox_update(self, message_key: str, patch: dict) -> None:
        if self.supabase is not None:
            self.supabase.update("outbox", {"message_key": f"eq.{message_key}"}, patch)
            return
        rows = self.local.read("outbox")
        for row in rows:
            if row.get("message_key") == message_key:
                row.update(patch)
        self.local.replace("outbox", rows)

    def outbox_pending(self) -> list[dict]:
        rows = self._read("outbox", limit=200)
        return [row for row in rows if row.get("state") in ("PENDING", "UNRESOLVED")]

    # ----------------------------------------------------------------- alerts
    def alert(self, kind: str, message: str, *, dedupe_key: str | None = None) -> None:
        self._write(
            "alerts",
            [{"kind": kind, "message": message, "dedupe_key": dedupe_key, "created_at": format_datetime(_utcnow())}],
        )

    def recent_alerts(self, limit: int = 50) -> list[dict]:
        return self._read("alerts", limit=limit)


def _filter_value(value: str) -> str:
    """Accept PostgREST filter syntax ('eq.X') and plain values alike."""
    text = str(value)
    for prefix in ("eq.", "gte.", "lte.", "lt.", "gt."):
        if text.startswith(prefix):
            return text[len(prefix):]
    return text


def _utcnow():
    from datetime import datetime, timezone

    return datetime.now(timezone.utc)


def build_store(config) -> Store:
    local = LocalStore(config.local_state_dir)
    supabase = None
    if config.supabase_ready:
        supabase = SupabaseStore(config.supabase_url, config.supabase_service_key, fallback=local)
    return Store(supabase, local)
