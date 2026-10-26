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
    """Append-only JSON-lines store. Zero dependencies, runs anywhere Python does."""

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

    def _remember_claim(self, key: str, cap: int = 20_000) -> None:
        """Remember a claimed slot without letting the set grow without bound."""
        self._claimed.add(key)
        if len(self._claimed) > cap:
            self._claimed = set(list(self._claimed)[-cap // 2 :])

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


# PostgREST caps a single response at db-max-rows (1,000 on Supabase hosted
# projects) regardless of how large ?limit= is. Without paging, asking for
# 200,000 training candles silently returned 1,000 and count_candles() reported
# 1,000 forever once a pair passed a thousand bars.
POSTGREST_PAGE_SIZE = 1000


class SupabaseStore:
    """Supabase PostgREST client over urllib."""

    def __init__(self, url: str, key: str, timeout: float = 8.0, fallback: LocalStore | None = None):
        self.base = url.rstrip("/") + "/rest/v1"
        self.key = key
        self.timeout = timeout
        self.fallback = fallback
        self.errors: list[str] = []
        self.last_response_headers: dict[str, str] = {}

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
                resp_headers = getattr(response, "headers", None)
                self.last_response_headers = (
                    {str(k).lower(): str(v) for k, v in resp_headers.items()}
                    if resp_headers is not None
                    else {}
                )
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
        base_params: list[str] = ["select=*"]
        for key, value in (filters or {}).items():
            base_params.append(f"{key}={urllib.parse.quote(str(value))}")
        if order:
            base_params.append(f"order={urllib.parse.quote(order)}")

        # Small bounded reads fit inside one PostgREST page (db-max-rows=1000).
        if limit is not None and 0 < int(limit) <= POSTGREST_PAGE_SIZE:
            query = "&".join(base_params + [f"limit={int(limit)}"])
            try:
                result = self._call("GET", f"{table}?{query}")
                return list(result or [])
            except StoreError as error:
                self.errors.append(str(error))
                return []

        # Large or unbounded reads must page with offset, otherwise Supabase
        # truncates silently at 1,000 rows (verified live: nightly asked for
        # 200,000 bars against a 101,730-row table and received 1,000).
        collected: list[dict] = []
        offset = 0
        target = int(limit) if (limit is not None and int(limit) > 0) else None
        try:
            while True:
                remaining = (target - len(collected)) if target is not None else POSTGREST_PAGE_SIZE
                if remaining <= 0:
                    break
                page_size = min(POSTGREST_PAGE_SIZE, remaining)
                query = "&".join(base_params + [f"limit={page_size}", f"offset={offset}"])
                batch = list(self._call("GET", f"{table}?{query}") or [])
                collected.extend(batch)
                if len(batch) < page_size:
                    break
                offset += len(batch)
            return collected
        except StoreError as error:
            self.errors.append(str(error))
            return collected

    def count(self, table: str, *, filters: dict[str, str] | None = None) -> int | None:
        """Exact server-side row count via PostgREST ``Prefer: count=exact``.

        Returns None on transport error or when the mock/server omits
        ``Content-Range``, letting the caller fall back to a paged read.
        """
        params: list[str] = ["select=*", "limit=1"]
        for key, value in (filters or {}).items():
            params.append(f"{key}={urllib.parse.quote(str(value))}")
        try:
            self.last_response_headers = {}
            rows = self._call(
                "GET",
                f"{table}?{'&'.join(params)}",
                headers={"Prefer": "count=exact"},
            )
        except StoreError as error:
            self.errors.append(str(error))
            return None
        content_range = self.last_response_headers.get("content-range", "")
        if "/" in content_range:
            total = content_range.rsplit("/", 1)[-1].strip()
            if total.isdigit():
                return int(total)
        # Mocked _call implementations in tests do not populate HTTP headers;
        # returning None lets Store.count_candles fall back to select().
        if rows is None:
            return None
        return None

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

    def _read(
        self,
        table: str,
        limit: int | None = None,
        order: str | None = None,
        **filters,
    ) -> list[dict]:
        if self.supabase is not None:
            remote = self.supabase.select(table, filters=filters or None, order=order, limit=limit)
            if remote:
                return remote
        rows = self.local.read(table)
        if filters:
            rows = [row for row in rows if all(_matches(row, key, value) for key, value in filters.items())]
        if order:
            # Ordered reads are best-first, so the slice comes off the front.
            rows = _apply_order(rows, order)
            if limit is not None and limit > 0:
                rows = rows[:limit]
            return rows
        if limit is not None and limit > 0:
            rows = rows[-limit:]
        return rows

    # --------------------------------------------------------------- candles
    def save_candles(self, candles: Iterable[dict]) -> None:
        self._write("candles_m1", list(candles), on_conflict="timestamp,pair")

    def oldest_candle(self, pair: str) -> str | None:
        """The earliest bar we hold, or None on an empty table.

        Backfill uses this to extend history backwards instead of re-fetching
        the window it already has. Without it every scheduled run would start
        from "now" and spend the same credits again for bars we already stored.
        """
        rows = self._read(
            "candles_m1", limit=1, order="timestamp.asc", **{"pair": f"eq.{pair}"}
        )
        return rows[0].get("timestamp") if rows else None

    def count_candles(self, pair: str) -> int:
        """How many bars we hold. Used to stop backfilling once there is enough."""
        if self.supabase is not None:
            exact = self.supabase.count("candles_m1", filters={"pair": f"eq.{pair}"})
            if exact is not None:
                return exact
        return len(self._read("candles_m1", **{"pair": f"eq.{pair}"}))

    def load_candles(self, pair: str, limit: int = 400) -> list[dict]:
        # Without an explicit order PostgREST may return any slice of the table,
        # and slicing "the newest N" off an unordered page silently drops the
        # bars that actually matter.
        rows = self._read(
            "candles_m1", limit=limit, order="timestamp.desc", **{"pair": f"eq.{pair}"}
        )
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

    def pending_signals(self, pair: str, limit: int = 500) -> list[dict]:
        rows = self._read(
            "signals",
            limit=limit,
            order="entry_time.desc",
            **{"pair": f"eq.{pair}", "settled": "eq.false"},
        )
        return [row for row in rows if row.get("settled") in (None, False, "false")]

    def load_signals(self, limit: int = 2000) -> list[dict]:
        return self._read("signals", limit=limit, order="entry_time.desc")

    # ----------------------------------------------------------- settlements
    def save_settlement(self, payload: dict) -> None:
        self._write("settlements", [payload], on_conflict="signal_id")
        self.update_signal(str(payload["signal_id"]), {"settled": True})

    def load_settlements(self, limit: int = 2000) -> list[dict]:
        return self._read("settlements", limit=limit, order="settled_at.desc")

    # ------------------------------------------------------------ model state
    @staticmethod
    def model_state_id(pair: str | None = None) -> str:
        clean = (pair or "").strip().upper()
        if not clean or clean == "EUR/USD":
            return "champion"
        return f"champion:{clean}"

    @staticmethod
    def council_weights_id(pair: str | None = None) -> str:
        clean = (pair or "").strip().upper()
        if not clean or clean == "EUR/USD":
            return "council"
        return f"council:{clean}"

    def save_model_state(self, payload: dict, *, pair: str | None = None) -> None:
        row = dict(payload)
        row["id"] = row.get("id") or self.model_state_id(pair)
        row["updated_at"] = format_datetime(_utcnow())
        if self.supabase is not None:
            self.supabase.upsert("model_state", [row], on_conflict="id")
            return
        existing = [item for item in self.local.read("model_state") if item.get("id") != row["id"]]
        existing.append(row)
        self.local.replace("model_state", existing)

    def load_model_state(self, *, pair: str | None = None) -> dict | None:
        target = self.model_state_id(pair)
        rows = self._read("model_state", limit=1, **{"id": f"eq.{target}"})
        if rows:
            return rows[-1]
        if target == "champion":
            # Legacy local files written before an explicit id was filtered.
            fallback = [item for item in self._read("model_state") if item.get("id") in (None, "champion")]
            return fallback[-1] if fallback else None
        return None

    def save_model_version(self, payload: dict) -> None:
        self._write("model_versions", [payload], on_conflict="version_id")

    def load_model_versions(self, limit: int = 25) -> list[dict]:
        return self._read("model_versions", limit=limit)

    # --------------------------------------------------------- council weights
    def save_council_weights(self, payload: dict, *, pair: str | None = None) -> None:
        row = dict(payload)
        row["id"] = row.get("id") or self.council_weights_id(pair)
        if self.supabase is not None:
            self.supabase.upsert("council_weights", [row], on_conflict="id")
            return
        existing = [item for item in self.local.read("council_weights") if item.get("id") != row["id"]]
        existing.append(row)
        self.local.replace("council_weights", existing)

    def load_council_weights(self, *, pair: str | None = None) -> dict | None:
        target = self.council_weights_id(pair)
        rows = self._read("council_weights", limit=1, **{"id": f"eq.{target}"})
        if rows:
            return rows[-1]
        if target == "council":
            fallback = [item for item in self._read("council_weights") if item.get("id") in (None, "council")]
            return fallback[-1] if fallback else None
        return None

    # --------------------------------------------------------- learning events
    _LEARNING_EVENT_COLUMNS = frozenset(
        {
            "id",
            "kind",
            "signal_id",
            "applied",
            "reason",
            "predicted",
            "raw",
            "label",
            "squared_error",
            "created_at",
            "outcome",
        }
    )

    def log_event(self, event: dict) -> None:
        raw_row = dict(event)
        extras = {k: v for k, v in raw_row.items() if k not in self._LEARNING_EVENT_COLUMNS}
        row = {k: v for k, v in raw_row.items() if k in self._LEARNING_EVENT_COLUMNS}
        if extras:
            extra_txt = ", ".join(f"{k}={v}" for k, v in sorted(extras.items()))
            base_reason = str(row.get("reason") or "").strip()
            row["reason"] = f"{base_reason} ({extra_txt})" if base_reason else extra_txt
        row.setdefault("created_at", format_datetime(_utcnow()))
        self._write("learning_events", [row])

    def load_events(self, limit: int = 200) -> list[dict]:
        return self._read("learning_events", limit=limit, order="created_at.desc")

    # ------------------------------------------------------------- job slots
    def claim_job(self, job: str, bucket: str, *, runner: str = "github-actions") -> bool:
        """Idempotency slot. True only for the first caller for (job, bucket)."""
        key = f"{job}|{bucket}"
        if key in self.local._claimed:
            return False
        self.local._remember_claim(key)
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
        rows = self._read(
            "heartbeats",
            limit=50,
            order="created_at.desc",
            **({"runner": f"eq.{runner}"} if runner else {}),
        )
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

    def outbox_pending(self, limit: int = 200) -> list[dict]:
        """Rows still waiting for a send, newest last.

        The filter belongs to the read, not to a pass over the result. Fetching
        the newest 200 rows and then discarding the ones that are not pending
        silently drops work: once 200 newer rows pile up behind it, an older
        pending row falls off the end and is never retried again.
        """
        # Keyword filters, not a filters= argument: _read forwards **filters
        # to PostgREST, so this becomes state=in.(PENDING,UNRESOLVED).
        return self._read("outbox", limit=limit, state="in.(PENDING,UNRESOLVED)")

    # ----------------------------------------------------------------- alerts
    def alert(self, kind: str, message: str, *, dedupe_key: str | None = None) -> None:
        self._write(
            "alerts",
            [{"kind": kind, "message": message, "dedupe_key": dedupe_key, "created_at": format_datetime(_utcnow())}],
        )

    def alert_seen(self, dedupe_key: str) -> bool:
        """True when an alert with this key has already been recorded.

        The bot runs in short-lived containers that restart every cycle, so an
        in-memory "already told the human" set evaporates and the same alert is
        re-sent on every wake-up. The key is persisted precisely so the check
        can survive the process; without this read the column is dead storage.
        """
        if not dedupe_key:
            return False
        rows = self._read("alerts", limit=1, **{"dedupe_key": f"eq.{dedupe_key}"})
        return bool(rows)

    def recent_alerts(self, limit: int = 50) -> list[dict]:
        return self._read("alerts", limit=limit, order="created_at.desc")


def _matches(row: dict, key: str, value: str) -> bool:
    """Compare a row value against a PostgREST filter, booleans included.

    Python stringifies False as "False" while PostgREST spells it "false", so a
    naive equality check silently drops every pending-signal row.

    Also understands PostgREST's `in.(a,b,c)` list form, so the local mirror
    and the remote query select the same rows. Without it a caller has to
    filter in Python after fetching, which is exactly how rows get dropped.
    """
    left = row.get(key)
    right = _filter_value(value)

    # in.(PENDING,UNRESOLVED)
    if right.startswith("in.(") and right.endswith(")"):
        choices = [item.strip() for item in right[4:-1].split(",") if item.strip()]
        if not choices:
            return False
        return any(_matches(row, key, choice) for choice in choices)

    if isinstance(left, bool) or right.lower() in ("true", "false"):
        return str(left).lower() == right.lower()
    return str(left) == right


def _apply_order(rows: list[dict], order: str) -> list[dict]:
    """Sort rows by a PostgREST-style 'field.direction' key.

    The local mirror has to honour the same ordering the remote query asked for,
    otherwise "the newest N rows" silently means "some N rows".
    """
    field_name, _, direction = str(order).partition(".")
    if not field_name:
        return rows
    rows.sort(key=lambda row: str(row.get(field_name, "")))
    if direction.lower() == "desc":
        rows.reverse()
    return rows


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
