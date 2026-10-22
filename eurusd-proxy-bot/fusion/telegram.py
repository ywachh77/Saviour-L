"""Telegram transport: owner-only delivery with honest delivery states.

The state machine is deliberately strict:

* ``SENT`` is set only when Telegram returns a message id;
* a network timeout after the request may have landed is ``UNRESOLVED``, not
  "probably sent" — the bot records it and tells the human;
* a clear API rejection is ``FAILED``.

Nothing is ever reported to the operator as delivered on the basis of a
successful HTTP request alone.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from . import charter

COMMANDS = ("/start", "/help", "/status", "/pause", "/resume")

HELP_TEXT = (
    "FUSION BOT — EUR/USD M1 research signals (paper only).\n"
    "\n"
    "Commands:\n"
    "/start — begin receiving signals\n"
    "/help — this message\n"
    "/status — health, ledger and learning state\n"
    "/pause — stop signals (bot keeps learning)\n"
    "/resume — resume signals\n"
    "\n"
    "Everything else is automatic: ingest, signals, settlement, online learning, "
    "nightly walk-forward, and a daily digest at 00:05 UTC.\n"
    "\n"
    "This is research output, not investment advice. Signals are cost-charged paper "
    "settlements; no broker order is ever placed."
)


@dataclass(frozen=True, slots=True)
class SendResult:
    ok: bool
    state: str
    message_id: str | None = None
    error: str = ""

    @property
    def confirmed(self) -> bool:
        return self.ok and self.state == charter.DELIVERY_SENT


@dataclass
class IncomingCommand:
    update_id: int
    chat_id: str
    command: str
    text: str = ""
    from_owner: bool = False


def _error_detail(error) -> str:
    """Read an HTTP error body whether it arrives as bytes, str or nothing."""
    try:
        body = error.read()
    except Exception:  # noqa: BLE001 - some transports expose no body at all
        return ""
    if isinstance(body, bytes):
        body = body.decode("utf-8", "replace")
    return str(body or "")[:300]


class TelegramClient:
    def __init__(
        self,
        token: str,
        owner_chat_id: str,
        *,
        timeout: float = 10.0,
        opener=urllib.request.urlopen,
    ):
        self.token = token.strip()
        self.owner_chat_id = str(owner_chat_id).strip()
        self.timeout = timeout
        self._opener = opener
        self.offset: int | None = None

    @property
    def configured(self) -> bool:
        return bool(self.token and self.owner_chat_id)

    def _call(self, method: str, payload: dict[str, Any]) -> tuple[dict | None, str]:
        url = f"https://api.telegram.org/bot{self.token}/{method}"
        data = urllib.parse.urlencode(payload).encode("utf-8")
        request = urllib.request.Request(url, data=data, headers={"User-Agent": "fusion-bot/2.0"})
        try:
            with self._opener(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
            return json.loads(body), ""
        except urllib.error.HTTPError as error:
            return None, f"HTTP {error.code}: {_error_detail(error)}"
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            return None, f"network: {error}"
        except json.JSONDecodeError as error:
            return None, f"bad json: {error}"

    # ---------------------------------------------------------------- sending
    def send(self, text: str, *, chat_id: str | None = None) -> SendResult:
        if not self.configured:
            return SendResult(False, charter.DELIVERY_FAILED, error="Telegram is not configured")
        payload = {
            "chat_id": chat_id or self.owner_chat_id,
            "text": text[:3900],
            "disable_web_page_preview": "true",
        }
        result, error = self._call("sendMessage", payload)
        if error:
            # A timeout means the message may or may not have been delivered.
            state = charter.DELIVERY_FAILED if error.startswith("HTTP") else charter.DELIVERY_UNKNOWN
            return SendResult(False, state, error=error)
        if not isinstance(result, dict) or not result.get("ok"):
            return SendResult(False, charter.DELIVERY_FAILED, error=str(result))
        message_id = (result.get("result") or {}).get("message_id")
        if message_id is None:
            return SendResult(False, charter.DELIVERY_UNKNOWN, error="Telegram returned no message id")
        return SendResult(True, charter.DELIVERY_SENT, message_id=str(message_id))

    # -------------------------------------------------------------- receiving
    def poll_commands(self, timeout_seconds: int = 0) -> list[IncomingCommand]:
        """Fetch pending updates and return owner commands only."""
        if not self.configured:
            return []
        payload: dict[str, Any] = {"timeout": int(timeout_seconds)}
        if self.offset is not None:
            payload["offset"] = self.offset
        result, error = self._call("getUpdates", payload)
        if error or not isinstance(result, dict) or not result.get("ok"):
            return []

        commands: list[IncomingCommand] = []
        for update in result.get("result") or []:
            update_id = int(update.get("update_id", 0))
            self.offset = max(self.offset or 0, update_id + 1)
            message = update.get("message") or update.get("edited_message") or {}
            text = str(message.get("text", "")).strip()
            if not text.startswith("/"):
                continue
            chat_id = str(((message.get("chat") or {}).get("id")) or "")
            command = text.split()[0].split("@")[0].lower()
            commands.append(
                IncomingCommand(
                    update_id=update_id,
                    chat_id=chat_id,
                    command=command,
                    text=text,
                    from_owner=bool(chat_id) and chat_id == self.owner_chat_id,
                )
            )
        return commands


@dataclass
class OutboxItem:
    """One message waiting to be delivered, with its honesty state attached."""

    message_key: str
    text: str
    kind: str = "signal"
    attempts: int = 0
    state: str = charter.DELIVERY_PENDING
    message_id: str | None = None
    created_at: str = ""
    last_error: str = ""

    def to_dict(self) -> dict:
        return {
            "message_key": self.message_key,
            "text": self.text,
            "kind": self.kind,
            "attempts": self.attempts,
            "state": self.state,
            "message_id": self.message_id,
            "created_at": self.created_at,
            "last_error": self.last_error,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "OutboxItem":
        return cls(
            message_key=str(payload.get("message_key", "")),
            text=str(payload.get("text", "")),
            kind=str(payload.get("kind", "signal")),
            attempts=int(payload.get("attempts", 0)),
            state=str(payload.get("state", charter.DELIVERY_PENDING)),
            message_id=payload.get("message_id"),
            created_at=str(payload.get("created_at", "")),
            last_error=str(payload.get("last_error", "")),
        )
