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
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from . import charter

COMMANDS = ("/start", "/help", "/signals", "/status", "/pause", "/resume")

# A long poll has to be allowed to outlive the window it is waiting on. The
# socket timeout used to be ten seconds while the poll window was twenty, so
# every getUpdates call was killed by its own client before Telegram could
# answer. The margin covers TLS setup and slow responses.
POLL_TIMEOUT_MARGIN_SECONDS = 10.0

HELP_TEXT = (
    "FUSION BOT — EUR/USD M1 research signals (paper only).\n"
    "\n"
    "Commands:\n"
    "/start — begin receiving signals\n"
    "/help — this message\n"
    "/signals — read the market now and answer at once\n"
    "/status — health, ledger and learning state\n"
    "/pause — stop signals (bot keeps learning)\n"
    "/resume — resume signals\n"
    "\n"
    "Everything else is automatic: ingest, signals, settlement, online learning, "
    "nightly walk-forward, and a daily digest at 00:05 UTC.\n\n"
    "/signals runs the same gates as the scheduled cycle. When they decline it "
    "says so, because a signal produced on demand would not be a prediction.\n"
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
        # Why a poll failed, so the caller can record it instead of the
        # transport silently reporting "nothing to do".
        self.last_poll_error: str | None = None
        # Commands that arrived from someone who is not the owner. Ignored, but
        # remembered: a bot that cannot hear the owner and a bot whose owner id
        # is wrong look identical from the outside unless this is recorded.
        self.ignored_senders: list[dict[str, str]] = []
        self.webhook_cleared = False
        self._webhook_recovery_tried = False

    @property
    def configured(self) -> bool:
        return bool(self.token and self.owner_chat_id)

    def _call(
        self,
        method: str,
        payload: dict[str, Any],
        *,
        timeout: float | None = None,
    ) -> tuple[dict | None, str]:
        url = f"https://api.telegram.org/bot{self.token}/{method}"
        data = urllib.parse.urlencode(payload).encode("utf-8")
        request = urllib.request.Request(url, data=data, headers={"User-Agent": "fusion-bot/2.0"})
        try:
            with self._opener(request, timeout=timeout if timeout is not None else self.timeout) as response:
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
        """Fetch pending updates and return the commands in them.

        Every caller filters on ``from_owner``; this returns everything so the
        caller can decide what to do with a stranger. Two failures used to be
        invisible because both ended in ``return []``:

        * a transport error. Most damaging is an active webhook, which makes
          getUpdates fail permanently while the bot looks perfectly healthy;
        * a command from the wrong chat id, which is dropped as a stranger.

        ``last_poll_error`` and ``ignored_senders`` exist so the caller can
        record both instead of reporting silence.
        """
        self.last_poll_error = None
        self.ignored_senders = []
        if not self.configured:
            return []
        payload: dict[str, Any] = {"timeout": int(timeout_seconds)}
        if self.offset is not None:
            payload["offset"] = self.offset
        # The socket has to stay open longer than the poll window, or the
        # client kills the request before Telegram can answer it.
        poll_timeout = max(float(self.timeout), float(timeout_seconds) + POLL_TIMEOUT_MARGIN_SECONDS)
        result, error = self._call("getUpdates", payload, timeout=poll_timeout)
        if not error and isinstance(result, dict) and not result.get("ok"):
            # Telegram answers a webhook conflict with HTTP 200 and ok:false,
            # not a transport error, so the description has to be read as a
            # failure reason too or the conflict is never noticed.
            error = str(result.get("description") or result)[:300]
        if error and self._looks_like_webhook_conflict(error):
            if self._clear_webhook():
                result, error = self._call("getUpdates", payload, timeout=poll_timeout)
        if error or not isinstance(result, dict) or not result.get("ok"):
            self.last_poll_error = (error or str(result))[:300]
            return []
        updates = result.get("result")
        if not isinstance(updates, list):
            # Guard the contract rather than iterating whatever came back: a
            # dict here used to raise AttributeError inside the loop.
            self.last_poll_error = f"unexpected getUpdates payload: {str(result)[:120]}"
            return []

        commands: list[IncomingCommand] = []
        for update in updates:
            update_id = int(update.get("update_id", 0))
            self.offset = max(self.offset or 0, update_id + 1)
            message = update.get("message") or update.get("edited_message") or {}
            text = str(message.get("text", "")).strip()
            if not text.startswith("/"):
                continue
            chat_id = str(((message.get("chat") or {}).get("id")) or "")
            command = text.split()[0].split("@")[0].lower()
            incoming = IncomingCommand(
                update_id=update_id,
                chat_id=chat_id,
                command=command,
                text=text,
                from_owner=bool(chat_id) and chat_id == self.owner_chat_id,
            )
            if not incoming.from_owner:
                # Remembered, not answered. The chat id is what makes a wrong
                # TELEGRAM_OWNER_CHAT_ID diagnosable from the database.
                self.ignored_senders.append({"chat_id": chat_id, "command": command})
            commands.append(incoming)
        return commands

    # ----------------------------------------------------------------- webhook
    @staticmethod
    def _looks_like_webhook_conflict(error: str) -> bool:
        """True when getUpdates failed because a webhook is active.

        Telegram refuses getUpdates while a webhook is set, and the bot has no
        webhook handler, so in that state it is permanently deaf and reports
        nothing. Recovering automatically is the only way out without asking
        the owner to run a curl command.
        """
        lowered = (error or "").lower()
        return "conflict" in lowered or "webhook" in lowered

    def _clear_webhook(self) -> bool:
        """Drop any webhook once per process, then let polling continue."""
        if self._webhook_recovery_tried:
            return False
        self._webhook_recovery_tried = True
        result, error = self._call("deleteWebhook", {"drop_pending_updates": "false"})
        self.webhook_cleared = not error and isinstance(result, dict) and bool(result.get("ok"))
        if error:
            self.last_poll_error = f"deleteWebhook failed: {error[:200]}"
        return self.webhook_cleared


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
