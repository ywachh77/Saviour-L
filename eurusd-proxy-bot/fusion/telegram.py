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
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping

from . import charter, instruments
from .domain import Signal

COMMANDS = ("/start", "/help", "/signals", "/status", "/pause", "/resume")

# A long poll has to be allowed to outlive the window it is waiting on. The
# socket timeout used to be ten seconds while the poll window was twenty, so
# every getUpdates call was killed by its own client before Telegram could
# answer. The margin covers TLS setup and slow responses.
POLL_TIMEOUT_MARGIN_SECONDS = 10.0

HELP_TEXT = (
    "FUSION BOT — FX & crypto M1 research signals (EUR/USD, BTC/USD, ETH/USD · paper only).\n"
    "\n"
    "Commands:\n"
    "/start — begin receiving signals\n"
    "/help — this message\n"
    "/signals — evaluate the upcoming next candle (UTC+05:30) and answer at once\n"
    "/status — health, ledger and learning state\n"
    "/pause — stop signals (bot keeps learning)\n"
    "/resume — resume signals\n"
    "\n"
    "Everything else is automatic: ingest, signals, settlement, online learning, "
    "nightly walk-forward, and a daily digest at 05:35 UTC+05:30.\n\n"
    "/signals runs the same gates as the scheduled cycle. When they decline it "
    "says so, because a signal produced on demand would not be a prediction.\n"
    "\n"
    "All entry, expiry, and system timings are reported in UTC+05:30. "
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
    def poll_commands(
        self,
        timeout_seconds: int = 0,
        *,
        include_plain_text: bool = False,
    ) -> list[IncomingCommand]:
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
            if not text:
                continue
            if not text.startswith("/") and not include_plain_text:
                continue
            chat_id = str(((message.get("chat") or {}).get("id")) or "")
            command = text.split()[0].split("@")[0].lower() if text.startswith("/") else "__needle__"
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


# ------------------------------------------------------------------ formatting
SIGNAL_STYLES = ("standard", "brief", "tactical", "compact", "custom")
RESULT_STYLES = ("standard", "compact", "custom")

DEFAULT_CUSTOM_SIGNAL_TEMPLATE = (
    "{pair} {side} ({horizon}m) #{short_id}\n"
    "Entry: {entry_hm} UTC+05:30  ->  Expiry: {expiry_hm} UTC+05:30\n"
    "Regime: {regime}  |  Model: {probability} ({calibration_state})\n"
    "Gate: {reason}\n"
    "Spread charged: {spread_pips} pips"
)

DEFAULT_CUSTOM_RESULT_TEMPLATE = (
    "RESULT {pair} {side} #{short_id} -> {verdict}\n"
    "Fill: {entry_price} -> {exit_price}\n"
    "Gross {gross_pips}p | Cost {spread_pips}p | Net {net_pips}p"
)

SIGNAL_PLACEHOLDERS: tuple[dict[str, str], ...] = (
    {"token": "{pair}", "desc": "Instrument symbol (EUR/USD, BTC/USD, ETH/USD)"},
    {"token": "{side}", "desc": "Direction (BUY or SELL)"},
    {"token": "{entry_hm}", "desc": "Entry candle HH:MM in UTC+05:30"},
    {"token": "{expiry_hm}", "desc": "Expiry candle HH:MM in UTC+05:30"},
    {"token": "{tz}", "desc": "System display timezone label (UTC+05:30)"},
    {"token": "{horizon}", "desc": "Hold duration in minutes"},
    {"token": "{regime}", "desc": "Market regime (TREND, RANGE, VOLATILE)"},
    {"token": "{probability}", "desc": "Calibrated/raw upward probability or n/a"},
    {"token": "{raw_probability}", "desc": "Raw logistic upward score or n/a"},
    {"token": "{calibration_state}", "desc": "RAW, CALIBRATED, or FLAT"},
    {"token": "{reason}", "desc": "Transparent council rule explanation"},
    {"token": "{spread_pips}", "desc": "Assumed round-turn spread in pips"},
    {"token": "{core_score}", "desc": "Core confluence checks passed (0-6)"},
    {"token": "{short_id}", "desc": "8-char signal identifier"},
    {"token": "{source}", "desc": "Feed source tag"},
)

RESULT_PLACEHOLDERS: tuple[dict[str, str], ...] = (
    {"token": "{pair}", "desc": "Instrument symbol"},
    {"token": "{side}", "desc": "Signal direction"},
    {"token": "{verdict}", "desc": "WIN, LOSS, TIED, or UNRESOLVED"},
    {"token": "{outcome}", "desc": "Ledger outcome constant"},
    {"token": "{entry_price}", "desc": "Realised entry candle open or n/a"},
    {"token": "{exit_price}", "desc": "Realised exit candle close or n/a"},
    {"token": "{gross_pips}", "desc": "Signed gross move in pips or n/a"},
    {"token": "{net_pips}", "desc": "Signed net move after spread or n/a"},
    {"token": "{spread_pips}", "desc": "Round-turn spread charged"},
    {"token": "{horizon}", "desc": "Hold duration in minutes"},
    {"token": "{short_id}", "desc": "8-char signal identifier"},
)

_TOKEN_RE = re.compile(r"\{([a-zA-Z0-9_]+)\}")
_MAX_TEMPLATE_CHARS = 1600


def safe_render_template(template: str, values: Mapping[str, Any]) -> str:
    """Substitute ``{identifier}`` placeholders without ``str.format`` traversal.

    Attribute access (``{a.__class__}``), index lookups (``{0}``), and stray
    braces are left untouched so user-supplied templates can never raise
    formatting exceptions or inspect Python internals.
    """
    clean = str(template or "")[:_MAX_TEMPLATE_CHARS]

    def _replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key in values:
            return str(values[key])
        return match.group(0)

    return _TOKEN_RE.sub(_replace, clean).strip()


@dataclass
class TelegramFormatConfig:
    signal_style: str = "standard"
    signal_template: str = ""
    result_style: str = "standard"
    result_template: str = ""
    apply_to_signals_cmd: bool = False
    append_paper_footer: bool = True

    def __post_init__(self) -> None:
        s_style = str(self.signal_style or "standard").strip().lower()
        r_style = str(self.result_style or "standard").strip().lower()
        self.signal_style = s_style if s_style in SIGNAL_STYLES else "standard"
        self.result_style = r_style if r_style in RESULT_STYLES else "standard"
        self.signal_template = str(self.signal_template or "").replace("\r\n", "\n")[:_MAX_TEMPLATE_CHARS]
        self.result_template = str(self.result_template or "").replace("\r\n", "\n")[:_MAX_TEMPLATE_CHARS]
        self.apply_to_signals_cmd = bool(self.apply_to_signals_cmd)
        self.append_paper_footer = bool(self.append_paper_footer)

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal_style": self.signal_style,
            "signal_template": self.signal_template,
            "result_style": self.result_style,
            "result_template": self.result_template,
            "apply_to_signals_cmd": self.apply_to_signals_cmd,
            "append_paper_footer": self.append_paper_footer,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "TelegramFormatConfig":
        if not isinstance(payload, Mapping):
            return cls()
        return cls(
            signal_style=str(payload.get("signal_style") or "standard"),
            signal_template=str(payload.get("signal_template") or ""),
            result_style=str(payload.get("result_style") or "standard"),
            result_template=str(payload.get("result_template") or ""),
            apply_to_signals_cmd=bool(payload.get("apply_to_signals_cmd", False)),
            append_paper_footer=bool(payload.get("append_paper_footer", True)),
        )


def signal_template_values(signal: Signal) -> dict[str, str]:
    from .timeutil import DISPLAY_TZ_LABEL, format_display_iso, to_display_tz

    entry_disp = to_display_tz(signal.entry_time)
    expiry_disp = to_display_tz(signal.expiry_time)
    probability = "n/a" if signal.probability_up is None else f"{signal.probability_up:.3f}"
    raw = "n/a" if signal.probability_raw is None else f"{signal.probability_raw:.3f}"
    cal_map = {
        "RAW": "raw (not enough outcomes to calibrate yet)",
        "CALIBRATED": "calibrated",
        "FLAT": "flat — abstained, never sent",
    }
    core_val = "n/a"
    for vote in getattr(signal, "council", ()) or ():
        if isinstance(vote, Mapping) and vote.get("name") == "orbix_confluence":
            core_val = str(vote.get("score", 0))
            break
    return {
        "pair": signal.pair,
        "side": signal.side,
        "entry_hm": entry_disp.strftime("%H:%M"),
        "expiry_hm": expiry_disp.strftime("%H:%M"),
        "tz": DISPLAY_TZ_LABEL,
        "entry_iso": format_display_iso(signal.entry_time),
        "expiry_iso": format_display_iso(signal.expiry_time),
        "horizon": str(signal.horizon_minutes),
        "regime": signal.regime,
        "probability": probability,
        "raw_probability": raw,
        "calibration_state": signal.calibration_state,
        "calibration_desc": cal_map.get(signal.calibration_state, signal.calibration_state),
        "reason": signal.reason,
        "spread_pips": f"{signal.spread_pips:.2f}",
        "core_score": core_val,
        "short_id": signal.signal_id[:8],
        "signal_id": signal.signal_id,
        "source": signal.source,
    }


def render_signal_message(
    signal: Signal,
    fmt: TelegramFormatConfig | None = None,
    *,
    brief: bool = False,
) -> str:
    """Format a sealed signal according to the active preset or custom template."""
    cfg = fmt or TelegramFormatConfig()
    vals = signal_template_values(signal)

    if brief and not cfg.apply_to_signals_cmd:
        style = "brief"
    else:
        style = cfg.signal_style

    if style == "brief":
        return "\n".join(
            [
                f"{vals['pair']} {vals['side']}",
                f"Entry  open of {vals['entry_hm']} {vals['tz']} candle",
                f"Expiry {vals['expiry_hm']} {vals['tz']}  ({vals['horizon']} min)",
                f"#{vals['short_id']}",
                "",
                "Paper — research only, no broker order.",
            ]
        )

    if style == "tactical":
        return "\n".join(
            [
                f"[{vals['side']}] {vals['pair']}  ·  {vals['horizon']}m  ·  #{vals['short_id']}",
                f"Entry {vals['entry_hm']} {vals['tz']}  ->  Expiry {vals['expiry_hm']} {vals['tz']}",
                f"Regime {vals['regime']}  |  Model {vals['probability']} ({vals['calibration_state']})  |  Spread {vals['spread_pips']}p",
                f"Gate: {vals['reason']}",
                "Paper — research only, no broker order.",
            ]
        )

    if style == "compact":
        return (
            f"{vals['pair']} {vals['side']} | in {vals['entry_hm']} {vals['tz']} -> exp {vals['expiry_hm']} {vals['tz']} ({vals['horizon']}m) "
            f"| {vals['regime']} | p={vals['probability']} | cost {vals['spread_pips']}p | #{vals['short_id']} [paper]"
        )

    if style == "custom":
        tmpl = cfg.signal_template.strip() or DEFAULT_CUSTOM_SIGNAL_TEMPLATE
        rendered = safe_render_template(tmpl, vals)
        if cfg.append_paper_footer and "paper" not in rendered.lower():
            rendered = f"{rendered}\nPaper — research only, no broker order."
        return rendered[:3800]

    # Default "standard" — all timestamps in UTC+05:30
    return "\n".join(
        [
            f"{vals['pair']} {vals['side']}",
            f"Entry: open of the {vals['entry_hm']} {vals['tz']} candle",
            f"Expires: {vals['expiry_hm']} {vals['tz']} ({vals['horizon']} min)",
            f"Regime: {vals['regime']}",
            f"Model: {vals['probability']} — {vals['calibration_desc']} (raw {vals['raw_probability']})",
            f"Why: {vals['reason']}",
            f"Cost: {vals['spread_pips']} pip spread charged on settlement.",
            "Paper research signal. No broker order is placed.",
        ]
    )


def render_result_message(
    signal: Signal,
    settlement: Any,
    fmt: TelegramFormatConfig | None = None,
) -> str:
    """Format a cost-charged settlement result while enforcing UNRESOLVED honesty."""
    cfg = fmt or TelegramFormatConfig()
    precision = instruments.for_symbol(signal.pair).precision
    short_id = signal.signal_id[:8]
    header = f"RESULT {signal.pair} {signal.side}  #{short_id}"
    outcome = str(getattr(settlement, "outcome", charter.UNRESOLVED))

    entry_px = getattr(settlement, "entry_price", None)
    exit_px = getattr(settlement, "exit_price", None)
    gross_val = getattr(settlement, "gross_pips", None)
    net_val = getattr(settlement, "net_pips", None)
    spread_val = float(getattr(settlement, "spread_pips", signal.spread_pips) or 0.0)

    if outcome == charter.UNRESOLVED:
        verdict = "UNRESOLVED"
    elif outcome == charter.TIED:
        verdict = "TIED"
    elif outcome == charter.WIN:
        verdict = "WIN"
    else:
        verdict = "LOSS"

    vals = {
        "pair": signal.pair,
        "side": signal.side,
        "short_id": short_id,
        "signal_id": signal.signal_id,
        "horizon": str(signal.horizon_minutes),
        "outcome": outcome,
        "verdict": verdict,
        "entry_price": "n/a" if entry_px is None else f"{float(entry_px):,.{precision}f}",
        "exit_price": "n/a" if exit_px is None else f"{float(exit_px):,.{precision}f}",
        "gross_pips": "n/a" if gross_val is None else f"{float(gross_val):+.1f}",
        "net_pips": "n/a" if net_val is None else f"{float(net_val):+.1f}",
        "spread_pips": f"{spread_val:.1f}",
    }

    style = cfg.result_style

    if style == "compact":
        if outcome == charter.UNRESOLVED:
            return f"RESULT {vals['pair']} {vals['side']} #{short_id} | UNRESOLVED — price data never arrived (no result claimed)"
        if outcome == charter.TIED:
            return f"RESULT {vals['pair']} {vals['side']} #{short_id} | {vals['entry_price']} -> {vals['exit_price']} | TIED"
        return (
            f"RESULT {vals['pair']} {vals['side']} #{short_id} | {vals['entry_price']} -> {vals['exit_price']} "
            f"| gross {vals['gross_pips']}p | cost {vals['spread_pips']}p | net {vals['net_pips']}p | {verdict}"
        )

    if style == "custom":
        tmpl = cfg.result_template.strip() or DEFAULT_CUSTOM_RESULT_TEMPLATE
        rendered = safe_render_template(tmpl, vals)
        if outcome == charter.UNRESOLVED and "unresolved" not in rendered.lower():
            rendered = f"{rendered}\nUNRESOLVED — the price data never arrived; no result is claimed."
        return rendered[:3800]

    # Default "standard" — byte-identical to original format_result(signal, settlement)
    if outcome == charter.UNRESOLVED:
        return "\n".join(
            [
                header,
                "UNRESOLVED — the price data never arrived.",
                "No result is claimed for this trade.",
            ]
        )
    if outcome == charter.TIED:
        return "\n".join(
            [
                header,
                f"Entry {vals['entry_price']} -> Exit {vals['exit_price']}",
                "TIED — no movement, no result either way.",
            ]
        )
    return "\n".join(
        [
            header,
            f"Entry {vals['entry_price']} -> Exit {vals['exit_price']}",
            f"Move {vals['gross_pips']} pips | cost {vals['spread_pips']} | net {vals['net_pips']}",
            verdict,
        ]
    )

