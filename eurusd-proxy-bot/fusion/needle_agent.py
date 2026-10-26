"""Cactus Compute Needle 3 (Apache-2.0) integration for SAVIOUR / FUSION BOT.

Why Needle 3 sits beside, never inside, the candle math
-------------------------------------------------------

Needle 3 (``cactus-needle``) is a sub-30 MB Laddered Simple Attention Network
purpose-built for **tool calling** and **structured JSON extraction** — not
open-ended chat and not floating-point time-series regression.

Accordingly, SAVIOUR keeps a strict separation of duties:

1. **Price-direction math stays 100% deterministic & causal**: M1 indicators,
   3-mode ``orbix_confluence``, 5m/15m/60m MTF cascade, walk-forward logistic
   veto, and isotonic calibration never pass through a language model.
2. **Needle 3 handles natural-language automation & structured extraction**:
   - Routing plain-English owner commands in Telegram and the Web Operator
     Console to safe, schema-checked SAVIOUR tools (``scan_market``,
     ``bot_status``, ``pause_signals``, ``resume_signals``,
     ``set_telegram_style``, ``configure_guards``).
   - Extracting structured macro/crypto shock metadata (``extract_news_shock``)
     from raw headlines into ``fusion.news`` blackout windows.
   - Refusing off-topic queries and Charter-prohibited actions (live broker
     orders, Quotex/Pocket Option automation, ``_otc`` feeds, Martingale, or
     guaranteed-win-rate requests) with an empty call list ``[]``.

Privacy & Telemetry
-------------------
Cactus Compute's binary enables telemetry unless explicitly disabled via
environment variables. This module sets ``NEEDLE_TELEMETRY=0`` and
``DO_NOT_TRACK=1`` before importing ``needle`` so zero telemetry ever leaves
the host.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

# Disable Needle binary & SDK telemetry before any needle import can occur.
os.environ.setdefault("NEEDLE_TELEMETRY", "0")
os.environ.setdefault("DO_NOT_TRACK", "1")

from . import instruments, telegram
from .timeutil import now as utc_now

DEFAULT_WEIGHTS_REL = "needle/saviour_needle3.cact"
DEFAULT_ADAPTER_REL = "needle/saviour_lora.safetensors"
DEFAULT_REPORT_REL = "needle/finetune_report.json"
DEFAULT_MIN_CONFIDENCE = 0.55

SUPPORTED_PAIRS = ("ALL", "EUR/USD", "BTC/USD", "ETH/USD")
SUPPORTED_SIGNAL_STYLES = ("standard", "brief", "tactical", "compact")
SUPPORTED_RESULT_STYLES = ("standard", "compact")
SUPPORTED_MTF_MINUTES = (5, 15, 60)
SUPPORTED_SEVERITIES = ("NONE", "LOW", "HIGH")

# Designed strictly per Needle 3 tool-calling guidelines:
# - Small toolset with plain, distinct names (no shared prefixes)
# - Explicit JSON enums & integer bounds for grammar-constrained decoding
# - Regex triggers to steer tool selection cleanly
SAVIOUR_NEEDLE_TOOLS: list[dict[str, Any]] = [
    {
        "name": "scan_market",
        "description": "Evaluate M1 market gates and report any signal.",
        "parameters": {
            "type": "object",
            "properties": {
                "pair": {"type": "string", "enum": list(SUPPORTED_PAIRS)},
                "mode": {"type": "string", "enum": ["now", "scan"]},
            },
            "required": ["pair", "mode"],
        },
    },
    {
        "name": "bot_status",
        "description": "Report bot health, heartbeat, ledger, and key pool.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "pause_signals",
        "description": "Pause outgoing signal delivery.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "resume_signals",
        "description": "Resume outgoing signal delivery.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "set_telegram_style",
        "description": "Set Telegram formatting preset.",
        "parameters": {
            "type": "object",
            "properties": {
                "signal_style": {"type": "string", "enum": list(SUPPORTED_SIGNAL_STYLES)},
                "result_style": {"type": "string", "enum": list(SUPPORTED_RESULT_STYLES)},
            },
            "required": ["signal_style"],
        },
    },
    {
        "name": "configure_guards",
        "description": "Update MTF minutes, cooldown, or horizon.",
        "parameters": {
            "type": "object",
            "properties": {
                "mtf_minutes": {"type": "integer", "enum": list(SUPPORTED_MTF_MINUTES)},
                "cooldown_minutes": {"type": "integer", "minimum": 0, "maximum": 120},
                "horizon_minutes": {"type": "integer", "minimum": 5, "maximum": 120},
            },
        },
    },
    {
        "name": "extract_news_shock",
        "description": "Extract headline shock severity and blackout minutes.",
        "parameters": {
            "type": "object",
            "properties": {
                "pair": {"type": "string", "enum": list(SUPPORTED_PAIRS)},
                "severity": {"type": "string", "enum": list(SUPPORTED_SEVERITIES)},
                "blackout_minutes": {"type": "integer", "minimum": 0, "maximum": 180},
                "event": {"type": "string"},
            },
            "required": ["pair", "severity", "blackout_minutes", "event"],
        },
    },
]

_TOOL_MAP: dict[str, dict[str, Any]] = {t["name"]: t for t in SAVIOUR_NEEDLE_TOOLS}

# Charter-prohibited patterns: always refused even if a user asks explicitly.
_CHARTER_PROHIBITED_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"\b(quotex|pocket\s*option|iq\s*option|binomo|olymp\s*trade)\b", re.IGNORECASE),
        "Charter refusal: SAVIOUR never connects to or automates binary-options brokers.",
    ),
    (
        re.compile(r"(_otc\b|\botc\s+(pair|market|chart|feed|signal|candle)s?\b)", re.IGNORECASE),
        "Charter refusal: synthetic OTC feeds are prohibited; SAVIOUR evaluates real spot/FX candles only.",
    ),
    (
        re.compile(r"\b(martingale|double\s+(the\s+)?(stake|lot|size|bet)|recover\s+loss\s+by\s+doubling)\b", re.IGNORECASE),
        "Charter refusal: Martingale and loss-doubling position sizing are strictly forbidden.",
    ),
    (
        re.compile(
            r"\b(place|execute|open|submit|send)\b.{0,40}\b(real|live)\b.{0,40}\b(order|trade|trades|position|buy|sell|account)\b"
            r"|\bconnect\s+(my\s+)?(mt4|mt5|metatrader|binance|broker\s+account)\b",
            re.IGNORECASE,
        ),
        "Charter refusal: SAVIOUR is a cost-charged paper research engine and never places real-money broker orders.",
    ),
    (
        re.compile(
            r"\b(guarantee[ds]?|100\s*%|99\s*%|95\s*%|never\s+lose|zero\s+loss|sure\s*shot)\b.*\b(win|accuracy|signal|profit)\b",
            re.IGNORECASE,
        ),
        "Charter refusal: SAVIOUR never fabricates guaranteed accuracy or relaxes gates to manufacture wins.",
    ),
)


def charter_refusal_reason(query: str) -> str | None:
    """Return a non-empty Charter refusal explanation when ``query`` crosses a hard boundary."""
    text = (query or "").strip()
    if not text:
        return None
    for pattern, message in _CHARTER_PROHIBITED_PATTERNS:
        if pattern.search(text):
            return message
    return None


def needle_package_info() -> dict[str, Any]:
    """Inspect installed ``cactus-needle`` runtime/training capabilities without network calls."""
    os.environ.setdefault("NEEDLE_TELEMETRY", "0")
    os.environ.setdefault("DO_NOT_TRACK", "1")
    info: dict[str, Any] = {
        "installed": False,
        "version": None,
        "train_installed": False,
        "native_engine_cached": False,
        "base_weights_cached": False,
        "telemetry_disabled": (
            os.environ.get("NEEDLE_TELEMETRY") == "0"
            and os.environ.get("DO_NOT_TRACK") == "1"
        ),
    }
    try:
        import importlib.metadata as md
        import needle  # type: ignore[import-not-found]

        info["installed"] = True
        try:
            info["version"] = md.version("cactus-needle")
        except Exception:
            info["version"] = getattr(needle, "__version__", "unknown")

        from needle.agent import fetch  # type: ignore[import-not-found]

        here = Path(needle.__file__).resolve().parent
        lib_name = fetch.lib_name(3)
        stem, suffix = os.path.splitext(lib_name)
        candidates = [
            os.environ.get(fetch.lib_path_env(3), ""),
            str(here / f"{stem}3{suffix}"),
            str(Path(fetch.cache_dir(3)) / lib_name),
        ]
        info["native_engine_cached"] = any(p and os.path.exists(p) for p in candidates)
        base_cact = Path(fetch.cache_dir(3)) / fetch.base_weights(3)
        info["base_weights_cached"] = base_cact.is_file()
    except Exception:
        return info

    try:
        import flax  # noqa: F401  # type: ignore[import-not-found]
        import jax  # noqa: F401  # type: ignore[import-not-found]
        import optax  # noqa: F401  # type: ignore[import-not-found]
        import safetensors  # noqa: F401  # type: ignore[import-not-found]

        info["train_installed"] = True
    except Exception:
        info["train_installed"] = False

    return info


def validate_tool_calls(raw_calls: list[Mapping[str, Any]] | None) -> list[dict[str, Any]]:
    """Sanitize and schema-check tool calls so invalid names or out-of-range args never execute."""
    if not raw_calls:
        return []
    cleaned: list[dict[str, Any]] = []
    for item in raw_calls:
        if not isinstance(item, Mapping):
            continue
        name = str(item.get("name") or "").strip()
        schema = _TOOL_MAP.get(name)
        if schema is None:
            continue
        raw_args = item.get("arguments") or {}
        if isinstance(raw_args, str):
            try:
                raw_args = json.loads(raw_args) if raw_args.strip() else {}
            except json.JSONDecodeError:
                continue
        if not isinstance(raw_args, Mapping):
            continue

        props = schema["parameters"].get("properties", {})
        required = schema["parameters"].get("required", [])
        out_args: dict[str, Any] = {}
        valid = True

        for key, prop_spec in props.items():
            if key not in raw_args or raw_args[key] is None:
                continue
            val = raw_args[key]
            ptype = prop_spec.get("type")
            if ptype == "string":
                sval = str(val).strip()
                allowed = prop_spec.get("enum")
                if allowed is not None:
                    matched = next((a for a in allowed if str(a).lower() == sval.lower()), None)
                    if matched is None:
                        valid = False
                        break
                    sval = matched
                out_args[key] = sval[:160]
            elif ptype == "integer":
                try:
                    ival = int(round(float(val)))
                except (TypeError, ValueError):
                    valid = False
                    break
                allowed = prop_spec.get("enum")
                if allowed is not None and ival not in allowed:
                    valid = False
                    break
                if "minimum" in prop_spec:
                    ival = max(int(prop_spec["minimum"]), ival)
                if "maximum" in prop_spec:
                    ival = min(int(prop_spec["maximum"]), ival)
                out_args[key] = ival

        if not valid:
            continue

        # Supply safe defaults for required scan_market / extract_news_shock fields if omitted
        if name == "scan_market":
            out_args.setdefault("pair", "ALL")
            out_args.setdefault("mode", "now")
        elif name == "set_telegram_style":
            if "signal_style" not in out_args:
                continue
            out_args.setdefault(
                "result_style",
                "compact" if out_args["signal_style"] in ("compact", "brief") else "standard",
            )
        elif name == "configure_guards":
            if not out_args:
                continue

        if any(req not in out_args for req in required):
            continue
        cleaned.append({"name": name, "arguments": out_args})
    return cleaned


# ---------------------------------------------------------------------------
# Deterministic schema-verified fallback router (used when C++ engine or HF
# cache is unavailable, ensuring <1ms offline parity with SAVIOUR_NEEDLE_TOOLS)
# ---------------------------------------------------------------------------

_PAIR_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b(btc(?:/|\s|-)?usd|bitcoin|btc|xbt)\b", re.IGNORECASE), "BTC/USD"),
    (re.compile(r"\b(eth(?:/|\s|-)?usd|ethereum|ether|eth)\b", re.IGNORECASE), "ETH/USD"),
    (re.compile(r"\b(eur(?:/|\s|-)?usd|eurusd|fiber|euro)\b", re.IGNORECASE), "EUR/USD"),
)

_HIGH_SHOCK_RE = re.compile(
    r"\b(cpi|nfp|non-?farm\s+payrolls?|fomc|rate\s+hike|rate\s+cut|emergency\s+(fed|ecb|meeting|rate)|"
    r"hot\s+inflation|surprise\s+inflation|powell|lagarde|ecb\s+decision|fed\s+decision|"
    r"exchange\s+halt|withdrawals?\s+halted|sec\s+(sues|charges|approves|rejects)|bridge\s+hack|"
    r"flash\s+crash|depeg)\b",
    re.IGNORECASE,
)
_LOW_SHOCK_RE = re.compile(
    r"\b(pmi|retail\s+sales|jobless\s+claims|housing\s+starts|consumer\s+confidence|"
    r"etf\s+inflows?|etf\s+outflows?|treasury\s+auction|fed\s+minutes)\b",
    re.IGNORECASE,
)
_HEADLINE_CONTEXT_RE = re.compile(
    r"\b(headline|breaking|news|release|prints?|announces?|surges?|plunges?|spikes?|whipsaws?|"
    r"inflation|payrolls?|fomc|cpi|nfp|ecb|powell|pmi|sec|withdrawals?\s+halted|"
    r"unchanged|steady\s+in\s+quiet|quiet\s+asian\s+session|commentary)\b",
    re.IGNORECASE,
)


def _detect_pair(text: str, *, default: str = "ALL") -> str:
    hits: list[str] = []
    for pattern, symbol in _PAIR_PATTERNS:
        if pattern.search(text) and symbol not in hits:
            hits.append(symbol)
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        return "ALL"
    return default


def _infer_event_label(text: str) -> str:
    stripped = re.sub(
        r"^\s*(?:breaking(?:\s+headline|\s+news)?|extract\s+(?:headline|news)\s+risk|classify\s+(?:headline|news)|headline)\s*:\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    checks = (
        (r"\bnfp\b|non-?farm\s+payrolls?", "US NFP"),
        (r"\bcpi\b|consumer\s+price", "US CPI"),
        (r"\becb\b|lagarde", "ECB"),
        (r"\bfomc\b|emergency\s+fed|fed\s+rate|rate\s+(hike|cut)|powell", "FOMC"),
        (r"\bsec\b", "SEC Crypto"),
        (r"withdrawals?\s+halted|exchange\s+halt|hack", "Crypto Liquidity Shock"),
        (r"\betf\b", "Crypto ETF Flow"),
        (r"\bpmi\b", "Macro PMI"),
        (r"jobless\s+claims", "US Jobless Claims"),
    )
    for pat, label in checks:
        if re.search(pat, stripped, re.IGNORECASE):
            return label
    words = re.findall(r"[A-Za-z0-9/]+", stripped)
    return " ".join(words[:3])[:40] if words else "Market Update"


def _route_single_clause(clause: str) -> list[dict[str, Any]]:
    text = clause.strip()
    lower = text.lower()
    if not lower:
        return []

    # 1. Explicit news / macro headline extraction
    is_command_verb = bool(
        re.search(
            r"^\s*(scan|check|show|get|give|run|pause|stop|halt|mute|resume|unpause|start|"
            r"set|switch|change|use|configure|update|what|how)\b",
            lower,
        )
    )
    is_extract_request = bool(re.search(r"\b(extract|classify|headline|news\s+risk|blackout\s+for)\b", lower))
    if is_extract_request or (_HEADLINE_CONTEXT_RE.search(text) and not is_command_verb):
        pair = _detect_pair(text, default="ALL")
        if pair == "ALL" and re.search(r"\b(eur|ecb|lagarde|eurozone)\b", lower):
            pair = "EUR/USD"
        elif pair == "ALL" and re.search(r"\b(us\s+cpi|nfp|fomc|powell|payrolls)\b", lower) and not re.search(r"\b(crypto|btc|eth)\b", lower):
            pair = "ALL"
        event = _infer_event_label(text)
        if _HIGH_SHOCK_RE.search(text):
            mins = 45 if event in ("FOMC", "Crypto Liquidity Shock") else 30
            return [
                {
                    "name": "extract_news_shock",
                    "arguments": {
                        "pair": pair,
                        "severity": "HIGH",
                        "blackout_minutes": mins,
                        "event": event,
                    },
                }
            ]
        if _LOW_SHOCK_RE.search(text):
            return [
                {
                    "name": "extract_news_shock",
                    "arguments": {
                        "pair": pair,
                        "severity": "LOW",
                        "blackout_minutes": 10,
                        "event": event,
                    },
                }
            ]
        if is_extract_request or re.search(r"\b(unchanged|quiet|routine|calm|steady)\b", lower):
            return [
                {
                    "name": "extract_news_shock",
                    "arguments": {
                        "pair": pair,
                        "severity": "NONE",
                        "blackout_minutes": 0,
                        "event": event,
                    },
                }
            ]

    calls: list[dict[str, Any]] = []

    # 2. Pause / Resume
    if re.search(
        r"\b(unpause|resume|turn\s+signals?\s+(back\s+)?on|(?:start|restart)\s+(?:sending\s+|signal\s+)?(?:signals?|delivery)|"
        r"re-?enable(?:\s+signal|\s+signals)?|continue\s+dispatching|enable\s+signals?\s+again)\b",
        lower,
    ):
        calls.append({"name": "resume_signals", "arguments": {}})
    elif re.search(
        r"\b(pause|halt|mute|suspend|stop\s+(?:sending\s+)?signals?|hold\s+signals?|quiet\s+mode)\b",
        lower,
    ):
        calls.append({"name": "pause_signals", "arguments": {}})

    # 3. Telegram formatting preset
    if re.search(r"\b(telegram|format|style|preset|template|layout|cards?|messages?|alerts?)\b", lower) and re.search(
        r"\b(standard|brief|tactical|compact)\b", lower
    ):
        sig_style = "standard"
        for candidate in ("tactical", "compact", "brief", "standard"):
            if re.search(rf"\b{candidate}\b", lower):
                sig_style = candidate
                break
        res_style = "compact" if ("compact" in lower or sig_style in ("compact", "brief")) else "standard"
        calls.append(
            {
                "name": "set_telegram_style",
                "arguments": {"signal_style": sig_style, "result_style": res_style},
            }
        )

    # 4. Runtime guard configuration (MTF / cooldown / horizon)
    guard_args: dict[str, Any] = {}
    mtf_match = re.search(
        r"\b(?:mtf|higher[-\s]?timeframe|trend\s+filter)(?:\s+\w+){0,3}?\s+(5|15|60)\s*(?:m|min|mins|minute|minutes)?\b"
        r"|\b(5|15|60)\s*(?:m|min|mins|minute|minutes)\s+(?:mtf|higher[-\s]?timeframe(?:\s+filter)?|trend\s+filter)\b",
        lower,
    )
    if mtf_match:
        val = int(mtf_match.group(1) or mtf_match.group(2))
        if val in SUPPORTED_MTF_MINUTES:
            guard_args["mtf_minutes"] = val

    cd_match = re.search(
        r"\bcooldown(?:\s+\w+){0,3}?\s+(\d{1,3})\s*(?:m|min|mins|minute|minutes)?\b"
        r"|\b(\d{1,3})\s*(?:m|min|mins|minute|minutes)\s+cooldown\b",
        lower,
    )
    if cd_match:
        guard_args["cooldown_minutes"] = max(0, min(120, int(cd_match.group(1) or cd_match.group(2))))

    hz_match = re.search(
        r"\b(?:horizon|expiry|hold)(?:\s+\w+){0,3}?\s+(\d{1,3})\s*(?:m|min|mins|minute|minutes)?\b"
        r"|\b(\d{1,3})\s*(?:m|min|mins|minute|minutes)\s+(?:horizon|expiry|hold)\b",
        lower,
    )
    if hz_match:
        guard_args["horizon_minutes"] = max(5, min(120, int(hz_match.group(1) or hz_match.group(2))))

    if guard_args:
        calls.append({"name": "configure_guards", "arguments": guard_args})

    # 5. Bot status / health / ledger
    if re.search(
        r"\b(status|health|heartbeat|win\s*rate|ledger|scorecard|model\s+mode|key\s*pool|"
        r"how\s+is\s+(the\s+)?(bot|saviour|fusion)|stats|performance)\b",
        lower,
    ):
        calls.append({"name": "bot_status", "arguments": {}})

    # 6. Market scan / signal check (only if this clause was not already a pause/resume/style/guard action mentioning 'signal')
    already_control_or_style = any(
        c["name"] in ("pause_signals", "resume_signals", "set_telegram_style", "configure_guards")
        for c in calls
    )
    if not already_control_or_style and re.search(
        r"\b(scan|signals?|setup|setups|opportunity|entry|evaluate|gate\s+check|read\s+(the\s+)?(?:\w+\s+)?market|"
        r"check\s+(the\s+)?(market|btc|bitcoin|eth|ether|ethereum|eur|eurusd|fiber|all\s+pairs))\b",
        lower,
    ):
        pair = _detect_pair(text, default="ALL")
        mode = "scan" if re.search(r"\b(scan\s+history|paper\s+scan|search\s+bars|find\s+a\s+setup|deep\s+scan)\b", lower) else "now"
        calls.append({"name": "scan_market", "arguments": {"pair": pair, "mode": mode}})

    return calls


def deterministic_route(query: str) -> list[dict[str, Any]]:
    """Deterministic multi-clause fallback router matching ``SAVIOUR_NEEDLE_TOOLS``."""
    if charter_refusal_reason(query):
        return []
    clean_query = (query or "").strip()
    is_command_verb = bool(
        re.search(
            r"^\s*(scan|check|show|get|give|run|pause|stop|halt|mute|resume|unpause|start|"
            r"set|switch|change|use|configure|update|what|how)\b",
            clean_query.lower(),
        )
    )
    if not is_command_verb and (_HEADLINE_CONTEXT_RE.search(clean_query) or re.search(r"\b(extract|classify)\b", clean_query.lower())):
        return validate_tool_calls(_route_single_clause(clean_query))
    if re.search(r"\bresults?\s+(?:to\s+)?compact\b", clean_query.lower()) and re.search(r"\btactical\b", clean_query.lower()):
        clauses = [clean_query]
    else:
        clauses = re.split(r"\s*,?\s+(?:and\s+then|then|and\s+also|and)\s+|;\s*", clean_query, flags=re.IGNORECASE)
    merged: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for clause in clauses:
        for call in _route_single_clause(clause):
            if call["name"] == "configure_guards":
                existing_guard = next((c for c in merged if c["name"] == "configure_guards"), None)
                if existing_guard is not None:
                    existing_guard["arguments"].update(call["arguments"])
                    continue
            dedup = f"{call['name']}:{json.dumps(call['arguments'], sort_keys=True)}"
            if dedup not in seen_keys:
                seen_keys.add(dedup)
                merged.append(call)
    if not merged and len(clauses) > 1:
        merged = _route_single_clause(query)
    return validate_tool_calls(merged)


@dataclass
class NeedleDecision:
    query: str
    type: str  # "tool_call" | "refusal"
    function_calls: list[dict[str, Any]]
    confidence: float | None
    engine: str  # "needle3_cact" | "needle3_jax_lora" | "deterministic_fallback" | "charter_shield"
    weights_path: str | None = None
    refusal_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "type": self.type,
            "function_calls": self.function_calls,
            "confidence": self.confidence,
            "engine": self.engine,
            "weights_path": self.weights_path,
            "refusal_reason": self.refusal_reason,
        }


class SaviourNeedleAgent:
    """Charter-guarded Needle 3 runtime for SAVIOUR operator commands & news extraction."""

    def __init__(
        self,
        *,
        state_dir: str | Path = "var",
        weights_path: str | Path | None = None,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    ) -> None:
        os.environ.setdefault("NEEDLE_TELEMETRY", "0")
        os.environ.setdefault("DO_NOT_TRACK", "1")
        self.state_dir = Path(state_dir)
        env_weights = os.environ.get("NEEDLE_WEIGHTS_PATH", "").strip()
        if weights_path is not None:
            self.weights_path: Path | None = Path(weights_path)
        elif env_weights:
            self.weights_path = Path(env_weights)
        else:
            self.weights_path = self.state_dir / DEFAULT_WEIGHTS_REL
        self.min_confidence = float(min_confidence)
        self._native_model: Any | None = None
        self._native_loaded_weights: str | None = None

    def finetune_report_path(self) -> Path:
        return self.state_dir / DEFAULT_REPORT_REL

    def load_finetune_report(self) -> dict[str, Any] | None:
        path = self.finetune_report_path()
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else None
        except Exception:
            return None

    def _resolve_cact_weights(self) -> Path | None:
        """Locate a runnable ``.cact`` archive (tuned SAVIOUR model first, then cached base)."""
        report = self.load_finetune_report() or {}
        if self.weights_path and self.weights_path.is_file():
            # Skip synthetic tiny bootstrap archives for live semantic inference unless explicitly forced.
            if not report.get("bootstrap_tiny_init", False) or os.environ.get("NEEDLE_FORCE_BOOTSTRAP_CACT") == "1":
                return self.weights_path
        try:
            from needle.agent import fetch  # type: ignore[import-not-found]

            cached = Path(fetch.cache_dir(3)) / fetch.base_weights(3)
            if cached.is_file():
                return cached
        except Exception:
            pass
        return None

    def _try_native_cact(self, query: str) -> NeedleDecision | None:
        pkg = needle_package_info()
        if not pkg.get("installed") or not pkg.get("native_engine_cached"):
            return None
        cact_path = self._resolve_cact_weights()
        if cact_path is None:
            return None
        try:
            if self._native_model is None or self._native_loaded_weights != str(cact_path):
                import needle  # type: ignore[import-not-found]

                self._native_model = needle.Needle(
                    tools=SAVIOUR_NEEDLE_TOOLS,
                    weights=str(cact_path),
                    auto_date=False,
                )
                self._native_loaded_weights = str(cact_path)
            else:
                self._native_model.reset()

            raw = self._native_model.complete(query)
            conf = raw.get("confidence")
            if conf is not None and float(conf) < self.min_confidence:
                return NeedleDecision(
                    query=query,
                    type="refusal",
                    function_calls=[],
                    confidence=round(float(conf), 4),
                    engine="needle3_cact",
                    weights_path=str(cact_path),
                    refusal_reason=(
                        f"Needle 3 confidence {float(conf):.2f} is below gate {self.min_confidence:.2f}; abstaining."
                    ),
                )
            calls = validate_tool_calls(raw.get("function_calls") or [])
            if not calls:
                return NeedleDecision(
                    query=query,
                    type="refusal",
                    function_calls=[],
                    confidence=(round(float(conf), 4) if conf is not None else None),
                    engine="needle3_cact",
                    weights_path=str(cact_path),
                    refusal_reason="Off-topic or unsupported request — no SAVIOUR tool matched.",
                )
            return NeedleDecision(
                query=query,
                type="tool_call",
                function_calls=calls,
                confidence=(round(float(conf), 4) if conf is not None else None),
                engine="needle3_cact",
                weights_path=str(cact_path),
            )
        except Exception:
            return None

    def status(self) -> dict[str, Any]:
        pkg = needle_package_info()
        report = self.load_finetune_report()
        cact_file = self.weights_path if (self.weights_path and self.weights_path.is_file()) else None
        local_cact = self.state_dir / "needle" / "saviour_needle3_local.cact"
        if cact_file is None and local_cact.is_file():
            cact_file = local_cact
        adapter_file = self.state_dir / DEFAULT_ADAPTER_REL

        active_engine = (
            "needle3_cact"
            if (pkg.get("native_engine_cached") and self._resolve_cact_weights() is not None)
            else "deterministic_schema_router"
        )
        return {
            "package": pkg,
            "active_engine": active_engine,
            "tools": [t["name"] for t in SAVIOUR_NEEDLE_TOOLS],
            "tool_count": len(SAVIOUR_NEEDLE_TOOLS),
            "min_confidence": self.min_confidence,
            "tuned_cact_present": bool(cact_file and cact_file.is_file()),
            "tuned_cact_path": str(cact_file) if cact_file else None,
            "tuned_cact_bytes": int(cact_file.stat().st_size) if (cact_file and cact_file.is_file()) else 0,
            "lora_adapter_present": adapter_file.is_file(),
            "lora_adapter_path": str(adapter_file) if adapter_file.is_file() else None,
            "finetune_report": report,
        }

    def route(self, query: str) -> NeedleDecision:
        """Route a plain-English command or headline to validated SAVIOUR tool calls."""
        clean = (query or "").strip()
        if not clean:
            return NeedleDecision(
                query="",
                type="refusal",
                function_calls=[],
                confidence=None,
                engine="charter_shield",
                refusal_reason="Empty input.",
            )

        prohibited = charter_refusal_reason(clean)
        if prohibited:
            return NeedleDecision(
                query=clean,
                type="refusal",
                function_calls=[],
                confidence=1.0,
                engine="charter_shield",
                refusal_reason=prohibited,
            )

        native = self._try_native_cact(clean)
        if native is not None:
            return native

        calls = deterministic_route(clean)
        if not calls:
            return NeedleDecision(
                query=clean,
                type="refusal",
                function_calls=[],
                confidence=None,
                engine="deterministic_fallback",
                refusal_reason=(
                    "No SAVIOUR action matched. Try asking for signals, bot status, pause/resume, "
                    "Telegram format preset, guard settings, or a market news headline."
                ),
            )
        return NeedleDecision(
            query=clean,
            type="tool_call",
            function_calls=calls,
            confidence=None,
            engine="deterministic_fallback",
        )


def format_blackout_window_entry(
    arguments: Mapping[str, Any],
    *,
    at: datetime | None = None,
) -> str | None:
    """Convert a ``HIGH`` ``extract_news_shock`` call into a ``NEWS_BLACKOUT_WINDOWS`` entry."""
    severity = str(arguments.get("severity") or "NONE").upper()
    minutes = int(arguments.get("blackout_minutes") or 0)
    if severity != "HIGH" or minutes <= 0:
        return None
    stamp = (at or utc_now()).astimezone(timezone.utc)
    start = (stamp - timedelta(minutes=2)).strftime("%Y-%m-%dT%H:%MZ")
    end = (stamp + timedelta(minutes=min(180, minutes))).strftime("%Y-%m-%dT%H:%MZ")
    raw_label = re.sub(r"[,@/]+", " ", str(arguments.get("event") or "News Shock")).strip()
    label = raw_label[:32] or "News Shock"
    return f"{label}@{start}/{end}"


def execute_on_bot(
    bot: Any,
    query: str,
    *,
    agent: SaviourNeedleAgent | None = None,
    at: datetime | None = None,
) -> tuple[NeedleDecision, str]:
    """Route ``query`` via Needle 3 and execute matched read/control tools on ``FusionBot``."""
    router = agent or SaviourNeedleAgent(state_dir=bot.config.local_state_dir)
    decision = router.route(query)
    if decision.type != "tool_call" or not decision.function_calls:
        return decision, decision.refusal_reason or "Request declined."

    outputs: list[str] = []
    for call in decision.function_calls:
        name = call["name"]
        args = call["arguments"]

        if name == "bot_status":
            outputs.append(bot.status_text())

        elif name == "pause_signals":
            bot._set_paused(True)
            bot._save_state(force=True)
            outputs.append("Signals paused. Settlement and learning continue.")

        elif name == "resume_signals":
            bot._set_paused(False)
            bot._save_state(force=True)
            outputs.append("Signals resumed.")

        elif name == "scan_market":
            pair = str(args.get("pair") or "ALL").upper()
            if pair != "ALL" and pair in bot.peer_bots:
                outputs.append(bot.peer_bots[pair].on_demand_signal(at=at))
            elif pair != "ALL" and pair == bot.config.pair and bot.peer_bots:
                # Evaluate only primary instrument without sweeping peers
                saved_peers = bot.peer_bots
                try:
                    bot.peer_bots = {}
                    outputs.append(bot.on_demand_signal(at=at))
                finally:
                    bot.peer_bots = saved_peers
            else:
                outputs.append(bot.on_demand_signal(at=at))

        elif name == "set_telegram_style":
            sig_style = str(args.get("signal_style") or "standard")
            res_style = str(args.get("result_style") or "standard")
            bot.telegram_format = telegram.TelegramFormatConfig(
                signal_style=sig_style,
                signal_template=bot.telegram_format.signal_template,
                result_style=res_style,
                result_template=bot.telegram_format.result_template,
                apply_to_signals_cmd=True,
                append_paper_footer=True,
            )
            bot.config.telegram_signal_style = bot.telegram_format.signal_style
            bot.config.telegram_result_style = bot.telegram_format.result_style
            for peer in bot.peer_bots.values():
                peer.telegram_format = bot.telegram_format
            bot._save_state(force=True)
            outputs.append(
                f"Telegram format updated: signal={bot.telegram_format.signal_style}, "
                f"result={bot.telegram_format.result_style}."
            )

        elif name == "configure_guards":
            changes: list[str] = []
            for member in (bot, *bot.peer_bots.values()):
                if "mtf_minutes" in args:
                    member.config.mtf_minutes = int(args["mtf_minutes"])
                    member.config.mtf_filter = True
                if "cooldown_minutes" in args:
                    member.config.signal_cooldown_minutes = int(args["cooldown_minutes"])
                if "horizon_minutes" in args:
                    member.config.horizon_minutes = int(args["horizon_minutes"])
            if "mtf_minutes" in args:
                changes.append(f"MTF={args['mtf_minutes']}m (enabled)")
            if "cooldown_minutes" in args:
                changes.append(f"cooldown={args['cooldown_minutes']}m")
            if "horizon_minutes" in args:
                changes.append(f"horizon={args['horizon_minutes']}m")
            outputs.append("Runtime guards updated: " + ", ".join(changes) + ".")

        elif name == "extract_news_shock":
            entry = format_blackout_window_entry(args, at=at)
            if entry:
                for member in (bot, *bot.peer_bots.values()):
                    existing = member.config.news_blackout_windows.strip()
                    member.config.news_blackout_windows = f"{existing},{entry}" if existing else entry
                    member.config.news_filter = True
                outputs.append(
                    f"HIGH news shock detected ({args['event']} on {args['pair']}): "
                    f"applied {args['blackout_minutes']}m blackout window ({entry})."
                )
            else:
                outputs.append(
                    f"Headline classified ({args['event']} on {args['pair']}): "
                    f"severity={args['severity']}, no blackout needed."
                )

    return decision, "\n\n".join(outputs)
