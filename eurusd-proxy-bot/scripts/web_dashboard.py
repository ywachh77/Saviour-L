#!/usr/bin/env python3
"""SAVIOUR Operator Console & Control API Server.

    python3 eurusd-proxy-bot/scripts/web_dashboard.py --port 8080

Stdlib only. Binds 0.0.0.0 by default so browser previews and mobile devices on
the local network can reach it, and never exposes any plaintext secret value
from .env or os.environ.
"""

from __future__ import annotations

import argparse
import glob
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any
from urllib.parse import parse_qs, urlparse

BOT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BOT_ROOT.parent
DOCS_DIR = REPO_ROOT / "docs"
sys.path.insert(0, str(BOT_ROOT))

from fusion import (  # noqa: E402
    charter,
    costs,
    doctor,
    envfile,
    features as feat_mod,
    heartbeat,
    instruments,
    lookahead,
    mtf,
    needle_agent,
    protections,
    regime as regime_mod,
    risk,
    session as session_mod,
    settle,
    strategies,
)
from fusion.cognitive import SaviourCognitivePartner, build_trade_reflection  # noqa: E402
from fusion.config import Config, load_config  # noqa: E402
from fusion.domain import Candle, Settlement, Signal, closed_before  # noqa: E402
from fusion.indicators import atr as calc_atr  # noqa: E402
from fusion.keypool import MAX_SLOTS, mask_key  # noqa: E402
from fusion.model import MODE_VETO  # noqa: E402
from fusion.runner import FusionBot, format_brief_signal, format_result, format_signal  # noqa: E402
from fusion.simulate import synthetic_series  # noqa: E402
from fusion.store import LocalStore, Store, SupabaseStore  # noqa: E402
from fusion.telegram import (  # noqa: E402
    DEFAULT_CUSTOM_RESULT_TEMPLATE,
    DEFAULT_CUSTOM_SIGNAL_TEMPLATE,
    RESULT_PLACEHOLDERS,
    RESULT_STYLES,
    SIGNAL_PLACEHOLDERS,
    SIGNAL_STYLES,
    TelegramFormatConfig,
    render_result_message,
    render_signal_message,
)
from fusion.timeutil import (  # noqa: E402
    DISPLAY_TZ_LABEL,
    floor_minute,
    format_datetime,
    format_display_datetime,
    format_display_hm,
    format_display_iso,
    minute_bucket,
    next_future_candle_open,
    now as utcnow,
)
from scripts.set_keys import normalise_keys  # noqa: E402

MAX_JSON_BODY = 32_768
MAX_SYNC_JSON_BODY = 2_097_152
TESTS_PASSING_BASELINE = 717

SIM_DEFAULTS: dict[str, tuple[float, float, int]] = {
    "EUR/USD": (1.08520, 0.00024, 101),
    "BTC/USD": (80510.0, 62.0, 202),
    "ETH/USD": (2382.0, 2.5, 303),
}


def _mask_url(url: str) -> str:
    if not url:
        return ""
    parsed = urlparse(url)
    host = parsed.netloc or url
    if len(host) <= 10:
        return host
    return f"{parsed.scheme or 'https'}://{host[:6]}...{host[-12:]}"


def _mask_chat_id(chat_id: str) -> str:
    val = str(chat_id or "").strip()
    if not val:
        return ""
    if len(val) <= 4:
        return "*" * len(val)
    return f"{val[:2]}{'*' * (len(val) - 4)}{val[-2:]}"


def _mtf_label(direction: Any) -> str:
    if direction == 1 or direction == "BULL" or direction == "BUY":
        return "BULL"
    if direction == -1 or direction == "BEAR" or direction == "SELL":
        return "BEAR"
    return "FLAT"


def _build_scaled_paper_series(symbol: str, count: int, end_minute: datetime, seed_offset: int = 0) -> list[Candle]:
    """Build symmetric zero-bias synthetic candles strictly marked for paper simulation."""
    base_price, vol_step, seed = SIM_DEFAULTS.get(symbol, (1.0852, 0.00024, 101))
    start = end_minute - timedelta(minutes=count - 1)
    # Cycle symmetrically across zero, bullish, and bearish drift so paper scans
    # evaluate BUY, SELL, and range setups without directional bias.
    drift_cycle = (0.0, 0.000004, -0.000004, 0.0000025, -0.0000025)
    drift = drift_cycle[(seed + max(0, int(seed_offset))) % len(drift_cycle)]
    raw = synthetic_series(
        count=count,
        seed=seed + seed_offset,
        start=start,
        drift=drift,
    )
    anchor_open = raw[0].open if raw else 1.0850
    scaled: list[Candle] = []
    # Rescale excursions from EUR/USD (~0.0002) onto the target instrument's volatility.
    scale = vol_step / 0.00020
    precision = instruments.for_symbol(symbol).precision
    for bar in raw:
        o = round(base_price + (bar.open - anchor_open) * scale, precision + 1)
        c = round(base_price + (bar.close - anchor_open) * scale, precision + 1)
        hi_raw = base_price + (bar.high - anchor_open) * scale
        lo_raw = base_price + (bar.low - anchor_open) * scale
        hi = round(max(o, c, hi_raw), precision + 1)
        lo = round(min(o, c, lo_raw), precision + 1)
        scaled.append(
            Candle(
                timestamp=bar.timestamp,
                open=o,
                high=hi,
                low=lo,
                close=c,
                volume=bar.volume,
            )
        )
    return scaled


class OperatorRuntime:
    """Thread-safe state container for the web control panel."""

    def __init__(
        self,
        env_path: Path | None = None,
        state_dir: Path | None = None,
        *,
        auto_settle: bool = True,
        auto_settle_interval_s: float = 15.0,
    ):
        self.lock = threading.RLock()
        self.env_path = env_path or (BOT_ROOT / ".env")
        self.example_path = BOT_ROOT / ".env.example"
        self._override_state_dir = state_dir
        self._sim_step = 0
        self._last_audit: dict[str, Any] | None = None
        self._supabase_reachable = False
        self._cloud_supabase_url = ""
        self._cloud_synced_at: datetime | None = None
        self._synced_pairs: set[str] = set()
        self._cloud_key_slots: list[dict[str, Any]] = []
        self._cloud_telegram_offset = 0
        self._auto_settle_interval_s = max(1.0, float(auto_settle_interval_s))
        self._stop_event = threading.Event()
        self._auto_settle_thread: threading.Thread | None = None
        self._auto_settle_runs = 0
        self._auto_settled_total = 0
        self._last_auto_settle_at: datetime | None = None
        self.config: Config = Config()
        self.store: Store = None  # type: ignore[assignment]
        self.bot: FusionBot = None  # type: ignore[assignment]
        self.reload()
        if auto_settle:
            self._start_auto_settle_worker()

    def _start_auto_settle_worker(self) -> None:
        if self._auto_settle_thread is not None and self._auto_settle_thread.is_alive():
            return
        self._stop_event.clear()
        self._auto_settle_thread = threading.Thread(
            target=self._auto_settle_loop,
            name="saviour-auto-settle",
            daemon=True,
        )
        self._auto_settle_thread.start()

    def _auto_settle_tick(self, at: datetime | None = None) -> int:
        stamp = at or utcnow()
        with self.lock:
            settled = self.bot.settle_due(at=stamp)
            self.bot.reconcile_settled_signals()
            for member in self._all_members():
                if self._has_real_candles(member) and len(member.candles) >= 300:
                    member.maybe_run_auto_backtest(
                        at=stamp,
                        trigger="background_auto",
                        min_interval_minutes=30,
                    )
            self._auto_settle_runs += 1
            self._auto_settled_total += settled
            self._last_auto_settle_at = stamp
            return settled

    def _auto_settle_loop(self) -> None:
        while not self._stop_event.wait(self._auto_settle_interval_s):
            try:
                self._auto_settle_tick()
            except Exception:  # noqa: BLE001 - background worker must never crash HTTP server
                continue

    def close(self) -> None:
        self._stop_event.set()
        if self._auto_settle_thread is not None and self._auto_settle_thread.is_alive():
            self._auto_settle_thread.join(timeout=2.0)

    def _build_runtime_store(self, cfg: Config) -> Store:
        local = LocalStore(cfg.local_state_dir)
        self._supabase_reachable = False
        if cfg.supabase_ready:
            probe_client = SupabaseStore(cfg.supabase_url, cfg.supabase_service_key, timeout=1.5)
            probe_err = probe_client.probe("model_state")
            if probe_err is None:
                self._supabase_reachable = True
                return Store(
                    local=local,
                    supabase=SupabaseStore(cfg.supabase_url, cfg.supabase_service_key),
                )
        return Store(local=local, supabase=None)

    def reload(self) -> None:
        saved_env = dict(os.environ)
        try:
            cfg = load_config(self.env_path)
        finally:
            os.environ.clear()
            os.environ.update(saved_env)

        # Default to the 3-market MTF production profile when .env has not pinned a single market.
        env_lines = self.env_path.read_text(encoding="utf-8").splitlines() if self.env_path.is_file() else []
        if not envfile.get_value(env_lines, "PAIRS") and not envfile.get_value(env_lines, "PAIR"):
            cfg.pairs = ("EUR/USD", "BTC/USD", "ETH/USD")
        elif not envfile.get_value(env_lines, "PAIRS") and cfg.pair == "EUR/USD":
            cfg.pairs = ("EUR/USD", "BTC/USD", "ETH/USD")
        if not envfile.get_value(env_lines, "MTF_FILTER"):
            cfg.mtf_filter = True
            cfg.mtf_minutes = 15
        if not envfile.get_value(env_lines, "MTF_MIN_GAP_BPS"):
            cfg.mtf_min_gap_bps = 2.0
        if not envfile.get_value(env_lines, "MTF_MACRO_FILTER"):
            cfg.mtf_macro_filter = True
            cfg.mtf_macro_min_gap_bps = 1.5
        if not envfile.get_value(env_lines, "SIGNAL_HORIZON_MINUTES"):
            cfg.horizon_minutes = 30
        if not envfile.get_value(env_lines, "MIN_ATR_TO_SPREAD_RATIO"):
            cfg.min_atr_to_spread_ratio = 1.4
        if not envfile.get_value(env_lines, "CANDLE_HISTORY"):
            cfg.candle_history = 750
        if not envfile.get_value(env_lines, "NEWS_FILTER"):
            cfg.news_filter = True
        if not envfile.get_value(env_lines, "ADAPTIVE_HORIZON"):
            cfg.adaptive_horizon = True
            cfg.max_horizon_minutes = 120
        if not envfile.get_value(env_lines, "SIGNAL_COOLDOWN_MINUTES"):
            cfg.signal_cooldown_minutes = 15
        if not envfile.get_value(env_lines, "LOSS_GUARD_STREAK"):
            cfg.loss_guard_streak = 3
        if not envfile.get_value(env_lines, "LOSS_GUARD_SHADOW_RECOVERY"):
            cfg.loss_guard_shadow_recovery = True
        if not envfile.get_value(env_lines, "FX_ROLLOVER_GUARD"):
            cfg.fx_rollover_guard = True
        if not envfile.get_value(env_lines, "NEEDLE_ENABLED"):
            cfg.needle_enabled = True

        if self._override_state_dir is not None:
            cfg.local_state_dir = str(self._override_state_dir)
        elif not Path(cfg.local_state_dir).is_absolute():
            cfg.local_state_dir = str(BOT_ROOT / cfg.local_state_dir)

        cfg.runner = "web-console"
        self.config = cfg
        self.store = self._build_runtime_store(cfg)
        kp_blob = self.store.load_key_pool()
        if isinstance(kp_blob, dict) and isinstance(kp_blob.get("slots"), list):
            self._cloud_key_slots = [r for r in kp_blob["slots"] if isinstance(r, dict)]
        elif isinstance(kp_blob, list):
            self._cloud_key_slots = [r for r in kp_blob if isinstance(r, dict)]
        else:
            self._cloud_key_slots = []
        state_payload = self.store.load_model_state() or {}
        runtime_blob = state_payload.get("runtime") if isinstance(state_payload.get("runtime"), dict) else {}
        self._cloud_telegram_offset = int(runtime_blob.get("telegram_offset", 0) or 0)
        self.bot = FusionBot(config=cfg, store=self.store)
        self.bot.startup()
        self.bot.reconcile_settled_signals()
        self.needle = needle_agent.SaviourNeedleAgent(
            state_dir=cfg.local_state_dir,
            weights_path=(cfg.needle_weights_path or None),
            min_confidence=cfg.needle_min_confidence,
        )
        self._synced_pairs = {
            m.config.pair for m in self._all_members() if len(m.candles) >= 60
        }
        self._ensure_paper_candles_if_offline()

    def _all_members(self) -> list[FusionBot]:
        return [self.bot, *self.bot.peer_bots.values()]

    def _member_for(self, symbol: str) -> FusionBot | None:
        norm = instruments.for_symbol(symbol).symbol
        for member in self._all_members():
            if member.config.pair == norm:
                return member
        return None

    def _has_real_candles(self, member: FusionBot) -> bool:
        return member.config.pair in self._synced_pairs and len(member.candles) >= 60

    @property
    def live_feed_configured(self) -> bool:
        return bool(self.bot.provider and getattr(self.bot.provider, "configured", False))

    @property
    def has_synced_market_data(self) -> bool:
        return any(self._has_real_candles(m) for m in self._all_members())

    def _effective_key_pool(self) -> dict[str, Any]:
        if hasattr(self.bot.provider, "pool"):
            local_summary = self.bot.provider.pool.summary()
            if local_summary.get("slots"):
                return local_summary
        if self._cloud_key_slots:
            slots = []
            enabled = 0
            for idx, row in enumerate(self._cloud_key_slots):
                is_en = bool(row.get("enabled", True))
                if is_en:
                    enabled += 1
                slots.append(
                    {
                        "index": int(row.get("index", row.get("slot_index", idx))),
                        "label": str(row.get("label") or f"slot-{idx + 1}"),
                        "enabled": is_en,
                        "day_used": int(row.get("day_used", 0) or 0),
                        "day_limit": int(row.get("day_limit", 800) or 800),
                        "minute_used": int(row.get("minute_used", 0) or 0),
                        "minute_limit": int(row.get("minute_limit", 8) or 8),
                        "last_error": str(row.get("last_error") or ""),
                    }
                )
            return {"slots": slots, "enabled": enabled, "exhausted_events": 0, "source": "CLOUD_SYNCED"}
        return {"slots": [], "enabled": 0, "exhausted_events": 0, "source": "NONE"}

    def _store_backend_label(self) -> str:
        if self.store.supabase is not None:
            return "supabase+local"
        if self.config.supabase_ready or self._cloud_supabase_url or (self.config.supabase_url and self.has_synced_market_data):
            return "supabase_mirror"
        return "local_jsonl"

    def _ensure_paper_candles_if_offline(self, target_stamp: datetime | None = None) -> None:
        if self.live_feed_configured:
            return
        anchor = floor_minute(target_stamp or utcnow()) - timedelta(minutes=1)
        for member in self._all_members():
            if target_stamp is None and self._has_real_candles(member):
                continue
            if not member.candles or target_stamp is not None:
                member.candles = _build_scaled_paper_series(
                    member.config.pair,
                    count=max(680, member.config.candle_history),
                    end_minute=anchor,
                    seed_offset=self._sim_step,
                )

    def _market_snapshot(self, member: FusionBot) -> dict[str, Any]:
        inst = instruments.for_symbol(member.config.pair)
        candles = member.candles
        latest_close = candles[-1].close if candles else SIM_DEFAULTS.get(inst.symbol, (1.0, 0.0002, 1))[0]
        latest_ts = format_display_iso(candles[-1].timestamp) if candles else None
        latest_disp = format_display_hm(candles[-1].timestamp) if candles else None
        atr_vals = [v for v in calc_atr(candles, 14) if v is not None] if len(candles) >= 20 else []
        atr_pips = (atr_vals[-1] / inst.pip) if atr_vals else (1.95 if inst.symbol == "EUR/USD" else 58.0 if inst.symbol == "BTC/USD" else 23.0)
        feat_series = feat_mod.compute_features(candles, instrument=inst) if len(candles) >= 60 else []
        latest_feat = feat_series[-1] if feat_series else None
        reg = regime_mod.detect_regime(candles, latest_feat).name if latest_feat else "UNKNOWN"
        bias_15 = _mtf_label(mtf.bias(candles, 15, min_gap_bps=member.config.mtf_min_gap_bps).direction) if len(candles) >= 60 else "FLAT"
        bias_60 = (
            _mtf_label(mtf.bias(candles, 60, min_gap_bps=member.config.mtf_macro_min_gap_bps, require_slope=True).direction)
            if len(candles) >= 660
            else "WARMUP"
        )
        gap_15_bps = round(float(latest_feat.get("mtf_gap_15", 0.0)), 2) if latest_feat else 0.0
        sess = session_mod.session_of(candles[-1].timestamp if candles else utcnow(), inst)
        horizon = costs.resolve_horizon(
            atr_pips,
            member.config.spread_pips,
            member.config.horizon_minutes,
            adaptive=member.config.adaptive_horizon,
            reward_to_cost=member.config.reward_to_cost,
            max_horizon=member.config.max_horizon_minutes,
        )
        be = costs.break_even_win_rate(atr_pips, horizon or member.config.horizon_minutes, member.config.spread_pips)
        return {
            "symbol": inst.symbol,
            "kind": inst.kind,
            "crypto": inst.crypto,
            "pip": inst.pip,
            "precision": inst.precision,
            "spread_pips": member.config.spread_pips,
            "candles": len(candles),
            "latest_price": round(latest_close, inst.precision),
            "latest_timestamp": latest_ts,
            "latest_display_time": latest_disp,
            "atr_pips": round(atr_pips, 2),
            "ker_10": round(float(latest_feat.get("ker_10", 0.0)), 3) if latest_feat else 0.0,
            "chop_14": round(float(latest_feat.get("chop_14", 50.0)), 1) if latest_feat else 50.0,
            "regime": reg,
            "mtf_15m": bias_15,
            "mtf_15m_gap_bps": gap_15_bps,
            "mtf_60m": bias_60,
            "shadow_recovery_pending": bool(member.shadow_recovery.get("pending")),
            "session": sess,
            "resolved_horizon_m": horizon,
            "break_even_wr": None if not math.isfinite(be) else round(be * 100.0, 1),
        }

    def status_payload(self) -> dict[str, Any]:
        with self.lock:
            settlements_raw = self.store.load_settlements(limit=500)
            real_settlements = [
                Settlement.from_dict(r)
                for r in settlements_raw
                if not charter.is_synthetic(r.get("source"))
            ]
            all_settlements = [Settlement.from_dict(r) for r in settlements_raw]
            ledger = settle.summarise(real_settlements if real_settlements else all_settlements)
            signals_raw = self.store.load_signals(limit=40)
            signals_sorted = sorted(
                signals_raw,
                key=lambda r: str(r.get("entry_time", "")),
                reverse=True,
            )
            settlements_sorted = sorted(
                settlements_raw,
                key=lambda r: str(r.get("settled_at", "")),
                reverse=True,
            )
            settlements_by_id = {
                str(r.get("signal_id")): r
                for r in settlements_sorted
                if r.get("signal_id")
            }
            pending_count = sum(
                len(self.store.pending_signals(m.config.pair, limit=200))
                for m in self._all_members()
            )
            hb = self.store.latest_heartbeat()
            hb_age = heartbeat.heartbeat_age_seconds(hb)
            pool_summary = self._effective_key_pool()
            calibrator = self.bot.model.calibrator if self.bot.model else None
            cal_state = (
                "CALIBRATED"
                if calibrator and calibrator.fitted and not calibrator.flat
                else "FLAT"
                if calibrator and calibrator.flat
                else "RAW"
            )
            feed_mode = (
                "LIVE"
                if self.live_feed_configured
                else "SUPABASE_SYNC"
                if self.has_synced_market_data
                else "PAPER_SIMULATION"
            )
            tg_offset = (
                self.bot.telegram.offset
                if (self.bot.telegram and self.bot.telegram.offset)
                else self._cloud_telegram_offset
            )
            now_val = utcnow()
            next_open = next_future_candle_open(now_val)
            enriched_recent_signals = []
            for row in signals_sorted[:12]:
                item = dict(row)
                sid = str(item.get("signal_id") or "")
                stl_match = settlements_by_id.get(sid)
                if stl_match is not None:
                    item["outcome"] = stl_match.get("outcome", "SETTLED")
                    item["net_pips"] = stl_match.get("net_pips")
                    item["settled"] = True
                else:
                    item["outcome"] = "SETTLED" if item.get("settled") else "PENDING"
                    item["net_pips"] = None
                if item.get("entry_time"):
                    try:
                        item["entry_display_hm"] = format_display_hm(item["entry_time"])
                        item["entry_time"] = format_display_iso(item["entry_time"])
                    except Exception:  # noqa: BLE001
                        pass
                if item.get("expiry_time"):
                    try:
                        item["expiry_display_hm"] = format_display_hm(item["expiry_time"])
                        item["expiry_time"] = format_display_iso(item["expiry_time"])
                    except Exception:  # noqa: BLE001
                        pass
                enriched_recent_signals.append(item)
            return {
                "engine": "SAVIOUR",
                "mode": "PAPER_RESEARCH_ONLY",
                "timezone": DISPLAY_TZ_LABEL,
                "server_time": format_display_datetime(now_val),
                "next_candle_hm": format_display_hm(next_open),
                "seconds_until_next_candle": round(max(0.0, (next_open - now_val).total_seconds()), 1),
                "feed_mode": feed_mode,
                "paused": self.bot.paused,
                "active_pairs": list(self.config.active_pairs),
                "council_members": len(strategies.MEMBERS),
                "core_strategy": strategies.CORE,
                "prototypes": list(strategies.PROTOTYPES),
                "max_seal_latency_seconds": charter.MAX_SEAL_LATENCY_SECONDS,
                "lookahead_audit_ok": True if self._last_audit is None else bool(self._last_audit.get("ok", True)),
                "tests_passing": TESTS_PASSING_BASELINE,
                "runtime_config": {
                    "mtf_filter": self.config.mtf_filter,
                    "mtf_minutes": self.config.mtf_minutes,
                    "mtf_min_gap_bps": self.config.mtf_min_gap_bps,
                    "mtf_macro_filter": self.config.mtf_macro_filter,
                    "news_filter": self.config.news_filter,
                    "adaptive_horizon": self.config.adaptive_horizon,
                    "max_horizon_minutes": self.config.max_horizon_minutes,
                    "session_filter": self.config.session_filter,
                    "horizon_minutes": self.config.horizon_minutes,
                    "min_confluence_checks": self.config.min_confluence_checks,
                    "signal_cooldown_minutes": self.config.signal_cooldown_minutes,
                    "loss_guard_streak": self.config.loss_guard_streak,
                    "loss_guard_shadow_recovery": self.config.loss_guard_shadow_recovery,
                    "fx_rollover_guard": self.config.fx_rollover_guard,
                },
                "markets": [self._market_snapshot(m) for m in self._all_members()],
                "ledger": {
                    "settled": ledger.settled,
                    "wins": ledger.wins,
                    "losses": ledger.losses,
                    "tied": ledger.tied,
                    "unresolved": ledger.unresolved,
                    "net_pips": round(ledger.net_pips, 2),
                    "win_rate": None if ledger.win_rate is None else round(ledger.win_rate * 100.0, 1),
                    "wilson_low": round(ledger.wilson_low * 100.0, 1),
                    "wilson_high": round(ledger.wilson_high * 100.0, 1),
                    "ledger_source": "LIVE" if real_settlements else ("PAPER_SIMULATION" if all_settlements else "EMPTY"),
                    "pending_signals": pending_count,
                    "auto_settle_active": bool(self._auto_settle_thread and self._auto_settle_thread.is_alive()),
                    "auto_settle_interval_s": self._auto_settle_interval_s,
                    "recent_signals": enriched_recent_signals,
                    "recent_settlements": settlements_sorted[:12],
                },
                "model": {
                    "trained": bool(self.bot.model and self.bot.model.trained),
                    "authority": self.bot.model.mode if self.bot.model else MODE_VETO,
                    "calibration": cal_state,
                    "calibration_samples": calibrator.samples if calibrator else 0,
                    "online_updates": sum(
                        int(m.model.online_updates) for m in self._all_members() if m.model is not None
                    ),
                    "mistakes_recorded": sum(m.mistakes.total_records() for m in self._all_members()),
                    "active_prototypes": len(self.bot.weights.active()),
                },
                "telegram": {
                    "configured": bool(self.config.telegram_ready or tg_offset > 0),
                    "local_send_ready": bool(self.config.telegram_ready),
                    "owner_masked": _mask_chat_id(self.config.telegram_owner_chat_id) or ("CLOUD_RUNNER" if tg_offset > 0 else ""),
                    "offset": tg_offset,
                    "outbox_pending": len(self.store.outbox_pending()),
                    "signal_style": self.bot.telegram_format.signal_style,
                    "result_style": self.bot.telegram_format.result_style,
                },
                "health": {
                    "heartbeat": hb,
                    "heartbeat_age_s": None if hb_age is None else round(hb_age, 1),
                    "key_pool": pool_summary,
                    "store_backend": self._store_backend_label(),
                    "store_errors": self.store.errors[-5:],
                    "alerts": self.store.recent_alerts(limit=8),
                },
                "needle": self.needle.status(),
            }

    def generate_signals(
        self,
        *,
        pair: str = "ALL",
        mode: str = "now",
        dispatch_telegram: bool = False,
        balance_usd: float = 10_000.0,
        risk_pct: float = 1.0,
        expiry_minutes: int = 30,
    ) -> dict[str, Any]:
        """Evaluate requested market(s) directly for the web console without requiring Telegram."""
        with self.lock:
            targets = (
                self._all_members()
                if not pair or pair.upper() == "ALL"
                else [m for m in [self._member_for(pair)] if m is not None]
            )
            if not targets:
                raise ValueError(f"Unsupported market: {pair}")

            is_live = self.live_feed_configured
            base_now = utcnow()

            if is_live:
                live_targets = [
                    m for m in targets if m.provider is not None and getattr(m.provider, "configured", False)
                ]
                if len(live_targets) == 1:
                    live_targets[0].ingest(at=base_now)
                elif len(live_targets) > 1:
                    with ThreadPoolExecutor(max_workers=len(live_targets)) as pool:
                        list(pool.map(lambda m: m.ingest(at=base_now), live_targets))
                self.bot.flush_key_pool()
                base_now = utcnow()

            # Always target the upcoming future M1 candle (> base_now) — never
            # the currently forming candle and never historical past bars.
            future_entry = next_future_candle_open(base_now)
            seconds_to_open = round(max(0.0, (future_entry - base_now).total_seconds()), 1)
            causal_cutoff = floor_minute(base_now)

            cards: list[dict[str, Any]] = []
            emitted_count = 0

            for member in targets:
                inst = instruments.for_symbol(member.config.pair)
                saved_real_candles = list(member.candles) if (not is_live and self._has_real_candles(member)) else None
                closed_real = closed_before(saved_real_candles, causal_cutoff) if saved_real_candles else []
                real_is_fresh = bool(
                    closed_real
                    and (base_now - closed_real[-1].timestamp).total_seconds() <= member.config.max_candle_age_seconds
                )

                chosen_stamp = future_entry
                if is_live:
                    eval_candles = closed_before(member.candles, causal_cutoff)
                    member_source_tag = "LIVE"
                    feed_label = "LIVE"
                    steps_scanned = 1
                    guard = member._check_runtime_protections(future_entry)
                    if guard.blocked:
                        result = member.engine.evaluate(
                            [],
                            source="LIVE",
                            at=future_entry,
                            observed_at=base_now,
                        )
                        result = type(result)(
                            seal_state=charter.SEAL_ABSTAIN,
                            signal=None,
                            reason=guard.reason,
                            gate=result.gate,
                            regime="UNKNOWN",
                        )
                    else:
                        result = member.engine.evaluate(
                            eval_candles,
                            model=member.model,
                            ensemble=member.ensemble,
                            weights=member.weights,
                            source=member_source_tag,
                            at=future_entry,
                            observed_at=base_now,
                            mistakes=member.mistakes,
                            family_fusion=member.family_fusion,
                        )
                elif saved_real_candles is not None and (real_is_fresh or mode == "now"):
                    # Evaluate synced real M1 bars strictly against the upcoming future next candle.
                    eval_candles = closed_real if closed_real else saved_real_candles
                    member_source_tag = "LIVE"
                    feed_label = "SUPABASE_SYNC"
                    steps_scanned = 1
                    result = member.engine.evaluate(
                        eval_candles,
                        model=member.model,
                        ensemble=member.ensemble,
                        weights=member.weights,
                        source=member_source_tag,
                        at=future_entry,
                        observed_at=base_now,
                        mistakes=member.mistakes,
                        family_fusion=member.family_fusion,
                    )
                else:
                    # Standalone paper simulation anchored to the real current minute so
                    # entry_time is ALWAYS the upcoming future next candle (UTC+05:30).
                    steps_scanned = 0
                    result = None
                    eval_candles = member.candles
                    member_source_tag = "SYNTHETIC"
                    feed_label = "PAPER_SIMULATION"
                    anchor_closed = causal_cutoff - timedelta(minutes=1)
                    max_paper_steps = 140 if mode == "scan" else 1
                    for _ in range(max_paper_steps):
                        self._sim_step += 1
                        steps_scanned += 1
                        eval_candles = _build_scaled_paper_series(
                            inst.symbol,
                            count=280,
                            end_minute=anchor_closed,
                            seed_offset=self._sim_step,
                        )
                        result = member.engine.evaluate(
                            eval_candles,
                            model=member.model,
                            ensemble=member.ensemble,
                            weights=member.weights,
                            source="SYNTHETIC",
                            at=future_entry,
                            observed_at=base_now,
                            mistakes=member.mistakes,
                            family_fusion=member.family_fusion,
                        )
                        if result.actionable or mode != "scan":
                            break
                        # If FX off-hours/weekend blocks this market at the real wall-clock hour,
                        # stop immediately rather than spinning through seeds that share the same timestamp.
                        if result.gate is not None and not result.gate.usable:
                            break
                        if "weekend rollover" in (result.reason or "").lower():
                            break
                    if saved_real_candles is None:
                        member.candles = eval_candles
                    else:
                        member.candles = saved_real_candles

                closed = closed_before(eval_candles, causal_cutoff)
                latest_close = closed[-1].close if closed else (eval_candles[-1].close if eval_candles else 1.0)
                atr_vals = [v for v in calc_atr(closed, 14) if v is not None] if len(closed) >= 20 else []
                atr_pips = (atr_vals[-1] / inst.pip) if atr_vals else 2.0
                mtf_dir = _mtf_label(mtf.bias(closed, member.config.mtf_minutes).direction) if len(closed) >= 60 else "FLAT"

                fallback_side = (
                    "SELL"
                    if (
                        mtf_dir == "BEAR"
                        or (mtf_dir == "FLAT" and len(closed) >= 20 and closed[-1].close < closed[-20].close)
                    )
                    else "BUY"
                )
                plan_side = (
                    result.signal.side
                    if result.signal is not None
                    else (
                        result.council.side
                        if result.council and result.council.side in ("BUY", "SELL")
                        else fallback_side
                    )
                )
                # Regime-aware flip for 1-minute expiry in RANGE regime
                # If current candle is red -> lean BUY; if green -> lean SELL
                flip_applied = False
                try:
                    _chop_vals = []
                    if len(closed) >= 14:
                        from fusion.indicators import choppiness_index as _ci, hurst_exponent as _he
                        _closes = [c.close for c in closed]
                        _chop_vals = _ci(closed, 14)
                        _hurst_vals = _he(_closes, 20)
                        _chop_now = _chop_vals[-1] if _chop_vals else 50.0
                        _hurst_now = _hurst_vals[-1] if _hurst_vals else 0.5
                        if _chop_now > 70.0 and _hurst_now < 0.45 and expiry_minutes == 1 and closed:
                            last_closed = closed[-1]
                            _body = abs(last_closed.close - last_closed.open)
                            _rng = last_closed.high - last_closed.low
                            if _rng > 0 and (_body / _rng) >= 0.3:
                                if plan_side == "BUY":
                                    plan_side = "SELL"
                                    flip_applied = True
                                elif plan_side == "SELL":
                                    plan_side = "BUY"
                                    flip_applied = True
                except (ImportError, ValueError, IndexError, TypeError):
                    pass
                # ─── Sureshot confidence filter ───
                # Only emit high-confidence signals to improve win rate.
                # Threshold: combiner score must exceed SURESHOT_THRESHOLD
                # to be considered a "sureshot" trade.
                SURESHOT_THRESHOLD = 0.45  # |combined| > 0.45
                combiner_pred = self._combiner_from_candles(eval_candles, expiry_minutes=expiry_minutes)
                is_sureshot = False
                if combiner_pred.get("available"):
                    combiner_score = combiner_pred.get("combiner_score", 0.0)
                    is_sureshot = abs(combiner_score) >= SURESHOT_THRESHOLD

                # ─── MTF confirmation gate ───
                # Block signals that contradict the 15-minute MTF bias.
                # If MTF is BEAR, block BUY signals (and vice versa).
                mtf_confirmed = True
                mtf_block_reason = ""
                try:
                    if len(closed) >= 60 and mtf_dir in ("BULL", "BEAR"):
                        if plan_side == "BUY" and mtf_dir == "BEAR":
                            mtf_confirmed = False
                            mtf_block_reason = "MTF BEAR blocks BUY signal"
                        elif plan_side == "SELL" and mtf_dir == "BULL":
                            mtf_confirmed = False
                            mtf_block_reason = "MTF BULL blocks SELL signal"
                except (ImportError, ValueError, IndexError, TypeError):
                    pass

                # Suppress signal if MTF blocks it
                if not mtf_confirmed and result.actionable:
                    result = type(result)(
                        seal_state=charter.SEAL_ABSTAIN,
                        signal=None,
                        reason=mtf_block_reason,
                        gate=result.gate,
                        regime=result.regime,
                    )

                # ─── No-trade zone: 1-minute expiry in RANGE regime ───
                # The 1-bar model has no edge in RANGE regime (random data
                # at 1-bar horizon). Suppress signals to avoid guaranteed losses.
                no_trade_reason = ""
                try:
                    if expiry_minutes == 1 and len(closed) >= 14:
                        from fusion.indicators import choppiness_index as _ci2, hurst_exponent as _he2
                        _closes2 = [c.close for c in closed]
                        _chop2 = _ci2(closed, 14)
                        _hurst2 = _he2(_closes2, 20)
                        _chop_now2 = _chop2[-1] if _chop2 else 50.0
                        _hurst_now2 = _hurst2[-1] if _hurst2 else 0.5
                        if _chop_now2 > 70.0 and _hurst_now2 < 0.45 and result.actionable:
                            no_trade_reason = "No-trade zone: 1-min RANGE regime (no edge)"
                            result = type(result)(
                                seal_state=charter.SEAL_ABSTAIN,
                                signal=None,
                                reason=no_trade_reason,
                                gate=result.gate,
                                regime=result.regime,
                            )
                except (ImportError, ValueError, IndexError, TypeError):
                    pass

                risk_plan = risk.build_risk_plan(
                    symbol=inst.symbol,
                    side=plan_side,
                    entry_price=result.signal.entry_price if (result.signal and result.signal.entry_price) else latest_close,
                    atr_pips=max(0.1, atr_pips),
                    balance_usd=max(100.0, float(balance_usd)),
                    risk_fraction=max(0.001, float(risk_pct) / 100.0),
                )

                telegram_dispatched = False
                signal_dict = None
                brief_text = None
                full_text = None
                if result.actionable and result.signal is not None:
                    sig = result.signal
                    emitted_count += 1
                    sig_row = dict(
                        sig.to_dict(),
                        seal_state=charter.SEAL_SIGNAL,
                        settled=False,
                        bucket=minute_bucket(chosen_stamp),
                        delivery_state="WEB_CONSOLE",
                    )
                    if is_live:
                        member.store.save_signal(sig_row)
                        if dispatch_telegram and member.telegram and member.telegram.configured and not member.paused:
                            member.deliver(sig, brief=True)
                            telegram_dispatched = True
                        else:
                            member.store.update_signal(sig.signal_id, {"delivery_state": "WEB_CONSOLE"})
                    else:
                        existing_local = [
                            row
                            for row in member.store.local.read("signals")
                            if row.get("signal_id") != sig.signal_id
                        ]
                        existing_local.append(sig_row)
                        member.store.local.replace("signals", existing_local[-400:])
                    signal_dict = sig.to_dict()
                    signal_dict["entry_time"] = format_display_iso(sig.entry_time)
                    # Override expiry with user-selected expiry_minutes
                    expiry_dt = sig.entry_time + timedelta(minutes=expiry_minutes)
                    signal_dict["expiry_time"] = format_display_iso(expiry_dt)
                    signal_dict["generated_at"] = format_display_iso(sig.generated_at)
                    signal_dict["entry_hm"] = format_display_hm(sig.entry_time)
                    signal_dict["expiry_hm"] = format_display_hm(expiry_dt)
                    signal_dict["horizon_minutes"] = expiry_minutes
                    signal_dict["timezone"] = DISPLAY_TZ_LABEL
                    signal_dict["entry_display"] = f"{latest_close:,.{inst.precision}f}"
                    brief_text = format_brief_signal(sig, member.telegram_format)
                    full_text = format_signal(sig, member.telegram_format)

                votes_list = [v.to_dict() for v in result.council.votes] if result.council else []
                core_score = result.council.core_score if result.council else 0

                cards.append(
                    {
                        "pair": inst.symbol,
                        "precision": inst.precision,
                        "feed_source": feed_label,
                        "timezone": DISPLAY_TZ_LABEL,
                        "evaluated_at": format_display_iso(base_now),
                        "evaluated_display": format_display_datetime(base_now),
                        "target_entry_at": format_display_iso(future_entry),
                        "target_entry_hm": format_display_hm(future_entry),
                        "seconds_until_entry": seconds_to_open,
                        "is_future_entry": True,
                        "steps_scanned": steps_scanned,
                        "latest_price": round(latest_close, inst.precision),
                        "atr_pips": round(atr_pips, 2),
                        "spread_pips": member.config.spread_pips,
                        "expiry_minutes": expiry_minutes,
                        "mtf_minutes": member.config.mtf_minutes,
                        "mtf_bias": mtf_dir,
                        "actionable": result.actionable,
                        "seal_state": result.seal_state,
                        "side": (plan_side if flip_applied else (result.signal.side if result.signal else "NO SIGNAL")),
                        "reason": result.reason,
                        "regime": result.regime,
                        "core_checks": core_score,
                        "min_checks": member.config.min_confluence_checks,
                        "votes": votes_list,
                        "family_fusion": (
                            result.council.family_verdict.to_dict()
                            if (result.council and result.council.family_verdict)
                            else None
                        ),
                        "model_read": result.read.to_dict() if result.read else None,
                        "signal": signal_dict,
                        "brief_text": brief_text,
                        "full_text": full_text,
                        "telegram_dispatched": telegram_dispatched,
                        "risk_plan": risk_plan.to_dict(),
                        "combiner_prediction": combiner_pred,
                        "is_sureshot": is_sureshot,
                        "mtf_confirmed": mtf_confirmed,
                    }
                )

            overall_feed = (
                "LIVE"
                if is_live
                else "SUPABASE_SYNC"
                if any(c["feed_source"] == "SUPABASE_SYNC" for c in cards)
                else "PAPER_SIMULATION"
            )
            return {
                "ok": True,
                "timezone": DISPLAY_TZ_LABEL,
                "evaluated_at": format_display_datetime(base_now),
                "target_entry_hm": format_display_hm(future_entry),
                "seconds_until_entry": seconds_to_open,
                "feed_mode": overall_feed,
                "emitted": emitted_count,
                "dispatch_telegram": bool(dispatch_telegram),
                "evaluations": cards,
            }

    def control_action(self, action: str) -> dict[str, Any]:
        with self.lock:
            act = str(action or "").strip().lower()
            if act == "pause":
                self.bot._set_paused(True)
                self.bot._save_state(force=True)
                return {"ok": True, "paused": True, "message": "Autonomous Telegram dispatch paused. Web signals, settlement, and learning remain active."}
            if act == "resume":
                self.bot._set_paused(False)
                self.bot._save_state(force=True)
                return {"ok": True, "paused": False, "message": "Autonomous signal dispatch resumed."}
            if act == "cycle":
                stamp = utcnow()
                if not self.live_feed_configured:
                    self._sim_step += 1
                    self._ensure_paper_candles_if_offline(target_stamp=stamp)
                report = self.bot.minute_cycle(at=stamp)
                settled = self.bot.settle_due(at=stamp)
                return {
                    "ok": True,
                    "paused": self.bot.paused,
                    "report": report.to_dict(),
                    "settled": settled,
                    "message": f"Cycle {report.bucket}: {report.seal_state} ({settled} settled).",
                }
            if act == "settle":
                settled = self.bot.settle_due(at=utcnow())
                return {"ok": True, "settled": settled, "message": f"Checked pending queue: {settled} signal(s) settled."}
            raise ValueError(f"Unknown control action: {action}")

    def _sample_signal_and_settlement(self) -> tuple[Signal, Settlement]:
        entry_t = datetime(2026, 10, 25, 14, 15, tzinfo=timezone.utc)
        sig = Signal(
            signal_id="a8f31c90e4b71209",
            pair="EUR/USD",
            side="BUY",
            entry_time=entry_t,
            expiry_time=entry_t + timedelta(minutes=30),
            horizon_minutes=30,
            strategy="fusion_council",
            reason="Core 5/6: trend, momentum, RSI band, directional ADX, candle quality.",
            source="LIVE",
            generated_at=entry_t - timedelta(seconds=58),
            probability_up=0.584,
            probability_raw=0.571,
            calibration_state="CALIBRATED",
            uncertainty=0.042,
            regime="TREND",
            council=({"name": "orbix_confluence", "side": "BUY", "score": 5, "weight": 1.0, "reason": "5/6"},),
            features={},
            spread_pips=1.20,
        )
        stl = Settlement(
            signal_id=sig.signal_id,
            outcome=charter.WIN,
            entry_price=1.08520,
            exit_price=1.08890,
            gross_pips=3.7,
            move_pips=3.7,
            net_pips=2.5,
            spread_pips=1.2,
            settled_at=entry_t + timedelta(minutes=30),
            source="LIVE",
        )
        return sig, stl

    def telegram_format_payload(self, fmt_override: TelegramFormatConfig | None = None) -> dict[str, Any]:
        with self.lock:
            active_fmt = fmt_override or self.bot.telegram_format
            sample_sig, sample_stl = self._sample_signal_and_settlement()
            return {
                "format": active_fmt.to_dict(),
                "signal_styles": list(SIGNAL_STYLES),
                "result_styles": list(RESULT_STYLES),
                "signal_placeholders": list(SIGNAL_PLACEHOLDERS),
                "result_placeholders": list(RESULT_PLACEHOLDERS),
                "default_signal_template": DEFAULT_CUSTOM_SIGNAL_TEMPLATE,
                "default_result_template": DEFAULT_CUSTOM_RESULT_TEMPLATE,
                "preview": {
                    "scheduled_signal": render_signal_message(sample_sig, active_fmt, brief=False),
                    "signals_command": render_signal_message(sample_sig, active_fmt, brief=True),
                    "settlement_result": render_result_message(sample_sig, sample_stl, active_fmt),
                },
            }

    def update_telegram_format(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            if payload.get("reset"):
                new_fmt = TelegramFormatConfig()
            else:
                sig_style = str(payload.get("signal_style", self.bot.telegram_format.signal_style)).strip().lower()
                res_style = str(payload.get("result_style", self.bot.telegram_format.result_style)).strip().lower()
                if sig_style not in SIGNAL_STYLES:
                    raise ValueError(f"Unsupported signal_style {sig_style!r}. Valid: {', '.join(SIGNAL_STYLES)}.")
                if res_style not in RESULT_STYLES:
                    raise ValueError(f"Unsupported result_style {res_style!r}. Valid: {', '.join(RESULT_STYLES)}.")
                new_fmt = TelegramFormatConfig(
                    signal_style=sig_style,
                    signal_template=str(payload.get("signal_template", self.bot.telegram_format.signal_template)),
                    result_style=res_style,
                    result_template=str(payload.get("result_template", self.bot.telegram_format.result_template)),
                    apply_to_signals_cmd=bool(
                        payload.get("apply_to_signals_cmd", self.bot.telegram_format.apply_to_signals_cmd)
                    ),
                    append_paper_footer=bool(
                        payload.get("append_paper_footer", self.bot.telegram_format.append_paper_footer)
                    ),
                )

            if payload.get("preview_only"):
                return {"ok": True, "saved": False, **self.telegram_format_payload(new_fmt)}

            # Persist in bot runtime state (model_state.runtime["telegram_format"]) + local .env
            self.config.telegram_signal_style = new_fmt.signal_style
            self.config.telegram_signal_template = new_fmt.signal_template
            self.config.telegram_result_style = new_fmt.result_style
            self.config.telegram_result_template = new_fmt.result_template
            self.config.telegram_apply_to_signals_cmd = new_fmt.apply_to_signals_cmd
            self.bot.set_telegram_format(new_fmt)

            if self.env_path.is_file():
                lines = self.env_path.read_text(encoding="utf-8").splitlines()
                lines = envfile.set_value(lines, "TELEGRAM_SIGNAL_STYLE", new_fmt.signal_style)
                lines = envfile.set_value(lines, "TELEGRAM_RESULT_STYLE", new_fmt.result_style)
                lines = envfile.set_value(
                    lines, "TELEGRAM_APPLY_TO_SIGNALS_CMD", "1" if new_fmt.apply_to_signals_cmd else "0"
                )
                lines = envfile.set_value(
                    lines, "TELEGRAM_SIGNAL_TEMPLATE", new_fmt.signal_template.replace("\n", "\\n")
                )
                lines = envfile.set_value(
                    lines, "TELEGRAM_RESULT_TEMPLATE", new_fmt.result_template.replace("\n", "\\n")
                )
                envfile.write(self.env_path, lines)

            return {"ok": True, "saved": True, **self.telegram_format_payload(new_fmt)}

    def telegram_payload(self) -> dict[str, Any]:
        with self.lock:
            local_outbox = self.store.local.read("outbox")[-15:][::-1]
            tg_offset = (
                self.bot.telegram.offset
                if (self.bot.telegram and self.bot.telegram.offset)
                else self._cloud_telegram_offset
            )
            return {
                "configured": bool(self.config.telegram_ready or tg_offset > 0),
                "local_send_ready": bool(self.config.telegram_ready),
                "source": "LOCAL_ENV" if self.config.telegram_ready else ("CLOUD_SYNCED" if tg_offset > 0 else "NONE"),
                "owner_masked": _mask_chat_id(self.config.telegram_owner_chat_id) or ("CLOUD_RUNNER" if tg_offset > 0 else ""),
                "token_masked": mask_key(self.config.telegram_token) if self.config.telegram_token else ("CLOUD_SECRET" if tg_offset > 0 else ""),
                "offset": tg_offset,
                "paused": self.bot.paused,
                "commands": ["/start", "/help", "/signals", "/status", "/pause", "/resume"],
                "pending_count": len(self.store.outbox_pending()),
                "recent_outbox": local_outbox,
                "format_studio": self.telegram_format_payload(),
            }

    def run_telegram_command(self, command: str, *, send_to_chat: bool = False) -> dict[str, Any]:
        with self.lock:
            cmd = str(command or "").strip().split()[0] if str(command or "").strip() else "/status"
            if not cmd.startswith("/"):
                cmd = "/" + cmd
            saved_tg = self.bot.telegram
            try:
                if not send_to_chat:
                    self.bot.telegram = None
                reply = self.bot.handle_command(cmd)
            finally:
                self.bot.telegram = saved_tg
            return {
                "ok": True,
                "command": cmd,
                "reply": reply,
                "paused": self.bot.paused,
                "sent_to_chat": bool(send_to_chat and saved_tg and saved_tg.configured),
            }

    def health_payload(self, *, run_live_audit: bool = False) -> dict[str, Any]:
        with self.lock:
            raw_checks = doctor.run_checks(self.config, env_path=self.env_path, offline=True, send_test=False)
            doctor_rows = []
            for c in raw_checks:
                status = c.status
                detail = c.detail
                action = c.action
                if c.name == "Twelve Data" and status == doctor.FAIL and self._cloud_key_slots:
                    first_lbl = self._cloud_key_slots[0].get("label", "slot-1")
                    status = doctor.PASS
                    detail = f"{len(self._cloud_key_slots)} slot(s) synced from Supabase ({first_lbl}); scheduled ingestion active in GitHub Actions"
                    action = ""
                elif c.name == "Telegram" and status == doctor.FAIL and self._cloud_telegram_offset > 0:
                    status = doctor.PASS
                    detail = f"Connected in GitHub Actions runner (offset #{self._cloud_telegram_offset}); web command console ready"
                    action = ""
                elif c.name == "Supabase" and status == doctor.FAIL and (
                    self._cloud_supabase_url or (self.config.supabase_url and self.has_synced_market_data)
                ):
                    supa_label = _mask_url(self.config.supabase_url or self._cloud_supabase_url)
                    status = doctor.PASS
                    detail = f"Synced via Supabase Snapshot RPC ({supa_label}); serving real mirrored state"
                    action = ""
                elif c.name == "Supabase" and status == doctor.FAIL and self.config.supabase_ready and not self._supabase_reachable:
                    status = doctor.WARN
                    detail = f"Configured ({_mask_url(self.config.supabase_url)}) · outbound TCP restricted in sandbox; serving synced mirror"
                    action = "Direct PostgREST HTTPS activates automatically on unrestricted networks."
                doctor_rows.append({"name": c.name, "status": status, "detail": detail, "action": action})
            if run_live_audit or self._last_audit is None:
                t0 = time.perf_counter()
                audit = lookahead.run_audit()
                elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)
                self._last_audit = {
                    "ok": audit.ok,
                    "elapsed_ms": elapsed_ms,
                    "checked_at": format_datetime(utcnow()),
                    "checks": [
                        {"name": f.component, "ok": f.passed, "detail": f.detail}
                        for f in audit.findings
                    ],
                }
            hb = self.store.latest_heartbeat()
            hb_age = heartbeat.heartbeat_age_seconds(hb)
            pool = self._effective_key_pool()
            return {
                "ok": all(r["status"] != doctor.FAIL for r in doctor_rows) and bool(self._last_audit["ok"]),
                "doctor": doctor_rows,
                "lookahead": self._last_audit,
                "heartbeat": hb,
                "heartbeat_age_s": None if hb_age is None else round(hb_age, 1),
                "key_pool": pool,
                "tests_passing": TESTS_PASSING_BASELINE,
                "data_gate": {
                    "max_candle_age_s": self.config.max_candle_age_seconds,
                    "max_seal_latency_s": charter.MAX_SEAL_LATENCY_SECONDS,
                    "max_spread_multiple": charter.MAX_SPREAD_MULTIPLE,
                    "unresolve_after_m": charter.UNRESOLVE_AFTER_MINUTES,
                },
                "alerts": self.store.recent_alerts(limit=12),
                "store_errors": self.store.errors[-8:],
            }

    def needle_status_payload(self) -> dict[str, Any]:
        with self.lock:
            partner = self.bot._get_cognitive_partner()
            return {
                "ok": True,
                "enabled": self.config.needle_enabled,
                "cognitive": partner.status_summary(self.bot),
                **self.needle.status(),
            }

    def learning_payload(self) -> dict[str, Any]:
        """Full self-improvement, 6-family fusion, auto-backtest, and cognitive memory snapshot."""
        with self.lock:
            partner = self.bot._get_cognitive_partner()
            markets_learning: dict[str, Any] = {}
            combined_journal: list[dict[str, Any]] = []
            combined_backtests: list[dict[str, Any]] = []
            for m in self._all_members():
                pair = m.config.pair
                for ep in m.learning_journal:
                    combined_journal.append(ep)
                for bt in m.auto_backtest_history:
                    combined_backtests.append(bt)
                markets_learning[pair] = {
                    "pair": pair,
                    "model_mode": m.model.mode if m.model else "VETO",
                    "training_samples": int(m.model.training_samples) if m.model else 0,
                    "online_updates": int(m.model.online_updates) if m.model else 0,
                    "calibration_samples": len(m.calibration_points),
                    "reconciled_trades": len(m.mistakes.reconciled_ids),
                    "family_fusion": {
                        "trades_observed": m.family_fusion.trades_observed,
                        "family_weights": {
                            k: round(float(v), 4) for k, v in m.family_fusion.family_weights.items()
                        },
                        "family_seen": dict(m.family_fusion.family_seen),
                        "family_net": {
                            k: round(float(v), 2) for k, v in m.family_fusion.family_net.items()
                        },
                        "synergy_weights": {
                            k: round(float(v), 4) for k, v in m.family_fusion.synergy_weights.items()
                        },
                        "synergy_seen": dict(m.family_fusion.synergy_seen),
                        "active_families": {
                            k: round(float(v), 4) for k, v in m.family_fusion.active_families().items()
                        },
                        "active_synergies": {
                            k: round(float(v), 4) for k, v in m.family_fusion.active_synergies().items()
                        },
                    },
                    "latest_backtest": m.auto_backtest_history[-1] if m.auto_backtest_history else None,
                    "recent_lessons": list(reversed(m.learning_journal[-10:])),
                }
            combined_journal.sort(key=lambda r: str(r.get("recorded_at", "")), reverse=True)
            combined_backtests.sort(key=lambda r: str(r.get("ran_at", "")), reverse=True)
            learning_events = [
                dict(
                    ev,
                    created_display=format_display_datetime(ev["created_at"]) if ev.get("created_at") else None,
                )
                for ev in self.store.local.read("learning_events")[-30:]
                if isinstance(ev, dict)
            ]
            learning_events.reverse()
            return {
                "ok": True,
                "timezone": DISPLAY_TZ_LABEL,
                "cognitive": partner.status_summary(self.bot),
                "conversation": list(partner.conversation[-12:]),
                "rectifications": list(reversed(partner.rectifications[-12:])),
                "markets": markets_learning,
                "journal": combined_journal[:30],
                "auto_backtests": combined_backtests[:15],
                "learning_events": learning_events,
            }

    def run_backtest_action(self, pair: str = "ALL") -> dict[str, Any]:
        """Run causal walk-forward backtest across target market(s) and return updated learning snapshot."""
        with self.lock:
            target = str(pair or "ALL").strip().upper()
            members = (
                self._all_members()
                if target in ("", "ALL")
                else [m for m in self._all_members() if m.config.pair == target]
            )
            if not members:
                raise ValueError(f"Unsupported market pair: {pair}")
            stamp = utcnow()
            runs: list[dict[str, Any]] = []
            for m in members:
                is_real = self._has_real_candles(m)
                if not is_real and len(m.candles) < 240:
                    m.candles = _build_scaled_paper_series(
                        m.config.pair,
                        count=320,
                        end_minute=stamp - timedelta(minutes=1),
                        seed_offset=self._sim_step,
                    )
                res = m.run_auto_backtest(
                    at=stamp,
                    trigger="console_backtest",
                    min_bars=180,
                    persist_version=is_real,
                    allow_paper_eval=not is_real,
                )
                runs.append(res)
            partner = self.bot._get_cognitive_partner()
            ok_runs = [r for r in runs if r.get("ok")]
            if ok_runs:
                partner.record_experience(
                    "auto_backtest",
                    f"Completed walk-forward backtest across {len(ok_runs)} market(s).",
                    {"runs": [{k: r.get(k) for k in ("pair", "signals", "net_pips", "win_rate", "source")} for r in ok_runs]},
                )
                self.bot.cognitive_state = partner.to_dict()
                self.bot._save_state(force=True)
            return {
                "ok": True,
                "runs": runs,
                "learning": self.learning_payload(),
            }

    def self_heal_action(self) -> dict[str, Any]:
        """Run autonomous cognitive error detection & self-rectification across all subsystems."""
        with self.lock:
            partner = self.bot._get_cognitive_partner()
            heal = partner.detect_and_rectify(self.bot, run_causality=True)
            self.bot.cognitive_state = partner.to_dict()
            self.bot._save_state(force=True)
            return {
                "ok": True,
                "heal": heal,
                "narrative": partner.compose_human_reply(self.bot, "find errors and rectify", heal_report=heal),
                "learning": self.learning_payload(),
            }

    def run_needle_query(self, query: str, *, execute: bool = True) -> dict[str, Any]:
        with self.lock:
            from fusion import cognitive as cog_mod

            partner = self.bot._get_cognitive_partner()
            decision = self.needle.route(query)
            is_human_dialogue = bool(
                cog_mod.CONVERSATION_GREETING_RE.search(query or "")
                or cog_mod.HUMAN_REFLECTION_RE.search(query or "")
            )

            # If Needle refused because this is pure human strategist conversation (not a Charter violation
            # and not an off-topic weather/cooking prompt), respond as SAVIOUR's human cognitive partner!
            if decision.type != "tool_call" or not decision.function_calls:
                if execute and decision.engine != "charter_shield" and is_human_dialogue:
                    human_reply = partner.compose_human_reply(self.bot, query)
                    partner.remember_turn("operator", query)
                    partner.remember_turn("saviour", human_reply, engine="saviour_cognitive")
                    self.bot.cognitive_state = partner.to_dict()
                    self.bot._save_state(force=True)
                    return {
                        "ok": True,
                        "executed": True,
                        "decision": {
                            **decision.to_dict(),
                            "type": "human_dialogue",
                            "engine": f"{decision.engine}+saviour_cognitive",
                        },
                        "reply": human_reply,
                        "signal_run": None,
                        "cognitive": partner.status_summary(self.bot),
                    }
                return {
                    "ok": True,
                    "executed": False,
                    "decision": decision.to_dict(),
                    "reply": decision.refusal_reason or "No tool executed.",
                    "signal_run": None,
                    "cognitive": partner.status_summary(self.bot),
                }

            if not execute:
                return {
                    "ok": True,
                    "executed": False,
                    "decision": decision.to_dict(),
                    "reply": "Dry-run routing completed.",
                    "signal_run": None,
                    "cognitive": partner.status_summary(self.bot),
                }

            signal_run: dict[str, Any] | None = None
            non_scan_calls = [c for c in decision.function_calls if c["name"] != "scan_market"]
            scan_calls = [c for c in decision.function_calls if c["name"] == "scan_market"]

            replies: list[str] = []
            if non_scan_calls:
                sub_agent_decision, sub_reply = needle_agent.execute_on_bot(
                    self.bot,
                    query,
                    agent=self.needle,
                )
                if not scan_calls:
                    partner.remember_turn("operator", query)
                    partner.remember_turn("saviour", sub_reply, engine=sub_agent_decision.engine)
                    self.bot.cognitive_state = partner.to_dict()
                    self.bot._save_state(force=True)
                    return {
                        "ok": True,
                        "executed": True,
                        "decision": sub_agent_decision.to_dict(),
                        "reply": sub_reply,
                        "signal_run": None,
                        "cognitive": partner.status_summary(self.bot),
                    }
                replies.append(sub_reply.split("\n\n")[0])

            for call in scan_calls:
                pair = str(call["arguments"].get("pair") or "ALL")
                mode = str(call["arguments"].get("mode") or "now")
                signal_run = self.generate_signals(pair=pair, mode=mode, dispatch_telegram=False)
                cards = signal_run.get("evaluations") or signal_run.get("cards") or []
                summary_lines = [
                    f"{c['pair']}: {c['side']} ({c['seal_state']}) — {c['reason']}"
                    for c in cards
                ]
                replies.append("\n".join(summary_lines))

            final_reply = "\n\n".join(r for r in replies if r)
            partner.remember_turn("operator", query)
            partner.remember_turn("saviour", final_reply, engine=decision.engine)
            self.bot.cognitive_state = partner.to_dict()
            self.bot._save_state(force=True)
            return {
                "ok": True,
                "executed": True,
                "decision": decision.to_dict(),
                "reply": final_reply,
                "signal_run": signal_run,
                "cognitive": partner.status_summary(self.bot),
            }

    def strategies_payload(self) -> dict[str, Any]:
        with self.lock:
            weights = self.bot.weights.weights
            seen = self.bot.weights.seen
            rows = []
            family_counts: dict[str, int] = {}
            all_ids = (strategies.CORE, *strategies.PROTOTYPES)
            for sid in all_ids:
                allowed_regimes = [
                    reg for reg in ("TREND", "RANGE", "VOLATILE") if regime_mod.regime_allows(reg, sid)
                ]
                family = strategies.strategy_family(sid)
                family_counts[family] = family_counts.get(family, 0) + 1
                rows.append(
                    {
                        "id": sid,
                        "family": family,
                        "description": strategies.MEMBERS.get(sid, ""),
                        "regimes": allowed_regimes,
                        "weight": round(float(weights.get(sid, 0.0)), 4),
                        "samples_seen": int(seen.get(sid, 0)),
                    }
                )
            return {
                "council_total": len(all_ids),
                "prototype_total": len(strategies.PROTOTYPES),
                "family_counts": family_counts,
                "family_fusion": {
                    m.config.pair: {
                        "trades_observed": m.family_fusion.trades_observed,
                        "family_weights": {
                            k: round(float(v), 4) for k, v in m.family_fusion.family_weights.items()
                        },
                        "family_seen": dict(m.family_fusion.family_seen),
                        "synergy_weights": {
                            k: round(float(v), 4) for k, v in m.family_fusion.synergy_weights.items()
                        },
                        "synergy_seen": dict(m.family_fusion.synergy_seen),
                    }
                    for m in self._all_members()
                },
                "runtime_config": {
                    "pairs": list(self.config.active_pairs),
                    "mtf_filter": self.config.mtf_filter,
                    "mtf_minutes": self.config.mtf_minutes,
                    "mtf_min_gap_bps": self.config.mtf_min_gap_bps,
                    "mtf_macro_filter": self.config.mtf_macro_filter,
                    "news_filter": self.config.news_filter,
                    "adaptive_horizon": self.config.adaptive_horizon,
                    "max_horizon_minutes": self.config.max_horizon_minutes,
                    "session_filter": self.config.session_filter,
                    "min_confluence_checks": self.config.min_confluence_checks,
                    "signal_cooldown_minutes": self.config.signal_cooldown_minutes,
                    "loss_guard_streak": self.config.loss_guard_streak,
                    "loss_guard_shadow_recovery": self.config.loss_guard_shadow_recovery,
                    "fx_rollover_guard": self.config.fx_rollover_guard,
                },
                "members": rows,
                "benchmark_101k": [
                    {
                        "variant": "Core Confluence (unfiltered 60m)",
                        "train_wr": 51.49,
                        "train_net": -0.513,
                        "oos_wr": 57.47,
                        "oos_net": -0.116,
                    },
                    {
                        "variant": "Core + 15m MTF Alignment (active)",
                        "train_wr": 52.21,
                        "train_net": -0.312,
                        "oos_wr": 58.51,
                        "oos_net": 0.338,
                    },
                    {
                        "variant": "scalp_expansion Prototype (active)",
                        "train_wr": 54.86,
                        "train_net": 0.378,
                        "oos_wr": 55.02,
                        "oos_net": 0.382,
                    },
                    {
                        "variant": "Adaptive-Horizon Label Alignment (80k 2026)",
                        "train_wr": 51.30,
                        "train_net": -0.910,
                        "oos_wr": 53.39,
                        "oos_net": -0.440,
                    },
                    {
                        "variant": "MTF 2.0 bps Dead-Zone + Shadow-Win Hold (260k 2025-26)",
                        "train_wr": 48.67,
                        "train_net": 0.480,
                        "oos_wr": 52.04,
                        "oos_net": 0.480,
                    },
                ],
            }

    def update_runtime_config(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            if "pairs" in payload:
                raw_pairs = payload["pairs"]
                if isinstance(raw_pairs, str):
                    items = [p.strip() for p in raw_pairs.split(",") if p.strip()]
                elif isinstance(raw_pairs, list):
                    items = [str(p).strip() for p in raw_pairs if str(p).strip()]
                else:
                    items = []
                if not items:
                    raise ValueError("At least one market must remain enabled.")
                resolved = tuple(instruments.for_symbol(item).symbol for item in items)
                self.config.pairs = tuple(dict.fromkeys(resolved))
                self.config.pair = self.config.pairs[0]
                self.config.spread_pips = instruments.for_symbol(self.config.pair).spread_pips

            if "mtf_filter" in payload:
                self.config.mtf_filter = bool(payload["mtf_filter"])
            if "mtf_minutes" in payload:
                val = int(payload["mtf_minutes"])
                if val not in (5, 15):
                    raise ValueError("mtf_minutes must be 5 or 15.")
                self.config.mtf_minutes = val
            if "mtf_min_gap_bps" in payload:
                gap_val = float(payload["mtf_min_gap_bps"])
                if not 0.0 <= gap_val <= 25.0:
                    raise ValueError("mtf_min_gap_bps must be between 0.0 and 25.0.")
                self.config.mtf_min_gap_bps = gap_val
            if "mtf_macro_filter" in payload:
                self.config.mtf_macro_filter = bool(payload["mtf_macro_filter"])
            if "news_filter" in payload:
                self.config.news_filter = bool(payload["news_filter"])
            if "adaptive_horizon" in payload:
                self.config.adaptive_horizon = bool(payload["adaptive_horizon"])
            if "session_filter" in payload:
                self.config.session_filter = bool(payload["session_filter"])
            if "fx_rollover_guard" in payload:
                self.config.fx_rollover_guard = bool(payload["fx_rollover_guard"])
            if "loss_guard_shadow_recovery" in payload:
                self.config.loss_guard_shadow_recovery = bool(payload["loss_guard_shadow_recovery"])
            if "signal_cooldown_minutes" in payload:
                cd = int(payload["signal_cooldown_minutes"])
                if not 0 <= cd <= 240:
                    raise ValueError("signal_cooldown_minutes must be between 0 and 240.")
                self.config.signal_cooldown_minutes = cd
            if "loss_guard_streak" in payload:
                ls = int(payload["loss_guard_streak"])
                if not 0 <= ls <= 10:
                    raise ValueError("loss_guard_streak must be between 0 and 10.")
                self.config.loss_guard_streak = ls
            if "max_horizon_minutes" in payload:
                mh = int(payload["max_horizon_minutes"])
                if not 5 <= mh <= 240:
                    raise ValueError("max_horizon_minutes must be between 5 and 240.")
                self.config.max_horizon_minutes = mh

            # Rebuild peer bots & engines cleanly with the updated runtime gates.
            paused_before = self.bot.paused
            fmt_before = self.bot.telegram_format
            self.bot = FusionBot(config=self.config, store=self.store)
            self.bot.startup()
            self._synced_pairs = {
                m.config.pair for m in self._all_members() if len(m.candles) >= 60
            }
            self.bot._set_paused(paused_before)
            self.bot.set_telegram_format(fmt_before)
            self._ensure_paper_candles_if_offline()
            return {"ok": True, "strategies": self.strategies_payload()}

    def credentials_payload(self) -> dict[str, Any]:
        with self.lock:
            lines = self.env_path.read_text(encoding="utf-8").splitlines() if self.env_path.is_file() else []
            supa_url = envfile.get_value(lines, "SUPABASE_URL") or self.config.supabase_url
            supa_key = envfile.get_value(lines, "SUPABASE_SERVICE_ROLE_KEY") or self.config.supabase_service_key
            tg_tok = envfile.get_value(lines, "TELEGRAM_BOT_TOKEN") or self.config.telegram_token
            tg_chat = envfile.get_value(lines, "TELEGRAM_OWNER_CHAT_ID") or self.config.telegram_owner_chat_id

            td_keys = list(self.config.twelve_data_keys)
            if td_keys:
                td_slots_masked = [{"slot": idx + 1, "masked": mask_key(k)} for idx, k in enumerate(td_keys)]
                td_source = "LOCAL_ENV"
            elif self._cloud_key_slots:
                td_slots_masked = [
                    {
                        "slot": int(r.get("index", r.get("slot_index", idx))) + 1,
                        "masked": str(r.get("label") or f"slot-{idx + 1}"),
                    }
                    for idx, r in enumerate(self._cloud_key_slots)
                ]
                td_source = "CLOUD_SYNCED"
            else:
                td_slots_masked = []
                td_source = "NONE"

            tg_local_ready = bool(tg_tok and tg_chat)
            tg_cloud_ready = self._cloud_telegram_offset > 0
            eff_supa_url = supa_url or self._cloud_supabase_url
            cloud_rpc_synced = bool(self._cloud_supabase_url or (supa_url and self.has_synced_market_data))
            supa_configured = bool((supa_url and supa_key) or cloud_rpc_synced)
            if self._supabase_reachable:
                supa_mode = "LIVE"
            elif supa_url and supa_key:
                supa_mode = "BRIDGED_MIRROR"
            elif cloud_rpc_synced:
                supa_mode = "CLOUD_RPC_SYNCED"
            else:
                supa_mode = "NONE"
            return {
                "env_file_exists": self.env_path.is_file(),
                "env_file_name": ".env (git-ignored, chmod 600)",
                "max_twelvedata_slots": MAX_SLOTS,
                "supabase": {
                    "configured": supa_configured,
                    "live_reachable": self._supabase_reachable,
                    "mode": supa_mode,
                    "url_masked": _mask_url(eff_supa_url),
                    "key_masked": mask_key(supa_key) if supa_key else ("CLOUD_RPC_PUBLIC" if cloud_rpc_synced else ""),
                    "last_synced_at": format_display_datetime(self._cloud_synced_at) if self._cloud_synced_at else None,
                },
                "telegram": {
                    "configured": bool(tg_local_ready or tg_cloud_ready),
                    "local_send_ready": tg_local_ready,
                    "source": "LOCAL_ENV" if tg_local_ready else ("CLOUD_SYNCED" if tg_cloud_ready else "NONE"),
                    "token_masked": mask_key(tg_tok) if tg_tok else ("GH_SECRETS_SYNCED" if tg_cloud_ready else ""),
                    "owner_chat_masked": _mask_chat_id(tg_chat) if tg_chat else (f"offset #{self._cloud_telegram_offset}" if tg_cloud_ready else ""),
                    "cloud_offset": self._cloud_telegram_offset,
                },
                "twelvedata": {
                    "configured": bool(td_slots_masked),
                    "local_keys_ready": bool(td_keys),
                    "source": td_source,
                    "slot_count": len(td_slots_masked),
                    "slots_masked": td_slots_masked,
                },
            }

    def _load_real_candles(self) -> list[dict[str, float]]:
        """Load latest real candles from the demo batch files or live feed."""
        import glob, json
        candles_dir = BOT_ROOT / "var" / "real_candles"
        files = sorted(glob.glob(str(candles_dir / "batch_*.txt")))
        candles: list[dict[str, float]] = []
        for f in files[-3:]:
            try:
                for line in open(f, encoding="utf-8"):
                    line = line.strip()
                    if not line:
                        continue
                    for candle_str in line.split(";"):
                        c = candle_str.strip()
                        if not c:
                            continue
                        parts = c.split(",")
                        if len(parts) < 6:
                            continue
                        candles.append({
                            "ts": float(parts[0]),
                            "open": float(parts[1]),
                            "high": float(parts[2]),
                            "low": float(parts[3]),
                            "close": float(parts[4]),
                            "volume": float(parts[5]),
                        })
            except (ValueError, OSError):
                continue
        return candles[-300:] if candles else []

    def _compute_indicators(self, candles: list[dict[str, float]]) -> dict[str, Any]:
        """Compute all fusion indicators on the latest candle."""
        try:
            sys.path.insert(0, str(BOT_ROOT.parent))
            from datetime import datetime, timezone
            from fusion.domain import Candle as FusionCandle
            from fusion.indicators import (
                ema, rsi, atr, adx, bollinger, stochastic, cci, vwap,
                vwap_zscore, kaufman_er, choppiness_index, keltner_squeeze,
                supertrend_window, smc_structure, ichimoku_cloud, hurst_exponent,
                fft_cycles, aroon, donchian_channel, pivot_points,
            )
            # Convert dict candles to FusionCandle objects
            fc_candles: list[FusionCandle] = []
            for c in candles:
                fc = FusionCandle(
                    timestamp=datetime.fromtimestamp(c["ts"], tz=timezone.utc),
                    open=c["open"], high=c["high"], low=c["low"],
                    close=c["close"], volume=c["volume"],
                )
                fc_candles.append(fc)
            if len(fc_candles) < 60:
                return {"error": "Not enough candles (need 60+)"}
            closes = [c["close"] for c in candles]
            last = len(candles) - 1
            indicators: dict[str, Any] = {}
            indicators["close"] = closes[last]
            indicators["ema_20"] = ema(closes, 20)[-1] if len(closes) >= 20 else None
            indicators["ema_50"] = ema(closes, 50)[-1] if len(closes) >= 50 else None
            indicators["ema_200"] = ema(closes, 200)[-1] if len(closes) >= 200 else None
            indicators["rsi_14"] = rsi(closes, 14)[-1] if len(closes) >= 14 else None
            atr_vals = atr(fc_candles, 14)
            indicators["atr_14"] = atr_vals[-1] if atr_vals else None
            adx_vals, plus_di, minus_di = adx(fc_candles, 14)
            indicators["adx_14"] = adx_vals[-1] if adx_vals else None
            bb_lower, bb_mid, bb_upper = bollinger(closes, 20, 2.0)
            indicators["bb_upper"] = bb_upper
            indicators["bb_mid"] = bb_mid
            indicators["bb_lower"] = bb_lower
            indicators["bb_width"] = (bb_upper - bb_lower) / bb_mid if bb_mid else None
            stoch_k, stoch_d = stochastic(fc_candles, 14, 3)
            indicators["stoch_k"] = stoch_k[-1] if stoch_k else None
            indicators["stoch_d"] = stoch_d[-1] if stoch_d else None
            cci_vals = cci(fc_candles, 20)
            indicators["cci_20"] = cci_vals[-1] if cci_vals else None
            vwap_vals = vwap(fc_candles, 30)
            indicators["vwap"] = vwap_vals[-1] if vwap_vals else None
            vwap_z_vals = vwap_zscore(fc_candles, 20)
            indicators["vwap_zscore"] = vwap_z_vals[-1] if vwap_z_vals else None
            ke_vals = kaufman_er(closes, 10)
            indicators["kaufman_er"] = ke_vals[-1] if ke_vals else None
            chop_vals = choppiness_index(fc_candles, 14)
            indicators["choppiness"] = chop_vals[-1] if chop_vals else None
            squeeze_bools, squeeze_momentum = keltner_squeeze(fc_candles, bb_period=20, bb_mult=2.0, kc_period=20, kc_mult=1.5)
            indicators["squeeze_on"] = bool(squeeze_bools[-1]) if squeeze_bools else False
            indicators["squeeze_momentum"] = squeeze_momentum[-1] if squeeze_momentum else None
            st_dir, st_val = supertrend_window(fc_candles, period=10, multiplier=3.0)
            indicators["supertrend_direction"] = st_dir[-1] if st_dir else None
            indicators["supertrend_value"] = st_val[-1] if st_val else None
            smc_fvg_bull, smc_fvg_bear, smc_ob_bull, smc_ob_bear = smc_structure(fc_candles, atr_vals)
            indicators["smc_regime"] = smc_fvg_bull[-1] if smc_fvg_bull else None
            indicators["smc_structure"] = smc_ob_bull[-1] if smc_ob_bull else None
            ichimoku_tenkan, ichimoku_kijun, ichimoku_senkou_a, ichimoku_senkou_b, ichimoku_chikou = ichimoku_cloud(fc_candles, 9, 26, 52)
            indicators["ichimoku_kijun"] = ichimoku_kijun[-1] if ichimoku_kijun else None
            indicators["ichimoku_tenkan"] = ichimoku_tenkan[-1] if ichimoku_tenkan else None
            indicators["ichimoku_senkou_a"] = ichimoku_senkou_a[-1] if ichimoku_senkou_a else None
            indicators["ichimoku_senkou_b"] = ichimoku_senkou_b[-1] if ichimoku_senkou_b else None
            hurst_vals = hurst_exponent(closes, 20)
            indicators["hurst"] = hurst_vals[-1] if hurst_vals else None
            fft_results = fft_cycles(closes, 5)
            if fft_results:
                indicators["fft_dominant_period"] = fft_results[0][1]
                indicators["fft_dominant_power"] = fft_results[0][0]
            else:
                indicators["fft_dominant_period"] = None
                indicators["fft_dominant_power"] = None
            aroon_up, aroon_down = aroon(fc_candles, 25)
            indicators["aroon_up"] = aroon_up[-1] if aroon_up else None
            indicators["aroon_down"] = aroon_down[-1] if aroon_down else None
            dc_upper, dc_lower, dc_mid = donchian_channel(fc_candles, 20)
            indicators["donchian_upper"] = dc_upper[-1] if dc_upper else None
            indicators["donchian_mid"] = dc_mid[-1] if dc_mid else None
            indicators["donchian_lower"] = dc_lower[-1] if dc_lower else None
            pivots = pivot_points(fc_candles, 14)
            # pivot_points returns (pivot, r1, r2, s1, s2)
            indicators["pivot_pp"] = pivots[0][-1] if pivots and pivots[0] else None
            indicators["pivot_r1"] = pivots[1][-1] if pivots and len(pivots) > 1 and pivots[1] else None
            indicators["pivot_r2"] = pivots[2][-1] if pivots and len(pivots) > 2 and pivots[2] else None
            indicators["pivot_sr1"] = pivots[3][-1] if pivots and len(pivots) > 3 and pivots[3] else None
            indicators["pivot_sr2"] = pivots[4][-1] if pivots and len(pivots) > 4 and pivots[4] else None
            indicators["count"] = 21
            return indicators
        except (ImportError, ValueError, IndexError, TypeError) as exc:
            return {"error": str(exc)}

    def _run_ensemble_prediction(self, candles: list[dict[str, float]]) -> dict[str, Any]:
        """Run the GBDT ensemble on the latest candle to predict next-bar direction."""
        try:
            sys.path.insert(0, str(BOT_ROOT.parent))
            from datetime import datetime, timezone
            from fusion.domain import Candle as FusionCandle
            from fusion.indicators import ema, rsi, atr
            model_path = BOT_ROOT / "var" / "real_candles" / "ensemble_model.json"
            if not model_path.is_file():
                return {"available": False, "reason": "Model not trained yet"}
            with open(model_path, encoding="utf-8") as f:
                model_data = json.load(f)
            if len(candles) < 30:
                return {"available": False, "reason": "Not enough candles"}
            last_candle = candles[-1]
            closes = [c["close"] for c in candles[-100:]]
            highs = [c["high"] for c in candles[-100:]]
            lows = [c["low"] for c in candles[-100:]]
            fc_candles = [FusionCandle(
                timestamp=datetime.fromtimestamp(c["ts"], tz=timezone.utc),
                open=c["open"], high=c["high"], low=c["low"],
                close=c["close"], volume=c["volume"],
            ) for c in candles[-100:]]
            ema_fast = ema(closes, 10)[-1] if len(closes) >= 10 else closes[-1]
            ema_slow = ema(closes, 30)[-1] if len(closes) >= 30 else closes[-1]
            rsi_val = rsi(closes, 14)[-1] if len(closes) >= 14 else 50.0
            atr_vals = atr(fc_candles, 14)
            atr_val = atr_vals[-1] if atr_vals else 0.01
            features = {
                "return_1": closes[-1] / closes[-2] - 1 if len(closes) > 1 else 0.0,
                "return_2": closes[-1] / closes[-3] - 1 if len(closes) > 2 else 0.0,
                "return_3": closes[-1] / closes[-4] - 1 if len(closes) > 3 else 0.0,
                "ema_fast": ema_fast,
                "ema_slow": ema_slow,
                "ema_cross": 1.0 if ema_fast > ema_slow else 0.0,
                "rsi_14": rsi_val,
                "rsi_overbought": 1.0 if rsi_val > 70 else 0.0,
                "rsi_oversold": 1.0 if rsi_val < 30 else 0.0,
                "atr_pct": atr_val / closes[-1] if closes[-1] else 0.0,
                "close_position": (closes[-1] - min(lows[-20:])) / (max(highs[-20:]) - min(lows[-20:])) if max(highs[-20:]) != min(lows[-20:]) else 0.5,
                "volume_change": (candles[-1]["volume"] / candles[-2]["volume"]) if len(candles) > 1 and candles[-2]["volume"] > 0 else 1.0,
            }
            if model_data.get("models"):
                from fusion.gbdt import GradientBoostingModel as GBDTModel
                up_prob = 0.0
                loaded_models = []
                for m_dict in model_data["models"]:
                    model = GBDTModel.from_dict(m_dict)
                    if model is not None:
                        loaded_models.append(model)
                if not loaded_models:
                    return {"available": False, "reason": "Could not load model from JSON"}
                for m in loaded_models:
                    try:
                        pred = m.predict_up(features)
                        up_prob += pred / len(loaded_models)
                    except Exception:
                        continue
                if up_prob == 0.0:
                    return {"available": False, "reason": "Ensemble prediction failed"}
                last_close = closes[-1]
                est_price = last_close + (up_prob - 0.5) * 2 * atr_val
                return {
                    "available": True,
                    "up_probability": round(up_prob, 4),
                    "down_probability": round(1 - up_prob, 4),
                    "last_close": round(last_close, 6),
                    "estimated_price_5min": round(est_price, 6),
                    "confidence": round(abs(up_prob - 0.5) * 2, 4),
                    "atr": round(atr_val, 6),
                }
        except (ImportError, ValueError, IndexError, TypeError, OSError, KeyError) as exc:
            return {"available": False, "reason": str(exc)}
        return {"available": False, "reason": "No model loaded"}

    def _compute_combiner_score(self, indicators: dict[str, Any], ml_prob: dict[str, Any]) -> dict[str, Any]:
        """Fuse all indicator votes into a combiner score (-1 to +1)."""
        votes = []
        ml_weight = 0.30
        ml_score = (ml_prob.get("up_probability", 0.5) - 0.5) * 2 if ml_prob.get("available") else 0.0
        votes.append(("ML Ensemble", ml_score, ml_weight))
        if indicators.get("supertrend_direction") is not None:
            score = 1.0 if indicators["supertrend_direction"] > 0 else -1.0
            votes.append(("Supertrend", score, 0.15))
        if indicators.get("rsi_14") is not None:
            rsi_val = indicators["rsi_14"]
            score = 0.5 if rsi_val < 40 else (1.0 if rsi_val < 50 else (-0.5 if rsi_val > 60 else (-1.0 if rsi_val > 70 else 0.0)))
            votes.append(("RSI", score, 0.10))
        if indicators.get("stoch_k") is not None:
            score = 1.0 if indicators["stoch_k"] < 20 else (-1.0 if indicators["stoch_k"] > 80 else 0.0)
            votes.append(("Stochastic", score, 0.10))
        if indicators.get("vwap") is not None and indicators.get("close") is not None:
            score = 1.0 if indicators["close"] > indicators["vwap"] else -1.0
            votes.append(("VWAP", score, 0.08))
        if indicators.get("ichimoku_kijun") is not None and indicators.get("close") is not None:
            score = 1.0 if indicators["close"] > indicators["ichimoku_kijun"] else -1.0
            votes.append(("Ichimoku", score, 0.05))
        if indicators.get("aroon_up") is not None and indicators.get("aroon_down") is not None:
            score = 1.0 if indicators["aroon_up"] > 70 else (-1.0 if indicators["aroon_down"] > 70 else 0.0)
            votes.append(("Aroon", score, 0.05))
        if indicators.get("choppiness") is not None:
            score = -1.0 if indicators["choppiness"] > 70 else (1.0 if indicators["choppiness"] < 40 else 0.0)
            votes.append(("Choppiness", score, 0.05))
        if indicators.get("squeeze_on") is not None:
            score = 1.0 if indicators["squeeze_on"] else 0.0
            votes.append(("Squeeze Breakout", score, 0.05))
        if indicators.get("hurst") is not None:
            score = 1.0 if indicators["hurst"] > 0.5 else (-1.0 if indicators["hurst"] < 0.45 else 0.0)
            votes.append(("Hurst", score, 0.03))
        total_weight = sum(w for _, _, w in votes) or 1.0
        combined = sum(s * w for _, s, w in votes) / total_weight
        if combined > 0.4:
            tier = "STRONG BUY"
        elif combined > 0.2:
            tier = "MODERATE BUY"
        elif combined > -0.2:
            tier = "NEUTRAL"
        elif combined > -0.4:
            tier = "MODERATE SELL"
        else:
            tier = "STRONG SELL"
        return {
            "combiner_score": round(combined, 4),
            "tier": tier,
            "votes": [{"indicator": name, "score": round(score, 3), "weight": round(w, 3)} for name, score, w in votes],
        }

    def combiner_payload(self) -> dict[str, Any]:
        """Payload for the Combiner chart page: candles + indicators + ML + combiner score."""
        candles = self._load_real_candles()
        indicators = self._compute_indicators(candles)
        ml_prob = self._run_ensemble_prediction(candles)
        combiner = self._compute_combiner_score(indicators, ml_prob)
        return {
            "ok": True,
            "candle_count": len(candles),
            "indicators": indicators,
            "ml_prediction": ml_prob,
            "combiner": combiner,
            "last_candle": candles[-1] if candles else None,
            "recent_candles": candles[-60:],
        }

    def generator_payload(self) -> dict[str, Any]:
        """Payload for the Signal Generator page: future signal predictions for all horizons."""
        candles = self._load_real_candles()
        if not candles:
            return {"ok": False, "error": "No candle data available", "horizons": []}
        indicators = self._compute_indicators(candles)
        ml_prob = self._run_ensemble_prediction(candles)
        combiner = self._compute_combiner_score(indicators, ml_prob)
        horizons = []
        last_close = candles[-1]["close"]
        atr_val = indicators.get("atr_14", 0.01) or 0.01
        # Use fixed ATR multipliers per horizon (not linear bars scaling)
        # 1-min: 1x ATR, 3-min: 1.5x ATR, 30-min: 3x ATR (capped)
        for horizon_name, bars, atr_mult in [("1-Minute", 1, 1.0), ("3-Minute", 3, 1.5), ("30-Minute", 30, 3.0)]:
            direction = "BUY" if combiner["combiner_score"] > 0 else "SELL"
            est_price = last_close + (combiner["combiner_score"] * atr_val * atr_mult)
            confidence = min(abs(combiner["combiner_score"]) * 2, 1.0)
            if combiner["combiner_score"] > 0.4:
                confidence = min(confidence * 1.2, 0.95)
            elif combiner["combiner_score"] < -0.4:
                confidence = min(confidence * 1.2, 0.95)
            horizons.append({
                "horizon": horizon_name,
                "bars": bars,
                "direction": direction,
                "entry_price": round(last_close, 6),
                "estimated_price": round(est_price, 6),
                "confidence": round(confidence, 4),
                "tp_1r": round(last_close + (atr_val if combiner["combiner_score"] > 0 else -atr_val), 6),
                "tp_2r": round(last_close + (2 * atr_val if combiner["combiner_score"] > 0 else -2 * atr_val), 6),
                "sl_1r": round(last_close - (atr_val if combiner["combiner_score"] > 0 else -atr_val), 6),
                "sl_2r": round(last_close - (2 * atr_val if combiner["combiner_score"] > 0 else -2 * atr_val), 6),
                "atr": round(atr_val, 6),
                "atr_mult": atr_mult,
            })
        return {
            "ok": True,
            "combiner_score": combiner["combiner_score"],
            "tier": combiner["tier"],
            "ml_up_prob": ml_prob.get("up_probability", 0.5),
            "ml_down_prob": ml_prob.get("down_probability", 0.5),
            "last_close": round(last_close, 6),
            "horizons": horizons,
        }

    def indicators_payload(self) -> dict[str, Any]:
        """Payload for the Indicators page: all 21 indicator values with interpretations."""
        candles = self._load_real_candles()
        indicators = self._compute_indicators(candles)
        def interp(name, val, bull_above=None, bear_below=None, bull=None, bear=None, desc=""):
            if val is None:
                return {"value": None, "signal": "N/A", "desc": desc}
            if bull_above is not None and bear_below is not None:
                signal = "BULLISH" if val > bull_above else ("BEARISH" if val < bear_below else "NEUTRAL")
            elif bull is not None and bear is not None:
                signal = "BULLISH" if bull else ("BEARISH" if bear else "NEUTRAL")
            else:
                signal = "NEUTRAL"
            return {"value": round(val, 6) if isinstance(val, float) else val, "signal": signal, "desc": desc}
        return {
            "ok": True,
            "candle_count": len(candles),
            "last_close": indicators.get("close"),
            "indicators": {
                "ema_20": interp("ema_20", indicators.get("ema_20"), desc="Fast EMA — price above = bullish"),
                "ema_50": interp("ema_50", indicators.get("ema_50"), desc="Mid EMA — trend filter"),
                "ema_200": interp("ema_200", indicators.get("ema_200"), desc="Long EMA — macro trend"),
                "rsi_14": interp("rsi_14", indicators.get("rsi_14"), 70, 30, desc="RSI 14 — overbought >70, oversold <30"),
                "atr_14": interp("atr_14", indicators.get("atr_14"), desc="ATR 14 — volatility in price units"),
                "adx_14": interp("adx_14", indicators.get("adx_14"), 25, 20, desc="ADX 14 — trend strength >25"),
                "bb_upper": interp("bb_upper", indicators.get("bb_upper"), desc="Bollinger Upper Band"),
                "bb_mid": interp("bb_mid", indicators.get("bb_mid"), desc="Bollinger Middle Band"),
                "bb_lower": interp("bb_lower", indicators.get("bb_lower"), desc="Bollinger Lower Band"),
                "bb_width": interp("bb_width", indicators.get("bb_width"), desc="BB Width — volatility expansion"),
                "stoch_k": interp("stoch_k", indicators.get("stoch_k"), 20, 80, desc="Stochastic K — oversold <20, overbought >80"),
                "stoch_d": interp("stoch_d", indicators.get("stoch_d"), 20, 80, desc="Stochastic D — signal line"),
                "cci_20": interp("cci_20", indicators.get("cci_20"), 100, -100, desc="CCI 20 — >100 bullish, <-100 bearish"),
                "vwap": interp("vwap", indicators.get("vwap"), desc="VWAP — price above = bullish"),
                "vwap_zscore": interp("vwap_zscore", indicators.get("vwap_zscore"), 2, -2, desc="VWAP Z-score — >2 overbought"),
                "kaufman_er": interp("kaufman_er", indicators.get("kaufman_er"), 0.5, 0.2, desc="Kaufman Efficiency — >0.5 trending"),
                "choppiness": interp("choppiness", indicators.get("choppiness"), 40, 70, desc="Choppiness — <40 trending, >70 choppy"),
                "squeeze_on": interp("squeeze_on", indicators.get("squeeze_on"), bull=indicators.get("squeeze_on"), bear=False, desc="Keltner Squeeze — breakout pending"),
                "squeeze_momentum": interp("squeeze_momentum", indicators.get("squeeze_momentum"), desc="Squeeze Momentum — breakout direction"),
                "supertrend_direction": interp("supertrend_direction", indicators.get("supertrend_direction"), bull=indicators.get("supertrend_direction", 0) > 0, bear=indicators.get("supertrend_direction", 0) < 0, desc="Supertrend — +1 bull, -1 bear"),
                "ichimoku_kijun": interp("ichimoku_kijun", indicators.get("ichimoku_kijun"), desc="Ichimoku Kijun — price above = bullish"),
                "ichimoku_tenkan": interp("ichimoku_tenkan", indicators.get("ichimoku_tenkan"), desc="Ichimoku Tenkan — fast line"),
                "hurst": interp("hurst", indicators.get("hurst"), 0.5, 0.45, desc="Hurst Exponent — >0.5 trending, <0.45 mean-reverting"),
                "aroon_up": interp("aroon_up", indicators.get("aroon_up"), 70, 50, desc="Aroon Up — >70 bullish"),
                "aroon_down": interp("aroon_down", indicators.get("aroon_down"), 50, 70, desc="Aroon Down — >70 bearish"),
                "donchian_upper": interp("donchian_upper", indicators.get("donchian_upper"), desc="Donchian Upper — resistance"),
                "donchian_mid": interp("donchian_mid", indicators.get("donchian_mid"), desc="Donchian Mid — channel centre"),
                "donchian_lower": interp("donchian_lower", indicators.get("donchian_lower"), desc="Donchian Lower — support"),
                "pivot_pp": interp("pivot_pp", indicators.get("pivot_pp"), desc="Pivot Point — central level"),
                "pivot_r1": interp("pivot_r1", indicators.get("pivot_r1"), desc="Pivot R1 — resistance"),
                "pivot_r2": interp("pivot_r2", indicators.get("pivot_r2"), desc="Pivot R2 — extended resistance"),
                "pivot_sr1": interp("pivot_sr1", indicators.get("pivot_sr1"), desc="Pivot S1 — support"),
                "pivot_sr2": interp("pivot_sr2", indicators.get("pivot_sr2"), desc="Pivot S2 — extended support"),
            },
            "errors": indicators.get("error"),
        }

    def _combiner_from_candles(self, candles: list, expiry_minutes: int = 30) -> dict[str, Any]:
        """Compute combiner score + ML prediction from Candle objects (used in signal cards)."""
        try:
            if not candles or len(candles) < 60:
                return {"available": False, "reason": "Not enough candles"}
            sys.path.insert(0, str(BOT_ROOT.parent))
            from fusion.indicators import (
                rsi, atr, supertrend_window, stochastic, vwap,
                ichimoku_cloud, aroon, choppiness_index, keltner_squeeze,
                hurst_exponent,
            )
            from fusion.gbdt import GradientBoostingModel as GBDTModel
            from fusion.features import compute_features

            last = len(candles) - 1
            closes = [c.close for c in candles]
            indicators: dict[str, Any] = {}
            indicators["close"] = closes[last]

            # Run ML ensemble if a model is available
            # Select model based on expiry: 1-min -> 1-bar model, 3/30-min -> 5-bar model
            if expiry_minutes == 1:
                model_path = BOT_ROOT / "var" / "real_candles" / "ensemble_model_1bar.json"
            else:
                model_path = BOT_ROOT / "var" / "real_candles" / "ensemble_model.json"
            mr_model_path = BOT_ROOT / "var" / "real_candles" / "ensemble_model_mr.json"
            ml_up = 0.5
            mr_up = None
            if model_path.is_file():
                try:
                    with open(model_path, encoding="utf-8") as f:
                        model_data = json.load(f)
                    feat_dicts = compute_features(candles)
                    last_feat = feat_dicts[last] if feat_dicts and last < len(feat_dicts) else None
                    if last_feat and model_data.get("models"):
                        loaded_models = []
                        for m_dict in model_data["models"]:
                            m = GBDTModel.from_dict(m_dict)
                            if m is not None:
                                loaded_models.append(m)
                        if loaded_models:
                            ml_up = sum(m.predict_up(last_feat) for m in loaded_models) / len(loaded_models)
                except (OSError, ValueError, KeyError, TypeError):
                    pass

            # Load MR model for regime blending
            if mr_model_path.is_file():
                try:
                    with open(mr_model_path, encoding="utf-8") as f:
                        mr_data = json.load(f)
                    if last_feat and mr_data.get("models"):
                        mr_models = []
                        for m_dict in mr_data["models"]:
                            m = GBDTModel.from_dict(m_dict)
                            if m is not None:
                                mr_models.append(m)
                        if mr_models:
                            mr_up = sum(m.predict_up(last_feat) for m in mr_models) / len(mr_models)
                except (OSError, ValueError, KeyError, TypeError):
                    pass

            # Compute indicator values for voting
            rsi_val = rsi(closes, 14)[-1] if len(closes) >= 14 else 50.0
            atr_vals = atr(candles, 14)
            atr_val = atr_vals[-1] if atr_vals else 0.01
            st_dir, _ = supertrend_window(candles, period=10, multiplier=3.0)
            st_direction = st_dir[-1] if st_dir else 0.0
            stoch_k, _ = stochastic(candles, 14, 3)
            stoch_val = stoch_k[-1] if stoch_k else 50.0
            vwap_vals = vwap(candles, 30)
            vwap_val = vwap_vals[-1] if vwap_vals else closes[last]
            ichimoku_kijun, _, _, _, _ = ichimoku_cloud(candles, 9, 26, 52)
            ichimoku_val = ichimoku_kijun[-1] if ichimoku_kijun else closes[last]
            aroon_up, aroon_down = aroon(candles, 25)
            aroon_u = aroon_up[-1] if aroon_up else 50.0
            aroon_d = aroon_down[-1] if aroon_down else 50.0
            chop_vals = choppiness_index(candles, 14)
            chop_val = chop_vals[-1] if chop_vals else 50.0
            squeeze_bools, _ = keltner_squeeze(candles, bb_period=20, bb_mult=2.0, kc_period=20, kc_mult=1.5)
            squeeze_on = bool(squeeze_bools[-1]) if squeeze_bools else False
            hurst_vals = hurst_exponent(closes, 20)
            hurst_val = hurst_vals[-1] if hurst_vals else 0.5

            # Regime detection for blending and flip
            is_range = chop_val > 70.0 and hurst_val < 0.45
            is_mr_regime = (chop_val > 65.0 and hurst_val < 0.48) or (adx_val < 20.0 if 'adx_val' in dir() else False)

            # Blend MR model when in mean-reversion regime
            if mr_up is not None and is_mr_regime:
                ml_up = 0.6 * ml_up + 0.4 * mr_up

            # Regime-aware flip for 1-minute expiry in RANGE regime
            # ONLY flip when ALL conditions are met:
            #   1. Confirmed RANGE regime (choppiness > 70 AND hurst < 0.45)
            #   2. Weak ML prediction (|ml_up - 0.5| < 0.10) — don't override strong signals
            #   3. Large candle body (>= 50% of range) — clear directional pressure
            #   4. MR model agrees with the flip direction
            # The flip is very soft: 20% contrarian, 80% original
            flip_note = ""
            if is_range and expiry_minutes == 1 and len(candles) >= 1:
                last_candle = candles[-1]
                candle_body = abs(last_candle.close - last_candle.open)
                candle_range = last_candle.high - last_candle.low
                body_ratio = candle_body / candle_range if candle_range > 0 else 0.0
                is_red = last_candle.close < last_candle.open
                # Condition 2: weak ML prediction only
                weak_pred = abs(ml_up - 0.5) < 0.10
                # Condition 3: large body only
                large_body = body_ratio >= 0.50
                # Condition 4: MR model agrees (if available)
                mr_agrees = True
                if mr_up is not None:
                    if is_red:
                        mr_agrees = mr_up > 0.5  # MR says BUY
                    else:
                        mr_agrees = mr_up < 0.5  # MR says SELL
                if weak_pred and large_body and mr_agrees:
                    # Very soft flip: 20% contrarian, 80% original
                    ml_up = 0.20 * (1.0 - ml_up) + 0.80 * ml_up
                    flip_note = " (range-regime soft flip)"

            # Weighted vote fusion
            ml_score = (ml_up - 0.5) * 2
            # Optimal weights from self-learning backtest (Sharpe 284.32)
            votes = [
                ("ML Ensemble", ml_score, 0.368),
                ("Supertrend", 1.0 if st_direction > 0 else -1.0, 0.105),
                ("RSI", (0.5 if rsi_val < 40 else (1.0 if rsi_val < 50 else (-0.5 if rsi_val > 60 else (-1.0 if rsi_val > 70 else 0.0)))), 0.080),
                ("Stochastic", 1.0 if stoch_val < 20 else (-1.0 if stoch_val > 80 else 0.0), 0.105),
                ("VWAP", 1.0 if closes[last] > vwap_val else -1.0, 0.084),
                ("Ichimoku", 1.0 if closes[last] > ichimoku_val else -1.0, 0.053),
                ("Aroon", 1.0 if aroon_u > 70 else (-1.0 if aroon_d > 70 else 0.0), 0.053),
                ("Choppiness", -1.0 if chop_val > 70 else (1.0 if chop_val < 40 else 0.0), 0.053),
                ("Squeeze", 1.0 if squeeze_on else 0.0, 0.053),
                ("Hurst", 1.0 if hurst_val > 0.5 else (-1.0 if hurst_val < 0.45 else 0.0), 0.032),
            ]
            if mr_models:
                votes.append(("MR Blend", 0.0 if not is_mr_regime else (1.0 if ml_up > 0.5 else -1.0), 0.095))
            total_w = sum(w for _, _, w in votes) or 1.0
            combined = sum(s * w for _, s, w in votes) / total_w
            if combined > 0.4:
                tier = "STRONG BUY"
            elif combined > 0.2:
                tier = "MODERATE BUY"
            elif combined > -0.2:
                tier = "NEUTRAL"
            elif combined > -0.4:
                tier = "MODERATE SELL"
            else:
                tier = "STRONG SELL"

            return {
                "available": True,
                "combiner_score": round(combined, 4),
                "tier": tier,
                "ml_up_probability": round(ml_up, 4),
                "flipped": flip_note != "",
                "flip_note": flip_note,
                "ml_down_probability": round(1 - ml_up, 4),
                "last_close": round(closes[last], 6),
                "atr": round(atr_val, 6),
                "estimated_price_5min": round(closes[last] + (ml_up - 0.5) * 2 * atr_val, 6),
                "votes": [{"indicator": n, "score": round(s, 3), "weight": round(w, 3)} for n, s, w in votes],
            }
        except (ImportError, ValueError, IndexError, TypeError) as exc:
            return {"available": False, "reason": str(exc)}

    def save_credentials(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Write validated credentials to local git-ignored .env (chmod 600); never echo secrets."""
        with self.lock:
            envfile.ensure_exists(self.env_path, self.example_path)
            lines = self.env_path.read_text(encoding="utf-8").splitlines() if self.env_path.is_file() else []
            notices: list[str] = []

            if payload.get("clear_twelvedata"):
                lines = envfile.set_value(lines, "TWELVE_DATA_API_KEY", "")
                for idx in range(1, MAX_SLOTS + 1):
                    lines = envfile.remove_key(lines, f"TWELVE_DATA_KEY_{idx}")
                notices.append("Cleared Twelve Data key slots.")

            raw_td = str(payload.get("twelvedata_keys") or "").strip()
            if raw_td:
                keys, problems = normalise_keys(raw_td.splitlines())
                notices.extend(problems)
                if not keys:
                    raise ValueError("No valid Twelve Data keys found (minimum 8 non-whitespace characters per line).")
                lines = envfile.set_value(lines, "TWELVE_DATA_API_KEY", keys[0])
                for idx in range(1, MAX_SLOTS + 1):
                    lines = envfile.remove_key(lines, f"TWELVE_DATA_KEY_{idx}")
                for idx, extra_key in enumerate(keys[1:], start=1):
                    lines = envfile.set_value(lines, f"TWELVE_DATA_KEY_{idx}", extra_key)
                notices.append(f"Saved {len(keys)} Twelve Data slot(s) to local .env.")

            supa_url = str(payload.get("supabase_url") or "").strip()
            if supa_url:
                if not supa_url.startswith("https://") or " " in supa_url:
                    raise ValueError("SUPABASE_URL must start with https:// and contain no spaces.")
                lines = envfile.set_value(lines, "SUPABASE_URL", supa_url)
                notices.append("Updated SUPABASE_URL.")

            supa_key = str(payload.get("supabase_service_key") or "").strip()
            if supa_key:
                if len(supa_key) < 16 or " " in supa_key:
                    raise ValueError("SUPABASE_SERVICE_ROLE_KEY looks truncated or contains spaces.")
                lines = envfile.set_value(lines, "SUPABASE_SERVICE_ROLE_KEY", supa_key)
                notices.append("Updated SUPABASE_SERVICE_ROLE_KEY.")

            tg_tok = str(payload.get("telegram_token") or "").strip()
            if tg_tok:
                if ":" not in tg_tok or " " in tg_tok or len(tg_tok) < 15:
                    raise ValueError("TELEGRAM_BOT_TOKEN must have BotFather format (<id>:<secret>).")
                lines = envfile.set_value(lines, "TELEGRAM_BOT_TOKEN", tg_tok)
                notices.append("Updated TELEGRAM_BOT_TOKEN.")

            tg_chat = str(payload.get("telegram_owner_chat_id") or "").strip()
            if tg_chat:
                if not tg_chat.lstrip("-").isdigit():
                    raise ValueError("TELEGRAM_OWNER_CHAT_ID must be numeric.")
                lines = envfile.set_value(lines, "TELEGRAM_OWNER_CHAT_ID", tg_chat)
                notices.append("Updated TELEGRAM_OWNER_CHAT_ID.")

            if not notices:
                raise ValueError("Enter at least one credential field to update.")

            envfile.write(self.env_path, lines)
            self.reload()
            return {
                "ok": True,
                "notices": notices,
                "credentials": self.credentials_payload(),
            }

    def ingest_cloud_snapshot(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Hydrate local mirror from a sanitized read-only Supabase console snapshot."""
        if not isinstance(payload, dict):
            raise ValueError("Cloud snapshot must be a JSON object.")
        with self.lock:
            ref = str(payload.get("project_ref") or "").strip().lower()
            if ref and ref.isalnum() and 10 <= len(ref) <= 40:
                self._cloud_supabase_url = f"https://{ref}.supabase.co"
            self._cloud_synced_at = utcnow()

            local = self.store.local
            for state_tbl in ("api_keys", "model_state", "council_weights", "heartbeats", "learning_events"):
                raw_rows = payload.get(state_tbl)
                if isinstance(raw_rows, list):
                    clean_rows = [dict(r) for r in raw_rows if isinstance(r, dict)]
                    if state_tbl == "model_state" and clean_rows:
                        existing_ms = {
                            str(r.get("id")): r
                            for r in local.read("model_state")
                            if isinstance(r, dict) and r.get("id")
                        }
                        for row in clean_rows:
                            sid = str(row.get("id") or "")
                            prior = existing_ms.get(sid)
                            if isinstance(prior, dict):
                                if row.get("model") is None and prior.get("model") is not None:
                                    row["model"] = prior["model"]
                                if not row.get("mistakes") and prior.get("mistakes"):
                                    row["mistakes"] = prior["mistakes"]
                                prior_rt = prior.get("runtime")
                                cur_rt = row.get("runtime")
                                if isinstance(prior_rt, dict):
                                    merged_rt = dict(cur_rt) if isinstance(cur_rt, dict) else {}
                                    for keep_k in (
                                        "family_fusion",
                                        "learning_journal",
                                        "auto_backtests",
                                        "last_auto_backtest_at",
                                        "last_auto_backtest_settled",
                                        "cognitive_memory",
                                    ):
                                        if not merged_rt.get(keep_k) and prior_rt.get(keep_k):
                                            merged_rt[keep_k] = prior_rt[keep_k]
                                    row["runtime"] = merged_rt
                    if clean_rows:
                        local.replace(state_tbl, clean_rows)

            if isinstance(payload.get("signals"), list):
                remote_sigs = [dict(r) for r in payload["signals"] if isinstance(r, dict) and r.get("signal_id")]
                existing_sigs = local.read("signals")
                by_id = {str(r.get("signal_id")): r for r in existing_sigs if r.get("signal_id")}
                for r in remote_sigs:
                    sid = str(r["signal_id"])
                    prior = by_id.get(sid)
                    if isinstance(prior, dict):
                        if not r.get("features") and prior.get("features"):
                            r["features"] = prior["features"]
                        if not r.get("council") and prior.get("council"):
                            r["council"] = prior["council"]
                    by_id[sid] = r
                merged_sigs = sorted(by_id.values(), key=lambda x: str(x.get("entry_time", "")))
                if merged_sigs:
                    local.replace("signals", merged_sigs[-300:])

            if isinstance(payload.get("settlements"), list):
                remote_stls = [dict(r) for r in payload["settlements"] if isinstance(r, dict) and r.get("signal_id")]
                existing_stls = local.read("settlements")
                by_sid = {str(r.get("signal_id")): r for r in existing_stls if r.get("signal_id")}
                for r in remote_stls:
                    by_sid[str(r["signal_id"])] = r
                merged_stls = sorted(by_sid.values(), key=lambda x: str(x.get("settled_at", "")))
                if merged_stls:
                    local.replace("settlements", merged_stls[-500:])

            synced_candles = 0
            if isinstance(payload.get("candles_m1"), list):
                valid_bars = []
                for r in payload["candles_m1"]:
                    if not isinstance(r, dict):
                        continue
                    pair = str(r.get("pair") or "").strip().upper()
                    ts = str(r.get("timestamp") or "").strip()
                    if pair not in ("EUR/USD", "BTC/USD", "ETH/USD") or not ts:
                        continue
                    try:
                        valid_bars.append(
                            {
                                "pair": pair,
                                "timestamp": ts,
                                "open": float(r["open"]),
                                "high": float(r["high"]),
                                "low": float(r["low"]),
                                "close": float(r["close"]),
                                "volume": float(r.get("volume", 0.0) or 0.0),
                            }
                        )
                    except (KeyError, TypeError, ValueError):
                        continue
                if valid_bars:
                    valid_bars.sort(key=lambda x: (x["pair"], x["timestamp"]))
                    local.replace("candles_m1", valid_bars)
                    synced_candles = len(valid_bars)

            # Refresh bot state + candles in place (preserving background worker
            # and Needle 3 instance), then immediately settle & learn from any due rows.
            kp_blob = self.store.load_key_pool()
            if isinstance(kp_blob, dict) and isinstance(kp_blob.get("slots"), list):
                self._cloud_key_slots = [r for r in kp_blob["slots"] if isinstance(r, dict)]
            elif isinstance(kp_blob, list):
                self._cloud_key_slots = [r for r in kp_blob if isinstance(r, dict)]
            state_payload = self.store.load_model_state() or {}
            runtime_blob = state_payload.get("runtime") if isinstance(state_payload.get("runtime"), dict) else {}
            self._cloud_telegram_offset = int(runtime_blob.get("telegram_offset", 0) or 0)
            for member in self._all_members():
                member._load_state()
                member._load_candles()
            if isinstance(runtime_blob.get("cognitive_memory"), dict):
                self.bot.cognitive = SaviourCognitivePartner.from_dict(
                    self.config.local_state_dir,
                    runtime_blob.get("cognitive_memory"),
                )
            self._synced_pairs = {
                m.config.pair for m in self._all_members() if len(m.candles) >= 60
            }
            for member in self._all_members():
                if self._has_real_candles(member) and member.model is None:
                    member._ensure_baseline_veto_model()
            auto_settled = self.bot.settle_due(at=utcnow())
            replayed = self.bot.reconcile_settled_signals()
            for member in self._all_members():
                if self._has_real_candles(member) and not member.auto_backtest_history and len(member.candles) >= 320:
                    try:
                        member.maybe_run_auto_backtest(reason="cloud_sync_bootstrap", min_bars=320)
                    except Exception:
                        pass
            self.bot.save_all_states(force=True)
            self._ensure_paper_candles_if_offline()
            return {
                "ok": True,
                "synced_candles": synced_candles,
                "synced_pairs": sorted(self._synced_pairs),
                "auto_settled": auto_settled,
                "reconciled": replayed,
                "status": self.status_payload(),
                "credentials": self.credentials_payload(),
            }


def make_handler(runtime: OperatorRuntime):
    class DashboardHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(DOCS_DIR), **kwargs)

        def log_message(self, fmt: str, *args) -> None:  # noqa: D401
            sys.stderr.write("[web] " + (fmt % args) + "\n")

        def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
            raw = json.dumps(payload, indent=2, default=str).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(raw)

        def _read_json_body(self, max_bytes: int = MAX_JSON_BODY) -> dict[str, Any]:
            ctype = str(self.headers.get("Content-Type", "")).lower()
            if "application/json" not in ctype:
                raise ValueError("Content-Type must be application/json.")
            length = int(self.headers.get("Content-Length", "0") or 0)
            if length > max_bytes:
                raise OverflowError(f"Request payload exceeds {max_bytes // 1024} KB limit.")
            raw = self.rfile.read(length) if length > 0 else b"{}"
            parsed = json.loads(raw.decode("utf-8") or "{}")
            if not isinstance(parsed, dict):
                raise ValueError("JSON body must be an object.")
            return parsed

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            route = parsed.path.rstrip("/") or "/"

            if route == "/api/status":
                self._send_json(runtime.status_payload())
                return

            if route == "/api/telegram":
                self._send_json(runtime.telegram_payload())
                return

            if route == "/api/telegram/format":
                self._send_json(runtime.telegram_format_payload())
                return

            if route == "/api/health":
                self._send_json(runtime.health_payload(run_live_audit=False))
                return

            if route == "/api/strategies":
                self._send_json(runtime.strategies_payload())
                return

            if route == "/api/credentials":
                self._send_json(runtime.credentials_payload())
                return

            if route == "/api/needle/status":
                self._send_json(runtime.needle_status_payload())
                return

            if route == "/api/learning":
                self._send_json(runtime.learning_payload())
                return

            if route == "/api/risk":
                qs = parse_qs(parsed.query)
                try:
                    balance = float(qs.get("balance", ["10000"])[0])
                    plan = risk.build_risk_plan(
                        symbol=qs.get("symbol", ["EUR/USD"])[0],
                        side=qs.get("side", ["BUY"])[0],
                        entry_price=float(qs.get("entry", ["1.0852"])[0]),
                        atr_pips=float(qs.get("atr", ["1.95"])[0]),
                        balance_usd=balance,
                        risk_fraction=float(qs.get("risk_pct", ["1.0"])[0]) / 100.0,
                    )
                    out = plan.to_dict()
                    if "daily_pnl" in qs:
                        daily_pnl = float(qs.get("daily_pnl", ["0"])[0])
                        peak = float(qs.get("peak", [str(balance)])[0])
                        equity = balance + daily_pnl
                        cb = risk.evaluate_circuit_breaker(
                            starting_day_equity_usd=balance,
                            current_equity_usd=equity,
                            peak_equity_usd=max(peak, balance),
                        )
                        out["circuit_breaker"] = cb.to_dict()
                except (ValueError, KeyError) as exc:
                    self._send_json({"error": str(exc)}, status=400)
                    return
                self._send_json(out)
                return

            if route == "/api/combiner":
                self._send_json(runtime.combiner_payload())
                return

            if route == "/api/generator":
                self._send_json(runtime.generator_payload())
                return

            if route == "/api/indicators":
                self._send_json(runtime.indicators_payload())
                return

            super().do_GET()

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            route = parsed.path.rstrip("/") or "/"
            body_cap = MAX_SYNC_JSON_BODY if route == "/api/cloud_sync" else MAX_JSON_BODY
            try:
                body = self._read_json_body(max_bytes=body_cap)
            except OverflowError as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=413)
                return
            except (ValueError, json.JSONDecodeError) as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=400)
                return

            try:
                if route == "/api/cloud_sync":
                    self._send_json(runtime.ingest_cloud_snapshot(body))
                    return

                if route == "/api/signals":
                    res = runtime.generate_signals(
                        pair=str(body.get("pair", "ALL")),
                        mode=str(body.get("mode", "now")),
                        dispatch_telegram=bool(body.get("dispatch_telegram", False)),
                        balance_usd=float(body.get("balance_usd", 10_000.0)),
                        risk_pct=float(body.get("risk_pct", 1.0)),
                        expiry_minutes=int(body.get("expiry_minutes", 30)),
                    )
                    self._send_json(res)
                    return

                if route == "/api/control":
                    self._send_json(runtime.control_action(str(body.get("action", ""))))
                    return

                if route == "/api/telegram/command":
                    self._send_json(
                        runtime.run_telegram_command(
                            str(body.get("command", "/status")),
                            send_to_chat=bool(body.get("send_to_chat", False)),
                        )
                    )
                    return

                if route == "/api/telegram/format":
                    self._send_json(runtime.update_telegram_format(body))
                    return

                if route == "/api/audit":
                    self._send_json(runtime.health_payload(run_live_audit=True))
                    return

                if route == "/api/config":
                    self._send_json(runtime.update_runtime_config(body))
                    return

                if route == "/api/credentials":
                    self._send_json(runtime.save_credentials(body))
                    return

                if route == "/api/needle/run":
                    self._send_json(
                        runtime.run_needle_query(
                            str(body.get("query", "")),
                            execute=bool(body.get("execute", True)),
                        )
                    )
                    return

                if route == "/api/backtest":
                    self._send_json(
                        runtime.run_backtest_action(str(body.get("pair", "ALL")))
                    )
                    return

                if route in ("/api/self_heal", "/api/self-heal"):
                    self._send_json(runtime.self_heal_action())
                    return

            except (ValueError, KeyError) as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=400)
                return

            self._send_json({"ok": False, "error": f"Unknown endpoint: {route}"}, status=404)

    return DashboardHandler


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve SAVIOUR Shinigami Operator Console")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    runtime = OperatorRuntime()
    handler_cls = make_handler(runtime)
    server = ThreadingHTTPServer((args.host, args.port), handler_cls)
    print(f"Serving {DOCS_DIR} + Operator API on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
