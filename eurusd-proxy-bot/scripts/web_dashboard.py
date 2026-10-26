#!/usr/bin/env python3
"""SAVIOUR Operator Console & Control API Server.

    python3 eurusd-proxy-bot/scripts/web_dashboard.py --port 8080

Stdlib only. Binds 0.0.0.0 by default so browser previews and mobile devices on
the local network can reach it, and never exposes any plaintext secret value
from .env or os.environ.
"""

from __future__ import annotations

import argparse
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
TESTS_PASSING_BASELINE = 702

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
    """Build instrument-scaled synthetic candles strictly marked for paper simulation."""
    base_price, vol_step, seed = SIM_DEFAULTS.get(symbol, (1.0852, 0.00024, 101))
    start = end_minute - timedelta(minutes=count - 1)
    raw = synthetic_series(
        count=count,
        seed=seed + seed_offset,
        start=start,
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

    def __init__(self, env_path: Path | None = None, state_dir: Path | None = None):
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
        self.config: Config = Config()
        self.store: Store = None  # type: ignore[assignment]
        self.bot: FusionBot = None  # type: ignore[assignment]
        self.reload()

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
        if self.config.supabase_ready or self._cloud_supabase_url:
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
            signals_raw = self.store.load_signals(limit=25)
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
            for row in signals_raw[-12:][::-1]:
                item = dict(row)
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
                    "recent_signals": enriched_recent_signals,
                    "recent_settlements": settlements_raw[-12:][::-1],
                },
                "model": {
                    "trained": bool(self.bot.model and self.bot.model.trained),
                    "authority": self.bot.model.mode if self.bot.model else MODE_VETO,
                    "calibration": cal_state,
                    "calibration_samples": calibrator.samples if calibrator else 0,
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

                plan_side = (
                    result.signal.side
                    if result.signal is not None
                    else (
                        result.council.side
                        if result.council and result.council.side in ("BUY", "SELL")
                        else "BUY"
                    )
                )
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
                        member.store.local.write("signals", [sig_row])
                    signal_dict = sig.to_dict()
                    signal_dict["entry_time"] = format_display_iso(sig.entry_time)
                    signal_dict["expiry_time"] = format_display_iso(sig.expiry_time)
                    signal_dict["generated_at"] = format_display_iso(sig.generated_at)
                    signal_dict["entry_hm"] = format_display_hm(sig.entry_time)
                    signal_dict["expiry_hm"] = format_display_hm(sig.expiry_time)
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
                        "mtf_minutes": member.config.mtf_minutes,
                        "mtf_bias": mtf_dir,
                        "actionable": result.actionable,
                        "seal_state": result.seal_state,
                        "side": result.signal.side if result.signal else "NO SIGNAL",
                        "reason": result.reason,
                        "regime": result.regime,
                        "core_checks": core_score,
                        "min_checks": member.config.min_confluence_checks,
                        "votes": votes_list,
                        "model_read": result.read.to_dict() if result.read else None,
                        "signal": signal_dict,
                        "brief_text": brief_text,
                        "full_text": full_text,
                        "telegram_dispatched": telegram_dispatched,
                        "risk_plan": risk_plan.to_dict(),
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
            return {
                "ok": True,
                "enabled": self.config.needle_enabled,
                **self.needle.status(),
            }

    def run_needle_query(self, query: str, *, execute: bool = True) -> dict[str, Any]:
        with self.lock:
            decision = self.needle.route(query)
            if not execute or decision.type != "tool_call" or not decision.function_calls:
                return {
                    "ok": True,
                    "executed": False,
                    "decision": decision.to_dict(),
                    "reply": decision.refusal_reason or "No tool executed.",
                    "signal_run": None,
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
                # Filter out duplicate scan_market text when web signal studio also runs below
                if not scan_calls:
                    return {
                        "ok": True,
                        "executed": True,
                        "decision": sub_agent_decision.to_dict(),
                        "reply": sub_reply,
                        "signal_run": None,
                    }
                # Re-run only non-scan outputs cleanly for reply banner
                replies.append(sub_reply.split("\n\n")[0])

            for call in scan_calls:
                pair = str(call["arguments"].get("pair") or "ALL")
                mode = str(call["arguments"].get("mode") or "now")
                signal_run = self.generate_signals(pair=pair, mode=mode, dispatch_telegram=False)
                cards = signal_run.get("cards") or []
                summary_lines = [
                    f"{c['pair']}: {c['side']} ({c['seal_state']}) — {c['reason']}"
                    for c in cards
                ]
                replies.append("\n".join(summary_lines))

            return {
                "ok": True,
                "executed": True,
                "decision": decision.to_dict(),
                "reply": "\n\n".join(r for r in replies if r),
                "signal_run": signal_run,
            }

    def strategies_payload(self) -> dict[str, Any]:
        with self.lock:
            weights = self.bot.weights.weights
            seen = self.bot.weights.seen
            rows = []
            all_ids = (strategies.CORE, *strategies.PROTOTYPES)
            for sid in all_ids:
                allowed_regimes = [
                    reg for reg in ("TREND", "RANGE", "VOLATILE") if regime_mod.regime_allows(reg, sid)
                ]
                family = (
                    "CORE_GATE"
                    if sid == strategies.CORE
                    else "SCALPING"
                    if sid.startswith("scalp_")
                    else "COUNCIL"
                )
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
            supa_configured = bool((supa_url and supa_key) or self._cloud_supabase_url)
            if self._supabase_reachable:
                supa_mode = "LIVE"
            elif supa_url and supa_key:
                supa_mode = "BRIDGED_MIRROR"
            elif self._cloud_supabase_url:
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
                    "key_masked": mask_key(supa_key) if supa_key else ("CLOUD_RPC_PUBLIC" if self._cloud_supabase_url else ""),
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
                    if clean_rows:
                        local.replace(state_tbl, clean_rows)

            if isinstance(payload.get("signals"), list):
                remote_sigs = [dict(r) for r in payload["signals"] if isinstance(r, dict) and r.get("signal_id")]
                existing_sigs = local.read("signals")
                by_id = {str(r.get("signal_id")): r for r in existing_sigs if r.get("signal_id")}
                for r in remote_sigs:
                    by_id[str(r["signal_id"])] = r
                merged_sigs = sorted(by_id.values(), key=lambda x: str(x.get("entry_time", "")))
                if merged_sigs:
                    local.replace("signals", merged_sigs[-200:])

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

            self.reload()
            return {
                "ok": True,
                "synced_candles": synced_candles,
                "synced_pairs": sorted(self._synced_pairs),
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


if __name__ == "__main__":
    raise SystemExit(main())
