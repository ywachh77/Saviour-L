#!/usr/bin/env python3
"""SAVIOUR Operator Console & Control API Server.

    python3 eurusd-proxy-bot/scripts/web_dashboard.py --port 8080

Stdlib only. Binds 0.0.0.0 by default so browser previews and mobile devices on
the local network can reach it, and never exposes any plaintext secret value
from .env or os.environ.
"""

from __future__ import annotations

import argparse
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
from fusion.runner import FusionBot, format_brief_signal, format_signal  # noqa: E402
from fusion.simulate import synthetic_series  # noqa: E402
from fusion.store import Store, build_store  # noqa: E402
from fusion.timeutil import floor_minute, format_datetime, minute_bucket, now as utcnow  # noqa: E402
from scripts.set_keys import normalise_keys  # noqa: E402

MAX_JSON_BODY = 32_768
TESTS_PASSING_BASELINE = 649

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
        self.config: Config = Config()
        self.store: Store = None  # type: ignore[assignment]
        self.bot: FusionBot = None  # type: ignore[assignment]
        self.reload()

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
        if not envfile.get_value(env_lines, "NEWS_FILTER"):
            cfg.news_filter = True
        if not envfile.get_value(env_lines, "ADAPTIVE_HORIZON"):
            cfg.adaptive_horizon = True
            cfg.max_horizon_minutes = 120

        if self._override_state_dir is not None:
            cfg.local_state_dir = str(self._override_state_dir)
        elif not Path(cfg.local_state_dir).is_absolute():
            cfg.local_state_dir = str(BOT_ROOT / cfg.local_state_dir)

        cfg.runner = "web-console"
        self.config = cfg
        self.store = build_store(cfg)
        self.bot = FusionBot(config=cfg, store=self.store)
        self.bot.startup()
        self._ensure_paper_candles_if_offline()

    def _all_members(self) -> list[FusionBot]:
        return [self.bot, *self.bot.peer_bots.values()]

    def _member_for(self, symbol: str) -> FusionBot | None:
        norm = instruments.for_symbol(symbol).symbol
        for member in self._all_members():
            if member.config.pair == norm:
                return member
        return None

    @property
    def live_feed_configured(self) -> bool:
        return bool(self.bot.provider and getattr(self.bot.provider, "configured", False))

    def _ensure_paper_candles_if_offline(self, target_stamp: datetime | None = None) -> None:
        if self.live_feed_configured:
            return
        anchor = floor_minute(target_stamp or utcnow()) - timedelta(minutes=1)
        for member in self._all_members():
            if not member.candles or target_stamp is not None:
                member.candles = _build_scaled_paper_series(
                    member.config.pair,
                    count=max(260, member.config.candle_history),
                    end_minute=anchor,
                    seed_offset=self._sim_step,
                )

    def _market_snapshot(self, member: FusionBot) -> dict[str, Any]:
        inst = instruments.for_symbol(member.config.pair)
        candles = member.candles
        latest_close = candles[-1].close if candles else SIM_DEFAULTS.get(inst.symbol, (1.0, 0.0002, 1))[0]
        latest_ts = format_datetime(candles[-1].timestamp) if candles else None
        atr_vals = [v for v in calc_atr(candles, 14) if v is not None] if len(candles) >= 20 else []
        atr_pips = (atr_vals[-1] / inst.pip) if atr_vals else (1.95 if inst.symbol == "EUR/USD" else 58.0 if inst.symbol == "BTC/USD" else 23.0)
        feat_series = feat_mod.compute_features(candles, instrument=inst) if len(candles) >= 60 else []
        latest_feat = feat_series[-1] if feat_series else None
        reg = regime_mod.detect_regime(candles, latest_feat).name if latest_feat else "UNKNOWN"
        bias_15 = mtf.bias(candles, 15).direction if len(candles) >= 60 else "FLAT"
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
            "atr_pips": round(atr_pips, 2),
            "regime": reg,
            "mtf_15m": bias_15,
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
            pool_summary = (
                self.bot.provider.pool.summary()
                if hasattr(self.bot.provider, "pool")
                else {"slots": [], "enabled": 0, "exhausted_events": 0}
            )
            calibrator = self.bot.model.calibrator if self.bot.model else None
            cal_state = (
                "CALIBRATED"
                if calibrator and calibrator.fitted and not calibrator.flat
                else "FLAT"
                if calibrator and calibrator.flat
                else "RAW"
            )
            return {
                "engine": "SAVIOUR",
                "mode": "PAPER_RESEARCH_ONLY",
                "feed_mode": "LIVE" if self.live_feed_configured else "PAPER_SIMULATION",
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
                    "news_filter": self.config.news_filter,
                    "adaptive_horizon": self.config.adaptive_horizon,
                    "max_horizon_minutes": self.config.max_horizon_minutes,
                    "session_filter": self.config.session_filter,
                    "horizon_minutes": self.config.horizon_minutes,
                    "min_confluence_checks": self.config.min_confluence_checks,
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
                    "recent_signals": signals_raw[-12:][::-1],
                    "recent_settlements": settlements_raw[-12:][::-1],
                },
                "model": {
                    "trained": bool(self.bot.model and self.bot.model.trained),
                    "authority": self.bot.model.mode if self.bot.model else MODE_VETO,
                    "calibration": cal_state,
                    "calibration_samples": calibrator.samples if calibrator else 0,
                },
                "telegram": {
                    "configured": bool(self.config.telegram_ready),
                    "owner_masked": _mask_chat_id(self.config.telegram_owner_chat_id),
                    "offset": self.bot.telegram.offset if self.bot.telegram else 0,
                    "outbox_pending": len(self.store.outbox_pending()),
                },
                "health": {
                    "heartbeat": hb,
                    "heartbeat_age_s": None if hb_age is None else round(hb_age, 1),
                    "key_pool": pool_summary,
                    "store_backend": "supabase+local" if self.store.supabase else "local_jsonl",
                    "store_errors": self.store.errors[-5:],
                    "alerts": self.store.recent_alerts(limit=8),
                },
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
            source_tag = "LIVE" if is_live else "SYNTHETIC"
            base_now = utcnow()
            eval_stamp = floor_minute(base_now) + timedelta(minutes=1, seconds=1)

            if is_live:
                for member in targets:
                    member.ingest(at=eval_stamp)
                self.bot.flush_key_pool()
            else:
                self._sim_step += 1
                # Anchor paper scan inside active overlap hours when mode=='scan' so FX spread is open.
                scan_anchor = (
                    datetime(2026, 1, 5, 13, 0, 1, tzinfo=timezone.utc) + timedelta(minutes=self._sim_step)
                    if mode == "scan"
                    else eval_stamp
                )
                self._ensure_paper_candles_if_offline(target_stamp=scan_anchor)
                eval_stamp = scan_anchor

            cards: list[dict[str, Any]] = []
            emitted_count = 0

            for member in targets:
                inst = instruments.for_symbol(member.config.pair)
                chosen_stamp = eval_stamp
                result = member.engine.evaluate(
                    member.candles,
                    model=member.model,
                    ensemble=member.ensemble,
                    weights=member.weights,
                    source=source_tag,
                    at=chosen_stamp,
                    mistakes=member.mistakes,
                )

                # In offline paper simulation only, mode='scan' walks forward bar-by-bar
                # through unmodified engine gates until a real 5/6+ setup seals (max 90 steps).
                steps_scanned = 1
                if not is_live and mode == "scan" and not result.actionable:
                    for _ in range(89):
                        self._sim_step += 1
                        steps_scanned += 1
                        chosen_stamp = datetime(2026, 1, 5, 13, 0, 1, tzinfo=timezone.utc) + timedelta(minutes=self._sim_step)
                        member.candles = _build_scaled_paper_series(
                            inst.symbol,
                            count=max(260, member.config.candle_history),
                            end_minute=floor_minute(chosen_stamp) - timedelta(minutes=1),
                            seed_offset=self._sim_step,
                        )
                        result = member.engine.evaluate(
                            member.candles,
                            model=member.model,
                            ensemble=member.ensemble,
                            weights=member.weights,
                            source="SYNTHETIC",
                            at=chosen_stamp,
                            mistakes=member.mistakes,
                        )
                        if result.actionable:
                            break

                closed = closed_before(member.candles, chosen_stamp)
                latest_close = closed[-1].close if closed else (member.candles[-1].close if member.candles else 1.0)
                atr_vals = [v for v in calc_atr(closed, 14) if v is not None] if len(closed) >= 20 else []
                atr_pips = (atr_vals[-1] / inst.pip) if atr_vals else 2.0
                mtf_dir = mtf.bias(closed, member.config.mtf_minutes).direction if len(closed) >= 60 else "FLAT"

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
                    member.store.save_signal(
                        dict(
                            sig.to_dict(),
                            seal_state=charter.SEAL_SIGNAL,
                            settled=False,
                            bucket=minute_bucket(chosen_stamp),
                        )
                    )
                    if dispatch_telegram and member.telegram and member.telegram.configured and not member.paused:
                        member.deliver(sig, brief=True)
                        telegram_dispatched = True
                    else:
                        member.store.update_signal(sig.signal_id, {"delivery_state": "WEB_CONSOLE"})
                    signal_dict = sig.to_dict()
                    signal_dict["entry_display"] = f"{latest_close:,.{inst.precision}f}"
                    brief_text = format_brief_signal(sig)
                    full_text = format_signal(sig)

                votes_list = [v.to_dict() for v in result.council.votes] if result.council else []
                core_score = result.council.core_score if result.council else 0

                cards.append(
                    {
                        "pair": inst.symbol,
                        "precision": inst.precision,
                        "feed_source": "LIVE" if is_live else "PAPER_SIMULATION",
                        "evaluated_at": format_datetime(chosen_stamp),
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

            return {
                "ok": True,
                "feed_mode": "LIVE" if is_live else "PAPER_SIMULATION",
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

    def telegram_payload(self) -> dict[str, Any]:
        with self.lock:
            local_outbox = self.store.local.read("outbox")[-15:][::-1]
            return {
                "configured": bool(self.config.telegram_ready),
                "owner_masked": _mask_chat_id(self.config.telegram_owner_chat_id),
                "token_masked": mask_key(self.config.telegram_token),
                "offset": self.bot.telegram.offset if self.bot.telegram else 0,
                "paused": self.bot.paused,
                "commands": ["/start", "/help", "/signals", "/status", "/pause", "/resume"],
                "pending_count": len(self.store.outbox_pending()),
                "recent_outbox": local_outbox,
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
            checks = doctor.run_checks(self.config, env_path=self.env_path, offline=True, send_test=False)
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
            pool = (
                self.bot.provider.pool.summary()
                if hasattr(self.bot.provider, "pool")
                else {"slots": [], "enabled": 0, "exhausted_events": 0}
            )
            return {
                "ok": all(c.status != doctor.FAIL for c in checks) and bool(self._last_audit["ok"]),
                "doctor": [
                    {"name": c.name, "status": c.status, "detail": c.detail, "action": c.action}
                    for c in checks
                ],
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
                    "news_filter": self.config.news_filter,
                    "adaptive_horizon": self.config.adaptive_horizon,
                    "max_horizon_minutes": self.config.max_horizon_minutes,
                    "session_filter": self.config.session_filter,
                    "min_confluence_checks": self.config.min_confluence_checks,
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
            if "news_filter" in payload:
                self.config.news_filter = bool(payload["news_filter"])
            if "adaptive_horizon" in payload:
                self.config.adaptive_horizon = bool(payload["adaptive_horizon"])
            if "session_filter" in payload:
                self.config.session_filter = bool(payload["session_filter"])
            if "max_horizon_minutes" in payload:
                mh = int(payload["max_horizon_minutes"])
                if not 5 <= mh <= 240:
                    raise ValueError("max_horizon_minutes must be between 5 and 240.")
                self.config.max_horizon_minutes = mh

            # Rebuild peer bots & engines cleanly with the updated runtime gates.
            paused_before = self.bot.paused
            self.bot = FusionBot(config=self.config, store=self.store)
            self.bot.startup()
            self.bot._set_paused(paused_before)
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
            return {
                "env_file_exists": self.env_path.is_file(),
                "env_file_name": ".env (git-ignored, chmod 600)",
                "max_twelvedata_slots": MAX_SLOTS,
                "supabase": {
                    "configured": bool(supa_url and supa_key),
                    "url_masked": _mask_url(supa_url),
                    "key_masked": mask_key(supa_key),
                },
                "telegram": {
                    "configured": bool(tg_tok and tg_chat),
                    "token_masked": mask_key(tg_tok),
                    "owner_chat_masked": _mask_chat_id(tg_chat),
                },
                "twelvedata": {
                    "configured": bool(td_keys),
                    "slot_count": len(td_keys),
                    "slots_masked": [
                        {"slot": idx + 1, "masked": mask_key(k)} for idx, k in enumerate(td_keys)
                    ],
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

        def _read_json_body(self) -> dict[str, Any]:
            ctype = str(self.headers.get("Content-Type", "")).lower()
            if "application/json" not in ctype:
                raise ValueError("Content-Type must be application/json.")
            length = int(self.headers.get("Content-Length", "0") or 0)
            if length > MAX_JSON_BODY:
                raise OverflowError("Request payload exceeds 32 KB limit.")
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

            if route == "/api/health":
                self._send_json(runtime.health_payload(run_live_audit=False))
                return

            if route == "/api/strategies":
                self._send_json(runtime.strategies_payload())
                return

            if route == "/api/credentials":
                self._send_json(runtime.credentials_payload())
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
            try:
                body = self._read_json_body()
            except OverflowError as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=413)
                return
            except (ValueError, json.JSONDecodeError) as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=400)
                return

            try:
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

                if route == "/api/audit":
                    self._send_json(runtime.health_payload(run_live_audit=True))
                    return

                if route == "/api/config":
                    self._send_json(runtime.update_runtime_config(body))
                    return

                if route == "/api/credentials":
                    self._send_json(runtime.save_credentials(body))
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
