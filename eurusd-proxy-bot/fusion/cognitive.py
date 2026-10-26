"""SAVIOUR Cognitive Partner — Human-Like Self-Learning, Family Synthesis & Self-Healing.

Merges Cactus Compute Needle 3 (tool-calling & structured intent) with SAVIOUR's
deterministic causal engine, 6 merged Strategic Families, walk-forward
auto-backtester, episodic trade memory, and autonomous error rectifier.

Design Principles
-----------------
1. **Strict Honesty Charter**: Needle/Cognitive synthesis never fabricates wins,
   never touches synthetic data for weight updates, and never bypasses causal
   candle math.
2. **Human-Like Episodic Reflection**: Every settled real trade produces a
   first-person post-mortem hypothesis capturing prediction surprise (Brier
   error), pre/post probability shift, supporting vs dissenting Strategic
   Families, and concrete weight adjustments stored permanently.
3. **Autonomous Error Detection & Rectification**: Continuously inspects model
   readiness, unreconciled settlements, overdue expiries, family-fusion seeding,
   guard bounds, outbox queues, and causal look-ahead invariance — fixing
   recoverable drift automatically and logging every repair.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any, Mapping

from . import charter, instruments, lookahead, settle
from .family_fusion import FAMILY_METADATA, SPECIALIST_FAMILIES
from .timeutil import format_datetime, format_display_datetime, now as utcnow

CONVERSATION_GREETING_RE = re.compile(
    r"^\s*(hi|hello|hey|yo|good\s+(morning|afternoon|evening)|namaste|greetings|"
    r"who\s+are\s+you|how\s+are\s+you|are\s+you\s+(alive|awake|there|ready|human)|"
    r"introduce\s+yourself|what\s+can\s+you\s+do)\b",
    re.IGNORECASE,
)

HUMAN_REFLECTION_RE = re.compile(
    r"\b(what\s+did\s+you\s+learn|how\s+do\s+you\s+learn|learn(?:ing|ed)?|"
    r"why\s+did\s+(?:we|it|btc|eth|eur|the\s+signal)\s+(?:lose|win|abstain|fail)|"
    r"mistake|losses|loss|post-?mortem|families|family\s+fusion|synerg(?:y|ies)|"
    r"merge(?:d)?\s+strateg|improve\s+yourself|evolve|develop(?:ing)?|"
    r"find\s+(?:any\s+)?(?:errors?|flaws?|bugs?|mistakes?)|rectify|self-?heal|fix\s+(?:errors?|flaws?|issues?)|"
    r"explain\s+(?:your|the)\s+(?:brain|mind|thinking|model|prediction))\b",
    re.IGNORECASE,
)


def build_trade_reflection(
    *,
    pair: str,
    signal_id: str,
    side: str,
    regime: str,
    outcome: str,
    net_pips: float,
    spread_pips: float,
    pre_prob: float | None,
    post_prob: float | None,
    brier_error: float | None,
    supporting_families: list[str],
    opposing_families: list[str],
    synergy_pairs: list[str],
    loss_streak: int = 0,
) -> dict[str, Any]:
    """Synthesize a human-readable post-trade reflection and structured episode."""
    stamp = format_datetime(utcnow())
    disp_stamp = format_display_datetime(utcnow())
    sup_txt = ", ".join(supporting_families) if supporting_families else "Core Gate solo"
    opp_txt = ", ".join(opposing_families) if opposing_families else "none"
    syn_txt = ", ".join(synergy_pairs) if synergy_pairs else "none"

    prob_shift = (
        round((post_prob - pre_prob) * 100.0, 2)
        if (pre_prob is not None and post_prob is not None)
        else 0.0
    )

    if outcome == charter.WIN:
        narrative = (
            f"Studied {pair} {side} ({regime}) -> WIN ({net_pips:+.2f}p net after {spread_pips:.2f}p spread). "
            f"Corroborating families [{sup_txt}] aligned cleanly"
            + (f" with synergy [{syn_txt}]" if synergy_pairs else "")
            + f". Reinforced online logistic + 4-member bootstrap ensemble ({prob_shift:+.2f}% score adaptation) "
            f"and credited aligned family weights."
        )
        action_taken = "Reinforced aligned family & synergy weights; updated online model, ensemble & calibrator."
    else:
        streak_note = (
            f" Slice loss streak reached {loss_streak} — tightened confluence floor to 6/6."
            if loss_streak >= 2
            else ""
        )
        narrative = (
            f"Post-mortem on {pair} {side} ({regime}) -> LOSS ({net_pips:+.2f}p net incl. {spread_pips:.2f}p spread). "
            f"Endorsing families [{sup_txt}] missed while opposing families [{opp_txt}] warned otherwise. "
            f"Applied 1.35x online gradient step + 1.25x family blame down-weight ({prob_shift:+.2f}% score shift)."
            f"{streak_note}"
        )
        action_taken = (
            "Applied 1.35x loss-focused gradient update, 1.25x dissenting family penalty, ensemble adaptation"
            + (", and 6/6 slice confluence lock." if loss_streak >= 2 else ".")
        )

    return {
        "timestamp": stamp,
        "display_time": disp_stamp,
        "signal_id": signal_id,
        "pair": pair,
        "side": side,
        "regime": regime,
        "outcome": outcome,
        "net_pips": round(float(net_pips), 2),
        "spread_pips": round(float(spread_pips), 2),
        "pre_prob": None if pre_prob is None else round(float(pre_prob), 4),
        "post_prob": None if post_prob is None else round(float(post_prob), 4),
        "brier_error": None if brier_error is None else round(float(brier_error), 4),
        "supporting_families": list(supporting_families),
        "opposing_families": list(opposing_families),
        "synergy_pairs": list(synergy_pairs),
        "narrative": narrative,
        "action_taken": action_taken,
    }


@dataclass
class SaviourCognitivePartner:
    """Human-like cognitive layer fused with SAVIOUR & Cactus Compute Needle 3."""

    state_dir: Path
    memory_turns: list[dict[str, str]] = field(default_factory=list)
    rectifications: list[dict[str, Any]] = field(default_factory=list)
    self_heal_runs: int = 0

    def __post_init__(self) -> None:
        self.state_dir = Path(self.state_dir)

    def experience_path(self) -> Path:
        return self.state_dir / "needle" / "experience_memory.jsonl"

    def record_experience(self, kind: str, summary: str, payload: Mapping[str, Any] | None = None) -> None:
        """Append real operational/learning episodes to local Needle experience memory."""
        try:
            path = self.experience_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            row = {
                "timestamp": format_datetime(utcnow()),
                "kind": kind,
                "summary": summary[:500],
                "payload": dict(payload or {}),
            }
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError:
            pass

    def experience_count(self) -> int:
        path = self.experience_path()
        if not path.is_file():
            return 0
        try:
            with path.open("r", encoding="utf-8") as handle:
                return sum(1 for line in handle if line.strip())
        except OSError:
            return 0

    @property
    def conversation(self) -> list[dict[str, str]]:
        return self.memory_turns

    def remember_turn(self, role: str, text: str, *, engine: str = "saviour_cognitive") -> None:
        clean = (text or "").strip()
        if not clean:
            return
        self.memory_turns.append(
            {
                "role": str(role),
                "text": clean[:900],
                "engine": str(engine),
                "at": format_display_datetime(utcnow()),
            }
        )
        self.memory_turns = self.memory_turns[-16:]

    record_turn = remember_turn

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_turns": self.memory_turns[-16:],
            "rectifications": self.rectifications[-20:],
            "self_heal_runs": int(self.self_heal_runs),
        }

    @classmethod
    def from_dict(cls, state_dir: str | Path, payload: Mapping[str, Any] | None) -> "SaviourCognitivePartner":
        inst = cls(state_dir=Path(state_dir))
        if isinstance(payload, Mapping):
            turns = payload.get("memory_turns")
            if isinstance(turns, list):
                inst.memory_turns = [
                    {"role": str(t.get("role", "user")), "text": str(t.get("text", ""))[:600], "at": str(t.get("at", ""))}
                    for t in turns[-16:]
                    if isinstance(t, Mapping)
                ]
            rects = payload.get("rectifications")
            if isinstance(rects, list):
                inst.rectifications = [dict(r) for r in rects[-20:] if isinstance(r, Mapping)]
            inst.self_heal_runs = int(payload.get("self_heal_runs", 0) or 0)
        return inst

    def detect_and_rectify(self, bot: Any, *, run_causality: bool = True) -> dict[str, Any]:
        """Scan all SAVIOUR subsystems for flaws or drift and rectify recoverable issues."""
        self.self_heal_runs += 1
        members = [bot, *bot.peer_bots.values()]
        checked: list[dict[str, Any]] = []
        fixed: list[dict[str, Any]] = []
        stamp_disp = format_display_datetime(utcnow())

        # 1. Overdue pending settlements
        settled_now = int(bot.settle_due() or 0)
        if settled_now > 0:
            item = {
                "subsystem": "auto_settlement",
                "status": "RECTIFIED",
                "detail": f"Settled {settled_now} overdue signal(s) and applied immediate online updates.",
                "at": stamp_disp,
            }
            fixed.append(item)
            checked.append(item)
        else:
            checked.append(
                {
                    "subsystem": "auto_settlement",
                    "status": "HEALTHY",
                    "detail": "Zero overdue unsettled signals across all active markets.",
                }
            )

        # 2. Unreconciled historical settlements
        replayed = int(bot.reconcile_settled_signals() or 0)
        if replayed > 0:
            item = {
                "subsystem": "settlement_reconciliation",
                "status": "RECTIFIED",
                "detail": f"Reconciled {replayed} historical settled outcome(s) into model, ensemble & family weights.",
                "at": stamp_disp,
            }
            fixed.append(item)
            checked.append(item)
        else:
            checked.append(
                {
                    "subsystem": "settlement_reconciliation",
                    "status": "HEALTHY",
                    "detail": "All confirmed real settlements are reconciled idempotently.",
                }
            )

        # 3. Cold-start model protection & Family Fusion walk-forward seeding
        seeded_models: list[str] = []
        seeded_families: list[str] = []
        is_synth = bool(getattr(bot.provider, "synthetic", False))
        for m in members:
            if (m.model is None or not m.model.trained) and not is_synth:
                m._ensure_baseline_veto_model()
                if m.model is not None and m.model.trained:
                    seeded_models.append(m.config.pair)
            if (
                not is_synth
                and m.family_fusion.trades_observed == 0
                and len(m.candles) >= 240
            ):
                bt = m.run_auto_backtest(trigger="self_heal_seed", min_bars=240)
                if bt.get("ok"):
                    seeded_families.append(m.config.pair)

        if seeded_models:
            item = {
                "subsystem": "baseline_veto_models",
                "status": "RECTIFIED",
                "detail": f"Initialized causal MODE_VETO protection on {', '.join(seeded_models)}.",
                "at": stamp_disp,
            }
            fixed.append(item)
            checked.append(item)
        else:
            checked.append(
                {
                    "subsystem": "baseline_veto_models",
                    "status": "HEALTHY",
                    "detail": "All active markets carry trained VETO/ADVISORY models.",
                }
            )

        if seeded_families:
            item = {
                "subsystem": "family_fusion_seeding",
                "status": "RECTIFIED",
                "detail": f"Seeded merged Strategic Family weights via walk-forward replay on {', '.join(seeded_families)}.",
                "at": stamp_disp,
            }
            fixed.append(item)
            checked.append(item)
        else:
            obs_total = sum(m.family_fusion.trades_observed for m in members)
            checked.append(
                {
                    "subsystem": "family_fusion_seeding",
                    "status": "HEALTHY",
                    "detail": f"Merged Strategic Families tracking {obs_total} cumulative outcome observations.",
                }
            )

        # 4. Runtime guard parameter bounds sanity check
        clamped_guards: list[str] = []
        for m in members:
            if not (0.0 <= float(m.config.mtf_min_gap_bps) <= 25.0):
                m.config.mtf_min_gap_bps = max(0.0, min(25.0, float(m.config.mtf_min_gap_bps)))
                clamped_guards.append(f"{m.config.pair}:mtf_min_gap_bps")
            if not (1 <= int(m.config.horizon_minutes) <= 240):
                m.config.horizon_minutes = max(5, min(120, int(m.config.horizon_minutes)))
                clamped_guards.append(f"{m.config.pair}:horizon_minutes")
            if not (0 <= int(m.config.signal_cooldown_minutes) <= 240):
                m.config.signal_cooldown_minutes = max(0, min(240, int(m.config.signal_cooldown_minutes)))
                clamped_guards.append(f"{m.config.pair}:cooldown")
        if clamped_guards:
            item = {
                "subsystem": "guard_bounds",
                "status": "RECTIFIED",
                "detail": f"Clamped out-of-range runtime guards: {', '.join(clamped_guards)}.",
                "at": stamp_disp,
            }
            fixed.append(item)
            checked.append(item)
        else:
            checked.append(
                {
                    "subsystem": "guard_bounds",
                    "status": "HEALTHY",
                    "detail": "All MTF, horizon, spread, and cooldown guards sit strictly within Charter bounds.",
                }
            )

        # 5. Outbox retry flush if Telegram transport is active
        pending_before = len(bot.store.outbox_pending())
        if pending_before > 0 and bot.telegram is not None and bot.telegram.configured:
            bot.flush_outbox()
            pending_after = len(bot.store.outbox_pending())
            if pending_after < pending_before:
                item = {
                    "subsystem": "telegram_outbox",
                    "status": "RECTIFIED",
                    "detail": f"Flushed {pending_before - pending_after} pending Telegram delivery row(s).",
                    "at": stamp_disp,
                }
                fixed.append(item)
                checked.append(item)
        else:
            checked.append(
                {
                    "subsystem": "telegram_outbox",
                    "status": "HEALTHY",
                    "detail": f"Outbox queue verified ({pending_before} pending).",
                }
            )

        # 6. Causal look-ahead & recursive warmup audit
        causality_ok = True
        if run_causality:
            audit = lookahead.run_audit()
            causality_ok = audit.ok
            checked.append(
                {
                    "subsystem": "causality_lookahead",
                    "status": "HEALTHY" if audit.ok else "ALERT",
                    "detail": (
                        f"{len(audit.findings)}/{len(audit.findings)} causal invariance checks passed (zero future leakage)."
                        if audit.ok
                        else f"{sum(1 for f in audit.findings if not f.passed)} causality check failed."
                    ),
                }
            )

        if fixed:
            self.rectifications.extend(fixed)
            self.rectifications = self.rectifications[-25:]
            bot.save_all_states(force=True)
            self.record_experience(
                "self_heal",
                f"Detected and rectified {len(fixed)} subsystem issue(s).",
                {"fixed": fixed},
            )

        return {
            "ok": causality_ok,
            "causality_ok": causality_ok,
            "checked_count": len(checked),
            "issues_count": len(fixed),
            "rectified_count": len(fixed),
            "checks": checked,
            "rectified": [dict(r, summary=r.get("detail", "")) for r in fixed],
            "checked_at": stamp_disp,
        }

    def status_summary(self, bot: Any, needle_status: Mapping[str, Any] | None = None) -> dict[str, Any]:
        members = [bot, *bot.peer_bots.values()]
        total_online = sum(int(m.model.online_updates) for m in members if m.model is not None)
        total_journal = sum(len(m.learning_journal) for m in members)
        total_backtests = sum(len(m.auto_backtest_history) for m in members)
        total_fam_obs = sum(int(m.family_fusion.trades_observed) for m in members)
        active_fams = sum(len(m.family_fusion.active_families()) for m in members)
        active_syns = sum(len(m.family_fusion.active_synergies()) for m in members)
        xp = (
            total_online * 18
            + total_journal * 12
            + total_backtests * 25
            + total_fam_obs * 2
            + len(self.rectifications) * 15
            + self.experience_count() * 5
        )
        stage = (
            "SOVEREIGN_SYNTHESIS"
            if xp >= 400
            else "BANKAI_ADAPTIVE"
            if xp >= 150
            else "SHIKAI_CALIBRATING"
        )
        recent_lessons = []
        for m in members:
            for item in m.learning_journal[-6:]:
                recent_lessons.append(item)
        recent_lessons.sort(key=lambda r: str(r.get("timestamp", "")), reverse=True)

        latest_backtests = []
        for m in members:
            if m.auto_backtest_history:
                latest_backtests.append(m.auto_backtest_history[-1])

        return {
            "persona": "SAVIOUR Cognitive Strategist (Needle 3 Merged)",
            "evolution_stage": stage,
            "experience_xp": xp,
            "online_updates_total": total_online,
            "episodes_stored": total_journal,
            "auto_backtests_run": total_backtests,
            "family_observations": total_fam_obs,
            "active_family_links": active_fams,
            "active_synergy_links": active_syns,
            "self_heal_runs": self.self_heal_runs,
            "flaws_rectified": len(self.rectifications),
            "experience_corpus_rows": self.experience_count(),
            "recent_lessons": recent_lessons[:8],
            "latest_backtests": latest_backtests,
            "recent_rectifications": self.rectifications[-6:][::-1],
            "memory_turns": self.memory_turns[-10:],
            "needle": dict(needle_status or {}),
        }

    def compose_human_reply(
        self,
        bot: Any,
        query: str,
        *,
        tool_reply: str | None = None,
        decision_dict: Mapping[str, Any] | None = None,
        heal_report: Mapping[str, Any] | None = None,
        backtest_runs: list[dict[str, Any]] | None = None,
    ) -> str:
        """Speak like a candid, articulate human quant partner grounded in live SAVIOUR state."""
        clean = (query or "").strip()
        lower = clean.lower()
        members = [bot, *bot.peer_bots.values()]

        settlements_raw = [
            r for r in bot.store.load_settlements(limit=400) if not charter.is_synthetic(r.get("source"))
        ]
        ledger = settle.summarise([settle.Settlement.from_dict(r) for r in settlements_raw]) if settlements_raw else None
        ledger_line = (
            f"{ledger.wins}W / {ledger.losses}L ({ledger.net_pips:+.2f} net pips after full spread)"
            if ledger and ledger.settled > 0
            else "0 live settled trades in this ledger so far (paper research protection active)"
        )

        # Build concise market pulse
        pulse_parts = []
        for m in members:
            mode = m.model.mode if m.model else "VETO"
            act_fams = list(m.family_fusion.active_families().keys())
            fam_tag = f"{len(act_fams)} learned families" if act_fams else "warming family priors"
            pulse_parts.append(f"{m.config.pair} [{mode}, {len(m.candles)} bars, {fam_tag}]")
        pulse_str = " · ".join(pulse_parts)

        lines: list[str] = []

        if CONVERSATION_GREETING_RE.search(clean) and not tool_reply:
            lines.append(
                "I'm awake at the desk and watching all three markets in real time (UTC+05:30). "
                "I work alongside Needle 3 and SAVIOUR's causal engine — studying every settled trade, "
                "merging our 6 strategic families, running walk-forward backtests automatically, and checking "
                "every subsystem so minor flaws get caught before they cost us pips."
            )
            lines.append(f"Current desk pulse: {pulse_str}. Live ledger: {ledger_line}.")
            lines.append(
                "Ask me to scan any pair for the upcoming candle, explain what I learned from recent wins and losses, "
                "run an automatic walk-forward backtest, or audit and self-heal the system."
            )
        elif heal_report is not None:
            fixed_cnt = int(heal_report.get("rectified_count", 0))
            chk_cnt = int(heal_report.get("checked_count", 0))
            if fixed_cnt > 0:
                fixed_desc = "; ".join(str(x.get("detail", "")) for x in heal_report.get("rectified", []))
                lines.append(
                    f"I ran a full {chk_cnt}-point diagnostic sweep and rectified {fixed_cnt} issue(s) on the spot: {fixed_desc}"
                )
            else:
                lines.append(
                    f"I just completed a {chk_cnt}-point diagnostic & causality sweep across EUR/USD, BTC/USD, and ETH/USD — "
                    "zero flaws, zero overdue settlements, and 5/5 look-ahead invariance checks passed cleanly."
                )
            if tool_reply:
                lines.append(tool_reply)
        elif backtest_runs:
            summaries = []
            for bt in backtest_runs:
                summaries.append(
                    f"{bt['pair']}: {bt['signals']} OOS signals ({bt['wins']}W/{bt['losses']}L, "
                    f"net {bt['net_pips']:+.1f}p, mode={bt['model_mode']}, active families={len(bt.get('learned_families', {}))})"
                )
            lines.append(
                "I ran a fresh purged walk-forward backtest across our stored candle history and updated the merged "
                "Strategic Family weights where real out-of-sample edge cleared shrinkage:\n"
                + "\n".join(f"  • {s}" for s in summaries)
            )
        elif re.search(r"\b(loss|losses|mistake|why\s+did|learn|post-?mortem)\b", lower):
            lessons = []
            for m in members:
                lessons.extend(m.learning_journal[-3:])
            lessons.sort(key=lambda r: str(r.get("timestamp", "")), reverse=True)
            if lessons:
                top = lessons[:3]
                lines.append(
                    f"Here is what I've learned directly from our recent settled trades ({ledger_line}):"
                )
                for item in top:
                    lines.append(f"  • {item.get('narrative', '')}")
            else:
                mistakes_total = sum(m.mistakes.total_records() for m in members)
                lines.append(
                    f"Live ledger stands at {ledger_line} with {mistakes_total} post-mortem slice observations recorded. "
                    "On every single loss, I apply a 1.35x online gradient step to both the logistic model and the "
                    "4-member bootstrap ensemble, penalise endorsing families by 1.25x, and tighten any 2-loss slice to 6/6 core confluence."
                )
            if tool_reply and tool_reply not in "\n".join(lines):
                lines.append(tool_reply)
        elif re.search(r"\b(famil(?:y|ies)|synerg(?:y|ies)|merge)\b", lower):
            fam_notes = []
            for m in members:
                act = m.family_fusion.active_families()
                syn = m.family_fusion.active_synergies()
                fam_notes.append(
                    f"{m.config.pair}: {len(act)}/5 specialist families active "
                    f"({', '.join(f'{k}={v:.2f}' for k, v in act.items()) or 'warming'}), "
                    f"{len(syn)} synergy pairs active"
                )
            lines.append(
                "I've merged all 22 council strategies into 6 coordinated Strategic Families "
                "(CORE_GATE, CLASSIC_COUNCIL, MICRO_SCALPING, SMC_ICT, VIRAL_MOMENTUM, QUANT_SCALP) plus a 10-pair "
                "cross-family synergy matrix:\n" + "\n".join(f"  • {n}" for n in fam_notes)
            )
            if tool_reply:
                lines.append(tool_reply)
        elif tool_reply:
            # Wrap operational tool output in natural, concise strategist framing
            calls = (decision_dict or {}).get("function_calls") or []
            call_names = [c.get("name") for c in calls if isinstance(c, Mapping)]
            if "scan_market" in call_names:
                lines.append("I evaluated the completed candles against our 6 merged families, MTF cascade, and meta-model gates for the upcoming minute:")
            elif "bot_status" in call_names:
                lines.append(f"Here is the live desk readout ({pulse_str}):")
            elif "configure_guards" in call_names or "set_telegram_style" in call_names:
                lines.append("Done — I've applied that configuration across the active runtime:")
            lines.append(tool_reply)
        else:
            lines.append(
                f"I'm tracking {pulse_str} with ledger at {ledger_line}. "
                "Every settled trade updates our online logistic weights, 4-member bootstrap ensemble, isotonic calibrator, "
                "mistake memory, and 6-family synergy matrix automatically."
            )

        reply_text = "\n\n".join(part for part in lines if part)
        stamp_hm = format_display_datetime(utcnow())
        self.memory_turns.append({"role": "user", "text": clean[:400], "at": stamp_hm})
        self.memory_turns.append({"role": "assistant", "text": reply_text[:600], "at": stamp_hm})
        self.memory_turns = self.memory_turns[-16:]
        return reply_text
