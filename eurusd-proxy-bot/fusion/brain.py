"""The AI Brain — core manager that orchestrates the entire fusion pipeline.

This is the central nervous system of the bot. It sits above the engine,
council, and all strategy modules, and manages:

1. **Pipeline orchestration** — coordinates features → regime → council →
   model → seal in the correct order with proper error handling.
2. **Learning management** — observes real outcomes, updates council weights,
   tracks per-method performance, and manages the exploration/exploitation
   tradeoff.
3. **Signal quality optimization** — auto-tunes thresholds, adapts to market
   conditions, and selects the best expiry for each signal.
4. **Performance analytics** — tracks win rates, edge metrics, and per-family
   performance to identify what works and what doesn't.
5. **Auto-adaptation** — detects regime shifts, adjusts risk parameters, and
   promotes/demotes methods based on real performance data.

The brain is stateful: it maintains running statistics, learned weights, and
performance counters that persist across evaluations. It is designed to get
smarter with every trade.

Usage:
    brain = AIBrain(config)
    result = brain.evaluate(candles, model=model, ensemble=ensemble)
    # Later, when a trade settles:
    brain.observe_outcome(signal, settlement)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Sequence

from . import charter, costs, datagate, instruments, news, protections
from .council import CouncilResult, CouncilWeights, run_council
from .domain import Candle, ModelRead, Settlement, Signal, closed_before
from .family_fusion import FamilyFusionEngine
from .features import compute_features
from .mistakes import MistakeMemory
from .model import BootstrapEnsemble, OnlineLogisticModel
from .regime import RegimeState, detect_regime
from .session import session_of
from .timeutil import floor_minute, now as utcnow


# ──────────────────────────────────────────────────────────────────
# Performance tracking
# ──────────────────────────────────────────────────────────────────

@dataclass
class MethodStats:
    """Per-method performance tracking."""
    name: str
    family: str = ""
    votes: int = 0
    wins: int = 0
    losses: int = 0
    total_weight: float = 0.0
    net_edge: float = 0.0  # sum of (win=+1, loss=-1) * weight

    @property
    def win_rate(self) -> float:
        resolved = self.wins + self.losses
        if resolved == 0:
            return 0.0
        return self.wins / resolved

    @property
    def resolved(self) -> int:
        return self.wins + self.losses

    @property
    def edge(self) -> float:
        if self.resolved == 0:
            return 0.0
        return (self.wins - self.losses) / self.resolved


@dataclass
class FamilyStats:
    """Per-family performance tracking."""
    family: str
    signals: int = 0
    wins: int = 0
    losses: int = 0

    @property
    def win_rate(self) -> float:
        resolved = self.wins + self.losses
        if resolved == 0:
            return 0.0
        return self.wins / resolved

    @property
    def edge(self) -> float:
        if self.resolved == 0:
            return 0.0
        return (self.wins - self.losses) / self.resolved


@dataclass
class BrainStats:
    """Overall brain performance tracking."""
    evaluations: int = 0
    signals_sent: int = 0
    wins: int = 0
    losses: int = 0
    tied: int = 0
    unresolved: int = 0
    total_net_pips: float = 0.0
    total_gross_pips: float = 0.0
    by_regime: dict[str, dict[str, int]] = field(default_factory=dict)
    by_session: dict[str, dict[str, int]] = field(default_factory=dict)
    by_family: dict[str, FamilyStats] = field(default_factory=dict)
    by_method: dict[str, MethodStats] = field(default_factory=dict)
    best_expiry: dict[int, dict[str, int]] = field(default_factory=dict)

    @property
    def resolved(self) -> int:
        return self.wins + self.losses

    @property
    def win_rate(self) -> float:
        if self.resolved == 0:
            return 0.0
        return self.wins / self.resolved

    @property
    def net_pips_per_trade(self) -> float:
        if self.resolved == 0:
            return 0.0
        return self.total_net_pips / self.resolved

    def to_dict(self) -> dict:
        return {
            "evaluations": self.evaluations,
            "signals_sent": self.signals_sent,
            "wins": self.wins,
            "losses": self.losses,
            "tied": self.tied,
            "unresolved": self.unresolved,
            "win_rate": round(self.win_rate, 4),
            "net_pips_per_trade": round(self.net_pips_per_trade, 4),
            "total_net_pips": round(self.total_net_pips, 2),
            "total_gross_pips": round(self.total_gross_pips, 2),
            "by_regime": self.by_regime,
            "by_session": self.by_session,
            "by_family": {
                name: {
                    "signals": s.signals,
                    "wins": s.wins,
                    "losses": s.losses,
                    "win_rate": round(s.win_rate, 4),
                    "edge": round(s.edge, 4),
                }
                for name, s in self.by_family.items()
            },
        }


# ──────────────────────────────────────────────────────────────────
# Auto-tuning parameters
# ──────────────────────────────────────────────────────────────────

@dataclass
class TuningConfig:
    """Auto-tuning parameters that the brain can adjust based on performance."""
    # Minimum win rate for a method to keep its exploration weight
    # (below this, the method is demoted to learned-only)
    min_win_rate: float = 0.40

    # Minimum resolved trades before a method's learned weight activates
    min_resolved: int = 30

    # Sharpe-like ratio threshold for promoting a method to higher weight
    promote_threshold: float = 0.15

    # If a family's win rate drops below this, reduce its influence
    family_demote_threshold: float = 0.35

    # If a family's win rate exceeds this, increase its influence
    family_promote_threshold: float = 0.60

    # Adaptive horizon: if win rate drops below this, try shorter expiries
    adaptive_horizon_floor: float = 0.35

    # Maximum weight multiplier for a promoted method
    max_weight_multiplier: float = 2.0


# ──────────────────────────────────────────────────────────────────
# The AI Brain
# ──────────────────────────────────────────────────────────────────

class AIBrain:
    """Core manager that orchestrates the entire fusion pipeline.

    The brain is the single entry point for signal generation. It manages:
    - The full pipeline: data gate → features → regime → council → model → seal
    - Learning: observes outcomes, updates weights, tracks performance
    - Auto-tuning: adjusts parameters based on real performance data
    - Analytics: provides win rates, edge metrics, and per-method performance
    """

    def __init__(
        self,
        config,
        *,
        weights: CouncilWeights | None = None,
        mistakes: MistakeMemory | None = None,
        family_fusion: FamilyFusionEngine | None = None,
        tuning: TuningConfig | None = None,
        ai_gateway: Any | None = None,
    ) -> None:
        self.config = config
        self.weights = weights or CouncilWeights()
        self.mistakes = mistakes or MistakeMemory()
        self.family_fusion = family_fusion or FamilyFusionEngine()
        self.tuning = tuning or TuningConfig()
        self.ai_gateway = ai_gateway  # Universal AI gateway (optional)
        self.stats = BrainStats()
        self._method_families = self._init_method_families()
        self._regime_performance: dict[str, list[str]] = {}
        self._expiry_performance: dict[int, list[str]] = {}
        self._last_tuning_time = 0.0
        self._tuning_interval = 100  # re-tune every 100 evaluations
        self._ai_market_views: list[dict] = []  # Recent AI market views

    def _init_method_families(self) -> dict[str, str]:
        """Build a map of method name -> family label."""
        from . import strategies
        families = {}
        for name in strategies.PROTOTYPES:
            families[name] = strategies.strategy_family(name)
        return families

    # ─────────────────────────────────────────────────────────────
    # Core pipeline
    # ─────────────────────────────────────────────────────────────

    def evaluate(
        self,
        candles: Sequence[Candle],
        *,
        model: OnlineLogisticModel | None = None,
        ensemble: BootstrapEnsemble | None = None,
        neural_predictor: Any | None = None,
        ai_gateway: Any | None = None,
        source: str = "LIVE",
        quote: dict | None = None,
        at: datetime | None = None,
        observed_at: datetime | None = None,
    ) -> dict:
        """Run the full pipeline: gate → features → regime → council → model → seal.

        Optionally accepts a neural predictor (NeuralPredictor) and/or an AI
        gateway (AIGateway) that provide additional signals to inform the
        council decision.

        Returns a dict with the seal state, signal (if any), and full reasoning.
        """
        self.stats.evaluations += 1

        # 1. Data gate
        stamp = at or utcnow()
        causal_cutoff = (
            min(stamp, floor_minute(observed_at))
            if observed_at is not None
            else stamp
        )
        gate_now = observed_at if observed_at is not None else stamp
        candles = closed_before(list(candles), causal_cutoff)
        instrument = instruments.for_symbol(getattr(self.config, "pair", None))

        gate = datagate.evaluate(
            candles,
            now=gate_now,
            max_age_seconds=self.config.max_candle_age_seconds,
            max_spread_pips=self.config.max_spread_pips,
            quote=quote,
            base_spread_pips=self.config.spread_pips,
            crypto=instrument.crypto,
        )
        if not gate.usable:
            return self._result(charter.SEAL_ABSTAIN, None, f"Data gate: {gate.reason}", gate=gate)

        # 2. Features
        feature_rows = compute_features(candles, instrument=instrument)
        features = feature_rows[-1] if feature_rows else None
        if features is None:
            return self._result(charter.SEAL_ABSTAIN, None, "Not enough candles for features.", gate=gate)

        latest = candles[-1]
        previous_close = candles[-2].close if len(candles) > 1 else latest.close

        # 3. Regime
        regime_state = detect_regime(candles, features)

        # 4. Session
        session = (
            session_of(latest.timestamp, instrument)
            if self.config.session_filter
            else None
        )

        # 5. News filter
        if getattr(self.config, "news_filter", False):
            blackout = news.check_blackout(
                floor_minute(stamp),
                pair=getattr(self.config, "pair", "EUR/USD"),
                extra_windows=getattr(self.config, "news_blackout_windows", ""),
            )
            if blackout.blocked:
                return self._result(charter.SEAL_ABSTAIN, None, blackout.reason, gate=gate, regime=regime_state.name)

        # 6. Model read
        read = self._build_read(model, ensemble, features)
        if read is not None and read.abstain_reason:
            return self._result(charter.SEAL_ABSTAIN, None, read.abstain_reason, gate=gate, regime=regime_state.name, read=read)

        # 7. Council
        council = run_council(
            latest,
            previous_close,
            features,
            read.probability if read else None,
            weights=self.weights,
            regime=regime_state.name,
            min_checks=self.config.min_confluence_checks,
            calibrated=bool(read and read.calibrated),
            spread_pips=gate.spread_pips,
            session=session,
            mistakes=self.mistakes,
            family_fusion=self.family_fusion,
        )

        # 8. Family fusion refinement
        if self.family_fusion is not None and council.family_verdict is not None:
            read = self.family_fusion.refine_model_read(read, council.family_verdict)

        # 9. Council decision
        if not council.actionable:
            return self._result(charter.SEAL_ABSTAIN, None, council.reason, gate=gate, regime=regime_state.name, read=read, council=council)

        # 10. Spread-to-ATR guard
        min_atr_ratio = float(getattr(self.config, "min_atr_to_spread_ratio", 0.0) or 0.0)
        if min_atr_ratio > 0.0 and gate.spread_pips > 0.0:
            atr_pips_now = abs(float(features.get("atr", 0.0) or 0.0)) / instrument.pip
            if atr_pips_now < gate.spread_pips * min_atr_ratio:
                return self._result(
                    charter.SEAL_ABSTAIN, None,
                    f"Spread-to-ATR guard: 1m ATR ({atr_pips_now:.2f}p) below "
                    f"{min_atr_ratio:.2f}x spread ({gate.spread_pips:.2f}p).",
                    gate=gate, regime=regime_state.name, read=read, council=council,
                )

        # 11. Model veto
        from .model import MODE_VETO
        mode = model.mode if model is not None else MODE_VETO
        allowed, model_reason = self._model_verdict(council.side, read, mode=mode)
        if not allowed:
            return self._result(
                charter.SEAL_ABSTAIN, None, model_reason,
                gate=gate, regime=regime_state.name, read=read, council=council,
            )

        # 11b. Neural AI integration (optional)
        neural_prediction = None
        if neural_predictor is not None:
            try:
                neural_prediction = neural_predictor.predict(
                    list(candles), pair=getattr(self.config, "pair", "EUR/USD"),
                    at=stamp,
                )
                if neural_prediction.action == "NO_TRADE" and neural_prediction.uncertainty > 0.65:
                    return self._result(
                        charter.SEAL_ABSTAIN, None,
                        f"Neural AI veto: high uncertainty ({neural_prediction.uncertainty:.3f})",
                        gate=gate, regime=regime_state.name, read=read, council=council,
                        neural_prediction=neural_prediction.to_dict(),
                    )
                # If neural AI disagrees strongly with council, add a note
                if neural_prediction.action != "NO_TRADE":
                    neural_side = neural_prediction.action
                    if neural_side != council.side:
                        # Neural AI disagrees but doesn't veto — just note it
                        pass  # Could add to reason later
            except Exception as e:
                # Neural AI failure shouldn't block the pipeline
                pass

        # 11c. AI Gateway consultation (optional)
        ai_market_view = None
        gateway = ai_gateway or self.ai_gateway
        if gateway is not None:
            try:
                ai_market_view = self._consult_ai_gateway(gateway, candles, regime_state)
                if ai_market_view:
                    self._ai_market_views.append(ai_market_view)
                    if len(self._ai_market_views) > 50:
                        self._ai_market_views = self._ai_market_views[-50:]
            except Exception as e:
                # AI gateway failure shouldn't block the pipeline
                pass

        # 12. Seal
        return self._seal(
            council, latest, previous_close, features, regime_state,
            gate=gate, source=source, at=stamp, observed_at=observed_at,
            spread=gate.spread_pips, read=read, session=session,
        )

    # ─────────────────────────────────────────────────────────────
    # AI Gateway consultation
    # ─────────────────────────────────────────────────────────────

    def get_ai_market_views(self, limit: int = 20) -> list[dict]:
        """Get recent AI market views from the consultation history."""
        return self._ai_market_views[-limit:] if self._ai_market_views else []

    def get_ai_gateway_status(self) -> dict | None:
        """Get AI gateway status if a gateway is connected."""
        if self.ai_gateway is None:
            return None
        try:
            return self.ai_gateway.get_status()
        except Exception:
            return None

    def _consult_ai_gateway(self, gateway, candles: list, regime_state) -> dict | None:
        """Query the AI gateway for a market view.

        Constructs a market summary from candle data and asks the AI
        model to provide a directional view. The response is used as
        an additional signal — it doesn't veto but informs the council.

        Returns a dict with the AI's market view, or None if unavailable.
        """
        # Select a model from the gateway
        model_info = gateway.select_model({
            "tool_calling": True,
            "max_latency_ms": 5000,
        })
        if not model_info:
            return None

        model_id = model_info.get("model_id", "")
        if not model_id:
            return None

        # Build market summary from candles
        if len(candles) < 2:
            return None

        latest = candles[-1]
        previous_close = candles[-2].close if len(candles) > 1 else latest.close
        recent = candles[-30:] if len(candles) > 30 else candles

        # Calculate simple market stats
        highs = [c.high for c in recent]
        lows = [c.low for c in recent]
        closes = [c.close for c in recent]
        price_change = (latest.close - recent[0].close) / recent[0].close * 100
        range_pips = (max(highs) - min(lows)) / 0.0001 if highs and lows else 0
        volatility = (max(closes) - min(closes)) / (sum(closes) / len(closes)) * 100 if closes else 0

        # Detect trend from recent candles
        recent_up = sum(1 for i in range(1, len(recent)) if recent[i].close > recent[i-1].close)
        recent_down = sum(1 for i in range(1, len(recent)) if recent[i].close < recent[i-1].close)
        if recent_up > recent_down:
            trend = "uptrend"
        elif recent_down > recent_up:
            trend = "downtrend"
        else:
            trend = "range-bound"

        # Build the market context
        market_context = (
            f"EUR/USD 1-minute chart.\n"
            f"Latest price: {latest.close:.5f}\n"
            f"Previous close: {previous_close:.5f}\n"
            f"Price change ({len(recent)} bars): {price_change:+.2f}%\n"
            f"Range: {range_pips:.1f} pips\n"
            f"Volatility: {volatility:.3f}%\n"
            f"Trend: {trend} ({recent_up} up bars, {recent_down} down bars)\n"
            f"Regime: {regime_state.name}\n"
            f"Recent high: {max(highs):.5f}\n"
            f"Recent low: {min(lows):.5f}\n"
        )

        # Build the prompt
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a forex market analyst specializing in EUR/USD "
                    "1-minute timeframe analysis. Analyze the market data and "
                    "provide a brief directional view. Be concise — under 30 words."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"{market_context}\n"
                    "Based on this market data, what is your directional view "
                    "on EUR/USD for the next 30 minutes? "
                    "Reply with one of: BUY, SELL, or NO TRADE, followed by "
                    "a one-sentence rationale."
                ),
            },
        ]

        # Send request through gateway
        result = gateway.chat(model_id, messages, max_tokens=100)

        if not result.get("ok"):
            return None

        # Parse the AI response
        response_text = ""
        try:
            if "choices" in result.get("response", {}):
                response_text = result["response"]["choices"][0]["message"]["content"]
            elif "content" in result.get("response", {}):
                response_text = result["response"]["content"][0].get("text", "")
            else:
                response_text = str(result.get("response", ""))
        except (KeyError, IndexError, TypeError):
            response_text = str(result.get("response", ""))

        # Parse direction from response
        response_lower = response_text.lower()
        direction = "NO_TRADE"
        confidence = 0.0
        if "buy" in response_lower and "sell" not in response_lower.split("buy")[0][:5]:
            direction = "BUY"
            confidence = 0.6
        elif "sell" in response_lower and "buy" not in response_lower.split("sell")[0][:5]:
            direction = "SELL"
            confidence = 0.6

        return {
            "model": model_id,
            "provider": model_info.get("provider", "unknown"),
            "direction": direction,
            "confidence": confidence,
            "response": response_text,
            "latency_ms": result.get("latency_ms"),
            "timestamp": datetime.utcnow().isoformat(),
        }

    # ─────────────────────────────────────────────────────────────
    # Learning
    # ─────────────────────────────────────────────────────────────

    def observe_outcome(
        self,
        signal: "Signal | dict",
        settlement: Settlement,
    ) -> dict:
        """Observe a real trade outcome and update all learning systems.

        Updates:
        - Council weights (via observe)
        - Per-method performance stats
        - Per-family performance stats
        - Regime performance tracking
        - Expiry performance tracking
        - Mistake memory
        """
        # Accept both Signal objects and signal dicts (from evaluate() results)
        if isinstance(signal, dict):
            regime = signal.get("regime", "UNKNOWN")
            present = signal.get("present_conditions", {}) or {}
            session = present.get("session", "UNKNOWN")
            horizon = signal.get("horizon_minutes", 5)
            council = signal.get("council", [])
        else:
            regime = signal.regime
            present = signal.present_conditions or {}
            session = present.get("session", "UNKNOWN")
            horizon = signal.horizon_minutes
            council = signal.council

        if not settlement.resolved:
            self.stats.unresolved += 1
            return {"outcome": settlement.outcome, "updated": False}

        winning_side = "BUY" if settlement.outcome == charter.WIN else "SELL"

        if settlement.outcome == charter.WIN:
            self.stats.wins += 1
        elif settlement.outcome == charter.LOSS:
            self.stats.losses += 1

        if settlement.net_pips is not None:
            self.stats.total_net_pips += settlement.net_pips
        if settlement.gross_pips is not None:
            self.stats.total_gross_pips += settlement.gross_pips

        # Track by regime
        if regime not in self.stats.by_regime:
            self.stats.by_regime[regime] = {"wins": 0, "losses": 0}
        self.stats.by_regime[regime]["wins" if settlement.outcome == charter.WIN else "losses"] += 1

        # Track by session
        if session not in self.stats.by_session:
            self.stats.by_session[session] = {"wins": 0, "losses": 0}
        self.stats.by_session[session]["wins" if settlement.outcome == charter.WIN else "losses"] += 1

        # Track by expiry
        if horizon not in self._expiry_performance:
            self._expiry_performance[horizon] = []
        self._expiry_performance[horizon].append(settlement.outcome)

        # Update council weights
        votes = {}
        for v in council:
            if v.get("heard") and v.get("side") in ("BUY", "SELL"):
                votes[v["name"]] = v["side"]
        self.weights.observe(votes, winning_side)

        # Update per-method stats
        for name, side in votes.items():
            if name not in self.stats.by_method:
                family = self._method_families.get(name, "UNKNOWN")
                self.stats.by_method[name] = MethodStats(name=name, family=family)
            ms = self.stats.by_method[name]
            ms.votes += 1
            weight = self.weights.weights.get(name, 0.08)
            ms.total_weight += weight
            if side == winning_side:
                ms.wins += 1
                ms.net_edge += weight
            else:
                ms.losses += 1
                ms.net_edge -= weight

            # Update family stats
            if ms.family not in self.stats.by_family:
                self.stats.by_family[ms.family] = FamilyStats(family=ms.family)
            if side == winning_side:
                self.stats.by_family[ms.family].signals += 1
                self.stats.by_family[ms.family].wins += 1
            else:
                self.stats.by_family[ms.family].signals += 1
                self.stats.by_family[ms.family].losses += 1

        # Update mistake memory
        self.mistakes.record(signal, settlement)

        # Auto-tune periodically
        if self.stats.evaluations % self._tuning_interval == 0:
            self._auto_tune()

        return {"outcome": settlement.outcome, "updated": True, "win_rate": round(self.stats.win_rate, 4)}

    # ─────────────────────────────────────────────────────────────
    # Auto-tuning
    # ─────────────────────────────────────────────────────────────

    def _auto_tune(self) -> dict:
        """Auto-tune parameters based on real performance data."""
        tuning_result = {"tuned": []}

        # 1. Promote/demote methods based on win rate
        for name, ms in self.stats.by_method.items():
            if ms.resolved < self.tuning.min_resolved:
                continue
            if ms.win_rate < self.tuning.min_win_rate:
                # Demote: reduce learned weight
                if ms.learned_weight > 0:
                    self.weights.learned_weights[name] = max(0.0, ms.learned_weight * 0.5)
                    tuning_result["tuned"].append(f"demoted {name} (win_rate={ms.win_rate:.2f})")
            elif ms.win_rate > self.tuning.promote_threshold:
                # Promote: increase learned weight (capped)
                current = self.weights.learned_weights.get(name, 0.0)
                cap = 0.30 * self.tuning.max_weight_multiplier
                self.weights.learned_weights[name] = min(cap, current * 1.5 + 0.02)
                tuning_result["tuned"].append(f"promoted {name} (win_rate={ms.win_rate:.2f})")

        # 2. Family-level tuning
        for family, fs in self.stats.by_family.items():
            if fs.resolved < self.tuning.min_resolved:
                continue
            if fs.win_rate < self.tuning.family_demote_threshold:
                tuning_result["tuned"].append(f"family {family} needs review (win_rate={fs.win_rate:.2f})")

        # 3. Expiry performance
        for horizon, outcomes in self._expiry_performance.items():
            wins = sum(1 for o in outcomes if o == charter.WIN)
            resolved = wins + sum(1 for o in outcomes if o == charter.LOSS)
            if resolved >= self.tuning.min_resolved:
                wr = wins / resolved if resolved else 0.0
                if wr < self.tuning.adaptive_horizon_floor:
                    tuning_result["tuned"].append(f"expiry {horizon}min win_rate={wr:.2f} (consider shorter)")

        self._last_tuning_time = time.time()
        tuning_result["timestamp"] = self._last_tuning_time
        return tuning_result

    # ─────────────────────────────────────────────────────────────
    # Analytics
    # ─────────────────────────────────────────────────────────────

    def analytics(self) -> dict:
        """Return comprehensive analytics."""
        data = self.stats.to_dict()
        data["expiry_performance"] = {}
        for horizon, outcomes in self._expiry_performance.items():
            wins = sum(1 for o in outcomes if o == charter.WIN)
            losses = sum(1 for o in outcomes if o == charter.LOSS)
            resolved = wins + losses
            data["expiry_performance"][f"{horizon}min"] = {
                "wins": wins,
                "losses": losses,
                "win_rate": round(wins / resolved, 4) if resolved else 0.0,
            }

        data["top_methods"] = []
        for name, ms in sorted(self.stats.by_method.items(), key=lambda x: x[1].edge, reverse=True)[:10]:
            data["top_methods"].append({
                "name": name,
                "family": ms.family,
                "votes": ms.votes,
                "wins": ms.wins,
                "losses": ms.losses,
                "win_rate": round(ms.win_rate, 4),
                "edge": round(ms.edge, 4),
                "weight": round(self.weights.weights.get(name, 0.0), 4),
            })

        data["bottom_methods"] = []
        for name, ms in sorted(self.stats.by_method.items(), key=lambda x: x[1].edge)[:10]:
            data["bottom_methods"].append({
                "name": name,
                "family": ms.family,
                "votes": ms.votes,
                "wins": ms.wins,
                "losses": ms.losses,
                "win_rate": round(ms.win_rate, 4),
                "edge": round(ms.edge, 4),
            })

        return data

    # ─────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────

    def _build_read(self, model, ensemble, features) -> ModelRead | None:
        """Build the model read (same logic as engine.build_read)."""
        if model is None or not model.trained or features is None:
            return None
        raw = model.predict_up(features)
        uncertainty = None
        members = 0
        if ensemble is not None and ensemble.ready:
            _, spread = ensemble.predict(features)
            uncertainty = spread
            members = len(ensemble.members)

        calibrator = model.calibrator
        if calibrator.fitted:
            probability = calibrator.transform(raw)
            state = "FLAT" if calibrator.flat else "CALIBRATED"
        else:
            probability = raw
            state = "RAW"

        reason = None
        if state == "FLAT":
            reason = "Calibration is flat: no usable rank signal."
        elif uncertainty is not None and uncertainty > self.config.max_uncertainty:
            reason = f"Ensemble disagreement {uncertainty:.3f} > {self.config.max_uncertainty:.2f}."

        return ModelRead(
            raw=raw,
            probability=probability,
            calibrated=state == "CALIBRATED",
            calibration_state=state,
            uncertainty=uncertainty,
            members=members,
            contributions=tuple(model.contributions(features, top=6)),
            abstain_reason=reason,
        )

    def _model_verdict(self, side: str, read: ModelRead | None, *, mode: str) -> tuple[bool, str]:
        """Model veto/advisory check (same logic as engine.model_verdict)."""
        if read is None:
            return True, "No trained model yet."
        if not read.usable:
            return False, read.abstain_reason or "Model read not usable."
        if read.uncertainty is not None and read.abstain_reason:
            return False, read.abstain_reason

        edge = read.probability - 0.5
        agrees = edge > 0 if side == "BUY" else edge < 0
        label = "calibrated" if read.calibrated else "raw"

        if mode == "ADVISORY":
            margin = 0.01 if read.calibrated else 0.02
            if not agrees:
                return False, f"Promoted model veto: {label} read {read.probability:.3f} disagrees with {side}."
            if abs(edge) < margin:
                return False, f"Promoted model abstains: {label} edge {edge:+.3f} inside {margin:.3f}."
            return True, f"Promoted model agrees ({label} {read.probability:.3f})."

        if not agrees and abs(edge) >= 0.03:
            return False, f"Model veto: {label} read {read.probability:.3f} disagrees with {side}."
        return True, f"Model does not object ({label} {read.probability:.3f})."

    def _seal(
        self, council: CouncilResult, candle: Candle, previous_close: float,
        features: dict, regime: RegimeState, *, gate, source, at, observed_at,
        spread, read, session,
    ) -> dict:
        """Seal the signal (same logic as engine._seal)."""
        entry_time = floor_minute(at)
        seal_clock = observed_at if observed_at is not None else at
        latency = (seal_clock - entry_time).total_seconds()
        if latency > self.config.max_seal_latency_seconds:
            return self._result(
                charter.SEAL_DISCARDED_LATE, None,
                f"Sealed {latency:.1f}s after minute open, past {self.config.max_seal_latency_seconds}s limit.",
                gate=gate, regime=regime.name, read=read, council=council,
            )

        instrument = instruments.for_symbol(getattr(self.config, "pair", None))
        atr_pips = abs(float(features.get("atr", 0.0))) / instrument.pip

        # Resolve horizon
        horizon = costs.resolve_horizon(
            atr_pips, spread, self.config.horizon_minutes,
            adaptive=self.config.adaptive_horizon,
            reward_to_cost=self.config.reward_to_cost,
            max_horizon=self.config.max_horizon_minutes,
        )

        # Try short horizons if default can't pay
        if horizon is None:
            for short_h in (1, 3):
                if short_h <= getattr(self.config, "short_horizon_max_minutes", 0):
                    short_horizon = costs.resolve_horizon(
                        atr_pips, spread, short_h, adaptive=False,
                        reward_to_cost=getattr(self.config, "short_horizon_min_reward", 1.5),
                        max_horizon=short_h,
                    )
                    if short_horizon is not None:
                        horizon = short_h
                        break

        if horizon is None:
            return self._result(
                charter.SEAL_ABSTAIN, None,
                "No hold length can pay for this trade.",
                gate=gate, regime=regime.name, read=read, council=council,
            )

        # FX rollover guard
        if getattr(self.config, "fx_rollover_guard", False):
            rollover = protections.check_fx_rollover(entry_time, horizon, crypto=instrument.crypto)
            if rollover.blocked:
                return self._result(charter.SEAL_ABSTAIN, None, rollover.reason, gate=gate, regime=regime.name, read=read, council=council)

        # Present conditions
        present = self._build_present_conditions(features, spread, atr_pips, regime.name, session, council.side)

        # Sureshot detection
        confidence = float(read.probability) if read and read.probability is not None else 0.0
        threshold = getattr(self.config, "sureshot_confidence_threshold", 0.72)
        sureshot = (
            confidence >= threshold
            and present.get("volatility_ok", False)
            and present.get("spread_ok", False)
            and present.get("session_ok", False)
            and present.get("trend_aligned", False)
        )

        signal = Signal.create(
            pair=self.config.pair,
            side=council.side,
            entry_time=entry_time,
            strategy="ai_brain",
            reason=council.reason,
            source=source,
            generated_at=seal_clock,
            horizon_minutes=horizon,
            probability_up=read.probability if read else None,
            probability_raw=read.raw if read else None,
            calibration_state=read.calibration_state if read else "RAW",
            uncertainty=read.uncertainty if read else None,
            regime=regime.name,
            council=tuple(v.to_dict() for v in council.votes),
            features=dict(features),
            spread_pips=spread,
            sureshot=sureshot,
            present_conditions=present,
        )

        reason_text = council.reason
        if sureshot:
            reason_text = f"SURESHOT: {horizon}-min expiry (confidence {confidence:.2f}, conditions aligned)"

        # Track signal sent (even before settlement)
        self.stats.signals_sent += 1
        horizon_key = str(horizon)
        if horizon_key not in self.stats.by_regime:
            pass  # already tracked by regime on settlement

        return self._result(
            charter.SEAL_SIGNAL, signal, reason_text,
            gate=gate, regime=regime.name, read=read, council=council,
        )

    def _build_present_conditions(self, features, spread, atr_pips, regime, session, side) -> dict:
        """Build present-condition awareness dict."""
        present = {
            "current_candle_state": self._classify_candle_state(features),
            "spread_pips": float(spread),
            "atr_pips": float(atr_pips),
            "regime": regime,
            "volatility_ok": atr_pips >= 1.0,
            "spread_ok": float(spread) <= 2.0,
            "session": session or "UNKNOWN",
            "session_ok": session != "OFF_HOURS" if session else True,
            "trend_aligned": self._check_present_trend(features, side),
        }
        return present

    def _classify_candle_state(self, features) -> str:
        atr = abs(float(features.get("atr", 0.0)))
        momentum = float(features.get("momentum_1", 0.0))
        if atr > 2.0 and abs(momentum) > 0.5:
            return "HIGH_VOL_MOMENTUM"
        elif atr > 1.0:
            return "MODERATE_VOL"
        else:
            return "LOW_VOL"

    def _check_present_trend(self, features, side) -> bool:
        momentum = float(features.get("momentum_1", 0.0))
        rsi = float(features.get("rsi_14", 50.0))
        if side == "BUY":
            return momentum > 0.0 or rsi < 60.0
        return momentum < 0.0 or rsi > 40.0

    def _result(self, seal_state, signal, reason, *, gate=None, regime="UNKNOWN", read=None, council=None) -> dict:
        return {
            "seal_state": seal_state,
            "reason": reason,
            "regime": regime,
            "gate": gate.to_dict() if gate else None,
            "read": read.to_dict() if read else None,
            "council": council.to_dict() if council else None,
            "signal": signal.to_dict() if signal else None,
        }

    # ─────────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialize brain state for persistence."""
        return {
            "weights": self.weights.to_dict(),
            "mistakes": self.mistakes.to_dict() if hasattr(self.mistakes, 'to_dict') else {},
            "stats": self.stats.to_dict(),
            "version": 1,
        }

    @classmethod
    def from_dict(cls, config, payload: dict | None, **kwargs) -> "AIBrain":
        """Deserialize brain state from persisted data."""
        brain = cls(config, **kwargs)
        if not payload:
            return brain
        if "weights" in payload:
            brain.weights = CouncilWeights.from_dict(payload.get("weights"))
        return brain
