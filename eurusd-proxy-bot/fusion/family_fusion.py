"""Strategic Family Merging & Cross-Family Synergy Engine for SAVIOUR.

Why merge strategic families instead of keeping 21 isolated rules?
------------------------------------------------------------------
Individual strategy prototypes fire sparsely on M1 candles, so waiting for 30
live signals per rule leaves specialist edge fragmented across small samples.
By pooling the 22 council members into 6 orthogonal Strategic Families:

1. ``CORE_GATE``         — 3-mode adaptive ``orbix_confluence`` anchor
2. ``CLASSIC_COUNCIL``   — Trend pullback, range fade, breakout, ML consensus, cadence
3. ``MICRO_SCALPING``    — 7 spread-cleared M1 microstructure scalpers
4. ``SMC_ICT``           — Liquidity sweep, FVG displacement, institutional order block
5. ``VIRAL_MOMENTUM``    — TTM Keltner squeeze, Supertrend flip, EMA ribbon pullback
6. ``QUANT_SCALP``       — Connors RSI-2, session VWAP +/-2 sigma, session box breakout

SAVIOUR learns at two coupled resolutions on every confirmed real outcome:
- **Hierarchical Family Shrinkage Weights** (fast statistical pooling across
  members of each family, conditioned on volatility regime);
- **Pairwise Cross-Family Synergy Matrix** (10 family pairs, rewarding setups
  where independent methodologies — e.g. ``SMC_ICT + QUANT_SCALP`` or
  ``CLASSIC_COUNCIL + VIRAL_MOMENTUM`` — confirm one another out of sample,
  and penalising losing coalitions 1.25x faster).

Strict Charter Invariants
-------------------------
- Cold-start weights begin at ``0.0`` until real outcomes are observed, so
  unproven priors never manufacture trades.
- Synthetic outcomes never touch family or synergy weights.
- Family fusion may refine probability within tight bounds (``+/-0.045``) and
  veto weak core reads when multi-family opposition exceeds ``FAMILY_VETO_MASS``;
  it never overrides the core gate to invent a trade out of thin air.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Mapping, Sequence

from . import strategies
from .domain import ModelRead
from .regime import regime_allows
from .session import session_allows

SPECIALIST_FAMILIES: tuple[str, ...] = (
    "CLASSIC_COUNCIL",
    "MICRO_SCALPING",
    "SMC_ICT",
    "VIRAL_MOMENTUM",
    "QUANT_SCALP",
)
ALL_FAMILIES: tuple[str, ...] = ("CORE_GATE", *SPECIALIST_FAMILIES)

FAMILY_METADATA: dict[str, dict[str, str]] = {
    "CORE_GATE": {
        "title": "Core Confluence Gate",
        "jp": "中核門",
        "focus": "3-mode adaptive EMA/MACD/RSI/ADX/body/wick primary execution gate",
    },
    "CLASSIC_COUNCIL": {
        "title": "Classic Institutional",
        "jp": "古典陣",
        "focus": "Trend pullback, range fade, 20-bar breakout, calibrated ML & cadence",
    },
    "MICRO_SCALPING": {
        "title": "Microstructure Scalping",
        "jp": "瞬刃陣",
        "focus": "7 spread-cleared M1 impulse, pivot, stochastic & micro-breakout rules",
    },
    "SMC_ICT": {
        "title": "Smart Money / ICT",
        "jp": "機関陣",
        "focus": "Liquidity sweep stop-hunts, fair-value-gap displacement & order blocks",
    },
    "VIRAL_MOMENTUM": {
        "title": "Volatility & Momentum",
        "jp": "疾風陣",
        "focus": "TTM Bollinger-inside-Keltner squeeze, Supertrend flip & EMA ribbon",
    },
    "QUANT_SCALP": {
        "title": "Quantitative Statistical",
        "jp": "数理陣",
        "focus": "Connors 2-period RSI, session VWAP +/-2 sigma & Asian box breakout",
    },
}

SYNERGY_PAIRS: tuple[str, ...] = tuple(
    f"{a}+{b}" for a, b in combinations(SPECIALIST_FAMILIES, 2)
)

# Learning & safety constants
MIN_SEEN_PER_FAMILY = 5
MIN_SEEN_PER_SYNERGY = 4
FAMILY_SHRINKAGE = 12.0
SYNERGY_SHRINKAGE = 10.0
PER_FAMILY_WEIGHT_CAP = 0.24
TOTAL_FAMILY_MASS_CAP = 0.72
PER_SYNERGY_BONUS_CAP = 0.06
TOTAL_SYNERGY_BONUS_CAP = 0.14
FAMILY_VETO_MASS = 0.30
MAX_PROBABILITY_DELTA = 0.045
LOSS_BLAME_MULTIPLIER = 1.25


def members_by_family() -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {fam: [] for fam in ALL_FAMILIES}
    for sid in (strategies.CORE, *strategies.PROTOTYPES):
        fam = strategies.strategy_family(sid)
        grouped.setdefault(fam, []).append(sid)
    return grouped


@dataclass(frozen=True, slots=True)
class FamilyVote:
    family: str
    side: str
    buy_count: int
    sell_count: int
    eligible_count: int
    conviction: float
    learned_weight: float
    effective_weight: float
    active_members: tuple[str, ...]

    @property
    def actionable(self) -> bool:
        return self.side in ("BUY", "SELL")

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "title": FAMILY_METADATA.get(self.family, {}).get("title", self.family),
            "jp": FAMILY_METADATA.get(self.family, {}).get("jp", ""),
            "side": self.side,
            "buy_count": self.buy_count,
            "sell_count": self.sell_count,
            "eligible_count": self.eligible_count,
            "conviction": round(self.conviction, 4),
            "learned_weight": round(self.learned_weight, 4),
            "effective_weight": round(self.effective_weight, 4),
            "active_members": list(self.active_members),
        }


@dataclass(frozen=True, slots=True)
class FamilyFusionVerdict:
    core_side: str
    family_votes: tuple[FamilyVote, ...]
    supporting_families: tuple[str, ...]
    opposing_families: tuple[str, ...]
    synergy_pairs: tuple[str, ...]
    support_mass: float
    opposition_mass: float
    synergy_bonus: float
    net_edge: float
    probability_delta: float
    vetoed: bool
    reason: str

    @property
    def net_score(self) -> float:
        return self.net_edge

    def to_dict(self) -> dict[str, Any]:
        return {
            "core_side": self.core_side,
            "family_votes": [fv.to_dict() for fv in self.family_votes],
            "supporting_families": list(self.supporting_families),
            "opposing_families": list(self.opposing_families),
            "synergy_pairs": list(self.synergy_pairs),
            "support_mass": round(self.support_mass, 4),
            "opposition_mass": round(self.opposition_mass, 4),
            "synergy_bonus": round(self.synergy_bonus, 4),
            "net_edge": round(self.net_edge, 4),
            "net_score": round(self.net_edge, 4),
            "probability_delta": round(self.probability_delta, 4),
            "vetoed": self.vetoed,
            "reason": self.reason,
        }


@dataclass
class FamilyFusionEngine:
    """Hierarchical family-pooling + cross-family synergy learner."""

    family_net: dict[str, float] = field(default_factory=dict)
    family_seen: dict[str, int] = field(default_factory=dict)
    family_ewma_acc: dict[str, float] = field(default_factory=dict)
    family_weights: dict[str, float] = field(default_factory=dict)
    regime_net: dict[str, float] = field(default_factory=dict)
    regime_seen: dict[str, int] = field(default_factory=dict)
    synergy_net: dict[str, float] = field(default_factory=dict)
    synergy_seen: dict[str, int] = field(default_factory=dict)
    synergy_weights: dict[str, float] = field(default_factory=dict)
    trades_observed: int = 0
    version: int = 1

    def __post_init__(self) -> None:
        for fam in ALL_FAMILIES:
            self.family_net.setdefault(fam, 0.0)
            self.family_seen.setdefault(fam, 0)
            self.family_ewma_acc.setdefault(fam, 0.5)
            self.family_weights.setdefault(fam, 0.0)
        for pair_key in SYNERGY_PAIRS:
            self.synergy_net.setdefault(pair_key, 0.0)
            self.synergy_seen.setdefault(pair_key, 0)
            self.synergy_weights.setdefault(pair_key, 0.0)

    def active_families(self) -> dict[str, float]:
        return {k: v for k, v in self.family_weights.items() if v > 0.0}

    def active_synergies(self) -> dict[str, float]:
        return {k: v for k, v in self.synergy_weights.items() if v > 0.0}

    def _regime_multiplier(self, regime: str, family: str) -> float:
        key = f"{regime or 'UNKNOWN'}|{family}"
        seen = int(self.regime_seen.get(key, 0))
        if seen < 4:
            return 1.0
        edge = float(self.regime_net.get(key, 0.0)) / (seen + 8.0)
        # Scale smoothly in [0.65, 1.30] based on regime-specific family edge.
        return max(0.65, min(1.30, 1.0 + 0.55 * edge))

    def recompute(self) -> None:
        raw_fam: dict[str, float] = {}
        for fam in SPECIALIST_FAMILIES:
            seen = int(self.family_seen.get(fam, 0))
            if seen < MIN_SEEN_PER_FAMILY:
                raw_fam[fam] = 0.0
                continue
            shrink_score = float(self.family_net.get(fam, 0.0)) / (seen + FAMILY_SHRINKAGE)
            ewma_tilt = max(-0.20, min(0.20, float(self.family_ewma_acc.get(fam, 0.5)) - 0.5))
            blended = 0.75 * shrink_score + 0.25 * ewma_tilt
            raw_fam[fam] = max(0.0, min(PER_FAMILY_WEIGHT_CAP, blended))

        total_fam = sum(raw_fam.values())
        if total_fam > TOTAL_FAMILY_MASS_CAP and total_fam > 0.0:
            scale = TOTAL_FAMILY_MASS_CAP / total_fam
            raw_fam = {k: v * scale for k, v in raw_fam.items()}
        c_seen = int(self.family_seen.get("CORE_GATE", 0))
        raw_fam["CORE_GATE"] = (
            max(0.0, min(0.35, float(self.family_net.get("CORE_GATE", 0.0)) / (c_seen + FAMILY_SHRINKAGE)))
            if c_seen >= MIN_SEEN_PER_FAMILY
            else 0.0
        )
        self.family_weights = raw_fam

        raw_syn: dict[str, float] = {}
        for pair_key in SYNERGY_PAIRS:
            seen = int(self.synergy_seen.get(pair_key, 0))
            if seen < MIN_SEEN_PER_SYNERGY:
                raw_syn[pair_key] = 0.0
                continue
            score = float(self.synergy_net.get(pair_key, 0.0)) / (seen + SYNERGY_SHRINKAGE)
            raw_syn[pair_key] = max(0.0, min(PER_SYNERGY_BONUS_CAP, 0.45 * score))

        total_syn = sum(raw_syn.values())
        if total_syn > TOTAL_SYNERGY_BONUS_CAP and total_syn > 0.0:
            scale = TOTAL_SYNERGY_BONUS_CAP / total_syn
            raw_syn = {k: v * scale for k, v in raw_syn.items()}
        self.synergy_weights = raw_syn
        self.version += 1

    def aggregate_member_votes(
        self,
        votes_by_member: Mapping[str, str],
        *,
        regime: str = "UNKNOWN",
        session: str | None = None,
    ) -> list[FamilyVote]:
        grouped = members_by_family()
        out: list[FamilyVote] = []
        for fam in SPECIALIST_FAMILIES:
            members = grouped.get(fam, [])
            eligible = [
                m
                for m in members
                if regime_allows(regime, m) and (session is None or session_allows(session, m))
            ]
            buy_members = [m for m in eligible if votes_by_member.get(m) == "BUY"]
            sell_members = [m for m in eligible if votes_by_member.get(m) == "SELL"]
            b_cnt = len(buy_members)
            s_cnt = len(sell_members)
            elig_cnt = len(eligible)

            if b_cnt > s_cnt:
                side = "BUY"
                active = tuple(buy_members)
                net_cnt = b_cnt - s_cnt
            elif s_cnt > b_cnt:
                side = "SELL"
                active = tuple(sell_members)
                net_cnt = s_cnt - b_cnt
            else:
                side = "NO SIGNAL"
                active = ()
                net_cnt = 0

            denom = max(1, min(elig_cnt, 3))
            conviction = min(1.0, net_cnt / denom) if side in ("BUY", "SELL") else 0.0
            base_w = float(self.family_weights.get(fam, 0.0))
            reg_mult = self._regime_multiplier(regime, fam)
            eff_w = base_w * conviction * reg_mult if side in ("BUY", "SELL") else 0.0

            out.append(
                FamilyVote(
                    family=fam,
                    side=side,
                    buy_count=b_cnt,
                    sell_count=s_cnt,
                    eligible_count=elig_cnt,
                    conviction=conviction,
                    learned_weight=base_w,
                    effective_weight=eff_w,
                    active_members=active,
                )
            )
        return out

    def evaluate_votes(
        self,
        core_side: str,
        votes_by_member: Mapping[str, str],
        *,
        regime: str = "UNKNOWN",
        session: str | None = None,
    ) -> FamilyFusionVerdict:
        f_votes = self.aggregate_member_votes(votes_by_member, regime=regime, session=session)
        supporting: list[str] = []
        opposing: list[str] = []
        support_mass = 0.0
        opposition_mass = 0.0

        # Also compute directional tilt for probability refinement (+ = BUY, - = SELL).
        buy_mass = 0.0
        sell_mass = 0.0

        by_fam_side: dict[str, str] = {}
        for fv in f_votes:
            if not fv.actionable:
                continue
            by_fam_side[fv.family] = fv.side
            if fv.side == "BUY":
                buy_mass += fv.effective_weight
            elif fv.side == "SELL":
                sell_mass += fv.effective_weight

            if core_side in ("BUY", "SELL"):
                if fv.side == core_side:
                    supporting.append(fv.family)
                    support_mass += fv.effective_weight
                else:
                    opposing.append(fv.family)
                    opposition_mass += fv.effective_weight

        active_synergies: list[str] = []
        synergy_support = 0.0
        synergy_oppose = 0.0
        synergy_dir = 0.0

        for pair_key in SYNERGY_PAIRS:
            fam_a, fam_b = pair_key.split("+", 1)
            side_a = by_fam_side.get(fam_a)
            side_b = by_fam_side.get(fam_b)
            if side_a in ("BUY", "SELL") and side_a == side_b:
                active_synergies.append(f"{pair_key}:{side_a}")
                syn_w = float(self.synergy_weights.get(pair_key, 0.0))
                if syn_w > 0.0:
                    synergy_dir += syn_w if side_a == "BUY" else -syn_w
                    if core_side in ("BUY", "SELL"):
                        if side_a == core_side:
                            synergy_support += syn_w
                        else:
                            synergy_oppose += syn_w

        total_support = support_mass + synergy_support
        total_opposition = opposition_mass + synergy_oppose
        synergy_bonus = synergy_support - synergy_oppose
        net_edge = total_support - total_opposition

        raw_dir_edge = (buy_mass - sell_mass) + synergy_dir
        prob_delta = max(-MAX_PROBABILITY_DELTA, min(MAX_PROBABILITY_DELTA, 0.09 * raw_dir_edge))

        vetoed = False
        if core_side in ("BUY", "SELL") and (total_opposition - total_support) > FAMILY_VETO_MASS:
            vetoed = True
            opp_names = ", ".join(opposing) if opposing else "specialist families"
            reason = (
                f"Merged family veto: core read was {core_side}, but learned strategic families "
                f"({opp_names}) opposed with net family mass {total_opposition - total_support:.2f}."
            )
        elif core_side in ("BUY", "SELL") and supporting:
            reason = (
                f"Merged families confirm {core_side}: {', '.join(supporting)} "
                f"(net mass {net_edge:+.2f})."
            )
        else:
            reason = "Merged family consensus neutral."

        return FamilyFusionVerdict(
            core_side=core_side,
            family_votes=tuple(f_votes),
            supporting_families=tuple(supporting),
            opposing_families=tuple(opposing),
            synergy_pairs=tuple(active_synergies),
            support_mass=total_support,
            opposition_mass=total_opposition,
            synergy_bonus=synergy_bonus,
            net_edge=net_edge,
            probability_delta=prob_delta,
            vetoed=vetoed,
            reason=reason,
        )

    def observe_outcome(
        self,
        votes_by_member: Mapping[str, str],
        winning_side: str | None,
        *,
        regime: str = "UNKNOWN",
        session: str | None = None,
        signal_side: str | None = None,
        was_loss: bool = False,
    ) -> dict[str, Any]:
        """Update family weights, regime affinity & pairwise synergies from one real outcome."""
        if winning_side not in ("BUY", "SELL"):
            return {"applied": False, "families_updated": []}

        f_votes = self.aggregate_member_votes(votes_by_member, regime=regime, session=session)
        updated_families: list[str] = []
        by_fam_side: dict[str, str] = {}

        for fv in f_votes:
            if not fv.actionable:
                continue
            fam = fv.family
            by_fam_side[fam] = fv.side
            correct = fv.side == winning_side
            # Penalise families that endorsed a losing live trade 1.25x faster
            # so weak setups get down-weighted immediately.
            penalty = (
                LOSS_BLAME_MULTIPLIER
                if (was_loss and signal_side in ("BUY", "SELL") and fv.side == signal_side and not correct)
                else 1.0
            )
            delta = (1.0 if correct else -penalty) * max(0.5, fv.conviction)

            self.family_seen[fam] = int(self.family_seen.get(fam, 0)) + 1
            self.family_net[fam] = float(self.family_net.get(fam, 0.0)) + delta
            prev_ewma = float(self.family_ewma_acc.get(fam, 0.5))
            alpha = 0.24 if was_loss else 0.18
            self.family_ewma_acc[fam] = (1.0 - alpha) * prev_ewma + alpha * (1.0 if correct else 0.0)

            reg_key = f"{regime or 'UNKNOWN'}|{fam}"
            self.regime_seen[reg_key] = int(self.regime_seen.get(reg_key, 0)) + 1
            self.regime_net[reg_key] = float(self.regime_net.get(reg_key, 0.0)) + (1.0 if correct else -penalty)
            updated_families.append(fam)

        updated_synergies: list[str] = []
        for pair_key in SYNERGY_PAIRS:
            fam_a, fam_b = pair_key.split("+", 1)
            side_a = by_fam_side.get(fam_a)
            side_b = by_fam_side.get(fam_b)
            if side_a in ("BUY", "SELL") and side_a == side_b:
                correct = side_a == winning_side
                penalty = LOSS_BLAME_MULTIPLIER if (was_loss and not correct) else 1.0
                self.synergy_seen[pair_key] = int(self.synergy_seen.get(pair_key, 0)) + 1
                self.synergy_net[pair_key] = float(self.synergy_net.get(pair_key, 0.0)) + (
                    1.0 if correct else -penalty
                )
                updated_synergies.append(pair_key)

        # Track CORE_GATE reliability alongside the 5 specialist families
        core_dir = str(votes_by_member.get(strategies.CORE, "")).upper()
        if core_dir in ("BUY", "SELL"):
            core_ok = core_dir == winning_side
            core_pen = LOSS_BLAME_MULTIPLIER if (was_loss and not core_ok) else 1.0
            self.family_seen["CORE_GATE"] = int(self.family_seen.get("CORE_GATE", 0)) + 1
            self.family_net["CORE_GATE"] = float(self.family_net.get("CORE_GATE", 0.0)) + (
                1.0 if core_ok else -core_pen
            )

        self.trades_observed += 1
        self.recompute()
        return {
            "applied": True,
            "families_updated": updated_families,
            "synergies_updated": updated_synergies,
            "active_family_weights": {k: round(v, 4) for k, v in self.active_families().items()},
        }

    def refine_model_read(
        self,
        read: ModelRead | None,
        verdict: FamilyFusionVerdict | None,
    ) -> ModelRead | None:
        """Blend calibrated ML probability with learned Strategic Family consensus."""
        if read is None or verdict is None or not read.usable or read.abstain_reason:
            return read
        if abs(verdict.probability_delta) < 1e-6:
            return read
        fused_prob = max(0.02, min(0.98, float(read.probability) + float(verdict.probability_delta)))
        return ModelRead(
            raw=read.raw,
            probability=fused_prob,
            calibrated=read.calibrated,
            calibration_state=read.calibration_state,
            uncertainty=read.uncertainty,
            members=read.members,
            contributions=read.contributions,
            abstain_reason=read.abstain_reason,
        )

    def summary(self) -> dict[str, Any]:
        grouped = members_by_family()
        families_out = []
        for fam in ALL_FAMILIES:
            meta = FAMILY_METADATA.get(fam, {})
            if fam == "CORE_GATE":
                families_out.append(
                    {
                        "family": fam,
                        "title": meta.get("title", fam),
                        "jp": meta.get("jp", ""),
                        "focus": meta.get("focus", ""),
                        "member_count": len(grouped.get(fam, [])),
                        "members": grouped.get(fam, []),
                        "seen": self.trades_observed,
                        "net": None,
                        "ewma_accuracy": None,
                        "weight": 1.0,
                        "role": "PRIMARY_GATE",
                    }
                )
            else:
                families_out.append(
                    {
                        "family": fam,
                        "title": meta.get("title", fam),
                        "jp": meta.get("jp", ""),
                        "focus": meta.get("focus", ""),
                        "member_count": len(grouped.get(fam, [])),
                        "members": grouped.get(fam, []),
                        "seen": int(self.family_seen.get(fam, 0)),
                        "net": round(float(self.family_net.get(fam, 0.0)), 3),
                        "ewma_accuracy": round(float(self.family_ewma_acc.get(fam, 0.5)) * 100.0, 1),
                        "weight": round(float(self.family_weights.get(fam, 0.0)), 4),
                        "role": "MERGED_SPECIALIST",
                    }
                )
        synergies_out = [
            {
                "pair": k,
                "seen": int(self.synergy_seen.get(k, 0)),
                "net": round(float(self.synergy_net.get(k, 0.0)), 3),
                "bonus_weight": round(float(self.synergy_weights.get(k, 0.0)), 4),
            }
            for k in SYNERGY_PAIRS
        ]
        synergies_out.sort(key=lambda r: (r["bonus_weight"], r["seen"]), reverse=True)
        return {
            "trades_observed": self.trades_observed,
            "version": self.version,
            "active_family_count": len(self.active_families()),
            "active_synergy_count": len(self.active_synergies()),
            "families": families_out,
            "top_synergies": synergies_out,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "family_net": {k: round(float(v), 5) for k, v in self.family_net.items()},
            "family_seen": {k: int(v) for k, v in self.family_seen.items()},
            "family_ewma_acc": {k: round(float(v), 5) for k, v in self.family_ewma_acc.items()},
            "family_weights": {k: round(float(v), 5) for k, v in self.family_weights.items()},
            "regime_net": {k: round(float(v), 5) for k, v in self.regime_net.items()},
            "regime_seen": {k: int(v) for k, v in self.regime_seen.items()},
            "synergy_net": {k: round(float(v), 5) for k, v in self.synergy_net.items()},
            "synergy_seen": {k: int(v) for k, v in self.synergy_seen.items()},
            "synergy_weights": {k: round(float(v), 5) for k, v in self.synergy_weights.items()},
            "trades_observed": int(self.trades_observed),
            "version": int(self.version),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "FamilyFusionEngine":
        inst = cls()
        if not isinstance(payload, Mapping):
            return inst
        for k, v in (payload.get("family_net") or {}).items():
            if k in SPECIALIST_FAMILIES:
                inst.family_net[str(k)] = float(v)
        for k, v in (payload.get("family_seen") or {}).items():
            if k in SPECIALIST_FAMILIES:
                inst.family_seen[str(k)] = int(v)
        for k, v in (payload.get("family_ewma_acc") or {}).items():
            if k in SPECIALIST_FAMILIES:
                inst.family_ewma_acc[str(k)] = float(v)
        for k, v in (payload.get("family_weights") or {}).items():
            if k in SPECIALIST_FAMILIES:
                inst.family_weights[str(k)] = float(v)
        for k, v in (payload.get("regime_net") or {}).items():
            inst.regime_net[str(k)] = float(v)
        for k, v in (payload.get("regime_seen") or {}).items():
            inst.regime_seen[str(k)] = int(v)
        for k, v in (payload.get("synergy_net") or {}).items():
            if k in SYNERGY_PAIRS:
                inst.synergy_net[str(k)] = float(v)
        for k, v in (payload.get("synergy_seen") or {}).items():
            if k in SYNERGY_PAIRS:
                inst.synergy_seen[str(k)] = int(v)
        for k, v in (payload.get("synergy_weights") or {}).items():
            if k in SYNERGY_PAIRS:
                inst.synergy_weights[str(k)] = float(v)
        inst.trades_observed = int(payload.get("trades_observed", 0) or 0)
        inst.version = int(payload.get("version", 1) or 1)
        return inst
