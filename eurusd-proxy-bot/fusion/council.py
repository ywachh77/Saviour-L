"""The strategy council: one authoritative core gate plus learned prototypes.

How the power is split, and why:

* the core 6-check gate decides whether there is a signal at all;
* prototypes are silent (weight 0) until learning has seen enough real,
  cost-charged outcomes to trust them;
* a learned prototype can block a core signal when the opposition mass passes
  ``VETO_MASS`` — it can never create one.

So the council gets smarter with data without ever letting an unproven rule
outvote the gate it was built around.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from . import charter, strategies
from .domain import Candle, Decision
from .family_fusion import FamilyFusionEngine, FamilyFusionVerdict
from .mistakes import MistakeMemory
from .regime import regime_allows
from .session import session_allows

# Learning gates
MIN_SEEN_PER_PROTOTYPE = 30       # real settled outcomes before learned weight kicks in
# Exploration weight: every prototype gets this before learning has seen enough
# data. It is SEPARATE from learned weight so recompute() never zeroes it out.
# This ensures the council always has voting members while learning is in progress.
EXPLORATION_WEIGHT = 0.08
# Learned weight cap: a prototype's learned weight cannot exceed this.
PER_PROTOTYPE_WEIGHT_CAP = 0.30   # no single prototype may dominate
# Total learned mass cap: learned weights together stay below the core.
TOTAL_LEARNED_MASS_CAP = 0.60
# Total effective mass cap (exploration + learned): keeps prototypes from
# overwhelming the core gate while unlearned.
TOTAL_PROTOTYPE_MASS_CAP = 1.20
# Net opposing mass that blocks a core signal. With exploration weight of 0.08
# per prototype, this requires at least 5 prototypes opposing the core.
VETO_MASS = 0.40
SHRINKAGE = 40.0                  # net / (seen + SHRINKAGE) keeps small samples humble
# Strategic consensus override: minimum total effective weight for strategic
# methods to override a core abstention. With ~117 prototypes at 0.08 each,
# this requires at least 4 methods agreeing in one direction.
CONSENSUS_THRESHOLD = 0.50


@dataclass
class CouncilWeights:
    """Per-prototype weights: exploration (always active) + learned (data-driven).

    The core gate is always weight 1.0. Prototypes have two weight components:
    - exploration_weight: always EXPLORATION_WEIGHT, never zeroed. Ensures the
      council always has voting members while learning is in progress.
    - learned_weight: adjusted by recompute() based on real outcomes. Starts at
      0.0, only becomes non-zero after MIN_SEEN_PER_PROTOTYPE real trades.
    Total weight = exploration_weight + learned_weight.
    """

    net: dict[str, float] = field(default_factory=dict)
    seen: dict[str, int] = field(default_factory=dict)
    learned_weights: dict[str, float] = field(default_factory=dict)
    version: int = 1

    def __post_init__(self) -> None:
        for name in strategies.PROTOTYPES:
            self.net.setdefault(name, 0.0)
            self.seen.setdefault(name, 0)
            self.learned_weights.setdefault(name, 0.0)

    @property
    def weights(self) -> dict[str, float]:
        """Total effective weight: exploration + learned."""
        return {
            name: EXPLORATION_WEIGHT + float(self.learned_weights.get(name, 0.0))
            for name in strategies.PROTOTYPES
        }

    def observe(self, votes: dict[str, str], winning_side: str | None) -> None:
        """Credit or blame every prototype that spoke on a settled real outcome."""
        if winning_side not in ("BUY", "SELL"):
            return
        for name, side in votes.items():
            if name not in self.net:
                continue
            self.seen[name] = int(self.seen.get(name, 0)) + 1
            self.net[name] = float(self.net.get(name, 0.0)) + (1.0 if side == winning_side else -1.0)
        self.recompute()

    def recompute(self) -> None:
        """Recompute learned weights. Exploration weight is never touched."""
        raw: dict[str, float] = {}
        for name in strategies.PROTOTYPES:
            seen = int(self.seen.get(name, 0))
            if seen < MIN_SEEN_PER_PROTOTYPE:
                raw[name] = 0.0
                continue
            score = float(self.net.get(name, 0.0)) / (seen + SHRINKAGE)
            raw[name] = max(0.0, min(PER_PROTOTYPE_WEIGHT_CAP, score))

        # Cap total learned mass
        total_learned = sum(raw.values())
        if total_learned > TOTAL_LEARNED_MASS_CAP:
            scale = TOTAL_LEARNED_MASS_CAP / total_learned
            raw = {name: value * scale for name, value in raw.items()}
        self.learned_weights = raw
        self.version += 1

    def active(self) -> dict[str, float]:
        return {name: value for name, value in self.weights.items() if value > 0.0}

    def to_dict(self) -> dict:
        return {
            "net": dict(self.net),
            "seen": dict(self.seen),
            "learned_weights": dict(self.learned_weights),
            "exploration_weight": EXPLORATION_WEIGHT,
            "weights": self.weights,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, payload: dict | None) -> "CouncilWeights":
        if not payload:
            return cls()
        result = cls()
        for k, v in (payload.get("net") or {}).items():
            result.net[str(k)] = float(v)
        for k, v in (payload.get("seen") or {}).items():
            result.seen[str(k)] = int(v)
        # Support both old "weights" key and new "learned_weights" key
        if "learned_weights" in payload:
            for k, v in (payload.get("learned_weights") or {}).items():
                result.learned_weights[str(k)] = float(v)
        else:
            for k, v in (payload.get("weights") or {}).items():
                result.learned_weights[str(k)] = max(0.0, float(v) - EXPLORATION_WEIGHT)
        result.version = int(payload.get("version", 1))
        return result


@dataclass(frozen=True, slots=True)
class Vote:
    name: str
    side: str
    weight: float
    reason: str
    heard: bool

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "side": self.side,
            "weight": round(self.weight, 4),
            "heard": self.heard,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class CouncilResult:
    side: str
    reason: str
    core_score: int
    votes: tuple[Vote, ...]
    support: float
    opposition: float
    vetoed: bool
    family_verdict: FamilyFusionVerdict | None = None

    @property
    def actionable(self) -> bool:
        return self.side in ("BUY", "SELL")

    def to_dict(self) -> dict:
        out = {
            "side": self.side,
            "reason": self.reason,
            "core_score": self.core_score,
            "votes": [vote.to_dict() for vote in self.votes],
            "support": round(self.support, 4),
            "opposition": round(self.opposition, 4),
            "vetoed": self.vetoed,
        }
        if self.family_verdict is not None:
            out["family_fusion"] = self.family_verdict.to_dict()
        return out


def run_council(
    candle: Candle,
    previous_close: float,
    features: dict[str, float] | None,
    probability_up: float | None,
    *,
    weights: CouncilWeights | None = None,
    regime: str = "UNKNOWN",
    min_checks: int = 5,
    calibrated: bool = False,
    spread_pips: float = charter.DEFAULT_SPREAD_PIPS,
    session: str | None = None,
    mistakes: MistakeMemory | None = None,
    family_fusion: FamilyFusionEngine | None = None,
) -> CouncilResult:
    """Run every member and fold them into one honest decision.

    ``spread_pips`` is the cost the scalping members must clear before they are
    allowed an opinion. It is the live spread when the quote supplied one.

    ``session`` narrows the field by time of day. ``None`` means no session
    filtering, which is the behaviour everything before this existed with.

    ``mistakes`` carries the post-mortem memory. When supplied, a setup the bot
    has already paid to learn is refused rather than re-tried.
    """
    weights = weights or CouncilWeights()
    core: Decision = strategies.REGISTRY[strategies.CORE](
        candle, previous_close, features, probability_up, min_checks=min_checks, regime=regime
    )

    votes: list[Vote] = []
    support = 0.0
    opposition = 0.0

    for name in strategies.PROTOTYPES:
        weight = float(weights.weights.get(name, 0.0))
        # A member must clear three gates to be heard: it needs a learned
        # weight, the volatility regime must suit it, and the session must suit
        # it. The session gate is about which KIND of rule fits the hour, which
        # is a conditional effect the model's smooth hour term cannot express.
        heard = (
            weight > 0.0
            and regime_allows(regime, name)
            and (session is None or session_allows(session, name))
        )
        decision: Decision = strategies.REGISTRY[name](
            candle,
            previous_close,
            features,
            probability_up,
            calibrated=calibrated,
            spread_pips=spread_pips,
        )
        if not heard:
            # Still recorded for learning bookkeeping, but it carries no weight.
            votes.append(Vote(name, decision.side, 0.0, decision.reason, False))
            continue
        votes.append(Vote(name, decision.side, round(weight, 4), decision.reason, True))
        if not decision.actionable:
            continue
        if decision.side == core.side:
            support += weight
        elif core.actionable:
            opposition += weight

    fam_engine = family_fusion or FamilyFusionEngine()
    member_sides = {v.name: v.side for v in votes if v.side in ("BUY", "SELL")}
    fam_verdict = fam_engine.evaluate_votes(
        core.side,
        member_sides,
        regime=regime,
        session=session,
    )

    if not core.actionable:
        # Strategic consensus override: if enough strategic methods agree
        # on a direction with positive weight, allow the signal.
        strategic_votes = [v for v in votes if v.heard and v.weight > 0.0]
        buy_mass = sum(v.weight for v in strategic_votes if v.side == "BUY")
        sell_mass = sum(v.weight for v in strategic_votes if v.side == "SELL")
        consensus_threshold = CONSENSUS_THRESHOLD  # minimum total weight for override

        if buy_mass >= consensus_threshold and buy_mass > sell_mass:
            supporters = ", ".join(
                v.name for v in strategic_votes
                if v.side == "BUY" and v.weight > 0.0
            )
            return CouncilResult(
                "BUY",
                f"Strategic consensus override: core abstained ({core.score}/6) but "
                f"{len([v for v in strategic_votes if v.side == 'BUY'])} strategic methods "
                f"voted BUY (mass {buy_mass:.2f}). Supporters: {supporters}.",
                core.score,
                tuple(votes),
                round(buy_mass, 4),
                round(sell_mass, 4),
                False,
                fam_verdict,
            )

        if sell_mass >= consensus_threshold and sell_mass > buy_mass:
            supporters = ", ".join(
                v.name for v in strategic_votes
                if v.side == "SELL" and v.weight > 0.0
            )
            return CouncilResult(
                "SELL",
                f"Strategic consensus override: core abstained ({core.score}/6) but "
                f"{len([v for v in strategic_votes if v.side == 'SELL'])} strategic methods "
                f"voted SELL (mass {sell_mass:.2f}). Supporters: {supporters}.",
                core.score,
                tuple(votes),
                round(buy_mass, 4),
                round(sell_mass, 4),
                False,
                fam_verdict,
            )

        return CouncilResult(
            "NO SIGNAL",
            core.reason,
            core.score,
            tuple(votes),
            round(support, 4),
            round(opposition, 4),
            False,
            fam_verdict,
        )

    if opposition - support > VETO_MASS:
        blockers = ", ".join(
            vote.name for vote in votes if vote.heard and vote.side not in ("NO SIGNAL", core.side)
        )
        return CouncilResult(
            "NO SIGNAL",
            f"Council veto: the core read was {core.side}, but learned prototypes "
            f"({blockers}) opposed it with net mass {opposition - support:.2f}.",
            core.score,
            tuple(votes),
            round(support, 4),
            round(opposition, 4),
            True,
            fam_verdict,
        )

    if fam_verdict.vetoed:
        return CouncilResult(
            "NO SIGNAL",
            fam_verdict.reason,
            core.score,
            tuple(votes),
            round(support, 4),
            round(opposition, 4),
            True,
            fam_verdict,
        )

    if mistakes is not None:
        key = mistakes.fingerprint(regime, session, core.side)
        refused, why = mistakes.refuses(key)
        if refused:
            return CouncilResult(
                "NO SIGNAL",
                why,
                core.score,
                tuple(votes),
                round(support, 4),
                round(opposition, 4),
                False,
                fam_verdict,
            )
        extra_checks = mistakes.extra_confluence_checks(key)
        required_checks = min(6, min_checks + extra_checks)
        if extra_checks > 0 and core.score < required_checks:
            streak = mistakes.loss_streak(key)
            return CouncilResult(
                "NO SIGNAL",
                f"Post-mortem tightening: setup {key} has {streak} consecutive losses; "
                f"requiring {required_checks}/6 core confluence (passed {core.score}/6).",
                core.score,
                tuple(votes),
                round(support, 4),
                round(opposition, 4),
                False,
                fam_verdict,
            )

    spoken = ", ".join(vote.name for vote in votes if vote.heard and vote.side == core.side)
    suffix = f" Supporting learned prototypes: {spoken}." if spoken else " No learned prototype is active yet."
    if fam_verdict.support_mass > 0.0 and fam_verdict.supporting_families:
        fam_txt = ", ".join(fam_verdict.supporting_families)
        suffix += f" Merged families ({fam_txt}): {fam_verdict.net_edge:+.2f}."
    return CouncilResult(
        core.side,
        core.reason + suffix,
        core.score,
        tuple(votes),
        round(support, 4),
        round(opposition, 4),
        False,
        fam_verdict,
    )


def spoken_votes(votes: Sequence[Vote]) -> dict[str, str]:
    """Map prototype -> side for the prototypes that actually had an opinion."""
    return {vote.name: vote.side for vote in votes if vote.side in ("BUY", "SELL")}
