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
from .mistakes import MistakeMemory
from .regime import regime_allows
from .session import session_allows

# Learning gates
MIN_SEEN_PER_PROTOTYPE = 30       # real settled outcomes before a voice is heard
PER_PROTOTYPE_WEIGHT_CAP = 0.30   # no single prototype may dominate
TOTAL_PROTOTYPE_MASS_CAP = 0.60   # prototypes together stay below the core
# Net opposing mass that blocks a core signal. It sits above the per-prototype
# cap, so no single learned rule can veto on its own — they must agree.
VETO_MASS = 0.35
SHRINKAGE = 40.0                  # net / (seen + SHRINKAGE) keeps small samples humble


@dataclass
class CouncilWeights:
    """Per-prototype learned weights. The core is always weight 1.0."""

    net: dict[str, float] = field(default_factory=dict)
    seen: dict[str, int] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)
    version: int = 1

    def __post_init__(self) -> None:
        for name in strategies.PROTOTYPES:
            self.net.setdefault(name, 0.0)
            self.seen.setdefault(name, 0)
            self.weights.setdefault(name, 0.0)

    def observe(self, votes: dict[str, str], winning_side: str | None) -> None:
        """Credit or blame every prototype that spoke on a settled real outcome.

        ``winning_side`` is the side the market actually took (BUY/SELL) or None
        when the bar was unresolved or tied; nothing is learned from those.
        """
        if winning_side not in ("BUY", "SELL"):
            return
        for name, side in votes.items():
            if name not in self.net:
                continue
            self.seen[name] = int(self.seen.get(name, 0)) + 1
            self.net[name] = float(self.net.get(name, 0.0)) + (1.0 if side == winning_side else -1.0)
        self.recompute()

    def recompute(self) -> None:
        raw: dict[str, float] = {}
        for name in strategies.PROTOTYPES:
            seen = int(self.seen.get(name, 0))
            if seen < MIN_SEEN_PER_PROTOTYPE:
                raw[name] = 0.0
                continue
            score = float(self.net.get(name, 0.0)) / (seen + SHRINKAGE)
            raw[name] = max(0.0, min(PER_PROTOTYPE_WEIGHT_CAP, score))

        total = sum(raw.values())
        if total > TOTAL_PROTOTYPE_MASS_CAP:
            scale = TOTAL_PROTOTYPE_MASS_CAP / total
            raw = {name: value * scale for name, value in raw.items()}
        self.weights = raw
        self.version += 1

    def active(self) -> dict[str, float]:
        return {name: value for name, value in self.weights.items() if value > 0.0}

    def to_dict(self) -> dict:
        return {
            "net": dict(self.net),
            "seen": dict(self.seen),
            "weights": dict(self.weights),
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, payload: dict | None) -> "CouncilWeights":
        if not payload:
            return cls()
        result = cls()
        result.net = {str(k): float(v) for k, v in (payload.get("net") or {}).items()}
        result.seen = {str(k): int(v) for k, v in (payload.get("seen") or {}).items()}
        result.weights = {str(k): float(v) for k, v in (payload.get("weights") or {}).items()}
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

    @property
    def actionable(self) -> bool:
        return self.side in ("BUY", "SELL")

    def to_dict(self) -> dict:
        return {
            "side": self.side,
            "reason": self.reason,
            "core_score": self.core_score,
            "votes": [vote.to_dict() for vote in self.votes],
            "support": round(self.support, 4),
            "opposition": round(self.opposition, 4),
            "vetoed": self.vetoed,
        }


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
        candle, previous_close, features, probability_up, min_checks=min_checks
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

    if not core.actionable:
        return CouncilResult(
            "NO SIGNAL",
            core.reason,
            core.score,
            tuple(votes),
            round(support, 4),
            round(opposition, 4),
            False,
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
            )

    spoken = ", ".join(vote.name for vote in votes if vote.heard and vote.side == core.side)
    suffix = f" Supporting learned prototypes: {spoken}." if spoken else " No learned prototype is active yet."
    return CouncilResult(
        core.side,
        core.reason + suffix,
        core.score,
        tuple(votes),
        round(support, 4),
        round(opposition, 4),
        False,
    )


def spoken_votes(votes: Sequence[Vote]) -> dict[str, str]:
    """Map prototype -> side for the prototypes that actually had an opinion."""
    return {vote.name: vote.side for vote in votes if vote.side in ("BUY", "SELL")}
