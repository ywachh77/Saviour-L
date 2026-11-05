"""One prediction engine, replacing a committee of 132.

Why this exists
---------------

The council is a vote: 132 rule sets each cast a BUY/SELL and the majority wins.
That design has three defects, all measured in ``scripts/direction_audit.py``:

1. **The vote double-counts.** ``breakout`` and ``quant_donchian_breakout``
   emitted an identical signal on all 484 bars they fired, in both directions.
   So did ``smc_ascending_wedge`` and ``scalp_breakout_retest``. Members that
   are the same rule under four names get four votes.
2. **Most members describe the past, not the future.** Measured over 11,756
   bars, the median member agreed with the *current* trend 77% of the time and
   with the *next* bar's direction 50.5% of the time. A majority vote of
   past-describers is a trend echo. That is why one member --
   ``smc_ascending_triangle`` -- was 87% trend-aligned and produced 259 BUYs and
   zero SELLs: it was not forecasting, it was confirming an uptrend.
3. **A vote cannot express uncertainty.** 60 of 132 members cannot be weighed
   against each other, and "the vote was 31-29" is not a probability.

This module is the alternative. It keeps every rule as an **input**, not as a
voter:

    features (FEATURE_NAMES)  +  member panel (132 stances, aggregated)
                              ->  one model
                              ->  one probability
                              ->  one decision

The panel summarises what all 132 rules think, per family, in a form the model
can weigh -- and, critically, one that cannot double-count, because identical
members collapse into the same family stance and the model learns how much that
family is worth. The rules are not deleted; they are demoted from voters to
evidence.

What comes out is a single number: the probability that price is higher over the
hold, and from it the expected **net** pips after spread. One engine, one
decision, one reason string.

Honesty rules, unchanged
------------------------

* The engine must beat the panel out of sample, or it is not an improvement.
  ``scripts/oracle_study.py`` measures it against the majority vote on bars the
  model never trained on, and reports the loss either way.
* It abstains when the expected net move does not clear the cost floor. An
  engine that always has an opinion is not a better engine.
* A probability is not a promise. Calibration is fitted out of fold, and the
  reported accuracy always carries a Wilson interval.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from . import charter
from .calibration import IsotonicCalibrator, fit_validated
from .features import FEATURE_NAMES
from .model import OnlineLogisticModel

# Families the 132 members collapse into. A family stance is the mean stance of
# its members, so four identical breakout rules contribute one unit of
# evidence, not four. This is the fix for the double-count defect as much as it
# is a feature-engineering choice.
# These must be exactly the labels ``strategies.strategy_family`` can return.
# An earlier draft invented SMC/VIRAL/QUANT/FLOW/METHOD names; the real labels
# are SMC_ICT, VIRAL_MOMENTUM, QUANT_SCALP and so on. The result was that five
# of eight panel inputs were constant zero -- the model was blind to 93
# members that there were supposed to be feeding it. tests/test_oracle.py now
# asserts the panel covers every family the registry can produce.
PANEL_FAMILIES = (
    "CORE_GATE",
    "CLASSIC_COUNCIL",
    "MICRO_SCALPING",
    "EXPIRY_SCALPING",
    "SMC_ICT",
    "VIRAL_MOMENTUM",
    "QUANT_SCALP",
)

PANEL_FEATURES = tuple(
    [f"panel_{family.lower()}_stance" for family in PANEL_FAMILIES]
    + [
        "panel_net_stance",       # signed balance of the whole panel
        "panel_agreement",        # share of speaking members on the winning side
        "panel_participation",    # share of members that spoke at all
        "panel_dispersion",       # how split the panel is (0 = unanimous)
        "panel_trend_gap",        # panel stance biased one way vs the other
    ]
)

ORACLE_FEATURES = tuple(FEATURE_NAMES) + PANEL_FEATURES

# How long the engine looks ahead when it is trained. One engine needs one
# horizon; a different one per caller would be 132 engines again.
DEFAULT_HORIZON_MINUTES = 15

# The engine will not speak unless the move it projects clears this share of
# the round-turn cost. Read from the charter so there is one cost policy.
MIN_NET_PIPS = 0.25


@dataclass(frozen=True)
class OracleRead:
    """One engine's single answer, with everything needed to audit it."""

    prob_up: float
    calibrated: float
    direction: str            # the raw directional call: "BUY" | "SELL"
    side: str                 # what is actually acted on: also "NO SIGNAL"
    take: bool
    expected_net_pips: float
    projected_move_pips: float
    spread_pips: float
    panel_net: float
    reason: str

    def to_dict(self) -> dict:
        return {
            "prob_up": round(self.prob_up, 4),
            "calibrated": round(self.calibrated, 4),
            "direction": self.direction,
            "side": self.side,
            "take": self.take,
            "expected_net_pips": round(self.expected_net_pips, 3),
            "projected_move_pips": round(self.projected_move_pips, 2),
            "spread_pips": round(self.spread_pips, 3),
            "panel_net": round(self.panel_net, 4),
            "reason": self.reason,
        }

    def __str__(self) -> str:
        return (
            f"{self.side} (raw {self.direction}) p_up={self.calibrated:.3f} "
            f"net={self.expected_net_pips:+.2f}p :: {self.reason}"
        )


# Members that are the same rule under different names. Measured by
# scripts/direction_audit.py: these emit an identical signal, bar for bar, in
# both directions, on 1,420 scored bars. One rule must cast one vote, so the
# panel counts each group once. Members listed here are still evaluated and
# still appear in the dashboard -- they simply do not get to double-count.
DUPLICATE_GROUPS: tuple[tuple[str, ...], ...] = (
    ("breakout", "quant_donchian_breakout", "scalp_breakout_retest"),
    ("smc_trend_resilience", "viral_rsi_extreme_follow"),
    ("smc_low_atr_candle", "smc_low_atr_trend"),
)

# member -> the single member that represents its group
_DUPLICATE_OF: dict[str, str] = {
    member: group[0]
    for group in DUPLICATE_GROUPS
    for member in group[1:]
}


def deduplicate(stances: Mapping[str, float]) -> dict[str, float]:
    """Fold identical members into one representative vote each.

    A duplicate that spoke while its twin did not is *not* dropped: the twin's
    silence is the twin's own information. Only an agreeing set collapses,
    because that is the case where counting both invents a second opinion.
    """
    out: dict[str, float] = {}
    for member, stance in stances.items():
        if stance is None:
            continue
        representative = _DUPLICATE_OF.get(member, member)
        existing = out.get(representative)
        if existing is None:
            out[representative] = float(stance)
        elif float(stance) != existing:
            # They disagree, which means they are not behaving identically on
            # this bar; keep the disagreement rather than hiding it.
            out[f"{member}~disagree"] = float(stance)
    return out


def family_stances(stances: Mapping[str, float]) -> dict[str, float]:
    """Collapse ``{member: stance}`` into per-family mean stances.

    Members that did not speak are absent, not zero: an abstention is not a
    neutral opinion and must not dilute the members that did speak.
    """
    from .strategies import strategy_family

    grouped: dict[str, list[float]] = {}
    for member, stance in stances.items():
        if stance is None:
            continue
        group = strategy_family(member)
        grouped.setdefault(group, []).append(float(stance))
    out: dict[str, float] = {}
    for family, values in grouped.items():
        out[family] = sum(values) / len(values)
    return out


def panel_vector(stances: Mapping[str, float]) -> dict[str, float]:
    """Turn 132 member stances into the panel half of the engine's input.

    ``stances`` maps member id -> -1.0 (sell), 0.0 (no signal) or +1.0 (buy).
    Members that are absent are treated as abstaining, which is different from
    disagreeing and is why participation is its own feature.
    """
    from .strategies import MEMBERS

    speaking = {
        name: float(value)
        for name, value in deduplicate(stances).items()
        if value is not None and float(value) != 0.0
    }
    # Participation is measured against the members that actually exist, so a
    # collapsed duplicate does not make the panel look quieter than it was.
    total = len(MEMBERS) - len(_DUPLICATE_OF)
    buys = sum(1 for value in speaking.values() if value > 0)
    sells = sum(1 for value in speaking.values() if value < 0)
    decided = buys + sells

    per_family = family_stances(speaking)
    vector: dict[str, float] = {}
    for family in PANEL_FAMILIES:
        vector[f"panel_{family.lower()}_stance"] = per_family.get(family, 0.0)

    net = (sum(speaking.values()) / len(speaking)) if speaking else 0.0
    vector["panel_net_stance"] = net
    vector["panel_agreement"] = (max(buys, sells) / decided) if decided else 0.0
    vector["panel_participation"] = decided / total if total else 0.0
    vector["panel_dispersion"] = (min(buys, sells) / decided) if decided else 0.0
    # A panel that is unanimous is informative; one that is split is noise. This
    # is the signed version, so the model can learn "strongly agreed both ways".
    vector["panel_trend_gap"] = abs(net) * vector["panel_agreement"]
    return vector


def majority_vote(stances: Mapping[str, float]) -> str:
    """The old design, kept only so the study can be a fair comparison."""
    buys = sum(1 for value in stances.values() if value is not None and value > 0)
    sells = sum(1 for value in stances.values() if value is not None and value < 0)
    if buys > sells:
        return "BUY"
    if sells > buys:
        return "SELL"
    return "NO SIGNAL"


@dataclass
class OracleEngine:
    """The single engine: all features, all members, one probability.

    ``horizon`` is the hold the engine is trained and judged on. It is a
    property of the engine, not a decision made per caller, because a per-caller
    horizon is how a single engine quietly becomes several again.
    """

    horizon: int = DEFAULT_HORIZON_MINUTES
    model: OnlineLogisticModel | None = None
    calibrator: IsotonicCalibrator | None = None
    trained_on: int = 0
    # Median raw score on the calibration tail. Direction is measured against
    # this, not against a fixed 0.5 -- see ``read`` for why.
    raw_centre: float = 0.5
    notes: list[str] = field(default_factory=list)

    # ------------------------------------------------------------- training
    def fit(
        self,
        samples: Iterable[tuple[dict[str, float], int]],
        *,
        min_samples: int = 200,
        calibration_fraction: float = 0.30,
    ) -> "OracleEngine":
        """Fit the model, then calibrate on a tail the model never saw.

        The split is chronological, not random: random splits leak the future
        into the past and produce a model that looks skilful and is not.
        """
        rows = [
            (dict(features), int(label))
            for features, label in samples
            if int(label) in (0, 1)
            and all(name in features for name in ORACLE_FEATURES)
        ]
        if len(rows) < min_samples:
            raise ValueError(
                f"the engine needs {min_samples} labelled bars to train, got {len(rows)}"
            )

        split = int(len(rows) * (1.0 - calibration_fraction))
        split = max(min_samples, min(split, len(rows) - max(50, min_samples // 4)))
        train_rows, calibration_rows = rows[:split], rows[split:]

        # The model must declare the engine's input set, or its own
        # ``compatible`` check (feature_names == FEATURE_NAMES) would reject it
        # and persistence would silently drop the engine.
        self.model = OnlineLogisticModel(feature_names=tuple(ORACLE_FEATURES)).fit(
            train_rows, min_samples=min_samples
        )
        raw = [self.model.predict_up(features) for features, _ in calibration_rows]
        labels = [label for _, label in calibration_rows]
        if raw:
            ordered = sorted(raw)
            self.raw_centre = ordered[len(ordered) // 2]
        calibration, report = fit_validated(
            list(zip(raw, labels)), minimum_points=max(50, len(raw) // 4)
        )
        self.calibrator = calibration if report.get("accepted") else None
        self.trained_on = len(rows)
        self.notes.append(
            f"trained on {len(train_rows)} bars, calibrated on {len(calibration_rows)}"
        )
        if self.calibrator is None or not self.calibrator.fitted:
            self.notes.append("calibration withheld: not enough out-of-fold evidence")
        return self

    @property
    def ready(self) -> bool:
        return self.model is not None and self.model.trained

    # ------------------------------------------------------------- inference
    def read(
        self,
        features: Mapping[str, float],
        stances: Mapping[str, float],
        *,
        spread_pips: float | None = None,
        min_net_pips: float = MIN_NET_PIPS,
    ) -> OracleRead:
        """One answer for one bar. Always returns a read; ``take`` may be False."""
        spread = float(charter.DEFAULT_SPREAD_PIPS if spread_pips is None else spread_pips)
        panel = panel_vector(stances)
        merged = {name: float(features[name]) for name in FEATURE_NAMES if name in features}
        merged.update(panel)

        if not self.ready:
            return OracleRead(
                prob_up=0.5,
                calibrated=0.5,
                direction="NO SIGNAL",
                side="NO SIGNAL",
                take=False,
                expected_net_pips=0.0,
                projected_move_pips=0.0,
                spread_pips=spread,
                panel_net=panel["panel_net_stance"],
                reason="engine is not trained",
            )

        raw = self.model.predict_up(merged)
        calibrated = raw
        if self.calibrator is not None and self.calibrator.fitted:
            calibrated = self.calibrator.transform(raw)

        # The projected move comes from the engine's own realised average at
        # this horizon, not from a per-call projection. Without a measurement
        # the engine declines to guess a payoff.
        published = getattr(self, "move_pips", None)
        if not published:
            return OracleRead(
                prob_up=raw,
                calibrated=calibrated,
                direction="NO SIGNAL",
                side="NO SIGNAL",
                take=False,
                expected_net_pips=0.0,
                projected_move_pips=0.0,
                spread_pips=spread,
                panel_net=panel["panel_net_stance"],
                reason="engine has no measured move for this horizon",
            )

        move = float(published)
        p = max(0.0, min(1.0, calibrated))

        # DIRECTION is measured against the score's OWN centre, not against a
        # fixed 0.5 and never against the calibrated value.
        #
        # Both alternatives are asymmetric and both failed in testing. Using
        # "calibrated >= 0.5" turned a 48.6%-up sample into an engine calling
        # BUY on 2.3% of bars, because calibration is a monotone map fitted on
        # a tail whose base rate is not 0.5 and whose isotonic fit collapsed to
        # four levels topping out at 0.458 -- so it could never cross 0.5.
        # Using "raw >= 0.5" was no better: the logistic intercept settles
        # wherever the data says, and here it drifted to bias -0.374, putting
        # the whole raw distribution below 0.5.
        #
        # Anchoring to the median raw score is two-sided by construction: half
        # the bars lean each way whatever the intercept does, and the engine can
        # only express a *relative* view -- which is the only thing an
        # uncalibrated score can honestly express. Whether that view is worth
        # money is then decided by the calibrated probability against the cost
        # floor, not by the direction rule.
        direction = "BUY" if raw >= self.raw_centre else "SELL"
        edge = p if direction == "BUY" else 1.0 - p
        expected = edge * (move - spread) - (1.0 - edge) * (move + spread)
        take = expected >= min_net_pips
        reason = (
            f"raw {raw:.3f} -> {direction}; calibrated p={p:.3f} vs "
            f"{required_probability(move, spread, min_net_pips):.3f} needed, "
            f"move {move:.2f}p, cost {spread:.2f}p -> {expected:+.2f}p net"
        )
        if not take:
            reason += " (below floor, abstaining)"
        return OracleRead(
            prob_up=raw,
            calibrated=calibrated,
            direction=direction,
            side=direction if take else "NO SIGNAL",
            take=take,
            expected_net_pips=expected,
            projected_move_pips=move,
            spread_pips=spread,
            panel_net=panel["panel_net_stance"],
            reason=reason,
        )

    # ------------------------------------------------------------ persistence
    def to_dict(self) -> dict:
        return {
            "horizon": self.horizon,
            "model": None if self.model is None else self.model.to_dict(),
            "calibrator": None if self.calibrator is None else self.calibrator.to_dict(),
            "move_pips": getattr(self, "move_pips", None),
            "raw_centre": float(self.raw_centre),
            "trained_on": self.trained_on,
            "notes": list(self.notes),
            "features": list(ORACLE_FEATURES),
        }

    @classmethod
    def from_dict(cls, payload: dict | None) -> "OracleEngine | None":
        if not payload:
            return None
        if list(payload.get("features") or []) != list(ORACLE_FEATURES):
            return None  # a model fitted on a different input set is not this engine
        model = OnlineLogisticModel.from_dict(payload.get("model"))
        if model is None:
            return None
        engine = cls(horizon=int(payload.get("horizon", DEFAULT_HORIZON_MINUTES)))
        engine.model = model
        engine.calibrator = IsotonicCalibrator.from_dict(payload.get("calibrator"))
        engine.move_pips = payload.get("move_pips")
        engine.raw_centre = float(payload.get("raw_centre", 0.5))
        engine.trained_on = int(payload.get("trained_on", 0))
        engine.notes = list(payload.get("notes") or [])
        return engine


def expected_net_pips(probability_up: float, move_pips: float, spread_pips: float) -> float:
    """Expected net pips of the trade this probability implies.

    A convenience wrapper for callers that already have the three numbers and
    want the same arithmetic the engine uses, rather than a second version of
    it living in the dashboard.
    """
    p = max(0.0, min(1.0, float(probability_up)))
    edge = p if p >= 0.5 else 1.0 - p
    move = float(move_pips)
    spread = float(spread_pips)
    return edge * (move - spread) - (1.0 - edge) * (move + spread)


def required_probability(move_pips: float, spread_pips: float, min_net_pips: float = MIN_NET_PIPS) -> float:
    """The probability the engine must reach before a trade is worth taking.

    Inverts :func:`expected_net_pips`. Reported alongside every engine decision
    so "the engine said take it" can always be checked against "it needed 0.62
    and it had 0.66".
    """
    move = float(move_pips)
    spread = float(spread_pips)
    if move <= spread or move <= 0.0:
        return float("inf")
    # edge*(m-s) - (1-edge)*(m+s) = min_net  ->  edge = (m + s + min_net) / (2m)
    edge = (move + spread + float(min_net_pips)) / (2.0 * move)
    if edge > 1.0:
        return float("inf")
    return max(0.5, edge)


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Delegated to the charter so there is one interval implementation."""
    return charter.wilson_interval(successes, total, z)


def summarise_reads(reads: Sequence[OracleRead]) -> dict:
    """A compact audit of what the engine did over a run."""
    taken = [read for read in reads if read.take]
    if not reads:
        return {"bars": 0, "taken": 0, "abstain_rate": 0.0}
    buys = sum(1 for read in taken if read.side == "BUY")
    sells = sum(1 for read in taken if read.side == "SELL")
    return {
        "bars": len(reads),
        "taken": len(taken),
        "abstain_rate": 1.0 - len(taken) / len(reads),
        "buys": buys,
        "sells": sells,
        "sell_share": sells / len(taken) if taken else 0.0,
        "mean_expected_net": (
            sum(read.expected_net_pips for read in taken) / len(taken) if taken else 0.0
        ),
    }


def _finite(value: float) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))
