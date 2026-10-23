"""Inference on trade outcomes — deciding when a number is evidence.

Why this module exists
----------------------

A replay of 220 signals that nets −0.35 pips per trade looks like a result.
It is not. It is **one draw** from a distribution whose width nobody measured.
Run the same system over eight different price paths and the same "result"
ranges from −2.96 to +0.12 pips per trade. The mean is about −1.3. The single
number that got reported was the luckiest of the eight.

That failure mode is not a modelling mistake, it is a *reporting* mistake, and
reporting mistakes are the dangerous ones: they are the difference between a
research tool and a machine that manufactures confidence.

So this module answers one question, and refuses to answer it when it cannot:

    given this sequence of trade outcomes, is there enough evidence to say
    which side of zero the true expectancy is on?

Three verdicts, and never two:

* ``EDGE_DEMONSTRATED``  — enough trades, and the whole interval is above zero.
* ``EDGE_REFUTED``       — enough trades, and the whole interval is below zero.
* ``EDGE_INCONCLUSIVE``  — everything else.

``EDGE_INCONCLUSIVE`` is not a polite way of saying "no edge". It means the
sample cannot tell. **Absence of evidence is not evidence of absence**, and the
distinction is enforced here rather than left to whoever is reading the output.

Why not a plain t-interval
--------------------------

Trade outcomes are autocorrelated: positions overlap in time, and a trending
hour produces a run of similar results. A t-interval assumes each observation
is independent, which understates the interval and manufactures significance.
The interval here comes from a **circular block bootstrap** — resampling runs
of consecutive trades — so persistence in the sequence survives into the
interval instead of being averaged away.

Stdlib only, on purpose: this runs on Termux.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence

# --- thresholds ---------------------------------------------------------------
# Fewer trades than this and the interval is too wide to say anything useful,
# whatever the point estimate claims. Matches backtest.MIN_SIGNALS_FOR_A_VIEW so
# the two places that judge "enough" agree with each other.
MIN_SAMPLES_FOR_EDGE = 200

# The floor above counts TRADES, and a trade is a noisy thing: hundreds are
# needed to see through one. An observation can also be a whole price path or a
# paired difference between two configurations, each of which already summarises
# hundreds of trades. Applying a 200-trade floor to eight such observations
# would dismiss a consistent, tightly-bounded effect for the crime of being
# cheap to measure, which is its own way of manufacturing the wrong conclusion.
MIN_PATHS_FOR_EDGE = 5

# Default run length for the block bootstrap. Ten consecutive trades keeps the
# short-range dependence that a per-trade shuffle would destroy.
DEFAULT_BLOCK_SIZE = 10

DEFAULT_BOOTSTRAP_ITERATIONS = 2000

DEFAULT_CONFIDENCE = 0.95

# Verdicts.
EDGE_DEMONSTRATED = "EDGE_DEMONSTRATED"
EDGE_REFUTED = "EDGE_REFUTED"
EDGE_INCONCLUSIVE = "EDGE_INCONCLUSIVE"


def _finite(values: Sequence[float]) -> list[float]:
    return [float(value) for value in values if isinstance(value, (int, float)) and math.isfinite(float(value))]


def _mean(values: Sequence[float]) -> float:
    return (sum(values) / len(values)) if values else 0.0


def stdev(values: Sequence[float]) -> float:
    """Sample standard deviation. Zero for fewer than two points."""
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(max(0.0, variance))


def _circular_block_sample(values: Sequence[float], block: int, rng: random.Random) -> list[float]:
    """Resample ``values`` in runs of ``block``, wrapping at the end.

    Wrapping means every position is an equally likely block start, so the
    resample has no edge effects and no need to truncate a final partial block.
    """
    size = len(values)
    out: list[float] = []
    while len(out) < size:
        start = rng.randrange(size)
        for offset in range(block):
            out.append(values[(start + offset) % size])
    return out[:size]


def block_bootstrap_means(
    values: Sequence[float],
    *,
    block_size: int | None = None,
    iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    seed: int = 20260101,
) -> list[float]:
    """The sampling distribution of the mean, by circular block bootstrap.

    Returns one mean per iteration, sorted. An empty input yields an empty list.
    """
    data = _finite(values)
    if not data:
        return []
    size = len(data)
    block = int(block_size) if block_size else DEFAULT_BLOCK_SIZE
    block = max(1, min(block, size))
    rng = random.Random(seed)
    means = []
    for _ in range(max(1, int(iterations))):
        resample = _circular_block_sample(data, block, rng)
        means.append(_mean(resample))
    means.sort()
    return means


def _percentile(sorted_values: Sequence[float], quantile: float) -> float:
    """Linear-interpolation percentile over an already-sorted sequence."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = min(1.0, max(0.0, quantile)) * (len(sorted_values) - 1)
    low_index = math.floor(position)
    high_index = math.ceil(position)
    if low_index == high_index:
        return float(sorted_values[int(position)])
    weight = position - low_index
    low_value = float(sorted_values[low_index])
    high_value = float(sorted_values[high_index])
    return low_value + (high_value - low_value) * weight


def confidence_interval(
    values: Sequence[float],
    *,
    confidence: float = DEFAULT_CONFIDENCE,
    block_size: int | None = None,
    iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    seed: int = 20260101,
) -> tuple[float, float, float]:
    """``(mean, low, high)`` for the mean of ``values``.

    Below three points the mean is reported with a zero-width interval, because
    pretending to measure the spread of two numbers is worse than admitting
    there is nothing to measure.
    """
    data = _finite(values)
    if not data:
        return (0.0, 0.0, 0.0)
    mean = _mean(data)
    if len(data) < 3:
        return (mean, mean, mean)
    alpha = 1.0 - min(0.999, max(0.5, float(confidence)))

    # Two estimates, and we report the wider of them.
    #
    # 1. Normal theory, which assumes each trade is independent. It is correct
    #    when outcomes really are independent, and it is the yardstick most
    #    readers will implicitly compare against.
    z = _z_for(confidence)
    half = z * stdev(data) / math.sqrt(len(data))
    normal_low, normal_high = mean - half, mean + half

    # 2. The block bootstrap, which keeps runs of consecutive trades together so
    #    persistence in the sequence is not averaged away.
    means = block_bootstrap_means(
        data, block_size=block_size, iterations=iterations, seed=seed
    )
    if not means:
        return (mean, normal_low, normal_high)
    boot_low = _percentile(means, alpha / 2.0)
    boot_high = _percentile(means, 1.0 - alpha / 2.0)

    # Taking the wider of the two is deliberate, and the direction matters.
    # Measured against iid data the block bootstrap is slightly NARROW at larger
    # block sizes — resampling long runs partly reproduces the original ordering
    # and so understates the variance. An interval that is too narrow invents
    # significance, which is the exact failure this module exists to prevent, so
    # the reported interval is never allowed to be tighter than the
    # independence-assuming estimate. When outcomes are autocorrelated the
    # bootstrap is wider and wins; when they are not, normal theory holds the
    # line. Every error this can make is in the cautious direction.
    return (mean, min(boot_low, normal_low), max(boot_high, normal_high))


def required_samples(
    stdev_pips: float,
    target_edge_pips: float,
    *,
    confidence: float = DEFAULT_CONFIDENCE,
) -> int:
    """How many trades it takes to resolve an edge of ``target_edge_pips``.

    Uses the normal approximation ``n = (z·sd / target)²``. Returns ``-1`` when
    the target is zero or negative, where the question has no finite answer.
    """
    if target_edge_pips <= 0 or stdev_pips <= 0:
        return -1
    z = _z_for(confidence)
    return int(math.ceil((z * float(stdev_pips) / float(target_edge_pips)) ** 2))


def _z_for(confidence: float) -> float:
    """Two-sided z value. Supports the levels anyone actually uses."""
    table = {0.80: 1.2816, 0.90: 1.6449, 0.95: 1.9600, 0.98: 2.3263, 0.99: 2.5758}
    return table.get(round(float(confidence), 4), 1.9600)


@dataclass(frozen=True, slots=True)
class EdgeVerdict:
    """The answer to "does this make money", or an honest refusal to answer."""

    n: int
    mean_net_pips: float
    ci_low: float
    ci_high: float
    confidence: float
    status: str
    stdev_pips: float = 0.0
    samples_needed: int = -1
    min_samples: int = MIN_SAMPLES_FOR_EDGE
    unit: str = "trades"

    @property
    def demonstrated(self) -> bool:
        return self.status == EDGE_DEMONSTRATED

    @property
    def refuted(self) -> bool:
        return self.status == EDGE_REFUTED

    @property
    def inconclusive(self) -> bool:
        return self.status == EDGE_INCONCLUSIVE

    def describe(self) -> str:
        """One line, safe to put in front of a human.

        Every branch states the sample size next to the claim, so a verdict can
        never be quoted without the evidence it rests on.
        """
        if self.n <= 0:
            return "no settled trades yet — nothing to judge"
        interval = f"95% CI {self.ci_low:+.3f} to {self.ci_high:+.3f} pips/trade"
        if self.demonstrated:
            return (
                f"edge demonstrated across {self.n} {self.unit}: "
                f"{self.mean_net_pips:+.3f} pips/trade after costs ({interval})"
            )
        if self.refuted:
            return (
                f"no edge — confidently negative across {self.n} {self.unit}: "
                f"{self.mean_net_pips:+.3f} pips/trade after costs ({interval})"
            )
        if self.n < self.min_samples:
            short = self.min_samples - self.n
            return (
                f"not enough evidence — {self.n} trades, {short} short of the "
                f"{self.min_samples} needed before a verdict is offered "
                f"(point estimate {self.mean_net_pips:+.3f}, {interval})"
            )
        return (
            f"inconclusive on {self.n} trades: {self.mean_net_pips:+.3f} "
            f"pips/trade, but the interval straddles zero ({interval}) — "
            f"this is not evidence of no edge, only of too little evidence"
        )

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "mean_net_pips": self.mean_net_pips,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "confidence": self.confidence,
            "status": self.status,
            "stdev_pips": self.stdev_pips,
            "samples_needed": self.samples_needed,
            "min_samples": self.min_samples,
            "unit": self.unit,
        }


def edge_verdict(
    net_pips: Sequence[float],
    *,
    confidence: float = DEFAULT_CONFIDENCE,
    min_samples: int | None = None,
    block_size: int | None = None,
    iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    seed: int = 20260101,
    target_edge_pips: float = 0.25,
    unit: str = "trades",
) -> EdgeVerdict:
    """Judge a sequence of per-trade net pip outcomes.

    ``target_edge_pips`` is the smallest edge worth detecting, used only to say
    how many more trades would settle the question.
    """
    # The floor follows the unit: 200 trades, or 5 aggregated observations.
    if min_samples is None:
        floor = MIN_PATHS_FOR_EDGE if unit == "paths" else MIN_SAMPLES_FOR_EDGE
    else:
        floor = int(min_samples)

    data = _finite(net_pips)
    spread_sd = stdev(data)
    mean, low, high = confidence_interval(
        data, confidence=confidence, block_size=block_size, iterations=iterations, seed=seed
    )

    # A single trade, or none, cannot produce an interval: report the point
    # estimate and refuse to judge it.
    if len(data) < 3:
        status = EDGE_INCONCLUSIVE
    elif len(data) < floor:
        # Under-powered. Refusing here is the whole point of the module: a
        # 30-trade hot streak must not be able to claim an edge.
        status = EDGE_INCONCLUSIVE
    elif low > 0.0:
        status = EDGE_DEMONSTRATED
    elif high < 0.0:
        status = EDGE_REFUTED
    else:
        status = EDGE_INCONCLUSIVE

    # The two thresholds answer different questions and can disagree: a power
    # calculation may say 62 trades resolve a 1.0 pip edge, while the floor says
    # no verdict is offered below 200. Quoting the smaller number would invite
    # someone to collect 62 and expect an answer, so report the stricter one.
    power_needed = required_samples(spread_sd, target_edge_pips, confidence=confidence)
    if power_needed < 0:
        needed = floor
    else:
        needed = max(power_needed, floor)

    return EdgeVerdict(
        n=len(data),
        mean_net_pips=mean,
        ci_low=low,
        ci_high=high,
        confidence=confidence,
        status=status,
        stdev_pips=spread_sd,
        samples_needed=needed,
        min_samples=floor,
        unit=unit,
    )


@dataclass(frozen=True, slots=True)
class MultiPathSummary:
    """The same system replayed over several independent price paths.

    One path answers nothing, because a single draw is a draw. This is the
    distribution over draws.
    """

    n_paths: int
    expectancies: tuple[float, ...]
    mean_expectancy: float
    stdev_expectancy: float
    best: float
    worst: float
    profitable_paths: int
    verdict: EdgeVerdict

    @property
    def single_path_is_misleading(self) -> bool:
        """True when paths disagree on the sign, i.e. a lone path told a story."""
        if not self.expectancies:
            return False
        return any(value > 0 for value in self.expectancies) and any(
            value < 0 for value in self.expectancies
        )

    def describe(self) -> str:
        """One line that leads with the spread across paths, not the average."""
        if not self.expectancies:
            return "no paths replayed — nothing to judge"
        spread_line = (
            f"{self.n_paths} price paths: mean {self.mean_expectancy:+.3f} "
            f"pips/trade (sd {self.stdev_expectancy:.3f}), "
            f"best {self.best:+.3f}, worst {self.worst:+.3f}"
        )
        if self.single_path_is_misleading:
            spread_line += (
                f" — paths disagree on the sign ({self.profitable_paths}/{self.n_paths} "
                f"positive), so any single one of them would have told a story"
            )
        return f"{spread_line}. Across-path verdict: {self.verdict.describe()}"

    def to_dict(self) -> dict:
        return {
            "paths": self.n_paths,
            "expectancies": list(self.expectancies),
            "mean_expectancy": self.mean_expectancy,
            "stdev_expectancy": self.stdev_expectancy,
            "best": self.best,
            "worst": self.worst,
            "profitable_paths": self.profitable_paths,
            "verdict": self.verdict.to_dict(),
        }


def summarise_paths(expectancies: Sequence[float]) -> MultiPathSummary:
    """Summarise per-path expectancies, judged across paths rather than within one.

    The verdict treats each path's expectancy as one observation, which is the
    honest unit: the question is not "did these trades average positive" but
    "would this system be positive on a path I have not seen".
    """
    data = _finite(expectancies)
    if not data:
        return MultiPathSummary(
            n_paths=0,
            expectancies=(),
            mean_expectancy=0.0,
            stdev_expectancy=0.0,
            best=0.0,
            worst=0.0,
            profitable_paths=0,
            verdict=EdgeVerdict(
                n=0, mean_net_pips=0.0, ci_low=0.0, ci_high=0.0,
                confidence=DEFAULT_CONFIDENCE, status=EDGE_INCONCLUSIVE,
            ),
        )
    mean, low, high = confidence_interval(
        data, block_size=1, iterations=DEFAULT_BOOTSTRAP_ITERATIONS
    )
    if low > 0.0:
        status = EDGE_DEMONSTRATED
    elif high < 0.0:
        status = EDGE_REFUTED
    else:
        status = EDGE_INCONCLUSIVE
    # The unit of observation here is a PRICE PATH, not a trade. Saying
    # "on 8 trades" when it is 8 paths would understate the evidence by orders
    # of magnitude, so the verdict carries its own label.
    return MultiPathSummary(
        n_paths=len(data),
        expectancies=tuple(data),
        mean_expectancy=mean,
        stdev_expectancy=stdev(data),
        best=max(data),
        worst=min(data),
        profitable_paths=sum(1 for value in data if value > 0),
        verdict=EdgeVerdict(
            n=len(data),
            mean_net_pips=mean,
            ci_low=low,
            ci_high=high,
            confidence=DEFAULT_CONFIDENCE,
            status=status,
            stdev_pips=stdev(data),
            unit="paths",
        ),
    )
