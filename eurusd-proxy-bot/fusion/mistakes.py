"""Post-mortem memory: refuse to repeat a mistake already paid for.

The council learns *who* to trust. This learns *what to avoid*. They are
different questions. A prototype weight says "breakout has been right lately";
a post-mortem says "buying breakouts in a ranging Asian session has lost 31 of
its last 40, and we are confident that is worse than a coin flip".

Design rules, all of them load-bearing:

* **Only real outcomes count.** Synthetic data trains nothing, so a replay must
  never write here. The caller is responsible; ``observe`` takes the settled
  outcome and the caller filters by source.
* **Confidence, not averages.** A 40% win rate over 10 trades is noise. The
  gate is the *upper* bound of a 95% Wilson interval falling below even money,
  so the bot only forbids a setup it can actually demonstrate is losing. Small
  samples are humble by construction, not by a hand-tuned prior.
* **It can only block.** Like the prototypes, post-mortems never create a
  signal. A quiet memory is a silent one.
* **It can forgive.** Counts are capped and decay, so a setup that stops
  losing earns its way back instead of being banned by history forever.

The fingerprint is deliberately coarse. Finer buckets fit the noise rather
than the market: with a hundred cells and twenty samples each, every one of
them looks like a mistake.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from . import charter

# Below this the interval is too wide to say anything, however bad the record
# looks. Twelve losses in a row still passes: that is the point.
MIN_SAMPLES = 15
# A cap and a decay, so a regime change can clear a stale ban. When a cell
# fills, both counts are halved rather than frozen at the limit.
MAX_SAMPLES = 200

# Refuse when even the optimistic end of the interval is below even money.
BREAK_EVEN = 0.5


def fingerprint(
    regime: str,
    session: str | None,
    side: str,
) -> str:
    """The coarse identity of a setup. ``session`` None means unfiltered.

    Deliberately regime x session x side and nothing finer. Adding a volatility
    bucket gives 120 cells, and at a sample floor of 25 that needs 3000 settled
    trades before the memory can speak at all -- years at this bot's rate, which
    would make the whole feature decorative. Volatility is already owned by the
    cost gate, so it is not unguarded here, just not duplicated.

    Forty cells at fifteen samples each is reachable in months rather than
    never, and coarse cells generalise: "the bot misreads ranging London
    mornings" is a real flaw, "it misreads them when ATR is 1.2 not 1.1" is a
    fit to noise.
    """
    return f"{regime}|{session or 'ANY'}|{side}"


@dataclass
class MistakeMemory:
    # Reachable both as ``memory.fingerprint(...)`` and
    # ``MistakeMemory.fingerprint(...)``, so a caller with an instance and one
    # with only the class build identical keys.
    fingerprint = staticmethod(fingerprint)
    """Per-fingerprint win/loss counts from real, cost-charged outcomes."""

    wins: dict[str, int] = field(default_factory=dict)
    losses: dict[str, int] = field(default_factory=dict)
    version: int = 1

    def observe(self, key: str, won: bool) -> None:
        """Record one settled outcome. ``won`` is gross direction, before cost."""
        if not key:
            return
        wins = int(self.wins.get(key, 0))
        losses = int(self.losses.get(key, 0))
        if won:
            wins += 1
        else:
            losses += 1
        total = wins + losses
        if total > MAX_SAMPLES:
            # Halve rather than clamp: the cell keeps learning, older evidence
            # just counts for less. A hard clamp would make the ban permanent.
            wins //= 2
            losses //= 2
        self.wins[key] = wins
        self.losses[key] = losses
        self.version += 1

    def record(self, outcome: str, key: str) -> None:
        """Record by outcome label, ignoring anything that was not decided."""
        if outcome == charter.WIN:
            self.observe(key, True)
        elif outcome == charter.LOSS:
            self.observe(key, False)
        # Tied and unresolved teach nothing; recording them would be a guess.

    def sample_size(self, key: str) -> int:
        return int(self.wins.get(key, 0)) + int(self.losses.get(key, 0))

    def win_rate(self, key: str) -> float | None:
        total = self.sample_size(key)
        if total == 0:
            return None
        return float(self.wins.get(key, 0)) / total

    def refuses(self, key: str) -> tuple[bool, str]:
        """Should this setup be refused, and why.

        The gate is deliberately hard to satisfy. It needs enough samples for
        the interval to mean something AND an upper bound below even money, so
        the bot is never banned by a bad week.
        """
        total = self.sample_size(key)
        if total < MIN_SAMPLES:
            return False, ""
        wins = int(self.wins.get(key, 0))
        _low, high = charter.wilson_interval(wins, total)
        if high >= BREAK_EVEN:
            return False, ""
        rate = wins / total
        return True, (
            f"Post-mortem: this setup has won {wins} of {total} "
            f"({rate * 100:.0f}%), and even the optimistic end of its 95% "
            f"interval ({high * 100:.0f}%) is below even money. Refusing to "
            f"repeat a mistake already paid for."
        )

    def banned(self) -> list[str]:
        """Keys currently refused, for the status report and the digest."""
        return [key for key in set(self.wins) | set(self.losses) if self.refuses(key)[0]]

    def to_dict(self) -> dict:
        return {
            "wins": dict(self.wins),
            "losses": dict(self.losses),
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, payload: dict | None) -> "MistakeMemory":
        if not payload:
            return cls()
        result = cls()
        result.wins = {str(k): int(v) for k, v in (payload.get("wins") or {}).items()}
        result.losses = {str(k): int(v) for k, v in (payload.get("losses") or {}).items()}
        result.version = int(payload.get("version", 1))
        return result

    def total_records(self) -> int:
        keys: Iterable[str] = set(self.wins) | set(self.losses)
        return sum(self.sample_size(key) for key in keys)
