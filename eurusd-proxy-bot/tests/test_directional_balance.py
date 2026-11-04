"""The bot must be able to sell.

Symptom this file exists for: every emitted signal was a BUY and the settled
trades lost money. The cause was not one bad rule but five separate defects,
each of which added a fixed amount of BUY weight to scores that are compared
against zero. A score with a constant positive offset is not a coin that
happens to land heads; it is a coin with a head on both sides.

Each test below pins one of those defects, so it cannot come back quietly:

1. ``hurst_exponent`` ran R/S analysis on prices instead of returns, so it read
   0.92 on average and never once fell below 0.5 -- the mean-reverting branch
   was dead code and the vote was BUY on 100% of bars.
2. The RSI vote was asymmetric: the mildly-oversold band was positive and the
   neutral band was zero, so it averaged +0.32 instead of 0.0.
3. The Keltner squeeze vote could only be +1.0 or 0.0, because compression is
   directionless and the vote ignored the momentum value that isn't.
4. Eight feature keys that council members read were never published, so
   ``features.get(key, 0.0)`` handed those rules a constant.
5. The Chaikin oscillator could not go negative, so its bearish branch was
   unreachable.

Together they made the dashboard's "sureshot" filter produce 149 BUY signals
and 36 SELL signals from the same 740 real bars where a balanced score
produces 94 and 84.
"""

from __future__ import annotations

import math
import random
import unittest

from fusion import indicators, methods, scalping, scalping_expiry, strategies
from fusion.domain import Candle
from fusion.features import compute_features
from fusion.indicators import (
    hurst_exponent,
    rsi_composite_vote,
    squeeze_composite_vote,
)

from .helpers import make_candles


def _random_walk(count: int = 2500, *, seed: int = 3, vol: float = 0.0002) -> list[float]:
    rng = random.Random(seed)
    price = 1.10
    series: list[float] = []
    for _ in range(count):
        price *= 1.0 + rng.gauss(0.0, vol)
        series.append(price)
    return series


class TestHurstEstimator(unittest.TestCase):
    """R/S belongs on returns, and a random walk must read 0.5."""

    def test_random_walk_reads_about_one_half(self) -> None:
        values = [v for v in hurst_exponent(_random_walk(), 20) if v is not None]
        self.assertTrue(values)
        mean = sum(values) / len(values)
        self.assertAlmostEqual(mean, 0.5, delta=0.06, msg=f"mean {mean:.3f} should be ~0.5")

    def test_random_walk_spans_both_sides_of_the_threshold(self) -> None:
        """The old price-domain form never produced a reading below 0.5."""
        values = [v for v in hurst_exponent(_random_walk(), 20) if v is not None]
        self.assertGreater(sum(1 for v in values if v > 0.5), 0)
        self.assertGreater(sum(1 for v in values if v < 0.5), 0)

    def test_mean_reverting_series_reads_below_a_trending_one(self) -> None:
        rng = random.Random(11)
        price, reverting = 1.10, []
        for _ in range(2500):
            price = 1.10 + 0.7 * (price - 1.10) + rng.gauss(0.0, 0.0002)
            reverting.append(price)
        trend_price, trending = 1.10, []
        for _ in range(2500):
            trend_price *= 1.0 + rng.gauss(0.00012, 0.0002)
            trending.append(trend_price)

        def mean_h(series: list[float]) -> float:
            values = [v for v in hurst_exponent(series, 20) if v is not None]
            return sum(values) / len(values)

        self.assertLess(mean_h(reverting), mean_h(trending))

    def test_expectation_matches_monte_carlo(self) -> None:
        """The Anis-Lloyd expectation is the thing that removes the bias."""
        rng = random.Random(5)
        for lag in (5, 12, 20):
            total = 0.0
            for _ in range(400):
                draws = [rng.gauss(0.0, 1.0) for _ in range(lag)]
                mean = sum(draws) / lag
                devs = [x - mean for x in draws]
                cumulative, running = [], 0.0
                for d in devs:
                    running += d
                    cumulative.append(running)
                spread = max(cumulative) - min(cumulative)
                sigma = math.sqrt(sum(d * d for d in devs) / lag)
                if sigma > 0:
                    total += spread / sigma
            empirical = total / 400
            expected = indicators._expected_rescaled_range(lag)
            self.assertAlmostEqual(
                empirical / expected, 1.0, delta=0.06,
                msg=f"lag {lag}: E[R/S] {empirical:.3f} vs formula {expected:.3f}",
            )

    def test_uniform_series_is_unknown_not_half(self) -> None:
        """A flat series has no spread to rescale; 0.5 would read as 'reverting'."""
        self.assertTrue(all(v is None for v in hurst_exponent([1.10] * 200, 20)))


class TestCompositeVotesAreBalanced(unittest.TestCase):
    """A vote that cannot reach both signs is a standing bias, not a signal."""

    def test_rsi_vote_reaches_both_extremes(self) -> None:
        self.assertEqual(rsi_composite_vote(10.0), 1.0)
        self.assertEqual(rsi_composite_vote(90.0), -1.0)

    def test_rsi_vote_is_symmetric_about_fifty(self) -> None:
        for offset in (5.0, 15.0, 25.0, 40.0):
            self.assertEqual(
                rsi_composite_vote(50.0 - offset),
                -rsi_composite_vote(50.0 + offset),
                msg=f"offset {offset} is not mirrored",
            )

    def test_rsi_vote_is_neutral_at_fifty(self) -> None:
        self.assertEqual(rsi_composite_vote(50.0), 0.0)

    def test_rsi_vote_averages_near_zero_on_a_two_sided_sample(self) -> None:
        """The old band set averaged +0.318 on real EUR/USD bars."""
        candles = make_candles(900, seed=17, volatility=0.00022)
        closes = [c.close for c in candles]
        values = [v for v in indicators.rsi(closes, 14) if v is not None]
        mean = sum(rsi_composite_vote(v) for v in values) / len(values)
        self.assertLess(abs(mean), 0.15, msg=f"mean vote {mean:+.3f} is biased")

    def test_rsi_vote_handles_missing_input(self) -> None:
        for value in (None, float("nan"), float("inf")):
            self.assertEqual(rsi_composite_vote(value), 0.0)

    def test_squeeze_vote_uses_momentum_for_direction(self) -> None:
        self.assertEqual(squeeze_composite_vote(True, 0.8), 1.0)
        self.assertEqual(squeeze_composite_vote(True, -0.8), -1.0)

    def test_squeeze_vote_is_silent_without_compression(self) -> None:
        self.assertEqual(squeeze_composite_vote(False, 3.0), 0.0)
        self.assertEqual(squeeze_composite_vote(False, -3.0), 0.0)

    def test_squeeze_vote_is_silent_without_momentum(self) -> None:
        self.assertEqual(squeeze_composite_vote(True, None), 0.0)
        self.assertEqual(squeeze_composite_vote(True, 0.0), 0.0)


class TestEveryRuleInputExists(unittest.TestCase):
    """A rule reading a key that is never published is not a rule."""

    def test_no_council_member_reads_a_missing_feature(self) -> None:
        rows = compute_features(make_candles(220, seed=21))
        published = set(rows[-1])
        missing: dict[str, list[str]] = {}
        for module in (methods, scalping, scalping_expiry):
            for name in dir(module):
                function = getattr(module, name)
                if not callable(function) or not name.startswith(("scalp_", "smc_", "viral_", "quant_")):
                    continue
                if not getattr(function, "__module__", "").startswith("fusion."):
                    continue
                code = getattr(function, "__code__", None)
                if code is None:
                    continue
                for key in code.co_consts:
                    if not isinstance(key, str):
                        continue
                    # Keys are used via features.get("<name>"); the literals
                    # land in co_consts, so check the ones shaped like keys.
                    if "_" in key and key.islower() and key not in published:
                        if key in getattr(module, "__dict__", {}):
                            continue
                        missing.setdefault(key, []).append(name)
        # Only report keys that a rule actually looks up.
        genuinely_missing = {
            key: users for key, users in missing.items()
            if key in _looked_up_keys(mod for mod in (methods, scalping, scalping_expiry))
        }
        self.assertEqual({}, genuinely_missing)

    def test_chaikin_oscillator_can_be_negative(self) -> None:
        """It could not be, so quant_chaikin_oscillator could only ever BUY."""
        candles = make_candles(400, seed=33)
        values = [indicators._chaikin_oscillator(candles, i) for i in range(4, len(candles))] \
            if hasattr(indicators, "_chaikin_oscillator") else None
        if values is None:
            from fusion.features import _chaikin_oscillator
            values = [_chaikin_oscillator(candles, i) for i in range(4, len(candles))]
        self.assertLess(min(values), 0.0)
        self.assertGreater(max(values), 0.0)

    def test_prev_aliases_agree_with_their_canonical_names(self) -> None:
        row = compute_features(make_candles(220, seed=21))[-1]
        self.assertEqual(row["ema_fast_prev"], row["ema_fast_previous"])
        self.assertEqual(row["ema_slow_prev"], row["ema_slow_previous"])
        self.assertEqual(row["rsi_14"], row["rsi"])


class TestCouncilCanTakeBothSides(unittest.TestCase):
    """End to end: the assembled council must emit both directions."""

    def test_council_emits_buy_and_sell_on_a_two_sided_market(self) -> None:
        from fusion import council

        candles = make_candles(1200, seed=23, volatility=0.00025)
        rows = compute_features(candles)
        weight = council.CouncilWeights()
        sides: list[str] = []
        for index in range(1, len(candles) - 3):
            features = rows[index]
            if features is None:
                continue
            decision = council.run_council(
                candles[index],
                candles[index - 1].close,
                features,
                None,
                weights=weight,
                spread_pips=1.2,
                session=None,
                regime="UNKNOWN",
                min_checks=5,
            )
            if decision.side in ("BUY", "SELL"):
                sides.append(decision.side)
        self.assertGreater(len(sides), 20, "no signals at all on this series")
        buys = sides.count("BUY")
        sells = sides.count("SELL")
        share = buys / len(sides)
        self.assertGreater(sells, 0, "the council never sold")
        self.assertLess(share, 0.75, f"BUY share {share:.0%} is lopsided")

    def test_no_registered_member_is_single_sided_by_construction(self) -> None:
        """A lone one-sided rule can only ever add mass to one side.

        Members that are deliberately one half of a pair (a bull flag beside a
        bear flag, an ascending triangle beside a descending one) are fine as
        long as the mirror is registered too; what must not happen is a
        one-sided rule registered on its own.
        """
        from fusion.strategies import PROTOTYPES, REGISTRY

        one_sided = {
            "smc_ascending_triangle": "smc_descending_triangle",
            "smc_bull_flag": "smc_bear_flag",
        }
        for name, mirror in one_sided.items():
            if name in PROTOTYPES:
                self.assertIn(mirror, PROTOTYPES, f"{name} has no {mirror} to balance it")
        self.assertNotIn("smc_bull_flag", PROTOTYPES, "the standalone one-sided bull flag is back")
        for name in PROTOTYPES:
            self.assertIn(name, REGISTRY)


def _looked_up_keys(modules) -> set[str]:
    """Feature keys each rule module actually reads via ``features.get(...)``."""
    import inspect
    import re

    keys: set[str] = set()
    for module in modules:
        source = inspect.getsource(module)
        keys.update(re.findall(r'features\.get\(\s*"([a-z0-9_]+)"', source))
    return keys


if __name__ == "__main__":
    unittest.main()
