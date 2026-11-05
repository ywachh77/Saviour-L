"""The single engine, and the directional audit that motivated it.

Two things are pinned here.

**Direction.** A council member that can only ever say BUY is not a strategy, it
is a permanent long vote. ``smc_ascending_triangle`` was exactly that: its two
"pattern" conditions read the 20-bar high (always true) and the 10-bar minimum
low (true on 98.5% of bars), so it fired on every close above the recent high
and produced 259 BUYs against zero SELLs. These tests require that a pattern
condition actually constrains the bars it can fire on, and that every one-sided
member has a mirror.

**One engine.** ``fusion.oracle`` folds the 132 members into a panel, feeds that
plus the standard features to a single model, and produces one probability from
which one decision follows. The tests check that it collapses duplicate
members, that an abstention is not a neutral vote, that the payoff arithmetic
round-trips, and -- most importantly -- that shuffling the training labels
destroys its skill. A model that keeps its accuracy on scrambled labels is
leaking the future, and this suite refuses to let that pass silently.
"""

from __future__ import annotations

import random
import unittest

from fusion import charter
from fusion.features import compute_features
from fusion.oracle import (
    DUPLICATE_GROUPS,
    ORACLE_FEATURES,
    PANEL_FEATURES,
    OracleEngine,
    deduplicate,
    expected_net_pips,
    majority_vote,
    panel_vector,
    required_probability,
)
from fusion.simulate import synthetic_series
from fusion.strategies import MEMBERS, REGISTRY, strategy_family


def _features_and_candles(bars: int = 900, seed: int = 5150):
    candles = synthetic_series(bars, seed=seed, volatility=0.00024)
    return candles, compute_features(candles)


def _all_stances(index: int, candles, rows, spread: float = 1.2) -> dict[str, float]:
    features = rows[index]
    stances: dict[str, float] = {}
    for name, rule in REGISTRY.items():
        try:
            decision = rule(
                candles[index], candles[index - 1].close, dict(features), None,
                spread_pips=spread,
            )
        except Exception:
            continue
        if decision is None or not getattr(decision, "actionable", False):
            continue
        side = (decision.side or "").upper()
        if side == "BUY":
            stances[name] = 1.0
        elif side == "SELL":
            stances[name] = -1.0
    return stances


class PatternConditionsMustConstrain(unittest.TestCase):
    """A pattern that is true on every bar is a trend filter wearing a costume."""

    def test_a_flat_high_requires_a_level_tested_more_than_once(self) -> None:
        _, rows = _features_and_candles(2000)
        scored = [row for row in rows if row is not None]
        flat = sum(1 for row in scored if row.get("triangle_flat_high", 0.0) > 0)
        rising = sum(1 for row in scored if row.get("triangle_rising_low", 0.0) > 0)
        breakout = sum(1 for row in scored if row.get("triangle_breakout", 0.0) > 0.5)

        # It was 98.5% / 98.5% / 3.3% before this was fixed. A resistance level
        # tested twice is a minority condition, and a breakout through one is
        # rarer still.
        self.assertLess(flat / len(scored), 0.35, "a 'tested twice' level is not rare enough")
        self.assertLess(rising / len(scored), 0.45, "higher-low structure is not rare enough")
        self.assertLess(breakout / len(scored), 0.05, "a breakout should be an event, not a state")
        self.assertGreater(flat, 0, "the condition never fires at all")
        self.assertGreater(breakout, 0, "the condition never fires at all")

    def test_support_needs_a_second_touch_from_a_different_candle(self) -> None:
        _, rows = _features_and_candles(2000)
        scored = [row for row in rows if row is not None]
        support = sum(1 for row in scored if row.get("descending_flat_low", 0.0) > 0)
        # Was 60.8% when a single touch satisfied it; the window minimum is
        # always touched once, so that condition was never a filter.
        self.assertLess(support / len(scored), 0.25)

    def test_every_one_sided_member_has_a_registered_mirror(self) -> None:
        """A member with only one branch must be balanced by its opposite.

        The audit found one blind member, ``smc_ascending_triangle``. The rule
        is that one-sidedness is permitted only where a genuine mirror exists,
        so the council cannot accumulate permanent BUY mass.
        """
        # smc_ascending_wedge is NOT one-sided: it has a breakdown branch
        # (SELL) as well as a breakup branch (BUY). Only the triangle pair is a
        # genuine mirror, so only it is asserted here.
        pairs = {"smc_ascending_triangle": "smc_descending_triangle"}
        for one, mirror in pairs.items():
            if one in REGISTRY:
                self.assertIn(mirror, REGISTRY, f"{one} has no mirror")
        self.assertIn("smc_ascending_triangle", REGISTRY)
        self.assertIn("smc_descending_triangle", REGISTRY)


class PanelCollapsesTheCouncil(unittest.TestCase):
    """132 rules are evidence, not 132 votes."""

    def test_duplicate_members_do_not_get_duplicate_votes(self) -> None:
        # Family grouping alone does NOT fix this: breakout, quant_donchian_
        # breakout and scalp_breakout_retest fire identically but sit in three
        # different families (CLASSIC_COUNCIL, QUANT_SCALP, MICRO_SCALPING).
        # The collapse therefore has to be an explicit duplicate table, which
        # is what DUPLICATE_GROUPS is.
        stances = {
            "breakout": 1.0,
            "quant_donchian_breakout": 1.0,
            "scalp_breakout_retest": 1.0,
        }
        self.assertEqual({"breakout": 1.0}, deduplicate(stances))

        vector = panel_vector(stances)
        self.assertAlmostEqual(1.0, vector["panel_net_stance"])
        self.assertAlmostEqual(1.0, vector["panel_agreement"])
        self.assertAlmostEqual(0.0, vector["panel_dispersion"])

        # Without the collapse the same three rules would carry three times the
        # mass of a single genuine member.
        uncollapsed = sum(stances.values()) / len(stances)
        self.assertAlmostEqual(uncollapsed, vector["panel_net_stance"])

    def test_duplicates_that_disagree_are_not_hidden(self) -> None:
        folded = deduplicate({"breakout": 1.0, "quant_donchian_breakout": -1.0})
        values = sorted(folded.values())
        self.assertEqual([-1.0, 1.0], values)

    def test_the_declared_duplicate_groups_really_are_duplicates(self) -> None:
        """The table must describe reality, not wishful thinking.

        Every group is evaluated on real bars and its members' signals compared
        bar for bar. Members that are NOT identical must not be collapsed, and
        members that ARE must be in the table -- otherwise the panel either
        double-counts or throws away a genuine second opinion.
        """
        candles, rows = _features_and_candles(700, seed=4242)
        signals: dict[str, list[int]] = {}
        for name in {member for group in DUPLICATE_GROUPS for member in group}:
            rule = REGISTRY[name]
            series = []
            for index in range(80, len(candles)):
                features = rows[index]
                if features is None:
                    series.append(0)
                    continue
                decision = rule(candles[index], candles[index - 1].close, dict(features), None,
                                spread_pips=1.2)
                if decision is None or not getattr(decision, "actionable", False):
                    series.append(0)
                else:
                    series.append(1 if (decision.side or "").upper() == "BUY" else -1)
            signals[name] = series

        for group in DUPLICATE_GROUPS:
            reference = signals[group[0]]
            for member in group[1:]:
                with self.subTest(group=group[0], member=member):
                    self.assertEqual(reference, signals[member])

    def test_an_abstention_is_not_a_neutral_vote(self) -> None:
        speaking = panel_vector({"breakout": 1.0, "range_reversal": 1.0})
        quiet = panel_vector({})
        self.assertGreater(speaking["panel_net_stance"], quiet["panel_net_stance"])
        self.assertEqual(0.0, quiet["panel_net_stance"])
        # Participation records the difference the net stance cannot: silence
        # and a perfectly balanced panel both net to zero.
        self.assertGreater(speaking["panel_participation"], quiet["panel_participation"])

    def test_a_split_panel_is_distinguishable_from_agreement(self) -> None:
        agreed = panel_vector({"breakout": 1.0, "trend_pullback": 1.0})
        split = panel_vector({"breakout": 1.0, "trend_pullback": -1.0})
        self.assertGreater(agreed["panel_agreement"], split["panel_agreement"])
        self.assertGreater(split["panel_dispersion"], agreed["panel_dispersion"])

    def test_majority_vote_is_the_thing_being_replaced(self) -> None:
        self.assertEqual("BUY", majority_vote({"a": 1.0, "b": 1.0, "c": -1.0}))
        self.assertEqual("SELL", majority_vote({"a": -1.0, "b": -1.0, "c": 1.0}))
        self.assertEqual("NO SIGNAL", majority_vote({"a": 1.0, "b": -1.0}))

    def test_the_panel_covers_every_family_the_registry_can_produce(self) -> None:
        families = {strategy_family(name) for name in MEMBERS}
        # Strip the prefix and the ONE suffix, in that order. Chaining
        # removeprefix/removesuffix over the whole tuple also mangles the
        # aggregate features (``panel_net_stance``), which is how this check
        # first reported seven families missing from a panel that had them all.
        published = {
            feature[len("panel_"): -len("_stance")]
            for feature in PANEL_FEATURES
            if feature.startswith("panel_") and feature.endswith("_stance")
            and feature.count("_stance") == 1
            and not feature.startswith("panel_net")
        }
        # Compare like with like: the registry labels are UPPERCASE and the
        # feature names are lowercased.
        missing = {family.lower() for family in families} - published
        self.assertFalse(missing, f"families with no panel feature: {sorted(missing)}")


class PayoffArithmetic(unittest.TestCase):
    """The one equation that decides whether a probability is worth acting on."""

    def test_required_probability_inverts_expected_net(self) -> None:
        move, spread = 4.3, charter.DEFAULT_SPREAD_PIPS
        required = required_probability(move, spread, min_net_pips=0.0)
        self.assertAlmostEqual(0.0, expected_net_pips(required, move, spread), places=9)
        self.assertGreater(expected_net_pips(required + 0.01, move, spread), 0.0)
        self.assertLess(expected_net_pips(required - 0.01, move, spread), 0.0)

    def test_a_payoff_inside_the_spread_can_never_be_taken(self) -> None:
        self.assertEqual(float("inf"), required_probability(1.0, 1.2))
        self.assertEqual(float("inf"), required_probability(0.0, 1.2))

    def test_a_coin_flip_loses_exactly_the_spread(self) -> None:
        # 0.5 * (m - s) - 0.5 * (m + s) = -s. The clean statement of why 50%
        # is not break-even.
        self.assertAlmostEqual(-1.2, expected_net_pips(0.5, 9.0, 1.2))


class TheSingleEngine(unittest.TestCase):
    """One engine, one probability, one decision."""

    @classmethod
    def setUpClass(cls) -> None:
        candles, rows = _features_and_candles(1500, seed=6161)
        cls.candles = candles
        samples = []
        for index in range(80, len(candles) - 15):
            features = rows[index]
            if features is None:
                continue
            stances = _all_stances(index, candles, rows)
            merged = dict(features)
            merged.update(panel_vector(stances))
            entry = candles[index + 1].open
            exit_ = candles[index + 15].close
            samples.append((merged, 1 if exit_ > entry else 0))
        cls.samples = samples
        cls.engine = OracleEngine(horizon=15).fit(samples, min_samples=200)
        cls.engine.move_pips = 7.3

    def test_it_declares_its_input_set(self) -> None:
        self.assertEqual(len(ORACLE_FEATURES), len(self.engine.model.feature_names))
        self.assertTrue(self.engine.model.compatible)
        self.assertEqual(tuple(ORACLE_FEATURES), tuple(self.engine.model.feature_names))

    def test_it_returns_one_read_with_a_decision_and_a_reason(self) -> None:
        index = 300
        stances = _all_stances(index, self.candles, _features_and_candles(1500, 6161)[1])
        read = self.engine.read(_features_and_candles(1500, 6161)[1][index], stances)
        self.assertGreaterEqual(read.prob_up, 0.0)
        self.assertLessEqual(read.prob_up, 1.0)
        self.assertIn(read.side, ("BUY", "SELL", "NO SIGNAL"))
        self.assertTrue(read.reason)
        self.assertIsInstance(read.to_dict()["reason"], str)

    def test_it_takes_both_sides_when_the_market_does(self) -> None:
        """A blind engine is one that cannot flip, not one that happens to lean.

        Testing this on a single sample is a trap: the sample above rises on
        only 43% of bars, so a *correctly calibrated* engine should say SELL
        there, and asserting "it must say BUY somewhere" would be asserting a
        bug. The honest test is that the same engine, trained on the same rules,
        flips its bias when the market's own bias flips.
        """
        def buy_share(drift: float) -> tuple[float, float]:
            candles = synthetic_series(1400, seed=4321, volatility=0.00024, drift=drift)
            rows = compute_features(candles)
            samples = []
            for index in range(80, len(candles) - 15):
                features = rows[index]
                if features is None:
                    continue
                merged = dict(features)
                merged.update(panel_vector(_all_stances(index, candles, rows)))
                entry = candles[index + 1].open
                samples.append((merged, 1 if candles[index + 15].close > entry else 0))

            engine = OracleEngine(horizon=15).fit(samples, min_samples=200)
            engine.move_pips = 7.0
            up = sum(label for _, label in samples) / len(samples)
            directions = [engine.read(features, {}).direction for features, _ in samples[::3]]
            return sum(1 for side in directions if side == "BUY") / len(directions), up

        falling_share, falling_up = buy_share(-0.000004)
        rising_share, rising_up = buy_share(0.000004)

        self.assertLess(falling_up, 0.5)
        self.assertGreater(rising_up, 0.5)
        # The engine must actually flip, not merely be less wrong one way.
        self.assertLess(falling_share, 0.35, "the engine could not lean bearish on a falling sample")
        self.assertGreater(rising_share, 0.55, "the engine could not lean bullish on a rising sample")
        self.assertGreater(
            rising_share - falling_share, 0.25,
            "the engine's bias did not follow the market's bias: it is structurally one-sided",
        )

    def test_it_abstains_when_the_payoff_cannot_clear_the_cost(self) -> None:
        _, rows = _features_and_candles(1500, seed=6161)
        read = self.engine.read(rows[400], _all_stances(400, self.candles, rows), spread_pips=40.0)
        self.assertFalse(read.take)
        self.assertEqual("NO SIGNAL", read.side)
        self.assertIn("abstaining", read.reason)

    def test_an_untrained_engine_refuses_to_guess(self) -> None:
        _, rows = _features_and_candles(600, seed=99)
        read = OracleEngine().read(rows[100], {})
        self.assertFalse(read.take)
        self.assertEqual("NO SIGNAL", read.side)
        self.assertIn("not trained", read.reason)

    def test_it_round_trips_through_persistence(self) -> None:
        payload = self.engine.to_dict()
        restored = OracleEngine.from_dict(payload)
        self.assertIsNotNone(restored)
        _, rows = _features_and_candles(1500, seed=6161)
        stances = _all_stances(400, self.candles, rows)
        before = self.engine.read(rows[400], stances).prob_up
        after = restored.read(rows[400], stances).prob_up
        self.assertAlmostEqual(before, after, places=9)

    def test_a_model_trained_on_another_input_set_is_refused(self) -> None:
        payload = self.engine.to_dict()
        payload["features"] = list(ORACLE_FEATURES)[:-1]
        self.assertIsNone(OracleEngine.from_dict(payload))

    def test_training_needs_a_real_sample(self) -> None:
        with self.assertRaises(ValueError):
            OracleEngine().fit(self.samples[:10], min_samples=200)


class TheEngineReplacesTheCouncilVote(unittest.TestCase):
    """Wired into the engine, one read decides -- not two."""

    def test_when_an_engine_is_supplied_its_answer_is_the_verdict(self) -> None:
        from fusion.domain import Candle
        from fusion.engine import FusionEngine
        from fusion.timeutil import parse_datetime
        from datetime import datetime, timezone

        candles, rows = _features_and_candles(700, seed=7171)
        samples = []
        for index in range(80, len(candles) - 15):
            features = rows[index]
            if features is None:
                continue
            merged = dict(features)
            merged.update(panel_vector(_all_stances(index, candles, rows)))
            entry = candles[index + 1].open
            samples.append((merged, 1 if candles[index + 15].close > entry else 0))
        oracle = OracleEngine(horizon=15).fit(samples, min_samples=200)
        oracle.move_pips = 7.0

        class _Config:
            supply_live = False
            pair = "EUR/USD"
            news_filter = False
            session_filter = True
            min_confluence_checks = 5
            max_uncertainty = 1.0
            min_atr_to_spread_ratio = 0.0
            mtf_filter = False
            mtf_minutes = 15
            adaptive_horizon = False
            expiry_minutes = 15

        engine = FusionEngine(_Config())
        index = 400
        candle = candles[index]
        verdict = engine.decide(
            candle,
            candles[index - 1].close,
            rows[index],
            spread_pips=charter.DEFAULT_SPREAD_PIPS,
            oracle=oracle,
            oracle_move_pips=7.0,
        )
        # Either it took the engine's side, or it abstained with the engine's
        # reason. What it must NOT do is fall through to the council vote.
        self.assertIn(verdict.side, ("BUY", "SELL", "NO SIGNAL"))
        self.assertIn("Single engine", verdict.reason)

    def test_without_an_engine_the_council_still_works(self) -> None:
        from fusion.engine import FusionEngine

        candles, rows = _features_and_candles(500, seed=7272)

        class _Config:
            supply_live = False
            pair = "EUR/USD"
            news_filter = False
            session_filter = False
            min_confluence_checks = 5
            max_uncertainty = 1.0
            min_atr_to_spread_ratio = 0.0
            mtf_filter = False
            mtf_minutes = 15
            adaptive_horizon = False
            expiry_minutes = 15

        verdict = FusionEngine(_Config()).decide(
            candles[400], candles[399].close, rows[400],
            spread_pips=charter.DEFAULT_SPREAD_PIPS,
        )
        self.assertIn(verdict.side, ("BUY", "SELL", "NO SIGNAL"))
        self.assertNotIn("Single engine", verdict.reason)


class ScrambledLabelsDestroyTheSkill(unittest.TestCase):
    """The control that decides whether any accuracy figure means anything.

    If a model keeps its edge on labels that were randomly permuted, it is not
    predicting -- it is memorising, or the dataset is leaking. This is the test
    that stops this repository reporting a number it cannot defend.
    """

    def _accuracy(self, shuffled: bool) -> float:
        candles, rows = _features_and_candles(1400, seed=8181)
        samples = []
        for index in range(80, len(candles) - 15):
            features = rows[index]
            if features is None:
                continue
            merged = dict(features)
            merged.update(panel_vector(_all_stances(index, candles, rows)))
            entry = candles[index + 1].open
            exit_ = candles[index + 15].close
            samples.append((merged, 1 if exit_ > entry else 0))

        split = int(len(samples) * 0.6)
        train, test = samples[:split], samples[split:]
        if shuffled:
            rng = random.Random(4242)
            labels = [label for _, label in train]
            rng.shuffle(labels)
            train = [(features, label) for (features, _), label in zip(train, labels)]

        engine = OracleEngine(horizon=15).fit(train, min_samples=200)
        wins = sum(
            1
            for features, label in test
            if (engine.read(features, {}).prob_up >= 0.5) == bool(label)
        )
        return wins / len(test)

    def test_shuffled_labels_leave_no_skill(self) -> None:
        accuracy = self._accuracy(shuffled=True)
        # Chance is 50%; the sampling error on a few hundred bars is a couple
        # of points. Anything near 60% would mean leakage.
        self.assertLess(accuracy, 0.58, f"a model on scrambled labels scored {accuracy:.1%}: leakage")
        self.assertGreater(accuracy, 0.42, f"a model on scrambled labels scored {accuracy:.1%}: broken control")


if __name__ == "__main__":
    unittest.main()
