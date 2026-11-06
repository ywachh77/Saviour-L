"""Consecutive losses: what is arithmetic, and what was a defect.

The system had a nightly routine that refitted towards recent losses whenever the
win rate was under **0.95**. At a live hit rate near 0.50 that condition was
always true, so the retrain ran every night whether there was anything to learn
from or not. These tests pin three things:

1. the trigger fires only when the shortfall is *statistically real* against the
   rate the spread actually demands, not against an unreachable number;
2. the retrain is refused outright when the payoff cannot clear the spread,
   because no amount of refitting fixes an arithmetic problem;
3. the learning-rate scaling is anchored to a fair coin, not to 0.95.

None of this removes consecutive losses -- at a 50% hit rate they are expected,
and `scripts/loss_streak_study.py` shows a run of two losses is near certain in a
20-trade block. What is removable is a nightly process that was reacting to noise
as if it were signal.
"""

from __future__ import annotations

import unittest

from fusion import charter
from fusion.loss_patterns import calculate_loss_learning_rate, should_trigger_loss_learning


def _ledger(wins: int, losses: int, *, gross: float = 3.0, spread: float = 1.2) -> list[dict]:
    rows = []
    for index in range(wins + losses):
        win = index < wins
        rows.append(
            {
                "source": "LIVE",
                "outcome": charter.WIN if win else charter.LOSS,
                "gross_pips": gross if win else -gross,
                "net_pips": (gross - spread) if win else (-gross - spread),
                "spread_pips": spread,
            }
        )
    return rows


class TheTriggerMeasuresAgainstTheRealBreakEven(unittest.TestCase):
    def test_an_ordinary_losing_run_does_not_trigger_a_retrain(self) -> None:
        """9 of 20 at a 50% base rate is noise, not a signal to refit on.

        The 95% target made this fire every time. Derived from the trades, a
        12-pip mean move against 1.2 pips needs only 55%, so a 45% run over 20
        trades -- whose Wilson upper bound is 65.8% -- is plainly inconclusive
        and must not trigger a retrain.
        """
        triggered, reason = should_trigger_loss_learning(_ledger(9, 11, gross=12.0))
        self.assertFalse(triggered, reason)
        self.assertIn("noise", reason)

    def test_a_thin_payoff_makes_the_same_run_conclusive(self) -> None:
        """The identical 9-of-20 becomes a genuine signal when the move is thin.

        A 3-pip move against 1.2 pips needs 70%; the same 65.8% upper bound now
        sits below the line, so the shortfall is real rather than noise. The
        trigger has to distinguish these two cases, which is exactly what a
        fixed 0.95 could not do.
        """
        triggered, reason = should_trigger_loss_learning(_ledger(9, 11, gross=3.0))
        self.assertTrue(triggered, reason)
        self.assertIn("70.0%", reason)

    def test_a_shortfall_that_is_real_does_trigger(self) -> None:
        triggered, reason = should_trigger_loss_learning(_ledger(4, 96, gross=3.0))
        self.assertTrue(triggered, reason)
        self.assertIn("break-even", reason)

    def test_a_profitable_payoff_needs_a_lower_rate_than_a_thin_one(self) -> None:
        """The same win rate is judged differently depending on the payoff.

        That is the whole point of deriving the target: a 55% win rate on a 3-pip
        move against 1.2 pips is healthy, and on a 1.5-pip move it is not.
        """
        healthy, healthy_reason = should_trigger_loss_learning(_ledger(60, 40, gross=4.0))
        thin, thin_reason = should_trigger_loss_learning(_ledger(60, 40, gross=2.6))
        self.assertFalse(healthy, healthy_reason)
        self.assertTrue(thin, thin_reason)

    def test_it_refuses_to_refit_when_the_payoff_cannot_cover_the_spread(self) -> None:
        """No win rate rescues a trade whose average move is under its toll."""
        triggered, reason = should_trigger_loss_learning(
            _ledger(20, 20, gross=0.5, spread=1.2)
        )
        self.assertFalse(triggered)
        self.assertIn("no win rate is profitable", reason)

    def test_it_still_respects_a_caller_supplied_target(self) -> None:
        """An explicit target overrides the derived one -- and is held to the
        same statistical bar, so 9/20 against a 50% target is still noise."""
        noise, _ = should_trigger_loss_learning(_ledger(9, 11, gross=3.0), target_win_rate=0.50)
        real, reason = should_trigger_loss_learning(_ledger(2, 18, gross=3.0), target_win_rate=0.50)
        self.assertFalse(noise)
        self.assertTrue(real, reason)

    def test_it_waits_for_enough_trades_to_judge(self) -> None:
        triggered, reason = should_trigger_loss_learning(_ledger(1, 5, gross=3.0))
        self.assertFalse(triggered)
        self.assertIn("needed to judge", reason)

    def test_synthetic_trades_are_ignored(self) -> None:
        rows = _ledger(2, 98, gross=3.0)
        for row in rows:
            row["source"] = "SYNTHETIC"
        triggered, reason = should_trigger_loss_learning(rows)
        self.assertFalse(triggered)
        self.assertIn("needed to judge", reason)


class TheLearningRateIsAnchoredToOneHundredPercentWrong(unittest.TestCase):
    """Scaling against 0.95 pinned the multiplier at its cap on every run."""

    def test_a_fair_coin_gets_no_special_learning_rate(self) -> None:
        self.assertEqual(0.02, calculate_loss_learning_rate(0.02, 10, 0.50))

    def test_a_win_rate_above_a_coin_does_not_amplify(self) -> None:
        self.assertEqual(0.02, calculate_loss_learning_rate(0.02, 10, 0.95))

    def test_a_genuinely_poor_rate_does_amplify(self) -> None:
        amplified = calculate_loss_learning_rate(0.02, 10, 0.20)
        self.assertGreater(amplified, 0.02)

    def test_it_stays_capped(self) -> None:
        self.assertLessEqual(calculate_loss_learning_rate(0.5, 500, 0.0), 0.12)


class StreaksAreArithmetic(unittest.TestCase):
    """Pinned so nobody "fixes" consecutive losses by suppressing the counter.

    At the measured ~50% hit rate, streaks are a property of the hit rate. These
    numbers are the reason a run of losses is not, on its own, evidence of a bug.
    """

    @staticmethod
    def _prob_of_run(k: int, trades: int, p_win: float) -> float:
        states = [[0.0] * k for _ in range(trades + 1)]
        states[0][0] = 1.0
        hit = 0.0
        for position in range(1, trades + 1):
            for run in range(k):
                weight = states[position - 1][run]
                if weight <= 0.0:
                    continue
                states[position][0] += weight * p_win
                if run + 1 >= k:
                    hit += weight * (1.0 - p_win)
                else:
                    states[position][run + 1] += weight * (1.0 - p_win)
        return hit

    def test_two_consecutive_losses_in_twenty_trades_is_the_normal_case(self) -> None:
        probability = self._prob_of_run(2, 20, 0.50)
        self.assertGreater(probability, 0.90, "a 2-loss run in 20 trades should be near-certain")
        self.assertLess(probability, 1.0)

    def test_a_five_loss_streak_is_expected_within_a_few_hundred_trades(self) -> None:
        self.assertGreater(self._prob_of_run(5, 500, 0.50), 0.95)

    def test_a_better_hit_rate_reduces_but_never_removes_streaks(self) -> None:
        at_coin = self._prob_of_run(3, 100, 0.50)
        at_edge = self._prob_of_run(3, 100, 0.60)
        self.assertLess(at_edge, at_coin)
        self.assertGreater(at_edge, 0.0, "even a 60% rate produces 3-loss runs")


if __name__ == "__main__":
    unittest.main()
