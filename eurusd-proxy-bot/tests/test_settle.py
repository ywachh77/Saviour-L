"""Settlement: real prices only, costs always, UNRESOLVED never invented."""

import unittest

from fusion import charter, settle
from fusion.domain import Candle, Signal
from fusion.timeutil import UTC, add_minutes

from .helpers import make_candles

PAIR = "EUR/USD"


def signal_at(stamp, side: str = "BUY", horizon: int = 3) -> Signal:
    return Signal.create(
        pair=PAIR,
        side=side,
        entry_time=stamp,
        strategy="fusion_council",
        reason="test",
        source="LIVE",
        generated_at=stamp,
        horizon_minutes=horizon,
        spread_pips=1.2,
    )


def ladder(start, prices: list[float]) -> list[Candle]:
    """Candles whose open and close both sit at the given price."""
    return [
        Candle(
            timestamp=add_minutes(start, index),
            open=price,
            high=price + 0.0002,
            low=price - 0.0002,
            close=price,
            volume=1000.0,
        )
        for index, price in enumerate(prices)
    ]


class TestSettlement(unittest.TestCase):
    def test_entry_is_the_open_of_the_named_candle(self):
        start = make_candles(1)[0].timestamp
        candles = ladder(start, [1.1000, 1.1010, 1.1020, 1.1030])
        signal = signal_at(add_minutes(start, 1))
        settlement = settle.settle(signal, candles)
        self.assertAlmostEqual(settlement.entry_price, 1.1010, places=6)
        self.assertAlmostEqual(settlement.exit_price, 1.1030, places=6)

    def test_buy_win_and_costs_are_charged(self):
        start = make_candles(1)[0].timestamp
        candles = ladder(start, [1.1000, 1.1000, 1.1000, 1.1003])
        signal = signal_at(add_minutes(start, 1))
        settlement = settle.settle(signal, candles)
        self.assertEqual(settlement.outcome, charter.WIN)
        self.assertAlmostEqual(settlement.gross_pips, 3.0, places=4)
        self.assertAlmostEqual(settlement.net_pips, 1.8, places=4)

    def test_sell_direction_is_inverted(self):
        start = make_candles(1)[0].timestamp
        candles = ladder(start, [1.1000, 1.1000, 1.1000, 1.0997])
        signal = signal_at(add_minutes(start, 1), side="SELL")
        settlement = settle.settle(signal, candles)
        self.assertEqual(settlement.outcome, charter.WIN)
        self.assertAlmostEqual(settlement.net_pips, 1.8, places=4)

    def test_a_small_win_can_be_a_net_loss(self):
        start = make_candles(1)[0].timestamp
        candles = ladder(start, [1.1000, 1.1000, 1.1000, 1.1001])
        signal = signal_at(add_minutes(start, 1))
        settlement = settle.settle(signal, candles)
        self.assertEqual(settlement.outcome, charter.WIN)  # direction was right
        self.assertLess(settlement.net_pips, 0.0)  # but it lost money

    def test_missing_prices_stay_unresolved(self):
        start = make_candles(1)[0].timestamp
        candles = ladder(start, [1.1000, 1.1000])
        signal = signal_at(add_minutes(start, 1))
        settlement = settle.settle(signal, candles)
        self.assertEqual(settlement.outcome, charter.UNRESOLVED)
        self.assertIsNone(settlement.net_pips)
        self.assertIsNone(settlement.gross_pips)
        self.assertFalse(settlement.resolved)

    def test_unresolved_is_described_not_claimed(self):
        start = make_candles(1)[0].timestamp
        candles = ladder(start, [1.1000])
        signal = signal_at(add_minutes(start, 1))
        settlement = settle.settle(signal, candles)
        text = settle.format_settlement(signal, settlement)
        self.assertIn("UNRESOLVED", text)
        self.assertNotIn("WIN", text)

    def test_a_tie_is_a_tie(self):
        start = make_candles(1)[0].timestamp
        candles = ladder(start, [1.1000, 1.1000, 1.1000, 1.1000])
        signal = signal_at(add_minutes(start, 1))
        settlement = settle.settle(signal, candles)
        self.assertEqual(settlement.outcome, charter.TIED)
        self.assertIsNone(settle.learning_label(signal, settlement))

    def test_learning_label_follows_price_direction_not_the_signal(self):
        start = make_candles(1)[0].timestamp
        candles = ladder(start, [1.1000, 1.1000, 1.1000, 1.0997])
        # A SELL that was right is still a DOWN bar for the model.
        signal = signal_at(add_minutes(start, 1), side="SELL")
        settlement = settle.settle(signal, candles)
        self.assertEqual(settle.learning_label(signal, settlement), 0)
        self.assertEqual(settle.winning_side(settlement), "SELL")


class TestLedger(unittest.TestCase):
    def _settlement(self, outcome: str, net: float):
        from datetime import datetime

        from fusion.domain import Settlement

        return Settlement(
            signal_id="x",
            outcome=outcome,
            entry_price=1.1,
            exit_price=1.1,
            gross_pips=None if net is None else net + 1.2,
            move_pips=None if net is None else net + 1.2,
            net_pips=net,
            spread_pips=1.2,
            settled_at=datetime.now(UTC),
            source="LIVE",
        )

    def test_summary_counts_unresolved_separately(self):
        rows = [
            self._settlement(charter.WIN, 2.0),
            self._settlement(charter.LOSS, -2.0),
            self._settlement(charter.UNRESOLVED, None),
        ]
        stats = settle.summarise(rows)
        self.assertEqual(stats.unresolved, 1)
        self.assertEqual(stats.wins, 1)
        self.assertEqual(stats.losses, 1)
        self.assertAlmostEqual(stats.net_pips, 0.0, places=6)

    def test_win_rate_excludes_unresolved_and_reports_wilson(self):
        rows = [self._settlement(charter.WIN, 1.0)] * 7 + [self._settlement(charter.LOSS, -1.0)] * 3
        rows += [self._settlement(charter.UNRESOLVED, None)] * 5
        stats = settle.summarise(rows)
        self.assertAlmostEqual(stats.win_rate, 0.7, places=9)
        self.assertLess(stats.wilson_low, 0.7)
        self.assertGreater(stats.wilson_high, 0.7)

    def test_empty_ledger_is_handled(self):
        stats = settle.summarise([])
        self.assertIsNone(stats.win_rate)
        self.assertEqual(stats.settled, 0)


if __name__ == "__main__":
    unittest.main()
