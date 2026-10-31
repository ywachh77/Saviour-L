"""The pip is not dimensionless, so this is where a unit bug would hide.

Everything else in the engine is a ratio -- a projected move against a cost --
and ratios survive a change of market untouched. The pip does not. EUR/USD
has a 0.0001 pip; a $1 move on bitcoin would be 10,000 pips if the wrong pip
were used. These tests pin the pip size and the measured constants to their
sources so a drift is visible.

EUR/USD only -- crypto has been removed per the project constraint.
"""

import unittest
from datetime import datetime, timezone

from fusion import instruments, session
from fusion.config import Config, load_config
from fusion.domain import Candle, Settlement, Signal
from fusion.features import compute_features
from fusion.runner import format_result
from fusion.settle import settle
from fusion.simulate import synthetic_series


def _at(hour: int) -> datetime:
    return datetime(2026, 1, 5, hour, 0, tzinfo=timezone.utc)


class TestTheRegistry(unittest.TestCase):
    def test_every_symbol_resolves_to_itself(self):
        for symbol in instruments.symbols():
            self.assertEqual(symbol, instruments.for_symbol(symbol).symbol)

    def test_the_pip_size_is_the_documented_one(self):
        self.assertEqual(0.0001, instruments.EURUSD.pip)

    def test_eur_is_not_marked_crypto(self):
        self.assertFalse(instruments.EURUSD.crypto)
        self.assertEqual("fx", instruments.EURUSD.kind)

    def test_an_unknown_symbol_falls_back_rather_than_raising(self):
        """Reading an old row must not crash the settlement path."""
        self.assertIs(instruments.EURUSD, instruments.for_symbol("DOGE/USD"))
        self.assertIs(instruments.EURUSD, instruments.for_symbol(None))

    def test_but_an_unknown_symbol_is_not_allowed(self):
        """Falling back to read a row is not the same as permitting a market."""
        self.assertTrue(instruments.is_allowed("EUR/USD"))
        self.assertFalse(instruments.is_allowed("DOGE/USD"))
        self.assertFalse(instruments.is_allowed(""))
        self.assertFalse(instruments.is_allowed(None))

    def test_the_default_is_eurusd_so_existing_callers_are_unmoved(self):
        self.assertIs(instruments.EURUSD, instruments.DEFAULT)
        self.assertEqual(0.0001, instruments.pip_for(None))


class TestPipConversion(unittest.TestCase):
    def test_a_fourth_decimal_on_eurusd_is_one_pip(self):
        self.assertAlmostEqual(1.0, instruments.EURUSD.pips(0.0001))
        self.assertAlmostEqual(0.61, instruments.EURUSD.pips(0.000061))

    def test_price_and_pips_round_trip(self):
        for move in (1.0, 12.5, 0.25):
            self.assertAlmostEqual(move, instruments.EURUSD.price(instruments.EURUSD.pips(move)))

    def test_the_wrong_pip_is_wrong_by_four_orders_of_magnitude(self):
        """The bug this module prevents, stated as a number.

        A 0.0001 move read with a $1 pip is 0.0001 pips. If that ever stops
        being true, this test is no longer describing the risk.
        """
        wrong = 0.0001 / 1.0
        right = instruments.EURUSD.pips(0.0001)
        self.assertAlmostEqual(0.0001, wrong)
        self.assertAlmostEqual(1.0, right)
        self.assertAlmostEqual(10_000.0, right / wrong)


class TestTheMeasuredConstants(unittest.TestCase):
    """Pin the figures to their sources so a drift is visible.

    Every number here came from a quoted market figure recorded in
    instruments.py. They are point estimates and will go stale; what must not
    drift unnoticed is that they are measured, not invented.
    """

    def test_an_instrument_with_no_movement_is_infinitely_expensive(self):
        self.assertEqual(float("inf"), instruments.spread_to_atr("EUR/USD", 0.0))

    def test_every_instrument_documents_where_its_numbers_came_from(self):
        self.assertTrue(instruments.EURUSD.note.strip(), "EUR/USD has no provenance")


class TestTheSessionCalendar(unittest.TestCase):
    def test_fx_still_uses_its_windows(self):
        self.assertEqual(session.ASIA, session.session_of(_at(3), instruments.EURUSD))
        self.assertEqual(session.LONDON, session.session_of(_at(12), instruments.EURUSD))
        self.assertEqual(session.OVERLAP, session.session_of(_at(15), instruments.EURUSD))
        self.assertEqual(session.OFF_HOURS, session.session_of(_at(23), instruments.EURUSD))

    def test_the_round_the_clock_window_is_all_day(self):
        self.assertEqual((0, 24), session.window_of(session.ROUND_THE_CLOCK))

    def test_the_asia_window_still_mutes_what_it_mutes_on_fx(self):
        self.assertFalse(session.session_allows(session.ASIA, "trend_pullback"))
        self.assertTrue(session.session_allows(session.ASIA, "range_reversal"))

    def test_a_new_strategy_is_still_heard(self):
        self.assertTrue(session.session_allows(session.ROUND_THE_CLOCK, "not_a_strategy"))


class TestConfigResolution(unittest.TestCase):
    def test_the_instrument_follows_the_pair(self):
        for symbol in instruments.symbols():
            config = Config(pair=symbol)
            self.assertEqual(symbol, config.instrument.symbol)
            self.assertEqual(config.instrument.pip, instruments.pip_for(symbol))

    def test_the_property_cannot_drift_from_the_pair(self):
        config = Config(pair="EUR/USD")
        self.assertEqual(0.0001, config.instrument.pip)

    def test_an_unsupported_market_is_refused_loudly(self):
        import os
        from tests.helpers import preserved_environ

        with preserved_environ():
            os.environ["PAIR"] = "DOGE/USD"
            with self.assertRaises(ValueError) as caught:
                load_config(".env.nonexistent")
            self.assertIn("EUR/USD", str(caught.exception))

    def test_an_explicit_spread_still_wins(self):
        import os
        from tests.helpers import preserved_environ

        with preserved_environ():
            os.environ["PAIR"] = "EUR/USD"
            os.environ["SPREAD_PIPS"] = "20"
            self.assertEqual(20.0, load_config(".env.nonexistent").spread_pips)


class TestSettlementUsesTheRightPip(unittest.TestCase):
    def _signal(self, pair, entry_time, expiry_time):
        return Signal(
            signal_id="instrument-test-1",
            pair=pair,
            side="BUY",
            entry_time=entry_time,
            expiry_time=expiry_time,
            generated_at=entry_time,
            strategy="fusion",
            reason="test",
            source="LIVE",
            spread_pips=instruments.for_symbol(pair).spread_pips,
        )

    def _candles(self, start_price, prices):
        return [
            Candle(
                timestamp=datetime(2026, 1, 5, 12, i, tzinfo=timezone.utc),
                open=start_price + price,
                high=start_price + price + 1,
                low=start_price + price - 1,
                close=start_price + price,
                volume=1.0,
            )
            for i, price in enumerate(prices)
        ]

    def test_a_fourth_decimal_eurusd_move_is_one_pip(self):
        candles = self._candles(1.0800, [0.0, 0.0, 0.0005, 0.0005, 0.0005, 0.0005])
        signal = self._signal(
            "EUR/USD", "2026-01-05T12:00:00+00:00", "2026-01-05T12:05:00+00:00"
        )
        result = settle(signal, candles)
        self.assertAlmostEqual(5.0, result.move_pips, places=6)


class TestFeaturesCarryTheRightPip(unittest.TestCase):
    def test_atr_pips_defaults_to_the_fx_pip(self):
        """Every existing caller passes no instrument and must be unmoved."""
        rows = compute_features(synthetic_series(220, seed=5))
        last = rows[-1]
        self.assertAlmostEqual(last["atr"] / 0.0001, last["atr_pips"])

    def test_the_row_stays_numeric(self):
        """The persistence round trip depends on this.

        Signal.from_dict floats every value in the row, so anything that is
        not a number silently stops the signal from ever settling.
        """
        import json

        from fusion.domain import Signal

        candles = synthetic_series(220, seed=11)
        row = compute_features(candles, instrument=instruments.EURUSD)[-1]
        for key, value in row.items():
            self.assertIsInstance(value, (int, float), f"EUR/USD.{key}")
        rebuilt = Signal.from_dict(
            {
                "signal_id": "t",
                "side": "BUY",
                "entry_time": "2026-01-05T12:00:00+00:00",
                "expiry_time": "2026-01-05T12:05:00+00:00",
                "features": json.loads(json.dumps(row, default=str)),
            }
        )
        self.assertAlmostEqual(rebuilt.features["atr_pips"], row["atr_pips"])


class TestResultMessageUsesTheRightPrecision(unittest.TestCase):
    def _result(self, pair, entry, exit_):
        signal = Signal(
            signal_id="precision-test-1",
            pair=pair,
            side="BUY",
            entry_time="2026-01-05T12:00:00+00:00",
            expiry_time="2026-01-05T12:05:00+00:00",
            generated_at="2026-01-05T12:00:00+00:00",
            strategy="fusion",
            reason="test",
            source="LIVE",
            spread_pips=1.2,
        )
        settlement = Settlement(
            signal_id="precision-test-1",
            outcome="WIN",
            entry_price=entry,
            exit_price=exit_,
            gross_pips=5.0,
            move_pips=5.0,
            net_pips=3.8,
            spread_pips=1.2,
            settled_at="2026-01-05T12:06:00+00:00",
            source="LIVE",
        )
        return format_result(signal, settlement)

    def test_eurusd_keeps_its_five_decimals(self):
        text = self._result("EUR/USD", 1.08521, 1.08571)
        self.assertIn("1.08521", text)

    def test_an_unresolved_result_names_no_price(self):
        signal = Signal(
            signal_id="precision-test-2",
            pair="EUR/USD",
            side="BUY",
            entry_time="2026-01-05T12:00:00+00:00",
            expiry_time="2026-01-05T12:05:00+00:00",
            generated_at="2026-01-05T12:00:00+00:00",
            strategy="fusion",
            reason="test",
            source="LIVE",
        )
        settlement = Settlement(
            signal_id="precision-test-2",
            outcome="UNRESOLVED",
            entry_price=None,
            exit_price=None,
            gross_pips=None,
            move_pips=None,
            net_pips=None,
            spread_pips=1.2,
            settled_at="2026-01-05T12:06:00+00:00",
            source="LIVE",
        )
        text = format_result(signal, settlement)
        self.assertIn("UNRESOLVED", text)
        self.assertIn("EUR/USD", text)


if __name__ == "__main__":
    unittest.main()
