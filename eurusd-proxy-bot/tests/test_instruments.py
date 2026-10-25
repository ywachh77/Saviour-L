"""The pip is not dimensionless, so this is where a unit bug would hide.

Everything else in the engine is a ratio -- a projected move against a cost --
and ratios survive a change of market untouched. The pip does not. Dividing a
bitcoin ATR by the FX pip turns a 58 dollar move into 580,000 "pips", and every
number downstream of it becomes nonsense that still looks like a number.

These tests exist to make that specific failure impossible, and to pin the
measured constants to their sources.
"""

import os
import unittest
from datetime import datetime, timezone

from fusion import instruments, session
from fusion.config import Config, load_config
from fusion.domain import Candle, Settlement, Signal
from fusion.features import compute_features
from fusion.runner import format_result
from fusion.settle import settle
from fusion.simulate import synthetic_series
from tests.helpers import preserved_environ


def _at(hour: int) -> datetime:
    return datetime(2026, 1, 5, hour, 0, tzinfo=timezone.utc)


class TestTheRegistry(unittest.TestCase):
    def test_every_symbol_resolves_to_itself(self):
        for symbol in instruments.symbols():
            self.assertEqual(symbol, instruments.for_symbol(symbol).symbol)

    def test_the_pip_sizes_are_the_documented_ones(self):
        self.assertEqual(0.0001, instruments.EURUSD.pip)
        self.assertEqual(1.0, instruments.BTCUSD.pip)
        self.assertEqual(0.1, instruments.ETHUSD.pip)

    def test_crypto_is_marked_and_fx_is_not(self):
        self.assertTrue(instruments.BTCUSD.crypto)
        self.assertTrue(instruments.ETHUSD.crypto)
        self.assertFalse(instruments.EURUSD.crypto)
        self.assertEqual("crypto", instruments.BTCUSD.kind)
        self.assertEqual("fx", instruments.EURUSD.kind)

    def test_an_unknown_symbol_falls_back_rather_than_raising(self):
        """Reading an old row must not crash the settlement path."""
        self.assertIs(instruments.EURUSD, instruments.for_symbol("DOGE/USD"))
        self.assertIs(instruments.EURUSD, instruments.for_symbol(None))

    def test_but_an_unknown_symbol_is_not_allowed(self):
        """Falling back to read a row is not the same as permitting a market."""
        self.assertTrue(instruments.is_allowed("BTC/USD"))
        self.assertFalse(instruments.is_allowed("DOGE/USD"))
        self.assertFalse(instruments.is_allowed(""))
        self.assertFalse(instruments.is_allowed(None))

    def test_the_default_is_eurusd_so_existing_callers_are_unmoved(self):
        self.assertIs(instruments.EURUSD, instruments.DEFAULT)
        self.assertEqual(0.0001, instruments.pip_for(None))


class TestPipConversion(unittest.TestCase):
    def test_a_dollar_move_on_bitcoin_is_one_pip(self):
        self.assertAlmostEqual(1.0, instruments.BTCUSD.pips(1.0))
        self.assertAlmostEqual(58.0, instruments.BTCUSD.pips(58.0))

    def test_a_fourth_decimal_on_eurusd_is_one_pip(self):
        self.assertAlmostEqual(1.0, instruments.EURUSD.pips(0.0001))
        self.assertAlmostEqual(0.61, instruments.EURUSD.pips(0.000061))

    def test_price_and_pips_round_trip(self):
        for item in (instruments.EURUSD, instruments.BTCUSD, instruments.ETHUSD):
            for move in (1.0, 12.5, 0.25):
                self.assertAlmostEqual(move, item.price(item.pips(move)))

    def test_the_same_relative_move_is_the_same_number_of_pips(self):
        """Two pips of EUR/USD and two pips of BTC are both two pips.

        That is the whole point of a per-instrument pip: the unit changes, the
        arithmetic on top of it does not.
        """
        self.assertAlmostEqual(
            instruments.EURUSD.pips(0.0002), instruments.BTCUSD.pips(2.0)
        )

    def test_using_the_wrong_pip_is_wrong_by_four_orders_of_magnitude(self):
        """The bug this module prevents, stated as a number.

        A $58 bitcoin move read with the FX pip is 580,000 pips. If that ever
        stops being true, this test is no longer describing the risk.
        """
        wrong = 58.0 / instruments.EURUSD.pip
        right = instruments.BTCUSD.pips(58.0)
        self.assertAlmostEqual(580_000.0, wrong)
        self.assertAlmostEqual(58.0, right)
        self.assertAlmostEqual(10_000.0, wrong / right)


class TestTheMeasuredConstants(unittest.TestCase):
    """Pin the figures to their sources so a drift is visible.

    Every number here came from a quoted market figure recorded in
    instruments.py. They are point estimates from late October 2026 and will
    go stale; what must not drift unnoticed is that they are measured, not
    invented.
    """

    def test_crypto_is_far_cheaper_to_trade_than_fx_on_one_minute_bars(self):
        """The reason crypto is worth adding at all.

        Cost as a fraction of one-minute movement. Lower is better. This is
        not a claim that crypto is easier to predict -- it is a claim that the
        spread does not eat the move before it can be measured.
        """
        fx = instruments.spread_to_atr("EUR/USD", 0.61)
        btc = instruments.spread_to_atr("BTC/USD", 58.0)
        eth = instruments.spread_to_atr("ETH/USD", 23.0)

        self.assertAlmostEqual(1.97, fx, places=1)
        self.assertAlmostEqual(0.14, btc, places=1)
        self.assertAlmostEqual(0.10, eth, places=1)

        self.assertLess(btc, fx / 10)
        self.assertLess(eth, fx / 10)

    def test_the_derived_spreads_match_a_tenth_of_a_percent(self):
        """0.01% of price, the figure the exchanges were quoting."""
        self.assertAlmostEqual(8.05, 0.0001 * 80_500, places=1)
        self.assertAlmostEqual(8.0, instruments.BTCUSD.spread_pips, delta=0.1)
        self.assertAlmostEqual(0.238, 0.0001 * 2_380, places=2)
        self.assertAlmostEqual(2.4, instruments.ETHUSD.spread_pips, delta=0.1)

    def test_an_instrument_with_no_movement_is_infinitely_expensive(self):
        self.assertEqual(float("inf"), instruments.spread_to_atr("BTC/USD", 0.0))

    def test_every_instrument_documents_where_its_numbers_came_from(self):
        for item in (instruments.EURUSD, instruments.BTCUSD, instruments.ETHUSD):
            self.assertTrue(item.note.strip(), f"{item.symbol} has no provenance")


class TestTheSessionCalendar(unittest.TestCase):
    def test_fx_still_uses_its_windows(self):
        self.assertEqual(session.ASIA, session.session_of(_at(3), instruments.EURUSD))
        self.assertEqual(session.LONDON, session.session_of(_at(12), instruments.EURUSD))
        self.assertEqual(session.OVERLAP, session.session_of(_at(15), instruments.EURUSD))
        self.assertEqual(session.OFF_HOURS, session.session_of(_at(23), instruments.EURUSD))

    def test_crypto_has_no_sessions_at_any_hour(self):
        """A London open for bitcoin would be invented structure."""
        for hour in range(24):
            self.assertEqual(
                session.ROUND_THE_CLOCK,
                session.session_of(_at(hour), instruments.BTCUSD),
                f"hour {hour} produced an FX session for crypto",
            )
            self.assertEqual(
                session.ROUND_THE_CLOCK,
                session.session_of(_at(hour), instruments.ETHUSD),
            )

    def test_the_round_the_clock_window_is_all_day(self):
        self.assertEqual((0, 24), session.window_of(session.ROUND_THE_CLOCK))

    def test_no_strategy_is_muted_on_crypto(self):
        """With no session structure, muting by hour would be silencing for
        no reason at all."""
        for name in session.ALLOWED[session.ROUND_THE_CLOCK]:
            self.assertTrue(session.session_allows(session.ROUND_THE_CLOCK, name))
        self.assertTrue(session.session_allows(session.ROUND_THE_CLOCK, "trend_pullback"))
        self.assertTrue(session.session_allows(session.ROUND_THE_CLOCK, "scalp_impulse"))
        self.assertTrue(session.session_allows(session.ROUND_THE_CLOCK, "range_reversal"))

    def test_the_asia_window_still_mutes_what_it_mutes_on_fx(self):
        """Adding crypto must not quietly loosen the FX rules."""
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
        config.pair = "BTC/USD"
        self.assertEqual(1.0, config.instrument.pip)

    def test_loading_a_crypto_config_picks_up_the_crypto_spread(self):
        with preserved_environ():
            for symbol, expected in (
                ("EUR/USD", 1.2),
                ("BTC/USD", 8.0),
                ("ETH/USD", 2.4),
            ):
                os.environ["PAIR"] = symbol
                # No SPREAD_PIPS anywhere: the instrument has to supply it.
                os.environ.pop("SPREAD_PIPS", None)
                self.assertAlmostEqual(expected, load_config(".env.nonexistent").spread_pips)

    def test_an_unsupported_market_is_refused_loudly(self):
        with preserved_environ():
            os.environ["PAIR"] = "DOGE/USD"
            with self.assertRaises(ValueError) as caught:
                load_config(".env.nonexistent")
            self.assertIn("BTC/USD", str(caught.exception))

    def test_the_pair_is_matched_case_insensitively(self):
        with preserved_environ():
            os.environ["PAIR"] = "btc/usd"
            self.assertEqual("BTC/USD", load_config(".env.nonexistent").pair)

    def test_an_explicit_spread_still_wins(self):
        with preserved_environ():
            os.environ["PAIR"] = "BTC/USD"
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

    def test_a_one_dollar_bitcoin_move_is_one_pip(self):
        """The headline case. A $1 win must be recorded as 1 pip, not 10,000."""
        candles = self._candles(80_000.0, [0.0, 0.0, 5.0, 5.0, 5.0, 5.0])
        signal = self._signal(
            "BTC/USD", "2026-01-05T12:00:00+00:00", "2026-01-05T12:05:00+00:00"
        )
        # Entry is the open of 12:00, exit is the close of 12:04.
        result = settle(signal, candles)
        self.assertEqual("WIN", result.outcome)
        self.assertAlmostEqual(5.0, result.move_pips, places=6)
        self.assertAlmostEqual(5.0, result.gross_pips, places=6)

    def test_a_fourth_decimal_eurusd_move_is_one_pip(self):
        candles = self._candles(1.0800, [0.0, 0.0, 0.0005, 0.0005, 0.0005, 0.0005])
        signal = self._signal(
            "EUR/USD", "2026-01-05T12:00:00+00:00", "2026-01-05T12:05:00+00:00"
        )
        result = settle(signal, candles)
        self.assertAlmostEqual(5.0, result.move_pips, places=6)

    def test_ether_settles_in_dimes(self):
        candles = self._candles(2_380.0, [0.0, 0.0, 1.5, 1.5, 1.5, 1.5])
        signal = self._signal(
            "ETH/USD", "2026-01-05T12:00:00+00:00", "2026-01-05T12:05:00+00:00"
        )
        result = settle(signal, candles)
        self.assertAlmostEqual(15.0, result.move_pips, places=6)

    def test_the_same_relative_move_costs_the_same_number_of_spreads(self):
        """The invariant is the move measured in spreads, not in pips.

        A pip is a fixed price increment, and it is *meant* to differ by
        market -- that is the whole point. So a 0.05% move is 5.4 EUR/USD
        pips and 40 bitcoin pips, and those numbers should not match. What has
        to match is the move expressed in the cost of getting in, because that
        is the quantity the strategy actually reasons about.

        If this fails, some part of the settlement path is still using an
        absolute pip instead of the instrument's.
        """
        cases = (
            ("EUR/USD", 1.0800, 0.00054),
            ("BTC/USD", 80_000.0, 40.0),
            ("ETH/USD", 2_380.0, 1.19),
        )
        in_spreads = []
        for pair, base, move in cases:
            candles = self._candles(base, [0.0, 0.0, move, move, move, move])
            signal = self._signal(pair, "2026-01-05T12:00:00+00:00", "2026-01-05T12:05:00+00:00")
            result = settle(signal, candles)
            # Every case moves 0.05% of its own price.
            self.assertAlmostEqual(0.05, 100.0 * move / base, delta=0.01)
            in_spreads.append(result.move_pips / result.spread_pips)

        for value in in_spreads[1:]:
            self.assertAlmostEqual(in_spreads[0], value, delta=0.6)


class TestFeaturesCarryTheRightPip(unittest.TestCase):
    def test_atr_pips_defaults_to_the_fx_pip(self):
        """Every existing caller passes no instrument and must be unmoved."""
        rows = compute_features(synthetic_series(220, seed=5))
        last = rows[-1]
        self.assertAlmostEqual(last["atr"] / 0.0001, last["atr_pips"])

    def test_atr_pips_follows_the_instrument(self):
        candles = synthetic_series(220, seed=5)
        btc = compute_features(candles, instrument=instruments.BTCUSD)[-1]
        eur = compute_features(candles, instrument=instruments.EURUSD)[-1]
        # Same bar series, so the raw ATR is identical; only the unit differs.
        self.assertAlmostEqual(btc["atr"], eur["atr"])
        self.assertAlmostEqual(10_000.0, eur["atr_pips"] / btc["atr_pips"], delta=1.0)

    def test_the_row_stays_numeric_for_every_instrument(self):
        """The persistence round trip depends on this.

        Signal.from_dict floats every value in the row, so anything that is
        not a number silently stops the signal from ever settling.
        """
        import json

        from fusion.domain import Signal

        candles = synthetic_series(220, seed=11)
        for item in (instruments.EURUSD, instruments.BTCUSD, instruments.ETHUSD):
            row = compute_features(candles, instrument=item)[-1]
            for key, value in row.items():
                self.assertIsInstance(value, (int, float), f"{item.symbol}.{key}")
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

    def test_bitcoin_prices_are_not_shown_to_five_decimals(self):
        text = self._result("BTC/USD", 80_512.40, 80_517.40)
        self.assertIn("80,512.40", text)
        self.assertNotIn("80512.40000", text)

    def test_eurusd_keeps_its_five_decimals(self):
        text = self._result("EUR/USD", 1.08521, 1.08571)
        self.assertIn("1.08521", text)

    def test_ether_prices_are_shown_in_dollars_and_cents(self):
        text = self._result("ETH/USD", 2_381.44, 2_386.44)
        self.assertIn("2,381.44", text)

    def test_an_unresolved_result_names_no_price(self):
        signal = Signal(
            signal_id="precision-test-2",
            pair="BTC/USD",
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
            spread_pips=8.0,
            settled_at="2026-01-05T12:06:00+00:00",
            source="LIVE",
        )
        text = format_result(signal, settlement)
        self.assertIn("UNRESOLVED", text)
        self.assertIn("BTC/USD", text)


class TestACryptoSeriesRunsThroughThePipeline(unittest.TestCase):
    def test_bitcoin_bars_produce_sane_pip_figures(self):
        """End to end: real-shaped prices, no invented scale.

        A synthetic random walk is not a model of bitcoin, so this proves
        nothing about an edge. It proves the plumbing is unit-correct, which
        is the part that can be wrong by a factor of ten thousand.
        """
        import random

        random.seed(4242)
        price = 80_000.0
        candles = []
        stamp = datetime(2026, 1, 5, 0, 0, tzinfo=timezone.utc)
        for minute in range(240):
            price = max(1.0, price * (1.0 + random.gauss(0.0, 0.0008)))
            candles.append(
                Candle(
                    timestamp=stamp.replace(hour=minute // 60, minute=minute % 60)
                    if minute < 1440
                    else stamp,
                    open=price,
                    high=price * 1.0004,
                    low=price * 0.9996,
                    close=price,
                    volume=1.0,
                )
            )
        rows = compute_features(candles, instrument=instruments.BTCUSD)
        last = rows[-1]
        self.assertIsNotNone(last)
        # On an $80,000 coin with a 0.08% per-minute sigma the M1 ATR should
        # land in the tens of dollars, not in the thousands or the fractions.
        self.assertGreater(last["atr_pips"], 1.0)
        self.assertLess(last["atr_pips"], 5_000.0)

    def test_the_cost_guard_sees_crypto_as_cheaper_than_fx(self):
        """Same shape of market, different verdict on whether it can be traded.

        This is the payoff, and it is a statement about cost rather than about
        skill: the spread is small enough relative to the movement that a
        one-minute trade is not doomed before it starts.
        """
        from fusion import charter

        fx_atr, fx_spread = 0.61, instruments.EURUSD.spread_pips
        btc_atr, btc_spread = 58.0, instruments.BTCUSD.spread_pips

        self.assertFalse(charter.scalp_cost_ok(fx_atr, fx_spread))
        self.assertTrue(charter.scalp_cost_ok(btc_atr, btc_spread))


class TestLiveCryptoPlumbingAndMultiMarket(unittest.TestCase):
    def test_provider_quote_uses_the_instrument_pip_not_fx_pip(self):
        import io
        import json

        from fusion.provider import TwelveDataProvider

        def fake_opener(_req, timeout=None):
            class Resp(io.BytesIO):
                status = 200

                def __enter__(self):
                    return self

                def __exit__(self, *a):
                    pass

            return Resp(json.dumps({"bid": "80500.00", "ask": "80508.00"}).encode())

        provider = TwelveDataProvider(["k1"], opener=fake_opener)
        quote = provider.quote("BTC/USD")
        self.assertIsNotNone(quote)
        # $8 spread on BTC/USD is 8 pips, not 80,000.
        self.assertAlmostEqual(8.0, quote["spread_pips"], places=4)

    def test_crypto_data_gate_stays_open_at_fx_off_hours(self):
        """At 03:00 UTC FX spreads widen 2.1x and trip the gate; 24/7 crypto does not."""
        from datetime import timedelta
        from fusion.engine import FusionEngine

        stamp = datetime(2026, 1, 4, 3, 0, 2, tzinfo=timezone.utc)  # Sunday 03:00 UTC
        start = stamp.replace(second=0) - timedelta(minutes=180)
        btc_bars = [
            Candle(
                timestamp=start + timedelta(minutes=i),
                open=80_000.0 + i,
                high=80_040.0 + i,
                low=79_960.0 + i,
                close=80_010.0 + i,
                volume=12.0,
            )
            for i in range(180)
        ]
        engine = FusionEngine(Config(pair="BTC/USD"))
        result = engine.evaluate(btc_bars, at=stamp)
        self.assertIsNotNone(result.gate)
        self.assertTrue(result.gate.usable, f"crypto gate blocked off-hours: {result.reason}")
        self.assertAlmostEqual(8.0, result.gate.spread_pips)

    def test_multi_market_bot_evaluates_eur_btc_and_eth_without_job_collision(self):
        import shutil
        import tempfile
        from datetime import timedelta
        from pathlib import Path
        from fusion.runner import FusionBot
        from fusion.store import LocalStore, Store

        directory = Path(tempfile.mkdtemp())
        try:
            cfg = Config(pair="EUR/USD", pairs=("EUR/USD", "BTC/USD", "ETH/USD"))
            store = Store(None, LocalStore(directory))
            bot = FusionBot(config=cfg, store=store, provider=None, telegram=None)
            bot.startup()
            self.assertEqual(set(bot.peer_bots), {"BTC/USD", "ETH/USD"})
            self.assertAlmostEqual(bot.peer_bots["BTC/USD"].config.spread_pips, 8.0)
            self.assertAlmostEqual(bot.peer_bots["ETH/USD"].config.spread_pips, 2.4)

            stamp = datetime(2026, 1, 5, 14, 0, 2, tzinfo=timezone.utc)
            start = stamp.replace(second=0) - timedelta(minutes=120)
            for target_bot, base, step in (
                (bot, 1.0850, 0.0002),
                (bot.peer_bots["BTC/USD"], 80_000.0, 25.0),
                (bot.peer_bots["ETH/USD"], 2_380.0, 4.0),
            ):
                target_bot.candles = [
                    Candle(
                        timestamp=start + timedelta(minutes=i),
                        open=base,
                        high=base + step,
                        low=base - step,
                        close=base,
                        volume=1.0,
                    )
                    for i in range(120)
                ]

            report = bot.minute_cycle(at=stamp)
            self.assertNotEqual(report.seal_state, "SKIPPED")
            self.assertEqual(bot.peer_bots["BTC/USD"].cycles, 1)
            self.assertEqual(bot.peer_bots["ETH/USD"].cycles, 1)

            reply = bot.on_demand_signal(at=stamp)
            for symbol in ("EUR/USD", "BTC/USD", "ETH/USD"):
                self.assertIn(symbol, reply)
        finally:
            shutil.rmtree(directory, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
