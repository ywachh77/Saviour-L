"""Higher-timeframe context resampled from the bars we already hold."""

import unittest
from datetime import datetime, timedelta, timezone

from fusion import mtf
from fusion.config import Config
from fusion.domain import Candle
from fusion.features import FEATURE_VERSION, compute_features
from fusion.mtf import Bias, DOWN, FLAT, UP, bias, bias_series, resample

from .helpers import make_candles

UTC = timezone.utc


def flat_series(count: int, price: float = 1.1000, start=None) -> list[Candle]:
    """A series that does not move, then whatever the caller appends."""
    first = start or datetime(2026, 1, 5, 8, 0, tzinfo=UTC)
    return [
        Candle(
            timestamp=first + timedelta(minutes=i),
            open=price,
            high=price + 0.0001,
            low=price - 0.0001,
            close=price,
            volume=100.0,
        )
        for i in range(count)
    ]


class TestResample(unittest.TestCase):
    def test_five_one_minute_bars_are_one_five_minute_bar(self):
        candles = make_candles(20, seed=3)
        bars = resample(candles, 5)
        self.assertEqual(len(bars), 4)

        first = bars[0]
        group = candles[:5]
        self.assertEqual(first.timestamp, group[0].timestamp)
        self.assertEqual(first.open, group[0].open)
        self.assertEqual(first.close, group[-1].close)
        self.assertEqual(first.high, max(c.high for c in group))
        self.assertEqual(first.low, min(c.low for c in group))
        self.assertAlmostEqual(first.volume, sum(c.volume for c in group))

    def test_buckets_align_to_wall_clock_boundaries(self):
        start = datetime(2026, 1, 5, 8, 3, tzinfo=UTC)
        candles = make_candles(20, seed=3, start=start)
        bars = resample(candles, 5)
        for bar in bars:
            self.assertEqual(bar.timestamp.minute % 5, 0)

    def test_a_forming_bucket_is_dropped(self):
        """A bar that has not finished is not a bar."""
        bars = resample(make_candles(23, seed=3), 5)
        self.assertEqual(len(bars), 4)  # 20 complete, 3 still forming
        bars = resample(make_candles(20, seed=3), 5)
        self.assertEqual(len(bars), 4)

    def test_one_minute_is_the_original_series(self):
        candles = make_candles(10, seed=3)
        self.assertEqual(resample(candles, 1), list(candles))


class TestBias(unittest.TestCase):
    def test_a_rising_series_tilts_up(self):
        candles = make_candles(200, seed=11, drift=0.00022, volatility=0.00002)
        self.assertEqual(bias(candles, 5).direction, UP)

    def test_a_falling_series_tilts_down(self):
        candles = make_candles(200, seed=11, drift=-0.00022, volatility=0.00002)
        self.assertEqual(bias(candles, 5).direction, DOWN)

    def test_too_few_bars_gives_no_view(self):
        result = bias(make_candles(10, seed=3), 5)
        self.assertEqual(result.direction, FLAT)
        self.assertFalse(result.known)
        self.assertIn("only", result.reason)

    def test_agrees_only_with_matching_sides(self):
        up = Bias(5, UP, 1.1, 1.09, 20, "test")
        self.assertTrue(up.agrees_with("BUY"))
        self.assertFalse(up.agrees_with("SELL"))
        self.assertFalse(up.agrees_with(None))

    def test_a_flat_tilt_agrees_with_nothing(self):
        """Unknown must block, never permit."""
        flat = Bias(5, FLAT, 1.1, 1.1, 20, "test")
        self.assertFalse(flat.agrees_with("BUY"))
        self.assertFalse(flat.agrees_with("SELL"))


class TestBiasSeriesNeverLooksAhead(unittest.TestCase):
    """The rule that matters: a bucket still forming cannot be read."""

    def test_mutating_the_future_cannot_move_a_reading(self):
        """Rewrite every bar after `index`; the reading at `index` must hold.

        The indices chosen are all the first minute of a five-minute bucket, so
        each one's own bucket is still forming. If a forming bucket were read,
        the rewritten bars sitting in it would move the answer.
        """
        base = make_candles(120, seed=7, drift=0.00005, volatility=0.0002)
        before = bias_series(base, 5)
        for index in (30, 55, 80, 100):
            variant = list(base[: index + 1])
            for offset in range(index + 1, len(base)):
                variant.append(
                    Candle(
                        timestamp=base[offset].timestamp,
                        open=1.1000,
                        high=1.3000,
                        low=0.9000,
                        close=0.9500,  # a collapse in every future bar
                        volume=9999.0,
                    )
                )
            after = bias_series(variant, 5)
            self.assertEqual(
                (before[index].direction, before[index].close),
                (after[index].direction, after[index].close),
                f"the reading at {index} moved when the future changed",
            )

    def test_a_reading_never_uses_a_bar_that_has_not_closed(self):
        candles = make_candles(120, seed=7)
        series = bias_series(candles, 5)
        for index, item in enumerate(series):
            if not item.known:
                continue
            # The bar the reading came from must have closed before this minute.
            self.assertLessEqual(index + 1, len(candles))
            self.assertIsNotNone(item.reason)

    def test_series_length_matches_the_input(self):
        candles = make_candles(97, seed=7)
        self.assertEqual(len(bias_series(candles, 5)), len(candles))


class TestFeaturesCarryTheTilt(unittest.TestCase):
    def setUp(self):
        self.candles = make_candles(200, seed=7)
        self.rows = compute_features(self.candles)

    def test_the_tilt_reaches_the_feature_row(self):
        row = self.rows[-1]
        self.assertIn("mtf_bias_5", row)
        self.assertIn("mtf_gap_5", row)
        self.assertIn("mtf_bias_15", row)
        self.assertIn("mtf_gap_15", row)
        self.assertIn(row["mtf_bias_5"], (-1.0, 0.0, 1.0))

    def test_the_row_holds_only_plain_numbers(self):
        """A rich object in the row silently killed every settlement.

        The row is persisted alongside the signal and re-read by
        Signal.from_dict, which runs float() over every value. A Bias object
        survives json.dumps only because the store falls back to str(), and
        returns as a string float() rejects -- so settle_due() caught the
        ValueError, skipped the signal and moved on. Signals stopped settling
        while still reporting themselves as delivered. Objects stay out.
        """
        for row in self.rows:
            if row is None:
                continue
            for key, value in row.items():
                self.assertIsInstance(
                    value,
                    (int, float),
                    f"feature {key!r} is {type(value).__name__}; the persisted row must be numeric",
                )
        self.assertNotIn("mtf_fast", self.rows[-1])
        self.assertNotIn("mtf_slow", self.rows[-1])

    def test_the_row_survives_the_persistence_round_trip(self):
        """The row must parse back through Signal.from_dict, not merely serialise.

        json.dumps(..., default=str) hides the problem: it accepts anything.
        The failure only appears on the way back in, which is why this asserts
        on the reload rather than on the write.
        """
        import json

        from fusion.domain import Signal

        row = self.rows[-1]
        persisted = json.loads(json.dumps(row, default=str))
        base = {
            "signal_id": "test-1",
            "side": "BUY",
            "entry_time": "2026-01-05T12:00:00+00:00",
            "expiry_time": "2026-01-05T12:05:00+00:00",
            "features": persisted,
        }
        rebuilt = Signal.from_dict(dict(base))
        self.assertEqual(rebuilt.features["mtf_bias_5"], row["mtf_bias_5"])

    def test_an_unknown_tilt_reports_a_zero_gap(self):
        """'We do not know' must never look like 'exactly on the average'."""
        for row in self.rows:
            if row is None:
                continue
            if row["mtf_bias_5"] == 0.0:
                self.assertEqual(row["mtf_gap_5"], 0.0)

    def test_the_tilt_is_not_silently_added_to_the_model_vector(self):
        """Unproven inputs must not redefine the model behind our backs.

        Measured across five 2,500-bar replays the tilt changed nothing at the
        aggregate level -- 891 signals, 51.2%, -734.2 net pips either way -- so
        it has not earned a place in FEATURE_NAMES. Adding it there would
        invalidate every trained model for no measured gain.
        """
        from fusion.features import FEATURE_NAMES

        for name in (
            "mtf_bias_5",
            "mtf_gap_5",
            "mtf_slope_5",
            "mtf_ready_5",
            "mtf_bias_15",
            "mtf_gap_15",
            "mtf_slope_15",
            "mtf_ready_15",
            "mtf_bias_60",
            "mtf_gap_60",
            "mtf_slope_60",
            "mtf_ready_60",
        ):
            self.assertNotIn(name, FEATURE_NAMES)
        self.assertEqual(5, FEATURE_VERSION)

    def test_clear_trend_dead_zone_and_slope_filter(self):
        gentle = make_candles(200, seed=11, drift=0.00001, volatility=0.000005)
        # Huge dead-zone forces FLAT rather than letting tiny EMA hug count as trend
        blocked = bias(gentle, minutes=5, min_gap_bps=50.0)
        self.assertEqual(blocked.direction, FLAT)
        self.assertIn("dead-zone", blocked.reason)

        strong = make_candles(200, seed=11, drift=0.00030, volatility=0.00002)
        clear = bias(strong, minutes=5, min_gap_bps=1.5, require_slope=True)
        self.assertEqual(clear.direction, UP)
        self.assertGreater(clear.slope_bps, 0.0)


class TestTheGateIsOffByDefault(unittest.TestCase):
    """Shipped off because it measured worse, not because it is unfinished."""

    def test_the_filter_is_off_until_real_data_settles_it(self):
        self.assertFalse(Config().mtf_filter)

    def test_it_can_be_switched_on(self):
        self.assertTrue(Config(mtf_filter=True).mtf_filter)


if __name__ == "__main__":
    unittest.main()
