"""Automated look-ahead and recursive-warmup audit tests."""

import unittest

from fusion import lookahead, strategies
from fusion.domain import Decision


class TestLookaheadAudit(unittest.TestCase):
    def test_full_audit_passes_on_production_pipeline(self):
        report = lookahead.run_audit()
        self.assertTrue(report.ok, f"audit failed: {report.to_dict()}")
        self.assertEqual(len(report.findings), 5)

    def test_audit_catches_a_strategy_that_peeks_ahead(self):
        """Sensitivity guard: an audit that cannot fail proves nothing."""
        calls = {"n": 0}

        def cheating_strategy(candle, previous_close, features, probability_up=None, **_kw):
            calls["n"] += 1
            # Cheat on the second invocation (the shocked future run) to mimic
            # a strategy whose state or closure read ahead into future bars.
            if calls["n"] == 2:
                return Decision("BUY", "peeked at shocked future", probability_up, 2)
            return Decision("NO SIGNAL", "clean prefix", probability_up, 0)

        original = dict(strategies.REGISTRY)
        strategies.REGISTRY["cheating_probe"] = cheating_strategy
        try:
            finding = lookahead.check_strategies_causality()
        finally:
            strategies.REGISTRY.clear()
            strategies.REGISTRY.update(original)

        self.assertFalse(finding.passed)
        self.assertIn("cheating_probe", finding.detail)

    def test_recursive_warmup_flags_insufficient_series(self):
        short = lookahead.synthetic_series(100, seed=5)
        finding = lookahead.check_recursive_warmup(short, short_window=80, production_window=320)
        self.assertFalse(finding.passed)


if __name__ == "__main__":
    unittest.main()
