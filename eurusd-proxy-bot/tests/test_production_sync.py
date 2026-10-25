"""Production sync guards: workflow parity, listen budget, KeyPool hydration, and multi-market telemetry."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import io
import json
from pathlib import Path
import tempfile
import unittest
import urllib.parse

from fusion.config import Config
from fusion.provider import TwelveDataProvider
from fusion.runner import FusionBot
from fusion.store import LocalStore, Store
from scripts.actions_cycle import resolve_listen_seconds

BOT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BOT_ROOT.parent


def _extract_env_flags(workflow_path: Path, keys: tuple[str, ...]) -> dict[str, str]:
    found: dict[str, str] = {}
    for raw_line in workflow_path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if ":" not in stripped or stripped.startswith("#"):
            continue
        name, value = stripped.split(":", 1)
        name = name.strip()
        if name in keys:
            found[name] = value.strip().strip('"').strip("'")
    return found


def _fake_twelvedata_opener(end_stamp: datetime):
    base_prices = {"EUR/USD": 1.0850, "BTC/USD": 80500.0, "ETH/USD": 2380.0}
    steps = {"EUR/USD": 0.0002, "BTC/USD": 25.0, "ETH/USD": 1.2}

    class _Response:
        status = 200

        def __init__(self, body: bytes):
            self._buf = io.BytesIO(body)

        def read(self) -> bytes:
            return self._buf.read()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def opener(request, timeout: float = 8.0):  # noqa: ARG001
        parsed = urllib.parse.urlparse(request.full_url)
        params = urllib.parse.parse_qs(parsed.query)
        symbol = params.get("symbol", ["EUR/USD"])[0]
        count = int(params.get("outputsize", ["80"])[0])
        base = base_prices[symbol]
        step = steps[symbol]
        values = []
        for idx in range(count):
            ts = end_stamp - timedelta(minutes=(count - 1 - idx))
            open_p = base + idx * step * 0.2
            close_p = open_p + step * (0.4 if idx % 3 else -0.2)
            high_p = max(open_p, close_p) + step * 0.3
            low_p = min(open_p, close_p) - step * 0.3
            values.append(
                {
                    "datetime": ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "open": f"{open_p:.5f}",
                    "high": f"{high_p:.5f}",
                    "low": f"{low_p:.5f}",
                    "close": f"{close_p:.5f}",
                    "volume": "100",
                }
            )
        payload = json.dumps({"status": "ok", "values": values}).encode("utf-8")
        return _Response(payload)

    return opener


class TestWorkflowParity(unittest.TestCase):
    SHARED_FLAGS = (
        "PAIRS",
        "ADAPTIVE_HORIZON",
        "MAX_HORIZON_MINUTES",
        "MTF_FILTER",
        "MTF_MINUTES",
        "NEWS_FILTER",
    )

    def test_cycle_and_nightly_workflows_share_identical_strategy_flags(self):
        root_cycle = _extract_env_flags(REPO_ROOT / ".github/workflows/fusion-cycle.yml", self.SHARED_FLAGS)
        root_nightly = _extract_env_flags(REPO_ROOT / ".github/workflows/fusion-nightly.yml", self.SHARED_FLAGS)
        sub_cycle = _extract_env_flags(BOT_ROOT / ".github/workflows/cycle.yml", self.SHARED_FLAGS)
        sub_nightly = _extract_env_flags(BOT_ROOT / ".github/workflows/nightly.yml", self.SHARED_FLAGS)

        self.assertEqual(set(root_cycle.keys()), set(self.SHARED_FLAGS))
        self.assertEqual(root_cycle, root_nightly)
        self.assertEqual(root_cycle, sub_cycle)
        self.assertEqual(root_cycle, sub_nightly)

        root_backfill = _extract_env_flags(REPO_ROOT / ".github/workflows/fusion-backfill.yml", ("PAIRS",))
        sub_backfill = _extract_env_flags(BOT_ROOT / ".github/workflows/backfill.yml", ("PAIRS",))
        self.assertEqual(root_backfill["PAIRS"], root_cycle["PAIRS"])
        self.assertEqual(sub_backfill["PAIRS"], root_cycle["PAIRS"])


class TestListenBudget(unittest.TestCase):
    def test_fast_cycle_keeps_full_listen_window(self):
        self.assertAlmostEqual(resolve_listen_seconds(720.0, elapsed=4.0, max_run_seconds=1080.0), 720.0)

    def test_long_scheduled_job_shrinks_listen_window_to_fit_budget(self):
        self.assertAlmostEqual(resolve_listen_seconds(720.0, elapsed=660.0, max_run_seconds=1080.0), 420.0)

    def test_exhausted_budget_skips_listening(self):
        self.assertEqual(resolve_listen_seconds(720.0, elapsed=1120.0, max_run_seconds=1080.0), 0.0)
        self.assertEqual(resolve_listen_seconds(0.0, elapsed=2.0, max_run_seconds=1080.0), 0.0)


class TestMultiMarketObservabilityAndKeyPoolHydration(unittest.TestCase):
    def test_ephemeral_containers_accumulate_keypool_usage_and_surface_all_markets(self):
        # Sunday 05:59 UTC: EUR/USD off-hours spread blocks FX, while BTC/USD and ETH/USD evaluate 24/7.
        sunday_stamp = datetime(2026, 10, 25, 5, 59, 2, tzinfo=timezone.utc)
        next_stamp = datetime(2026, 10, 25, 6, 14, 2, tzinfo=timezone.utc)

        with tempfile.TemporaryDirectory() as tmp:
            local = LocalStore(Path(tmp))
            store = Store(local=local, supabase=None)
            cfg = Config(
                pair="EUR/USD",
                pairs=("EUR/USD", "BTC/USD", "ETH/USD"),
                candle_history=90,
                twelve_data_keys=["test-live-key-0123456789abcdef"],
            )

            # Ephemeral container #1
            provider_1 = TwelveDataProvider(
                cfg.twelve_data_keys,
                opener=_fake_twelvedata_opener(sunday_stamp - timedelta(minutes=1)),
            )
            bot_1 = FusionBot(config=cfg, store=store, provider=provider_1)
            bot_1.startup()
            report_1 = bot_1.minute_cycle(at=sunday_stamp)

            # All 3 markets appear in the summary report & heartbeat; crypto regime is not hidden by FX UNKNOWN.
            self.assertIn("EUR/USD:", report_1.reason)
            self.assertIn("BTC/USD", report_1.reason)
            self.assertIn("ETH/USD", report_1.reason)
            self.assertNotEqual(report_1.regime, "UNKNOWN")

            saved_pool_1 = store.load_key_pool()
            self.assertIsNotNone(saved_pool_1)
            self.assertEqual(saved_pool_1["slots"][0]["day_used"], 3)

            # Peer heartbeats carry pair-scoped buckets.
            hb_rows = local.read("heartbeats")
            buckets = [row.get("bucket") for row in hb_rows]
            self.assertIn("BTC/USD|2026-10-25T05:59Z", buckets)
            self.assertIn("ETH/USD|2026-10-25T05:59Z", buckets)

            # Ephemeral container #2 boots fresh and hydrates day_used == 3 before adding 3 more calls.
            provider_2 = TwelveDataProvider(
                cfg.twelve_data_keys,
                opener=_fake_twelvedata_opener(next_stamp - timedelta(minutes=1)),
            )
            bot_2 = FusionBot(config=cfg, store=store, provider=provider_2)
            bot_2.startup()
            self.assertEqual(bot_2.provider.pool.slots[0].day_used, 3)

            bot_2.minute_cycle(at=next_stamp)
            saved_pool_2 = store.load_key_pool()
            self.assertEqual(saved_pool_2["slots"][0]["day_used"], 6)

            status = bot_2.status_text()
            self.assertIn("pairs EUR/USD, BTC/USD, ETH/USD", status)
            self.assertIn("6/800 credits today", status)
            self.assertIn("ETH/USD", status)


if __name__ == "__main__":
    unittest.main()
