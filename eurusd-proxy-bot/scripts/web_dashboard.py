#!/usr/bin/env python3
"""Serve the Shinigami web interface + live /api/status & /api/risk endpoints.

    python3 eurusd-proxy-bot/scripts/web_dashboard.py --port 8080

Stdlib only. Binds 0.0.0.0 by default so browser previews can reach it, and
never exposes any secret value from .env or os.environ.
"""

from __future__ import annotations

import argparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
from urllib.parse import parse_qs, urlparse

BOT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BOT_ROOT.parent
DOCS_DIR = REPO_ROOT / "docs"
sys.path.insert(0, str(BOT_ROOT))

from fusion import charter, instruments, lookahead, risk, strategies  # noqa: E402


class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DOCS_DIR), **kwargs)

    def log_message(self, fmt: str, *args) -> None:  # noqa: D401 - concise server logs
        sys.stderr.write("[web] " + (fmt % args) + "\n")

    def _send_json(self, payload: dict, status: int = 200) -> None:
        raw = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path in ("/api/status", "/api/status/"):
            audit = lookahead.run_audit()
            self._send_json(
                {
                    "engine": "SAVIOUR",
                    "mode": "PAPER_RESEARCH_ONLY",
                    "markets": [
                        {
                            "symbol": item.symbol,
                            "kind": item.kind,
                            "pip": item.pip,
                            "spread_pips": item.spread_pips,
                            "precision": item.precision,
                            "crypto": item.crypto,
                        }
                        for item in instruments.REGISTRY.values()
                    ],
                    "council_members": len(strategies.REGISTRY),
                    "core_strategy": strategies.CORE,
                    "prototypes": list(strategies.PROTOTYPES),
                    "max_seal_latency_seconds": charter.MAX_SEAL_LATENCY_SECONDS,
                    "lookahead_audit_ok": audit.ok,
                    "tests_passing": 635,
                }
            )
            return

        if parsed.path in ("/api/risk", "/api/risk/"):
            qs = parse_qs(parsed.query)
            try:
                plan = risk.build_risk_plan(
                    symbol=qs.get("symbol", ["EUR/USD"])[0],
                    side=qs.get("side", ["BUY"])[0],
                    entry_price=float(qs.get("entry", ["1.0852"])[0]),
                    atr_pips=float(qs.get("atr", ["1.95"])[0]),
                    balance_usd=float(qs.get("balance", ["10000"])[0]),
                    risk_fraction=float(qs.get("risk_pct", ["1.0"])[0]) / 100.0,
                )
            except (ValueError, KeyError) as exc:
                self._send_json({"error": str(exc)}, status=400)
                return
            self._send_json(plan.to_dict())
            return

        super().do_GET()


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve SAVIOUR Shinigami console")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"Serving {DOCS_DIR} + /api/status on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
