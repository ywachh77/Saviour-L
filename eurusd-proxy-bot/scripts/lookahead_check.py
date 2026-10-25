#!/usr/bin/env python3
"""CLI look-ahead and recursive-warmup audit (Freqtrade-style).

    python scripts/lookahead_check.py

Exits 0 only when every feature, higher-timeframe resampler, council strategy,
live engine seal guard, and recursive warmup check passes.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fusion.lookahead import run_audit  # noqa: E402


def main() -> int:
    report = run_audit()
    print("SAVIOUR lookahead-analysis + recursive-analysis")
    for item in report.findings:
        status = "PASS" if item.passed else "FAIL"
        print(f"  [{status}] {item.component:18s} {item.detail}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
