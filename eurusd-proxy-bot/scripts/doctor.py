#!/usr/bin/env python3
"""Check your setup and tell you exactly which step is broken.

    python scripts/doctor.py            # check everything
    python scripts/doctor.py --send     # also put a test message in your chat
    python scripts/doctor.py --offline  # skip the live Twelve Data call

The Twelve Data check costs one API credit per run. Everything else is free.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fusion.config import load_config  # noqa: E402
from fusion.doctor import run_checks, summarise  # noqa: E402

BOUNDARY = (
    "Reminder: this bot is research and paper-settlement only. "
    "It never places a broker order."
)


def main(argv: list[str]) -> int:
    send_test = "--send" in argv
    offline = "--offline" in argv
    config = load_config(ROOT / ".env")

    print()
    print("FUSION BOT — setup doctor")
    print("=" * 62)
    checks = run_checks(config, env_path=ROOT / ".env", offline=offline, send_test=send_test)
    for check in checks:
        print(check.line())
    passed, failed = summarise(checks)

    print()
    print("-" * 62)
    if failed == 0:
        print(f"All {passed} checks passed. Run:  python scripts/run_bot.py")
    else:
        print(f"{passed} passed, {failed} need attention. Fix them in the order printed above.")
    print()
    print(BOUNDARY)
    print()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
