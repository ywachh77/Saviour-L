#!/usr/bin/env python3
"""Can this configuration ever pay for itself?

    python scripts/economics.py                # judge the current .env
    python scripts/economics.py 1.5            # assume an M1 ATR of 1.5 pips
    python scripts/economics.py 1.5 --spread 0.8

This answers a question the accuracy numbers cannot. A model can be right 56%
of the time and still lose money forever, because being right about the
direction is worthless when the move is smaller than the toll charged to travel
it. That is a *cost* problem, not a prediction problem, and more accuracy does
not fix it.

It runs with no keys, no database and no network, so it is the first thing worth
running on a fresh install.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fusion import costs  # noqa: E402
from fusion.config import load_config  # noqa: E402

# A reference scale for EUR/USD on a one-minute chart. Quiet Asia is lower,
# London open is higher. Used only when the caller does not supply a number.
REALISTIC_M1_ATR_PIPS = 1.5

BAR = "─" * 62


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Can this configuration pay its own costs?")
    parser.add_argument("atr", nargs="?", type=float, default=None,
                        help=f"M1 ATR in pips (default {REALISTIC_M1_ATR_PIPS})")
    parser.add_argument("--spread", type=float, default=None, help="round-turn spread in pips")
    parser.add_argument("--horizon", type=int, default=None, help="hold length in minutes")
    args = parser.parse_args(argv)

    config = load_config(ROOT / ".env")
    spread = float(args.spread if args.spread is not None else config.spread_pips)
    horizon = int(args.horizon if args.horizon is not None else config.horizon_minutes)
    atr_pips = float(args.atr if args.atr is not None else REALISTIC_M1_ATR_PIPS)

    print()
    print("FUSION BOT — can this configuration pay for itself?")
    print("=" * 62)
    print(f"  pair              : {config.pair}")
    print(f"  round-turn cost   : {spread:.2f} pips")
    print(f"  configured hold   : {horizon} minutes")
    print(f"  market volatility : {atr_pips:.2f} pips M1 ATR")
    print(f"  cost multiple     : {config.reward_to_cost:.1f}x "
          f"(the move must be this many times the cost)")
    print()
    print(BAR)

    verdict = costs.evaluate(atr_pips, horizon, spread, config.reward_to_cost)
    print()
    print(f"  projected move over {horizon}m : {verdict.projected_move_pips:6.2f} pips")
    print(f"  move needed to pay        : {verdict.required_move_pips:6.2f} pips")
    print(f"  M1 ATR needed for {horizon}m    : {verdict.required_atr_pips:6.2f} pips "
          f"(market offers {atr_pips:.2f})")
    print()

    needed = costs.break_even_win_rate(atr_pips, horizon, spread)
    if needed == float("inf"):
        print("  hit rate needed to break even : impossible at any accuracy")
        print("    the projected move is smaller than the cost, so there is no")
        print("    win rate that rescues this configuration.")
    else:
        print(f"  hit rate needed to break even : {needed * 100:.1f}%")
        print("    This is the number to compare any reported win rate against.")
    print()

    if verdict.never_viable:
        print("  VERDICT — DO NOT TRADE THIS")
        print(f"    {verdict.describe()}")
        print()
        print("    This is not a prediction problem. The market is not moving enough")
        print("    for ANY hold length to pay the toll, so no amount of model accuracy")
        print("    can make this configuration profitable. Wait for volatility, or")
        print("    reduce your costs.")
        print()
        return 1

    if verdict.ok:
        print("  VERDICT — affordable")
        print(f"    {verdict.describe()}")
        print()
        print("    Affordable means the trade CAN pay for itself, not that it will.")
        print("    Whether it does is a question about the model, and it is answered by")
        print("    scripts/backtest.py --paths — not by a single replay.")
        print()
        return 0

    viable = verdict.viable_horizon
    print("  VERDICT — NOT affordable at this hold length")
    print(f"    {verdict.describe()}")
    print()
    print(f"    A {viable}-minute hold would clear the cost at this volatility.")
    if config.adaptive_horizon:
        print("    ADAPTIVE_HORIZON is already on, so signals are being stretched to")
        print("    a horizon that can pay rather than dropped.")
    else:
        print("    Set ADAPTIVE_HORIZON=1 in .env to stretch each signal's hold to a")
        print("    length that can pay, instead of trading a horizon that cannot.")
    print()
    print("    The alternative is worse: at this volatility a shorter hold simply")
    print("    pays the same toll for a smaller move.")
    print()

    print(BAR)
    print("  At a fixed 1.2 pip cost, what each M1 ATR can afford:")
    print(f"    {'M1 ATR':>8} {'min hold to pay':>18}")
    for scale in (0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0):
        needed = costs.minimum_viable_horizon(scale, 1.2, config.reward_to_cost)
        shown = "never" if needed is None else f"{needed}m"
        marker = "  <- your market" if abs(scale - atr_pips) < 0.26 else ""
        print(f"    {scale:>8.1f} {shown:>18}{marker}")
    print()
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
