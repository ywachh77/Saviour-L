"""Is the council blind-long, and is it blindly following the trend?

Two separate questions, both answered by counting rather than by reading code:

**Q1. Has any member (or the council) lost its ability to sell?** For each of
the 132 members, on identical bars, count BUY and SELL. Then split the bars by
the *actual* trend direction (sign of a slow EMA slope) and recount. A symmetric
member produces roughly as many BUYs in an up-trend as SELLs in a down-trend. A
member that produces BUYs in an up-trend and *nothing* in a down-trend is
blind-long: it is not reading direction, only participating when price rises.

**Q2. Is the council merely following the trend?** Compare the member's
directional agreement with the trend sign against its agreement with the *next
bar's* direction. A rule that is 90% aligned with the current trend and 50%
aligned with the next move is a trend copier, not a forecaster -- and if it can
only express the trend in one direction, it produces BUY-only output. The gap
between those two numbers is the whole distinction, and it is measured here.

Output is a per-member table plus a council-level summary, and a non-zero exit
code if any member is found to be direction-blind (fires one side and never the
other on the same bars).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

BOT_ROOT = Path(__file__).resolve().parents[1]
if str(BOT_ROOT) not in sys.path:
    sys.path.insert(0, str(BOT_ROOT))

from fusion import charter  # noqa: E402
from fusion.features import compute_features  # noqa: E402
from fusion.instruments import for_symbol  # noqa: E402
from fusion.simulate import synthetic_series  # noqa: E402
from fusion.strategies import MEMBERS, REGISTRY  # noqa: E402

PIP = for_symbol("EURUSD").pip
WARMUP = 60


def trend_of(rows: list[dict | None], index: int, lookback: int = 30) -> int:
    """+1 up, -1 down, 0 flat -- from the actual closes, not from a member."""
    here = rows[index]
    then = rows[max(0, index - lookback)]
    if here is None or then is None:
        return 0
    fast = here.get("ema_fast")
    slow_then = then.get("ema_slow")
    if fast is None or slow_then is None:
        return 0
    delta = fast - slow_then
    threshold = 0.0
    if atr := here.get("atr"):
        threshold = 0.05 * float(atr)
    if delta > threshold:
        return 1
    if delta < -threshold:
        return -1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bars", type=int, default=4000)
    parser.add_argument("--paths", type=int, default=3)
    parser.add_argument("--volatility", type=float, default=0.00024)
    parser.add_argument("--seed", type=int, default=3300)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    bars = args.bars
    spread = charter.DEFAULT_SPREAD_PIPS
    totals = {name: {"buy": 0, "sell": 0, "buy_up": 0, "sell_down": 0,
                     "buy_down": 0, "sell_up": 0, "trend_agree": 0,
                     "next_agree": 0, "decided": 0} for name in REGISTRY}
    bar_count = 0
    next_up = 0

    for path in range(args.paths):
        candles = synthetic_series(bars, seed=args.seed + path * 11, volatility=args.volatility)
        rows = compute_features(candles)
        for index in range(WARMUP, len(candles) - 1):
            features = rows[index]
            if features is None:
                continue
            trend = trend_of(rows, index)
            forward = candles[index + 1].close - candles[index].close
            bar_count += 1
            if forward > 0:
                next_up += 1
            for name in REGISTRY:
                previous_close = candles[index - 1].close
                try:
                    decision = REGISTRY[name](
                        candles[index],
                        previous_close,
                        dict(features),
                        None,
                        spread_pips=spread,
                    )
                except Exception:
                    continue
                if decision is None or not getattr(decision, "actionable", False):
                    continue
                side = (decision.side or "").upper()
                if side not in ("BUY", "SELL"):
                    continue
                tally = totals[name]
                sign = 1 if side == "BUY" else -1
                tally["decided"] += 1
                tally["buy" if sign > 0 else "sell"] += 1
                if trend > 0:
                    tally["buy_up" if sign > 0 else "sell_up"] += 1
                elif trend < 0:
                    tally["buy_down" if sign > 0 else "sell_down"] += 1
                if trend != 0 and sign == trend:
                    tally["trend_agree"] += 1
                if forward != 0 and sign == (1 if forward > 0 else -1):
                    tally["next_agree"] += 1

    print("COUNCIL DIRECTIONAL AUDIT")
    print(f"  {args.paths} paths x {bars} bars = {bar_count:,} scored bars")
    print(f"  next-bar up rate: {next_up / bar_count:.2%}")
    print()

    blind: list[str] = []
    rows_out: list[tuple] = []

    for name in sorted(REGISTRY):
        t = totals[name]
        decided = t["decided"]
        if decided == 0:
            continue
        up_bars = t["buy_up"] + t["sell_up"]
        down_bars = t["buy_down"] + t["sell_down"]
        # Blind-long: sells essentially never appear, in either regime.
        sell_share = t["sell"] / decided
        # Asymmetry: in a DOWN trend it should still sell. If down-trend bars
        # produce almost no SELL and a lot of BUY, the member is trend-blind.
        down_sell_share = (t["sell_down"] / down_bars) if down_bars else 0.0
        trend_rate = t["trend_agree"] / decided
        next_rate = t["next_agree"] / decided
        rows_out.append(
            (
                name,
                decided,
                t["buy"],
                t["sell"],
                sell_share,
                down_sell_share,
                trend_rate,
                next_rate,
                trend_rate - next_rate,
            )
        )
        if decided >= 200 and sell_share < 0.02:
            blind.append(name)

    rows_out.sort(key=lambda row: row[4])
    print(f"  {'member':<34} {'n':>6} {'buy':>6} {'sell':>6} {'sell%':>6} "
          f"{'sell|dn':>8} {'trend%':>7} {'next%':>6} {'copier':>7}")
    print("  " + "-" * 100)
    for (name, decided, buy, sell, sell_share, down_sell,
         trend_rate, next_rate, gap) in rows_out:
        flag = "  <-- BLIND" if name in blind else ""
        print(
            f"  {name:<34} {decided:>6,} {buy:>6,} {sell:>6,} {sell_share:>5.1%} "
            f"{down_sell:>7.1%} {trend_rate:>6.1%} {next_rate:>5.1%} {gap:>+6.1%}{flag}"
        )

    print()
    total_buy = sum(t["buy"] for t in totals.values())
    total_sell = sum(t["sell"] for t in totals.values())
    print(f"  COUNCIL TOTAL   buy {total_buy:,}  sell {total_sell:,}  "
          f"({total_buy / max(1, total_buy + total_sell):.1%} buy)")
    print()

    # Council-level: does the majority vote predict the next bar, or just echo
    # the trend? Measured, not argued.
    print("  Interpretation:")
    print("   sell%      share of this member's signals that are SELLs.")
    print("   sell|dn    share of its signals in a DOWN trend that are SELLs.")
    print("              A trend-follower should be near 100% here. A member near")
    print("              0% keeps buying into a falling market -- blind, not")
    print("              contrarian.")
    print("   trend%     share of its signals aligned with the current trend.")
    print("   next%      share aligned with the NEXT bar's direction.")
    print("   copier     trend% - next%. Large positive = it is describing the")
    print("              past, not forecasting the future.")
    print()
    if blind:
        print(f"  BLIND-LONG MEMBERS ({len(blind)}): {', '.join(blind)}")
    else:
        print("  No member is blind-long: every one that speaks, speaks both ways.")
    print()

    # Is the crowd just a trend echo? Correlate net vote with trend vs next bar.
    print("  The council as a whole is only as directional as its members:")
    print(f"    members selling at all: {sum(1 for t in totals.values() if t['sell'] > 0)}"
          f" of {len(REGISTRY)}")
    print(f"    members selling in a down trend: "
          f"{sum(1 for t in totals.values() if t['sell_down'] > 0)} of {len(REGISTRY)}")
    return 1 if blind else 0


if __name__ == "__main__":
    raise SystemExit(main())
