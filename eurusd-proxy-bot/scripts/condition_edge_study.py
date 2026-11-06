"""Where do the signals actually win, and where do they fail?

Every rule in this repository claims a condition: buy when RSI confirms, sell
when ADX is high, trade the London session, avoid dead volatility. Those claims
are testable. This script tests them, and it is built to stop you believing a
result that is not there.

It answers one question per condition:

    given this state of the market, what is the win rate of a signal taken now,
    and is that win rate above the rate the spread demands?

Two things make the answer trustworthy rather than decorative:

* **The null is the base rate, not 50%.** If up-moves happen 51% of the time,
  a "57% winner" has earned 6 points, not 7. Every rate is compared against the
  rate a coin flip would have scored in the same sample.
* **Many conditions are tested at once.** Split 5,000 bars by 6 conditions into
  3 buckets each and you have 18 cells, and the best of 18 will look good by
  chance. So each finding carries a p-value and a Benjamini-Hochberg pass, and
  the script says plainly which rows survive.

Output is a table per condition: n, win rate, the base rate it must beat, net
pips per trade after spread, and whether the difference is real.

Run:

    python3 scripts/condition_edge_study.py                    # synthetic sample
    python3 scripts/condition_edge_study.py --real             # the shipped bars
    python3 scripts/condition_edge_study.py --bars 20000 --paths 4
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

BOT_ROOT = Path(__file__).resolve().parents[1]
if str(BOT_ROOT) not in sys.path:
    sys.path.insert(0, str(BOT_ROOT))

from fusion import charter  # noqa: E402
from fusion.domain import Candle  # noqa: E402
from fusion.features import compute_features  # noqa: E402
from fusion.instruments import for_symbol  # noqa: E402
from fusion.simulate import synthetic_series  # noqa: E402
from fusion.timeutil import parse_datetime  # noqa: E402

PIP = for_symbol("EURUSD").pip
WARMUP = 80


# --------------------------------------------------------------------------
# statistics that refuse to flatter
# --------------------------------------------------------------------------


def normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def two_sided_p(wins: int, n: int, base_rate: float) -> float:
    """Is this win rate distinguishable from the sample's own base rate?"""
    if n <= 0:
        return 1.0
    base = min(max(base_rate, 1e-9), 1.0 - 1e-9)
    variance = n * base * (1.0 - base)
    if variance <= 0:
        return 1.0
    z = (wins - n * base) / math.sqrt(variance)
    return 2.0 * (1.0 - normal_cdf(abs(z)))


def benjamini_hochberg(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Which of these findings survive multiple-comparison correction?

    Testing 18 conditions and reporting the best is how a study lies without
    anyone lying. This returns the subset that stays significant after the
    number of tests is accounted for.
    """
    n = len(p_values)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: p_values[i])
    keep = [False] * n
    threshold_index = -1
    for rank, index in enumerate(order, start=1):
        if p_values[index] <= alpha * rank / n:
            threshold_index = rank
    for rank, index in enumerate(order, start=1):
        if rank <= threshold_index:
            keep[index] = True
    return keep


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------


def load_real(path: Path) -> list[Candle]:
    rows = json.loads(path.read_text())
    rows.sort(key=lambda row: row["ts"])
    out = []
    for row in rows:
        out.append(
            Candle(
                timestamp=parse_datetime(row["ts"]),
                open=float(row["o"]),
                high=float(row["h"]),
                low=float(row["l"]),
                close=float(row["c"]),
                volume=float(row.get("v") or 0.0),
            )
        )
    return out


def build_dataset(paths: list[list[Candle]], horizon: int, stride: int) -> list[dict]:
    """One row per *non-overlapping* window: conditions, and what price did next.

    ``stride`` defaults to the hold. This matters more than anything else in the
    script. Consecutive 15-minute windows share 14 of their 15 minutes, so a
    table built from every bar has ~15x too many "independent" observations and
    p-values that are nonsense. The control run proved it: on a pure random walk
    the overlapping table produced five confident "findings". Sampling every
    ``stride`` bars is what makes n mean what it says.
    """
    rows: list[dict] = []
    for candles in paths:
        features = compute_features(candles)
        for index in range(WARMUP, len(candles) - horizon, max(1, stride)):
            here = features[index]
            if here is None:
                continue
            entry = candles[index + 1].open
            exit_ = candles[index + horizon].close
            move = (exit_ - entry) / PIP
            if abs(move) < 1e-9:
                continue
            rows.append(
                {
                    "rsi": float(here.get("rsi") or 50.0),
                    "adx": float(here.get("adx") or 0.0),
                    "di_spread": float(here.get("di_spread") or 0.0),
                    "atr_pips": abs(float(here.get("atr") or 0.0)) / PIP,
                    "ema_gap": float(here.get("ema_gap") or 0.0),
                    "ema_slope": float(here.get("ema_slope") or 0.0),
                    "close_frac": (candles[index].close - candles[index].low)
                    / max(candles[index].high - candles[index].low, 1e-9),
                    "hour": candles[index].timestamp.hour,
                    "return_1": float(here.get("return_1") or 0.0),
                    "return_3": float(here.get("return_3") or 0.0),
                    "volatility_ratio": float(here.get("volatility_ratio") or 1.0),
                    "move": move,
                    "up": move > 0,
                    "sample_index": index,
                }
            )
    return rows


def buckets(values: list[float], count: int = 3) -> list[float]:
    ordered = sorted(values)
    cuts = []
    for k in range(1, count):
        cuts.append(ordered[int(len(ordered) * k / count)])
    return cuts


def assign(value: float, cuts: list[float], labels: list[str]) -> str:
    for cut, label in zip(cuts, labels[:-1]):
        if value <= cut:
            return label
    return labels[-1]


# --------------------------------------------------------------------------
# the conditions, as a trader would state them
# --------------------------------------------------------------------------

CONDITIONS = {
    "RSI": ("rsi", ["RSI < 30", "RSI 30-50", "RSI 50-70", "RSI > 70"]),
    "ADX": ("adx", ["ADX < 20", "ADX 20-25", "ADX 25-35", "ADX > 35"]),
    "ATR (volatility)": ("atr_pips", ["ATR low", "ATR mid", "ATR high"]),
    "EMA slope (trend)": ("ema_slope", ["falling", "flat", "rising"]),
    "Close in range": ("close_frac", ["closed low", "closed mid", "closed high"]),
    "Session (UTC)": ("hour", ["ASIA 00-07", "LONDON 08-12", "NY 13-20", "LATE 21-23"]),
    "Volatility ratio": ("volatility_ratio", ["contracting", "normal", "expanding"]),
}

SESSION_BINS = [
    ("ASIA 00-07", range(0, 8)),
    ("LONDON 08-12", range(8, 13)),
    ("NY 13-20", range(13, 21)),
    ("LATE 21-23", range(21, 24)),
]


def label_for(condition: str, row: dict, cuts: dict[str, list[float]]) -> str:
    if condition == "RSI":
        value = row["rsi"]
        for cut, label in zip([30, 50, 70], ["RSI < 30", "RSI 30-50", "RSI 50-70"]):
            if value < cut:
                return label
        return "RSI > 70"
    if condition == "ADX":
        value = row["adx"]
        for cut, label in zip([20, 25, 35], ["ADX < 20", "ADX 20-25", "ADX 25-35"]):
            if value < cut:
                return label
        return "ADX > 35"
    if condition == "Session (UTC)":
        for label, hours in SESSION_BINS:
            if row["hour"] in hours:
                return label
        return "LATE 21-23"
    field, labels = CONDITIONS[condition]
    return assign(row[field], cuts[condition], labels)


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def analyse(rows: list[dict], spread: float, horizon: int) -> None:
    base_up = sum(1 for row in rows if row["up"]) / len(rows)
    mean_move = statistics.fmean(abs(row["move"]) for row in rows)
    # The rate a trade needs to clear its cost, on the directional bet.
    required = 0.5 + spread / (2.0 * mean_move) if mean_move > spread else float("inf")

    print(f"  bars scored:            {len(rows):,}")
    print(f"  base rate (price up):   {base_up:.1%}  <- the number every condition must beat")
    print(f"  mean |move| over {horizon:>2}m:   {mean_move:.2f} pips")
    print(
        f"  required to break even: "
        + ("never (payoff < spread)" if math.isinf(required) else f"{required:.1%}")
    )
    print()

    cuts: dict[str, list[float]] = {}
    for condition, (field, labels) in CONDITIONS.items():
        if condition in ("RSI", "ADX", "Session (UTC)"):
            continue
        cuts[condition] = buckets([row[field] for row in rows], len(labels))

    findings = []
    for condition in CONDITIONS:
        print(f"{condition}")
        print(f"  {'bucket':<16} {'n':>6} {'up%':>7} {'vs base':>8} {'net pips':>9} {'p':>7} {'real?':>6}")
        print("  " + "-" * 66)
        for label in CONDITIONS[condition][1]:
            subset = [row for row in rows if label_for(condition, row, cuts) == label]
            if not subset:
                continue
            ups = sum(1 for row in subset if row["up"])
            n = len(subset)
            rate = ups / n
            p = two_sided_p(ups, n, base_up)
            # Net if you always bought in this bucket.
            net = statistics.fmean(
                (row["move"] - spread) for row in subset
            )
            findings.append((condition, label, n, rate, p))
            print(
                f"  {label:<16} {n:>6} {rate:>6.1%} {rate - base_up:>+7.1%} "
                f"{net:>+8.2f}p {p:>7.3f}"
            )
        print()

    # Multiple-comparison control across everything tested.
    p_values = [p for _, _, _, _, p in findings]
    keep = benjamini_hochberg(p_values)
    survivors = [
        (condition, label, n, rate, p)
        for (condition, label, n, rate, p), ok in zip(findings, keep)
        if ok
    ]
    print("AFTER CORRECTING FOR TESTING ALL OF THESE AT ONCE (Benjamini-Hochberg, 5%)")
    if survivors:
        for condition, label, n, rate, p in survivors:
            print(f"  {condition:<18} {label:<16} n={n:>6} up={rate:>6.1%} p={p:.4f}")
    else:
        print("  none. Not one bucket in this table is distinguishable from the base")
        print("  rate once you account for how many buckets were examined.")
    print()
    print("HOW TO READ THIS")
    print("  'up%' is how often price rose over the hold, NOT how often a signal won.")
    print("  A bucket is only interesting if 'vs base' is large AND the p-value is")
    print("  small AND it survives the correction at the bottom. A row that looks")
    print("  strong alone but drops out of that list was found by looking, not by")
    print("  the market telling you.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true", help="use the shipped real candles")
    parser.add_argument("--bars", type=int, default=6000)
    parser.add_argument("--paths", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--volatility", type=float, default=0.00024)
    parser.add_argument("--spread", type=float, default=charter.DEFAULT_SPREAD_PIPS)
    parser.add_argument("--stride", type=int, default=0,
                        help="bars between samples; 0 = the hold (non-overlapping)")
    args = parser.parse_args()

    print("WHERE SIGNALS WIN, WHERE THEY FAIL")
    if args.real:
        path = BOT_ROOT / "var" / "real_candles" / "eurusd_real_20261030_live.json"
        candles = load_real(path)
        print(f"  source: REAL EUR/USD M1, {path.name}")
        print(f"  {len(candles)} bars, {candles[0].timestamp:%Y-%m-%d %H:%M}Z -> "
              f"{candles[-1].timestamp:%H:%M}Z")
        paths = [candles]
    else:
        paths = [
            synthetic_series(
                args.bars, seed=4400 + index * 31, volatility=args.volatility, drift=0.0
            )
            for index in range(args.paths)
        ]
        print(f"  source: SYNTHETIC random walk, {args.paths} x {args.bars} bars")
        print("  (synthetic data has no real structure by construction: this run is")
        print("   the CONTROL. Any 'finding' here is noise, and the corrected list")
        print("   below should be empty. Compare it with --real.)")
    print(f"  spread: {args.spread:.2f} pips, hold: {args.horizon}m")
    print()
    stride = args.stride if args.stride > 0 else args.horizon
    rows = build_dataset(paths, args.horizon, stride)
    print(f"  sampling: every {stride} bars (non-overlapping windows)")
    if not rows:
        print("  no rows")
        return 1
    analyse(rows, args.spread, args.horizon)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
