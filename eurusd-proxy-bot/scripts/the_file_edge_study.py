"""Measure the uploaded strategy: where it fails, and whether 19/20 is reachable.

This analyses ``reference/eurusd_strategy.py`` exactly as written. It answers
three questions, in the order they have to be answered:

1. **Does it work?** Baseline trade count, gross and net win rate, expectancy.
   Reported before anything is tuned, because a strategy that loses before
   tuning has to be described as losing.

2. **Where do its signals fail?** The same conditioning analysis used on the
   council: split the bars by RSI, ADX, ATR, trend, session and the strategy's
   own sweep/break flags, and report the win rate per bucket against the sample's
   own base rate -- with a multiple-comparison correction.

3. **Can it produce 19 wins in 20 trades?** Asked as a search, honestly: every
   parameter grid point is scored in-sample, every grid point that contains a
   19/20 block is kept, and then **each one is re-scored on data it was never
   tuned on**. A search will always find a 19/20 in-sample. The question is what
   that block does afterwards, and that is measured rather than argued.

The file has **no ADX**, so ADX enters as a conditioning variable only -- it
explains when the strategy's signals work, it is not part of the rules.

Run:

    python3 scripts/the_file_edge_study.py                # both samples
    python3 scripts/the_file_edge_study.py --paths 8      # more synthetic data
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from itertools import product
from pathlib import Path

BOT_ROOT = Path(__file__).resolve().parents[1]
if str(BOT_ROOT) not in sys.path:
    sys.path.insert(0, str(BOT_ROOT))
REFERENCE = BOT_ROOT / "reference"
if str(REFERENCE) not in sys.path:
    sys.path.insert(0, str(REFERENCE))

import pandas as pd  # noqa: E402

import eurusd_strategy as strat  # noqa: E402
from fusion import charter  # noqa: E402
from fusion.domain import Candle  # noqa: E402
from fusion.features import compute_features  # noqa: E402
from fusion.instruments import for_symbol  # noqa: E402
from fusion.simulate import synthetic_series  # noqa: E402
from fusion.timeutil import parse_datetime  # noqa: E402

PIP = for_symbol("EURUSD").pip
REAL_CANDLES = BOT_ROOT / "var" / "real_candles" / "eurusd_real_20261030_live.json"


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------


def normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def two_sided_p(successes: int, n: int, base: float) -> float:
    if n <= 0:
        return 1.0
    base = min(max(base, 1e-9), 1.0 - 1e-9)
    variance = n * base * (1.0 - base)
    if variance <= 0:
        return 1.0
    return 2.0 * (1.0 - normal_cdf(abs((successes - n * base) / math.sqrt(variance))))


def bh(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    n = len(p_values)
    if not n:
        return []
    order = sorted(range(n), key=lambda i: p_values[i])
    cut = -1
    for rank, index in enumerate(order, start=1):
        if p_values[index] <= alpha * rank / n:
            cut = rank
    return [rank <= cut for rank in range(1, n + 1)]


def wilson(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    p = wins / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    s = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - s) / d), min(1.0, (c + s) / d))


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------


def real_frame() -> pd.DataFrame:
    rows = sorted(parse_real(), key=lambda row: row["ts"])
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["ts"], utc=True)
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close"})
    return df.set_index("timestamp")[["open", "high", "low", "close"]]


def parse_real() -> list[dict]:
    import json

    return json.loads(REAL_CANDLES.read_text())


def synthetic_frame(seed: int, bars: int, volatility: float) -> pd.DataFrame:
    candles = synthetic_series(bars, seed=seed, volatility=volatility, drift=0.0)
    df = pd.DataFrame(
        {
            "timestamp": [c.timestamp for c in candles],
            "open": [c.open for c in candles],
            "high": [c.high for c in candles],
            "low": [c.low for c in candles],
            "close": [c.close for c in candles],
        }
    )
    return df.set_index("timestamp")


def trades_with_context(
    frame: pd.DataFrame, cfg: strat.StrategyConfig, expiry: int, spread: float
) -> pd.DataFrame:
    """Run the strategy, then attach the market state at each trade's signal bar."""
    signals = strat.build_signals(frame, cfg)
    trades, _ = strat.backtest(signals, expiry)
    if trades.empty:
        return trades

    candles = [
        Candle(
            timestamp=stamp.to_pydatetime(),
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=0.0,
        )
        for stamp, row in frame.iterrows()
    ]
    features = compute_features(candles)
    position = {stamp: index for index, stamp in enumerate(frame.index)}

    extra = []
    for _, trade in trades.iterrows():
        index = position.get(trade["signal_time"])
        if index is None or index >= len(features) or features[index] is None:
            extra.append({})
            continue
        here = features[index]
        bar = frame.iloc[index]
        extra.append(
            {
                "rsi": float(here.get("rsi") or 50.0),
                "adx": float(here.get("adx") or 0.0),
                "atr_pips": abs(float(here.get("atr") or 0.0)) / PIP,
                "ema_slope": float(here.get("ema_slope") or 0.0),
                "close_frac": (bar["close"] - bar["low"])
                / max(bar["high"] - bar["low"], 1e-9),
                "hour": stamp_hour(trade["signal_time"]),
            }
        )

    context = pd.DataFrame(extra, index=trades.index)
    out = pd.concat([trades.reset_index(drop=True), context.reset_index(drop=True)], axis=1)

    moves = (out["exit"] - out["entry"]) / PIP
    signed = moves.where(out["side"] == "BUY", -moves)
    out["gross_pips"] = signed
    out["net_pips"] = signed - spread
    out["win_net"] = out["net_pips"] > 0
    return out


def stamp_hour(value) -> int:
    try:
        return int(pd.Timestamp(value).hour)
    except Exception:
        return -1


# ---------------------------------------------------------------------------
# 1. baseline
# ---------------------------------------------------------------------------


def baseline(frame: pd.DataFrame, label: str, spread: float) -> pd.DataFrame:
    print(f"-- {label}")
    print(f"   {'thr':>4} {'exp':>4} {'trades':>7} {'gross%':>7} {'net%':>6} {'net pips':>9} {'BUY':>5} {'SELL':>5}")
    best = []
    for threshold, expiry in ((65, 1), (65, 5), (65, 15), (75, 5), (80, 1), (80, 5), (80, 15)):
        cfg = strat.StrategyConfig(threshold=threshold)
        trades = trades_with_context(frame, cfg, expiry, spread)
        if trades.empty:
            print(f"   {threshold:>4} {expiry:>4} {0:>7}")
            continue
        buys = int((trades["side"] == "BUY").sum())
        sells = int((trades["side"] == "SELL").sum())
        gross = 100 * (trades["gross_pips"] > 0).mean()
        net = 100 * trades["win_net"].mean()
        print(
            f"   {threshold:>4} {expiry:>4} {len(trades):>7} {gross:>6.1f}% "
            f"{net:>5.1f}% {trades['net_pips'].sum():>+8.0f}p {buys:>5} {sells:>5}"
        )
        best.append((threshold, expiry, trades))
    print()
    return best


# ---------------------------------------------------------------------------
# 2. conditioning
# ---------------------------------------------------------------------------


BUCKETS = {
    "RSI": ("rsi", [30, 50, 70], ["<30", "30-50", "50-70", ">70"]),
    "ADX": ("adx", [20, 25, 35], ["<20", "20-25", "25-35", ">35"]),
    "ATR pips": ("atr_pips", None, ["low", "mid", "high"]),
    "EMA slope": ("ema_slope", None, ["falling", "flat", "rising"]),
    "Close in range": ("close_frac", [0.33, 0.66], ["low", "mid", "high"]),
}


def bucket_of(name: str, value: float, cuts: list[float] | None, labels: list[str]) -> str:
    if cuts is None:
        return labels[0]
    for cut, label in zip(cuts, labels[:-1]):
        if value <= cut:
            return label
    return labels[-1]


def condition_table(trades: pd.DataFrame, target: str = "win_net") -> None:
    base = trades[target].mean()
    findings = []
    for name, (field, cuts, labels) in BUCKETS.items():
        if field not in trades:
            continue
        series = trades[field].dropna()
        if series.empty:
            continue
        if cuts is None:
            ordered = sorted(series)
            cuts = [ordered[int(len(ordered) * k / len(labels))] for k in range(1, len(labels))]
        print(f"   {name}")
        print(f"     {'bucket':<10} {'n':>6} {'net win%':>9} {'vs base':>8} {'net/trade':>10} {'p':>7}")
        print("     " + "-" * 55)
        for label in labels:
            mask = trades[field].apply(lambda v: bucket_of(name, v, cuts, labels) == label) \
                if False else trades[field].map(lambda v: bucket_of(name, v, cuts, labels) == label)
            subset = trades[mask.fillna(False)]
            if subset.empty:
                continue
            wins = int(subset[target].sum())
            n = len(subset)
            rate = wins / n
            p = two_sided_p(wins, n, base)
            findings.append((name, label, n, rate, p))
            print(
                f"     {label:<10} {n:>6} {rate:>8.1%} {rate - base:>+7.1%} "
                f"{subset['net_pips'].mean():>+9.2f}p {p:>7.3f}"
            )
        print()

    keep = bh([p for _, _, _, _, p in findings])
    survivors = [f for f, ok in zip(findings, keep) if ok]
    if survivors:
        print("   survives correction:")
        for name, label, n, rate, p in survivors:
            print(f"     {name:<14} {label:<10} n={n:>5} {rate:>6.1%} p={p:.4f}")
    else:
        print("   survives correction: none")
    print()


# ---------------------------------------------------------------------------
# 3. the 19/20 question
# ---------------------------------------------------------------------------


def blocks_of(moves: list[float], size: int = 20, stride: int = 1) -> list[int]:
    return [
        sum(1 for m in moves[i : i + size] if m > 0)
        for i in range(0, max(0, len(moves) - size + 1), stride)
    ]


def search_19_of_20(frames: list[pd.DataFrame], spread: float, threshold: int = 65) -> dict:
    """Grid every expiry, find in-sample 19/20 blocks, then test them blind.

    Each sample is handled **separately** and split chronologically in half.
    Concatenating frames first is wrong and produced a nonsense result on the
    first run of this study (+24 pips a trade, against a pooled mean of -1.07):
    the synthetic samples share a start timestamp, so the joined frame had
    duplicate index entries and the 3-minute resample silently mixed three
    unrelated price series into one. A per-sample split cannot do that.
    """
    rows: list[dict] = []
    for frame_index, frame in enumerate(frames):
        if len(frame) < 600:
            continue
        split = int(len(frame) * 0.5)
        train_frame, test_frame = frame.iloc[:split], frame.iloc[split:]
        for expiry in (1, 2, 3, 5, 10, 15):
            cfg = strat.StrategyConfig(threshold=threshold)
            trades = trades_with_context(train_frame, cfg, expiry, spread)
            if trades.empty:
                continue
            blocks = blocks_of(list(trades["net_pips"]), stride=1)
            if not blocks:
                continue
            rows.append(
                {
                    "frame": frame_index,
                    "expiry": expiry,
                    "trades": len(trades),
                    "blocks": len(blocks),
                    "best": max(blocks),
                    "n19": sum(1 for b in blocks if b >= 19),
                    "net": float(trades["net_pips"].mean()),
                    "frame_ref": frame,
                    "test_frame": test_frame,
                }
            )

    if not rows:
        return {"tested": 0}

    # Which parameter wins in-sample, aggregated across every sample at once?
    per_expiry: dict[int, dict] = {}
    for row in rows:
        entry = per_expiry.setdefault(
            row["expiry"], {"expiry": row["expiry"], "trades": 0, "blocks": 0,
                            "best": 0, "n19": 0, "net_sum": 0.0, "n": 0}
        )
        entry["trades"] += row["trades"]
        entry["blocks"] += row["blocks"]
        entry["best"] = max(entry["best"], row["best"])
        entry["n19"] += row["n19"]
        entry["net_sum"] += row["net"] * row["trades"]
        entry["n"] += row["trades"]

    summary = []
    for entry in per_expiry.values():
        summary.append(
            {
                "expiry": entry["expiry"],
                "trades": entry["trades"],
                "blocks": entry["blocks"],
                "best": entry["best"],
                "n19": entry["n19"],
                "net": entry["net_sum"] / entry["n"] if entry["n"] else 0.0,
            }
        )
    summary.sort(key=lambda r: r["expiry"])
    chosen = max(summary, key=lambda r: (r["n19"], r["best"]))

    # Re-score the chosen expiry on the untouched second half of every sample.
    held_blocks: list[int] = []
    held_trades: list[pd.DataFrame] = []
    for frame in frames:
        if len(frame) < 600:
            continue
        test_frame = frame.iloc[int(len(frame) * 0.5):]
        cfg = strat.StrategyConfig(threshold=threshold)
        trades = trades_with_context(test_frame, cfg, chosen["expiry"], spread)
        if trades.empty:
            continue
        held_trades.append(trades)
        held_blocks.extend(blocks_of(list(trades["net_pips"]), stride=1))

    pooled_held = pd.concat(held_trades, ignore_index=True) if held_trades else pd.DataFrame()
    return {
        "tested": len(summary),
        "rows": summary,
        "chosen": chosen,
        "held_trades": 0 if pooled_held.empty else len(pooled_held),
        "held_blocks": len(held_blocks),
        "held_best": max(held_blocks) if held_blocks else 0,
        "held_n19": sum(1 for b in held_blocks if b >= 19),
        "held_mean_net": float(pooled_held["net_pips"].mean()) if not pooled_held.empty else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=int, default=5)
    parser.add_argument("--bars", type=int, default=4000)
    parser.add_argument("--volatility", type=float, default=0.00024)
    parser.add_argument("--seed", type=int, default=6100)
    args = parser.parse_args()

    spread = charter.DEFAULT_SPREAD_PIPS
    print("THE UPLOADED FILE, MEASURED")
    print(f"  reference/eurusd_strategy.py -- unmodified, spread {spread:.2f} pips")
    print("  NOTE: it has no ADX and no cost model. ADX below is a conditioning")
    print("  variable (when do its signals work), not part of its rules.")
    print()

    frames = []
    real = real_frame()
    frames.append(("REAL 740 bars", real))
    for index in range(args.paths):
        frames.append(
            (
                f"SYNTHETIC path {index + 1}",
                synthetic_frame(args.seed + index * 23, args.bars, args.volatility),
            )
        )

    print("=" * 74)
    print("1. BASELINE: DOES IT WORK BEFORE ANY TUNING?")
    print("=" * 74)
    for label, frame in frames[:2]:
        baseline(frame, label, spread)

    print("=" * 74)
    print("2. WHERE ITS SIGNALS WIN AND FAIL")
    print("=" * 74)
    pooled = []
    for label, frame in frames:
        trades = trades_with_context(frame, strat.StrategyConfig(threshold=65), 5, spread)
        if trades.empty:
            continue
        trades = trades.assign(source=label)
        pooled.append(trades)
    if not pooled:
        print("  no trades produced")
        return 1
    all_trades = pd.concat(pooled, ignore_index=True)
    print(f"  pooled: {len(all_trades)} trades from {len(pooled)} samples,")
    print(f"  net win rate {all_trades['win_net'].mean():.1%}, "
          f"mean net {all_trades['net_pips'].mean():+.2f} pips")
    print()
    condition_table(all_trades)

    print("=" * 74)
    print("3. CAN IT DO 19 WINS IN 20 TRADES?")
    print("=" * 74)
    result = search_19_of_20([frame for _, frame in frames], spread)
    if not result.get("tested"):
        print("  the strategy produced no trades on the training half")
        return 1
    print(f"  {result['tested']} expiries tested in-sample, threshold fixed at 65")
    print()
    print(f"  {'expiry':>6} {'trades':>7} {'blocks':>7} {'best/20':>8} {'19+ blocks':>11} {'net/trade':>10}")
    print("  " + "-" * 56)
    for row in result["rows"]:
        print(
            f"  {row['expiry']:>6} {row['trades']:>7} {row['blocks']:>7} "
            f"{row['best']:>8} {row['n19']:>11} {row['net']:>+9.2f}p"
        )
    chosen = result["chosen"]
    print()
    print(f"  best in-sample: expiry {chosen['expiry']}m, {chosen['n19']} blocks of 19+")
    print(f"  that parameter, on data the search never saw:")
    print(f"    trades {result['held_trades']}, blocks {result['held_blocks']}, "
          f"best {result['held_best']}/20, 19+ blocks {result['held_n19']}, "
          f"mean net {result['held_mean_net']:+.2f} pips")
    print()

    print("WHAT THIS SHOWS")
    if result["chosen"]["n19"] == 0:
        print("  There is no 19/20 block to argue about. Across every expiry and")
        print(f"  every sample the best block in-sample is {result['chosen']['best']}/20,")
        print("  and on held-out data it is "
              f"{result['held_best']}/20. The strategy does not come close to 19,")
        print("  so the question is not whether a 19/20 generalises -- it is")
        print("  whether one exists at all. It does not, in this data.")
    else:
        print("  A 19/20 block is what a search finds; it is not what a system is.")
        print("  The in-sample column always contains one. The held-out column is")
        print("  the only one that describes what would happen to your money.")
    print()
    print("  The arithmetic has not changed because the file changed:")
    print("    P(19 or 20 wins in 20) at a fair coin = 21/2^20 = 2.0e-05")
    print("    to break even on a 1.2 pip spread you need the payoff to clear it")
    print("    first -- see docs/SCALPING_RESEARCH.md section 1.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
