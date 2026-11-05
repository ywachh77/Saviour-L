"""The genuine 19/20 hunt.

A "genuine" 19/20 means: >=95% win rate SUSTAINED OUT OF SAMPLE with
positive net expectancy after spread. The barrier geometry alone can
manufacture the 95% (a stop 19x the target does it for a coin flip), but
at that geometry break-even is 95% + spread/(tp+sl), so a genuine result
requires a real directional edge of exactly that size:

    tp 2 / sl 38   -> break-even 98.00%, edge needed +3.0pp over driftless
    tp 5 / sl 95   -> break-even 96.20%, edge needed +1.2pp
    tp 10 / sl 190 -> break-even 95.60%, edge needed +0.6pp

This script hunts for that edge on REAL tick-aggregated EUR/USD M1 data
with rules that are declared here, before any numbers are seen:

    coinflip   control (fixed seed) -- must NOT pass; if it does, the
               test itself is broken
    ema        EMA9 vs EMA21 trend following
    momentum   5-bar momentum sign
    meanrev    RSI14 < 30 buy / > 70 sell
    ema+mom    both agree, else skip

Protocol: causal features only, entry at next bar open, stop-first on a
bar touching both barriers, spread charged once, non-overlapping trades,
chronological 50/50 split, only the held-out half is evidence.

VERDICT RULE (a rule is a genuine 19/20 only if ALL hold, held out):
    win rate >= 95%, Wilson 95% LOWER bound > break-even win rate,
    net pips/trade > 0, and n >= 100 trades.

    python3 scripts/genuine_19_hunt.py --csv /path/to/eurusd_m1.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from datetime import datetime
from pathlib import Path

PIP = 0.0001
GEOMETRIES = [(2.0, 38.0), (5.0, 95.0), (10.0, 190.0)]  # all 1:19 -> ~95%
RULES = ["coinflip", "ema", "momentum", "meanrev", "ema+mom"]


def load(path: Path):
    ts, o, h, l, c = [], [], [], [], []
    with path.open() as fh:
        for row in csv.DictReader(fh):
            ts.append(row["timestamp"])
            o.append(float(row["open"])); h.append(float(row["high"]))
            l.append(float(row["low"])); c.append(float(row["close"]))
    return ts, o, h, l, c


def ema_series(v, span):
    a = 2.0 / (span + 1)
    out = [v[0]]
    for x in v[1:]:
        out.append(a * x + (1 - a) * out[-1])
    return out


def rsi_series(closes, period=14):
    out = [50.0] * len(closes)
    for i in range(period + 1, len(closes)):
        gains = losses = 0.0
        for j in range(i - period + 1, i + 1):
            d = closes[j] - closes[j - 1]
            if d > 0:
                gains += d
            else:
                losses -= d
        out[i] = 100.0 if losses == 0 else 100.0 - 100.0 / (1.0 + gains / losses)
    return out


def wilson(k, n, z=1.959963984540054):
    if not n:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    cen = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (cen - half), 100 * (cen + half))


def direction(rule, i, fast, slow, rsi, closes, rng):
    if rule == "coinflip":
        return rng.choice([-1, 1])
    if rule == "ema":
        return 1 if fast[i] > slow[i] else (-1 if fast[i] < slow[i] else 0)
    if rule == "momentum":
        d = closes[i] - closes[max(0, i - 5)]
        return 1 if d > 0 else (-1 if d < 0 else 0)
    if rule == "meanrev":
        if rsi[i] < 30:
            return 1
        if rsi[i] > 70:
            return -1
        return 0
    if rule == "ema+mom":
        e = 1 if fast[i] > slow[i] else (-1 if fast[i] < slow[i] else 0)
        d = closes[i] - closes[max(0, i - 5)]
        m = 1 if d > 0 else (-1 if d < 0 else 0)
        return e if e == m else 0
    raise ValueError(rule)


def simulate(rule, start, end, tp, sl, spread, o, h, l, c, fast, slow, rsi, seed):
    rng = random.Random(seed)
    nets = []
    i = max(start, 21)
    while i + 1 < end:
        d = direction(rule, i, fast, slow, rsi, c, rng)
        if d == 0:
            i += 1
            continue
        entry = o[i + 1]
        j = i + 1
        net = None
        while j < end:
            if d > 0:
                hit_stop = l[j] <= entry - sl * PIP
                hit_take = h[j] >= entry + tp * PIP
            else:
                hit_stop = h[j] >= entry + sl * PIP
                hit_take = l[j] <= entry - tp * PIP
            if hit_stop:
                net = -(sl + spread)
                break
            if hit_take:
                net = tp - spread
                break
            j += 1
        if net is None:
            break
        nets.append(net)
        i = j + 1
    return nets


def summarise(nets, tp, sl, spread):
    n = len(nets)
    wins = sum(1 for x in nets if x > 0)
    wr = 100.0 * wins / n if n else None
    be = 100.0 * (sl + spread) / (tp + sl)
    lo, hi = wilson(wins, n)
    eq = peak = dd = 0.0
    for x in nets:
        eq += x
        peak = max(peak, eq)
        dd = max(dd, peak - eq)
    flags = [x > 0 for x in nets]
    return {
        "n": n, "win_rate": wr, "wilson_lo": lo, "wilson_hi": hi,
        "break_even": be,
        "net_per_trade": (sum(nets) / n) if n else None,
        "worst": min(nets) if n else None, "max_dd": dd,
        "w19": sum(1 for k in range(max(0, n - 19)) if sum(flags[k:k + 20]) >= 19),
        "w20": sum(1 for k in range(max(0, n - 19)) if sum(flags[k:k + 20]) == 20),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--spread", type=float, default=1.2)
    p.add_argument("--seed", type=int, default=99)
    a = p.parse_args()

    ts, o, h, l, c = load(a.csv)
    print(f"data: {len(c)} M1 bars  {ts[0]} .. {ts[-1]}  spread charged={a.spread}p")
    fast, slow, rsi = ema_series(c, 9), ema_series(c, 21), rsi_series(c)
    cut = len(c) // 2
    print(f"split: train {cut} bars | held out {len(c) - cut} bars "
          f"(held out starts {ts[cut]})\n")

    genuine = []
    hdr = (f"{'rule':<9} {'tp/sl':<10} {'n':>5} {'win%':>7} {'WilsonLo':>9} "
           f"{'need%':>7} {'net/tr':>8} {'worst':>8} {'maxDD':>8} {'19+w':>6} {'20/20':>6}  verdict")
    print(hdr)
    print("-" * len(hdr))
    for gi, (tp, sl) in enumerate(GEOMETRIES):
        for rule in RULES:
            nets = simulate(rule, cut, len(c), tp, sl, a.spread,
                            o, h, l, c, fast, slow, rsi, a.seed + gi)
            s = summarise(nets, tp, sl, a.spread)
            ok = (s["n"] >= 100 and s["win_rate"] is not None
                  and s["win_rate"] >= 95.0
                  and s["wilson_lo"] is not None and s["wilson_lo"] > s["break_even"]
                  and s["net_per_trade"] is not None and s["net_per_trade"] > 0)
            if ok:
                genuine.append((rule, tp, sl, s))
            print(f"{rule:<9} {f'{tp:g}/{sl:g}':<10} {s['n']:>5} "
                  f"{s['win_rate']:>6.2f}% {s['wilson_lo']:>8.2f}% "
                  f"{s['break_even']:>6.2f}% {s['net_per_trade']:>+7.3f}p "
                  f"{s['worst']:>7.1f}p {s['max_dd']:>7.1f}p {s['w19']:>6} {s['w20']:>6}  "
                  f"{'GENUINE' if ok else '-'}")
    print()
    if genuine:
        print("GENUINE 19/20 FOUND (held out):")
        for rule, tp, sl, s in genuine:
            print(f"  {rule} tp={tp} sl={sl}: {s['win_rate']:.2f}% win "
                  f"(Wilson lo {s['wilson_lo']:.2f}% > break-even {s['break_even']:.2f}%), "
                  f"net {s['net_per_trade']:+.3f}p/trade over {s['n']} trades")
    else:
        print("NO GENUINE 19/20: no predeclared rule cleared the held-out bar.")
        print("Any 95% win rates above sit below their break-even rate or fail")
        print("the Wilson bound -- geometry wearing a costume, not edge.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
