"""Build supervisor fine-tuning data from candles.

The supervisor is an LLM (or any classifier) placed ABOVE the barrier engine:
at each candidate signal it sees only information available at the signal bar
and must answer APPROVE or VETO. A trade is labelled APPROVE if taking it
would have won net of spread, VETO if it would have lost. Training a model on
anything else -- raw direction, gross outcome, same-candle information --
would teach it the wrong job.

Labels come from the same barrier logic as reference/eurusd_m1.py: entry at
the next bar's open, stop-first on a bar touching both barriers, spread
charged once. Candidates are the EMA 9/21 signals of that engine, taken
non-overlapping (next candidate only after the previous exit), so the dataset
matches what the live supervisor would actually be asked about.

Output: JSONL with `prompt` / `completion` pairs for SFT, split
chronologically (first half train, second half held out), plus a printed
summary. The summary matters more than the file: it states how many labelled
trades exist, because fine-tuning on a few dozen examples teaches the model
the trainer's noise, not the market.

    python3 scripts/build_supervisor_dataset.py \
        --tp 3.0 --sl 17.0 --out var/supervisor_dataset.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

BOT_ROOT = Path(__file__).resolve().parents[1]
PIP = 0.0001


def load_candles(path: Path) -> list[dict]:
    if path.suffix == ".json":
        rows = json.loads(path.read_text())
        rows.sort(key=lambda r: r["ts"])
        return [
            {"ts": r["ts"], "o": float(r["o"]), "h": float(r["h"]),
             "l": float(r["l"]), "c": float(r["c"])}
            for r in rows
        ]
    import csv

    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    out = [
        {"ts": r["timestamp"], "o": float(r["open"]), "h": float(r["high"]),
         "l": float(r["low"]), "c": float(r["close"])}
        for r in rows
    ]
    out.sort(key=lambda r: r["ts"])
    return out


def ema(values: list[float], span: int) -> list[float]:
    alpha = 2.0 / (span + 1)
    out = [values[0]]
    for v in values[1:]:
        out.append(alpha * v + (1 - alpha) * out[-1])
    return out


def rsi(closes: list[float], period: int = 14) -> list[float]:
    out = [50.0] * len(closes)
    gains = [max(closes[i] - closes[i - 1], 0.0) for i in range(1, len(closes))]
    losses = [max(closes[i - 1] - closes[i], 0.0) for i in range(1, len(closes))]
    for i in range(period, len(gains)):
        g = sum(gains[i - period + 1: i + 1]) / period
        loss = sum(losses[i - period + 1: i + 1]) / period
        out[i + 1] = 100.0 if loss == 0 else 100.0 - 100.0 / (1.0 + g / loss)
    return out


def atr(candles: list[dict], period: int = 14) -> list[float]:
    trs = []
    for i, c in enumerate(candles):
        if i == 0:
            trs.append(c["h"] - c["l"])
        else:
            prev = candles[i - 1]["c"]
            trs.append(max(c["h"] - c["l"], abs(c["h"] - prev), abs(c["l"] - prev)))
    out, run = [], 0.0
    for i, tr in enumerate(trs):
        run += tr
        if i >= period:
            run -= trs[i - period]
        out.append(run / min(i + 1, period) / PIP)
    return out


def barrier_outcome(candles, entry_i, direction, tp, sl, spread, end):
    """Return net pips for a trade entered at candles[entry_i] open, or None."""
    entry = candles[entry_i]["o"]
    j = entry_i
    while j < end:
        b = candles[j]
        if direction > 0:
            hit_stop = b["l"] <= entry - sl * PIP
            hit_take = b["h"] >= entry + tp * PIP
        else:
            hit_stop = b["h"] >= entry + sl * PIP
            hit_take = b["l"] <= entry - tp * PIP
        if hit_stop:
            return -(sl + spread), j
        if hit_take:
            return (tp - spread), j
        j += 1
    return None, None


def fmt(x: float, digits: int = 2) -> str:
    return f"{x:.{digits}f}"


def build(candles, tp, sl, spread):
    closes = [c["c"] for c in candles]
    fast, slow = ema(closes, 9), ema(closes, 21)
    rsi_v, atr_v = rsi(closes), atr(candles)
    examples, i, end = [], 21, len(candles)
    while i + 1 < end:
        direction = 1 if fast[i] > slow[i] else (-1 if fast[i] < slow[i] else 0)
        if direction == 0:
            i += 1
            continue
        net, exit_i = barrier_outcome(candles, i + 1, direction, tp, sl, spread, end)
        if net is None:
            break
        c = candles[i]
        rng_pips = (c["h"] - c["l"]) / PIP
        close_pos = 0.0 if rng_pips == 0 else (c["c"] - c["l"]) / PIP / rng_pips
        ret5 = (closes[i] - closes[max(0, i - 5)]) / PIP
        prompt = (
            "You supervise a EUR/USD M1 barrier strategy "
            f"(target {tp} pips, stop {sl} pips, spread {spread} pips).\n"
            f"Candidate signal at {c['ts']}: {'BUY' if direction > 0 else 'SELL'}.\n"
            f"Signal-bar features (all known at bar close): "
            f"RSI14={fmt(rsi_v[i])}, ATR14={fmt(atr_v[i])} pips, "
            f"EMA9-EMA21 gap={fmt((fast[i] - slow[i]) / PIP)} pips, "
            f"5-bar return={fmt(ret5)} pips, close position in bar={fmt(close_pos)}, "
            f"bar range={fmt(rng_pips)} pips.\n"
            "Answer with exactly one word: APPROVE or VETO."
        )
        examples.append({
            "prompt": prompt,
            "completion": " APPROVE" if net > 0 else " VETO",
            "net_pips": net,
            "signal_time": c["ts"],
            "side": "BUY" if direction > 0 else "SELL",
        })
        i = exit_i + 1  # no overlap, no same-candle re-entry
    return examples


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candles", type=Path,
                   default=BOT_ROOT / "var/real_candles/eurusd_real_20261030_live.json")
    p.add_argument("--tp", type=float, default=3.0)
    p.add_argument("--sl", type=float, default=17.0)
    p.add_argument("--spread", type=float, default=1.2)
    p.add_argument("--out", type=Path, default=BOT_ROOT / "var/supervisor_dataset.jsonl")
    a = p.parse_args()

    candles = load_candles(a.candles)
    examples = build(candles, a.tp, a.sl, a.spread)
    cut = len(examples) // 2
    train, held = examples[:cut], examples[cut:]

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w") as fh:
        for ex in examples:
            fh.write(json.dumps({"prompt": ex["prompt"], "completion": ex["completion"],
                                 "held_out": ex in held}) + "\n")

    def base_rate(rows):
        return 100.0 * sum(r["net_pips"] > 0 for r in rows) / len(rows) if rows else float("nan")

    print(f"candles: {len(candles)}   tp={a.tp} sl={a.sl} spread={a.spread}")
    print(f"labelled candidate trades: {len(examples)} "
          f"(train {len(train)}, held out {len(held)})")
    print(f"approve base rate: train {base_rate(train):.1f}%  held out {base_rate(held):.1f}%")
    print(f"net if ALL approved: train {sum(r['net_pips'] for r in train):+.1f}p  "
          f"held out {sum(r['net_pips'] for r in held):+.1f}p")
    print(f"saved: {a.out}")
    if len(examples) < 200:
        print()
        print(f"WARNING: {len(examples)} examples is not fine-tuning data. It is")
        print("enough to EVALUATE a supervisor (does it beat approve-all?), not")
        print("to train one. A supervisor trained here would memorise these")
        print("trades. Collect months of M1 data before any fine-tune is run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
