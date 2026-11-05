#!/usr/bin/env python3
"""Standalone EUR/USD M1 barrier strategy research engine. Offline only.

REFERENCE COPY -- uploaded by the project owner, stored unmodified.

This is the honest implementation of docs/PROMPT_19_OF_20.md: it charges the
spread, reports net pips, splits chronologically, reports Wilson and bootstrap
intervals, forbids martingale, and states its own limitations. It is kept
verbatim because it is the CLEAN version of the experiment -- the counterpart to
scripts/prove_19_of_20_prompt.py, which asserts the outcome analytically.

Verified by scripts/verify_eurusd_m1.py: causal (0 truncation mismatches),
net_pips arithmetic correct, and the stop-first assumption on a bar that touches
both barriers is conservative and never fired in 932 trades.

Its result is the point, not a disappointment: 99%+ win rates, negative net
expectancy, and a break-even requirement of 99.96% that no edge can supply.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd

PIP = 0.0001

def wilson(k: int, n: int, z: float = 1.959963984540054):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z*z/n
    c = (p + z*z/(2*n))/d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return [c-h, c+h]

def bootstrap_ci(values, seed=7319, reps=4000):
    a = np.asarray(values, dtype=float)
    if len(a) < 2:
        return [None, None]
    rng = np.random.default_rng(seed)
    means = np.empty(reps)
    for i in range(reps):
        means[i] = rng.choice(a, len(a), replace=True).mean()
    return [float(np.quantile(means, .025)), float(np.quantile(means, .975))]

def max_drawdown(pnls):
    if not pnls:
        return 0.0
    eq = np.cumsum(np.asarray(pnls, dtype=float))
    peak = np.maximum.accumulate(np.r_[0.0, eq])[1:]
    return float(np.max(peak-eq))

def longest_loss_run(pnls):
    best = run = 0
    for x in pnls:
        run = run + 1 if x < 0 else 0
        best = max(best, run)
    return best

def load_csv(path):
    d = pd.read_csv(path)
    d.columns = [str(c).strip().lower() for c in d.columns]
    required = {"timestamp", "open", "high", "low", "close"}
    missing = required - set(d.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True, errors="raise")
    for c in ("open", "high", "low", "close"):
        d[c] = pd.to_numeric(d[c], errors="raise")
    d = d.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    if len(d) < 300:
        raise ValueError("At least 300 unique M1 bars are required.")
    if not (d.high >= d[["open", "close", "low"]].max(axis=1)).all():
        raise ValueError("Invalid OHLC high values.")
    if not (d.low <= d[["open", "close", "high"]].min(axis=1)).all():
        raise ValueError("Invalid OHLC low values.")
    return d

def signal_at(d, i, mode, rng):
    """Signal from completed candle i; entry at candle i+1 open."""
    if mode == "coinflip":
        return int(rng.choice([-1, 1]))
    if mode == "ema":
        close = d.close
        fast = close.ewm(span=9, adjust=False).mean()
        slow = close.ewm(span=21, adjust=False).mean()
        if fast.iloc[i] > slow.iloc[i]:
            return 1
        if fast.iloc[i] < slow.iloc[i]:
            return -1
        return 0
    if mode == "momentum":
        delta = float(d.close.iloc[i] - d.close.iloc[max(0, i-5)])
        return 1 if delta > 0 else (-1 if delta < 0 else 0)
    raise ValueError(f"Unknown signal mode: {mode}")

def simulate(d, start, end, tp, sl, spread, mode, seed):
    rng = np.random.default_rng(seed)
    out = []
    i = max(start, 21)
    end = min(end, len(d))
    while i + 1 < end:
        direction = signal_at(d, i, mode, rng)
        if direction == 0:
            i += 1
            continue
        entry_i = i + 1
        entry = float(d.open.iloc[entry_i])
        exit_i = None
        gross = None
        why = None
        j = entry_i
        while j < end:
            b = d.iloc[j]
            if direction > 0:
                hit_stop = b.low <= entry - sl*PIP
                hit_take = b.high >= entry + tp*PIP
            else:
                hit_stop = b.high >= entry + sl*PIP
                hit_take = b.low <= entry - tp*PIP
            # OHLC cannot identify intrabar order: conservative stop-first.
            if hit_stop:
                exit_i, gross, why = j, -sl, "STOP" if not hit_take else "STOP_AND_TARGET_SAME_BAR_STOP_FIRST"
                break
            if hit_take:
                exit_i, gross, why = j, tp, "TARGET"
                break
            j += 1
        if exit_i is None:
            # Discard unresolved position at the split boundary.
            break
        net = gross - spread
        out.append({
            "entry_time": d.timestamp.iloc[entry_i].isoformat(),
            "exit_time": d.timestamp.iloc[exit_i].isoformat(),
            "side": "BUY" if direction > 0 else "SELL",
            "gross_pips": float(gross), "spread_pips": float(spread),
            "net_pips": float(net), "exit_reason": why,
            "bars_held": int(exit_i-entry_i+1)
        })
        # Next eligible signal is after the exit candle: no same-candle re-entry.
        i = exit_i + 1
    return out

def metrics(trades):
    p = [t["net_pips"] for t in trades]
    n = len(p)
    wins = sum(x > 0 for x in p)
    win_values = [x for x in p if x > 0]
    loss_values = [-x for x in p if x < 0]
    aw = float(np.mean(win_values)) if win_values else None
    al = float(np.mean(loss_values)) if loss_values else None
    be = al/(aw+al) if aw is not None and al is not None else None
    return {
        "trades": n, "wins": wins, "losses": n-wins,
        "net_win_rate_pct": 100*wins/n if n else None,
        "wilson_95_win_rate_pct": [100*x if x is not None else None for x in wilson(wins,n)],
        "net_pips_total": float(sum(p)),
        "net_expectancy_pips_per_trade": float(np.mean(p)) if n else None,
        "expectancy_bootstrap_95_ci_pips": bootstrap_ci(p),
        "average_net_win_pips": aw, "average_net_loss_pips": al,
        "realised_break_even_win_rate_pct": 100*be if be is not None else None,
        "worst_single_trade_pips": min(p) if p else None,
        "longest_losing_run": longest_loss_run(p),
        "max_drawdown_pips": max_drawdown(p),
        "non_overlapping_trades": n
    }

def windows_19_20(trades):
    w = [t["net_pips"] > 0 for t in trades]
    return sum(sum(w[i:i+20]) >= 19 for i in range(max(0, len(w)-19)))

def evaluate(d, tp, sl, spread, mode, seed):
    cut = len(d)//2
    train = simulate(d, 0, cut, tp, sl, spread, mode, seed)
    test = simulate(d, cut, len(d), tp, sl, spread, mode, seed+1)
    return {
        "project": "EURUSD_M1_BARRIER_STRATEGY",
        "mode": "OFFLINE_RESEARCH_ONLY_NO_BROKER_CONNECTION",
        "dataset": {"bars": len(d), "first_timestamp": d.timestamp.iloc[0].isoformat(),
                    "last_timestamp": d.timestamp.iloc[-1].isoformat()},
        "parameters": {"take_profit_pips": tp, "stop_loss_pips": sl,
                       "round_trip_spread_pips": spread, "direction_mode": mode},
        "chronological_split": {"train_bars": cut, "held_out_bars": len(d)-cut},
        "train": {**metrics(train), "rolling_20_trade_windows_19plus_wins": windows_19_20(train)},
        "held_out": {**metrics(test), "rolling_20_trade_windows_19plus_wins": windows_19_20(test)},
        "trades": {"train": train, "held_out": test},
        "safety": {"martingale": False, "loss_based_size_increase": False,
                   "averaging_down": False, "adding_to_loser": False,
                   "same_candle_reentry": False, "live_execution": False},
        "limitations": [
            "OHLC bars do not reveal intrabar event order; stop is assumed first if both barriers touch.",
            "Spread is deducted once per round trip; commissions, slippage, swaps and variable spread are not included.",
            "Signals use completed candles and enter at the next candle open.",
            "Unresolved trades at each split boundary are excluded.",
            "Results depend on data quality and broker-specific EUR/USD pricing."
        ]
    }

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv")
    p.add_argument("--tp", type=float, default=1.25)
    p.add_argument("--sl", type=float, default=116.0)
    p.add_argument("--spread", type=float, default=1.2)
    p.add_argument("--signal", choices=["coinflip","ema","momentum"], default="ema")
    p.add_argument("--seed", type=int, default=20261105)
    p.add_argument("--out", default="eurusd_m1_report.json")
    a = p.parse_args()
    if a.tp <= 0 or a.sl <= 0 or a.spread < 0:
        p.error("TP/SL must be positive and spread cannot be negative.")
    if a.tp <= a.spread:
        print("WARNING: target is no larger than spread; target-winning trades are net-negative.")
    d = load_csv(a.csv)
    result = evaluate(d, a.tp, a.sl, a.spread, a.signal, a.seed)
    Path(a.out).write_text(json.dumps(result, indent=2))
    for name in ("train", "held_out"):
        m = result[name]
        print(f"{name}: n={m['trades']} wins={m['wins']} "
              f"win_rate={m['net_win_rate_pct']}% "
              f"net/trade={m['net_expectancy_pips_per_trade']} "
              f"worst={m['worst_single_trade_pips']} "
              f"max_DD={m['max_drawdown_pips']} "
              f"19/20_windows={m['rolling_20_trade_windows_19plus_wins']}")
        print("  expectancy bootstrap 95% CI:", m["expectancy_bootstrap_95_ci_pips"])
    print("Saved:", a.out)
    print("RESEARCH ONLY. No live orders placed.")

if __name__ == "__main__":
    main()
