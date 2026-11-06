#!/usr/bin/env python3
"""
REFERENCE COPY -- uploaded by the project owner, analysed unmodified.

This file is kept verbatim so that `scripts/the_file_edge_study.py` can measure
it reproducibly. Do not "fix" it here: the point of the reference copy is that
the study is measuring the strategy as written. Findings and the corrected
version (if any) belong in fusion/, not in this file.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class StrategyConfig:
    sweep_lookback: int = 10
    break_lookback: int = 5
    rsi_period: int = 14
    atr_period: int = 14
    ema_fast: int = 20
    ema_slow: int = 50
    threshold: int = 80
    min_atr_pct: float = 0.00002  # 0.2 pip relative to price; tune per feed
    max_atr_pct: float = 0.003    # avoid extreme volatility


def rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, pd.NA)
    out = 100 - (100 / (1 + rs))
    out = out.mask((loss == 0) & (gain > 0), 100)
    out = out.mask((gain == 0) & (loss > 0), 0)
    return out


def add_indicators(df: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    x = df.copy()
    x["ema_fast"] = x["close"].ewm(span=cfg.ema_fast, adjust=False).mean()
    x["ema_slow"] = x["close"].ewm(span=cfg.ema_slow, adjust=False).mean()
    x["rsi"] = rsi(x["close"], cfg.rsi_period)

    previous_close = x["close"].shift(1)
    true_range = pd.concat(
        [
            x["high"] - x["low"],
            (x["high"] - previous_close).abs(),
            (x["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    x["atr"] = true_range.ewm(alpha=1 / cfg.atr_period, adjust=False).mean()
    x["atr_pct"] = x["atr"] / x["close"]
    return x


def build_signals(m1: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    """Return M1 rows with signal, score, and explanatory conditions."""
    m1 = add_indicators(m1, cfg)

    # Resample into M3 candles. Label at candle close; only completed M3
    # values are joined to M1 timestamps, preventing use of unfinished bars.
    m3 = m1[["open", "high", "low", "close"]].resample(
        "3min", label="right", closed="left"
    ).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
    ).dropna()
    m3["ema_fast"] = m3["close"].ewm(span=cfg.ema_fast, adjust=False).mean()
    m3["ema_slow"] = m3["close"].ewm(span=cfg.ema_slow, adjust=False).mean()
    m3["trend_up"] = (m3["ema_fast"] > m3["ema_slow"]) & (
        m3["close"] > m3["ema_fast"]
    )
    m3["trend_down"] = (m3["ema_fast"] < m3["ema_slow"]) & (
        m3["close"] < m3["ema_fast"]
    )

    # At an M1 close, use only the most recently completed M3 candle.
    left = m1.reset_index().rename(columns={m1.index.name or "index": "timestamp"})
    # CSV timestamps mark M1 candle opens, so indicators become available
    # one minute later, at that candle's close.
    left["bar_close_time"] = left["timestamp"] + pd.Timedelta(minutes=1)
    right = m3[["trend_up", "trend_down"]].reset_index()
    right = right.rename(columns={right.columns[0]: "m3_close_time"})
    left = pd.merge_asof(
        left.sort_values("bar_close_time"),
        right.sort_values("m3_close_time"),
        left_on="bar_close_time",
        right_on="m3_close_time",
        direction="backward",
        allow_exact_matches=True,
    ).set_index("timestamp")

    # Exclude current candle from prior swing levels.
    prior_low = left["low"].shift(1).rolling(cfg.sweep_lookback).min()
    prior_high = left["high"].shift(1).rolling(cfg.sweep_lookback).max()
    prior_break_high = left["high"].shift(1).rolling(cfg.break_lookback).max()
    prior_break_low = left["low"].shift(1).rolling(cfg.break_lookback).min()

    bullish_sweep = (left["low"] < prior_low) & (left["close"] > prior_low)
    bearish_sweep = (left["high"] > prior_high) & (left["close"] < prior_high)
    bullish_break = left["close"] > prior_break_high
    bearish_break = left["close"] < prior_break_low
    bullish_momentum = left["rsi"] > 50
    bearish_momentum = left["rsi"] < 50
    volatility_ok = (
        (left["atr_pct"] >= cfg.min_atr_pct)
        & (left["atr_pct"] <= cfg.max_atr_pct)
    )

    buy_score = (
        left["trend_up"].fillna(False).astype(int) * 25
        + bullish_sweep.fillna(False).astype(int) * 25
        + bullish_break.fillna(False).astype(int) * 25
        + bullish_momentum.fillna(False).astype(int) * 15
        + volatility_ok.fillna(False).astype(int) * 10
    )
    sell_score = (
        left["trend_down"].fillna(False).astype(int) * 25
        + bearish_sweep.fillna(False).astype(int) * 25
        + bearish_break.fillna(False).astype(int) * 25
        + bearish_momentum.fillna(False).astype(int) * 15
        + volatility_ok.fillna(False).astype(int) * 10
    )

    left["buy_score"] = buy_score
    left["sell_score"] = sell_score
    left["signal"] = "NONE"
    left.loc[buy_score >= cfg.threshold, "signal"] = "BUY"
    left.loc[sell_score >= cfg.threshold, "signal"] = "SELL"

    # If both sides somehow qualify on one candle, reject the ambiguous setup.
    both = (buy_score >= cfg.threshold) & (sell_score >= cfg.threshold)
    left.loc[both, "signal"] = "NONE"
    left["score"] = left[["buy_score", "sell_score"]].max(axis=1)
    return left


def backtest(
    signals: pd.DataFrame, expiry_candles: int = 1
) -> tuple[pd.DataFrame, dict]:
    """
    Simple fixed-expiry directional test.
    Signal at candle close -> entry at next candle open -> compare with close
    after expiry_candles. Ties are recorded as DRAW.
    """
    if expiry_candles < 1:
        raise ValueError("expiry_candles must be at least 1")

    rows = []
    n = len(signals)
    for i in range(n - expiry_candles - 1):
        side = signals["signal"].iloc[i]
        if side not in ("BUY", "SELL"):
            continue

        entry_i = i + 1
        exit_i = entry_i + expiry_candles - 1
        entry = float(signals["open"].iloc[entry_i])
        exit_price = float(signals["close"].iloc[exit_i])

        if exit_price == entry:
            result = "DRAW"
        elif (side == "BUY" and exit_price > entry) or (
            side == "SELL" and exit_price < entry
        ):
            result = "WIN"
        else:
            result = "LOSS"

        rows.append(
            {
                "signal_time": signals.index[i],
                "side": side,
                "score": int(signals["score"].iloc[i]),
                "entry_time": signals.index[entry_i],
                "entry": entry,
                "exit_time": signals.index[exit_i],
                "exit": exit_price,
                "result": result,
            }
        )

    trades = pd.DataFrame(rows)
    if trades.empty:
        stats = {"trades": 0, "wins": 0, "losses": 0, "draws": 0, "win_rate_pct": 0.0}
    else:
        wins = int((trades["result"] == "WIN").sum())
        losses = int((trades["result"] == "LOSS").sum())
        draws = int((trades["result"] == "DRAW").sum())
        decided = wins + losses
        stats = {
            "trades": len(trades),
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_rate_pct": round(100 * wins / decided, 2) if decided else 0.0,
        }
    return trades, stats


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"timestamp", "open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {', '.join(sorted(missing))}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise", utc=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    df = df.set_index("timestamp")
    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="raise")
    if (df[["high", "low", "open", "close"]] <= 0).any().any():
        raise ValueError("OHLC prices must be positive")
    if (df["high"] < df[["open", "close", "low"]].max(axis=1)).any():
        raise ValueError("Invalid candle: high is below another OHLC value")
    if (df["low"] > df[["open", "close", "high"]].min(axis=1)).any():
        raise ValueError("Invalid candle: low is above another OHLC value")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="EUR/USD M1 + completed M3 strategy")
    parser.add_argument("csv", help="M1 OHLC CSV file")
    parser.add_argument("--expiry", type=int, default=1, help="M1 candles to evaluate (default: 1)")
    parser.add_argument("--threshold", type=int, default=80, help="Minimum score (default: 80)")
    parser.add_argument("--signals-out", default="eurusd_signals.csv")
    parser.add_argument("--trades-out", default="eurusd_backtest.csv")
    args = parser.parse_args()

    cfg = StrategyConfig(threshold=args.threshold)
    candles = load_csv(args.csv)
    result = build_signals(candles, cfg)
    trades, stats = backtest(result, args.expiry)

    result[["signal", "buy_score", "sell_score", "score"]].to_csv(args.signals_out)
    trades.to_csv(args.trades_out, index=False)

    print("Backtest:", stats)
    print("Signals saved:", args.signals_out)
    print("Trades saved:", args.trades_out)


if __name__ == "__main__":
    main()
