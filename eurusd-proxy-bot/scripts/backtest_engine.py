#!/usr/bin/env python3
"""
Walk-Forward Backtester + Self-Learning Weight Optimizer
==========================================================

Backtests the full signal pipeline (ML ensemble + combiner + gates) on real
candle data, walking forward one candle at a time with no lookahead.

Outputs:
  - Backtest report (win rate, pips, Sharpe, drawdown)
  - Optimal combiner weights (self-learning)
  - Per-regime performance breakdown
  - Recommended confidence threshold for "sureshot" filtering

Usage:
  python3 scripts/backtest_engine.py                    # Full backtest
  python3 scripts/backtest_engine.py --weights          # Optimize weights only
  python3 scripts/backtest_engine.py --candles 10000    # Use N candles
"""

import argparse
import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fusion.domain import Candle
from fusion.features import compute_features, FEATURE_NAMES, FEATURE_VERSION
from fusion.gbdt import GradientBoostingModel
from fusion.indicators import (
    adx, atr, rsi, stochastic, vwap, supertrend_window,
    ichimoku_cloud, aroon, choppiness_index, keltner_squeeze,
    hurst_exponent,
)

CANDLE_DIR = PROJECT_ROOT / "var" / "real_candles"
REPORT_JSON = CANDLE_DIR / "backtest_report.json"
WEIGHTS_JSON = CANDLE_DIR / "optimal_weights.json"
MODEL_PATH = CANDLE_DIR / "ensemble_model.json"
MODEL_1BAR_PATH = CANDLE_DIR / "ensemble_model_1bar.json"
MODEL_MR_PATH = CANDLE_DIR / "ensemble_model_mr.json"

SPREAD_PIPS = 1.2  # EUR/USD typical spread
PIPS_PER_MOVE = 0.0001  # 1 pip = 0.0001 for EUR/USD


# ─── Default combiner weights ─────────────────────────────────────────
DEFAULT_WEIGHTS = {
    "ml_ensemble": 0.30,
    "supertrend": 0.15,
    "rsi": 0.10,
    "stochastic": 0.10,
    "vwap": 0.08,
    "ichimoku": 0.05,
    "aroon": 0.05,
    "choppiness": 0.05,
    "squeeze": 0.05,
    "hurst": 0.03,
    "mr_blend": 0.09,  # mean-reversion model blend
}


def load_candles(n_candles=None):
    """Load candles from batch files, sorted by timestamp."""
    candles = []
    for batch_file in sorted(CANDLE_DIR.glob("batch_*.txt")):
        for line in batch_file.read_text().strip().split(";"):
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 5:
                continue
            ts = int(float(parts[0]))
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            candles.append(Candle(
                timestamp=dt,
                open=float(parts[1]),
                high=float(parts[2]),
                low=float(parts[3]),
                close=float(parts[4]),
                volume=float(parts[5]) if len(parts) > 5 else 0.0,
            ))
    candles.sort(key=lambda c: c.timestamp)
    if n_candles:
        candles = candles[-n_candles:]
    return candles


def load_models(expiry_minutes=30):
    """Load the appropriate model ensemble based on expiry."""
    model_path = MODEL_1BAR_PATH if expiry_minutes == 1 else MODEL_PATH
    models = []
    if model_path.is_file():
        with open(model_path, encoding="utf-8") as f:
            data = json.load(f)
        for m_dict in data.get("models", []):
            m = GradientBoostingModel.from_dict(m_dict)
            if m is not None:
                models.append(m)
    return models


def load_mr_model():
    """Load the mean-reversion model."""
    models = []
    if MODEL_MR_PATH.is_file():
        with open(MODEL_MR_PATH, encoding="utf-8") as f:
            data = json.load(f)
        for m_dict in data.get("models", []):
            m = GradientBoostingModel.from_dict(m_dict)
            if m is not None:
                models.append(m)
    return models


def compute_indicator_votes(candles, index):
    """Compute all indicator votes for a given candle index."""
    window = candles[max(0, index - 99): index + 1]
    closes = [c.close for c in window]
    last = len(closes) - 1

    rsi_val = rsi(closes, 14)[last] if len(closes) >= 14 else 50.0
    atr_vals = atr(window, 14)
    atr_val = atr_vals[last] if atr_vals else 0.01
    st_dir, _ = supertrend_window(window, period=10, multiplier=3.0)
    st_direction = st_dir[last] if st_dir else 0.0
    stoch_k, _ = stochastic(window, 14, 3)
    stoch_val = stoch_k[last] if stoch_k else 50.0
    vwap_vals = vwap(window, 30)
    vwap_val = vwap_vals[last] if vwap_vals else closes[last]
    ichimoku_kijun, _, _, _, _ = ichimoku_cloud(window, 9, 26, 52)
    ichimoku_val = ichimoku_kijun[last] if ichimoku_kijun else closes[last]
    aroon_up, aroon_down = aroon(window, 25)
    aroon_u = aroon_up[last] if aroon_up else 50.0
    aroon_d = aroon_down[last] if aroon_down else 50.0
    chop_vals = choppiness_index(window, 14)
    chop_val = chop_vals[last] if chop_vals else 50.0
    squeeze_bools, _ = keltner_squeeze(window, bb_period=20, bb_mult=2.0, kc_period=20, kc_mult=1.5)
    squeeze_on = bool(squeeze_bools[last]) if squeeze_bools else False
    hurst_vals = hurst_exponent(closes, 20)
    hurst_val = hurst_vals[last] if hurst_vals else 0.5

    votes = {
        "supertrend": 1.0 if st_direction > 0 else -1.0,
        "rsi": (0.5 if rsi_val < 40 else (1.0 if rsi_val < 50 else (-0.5 if rsi_val > 60 else (-1.0 if rsi_val > 70 else 0.0)))),
        "stochastic": 1.0 if stoch_val < 20 else (-1.0 if stoch_val > 80 else 0.0),
        "vwap": 1.0 if closes[last] > vwap_val else -1.0,
        "ichimoku": 1.0 if closes[last] > ichimoku_val else -1.0,
        "aroon": 1.0 if aroon_u > 70 else (-1.0 if aroon_d > 70 else 0.0),
        "choppiness": -1.0 if chop_val > 70 else (1.0 if chop_val < 40 else 0.0),
        "squeeze": 1.0 if squeeze_on else 0.0,
        "hurst": 1.0 if hurst_val > 0.5 else (-1.0 if hurst_val < 0.45 else 0.0),
    }

    regime = {
        "chop": chop_val,
        "hurst": hurst_val,
        "is_range": chop_val > 70.0 and hurst_val < 0.45,
        "is_mr_regime": (chop_val > 65.0 and hurst_val < 0.48),
    }

    return votes, regime, {
        "rsi": rsi_val, "atr": atr_val, "chop": chop_val,
        "hurst": hurst_val, "close": closes[last],
    }


def run_backtest(candles, weights=None, expiry_minutes=30, confidence_threshold=0.0):
    """Run walk-forward backtest.

    For each candle i (with enough history):
      1. Compute features up to i
      2. Run ML ensemble prediction
      3. Compute combiner score
      4. Check confidence threshold
      5. Compare prediction with actual next-bar movement
      6. Track wins/losses/pips

    Returns detailed results dict.
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    models = load_models(expiry_minutes)
    mr_models = load_mr_model()

    if not models:
        print("ERROR: No models loaded. Run train_robust_model.py first.")
        return None

    # Compute features once for all candles
    print(f"Computing features for {len(candles)} candles...")
    feat_dicts = compute_features(candles)

    horizon = expiry_minutes
    min_bars = 60  # Need enough history for indicators

    trades = []
    equity = 10000.0  # Starting equity in USD
    peak_equity = equity
    max_drawdown = 0.0

    per_regime = {"trend": [], "range": [], "unknown": []}
    per_regime_results = {"trend": {"wins": 0, "losses": 0, "pips": 0.0},
                          "range": {"wins": 0, "losses": 0, "pips": 0.0},
                          "unknown": {"wins": 0, "losses": 0, "pips": 0.0}}

    for i in range(min_bars, len(candles) - horizon):
        feat = feat_dicts[i]
        if feat is None:
            continue

        # ML prediction
        ml_up = sum(m.predict_up(feat) for m in models) / len(models)

        # MR blending if in mean-reversion regime
        votes, regime, _ = compute_indicator_votes(candles, i)
        if mr_models and regime["is_mr_regime"]:
            mr_up = sum(m.predict_up(feat) for m in mr_models) / len(mr_models)
            ml_up = 0.6 * ml_up + 0.4 * mr_up

        # Regime-aware flip for 1-minute expiry in RANGE regime
        flip_applied = False
        if regime["is_range"] and expiry_minutes == 1:
            candle = candles[i]
            body = abs(candle.close - candle.open)
            rng = candle.high - candle.low
            if rng > 0 and (body / rng) >= 0.3:
                ml_up = 0.65 * (1.0 - ml_up) + 0.35 * ml_up
                flip_applied = True

        # ML vote score
        ml_score = (ml_up - 0.5) * 2

        # Combiner score
        vote_map = {
            "ml_ensemble": ml_score,
            **votes,
        }
        if mr_models:
            vote_map["mr_blend"] = 0.0 if not regime["is_mr_regime"] else (1.0 if ml_up > 0.5 else -1.0)

        total_w = sum(weights.get(k, 0) for k in vote_map) or 1.0
        combined = sum(vote_map[k] * weights.get(k, 0) for k in vote_map) / total_w

        # Confidence threshold (sureshot filter)
        confidence = abs(combined)
        if confidence_threshold > 0 and confidence < confidence_threshold:
            continue  # Skip low-confidence signals

        # No-trade zone: skip 1-minute expiry in RANGE regime
        # The 1-bar model has no edge in RANGE regime (random data at 1-bar horizon)
        if expiry_minutes == 1 and regime["is_range"]:
            continue

        # Determine signal side
        if combined > 0.2:
            side = "BUY"
        elif combined < -0.2:
            side = "SELL"
        else:
            continue  # NEUTRAL — no signal

        # Check outcome
        actual_close = candles[i + horizon].close
        entry_price = candles[i].close
        if side == "BUY":
            move_pips = (actual_close - entry_price) / PIPS_PER_MOVE
        else:
            move_pips = (entry_price - actual_close) / PIPS_PER_MOVE

        # Net pips after spread
        net_pips = move_pips - SPREAD_PIPS

        # Track equity: fixed 1% risk per trade
        # Use a fixed pip value per trade to avoid synthetic data artifacts
        atr_pips = atr_val_pips(candles, i) or 10.0
        risk_usd = equity * 0.01  # 1% risk per trade
        # Scale by ratio of actual move to expected move (ATR)
        expected_pips = max(1.0, atr_pips)
        move_ratio = max(-3.0, min(3.0, net_pips / expected_pips))  # cap at 3x ATR
        trade_usd = risk_usd * move_ratio
        equity += trade_usd
        if equity > peak_equity:
            peak_equity = equity
        dd = (peak_equity - equity) / peak_equity
        if dd > max_drawdown:
            max_drawdown = dd

        win = net_pips > 0
        trades.append({
            "index": i,
            "side": side,
            "confidence": round(confidence, 4),
            "ml_up": round(ml_up, 4),
            "combined": round(combined, 4),
            "net_pips": round(net_pips, 2),
            "win": win,
            "regime": "range" if regime["is_range"] else "trend",
            "flipped": flip_applied,
        })

        # Per-regime tracking
        r = "range" if regime["is_range"] else "trend"
        if win:
            per_regime_results[r]["wins"] += 1
        else:
            per_regime_results[r]["losses"] += 1
        per_regime_results[r]["pips"] += net_pips
        per_regime[r].append(net_pips)

    # Compute summary stats
    total_trades = len(trades)
    wins = sum(1 for t in trades if t["win"])
    losses = total_trades - wins
    win_rate = wins / total_trades if total_trades else 0.0

    total_pips = sum(t["net_pips"] for t in trades)
    avg_pips = total_pips / total_trades if total_trades else 0.0

    # Sharpe ratio (annualized from 1-min bars)
    if len(trades) > 1:
        returns = [t["net_pips"] for t in trades]
        mean_r = sum(returns) / len(returns)
        std_r = math.sqrt(sum((r - mean_r) ** 2 for r in returns) / (len(returns) - 1))
        sharpe = (mean_r / std_r) * math.sqrt(252 * 240) if std_r > 0 else 0.0  # 240 1-min bars/day
    else:
        sharpe = 0.0

    # Best/worst trades
    best_trade = max(trades, key=lambda t: t["net_pips"]) if trades else None
    worst_trade = min(trades, key=lambda t: t["net_pips"]) if trades else None

    # Sureshot analysis — what if we only traded high-confidence signals?
    for threshold in [0.3, 0.4, 0.5, 0.6]:
        high_conf_trades = [t for t in trades if t["confidence"] >= threshold]
        if len(high_conf_trades) > 0:
            hc_wins = sum(1 for t in high_conf_trades if t["win"])
            hc_wr = hc_wins / len(high_conf_trades)
            hc_pips = sum(t["net_pips"] for t in high_conf_trades)
        else:
            hc_wr = 0.0
            hc_pips = 0.0

    report = {
        "expiry_minutes": expiry_minutes,
        "horizon": horizon,
        "total_candles": len(candles),
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate, 4),
        "total_pips": round(total_pips, 2),
        "avg_pips_per_trade": round(avg_pips, 2),
        "sharpe_ratio": round(sharpe, 2),
        "max_drawdown": round(max_drawdown, 4),
        "final_equity": round(equity, 2),
        "starting_equity": 10000.0,
        "spread_pips": SPREAD_PIPS,
        "best_trade": best_trade,
        "worst_trade": worst_trade,
        "per_regime": {
            k: {
                "trades": len(per_regime[k]),
                "wins": per_regime_results[k]["wins"],
                "losses": per_regime_results[k]["losses"],
                "win_rate": round(per_regime_results[k]["wins"] / len(per_regime[k]), 4) if per_regime[k] else 0,
                "total_pips": round(per_regime_results[k]["pips"], 2),
            }
            for k in per_regime
        },
        "sureshot_analysis": {
            f"threshold_{t}": {
                "trades": len([t2 for t2 in trades if t2["confidence"] >= t]),
                "win_rate": round(sum(1 for t2 in trades if t2["confidence"] >= t and t2["win"]) / max(1, len([t2 for t2 in trades if t2["confidence"] >= t])), 4),
                "total_pips": round(sum(t2["net_pips"] for t2 in trades if t2["confidence"] >= t), 2),
            }
            for t in [0.3, 0.4, 0.5, 0.6]
        },
        "flipped_trades": sum(1 for t in trades if t["flipped"]),
        "flipped_win_rate": round(sum(1 for t in trades if t["flipped"] and t["win"]) / max(1, sum(1 for t in trades if t["flipped"])), 4),
    }

    return report


def atr_val_pips(candles, index):
    """Get ATR in pips at a given index."""
    window = candles[max(0, index - 19): index + 1]
    atr_vals = atr(window, 14)
    if atr_vals and atr_vals[-1]:
        return atr_vals[-1] / PIPS_PER_MOVE
    return 10.0


def optimize_weights(candles, expiry_minutes=30):
    """Grid-search optimal combiner weights.

    Tests multiple weight configurations and picks the best by Sharpe ratio.
    """
    print(f"\nOptimizing weights for {expiry_minutes}-min expiry...")

    weight_space = {
        "ml_ensemble": [0.25, 0.30, 0.35],
        "supertrend": [0.10, 0.15],
        "rsi": [0.08, 0.10],
        "mr_blend": [0.0, 0.09],
    }

    # Keep other weights fixed at default
    fixed_weights = {k: v for k, v in DEFAULT_WEIGHTS.items()
                     if k not in weight_space}

    best_result = None
    best_weights = None
    total_configs = 0

    # Limit grid search to keep it fast
    ml_vals = weight_space["ml_ensemble"]
    st_vals = weight_space["supertrend"]
    mr_vals = weight_space["mr_blend"]

    for ml_w in ml_vals:
        for st_w in st_vals:
            for mr_w in mr_vals:
                total_configs += 1
                w = {**fixed_weights, "ml_ensemble": ml_w, "supertrend": st_w, "mr_blend": mr_w}
                # Normalize
                total = sum(w.values())
                w = {k: v / total for k, v in w.items()}

                result = run_backtest(candles, weights=w, expiry_minutes=expiry_minutes)
                if result is None:
                    continue

                score = result["sharpe_ratio"]
                if best_result is None or score > best_result["sharpe_ratio"]:
                    best_result = result
                    best_weights = w

                if total_configs % 10 == 0:
                    print(f"  Tested {total_configs} configs... best Sharpe: {score:.2f}")

    print(f"\nTested {total_configs} weight configurations.")
    if best_weights:
        print(f"Best Sharpe: {best_result['sharpe_ratio']:.2f}")
        print(f"Best weights: {json.dumps(best_weights, indent=2)}")

    return best_weights, best_result


def main():
    parser = argparse.ArgumentParser(description="Walk-forward backtester + self-learning weight optimizer")
    parser.add_argument("--candles", type=int, default=0, help="Number of candles to use (0 = all)")
    parser.add_argument("--candle-dir", type=str, default=None, help="Directory containing candle batch files")
    parser.add_argument("--weights", action="store_true", help="Run weight optimization only")
    parser.add_argument("--expiry", type=int, default=30, help="Expiry minutes (1, 3, or 30)")
    parser.add_argument("--all-expiries", action="store_true", help="Test all expiry options")
    args = parser.parse_args()

    # Allow overriding CANDLE_DIR from CLI argument
    global CANDLE_DIR, REPORT_JSON, WEIGHTS_JSON, MODEL_PATH, MODEL_1BAR_PATH, MODEL_MR_PATH
    if args.candle_dir:
        CANDLE_DIR = Path(args.candle_dir)
        REPORT_JSON = CANDLE_DIR / "backtest_report.json"
        WEIGHTS_JSON = CANDLE_DIR / "optimal_weights.json"
        MODEL_PATH = CANDLE_DIR / "ensemble_model.json"
        MODEL_1BAR_PATH = CANDLE_DIR / "ensemble_model_1bar.json"
        MODEL_MR_PATH = CANDLE_DIR / "ensemble_model_mr.json"

    print("=" * 60)
    print("WALK-FORWARD BACKTESTER + SELF-LEARNING ENGINE")
    print("=" * 60)

    # Load candles
    n = args.candles if args.candles > 0 else None
    candles = load_candles(n)
    print(f"Loaded {len(candles)} candles")

    if not candles:
        print("ERROR: No candles found. Run train_robust_model.py or generate_micro_trends.py first.")
        sys.exit(1)

    expiries = [1, 3, 30] if args.all_expiries else [args.expiry]
    all_reports = {}

    for expiry in expiries:
        print(f"\n{'=' * 40}")
        print(f"BACKTESTING: {expiry}-MINUTE EXPIRY")
        print(f"{'=' * 40}")

        if args.weights:
            best_weights, best_result = optimize_weights(candles, expiry_minutes=expiry)
            if best_weights:
                all_reports[f"{expiry}min_optimized"] = {
                    "weights": best_weights,
                    "report": best_result,
                }
        else:
            result = run_backtest(candles, expiry_minutes=expiry)
            if result:
                all_reports[f"{expiry}min"] = result

                print(f"\n{'─' * 40}")
                print(f"RESULTS: {expiry}-MINUTE EXPIRY")
                print(f"{'─' * 40}")
                print(f"  Total trades:  {result['total_trades']}")
                print(f"  Wins:          {result['wins']}")
                print(f"  Losses:        {result['losses']}")
                print(f"  Win rate:      {result['win_rate']:.1%}")
                print(f"  Total pips:    {result['total_pips']:.2f}")
                print(f"  Avg pips/trade: {result['avg_pips_per_trade']:.2f}")
                print(f"  Sharpe ratio:  {result['sharpe_ratio']:.2f}")
                print(f"  Max drawdown:  {result['max_drawdown']:.2%}")
                print(f"  Final equity:  ${result['final_equity']:.2f}")

                print(f"\n  Per-regime:")
                for regime, data in result["per_regime"].items():
                    print(f"    {regime:8s}: {data['trades']} trades, "
                          f"{data['win_rate']:.1%} win rate, "
                          f"{data['total_pips']:.2f} pips")

                print(f"\n  Sureshot analysis (confidence threshold):")
                for threshold, data in result["sureshot_analysis"].items():
                    print(f"    {threshold:12s}: {data['trades']} trades, "
                          f"{data['win_rate']:.1%} win rate, "
                          f"{data['total_pips']:.2f} pips")

                if result["flipped_trades"] > 0:
                    print(f"\n  Regime-aware flips: {result['flipped_trades']} trades, "
                          f"{result['flipped_win_rate']:.1%} win rate")

    # Save reports
    REPORT_JSON.write_text(json.dumps(all_reports, indent=2))
    print(f"\nReport saved to {REPORT_JSON}")

    # Summary
    print(f"\n{'=' * 60}")
    print("SELF-LEARNING SUMMARY")
    print("=" * 60)
    for key, data in all_reports.items():
        if isinstance(data, dict) and "report" in data:
            report = data["report"]
            weights = data["weights"]
            print(f"\n{key}:")
            print(f"  Win rate:    {report['win_rate']:.1%}")
            print(f"  Sharpe:      {report['sharpe_ratio']:.2f}")
            print(f"  Total pips:  {report['total_pips']:.2f}")
            print(f"  Optimal weights: {json.dumps(weights)}")
        else:
            print(f"\n{key}:")
            print(f"  Win rate:    {data['win_rate']:.1%}")
            print(f"  Sharpe:      {data['sharpe_ratio']:.2f}")
            print(f"  Total pips:  {data['total_pips']:.2f}")

    print(f"\n{'=' * 60}")
    print("BACKTEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
