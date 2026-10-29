#!/usr/bin/env python3
"""
Win Streak Finder — Uses actual trained ensemble model
=======================================================
Tests the real GBDT ensemble model with various confidence thresholds
and filter combinations to find setups that achieve 65%+ win rate
for 5-consecutive-win sessions.

Uses the actual trained ensemble_model_fb.json (main, 30-min horizon).
"""

import sys, math, json
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from fusion.domain import Candle
from fusion.features import compute_features, FEATURE_NAMES
from backtest_engine import load_candles, load_models

CANDLE_DIR = PROJECT_ROOT / "var" / "real_candles"
PIPS_PER_MOVE = 0.0001
SPREAD_PIPS = 1.2
HORIZON = 5  # 5-bar horizon = 30-min expiry


def is_london_or_ny(dt):
    h = dt.hour
    return (7 <= h <= 16) or (12 <= h <= 20)


def get_adx(candles, index, period=14):
    window = candles[max(0, index - period): index + 1]
    if len(window) < period + 1:
        return 0.0
    trs = []
    plus_dm = []
    minus_dm = []
    for i in range(1, len(window)):
        high = window[i].high
        low = window[i].low
        prev_high = window[i-1].high
        prev_low = window[i-1].low
        tr = max(high - low, abs(high - prev_low), abs(low - prev_high))
        plus_d = high - prev_high
        minus_d = prev_low - low
        trs.append(tr)
        plus_dm.append(plus_d if plus_d > minus_d and plus_d > 0 else 0)
        minus_dm.append(minus_d if minus_d > plus_d and minus_d > 0 else 0)
    if not trs:
        return 0.0
    atr = sum(trs) / len(trs)
    plus_dm_avg = sum(plus_dm) / len(plus_dm)
    minus_dm_avg = sum(minus_dm) / len(minus_dm)
    if atr < 1e-10:
        return 0.0
    plus_di = 100 * plus_dm_avg / atr
    minus_di = 100 * minus_dm_avg / atr
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di) if (plus_di + minus_di) > 0 else 0
    return dx


def get_atr_pips(candles, index):
    window = candles[max(0, index - 19): index + 1]
    trs = []
    for i in range(1, len(window)):
        tr = max(window[i].high - window[i].low,
                 abs(window[i].high - window[i-1].low),
                 abs(window[i].low - window[i-1].high))
        trs.append(tr)
    if not trs:
        return 0.0
    return sum(trs) / len(trs) / PIPS_PER_MOVE


def run_test(candles, feats):
    models = load_models(30)
    if not models:
        print("ERROR: Could not load models")
        return []
    print(f"Loaded {len(candles)} candles, {len(models)} model members")
    print(f"Testing {HORIZON}-bar horizon (30-min expiry)")
    print()

    # Pre-compute ADX
    adx_vals = [get_adx(candles, i) for i in range(len(candles))]

    # Pre-compute ML predictions for all candles
    ml_probs = []
    for i in range(len(candles)):
        feat = feats[i]
        if feat is None:
            ml_probs.append(None)
            continue
        prob = sum(m.predict_up(feat) for m in models) / len(models)
        ml_probs.append(prob)

    # Test: ML confidence threshold × session filter × ADX filter
    ml_thresholds = [0.55, 0.60, 0.62, 0.65, 0.68, 0.70, 0.72, 0.75, 0.78, 0.80]
    adx_filters = [0, 20, 25, 30]
    session_filters = [True, False]

    results = []

    for ml_min in ml_thresholds:
        for adx_min in adx_filters:
            for use_session in session_filters:
                trades = []
                for i in range(60, len(candles) - HORIZON):
                    feat = feats[i]
                    if feat is None:
                        continue
                    ml_up = ml_probs[i]
                    if ml_up is None:
                        continue

                    # ML confidence filter
                    confidence = max(ml_up, 1 - ml_up)
                    if confidence < ml_min:
                        continue

                    # ADX filter
                    if adx_min > 0 and adx_vals[i] < adx_min:
                        continue

                    # Session filter
                    if use_session and not is_london_or_ny(candles[i].timestamp):
                        continue

                    # Signal direction
                    if ml_up > 0.5:
                        side = "BUY"
                    else:
                        side = "SELL"

                    # Outcome
                    future_close = candles[i + HORIZON].close
                    entry = candles[i].close
                    if side == "BUY":
                        pips = (future_close - entry) / PIPS_PER_MOVE
                    else:
                        pips = (entry - future_close) / PIPS_PER_MOVE
                    net_pips = pips - SPREAD_PIPS

                    trades.append({
                        'win': net_pips > 0,
                        'pips': net_pips,
                        'confidence': confidence,
                        'timestamp': candles[i].timestamp,
                    })

                if len(trades) < 5:
                    continue

                wins = sum(1 for t in trades if t['win'])
                wr = wins / len(trades)
                total_pips = sum(t['pips'] for t in trades)

                # Streak stats
                max_streak = 0
                current_streak = 0
                streaks_of_5 = 0
                day_streaks = defaultdict(list)

                for t in trades:
                    day = t['timestamp'].strftime('%Y-%m-%d')
                    if t['win']:
                        current_streak += 1
                        if current_streak > max_streak:
                            max_streak = current_streak
                        if current_streak == 5:
                            streaks_of_5 += 1
                    else:
                        current_streak = 0
                    day_streaks[day].append(t['win'])

                days_with_5 = 0
                for day, results_list in day_streaks.items():
                    streak = 0
                    for r in results_list:
                        if r:
                            streak += 1
                            if streak >= 5:
                                days_with_5 += 1
                                break
                        else:
                            streak = 0

                avg_trades_per_day = len(trades) / max(1, len(day_streaks))

                results.append({
                    'ml_min': ml_min,
                    'adx_min': adx_min,
                    'use_session': use_session,
                    'trades': len(trades),
                    'win_rate': wr,
                    'total_pips': total_pips,
                    'max_streak': max_streak,
                    'streaks_of_5': streaks_of_5,
                    'days_with_5': days_with_5,
                    'total_days': len(day_streaks),
                    'avg_trades_per_day': round(avg_trades_per_day, 1),
                })

    results.sort(key=lambda x: (-x['win_rate'], -x['trades']))
    return results


def main():
    candles = load_candles(2000)
    feats = compute_features(candles)

    results = run_test(candles, feats)

    print(f"{'MLConf':>6s} {'ADX':>4s} {'Session':>7s} {'Trades':>6s} {'WR':>6s} {'Pips':>8s} {'MaxStk':>7s} {'5-Stk':>5s} {'Days5':>5s} {'T/D':>5s}")
    print("=" * 75)

    for r in results[:30]:
        print(f"{r['ml_min']:6.2f} {r['adx_min']:4d} {'Y' if r['use_session'] else 'N':>7s} "
              f"{r['trades']:6d} {r['win_rate']*100:5.1f}% {r['total_pips']:+8.2f} "
              f"{r['max_streak']:7d} {r['streaks_of_5']:5d} {r['days_with_5']:5d} {r['avg_trades_per_day']:5.1f}")

    print()
    print("=== BEST WIN RATE (min 10 trades) ===")
    qualified = [r for r in results if r['trades'] >= 10]
    if qualified:
        best = qualified[0]
        print(f"MLConf>={best['ml_min']}, ADX>={best['adx_min']}, Session={best['use_session']}")
        print(f"  WR={best['win_rate']*100:.1f}%, Trades={best['trades']}, "
              f"Pips={best['total_pips']:+.2f}, Max Streak={best['max_streak']}, "
              f"5-streaks={best['streaks_of_5']}, Days with 5={best['days_with_5']}/{best['total_days']}")

    print()
    print("=== BEST FOR 5-STREAK (min 5 trades, most 5-streaks) ===")
    qualified2 = [r for r in results if r['trades'] >= 5]
    if qualified2:
        best = max(qualified2, key=lambda x: x['streaks_of_5'])
        print(f"MLConf>={best['ml_min']}, ADX>={best['adx_min']}, Session={best['use_session']}")
        print(f"  WR={best['win_rate']*100:.1f}%, Trades={best['trades']}, "
              f"Pips={best['total_pips']:+.2f}, Max Streak={best['max_streak']}, "
              f"5-streaks={best['streaks_of_5']}, Days with 5={best['days_with_5']}/{best['total_days']}")

    # Save results
    out_path = CANDLE_DIR / "win_streak_results.json"
    out_path.write_text(json.dumps(results[:100], indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
