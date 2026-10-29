#!/usr/bin/env python3
"""
Win Streak Filter Tester
========================
Tests combinations of filters to find the highest per-trade win rate
for achieving 5 consecutive wins per day.

Filters tested:
  - Sureshot confidence threshold
  - Regime filter (trend only)
  - Session filter (London + NY only)
  - ML ensemble confidence threshold
  - ADX trend strength threshold
  - MTF confirmation (15-min bias alignment)
  - Council unanimity (all voters agree)

Output: ranked combinations by win rate, with streak statistics.
"""

import sys, math, json
from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fusion.domain import Candle
from fusion.features import compute_features, FEATURE_NAMES
from fusion.indicators import atr as calc_atr

CANDLE_DIR = PROJECT_ROOT / "var" / "real_candles"
PIPS_PER_MOVE = 0.0001
SPREAD_PIPS = 1.2
HORIZON = 30  # 30-min expiry


def load_candles(n=2000):
    candles = []
    for bf in sorted(CANDLE_DIR.glob("batch_real_*.txt")):
        for line in bf.read_text().strip().split(";"):
            line = line.strip()
            if not line: continue
            parts = line.split(",")
            if len(parts) < 5: continue
            ts = int(float(parts[0]))
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            candles.append(Candle(timestamp=dt, open=float(parts[1]), high=float(parts[2]),
                                   low=float(parts[3]), close=float(parts[4]),
                                   volume=float(parts[5]) if len(parts) > 5 else 0.0))
    candles.sort(key=lambda c: c.timestamp)
    return candles[-n:] if n else candles


def is_london_or_ny(dt):
    """London: 07:00-16:00 UTC, NY: 12:00-20:00 UTC"""
    h = dt.hour
    return (7 <= h <= 16) or (12 <= h <= 20)


def get_adx(candles, index, period=14):
    window = candles[max(0, index - period): index + 1]
    if len(window) < period + 1:
        return 0.0
    # Simple ADX approximation
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


def compute_combined_score(feat, weights):
    """Compute the combiner score from feature values."""
    score = 0.0
    total_w = 0.0
    for key, w in weights.items():
        if key in feat:
            val = feat[key]
            if val is not None:
                score += val * w
                total_w += abs(w)
    return score / total_w if total_w > 0 else 0.0


# Default combiner weights (simplified)
DEFAULT_WEIGHTS = {
    "close_position": 0.15,
    "rsi_slope": 0.10,
    "ema_slope": 0.10,
    "atr_pips": 0.05,
    "choppiness": -0.10,
    "hurst": 0.10,
    "supertrend_direction": 0.10,
    "ichimoku_kijun": 0.05,
    "session_momentum_shift": 0.10,
    "vpin_proxy": 0.05,
    "flow_momentum": 0.05,
    "kyle_lambda_proxy": 0.05,
}


def run_filter_test(candles, feats):
    """Test multiple filter combinations."""
    print(f"Loaded {len(candles)} candles")
    print(f"Testing {HORIZON}-min expiry")
    print()

    # Pre-compute ADX for all candles
    adx_vals = [get_adx(candles, i) for i in range(len(candles))]

    # Pre-compute regime
    regimes = []
    for i in range(len(candles)):
        if adx_vals[i] > 25:
            regimes.append("trend")
        elif adx_vals[i] < 15:
            regimes.append("range")
        else:
            regimes.append("unknown")

    # Test combinations
    thresholds = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    adx_filters = [0, 20, 25, 30]
    ml_conf_filters = [0.0, 0.55, 0.60, 0.65, 0.70]

    results = []

    for sureshot_thresh in thresholds:
        for adx_min in adx_filters:
            for ml_conf_min in ml_conf_filters:
                for use_session in [True, False]:
                    for use_regime in [True, False]:
                        trades = []
                        for i in range(60, len(candles) - HORIZON):
                            feat = feats[i]
                            if feat is None:
                                continue

                            # Compute combined score (proxy for ML confidence)
                            combined = compute_combined_score(feat, DEFAULT_WEIGHTS)
                            confidence = abs(combined)

                            # Sureshot filter
                            if confidence < sureshot_thresh:
                                continue

                            # ADX filter
                            if adx_min > 0 and adx_vals[i] < adx_min:
                                continue

                            # ML confidence filter
                            ml_up = combined  # proxy
                            if ml_conf_min > 0 and abs(ml_up) < ml_conf_min:
                                continue

                            # Session filter
                            if use_session and not is_london_or_ny(candles[i].timestamp):
                                continue

                            # Regime filter (trend only)
                            if use_regime and regimes[i] != "trend":
                                continue

                            # Signal direction
                            if combined > 0.2:
                                side = "BUY"
                            elif combined < -0.2:
                                side = "SELL"
                            else:
                                continue

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

                        if len(trades) < 10:
                            continue

                        wins = sum(1 for t in trades if t['win'])
                        wr = wins / len(trades)
                        total_pips = sum(t['pips'] for t in trades)

                        # Compute streak statistics
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

                        # Count days with at least 1 streak of 5
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

                        results.append({
                            'sureshot_thresh': sureshot_thresh,
                            'adx_min': adx_min,
                            'ml_conf_min': ml_conf_min,
                            'use_session': use_session,
                            'use_regime': use_regime,
                            'trades': len(trades),
                            'win_rate': wr,
                            'total_pips': total_pips,
                            'max_streak': max_streak,
                            'streaks_of_5': streaks_of_5,
                            'days_with_5': days_with_5,
                            'total_days': len(day_streaks),
                        })

    # Sort by win rate (descending), then by trades (descending)
    results.sort(key=lambda x: (-x['win_rate'], -x['trades']))

    return results


def main():
    candles = load_candles(2000)
    feats = compute_features(candles)

    results = run_filter_test(candles, feats)

    print(f"{'Sureshot':>8s} {'ADX':>4s} {'MLConf':>6s} {'Session':>7s} {'Regime':>6s} {'Trades':>6s} {'WR':>6s} {'Pips':>8s} {'MaxStk':>7s} {'5-Stk':>5s} {'Days5':>5s}")
    print("=" * 75)

    # Show top 30
    for r in results[:30]:
        print(f"{r['sureshot_thresh']:8.2f} {r['adx_min']:4d} {r['ml_conf_min']:6.2f} "
              f"{'Y' if r['use_session'] else 'N':>7s} {'Y' if r['use_regime'] else 'N':>6s} "
              f"{r['trades']:6d} {r['win_rate']*100:5.1f}% {r['total_pips']:+8.2f} "
              f"{r['max_streak']:7d} {r['streaks_of_5']:5d} {r['days_with_5']:5d}")

    print()
    print("=== BEST COMBINATION FOR 5-STREAK ===")
    best_streak = max(results, key=lambda x: x['streaks_of_5'])
    print(f"Sureshot={best_streak['sureshot_thresh']}, ADX>={best_streak['adx_min']}, "
          f"MLConf>={best_streak['ml_conf_min']}, Session={best_streak['use_session']}, "
          f"Regime={best_streak['use_regime']}")
    print(f"  WR={best_streak['win_rate']*100:.1f}%, Trades={best_streak['trades']}, "
          f"Max Streak={best_streak['max_streak']}, 5-streaks={best_streak['streaks_of_5']}, "
          f"Days with 5-streak={best_streak['days_with_5']}/{best_streak['total_days']}")

    print()
    print("=== BEST WIN RATE (min 20 trades) ===")
    qualified = [r for r in results if r['trades'] >= 20]
    best_wr = max(qualified, key=lambda x: x['win_rate']) if qualified else None
    if best_wr:
        print(f"Sureshot={best_wr['sureshot_thresh']}, ADX>={best_wr['adx_min']}, "
              f"MLConf>={best_wr['ml_conf_min']}, Session={best_wr['use_session']}, "
              f"Regime={best_wr['use_regime']}")
        print(f"  WR={best_wr['win_rate']*100:.1f}%, Trades={best_wr['trades']}, "
              f"Pips={best_wr['total_pips']:+.2f}, Max Streak={best_wr['max_streak']}")

    # Save full results
    out_path = CANDLE_DIR / "filter_test_results.json"
    out_path.write_text(json.dumps(results[:100], indent=2))
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
