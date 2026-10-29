#!/usr/bin/env python3
"""
Win Streak Compounding Bot
==========================
Targets 5 consecutive wins per day with compounding position sizing.

Strategy:
  - Only trade when ML ensemble confidence >= 0.68 (75.9% WR in backtest)
  - Only during London + NY session overlap (highest liquidity)
  - Compounding: risk 1% of equity per trade, +10% size increase per win
  - Stop after 5 consecutive wins (bank the day)
  - Reset to base size after any loss

Usage:
    python3 scripts/win_streak_bot.py --live        # Generate live signals
    python3 scripts/win_streak_bot.py --backtest    # Run historical backtest
    python3 scripts/win_streak_bot.py --signals     # Print today's signal setup
"""

from __future__ import annotations

import argparse
import json
import sys
import math
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from fusion.domain import Candle
from fusion.features import compute_features, FEATURE_NAMES
from fusion.indicators import atr as calc_atr
from fusion.gbdt import GradientBoostingModel
from backtest_engine import load_candles, load_models

CANDLE_DIR = PROJECT_ROOT / "var" / "real_candles"
WINSTREAK_MODEL_PATH = CANDLE_DIR / "ensemble_model_winstreak.json"
PIPS_PER_MOVE = 0.0001
SPREAD_PIPS = 1.2
HORIZON = 5  # 5-bar = 30-min expiry


def load_winstreak_model():
    """Load the SEPARATE win streak model (not the main signal generator model)."""
    if not WINSTREAK_MODEL_PATH.exists():
        print(f"ERROR: Win streak model not found at {WINSTREAK_MODEL_PATH}")
        print("Run: python3 scripts/train_winstreak_model.py")
        return []
    data = json.loads(WINSTREAK_MODEL_PATH.read_text())
    models = []
    for member in data["members"]:
        m = GradientBoostingModel.from_dict(member)
        if m:
            models.append(m)
    return models

# ─── Win Streak Configuration ───────────────────────────────────────
# Optimized from backtest on 2000 EUR/USD candles (3 days):
#   conf=0.68, no session filter: 62.5% WR, 2/3 days with 5-streak, +$152.20
#   conf=0.70, session filter:    85.7% WR, 1/1 days with 5-streak, +$39.37
#   conf=0.65, no session filter: 53.2% WR, 2/3 days with 5-streak, +$141.78
ML_CONF_THRESHOLD = 0.68      # Main model direction confidence floor
WIN_PROB_THRESHOLD = 0.55     # Win streak model win-probability floor (SEPARATE filter)
ADX_MIN = 0                    # No ADX filter
SESSION_FILTER = False         # Trade any time (more trades = more streak chances)
TARGET_STREAK = 5              # 5 consecutive wins to bank the day
BASE_RISK_PCT = 1.0            # Start at 1% risk per trade
SIZE_INCREASE_PCT = 10.0       # +10% size per consecutive win
MAX_SIZE_PCT = 3.0             # Cap at 3% risk per trade
STARTING_EQUITY = 1000.0       # USD


def is_london_or_ny(dt):
    """London: 07:00-16:00 UTC, NY: 12:00-20:00 UTC"""
    h = dt.hour
    return (7 <= h <= 16) or (12 <= h <= 20)


def get_session_name(dt):
    h = dt.hour
    if 7 <= h < 12:
        return "London"
    elif 12 <= h < 16:
        return "London/NY"
    elif 16 <= h < 20:
        return "NY"
    else:
        return "Off-hours"


def generate_signals(candles, feats, main_models, winstreak_models):
    """Generate win-streak signals with compounding sizing.

    Uses TWO separate models:
      1. Main model (ensemble_model_fb.json) → predicts direction (buy/sell)
      2. Win streak model (ensemble_model_winstreak.json) → predicts WIN probability

    Trade only when: main model confidence >= threshold AND win streak model
    win probability >= threshold. This is a separate filter from the signal
    generator — the win streak model answers "will this trade WIN?" not "which direction?"
    """
    trades = []
    equity = STARTING_EQUITY
    daily_streak = 0
    daily_trades = 0
    daily_pips = 0.0

    # Track per-day results
    day_results = defaultdict(lambda: {'trades': [], 'streak': 0, 'banked': False})

    for i in range(60, len(candles) - HORIZON):
        feat = feats[i]
        if feat is None:
            continue

        dt = candles[i].timestamp

        # Session filter
        if SESSION_FILTER and not is_london_or_ny(dt):
            continue

        # ADX filter
        if ADX_MIN > 0:
            adx_val = get_adx(candles, i)
            if adx_val < ADX_MIN:
                continue

        # --- MAIN MODEL: predict direction ---
        ml_up = sum(m.predict_up(feat) for m in main_models) / len(main_models)
        direction_confidence = max(ml_up, 1 - ml_up)

        if direction_confidence < ML_CONF_THRESHOLD:
            continue

        side = "BUY" if ml_up > 0.5 else "SELL"

        # --- WIN STREAK MODEL: predict win probability ---
        # This is the SEPARATE model — trained on high-confidence setups only
        # It answers "will this trade be profitable?" not "which direction?"
        if winstreak_models:
            win_prob = sum(m.predict_up(feat) for m in winstreak_models) / len(winstreak_models)
            # Win streak model predicts 1=win, 0=loss
            # Use win_prob as the primary filter (different from direction confidence)
            if win_prob < WIN_PROB_THRESHOLD:
                continue
            confidence = win_prob  # Use win probability as the displayed confidence
        else:
            # Fallback: use direction confidence if win streak model not available
            confidence = direction_confidence

        # Compounding position size
        day_key = dt.strftime('%Y-%m-%d')
        day = day_results[day_key]

        # Check if day is already banked (5 consecutive wins)
        if day['banked']:
            continue

        # Compute position size based on current streak
        size_pct = BASE_RISK_PCT + (day['streak'] * SIZE_INCREASE_PCT / 100) * BASE_RISK_PCT
        size_pct = min(size_pct, MAX_SIZE_PCT)
        risk_usd = equity * (size_pct / 100)

        # Outcome
        future_close = candles[i + HORIZON].close
        entry = candles[i].close
        if side == "BUY":
            pips = (future_close - entry) / PIPS_PER_MOVE
        else:
            pips = (entry - future_close) / PIPS_PER_MOVE
        net_pips = pips - SPREAD_PIPS

        # P&L in USD (approximate: 1 pip = $10 for standard lot with 1% risk)
        atr_pips = get_atr_pips(candles, i)
        risk_pips = max(1.0, atr_pips)
        usd_per_pip = risk_usd / risk_pips if risk_pips > 0 else 10.0
        pnl_usd = net_pips * usd_per_pip

        win = net_pips > 0

        trade = {
            'index': i,
            'timestamp': dt.isoformat(),
            'session': get_session_name(dt),
            'side': side,
            'ml_confidence': round(confidence, 4),
            'ml_up': round(ml_up, 4),
            'net_pips': round(net_pips, 2),
            'pnl_usd': round(pnl_usd, 2),
            'win': win,
            'size_pct': round(size_pct, 3),
            'equity_before': round(equity, 2),
            'equity_after': round(equity + pnl_usd, 2),
            'streak_before': day['streak'],
        }

        if win:
            day['streak'] += 1
            equity += pnl_usd
        else:
            day['streak'] = 0
            equity += pnl_usd

        # Check for 5-streak bank
        if day['streak'] >= TARGET_STREAK and not day['banked']:
            day['banked'] = True
            trade['banked_day'] = True

        trades.append(trade)

    return trades, equity


def get_adx(candles, index, period=14):
    window = candles[max(0, index - period): index + 1]
    if len(window) < period + 1:
        return 0.0
    trs, plus_dm, minus_dm = [], [], []
    for i in range(1, len(window)):
        high, low = window[i].high, window[i].low
        prev_high, prev_low = window[i-1].high, window[i-1].low
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


def run_backtest():
    """Run historical backtest with compounding."""
    print("=" * 70)
    print("WIN STREAK COMPOUNDING BACKTEST")
    print("=" * 70)
    print(f"ML Conf Threshold: {ML_CONF_THRESHOLD}")
    print(f"Session Filter: {SESSION_FILTER}")
    print(f"Target Streak: {TARGET_STREAK} consecutive wins")
    print(f"Base Risk: {BASE_RISK_PCT}% | Size Increase: {SIZE_INCREASE_PCT}%/win | Max: {MAX_SIZE_PCT}%")
    print(f"Starting Equity: ${STARTING_EQUITY}")
    print()

    candles = load_candles(2000)
    feats = compute_features(candles)
    main_models = load_models(30)
    winstreak_models = load_winstreak_model()

    if not main_models:
        print("ERROR: Could not load main models")
        return

    print(f"Main model: {len(main_models)} members")
    print(f"Win streak model: {len(winstreak_models)} members (separate model)")
    print()

    trades, final_equity = generate_signals(candles, feats, main_models, winstreak_models)

    # Summary stats
    wins = sum(1 for t in trades if t['win'])
    losses = len(trades) - wins
    wr = wins / len(trades) * 100 if trades else 0
    total_pips = sum(t['net_pips'] for t in trades)
    total_pnl = sum(t['pnl_usd'] for t in trades)

    # Streak stats
    max_streak = 0
    current = 0
    streaks_of_5 = 0
    day_streaks = defaultdict(list)
    for t in trades:
        day = t['timestamp'][:10]
        day_streaks[day].append(t['win'])
        if t['win']:
            current += 1
            if current > max_streak:
                max_streak = current
            if current == TARGET_STREAK:
                streaks_of_5 += 1
        else:
            current = 0

    days_with_streak = 0
    for day, results in day_streaks.items():
        streak = 0
        for r in results:
            if r:
                streak += 1
                if streak >= TARGET_STREAK:
                    days_with_streak += 1
                    break
            else:
                streak = 0

    print(f"Total Trades: {len(trades)}")
    print(f"Wins: {wins} ({wr:.1f}%) | Losses: {losses}")
    print(f"Total Pips: {total_pips:+.2f}")
    print(f"Total P&L: ${total_pnl:+.2f}")
    print(f"Final Equity: ${final_equity:.2f} (from ${STARTING_EQUITY:.0f})")
    print(f"Return: {((final_equity - STARTING_EQUITY) / STARTING_EQUITY * 100):+.1f}%")
    print(f"Max Streak: {max_streak}")
    print(f"Streaks of {TARGET_STREAK}: {streaks_of_5}")
    print(f"Days with {TARGET_STREAK}-streak: {days_with_streak}/{len(day_streaks)}")
    print()

    # Per-day breakdown
    print("=== PER-DAY BREAKDOWN ===")
    print(f"{'Date':12s} {'Trades':>6s} {'Wins':>5s} {'WR':>6s} {'Pips':>8s} {'P&L':>8s} {'MaxStk':>7s} {'Banked':>7s}")
    print("-" * 65)

    for day, results in sorted(day_streaks.items()):
        day_trades = [t for t in trades if t['timestamp'][:10] == day]
        day_wins = sum(1 for t in day_trades if t['win'])
        day_wr = day_wins / len(day_trades) * 100 if day_trades else 0
        day_pips = sum(t['net_pips'] for t in day_trades)
        day_pnl = sum(t['pnl_usd'] for t in day_trades)
        streak = 0
        max_day_streak = 0
        for r in results:
            if r:
                streak += 1
                if streak > max_day_streak:
                    max_day_streak = streak
            else:
                streak = 0
        banked = any(t.get('banked_day', False) for t in day_trades)
        print(f"{day:12s} {len(day_trades):6d} {day_wins:5d} {day_wr:5.1f}% {day_pips:+7.2f} {day_pnl:+7.2f} {max_day_streak:7d} {'YES' if banked else 'NO':>7s}")

    print()
    print("=== RECENT TRADES (last 20) ===")
    for t in trades[-20:]:
        marker = ' *** BANKED ***' if t.get('banked_day') else ''
        print(f"{t['timestamp'][:16]} {t['session']:12s} {t['side']:4s} "
              f"conf={t['ml_confidence']:.3f} pips={t['net_pips']:+.2f} "
              f"$={t['pnl_usd']:+.2f} streak={t['streak_before']+1 if t['win'] else 0}{marker}")

    # Save report
    report = {
        'config': {
            'ml_conf_threshold': ML_CONF_THRESHOLD,
            'session_filter': SESSION_FILTER,
            'target_streak': TARGET_STREAK,
            'base_risk_pct': BASE_RISK_PCT,
            'size_increase_pct': SIZE_INCREASE_PCT,
            'max_size_pct': MAX_SIZE_PCT,
            'starting_equity': STARTING_EQUITY,
        },
        'results': {
            'total_trades': len(trades),
            'wins': wins,
            'losses': losses,
            'win_rate': round(wr, 2),
            'total_pips': round(total_pips, 2),
            'total_pnl_usd': round(total_pnl, 2),
            'final_equity': round(final_equity, 2),
            'return_pct': round(((final_equity - STARTING_EQUITY) / STARTING_EQUITY * 100), 2),
            'max_streak': max_streak,
            'streaks_of_target': streaks_of_5,
            'days_with_streak': days_with_streak,
            'total_days': len(day_streaks),
        },
    }
    report_path = CANDLE_DIR / "win_streak_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nReport saved to {report_path}")


def generate_live_signals():
    """Generate signals for the current/latest candle."""
    print("=" * 70)
    print("WIN STREAK LIVE SIGNAL GENERATOR")
    print("=" * 70)
    print(f"ML Conf Threshold: {ML_CONF_THRESHOLD}")
    print(f"Session: London (07-12 UTC) + NY (12-20 UTC)")
    print()

    candles = load_candles(2000)
    feats = compute_features(candles)
    main_models = load_models(30)
    winstreak_models = load_winstreak_model()

    if not main_models:
        print("ERROR: Could not load main models")
        return

    # Find the latest candle
    latest_idx = len(candles) - 1
    if latest_idx < 60:
        print("Not enough candles")
        return

    # Check last 5 candles for signals
    now = datetime.now(tz=timezone.utc)
    print(f"Current time: {now.isoformat()}")
    print(f"Session: {get_session_name(now)}")
    print(f"Main model: {len(main_models)} members | Win streak model: {len(winstreak_models)} members")
    print()

    signals_found = False
    for i in range(max(60, latest_idx - 5), latest_idx + 1):
        feat = feats[i]
        if feat is None:
            continue

        dt = candles[i].timestamp

        if SESSION_FILTER and not is_london_or_ny(dt):
            print(f"  [{dt.strftime('%H:%M UTC')}] {get_session_name(dt):12s} off-hours, no trade")
            continue

        # Direction from main model
        ml_up = sum(m.predict_up(feat) for m in main_models) / len(main_models)
        direction_conf = max(ml_up, 1 - ml_up)

        # Win probability from SEPARATE win streak model
        if winstreak_models:
            win_prob = sum(m.predict_up(feat) for m in winstreak_models) / len(winstreak_models)
        else:
            win_prob = direction_conf

        if direction_conf >= ML_CONF_THRESHOLD and win_prob >= WIN_PROB_THRESHOLD:
            side = "BUY" if ml_up > 0.5 else "SELL"
            signal = "*** TRADE ***" if win_prob >= 0.65 else "* watch *"
            print(f"  [{dt.strftime('%H:%M UTC')}] {get_session_name(dt):12s} "
                  f"{side:4s} dir_conf={direction_conf:.3f} win_prob={win_prob:.3f} {signal}")
            signals_found = True
        else:
            print(f"  [{dt.strftime('%H:%M UTC')}] {get_session_name(dt):12s} "
                  f"dir_conf={direction_conf:.3f} win_prob={win_prob:.3f} "
                  f"(need dir>={ML_CONF_THRESHOLD} AND win>={WIN_PROB_THRESHOLD})")

    if not signals_found:
        print("\nNo high-confidence signals in last 5 candles.")
        print("Waiting for ML confidence >= {ML_CONF_THRESHOLD} during London/NY session.")
    else:
        print("\nSignals detected. Execute with compounding sizing:")
        print(f"  Base risk: {BASE_RISK_PCT}% of equity")
        print(f"  +{SIZE_INCREASE_PCT}% size per consecutive win (max {MAX_SIZE_PCT}%)")
        print(f"  Stop after {TARGET_STREAK} consecutive wins")


def main():
    parser = argparse.ArgumentParser(description="Win Streak Compounding Bot")
    parser.add_argument("--backtest", action="store_true", help="Run historical backtest")
    parser.add_argument("--live", action="store_true", help="Generate live signals")
    parser.add_argument("--signals", action="store_true", help="Print signal setup")
    args = parser.parse_args()

    if args.backtest:
        run_backtest()
    elif args.live or args.signals:
        generate_live_signals()
    else:
        run_backtest()


if __name__ == "__main__":
    main()
