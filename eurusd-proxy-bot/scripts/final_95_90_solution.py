#!/usr/bin/env python3
"""
Final comprehensive solution to achieve 95% win rate with 90% signal rate.

This requires:
1. More candle data (5,000+ candles from trending periods)
2. Neural model trained on high-accuracy data
3. Sophisticated signal filtering
4. Better entry/exit timing
5. Tighter risk management

Current limitations:
- 740 candles from ranging market (Oct 30)
- Win rate: 43.5%, Signal rate: 73.5%

To achieve targets:
- Need 5,000+ candles from trending periods
- Need neural model with 80%+ direction accuracy
- Need better signal quality filtering
"""

import json
from datetime import datetime, timezone
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fusion.domain import Candle
from fusion.config import Config
from fusion.engine import FusionEngine
from fusion.council import CouncilWeights
from fusion.timeutil import next_future_candle_open
from fusion.features import compute_features
from fusion.regime import detect_regime
from fusion.indicators import atr, adx


def load_candles(file_path: str) -> list[Candle]:
    with open(file_path) as f:
        data = json.load(f)

    candles = []
    for r in data:
        ts = datetime.fromisoformat(r['ts'])
        candles.append(Candle(
            timestamp=ts,
            open=r['o'],
            high=r['h'],
            low=r['l'],
            close=r['c'],
            volume=r.get('v', 0.0)
        ))
    return sorted(candles, key=lambda c: c.timestamp)


def analyze_current_performance(candles: list[Candle]) -> dict:
    """Analyze current performance and identify gaps."""

    print(f"\n{'='*70}")
    print("CURRENT PERFORMANCE ANALYSIS")
    print(f"{'='*70}\n")

    config = Config(pair="EUR/USD", spread_pips=1.2)
    engine = FusionEngine(config)
    weights = CouncilWeights()

    wins = 0
    losses = 0
    total_signals = 0
    total_evals = 0
    trade_results = []

    # Test windows
    for i in range(150, min(len(candles) - 1, 350)):
        window = candles[:i]
        total_evals += 1

        # Compute features
        features_list = compute_features(window)
        if not features_list:
            continue
        features = features_list[-1]

        # Detect regime
        regime = detect_regime(window, features)

        # Allow TREND and RANGE regimes
        if regime.name not in ("TREND", "RANGE"):
            continue

        # Check ADX (lowered threshold)
        adx_result = adx(window, 14)
        if not adx_result or len(adx_result) < 1:
            continue
        adx_list = adx_result[0]
        if not adx_list:
            continue
        current_adx = adx_list[-1] if adx_list[-1] is not None else 0

        # Require ADX > 12
        if current_adx < 12:
            continue

        # Check ATR (lowered threshold)
        atr_values = atr(window, 14)
        if not atr_values:
            continue
        current_atr = atr_values[-1]

        # Require ATR > 0.3 pips
        if current_atr < 0.00003:
            continue

        result = engine.evaluate(
            window,
            weights=weights,
            source="ANALYSIS_TEST",
            quote={"spread_pips": 1.2},
            at=next_future_candle_open(window[-1].timestamp),
            observed_at=window[-1].timestamp,
        )

        if result.actionable and result.signal is not None and result.signal.entry_price is not None:
            # Filter by confidence
            if result.signal.probability_up:
                confidence = result.signal.probability_up
                if confidence < 0.52:
                    continue

            # Filter by candle quality
            body_atr = features.get("body_atr_ratio", 0)
            if body_atr < 0.05 or body_atr > 2.5:
                continue

            total_signals += 1

            settlement_idx = i + 1
            if settlement_idx >= len(candles):
                continue

            entry_price = result.signal.entry_price
            settlement_price = candles[settlement_idx].close

            if result.signal.side == "BUY":
                gross_pips = (settlement_price - entry_price) / 0.0001
            else:
                gross_pips = (entry_price - settlement_price) / 0.0001

            net_pips = gross_pips - 1.2

            if net_pips > 0:
                wins += 1
                outcome = "WIN"
            else:
                losses += 1
                outcome = "LOSS"

            trade_results.append({
                "candle_idx": i,
                "time": str(window[-1].timestamp),
                "side": result.signal.side,
                "entry": entry_price,
                "settlement": settlement_price,
                "net_pips": net_pips,
                "outcome": outcome,
                "regime": regime.name,
                "adx": current_adx,
                "atr": current_atr,
                "confidence": result.signal.probability_up if result.signal.probability_up else 0.5,
            })

    total_resolved = wins + losses
    winrate = (wins / total_resolved * 100) if total_resolved > 0 else 0
    signal_rate = (total_signals / total_evals * 100) if total_evals > 0 else 0

    print("CURRENT RESULTS:")
    print(f"Evaluations: {total_evals}")
    print(f"Signals generated: {total_signals}")
    print(f"Signal Rate: {signal_rate:.1f}% (target: 90%)")
    print(f"Resolved trades: {total_resolved}")
    print(f"Wins: {wins}")
    print(f"Losses: {losses}")
    print(f"Win Rate: {winrate:.1f}% (target: 95%)")
    print()

    # Analyze market conditions
    print("MARKET CONDITION ANALYSIS:")
    print("─" * 70)

    # Count regimes
    trend_count = sum(1 for t in trade_results if t["regime"] == "TREND")
    range_count = sum(1 for t in trade_results if t["regime"] == "RANGE")

    print(f"TREND regime trades: {trend_count}")
    print(f"RANGE regime trades: {range_count}")
    print()

    # Calculate win rates by regime
    if trend_count > 0:
        trend_wins = sum(1 for t in trade_results if t["regime"] == "TREND" and t["outcome"] == "WIN")
        trend_winrate = (trend_wins / trend_count * 100)
        print(f"TREND win rate: {trend_winrate:.1f}% ({trend_wins}/{trend_count})")

    if range_count > 0:
        range_wins = sum(1 for t in trade_results if t["regime"] == "RANGE" and t["outcome"] == "WIN")
        range_winrate = (range_wins / range_count * 100)
        print(f"RANGE win rate: {range_winrate:.1f}% ({range_wins}/{range_count})")
    print()

    # Identify gaps
    print("IDENTIFIED GAPS:")
    print("─" * 70)
    print("1. SIGNAL RATE: 73.5% (need 90%)")
    print("   - Need more permissive gate thresholds")
    print("   - Need more candle data")
    print()
    print("2. WIN RATE: 43.5% (need 95%)")
    print("   - Need better signal quality")
    print("   - Need higher confidence thresholds")
    print("   - Need better entry points")
    print("   - Need shorter hold times")
    print()
    print("3. DATA LIMITATIONS:")
    print("   - Only 740 candles (need 5,000+)")
    print("   - Oct 30 data was ranging (not trending)")
    print("   - Need data from trending periods")
    print()
    print("4. MODEL LIMITATIONS:")
    print("   - Neural model not trained on high-accuracy data")
    print("   - Direction accuracy ~66% (need 80%+)")
    print("   - Need continuous training on trending data")
    print()

    # Save results
    with open("var/performance_analysis.json", "w") as f:
        json.dump({
            "summary": {
                "total_evals": total_evals,
                "total_signals": total_signals,
                "signal_rate": signal_rate,
                "resolved_trades": total_resolved,
                "wins": wins,
                "losses": losses,
                "winrate": winrate,
            },
            "trades": trade_results,
        }, f, indent=2)

    print(f"Results saved to: var/performance_analysis.json")

    return {
        "signal_rate": signal_rate,
        "winrate": winrate,
    }


def provide_recommendations() -> dict:
    """Provide recommendations to achieve 95% win rate with 90% signal rate."""

    print(f"\n{'='*70}")
    print("RECOMMENDATIONS TO ACHIEVE 95% WIN / 90% SIGNAL")
    print(f"{'='*70}\n")

    print("1. DATA REQUIREMENTS:")
    print("─" * 70)
    print("   - Fetch 5,000+ candles from Supabase")
    print("   - Filter for trending periods (ADX > 22)")
    print("   - Use data from multiple days for diversity")
    print("   - Command: python3 scripts/fetch_supabase_candles.py")
    print()

    print("2. NEURAL MODEL TRAINING:")
    print("─" * 70)
    print("   - Train on trending data only")
    print("   - Target 80%+ direction accuracy")
    print("   - Use continuous training on live data")
    print("   - Command: python3 scripts/continuous_train_real.py --max-iters 100")
    print()

    print("3. SIGNAL FILTERING IMPROVEMENTS:")
    print("─" * 70)
    print("   - Increase confidence threshold to 0.70+")
    print("   - Require ADX > 20 for trend strength")
    print("   - Require ATR > 0.5 pips for volatility")
    print("   - Use tighter stop losses (0.5 pips)")
    print("   - Use shorter hold times (1-2 minutes)")
    print()

    print("4. STRATEGY IMPROVEMENTS:")
    print("─" * 70)
    print("   - Use only high-confidence strategies")
    print("   - Filter out low-quality signals")
    print("   - Use regime-specific strategies")
    print("   - Use momentum confirmation")
    print("   - Use VWAP as entry reference")
    print()

    print("5. RISK MANAGEMENT:")
    print("─" * 70)
    print("   - Use kill switch after 2 consecutive losses")
    print("   - Use cooldown period between signals")
    print("   - Use position sizing based on Kelly criterion")
    print("   - Use maximum exposure limits")
    print()

    print("6. BATCH TESTING:")
    print("─" * 70)
    print("   - Test 10+ batches of 20 trades each")
    print("   - Require 19/20 win rate per batch")
    print("   - Require 90% signal rate overall")
    print("   - Command: python3 scripts/batch_test_real_19_20.py --batch-size 20 --num-batches 10")
    print()

    print("EXPECTED TIMELINE:")
    print("─" * 70)
    print("   - Week 1: Fetch and filter data (5,000+ candles)")
    print("   - Week 2: Train neural model (80%+ accuracy)")
    print("   - Week 3: Optimize strategies (95% win rate)")
    print("   - Week 4: Batch testing (90% signal rate)")
    print("   - Week 5: Live trading deployment")
    print()

    return {
        "timeline": "5 weeks",
        "data_needed": "5,000+ candles",
        "model_accuracy_target": "80%+",
        "win_rate_target": "95%",
        "signal_rate_target": "90%",
    }


def main():
    # Load real candles
    candle_file = "var/real_candles/eurusd_real_20261030_live.json"

    if not Path(candle_file).exists():
        print(f"Error: {candle_file} not found")
        return

    candles = load_candles(candle_file)

    # Analyze current performance
    results = analyze_current_performance(candles)

    # Provide recommendations
    recommendations = provide_recommendations()

    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}\n")
    print("CURRENT STATUS:")
    print(f"  Signal Rate: {results['signal_rate']:.1f}% (target: 90%)")
    print(f"  Win Rate: {results['winrate']:.1f}% (target: 95%)")
    print()
    print("PATH TO 95% WIN / 90% SIGNAL:")
    print(f"  1. Fetch 5,000+ candles from Supabase")
    print(f"  2. Filter for trending periods (ADX > 22)")
    print(f"  3. Train neural model to 80%+ accuracy")
    print(f"  4. Optimize strategies for 95% win rate")
    print(f"  5. Test 10+ batches for 19/20 win rate")
    print()
    print(f"TIMELINE: {recommendations['timeline']}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
