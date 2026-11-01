"""AI Brain command-line interface — the primary entry point.

Usage:
    python -m fusion.brain_cli evaluate --pair EUR/USD --candles-file ...
    python -m fusion.brain_cli backtest --pair EUR/USD --candles-file ...
    python -m fusion.brain_cli stats --state-file ...
    python -m fusion.brain_cli run --pair EUR/USD --live

The brain orchestrates the full pipeline: data gate → features → regime →
council → model → seal, plus learning, auto-tuning, and analytics.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add parent to path when running as script
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fusion.brain import AIBrain, TuningConfig
from fusion.config import Config
from fusion.domain import Candle, Settlement
from fusion.charter import WIN, LOSS


def load_candles(filepath: str) -> list[Candle]:
    """Load candles from a JSON file (Supabase format)."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Candle file not found: {filepath}")
    with open(filepath) as f:
        rows = json.load(f)
    if not rows:
        raise ValueError(f"Candle file is empty: {filepath}")
    rows.reverse()  # oldest first
    candles = []
    for r in rows:
        ts = datetime.fromisoformat(r["ts"])
        candles.append(Candle(
            timestamp=ts,
            open=r["o"], high=r["h"], low=r["l"], close=r["c"], volume=r["v"],
        ))
    return candles


def load_candles_simple(filepath: str) -> list[Candle]:
    """Load candles from a simple JSON format: [{"ts": "...", "o": ..., ...}]"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Candle file not found: {filepath}")
    with open(filepath) as f:
        rows = json.load(f)
    if not rows:
        raise ValueError(f"Candle file is empty: {filepath}")
    rows.reverse()
    candles = []
    for r in rows:
        ts = datetime.fromisoformat(r["ts"])
        candles.append(Candle(
            timestamp=ts,
            open=r["o"], high=r["h"], low=r["l"], close=r["c"], volume=r["v"],
        ))
    return candles


def cmd_evaluate(args):
    """Evaluate the brain on candle data and print signals."""
    candles = load_candles(args.candles)
    config = Config(pair=args.pair)
    brain = AIBrain(config, tuning=TuningConfig())

    print(f"=== AI BRAIN EVALUATION: {args.pair} ===\n")
    print(f"Loaded {len(candles)} candles")
    print(f"Starting from candle {args.start} (0-indexed)")
    print()

    start_idx = int(args.start)
    interval = int(args.interval)

    for idx in range(start_idx, len(candles), interval):
        sub = candles[:idx + 1]
        result = brain.evaluate(sub, source="BACKTEST", at=sub[-1].timestamp)

        if result["seal_state"] == "SIGNAL":
            sig = result["signal"]
            print(f"✅ SIGNAL {sig['entry_time'][11:16]} UTC | {sig['side']:4s} | {sig['horizon_minutes']}min | {sig['regime']}")
            if args.verbose:
                print(f"   Reason: {sig['reason'][:100]}")
                print(f"   Council: {len(sig.get('council', []))} votes")
        else:
            if args.verbose:
                ts = sub[-1].timestamp.strftime("%H:%M")
                print(f"⏸️ {ts} UTC | {result['reason'][:80]}")

    # Print stats
    print(f"\n=== BRAIN STATS ===")
    print(f"Evaluations: {brain.stats.evaluations}")
    print(f"Signals sent: {brain.stats.signals_sent}")
    print(f"Wins: {brain.stats.wins}, Losses: {brain.stats.losses}")
    print(f"Win rate: {brain.stats.win_rate:.0%}")
    print(f"Total net pips: {brain.stats.total_net_pips:.2f}")
    print(f"Methods tracked: {len(brain.stats.by_method)}")
    print(f"Families tracked: {len(brain.stats.by_family)}")


def cmd_backtest(args):
    """Run a full backtest with settlements and print performance report."""
    candles = load_candles(args.candles)
    config = Config(pair=args.pair)
    brain = AIBrain(config, tuning=TuningConfig())

    print(f"=== AI BRAIN BACKTEST: {args.pair} ===\n")
    print(f"Loaded {len(candles)} candles")
    print(f"Starting from candle {args.start}, interval {args.interval} min\n")

    start_idx = int(args.start)
    interval = int(args.interval)
    spread = args.spread

    for idx in range(start_idx, len(candles), interval):
        sub = candles[:idx + 1]
        result = brain.evaluate(sub, source="BACKTEST", at=sub[-1].timestamp)

        if result["seal_state"] == "SIGNAL":
            sig = result["signal"]
            entry_price = candles[idx].open
            horizon = sig["horizon_minutes"]
            future_idx = idx + horizon

            if future_idx < len(candles):
                future_close = candles[future_idx].close

                if sig["side"] == "BUY":
                    outcome = WIN if future_close > entry_price else LOSS
                else:
                    outcome = WIN if future_close < entry_price else LOSS

                pips = abs(future_close - entry_price) / 0.0001

                entry_time = datetime.fromisoformat(sig["entry_time"])
                settle_time = entry_time + timedelta(minutes=horizon)

                settlement = Settlement(
                    signal_id=sig["signal_id"],
                    outcome=outcome,
                    entry_price=entry_price,
                    exit_price=future_close,
                    gross_pips=pips,
                    move_pips=pips,
                    net_pips=pips - spread,
                    spread_pips=spread,
                    settled_at=settle_time,
                    source="BACKTEST",
                )
                brain.observe_outcome(sig, settlement)

                status = "✅" if outcome == WIN else "❌"
                print(f"{status} {sig['entry_time'][11:16]} UTC | {sig['side']:4s} {horizon:2d}min | {pips:6.2f} pips | {sig['regime']:8s}")

    # Print comprehensive report
    print(f"\n{'='*70}")
    print(f"=== AI BRAIN BACKTEST REPORT ===")
    print(f"{'='*70}")
    print(f"Evaluations: {brain.stats.evaluations}")
    print(f"Signals sent: {brain.stats.signals_sent}")
    print(f"Wins: {brain.stats.wins}, Losses: {brain.stats.losses}")
    print(f"Win rate: {brain.stats.win_rate:.1%}")
    print(f"Total net pips: {brain.stats.total_net_pips:.2f}")
    print(f"Methods tracked: {len(brain.stats.by_method)}")
    print(f"Families tracked: {len(brain.stats.by_family)}")

    # Top methods
    print(f"\n=== TOP 10 METHODS ===")
    methods = sorted(brain.stats.by_method.values(), key=lambda m: m.net_edge, reverse=True)
    for m in methods[:10]:
        avg_w = m.total_weight / max(1, m.votes)
        print(f"  {m.name:40s} | {m.wins:3d}W {m.losses:3d}L | {m.win_rate:5.0%} | edge={m.net_edge:+.3f} | avg_w={avg_w:.3f}")

    # Family performance
    print(f"\n=== FAMILY PERFORMANCE ===")
    for name, fs in sorted(brain.stats.by_family.items(), key=lambda x: x[1].signals, reverse=True):
        wr = fs.win_rate if fs.signals > 0 else 0
        print(f"  {name:30s} | {fs.signals:3d} signals | {fs.wins}W {fs.losses}L | {wr:.0%} win rate")

    # Regime performance
    print(f"\n=== REGIME PERFORMANCE ===")
    for name, rs in sorted(brain.stats.by_regime.items(), key=lambda x: x[1]["wins"] + x[1]["losses"], reverse=True):
        total = rs["wins"] + rs["losses"]
        wr = rs["wins"] / total if total > 0 else 0
        print(f"  {name:12s} | {rs['wins']}W {rs['losses']}L | {wr:.0%}")

    # Expiry performance
    print(f"\n=== EXPIRY PERFORMANCE ===")
    for horizon, outcomes in sorted(brain._expiry_performance.items()):
        wins = outcomes.count("WIN")
        losses = outcomes.count("LOSS")
        total = wins + losses
        wr = wins / total if total > 0 else 0
        print(f"  {horizon}min | {wins}W {losses}L | {wr:.0%} | {total} trades")

    # Save state
    if args.output:
        state = brain.to_dict()
        with open(args.output, "w") as f:
            json.dump(state, f, indent=2, default=str)
        print(f"\nBrain state saved to {args.output}")


def cmd_stats(args):
    """Print brain statistics from a saved state file."""
    with open(args.state_file) as f:
        state = json.load(f)

    brain = AIBrain.from_dict(state, Config(pair=args.pair))

    print(f"=== AI BRAIN STATS ===\n")
    print(f"Evaluations: {brain.stats.evaluations}")
    print(f"Signals sent: {brain.stats.signals_sent}")
    print(f"Wins: {brain.stats.wins}, Losses: {brain.stats.losses}")
    print(f"Win rate: {brain.stats.win_rate:.1%}")
    print(f"Total net pips: {brain.stats.total_net_pips:.2f}")
    print(f"Methods tracked: {len(brain.stats.by_method)}")
    print(f"Families tracked: {len(brain.stats.by_family)}")
    print(f"Auto-tunes performed: {brain.stats.tuning_runs}")

    print(f"\n=== TOP 10 METHODS ===")
    methods = sorted(brain.stats.by_method.values(), key=lambda m: m.net_edge, reverse=True)
    for m in methods[:10]:
        print(f"  {m.name:40s} | {m.wins:3d}W {m.losses:3d}L | {m.win_rate:5.0%} | edge={m.net_edge:+.3f}")

    print(f"\n=== FAMILY PERFORMANCE ===")
    for name, fs in sorted(brain.stats.by_family.items(), key=lambda x: x[1].signals, reverse=True):
        wr = fs.win_rate if fs.signals > 0 else 0
        print(f"  {name:30s} | {fs.signals:3d} signals | {fs.wins}W {fs.losses}L | {wr:.0%}")


def cmd_run(args):
    """Run the brain in live mode (continuous evaluation)."""
    config = Config(pair=args.pair)
    brain = AIBrain(config, tuning=TuningConfig())

    print(f"=== AI BRAIN LIVE: {args.pair} ===")
    print(f"Monitoring... Press Ctrl+C to stop.\n")

    # In live mode, the brain would connect to a data feed
    # For now, print instructions
    print("Live mode requires a real-time data feed connection.")
    print("Use 'evaluate' or 'backtest' with candle data files.")


def main():
    parser = argparse.ArgumentParser(
        description="AI Brain — Central orchestrator for signal generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate on real candle data
  python -m fusion.brain_cli evaluate --candles var/real_candles/eurusd_real_20261030.json

  # Full backtest with settlements
  python -m fusion.brain_cli backtest --candles var/real_candles/eurusd_real_20261030.json --start 60 --interval 3

  # Print saved brain stats
  python -m fusion.brain_cli stats --state-file brain_state.json
        """,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Evaluate
    p_eval = sub.add_parser("evaluate", help="Evaluate brain on candle data")
    p_eval.add_argument("--candles", required=True, help="Path to candle JSON file")
    p_eval.add_argument("--pair", default="EUR/USD", help="Trading pair")
    p_eval.add_argument("--start", default="60", help="Starting candle index")
    p_eval.add_argument("--interval", default="3", help="Evaluation interval in minutes")
    p_eval.add_argument("--verbose", action="store_true", help="Show abstentions too")
    p_eval.set_defaults(func=cmd_evaluate)

    # Backtest
    p_bt = sub.add_parser("backtest", help="Full backtest with settlements")
    p_bt.add_argument("--candles", required=True, help="Path to candle JSON file")
    p_bt.add_argument("--pair", default="EUR/USD", help="Trading pair")
    p_bt.add_argument("--start", default="60", help="Starting candle index")
    p_bt.add_argument("--interval", default="3", help="Evaluation interval in minutes")
    p_bt.add_argument("--spread", type=float, default=1.2, help="Spread in pips")
    p_bt.add_argument("--output", help="Save brain state to JSON file")
    p_bt.set_defaults(func=cmd_backtest)

    # Stats
    p_stats = sub.add_parser("stats", help="Print brain statistics from saved state")
    p_stats.add_argument("--state-file", required=True, help="Path to saved brain state JSON")
    p_stats.add_argument("--pair", default="EUR/USD", help="Trading pair")
    p_stats.set_defaults(func=cmd_stats)

    # Run (live)
    p_run = sub.add_parser("run", help="Run brain in live mode")
    p_run.add_argument("--pair", default="EUR/USD", help="Trading pair")
    p_run.set_defaults(func=cmd_run)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
