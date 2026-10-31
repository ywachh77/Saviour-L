#!/usr/bin/env python3
"""
Continuous training loop for the neural model on real EUR/USD candles.

Repeatedly trains the Temporal Transformer on real candles with
walk-forward splits and varying seeds. Keeps the best checkpoint
based on test direction accuracy.

Usage: python3 scripts/continuous_train_real.py [--max-iters N] [--interval SECONDS]
"""

import argparse
import json
import os
import random
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CANDLES_FILE = PROJECT_ROOT / "var" / "real_candles" / "eurusd_real_20261030_live.json"
CHECKPOINT_DIR = PROJECT_ROOT / "neural" / "checkpoints"
LOG_FILE = PROJECT_ROOT / "var" / "continuous_train_log.jsonl"


class StopRequested:
    pass


def signal_handler(signum, frame):
    raise StopRequested()


def run_training_iteration(iteration: int, seed: int) -> dict:
    """Run one neural training iteration and return metrics."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)

    cmd = [
        sys.executable, "-m", "fusion.neural.operations",
        "--mode", "train",
        "--candles", str(CANDLES_FILE),
    ]
    env["NEURAL_SEED"] = str(seed)

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        env=env,
        timeout=1200,
    )

    if result.returncode != 0:
        return {"error": result.stderr[-500:], "iteration": iteration, "seed": seed}

    # Parse metrics from the JSON output (last JSON block in stdout)
    try:
        # Find the JSON object in stdout
        lines = result.stdout.strip().split("\n")
        json_start = None
        for i, line in enumerate(lines):
            if line.strip() == "{":
                json_start = i
                break
        if json_start is not None:
            report = json.loads("\n".join(lines[json_start:]))
            return {
                "iteration": iteration,
                "seed": seed,
                "model_version": report.get("model_version"),
                "best_epoch": report.get("best_epoch"),
                "best_val_loss": report.get("best_val_loss"),
                "best_val_direction_acc": report.get("best_val_direction_acc"),
                "test_direction_acc": report.get("test_metrics", {}).get("direction_acc"),
                "test_return_mae": report.get("test_metrics", {}).get("return_mae"),
                "test_loss": report.get("test_metrics", {}).get("loss"),
                "finished_at": datetime.now(timezone.utc).isoformat(),
            }
    except (json.JSONDecodeError, KeyError) as e:
        return {"error": f"Parse error: {e}", "iteration": iteration, "seed": seed}

    return {"error": "No JSON output", "iteration": iteration, "seed": seed}


def find_best_checkpoint() -> str | None:
    """Find the best checkpoint by reading report JSON files."""
    best = None
    best_acc = 0.0
    for ckpt_file in CHECKPOINT_DIR.glob("v1-*_report.json"):
        try:
            report = json.loads(ckpt_file.read_text())
            acc = report.get("test_metrics", {}).get("direction_acc", 0)
            if acc > best_acc:
                best_acc = acc
                best = ckpt_file.name.replace("_report.json", "")
        except (json.JSONDecodeError, KeyError):
            continue
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-iters", type=int, default=0, help="Max iterations (0=unlimited)")
    parser.add_argument("--interval", type=int, default=60, help="Seconds between iterations")
    parser.add_argument("--seed-start", type=int, default=100, help="Starting seed")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Verify candle file exists
    if not CANDLES_FILE.exists():
        print(f"ERROR: Candle file not found: {CANDLES_FILE}")
        sys.exit(1)

    with open(CANDLES_FILE) as f:
        candles = json.load(f)
    print(f"Loaded {len(candles)} real candles from {CANDLES_FILE}")
    print(f"Continuous training started at {datetime.now(timezone.utc).isoformat()}")
    print(f"Checkpoint dir: {CHECKPOINT_DIR}")
    print(f"Log file: {LOG_FILE}")
    print(f"Max iterations: {args.max_iters or 'unlimited'}")
    print(f"Interval: {args.interval}s")
    print("---")

    iteration = 0
    while True:
        if args.max_iters > 0 and iteration >= args.max_iters:
            break

        seed = args.seed_start + iteration * 7  # Vary seed each iteration
        iteration += 1

        print(f"\n[{datetime.now(timezone.utc).isoformat()}] Iteration {iteration} (seed={seed})")
        print("Running neural training...")

        try:
            result = run_training_iteration(iteration, seed)

            if "error" in result:
                print(f"ERROR: {result['error']}")
            else:
                print(f"Model: {result['model_version']}")
                print(f"Best epoch: {result['best_epoch']}")
                print(f"Val loss: {result['best_val_loss']:.4f}")
                print(f"Val dir acc: {result['best_val_direction_acc']:.2%}")
                print(f"Test dir acc: {result['test_direction_acc']:.2%}")
                print(f"Test return MAE: {result['test_return_mae']:.4f}")

                # Log to file
                with open(LOG_FILE, "a") as f:
                    f.write(json.dumps(result) + "\n")

                # Find best checkpoint
                best = find_best_checkpoint()
                print(f"Best checkpoint: {best}")
        except subprocess.TimeoutExpired:
            print("TIMEOUT: Training exceeded 20 minutes")
        except StopRequested:
            print("\nStop requested. Exiting...")
            break
        except Exception as e:
            print(f"EXCEPTION: {e}")

        if args.interval > 0:
            print(f"Waiting {args.interval}s before next iteration...")
            try:
                time.sleep(args.interval)
            except StopRequested:
                print("\nStop requested during wait. Exiting...")
                break

    print(f"\nTraining complete. {iteration} iterations ran.")
    best = find_best_checkpoint()
    if best:
        print(f"Best model: {best}")


if __name__ == "__main__":
    main()
