"""Autonomous Operations Daemon — Background service for the neural AI.

Runs continuously to:
  - Collect new market data
  - Evaluate prediction outcomes
  - Trigger scheduled training
  - Run validation
  - Promote qualified models
  - Roll back failed upgrades
  - Monitor model health
  - Recover safely after restarts
  - Resume unfinished jobs

Uses a persistent Python server. Can be started with:
    python -m fusion.neural.operations --mode daemon
"""

from __future__ import annotations

import json
import os
import signal
import sys
import time
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

from .architecture import NeuralConfig
from .memory import NeuralMemory
from .predictor import NeuralPredictor
from .registry import ModelRegistry, PromotionRequirements
from .trainer import NeuralTrainer, TrainingConfig, TrainingReport


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class DaemonState:
    """Persistent daemon state for crash recovery."""
    last_run_id: str = ""
    last_training_at: str = ""
    last_prediction_at: str = ""
    last_outcome_eval_at: str = ""
    last_health_check_at: str = ""
    completed_jobs: list[str] = field(default_factory=list)
    pending_jobs: list[str] = field(default_factory=list)
    is_training: bool = False
    is_validating: bool = False
    is_promoting: bool = False
    consecutive_errors: int = 0
    last_error: str = ""
    started_at: str = ""
    state_version: int = 1
    # Safety guards
    training_failures: int = 0
    last_successful_training: str = ""
    safety_mode: bool = False
    safety_reason: str = ""
    max_consecutive_training_failures: int = 3
    max_model_age_hours: int = 48

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: dict) -> "DaemonState":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Operations Daemon
# ---------------------------------------------------------------------------

class NeuralOperations:
    """Autonomous operations daemon for the neural AI system.

    Usage:
        ops = NeuralOperations()
        ops.run_once()  # Single cycle
        ops.run_daemon()  # Continuous
    """

    def __init__(
        self,
        state_path: str = "neural/operations_state.json",
        checkpoint_dir: str = "neural/checkpoints",
        registry_path: str = "neural/registry.json",
        candles_path: str = "var/real_candles/eurusd_real_20261030.json",
        poll_interval_seconds: int = 60,
        training_interval_hours: int = 6,
        outcome_eval_interval_minutes: int = 5,
    ):
        self.state_path = Path(state_path)
        self.checkpoint_dir = checkpoint_dir
        self.registry_path = registry_path
        self.candles_path = candles_path
        self.poll_interval = poll_interval_seconds
        self.training_interval = timedelta(hours=training_interval_hours)
        self.outcome_eval_interval = timedelta(minutes=outcome_eval_interval_minutes)

        self.registry = ModelRegistry(registry_path, checkpoint_dir)
        self.memory = NeuralMemory()
        self.state = self._load_state()
        # Predictor cache — avoid reloading model every cycle
        self._cached_predictor: NeuralPredictor | None = None
        self._cached_prod_version: str | None = None

    def _load_state(self) -> DaemonState:
        if self.state_path.exists():
            try:
                with open(self.state_path) as f:
                    return DaemonState.from_dict(json.load(f))
            except Exception:
                pass
        state = DaemonState(started_at=datetime.now(timezone.utc).isoformat())
        self._save_state(state)
        return state

    def _save_state(self, state: DaemonState):
        with open(self.state_path, "w") as f:
            json.dump(state.to_dict(), f, indent=2, default=str)

    # ------------------------------------------------------------------ main cycle

    def run_once(self) -> dict:
        """Run one operations cycle. Returns a report dict."""
        report = {
            "cycle_started": datetime.now(timezone.utc).isoformat(),
            "actions": [],
            "errors": [],
        }

        try:
            # 0. Safety guard check
            safety = self._check_safety()
            if safety:
                report["actions"].append(f"Safety mode: {safety}")
                report["cycle_finished"] = datetime.now(timezone.utc).isoformat()
                return report

            # 1. Health check
            health = self._check_health()
            report["actions"].append(f"Health check: {'OK' if health else 'DEGRADED'}")

            # 2. Evaluate outcomes
            if self._should_run("last_outcome_eval_at", self.outcome_eval_interval):
                try:
                    outcomes = self._evaluate_outcomes()
                    report["actions"].append(f"Evaluated {outcomes} outcomes")
                except Exception as e:
                    report["errors"].append(f"Outcome eval failed: {e}")

            # 3. Check if training is needed
            if self._should_run("last_training_at", self.training_interval):
                if not self.state.is_training:
                    try:
                        self._trigger_training()
                        report["actions"].append("Training triggered")
                    except Exception as e:
                        report["errors"].append(f"Training trigger failed: {e}")

            # 4. Generate prediction
            try:
                prediction = self._generate_prediction()
                if prediction:
                    report["actions"].append(f"Prediction: {prediction['action']}")
            except Exception as e:
                report["errors"].append(f"Prediction failed: {e}")

            # 5. Check for pending upgrades
            try:
                upgrade_result = self._check_upgrades()
                if upgrade_result:
                    report["actions"].append(f"Upgrade check: {upgrade_result}")
            except Exception as e:
                report["errors"].append(f"Upgrade check failed: {e}")

        except Exception as e:
            report["errors"].append(f"Cycle failed: {e}")
            traceback.print_exc()

        report["cycle_finished"] = datetime.now(timezone.utc).isoformat()
        return report

    def run_daemon(self, max_cycles: int = 0):
        """Run the daemon continuously. max_cycles=0 means unlimited."""
        cycle = 0
        print(f"Neural Operations Daemon starting at {datetime.now(timezone.utc).isoformat()}")
        print(f"Poll interval: {self.poll_interval}s")
        print(f"Training interval: {self.training_interval}")
        print(f"Outcome eval interval: {self.outcome_eval_interval}")

        def handle_signal(signum, frame):
            print(f"\nReceived signal {signum}, shutting down...")
            self._save_state(self.state)
            sys.exit(0)

        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)

        while True:
            cycle += 1
            try:
                report = self.run_once()
                if report["actions"]:
                    for action in report["actions"]:
                        print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {action}")
                if report["errors"]:
                    for error in report["errors"]:
                        print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] ERROR: {error}")
                    self.state.consecutive_errors += 1
                    self.state.last_error = "; ".join(report["errors"])
                    self._save_state(self.state)
                else:
                    self.state.consecutive_errors = 0

                self._save_state(self.state)

            except Exception as e:
                print(f"FATAL: {e}")
                traceback.print_exc()
                self.state.consecutive_errors += 1
                self.state.last_error = str(e)
                self._save_state(self.state)

            if max_cycles > 0 and cycle >= max_cycles:
                print(f"Completed {cycle} cycles, shutting down.")
                break

            time.sleep(self.poll_interval)

    # ------------------------------------------------------------------ subsystems

    def _check_health(self) -> bool:
        """Check if the production model exists and is healthy."""
        prod = self.registry.get_production_version()
        if prod is None:
            return False
        if not os.path.exists(prod.checkpoint_path):
            return False
        return True

    def _check_safety(self) -> str | None:
        """Check autonomous safety guards. Returns a reason string if in
        safety mode, or None if safe to proceed.

        Guards:
        1. Consecutive training failures — pause if exceeded
        2. Model staleness — alert if production model is too old
        3. Model health — pause if production model health check fails
        """
        # Guard 1: Consecutive training failures
        if self.state.training_failures >= self.state.max_consecutive_training_failures:
            if not self.state.safety_mode:
                self._enter_safety_mode(
                    f"Training failed {self.state.training_failures} consecutive times"
                )
            return self.state.safety_reason

        # Guard 2: Model staleness
        prod = self.registry.get_production_version()
        if prod is not None:
            try:
                model_age = datetime.now(timezone.utc) - datetime.fromisoformat(
                    prod.created_at
                )
                if model_age > timedelta(hours=self.state.max_model_age_hours):
                    return f"Model stale: {model_age.total_seconds() // 3600:.0f}h old"
            except (ValueError, TypeError):
                pass

        # Guard 3: Model health — don't predict if health check fails
        if prod is not None:
            if not os.path.exists(prod.checkpoint_path):
                if not self.state.safety_mode:
                    self._enter_safety_mode("Production model checkpoint missing")
                return self.state.safety_reason

        return None

    def _enter_safety_mode(self, reason: str) -> None:
        """Enter safety mode — pauses training and predictions."""
        self.state.safety_mode = True
        self.state.safety_reason = reason
        self._save_state(self.state)
        print(f"SAFETY MODE ENGAGED: {reason}")

    def _exit_safety_mode(self) -> None:
        """Exit safety mode — clears the flag and resets counters."""
        self.state.safety_mode = False
        self.state.safety_reason = ""
        self.state.training_failures = 0
        self._save_state(self.state)
        print("SAFETY MODE DISENGAGED")

    def _should_run(self, attr: str, interval: timedelta) -> bool:
        """Check if an action should run based on the last run time."""
        last = self.state.__dict__.get(attr, "")
        if not last:
            return True
        try:
            last_dt = datetime.fromisoformat(last)
            return datetime.now(timezone.utc) - last_dt >= interval
        except Exception:
            return True

    def _evaluate_outcomes(self) -> int:
        """Evaluate settled outcomes and update learning.

        Fetches current price for each unresolved signal, settles the
        outcome, and saves it to memory. Only settles signals whose
        expiry time has passed — never trains on unresolved outcomes.
        """
        unresolved = self.memory.get_unresolved_outcomes(limit=50)
        count = 0
        for signal in unresolved:
            # Check if the signal has expired
            signal_ts = signal.get("timestamp", "")
            horizon = signal.get("horizon_minutes", 5)
            if signal_ts:
                try:
                    signal_time = datetime.fromisoformat(signal_ts)
                    expiry = signal_time + timedelta(minutes=horizon)
                    if datetime.now(timezone.utc) < expiry:
                        continue  # Not yet expired
                except (ValueError, TypeError):
                    continue  # Bad timestamp, skip

            # Get the entry price and side
            entry_price = signal.get("entry_price")
            side = signal.get("side", "BUY")
            signal_id = signal.get("signal_id", "")
            pair = signal.get("pair", "EUR/USD")

            if entry_price is None:
                continue

            # Fetch current price from candle data
            current_price = self._get_current_price()
            if current_price is None:
                continue  # No price data available

            # Calculate PnL
            spread_pips = signal.get("spread_pips", 1.2)
            if side == "BUY":
                pnl_pips = (current_price - entry_price) / 0.0001 - spread_pips
            else:
                pnl_pips = (entry_price - current_price) / 0.0001 - spread_pips

            # Determine outcome
            if pnl_pips > 0:
                outcome = "WIN"
            elif pnl_pips < 0:
                outcome = "LOSS"
            else:
                outcome = "BREAKEVEN"

            # Create settlement
            from ..domain import Settlement
            settlement = Settlement(
                signal_id=signal_id,
                outcome=outcome,
                entry_price=entry_price,
                exit_price=current_price,
                gross_pips=pnl_pips + spread_pips,
                move_pips=pnl_pips + spread_pips,
                net_pips=pnl_pips,
                spread_pips=spread_pips,
                settled_at=datetime.now(timezone.utc),
                source="neural_operations",
            )

            # Save outcome to memory
            settlement_dict = {
                "outcome": outcome,
                "entry_price": entry_price,
                "exit_price": current_price,
                "gross_pips": pnl_pips + spread_pips,
                "net_pips": pnl_pips,
                "spread_pips": spread_pips,
                "settled_at": datetime.now(timezone.utc).isoformat(),
            }
            self.memory.save_outcome(signal_id, settlement_dict)
            count += 1

        self.state.last_outcome_eval_at = datetime.now(timezone.utc).isoformat()
        return count

    def _get_current_price(self) -> float | None:
        """Get the latest close price from candle data."""
        try:
            if not os.path.exists(self.candles_path):
                return None
            with open(self.candles_path) as f:
                rows = json.load(f)
            if not rows:
                return None
            # rows are newest-first from Supabase, so [0] is latest
            return float(rows[0].get("c", 0))
        except (ValueError, KeyError, OSError, json.JSONDecodeError):
            return None

    def _trigger_training(self) -> TrainingReport | None:
        """Trigger a new training run. Tracks failures for safety mode."""
        self.state.is_training = True
        try:
            # Load candles
            if not os.path.exists(self.candles_path):
                self.state.last_training_at = datetime.now(timezone.utc).isoformat()
                return None  # Skip training, don't retry immediately

            with open(self.candles_path) as f:
                rows = json.load(f)
            rows.reverse()

            from ..domain import Candle
            candles = []
            for r in rows:
                ts = datetime.fromisoformat(r["ts"])
                candles.append(Candle(
                    timestamp=ts, open=r["o"], high=r["h"], low=r["l"],
                    close=r["c"], volume=r["v"],
                ))

            # Train
            trainer = NeuralTrainer(
                TrainingConfig(epochs=30, patience=5,
                               checkpoint_dir=self.checkpoint_dir),
                NeuralConfig(),
            )
            report = trainer.train(candles)

            # Register candidate
            self.registry.register_candidate(
                report.model_version, report,
                f"{self.checkpoint_dir}/{report.model_version}.pt",
            )

            # Save to memory
            self.memory.save_training_report(report.to_dict())
            self.memory.save_model_version(
                self.registry.versions[-1].to_dict()
            )

            self.state.last_training_at = datetime.now(timezone.utc).isoformat()
            self.state.last_successful_training = self.state.last_training_at
            self.state.training_failures = 0
            if self.state.safety_mode:
                self._exit_safety_mode()
            return report

        except Exception as e:
            self.state.training_failures += 1
            self.state.last_error = str(e)
            print(f"Training failed ({self.state.training_failures}/"
                  f"{self.state.max_consecutive_training_failures}): {e}")
            raise

        finally:
            self.state.is_training = False

    def _generate_prediction(self) -> dict | None:
        """Generate a live prediction."""
        prod = self.registry.get_production_version()
        if prod is None:
            return None

        # Use cached predictor if production version hasn't changed
        cached_version = getattr(self, "_cached_prod_version", None)
        if self._cached_predictor is not None and cached_version == prod.version:
            predictor = self._cached_predictor
        else:
            predictor = NeuralPredictor.from_checkpoint(prod.checkpoint_path)
            self._cached_predictor = predictor
            self._cached_prod_version = prod.version

        # Load recent candles
        if not os.path.exists(self.candles_path):
            return None

        with open(self.candles_path) as f:
            rows = json.load(f)
        rows.reverse()

        from ..domain import Candle
        candles = []
        for r in rows[-200:]:  # Last 200 candles
            ts = datetime.fromisoformat(r["ts"])
            candles.append(Candle(
                timestamp=ts, open=r["o"], high=r["h"], low=r["l"],
                close=r["c"], volume=r["v"],
            ))

        prediction = predictor.predict(candles, pair="EUR/USD")
        pred_dict = prediction.to_dict()

        # Save signal
        self.memory.save_signal(pred_dict)

        # Save health
        health = predictor.get_health()
        health["model_version"] = prod.version
        self.memory.save_health(health)

        self.state.last_prediction_at = datetime.now(timezone.utc).isoformat()
        return pred_dict

    def _check_upgrades(self) -> str | None:
        """Check if any candidate should be promoted."""
        # Find CANDIDATE versions
        candidates = [v for v in self.registry.versions if v.status == "CANDIDATE"]
        if not candidates:
            return None

        # Load candles for validation
        if not os.path.exists(self.candles_path):
            return "No candle data for validation"

        with open(self.candles_path) as f:
            rows = json.load(f)
        rows.reverse()

        from ..domain import Candle
        candles = []
        for r in rows:
            ts = datetime.fromisoformat(r["ts"])
            candles.append(Candle(
                timestamp=ts, open=r["o"], high=r["h"], low=r["l"],
                close=r["c"], volume=r["v"],
            ))

        # Evaluate each candidate
        for candidate in candidates:
            result = self.registry.evaluate_candidate(candles, candidate.version)

            if result.get("promotion_decision") == "PRODUCTION":
                # Record upgrade
                self.memory.save_upgrade({
                    "from_version": self.registry.get_production_version().version
                    if self.registry.get_production_version() else "",
                    "to_version": candidate.version,
                    "status": "PROMOTED",
                    "metrics": result,
                    "notes": "Auto-promoted after walk-forward validation",
                })
                return f"Promoted {candidate.version}"
            elif result.get("promotion_decision") == "REJECTED":
                self.memory.save_upgrade({
                    "from_version": "",
                    "to_version": candidate.version,
                    "status": "REJECTED",
                    "metrics": result,
                    "notes": "Failed promotion requirements",
                })
                return f"Rejected {candidate.version}"

        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="SAVIOUR Neural AI Operations Daemon",
    )
    parser.add_argument("--mode", choices=["daemon", "once", "train", "predict", "stats"],
                        default="once", help="Operation mode")
    parser.add_argument("--candles", default="var/real_candles/eurusd_real_20261030.json",
                        help="Path to candle data")
    parser.add_argument("--poll", type=int, default=60, help="Poll interval (seconds)")
    parser.add_argument("--max-cycles", type=int, default=0, help="Max cycles (0=unlimited)")

    args = parser.parse_args()

    ops = NeuralOperations(candles_path=args.candles, poll_interval_seconds=args.poll)

    if args.mode == "once":
        report = ops.run_once()
        print(json.dumps(report, indent=2, default=str))
    elif args.mode == "daemon":
        ops.run_daemon(max_cycles=args.max_cycles)
    elif args.mode == "train":
        # Load candles and train
        with open(args.candles) as f:
            rows = json.load(f)
        rows.reverse()
        from ..domain import Candle
        candles = []
        for r in rows:
            ts = datetime.fromisoformat(r["ts"])
            candles.append(Candle(
                timestamp=ts, open=r["o"], high=r["h"], low=r["l"],
                close=r["c"], volume=r["v"],
            ))
        trainer = NeuralTrainer(TrainingConfig(epochs=30), NeuralConfig())
        report = trainer.train(candles)
        print(json.dumps(report.to_dict(), indent=2, default=str))
    elif args.mode == "predict":
        ops._generate_prediction()
    elif args.mode == "stats":
        registry = ModelRegistry(ops.registry_path, ops.checkpoint_dir)
        versions = registry.list_versions()
        print(f"Model versions: {len(versions)}")
        for v in versions:
            print(f"  {v['version']} | {v['status']} | {v.get('metrics', {}).get('mean_direction_acc', 0):.1%}")
        prod = registry.get_production_version()
        if prod:
            print(f"\nProduction: {prod.version}")


if __name__ == "__main__":
    main()
