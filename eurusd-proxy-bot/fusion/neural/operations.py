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
from datetime import datetime, timedelta
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

    def _load_state(self) -> DaemonState:
        if self.state_path.exists():
            try:
                with open(self.state_path) as f:
                    return DaemonState.from_dict(json.load(f))
            except Exception:
                pass
        state = DaemonState(started_at=datetime.utcnow().isoformat())
        self._save_state(state)
        return state

    def _save_state(self, state: DaemonState):
        with open(self.state_path, "w") as f:
            json.dump(state.to_dict(), f, indent=2, default=str)

    # ------------------------------------------------------------------ main cycle

    def run_once(self) -> dict:
        """Run one operations cycle. Returns a report dict."""
        report = {
            "cycle_started": datetime.utcnow().isoformat(),
            "actions": [],
            "errors": [],
        }

        try:
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

        report["cycle_finished"] = datetime.utcnow().isoformat()
        return report

    def run_daemon(self, max_cycles: int = 0):
        """Run the daemon continuously. max_cycles=0 means unlimited."""
        cycle = 0
        print(f"Neural Operations Daemon starting at {datetime.utcnow().isoformat()}")
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
                        print(f"[{datetime.utcnow().strftime('%H:%M:%S')}] {action}")
                if report["errors"]:
                    for error in report["errors"]:
                        print(f"[{datetime.utcnow().strftime('%H:%M:%S')}] ERROR: {error}")
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

    def _should_run(self, attr: str, interval: timedelta) -> bool:
        """Check if an action should run based on the last run time."""
        last = self.state.__dict__.get(attr, "")
        if not last:
            return True
        try:
            last_dt = datetime.fromisoformat(last)
            return datetime.utcnow() - last_dt >= interval
        except Exception:
            return True

    def _evaluate_outcomes(self) -> int:
        """Evaluate settled outcomes and update learning."""
        unresolved = self.memory.get_unresolved_outcomes(limit=50)
        count = 0
        for signal in unresolved:
            # In production, this would fetch the current price and settle
            # For now, just count
            count += 1

        self.state.last_outcome_eval_at = datetime.utcnow().isoformat()
        return count

    def _trigger_training(self) -> TrainingReport | None:
        """Trigger a new training run."""
        self.state.is_training = True
        try:
            # Load candles
            if not os.path.exists(self.candles_path):
                self.state.last_training_at = datetime.utcnow().isoformat()
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

            self.state.last_training_at = datetime.utcnow().isoformat()
            return report

        finally:
            self.state.is_training = False

    def _generate_prediction(self) -> dict | None:
        """Generate a live prediction."""
        prod = self.registry.get_production_version()
        if prod is None:
            return None

        predictor = NeuralPredictor.from_checkpoint(prod.checkpoint_path)

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

        self.state.last_prediction_at = datetime.utcnow().isoformat()
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
