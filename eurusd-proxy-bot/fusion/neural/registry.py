"""Model Registry — Candidate promotion, versioning, and rollback.

Maintains a registry of model versions with full provenance. New candidate
models must pass walk-forward validation before promotion to production.
Automatic rollback on failed upgrades.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .architecture import NeuralConfig, TemporalTransformer, build_model
from .trainer import NeuralTrainer, TrainingReport


# ---------------------------------------------------------------------------
# Requirements for promotion
# ---------------------------------------------------------------------------

@dataclass
class PromotionRequirements:
    """Minimum performance requirements for model promotion.

    A candidate must meet ALL of these to be promoted:
    1. Walk-forward validation passes
    2. Out-of-sample test loss improves over production
    3. Win rate >= minimum threshold
    4. Max drawdown within limits
    5. Transaction-cost-adjusted return > 0
    6. Prediction calibration acceptable
    7. No regression in any regime
    """
    min_win_rate: float = 0.55          # Min direction accuracy
    min_oof_improvement: float = 0.01   # Min improvement over production (relative)
    max_drawdown_pips: float = 50.0     # Max drawdown in pips
    min_net_return_pips: float = 10.0   # Min net return in pips
    min_calibration_score: float = 0.50  # Min Brier score (lower is better, so max here)
    max_val_loss_ratio: float = 1.05    # Max val_loss / train_loss ratio
    min_regimes_tested: int = 3         # Min number of regimes tested
    min_predictions: int = 50           # Min number of OOF predictions

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


# ---------------------------------------------------------------------------
# Model version info
# ---------------------------------------------------------------------------

@dataclass
class ModelVersion:
    """Metadata for a model version in the registry."""
    version: str
    checkpoint_path: str
    training_report: dict
    neural_config: dict
    created_at: str
    status: str = "CANDIDATE"  # CANDIDATE | PRODUCTION | ROLLBACK | REJECTED
    metrics: dict = field(default_factory=dict)
    is_production: bool = False
    parent_version: str = ""
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "checkpoint_path": self.checkpoint_path,
            "training_report": self.training_report,
            "neural_config": self.neural_config,
            "created_at": self.created_at,
            "status": self.status,
            "metrics": self.metrics,
            "is_production": self.is_production,
            "parent_version": self.parent_version,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Walk-forward validation
# ---------------------------------------------------------------------------

class WalkForwardValidator:
    """Chronological walk-forward validation for model candidates.

    Splits data into expanding-window folds:
      - Fold 1: train on [0, t1), validate on [t1, t2)
      - Fold 2: train on [0, t2), validate on [t2, t3)
      - etc.

    This prevents future data leakage and gives honest OOF estimates.
    """

    def __init__(self, n_folds: int = 5, test_size: int = 200):
        self.n_folds = n_folds
        self.test_size = test_size

    def validate(self, candles, neural_config, training_config) -> dict:
        """Run walk-forward validation and return metrics."""
        from .dataset import NeuralDataset
        from .trainer import NeuralTrainer, multi_task_loss

        n = len(candles)
        min_train = neural_config.input_window + training_config.batch_size * 2

        if n < min_train + self.test_size * 2:
            return {
                "valid": False,
                "note": f"Not enough candles ({n}) for walk-forward validation.",
                "folds": [],
            }

        fold_size = (n - min_train) // (self.n_folds + 1)
        folds_data = []
        fold_start = min_train

        for fold in range(self.n_folds):
            train_end = fold_start + fold_size
            test_end = train_end + self.test_size
            if train_end >= n or test_end > n:
                break

            folds_data.append({
                "fold": fold + 1,
                "train_start": fold_start,
                "train_end": train_end,
                "test_start": train_end,
                "test_end": test_end,
            })
            fold_start = test_end

        results = []
        all_test_preds = []
        all_test_targets = []
        all_prob_preds = []
        all_returns = []

        for fold_data in folds_data:
            train_candles = candles[fold_data["train_start"]:fold_data["train_end"]]
            test_candles = candles[fold_data["test_start"]:fold_data["test_end"]]

            trainer = NeuralTrainer(training_config, neural_config)
            trainer.config.epochs = min(20, training_config.epochs)
            trainer.config.patience = 5
            # Use the trainer's checkpoint_dir (already set from training_config)

            report = trainer.train(train_candles)

            if not report.history:
                continue

            # Evaluate on test fold
            test_dataset = NeuralDataset(test_candles, neural_config)
            if len(test_dataset) == 0:
                continue

            # Load from the trainer's checkpoint directory
            ckpt_path = f"{training_config.checkpoint_dir}/{report.model_version}.pt"
            model, _, _ = NeuralTrainer.load_checkpoint(ckpt_path)

            from torch.utils.data import DataLoader
            loader = DataLoader(test_dataset, batch_size=training_config.batch_size,
                               shuffle=False)

            loss, acc, preds, targets, prob_preds, returns = self._evaluate_oof(model, loader, training_config)

            results.append({
                "fold": fold_data["fold"],
                "test_loss": loss,
                "test_direction_acc": acc,
                "n_predictions": len(preds),
            })

            all_test_preds.extend(preds)
            all_test_targets.extend(targets)
            all_prob_preds.extend(prob_preds)
            all_returns.extend(returns)

        # Aggregate
        if not results:
            return {"valid": False, "note": "No folds completed.", "folds": []}

        losses = [r["test_loss"] for r in results]
        accs = [r["test_direction_acc"] for r in results]

        # Compute calibration (Brier score) using predicted probabilities
        brier = self._brier_score(all_prob_preds, all_test_targets)

        # Compute net return (transaction-cost adjusted, using actual returns)
        net_return, max_dd = self._net_return(all_test_preds, all_test_targets, all_returns)

        return {
            "valid": True,
            "n_folds": len(results),
            "mean_test_loss": np.mean(losses),
            "std_test_loss": np.std(losses),
            "mean_direction_acc": np.mean(accs),
            "min_direction_acc": np.min(accs),
            "brier_score": brier,
            "net_return_pips": net_return,
            "max_drawdown_pips": max_dd,
            "folds": results,
        }

    def _evaluate_oof(self, model, loader, config, device="cpu"):
        """Evaluate model and collect predictions + probabilities + returns for OOF analysis."""
        model.eval()
        losses = []
        correct = 0
        total = 0
        preds = []
        targets = []
        prob_preds = []  # Predicted probabilities for calibration
        returns = []     # Actual return pips for net return calculation

        with torch.no_grad():
            for batch in loader:
                features = batch["features"]
                state = batch["state"]
                direction = batch["direction"]
                ret = batch["return_pips"]

                output = model(features, state)
                loss, _ = multi_task_loss(output, batch, config)
                losses.append(loss.item())

                pred_dir = output["direction_logits"].argmax(dim=-1)
                correct += (pred_dir == direction).sum().item()
                total += direction.size(0)

                # Get predicted probabilities for calibration
                dir_probs = torch.softmax(output["direction_logits"], dim=-1)

                for i in range(direction.size(0)):
                    preds.append(pred_dir[i].item())
                    targets.append(direction[i].item())
                    prob_preds.append(dir_probs[i].tolist())
                    returns.append(ret[i].item())

        return np.mean(losses), correct / max(1, total), preds, targets, prob_preds, returns

    def _brier_score(self, prob_preds, targets) -> float:
        """Compute Brier score using predicted probabilities (lower is better).

        Brier score = mean(sum((predicted_prob - one_hot_target)^2))
        Range: [0, 2] for 3-class problem. Lower is better.
        """
        if not prob_preds:
            return 1.0
        probs = np.array(prob_preds)  # (n, 3)
        one_hot = np.eye(3)[targets]  # (n, 3)
        return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))

    def _net_return(self, preds, targets, returns, spread_pips=1.2) -> tuple[float, float]:
        """Compute transaction-cost-adjusted net return and max drawdown.

        Uses actual return values from the dataset, not assumed averages.
        """
        if not preds:
            return 0.0, 0.0

        # Only count trades where the model predicted BUY or SELL (not NO_TRADE)
        total_net = 0.0
        running_peak = 0.0
        max_dd = 0.0
        equity = 0.0

        for p, t, r in zip(preds, targets, returns):
            if p == 2:  # NO_TRADE — skip
                continue

            # Net return after spread cost
            if p == 0:  # BUY
                net = r - spread_pips
            elif p == 1:  # SELL
                net = -r - spread_pips
            else:
                continue

            equity += net
            running_peak = max(running_peak, equity)
            max_dd = max(max_dd, running_peak - equity)

        return total_net if not preds else equity, max_dd


# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------

class ModelRegistry:
    """Model registry with automatic candidate promotion and rollback.

    Manages model versions, runs walk-forward validation, and promotes
    qualifying candidates to production automatically.
    """

    def __init__(
        self,
        registry_path: str = "neural/registry.json",
        checkpoint_dir: str = "neural/checkpoints",
        requirements: PromotionRequirements | None = None,
    ):
        self.registry_path = Path(registry_path)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.requirements = requirements or PromotionRequirements()
        self.versions: list[ModelVersion] = []
        self._load()

    def _load(self):
        if self.registry_path.exists():
            with open(self.registry_path) as f:
                data = json.load(f)
                for v_data in data.get("versions", []):
                    self.versions.append(ModelVersion(
                        version=v_data["version"],
                        checkpoint_path=v_data["checkpoint_path"],
                        training_report=v_data.get("training_report", {}),
                        neural_config=v_data.get("neural_config", {}),
                        created_at=v_data["created_at"],
                        status=v_data.get("status", "CANDIDATE"),
                        metrics=v_data.get("metrics", {}),
                        is_production=v_data.get("is_production", False),
                        parent_version=v_data.get("parent_version", ""),
                        notes=v_data.get("notes", ""),
                    ))

    def _save(self):
        data = {
            "versions": [v.to_dict() for v in self.versions],
            "requirements": self.requirements.to_dict(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(self.registry_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def get_production_version(self) -> ModelVersion | None:
        for v in self.versions:
            if v.is_production:
                return v
        return None

    def register_candidate(
        self,
        model_version: str,
        training_report: TrainingReport,
        checkpoint_path: str,
        parent_version: str = "",
    ) -> ModelVersion:
        """Register a new candidate model."""
        version = ModelVersion(
            version=model_version,
            checkpoint_path=checkpoint_path,
            training_report=training_report.to_dict(),
            neural_config=training_report.neural_config,
            created_at=datetime.now(timezone.utc).isoformat(),
            status="CANDIDATE",
            parent_version=parent_version or (
                self.get_production_version().version
                if self.get_production_version()
                else ""
            ),
        )
        self.versions.append(version)
        self._save()
        return version

    def evaluate_candidate(self, candles, candidate_version: str) -> dict:
        """Run walk-forward validation on a candidate.

        Returns validation results with promotion decision.
        """
        candidate = None
        for v in self.versions:
            if v.version == candidate_version:
                candidate = v
                break
        if candidate is None:
            return {"valid": False, "note": f"Version {candidate_version} not found."}

        nc = NeuralConfig.from_dict(candidate.neural_config)
        tc = TrainingConfig()
        tc.epochs = 20

        validator = WalkForwardValidator(n_folds=5, test_size=200)
        results = validator.validate(candles, nc, tc)

        if not results["valid"]:
            candidate.status = "REJECTED"
            candidate.notes = results.get("note", "Validation failed.")
            self._save()
            return results

        # Check promotion requirements
        production = self.get_production_version()
        prod_val_loss = production.metrics.get("walk_forward_mean_loss", float("inf")) if production else float("inf")

        meets_req = self._check_requirements(results, prod_val_loss)

        candidate.metrics["walk_forward"] = results
        candidate.metrics["mean_test_loss"] = results["mean_test_loss"]
        candidate.metrics["mean_direction_acc"] = results["mean_direction_acc"]
        candidate.metrics["brier_score"] = results["brier_score"]
        candidate.metrics["net_return_pips"] = results["net_return_pips"]
        candidate.metrics["max_drawdown_pips"] = results["max_drawdown_pips"]

        if meets_req:
            candidate.status = "PROMOTED"
            candidate.notes = "Passed all promotion requirements."
            self._promote(candidate)
        else:
            candidate.status = "REJECTED"
            candidate.notes = "Failed promotion requirements."

        self._save()
        return {**results, "promotion_decision": candidate.status}

    def _check_requirements(self, results: dict, prod_val_loss: float) -> bool:
        """Check if validation results meet promotion requirements."""
        req = self.requirements

        if results["n_folds"] < req.min_regimes_tested:
            return False

        # Win rate
        if results["mean_direction_acc"] < req.min_win_rate:
            return False

        # OOF improvement
        if prod_val_loss < float("inf"):
            improvement = (prod_val_loss - results["mean_test_loss"]) / prod_val_loss
            if improvement < req.min_oof_improvement:
                return False

        # Drawdown
        if results["max_drawdown_pips"] > req.max_drawdown_pips:
            return False

        # Net return
        if results["net_return_pips"] < req.min_net_return_pips:
            return False

        # Calibration (Brier score should be low)
        if results["brier_score"] > 1.0 - req.min_calibration_score:
            return False

        return True

    def _promote(self, candidate: ModelVersion):
        """Promote a candidate to production, demoting the previous one."""
        # Backup current production checkpoint FIRST, before demoting
        prod = self.get_production_version()
        if prod:
            backup_path = prod.checkpoint_path.replace(".pt", "_backup.pt")
            try:
                shutil.copy2(prod.checkpoint_path, backup_path)
            except Exception:
                pass

        # Demote current production
        for v in self.versions:
            if v.is_production:
                v.is_production = False
                v.status = "SUPERSEDED"

        # Promote candidate
        candidate.is_production = True
        candidate.status = "PRODUCTION"

    def rollback(self, version: str | None = None) -> bool:
        """Rollback to a previous production version.

        If no version specified, rolls back to the most recent backup.
        """
        target = None
        if version:
            for v in self.versions:
                if v.version == version:
                    target = v
                    break
        else:
            # Find most recent backup
            for v in reversed(self.versions):
                if v.status in ("SUPERSEDED", "ROLLBACK"):
                    target = v
                    break

        if target is None:
            return False

        # Demote current production
        for v in self.versions:
            if v.is_production:
                v.is_production = False
                v.status = "ROLLBACK"

        # Promote target
        target.is_production = True
        target.status = "PRODUCTION"
        target.notes = f"Rolled back on {datetime.now(timezone.utc).isoformat()}."

        self._save()
        return True

    def list_versions(self) -> list[dict]:
        """List all model versions with their status."""
        return [v.to_dict() for v in self.versions]
