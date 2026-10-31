"""Autonomous Neural Training Engine.

Trains the Temporal Transformer from randomly initialized weights using:
  - AdamW optimization with learning-rate scheduling
  - Multi-task loss (direction CE + return MSE + uncertainty BCE + regime CE)
  - Early stopping on validation loss
  - Checkpoint saving
  - Training/validation metrics recording
  - Incremental dataset expansion
  - Walk-forward validation splits

No manual intervention required for routine training.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .architecture import NeuralConfig, TemporalTransformer, build_model, count_parameters
from .dataset import NeuralDataset


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """Hyperparameters for the training pipeline."""
    # Optimization
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    lr_scheduler: str = "cosine"  # cosine | plateau | step
    lr_min: float = 1e-5
    warmup_steps: int = 100
    grad_clip: float = 1.0

    # Early stopping
    patience: int = 8
    min_delta: float = 0.0005

    # Data
    train_fraction: float = 0.8
    val_fraction: float = 0.1
    test_fraction: float = 0.1
    shuffle_train: bool = True

    # Loss weights
    w_direction: float = 1.0
    w_return: float = 0.3
    w_uncertainty: float = 0.2
    w_regime: float = 0.2

    # Checkpointing
    checkpoint_dir: str = "neural/checkpoints"
    save_best_only: bool = True

    # Reproducibility
    seed: int = 42

    # Misc
    num_workers: int = 0
    log_every: int = 10
    device: str = "cpu"  # cpu | cuda

    def to_dict(self) -> dict:
        return {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "lr_scheduler": self.lr_scheduler,
            "lr_min": self.lr_min,
            "warmup_steps": self.warmup_steps,
            "grad_clip": self.grad_clip,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "train_fraction": self.train_fraction,
            "val_fraction": self.val_fraction,
            "test_fraction": self.test_fraction,
            "shuffle_train": self.shuffle_train,
            "w_direction": self.w_direction,
            "w_return": self.w_return,
            "w_uncertainty": self.w_uncertainty,
            "w_regime": self.w_regime,
            "checkpoint_dir": self.checkpoint_dir,
            "save_best_only": self.save_best_only,
            "seed": self.seed,
            "num_workers": self.num_workers,
            "log_every": self.log_every,
            "device": self.device,
        }


# ---------------------------------------------------------------------------
# Multi-task loss
# ---------------------------------------------------------------------------

def multi_task_loss(
    output: dict,
    batch: dict,
    config: TrainingConfig,
) -> tuple[torch.Tensor, dict]:
    """Compute the multi-task loss.

    Loss = w_dir * CE(direction) + w_ret * MSE(return)
         + w_unc * BCE(uncertainty) + w_reg * CE(regime)
    """
    direction_logits = output["direction_logits"]
    expected_return = output["expected_return"]
    uncertainty = output["uncertainty"]
    regime_logits = output["regime_logits"]

    target_dir = batch["direction"]
    target_ret = batch["return_pips"]
    target_unc = batch["uncertainty"]
    target_regime = batch["regime"]

    # Direction cross-entropy
    loss_dir = F.cross_entropy(direction_logits, target_dir)

    # Return MSE
    loss_ret = F.mse_loss(expected_return, target_ret)

    # Uncertainty BCE
    loss_unc = F.binary_cross_entropy(uncertainty, target_unc)

    # Regime cross-entropy
    loss_reg = F.cross_entropy(regime_logits, target_regime)

    total = (
        config.w_direction * loss_dir
        + config.w_return * loss_ret
        + config.w_uncertainty * loss_unc
        + config.w_regime * loss_reg
    )

    metrics = {
        "loss": total.item(),
        "loss_direction": loss_dir.item(),
        "loss_return": loss_ret.item(),
        "loss_uncertainty": loss_unc.item(),
        "loss_regime": loss_reg.item(),
    }

    return total, metrics


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class TrainingMetrics:
    """Records training and validation metrics."""
    epoch: int = 0
    train_loss: float = 0.0
    train_direction_acc: float = 0.0
    train_return_mae: float = 0.0
    val_loss: float = 0.0
    val_direction_acc: float = 0.0
    val_return_mae: float = 0.0
    lr: float = 0.0
    time_seconds: float = 0.0
    is_best: bool = False
    stopped_early: bool = False

    def to_dict(self) -> dict:
        return {
            "epoch": self.epoch,
            "train_loss": self.train_loss,
            "train_direction_acc": self.train_direction_acc,
            "train_return_mae": self.train_return_mae,
            "val_loss": self.val_loss,
            "val_direction_acc": self.val_direction_acc,
            "val_return_mae": self.val_return_mae,
            "lr": self.lr,
            "time_seconds": self.time_seconds,
            "is_best": self.is_best,
            "stopped_early": self.stopped_early,
        }


@dataclass
class TrainingReport:
    """Complete training report."""
    run_id: str
    model_version: str
    started_at: str
    finished_at: str = ""
    total_epochs: int = 0
    best_epoch: int = 0
    best_val_loss: float = float("inf")
    best_val_direction_acc: float = 0.0
    final_metrics: dict = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)
    config: dict = field(default_factory=dict)
    neural_config: dict = field(default_factory=dict)
    test_metrics: dict = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "model_version": self.model_version,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "total_epochs": self.total_epochs,
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "best_val_direction_acc": self.best_val_direction_acc,
            "final_metrics": self.final_metrics,
            "history": self.history,
            "config": self.config,
            "neural_config": self.neural_config,
            "test_metrics": self.test_metrics,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Learning rate schedulers
# ---------------------------------------------------------------------------

class WarmupCosineScheduler:
    """Linear warmup + cosine decay scheduler."""

    def __init__(self, optimizer, warmup_steps: int, total_steps: int,
                 initial_lr: float, min_lr: float):
        self.optimizer = optimizer
        self.warmup_steps = max(1, warmup_steps)
        self.total_steps = max(1, total_steps)
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.current_step = 0

    def step(self):
        self.current_step += 1
        lr = self._get_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _get_lr(self) -> float:
        if self.current_step <= self.warmup_steps:
            # Start at a small fraction to avoid LR=0 on first step
            if self.current_step == 0:
                return self.initial_lr * 0.1
            return self.initial_lr * (self.current_step / self.warmup_steps)
        progress = (self.current_step - self.warmup_steps) / max(
            1, self.total_steps - self.warmup_steps
        )
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return self.min_lr + (self.initial_lr - self.min_lr) * cosine

    def get_lr(self) -> float:
        return self._get_lr()


class PlateauScheduler:
    """ReduceLROnPlateau-style scheduler."""

    def __init__(self, optimizer, factor: float = 0.5, patience: int = 3,
                 min_lr: float = 1e-5):
        self.optimizer = optimizer
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.best_val = float("inf")
        self.wait = 0

    def step(self, val_loss: float):
        if val_loss < self.best_val:
            self.best_val = val_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                for group in self.optimizer.param_groups:
                    new_lr = max(self.min_lr, group["lr"] * self.factor)
                    group["lr"] = new_lr
                self.wait = 0

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class NeuralTrainer:
    """Autonomous training pipeline for the Temporal Transformer.

    Usage:
        trainer = NeuralTrainer(config, neural_config)
        report = trainer.train(candles)
    """

    def __init__(
        self,
        config: TrainingConfig | None = None,
        neural_config: NeuralConfig | None = None,
    ):
        self.config = config or TrainingConfig()
        self.neural_config = neural_config or NeuralConfig()
        self.device = torch.device(self.config.device)
        self._set_seed(self.config.seed)

        # Checkpoint dir
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def _set_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def train(self, candles: Sequence) -> TrainingReport:
        """Train the model from scratch on candle data."""
        cfg = self.config
        nc = self.neural_config

        # 1. Build dataset
        dataset = NeuralDataset(candles, nc, spread_pips=1.2)
        if len(dataset) == 0:
            report = TrainingReport(
                run_id=uuid.uuid4().hex,
                model_version="v0",
                started_at=datetime.now(timezone.utc).isoformat(),
                notes="Not enough candles to build training sequences.",
            )
            return report

        # 2. Split into train/val/test (chronological, no shuffle for val/test)
        n = len(dataset)
        train_end = int(n * cfg.train_fraction)
        val_end = int(n * (cfg.train_fraction + cfg.val_fraction))

        train_indices = list(range(train_end))
        val_indices = list(range(train_end, val_end))
        test_indices = list(range(val_end, n))

        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)
        test_dataset = torch.utils.data.Subset(dataset, test_indices)

        train_loader = DataLoader(
            train_dataset, batch_size=cfg.batch_size, shuffle=cfg.shuffle_train,
            num_workers=cfg.num_workers,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.num_workers,
        )
        test_loader = DataLoader(
            test_dataset, batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.num_workers,
        )

        # 3. Build model from scratch
        model = build_model(nc).to(self.device)
        n_params = count_parameters(model)

        # 4. Optimizer (AdamW)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

        # 5. LR scheduler (scale warmup to dataset size)
        total_steps = len(train_loader) * cfg.epochs
        warmup = min(cfg.warmup_steps, max(1, total_steps // 5))
        if cfg.lr_scheduler == "cosine":
            scheduler = WarmupCosineScheduler(
                optimizer, warmup, total_steps,
                cfg.learning_rate, cfg.lr_min,
            )
        elif cfg.lr_scheduler == "plateau":
            scheduler = PlateauScheduler(optimizer, min_lr=cfg.lr_min)
        else:
            scheduler = None

        # 6. Training loop
        report = TrainingReport(
            run_id=uuid.uuid4().hex,
            model_version=f"v1-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
            started_at=datetime.now(timezone.utc).isoformat(),
            config=cfg.to_dict(),
            neural_config=nc.to_dict(),
        )

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(cfg.epochs):
            epoch_start = time.time()

            # Train
            model.train()
            train_losses = []
            train_correct = 0
            train_total = 0
            train_ret_errors = []

            for batch in train_loader:
                batch = self._to_device(batch)
                optimizer.zero_grad()
                output = model(batch["features"], batch["state"])
                loss, metrics = multi_task_loss(output, batch, cfg)
                loss.backward()

                if cfg.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

                optimizer.step()
                if scheduler:
                    scheduler.step()

                train_losses.append(metrics["loss"])

                # Track direction accuracy
                pred_dir = output["direction_logits"].argmax(dim=-1)
                train_correct += (pred_dir == batch["direction"]).sum().item()
                train_total += batch["direction"].size(0)

                # Track return MAE
                train_ret_errors.append(
                    (output["expected_return"] - batch["return_pips"]).abs().mean().item()
                )

            # Validate
            val_metrics = self._evaluate(model, val_loader, cfg)

            # Scheduler with plateau
            if isinstance(scheduler, PlateauScheduler):
                scheduler.step(val_metrics["loss"])

            # Compute epoch metrics
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - epoch_start
            epoch_metrics = TrainingMetrics(
                epoch=epoch + 1,
                train_loss=np.mean(train_losses),
                train_direction_acc=train_correct / max(1, train_total),
                train_return_mae=np.mean(train_ret_errors),
                val_loss=val_metrics["loss"],
                val_direction_acc=val_metrics["direction_acc"],
                val_return_mae=val_metrics["return_mae"],
                lr=lr,
                time_seconds=elapsed,
            )

            # Check best
            is_best = val_metrics["loss"] < best_val_loss - cfg.min_delta
            if is_best:
                best_val_loss = val_metrics["loss"]
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epoch_metrics.is_best = True
                patience_counter = 0
                # Save checkpoint
                self._save_checkpoint(model, nc, cfg, report, epoch_metrics,
                                      norm_stats=dataset.get_normalization_stats())
            else:
                patience_counter += 1

            report.history.append(epoch_metrics.to_dict())

            # Log
            if (epoch + 1) % cfg.log_every == 0 or is_best:
                print(f"  Epoch {epoch+1:3d} | "
                      f"train_loss={epoch_metrics.train_loss:.4f} | "
                      f"val_loss={epoch_metrics.val_loss:.4f} | "
                      f"val_dir_acc={epoch_metrics.val_direction_acc:.1%} | "
                      f"lr={lr:.2e} | {elapsed:.1f}s")

            # Early stopping
            if patience_counter >= cfg.patience:
                report.notes = f"Early stopped at epoch {epoch+1} (patience={cfg.patience})"
                break

        # 7. Restore best model
        if best_state is not None:
            model.load_state_dict(best_state)
            report.best_epoch = max(1, len(report.history) - patience_counter)
            report.best_val_loss = best_val_loss
            # Find best direction accuracy
            best_acc = max(
                (m["val_direction_acc"] for m in report.history), default=0.0
            )
            report.best_val_direction_acc = best_acc

        # 8. Test evaluation
        test_metrics = self._evaluate(model, test_loader, cfg)
        report.test_metrics = test_metrics
        report.total_epochs = len(report.history)
        report.final_metrics = {
            "best_val_loss": best_val_loss,
            "test_loss": test_metrics["loss"],
            "test_direction_acc": test_metrics["direction_acc"],
            "test_return_mae": test_metrics["return_mae"],
        }
        report.finished_at = datetime.now(timezone.utc).isoformat()

        # Save final checkpoint
        self._save_checkpoint(model, nc, cfg, report, None,
                              norm_stats=dataset.get_normalization_stats(),
                              is_final=True)

        # Save report
        self._save_report(report)

        return report

    def _evaluate(self, model, loader, cfg: TrainingConfig) -> dict:
        """Evaluate model on a dataset."""
        if model is None or loader is None:
            return {"loss": float("inf"), "direction_acc": 0.0, "return_mae": 0.0}
        model.eval()
        losses = []
        correct = 0
        total = 0
        ret_errors = []

        with torch.no_grad():
            for batch in loader:
                batch = self._to_device(batch)
                output = model(batch["features"], batch["state"])
                loss, _ = multi_task_loss(output, batch, cfg)
                losses.append(loss.item())

                pred_dir = output["direction_logits"].argmax(dim=-1)
                correct += (pred_dir == batch["direction"]).sum().item()
                total += batch["direction"].size(0)

                ret_errors.append(
                    (output["expected_return"] - batch["return_pips"]).abs().mean().item()
                )

        return {
            "loss": np.mean(losses),
            "direction_acc": correct / max(1, total),
            "return_mae": np.mean(ret_errors),
        }

    def _to_device(self, batch: dict) -> dict:
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(self.device)
        return batch

    def _save_checkpoint(
        self, model, neural_config, training_config, report, epoch_metrics,
        norm_stats: dict, is_final: bool = False,
    ):
        """Save model checkpoint."""
        ckpt_path = Path(self.config.checkpoint_dir) / f"{report.model_version}.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "neural_config": neural_config.to_dict(),
            "training_config": training_config.to_dict(),
            "norm_stats": norm_stats,
            "run_id": report.run_id,
            "model_version": report.model_version,
            "epoch": epoch_metrics.epoch if epoch_metrics else 0,
            "val_loss": epoch_metrics.val_loss if epoch_metrics else 0.0,
            "is_final": is_final,
        }, ckpt_path)

    def _save_report(self, report: TrainingReport):
        """Save training report as JSON."""
        report_path = Path(self.config.checkpoint_dir) / f"{report.model_version}_report.json"
        with open(report_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2, default=str)

    @staticmethod
    def load_checkpoint(checkpoint_path: str, device: str = "cpu") -> tuple:
        """Load a model from checkpoint."""
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        nc = NeuralConfig.from_dict(ckpt["neural_config"])
        model = build_model(nc).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model, nc, ckpt
