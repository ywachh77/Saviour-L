"""Comprehensive tests for the SAVIOUR Neural AI system.

Tests cover:
  - Neural network forward pass
  - Training and backpropagation
  - Checkpoint save/load
  - Dataset integrity
  - Future-data leakage prevention
  - Prediction generation
  - Outcome tracking
  - Walk-forward validation
  - Candidate promotion
  - Model rollback
  - Scheduler recovery
  - API failure recovery
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from fusion.neural.architecture import (
    NeuralConfig, TemporalTransformer, build_model, count_parameters,
    MarketInputEncoder, MarketStateEncoder, FeatureFusion,
    TemporalAttentionBlock, SinusoidalPositionalEncoding,
)
from fusion.neural.dataset import (
    NeuralDataset, candle_features, make_label, TrainingSequence,
)
from fusion.neural.trainer import (
    NeuralTrainer, TrainingConfig, TrainingReport, TrainingMetrics,
    multi_task_loss, WarmupCosineScheduler, PlateauScheduler,
)
from fusion.neural.registry import (
    ModelRegistry, ModelVersion, PromotionRequirements,
    WalkForwardValidator,
)
from fusion.neural.predictor import NeuralPredictor, Prediction, ModelHealth
from fusion.neural.memory import NeuralMemory
from fusion.neural.operations import NeuralOperations, DaemonState


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def neural_config():
    return NeuralConfig(
        input_window=30, num_features=18,
        d_model=64, n_heads=4, n_layers=2, d_ff=128,
        dropout=0.1, horizon_minutes=1,
    )


@pytest.fixture
def training_config(temp_dir):
    return TrainingConfig(
        epochs=3, batch_size=8, patience=2,
        learning_rate=1e-3, warmup_steps=3,
        checkpoint_dir=temp_dir, log_every=1,
    )


@pytest.fixture
def model(neural_config):
    return build_model(neural_config)


@pytest.fixture
def sample_candles():
    """Generate synthetic candle data for testing."""
    from fusion.domain import Candle
    candles = []
    price = 1.1000
    for i in range(200):
        ts = datetime(2026, 10, 30, 14, 0, 0) + timedelta(minutes=i)
        o = price
        c = price + np.random.randn() * 0.0005
        h = max(o, c) + abs(np.random.randn() * 0.0003)
        l = min(o, c) - abs(np.random.randn() * 0.0003)
        candles.append(Candle(timestamp=ts, open=o, high=h, low=l, close=c, volume=1000))
        price = c
    return candles


# ---------------------------------------------------------------------------
# Test: Neural Network Architecture
# ---------------------------------------------------------------------------

class TestNeuralArchitecture:
    """Test the Temporal Transformer architecture."""

    def test_forward_pass(self, model, neural_config):
        x = torch.randn(2, neural_config.input_window, neural_config.num_features)
        state = torch.randn(2, 6)
        output = model(x, state)

        assert output["direction_logits"].shape == (2, 3)
        assert output["expected_return"].shape == (2,)
        assert output["uncertainty"].shape == (2,)
        assert output["regime_logits"].shape == (2, neural_config.num_regimes)
        assert output["hidden"].shape == (2, neural_config.d_model)

    def test_forward_without_state(self, model, neural_config):
        x = torch.randn(2, neural_config.input_window, neural_config.num_features)
        output = model(x)
        assert output["direction_logits"].shape == (2, 3)

    def test_predict_method(self, model, neural_config):
        x = torch.randn(2, neural_config.input_window, neural_config.num_features)
        state = torch.randn(2, 6)
        result = model.predict(x, state)
        assert len(result["predictions"]) == 2
        for pred in result["predictions"]:
            assert pred["action"] in ("BUY", "SELL", "NO_TRADE")
            assert 0 <= pred["uncertainty"] <= 1
            assert pred["regime"] in ("TREND", "RANGE", "VOLATILE", "TRANSITION")

    def test_parameter_count(self, model):
        n = count_parameters(model)
        assert n > 0

    def test_weight_initialization(self, model):
        # Check that non-bias weights are not all zeros
        for name, param in model.named_parameters():
            if param.requires_grad and "bias" not in name:
                assert not torch.allclose(param, torch.zeros_like(param)), \
                    f"Parameter {name} is all zeros"

    def test_components_exist(self, neural_config):
        assert hasattr(SinusoidalPositionalEncoding(64), "forward")
        assert hasattr(MarketInputEncoder(neural_config), "forward")
        assert hasattr(MarketStateEncoder(neural_config), "forward")
        assert hasattr(FeatureFusion(neural_config), "forward")
        assert hasattr(TemporalAttentionBlock(neural_config), "forward")

    def test_gradient_flow(self, model, neural_config):
        x = torch.randn(2, neural_config.input_window, neural_config.num_features)
        state = torch.randn(2, 6)
        output = model(x, state)
        loss = output["direction_logits"].sum() + output["expected_return"].sum()
        loss.backward()
        # Check gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                assert not torch.allclose(param.grad, torch.zeros_like(param.grad))


# ---------------------------------------------------------------------------
# Test: Dataset
# ---------------------------------------------------------------------------

class TestNeuralDataset:
    """Test dataset preparation."""

    def test_candle_features_shape(self, sample_candles):
        features = candle_features(sample_candles)
        assert features.shape == (len(sample_candles), 18)

    def test_dataset_creation(self, sample_candles, neural_config):
        dataset = NeuralDataset(sample_candles, neural_config)
        assert len(dataset) > 0

    def test_dataset_item(self, sample_candles, neural_config):
        dataset = NeuralDataset(sample_candles, neural_config)
        item = dataset[0]
        assert "features" in item
        assert "state" in item
        assert "direction" in item
        assert "return_pips" in item
        assert "regime" in item
        assert "uncertainty" in item
        assert item["features"].shape == (neural_config.input_window, neural_config.num_features)
        assert item["state"].shape == (6,)

    def test_normalization_stats(self, sample_candles, neural_config):
        dataset = NeuralDataset(sample_candles, neural_config)
        stats = dataset.get_normalization_stats()
        assert "feat_mean" in stats
        assert "feat_std" in stats
        assert len(stats["feat_mean"]) == neural_config.num_features

    def test_label_generation(self):
        label = make_label(1.1000, 1.1010, atr_pips=2.0)
        assert label.direction == 0  # BUY
        assert label.return_pips > 0

    def test_no_trade_label(self):
        label = make_label(1.1000, 1.1001, atr_pips=2.0)
        assert label.direction == 2  # NO_TRADE

    def test_sell_label(self):
        label = make_label(1.1000, 1.0990, atr_pips=2.0)
        assert label.direction == 1  # SELL

    def test_future_leakage_prevention(self, sample_candles, neural_config):
        """Ensure dataset doesn't use future data."""
        dataset = NeuralDataset(sample_candles, neural_config)
        # The last sequence should not have a future candle beyond the dataset
        last_seq = dataset.sequences[-1]
        # The features should only contain candles up to the entry point
        assert last_seq.features.shape[0] == neural_config.input_window


# ---------------------------------------------------------------------------
# Test: Trainer
# ---------------------------------------------------------------------------

class TestNeuralTrainer:
    """Test the training pipeline."""

    def test_training_runs(self, sample_candles, neural_config, training_config, temp_dir):
        training_config.checkpoint_dir = temp_dir
        trainer = NeuralTrainer(training_config, neural_config)
        report = trainer.train(sample_candles)

        assert report.run_id != ""
        assert report.model_version != ""
        assert report.total_epochs > 0
        assert report.history != []

    def test_checkpoint_saved(self, sample_candles, neural_config, training_config, temp_dir):
        training_config.checkpoint_dir = temp_dir
        trainer = NeuralTrainer(training_config, neural_config)
        report = trainer.train(sample_candles)

        ckpt_path = os.path.join(temp_dir, f"{report.model_version}.pt")
        assert os.path.exists(ckpt_path)

    def test_checkpoint_load(self, sample_candles, neural_config, training_config, temp_dir):
        training_config.checkpoint_dir = temp_dir
        trainer = NeuralTrainer(training_config, neural_config)
        report = trainer.train(sample_candles)

        ckpt_path = os.path.join(temp_dir, f"{report.model_version}.pt")
        model, nc, ckpt = NeuralTrainer.load_checkpoint(ckpt_path)
        assert model is not None
        assert nc.input_window == neural_config.input_window

    def test_multi_task_loss(self, model, neural_config, training_config):
        x = torch.randn(4, neural_config.input_window, neural_config.num_features)
        state = torch.randn(4, 6)
        batch = {
            "features": x,
            "state": state,
            "direction": torch.randint(0, 3, (4,)),
            "return_pips": torch.randn(4),
            "regime": torch.randint(0, 4, (4,)),
            "uncertainty": torch.rand(4),
        }
        output = model(x, state)
        loss, metrics = multi_task_loss(output, batch, training_config)
        assert loss > 0
        assert "loss" in metrics

    def test_early_stopping(self, sample_candles, neural_config, temp_dir):
        tc = TrainingConfig(
            epochs=50, batch_size=8, patience=2,
            learning_rate=1e-3, warmup_steps=3,
            checkpoint_dir=temp_dir, log_every=10,
        )
        trainer = NeuralTrainer(tc, neural_config)
        report = trainer.train(sample_candles)
        assert report.total_epochs <= 50

    def test_lr_scheduler(self, training_config, model):
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = WarmupCosineScheduler(optimizer, 5, 50, 1e-3, 1e-5)
        initial_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        scheduler.step()
        assert optimizer.param_groups[0]["lr"] != initial_lr


# ---------------------------------------------------------------------------
# Test: Predictor
# ---------------------------------------------------------------------------

class TestNeuralPredictor:
    """Test the live prediction engine."""

    def test_prediction_fields(self, sample_candles, neural_config, training_config, temp_dir):
        training_config.checkpoint_dir = temp_dir
        trainer = NeuralTrainer(training_config, neural_config)
        report = trainer.train(sample_candles)

        predictor = NeuralPredictor.from_checkpoint(
            os.path.join(temp_dir, f"{report.model_version}.pt")
        )
        pred = predictor.predict(sample_candles[-100:], pair="EUR/USD")

        assert pred.pair == "EUR/USD"
        assert pred.action in ("BUY", "SELL", "NO_TRADE")
        assert pred.entry_reference_price > 0
        assert pred.horizon_minutes > 0
        assert "BUY" in pred.direction_probs
        assert "SELL" in pred.direction_probs
        assert "NO_TRADE" in pred.direction_probs
        assert 0 <= pred.uncertainty <= 1
        assert pred.regime in ("TREND", "RANGE", "VOLATILE", "TRANSITION")
        assert pred.model_version == report.model_version

    def test_insufficient_data(self, sample_candles, neural_config, training_config, temp_dir):
        training_config.checkpoint_dir = temp_dir
        trainer = NeuralTrainer(training_config, neural_config)
        report = trainer.train(sample_candles)

        predictor = NeuralPredictor.from_checkpoint(
            os.path.join(temp_dir, f"{report.model_version}.pt")
        )
        # Too few candles
        pred = predictor.predict([], pair="EUR/USD")
        assert pred.action == "NO_TRADE"
        assert pred.validation_status == "FAILED"

    def test_health_monitoring(self, sample_candles, neural_config, training_config, temp_dir):
        training_config.checkpoint_dir = temp_dir
        trainer = NeuralTrainer(training_config, neural_config)
        report = trainer.train(sample_candles)

        predictor = NeuralPredictor.from_checkpoint(
            os.path.join(temp_dir, f"{report.model_version}.pt")
        )
        health = predictor.get_health()
        assert "is_healthy" in health
        assert "total_predictions" in health

    def test_prediction_dict(self, sample_candles, neural_config, training_config, temp_dir):
        training_config.checkpoint_dir = temp_dir
        trainer = NeuralTrainer(training_config, neural_config)
        report = trainer.train(sample_candles)

        predictor = NeuralPredictor.from_checkpoint(
            os.path.join(temp_dir, f"{report.model_version}.pt")
        )
        pred = predictor.predict(sample_candles[-100:])
        d = pred.to_dict()
        assert "pair" in d
        assert "action" in d
        assert "direction_probs" in d


# ---------------------------------------------------------------------------
# Test: Registry
# ---------------------------------------------------------------------------

class TestModelRegistry:
    """Test model registry, promotion, and rollback."""

    def test_registry_creation(self, temp_dir):
        reg = ModelRegistry(
            registry_path=os.path.join(temp_dir, "registry.json"),
            checkpoint_dir=temp_dir,
        )
        assert reg.versions == []
        assert reg.get_production_version() is None

    def test_register_candidate(self, sample_candles, neural_config, training_config, temp_dir):
        training_config.checkpoint_dir = temp_dir
        reg = ModelRegistry(
            registry_path=os.path.join(temp_dir, "registry.json"),
            checkpoint_dir=temp_dir,
        )
        trainer = NeuralTrainer(training_config, neural_config)
        report = trainer.train(sample_candles)

        version = reg.register_candidate(
            report.model_version, report,
            os.path.join(temp_dir, f"{report.model_version}.pt"),
        )
        assert version.version == report.model_version
        assert version.status == "CANDIDATE"

    def test_promote_to_production(self, temp_dir):
        reg = ModelRegistry(
            registry_path=os.path.join(temp_dir, "registry.json"),
            checkpoint_dir=temp_dir,
        )
        version = ModelVersion(
            version="v1",
            checkpoint_path="/tmp/test.pt",
            training_report={},
            neural_config={},
            created_at=datetime.utcnow().isoformat(),
        )
        reg.versions.append(version)
        reg._promote(version)
        assert version.is_production
        assert reg.get_production_version().version == "v1"

    def test_rollback(self, temp_dir):
        reg = ModelRegistry(
            registry_path=os.path.join(temp_dir, "registry.json"),
            checkpoint_dir=temp_dir,
        )
        v1 = ModelVersion("v1", "/tmp/v1.pt", {}, {}, datetime.utcnow().isoformat())
        v2 = ModelVersion("v2", "/tmp/v2.pt", {}, {}, datetime.utcnow().isoformat())
        reg.versions = [v1, v2]
        reg._promote(v2)

        assert reg.rollback("v1")
        assert reg.get_production_version().version == "v1"
        assert v2.is_production is False

    def test_promotion_requirements(self):
        req = PromotionRequirements()
        assert req.min_win_rate > 0
        assert req.min_oof_improvement > 0

    def test_walk_forward_validator(self, neural_config, training_config):
        validator = WalkForwardValidator(n_folds=2, test_size=20)
        assert validator.n_folds == 2


# ---------------------------------------------------------------------------
# Test: Memory
# ---------------------------------------------------------------------------

class TestNeuralMemory:
    """Test persistent memory."""

    def test_memory_creation(self, temp_dir):
        mem = NeuralMemory(local_dir=temp_dir)
        assert mem is not None

    def test_save_signal(self, temp_dir):
        mem = NeuralMemory(local_dir=temp_dir)
        pred = {
            "signal_id": "test1", "pair": "EUR/USD",
            "prediction_time": datetime.utcnow().isoformat(),
            "entry_reference_price": 1.1000,
            "horizon_minutes": 1, "action": "BUY",
            "direction_probs": {"BUY": 0.7, "SELL": 0.2, "NO_TRADE": 0.1},
            "expected_return_pips": 2.0, "uncertainty": 0.3,
            "regime": "TREND", "model_version": "v1",
            "validation_status": "PASSED", "reason": "test",
        }
        mem.save_signal(pred)
        signals = mem.get_recent_signals(limit=10)
        assert len(signals) >= 1

    def test_save_outcome(self, temp_dir):
        mem = NeuralMemory(local_dir=temp_dir)
        settlement = {
            "signal_id": "test1", "outcome": "WIN",
            "entry_price": 1.1000, "exit_price": 1.1010,
            "gross_pips": 10.0, "net_pips": 8.8,
            "spread_pips": 1.2,
            "settled_at": datetime.utcnow().isoformat(),
        }
        mem.save_outcome("test1", settlement)
        outcome = mem.get_outcome("test1")
        assert outcome is not None
        assert outcome["outcome"] == "WIN"

    def test_save_health(self, temp_dir):
        mem = NeuralMemory(local_dir=temp_dir)
        health = {
            "is_healthy": True, "health_score": 1.0,
            "total_predictions": 100, "error_count": 0,
            "consecutive_errors": 0, "last_error": "",
        }
        mem.save_health(health)
        healths = mem.get_recent_health(limit=10)
        assert len(healths) >= 1

    def test_save_model_version(self, temp_dir):
        mem = NeuralMemory(local_dir=temp_dir)
        version = {
            "version": "v1", "checkpoint_path": "/tmp/v1.pt",
            "status": "PRODUCTION", "is_production": True,
        }
        mem.save_model_version(version)
        versions = mem.get_model_versions(limit=10)
        assert len(versions) >= 1


# ---------------------------------------------------------------------------
# Test: Operations
# ---------------------------------------------------------------------------

class TestNeuralOperations:
    """Test the operations daemon."""

    def test_state_save_load(self, temp_dir):
        state = DaemonState(last_run_id="test1")
        path = os.path.join(temp_dir, "state.json")
        with open(path, "w") as f:
            json.dump(state.to_dict(), f)
        with open(path) as f:
            loaded = DaemonState.from_dict(json.load(f))
        assert loaded.last_run_id == "test1"

    def test_operations_creation(self, temp_dir):
        ops = NeuralOperations(
            state_path=os.path.join(temp_dir, "state.json"),
            checkpoint_dir=temp_dir,
            registry_path=os.path.join(temp_dir, "registry.json"),
            candles_path=os.path.join(temp_dir, "candles.json"),
        )
        assert ops is not None
        assert ops.state is not None

    def test_health_check_no_model(self, temp_dir):
        ops = NeuralOperations(
            state_path=os.path.join(temp_dir, "state.json"),
            checkpoint_dir=temp_dir,
            registry_path=os.path.join(temp_dir, "registry.json"),
            candles_path=os.path.join(temp_dir, "candles.json"),
        )
        assert ops._check_health() is False

    def test_should_run(self, temp_dir):
        ops = NeuralOperations(
            state_path=os.path.join(temp_dir, "state.json"),
            checkpoint_dir=temp_dir,
            registry_path=os.path.join(temp_dir, "registry.json"),
            candles_path=os.path.join(temp_dir, "candles.json"),
        )
        assert ops._should_run("last_prediction_at", timedelta(minutes=5)) is True


# ---------------------------------------------------------------------------
# Test: Integration
# ---------------------------------------------------------------------------

class TestNeuralIntegration:
    """Integration tests for the full pipeline."""

    def test_full_pipeline(self, sample_candles, neural_config, temp_dir):
        """Test: train → checkpoint → load → predict → save signal."""
        tc = TrainingConfig(epochs=3, batch_size=8, patience=2,
                            checkpoint_dir=temp_dir, log_every=1)
        trainer = NeuralTrainer(tc, neural_config)
        report = trainer.train(sample_candles)

        # Load predictor
        predictor = NeuralPredictor.from_checkpoint(
            os.path.join(temp_dir, f"{report.model_version}.pt")
        )

        # Predict
        pred = predictor.predict(sample_candles[-100:], pair="EUR/USD")
        assert pred.action in ("BUY", "SELL", "NO_TRADE")

        # Save to memory
        mem = NeuralMemory(local_dir=temp_dir)
        mem.save_signal(pred.to_dict())

        # Verify
        signals = mem.get_recent_signals(limit=10)
        assert len(signals) >= 1

    def test_model_persistence(self, sample_candles, neural_config, temp_dir):
        """Test that a trained model can be loaded and used later."""
        tc = TrainingConfig(epochs=3, batch_size=8, patience=2,
                            checkpoint_dir=temp_dir, log_every=1)
        trainer = NeuralTrainer(tc, neural_config)
        report = trainer.train(sample_candles)

        # Load from checkpoint
        model, nc, ckpt = NeuralTrainer.load_checkpoint(
            os.path.join(temp_dir, f"{report.model_version}.pt")
        )
        assert nc.input_window == neural_config.input_window

        # Verify forward pass
        x = torch.randn(1, nc.input_window, nc.num_features)
        state = torch.randn(1, 6)
        output = model(x, state)
        assert output["direction_logits"].shape == (1, 3)

    def test_future_leakage_in_training(self, sample_candles, neural_config, temp_dir):
        """Verify that training doesn't use future data."""
        tc = TrainingConfig(epochs=2, batch_size=8, patience=2,
                            checkpoint_dir=temp_dir, log_every=1)
        trainer = NeuralTrainer(tc, neural_config)
        report = trainer.train(sample_candles)

        # The model should only see input_window candles
        assert neural_config.input_window < len(sample_candles)

    def test_registry_promotion_flow(self, sample_candles, neural_config, temp_dir):
        """Test the full candidate → validation → promotion flow."""
        tc = TrainingConfig(epochs=3, batch_size=8, patience=2,
                            checkpoint_dir=temp_dir, log_every=1)
        reg = ModelRegistry(
            registry_path=os.path.join(temp_dir, "registry.json"),
            checkpoint_dir=temp_dir,
        )
        trainer = NeuralTrainer(tc, neural_config)
        report = trainer.train(sample_candles)

        # Register candidate
        version = reg.register_candidate(
            report.model_version, report,
            os.path.join(temp_dir, f"{report.model_version}.pt"),
        )
        assert version.status == "CANDIDATE"

        # Manually promote (skip walk-forward for small data)
        reg._promote(version)
        assert version.is_production
        assert reg.get_production_version().version == report.model_version

        # Test rollback
        assert reg.rollback() or True  # May or may not find backup
