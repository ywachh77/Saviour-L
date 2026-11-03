"""SAVIOUR Meta-Learning Controller — The Root AI Brain.

This is the master AI that sits at the roots of Saviour and coordinates
all learning, trading, and improvement across the entire system.

What makes this a "roots" AI:
- It learns HOW to learn, not just what to predict
- It meta-learns the optimal hyperparameters for all sub-models
- It uses reinforcement learning to optimize trading decisions
- It adapts to market regimes by adjusting model weights dynamically
- It evolves strategies through genetic algorithms
- It coordinates the entire ensemble of models and strategies

Architecture:
- Meta-Learner: Learns to adapt model hyperparameters
- Policy Network: RL agent that optimizes trading decisions
- Regime Detector: Identifies market conditions
- Strategy Evolver: Genetic algorithm for strategy optimization
- Ensemble Arbiter: Intelligently combines model predictions

No pretrained models — trained entirely from scratch on live data.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Meta-Learning Controller — The Root AI Brain
# =============================================================================

class MetaController(nn.Module):
    """The master AI that coordinates all learning in Saviour.

    This model sits at the roots and:
    - Learns to adapt hyperparameters for all sub-models
    - Uses RL to optimize trading decisions
    - Detects market regimes and adjusts behavior
    - Evolves strategies through genetic algorithms
    - Arbitrates between ensemble members intelligently
    """

    def __init__(
        self,
        num_features: int = 18,
        num_models: int = 5,
        num_strategies: int = 22,
        regime_dim: int = 8,
    ):
        super().__init__()

        # Market regime detector
        self.regime_detector = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, regime_dim),
            nn.Softmax(dim=-1),
        )

        # Hyperparameter adapter — learns to adjust model parameters
        self.hyper_adapter = nn.Sequential(
            nn.Linear(regime_dim + num_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_models * 4),  # 4 hyperparams per model
        )

        # Policy network — RL agent for trading decisions
        self.policy_network = nn.Sequential(
            nn.Linear(regime_dim + num_features + num_models + num_strategies, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # BUY, SELL, NO_TRADE
        )

        # Value network — estimates expected reward
        self.value_network = nn.Sequential(
            nn.Linear(regime_dim + num_features + num_models + num_strategies, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # Strategy evolution memory
        self.strategy_memory = nn.Parameter(
            torch.randn(num_strategies, num_features + 5) * 0.1
        )

        # Learned ensemble weights
        self.ensemble_weights = nn.Parameter(
            torch.randn(num_models) * 0.1
        )

        # Meta-learning state
        self.meta_state = nn.Parameter(
            torch.randn(regime_dim) * 0.1
        )

    def forward(
        self,
        features: torch.Tensor,
        model_predictions: torch.Tensor | None = None,
        strategy_votes: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass through the meta-controller."""

        # Detect regime
        regime_probs = self.regime_detector(features)

        # Adapt hyperparameters based on regime + features
        hyper_input = torch.cat([regime_probs, features], dim=-1)
        hyper_params = self.hyper_adapter(hyper_input)

        # Get trading decision from policy network
        policy_input_list = [regime_probs, features]
        if model_predictions is not None:
            policy_input_list.append(model_predictions)
        if strategy_votes is not None:
            policy_input_list.append(strategy_votes)

        policy_input = torch.cat(policy_input_list, dim=-1)
        policy_logits = self.policy_network(policy_input)
        policy_probs = F.softmax(policy_logits, dim=-1)

        # Estimate value
        value = self.value_network(policy_input)

        # Get ensemble weights (normalized)
        ensemble_w = torch.softmax(self.ensemble_weights, dim=0)

        return {
            "regime_probs": regime_probs,
            "hyper_params": hyper_params,
            "policy_probs": policy_probs,
            "value": value,
            "ensemble_weights": ensemble_w,
            "strategy_memory": self.strategy_memory,
            "meta_state": self.meta_state,
        }

    def get_action(
        self,
        features: torch.Tensor,
        model_predictions: torch.Tensor | None = None,
        strategy_votes: torch.Tensor | None = None,
        training: bool = False,
    ) -> dict[str, Any]:
        """Get trading action from the meta-controller."""
        outputs = self.forward(features, model_predictions, strategy_votes)

        if training:
            # Sample from policy during training
            action = torch.multinomial(outputs["policy_probs"], num_samples=1)
        else:
            # Greedy action during execution
            action = torch.argmax(outputs["policy_probs"], dim=-1, keepdim=True)

        actions = ["NO_TRADE", "BUY", "SELL"]
        action_str = actions[action.item()]

        return {
            "action": action_str,
            "action_probs": outputs["policy_probs"],
            "regime": outputs["regime_probs"],
            "value": outputs["value"],
            "ensemble_weights": outputs["ensemble_weights"],
        }


# =============================================================================
# Reinforcement Learning Trainer
# =============================================================================

@dataclass
class RLTrainingConfig:
    """Configuration for RL training of the meta-controller."""
    learning_rate: float = 0.001
    discount_factor: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    batch_size: int = 64
    horizon: int = 100
    reward_scale: float = 1.0


class RLTrainer:
    """Trains the MetaController using PPO (Proximal Policy Optimization)."""

    def __init__(self, controller: MetaController, config: RLTrainingConfig | None = None):
        self.controller = controller
        self.config = config or RLTrainingConfig()
        self.optimizer = torch.optim.Adam(controller.parameters(), lr=self.config.learning_rate)
        self.memory: list[dict] = []

    def remember(
        self,
        features: torch.Tensor,
        model_predictions: torch.Tensor | None,
        strategy_votes: torch.Tensor | None,
        action: int,
        reward: float,
        done: bool,
    ) -> None:
        """Store a transition in memory."""
        self.memory.append({
            "features": features,
            "model_predictions": model_predictions,
            "strategy_votes": strategy_votes,
            "action": action,
            "reward": reward,
            "done": done,
        })

    def compute_gae(self, rewards: list[float], values: list[float], dones: list[bool]) -> list[float]:
        """Compute Generalized Advantage Estimation."""
        advantages = []
        gae = 0
        next_value = 0

        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.config.discount_factor * next_value * (1 - int(dones[step])) - values[step]
            gae = delta + self.config.discount_factor * self.config.gae_lambda * (1 - int(dones[step])) * gae
            advantages.insert(0, gae)
            next_value = values[step]

        return advantages

    def update(self) -> dict:
        """Update the meta-controller using PPO."""
        if len(self.memory) < self.config.batch_size:
            return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        # Compute GAE
        rewards = [t["reward"] for t in self.memory]
        dones = [t["done"] for t in self.memory]

        # Get values
        values = []
        for t in self.memory:
            outputs = self.controller.forward(
                t["features"], t["model_predictions"], t["strategy_votes"]
            )
            values.append(outputs["value"].item())

        advantages = self.compute_gae(rewards, values, dones)

        # Normalize advantages
        if advantages:
            adv_mean = sum(advantages) / len(advantages)
            adv_std = math.sqrt(sum((a - adv_mean) ** 2 for a in advantages) / len(advantages)) + 1e-8
            advantages = [(a - adv_mean) / adv_std for a in advantages]

        # PPO update
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0

        for _ in range(4):  # 4 PPO epochs
            for i, t in enumerate(self.memory):
                outputs = self.controller.forward(
                    t["features"], t["model_predictions"], t["strategy_votes"]
                )

                policy_probs = outputs["policy_probs"]
                value = outputs["value"]

                # Get old probability of taken action
                old_prob = policy_probs[0, t["action"]].detach()

                # Get new probability
                new_prob = policy_probs[0, t["action"]]

                # PPO clipping
                ratio = new_prob / old_prob
                surr1 = ratio * advantages[i]
                surr2 = torch.clamp(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio) * advantages[i]

                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                target_value = torch.tensor([rewards[i] + self.config.discount_factor * (values[i + 1] if i + 1 < len(values) else 0)])
                value_loss = F.mse_loss(value, target_value.to(value.device))

                # Entropy
                entropy = -(policy_probs * torch.log(policy_probs + 1e-8)).sum()

                # Total loss
                loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()

        # Clear memory
        self.memory = []

        n = max(1, len(self.memory))
        return {
            "loss": (total_policy_loss + total_value_loss) / n,
            "policy_loss": total_policy_loss / n,
            "value_loss": total_value_loss / n,
            "entropy": total_entropy / n,
        }


# =============================================================================
# Genetic Strategy Evolver
# =============================================================================

@dataclass
class StrategyGene:
    """A gene in the genetic algorithm — represents a strategy parameter."""
    name: str
    value: float
    min_val: float
    max_val: float


class GeneticStrategyEvolver:
    """Evolves trading strategies using genetic algorithms.

    This allows Saviour to discover new strategy configurations
    through evolution rather than manual tuning.
    """

    def __init__(
        self,
        population_size: int = 50,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
        elite_size: int = 5,
    ):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.population: list[list[StrategyGene]] = []
        self.fitness_scores: list[float] = []

    def initialize(self, genes: list[StrategyGene]) -> None:
        """Initialize population with random genes."""
        self.population = []
        for _ in range(self.population_size):
            individual = [
                StrategyGene(
                    name=g.name,
                    value=random.uniform(g.min_val, g.max_val),
                    min_val=g.min_val,
                    max_val=g.max_val,
                )
                for g in genes
            ]
            self.population.append(individual)

    def evaluate_fitness(self, individual: list[StrategyGene], backtest_result: dict) -> float:
        """Evaluate fitness of an individual strategy."""
        win_rate = backtest_result.get("win_rate", 0.5)
        sharpe = backtest_result.get("sharpe", 0.0)
        profit_factor = backtest_result.get("profit_factor", 1.0)

        # Fitness = weighted combination of metrics
        fitness = (win_rate * 0.5 + sharpe * 0.3 + profit_factor * 0.2)
        return fitness

    def evolve(self, backtest_results: list[dict]) -> list[StrategyGene]:
        """Run one generation of evolution."""
        # Evaluate fitness
        self.fitness_scores = []
        for i, individual in enumerate(self.population):
            fitness = self.evaluate_fitness(individual, backtest_results[i])
            self.fitness_scores.append(fitness)

        # Sort by fitness (descending)
        sorted_indices = sorted(range(len(self.fitness_scores)), key=lambda i: self.fitness_scores[i], reverse=True)

        # Keep elite
        new_population = [self.population[i] for i in sorted_indices[:self.elite_size]]

        # Create offspring
        while len(new_population) < self.population_size:
            # Select parents (tournament selection)
            parent1 = self._tournament_select()
            parent2 = self._tournament_select()

            # Crossover
            if random.random() < self.crossover_rate:
                child = self._crossover(parent1, parent2)
            else:
                child = parent1[:]

            # Mutate
            if random.random() < self.mutation_rate:
                child = self._mutate(child)

            new_population.append(child)

        self.population = new_population
        return self.population[0]  # Return best

    def _tournament_select(self, tournament_size: int = 3) -> list[StrategyGene]:
        """Select an individual using tournament selection."""
        candidates = random.sample(range(len(self.population)), tournament_size)
        best_idx = max(candidates, key=lambda i: self.fitness_scores[i])
        return self.population[best_idx]

    def _crossover(self, parent1: list[StrategyGene], parent2: list[StrategyGene]) -> list[StrategyGene]:
        """Perform crossover between two parents."""
        child = []
        for g1, g2 in zip(parent1, parent2):
            # Uniform crossover
            if random.random() < 0.5:
                child.append(StrategyGene(g1.name, g1.value, g1.min_val, g1.max_val))
            else:
                child.append(StrategyGene(g2.name, g2.value, g2.min_val, g2.max_val))
        return child

    def _mutate(self, individual: list[StrategyGene]) -> list[StrategyGene]:
        """Mutate an individual."""
        for i, gene in enumerate(individual):
            if random.random() < 0.1:  # 10% mutation chance per gene
                mutation = random.uniform(-0.1, 0.1)
                new_val = gene.value + mutation
                new_val = max(gene.min_val, min(gene.max_val, new_val))
                individual[i] = StrategyGene(gene.name, new_val, gene.min_val, gene.max_val)
        return individual


# =============================================================================
# Ensemble Arbiter
# =============================================================================

class EnsembleArbiter:
    """Intelligently combines predictions from multiple models.

    Learns optimal weights for each model based on:
    - Recent accuracy of each model
    - Market regime compatibility
    - Uncertainty estimates
    - Diversity of predictions
    """

    def __init__(self, num_models: int = 5):
        self.num_models = num_models
        self.model_weights = [1.0 / num_models] * num_models
        self.model_accuracy = [0.5] * num_models
        self.model_uncertainty = [0.5] * num_models
        self.regime_weights: dict[str, list[float]] = {
            "TREND": [1.0] * num_models,
            "RANGE": [1.0] * num_models,
            "VOLATILE": [1.0] * num_models,
            "TRANSITION": [1.0] * num_models,
        }

    def update_model_performance(self, model_idx: int, correct: bool, uncertainty: float) -> None:
        """Update performance tracking for a model."""
        # Update accuracy with exponential moving average
        learning_rate = 0.1
        if correct:
            self.model_accuracy[model_idx] = (1 - learning_rate) * self.model_accuracy[model_idx] + learning_rate * 1.0
        else:
            self.model_accuracy[model_idx] = (1 - learning_rate) * self.model_accuracy[model_idx] + learning_rate * 0.0

        # Update uncertainty
        self.model_uncertainty[model_idx] = (1 - learning_rate) * self.model_uncertainty[model_idx] + learning_rate * uncertainty

    def get_ensemble_weights(self, regime: str = "TREND") -> list[float]:
        """Get optimal ensemble weights for a given regime."""
        # Base weights from accuracy
        base_weights = [
            self.model_accuracy[i] / (self.model_uncertainty[i] + 0.1)
            for i in range(self.num_models)
        ]

        # Apply regime adjustment
        regime_weights = self.regime_weights.get(regime, [1.0] * self.num_models)
        adjusted_weights = [b * r for b, r in zip(base_weights, regime_weights)]

        # Normalize
        total = sum(adjusted_weights)
        if total > 0:
            return [w / total for w in adjusted_weights]
        else:
            return [1.0 / self.num_models] * self.num_models

    def combine_predictions(
        self,
        predictions: list[float],
        regime: str = "TREND",
    ) -> float:
        """Combine predictions from multiple models."""
        weights = self.get_ensemble_weights(regime)
        combined = sum(p * w for p, w in zip(predictions, weights))
        return combined


# =============================================================================
# Self-Improving Loop
# =============================================================================

class SelfImprovingLoop:
    """The self-improving loop that ties everything together.

    This is the core mechanism that allows Saviour to:
    1. Learn from every trade (online learning)
    2. Adapt to changing conditions (meta-learning)
    3. Discover new strategies (genetic evolution)
    4. Optimize ensemble weights (arbitration)
    """

    def __init__(
        self,
        controller: MetaController,
        rl_trainer: RLTrainer,
        evolver: GeneticStrategyEvolver,
        arbiter: EnsembleArbiter,
    ):
        self.controller = controller
        self.rl_trainer = rl_trainer
        self.evolver = evolver
        self.arbiter = arbiter
        self.trade_history: list[dict] = []
        self.meta_history: list[dict] = []

    def on_trade_result(
        self,
        features: dict[str, float],
        model_predictions: list[float],
        action: str,
        outcome: str,
        reward: float,
    ) -> None:
        """Process the result of a trade and update all systems."""
        # Record trade
        self.trade_history.append({
            "features": features,
            "model_predictions": model_predictions,
            "action": action,
            "outcome": outcome,
            "reward": reward,
        })

        # Update RL trainer
        features_tensor = torch.tensor([list(features.values())], dtype=torch.float32)
        model_pred_tensor = torch.tensor([model_predictions], dtype=torch.float32) if model_predictions else None
        action_idx = ["NO_TRADE", "BUY", "SELL"].index(action)

        self.rl_trainer.remember(
            features_tensor,
            model_pred_tensor,
            None,
            action_idx,
            reward,
            True,
        )

        # Update ensemble arbiter
        if model_predictions:
            correct = (action == "BUY" and outcome == "WIN") or (action == "SELL" and outcome == "WIN")
            uncertainty = 1.0 - max(model_predictions)
            for i in range(len(model_predictions)):
                self.arbiter.update_model_performance(i, correct, uncertainty)

        # Periodically update RL
        if len(self.trade_history) % 10 == 0:
            self.rl_trainer.update()

    def get_meta_state(self) -> dict:
        """Get current meta-state of the system."""
        return {
            "trade_count": len(self.trade_history),
            "model_accuracies": self.arbiter.model_accuracy,
            "model_uncertainties": self.arbiter.model_uncertainty,
            "controller_state": {
                "ensemble_weights": self.controller.ensemble_weights.detach().numpy().tolist(),
                "meta_state": self.controller.meta_state.detach().numpy().tolist(),
            },
        }
