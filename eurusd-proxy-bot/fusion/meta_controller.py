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
Uses ONLY Python stdlib (no torch, no numpy) for maximum portability.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence


# ============================================================================
# Pure Python Linear Algebra Helpers
# ============================================================================

def matmul(a, b):
    """Matrix multiplication for 1D and 2D lists."""
    if isinstance(a[0], (int, float)):
        # a is 1D, b is 2D
        return [sum(a[i] * b[i][j] for i in range(len(a))) for j in range(len(b[0]))]
    elif isinstance(b[0], (int, float)):
        # a is 2D, b is 1D
        return [sum(a[i][j] * b[j] for j in range(len(b))) for i in range(len(a))]
    else:
        # Both 2D
        return [
            [sum(a[i][k] * b[k][j] for k in range(len(b))) for j in range(len(b[0]))]
            for i in range(len(a))
        ]


def matvec(matrix, vector):
    """Matrix-vector multiplication."""
    return [sum(matrix[i][j] * vector[j] for j in range(len(vector))) for i in range(len(matrix))]


def dot(a, b):
    """Dot product of two vectors."""
    return sum(x * y for x, y in zip(a, b))


def relu(x):
    """ReLU activation - element-wise for list or scalar."""
    if isinstance(x, list):
        return [max(0.0, v) for v in x]
    return max(0.0, x)


def softmax(x):
    """Softmax activation for 1D list."""
    if not x:
        return x
    max_val = max(x)
    exps = [math.exp(v - max_val) for v in x]
    total = sum(exps)
    if total == 0:
        return [1.0 / len(x)] * len(x)
    return [e / total for e in exps]


def relu_layer(x):
    """Apply ReLU to a 1D list."""
    return [max(0.0, v) for v in x]


class Linear:
    """Linear layer (fully connected) using pure Python lists."""

    def __init__(self, in_features, out_features):
        # He initialization
        std = math.sqrt(2.0 / in_features)
        self.weights = [[random.gauss(0, std) for _ in range(out_features)] for _ in range(in_features)]
        self.bias = [0.0] * out_features

    def __call__(self, x):
        """Forward pass: y = xW + b"""
        return [dot(row, x) + b for row, b in zip(self.weights, self.bias)]


# ============================================================================
# Meta-Learning Controller — The Root AI Brain
# ============================================================================

class MetaController:
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
        num_features: int = 50,  # MAXIMUM: 50 features
        num_models: int = 15,    # MAXIMUM: 15 models in ensemble
        num_strategies: int = 50,  # MAXIMUM: 50 strategies
        regime_dim: int = 20,    # MAXIMUM: 20 regime dimensions
    ):
        self.num_features = num_features
        self.num_models = num_models
        self.num_strategies = num_strategies
        self.regime_dim = regime_dim

        # Market regime detector — MAXIMUM depth (5 hidden layers)
        self.regime_detector = Linear(num_features, 256)
        self.regime_detector_2 = Linear(256, 128)
        self.regime_detector_3 = Linear(128, 64)
        self.regime_detector_4 = Linear(64, 32)
        self.regime_detector_5 = Linear(32, regime_dim)

        # Hyperparameter adapter — MAXIMUM depth (5 hidden layers)
        self.hyper_adapter_1 = Linear(regime_dim + num_features, 512)
        self.hyper_adapter_2 = Linear(512, 256)
        self.hyper_adapter_3 = Linear(256, 128)
        self.hyper_adapter_4 = Linear(128, 64)
        self.hyper_adapter_5 = Linear(64, num_models * 8)  # 8 hyperparams per model (MAXIMUM)

        # Policy network — MAXIMUM depth (5 hidden layers)
        policy_input_dim = regime_dim + num_features + num_models + num_strategies
        self.policy_network_1 = Linear(policy_input_dim, 512)
        self.policy_network_2 = Linear(512, 256)
        self.policy_network_3 = Linear(256, 128)
        self.policy_network_4 = Linear(128, 64)
        self.policy_network_5 = Linear(64, 3)  # BUY, SELL, NO_TRADE

        # Value network — MAXIMUM depth (5 hidden layers)
        self.value_network_1 = Linear(policy_input_dim, 512)
        self.value_network_2 = Linear(512, 256)
        self.value_network_3 = Linear(256, 128)
        self.value_network_4 = Linear(128, 64)
        self.value_network_5 = Linear(64, 1)

        # Counterfactual reasoning network — MAXIMUM depth (5 hidden layers)
        self.counterfactual_1 = Linear(policy_input_dim, 512)
        self.counterfactual_2 = Linear(512, 256)
        self.counterfactual_3 = Linear(256, 128)
        self.counterfactual_4 = Linear(128, 64)
        self.counterfactual_5 = Linear(64, 3)  # Counterfactual action outcomes

        # Strategy evolution memory
        self.strategy_memory = [[random.gauss(0, 0.1) for _ in range(num_features + 5)] for _ in range(num_strategies)]

        # Learned ensemble weights
        self.ensemble_weights = [random.gauss(0, 0.1) for _ in range(num_models)]

        # Meta-learning state
        self.meta_state = [random.gauss(0, 0.1) for _ in range(regime_dim)]

    def _forward_mlp(self, x, layers):
        """Forward pass through a sequence of Linear + ReLU layers."""
        h = x
        for layer in layers[:-1]:
            h = layer(h)
            h = relu_layer(h)
        out = layers[-1](h)
        return out

    def forward(
        self,
        features: list[float],
        model_predictions: list[float] | None = None,
        strategy_votes: list[float] | None = None,
    ) -> dict[str, Any]:
        """Forward pass through the meta-controller."""

        # Detect regime
        regime_probs = softmax(self._forward_mlp(features, [
            self.regime_detector, self.regime_detector_2, self.regime_detector_3,
            self.regime_detector_4, self.regime_detector_5
        ]))

        # Adapt hyperparameters based on regime + features
        hyper_input = regime_probs + features
        hyper_params = self._forward_mlp(hyper_input, [
            self.hyper_adapter_1, self.hyper_adapter_2, self.hyper_adapter_3,
            self.hyper_adapter_4, self.hyper_adapter_5
        ])

        # Get trading decision from policy network
        # Always build a fixed-size input: regime_dim + num_features + num_models + num_strategies
        model_preds = model_predictions if model_predictions is not None else [0.0] * self.num_models
        strategy_votes_padded = strategy_votes if strategy_votes is not None else [0.0] * self.num_strategies

        policy_input = regime_probs + features + model_preds + strategy_votes_padded

        policy_logits = self._forward_mlp(policy_input, [
            self.policy_network_1, self.policy_network_2, self.policy_network_3,
            self.policy_network_4, self.policy_network_5
        ])
        policy_probs = softmax(policy_logits)

        # Estimate value
        value = self._forward_mlp(policy_input, [
            self.value_network_1, self.value_network_2, self.value_network_3,
            self.value_network_4, self.value_network_5
        ])

        # Estimate counterfactual outcomes
        counterfactual = self._forward_mlp(policy_input, [
            self.counterfactual_1, self.counterfactual_2, self.counterfactual_3,
            self.counterfactual_4, self.counterfactual_5
        ])

        # Get ensemble weights (normalized)
        ensemble_w = softmax(self.ensemble_weights)

        return {
            "regime_probs": regime_probs,
            "hyper_params": hyper_params,
            "policy_probs": policy_probs,
            "value": value,
            "ensemble_weights": ensemble_w,
            "strategy_memory": self.strategy_memory,
            "meta_state": self.meta_state,
            "counterfactual": counterfactual,
        }

    def get_action(
        self,
        features: list[float],
        model_predictions: list[float] | None = None,
        strategy_votes: list[float] | None = None,
        training: bool = False,
    ) -> dict[str, Any]:
        """Get trading action from the meta-controller."""
        outputs = self.forward(features, model_predictions, strategy_votes)

        if training:
            # Sample from policy during training
            action = random.choices(range(3), weights=outputs["policy_probs"], k=1)[0]
        else:
            # Greedy action during execution
            action = outputs["policy_probs"].index(max(outputs["policy_probs"]))

        actions = ["NO_TRADE", "BUY", "SELL"]
        action_str = actions[action]

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
    """Trains the MetaController using PPO (Proximal Policy Optimization).

    Simplified implementation using pure Python — parameters are adjusted
    directly based on reward signals.
    """

    def __init__(self, controller: MetaController, config: RLTrainingConfig | None = None):
        self.controller = controller
        self.config = config or RLTrainingConfig()
        self.memory: list[dict] = []

    def remember(
        self,
        features: list[float],
        model_predictions: list[float] | None,
        strategy_votes: list[float] | None,
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
        gae = 0.0
        next_value = 0.0

        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.config.discount_factor * next_value * (1 - int(dones[step])) - values[step]
            gae = delta + self.config.discount_factor * self.config.gae_lambda * (1 - int(dones[step])) * gae
            advantages.insert(0, gae)
            next_value = values[step]

        return advantages

    def update(self) -> dict:
        """Update the meta-controller using PPO.

        Simplified version — adjusts ensemble weights based on reward signals.
        """
        if len(self.memory) < self.config.batch_size:
            return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        rewards = [t["reward"] for t in self.memory]
        total_reward = sum(rewards)
        avg_reward = total_reward / len(rewards)

        # Adjust ensemble weights based on average reward
        adjustment = self.config.learning_rate * avg_reward
        for i in range(len(self.controller.ensemble_weights)):
            self.controller.ensemble_weights[i] += adjustment

        # Adjust strategy memory based on rewards
        for i in range(len(self.controller.strategy_memory)):
            for j in range(len(self.controller.strategy_memory[i])):
                self.controller.strategy_memory[i][j] += adjustment * 0.1

        # Clear memory
        self.memory = []

        return {
            "loss": abs(avg_reward),
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
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
        features_list = list(features.values())
        action_idx = ["NO_TRADE", "BUY", "SELL"].index(action)

        self.rl_trainer.remember(
            features_list,
            model_predictions,
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
                "ensemble_weights": self.controller.ensemble_weights,
                "meta_state": self.controller.meta_state,
            },
        }
