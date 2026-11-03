"""Paper trading feedback loop: learn from every loss before going live.

This module implements the core requirement: "make it actually learn from every
loss through paper tests." Every losing paper trade feeds back into the model
weights, creating a continuous improvement loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from . import charter, learning, loss_patterns
from .domain import Candle
from .model import OnlineLogisticModel, GradientBoostingModel


@dataclass
class PaperTradeResult:
    """Result of a paper trade simulation."""
    signal_id: str
    features: dict
    label: int  # 1 = win, 0 = loss
    gross_pips: float
    net_pips: float
    learned: bool = False
    learning_detail: str = ""


@dataclass
class PaperTradingFeedback:
    """Feedback from paper trading loop."""
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    learned_from_losses: int = 0
    loss_analysis: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "total_trades": self.total_trades,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": self.win_rate,
            "learned_from_losses": self.learned_from_losses,
            "loss_analysis": self.loss_analysis,
        }


def simulate_paper_trades(
    candles: Sequence[Candle],
    config,
    model: OnlineLogisticModel | GradientBoostingModel | None,
    *,
    max_trades: int = 100,
) -> tuple[list[PaperTradeResult], PaperTradingFeedback]:
    """Simulate paper trades on recent candles and learn from each outcome.

    This is the core "learn from every loss" mechanism:
    1. Walk through recent candles as if they were live trades
    2. For each simulated trade, check the outcome
    3. If it's a loss, immediately apply online learning with higher rate
    4. Track win rate and loss patterns
    """
    instrument = None
    from . import instruments
    instrument = instruments.for_symbol(getattr(config, "pair", None))

    rep_horizon = learning._representative_horizon(candles, config)
    samples = learning.build_training_samples(candles, horizon=rep_horizon, instrument=instrument)

    if len(samples) < 20:
        return [], PaperTradingFeedback()

    results: list[PaperTradeResult] = []
    settlements_sim: list[dict] = []

    # Use the last portion as "paper trading" window
    test_samples = samples[-max_trades:] if len(samples) > max_trades else samples

    for sample in test_samples:
        signal_id = sample.timestamp
        features = sample.features
        label = sample.label
        gross_pips = sample.gross_pips

        # Simulate spread cost
        net_pips = gross_pips - config.spread_pips if label == 1 else gross_pips + config.spread_pips

        result = PaperTradeResult(
            signal_id=signal_id,
            features=features,
            label=label,
            gross_pips=gross_pips,
            net_pips=net_pips,
        )

        # Learn from this outcome
        if model is not None and model.trained:
            # Loss learning: higher rate for losses
            base_lr = float(getattr(config, "online_learning_rate", 0.02))
            effective_lr = base_lr

            if label == 0:  # Loss
                multiplier = float(getattr(config, "loss_learning_multiplier", 2.0))
                cap = float(getattr(config, "loss_learning_rate_cap", 0.08))
                effective_lr = min(cap, base_lr * multiplier)
                result.learned = True
                result.learning_detail = f"Loss learning at {effective_lr:.3f} rate"
            else:  # Win
                result.learned = True
                result.learning_detail = f"Win learning at {effective_lr:.3f} rate"

            step = learning.online_step(
                model,
                features,
                label,
                source="PAPER_SIMULATION",
                learning_rate=effective_lr,
                allow_synthetic=False,  # Paper sim is not synthetic data
            )
            if step.applied:
                result.learned = True
                settlements_sim.append({
                    "signal_id": signal_id,
                    "outcome": charter.WIN if label == 1 else charter.LOSS,
                    "net_pips": net_pips,
                    "features": features,
                })

        results.append(result)

    # Build feedback summary
    feedback = PaperTradingFeedback(
        total_trades=len(results),
        wins=sum(1 for r in results if r.label == 1),
        losses=sum(1 for r in results if r.label == 0),
        win_rate=sum(1 for r in results if r.label == 1) / max(1, len(results)),
        learned_from_losses=sum(1 for r in results if r.learned and r.label == 0),
    )

    # Analyze loss patterns
    if settlements_sim:
        analysis = loss_patterns.analyze_losses(settlements_sim)
        feedback.loss_analysis = analysis.to_dict()

    return results, feedback


def paper_trading_loop(
    candles: Sequence[Candle],
    settlements: Sequence[dict],
    config,
    model: OnlineLogisticModel | GradientBoostingModel | None,
) -> tuple[OnlineLogisticModel | GradientBoostingModel | None, PaperTradingFeedback]:
    """Run paper trading loop and return updated model.

    This is called from the nightly job as an additional learning step:
    1. Simulate recent trades as paper trades
    2. Learn from each outcome (especially losses)
    3. Return the updated model and feedback
    """
    # Check if we should run paper trading learning
    should_learn, _ = loss_patterns.should_trigger_loss_learning(settlements)

    if not should_learn and model is not None and model.trained:
        # Win rate is already at target, skip paper trading loop
        return model, PaperTradingFeedback()

    # Run paper trading simulation
    results, feedback = simulate_paper_trades(
        candles,
        config,
        model,
        max_trades=50,
    )

    return model, feedback
