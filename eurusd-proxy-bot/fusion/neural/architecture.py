"""SAVIOUR Temporal Transformer — Custom Neural Network Architecture.

A multi-task Temporal Transformer that learns to predict:
  - Direction (BUY / SELL / NO TRADE)
  - Expected return (in pips)
  - Uncertainty (predictive variance proxy)
  - Market regime (TREND / RANGE / VOLATILE / TRANSITION)

All weights are initialized randomly and trained by Saviour from scratch.
No pretrained models, no external weights.

Architecture (per spec):
  Input:    120 one-minute candles (OHLCV + engineered features)
  Embedding: 128 dimensions
  Attention: 4 heads, 4 layers
  FF dimension: 256
  Dropout:   0.1
  Horizon:   1 minute (extendable)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class NeuralConfig:
    """Hyperparameters for the Temporal Transformer."""
    # Input
    input_window: int = 120          # Number of one-minute candles
    num_features: int = 18           # OHLCV (5) + technical features (13)

    # Transformer
    d_model: int = 128               # Embedding dimension
    n_heads: int = 4                 # Attention heads
    n_layers: int = 4                # Transformer layers
    d_ff: int = 256                  # Feed-forward dimension
    dropout: float = 0.1             # Dropout rate

    # Heads
    horizon_minutes: int = 1         # Initial prediction horizon
    num_regimes: int = 4             # TREND, RANGE, VOLATILE, TRANSITION

    # Normalization
    normalize_input: bool = True     # Use training-only statistics

    # Output
    direction_threshold: float = 0.55  # Min probability to act (after cost guard)
    min_expected_return_pips: float = 1.5  # Min expected return to act
    max_uncertainty: float = 0.65    # Max uncertainty before NO TRADE

    def __post_init__(self) -> None:
        """Validate config values are in sensible ranges."""
        if self.input_window < 10:
            raise ValueError(f"input_window must be >= 10, got {self.input_window}")
        if self.horizon_minutes < 1:
            raise ValueError(f"horizon_minutes must be >= 1, got {self.horizon_minutes}")
        if self.d_model < 16:
            raise ValueError(f"d_model must be >= 16, got {self.d_model}")
        if self.n_heads < 1 or self.n_heads > 16:
            raise ValueError(f"n_heads must be between 1 and 16, got {self.n_heads}")
        if self.n_layers < 1 or self.n_layers > 12:
            raise ValueError(f"n_layers must be between 1 and 12, got {self.n_layers}")
        if self.d_ff < 16:
            raise ValueError(f"d_ff must be >= 16, got {self.d_ff}")
        if self.dropout < 0 or self.dropout > 0.9:
            raise ValueError(f"dropout must be between 0 and 0.9, got {self.dropout}")
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")

    def to_dict(self) -> dict:
        return {
            "input_window": self.input_window,
            "num_features": self.num_features,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "d_ff": self.d_ff,
            "dropout": self.dropout,
            "horizon_minutes": self.horizon_minutes,
            "num_regimes": self.num_regimes,
            "normalize_input": self.normalize_input,
            "direction_threshold": self.direction_threshold,
            "min_expected_return_pips": self.min_expected_return_pips,
            "max_uncertainty": self.max_uncertainty,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "NeuralConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------

class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding — no learned parameters needed."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, d_model)
        return x + self.pe[:, : x.size(1)]


# ---------------------------------------------------------------------------
# Market Input Encoder
# ---------------------------------------------------------------------------

class MarketInputEncoder(nn.Module):
    """Encodes raw candle features into dense embeddings.

    Each candle's feature vector (OHLCV + technical indicators) is projected
    into the model dimension via a linear layer followed by layer normalization
    and a GELU activation.
    """

    def __init__(self, config: NeuralConfig):
        super().__init__()
        self.config = config
        self.input_projection = nn.Linear(config.num_features, config.d_model)
        self.norm = nn.LayerNorm(config.d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, num_features)
        h = self.input_projection(x)
        h = self.norm(h)
        h = self.act(h)
        h = self.dropout(h)
        return h


# ---------------------------------------------------------------------------
# Market-State Encoder
# ---------------------------------------------------------------------------

class MarketStateEncoder(nn.Module):
    """Encodes market-state context (regime, volatility regime, session).

    Produces a context vector that modulates the temporal features.
    """

    def __init__(self, config: NeuralConfig):
        super().__init__()
        self.config = config
        # State features: atr_pct, adx, rsi, momentum, volatility_ratio, session_code
        self.state_projection = nn.Linear(6, config.d_model)
        self.norm = nn.LayerNorm(config.d_model)
        self.act = nn.GELU()

    def forward(self, state_features: torch.Tensor) -> torch.Tensor:
        # state_features: (batch, 6)
        h = self.state_projection(state_features)
        h = self.norm(h)
        h = self.act(h)
        return h  # (batch, d_model)


# ---------------------------------------------------------------------------
# Neural Feature Fusion
# ---------------------------------------------------------------------------

class FeatureFusion(nn.Module):
    """Fuses temporal features with market-state context.

    Uses a FiLM-like (Feature-wise Linear Modulation) approach where
    the state context generates scale and shift parameters that modulate
    the temporal features.
    """

    def __init__(self, config: NeuralConfig):
        super().__init__()
        self.config = config
        self.gamma = nn.Linear(config.d_model, config.d_model)
        self.beta = nn.Linear(config.d_model, config.d_model)
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, temporal: torch.Tensor, state_ctx: torch.Tensor) -> torch.Tensor:
        # temporal: (batch, seq, d_model)
        # state_ctx: (batch, d_model)
        gamma = self.gamma(state_ctx).unsqueeze(1)  # (batch, 1, d_model)
        beta = self.beta(state_ctx).unsqueeze(1)    # (batch, 1, d_model)
        return self.norm(temporal * gamma + beta)


# ---------------------------------------------------------------------------
# Temporal Attention Block
# ---------------------------------------------------------------------------

class TemporalAttentionBlock(nn.Module):
    """One transformer encoder layer: self-attention + feed-forward."""

    def __init__(self, config: NeuralConfig):
        super().__init__()
        self.config = config

        self.attention = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.n_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)

        self.ff = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        attn_out, _ = self.attention(x, x, x, need_weights=False)
        x = self.norm1(x + attn_out)

        # Feed-forward with residual
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x


# ---------------------------------------------------------------------------
# Multi-Task Prediction Heads
# ---------------------------------------------------------------------------

class DirectionHead(nn.Module):
    """Predicts direction: BUY / SELL / NO TRADE.

    Uses a 3-class softmax output. NO TRADE is predicted when the model
    is uncertain or expects sub-threshold returns.
    """

    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 3),  # BUY, SELL, NO_TRADE
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReturnHead(nn.Module):
    """Predicts expected return in pips (scalar regression)."""

    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class UncertaintyHead(nn.Module):
    """Predicts uncertainty (predictive variance proxy, bounded to [0, 1])."""

    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Sigmoid to bound to [0, 1]
        return torch.sigmoid(self.net(x).squeeze(-1))


class RegimeHead(nn.Module):
    """Classifies market regime: TREND / RANGE / VOLATILE / TRANSITION."""

    def __init__(self, d_model: int, num_regimes: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_regimes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Full Temporal Transformer
# ---------------------------------------------------------------------------

class TemporalTransformer(nn.Module):
    """The complete SAVIOUR neural network.

    Pipeline:
      1. MarketInputEncoder — candle features → embeddings
      2. SinusoidalPositionalEncoding — add positional info
      3. N TemporalAttentionBlocks — temporal reasoning
      4. FeatureFusion — modulate with market state
      5. Multi-task heads — direction, return, uncertainty, regime

    The final timestep representation is used for all heads.
    """

    def __init__(self, config: NeuralConfig):
        super().__init__()
        self.config = config

        # Encoders
        self.input_encoder = MarketInputEncoder(config)
        self.pos_encoding = SinusoidalPositionalEncoding(config.d_model)
        self.state_encoder = MarketStateEncoder(config)
        self.fusion = FeatureFusion(config)

        # Temporal attention layers
        self.attention_layers = nn.ModuleList([
            TemporalAttentionBlock(config) for _ in range(config.n_layers)
        ])

        # Final norm
        self.final_norm = nn.LayerNorm(config.d_model)

        # Prediction heads
        self.direction_head = DirectionHead(config.d_model, config.dropout)
        self.return_head = ReturnHead(config.d_model, config.dropout)
        self.uncertainty_head = UncertaintyHead(config.d_model, config.dropout)
        self.regime_head = RegimeHead(config.d_model, config.num_regimes, config.dropout)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.MultiheadAttention):
                # Xavier init for the out_proj and in_proj
                for proj in [module.in_proj_weight, module.out_proj.weight]:
                    if proj is not None:
                        nn.init.xavier_uniform_(proj)
                if module.in_proj_bias is not None:
                    nn.init.zeros_(module.in_proj_bias)
                if module.out_proj.bias is not None:
                    nn.init.zeros_(module.out_proj.bias)

    def forward(self, x: torch.Tensor, state_features: torch.Tensor | None = None) -> dict:
        """Forward pass.

        Args:
            x: (batch, seq, num_features) — candle feature sequences
            state_features: (batch, 6) — market-state context features

        Returns:
            dict with:
              direction_logits: (batch, 3)
              expected_return: (batch,)
              uncertainty: (batch,)
              regime_logits: (batch, num_regimes)
              hidden: (batch, d_model) — final hidden representation
        """
        # 1. Encode input
        h = self.input_encoder(x)

        # 2. Add positional encoding
        h = self.pos_encoding(h)

        # 3. Temporal attention
        for layer in self.attention_layers:
            h = layer(h)

        # 4. Fusion with market state
        if state_features is not None:
            state_ctx = self.state_encoder(state_features)
            h = self.fusion(h, state_ctx)

        # 5. Final norm
        h = self.final_norm(h)

        # Use the last timestep for prediction
        last_h = h[:, -1, :]

        # 6. Multi-task heads
        direction_logits = self.direction_head(last_h)
        expected_return = self.return_head(last_h)
        uncertainty = self.uncertainty_head(last_h)
        regime_logits = self.regime_head(last_h)

        return {
            "direction_logits": direction_logits,
            "expected_return": expected_return,
            "uncertainty": uncertainty,
            "regime_logits": regime_logits,
            "hidden": last_h,
        }

    def predict(self, x: torch.Tensor, state_features: torch.Tensor | None = None) -> dict:
        """Forward pass with post-processing for live prediction.

        Returns a structured prediction dict with all required fields.
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(x, state_features)

        direction_probs = F.softmax(output["direction_logits"], dim=-1)
        # Classes: 0=BUY, 1=SELL, 2=NO_TRADE
        buy_prob = direction_probs[:, 0]
        sell_prob = direction_probs[:, 1]
        no_trade_prob = direction_probs[:, 2]

        regime_probs = F.softmax(output["regime_logits"], dim=-1)
        regime_idx = torch.argmax(regime_probs, dim=-1)

        # Decide action
        cfg = self.config
        actions = []
        for i in range(x.size(0)):
            buy_p = buy_prob[i].item()
            sell_p = sell_prob[i].item()
            no_p = no_trade_prob[i].item()
            ret = output["expected_return"][i].item()
            unc = output["uncertainty"][i].item()

            # Cost-aware decision
            # NO TRADE if uncertainty too high
            if unc > cfg.max_uncertainty:
                action = "NO_TRADE"
            elif buy_p > cfg.direction_threshold and ret > cfg.min_expected_return_pips:
                action = "BUY"
            elif sell_p > cfg.direction_threshold and -ret > cfg.min_expected_return_pips:
                action = "SELL"
            elif buy_p > sell_p and ret > 0:
                action = "BUY"  # weak signal but positive expected return
            elif sell_p > buy_p and ret < 0:
                action = "SELL"
            else:
                action = "NO_TRADE"

            actions.append({
                "action": action,
                "direction_probs": {
                    "BUY": buy_p,
                    "SELL": sell_p,
                    "NO_TRADE": no_p,
                },
                "expected_return_pips": ret,
                "uncertainty": unc,
                "regime": ["TREND", "RANGE", "VOLATILE", "TRANSITION"][regime_idx[i].item()],
                "regime_probs": {
                    "TREND": regime_probs[i, 0].item(),
                    "RANGE": regime_probs[i, 1].item(),
                    "VOLATILE": regime_probs[i, 2].item(),
                    "TRANSITION": regime_probs[i, 3].item(),
                },
            })

        return {
            "predictions": actions,
            "direction_probs": direction_probs,
            "expected_return": output["expected_return"],
            "uncertainty": output["uncertainty"],
            "regime_logits": output["regime_logits"],
        }


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def build_model(config: NeuralConfig | None = None) -> TemporalTransformer:
    """Create a TemporalTransformer with the given config (or defaults)."""
    return TemporalTransformer(config or NeuralConfig())


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
