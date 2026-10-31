"""Neural dataset preparation — Candle → training sequences.

Converts raw OHLCV candles into normalized training sequences with proper
labels. Prevents future-data leakage by only using candles that have closed.

Label scheme:
  Direction: 0=BUY, 1=SELL, 2=NO_TRADE
  Return: expected return in pips (target)
  Uncertainty: derived from recent error rate
  Regime: 0=TREND, 1=RANGE, 2=VOLATILE, 3=TRANSITION
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .architecture import NeuralConfig


# ---------------------------------------------------------------------------
# Feature extraction from candles
# ---------------------------------------------------------------------------

def candle_features(candles: Sequence) -> np.ndarray:
    """Extract feature vectors from a sequence of Candle objects.

    Returns: (num_candles, num_features) array

    Features:
      0-4:   OHLC + volume (normalized)
      5:     log return
      6:     high-low range (in pips)
      7:     close position in range [0, 1]
      8:     body size (in pips)
      9:     upper shadow (in pips)
     10:     lower shadow (in pips)
     11:     RSI (14)
     12:     ATR (normalized)
     13:     EMA fast - slow (normalized)
     14:     ADX (14)
     15:     Momentum (10-period)
     16:     Volatility ratio (ATR / SMA(ATR))
     17:     Consecutive direction count
    """
    rows = []
    n = len(candles)
    if n == 0:
        return np.zeros((0, 18), dtype=np.float32)

    closes = np.array([c.close for c in candles], dtype=np.float64)
    highs = np.array([c.high for c in candles], dtype=np.float64)
    lows = np.array([c.low for c in candles], dtype=np.float64)
    opens = np.array([c.open for c in candles], dtype=np.float64)
    volumes = np.array([c.volume or 0.0 for c in candles], dtype=np.float64)

    # Pip size for EUR/USD
    pip = 0.0001

    for i in range(n):
        o = opens[i]
        h = highs[i]
        l = lows[i]
        c = closes[i]
        v = volumes[i]

        # Normalized OHLCV: use log returns from a reference price
        # Raw prices (~1.1) would dominate the feature vector without normalization
        base_price = closes[0] if n > 0 else 1.0
        feats = [
            math.log(o / base_price) * 1000,  # 0: log open vs base
            math.log(h / base_price) * 1000,  # 1: log high vs base
            math.log(l / base_price) * 1000,  # 2: log low vs base
            math.log(c / base_price) * 1000,  # 3: log close vs base
            v / (1000.0 + 1.0),              # 4: volume (scaled)
        ]

        # Log return
        if i > 0:
            log_ret = math.log(c / closes[i - 1])
        else:
            log_ret = 0.0
        feats.append(log_ret * 1000)  # 5

        # Range
        rng = (h - l) / pip if h > l else 0.0
        feats.append(rng / 10.0)  # 6: normalized range

        # Close position in range
        cp = (c - l) / (h - l) if h > l else 0.5
        feats.append(cp)  # 7

        # Body
        body = abs(c - o) / pip
        feats.append(body / 10.0)  # 8

        # Shadows
        upper = (h - max(o, c)) / pip
        lower = (min(o, c) - l) / pip
        feats.append(upper / 10.0)  # 9
        feats.append(lower / 10.0)  # 10

        # RSI (14)
        if i >= 14:
            gains, losses = [], []
            for j in range(i - 13, i + 1):
                diff = closes[j] - closes[j - 1]
                if diff > 0:
                    gains.append(diff)
                    losses.append(0.0)
                else:
                    gains.append(0.0)
                    losses.append(abs(diff))
            avg_gain = np.mean(gains)
            avg_loss = np.mean(losses)
            if avg_loss == 0:
                rsi = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi = 50.0
        feats.append(rsi / 100.0)  # 11

        # ATR (14)
        if i >= 14:
            tr_vals = []
            for j in range(i - 13, i + 1):
                tr = max(
                    highs[j] - lows[j],
                    abs(highs[j] - closes[j - 1]),
                    abs(lows[j] - closes[j - 1]),
                )
                tr_vals.append(tr)
            atr = np.mean(tr_vals) / pip
        else:
            atr = (h - l) / pip
        feats.append(atr / 10.0)  # 12

        # EMA fast - slow (normalized)
        if i >= 30:
            ema_fast = _ema(closes[: i + 1], period=12)
            ema_slow = _ema(closes[: i + 1], period=26)
            diff = (ema_fast - ema_slow) / pip
        else:
            diff = 0.0
        feats.append(diff / 10.0)  # 13

        # ADX (14)
        if i >= 28:
            adx = _adx(highs, lows, closes, i, period=14)
        else:
            adx = 0.0
        feats.append(adx / 100.0)  # 14

        # Momentum (10-period)
        if i >= 10:
            mom = (c - closes[i - 10]) / pip
        else:
            mom = 0.0
        feats.append(mom / 10.0)  # 15

        # Volatility ratio
        if i >= 20:
            recent_atr = atr
            past_trs = []
            for j in range(i - 19, i):
                tr = max(
                    highs[j] - lows[j],
                    abs(highs[j] - closes[j - 1]),
                    abs(lows[j] - closes[j - 1]),
                )
                past_trs.append(tr / pip)
            past_avg = np.mean(past_trs) if past_trs else 1.0
            vol_ratio = recent_atr / (past_avg + 1e-10)
        else:
            vol_ratio = 1.0
        feats.append(vol_ratio / 2.0)  # 16

        # Consecutive direction streak (count consecutive same-direction moves)
        if i > 0:
            streak = 1
            current_dir = closes[i] > closes[i - 1]  # True=up, False=down
            for j in range(i - 1, -1, -1):
                dir_j = closes[j] > closes[j - 1] if j > 0 else None
                if dir_j == current_dir:
                    streak += 1
                else:
                    break
                if streak > 10:
                    streak = 10
                    break
        else:
            streak = 0
        feats.append(streak / 10.0)  # 17

        rows.append(feats)

    arr = np.array(rows, dtype=np.float32)
    # Clean NaN/inf values — replace with 0.0 to prevent propagation
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def _ema(values: np.ndarray, period: int) -> float:
    k = 2.0 / (period + 1)
    ema = values[0]
    for v in values[1:]:
        ema = v * k + ema * (1 - k)
    return ema


def _adx(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
         index: int, period: int = 14) -> float:
    if index < period:
        return 0.0

    up_moves = np.diff(highs[: index + 1])
    down_moves = -np.diff(lows[: index + 1])

    # Wilder's smoothing
    up_smooth = np.zeros(len(up_moves))
    down_smooth = np.zeros(len(down_moves))

    if len(up_moves) >= period:
        up_smooth[period - 1] = np.sum(up_moves[:period])
        down_smooth[period - 1] = np.sum(down_moves[:period])
        for i in range(period, len(up_moves)):
            up_smooth[i] = (up_smooth[i - 1] * (period - 1) + up_moves[i]) / period
            down_smooth[i] = (down_smooth[i - 1] * (period - 1) + down_moves[i]) / period

    # Handle edge cases
    up_val = up_smooth[-1]
    down_val = down_smooth[-1]

    # NaN or inf check
    if not np.isfinite(up_val) or not np.isfinite(down_val):
        return 0.0

    if up_val == 0 and down_val == 0:
        return 0.0

    # Handle extreme cases: when one direction has no movement
    if down_val == 0:
        return 100.0  # Strong uptrend, maximum directional movement
    if up_val == 0:
        return 100.0  # Strong downtrend, maximum directional movement

    denom = up_val + down_val
    if denom <= 0:
        return 0.0

    dx = 100.0 * abs(up_val - down_val) / denom
    # Clamp to valid ADX range [0, 100]
    return float(np.clip(dx, 0.0, 100.0))


# ---------------------------------------------------------------------------
# Label generation
# ---------------------------------------------------------------------------

@dataclass
class SequenceLabel:
    direction: int       # 0=BUY, 1=SELL, 2=NO_TRADE
    return_pips: float   # expected return in pips
    regime: int          # 0=TREND, 1=RANGE, 2=VOLATILE, 3=TRANSITION
    uncertainty: float   # 0.0 = certain, 1.0 = uncertain


def make_label(
    entry_price: float,
    future_price: float,
    atr_pips: float,
    spread_pips: float = 1.2,
    min_return_pips: float = 3.0,
) -> SequenceLabel:
    """Generate a training label from entry and future prices.

    High-conviction labeling: BUY/SELL labels are only assigned when
    the move is substantial — both in absolute pips AND relative to
    ATR. This teaches the model that most candles are NO_TRADE,
    directly supporting high win rates on the trades it does fire.

    Note: ``direction_threshold`` from NeuralConfig is not used here
    because labels are based on actual price movement, not model
    probabilities. The threshold is used at inference time only.
    """
    ret = (future_price - entry_price) / 0.0001  # pips

    # Net return after cost
    net_ret = abs(ret) - spread_pips

    # High-conviction: require move to clear both absolute and ATR-relative
    # thresholds. This ensures BUY/SELL labels are only for strong moves.
    atr_relative = abs(ret) / (atr_pips + 1e-10)

    if net_ret < min_return_pips or atr_relative < 0.3:
        direction = 2  # NO_TRADE
    elif ret > 0:
        direction = 0  # BUY
    else:
        direction = 1  # SELL

    # Regime classification
    if atr_pips > 3.0:
        regime = 2  # VOLATILE
    elif atr_pips < 1.5:
        regime = 1  # RANGE
    else:
        # Check for trend using |ret| / atr
        if abs(ret) / (atr_pips + 1e-10) > 0.5:
            regime = 0  # TREND
        else:
            regime = 3  # TRANSITION

    # Uncertainty: higher when ATR is low relative to spread
    if atr_pips > 0:
        uncertainty = min(1.0, spread_pips / atr_pips)
    else:
        uncertainty = 1.0

    return SequenceLabel(
        direction=direction,
        return_pips=ret,
        regime=regime,
        uncertainty=uncertainty,
    )


# ---------------------------------------------------------------------------
# Dataset classes
# ---------------------------------------------------------------------------

@dataclass
class TrainingSequence:
    """A single training sample: input window + labels."""
    features: np.ndarray       # (input_window, num_features)
    state: np.ndarray          # (6,) — market-state features
    direction: int
    return_pips: float
    regime: int
    uncertainty: float
    timestamp: str


class NeuralDataset(Dataset):
    """PyTorch Dataset for Temporal Transformer training.

    Prepares sliding windows from candle data with proper labels.
    Prevents future leakage by only labeling with future data that
    has already been observed.
    """

    def __init__(
        self,
        candles: Sequence,
        config: NeuralConfig,
        horizon_minutes: int | None = None,
        spread_pips: float = 1.2,
    ):
        self.config = config
        self.horizon = horizon_minutes or config.horizon_minutes
        self.spread = spread_pips

        # Extract all features
        all_features = candle_features(candles)
        n = len(candles)

        # Build sliding windows
        self.sequences: list[TrainingSequence] = []
        min_idx = config.input_window + self.horizon

        for i in range(min_idx, n):
            # Input window: candles [i - input_window + 1, i]
            # This gives exactly input_window candles (the current + input_window-1 previous)
            window = all_features[i - config.input_window + 1 : i + 1]

            # State features at the end of the window
            state = self._extract_state(all_features, i)

            # Label: future price at horizon
            entry_price = candles[i].close
            future_idx = i + self.horizon
            if future_idx >= n:
                continue  # Skip this index, try the next one
            future_price = candles[future_idx].close
            atr_pips = all_features[i, 12] * 10.0  # ATR was normalized

            label = make_label(
                entry_price,
                future_price,
                atr_pips,
                spread_pips=self.spread,
                min_return_pips=config.min_expected_return_pips,
            )

            self.sequences.append(TrainingSequence(
                features=window,
                state=state,
                direction=label.direction,
                return_pips=label.return_pips,
                regime=label.regime,
                uncertainty=label.uncertainty,
                timestamp=candles[i].timestamp.isoformat(),
            ))

        # Compute normalization statistics (training-only)
        if self.sequences:
            all_feats = np.stack([s.features for s in self.sequences])
            self.feat_mean = all_feats.reshape(-1, all_feats.shape[-1]).mean(axis=0)
            self.feat_std = all_feats.reshape(-1, all_feats.shape[-1]).std(axis=0) + 1e-8
            self.state_mean = np.mean([s.state for s in self.sequences], axis=0)
            self.state_std = np.std([s.state for s in self.sequences], axis=0) + 1e-8
        else:
            self.feat_mean = np.zeros(config.num_features, dtype=np.float32)
            self.feat_std = np.ones(config.num_features, dtype=np.float32)
            self.state_mean = np.zeros(6, dtype=np.float32)
            self.state_std = np.ones(6, dtype=np.float32)

    def _extract_state(self, features: np.ndarray, idx: int) -> np.ndarray:
        """Extract 6-dim market-state features at a given index."""
        atr_pct = features[idx, 12] * 10.0  # ATR in pips
        rsi = features[idx, 11] * 100.0     # RSI in 0-100
        adx = features[idx, 14] * 100.0    # ADX in 0-100
        vol_ratio = features[idx, 16] * 2.0  # Volatility ratio
        momentum = features[idx, 15] * 10.0  # Momentum in pips
        # Session code: 0=Asia, 1=London, 2=NY, 3=overlap (simplified)
        # We'll use 0 as default since session info isn't in features
        session = 0.0
        return np.array([
            atr_pct / 10.0,
            rsi / 100.0,
            adx / 100.0,
            vol_ratio / 2.0,
            momentum / 10.0,
            session,
        ], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict:
        seq = self.sequences[idx]

        # Normalize features
        if self.config.normalize_input:
            feats = (seq.features - self.feat_mean) / self.feat_std
            state = (seq.state - self.state_mean) / self.state_std
        else:
            feats = seq.features
            state = seq.state

        return {
            "features": torch.from_numpy(feats).float(),
            "state": torch.from_numpy(state).float(),
            "direction": torch.tensor(seq.direction, dtype=torch.long),
            "return_pips": torch.tensor(seq.return_pips, dtype=torch.float32),
            "regime": torch.tensor(seq.regime, dtype=torch.long),
            "uncertainty": torch.tensor(seq.uncertainty, dtype=torch.float32),
        }

    def get_normalization_stats(self) -> dict:
        """Return normalization statistics for inference."""
        return {
            "feat_mean": self.feat_mean.tolist(),
            "feat_std": self.feat_std.tolist(),
            "state_mean": self.state_mean.tolist(),
            "state_std": self.state_std.tolist(),
        }
