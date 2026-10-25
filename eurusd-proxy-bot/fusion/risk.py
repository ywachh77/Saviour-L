"""Paper risk management: ATR invalidation stops, targets, and 1-2% sizing.

This repository never connects to a broker or places a real-money order. Risk
rules still belong in a research engine for two reasons:

1. High hit rates without explicit stops are the oldest trick in retail EA
   marketing: take tiny profits and let losers run unboundedly until one move
   erases everything. Defining hard ATR invalidation levels makes risk-reward
   visible rather than hidden.
2. Anyone translating paper signals into a real risk plan needs position sizing
   bounded to a small fraction of equity (typically 1% to 2%), plus a daily
   drawdown / losing-streak circuit breaker that halts new entries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from . import instruments

DEFAULT_RISK_FRACTION = 0.01       # 1% of equity per trade
MAX_SAFE_RISK_FRACTION = 0.02      # 2% hard ceiling unless explicitly overridden
DEFAULT_STOP_ATR = 1.5
DEFAULT_TARGET_ATR = 2.25          # 1.5 : 1 reward-to-risk before spread
DEFAULT_MAX_DAILY_LOSS_R = 3.0     # stop trading for the day at -3R
DEFAULT_MAX_LOSS_STREAK = 4


@dataclass(frozen=True, slots=True)
class TradeRiskPlan:
    symbol: str
    side: str
    entry_price: float
    stop_price: float
    target_price: float
    stop_pips: float
    target_pips: float
    spread_pips: float
    net_reward_to_risk: float
    risk_amount_usd: float
    position_units: float
    risk_fraction: float

    def to_dict(self) -> dict:
        inst = instruments.for_symbol(self.symbol)
        return {
            "symbol": self.symbol,
            "side": self.side,
            "entry_price": round(self.entry_price, inst.precision),
            "stop_price": round(self.stop_price, inst.precision),
            "target_price": round(self.target_price, inst.precision),
            "stop_pips": round(self.stop_pips, 2),
            "target_pips": round(self.target_pips, 2),
            "spread_pips": round(self.spread_pips, 2),
            "net_reward_to_risk": round(self.net_reward_to_risk, 3),
            "risk_amount_usd": round(self.risk_amount_usd, 2),
            "position_units": round(self.position_units, 6),
            "risk_fraction": round(self.risk_fraction, 4),
        }


@dataclass(frozen=True, slots=True)
class CircuitBreakerState:
    halted: bool
    reason: str
    consecutive_losses: int
    net_pips_today: float


def build_risk_plan(
    *,
    symbol: str,
    side: str,
    entry_price: float,
    atr_pips: float,
    balance_usd: float = 10_000.0,
    risk_fraction: float = DEFAULT_RISK_FRACTION,
    stop_atr_multiple: float = DEFAULT_STOP_ATR,
    target_atr_multiple: float = DEFAULT_TARGET_ATR,
    spread_pips: float | None = None,
    cap_risk_fraction: bool = True,
) -> TradeRiskPlan:
    """Compute hard stop, target, net R:R, and 1-2% sized units for one setup."""
    clean_side = side.strip().upper()
    if clean_side not in ("BUY", "SELL"):
        raise ValueError(f"side must be BUY or SELL, got {side!r}")
    if entry_price <= 0:
        raise ValueError("entry_price must be positive")
    if atr_pips <= 0:
        raise ValueError("atr_pips must be positive to place an ATR stop")
    if balance_usd <= 0:
        raise ValueError("balance_usd must be positive")
    if risk_fraction <= 0:
        raise ValueError("risk_fraction must be positive")

    inst = instruments.for_symbol(symbol)
    effective_fraction = (
        min(float(risk_fraction), MAX_SAFE_RISK_FRACTION)
        if cap_risk_fraction
        else float(risk_fraction)
    )
    spread = float(inst.spread_pips if spread_pips is None else spread_pips)

    stop_pips = max(spread * 1.5, float(atr_pips) * float(stop_atr_multiple))
    target_pips = max(stop_pips, float(atr_pips) * float(target_atr_multiple))

    stop_delta = inst.price(stop_pips)
    target_delta = inst.price(target_pips)

    if clean_side == "BUY":
        stop_price = entry_price - stop_delta
        target_price = entry_price + target_delta
    else:
        stop_price = entry_price + stop_delta
        target_price = entry_price - target_delta

    # Spread hurts the target and hastens the stop on a market order.
    effective_risk_pips = stop_pips + spread
    effective_reward_pips = max(0.0, target_pips - spread)
    net_rr = effective_reward_pips / effective_risk_pips if effective_risk_pips > 0 else 0.0

    risk_amount = float(balance_usd) * effective_fraction
    # Each unit of base currency (1 EUR, 1 BTC, 1 ETH) quoted in USD loses
    # inst.price(effective_risk_pips) USD if stopped out after paying spread.
    loss_per_unit_usd = inst.price(effective_risk_pips)
    units = risk_amount / loss_per_unit_usd if loss_per_unit_usd > 0 else 0.0

    return TradeRiskPlan(
        symbol=inst.symbol,
        side=clean_side,
        entry_price=entry_price,
        stop_price=stop_price,
        target_price=target_price,
        stop_pips=stop_pips,
        target_pips=target_pips,
        spread_pips=spread,
        net_reward_to_risk=net_rr,
        risk_amount_usd=risk_amount,
        position_units=units,
        risk_fraction=effective_fraction,
    )


def evaluate_circuit_breaker(
    settlements: Sequence[dict],
    *,
    stop_pips_per_r: float = 10.0,
    max_daily_loss_r: float = DEFAULT_MAX_DAILY_LOSS_R,
    max_loss_streak: int = DEFAULT_MAX_LOSS_STREAK,
) -> CircuitBreakerState:
    """Check whether recent settled paper outcomes trip a daily stop."""
    consecutive_losses = 0
    for row in settlements:
        outcome = str(row.get("outcome", "")).upper()
        if outcome == "LOSS":
            consecutive_losses += 1
        elif outcome == "WIN":
            break

    net_today = sum(
        float(row["net_pips"])
        for row in settlements
        if row.get("net_pips") is not None
    )

    if consecutive_losses >= int(max_loss_streak):
        return CircuitBreakerState(
            True,
            f"Circuit breaker: {consecutive_losses} consecutive losses reached the {max_loss_streak}-loss limit.",
            consecutive_losses,
            net_today,
        )

    max_loss_pips = abs(float(stop_pips_per_r)) * abs(float(max_daily_loss_r))
    if max_loss_pips > 0 and net_today <= -max_loss_pips:
        return CircuitBreakerState(
            True,
            f"Circuit breaker: cumulative net {net_today:+.1f} pips breached the -{max_loss_pips:.1f} pip daily limit.",
            consecutive_losses,
            net_today,
        )

    return CircuitBreakerState(False, "Within risk limits.", consecutive_losses, net_today)
