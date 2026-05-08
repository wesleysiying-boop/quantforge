"""Risk / sizing layer.

Translates a `SignalEvent` (a target portfolio weight in [-1, 1]) into a
concrete `OrderEvent` (a sized buy/sell). Two sizing schemes provided:

* `TargetWeightRisk` — sizes orders to bring the position to `target_pct` of
  current equity. Simple, transparent, the default.
* `VolatilityTargetRisk` — scales nominal position size inversely with the
  symbol's recent realized vol so that each name contributes roughly equal
  risk to the book. Used in dual-momentum / vol-targeted strategies.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

from quantforge.core.events import OrderEvent, SignalEvent
from quantforge.core.types import Order, OrderSide, Symbol
from quantforge.portfolio.portfolio import Portfolio


class RiskModel(ABC):
    @abstractmethod
    def size(
        self,
        signal: SignalEvent,
        portfolio: Portfolio,
        last_price: dict[Symbol, float],
    ) -> OrderEvent | None: ...


class TargetWeightRisk(RiskModel):
    """Order to bring position weight to `signal.target_pct` of current equity."""

    def __init__(self, max_leverage: float = 1.0) -> None:
        if max_leverage <= 0:
            raise ValueError("max_leverage must be positive")
        self.max_leverage = max_leverage

    def size(
        self,
        signal: SignalEvent,
        portfolio: Portfolio,
        last_price: dict[Symbol, float],
    ) -> OrderEvent | None:
        price = last_price.get(signal.symbol)
        if price is None or price <= 0:
            return None

        target = max(-self.max_leverage, min(self.max_leverage, signal.target_pct))
        equity = portfolio.equity()
        target_notional = target * equity
        target_qty = int(target_notional / price)

        current_qty = portfolio.position(signal.symbol).quantity
        delta = target_qty - current_qty
        if delta == 0:
            return None

        side = OrderSide.BUY if delta > 0 else OrderSide.SELL
        order = Order(
            symbol=signal.symbol,
            side=side,
            quantity=abs(delta),
            timestamp=signal.timestamp,
        )
        return OrderEvent(timestamp=signal.timestamp, order=order)


class VolatilityTargetRisk(RiskModel):
    """Size such that each position contributes `target_vol_annualized` to book risk."""

    def __init__(
        self,
        target_vol_annualized: float = 0.10,
        lookback_returns: int = 21,
        bars_per_year: int = 252,
        max_weight: float = 1.0,
    ) -> None:
        self.target_vol = target_vol_annualized
        self.lookback = lookback_returns
        self.bars_per_year = bars_per_year
        self.max_weight = max_weight
        self._history: dict[Symbol, list[float]] = {}

    def update_returns(self, symbol: Symbol, ret: float) -> None:
        hist = self._history.setdefault(symbol, [])
        hist.append(ret)
        if len(hist) > self.lookback * 4:
            del hist[: -self.lookback * 2]

    def _realized_vol(self, symbol: Symbol) -> float | None:
        hist = self._history.get(symbol, [])
        if len(hist) < self.lookback:
            return None
        window = hist[-self.lookback :]
        mean = sum(window) / len(window)
        var = sum((r - mean) ** 2 for r in window) / max(1, len(window) - 1)
        return math.sqrt(var * self.bars_per_year)

    def size(
        self,
        signal: SignalEvent,
        portfolio: Portfolio,
        last_price: dict[Symbol, float],
    ) -> OrderEvent | None:
        price = last_price.get(signal.symbol)
        if price is None or price <= 0:
            return None
        vol = self._realized_vol(signal.symbol)
        if vol is None or vol <= 0:
            return None

        scale = self.target_vol / vol
        target = max(-self.max_weight, min(self.max_weight, signal.target_pct * scale))

        equity = portfolio.equity()
        target_qty = int(target * equity / price)
        current_qty = portfolio.position(signal.symbol).quantity
        delta = target_qty - current_qty
        if delta == 0:
            return None

        side = OrderSide.BUY if delta > 0 else OrderSide.SELL
        order = Order(
            symbol=signal.symbol,
            side=side,
            quantity=abs(delta),
            timestamp=signal.timestamp,
        )
        return OrderEvent(timestamp=signal.timestamp, order=order)
