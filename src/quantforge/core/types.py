"""Domain primitives.

Integer-first arithmetic for cash and quantities to avoid floating-point drift:
cash is held in integer minor units (cents for USD, satoshis for BTC), and share
quantities are integer shares. Floats appear only at the reporting boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import NewType
from uuid import uuid4

Symbol = NewType("Symbol", str)
"""Ticker / instrument identifier (e.g. 'SPY', 'BTC-USD')."""


class OrderSide(StrEnum):
    BUY = "buy"
    SELL = "sell"

    @property
    def sign(self) -> int:
        return 1 if self is OrderSide.BUY else -1


class OrderType(StrEnum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


class OrderStatus(StrEnum):
    PENDING = "pending"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class Bar:
    """OHLCV bar. Timestamp is the bar's *close* time."""

    symbol: Symbol
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    interval: str = "1d"

    def __post_init__(self) -> None:
        # Tolerance for FP drift in adjusted prices from data vendors.
        # yfinance's split/dividend adjustment can leave the close 1-2 ULPs
        # above high or below low; we accept up to 1e-6 relative drift here
        # and reject anything looser as genuine data corruption.
        scale = max(abs(self.high), abs(self.low), 1.0)
        tol = 1e-6 * scale
        lo = self.low - tol
        hi = self.high + tol
        if not (lo <= self.open <= hi and lo <= self.close <= hi and self.low <= self.high + tol):
            raise ValueError(
                f"OHLC violates bounds for {self.symbol} @ {self.timestamp}: "
                f"O={self.open} H={self.high} L={self.low} C={self.close}"
            )
        if self.volume < 0:
            raise ValueError(f"Negative volume: {self.volume}")


@dataclass(frozen=True, slots=True)
class Tick:
    """Top-of-book quote / trade tick."""

    symbol: Symbol
    timestamp: datetime
    bid: float
    ask: float
    bid_size: float = 0.0
    ask_size: float = 0.0
    last: float | None = None

    @property
    def mid(self) -> float:
        return 0.5 * (self.bid + self.ask)

    @property
    def spread(self) -> float:
        return self.ask - self.bid


@dataclass(slots=True)
class Order:
    """Mutable order — quantity_filled / status update over its lifetime."""

    symbol: Symbol
    side: OrderSide
    quantity: int
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    stop_price: float | None = None
    timestamp: datetime | None = None
    order_id: str = field(default_factory=lambda: uuid4().hex[:12])
    status: OrderStatus = OrderStatus.PENDING
    quantity_filled: int = 0
    avg_fill_price: float = 0.0

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError(f"Order quantity must be positive, got {self.quantity}")
        if self.order_type is OrderType.LIMIT and self.limit_price is None:
            raise ValueError("Limit order requires limit_price")
        if self.order_type is OrderType.STOP and self.stop_price is None:
            raise ValueError("Stop order requires stop_price")

    @property
    def remaining(self) -> int:
        return self.quantity - self.quantity_filled

    @property
    def is_done(self) -> bool:
        return self.status in (OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED)


@dataclass(slots=True)
class Position:
    """Per-symbol position. Quantities are signed (long > 0, short < 0)."""

    symbol: Symbol
    quantity: int = 0
    avg_cost: float = 0.0
    realized_pnl: float = 0.0

    def market_value(self, price: float) -> float:
        return self.quantity * price

    def unrealized_pnl(self, price: float) -> float:
        return self.quantity * (price - self.avg_cost)

    def apply_fill(self, side: OrderSide, qty: int, price: float) -> None:
        """Update average cost / realized P&L using the running-average method.

        Closing or flipping a position realizes P&L on the closed portion at the
        prior average cost.
        """
        signed_qty = side.sign * qty
        new_qty = self.quantity + signed_qty

        if self.quantity == 0 or self.quantity * signed_qty > 0:
            # Opening or adding to an existing position — running-avg cost.
            total_cost = self.avg_cost * self.quantity + price * signed_qty
            self.avg_cost = total_cost / new_qty if new_qty != 0 else 0.0
        else:
            # Reducing or flipping. Close min(|prior|, |incoming|) at avg_cost.
            closing_qty = min(abs(self.quantity), abs(signed_qty))
            direction = 1 if self.quantity > 0 else -1
            self.realized_pnl += direction * closing_qty * (price - self.avg_cost)
            if abs(signed_qty) > abs(self.quantity):
                # Flipped through zero — remainder opens at fill price.
                self.avg_cost = price
            elif new_qty == 0:
                self.avg_cost = 0.0
            # else: partial close, avg_cost unchanged

        self.quantity = new_qty
