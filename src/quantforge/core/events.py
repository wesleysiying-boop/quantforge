"""Event types flowing through the engine's queue.

Four event kinds, processed in this order at any given timestamp:

    MarketEvent   ─►  Strategy emits  ─►  SignalEvent
    SignalEvent   ─►  Risk emits      ─►  OrderEvent
    OrderEvent    ─►  Broker emits    ─►  FillEvent
    FillEvent     ─►  Portfolio updates state

The priority field on each event preserves this ordering when timestamps tie.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum

from quantforge.core.types import Bar, Order, OrderSide, Symbol, Tick


class EventType(IntEnum):
    """Lower value = higher priority within the same timestamp."""

    MARKET = 0
    SIGNAL = 1
    ORDER = 2
    FILL = 3


@dataclass(frozen=True, slots=True)
class Event:
    timestamp: datetime
    event_type: EventType


@dataclass(frozen=True, slots=True, kw_only=True)
class MarketEvent(Event):
    bar: Bar | None = None
    tick: Tick | None = None
    event_type: EventType = field(default=EventType.MARKET, init=False)

    def __post_init__(self) -> None:
        if (self.bar is None) == (self.tick is None):
            raise ValueError("MarketEvent must carry exactly one of bar/tick")

    @property
    def symbol(self) -> Symbol:
        if self.bar is not None:
            return self.bar.symbol
        assert self.tick is not None
        return self.tick.symbol


@dataclass(frozen=True, slots=True, kw_only=True)
class SignalEvent(Event):
    """Strategy intent. `target_pct` in [-1, 1] expresses target portfolio weight."""

    symbol: Symbol
    target_pct: float
    strength: float = 1.0
    event_type: EventType = field(default=EventType.SIGNAL, init=False)


@dataclass(frozen=True, slots=True, kw_only=True)
class OrderEvent(Event):
    order: Order
    event_type: EventType = field(default=EventType.ORDER, init=False)


@dataclass(frozen=True, slots=True, kw_only=True)
class FillEvent(Event):
    symbol: Symbol
    side: OrderSide
    quantity: int
    price: float
    commission: float = 0.0
    slippage: float = 0.0
    order_id: str = ""
    event_type: EventType = field(default=EventType.FILL, init=False)

    @property
    def gross_value(self) -> float:
        return self.price * self.quantity

    @property
    def net_value(self) -> float:
        """Cash impact: negative for buys (cash out), positive for sells (cash in)."""
        return -self.side.sign * self.gross_value - self.commission
