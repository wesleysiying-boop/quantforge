"""Event queue ordering tests."""

from __future__ import annotations

from datetime import datetime, timedelta

from quantforge.core.events import (
    EventType,
    FillEvent,
    MarketEvent,
    OrderEvent,
    SignalEvent,
)
from quantforge.core.queue import EventQueue
from quantforge.core.types import Bar, Order, OrderSide, Symbol


def _bar(t: datetime) -> Bar:
    return Bar(Symbol("X"), t, open=10, high=11, low=9, close=10, volume=1)


def test_pops_in_timestamp_order() -> None:
    q = EventQueue()
    t0 = datetime(2024, 1, 1)
    q.push(MarketEvent(timestamp=t0 + timedelta(days=2), bar=_bar(t0 + timedelta(days=2))))
    q.push(MarketEvent(timestamp=t0, bar=_bar(t0)))
    q.push(MarketEvent(timestamp=t0 + timedelta(days=1), bar=_bar(t0 + timedelta(days=1))))

    out = [q.pop() for _ in range(3)]
    assert [e.timestamp for e in out] == [
        t0,
        t0 + timedelta(days=1),
        t0 + timedelta(days=2),
    ]


def test_priority_breaks_timestamp_ties() -> None:
    """Same timestamp: MARKET → SIGNAL → ORDER → FILL."""
    q = EventQueue()
    t = datetime(2024, 1, 1)
    order = Order(symbol=Symbol("X"), side=OrderSide.BUY, quantity=1)
    q.push(FillEvent(timestamp=t, symbol=Symbol("X"), side=OrderSide.BUY, quantity=1, price=10))
    q.push(OrderEvent(timestamp=t, order=order))
    q.push(SignalEvent(timestamp=t, symbol=Symbol("X"), target_pct=0.5))
    q.push(MarketEvent(timestamp=t, bar=_bar(t)))

    types = [q.pop().event_type for _ in range(4)]
    assert types == [EventType.MARKET, EventType.SIGNAL, EventType.ORDER, EventType.FILL]


def test_fifo_within_same_timestamp_and_type() -> None:
    """Stable order: insertion order is preserved when (ts, type) tie."""
    q = EventQueue()
    t = datetime(2024, 1, 1)
    sig_a = SignalEvent(timestamp=t, symbol=Symbol("A"), target_pct=0.1)
    sig_b = SignalEvent(timestamp=t, symbol=Symbol("B"), target_pct=0.2)
    sig_c = SignalEvent(timestamp=t, symbol=Symbol("C"), target_pct=0.3)
    q.push(sig_a)
    q.push(sig_b)
    q.push(sig_c)
    out = [q.pop() for _ in range(3)]
    assert [s.symbol for s in out] == [Symbol("A"), Symbol("B"), Symbol("C")]


def test_truthiness_and_len() -> None:
    q = EventQueue()
    assert not q
    assert len(q) == 0
    q.push(MarketEvent(timestamp=datetime(2024, 1, 1), bar=_bar(datetime(2024, 1, 1))))
    assert q
    assert len(q) == 1
