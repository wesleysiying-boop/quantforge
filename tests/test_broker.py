"""Simulated broker tests — fills, slippage, commission, look-ahead protection."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from quantforge.core.events import OrderEvent
from quantforge.core.types import Bar, Order, OrderSide, OrderStatus, OrderType, Symbol
from quantforge.execution.broker import SimulatedBroker
from quantforge.execution.commission import PerTradeCommission
from quantforge.execution.slippage import FixedBpsSlippage, NoSlippage


def _bar(
    t: datetime,
    open_: float = 100.0,
    close: float = 100.0,
    high: float | None = None,
    low: float | None = None,
) -> Bar:
    return Bar(
        symbol=Symbol("X"),
        timestamp=t,
        open=open_,
        high=high if high is not None else max(open_, close) * 1.01,
        low=low if low is not None else min(open_, close) * 0.99,
        close=close,
        volume=1_000_000,
    )


def test_market_order_fills_next_bar_at_open() -> None:
    broker = SimulatedBroker(slippage=NoSlippage(), commission=PerTradeCommission(fee=0.0))
    t0 = datetime(2024, 1, 1)
    b0 = _bar(t0, open_=100, close=101)
    b1 = _bar(t0 + timedelta(days=1), open_=102, close=103)

    broker.on_bar(b0)  # broker now knows about b0
    order = Order(symbol=Symbol("X"), side=OrderSide.BUY, quantity=10)
    broker.submit(OrderEvent(timestamp=t0, order=order))

    fills_b0 = broker.on_bar(b0)  # same bar — should NOT fill
    assert fills_b0 == []

    fills_b1 = broker.on_bar(b1)
    assert len(fills_b1) == 1
    assert fills_b1[0].price == 102.0
    assert order.status is OrderStatus.FILLED


def test_no_same_bar_lookahead() -> None:
    """Order submitted during bar t MUST NOT fill on bar t."""
    broker = SimulatedBroker()
    t = datetime(2024, 1, 1)
    b = _bar(t)
    broker.on_bar(b)
    order = Order(symbol=Symbol("X"), side=OrderSide.BUY, quantity=1)
    broker.submit(OrderEvent(timestamp=t, order=order))
    fills = broker.on_bar(b)
    assert fills == []


def test_slippage_increases_buy_price() -> None:
    broker = SimulatedBroker(slippage=FixedBpsSlippage(bps=10), commission=PerTradeCommission(0.0))
    t = datetime(2024, 1, 1)
    broker.on_bar(_bar(t))
    order = Order(symbol=Symbol("X"), side=OrderSide.BUY, quantity=10)
    broker.submit(OrderEvent(timestamp=t, order=order))
    fills = broker.on_bar(_bar(t + timedelta(days=1), open_=100, close=100))
    assert fills[0].price == pytest.approx(100 * (1 + 10 / 10_000))


def test_slippage_decreases_sell_price() -> None:
    broker = SimulatedBroker(slippage=FixedBpsSlippage(bps=10), commission=PerTradeCommission(0.0))
    t = datetime(2024, 1, 1)
    broker.on_bar(_bar(t))
    order = Order(symbol=Symbol("X"), side=OrderSide.SELL, quantity=10)
    broker.submit(OrderEvent(timestamp=t, order=order))
    fills = broker.on_bar(_bar(t + timedelta(days=1), open_=100, close=100))
    assert fills[0].price == pytest.approx(100 * (1 - 10 / 10_000))


def test_limit_order_fills_when_price_crosses() -> None:
    broker = SimulatedBroker(slippage=NoSlippage(), commission=PerTradeCommission(0.0))
    t = datetime(2024, 1, 1)
    broker.on_bar(_bar(t))
    order = Order(
        symbol=Symbol("X"),
        side=OrderSide.BUY,
        quantity=10,
        order_type=OrderType.LIMIT,
        limit_price=99.0,
    )
    broker.submit(OrderEvent(timestamp=t, order=order))
    # Next bar low is below 99 → fills.
    fills = broker.on_bar(_bar(t + timedelta(days=1), open_=100, close=100, high=101, low=98))
    assert len(fills) == 1
    assert fills[0].price == 99.0


def test_limit_order_rests_when_price_does_not_cross() -> None:
    broker = SimulatedBroker(slippage=NoSlippage(), commission=PerTradeCommission(0.0))
    t = datetime(2024, 1, 1)
    broker.on_bar(_bar(t))
    order = Order(
        symbol=Symbol("X"),
        side=OrderSide.BUY,
        quantity=10,
        order_type=OrderType.LIMIT,
        limit_price=90.0,
    )
    broker.submit(OrderEvent(timestamp=t, order=order))
    fills = broker.on_bar(_bar(t + timedelta(days=1), open_=100, close=100, high=101, low=99))
    assert fills == []
    assert order.status is OrderStatus.PENDING


def test_submit_with_no_history_rejects() -> None:
    """Submitting before the broker has seen any bar must reject the order."""
    broker = SimulatedBroker()
    order = Order(symbol=Symbol("X"), side=OrderSide.BUY, quantity=1)
    broker.submit(OrderEvent(timestamp=datetime(2024, 1, 1), order=order))
    assert order.status is OrderStatus.REJECTED
