"""Tests for core domain primitives."""

from __future__ import annotations

from datetime import datetime

import pytest
from hypothesis import given
from hypothesis import strategies as st

from quantforge.core.types import (
    Bar,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    Symbol,
)


class TestBar:
    def test_valid(self) -> None:
        Bar(Symbol("X"), datetime(2024, 1, 1), open=10, high=12, low=9, close=11, volume=100)

    def test_invalid_ohlc(self) -> None:
        with pytest.raises(ValueError, match="OHLC"):
            Bar(Symbol("X"), datetime(2024, 1, 1), open=10, high=9, low=11, close=10.5, volume=100)

    def test_negative_volume(self) -> None:
        with pytest.raises(ValueError, match="volume"):
            Bar(Symbol("X"), datetime(2024, 1, 1), open=10, high=11, low=9, close=10, volume=-1)


class TestOrder:
    def test_market_order_ok(self) -> None:
        o = Order(symbol=Symbol("X"), side=OrderSide.BUY, quantity=10)
        assert o.status is OrderStatus.PENDING
        assert o.remaining == 10

    def test_limit_requires_price(self) -> None:
        with pytest.raises(ValueError, match="limit_price"):
            Order(symbol=Symbol("X"), side=OrderSide.BUY, quantity=1, order_type=OrderType.LIMIT)

    def test_zero_qty(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            Order(symbol=Symbol("X"), side=OrderSide.BUY, quantity=0)


class TestPosition:
    def test_open_long(self) -> None:
        p = Position(Symbol("X"))
        p.apply_fill(OrderSide.BUY, 100, 50.0)
        assert p.quantity == 100
        assert p.avg_cost == 50.0
        assert p.realized_pnl == 0.0

    def test_add_to_long(self) -> None:
        p = Position(Symbol("X"))
        p.apply_fill(OrderSide.BUY, 100, 50.0)
        p.apply_fill(OrderSide.BUY, 100, 60.0)
        assert p.quantity == 200
        assert p.avg_cost == pytest.approx(55.0)

    def test_close_long_realizes_pnl(self) -> None:
        p = Position(Symbol("X"))
        p.apply_fill(OrderSide.BUY, 100, 50.0)
        p.apply_fill(OrderSide.SELL, 100, 60.0)
        assert p.quantity == 0
        assert p.realized_pnl == pytest.approx(1000.0)

    def test_partial_close_long(self) -> None:
        p = Position(Symbol("X"))
        p.apply_fill(OrderSide.BUY, 100, 50.0)
        p.apply_fill(OrderSide.SELL, 40, 60.0)
        assert p.quantity == 60
        assert p.avg_cost == pytest.approx(50.0)
        assert p.realized_pnl == pytest.approx(400.0)

    def test_flip_long_to_short(self) -> None:
        p = Position(Symbol("X"))
        p.apply_fill(OrderSide.BUY, 100, 50.0)
        p.apply_fill(OrderSide.SELL, 150, 60.0)
        assert p.quantity == -50
        assert p.avg_cost == pytest.approx(60.0)
        assert p.realized_pnl == pytest.approx(1000.0)

    def test_unrealized_pnl(self) -> None:
        p = Position(Symbol("X"))
        p.apply_fill(OrderSide.BUY, 100, 50.0)
        assert p.unrealized_pnl(55.0) == pytest.approx(500.0)

    @given(
        qty1=st.integers(min_value=1, max_value=1000),
        qty2=st.integers(min_value=1, max_value=1000),
        p1=st.floats(min_value=1.0, max_value=1000.0),
        p2=st.floats(min_value=1.0, max_value=1000.0),
    )
    def test_open_then_close_pnl_equals_price_diff(
        self, qty1: int, qty2: int, p1: float, p2: float
    ) -> None:
        """Closing the entire position must realize qty * (sell_price - buy_price)."""
        p = Position(Symbol("X"))
        p.apply_fill(OrderSide.BUY, qty1, p1)
        # Close exactly the open quantity.
        p.apply_fill(OrderSide.SELL, qty1, p2)
        assert p.quantity == 0
        assert p.realized_pnl == pytest.approx(qty1 * (p2 - p1), rel=1e-9, abs=1e-6)
