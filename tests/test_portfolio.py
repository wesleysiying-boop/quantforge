"""Portfolio accounting tests."""

from __future__ import annotations

from datetime import datetime

import pytest

from quantforge.core.events import FillEvent
from quantforge.core.types import OrderSide, Symbol
from quantforge.portfolio.portfolio import Portfolio


def _fill(side: OrderSide, qty: int, price: float, commission: float = 0.0) -> FillEvent:
    return FillEvent(
        timestamp=datetime(2024, 1, 1),
        symbol=Symbol("X"),
        side=side,
        quantity=qty,
        price=price,
        commission=commission,
    )


def test_buy_decreases_cash_increases_position() -> None:
    p = Portfolio(starting_cash=100_000)
    p.apply_fill(_fill(OrderSide.BUY, 100, 50.0))
    assert p.cash == 95_000
    assert p.position(Symbol("X")).quantity == 100


def test_sell_increases_cash() -> None:
    p = Portfolio(starting_cash=100_000)
    p.apply_fill(_fill(OrderSide.BUY, 100, 50.0))
    p.apply_fill(_fill(OrderSide.SELL, 100, 60.0))
    assert p.cash == 101_000


def test_commission_charged_to_cash() -> None:
    p = Portfolio(starting_cash=100_000)
    p.apply_fill(_fill(OrderSide.BUY, 100, 50.0, commission=1.0))
    assert p.cash == 95_000 - 1.0


def test_equity_invariant_round_trip() -> None:
    """Buy then sell at same price (no commission) leaves equity unchanged."""
    p = Portfolio(starting_cash=100_000)
    p.apply_fill(_fill(OrderSide.BUY, 100, 50.0))
    p.mark(Symbol("X"), 50.0)
    p.apply_fill(_fill(OrderSide.SELL, 100, 50.0))
    assert p.equity() == pytest.approx(100_000)


def test_record_appends_history() -> None:
    p = Portfolio(starting_cash=100_000)
    p.apply_fill(_fill(OrderSide.BUY, 100, 50.0))
    p.mark(Symbol("X"), 55.0)
    snap = p.record(datetime(2024, 1, 2))
    assert snap.cash == 95_000
    assert snap.market_value == pytest.approx(5_500)
    assert snap.equity == pytest.approx(100_500)
    assert len(p.history) == 1
