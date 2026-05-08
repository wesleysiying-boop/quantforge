"""quantforge — event-driven multi-market backtesting and paper-trading framework."""

from __future__ import annotations

from quantforge.core.types import Bar, Order, OrderSide, OrderType, Position, Tick
from quantforge.data.yfinance_feed import YFinanceFeed
from quantforge.engine.backtester import Backtester, BacktestResult
from quantforge.strategy.base import Context, Strategy

__version__ = "0.1.0"

__all__ = [
    "BacktestResult",
    "Backtester",
    "Bar",
    "Context",
    "Order",
    "OrderSide",
    "OrderType",
    "Position",
    "Strategy",
    "Tick",
    "YFinanceFeed",
    "__version__",
]
