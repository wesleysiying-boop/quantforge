"""Portfolio — holdings, cash, equity curve.

Holds the canonical state that drives mark-to-market, P&L, and equity curve.
The portfolio is updated only by `apply_fill` (on FillEvent) and `mark`
(on MarketEvent); it never sees orders or signals directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from quantforge.core.events import FillEvent
from quantforge.core.types import Position, Symbol


@dataclass(slots=True)
class EquitySnapshot:
    timestamp: datetime
    cash: float
    market_value: float
    equity: float
    positions: dict[Symbol, int]


class Portfolio:
    """Cash + positions + history. The single source of truth for performance."""

    def __init__(self, starting_cash: float = 100_000.0) -> None:
        self.starting_cash = starting_cash
        self._cash = starting_cash
        self._positions: dict[Symbol, Position] = {}
        self._last_price: dict[Symbol, float] = {}
        self._history: list[EquitySnapshot] = []
        self._fills: list[FillEvent] = []

    @property
    def cash(self) -> float:
        return self._cash

    @property
    def positions(self) -> dict[Symbol, Position]:
        return self._positions

    @property
    def history(self) -> list[EquitySnapshot]:
        return self._history

    @property
    def fills(self) -> list[FillEvent]:
        return self._fills

    def position(self, symbol: Symbol) -> Position:
        if symbol not in self._positions:
            self._positions[symbol] = Position(symbol=symbol)
        return self._positions[symbol]

    def market_value(self) -> float:
        return sum(
            pos.market_value(self._last_price.get(sym, pos.avg_cost))
            for sym, pos in self._positions.items()
        )

    def equity(self) -> float:
        return self._cash + self.market_value()

    def apply_fill(self, fill: FillEvent) -> None:
        pos = self.position(fill.symbol)
        pos.apply_fill(fill.side, fill.quantity, fill.price)
        self._cash += fill.net_value
        self._last_price[fill.symbol] = fill.price
        self._fills.append(fill)

    def mark(self, symbol: Symbol, price: float) -> None:
        self._last_price[symbol] = price

    def record(self, timestamp: datetime) -> EquitySnapshot:
        snap = EquitySnapshot(
            timestamp=timestamp,
            cash=self._cash,
            market_value=self.market_value(),
            equity=self.equity(),
            positions={sym: pos.quantity for sym, pos in self._positions.items() if pos.quantity},
        )
        self._history.append(snap)
        return snap
