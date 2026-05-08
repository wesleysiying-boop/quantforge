"""Broker — order routing + fill simulation.

`Broker` is the abstract interface; `SimulatedBroker` is the backtest
implementation. A live broker (Alpaca, Binance) implements the same interface
but routes orders to the exchange and listens for fill events on a websocket.

Fill semantics for the simulated broker:

* Market orders submitted on bar `t`'s close fill at bar `t+1`'s open, with
  slippage applied against the open price. This eliminates same-bar look-ahead.
* Limit orders fill at the limit price if the next bar's range crosses it.
* Stop orders trigger if the next bar's range touches the stop, then fill at
  the stop price (no further adverse slippage modeled).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass

from quantforge.core.events import FillEvent, OrderEvent
from quantforge.core.types import Bar, Order, OrderSide, OrderStatus, OrderType, Symbol
from quantforge.execution.commission import CommissionModel, PerShareCommission
from quantforge.execution.slippage import FixedBpsSlippage, SlippageModel


@dataclass(slots=True)
class _PendingOrder:
    order: Order
    submitted_at_bar: Bar


class Broker(ABC):
    @abstractmethod
    def submit(self, event: OrderEvent) -> None: ...

    @abstractmethod
    def on_bar(self, bar: Bar) -> list[FillEvent]: ...


class SimulatedBroker(Broker):
    """Bar-driven simulated broker with no same-bar look-ahead."""

    def __init__(
        self,
        slippage: SlippageModel | None = None,
        commission: CommissionModel | None = None,
    ) -> None:
        self.slippage = slippage if slippage is not None else FixedBpsSlippage(bps=5.0)
        self.commission = commission if commission is not None else PerShareCommission()
        self._pending: dict[Symbol, list[_PendingOrder]] = defaultdict(list)
        self._last_bar: dict[Symbol, Bar] = {}

    def submit(self, event: OrderEvent) -> None:
        order = event.order
        assert order is not None
        last = self._last_bar.get(order.symbol)
        if last is None:
            order.status = OrderStatus.REJECTED
            return
        self._pending[order.symbol].append(_PendingOrder(order=order, submitted_at_bar=last))

    def on_bar(self, bar: Bar) -> list[FillEvent]:
        """Process pending orders for this symbol against the new bar."""
        self._last_bar[bar.symbol] = bar
        fills: list[FillEvent] = []
        still_pending: list[_PendingOrder] = []

        for po in self._pending[bar.symbol]:
            # Orders submitted during bar `t` can only fill on bar `t+1` or later.
            if po.submitted_at_bar.timestamp >= bar.timestamp:
                still_pending.append(po)
                continue

            fill = self._try_fill(po.order, bar)
            if fill is None:
                still_pending.append(po)
            else:
                fills.append(fill)

        self._pending[bar.symbol] = still_pending
        return fills

    def _try_fill(self, order: Order, bar: Bar) -> FillEvent | None:
        if order.is_done:
            return None

        side = order.side
        qty = order.remaining

        if order.order_type is OrderType.MARKET:
            mid = bar.open
            fill_price = self.slippage.apply(bar, side, qty, mid)
        elif order.order_type is OrderType.LIMIT:
            assert order.limit_price is not None
            crossed = (side is OrderSide.BUY and bar.low <= order.limit_price) or (
                side is OrderSide.SELL and bar.high >= order.limit_price
            )
            if not crossed:
                return None
            fill_price = order.limit_price  # No adverse slippage on a marketable limit.
        elif order.order_type is OrderType.STOP:
            assert order.stop_price is not None
            triggered = (side is OrderSide.BUY and bar.high >= order.stop_price) or (
                side is OrderSide.SELL and bar.low <= order.stop_price
            )
            if not triggered:
                return None
            fill_price = self.slippage.apply(bar, side, qty, order.stop_price)
        else:  # pragma: no cover
            raise ValueError(f"Unknown order type: {order.order_type}")

        commission = self.commission.calculate(qty, fill_price)
        order.quantity_filled += qty
        order.avg_fill_price = fill_price
        order.status = OrderStatus.FILLED

        return FillEvent(
            timestamp=bar.timestamp,
            symbol=bar.symbol,
            side=side,
            quantity=qty,
            price=fill_price,
            commission=commission,
            order_id=order.order_id,
        )
