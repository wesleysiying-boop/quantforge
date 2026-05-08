"""Event-driven backtest engine.

The single event loop:

    while events_pending:
        event = queue.pop()
        clock.advance_to(event.timestamp)
        match event.type:
            MARKET → broker.on_bar() → push fills
                   → strategy.on_bar() → push signals
            SIGNAL → risk.size()      → push order
            ORDER  → broker.submit()
            FILL   → portfolio.apply_fill()

Determinism comes from `EventQueue`'s stable (timestamp, type, seq) ordering.
The same strategy, on the same data, MUST produce the same equity curve on
every run — this is asserted in `tests/test_determinism.py`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from quantforge.analytics.metrics import PerformanceReport, build_report
from quantforge.core.clock import SimulationClock
from quantforge.core.events import (
    EventType,
    FillEvent,
    MarketEvent,
    SignalEvent,
)
from quantforge.core.queue import EventQueue
from quantforge.core.types import Symbol
from quantforge.execution.broker import SimulatedBroker
from quantforge.portfolio.portfolio import Portfolio
from quantforge.portfolio.risk import RiskModel, TargetWeightRisk
from quantforge.strategy.base import Context, Strategy

if TYPE_CHECKING:
    from quantforge.data.base import DataFeed
    from quantforge.execution.broker import Broker

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class BacktestResult:
    equity_curve: pd.Series
    fills: list[FillEvent]
    report: PerformanceReport
    portfolio: Portfolio
    bars_processed: int = 0
    signals_generated: int = 0
    orders_submitted: int = 0
    metadata: dict[str, object] = field(default_factory=dict)

    def summary(self) -> str:
        return self.report.pretty()

    def to_frame(self) -> pd.DataFrame:
        return self.equity_curve.to_frame("equity").assign(
            returns=self.equity_curve.pct_change().fillna(0.0),
            drawdown=self.equity_curve / self.equity_curve.cummax() - 1.0,
        )


class Backtester:
    """Wires data feed → strategy → risk → broker → portfolio."""

    def __init__(
        self,
        feed: DataFeed,
        strategy: Strategy,
        starting_cash: float = 100_000.0,
        broker: Broker | None = None,
        risk: RiskModel | None = None,
        periods_per_year: int = 252,
        risk_free: float = 0.0,
    ) -> None:
        self.feed = feed
        self.strategy = strategy
        self.broker = broker if broker is not None else SimulatedBroker()
        self.risk = risk if risk is not None else TargetWeightRisk()
        self.portfolio = Portfolio(starting_cash=starting_cash)
        self.queue = EventQueue()
        self.context = Context()
        self.periods_per_year = periods_per_year
        self.risk_free = risk_free

        self._last_price: dict[Symbol, float] = {}
        self._clock: SimulationClock | None = None

    def run(self) -> BacktestResult:
        bars = 0
        signals = 0
        orders = 0
        first_ts: datetime | None = None
        last_ts: datetime | None = None

        for market_event in self.feed.stream():
            self.queue.push(market_event)
            if first_ts is None:
                first_ts = market_event.timestamp
                self._clock = SimulationClock(first_ts)
                self.strategy.on_start(self.context)

            while self.queue:
                event = self.queue.pop()
                assert self._clock is not None
                self._clock.advance_to(event.timestamp)

                if event.event_type is EventType.MARKET:
                    bars += self._handle_market(event)  # type: ignore[arg-type]
                elif event.event_type is EventType.SIGNAL:
                    if self._handle_signal(event):  # type: ignore[arg-type]
                        signals += 1
                        orders += 1
                elif event.event_type is EventType.ORDER:
                    self.broker.submit(event)  # type: ignore[arg-type]
                elif event.event_type is EventType.FILL:
                    self.portfolio.apply_fill(event)  # type: ignore[arg-type]

            last_ts = market_event.timestamp

        if first_ts is not None:
            self.strategy.on_finish(self.context)

        equity_curve = self._build_equity_curve()
        report = build_report(
            equity_curve.to_numpy(),
            n_fills=len(self.portfolio.fills),
            periods_per_year=self.periods_per_year,
            risk_free=self.risk_free,
        )
        return BacktestResult(
            equity_curve=equity_curve,
            fills=self.portfolio.fills,
            report=report,
            portfolio=self.portfolio,
            bars_processed=bars,
            signals_generated=signals,
            orders_submitted=orders,
            metadata={
                "start": first_ts,
                "end": last_ts,
                "starting_cash": self.portfolio.starting_cash,
                "strategy": type(self.strategy).__name__,
            },
        )

    def _handle_market(self, event: MarketEvent) -> int:
        if event.bar is None:
            return 0
        bar = event.bar

        # 1. Apply any fills from orders submitted on prior bars BEFORE the
        #    strategy sees the new bar — otherwise the strategy reads stale
        #    portfolio state.
        for fill in self.broker.on_bar(bar):
            self.portfolio.apply_fill(fill)

        # 2. Mark to bar close.
        self._last_price[bar.symbol] = bar.close
        self.portfolio.mark(bar.symbol, bar.close)

        # 3. Strategy reacts to the bar — may emit signals (queued).
        self.context._ingest(bar)
        self.strategy.on_bar(self.context, bar)
        for sig in self.context._drain_signals():
            self.queue.push(sig)

        # 4. Snapshot equity at bar close.
        self.portfolio.record(bar.timestamp)
        return 1

    def _handle_signal(self, event: SignalEvent) -> bool:
        order_event = self.risk.size(event, self.portfolio, self._last_price)
        if order_event is None:
            return False
        self.queue.push(order_event)
        return True

    def _build_equity_curve(self) -> pd.Series:
        if not self.portfolio.history:
            return pd.Series(dtype=np.float64, name="equity")
        idx = [s.timestamp for s in self.portfolio.history]
        vals = [s.equity for s in self.portfolio.history]
        return pd.Series(vals, index=pd.DatetimeIndex(idx), name="equity")
