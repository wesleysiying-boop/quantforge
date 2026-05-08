"""DataFeed abstract base class.

A `DataFeed` yields `MarketEvent`s in strict timestamp order. The same
interface is implemented by historical replay sources (yfinance, Polygon,
CSV) and by live websocket sources — the engine doesn't care which.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from datetime import datetime

from quantforge.core.events import MarketEvent
from quantforge.core.types import Symbol


class DataFeed(ABC):
    """Stream of market events for one or more symbols, in time order."""

    symbols: list[Symbol]
    start: datetime | None
    end: datetime | None

    @abstractmethod
    def stream(self) -> Iterator[MarketEvent]:
        """Yield MarketEvents in non-decreasing timestamp order.

        Implementations MUST guarantee monotonic timestamps; the engine relies
        on this for correctness of the event-queue ordering invariants.
        """
        ...

    @abstractmethod
    def warmup_bars(self, symbol: Symbol, n: int) -> list[float]:
        """Return up to `n` bars of close prices ending just before `start`.

        Strategies that compute indicators over a lookback window need history
        before the first bar of the test period — this provides it without
        introducing a survivorship-biased peek into the test window.
        """
        ...
