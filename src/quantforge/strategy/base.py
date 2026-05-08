"""Strategy ABC + Context.

Strategies are intentionally narrow: they receive bars and emit signals.
They never see the broker, portfolio, queue, or wall clock — only a
`Context` object that exposes a curated, read-only-ish API.

This separation is what makes strategies trivially unit-testable: feed a
context with mocked history, assert the signals are what you expect.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict, deque
from datetime import datetime
from typing import Final

import numpy as np

from quantforge.core.events import SignalEvent
from quantforge.core.types import Bar, Symbol


class Context:
    """Strategy-facing API.

    Holds a rolling window of bar history per symbol and acts as a sink for
    `target_pct` calls. Backtester drains `pending_signals` after each
    `on_bar` invocation.
    """

    def __init__(self, history_size: int = 1024) -> None:
        self._history: dict[Symbol, deque[Bar]] = defaultdict(lambda: deque(maxlen=history_size))
        self.pending_signals: list[SignalEvent] = []
        self._now: datetime | None = None
        self.history_size: Final = history_size

    @property
    def now(self) -> datetime:
        if self._now is None:
            raise RuntimeError("Context.now read before any bar was processed")
        return self._now

    def _ingest(self, bar: Bar) -> None:
        self._history[bar.symbol].append(bar)
        self._now = bar.timestamp

    def _drain_signals(self) -> list[SignalEvent]:
        out, self.pending_signals = self.pending_signals, []
        return out

    def history(self, symbol: Symbol, field: str = "close", n: int | None = None) -> np.ndarray:
        """Return the last `n` values of `field` for `symbol`. Defaults to all."""
        bars = self._history.get(symbol)
        if not bars:
            return np.array([], dtype=np.float64)
        slice_ = list(bars)[-n:] if n is not None else list(bars)
        return np.array([getattr(b, field) for b in slice_], dtype=np.float64)

    def target_pct(self, symbol: Symbol, weight: float, *, strength: float = 1.0) -> None:
        """Express intent: `symbol` should be `weight` of equity. weight in [-1, 1]."""
        if not -1.0 <= weight <= 1.0:
            raise ValueError(f"weight must be in [-1, 1], got {weight}")
        self.pending_signals.append(
            SignalEvent(
                timestamp=self.now,
                symbol=symbol,
                target_pct=weight,
                strength=strength,
            )
        )


class Strategy(ABC):
    """Override `on_bar` (and optionally `on_start` / `on_finish`)."""

    def on_start(self, ctx: Context) -> None:  # noqa: B027 - intentional empty hook
        """Called once before the first bar."""

    def on_finish(self, ctx: Context) -> None:  # noqa: B027 - intentional empty hook
        """Called once after the last bar."""

    @abstractmethod
    def on_bar(self, ctx: Context, bar: Bar) -> None: ...
