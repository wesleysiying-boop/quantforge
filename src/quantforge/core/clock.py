"""Clock abstraction.

The same engine drives backtest and live modes; the only difference is which
clock it holds. Strategies and brokers MUST get `now()` from the clock — never
from `datetime.now()` directly — so that backtests are deterministic and
replayable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import UTC, datetime


class Clock(ABC):
    @abstractmethod
    def now(self) -> datetime: ...


class SimulationClock(Clock):
    """Clock advanced explicitly by the backtest event loop."""

    __slots__ = ("_now",)

    def __init__(self, start: datetime) -> None:
        self._now = start

    def now(self) -> datetime:
        return self._now

    def advance_to(self, t: datetime) -> None:
        if t < self._now:
            raise ValueError(f"Clock cannot move backwards: {self._now} -> {t}")
        self._now = t


class WallClock(Clock):
    """Clock that returns real wall-clock time — used in live / paper modes."""

    def now(self) -> datetime:
        return datetime.now(UTC)
