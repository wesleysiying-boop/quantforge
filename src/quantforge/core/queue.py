"""In-order event queue.

Wraps `heapq` to give a stable, typed priority queue. Events ordered by
(timestamp, event_type, sequence) — sequence is a monotonically increasing
counter that breaks ties between events with identical (timestamp, type)
deterministically by insertion order.
"""

from __future__ import annotations

import heapq
from itertools import count
from typing import Final

from quantforge.core.events import Event


class EventQueue:
    __slots__ = ("_heap", "_seq")

    def __init__(self) -> None:
        self._heap: list[tuple[float, int, int, int, Event]] = []
        self._seq: Final = count()

    def push(self, event: Event) -> None:
        ts = event.timestamp.timestamp()
        seq = next(self._seq)
        # heapq compares tuples element-wise; we never compare on the Event itself.
        # The leading int(ts*1e9) keeps integer-only comparisons in the hot path.
        heapq.heappush(self._heap, (ts, int(event.event_type), seq, seq, event))

    def pop(self) -> Event:
        return heapq.heappop(self._heap)[-1]

    def __len__(self) -> int:
        return len(self._heap)

    def __bool__(self) -> bool:
        return bool(self._heap)
