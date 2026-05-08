"""Shared fixtures: synthetic data feeds for fast, deterministic tests."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timedelta

import numpy as np
import pytest

from quantforge.core.events import MarketEvent
from quantforge.core.types import Bar, Symbol
from quantforge.data.base import DataFeed


class SyntheticFeed(DataFeed):
    """Deterministic price path — geometric Brownian motion with fixed seed."""

    def __init__(
        self,
        symbol: str = "TEST",
        n_bars: int = 250,
        start: datetime | None = None,
        s0: float = 100.0,
        mu: float = 0.0,
        sigma: float = 0.01,
        seed: int = 42,
        warmup: int = 50,
    ) -> None:
        self.symbols = [Symbol(symbol)]
        self.start = start if start is not None else datetime(2022, 1, 1)
        self.end = self.start + timedelta(days=n_bars)
        self._s0 = s0
        self._n = n_bars + warmup
        self._mu = mu
        self._sigma = sigma
        self._seed = seed
        self._warmup = warmup

    def _generate(self) -> list[Bar]:
        rng = np.random.default_rng(self._seed)
        rets = rng.normal(self._mu, self._sigma, size=self._n)
        prices = self._s0 * np.exp(np.cumsum(rets))
        bars: list[Bar] = []
        ts0 = self.start - timedelta(days=self._warmup)
        sym = self.symbols[0]
        for i, p in enumerate(prices):
            ts = ts0 + timedelta(days=i)
            o = float(p * (1 - 0.001))
            h = float(p * (1 + 0.003))
            l_ = float(p * (1 - 0.003))
            c = float(p)
            bars.append(
                Bar(symbol=sym, timestamp=ts, open=o, high=h, low=l_, close=c, volume=1_000_000)
            )
        return bars

    def stream(self) -> Iterator[MarketEvent]:
        for bar in self._generate():
            if bar.timestamp >= self.start:
                yield MarketEvent(timestamp=bar.timestamp, bar=bar)

    def warmup_bars(self, symbol: Symbol, n: int) -> list[float]:
        all_bars = self._generate()
        before = [b.close for b in all_bars if b.timestamp < self.start]
        return before[-n:]


@pytest.fixture
def synthetic_feed() -> SyntheticFeed:
    return SyntheticFeed()


@pytest.fixture
def trending_feed() -> SyntheticFeed:
    return SyntheticFeed(mu=0.001, sigma=0.005, seed=7)
