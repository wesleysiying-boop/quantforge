"""Slippage models.

Slippage is the difference between the price assumed at signal time and the
price actually realized. Modeling it conservatively is the single biggest
factor in keeping a backtest honest — strategies that look brilliant under
zero-slippage often disappear once realistic costs are charged.

Three models, in increasing fidelity:

* `FixedBpsSlippage`  - flat basis-point haircut. Crude but auditable.
* `VolumeParticipationSlippage` - linear in (size / bar volume).
* `SquareRootImpactSlippage`    - Almgren-style sqrt(participation) impact.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

from quantforge.core.types import Bar, OrderSide


class SlippageModel(ABC):
    @abstractmethod
    def apply(self, bar: Bar, side: OrderSide, quantity: int, mid_price: float) -> float:
        """Return the realized fill price after slippage."""
        ...


class NoSlippage(SlippageModel):
    """Zero-slippage baseline — useful for unit tests, never for research."""

    def apply(self, bar: Bar, side: OrderSide, quantity: int, mid_price: float) -> float:
        return mid_price


class FixedBpsSlippage(SlippageModel):
    """Symmetric cost of `bps` basis points applied against the trade direction."""

    def __init__(self, bps: float = 5.0) -> None:
        if bps < 0:
            raise ValueError("bps must be non-negative")
        self.bps = bps

    def apply(self, bar: Bar, side: OrderSide, quantity: int, mid_price: float) -> float:
        return mid_price * (1.0 + side.sign * self.bps / 10_000.0)


class VolumeParticipationSlippage(SlippageModel):
    """Linear impact: slippage_bps = k * (qty / bar_volume) * 10_000.

    `k` is the impact coefficient - typical empirical values are 0.1 to 0.5
    for liquid US equities at the daily horizon.
    """

    def __init__(self, k: float = 0.1, min_bps: float = 1.0) -> None:
        self.k = k
        self.min_bps = min_bps

    def apply(self, bar: Bar, side: OrderSide, quantity: int, mid_price: float) -> float:
        participation = 1.0 if bar.volume <= 0 else quantity / bar.volume
        bps = max(self.min_bps, self.k * participation * 10_000.0)
        return mid_price * (1.0 + side.sign * bps / 10_000.0)


class SquareRootImpactSlippage(SlippageModel):
    """Almgren-style sqrt-impact: slippage_bps = sigma * eta * sqrt(qty / ADV) * 10_000.

    * `sigma` is the per-bar return volatility (passed in or estimated externally).
    * `eta` is the impact coefficient; literature often cites ~1 for institutional flow.
    """

    def __init__(self, sigma: float = 0.01, eta: float = 1.0, min_bps: float = 1.0) -> None:
        self.sigma = sigma
        self.eta = eta
        self.min_bps = min_bps

    def apply(self, bar: Bar, side: OrderSide, quantity: int, mid_price: float) -> float:
        participation = 1.0 if bar.volume <= 0 else quantity / bar.volume
        bps = max(self.min_bps, self.sigma * self.eta * math.sqrt(participation) * 10_000.0)
        return mid_price * (1.0 + side.sign * bps / 10_000.0)
