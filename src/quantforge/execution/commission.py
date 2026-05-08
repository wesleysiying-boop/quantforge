"""Commission models.

Commission is symmetric across buys and sells and is debited from cash on
fill. Three common shapes:

* `PerShareCommission`  - $/share, with min and max per trade (Interactive Brokers fixed schedule).
* `PerTradeCommission`  - flat $/trade.
* `TieredCommission`    - bps of notional with floor (typical for crypto exchanges).
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class CommissionModel(ABC):
    @abstractmethod
    def calculate(self, quantity: int, fill_price: float) -> float: ...


class PerShareCommission(CommissionModel):
    def __init__(self, rate: float = 0.005, minimum: float = 1.0, maximum: float = 1e9) -> None:
        self.rate = rate
        self.minimum = minimum
        self.maximum = maximum

    def calculate(self, quantity: int, fill_price: float) -> float:
        gross = quantity * self.rate
        return min(self.maximum, max(self.minimum, gross))


class PerTradeCommission(CommissionModel):
    def __init__(self, fee: float = 1.0) -> None:
        self.fee = fee

    def calculate(self, quantity: int, fill_price: float) -> float:
        return self.fee


class TieredCommission(CommissionModel):
    """Notional-based: bps of (qty * price), with a per-trade floor."""

    def __init__(self, bps: float = 10.0, minimum: float = 0.0) -> None:
        self.bps = bps
        self.minimum = minimum

    def calculate(self, quantity: int, fill_price: float) -> float:
        notional = quantity * fill_price
        return max(self.minimum, notional * self.bps / 10_000.0)
