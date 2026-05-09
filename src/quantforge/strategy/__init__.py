"""Strategy framework + bundled strategy library."""

from quantforge.strategy.base import Context, Strategy
from quantforge.strategy.indicators import ema, returns, rolling_zscore, sma
from quantforge.strategy.library.mean_reversion import MeanReversionZScore
from quantforge.strategy.library.sma_crossover import SmaCrossover
from quantforge.strategy.regime_filter import RegimeClassifier, RegimeFilteredStrategy

__all__ = [
    "Context",
    "MeanReversionZScore",
    "RegimeClassifier",
    "RegimeFilteredStrategy",
    "SmaCrossover",
    "Strategy",
    "ema",
    "returns",
    "rolling_zscore",
    "sma",
]
