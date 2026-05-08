"""SMA crossover backtest on SPY — the canonical "does this thing run?" demo.

Run:  python examples/01_sma_crossover.py
"""

from __future__ import annotations

from quantforge import Backtester, YFinanceFeed
from quantforge.strategy.library.sma_crossover import SmaCrossover


def main() -> None:
    feed = YFinanceFeed(
        symbols=["SPY"],
        start="2018-01-01",
        end="2024-12-31",
    )
    strategy = SmaCrossover(symbols=["SPY"], fast=20, slow=50)
    bt = Backtester(feed=feed, strategy=strategy, starting_cash=100_000)

    result = bt.run()

    print(f"\nStrategy: {type(strategy).__name__}")
    print(f"Period:   {result.metadata['start']} → {result.metadata['end']}")
    print(f"Bars:     {result.bars_processed}\n")
    print(result.summary())


if __name__ == "__main__":
    main()
