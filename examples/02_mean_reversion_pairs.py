"""Mean-reversion z-score on a single ETF.

Looks for closes that deviate more than 2 standard deviations from a 20-day
mean and bets on reversion. This is intentionally a 'simple' setup — the
goal is to demonstrate the strategy framework, not to claim alpha.

Run:  python examples/02_mean_reversion_pairs.py
"""

from __future__ import annotations

from quantforge import Backtester, YFinanceFeed
from quantforge.execution.broker import SimulatedBroker
from quantforge.execution.commission import TieredCommission
from quantforge.execution.slippage import VolumeParticipationSlippage
from quantforge.strategy.library.mean_reversion import MeanReversionZScore


def main() -> None:
    feed = YFinanceFeed(symbols=["IWM"], start="2018-01-01", end="2024-12-31")
    strategy = MeanReversionZScore(symbols=["IWM"], lookback=20, entry_z=2.0, exit_z=0.5)
    broker = SimulatedBroker(
        slippage=VolumeParticipationSlippage(k=0.1, min_bps=2.0),
        commission=TieredCommission(bps=2.0, minimum=1.0),
    )
    bt = Backtester(feed=feed, strategy=strategy, starting_cash=100_000, broker=broker)
    result = bt.run()

    print(f"\nStrategy: {type(strategy).__name__}")
    print(f"Period:   {result.metadata['start']} → {result.metadata['end']}\n")
    print(result.summary())


if __name__ == "__main__":
    main()
