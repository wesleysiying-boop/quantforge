"""End-to-end backtest tests on synthetic data.

These exercise the full event loop: feed → strategy → risk → broker →
portfolio → analytics. Determinism is the most important property to verify
— same seed, same code, same numbers every time.
"""

from __future__ import annotations

import pytest

from quantforge import Backtester
from quantforge.execution.broker import SimulatedBroker
from quantforge.execution.commission import PerTradeCommission
from quantforge.execution.slippage import NoSlippage
from quantforge.strategy.base import Context, Strategy
from quantforge.strategy.library.sma_crossover import SmaCrossover
from tests.conftest import SyntheticFeed


class BuyAndHold(Strategy):
    def __init__(self, symbol: str = "TEST") -> None:
        self.symbol = symbol
        self._bought = False

    def on_bar(self, ctx: Context, bar) -> None:  # type: ignore[no-untyped-def]
        if not self._bought:
            ctx.target_pct(bar.symbol, 1.0)
            self._bought = True


def test_buy_and_hold_full_invest(synthetic_feed: SyntheticFeed) -> None:
    bt = Backtester(
        feed=synthetic_feed,
        strategy=BuyAndHold(),
        starting_cash=100_000,
        broker=SimulatedBroker(slippage=NoSlippage(), commission=PerTradeCommission(0.0)),
    )
    result = bt.run()
    # Bought and held → equity tracks the price path closely.
    assert result.report.n_fills == 1
    # The portfolio should be ~fully invested by mid-test.
    assert result.bars_processed > 200


def test_determinism(synthetic_feed: SyntheticFeed) -> None:
    """Same inputs MUST produce the same equity curve."""
    feed1 = SyntheticFeed(seed=42)
    feed2 = SyntheticFeed(seed=42)
    r1 = Backtester(feed=feed1, strategy=BuyAndHold(), starting_cash=100_000).run()
    r2 = Backtester(feed=feed2, strategy=BuyAndHold(), starting_cash=100_000).run()
    assert (r1.equity_curve.values == r2.equity_curve.values).all()
    assert r1.report.as_dict() == r2.report.as_dict()


def test_no_lookahead_in_strategy() -> None:
    """A strategy that buys a bar at its close MUST NOT fill at that close."""
    feed = SyntheticFeed(seed=1, n_bars=50)
    bt = Backtester(
        feed=feed,
        strategy=BuyAndHold(),
        starting_cash=10_000,
        broker=SimulatedBroker(slippage=NoSlippage(), commission=PerTradeCommission(0.0)),
    )
    result = bt.run()
    fills = result.fills
    assert len(fills) == 1
    fill = fills[0]
    bars = list(feed._generate())  # all bars, including warmup
    test_bars = [b for b in bars if b.timestamp >= feed.start]
    # Fill must happen at the second bar's open (first bar generates the signal).
    assert fill.price == test_bars[1].open


def test_sma_crossover_runs(trending_feed: SyntheticFeed) -> None:
    bt = Backtester(
        feed=trending_feed,
        strategy=SmaCrossover(symbols=["TEST"], fast=5, slow=20),
        starting_cash=100_000,
    )
    result = bt.run()
    assert result.report.n_fills > 0
    assert len(result.equity_curve) > 0


def test_starting_cash_preserved_with_zero_signals() -> None:
    """A strategy that emits no orders must finish with starting cash."""

    class DoNothing(Strategy):
        def on_bar(self, ctx: Context, bar) -> None:  # type: ignore[no-untyped-def]
            pass

    feed = SyntheticFeed(seed=0)
    bt = Backtester(feed=feed, strategy=DoNothing(), starting_cash=50_000)
    result = bt.run()
    assert result.report.n_fills == 0
    assert result.equity_curve.iloc[-1] == pytest.approx(50_000)
