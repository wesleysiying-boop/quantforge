"""quantforge CLI — `qf <command>`.

Thin wrapper around the library so smoke tests and demos can run without a
full Python script. The interesting logic lives in the library.
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import pandas as pd
from rich.console import Console
from rich.table import Table

from quantforge import Backtester, YFinanceFeed
from quantforge.analytics.metrics import build_report
from quantforge.strategy.library.mean_reversion import MeanReversionZScore
from quantforge.strategy.library.sma_crossover import SmaCrossover

console = Console()

STRATEGIES = {
    "sma": SmaCrossover,
    "mean_reversion": MeanReversionZScore,
}


@click.group()
def main() -> None:
    """quantforge — event-driven backtesting from the command line."""


@main.command()
@click.option("--strategy", "-s", type=click.Choice(list(STRATEGIES)), default="sma")
@click.option("--symbol", "-S", multiple=True, default=["SPY"], help="Ticker(s).")
@click.option("--start", default="2018-01-01")
@click.option("--end", default="2024-12-31")
@click.option("--cash", type=float, default=100_000.0)
@click.option("--fast", type=int, default=20, help="SMA only — fast window.")
@click.option("--slow", type=int, default=50, help="SMA only — slow window.")
@click.option("--lookback", type=int, default=20, help="Mean-reversion lookback.")
@click.option("--out", type=click.Path(), default=None, help="Write equity curve CSV.")
def backtest(
    strategy: str,
    symbol: tuple[str, ...],
    start: str,
    end: str,
    cash: float,
    fast: int,
    slow: int,
    lookback: int,
    out: str | None,
) -> None:
    """Run a backtest and print a performance report."""
    feed = YFinanceFeed(symbols=list(symbol), start=start, end=end)

    strat: SmaCrossover | MeanReversionZScore
    if strategy == "sma":
        strat = SmaCrossover(symbols=list(symbol), fast=fast, slow=slow)
    elif strategy == "mean_reversion":
        strat = MeanReversionZScore(symbols=list(symbol), lookback=lookback)
    else:  # pragma: no cover
        raise click.BadParameter(f"unknown strategy: {strategy}")

    bt = Backtester(feed=feed, strategy=strat, starting_cash=cash)
    result = bt.run()

    table = Table(title=f"{type(strat).__name__}  {','.join(symbol)}  {start} → {end}")
    table.add_column("metric")
    table.add_column("value", justify="right")
    for k, v in result.report.as_dict().items():
        if isinstance(v, float):
            display = f"{v:.2%}" if "ratio" in k or "return" in k or "drawdown" in k else f"{v:.4f}"
        else:
            display = str(v)
        table.add_row(k, display)
    console.print(table)

    if out is not None:
        path = Path(out)
        result.to_frame().to_csv(path)
        console.print(f"[green]wrote equity curve to[/] {path}")


@main.command(name="report")
@click.argument("path", type=click.Path(exists=True))
def report_cmd(path: str) -> None:
    """Print metrics from a saved equity-curve CSV."""
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    report = build_report(df["equity"].to_numpy(), n_fills=0)
    console.print(json.dumps(report.as_dict(), indent=2, default=str))


if __name__ == "__main__":  # pragma: no cover
    main()
