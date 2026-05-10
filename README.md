# quantforge

> Event-driven, multi-market backtesting and paper-trading framework — built to study, not to oversell.

[![CI](https://github.com/wesleysiying-boop/quantforge/actions/workflows/ci.yml/badge.svg)](https://github.com/wesleysiying-boop/quantforge/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-261230.svg)](https://github.com/astral-sh/ruff)
[![Type checked: mypy](https://img.shields.io/badge/type%20checked-mypy_strict-blue.svg)](http://mypy-lang.org/)

`quantforge` is a research-grade, event-driven backtesting engine that runs the same strategy code in three modes — historical replay, paper trading, and (eventually) live trading — across US equities and crypto markets.

The core design goal is **fidelity**: backtest results must match paper-trading results bar-for-bar when given identical data, because the same event loop drives both. No vectorized shortcuts, no look-ahead leaks, no hand-waved fills.

---

## Why another backtester?

Most open-source Python backtesters fall into one of two camps:

| Camp                          | Trade-off                                                  |
| ----------------------------- | ---------------------------------------------------------- |
| Vectorized (`vectorbt`, `bt`) | Fast, but leaks future information unless you're careful.  |
| Event-driven (`zipline`, `backtrader`) | Realistic, but heavy, opinionated, and slow.       |

`quantforge` aims for the middle: an event-driven core that's **strict about causality**, lean enough to read end-to-end in an afternoon, and fast where it matters via a Rust hot-path extension (planned, see [Roadmap](#roadmap)).

---

## Features

- **Single event loop** for backtest, paper, and live — strategy code is identical across modes.
- **Pluggable data feeds** — Yahoo Finance, Polygon (planned), Binance / Coinbase via `ccxt` (planned), CSV/Parquet replay.
- **Realistic execution** — configurable slippage models (fixed bps, volume-participation, square-root impact), commission models (per-share, per-trade, exchange-tier), partial fills.
- **Portfolio & risk** — position-level P&L, exposure caps, per-symbol stop-loss, volatility targeting.
- **Analytics** — Sharpe, Sortino, Calmar, max drawdown, hit ratio, turnover, rolling metrics.
- **Strategy library** — SMA crossover, dual-momentum, mean reversion (z-score), pairs trading, HMM-regime filter (via [`regime-hmm`](https://github.com/wesleysiying-boop/regime-hmm) — planned).
- **Visualization** — Streamlit dashboard with equity curve, drawdown, rolling Sharpe, trade scatter.
- **Strict tooling** — `ruff`, `mypy --strict`, `pytest` with property-based tests via `hypothesis`.

---

## Architecture

```
                            ┌────────────────────┐
                            │   DataFeed (ABC)   │
                            │  yfinance / ccxt / │
                            │  csv / live ws     │
                            └─────────┬──────────┘
                                      │  MarketEvent
                                      ▼
        ┌──────────────┐      ┌────────────────┐      ┌──────────────┐
        │   Strategy   │◄─────┤  Event Queue   ├─────►│  Portfolio   │
        │  (user code) │      │   (in-order)   │      │   (state)    │
        └──────┬───────┘      └────────┬───────┘      └──────┬───────┘
               │ SignalEvent           │ FillEvent           │
               ▼                       │                     │ OrderEvent
        ┌──────────────┐               │              ┌──────▼───────┐
        │  Risk Layer  │───────────────┴─────────────►│    Broker    │
        │  (sizing)    │      OrderEvent              │ sim / paper  │
        └──────────────┘                              │  / live      │
                                                      └──────────────┘
```

Four event types — `MarketEvent`, `SignalEvent`, `OrderEvent`, `FillEvent` — flow through one priority queue. The `Backtester` and `LiveTrader` differ only in their data source and broker; everything in between is shared.

---

## Quick start

```bash
git clone https://github.com/wesleysiying-boop/quantforge.git
cd quantforge
python -m pip install -e ".[dev,viz]"
make run-example      # runs an SMA crossover on SPY 2018–2024
make dashboard        # opens Streamlit results viewer
```

Minimal strategy:

```python
from quantforge import Strategy, Backtester, YFinanceFeed
from quantforge.indicators import sma

class SmaCrossover(Strategy):
    def __init__(self, fast: int = 20, slow: int = 50) -> None:
        self.fast, self.slow = fast, slow

    def on_bar(self, ctx, bar) -> None:
        f = sma(ctx.history(bar.symbol, "close", self.slow), self.fast)
        s = sma(ctx.history(bar.symbol, "close", self.slow), self.slow)
        if f[-1] > s[-1] and f[-2] <= s[-2]:
            ctx.target_pct(bar.symbol, 1.0)
        elif f[-1] < s[-1] and f[-2] >= s[-2]:
            ctx.target_pct(bar.symbol, 0.0)

bt = Backtester(
    feed=YFinanceFeed(symbols=["SPY"], start="2018-01-01", end="2024-12-31"),
    strategy=SmaCrossover(fast=20, slow=50),
    starting_cash=100_000,
)
result = bt.run()
print(result.summary())
```

---

## Project status

This is a **work in progress**, built in the open. The roadmap below tracks reality, not aspiration.

### Roadmap

- [x] Project scaffolding, packaging, CI, lint/type/test toolchain
- [x] Core event types & priority queue
- [x] `DataFeed` ABC + `YFinanceFeed` + Parquet cache
- [x] Simulated broker with slippage + commission models
- [x] `Portfolio` with position tracking, mark-to-market, P&L
- [x] Backtest engine (event loop + clock)
- [x] First strategy: SMA crossover
- [x] Analytics: Sharpe, Sortino, max drawdown, equity curve
- [x] Streamlit dashboard
- [x] HMM regime-filter integration via `regime-hmm` ([example](examples/03_regime_filtered_momentum.py))
- [x] Rust hot-path extension (`crates/quantforge-rust`) for indicator math — measured speedups below
- [ ] Property-based tests for fill semantics (hypothesis)
- [ ] Polygon.io data feed (free tier)
- [ ] CCXT crypto feed + Binance / Coinbase paper-trading adapter
- [ ] Alpaca paper-trading adapter (US equities)
- [ ] Walk-forward validation harness
- [ ] Limit order book matching from `lobster-cpp` (companion project)
- [ ] Monte Carlo bootstrap for confidence intervals on Sharpe

### Rust hot-path extension

Optional companion crate `crates/quantforge-rust` ships drop-in replacements
for the indicator math via PyO3. Build once, the Python `indicators` module
picks them up automatically:

```bash
cd crates/quantforge-rust
maturin develop --release
```

Measured on Python 3.14 / NumPy 2.4 / Apple M-series, n=10,000, window=20:

| Indicator              |    NumPy   |    Rust    |  Speedup |
| ---------------------- | ---------: | ---------: | -------: |
| `sma`                  |    62.4 us |    14.3 us |    4.4x  |
| `ema`                  |   2536 us  |    27.1 us |   93.5x  |
| `rolling_zscore`       | 89,285 us  |    27.7 us | 3,200x\* |
| `returns(log=True)`    |    40.7 us |    37.3 us |    1.1x  |

\* The `rolling_zscore` win is partly algorithmic (Rust uses a sliding
sum/sum-of-squares accumulator, the NumPy path recomputes per step). The
Python loop in `ema` accounts for the ~100x figure there. `returns` uses
`np.diff(np.log(...))` which is already C-vectorized — Rust matches but
doesn't beat. Honest numbers, no cherry-picking.

### Companion projects

- [`regime-hmm`](https://github.com/wesleysiying-boop/regime-hmm) — Hidden Markov Model market-regime detector (planned).
- [`lobster-cpp`](https://github.com/wesleysiying-boop/lobster-cpp) — Sub-microsecond limit order book in C++20, with `pybind11` bindings (planned).

---

## Design notes

A few design choices that aren't obvious from the code:

1. **No look-ahead, ever.** The event queue guarantees a strategy sees a bar at time `t` only after the broker has finished processing fills from time `t-1`. The `Bar.timestamp` is the bar's *close*, and signals generated on that bar can fill no earlier than the next bar's open.
2. **Cash and positions are integers internally.** Cash is stored as integer cents; share quantities as integer shares (or integer satoshis for crypto). Floating-point P&L is computed only at the reporting boundary. This avoids the cumulative-rounding bugs that have bitten every backtester I've used.
3. **The clock is explicit.** `SimulationClock` is a parameter, not a global. This is what lets the same engine drive both backtest and live modes — in live mode, the clock is `WallClock`.
4. **Strategies are stateless w.r.t. the framework.** They never hold references to the broker, portfolio, or queue — only to a `Context` object that exposes a narrow, read-mostly API. This keeps strategies trivially testable.

---

## License

[MIT](LICENSE) © 2026 Wesley Si Ying
