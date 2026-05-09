"""SMA crossover with HMM regime filter — quantforge × regime-hmm.

Demonstrates how the two projects compose: the SMA crossover strategy
trades when the HMM says we're in a "trending" regime (state 0 or 2 in
the SPY fit, depending on how the EM converges), and stays flat when
the HMM says we're in chop.

Requires:  pip install -e '.[regime]'

Run:       python examples/03_regime_filtered_momentum.py
"""

from __future__ import annotations

import sys

import numpy as np

from quantforge import Backtester, YFinanceFeed
from quantforge.strategy.library.sma_crossover import SmaCrossover
from quantforge.strategy.regime_filter import RegimeFilteredStrategy

try:
    from regime_hmm import GaussianHMM
except ImportError:
    sys.stderr.write(
        "regime-hmm is required for this example.\n"
        "Install via: pip install regime-hmm\n"
        "Or from a sibling clone: pip install -e ../regime-hmm\n"
    )
    raise


def feature_log_returns(history: np.ndarray) -> np.ndarray:
    """Log returns shaped (T-1, 1) — feeds straight into a univariate HMM."""
    if history.size < 2:
        return np.zeros((0, 1), dtype=np.float64)
    return np.diff(np.log(history))[:, None]


def main() -> None:
    symbol = "SPY"
    feed = YFinanceFeed(symbols=[symbol], start="2010-01-01", end="2024-12-31")

    inner = SmaCrossover(symbols=[symbol], fast=20, slow=50)
    classifier = GaussianHMM(n_states=3, n_restarts=5, max_iter=100, random_state=0)

    # First pass: allow ALL regimes so we can see what the HMM labels look
    # like in an unfiltered run, then pick the trending ones below.
    discovery = RegimeFilteredStrategy(
        inner=SmaCrossover(symbols=[symbol], fast=20, slow=50),
        classifier=GaussianHMM(n_states=3, n_restarts=5, max_iter=100, random_state=0),
        allowed_regimes=[0, 1, 2],
        calibration_symbol=symbol,
        calibration_bars=504,
        feature_fn=feature_log_returns,
    )
    bt_disc = Backtester(feed=feed, strategy=discovery, starting_cash=100_000)
    res_disc = bt_disc.run()

    # Inspect what the HMM learned — pick the state whose mean log-return is
    # most positive as the "trending bull" regime. (This is heuristic; in a
    # real research workflow you'd inspect the per-regime return tables and
    # pick by hand.)
    means = discovery.classifier.means_
    assert means is not None
    bull_regime = int(np.argmax(means[:, 0]))
    print("\nDiscovery pass — HMM regime means (log-returns):")
    for k, m in enumerate(means[:, 0]):
        marker = "  ← bull" if k == bull_regime else ""
        print(f"  state {k}: mean log-return = {m:+.5f}{marker}")
    print(f"\nUnfiltered SMA(20,50) on {symbol}:")
    print(res_disc.summary())

    # Second pass: reuse the same fitted classifier, but only allow trades
    # when in the bull regime.
    feed2 = YFinanceFeed(symbols=[symbol], start="2010-01-01", end="2024-12-31")
    filtered = RegimeFilteredStrategy(
        inner=inner,
        classifier=classifier,
        allowed_regimes=[bull_regime],
        calibration_symbol=symbol,
        calibration_bars=504,
        feature_fn=feature_log_returns,
    )
    bt = Backtester(feed=feed2, strategy=filtered, starting_cash=100_000)
    result = bt.run()

    print(f"\nFiltered (bull-only) SMA(20,50) on {symbol}:")
    print(result.summary())
    print(f"\nFilter stats: {filtered.stats}")


if __name__ == "__main__":
    main()
