"""Tests for RegimeFilteredStrategy.

The wrapper uses a `RegimeClassifier` Protocol; we test against a tiny
hand-rolled stub here so the test stays self-contained (no regime-hmm
dependency for unit tests).
"""

from __future__ import annotations

import numpy as np

from quantforge import Backtester
from quantforge.strategy.base import Context, Strategy
from quantforge.strategy.regime_filter import RegimeFilteredStrategy
from tests.conftest import SyntheticFeed


class _AlwaysBuyStrategy(Strategy):
    """Inner strategy that emits target_pct=1.0 on every bar."""

    def __init__(self, symbol: str = "TEST") -> None:
        self.symbol = symbol

    def on_bar(self, ctx: Context, bar) -> None:  # type: ignore[no-untyped-def]
        ctx.target_pct(bar.symbol, 1.0)


class _ConstantRegimeClassifier:
    """Stub: always returns the same one-hot regime distribution."""

    def __init__(self, regime: int, n_states: int = 2) -> None:
        self.regime = regime
        self.n_states = n_states
        self.fit_count = 0

    def fit(self, X):  # type: ignore[no-untyped-def]
        self.fit_count += 1
        return self

    def predict_proba(self, X):  # type: ignore[no-untyped-def]
        T = len(X)
        out = np.zeros((T, self.n_states), dtype=np.float64)
        out[:, self.regime] = 1.0
        return out


def _run(inner: Strategy, allowed: list[int], regime: int, calibration_bars: int = 50):
    feed = SyntheticFeed(seed=1, n_bars=200)
    classifier = _ConstantRegimeClassifier(regime=regime, n_states=2)
    wrapped = RegimeFilteredStrategy(
        inner=inner,
        classifier=classifier,
        allowed_regimes=allowed,
        calibration_symbol="TEST",
        calibration_bars=calibration_bars,
    )
    bt = Backtester(feed=feed, strategy=wrapped, starting_cash=100_000)
    return bt.run(), wrapped, classifier


def test_signals_pass_when_regime_is_allowed() -> None:
    result, wrapped, classifier = _run(_AlwaysBuyStrategy(), allowed=[1], regime=1)
    assert classifier.fit_count == 1
    assert wrapped.calibrated
    assert wrapped.stats["allowed"] > 0
    assert wrapped.stats["blocked"] == 0
    # The strategy was buying — fills should exist.
    assert result.report.n_fills > 0


def test_signals_blocked_when_regime_not_allowed() -> None:
    result, wrapped, _classifier = _run(_AlwaysBuyStrategy(), allowed=[0], regime=1)
    assert wrapped.stats["allowed"] == 0
    assert wrapped.stats["blocked"] > 0
    # All signals rewritten to flat — no resting position, no real trades that hold.
    # We may see one fill if the wrapper rewrote a target=1.0 to target=0.0
    # while there's no existing position (zero-delta), in which case no order.
    assert result.report.n_fills == 0


def test_no_action_before_calibration() -> None:
    """Pre-calibration signals must be silently dropped, not passed through."""
    feed = SyntheticFeed(seed=1, n_bars=200)
    classifier = _ConstantRegimeClassifier(regime=0)
    wrapped = RegimeFilteredStrategy(
        inner=_AlwaysBuyStrategy(),
        classifier=classifier,
        allowed_regimes=[0],
        calibration_symbol="TEST",
        # Set huge: calibration never finishes within the test data.
        calibration_bars=10_000,
    )
    bt = Backtester(feed=feed, strategy=wrapped, starting_cash=100_000)
    result = bt.run()
    assert not wrapped.calibrated
    assert wrapped.stats["pre_calibration"] > 0
    assert wrapped.stats["allowed"] == 0
    assert result.report.n_fills == 0


def test_refit_every_triggers_extra_fits() -> None:
    feed = SyntheticFeed(seed=1, n_bars=300)
    classifier = _ConstantRegimeClassifier(regime=0)
    wrapped = RegimeFilteredStrategy(
        inner=_AlwaysBuyStrategy(),
        classifier=classifier,
        allowed_regimes=[0],
        calibration_symbol="TEST",
        calibration_bars=50,
        refit_every=50,
    )
    bt = Backtester(feed=feed, strategy=wrapped, starting_cash=100_000)
    bt.run()
    # Initial calibration + at least 2 refits over 250 post-calibration bars.
    assert classifier.fit_count >= 3
