"""Performance metric tests."""

from __future__ import annotations

import numpy as np
import pytest

from quantforge.analytics.metrics import (
    annualized_return,
    annualized_volatility,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
)


def test_max_drawdown_identifies_peak_and_trough() -> None:
    equity = np.array([100, 110, 120, 80, 90, 100, 130], dtype=float)
    mdd, peak, trough = max_drawdown(equity)
    assert peak == 2  # value 120
    assert trough == 3  # value 80
    assert mdd == pytest.approx(80 / 120 - 1.0)


def test_max_drawdown_monotonically_increasing() -> None:
    mdd, _peak, _trough = max_drawdown(np.array([100, 110, 120, 130], dtype=float))
    assert mdd == 0.0


def test_annualized_return_with_known_path() -> None:
    # Doubled over 252 daily bars → 100% annualized.
    eq = np.linspace(1.0, 2.0, 253)
    assert annualized_return(eq) == pytest.approx(1.0, abs=0.01)


def test_sharpe_zero_for_constant_returns() -> None:
    # Std is 0 → handler returns 0 to avoid div by zero.
    assert sharpe_ratio(np.array([0.001] * 10)) == 0.0


def test_sharpe_positive_for_positive_drift() -> None:
    rng = np.random.default_rng(0)
    rets = rng.normal(0.001, 0.01, size=2520)
    s = sharpe_ratio(rets)
    assert s > 0


def test_sortino_only_penalizes_downside() -> None:
    # Equal magnitude but only 1 of 4 is negative — Sortino > Sharpe.
    rets = np.array([0.01, 0.01, 0.01, -0.01])
    s = sharpe_ratio(rets)
    so = sortino_ratio(rets)
    assert so > s


def test_annualized_vol_scales_with_sqrt_t() -> None:
    rng = np.random.default_rng(1)
    rets = rng.normal(0, 0.01, size=2520)
    daily_vol = rets.std(ddof=1)
    assert annualized_volatility(rets) == pytest.approx(daily_vol * np.sqrt(252))
