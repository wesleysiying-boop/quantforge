"""Indicator math correctness."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from quantforge.strategy.indicators import ema, returns, rolling_zscore, sma


class TestSMA:
    def test_constant_input(self) -> None:
        out = sma([5.0] * 10, 3)
        assert np.all(np.isnan(out[:2]))
        assert np.allclose(out[2:], 5.0)

    def test_rolling_average(self) -> None:
        out = sma([1, 2, 3, 4, 5], 3)
        assert np.allclose(out[2:], [2.0, 3.0, 4.0])

    def test_window_too_large(self) -> None:
        out = sma([1, 2], 5)
        assert np.all(np.isnan(out))

    def test_invalid_window(self) -> None:
        with pytest.raises(ValueError):
            sma([1, 2, 3], 0)


class TestEMA:
    def test_first_value_equals_input(self) -> None:
        out = ema([10, 11, 12], 3)
        assert out[0] == 10.0

    def test_constant_input_converges(self) -> None:
        out = ema([5.0] * 100, 5)
        assert np.allclose(out[-1], 5.0)


class TestReturns:
    def test_simple(self) -> None:
        out = returns([100, 110, 99])
        assert out[0] == 0.0
        assert out[1] == pytest.approx(0.10)
        assert out[2] == pytest.approx(-0.10)

    @given(
        n=st.integers(min_value=2, max_value=200),
        s0=st.floats(min_value=1.0, max_value=1000.0),
    )
    def test_log_returns_match_log_diff(self, n: int, s0: float) -> None:
        rng = np.random.default_rng(0)
        prices = s0 * np.exp(np.cumsum(rng.normal(0, 0.01, size=n)))
        log_r = returns(prices, log=True)
        manual = np.concatenate([[0.0], np.diff(np.log(prices))])
        assert np.allclose(log_r, manual)


class TestZScore:
    def test_zero_variance(self) -> None:
        out = rolling_zscore([5.0] * 10, 5)
        assert out[-1] == 0.0

    def test_known_value(self) -> None:
        prices = list(range(1, 21))
        out = rolling_zscore(prices, 5)
        assert not np.isnan(out[-1])
        assert out[-1] > 0
