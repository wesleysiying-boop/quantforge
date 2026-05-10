"""Indicator primitives.

Pure NumPy by default, with an optional Rust fast-path when the
``quantforge-rust`` companion extension is installed. The Rust path is
roughly 10-50x faster on long sequences (see ``benchmarks/bench_indicators.py``)
but is strictly opt-in — the package itself depends only on NumPy.

To install the Rust extension:

    cd crates/quantforge-rust && maturin develop --release

The integration uses a try/except import so a deployment without the Rust
build still works; only the Python path is exercised.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

try:
    import quantforge_rust as _rust

    HAS_RUST = True
except ImportError:
    HAS_RUST = False
    _rust = None


def sma(prices: FloatArray | list[float], window: int) -> FloatArray:
    """Simple moving average. Returns NaN for the first `window-1` elements."""
    if window <= 0:
        raise ValueError("window must be positive")
    arr = np.ascontiguousarray(np.asarray(prices, dtype=np.float64))
    if HAS_RUST:
        return _rust.sma(arr, window)  # type: ignore[no-any-return]
    return _sma_numpy(arr, window)


def _sma_numpy(arr: FloatArray, window: int) -> FloatArray:
    if arr.size < window:
        return np.full_like(arr, np.nan)
    cumsum = np.cumsum(np.insert(arr, 0, 0.0))
    out = np.full_like(arr, np.nan)
    out[window - 1 :] = (cumsum[window:] - cumsum[:-window]) / window
    return out


def ema(prices: FloatArray | list[float], window: int) -> FloatArray:
    """Exponential moving average with alpha = 2 / (window + 1)."""
    if window <= 0:
        raise ValueError("window must be positive")
    arr = np.ascontiguousarray(np.asarray(prices, dtype=np.float64))
    if HAS_RUST:
        return _rust.ema(arr, window)  # type: ignore[no-any-return]
    return _ema_numpy(arr, window)


def _ema_numpy(arr: FloatArray, window: int) -> FloatArray:
    alpha = 2.0 / (window + 1)
    out = np.empty_like(arr)
    if arr.size == 0:
        return out
    out[0] = arr[0]
    for i in range(1, arr.size):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
    return out


def returns(prices: FloatArray | list[float], log: bool = False) -> FloatArray:
    """Period-over-period returns. Length matches input; first element is 0."""
    arr = np.ascontiguousarray(np.asarray(prices, dtype=np.float64))
    if HAS_RUST:
        return _rust.returns(arr, log)  # type: ignore[no-any-return]
    return _returns_numpy(arr, log)


def _returns_numpy(arr: FloatArray, log: bool) -> FloatArray:
    if arr.size < 2:
        return np.zeros_like(arr)
    out = np.empty_like(arr)
    out[0] = 0.0
    if log:
        out[1:] = np.diff(np.log(arr))
    else:
        out[1:] = arr[1:] / arr[:-1] - 1.0
    return out


def rolling_zscore(prices: FloatArray | list[float], window: int) -> FloatArray:
    """Z-score of the latest value vs. rolling mean / std over `window` bars."""
    arr = np.ascontiguousarray(np.asarray(prices, dtype=np.float64))
    if HAS_RUST and window >= 2:
        return _rust.rolling_zscore(arr, window)  # type: ignore[no-any-return]
    return _rolling_zscore_numpy(arr, window)


def _rolling_zscore_numpy(arr: FloatArray, window: int) -> FloatArray:
    out = np.full_like(arr, np.nan)
    if arr.size < window:
        return out
    for i in range(window - 1, arr.size):
        w = arr[i - window + 1 : i + 1]
        mu = w.mean()
        sd = w.std(ddof=1)
        out[i] = 0.0 if sd == 0 else (arr[i] - mu) / sd
    return out
