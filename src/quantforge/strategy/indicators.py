"""Indicator primitives.

Pure NumPy, no pandas, no global state. Functions take a 1-D array of prices
and return either a scalar (latest value) or a same-length array. Designed
to be drop-in replaceable by a Rust extension later — see `quantforge-core`
in the roadmap.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def sma(prices: FloatArray | list[float], window: int) -> FloatArray:
    """Simple moving average. Returns NaN for the first `window-1` elements."""
    arr = np.asarray(prices, dtype=np.float64)
    if window <= 0:
        raise ValueError("window must be positive")
    if arr.size < window:
        return np.full_like(arr, np.nan)
    cumsum = np.cumsum(np.insert(arr, 0, 0.0))
    out = np.full_like(arr, np.nan)
    out[window - 1 :] = (cumsum[window:] - cumsum[:-window]) / window
    return out


def ema(prices: FloatArray | list[float], window: int) -> FloatArray:
    """Exponential moving average with alpha = 2 / (window + 1)."""
    arr = np.asarray(prices, dtype=np.float64)
    if window <= 0:
        raise ValueError("window must be positive")
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
    arr = np.asarray(prices, dtype=np.float64)
    if arr.size < 2:
        return np.zeros_like(arr)
    if log:
        out = np.empty_like(arr)
        out[0] = 0.0
        out[1:] = np.diff(np.log(arr))
    else:
        out = np.empty_like(arr)
        out[0] = 0.0
        out[1:] = arr[1:] / arr[:-1] - 1.0
    return out


def rolling_zscore(prices: FloatArray | list[float], window: int) -> FloatArray:
    """Z-score of the latest value vs. rolling mean / std over `window` bars."""
    arr = np.asarray(prices, dtype=np.float64)
    out = np.full_like(arr, np.nan)
    if arr.size < window:
        return out
    for i in range(window - 1, arr.size):
        w = arr[i - window + 1 : i + 1]
        mu = w.mean()
        sd = w.std(ddof=1)
        out[i] = 0.0 if sd == 0 else (arr[i] - mu) / sd
    return out
