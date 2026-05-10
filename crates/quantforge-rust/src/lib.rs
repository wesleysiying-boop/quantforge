//! quantforge_rust — hot-path indicator math.
//!
//! These functions match the signatures and semantics of the corresponding
//! NumPy implementations in `quantforge.strategy.indicators`. The Python
//! caller transparently picks the Rust path when this extension is
//! available; if not, it falls back to NumPy.
//!
//! All inputs and outputs are float64 NumPy arrays (no copies — we read
//! the input as a contiguous slice and write into a freshly-allocated
//! output owned by Python).

use ndarray::Array1;
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

const NAN: f64 = f64::NAN;

/// Simple moving average. NaN for the first `window-1` elements.
#[pyfunction]
#[pyo3(signature = (prices, window))]
fn sma<'py>(
    py: Python<'py>,
    prices: PyReadonlyArray1<'py, f64>,
    window: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    if window == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "window must be positive",
        ));
    }
    let p = prices.as_slice()?;
    let n = p.len();
    let mut out = vec![NAN; n];
    if n < window {
        return Ok(PyArray1::from_vec_bound(py, out));
    }
    // Single-pass cumulative sum keeps this O(n) regardless of window size.
    let inv_w = 1.0 / window as f64;
    let mut sum: f64 = p[..window].iter().sum();
    out[window - 1] = sum * inv_w;
    for i in window..n {
        sum += p[i] - p[i - window];
        out[i] = sum * inv_w;
    }
    Ok(PyArray1::from_vec_bound(py, out))
}

/// Exponential moving average with alpha = 2 / (window + 1).
#[pyfunction]
#[pyo3(signature = (prices, window))]
fn ema<'py>(
    py: Python<'py>,
    prices: PyReadonlyArray1<'py, f64>,
    window: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    if window == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "window must be positive",
        ));
    }
    let p = prices.as_slice()?;
    let n = p.len();
    let mut out = vec![0.0_f64; n];
    if n == 0 {
        return Ok(PyArray1::from_vec_bound(py, out));
    }
    let alpha = 2.0 / (window as f64 + 1.0);
    let one_m_alpha = 1.0 - alpha;
    out[0] = p[0];
    for i in 1..n {
        out[i] = alpha * p[i] + one_m_alpha * out[i - 1];
    }
    Ok(PyArray1::from_vec_bound(py, out))
}

/// Rolling z-score: (x - rolling_mean) / rolling_std with ddof=1.
/// NaN for the first `window-1` elements.
#[pyfunction]
#[pyo3(signature = (prices, window))]
fn rolling_zscore<'py>(
    py: Python<'py>,
    prices: PyReadonlyArray1<'py, f64>,
    window: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    if window == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "window must be positive",
        ));
    }
    if window < 2 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "rolling_zscore requires window >= 2 (ddof=1 needs >=2 samples)",
        ));
    }
    let p = prices.as_slice()?;
    let n = p.len();
    let mut out = vec![NAN; n];
    if n < window {
        return Ok(PyArray1::from_vec_bound(py, out));
    }
    let w = window as f64;
    // Welford's accumulator scaled to a sliding window — we maintain the
    // running sum and sum-of-squares so each step is O(1).
    let mut s: f64 = p[..window].iter().sum();
    let mut ss: f64 = p[..window].iter().map(|x| x * x).sum();
    let mut mean = s / w;
    let mut var = (ss - s * mean) / (w - 1.0);
    out[window - 1] = if var > 0.0 {
        (p[window - 1] - mean) / var.sqrt()
    } else {
        0.0
    };
    for i in window..n {
        let drop = p[i - window];
        let add = p[i];
        s += add - drop;
        ss += add * add - drop * drop;
        mean = s / w;
        var = (ss - s * mean) / (w - 1.0);
        // Numerical floor — accumulating floats can drive var slightly
        // negative when the window is constant. Treat as zero variance.
        out[i] = if var > 1e-15 {
            (p[i] - mean) / var.sqrt()
        } else {
            0.0
        };
    }
    Ok(PyArray1::from_vec_bound(py, out))
}

/// Period-over-period returns. Length matches input; first element is 0.
/// `log=true` for log-returns, false for simple.
#[pyfunction]
#[pyo3(signature = (prices, log = false))]
fn returns<'py>(
    py: Python<'py>,
    prices: PyReadonlyArray1<'py, f64>,
    log: bool,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let p = prices.as_slice()?;
    let n = p.len();
    let mut out = Array1::<f64>::zeros(n);
    if n < 2 {
        return Ok(PyArray1::from_owned_array_bound(py, out));
    }
    if log {
        for i in 1..n {
            out[i] = (p[i] / p[i - 1]).ln();
        }
    } else {
        for i in 1..n {
            out[i] = p[i] / p[i - 1] - 1.0;
        }
    }
    Ok(PyArray1::from_owned_array_bound(py, out))
}

#[pymodule]
fn quantforge_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sma, m)?)?;
    m.add_function(wrap_pyfunction!(ema, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_zscore, m)?)?;
    m.add_function(wrap_pyfunction!(returns, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
