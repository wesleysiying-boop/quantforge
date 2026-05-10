"""Benchmark Rust vs NumPy indicator implementations.

Run:  python benchmarks/bench_indicators.py
"""

from __future__ import annotations

import statistics
import sys
import time

import numpy as np

from quantforge.strategy import indicators


def _measure(fn, args, kwargs, repeats: int = 5, inner: int = 100) -> tuple[float, float]:
    """Return (median_us_per_call, min_us_per_call)."""
    timings = []
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        for _ in range(inner):
            fn(*args, **kwargs)
        elapsed_ns = time.perf_counter_ns() - t0
        timings.append(elapsed_ns / inner / 1_000.0)  # us per call
    return statistics.median(timings), min(timings)


def main() -> None:
    if not indicators.HAS_RUST:
        print("quantforge_rust is not installed.")
        print("Build it via:  cd crates/quantforge-rust && maturin develop --release")
        sys.exit(1)

    print(f"Python: {sys.version.split()[0]}")
    print(f"NumPy : {np.__version__}")
    print("Rust  : enabled\n")

    rng = np.random.default_rng(0)
    sizes = [1_000, 10_000, 100_000]

    for n in sizes:
        prices = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, size=n)))
        prices = np.ascontiguousarray(prices)

        print(f"--- n = {n:>7,} ---")
        for name, py_fn, rust_fn, kwargs in [
            ("sma(window=20)", indicators._sma_numpy, indicators.sma, {"window": 20}),
            ("ema(window=20)", indicators._ema_numpy, indicators.ema, {"window": 20}),
            (
                "rolling_zscore(window=20)",
                indicators._rolling_zscore_numpy,
                indicators.rolling_zscore,
                {"window": 20},
            ),
            ("returns(log=True)", indicators._returns_numpy, indicators.returns, {"log": True}),
        ]:
            # Python path: call the underscore-prefixed numpy function directly.
            py_med, _ = _measure(py_fn, (prices,), kwargs)
            # Rust path: call the public function (which dispatches to Rust).
            rs_med, _ = _measure(rust_fn, (prices,), kwargs)
            speedup = py_med / rs_med if rs_med > 0 else float("inf")
            print(
                f"  {name:<28}  numpy={py_med:>8.1f} us   "
                f"rust={rs_med:>7.1f} us   speedup={speedup:>5.1f}x"
            )
        print()


if __name__ == "__main__":
    main()
