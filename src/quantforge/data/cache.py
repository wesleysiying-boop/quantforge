"""On-disk Parquet cache for OHLCV bars.

Network calls (yfinance, Polygon, etc.) are slow and rate-limited. The cache
keys on `(source, symbol, interval)` and stores one Parquet file per key with
a `timestamp` index. Reads are cheap and zero-copy via `pyarrow`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "quantforge"


class ParquetCache:
    """Filesystem-backed cache. Thread-safe for distinct keys, not within one key."""

    def __init__(self, base_dir: Path | str | None = None) -> None:
        self.base_dir = Path(base_dir) if base_dir is not None else DEFAULT_CACHE_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, source: str, symbol: str, interval: str) -> Path:
        safe_sym = symbol.replace("/", "_").replace(":", "_")
        return self.base_dir / source / interval / f"{safe_sym}.parquet"

    def get(self, source: str, symbol: str, interval: str) -> pd.DataFrame | None:
        path = self._path(source, symbol, interval)
        if not path.exists():
            return None
        return pd.read_parquet(path)

    def put(self, source: str, symbol: str, interval: str, df: pd.DataFrame) -> None:
        path = self._path(source, symbol, interval)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, engine="pyarrow", compression="snappy")

    def has(self, source: str, symbol: str, interval: str) -> bool:
        return self._path(source, symbol, interval).exists()
