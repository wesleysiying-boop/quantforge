"""Yahoo Finance historical data feed.

Pulls daily / intraday OHLCV bars via the `yfinance` package and caches them
to disk. yfinance returns `Close` already adjusted for splits but not
dividends — we use `auto_adjust=True` so closes are total-return-adjusted,
which matches what most public-data backtests assume.

Limitations (documented honestly because they affect strategy validity):
  * Yahoo's intraday history is short (~60 days for 1m, ~730 for 1h).
  * Volume is best-effort; for serious work use Polygon or a paid feed.
  * Survivorship bias: delisted tickers vanish silently.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast

import pandas as pd

from quantforge.core.events import MarketEvent
from quantforge.core.types import Bar, Symbol
from quantforge.data.base import DataFeed
from quantforge.data.cache import ParquetCache

if TYPE_CHECKING:
    pass


class YFinanceFeed(DataFeed):
    """Daily / intraday bars from Yahoo Finance."""

    def __init__(
        self,
        symbols: list[str],
        start: str | datetime,
        end: str | datetime,
        interval: str = "1d",
        cache: ParquetCache | None = None,
        warmup_days: int = 252,
    ) -> None:
        self.symbols = [Symbol(s) for s in symbols]
        # Yahoo returns tz-aware (UTC) indexes; normalize bounds to UTC so
        # downstream comparisons don't trip over naive/aware mismatches.
        self._start_ts = pd.Timestamp(start, tz="UTC")
        self._end_ts = pd.Timestamp(end, tz="UTC")
        self.start = self._start_ts.to_pydatetime()
        self.end = self._end_ts.to_pydatetime()
        self.interval = interval
        self.cache = cache if cache is not None else ParquetCache()
        self.warmup_days = warmup_days
        self._frames: dict[Symbol, pd.DataFrame] = {}

    def _fetch(self, symbol: Symbol) -> pd.DataFrame:
        cached = self.cache.get("yfinance", symbol, self.interval)
        # Determine the window we need including warmup.
        window_start = self._start_ts - pd.Timedelta(days=self.warmup_days * 2)
        if cached is not None and not cached.empty:
            cached_start = cached.index.min()
            cached_end = cached.index.max()
            if cached_start <= window_start and cached_end >= self._end_ts:
                return cached

        try:
            import yfinance as yf  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError("yfinance is required: pip install yfinance") from exc

        raw: Any = yf.download(
            symbol,
            start=window_start.tz_convert(None),
            end=(self._end_ts + pd.Timedelta(days=1)).tz_convert(None),
            interval=self.interval,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        if raw is None or raw.empty:
            raise RuntimeError(f"No data returned for {symbol}")

        df = cast(pd.DataFrame, raw)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
        df.index = pd.to_datetime(df.index, utc=True)
        df = df.dropna()

        self.cache.put("yfinance", symbol, self.interval, df)
        return df

    def _ensure_loaded(self) -> None:
        for sym in self.symbols:
            if sym not in self._frames:
                self._frames[sym] = self._fetch(sym)

    def stream(self) -> Iterator[MarketEvent]:
        self._ensure_loaded()

        # Merge all symbols into a single time-sorted iterator.
        merged: list[tuple[datetime, Bar]] = []
        for sym, df in self._frames.items():
            window = df.loc[(df.index >= self._start_ts) & (df.index <= self._end_ts)]
            for ts, row in window.iterrows():
                py_ts: datetime = (
                    ts.to_pydatetime() if isinstance(ts, pd.Timestamp) else cast(datetime, ts)
                )
                bar = Bar(
                    symbol=sym,
                    timestamp=py_ts,
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                    interval=self.interval,
                )
                merged.append((py_ts, bar))

        merged.sort(key=lambda x: x[0])
        for ts, bar in merged:
            yield MarketEvent(timestamp=ts, bar=bar)

    def warmup_bars(self, symbol: Symbol, n: int) -> list[float]:
        self._ensure_loaded()
        df = self._frames[symbol]
        before = df.loc[df.index < self._start_ts]
        return [float(x) for x in before["close"].tail(n).tolist()]
