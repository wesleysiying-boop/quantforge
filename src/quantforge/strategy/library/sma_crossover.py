"""SMA crossover — the canonical "hello world" of trend following.

Buy when the fast SMA crosses above the slow SMA; flatten when it crosses
back below. Long-only by default; flip `allow_short=True` to go short on the
bearish cross. Useful as a sanity baseline more than a research strategy.
"""

from __future__ import annotations

from quantforge.core.types import Bar, Symbol
from quantforge.strategy.base import Context, Strategy
from quantforge.strategy.indicators import sma


class SmaCrossover(Strategy):
    def __init__(
        self,
        symbols: list[str],
        fast: int = 20,
        slow: int = 50,
        allow_short: bool = False,
    ) -> None:
        if fast >= slow:
            raise ValueError(f"fast ({fast}) must be < slow ({slow})")
        self.symbols = [Symbol(s) for s in symbols]
        self.fast = fast
        self.slow = slow
        self.allow_short = allow_short

    def on_bar(self, ctx: Context, bar: Bar) -> None:
        closes = ctx.history(bar.symbol, "close")
        if closes.size < self.slow + 1:
            return

        fast_line = sma(closes, self.fast)
        slow_line = sma(closes, self.slow)
        f_now, f_prev = fast_line[-1], fast_line[-2]
        s_now, s_prev = slow_line[-1], slow_line[-2]

        bullish_cross = f_prev <= s_prev and f_now > s_now
        bearish_cross = f_prev >= s_prev and f_now < s_now

        if bullish_cross:
            ctx.target_pct(bar.symbol, 1.0)
        elif bearish_cross:
            ctx.target_pct(bar.symbol, -1.0 if self.allow_short else 0.0)
