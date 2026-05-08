"""Mean-reversion via rolling z-score.

Enter long when price drops below `-entry_z` standard deviations of its
rolling mean; exit when it returns to within `exit_z`. Symmetric short side
optional.
"""

from __future__ import annotations

from quantforge.core.types import Bar, Symbol
from quantforge.strategy.base import Context, Strategy
from quantforge.strategy.indicators import rolling_zscore


class MeanReversionZScore(Strategy):
    def __init__(
        self,
        symbols: list[str],
        lookback: int = 20,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        allow_short: bool = True,
    ) -> None:
        if entry_z <= exit_z:
            raise ValueError("entry_z must be > exit_z")
        self.symbols = [Symbol(s) for s in symbols]
        self.lookback = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.allow_short = allow_short
        self._in_position: dict[Symbol, int] = {}  # +1 long, -1 short, 0 flat

    def on_bar(self, ctx: Context, bar: Bar) -> None:
        closes = ctx.history(bar.symbol, "close")
        if closes.size < self.lookback + 1:
            return

        z = rolling_zscore(closes, self.lookback)[-1]
        state = self._in_position.get(bar.symbol, 0)

        if state == 0:
            if z < -self.entry_z:
                ctx.target_pct(bar.symbol, 1.0)
                self._in_position[bar.symbol] = 1
            elif z > self.entry_z and self.allow_short:
                ctx.target_pct(bar.symbol, -1.0)
                self._in_position[bar.symbol] = -1
        elif (state == 1 and z > -self.exit_z) or (state == -1 and z < self.exit_z):
            ctx.target_pct(bar.symbol, 0.0)
            self._in_position[bar.symbol] = 0
