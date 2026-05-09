"""Regime-filter strategy wrapper.

Wraps an inner strategy and gates its signals on a market regime classifier.
The classifier is anything that exposes ``predict_proba(X) -> (T, K)`` —
matching the sklearn-style API of `regime-hmm`.

The wrapper works in two phases:

1. **Calibration** — collects `calibration_bars` of feature observations
   before allowing the inner strategy to act. During calibration, the
   classifier is fit once.
2. **Live filter** — after calibration, every emitted signal from the
   inner strategy is intercepted; if the current regime is in the
   allowed set, the signal passes through, otherwise it is rewritten to
   ``target_pct=0`` (force flat).

The default feature is a single column of log returns; pass a custom
``feature_fn`` for richer feature stacks (volatility, drawdown, etc.).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import numpy.typing as npt

from quantforge.core.events import SignalEvent
from quantforge.core.types import Bar, Symbol
from quantforge.strategy.base import Context, Strategy

if TYPE_CHECKING:
    pass

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int_]


class RegimeClassifier(Protocol):
    """Minimal interface a regime classifier must satisfy."""

    def fit(self, X: FloatArray) -> Any: ...
    def predict_proba(self, X: FloatArray) -> FloatArray: ...


def default_feature(history: FloatArray) -> FloatArray:
    """Default feature: 1-D log returns of close prices."""
    if history.size < 2:
        return np.zeros((0, 1), dtype=np.float64)
    log_ret = np.diff(np.log(history))
    return log_ret[:, None]


class RegimeFilteredStrategy(Strategy):
    """Wrap an inner strategy with a regime gate.

    Parameters
    ----------
    inner
        The strategy whose signals are filtered.
    classifier
        Any object implementing ``fit(X)`` and ``predict_proba(X) -> (T, K)``.
        ``regime_hmm.GaussianHMM`` is the canonical fit.
    allowed_regimes
        Iterable of regime indices (after calibration) in which the inner
        strategy's signals are allowed through. The mapping from regime
        index to "what the regime means" is determined by the calibration
        data, so check after fit (e.g. by running once with all regimes
        allowed and inspecting per-regime returns).
    calibration_symbol
        The symbol whose price series drives feature extraction. The wrapper
        uses one symbol's bars to fit the classifier; multi-symbol regime
        modeling is out of scope.
    calibration_bars
        Minimum number of bars to observe before fitting. Until this is
        reached, the inner strategy is silenced.
    feature_fn
        Maps a 1-D close-price array (length T) to a 2-D feature matrix
        (T-k by d). Defaults to ``default_feature`` (log returns).
    refit_every
        If positive, refit the classifier every ``refit_every`` bars after
        the initial calibration. Useful for adapting to slow regime drift;
        set to 0 to fit once and never again.
    """

    def __init__(
        self,
        inner: Strategy,
        classifier: RegimeClassifier,
        allowed_regimes: Iterable[int],
        calibration_symbol: str,
        *,
        calibration_bars: int = 504,
        feature_fn: Callable[[FloatArray], FloatArray] = default_feature,
        refit_every: int = 0,
    ) -> None:
        self.inner = inner
        self.classifier = classifier
        self.allowed_regimes = frozenset(int(r) for r in allowed_regimes)
        self.calibration_symbol = Symbol(calibration_symbol)
        self.calibration_bars = calibration_bars
        self.feature_fn = feature_fn
        self.refit_every = refit_every

        self._calibrated = False
        self._bars_since_fit = 0
        self._allowed_count = 0
        self._blocked_count = 0
        self._never_acted = 0  # bars where calibration wasn't done yet

    @property
    def calibrated(self) -> bool:
        return self._calibrated

    @property
    def stats(self) -> dict[str, int]:
        return {
            "allowed": self._allowed_count,
            "blocked": self._blocked_count,
            "pre_calibration": self._never_acted,
        }

    def on_start(self, ctx: Context) -> None:
        self.inner.on_start(ctx)

    def on_finish(self, ctx: Context) -> None:
        self.inner.on_finish(ctx)

    def on_bar(self, ctx: Context, bar: Bar) -> None:
        # Snapshot how many signals the inner already had pending so we can
        # tell which ones came from this bar's invocation.
        before = len(ctx.pending_signals)
        self.inner.on_bar(ctx, bar)
        new_signals = ctx.pending_signals[before:]
        if not new_signals:
            return

        if not self._maybe_calibrate(ctx):
            # Drop pre-calibration signals — we can't gate them honestly.
            ctx.pending_signals = ctx.pending_signals[:before]
            self._never_acted += len(new_signals)
            return

        regime = self._current_regime(ctx)
        if regime in self.allowed_regimes:
            self._allowed_count += len(new_signals)
            return  # let signals stand

        # Block: rewrite to flat.
        ctx.pending_signals = ctx.pending_signals[:before]
        for sig in new_signals:
            ctx.pending_signals.append(
                SignalEvent(
                    timestamp=sig.timestamp,
                    symbol=sig.symbol,
                    target_pct=0.0,
                    strength=sig.strength,
                )
            )
            self._blocked_count += 1

    def _maybe_calibrate(self, ctx: Context) -> bool:
        closes = ctx.history(self.calibration_symbol, "close")
        if closes.size < self.calibration_bars:
            return False

        if not self._calibrated:
            X = self.feature_fn(closes)
            self.classifier.fit(X)
            self._calibrated = True
            self._bars_since_fit = 0
            return True

        if self.refit_every > 0:
            self._bars_since_fit += 1
            if self._bars_since_fit >= self.refit_every:
                X = self.feature_fn(closes)
                self.classifier.fit(X)
                self._bars_since_fit = 0
        return True

    def _current_regime(self, ctx: Context) -> int:
        closes = ctx.history(self.calibration_symbol, "close")
        X = self.feature_fn(closes)
        proba = self.classifier.predict_proba(X)
        return int(np.argmax(proba[-1]))
