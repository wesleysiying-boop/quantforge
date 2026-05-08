"""Performance metrics.

All ratios computed from the equity curve, not from per-trade P&L. The
inference period is implied by the bar interval — daily bars give 252
trading days/year, hourly bars give 252*6.5, etc.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def cumulative_returns(equity: FloatArray | list[float]) -> FloatArray:
    arr = np.asarray(equity, dtype=np.float64)
    if arr.size == 0:
        return arr
    out: FloatArray = arr / arr[0] - 1.0
    return out


def annualized_return(equity: FloatArray | list[float], periods_per_year: int = 252) -> float:
    arr = np.asarray(equity, dtype=np.float64)
    if arr.size < 2 or arr[0] <= 0:
        return 0.0
    total = arr[-1] / arr[0]
    years = (arr.size - 1) / periods_per_year
    if years <= 0:
        return 0.0
    return float(total ** (1.0 / years) - 1.0)


def annualized_volatility(returns: FloatArray | list[float], periods_per_year: int = 252) -> float:
    arr = np.asarray(returns, dtype=np.float64)
    if arr.size < 2:
        return 0.0
    return float(arr.std(ddof=1) * np.sqrt(periods_per_year))


_STD_EPS = 1e-12


def sharpe_ratio(
    returns: FloatArray | list[float],
    risk_free: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    arr = np.asarray(returns, dtype=np.float64)
    if arr.size < 2:
        return 0.0
    excess = arr - risk_free / periods_per_year
    sd = float(excess.std(ddof=1))
    if sd < _STD_EPS:
        return 0.0
    return float(excess.mean() / sd * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: FloatArray | list[float],
    risk_free: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    arr = np.asarray(returns, dtype=np.float64)
    if arr.size < 2:
        return 0.0
    excess = arr - risk_free / periods_per_year
    # Downside deviation conventionally squares only sub-MAR returns and
    # divides by the *full* sample size — this stays defined even when
    # downside is sparse.
    downside_sq = np.where(excess < 0, excess, 0.0) ** 2
    dd = float(np.sqrt(downside_sq.mean()))
    if dd < _STD_EPS:
        return 0.0
    return float(excess.mean() / dd * np.sqrt(periods_per_year))


def max_drawdown(equity: FloatArray | list[float]) -> tuple[float, int, int]:
    """Return `(max_dd, peak_idx, trough_idx)`. max_dd is negative."""
    arr = np.asarray(equity, dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0, 0
    running_max = np.maximum.accumulate(arr)
    dd = arr / running_max - 1.0
    trough = int(np.argmin(dd))
    peak = int(np.argmax(arr[: trough + 1])) if trough > 0 else 0
    return float(dd[trough]), peak, trough


def calmar_ratio(equity: FloatArray | list[float], periods_per_year: int = 252) -> float:
    ann = annualized_return(equity, periods_per_year)
    mdd, _, _ = max_drawdown(equity)
    if mdd == 0:
        return 0.0
    return float(ann / abs(mdd))


def hit_ratio(returns: FloatArray | list[float]) -> float:
    arr = np.asarray(returns, dtype=np.float64)
    arr = arr[arr != 0.0]
    if arr.size == 0:
        return 0.0
    return float((arr > 0).sum() / arr.size)


@dataclass(frozen=True, slots=True)
class PerformanceReport:
    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    hit_ratio: float
    n_periods: int
    n_fills: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "annualized_volatility": self.annualized_volatility,
            "sharpe": self.sharpe,
            "sortino": self.sortino,
            "calmar": self.calmar,
            "max_drawdown": self.max_drawdown,
            "hit_ratio": self.hit_ratio,
            "n_periods": self.n_periods,
            "n_fills": self.n_fills,
        }

    def pretty(self) -> str:
        rows = [
            ("Total return", f"{self.total_return:>10.2%}"),
            ("Annualized return", f"{self.annualized_return:>10.2%}"),
            ("Annualized vol", f"{self.annualized_volatility:>10.2%}"),
            ("Sharpe", f"{self.sharpe:>10.2f}"),
            ("Sortino", f"{self.sortino:>10.2f}"),
            ("Calmar", f"{self.calmar:>10.2f}"),
            ("Max drawdown", f"{self.max_drawdown:>10.2%}"),
            ("Hit ratio", f"{self.hit_ratio:>10.2%}"),
            ("Periods", f"{self.n_periods:>10d}"),
            ("Fills", f"{self.n_fills:>10d}"),
        ]
        width = max(len(k) for k, _ in rows)
        lines = [f"  {k:<{width}}  {v}" for k, v in rows]
        return "\n".join(lines)


def build_report(
    equity: FloatArray | list[float],
    n_fills: int,
    periods_per_year: int = 252,
    risk_free: float = 0.0,
) -> PerformanceReport:
    arr = np.asarray(equity, dtype=np.float64)
    if arr.size < 2:
        return PerformanceReport(0, 0, 0, 0, 0, 0, 0, 0, int(arr.size), n_fills)

    rets = arr[1:] / arr[:-1] - 1.0
    mdd, _, _ = max_drawdown(arr)
    return PerformanceReport(
        total_return=float(arr[-1] / arr[0] - 1.0),
        annualized_return=annualized_return(arr, periods_per_year),
        annualized_volatility=annualized_volatility(rets, periods_per_year),
        sharpe=sharpe_ratio(rets, risk_free, periods_per_year),
        sortino=sortino_ratio(rets, risk_free, periods_per_year),
        calmar=calmar_ratio(arr, periods_per_year),
        max_drawdown=mdd,
        hit_ratio=hit_ratio(rets),
        n_periods=int(arr.size),
        n_fills=n_fills,
    )
