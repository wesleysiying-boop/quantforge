"""Streamlit dashboard for browsing backtest results.

Run with:  ``streamlit run src/quantforge/viz/dashboard.py``

Lets you pick a strategy + universe + date range from the sidebar, runs the
backtest in-process, and renders equity curve, drawdown, rolling Sharpe, and
trade-level scatter. Useful for fast iteration on parameter sweeps.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go  # type: ignore[import-not-found]
    import streamlit as st  # type: ignore[import-not-found]
    from plotly.subplots import make_subplots  # type: ignore[import-not-found]
except ImportError as exc:  # pragma: no cover
    raise SystemExit("viz extras not installed. Run:  pip install -e '.[viz]'") from exc

from quantforge import Backtester, YFinanceFeed
from quantforge.strategy.library.mean_reversion import MeanReversionZScore
from quantforge.strategy.library.sma_crossover import SmaCrossover


def _rolling_sharpe(returns: pd.Series, window: int = 63, periods_per_year: int = 252) -> pd.Series:
    mean = returns.rolling(window).mean()
    std = returns.rolling(window).std()
    out: pd.Series = (mean / std * np.sqrt(periods_per_year)).rename("rolling_sharpe")
    return out


def main() -> None:
    st.set_page_config(page_title="quantforge", layout="wide", page_icon="📈")
    st.title("📈 quantforge — backtest dashboard")

    with st.sidebar:
        st.header("Configuration")
        strategy_name = st.selectbox("Strategy", ["SMA crossover", "Mean reversion (z-score)"])
        symbols_raw = st.text_input("Symbols (comma-separated)", "SPY,QQQ")
        symbols = [s.strip().upper() for s in symbols_raw.split(",") if s.strip()]
        start = st.date_input("Start", value=date(2018, 1, 1))
        end = st.date_input("End", value=date(2024, 12, 31))
        cash = st.number_input("Starting cash ($)", value=100_000, step=10_000, min_value=1)

        strategy: SmaCrossover | MeanReversionZScore
        if strategy_name == "SMA crossover":
            fast = st.slider("Fast SMA", 5, 100, 20)
            slow = st.slider("Slow SMA", 20, 300, 50)
            allow_short = st.checkbox("Allow short", value=False)
            strategy = SmaCrossover(symbols=symbols, fast=fast, slow=slow, allow_short=allow_short)
        else:
            lookback = st.slider("Lookback", 5, 60, 20)
            entry_z = st.slider("Entry z", 1.0, 4.0, 2.0, 0.1)
            exit_z = st.slider("Exit z", 0.0, 2.0, 0.5, 0.1)
            strategy = MeanReversionZScore(
                symbols=symbols, lookback=lookback, entry_z=entry_z, exit_z=exit_z
            )

        run = st.button("Run backtest", type="primary", use_container_width=True)

    if not run:
        st.info("Configure parameters in the sidebar, then click **Run backtest**.")
        return

    with st.spinner("Loading data and running backtest..."):
        feed = YFinanceFeed(symbols=symbols, start=str(start), end=str(end))
        bt = Backtester(feed=feed, strategy=strategy, starting_cash=float(cash))
        result = bt.run()

    cols = st.columns(4)
    cols[0].metric("Total return", f"{result.report.total_return:.2%}")
    cols[1].metric("Sharpe", f"{result.report.sharpe:.2f}")
    cols[2].metric("Max drawdown", f"{result.report.max_drawdown:.2%}")
    cols[3].metric("Fills", str(result.report.n_fills))

    df = result.to_frame()
    df["rolling_sharpe"] = _rolling_sharpe(df["returns"])

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=("Equity curve", "Drawdown", "Rolling Sharpe (3M)"),
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df["equity"], name="equity", line={"color": "#22d3ee"}),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["drawdown"], name="drawdown", fill="tozeroy", line={"color": "#f43f5e"}
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["rolling_sharpe"], name="rolling sharpe", line={"color": "#a78bfa"}
        ),
        row=3,
        col=1,
    )
    fig.update_layout(height=720, showlegend=False, template="plotly_dark", margin={"t": 40})
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Performance report"):
        st.json(result.report.as_dict())

    with st.expander(f"Fills ({len(result.fills)})"):
        if result.fills:
            fills_df = pd.DataFrame(
                [
                    {
                        "timestamp": f.timestamp,
                        "symbol": f.symbol,
                        "side": f.side.value,
                        "qty": f.quantity,
                        "price": f.price,
                        "commission": f.commission,
                    }
                    for f in result.fills
                ]
            )
            st.dataframe(fills_df, use_container_width=True)
        else:
            st.write("No fills.")


if __name__ == "__main__":
    main()
