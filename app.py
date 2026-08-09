"""
dashboard/app.py

Streamlit dashboard for the Trading Strategy Backtester.

Run with:
    streamlit run dashboard/app.py

Lets the user pick a ticker, date range, and strategy (or all three at once
for a side-by-side comparison), then shows:
  - An interactive candlestick chart with the relevant indicator overlaid
  - The equity curve for the selected strategy
  - A key-metrics table (Total Return, CAGR, Sharpe, Max Drawdown, Win Rate, # Trades)
  - A side-by-side comparison table across all three strategies
"""

import os
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from backtesting import Backtest
from plotly.subplots import make_subplots

# Allow running via `streamlit run dashboard/app.py` from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import fetch_data, STRATEGIES
from utils.indicators import sma, rsi, macd
from utils.metrics import summarize

st.set_page_config(page_title="Trading Strategy Backtester", layout="wide")

st.title("📈 Trading Strategy Backtester")
st.caption("Backtest SMA crossover, RSI mean-reversion, and MACD strategies on any Yahoo Finance ticker.")

# ---------------------------------------------------------------- Sidebar --
with st.sidebar:
    st.header("Settings")
    ticker = st.text_input("Ticker", value="AAPL", help="e.g. AAPL, MSFT, RELIANCE.NS, TCS.NS")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start", value=pd.Timestamp("2020-01-01"))
    with col2:
        end_date = st.date_input("End", value=pd.Timestamp.today())

    strategy_label = st.selectbox(
        "Strategy",
        options=["SMA Crossover", "RSI Mean-Reversion", "MACD Crossover", "Compare All"],
    )
    strategy_map = {"SMA Crossover": "sma", "RSI Mean-Reversion": "rsi", "MACD Crossover": "macd"}

    cash = st.number_input("Starting Capital", value=100_000, step=10_000)
    commission = st.number_input("Commission (fraction per trade)", value=0.001, step=0.0005, format="%.4f")

    run_button = st.button("Run Backtest", type="primary", use_container_width=True)


def load_data(ticker, start_date, end_date):
    with st.spinner(f"Fetching {ticker} ..."):
        return fetch_data(ticker, str(start_date), str(end_date))


def plot_candles_with_indicator(df: pd.DataFrame, strategy_key: str):
    """Candlestick chart with the relevant indicator plotted below it."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05,
        subplot_titles=("Price", "Indicator"),
    )
    fig.add_trace(
        go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"
        ),
        row=1, col=1,
    )

    if strategy_key == "sma":
        fig.add_trace(go.Scatter(x=df.index, y=sma(df["Close"], 20), name="SMA 20", line=dict(width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=sma(df["Close"], 50), name="SMA 50", line=dict(width=1)), row=1, col=1)
        # No separate indicator subplot needed for SMA -- both lines sit on the price panel above
        fig.add_annotation(text="(SMA lines shown on price chart above)", row=2, col=1,
                            showarrow=False, xref="x2", yref="y2 domain", y=0.5)
    elif strategy_key == "rsi":
        rsi_vals = rsi(df["Close"], 14)
        fig.add_trace(go.Scatter(x=df.index, y=rsi_vals, name="RSI 14", line=dict(color="purple")), row=2, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)
    elif strategy_key == "macd":
        macd_df = macd(df["Close"])
        fig.add_trace(go.Scatter(x=df.index, y=macd_df["macd"], name="MACD", line=dict(color="blue")), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=macd_df["signal"], name="Signal", line=dict(color="orange")), row=2, col=1)
        fig.add_trace(go.Bar(x=df.index, y=macd_df["histogram"], name="Histogram"), row=2, col=1)

    fig.update_layout(height=600, xaxis_rangeslider_visible=False, legend=dict(orientation="h"))
    return fig


def run_one(df, strategy_key, cash, commission):
    strategy_cls = STRATEGIES[strategy_key]
    bt = Backtest(df, strategy_cls, cash=cash, commission=commission, finalize_trades=True)
    stats = bt.run()
    return bt, stats


def metrics_table(stats, cash) -> dict:
    equity_curve = stats["_equity_curve"]["Equity"]
    trades = stats["_trades"]
    trade_pnls = trades["PnL"] if len(trades) else pd.Series(dtype=float)
    return summarize(equity_curve, trade_pnls)


# ------------------------------------------------------------------- Main --
if run_button:
    try:
        df = load_data(ticker, start_date, end_date)
    except Exception as e:
        st.error(f"Couldn't load data: {e}")
        st.stop()

    st.success(f"Loaded {len(df)} rows for {ticker} ({df.index[0].date()} to {df.index[-1].date()})")

    if strategy_label == "Compare All":
        st.subheader("Strategy Comparison")
        results = {}
        for label, key in strategy_map.items():
            bt, stats = run_one(df, key, cash, commission)
            results[label] = metrics_table(stats, cash)

        comparison_df = pd.DataFrame(results).T
        st.dataframe(comparison_df, use_container_width=True)

        st.subheader("Price Chart")
        st.plotly_chart(plot_candles_with_indicator(df, "sma"), use_container_width=True)

    else:
        strategy_key = strategy_map[strategy_label]
        bt, stats = run_one(df, strategy_key, cash, commission)

        st.subheader("Key Metrics")
        m = metrics_table(stats, cash)
        cols = st.columns(len(m))
        for col, (k, v) in zip(cols, m.items()):
            col.metric(k, v)

        st.subheader("Price Chart")
        st.plotly_chart(plot_candles_with_indicator(df, strategy_key), use_container_width=True)

        st.subheader("Equity Curve")
        equity_curve = stats["_equity_curve"]["Equity"]
        st.line_chart(equity_curve)

        st.subheader("Trade Log")
        trades = stats["_trades"]
        if len(trades):
            st.dataframe(trades, use_container_width=True)
        else:
            st.info("No trades were executed in this period.")

else:
    st.info("Set your parameters in the sidebar and click **Run Backtest** to get started.")
