"""
main.py

Command-line entry point for the backtester.

Usage:
    python main.py --ticker AAPL --strategy sma --start 2020-01-01 --end 2024-01-01
    python main.py --ticker RELIANCE.NS --strategy rsi --cash 100000 --commission 0.001
    python main.py --ticker TCS.NS --strategy macd --start 2021-01-01

Run with --help to see all options.
"""

import argparse
import os
import sys

import pandas as pd
import yfinance as yf
from backtesting import Backtest

from strategies.moving_average import SmaCross
from strategies.rsi_strategy import RsiStrategy
from strategies.macd_strategy import MacdStrategy

STRATEGIES = {
    "sma": SmaCross,
    "rsi": RsiStrategy,
    "macd": MacdStrategy,
}

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
IMAGES_DIR = os.path.join(os.path.dirname(__file__), "images")


def fetch_data(ticker: str, start: str, end: str, use_cache: bool = True) -> pd.DataFrame:
    """
    Download OHLCV data from Yahoo Finance via yfinance, with simple CSV
    caching so repeated runs during development don't re-hit the API.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    cache_path = os.path.join(DATA_DIR, f"{ticker}_{start}_{end}.csv")

    if use_cache and os.path.exists(cache_path):
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
    else:
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if df.empty:
            raise ValueError(
                f"No data returned for ticker '{ticker}'. Check the symbol "
                f"(NSE tickers need a '.NS' suffix, e.g. 'RELIANCE.NS')."
            )
        # yfinance can return MultiIndex columns for a single ticker in some versions
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.to_csv(cache_path)

    # backtesting.py requires exactly these column names, no NaNs
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    return df


def run_backtest(df: pd.DataFrame, strategy_name: str, cash: float, commission: float):
    strategy_cls = STRATEGIES[strategy_name]
    bt = Backtest(df, strategy_cls, cash=cash, commission=commission, finalize_trades=True)
    stats = bt.run()
    return bt, stats


def main():
    parser = argparse.ArgumentParser(description="Trading Strategy Backtester")
    parser.add_argument("--ticker", required=True, help="Yahoo Finance ticker, e.g. AAPL or RELIANCE.NS")
    parser.add_argument("--strategy", choices=STRATEGIES.keys(), default="sma")
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    parser.add_argument("--cash", type=float, default=100_000)
    parser.add_argument("--commission", type=float, default=0.001, help="Fraction per trade, e.g. 0.001 = 0.1%")
    parser.add_argument("--no-cache", action="store_true", help="Force re-download instead of using cached CSV")
    parser.add_argument("--plot", action="store_true", help="Save an interactive HTML plot to images/")
    args = parser.parse_args()

    print(f"Fetching {args.ticker} from {args.start} to {args.end} ...")
    df = fetch_data(args.ticker, args.start, args.end, use_cache=not args.no_cache)
    print(f"Loaded {len(df)} rows.")

    print(f"Running '{args.strategy}' strategy ...")
    bt, stats = run_backtest(df, args.strategy, args.cash, args.commission)

    print("\n" + "=" * 50)
    print(f"RESULTS: {args.ticker} | {args.strategy.upper()}")
    print("=" * 50)
    print(stats)

    if args.plot:
        os.makedirs(IMAGES_DIR, exist_ok=True)
        plot_path = os.path.join(IMAGES_DIR, f"{args.ticker}_{args.strategy}.html")
        bt.plot(filename=plot_path, open_browser=False)
        print(f"\nPlot saved to {plot_path}")


if __name__ == "__main__":
    sys.exit(main())
