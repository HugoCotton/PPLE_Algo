"""
visualiser.py
=============
PPLE Investment Club · Algo Team
-------------------------------------------------
Charting utilities for individual tickers and top-N ranked stocks.

plot_ticker(ticker, df)              — interactive price + indicator chart
plot_top_n(ranked_df, stock_data, n) — batch-save PNG charts
-------------------------------------------------
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from indicators import add_all_indicators


def plot_ticker(ticker: str, df: pd.DataFrame, save_path: str = None) -> None:
    """
    Plot a 3-panel chart for one ticker:
      Top    : Close price with SMA20, SMA50, and Bollinger Bands
      Middle : RSI with overbought/oversold lines
      Bottom : Volume bars

    Parameters
    ----------
    ticker    : ticker symbol (used in title)
    df        : cleaned OHLCV DataFrame
    save_path : if given, save the figure to this file path (PNG) instead of showing
    """
    df = add_all_indicators(df.copy())

    fig = plt.figure(figsize=(14, 9))
    fig.suptitle(f"{ticker}  —  PPLE Algo", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(3, 1, height_ratios=[3, 1.5, 1], hspace=0.05)

    ax_price  = fig.add_subplot(gs[0])
    ax_rsi    = fig.add_subplot(gs[1], sharex=ax_price)
    ax_volume = fig.add_subplot(gs[2], sharex=ax_price)

    dates = df.index

    # ── Price panel ────────────────────────────────────────────────
    ax_price.plot(dates, df["Close"],    color="#1f77b4", linewidth=1.5, label="Close")
    ax_price.plot(dates, df["SMA_20"],   color="#ff7f0e", linewidth=1,   label="SMA 20", linestyle="--")
    ax_price.plot(dates, df["SMA_50"],   color="#2ca02c", linewidth=1,   label="SMA 50", linestyle="--")
    ax_price.fill_between(dates, df["BB_lower"], df["BB_upper"],
                          alpha=0.12, color="grey", label="Bollinger Bands")
    ax_price.plot(dates, df["BB_upper"], color="grey", linewidth=0.6, linestyle=":")
    ax_price.plot(dates, df["BB_lower"], color="grey", linewidth=0.6, linestyle=":")
    ax_price.set_ylabel("Price (USD)")
    ax_price.legend(loc="upper left", fontsize=8)
    ax_price.grid(True, alpha=0.3)
    plt.setp(ax_price.get_xticklabels(), visible=False)

    # ── RSI panel ──────────────────────────────────────────────────
    ax_rsi.plot(dates, df["rsi"], color="#9467bd", linewidth=1.2, label="RSI 14")
    ax_rsi.axhline(70, color="red",   linestyle="--", linewidth=0.8, alpha=0.7)
    ax_rsi.axhline(30, color="green", linestyle="--", linewidth=0.8, alpha=0.7)
    ax_rsi.fill_between(dates, 70, df["rsi"].clip(upper=100),
                        where=df["rsi"] > 70, alpha=0.15, color="red")
    ax_rsi.fill_between(dates, df["rsi"].clip(lower=0), 30,
                        where=df["rsi"] < 30, alpha=0.15, color="green")
    ax_rsi.set_ylim(0, 100)
    ax_rsi.set_ylabel("RSI")
    ax_rsi.legend(loc="upper left", fontsize=8)
    ax_rsi.grid(True, alpha=0.3)
    plt.setp(ax_rsi.get_xticklabels(), visible=False)

    # ── Volume panel ───────────────────────────────────────────────
    colors = ["#2ca02c" if c >= o else "#d62728"
              for c, o in zip(df["Close"], df["Open"])]
    ax_volume.bar(dates, df["Volume"], color=colors, width=0.8, alpha=0.7)
    ax_volume.set_ylabel("Volume")
    ax_volume.grid(True, alpha=0.3)

    fig.autofmt_xdate(rotation=30)

    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
    # else: caller does plt.show()


def plot_top_n(ranked_df: pd.DataFrame, stock_data: dict,
               n: int = 10, save_dir: str = "charts") -> None:
    """
    Save PNG charts for the top N and bottom N tickers in ranked_df.

    Parameters
    ----------
    ranked_df  : output of scorer.build_rankings()
    stock_data : dict {ticker: DataFrame}
    n          : number of top + bottom tickers to chart
    save_dir   : directory to write PNG files into
    """
    os.makedirs(save_dir, exist_ok=True)

    top_tickers    = ranked_df.head(n)["ticker"].tolist()
    bottom_tickers = ranked_df.tail(n)["ticker"].tolist()

    for ticker in set(top_tickers + bottom_tickers):
        if ticker not in stock_data:
            continue
        path = os.path.join(save_dir, f"{ticker}.png")
        plot_ticker(ticker, stock_data[ticker], save_path=path)
