"""
visualizer.py
=============
Generates matplotlib charts for a single ticker.

Each chart has two subplots stacked vertically:
  1. Price panel:  Candlestick-style close line, 20/50-day SMAs,
                   and Bollinger Bands as a shaded region.
  2. RSI panel:    RSI line with overbought (70) and oversold (30)
                   reference lines and a shaded neutral zone.

The chart is designed to be readable at standard screen resolution and
uses a dark background theme for clarity.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

from indicators import (
    compute_sma_crossover,
    compute_bollinger_bands,
    compute_rsi,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _apply_dark_theme(fig: plt.Figure, axes: list) -> None:
    """
    Apply a consistent dark background theme to a figure and its axes.

    Sets the figure background, axis backgrounds, tick colours, spine
    colours, and label colours to a dark palette. Called once at the
    start of plot_ticker() before any data is drawn.

    Args:
        fig:  The matplotlib Figure object.
        axes: List of Axes objects on the figure.
    """
    fig.patch.set_facecolor("#0f1117")
    for ax in axes:
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#c9d1d9", labelsize=9)
        ax.xaxis.label.set_color("#c9d1d9")
        ax.yaxis.label.set_color("#c9d1d9")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")


def _format_date_axis(ax: plt.Axes, df: pd.DataFrame) -> None:
    """
    Format the x-axis of a price/RSI panel with readable date labels.

    Uses matplotlib's AutoDateLocator to space ticks automatically based
    on the date range, and formats them as "Mon DD" (e.g. "Jan 14").
    Rotates labels 30° to prevent overlap.

    Args:
        ax: The Axes whose x-axis should be formatted.
        df: The DataFrame whose index is used (to confirm DatetimeIndex).
    """
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def plot_ticker(
    ticker: str,
    df: pd.DataFrame,
    lookback_days: int = 90,
    save_path: str | None = None,
) -> None:
    """
    Generate and display (or save) a two-panel technical chart for one ticker.

    Panel 1 — Price + Indicators:
      - Closing price as a thin white line
      - 20-day SMA as an orange line
      - 50-day SMA as a cyan line
      - Bollinger Bands upper/lower as dashed grey lines
      - The area between the bands is lightly shaded blue

    Panel 2 — RSI:
      - RSI line coloured dynamically:
          * Green when RSI < 30 (oversold)
          * Red when RSI > 70 (overbought)
          * Yellow otherwise (neutral)
      - Horizontal dashed lines at 30 and 70
      - The 30–70 neutral zone is lightly shaded

    The function re-computes all indicators from the raw DataFrame so that
    the chart always reflects the same calculations as the scorer.

    Args:
        ticker:       Ticker symbol string used as the chart title.
        df:           Full historical DataFrame for the ticker (data contract).
        lookback_days: How many of the most recent trading days to show
                       on the chart (default 90 ≈ 3 months). The full
                       history is still used for indicator calculation;
                       only the display window is trimmed.
        save_path:    If provided, saves the figure to this file path
                      (e.g. "charts/AAPL.png") instead of showing it.
                      Useful for batch-generating charts.
    """
    # --- Compute indicators on the full dataset ---
    sma_data = compute_sma_crossover(df)
    bb_data  = compute_bollinger_bands(df)
    rsi_data = compute_rsi(df)

    # --- Trim to the display window ---
    display_df   = df.iloc[-lookback_days:]
    sma20_series = sma_data["sma20_series"].iloc[-lookback_days:]
    sma50_series = sma_data["sma50_series"].iloc[-lookback_days:]
    bb_upper     = bb_data["upper_series"].iloc[-lookback_days:]
    bb_lower     = bb_data["lower_series"].iloc[-lookback_days:]
    bb_middle    = bb_data["middle_series"].iloc[-lookback_days:]
    rsi_series   = rsi_data["rsi_series"].iloc[-lookback_days:]
    dates        = display_df.index

    # --- Build figure: 2 rows, price panel is 3× taller than RSI panel ---
    fig = plt.figure(figsize=(14, 8))
    gs  = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.08)
    ax_price = fig.add_subplot(gs[0])
    ax_rsi   = fig.add_subplot(gs[1], sharex=ax_price)

    _apply_dark_theme(fig, [ax_price, ax_rsi])

    # -----------------------------------------------------------------------
    # Panel 1: Price
    # -----------------------------------------------------------------------

    # Closing price line
    ax_price.plot(dates, display_df["Close"], color="#e6edf3", linewidth=1.2,
                  label="Close", zorder=3)

    # 20-day SMA
    ax_price.plot(dates, sma20_series, color="#f0883e", linewidth=1.0,
                  label="SMA 20", zorder=2)

    # 50-day SMA
    ax_price.plot(dates, sma50_series, color="#58a6ff", linewidth=1.0,
                  label="SMA 50", zorder=2)

    # Bollinger Bands — upper and lower as dashed lines
    ax_price.plot(dates, bb_upper, color="#8b949e", linewidth=0.7,
                  linestyle="--", label="BB Upper", zorder=1)
    ax_price.plot(dates, bb_lower, color="#8b949e", linewidth=0.7,
                  linestyle="--", label="BB Lower", zorder=1)

    # Shaded region between bands
    ax_price.fill_between(dates, bb_lower, bb_upper,
                          alpha=0.07, color="#58a6ff", zorder=0)

    # Mark the SMA crossover point if it occurred in the display window
    _mark_crossovers(ax_price, dates, sma20_series, sma50_series)

    ax_price.set_ylabel("Price (USD)", color="#c9d1d9")
    ax_price.legend(loc="upper left", fontsize=8,
                    facecolor="#161b22", labelcolor="#c9d1d9", framealpha=0.8)
    ax_price.set_title(
        f"{ticker}  |  SMA 20/50 · Bollinger Bands · RSI(14)",
        color="#e6edf3", fontsize=13, fontweight="bold", pad=10
    )

    # Hide x-axis tick labels on the price panel (shared with RSI)
    plt.setp(ax_price.xaxis.get_majorticklabels(), visible=False)

    # -----------------------------------------------------------------------
    # Panel 2: RSI
    # -----------------------------------------------------------------------

    # Colour the RSI line by region
    _plot_rsi_coloured(ax_rsi, dates, rsi_series)

    # Reference lines
    ax_rsi.axhline(70, color="#f85149", linewidth=0.8, linestyle="--", alpha=0.7)
    ax_rsi.axhline(30, color="#3fb950", linewidth=0.8, linestyle="--", alpha=0.7)
    ax_rsi.fill_between(dates, 30, 70, alpha=0.05, color="#c9d1d9")

    # Labels
    ax_rsi.text(dates[-1], 71, "Overbought", color="#f85149",
                fontsize=7, ha="right", va="bottom")
    ax_rsi.text(dates[-1], 29, "Oversold", color="#3fb950",
                fontsize=7, ha="right", va="top")

    ax_rsi.set_ylim(0, 100)
    ax_rsi.set_ylabel("RSI", color="#c9d1d9")
    ax_rsi.set_yticks([0, 30, 50, 70, 100])

    _format_date_axis(ax_rsi, display_df)

    # -----------------------------------------------------------------------
    # Output
    # -----------------------------------------------------------------------
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"[CHART] Saved → {save_path}")
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Internal drawing helpers
# ---------------------------------------------------------------------------

def _mark_crossovers(
    ax: plt.Axes,
    dates: pd.DatetimeIndex,
    sma20: pd.Series,
    sma50: pd.Series,
) -> None:
    """
    Draw vertical marker lines at every SMA crossover point in the chart window.

    A crossover is detected wherever the sign of (sma20 - sma50) changes
    from one day to the next. Bullish crosses (20 crosses above 50) are
    marked with a semi-transparent green vertical line; bearish crosses
    with a red vertical line.

    Args:
        ax:    The price Axes to draw on.
        dates: The DatetimeIndex for the x-axis.
        sma20: 20-day SMA series (aligned to dates).
        sma50: 50-day SMA series (aligned to dates).
    """
    diff      = (sma20 - sma50).dropna()
    prev_diff = diff.shift(1)

    bullish_cross = (diff > 0) & (prev_diff <= 0)
    bearish_cross = (diff < 0) & (prev_diff >= 0)

    for date in diff[bullish_cross].index:
        ax.axvline(date, color="#3fb950", linewidth=0.8, alpha=0.5, linestyle=":")

    for date in diff[bearish_cross].index:
        ax.axvline(date, color="#f85149", linewidth=0.8, alpha=0.5, linestyle=":")


def _plot_rsi_coloured(
    ax: plt.Axes,
    dates: pd.DatetimeIndex,
    rsi: pd.Series,
) -> None:
    """
    Plot the RSI line with dynamic colour based on the RSI value.

    Rather than a single flat colour, this function splits the RSI series
    into three segments and plots each in a different colour:
      - Green  (#3fb950) where RSI < 30 (oversold territory)
      - Red    (#f85149) where RSI > 70 (overbought territory)
      - Yellow (#e3b341) for the neutral 30–70 band

    Splitting is done day-by-day: wherever the colour changes, a new
    line segment is started. This avoids artefacts at transition points.

    Args:
        ax:    The RSI Axes to draw on.
        dates: DatetimeIndex for the x-axis.
        rsi:   RSI values series aligned to dates.
    """
    def _colour(val: float) -> str:
        if val < 30:
            return "#3fb950"
        elif val > 70:
            return "#f85149"
        return "#e3b341"

    vals   = rsi.values
    d_list = list(dates)

    seg_x   = [d_list[0]]
    seg_y   = [vals[0]]
    seg_col = _colour(vals[0])

    for i in range(1, len(vals)):
        col = _colour(vals[i])
        seg_x.append(d_list[i])
        seg_y.append(vals[i])
        if col != seg_col or i == len(vals) - 1:
            ax.plot(seg_x, seg_y, color=seg_col, linewidth=1.2)
            seg_x   = [d_list[i]]
            seg_y   = [vals[i]]
            seg_col = col


# ---------------------------------------------------------------------------
# Batch chart generation
# ---------------------------------------------------------------------------

def plot_top_n(
    ranked_df: pd.DataFrame,
    stock_data: dict[str, pd.DataFrame],
    n: int = 10,
    save_dir: str | None = None,
) -> None:
    """
    Generate charts for the top N and bottom N tickers from the ranked table.

    Useful for quickly reviewing the most and least bullish setups without
    manually specifying each ticker.

    Args:
        ranked_df:  The DataFrame returned by scorer.build_rankings().
        stock_data: The full dict of ticker DataFrames (from data_interface).
        n:          How many tickers to chart from each end (default 10).
        save_dir:   Directory to save charts into (e.g. "output/charts/").
                    If None, charts are shown interactively one by one.
    """
    top_tickers    = ranked_df.head(n)["ticker"].tolist()
    bottom_tickers = ranked_df.tail(n)["ticker"].tolist()
    tickers_to_plot = list(dict.fromkeys(top_tickers + bottom_tickers))

    for ticker in tickers_to_plot:
        if ticker not in stock_data:
            continue
        save_path = f"{save_dir}/{ticker}.png" if save_dir else None
        plot_ticker(ticker, stock_data[ticker], save_path=save_path)