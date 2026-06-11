"""
indicators.py
=============
Computes all technical indicators for a single ticker's DataFrame.

Each function takes a DataFrame (matching the data contract) and returns
either a modified DataFrame with new columns, or a dict of signal values.

All functions are pure — they never modify the input DataFrame in place.
"""

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# SMA CROSSOVER (20-day / 50-day)
# ---------------------------------------------------------------------------

def compute_sma(df: pd.DataFrame, window: int) -> pd.Series:
    """
    Compute a Simple Moving Average (SMA) over a rolling window.

    A Simple Moving Average smooths price data by averaging the closing
    price over the last `window` trading days. The 20-day SMA reacts
    quickly to recent price changes; the 50-day SMA moves more slowly
    and represents the medium-term trend.

    Args:
        df:     DataFrame with at least a "Close" column.
        window: Number of periods to average (e.g. 20 or 50).

    Returns:
        A pd.Series of the same length as df, with NaN for the first
        (window - 1) rows where there isn't enough data yet.
    """
    return df["Close"].rolling(window=window).mean()


def compute_sma_crossover(df: pd.DataFrame) -> dict:
    """
    Detect the 20/50-day SMA crossover signal.

    Logic:
      - If the 20-day SMA is currently ABOVE the 50-day SMA → bullish (+1)
      - If the 20-day SMA is currently BELOW the 50-day SMA → bearish (-1)
      - If they are equal (rare) → neutral (0)

    We also detect whether a crossover *just happened* in the last 3 days
    (a fresh cross carries more weight than one that happened weeks ago),
    but the base signal is simply the current relative position.

    Args:
        df: DataFrame with a "Close" column and at least 50 rows.

    Returns:
        A dict with:
          "sma20":         the current 20-day SMA value
          "sma50":         the current 50-day SMA value
          "signal":        +1 (bullish), -1 (bearish), or 0 (neutral)
          "fresh_cross":   True if a crossover occurred in the last 3 days
          "sma20_series":  full pd.Series (for charting)
          "sma50_series":  full pd.Series (for charting)
    """
    sma20 = compute_sma(df, 20)
    sma50 = compute_sma(df, 50)

    current_20 = sma20.iloc[-1]
    current_50 = sma50.iloc[-1]

    if current_20 > current_50:
        signal = 1
    elif current_20 < current_50:
        signal = -1
    else:
        signal = 0

    # Detect a fresh crossover: the sign of (sma20 - sma50) changed
    # in the last 3 days.
    diff = sma20 - sma50
    recent_diff = diff.dropna().iloc[-4:]          # last 4 valid values
    sign_changes = (recent_diff > 0) != (recent_diff > 0).shift(1)
    fresh_cross = bool(sign_changes.any())

    return {
        "sma20": round(current_20, 4),
        "sma50": round(current_50, 4),
        "signal": signal,
        "fresh_cross": fresh_cross,
        "sma20_series": sma20,
        "sma50_series": sma50,
    }


# ---------------------------------------------------------------------------
# BOLLINGER BANDS
# ---------------------------------------------------------------------------

def compute_bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> dict:
    """
    Compute Bollinger Bands and derive a trading signal.

    Bollinger Bands consist of three lines plotted around a moving average:
      - Middle Band: 20-day SMA of closing price
      - Upper Band:  Middle Band + (2 × 20-day rolling standard deviation)
      - Lower Band:  Middle Band - (2 × 20-day rolling standard deviation)

    The bands expand during volatile periods and contract during calm periods.
    Price touching or breaching a band is a potential reversal signal.

    Signal logic (based on where today's close sits relative to the bands):
      - Close <= Lower Band  → oversold, potential buy  → +1
      - Close >= Upper Band  → overbought, potential sell → -1
      - Close inside bands   → neutral                   →  0

    We also compute a "% B" value: where the price sits within the bands
    as a 0–1 ratio (0 = at lower band, 1 = at upper band, 0.5 = at midpoint).
    This gives a finer-grained view than just a binary breach.

    Args:
        df:      DataFrame with "Close" column and at least `window` rows.
        window:  Rolling window for the SMA and std dev (default 20).
        num_std: Number of standard deviations for band width (default 2.0).

    Returns:
        A dict with:
          "upper":         current upper band value
          "middle":        current middle band (SMA) value
          "lower":         current lower band value
          "pct_b":         %B value (0–1 scale)
          "bandwidth":     (upper - lower) / middle — measures volatility
          "signal":        +1 (oversold), -1 (overbought), 0 (neutral)
          "upper_series":  full pd.Series (for charting)
          "middle_series": full pd.Series (for charting)
          "lower_series":  full pd.Series (for charting)
    """
    middle = df["Close"].rolling(window=window).mean()
    std    = df["Close"].rolling(window=window).std()
    upper  = middle + num_std * std
    lower  = middle - num_std * std

    close   = df["Close"].iloc[-1]
    u       = upper.iloc[-1]
    m       = middle.iloc[-1]
    l       = lower.iloc[-1]

    # %B: 0 at lower band, 1 at upper band
    pct_b = (close - l) / (u - l) if (u - l) != 0 else 0.5

    # Bandwidth: normalised width of the bands
    bandwidth = (u - l) / m if m != 0 else 0

    if close <= l:
        signal = 1
    elif close >= u:
        signal = -1
    else:
        signal = 0

    return {
        "upper":         round(u, 4),
        "middle":        round(m, 4),
        "lower":         round(l, 4),
        "pct_b":         round(pct_b, 4),
        "bandwidth":     round(bandwidth, 4),
        "signal":        signal,
        "upper_series":  upper,
        "middle_series": middle,
        "lower_series":  lower,
    }


# ---------------------------------------------------------------------------
# RSI — Relative Strength Index
# ---------------------------------------------------------------------------

def compute_rsi(df: pd.DataFrame, window: int = 14) -> dict:
    """
    Compute the Relative Strength Index (RSI) and derive a signal.

    RSI is a momentum oscillator that measures the speed and magnitude of
    recent price changes. It oscillates between 0 and 100.

    Calculation steps:
      1. Compute daily price changes (delta).
      2. Separate into gains (positive deltas) and losses (absolute negative deltas).
      3. Compute the initial average gain and loss over the first `window` periods
         using a simple mean (Wilder's smoothing seed).
      4. For subsequent periods, use Wilder's exponential smoothing:
             avg_gain = (prev_avg_gain × (window-1) + current_gain) / window
      5. RS  = avg_gain / avg_loss
         RSI = 100 - (100 / (1 + RS))

    Signal logic:
      - RSI < 30  → oversold, potential buy  → +1
      - RSI > 70  → overbought, potential sell → -1
      - 30–70     → neutral                   →  0

    Args:
        df:     DataFrame with "Close" column and at least (window + 1) rows.
        window: Lookback period (default 14, as defined by J. Welles Wilder).

    Returns:
        A dict with:
          "rsi":        current RSI value (0–100)
          "signal":     +1 (oversold), -1 (overbought), 0 (neutral)
          "rsi_series": full pd.Series of RSI values (for charting)
    """
    delta = df["Close"].diff()

    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    # Wilder's smoothing (equivalent to EWM with alpha = 1/window)
    avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
    avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()

    rs  = avg_gain / avg_loss.replace(0, np.nan)  # avoid division by zero
    rsi = 100 - (100 / (1 + rs))

    current_rsi = rsi.iloc[-1]

    if current_rsi < 30:
        signal = 1
    elif current_rsi > 70:
        signal = -1
    else:
        signal = 0

    return {
        "rsi":        round(current_rsi, 2),
        "signal":     signal,
        "rsi_series": rsi,
    }


# ---------------------------------------------------------------------------
# COMBINED — run all indicators for one ticker
# ---------------------------------------------------------------------------

def compute_all_indicators(df: pd.DataFrame) -> dict | None:
    """
    Run all three indicator functions on a single ticker's DataFrame.

    This is the main entry point called by scorer.py. It validates the
    DataFrame first, then runs each indicator and bundles the results.

    Validation checks:
      - "Close" column must be present
      - At least 60 rows (50 for SMA + 10 buffer)
      - If "High" or "Low" are missing, Bollinger Bands fall back to "Close"

    Args:
        df: A single ticker's DataFrame (matching the data contract).

    Returns:
        A dict with keys "sma", "bb", "rsi" — each containing the dict
        returned by the corresponding compute_* function.
        Returns None if validation fails (caller will skip the ticker).
    """
    if "Close" not in df.columns:
        return None
    if len(df) < 60:
        return None

    # Drop NaN closes before computing
    df = df.dropna(subset=["Close"]).copy()

    sma_result = compute_sma_crossover(df)
    bb_result  = compute_bollinger_bands(df)
    rsi_result = compute_rsi(df)

    return {
        "sma": sma_result,
        "bb":  bb_result,
        "rsi": rsi_result,
    }