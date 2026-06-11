"""
indicators.py
=============
Member B: Signals — Hugo Cotton
PPLE Investment Club · Algo Team
-------------------------------------------------
Computes technical indicators for a single ticker's OHLCV DataFrame:
  - SMA 20/50 crossover
  - Bollinger Bands (20-day, 2σ)
  - RSI (14-day, Wilder's method)
-------------------------------------------------
"""

import pandas as pd
import numpy as np


# ──────────────────────────────────────────────
# 1. SMA CROSSOVER
# ──────────────────────────────────────────────
def compute_sma(df: pd.DataFrame, fast: int = 20, slow: int = 50) -> pd.DataFrame:
    """
    Add SMA_fast, SMA_slow columns and a sma_signal column.

    sma_signal:
      +1  →  SMA_fast > SMA_slow  (bullish)
      -1  →  SMA_fast < SMA_slow  (bearish)
       0  →  equal
    """
    df = df.copy()
    close = df["Close"]
    df[f"SMA_{fast}"] = close.rolling(fast).mean()
    df[f"SMA_{slow}"] = close.rolling(slow).mean()

    df["sma_signal"] = 0
    df.loc[df[f"SMA_{fast}"] > df[f"SMA_{slow}"], "sma_signal"] = 1
    df.loc[df[f"SMA_{fast}"] < df[f"SMA_{slow}"], "sma_signal"] = -1
    return df


def fresh_cross_bonus(df: pd.DataFrame, fast: int = 20, slow: int = 50,
                      window: int = 3) -> float:
    """
    Returns +0.5 if a bullish crossover happened in the last `window` days,
    -0.5 for a bearish crossover, 0 otherwise.
    """
    if f"SMA_{fast}" not in df.columns or f"SMA_{slow}" not in df.columns:
        df = compute_sma(df, fast, slow)

    recent = df.tail(window + 1)
    prev_signal = recent["sma_signal"].iloc[:-1]
    curr_signal = recent["sma_signal"].iloc[-1]

    if curr_signal == 1 and (prev_signal == -1).any():
        return 0.5
    if curr_signal == -1 and (prev_signal == 1).any():
        return -0.5
    return 0.0


# ──────────────────────────────────────────────
# 2. BOLLINGER BANDS
# ──────────────────────────────────────────────
def compute_bollinger(df: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """
    Add BB_mid, BB_upper, BB_lower, pct_b, and bb_signal columns.

    pct_b = (close - lower) / (upper - lower)  — 0 at lower band, 1 at upper band

    bb_signal:
      +1  →  close ≤ lower band (oversold / buy)
      -1  →  close ≥ upper band (overbought / sell)
       0  →  inside bands
    """
    df = df.copy()
    close = df["Close"]
    df["BB_mid"]   = close.rolling(window).mean()
    rolling_std    = close.rolling(window).std()
    df["BB_upper"] = df["BB_mid"] + num_std * rolling_std
    df["BB_lower"] = df["BB_mid"] - num_std * rolling_std

    band_width = df["BB_upper"] - df["BB_lower"]
    df["pct_b"] = (close - df["BB_lower"]) / band_width.replace(0, np.nan)

    df["bb_signal"] = 0
    df.loc[close <= df["BB_lower"], "bb_signal"] = 1
    df.loc[close >= df["BB_upper"], "bb_signal"] = -1
    return df


# ──────────────────────────────────────────────
# 3. RSI (Wilder's smoothing)
# ──────────────────────────────────────────────
def compute_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Add rsi and rsi_signal columns using Wilder's exponential smoothing.

    rsi_signal:
      +1  →  RSI < 30 (oversold)
      -1  →  RSI > 70 (overbought)
       0  →  30–70 (neutral)
    """
    df = df.copy()
    delta = df["Close"].diff()

    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    # Wilder's smoothing: first value is simple average, rest are EWMA
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    df["rsi_signal"] = 0
    df.loc[df["rsi"] < 30, "rsi_signal"] = 1
    df.loc[df["rsi"] > 70, "rsi_signal"] = -1
    return df


# ──────────────────────────────────────────────
# 4. FULL INDICATOR SUITE (convenience wrapper)
# ──────────────────────────────────────────────
def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply SMA crossover, Bollinger Bands, and RSI in one call.
    Returns the enriched DataFrame.
    """
    df = compute_sma(df)
    df = compute_bollinger(df)
    df = compute_rsi(df)
    return df
