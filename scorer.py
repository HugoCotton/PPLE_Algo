"""
scorer.py
=========
Converts indicator results into scores and produces a ranked DataFrame.

Scoring philosophy:
  Each of the three indicators contributes a signal of +1, 0, or -1.
  These are summed into a composite score ranging from -3 to +3:
    +3 = all three indicators bullish  (strong buy)
    +2 = two bullish, one neutral
    +1 = one bullish, two neutral  (or mixed)
     0 = perfectly neutral / mixed
    -1 = one bearish signal
    -2 = two bearish signals
    -3 = all three bearish (strong sell)

  A "fresh SMA crossover" bonus of +0.5 / -0.5 is added when the 20/50
  crossover happened within the last 3 days, rewarding recency of trend change.
  This can push the composite score slightly outside the -3…+3 integer range.

  The final ranking is by composite score descending (highest = most bullish).
"""

import pandas as pd
from indicators import compute_all_indicators


# ---------------------------------------------------------------------------
# Score a single ticker
# ---------------------------------------------------------------------------

def score_ticker(indicators: dict) -> dict:
    """
    Convert a single ticker's indicator results into a numeric score.

    Takes the output of compute_all_indicators() and maps each indicator's
    signal to a numeric contribution, then sums them.

    Scoring breakdown:
      - SMA crossover signal:  +1 (20 > 50), -1 (20 < 50), or 0
      - SMA fresh cross bonus: +0.5 if bullish cross just happened,
                               -0.5 if bearish cross just happened
      - Bollinger Band signal: +1 (at lower band), -1 (at upper band), 0
      - RSI signal:            +1 (RSI < 30), -1 (RSI > 70), 0

    Args:
        indicators: The dict returned by compute_all_indicators().
                    Must have keys "sma", "bb", "rsi".

    Returns:
        A dict with:
          "sma_signal":   raw SMA signal (-1, 0, +1)
          "bb_signal":    raw BB signal (-1, 0, +1)
          "rsi_signal":   raw RSI signal (-1, 0, +1)
          "fresh_cross":  bool — whether a fresh crossover occurred
          "composite":    float — final weighted score
    """
    sma = indicators["sma"]
    bb  = indicators["bb"]
    rsi = indicators["rsi"]

    sma_signal = sma["signal"]
    bb_signal  = bb["signal"]
    rsi_signal = rsi["signal"]

    # Fresh cross bonus: adds urgency to a recent crossover
    fresh_bonus = 0.0
    if sma["fresh_cross"]:
        fresh_bonus = 0.5 * sma_signal   # +0.5 for bullish cross, -0.5 for bearish

    composite = sma_signal + bb_signal + rsi_signal + fresh_bonus

    return {
        "sma_signal":  sma_signal,
        "bb_signal":   bb_signal,
        "rsi_signal":  rsi_signal,
        "fresh_cross": sma["fresh_cross"],
        "composite":   round(composite, 2),
    }


# ---------------------------------------------------------------------------
# Build the full ranked table
# ---------------------------------------------------------------------------

def build_rankings(stock_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Run all indicators and scoring for every ticker, then rank them.

    This is the top-level function called by main.py. It iterates over
    all tickers in the stock_data dict, computes indicators, scores each
    one, and assembles a ranked DataFrame.

    Tickers that fail validation (too few rows, missing Close) are skipped
    with a printed warning rather than crashing the program.

    Args:
        stock_data: dict mapping ticker strings to DataFrames.
                    Matches the output of data_interface.get_stock_data().

    Returns:
        A pd.DataFrame with one row per valid ticker, sorted by "composite"
        score descending. Columns:
          ticker       | str   — ticker symbol
          composite    | float — overall score (-3.5 to +3.5)
          sma_signal   | int   — SMA crossover signal
          bb_signal    | int   — Bollinger Band signal
          rsi_signal   | int   — RSI signal
          fresh_cross  | bool  — recent SMA crossover flag
          sma20        | float — current 20-day SMA
          sma50        | float — current 50-day SMA
          bb_upper     | float — upper Bollinger Band
          bb_lower     | float — lower Bollinger Band
          pct_b        | float — %B position within bands
          rsi          | float — current RSI value
          close        | float — latest closing price
    """
    rows = []

    for ticker, df in stock_data.items():
        indicators = compute_all_indicators(df)

        if indicators is None:
            print(f"[SKIP] {ticker}: insufficient data or missing Close column.")
            continue

        score = score_ticker(indicators)

        rows.append({
            "ticker":      ticker,
            "composite":   score["composite"],
            "sma_signal":  score["sma_signal"],
            "bb_signal":   score["bb_signal"],
            "rsi_signal":  score["rsi_signal"],
            "fresh_cross": score["fresh_cross"],
            "sma20":       indicators["sma"]["sma20"],
            "sma50":       indicators["sma"]["sma50"],
            "bb_upper":    indicators["bb"]["upper"],
            "bb_lower":    indicators["bb"]["lower"],
            "pct_b":       indicators["bb"]["pct_b"],
            "rsi":         indicators["rsi"]["rsi"],
            "close":       round(df["Close"].iloc[-1], 4),
        })

    if not rows:
        print("[ERROR] No tickers produced valid results.")
        return pd.DataFrame()

    ranked = (
        pd.DataFrame(rows)
        .sort_values("composite", ascending=False)
        .reset_index(drop=True)
    )
    ranked.index += 1   # rank starts at 1, not 0
    ranked.index.name = "rank"

    return ranked


# ---------------------------------------------------------------------------
# Helpers for display
# ---------------------------------------------------------------------------

def signal_label(value: int) -> str:
    """
    Convert a raw signal integer to a human-readable label.

    Used by main.py when printing the ranked table so that +1/-1/0
    are shown as readable strings rather than bare numbers.

    Args:
        value: -1, 0, or +1

    Returns:
        "BUY", "SELL", or "NEUT"
    """
    return {1: "BUY", -1: "SELL", 0: "NEUT"}.get(value, "?")


def composite_label(score: float) -> str:
    """
    Map a composite score to a summary sentiment label.

    Used in the printed table as an at-a-glance summary column.

    Score bands:
      +2.0 to +3.5 → "STRONG BUY"
      +0.5 to +1.5 → "BUY"
       -0.5 to 0.5 → "NEUTRAL"
      -1.5 to -0.5 → "SELL"
      -3.5 to -2.0 → "STRONG SELL"

    Args:
        score: The composite float score.

    Returns:
        A string sentiment label.
    """
    if score >= 2.0:
        return "STRONG BUY"
    elif score >= 0.5:
        return "BUY"
    elif score <= -2.0:
        return "STRONG SELL"
    elif score <= -0.5:
        return "SELL"
    else:
        return "NEUTRAL"