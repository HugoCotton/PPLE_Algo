"""
scorer.py
=========
Member B: Signals — Hugo Cotton
PPLE Investment Club · Algo Team
-------------------------------------------------
Builds composite scores for each ticker and returns a ranked DataFrame.

Composite score range: −3.5 (strong sell) → +3.5 (strong buy)
  SMA signal   : −1 / 0 / +1
  BB  signal   : −1 / 0 / +1
  RSI signal   : −1 / 0 / +1
  Fresh-cross  : −0.5 / 0 / +0.5
-------------------------------------------------
"""

import pandas as pd
import logging

from indicators import add_all_indicators, fresh_cross_bonus

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# SCORE A SINGLE TICKER
# ──────────────────────────────────────────────
def score_ticker(df: pd.DataFrame) -> dict:
    """
    Compute indicator signals and composite score for one ticker.

    Parameters
    ----------
    df : cleaned OHLCV DataFrame (output of data_interface)

    Returns
    -------
    dict with keys: sma_signal, bb_signal, rsi_signal, cross_bonus,
                    composite, close, rsi, pct_b
    """
    df = add_all_indicators(df)
    last = df.iloc[-1]

    sma_sig   = int(last.get("sma_signal", 0))
    bb_sig    = int(last.get("bb_signal",  0))
    rsi_sig   = int(last.get("rsi_signal", 0))
    cross     = fresh_cross_bonus(df)
    composite = sma_sig + bb_sig + rsi_sig + cross

    return {
        "sma_signal":  sma_sig,
        "bb_signal":   bb_sig,
        "rsi_signal":  rsi_sig,
        "cross_bonus": cross,
        "composite":   composite,
        "close":       float(last["Close"]),
        "rsi":         float(last.get("rsi", float("nan"))),
        "pct_b":       float(last.get("pct_b", float("nan"))),
    }


# ──────────────────────────────────────────────
# BUILD FULL RANKINGS TABLE
# ──────────────────────────────────────────────
def build_rankings(stock_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Score every ticker and return a DataFrame sorted best → worst.

    Parameters
    ----------
    stock_data : dict {ticker: cleaned_DataFrame}  from data_interface

    Returns
    -------
    pd.DataFrame with columns:
        ticker, composite, sma_signal, bb_signal, rsi_signal,
        cross_bonus, close, rsi, pct_b
    Sorted by composite score descending.
    """
    rows = []
    for ticker, df in stock_data.items():
        try:
            scores = score_ticker(df)
            scores["ticker"] = ticker
            rows.append(scores)
        except Exception as e:
            logger.warning(f"[SCORER] {ticker}: scoring failed — {e}")

    if not rows:
        return pd.DataFrame()

    ranked = (
        pd.DataFrame(rows)
        .sort_values("composite", ascending=False)
        .reset_index(drop=True)
    )
    logger.info(f"[SCORER] Ranked {len(ranked)} tickers.")
    return ranked


# ──────────────────────────────────────────────
# LABEL HELPERS  (used by main.py for display)
# ──────────────────────────────────────────────
def signal_label(val: int) -> str:
    """Convert a −1/0/+1 signal integer to a readable string."""
    return {1: "BUY", -1: "SELL", 0: "NEUTRAL"}.get(int(val), "—")


def composite_label(score: float) -> str:
    """Map a composite score to a sentiment label."""
    if score is None or score != score:  # NaN guard
        return "N/A"
    if score >= 2.5:
        return "STRONG BUY"
    elif score >= 1.0:
        return "BUY"
    elif score <= -2.5:
        return "STRONG SELL"
    elif score <= -1.0:
        return "SELL"
    return "NEUTRAL"


# ──────────────────────────────────────────────
# Standalone test
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import logging
    from data_interface import get_stock_data

    logging.basicConfig(level=logging.INFO)
    stock_data = get_stock_data()
    ranked = build_rankings(stock_data)
    print(ranked.head(10).to_string())
