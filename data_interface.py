"""
data_interface.py
=================
Member A: Data — Alice Galzin
PPLE Investment Club · Algo Team
-------------------------------------------------
Responsibilities:
  1. Screen the S&P 500 universe daily by avg. daily volume
  2. Fetch OHLCV prices via yFinance
  3. Clean and validate data for signals module
-------------------------------------------------
"""

import pandas as pd
import yfinance as yf
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
TOP_N_TICKERS   = 100    # how many most-liquid tickers to screen
HISTORY_DAYS    = "6mo"  # how much history to download (needs ≥60 rows for indicators)
MIN_ROWS        = 60     # minimum trading days required for indicator calc
SP500_URL       = "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv"


# ──────────────────────────────────────────────
# 1. FETCH S&P 500 TICKER LIST
# ──────────────────────────────────────────────
def get_sp500_tickers() -> list[str]:
    """
    Download the current S&P 500 constituent list from DataHub.

    Returns
    -------
    list of ticker strings, with dots replaced by hyphens (Yahoo Finance format)
    e.g. "BRK.B" → "BRK-B"
    """
    try:
        sp500 = pd.read_csv(SP500_URL)
        tickers = sp500["Symbol"].tolist()
        tickers = [t.replace(".", "-") for t in tickers]
        logger.info(f"[DATA] Fetched {len(tickers)} S&P 500 tickers.")
        return tickers
    except Exception as e:
        logger.error(f"[DATA] Failed to fetch S&P 500 list: {e}")
        # Fallback: a hardcoded set of large-cap tickers
        logger.warning("[DATA] Using fallback ticker list.")
        return [
            "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG",
            "TSLA", "BRK-B", "JPM", "LLY", "V", "UNH", "XOM", "MA",
            "AVGO", "JNJ", "HD", "PG", "COST", "MRK", "ABBV", "CVX",
            "KO", "PEP", "WMT", "NFLX", "BAC", "CRM", "TMO",
        ]


# ──────────────────────────────────────────────
# 2. SCREEN BY LIQUIDITY (avg. daily volume)
# ──────────────────────────────────────────────
def screen_by_volume(tickers: list[str], top_n: int = TOP_N_TICKERS) -> list[str]:
    """
    Download 3 months of volume data for all tickers and rank by
    average daily volume. Returns the top_n most liquid symbols.

    Parameters
    ----------
    tickers : full S&P 500 ticker list
    top_n   : number of tickers to keep

    Returns
    -------
    list of the top_n ticker strings by liquidity
    """
    logger.info(f"[DATA] Screening {len(tickers)} tickers by volume...")

    try:
        raw = yf.download(
            tickers,
            period="3mo",
            group_by="ticker",
            auto_adjust=False,
            progress=False,
        )
    except Exception as e:
        logger.error(f"[DATA] Volume screen download failed: {e}")
        return tickers[:top_n]

    # Handle yfinance column ordering quirk
    if raw.columns.nlevels == 2 and raw.columns[0][0] in ["Open","High","Low","Close","Adj Close","Volume"]:
        raw = raw.swaplevel(axis=1)
    raw = raw.sort_index(axis=1)

    volume_dict = {}
    for ticker in tickers:
        try:
            avg_vol = raw[ticker]["Volume"].dropna().mean()
            if pd.notna(avg_vol) and avg_vol > 0:
                volume_dict[ticker] = avg_vol
        except Exception:
            continue

    sorted_tickers = sorted(volume_dict.items(), key=lambda x: x[1], reverse=True)
    top_tickers = [t[0] for t in sorted_tickers[:top_n]]

    logger.info(f"[DATA] Top {len(top_tickers)} tickers by avg volume selected.")
    return top_tickers


# ──────────────────────────────────────────────
# 3. FETCH OHLCV HISTORY FOR SCREENED TICKERS
# ──────────────────────────────────────────────
def fetch_ohlcv(tickers: list[str], period: str = HISTORY_DAYS) -> dict[str, pd.DataFrame]:
    """
    Download full OHLCV data for a list of tickers.

    Parameters
    ----------
    tickers : list of ticker strings
    period  : yfinance period string e.g. "6mo", "1y"

    Returns
    -------
    dict mapping ticker → clean DataFrame with columns:
        Date, Open, High, Low, Close, Adj Close, Volume
    """
    logger.info(f"[DATA] Downloading {period} OHLCV for {len(tickers)} tickers...")

    try:
        raw = yf.download(
            tickers,
            period=period,
            group_by="ticker",
            auto_adjust=False,
            progress=False,
        )
    except Exception as e:
        logger.error(f"[DATA] OHLCV download failed: {e}")
        return {}

    # Fix column ordering quirk
    if raw.columns.nlevels == 2 and raw.columns[0][0] in ["Open","High","Low","Close","Adj Close","Volume"]:
        raw = raw.swaplevel(axis=1)
    raw = raw.sort_index(axis=1)

    stock_data = {}
    for ticker in tickers:
        try:
            df = raw[ticker].copy().dropna(how="all")
            if len(df) < MIN_ROWS:
                logger.warning(f"[DATA] {ticker}: only {len(df)} rows — skipping (need {MIN_ROWS})")
                continue
            stock_data[ticker] = df
        except Exception as e:
            logger.warning(f"[DATA] {ticker}: failed to extract — {e}")
            continue

    logger.info(f"[DATA] Clean data ready for {len(stock_data)} tickers.")
    return stock_data


# ──────────────────────────────────────────────
# 4. CLEAN & VALIDATE
# ──────────────────────────────────────────────
def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply standard cleaning to a single ticker's DataFrame:
      - Ensure DatetimeIndex
      - Forward-fill small gaps (max 2 days, weekend gaps etc.)
      - Drop rows where Close is NaN after fill
      - Sort by date ascending

    Parameters
    ----------
    df : raw OHLCV DataFrame

    Returns
    -------
    cleaned DataFrame
    """
    df = df.copy()

    # Ensure DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    df = df.sort_index()
    df["Close"] = df["Close"].ffill(limit=2)
    df = df.dropna(subset=["Close"])
    return df


# ──────────────────────────────────────────────
# 5. MAIN ENTRY POINT  (called by main.py)
# ──────────────────────────────────────────────
def get_stock_data() -> dict[str, pd.DataFrame]:
    """
    Full pipeline: screen → fetch → clean.

    Called by main.py as:
        stock_data = get_stock_data()

    Returns
    -------
    dict {ticker: cleaned_DataFrame}  ready for scorer.build_rankings()
    """
    tickers     = get_sp500_tickers()
    top_tickers = screen_by_volume(tickers, top_n=TOP_N_TICKERS)
    raw_data    = fetch_ohlcv(top_tickers)

    clean_data = {}
    for ticker, df in raw_data.items():
        try:
            clean_data[ticker] = clean_dataframe(df)
        except Exception as e:
            logger.warning(f"[DATA] Cleaning failed for {ticker}: {e}")

    logger.info(f"[DATA] Pipeline complete — {len(clean_data)} tickers ready.")
    return clean_data


# ──────────────────────────────────────────────
# Standalone test
# ──────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data = get_stock_data()
    print(f"\nReady tickers: {len(data)}")
    for t, df in list(data.items())[:3]:
        print(f"  {t}: {len(df)} rows, last close={df['Close'].iloc[-1]:.2f}")
