"""
data_interface.py
=================
Loads stock data from a local CSV file.

Expected CSV format (matches top_100_clean_ohlcv.csv):
  Date        — string, parseable as a date (e.g. "2026-01-16")
  Ticker      — string ticker symbol (e.g. "AAPL")
  Open        — float
  High        — float
  Low         — float
  Close       — float
  Adj Close   — float  (loaded but not used in analysis)
  Volume      — int

The file contains all tickers stacked in one flat table (long format).
get_stock_data() splits it into a dict of per-ticker DataFrames, each
with a DatetimeIndex sorted ascending, ready for the indicators pipeline.

To point at a different file, change CSV_PATH below.
"""

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration — change this path if your CSV lives elsewhere
# ---------------------------------------------------------------------------

CSV_PATH = "top_100_clean_ohlcv.csv"

# Column name mapping: CSV column → internal name expected by indicators.py
# If your CSV uses different column headers, edit the keys here.
COLUMN_MAP = {
    "Date":      "Date",        # will become the DatetimeIndex
    "Ticker":    "Ticker",      # used to split into per-ticker DataFrames
    "Open":      "Open",
    "High":      "High",
    "Low":       "Low",
    "Close":     "Close",
    "Adj Close": "Adj Close",   # retained but not used in scoring
    "Volume":    "Volume",
}


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def get_ticker_list() -> list[str]:
    """
    Return a sorted list of all unique ticker symbols found in the CSV.

    Reads only the "Ticker" column (no need to load the full file) and
    returns deduplicated, uppercase-sorted symbols.

    Returns:
        A list of strings, e.g. ["AAPL", "AMZN", "GOOG", ...]
    """
    tickers = pd.read_csv(CSV_PATH, usecols=["Ticker"])["Ticker"]
    return sorted(tickers.unique().tolist())


def get_stock_data() -> dict[str, pd.DataFrame]:
    """
    Load the full CSV and split it into a dict of per-ticker DataFrames.

    Steps:
      1. Read the entire CSV into a single DataFrame.
      2. Rename columns according to COLUMN_MAP (no-op if names already match).
      3. Parse the Date column as datetime and set it as the index.
      4. Sort each ticker's rows by date ascending (oldest first).
      5. Split by ticker into individual DataFrames.
      6. Warn and skip any ticker with fewer than 60 rows (indicators need
         at least 50 for SMA50 plus a warmup buffer).

    Returns:
        dict mapping ticker string → pd.DataFrame with:
          - DatetimeIndex (ascending)
          - Columns: Open, High, Low, Close, Adj Close, Volume
    """
    # --- Load ---
    raw = pd.read_csv(CSV_PATH)

    # --- Validate expected columns are present ---
    missing = [col for col in COLUMN_MAP.keys() if col not in raw.columns]
    if missing:
        raise ValueError(
            f"CSV is missing expected columns: {missing}\n"
            f"Found columns: {list(raw.columns)}"
        )

    # --- Rename to internal names (harmless if names already match) ---
    raw = raw.rename(columns=COLUMN_MAP)

    # --- Parse dates and set as index ---
    raw["Date"] = pd.to_datetime(raw["Date"])
    raw = raw.sort_values(["Ticker", "Date"])

    # --- Split into per-ticker DataFrames ---
    result: dict[str, pd.DataFrame] = {}

    for ticker, group in raw.groupby("Ticker"):
        df = (
            group
            .drop(columns=["Ticker"])   # ticker identity now lives in the dict key
            .set_index("Date")
            .sort_index()               # ensure ascending date order
        )

        if len(df) < 60:
            print(
                f"[WARN] {ticker}: only {len(df)} rows — "
                f"need at least 60 for reliable indicator calculation. Skipping."
            )
            continue

        result[ticker] = df

    print(f"[INFO] Loaded {len(result)} tickers from '{CSV_PATH}'.")
    return result