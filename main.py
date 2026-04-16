"""
main.py
=======
Entry point for the stock analysis program.

Usage examples:
  python main.py                        # rank all tickers, print table
  python main.py --top 20               # show only top 20 in table
  python main.py --plot AAPL            # show chart for one ticker
  python main.py --plot AAPL MSFT TSLA  # show charts for several tickers
  python main.py --plot-top 5           # chart the top 5 ranked tickers
  python main.py --save-charts ./charts # save all top/bottom charts to folder
  python main.py --export rankings.csv  # save rankings table to CSV
"""

import argparse
import os
import sys

import pandas as pd
from tabulate import tabulate

from data_interface import get_stock_data
from scorer import build_rankings, signal_label, composite_label
from visualiser import plot_ticker, plot_top_n


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """
    Define and parse command-line arguments.

    All arguments are optional — running with no arguments prints the
    full ranked table to the terminal.

    Returns:
        argparse.Namespace with attributes:
          top (int):          Show only the top N rows in the table.
          plot (list[str]):   List of ticker symbols to chart individually.
          plot_top (int):     Chart the top N tickers from the rankings.
          save_charts (str):  Directory path to save chart PNGs into.
          export (str):       File path to write the rankings CSV to.
          no_color (bool):    Disable ANSI colour in terminal output.
    """
    parser = argparse.ArgumentParser(
        description="Stock technical analysis ranker (SMA crossover, Bollinger Bands, RSI)"
    )
    parser.add_argument(
        "--top", type=int, default=None,
        help="Show only the top N tickers in the rankings table."
    )
    parser.add_argument(
        "--plot", nargs="+", metavar="TICKER",
        help="Show interactive chart(s) for the specified ticker(s)."
    )
    parser.add_argument(
        "--plot-top", type=int, default=None, metavar="N",
        help="Show charts for the top N ranked tickers."
    )
    parser.add_argument(
        "--save-charts", type=str, default=None, metavar="DIR",
        help="Save charts for top/bottom 10 tickers to this directory."
    )
    parser.add_argument(
        "--export", type=str, default=None, metavar="FILE",
        help="Export the rankings table to a CSV file."
    )
    parser.add_argument(
        "--no-color", action="store_true",
        help="Disable ANSI colour codes in terminal output."
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Terminal output helpers
# ---------------------------------------------------------------------------

# ANSI colour codes — used to colour-code the sentiment column
ANSI = {
    "green":  "\033[92m",
    "red":    "\033[91m",
    "yellow": "\033[93m",
    "reset":  "\033[0m",
}

def _sentiment_colour(label: str, no_color: bool) -> str:
    """
    Wrap a sentiment label string in ANSI colour codes for terminal output.

    Colours used:
      STRONG BUY / BUY   → green
      STRONG SELL / SELL → red
      NEUTRAL            → yellow

    Args:
        label:    The sentiment string (e.g. "STRONG BUY").
        no_color: If True, returns the label unchanged (no ANSI codes).

    Returns:
        The label string, optionally wrapped in ANSI escape sequences.
    """
    if no_color:
        return label
    if "BUY" in label:
        return f"{ANSI['green']}{label}{ANSI['reset']}"
    elif "SELL" in label:
        return f"{ANSI['red']}{label}{ANSI['reset']}"
    return f"{ANSI['yellow']}{label}{ANSI['reset']}"


def print_rankings(ranked_df: pd.DataFrame, top: int | None, no_color: bool) -> None:
    """
    Print the ranked ticker table to stdout using tabulate.

    Selects a subset of columns for readability, adds a human-readable
    "sentiment" column, and formats numeric values to 2 decimal places.

    The table is printed in "rounded_outline" tabulate style, which draws
    a clean box around the table with rounded corners.

    Args:
        ranked_df: The full ranked DataFrame from build_rankings().
        top:       If provided, only the top N rows are shown.
        no_color:  If True, no ANSI colour codes are used.
    """
    df = ranked_df.copy()

    if top is not None:
        df = df.head(top)

    # Add readable sentiment column
    df["sentiment"] = df["composite"].apply(
        lambda s: _sentiment_colour(composite_label(s), no_color)
    )

    # Format signal columns
    for col in ["sma_signal", "bb_signal", "rsi_signal"]:
        df[col] = df[col].apply(signal_label)

    # Select display columns
    display_cols = [
        "ticker", "composite", "sentiment",
        "sma_signal", "bb_signal", "rsi_signal",
        "close", "rsi", "pct_b",
    ]

    display_df = df[display_cols].copy()
    display_df.columns = [
        "Ticker", "Score", "Sentiment",
        "SMA", "BB", "RSI_sig",
        "Close", "RSI", "%B",
    ]

    print("\n" + "═" * 72)
    print("  STOCK TECHNICAL ANALYSIS — RANKED RESULTS")
    print("═" * 72)
    print(tabulate(
        display_df,
        headers="keys",
        tablefmt="rounded_outline",
        floatfmt=".2f",
        showindex=True,
    ))
    print(f"\n  Total tickers ranked: {len(ranked_df)}")
    print("  Score range: -3.5 (strong sell) → +3.5 (strong buy)\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Orchestrate the full analysis pipeline.

    Steps:
      1. Parse CLI arguments.
      2. Fetch stock data via data_interface.get_stock_data().
      3. Build rankings via scorer.build_rankings().
      4. Print the ranked table.
      5. Handle optional chart display / saving / CSV export.

    Any step that fails prints a message and exits cleanly rather than
    crashing with a raw Python traceback.
    """
    args = parse_args()

    # --- Step 1: Fetch data ---
    print("[INFO] Fetching stock data...")
    try:
        stock_data = get_stock_data()
    except NotImplementedError:
        print("[ERROR] get_stock_data() is not implemented yet.")
        print("        Edit data_interface.py to plug in your data source.")
        sys.exit(1)
    except Exception as exc:
        print(f"[ERROR] Failed to fetch stock data: {exc}")
        sys.exit(1)

    if not stock_data:
        print("[ERROR] get_stock_data() returned an empty dict.")
        sys.exit(1)

    print(f"[INFO] Loaded data for {len(stock_data)} tickers.")

    # --- Step 2: Build rankings ---
    print("[INFO] Computing indicators and scoring...")
    ranked_df = build_rankings(stock_data)

    if ranked_df.empty:
        print("[ERROR] Ranking produced no results.")
        sys.exit(1)

    # --- Step 3: Print table ---
    print_rankings(ranked_df, top=args.top, no_color=args.no_color)

    # --- Step 4: Export CSV ---
    if args.export:
        ranked_df.to_csv(args.export)
        print(f"[INFO] Rankings exported to {args.export}")

    # --- Step 5: Individual ticker charts ---
    if args.plot:
        for ticker in args.plot:
            ticker = ticker.upper()
            if ticker not in stock_data:
                print(f"[WARN] {ticker} not found in stock data, skipping.")
                continue
            print(f"[CHART] Plotting {ticker}...")
            plot_ticker(ticker, stock_data[ticker])

    # --- Step 6: Top-N charts ---
    if args.plot_top:
        top_tickers = ranked_df.head(args.plot_top)["ticker"].tolist()
        for ticker in top_tickers:
            print(f"[CHART] Plotting {ticker}...")
            plot_ticker(ticker, stock_data[ticker])

    # --- Step 7: Save charts ---
    if args.save_charts:
        os.makedirs(args.save_charts, exist_ok=True)
        print(f"[INFO] Saving charts for top/bottom 10 to {args.save_charts}/")
        plot_top_n(ranked_df, stock_data, n=10, save_dir=args.save_charts)


if __name__ == "__main__":
    main()