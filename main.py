"""
main.py
=======
PPLE Investment Club · Algo Team
-------------------------------------------------
Entry point for the autonomous paper trading system.

Pipeline:
  1. DATA      — Alice:  screen top 100 S&P 500 by liquidity, fetch OHLCV
  2. SIGNALS   — Hugo:   compute SMA / Bollinger / RSI indicators, rank stocks
  3. RISK      — Jannis: size positions, run circuit breaker, log metrics
  4. EXECUTION          — submit bracket orders to Alpaca paper account
  5. REPORTING          — daily log, CSV export, benchmark vs SPY

Usage:
  python main.py                    # full daily run
  python main.py --report-only      # print report without trading
  python main.py --plot AAPL MSFT   # chart specific tickers
  python main.py --plot-top 5       # chart top 5 ranked tickers
  python main.py --export out.csv   # save rankings to CSV
  python main.py --no-execute       # run pipeline but skip order submission
  python main.py --no-color         # disable ANSI colors
"""

import argparse
import logging
import os
import sys
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate

from data_interface import get_stock_data
from scorer import build_rankings, signal_label, composite_label
from visualiser import plot_ticker, plot_top_n
from risk import run_risk_layer, execute_trades, daily_report


# ──────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            os.path.join(LOG_DIR, f"algo_{datetime.now().strftime('%Y%m%d')}.log")
        ),
    ],
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# ANSI COLOURS
# ──────────────────────────────────────────────
ANSI = {
    "green":  "\033[92m",
    "red":    "\033[91m",
    "yellow": "\033[93m",
    "cyan":   "\033[96m",
    "bold":   "\033[1m",
    "reset":  "\033[0m",
}

def colour(text: str, code: str, no_color: bool) -> str:
    return text if no_color else f"{ANSI[code]}{text}{ANSI['reset']}"


# ──────────────────────────────────────────────
# CLI ARGUMENTS
# ──────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PPLE Algo · Autonomous S&P 500 paper trading system"
    )
    p.add_argument("--top",          type=int,   default=None,  help="Show only top N in rankings table.")
    p.add_argument("--plot",         nargs="+",  metavar="TICKER", help="Plot chart(s) for specific ticker(s).")
    p.add_argument("--plot-top",     type=int,   default=None,  metavar="N", help="Plot charts for top N ranked tickers.")
    p.add_argument("--save-charts",  type=str,   default=None,  metavar="DIR", help="Save top/bottom charts to directory.")
    p.add_argument("--export",       type=str,   default=None,  metavar="FILE", help="Export rankings to CSV.")
    p.add_argument("--no-color",     action="store_true", help="Disable ANSI colours.")
    p.add_argument("--no-execute",   action="store_true", help="Skip order submission (dry run).")
    p.add_argument("--report-only",  action="store_true", help="Print daily report only, skip full pipeline.")
    return p.parse_args()


# ──────────────────────────────────────────────
# RANKINGS TABLE
# ──────────────────────────────────────────────
def _sentiment_colour(label: str, no_color: bool) -> str:
    if no_color:
        return label
    if "STRONG BUY" in label:
        return f"{ANSI['green']}{ANSI['bold']}{label}{ANSI['reset']}"
    elif "BUY" in label:
        return f"{ANSI['green']}{label}{ANSI['reset']}"
    elif "STRONG SELL" in label:
        return f"{ANSI['red']}{ANSI['bold']}{label}{ANSI['reset']}"
    elif "SELL" in label:
        return f"{ANSI['red']}{label}{ANSI['reset']}"
    return f"{ANSI['yellow']}{label}{ANSI['reset']}"


def print_rankings(ranked_df: pd.DataFrame, top: int | None, no_color: bool) -> None:
    df = ranked_df.copy()
    if top is not None:
        df = df.head(top)

    df["Sentiment"] = df["composite"].apply(
        lambda s: _sentiment_colour(composite_label(s), no_color)
    )
    for col in ["sma_signal", "bb_signal", "rsi_signal"]:
        df[col] = df[col].apply(signal_label)

    display_df = df[[
        "ticker", "composite", "Sentiment",
        "sma_signal", "bb_signal", "rsi_signal",
        "close", "rsi", "pct_b",
    ]].copy()
    display_df.columns = ["Ticker", "Score", "Sentiment", "SMA", "BB", "RSI_sig", "Close", "RSI", "%B"]

    header = colour("  PPLE ALGO · STOCK RANKINGS", "cyan", no_color)
    print(f"\n{'═'*72}")
    print(header)
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}   |   {len(ranked_df)} tickers ranked")
    print(f"{'═'*72}")
    print(tabulate(display_df, headers="keys", tablefmt="rounded_outline", floatfmt=".2f", showindex=True))
    print(f"\n  Score: −3.5 (strong sell) → +3.5 (strong buy)\n")


# ──────────────────────────────────────────────
# BENCHMARK: compare vs SPY
# ──────────────────────────────────────────────
def print_benchmark(no_color: bool) -> None:
    """
    Pull SPY daily return from yfinance and compare to portfolio P&L.
    Prints a simple one-line benchmark summary.
    """
    try:
        import yfinance as yf
        spy = yf.download("SPY", period="2d", progress=False, auto_adjust=True)
        if len(spy) >= 2:
            spy_ret = (spy["Close"].iloc[-1] / spy["Close"].iloc[-2] - 1) * 100
            bench_line = f"  SPY today: {spy_ret:+.2f}%"
            col = "green" if spy_ret >= 0 else "red"
            print(colour(bench_line, col, no_color))
    except Exception:
        pass


# ──────────────────────────────────────────────
# TRADE LOG  (appended to daily CSV)
# ──────────────────────────────────────────────
def log_trades(sized_positions: dict, ranked_df: pd.DataFrame) -> None:
    """
    Write a trade log entry for today's selected positions.
    Saved to logs/trades_YYYYMMDD.csv
    """
    today = datetime.now().strftime("%Y%m%d")
    log_path = os.path.join(LOG_DIR, f"trades_{today}.csv")

    rows = []
    for ticker, dollar_amt in sized_positions.items():
        score = ranked_df.loc[ranked_df["ticker"] == ticker, "composite"]
        score_val = float(score.iloc[0]) if not score.empty else None
        rows.append({
            "timestamp": datetime.now().isoformat(),
            "ticker":    ticker,
            "dollar_amt": dollar_amt,
            "score":     score_val,
            "sentiment": composite_label(score_val) if score_val is not None else "N/A",
        })

    if rows:
        pd.DataFrame(rows).to_csv(log_path, index=False)
        logger.info(f"[LOG] Trade log saved → {log_path}")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    logger.info("=" * 60)
    logger.info("  PPLE INVESTMENT CLUB · ALGO SYSTEM  STARTING")
    logger.info("=" * 60)

    # ── Report-only mode ────────────────────────────────────────
    if args.report_only:
        logger.info("[MODE] Report-only — fetching Alpaca metrics...")
        try:
            daily_report()
            print_benchmark(args.no_color)
        except Exception as e:
            logger.error(f"[ERROR] Report failed: {e}")
        return

    # ── Step 1: Fetch data ───────────────────────────────────────
    logger.info("[STEP 1/4] Fetching & screening stock data...")
    try:
        stock_data = get_stock_data()
    except Exception as e:
        logger.error(f"[ERROR] Data fetch failed: {e}")
        sys.exit(1)

    if not stock_data:
        logger.error("[ERROR] No data returned from get_stock_data(). Exiting.")
        sys.exit(1)
    logger.info(f"[STEP 1] ✓ Data ready for {len(stock_data)} tickers.")

    # ── Step 2: Build rankings ───────────────────────────────────
    logger.info("[STEP 2/4] Computing indicators and scoring...")
    ranked_df = build_rankings(stock_data)

    if ranked_df.empty:
        logger.error("[ERROR] Rankings produced no results. Exiting.")
        sys.exit(1)
    logger.info(f"[STEP 2] ✓ Ranked {len(ranked_df)} tickers.")

    # Print the table
    print_rankings(ranked_df, top=args.top, no_color=args.no_color)
    print_benchmark(args.no_color)

    # Export CSV if requested
    if args.export:
        ranked_df.to_csv(args.export, index=True)
        logger.info(f"[EXPORT] Rankings saved → {args.export}")

    # Auto-save dated rankings CSV to data/ folder
    dated_csv = os.path.join("data", f"rankings_{datetime.now().strftime('%Y%m%d')}.csv")
    os.makedirs("data", exist_ok=True)
    ranked_df.to_csv(dated_csv, index=True)
    logger.info(f"[DATA] Rankings auto-saved → {dated_csv}")

    # ── Step 3: Risk layer ───────────────────────────────────────
    logger.info("[STEP 3/4] Running risk layer...")
    ranked_candidates = ranked_df["ticker"].tolist()

    try:
        sized_positions, cb_triggered = run_risk_layer(ranked_candidates)
    except Exception as e:
        logger.error(f"[ERROR] Risk layer failed: {e}")
        logger.warning("[WARN] Continuing without execution.")
        sized_positions, cb_triggered = {}, True

    if cb_triggered:
        logger.warning("[RISK] Circuit breaker ACTIVE — no trades executed today.")
    else:
        logger.info(f"[STEP 3] ✓ {len(sized_positions)} positions sized.")

    # ── Step 4: Execute ──────────────────────────────────────────
    logger.info("[STEP 4/4] Submitting orders to Alpaca...")
    if args.no_execute:
        logger.info("[MODE] --no-execute flag set — skipping order submission.")
        print("\n  [DRY RUN] Would submit the following positions:")
        for t, amt in sized_positions.items():
            print(f"    {t:<6}  ${amt:,.2f}")
    elif cb_triggered:
        logger.warning("[EXEC] Skipping execution — circuit breaker is active.")
    else:
        try:
            execute_trades(sized_positions)
            log_trades(sized_positions, ranked_df)
            logger.info("[STEP 4] ✓ Orders submitted.")
        except Exception as e:
            logger.error(f"[ERROR] Execution failed: {e}")

    # ── Charts ───────────────────────────────────────────────────
    if args.plot:
        for ticker in args.plot:
            ticker = ticker.upper()
            if ticker not in stock_data:
                logger.warning(f"[CHART] {ticker} not in data, skipping.")
                continue
            plot_ticker(ticker, stock_data[ticker])

    if args.plot_top:
        for ticker in ranked_df.head(args.plot_top)["ticker"].tolist():
            plot_ticker(ticker, stock_data[ticker])

    if args.save_charts:
        os.makedirs(args.save_charts, exist_ok=True)
        plot_top_n(ranked_df, stock_data, n=10, save_dir=args.save_charts)
        logger.info(f"[CHART] Charts saved to {args.save_charts}/")

    if args.plot or args.plot_top:
        plt.show()

    logger.info("=" * 60)
    logger.info("  PPLE ALGO · RUN COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
