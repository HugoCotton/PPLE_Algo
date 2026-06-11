"""
risk.py — Member C: Risk & Metrics
PPLE Investment Club · Algo Team
-------------------------------------------------
Inputs  : ranked candidate list from signals.py
          Alpaca API object (passed in from main.py)
Outputs : sized positions ready for execution
          daily Sharpe ratio + max drawdown report
          circuit breaker flag (v2)
-------------------------------------------------
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import alpaca_trade_api as tradeapi

# ──────────────────────────────────────────────
# 1. CONNECT TO ALPACA
# ──────────────────────────────────────────────
API_KEY    = os.environ.get("ALPACA_API_KEY")
SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY")
BASE_URL   = "https://paper-api.alpaca.markets"

api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version="v2")


# ──────────────────────────────────────────────
# 2. POSITION SIZER
# ──────────────────────────────────────────────
def size_positions(ranked_candidates: list[str]) -> dict:
    """
    Takes the ranked list from signals.py.
    Returns a dict of {ticker: dollar_amount} for up to 10 positions.
    Each position gets an equal 10% of total equity.

    Parameters
    ----------
    ranked_candidates : list of ticker strings, best first

    Returns
    -------
    dict  e.g. {"AAPL": 1500.0, "MSFT": 1500.0, ...}
    """
    account     = api.get_account()
    equity      = float(account.equity)
    max_pos     = 10
    weight      = 0.10                          # 10% per position (equal weight)

    # Only take top 10 candidates
    candidates  = ranked_candidates[:max_pos]
    dollar_size = equity * weight

    sized = {ticker: round(dollar_size, 2) for ticker in candidates}

    print(f"\n[RISK] Account equity : ${equity:,.2f}")
    print(f"[RISK] Position size  : ${dollar_size:,.2f} per stock (10%)")
    print(f"[RISK] Candidates     : {list(sized.keys())}")
    return sized


# ──────────────────────────────────────────────
# 3. STOP-LOSS CALCULATOR
# ──────────────────────────────────────────────
def calculate_stop_loss(ticker: str, stop_pct: float = 0.05) -> float:
    """
    Fetches the latest price for a ticker and returns
    the stop-loss price (default: 5% below entry).

    Parameters
    ----------
    ticker   : stock symbol e.g. "AAPL"
    stop_pct : fraction below entry to place stop (default 0.05 = 5%)

    Returns
    -------
    float  stop-loss price
    """
    latest      = api.get_latest_trade(ticker)
    entry_price = float(latest.price)
    stop_price  = round(entry_price * (1 - stop_pct), 2)

    print(f"[RISK] {ticker} entry=${entry_price:.2f}  stop=${stop_price:.2f} (-{stop_pct*100:.0f}%)")
    return stop_price


# ──────────────────────────────────────────────
# 4. CIRCUIT BREAKER  (v2)
# ──────────────────────────────────────────────
def circuit_breaker_triggered(drawdown_limit: float = 0.10) -> bool:
    """
    Checks whether the total portfolio is down more than
    drawdown_limit (default 10%) from the starting equity.

    If triggered → returns True and main.py should skip execution.

    Parameters
    ----------
    drawdown_limit : float, fraction of portfolio loss to trigger halt

    Returns
    -------
    bool  True = stop all trading, False = safe to trade
    """
    account        = api.get_account()
    equity         = float(account.equity)
    last_equity    = float(account.last_equity)   # previous day's equity

    daily_change   = (equity - last_equity) / last_equity

    if daily_change <= -drawdown_limit:
        print(f"\n🚨 [CIRCUIT BREAKER] Portfolio down {daily_change*100:.1f}% — ALL TRADING PAUSED.")
        return True

    print(f"[RISK] Daily P&L: {daily_change*100:+.2f}% — Circuit breaker: OFF")
    return False


# ──────────────────────────────────────────────
# 5. PERFORMANCE METRICS
# ──────────────────────────────────────────────
def get_portfolio_history(days: int = 30) -> pd.DataFrame:
    """
    Pulls portfolio equity history from Alpaca for the last N days.

    Returns
    -------
    DataFrame with columns: ['timestamp', 'equity', 'daily_return']
    """
    end   = datetime.now()
    start = end - timedelta(days=days)

    history = api.get_portfolio_history(
        date_start = start.strftime("%Y-%m-%d"),
        date_end   = end.strftime("%Y-%m-%d"),
        timeframe  = "1D"
    )

    df = pd.DataFrame({
        "timestamp" : pd.to_datetime(history.timestamp, unit="s"),
        "equity"    : history.equity
    })

    df["daily_return"] = df["equity"].pct_change().fillna(0)
    return df


def calculate_sharpe(df: pd.DataFrame, risk_free_rate: float = 0.0) -> float:
    """
    Sharpe Ratio = (mean daily return - risk free rate) / std of daily returns
    Annualised by multiplying by sqrt(252).

    A Sharpe > 1 is generally considered good.
    """
    returns = df["daily_return"]
    excess  = returns - (risk_free_rate / 252)

    if returns.std() == 0:
        return 0.0

    sharpe = (excess.mean() / returns.std()) * np.sqrt(252)
    return round(sharpe, 3)


def calculate_max_drawdown(df: pd.DataFrame) -> float:
    """
    Max Drawdown = largest peak-to-trough drop in portfolio value.
    Expressed as a negative percentage, e.g. -0.12 = -12%.

    This is the "pain metric" — how bad did it get at worst?
    """
    equity      = df["equity"]
    rolling_max = equity.cummax()
    drawdown    = (equity - rolling_max) / rolling_max
    return round(drawdown.min(), 4)


def daily_report():
    """
    Prints the full daily performance report:
      - Current equity
      - Daily P&L vs yesterday
      - Sharpe ratio (30-day)
      - Max drawdown (30-day)
      - Open positions
    """
    print("\n" + "="*50)
    print("  PPLE ALGO · DAILY RISK REPORT")
    print("  " + datetime.now().strftime("%Y-%m-%d %H:%M"))
    print("="*50)

    # Account snapshot
    account      = api.get_account()
    equity       = float(account.equity)
    last_equity  = float(account.last_equity)
    daily_pnl    = equity - last_equity
    daily_pct    = (daily_pnl / last_equity) * 100

    print(f"\n  Equity today     : ${equity:>12,.2f}")
    print(f"  Yesterday equity : ${last_equity:>12,.2f}")
    print(f"  Daily P&L        : ${daily_pnl:>+12,.2f}  ({daily_pct:+.2f}%)")

    # Sharpe + Drawdown
    df      = get_portfolio_history(days=30)
    sharpe  = calculate_sharpe(df)
    mdd     = calculate_max_drawdown(df)

    print(f"\n  Sharpe ratio     : {sharpe:>8.3f}  (30-day, annualised)")
    print(f"  Max drawdown     : {mdd*100:>7.2f}%  (30-day)")

    # Open positions
    positions = api.list_positions()
    print(f"\n  Open positions   : {len(positions)}/10")
    for p in positions:
        unrealised = float(p.unrealized_plpc) * 100
        print(f"    {p.symbol:<6}  qty={p.qty:<4}  P&L={unrealised:+.2f}%")

    # Circuit breaker status
    triggered = circuit_breaker_triggered()
    print(f"\n  Circuit breaker  : {'🔴 TRIGGERED' if triggered else '🟢 OK'}")
    print("="*50 + "\n")

    return {
        "equity"      : equity,
        "daily_pnl"   : daily_pnl,
        "daily_pct"   : daily_pct,
        "sharpe"      : sharpe,
        "max_drawdown": mdd,
        "cb_triggered": triggered
    }


# ──────────────────────────────────────────────
# 6. MAIN ENTRY POINT
#    Called by main.py like:
#    from risk import run_risk_layer
#    positions, cb = run_risk_layer(ranked_candidates)
# ──────────────────────────────────────────────
def run_risk_layer(ranked_candidates: list[str]) -> tuple[dict, bool]:
    """
    Master function called by main.py.

    Steps:
      1. Check circuit breaker — if triggered, return empty + True
      2. Size positions (10% equal weight, max 10)
      3. Print daily report

    Returns
    -------
    sized_positions : dict {ticker: dollar_amount}
    cb_triggered    : bool (True = don't execute trades today)
    """
    print("\n[RISK] Running risk layer...")

    # Step 1 — Circuit breaker check
    if circuit_breaker_triggered():
        print("[RISK] Circuit breaker active — returning empty positions.")
        return {}, True

    # Step 2 — Size positions
    sized = size_positions(ranked_candidates)

    # Step 3 — Daily report
    daily_report()

    return sized, False

# ──────────────────────────────────────────────
# 7. EXECUTION — submits orders to Alpaca
# ──────────────────────────────────────────────
def execute_trades(sized_positions: dict):
    """
    Takes sized positions from run_risk_layer and submits
    bracket orders to Alpaca (buy + stop-loss in one go).
    """
    existing = [p.symbol for p in api.list_positions()]

    for ticker, dollar_amount in sized_positions.items():

        # Skip if we already own this stock
        if ticker in existing:
            print(f"[EXEC] {ticker} — already held, skipping")
            continue

        # Get current price and calculate how many shares to buy
        latest      = api.get_latest_trade(ticker)
        price       = float(latest.price)
        qty         = int(dollar_amount // price)
        stop_price  = round(price * 0.95, 2)  # 5% stop-loss

        if qty < 1:
            print(f"[EXEC] {ticker} — not enough funds for 1 share, skipping")
            continue

        # Submit bracket order (buy + stop-loss together)
        api.submit_order(
            symbol        = ticker,
            qty           = qty,
            side          = "buy",
            type          = "market",
            time_in_force = "day",
            order_class   = "bracket",
            stop_loss     = {"stop_price": stop_price}
        )

        print(f"[EXEC] ✅ Order submitted: {ticker}  qty={qty}  stop=${stop_price}")

# ──────────────────────────────────────────────
# Quick standalone test
# ──────────────────────────────────────────────
if __name__ == "__main__":
    # Simulate a ranked list from signals.py
    test_candidates = ["AAPL", "MSFT", "NVDA", "AMZN", "META",
                       "GOOGL", "TSLA", "JPM", "V", "UNH", "LLY"]

    sized_positions, halted = run_risk_layer(test_candidates)

    if halted:
        print("Trading halted today.")
    else:
        print("\nFinal sized positions:")
        for ticker, amount in sized_positions.items():
            print(f"  {ticker}: ${amount:,.2f}")

