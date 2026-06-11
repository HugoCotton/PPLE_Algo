# PPLE_Algo
PPLE Investment Club's Algorithmic Portfolio's trading system

This autonomous paper trading system (using Alapaca AI) will operate daily using S&P 500 data
Our finished system aims to:
    1. Select its own stocks (screening top 100 most liquid S&P 500 tickers)
    2. Generates trade signals (uses a variety of indicators to rank held and potential stocks)
    3. Manages its own risk (sizes positions equally, exiting poorly-performing positions)
    4. Executes autonomously (operates on dedicated server, logs trades every trading day)
    5. Reports own performance (outputs daily measures of success, and benchmarks against S&P 500)

3 Main Elements to the program: Data, Signals and Risk & Metrics

*DATA* - Alice Galzin
- Screens the S&P 500 Universe daily by avg. daily volume
- Fetches prices using the yFinance library (OHLCV)
- Cleans up data for use by signals

*SIGNALS* - Hugo Cotton
- Performs (simple momentum) analyses upon given stocks
    - 20/50 day SMA Crossover - Measure of recent vs longer term volatility, demonstrates changes in trends
    - Bollinger Bands - Measure of volatility, using a historical moving SD to indicate relative highs or lows
    - RSI Filter - Measure of the oversold or overbought nature of the stock based off of recent avg. gains and losses
- Combines analyses to rank the stocks, and identify candidates for entry

*RISK & METRICS* - Jannis Lommen
- Appropriately weights the approx. 10 open positions
- Stop-loss guards any position that drops >5%
- Monitors daily performance using Sharpe ratios and drawdown outputs

Following these modules, orders are generated and executed using the Alpaca API, logging each trade with timestamp and reason



# PPLE Investment Club · Algo Trading System
## Algorithmic Paper Trading on S&P 500

A fully autonomous paper-trading system built by the PPLE Investment Club Algo Team.  
Runs daily against Alpaca's paper trading API. No real money is at risk.

---

## Team

| Module | Member | Responsibility |
|---|---|---|
| `data_interface.py` | Alice Galzin | S&P 500 screening, OHLCV fetching, data cleaning |
| `indicators.py` + `scorer.py` | Hugo Cotton | SMA crossover, Bollinger Bands, RSI, ranking |
| `risk.py` | Jannis Lommen | Position sizing, stop-losses, circuit breaker, metrics |
| `main.py` + `scheduler.py` | Shared | Orchestration, logging, execution, scheduling |

---

## System Overview

```
S&P 500 Universe (~500 tickers)
        │
        ▼
[DATA] Screen top 100 by avg daily volume
        │
        ▼
[SIGNALS] Compute SMA 20/50 crossover + Bollinger Bands + RSI
          → Score each ticker −3.5 to +3.5
          → Rank candidates, best first
        │
        ▼
[RISK] Circuit breaker check (−10% daily drawdown halts all trades)
       Size top 10 positions at 10% equity each
       Bracket orders with 5% stop-loss
        │
        ▼
[ALPACA] Submit bracket orders to paper account
        │
        ▼
[LOG] Daily report: Sharpe ratio, max drawdown, P&L vs SPY
      Trade log CSV saved to logs/
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Alpaca API keys
```bash
export ALPACA_API_KEY="your_key_here"
export ALPACA_SECRET_KEY="your_secret_here"
```
Get paper trading keys free at: https://alpaca.markets

### 3. Run the pipeline once
```bash
python main.py
```

### 4. Run daily at 09:35 ET automatically (server mode)
```bash
python scheduler.py
```

---

## CLI Options

### main.py
```
python main.py                    # full pipeline: screen → rank → size → execute
python main.py --no-execute       # dry run (no orders sent to Alpaca)
python main.py --report-only      # print today's Alpaca metrics only
python main.py --top 20           # show only top 20 in rankings table
python main.py --plot AAPL MSFT   # show interactive charts
python main.py --plot-top 5       # chart the top 5 ranked tickers
python main.py --save-charts ./charts  # save top/bottom charts as PNGs
python main.py --export out.csv   # save rankings to CSV
python main.py --no-color         # disable ANSI colours (for log files)
```

### scheduler.py
```
python scheduler.py               # run daily at 09:35 ET
python scheduler.py --time 15:55  # run near market close instead
python scheduler.py --run-now     # fire once immediately, then schedule
python scheduler.py --no-execute  # dry run mode
```

---

## Scoring Logic

Each ticker receives a composite score from three independent signals:

| Signal | Bullish (+1) | Bearish (−1) | Neutral (0) |
|---|---|---|---|
| SMA Crossover | SMA20 > SMA50 | SMA20 < SMA50 | Equal |
| Bollinger Bands | Close ≤ Lower Band | Close ≥ Upper Band | Inside bands |
| RSI (14) | RSI < 30 | RSI > 70 | 30–70 |
| Fresh Cross Bonus | +0.5 (recent bullish) | −0.5 (recent bearish) | — |

**Composite range: −3.5 (strong sell) → +3.5 (strong buy)**

---

## Risk Controls

- **Equal weight**: 10% of portfolio per position, max 10 positions
- **Stop-loss**: 5% below entry price (bracket order)
- **Circuit breaker**: halts all trading if daily portfolio drawdown exceeds −10%
- **Daily metrics**: Sharpe ratio (30-day annualised) + max drawdown

---

## Output Files

```
logs/
  algo_YYYYMMDD.log       # full run log
  trades_YYYYMMDD.csv     # positions entered today

data/
  rankings_YYYYMMDD.csv   # full scored + ranked table

charts/
  AAPL.png, MSFT.png …   # saved chart images (if --save-charts used)
```

---

## Technical Indicators — Detail

### SMA Crossover (20/50-day)
The 20-day SMA responds quickly to recent price action; the 50-day represents medium-term trend.  
When 20 crosses above 50 → trend turning bullish. A **fresh cross** (within 3 days) earns a +0.5 bonus.

### Bollinger Bands (20-day, 2σ)
Bands are set 2 standard deviations above/below a 20-day moving average.  
Price at the lower band → oversold (buy signal). Price at upper band → overbought (sell signal).  
%B metric shows exactly where price sits within the bands (0 = lower, 1 = upper).

### RSI — Relative Strength Index (14-day)
Wilder's momentum oscillator. RSI < 30 = oversold (buy). RSI > 70 = overbought (sell).  
Calculated using exponential smoothing of average gains vs average losses.

---

*Built for the PPLE Investment Club, University of Amsterdam · 2025*
