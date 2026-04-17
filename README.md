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
        Screens the S&P 500 Universe daily by avg. daily volume
        Fetches prices using the yFinance library (OHLCV)
        Cleans up data for use by signals

    *SIGNALS* - Hugo Cotton
        Performs (simple momentum) analyses upon given stocks
            20/50 day SMA Crossover - Measure of recent vs longer term volatility, demonstrates changes in trends
            Bollinger Bands - Measure of volatility, using a historical moving SD to indicate relative highs or lows
            RSI Filter - Measure of the oversold or overbought nature of the stock based off of recent avg. gains and losses
        Combines analyses to rank the stocks, and identify candidates for entry

    *RISK & METRICS* - Jannis Lommen
        Appropriately weights the approx. 10 open positions
        Stop-loss guards any position that drops >5%
        Monitors daily performance using Sharpe ratios and drawdown outputs

Following these modules, orders are generated and executed using the Alpaca API, logging each trade with timestamp and reason