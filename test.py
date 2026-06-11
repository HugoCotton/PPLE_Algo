import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta

class MomentumStrategy:
    def __init__(self, symbol, start_date, end_date, initial_capital=10000, 
                 momentum_period=20, position_size=1.0):
        """
        Initialize momentum trading strategy
        
        Parameters:
        - symbol: Stock ticker symbol
        - start_date: Backtest start date
        - end_date: Backtest end date
        - initial_capital: Starting capital
        - momentum_period: Lookback period for momentum calculation
        - position_size: Fraction of capital to use per trade (0-1)
        """
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.momentum_period = momentum_period
        self.position_size = position_size
        self.data = None
        self.results = None
        
    def fetch_data(self):
        """Download historical price data"""
        print(f"Fetching data for {self.symbol}...")
        self.data = yf.download(self.symbol, start=self.start_date, 
                                end=self.end_date, progress=False)
        return self.data
    
    def calculate_momentum(self):
        """Calculate momentum indicator (Rate of Change)"""
        # ROC = (Current Price - Price n periods ago) / Price n periods ago * 100
        self.data['ROC'] = ((self.data['Close'] - self.data['Close'].shift(self.momentum_period)) 
                            / self.data['Close'].shift(self.momentum_period) * 100)
        
        # Also calculate moving average of ROC for smoother signals
        self.data['ROC_MA'] = self.data['ROC'].rolling(window=5).mean()
        
    def generate_signals(self):
        """Generate buy/sell signals based on momentum"""
        self.data['Signal'] = 0
        
        # Buy signal: positive momentum
        # Sell signal: negative momentum
        self.data.loc[self.data['ROC'] > 0, 'Signal'] = 1  # Buy
        self.data.loc[self.data['ROC'] < 0, 'Signal'] = -1  # Sell
        
        # Generate positions (1 = long, 0 = no position, -1 = short - if shorting allowed)
        # For simplicity, we'll only go long or hold cash
        self.data['Position'] = 0
        self.data.loc[self.data['Signal'] == 1, 'Position'] = 1
        
    def backtest(self):
        """Run backtest and calculate returns"""
        # Calculate daily returns
        self.data['Market_Return'] = self.data['Close'].pct_change()
        
        # Strategy returns (only earn returns when in position)
        self.data['Strategy_Return'] = self.data['Position'].shift(1) * self.data['Market_Return']
        
        # Calculate cumulative returns
        self.data['Cumulative_Market_Return'] = (1 + self.data['Market_Return']).cumprod()
        self.data['Cumulative_Strategy_Return'] = (1 + self.data['Strategy_Return']).cumprod()
        
        # Calculate portfolio value
        self.data['Portfolio_Value'] = self.initial_capital * self.data['Cumulative_Strategy_Return']
        
        # Drop NaN values
        self.data = self.data.dropna()
        
    def calculate_metrics(self):
        """Calculate performance metrics"""
        # Total returns
        total_market_return = (self.data['Cumulative_Market_Return'].iloc[-1] - 1) * 100
        total_strategy_return = (self.data['Cumulative_Strategy_Return'].iloc[-1] - 1) * 100
        
        # Annualized returns (assuming 252 trading days)
        n_days = len(self.data)
        annual_market_return = ((self.data['Cumulative_Market_Return'].iloc[-1] ** (252/n_days)) - 1) * 100
        annual_strategy_return = ((self.data['Cumulative_Strategy_Return'].iloc[-1] ** (252/n_days)) - 1) * 100
        
        # Sharpe ratio (assuming risk-free rate = 0 for simplicity)
        sharpe_ratio = (self.data['Strategy_Return'].mean() / self.data['Strategy_Return'].std()) * np.sqrt(252)
        
        # Maximum drawdown
        cumulative = self.data['Cumulative_Strategy_Return']
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        
        self.results = {
            'Total Market Return (%)': round(total_market_return, 2),
            'Total Strategy Return (%)': round(total_strategy_return, 2),
            'Annual Market Return (%)': round(annual_market_return, 2),
            'Annual Strategy Return (%)': round(annual_strategy_return, 2),
            'Sharpe Ratio': round(sharpe_ratio, 2),
            'Max Drawdown (%)': round(max_drawdown, 2),
            'Final Portfolio Value ($)': round(self.data['Portfolio_Value'].iloc[-1], 2)
        }
        
        return self.results
    
    def plot_results(self):
        """Plot backtest results"""
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        # Plot 1: Price and positions
        axes[0].plot(self.data.index, self.data['Close'], label='Price', alpha=0.7)
        buy_signals = self.data[self.data['Position'].diff() == 1]
        sell_signals = self.data[self.data['Position'].diff() == -1]
        axes[0].scatter(buy_signals.index, buy_signals['Close'], 
                       color='green', marker='^', s=100, label='Buy', alpha=0.7)
        axes[0].scatter(sell_signals.index, sell_signals['Close'], 
                       color='red', marker='v', s=100, label='Sell', alpha=0.7)
        axes[0].set_title(f'{self.symbol} - Price and Signals')
        axes[0].set_ylabel('Price ($)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Momentum indicator
        axes[1].plot(self.data.index, self.data['ROC'], label='ROC', alpha=0.7)
        axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1].set_title('Momentum Indicator (ROC)')
        axes[1].set_ylabel('ROC (%)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Cumulative returns
        axes[2].plot(self.data.index, 
                    (self.data['Cumulative_Market_Return'] - 1) * 100, 
                    label='Buy & Hold', alpha=0.7)
        axes[2].plot(self.data.index, 
                    (self.data['Cumulative_Strategy_Return'] - 1) * 100, 
                    label='Momentum Strategy', alpha=0.7)
        axes[2].set_title('Strategy Performance')
        axes[2].set_xlabel('Date')
        axes[2].set_ylabel('Cumulative Return (%)')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def run(self):
        """Execute complete backtest workflow"""
        self.fetch_data()
        self.calculate_momentum()
        self.generate_signals()
        self.backtest()
        metrics = self.calculate_metrics()
        
        print("\n" + "="*50)
        print("BACKTEST RESULTS")
        print("="*50)
        for key, value in metrics.items():
            print(f"{key}: {value}")
        print("="*50 + "\n")
        
        self.plot_results()
        
        return self.data, metrics


# Example usage
if __name__ == "__main__":
    # Set parameters
    SYMBOL = 'SPY'
    START_DATE = '2020-01-01'
    END_DATE = '2024-01-01'
    INITIAL_CAPITAL = 10000
    MOMENTUM_PERIOD = 45  # 20-day momentum
    
    # Create and run strategy
    strategy = MomentumStrategy(
        symbol=SYMBOL,
        start_date=START_DATE,
        end_date=END_DATE,
        initial_capital=INITIAL_CAPITAL,
        momentum_period=MOMENTUM_PERIOD
    )
    
    data, metrics = strategy.run()