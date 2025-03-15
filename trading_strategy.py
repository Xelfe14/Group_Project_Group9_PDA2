import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# store trade information
class TradeAction:
    def __init__(self, date, action, price, shares, cash_value, total_shares, total_cash, portfolio_value):
        self.date = date
        self.action = action
        self.price = price
        self.shares = shares
        self.cash_value = cash_value
        self.total_shares = total_shares
        self.total_cash = total_cash
        self.portfolio_value = portfolio_value

# Base trading strategy
class TradingStrategy:
    def __init__(self, initial_capital=10000.0):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.shares = 0
        self.trade_history = []

    def reset(self):
        self.cash = self.initial_capital
        self.shares = 0
        self.trade_history = []

    def record_trade(self, date, action, price, shares):
        # Calculate the cash value of this trade (negative if BUY, positive if SELL)
        cash_value = shares * price * (-1 if action == 'BUY' else 1)
        self.cash += cash_value

        # Update share count
        if action == 'BUY':
            self.shares += shares
        elif action == 'SELL':
            self.shares -= shares

        # Calculate the new portfolio value
        portfolio_value = self.cash + (self.shares * price)

        # Create and store a TradeAction
        trade = TradeAction(
            date,
            action,
            price,
            shares,
            cash_value,
            self.shares,
            self.cash,
            portfolio_value
        )
        self.trade_history.append(trade)
        return trade

# Buy and hold until profit target is reached
class BuyAndHoldStrategy(TradingStrategy):
    def __init__(self, initial_capital=10000.0, profit_target=0.1):
        super().__init__(initial_capital)
        self.profit_target = profit_target
        self.entry_price = None

    def decide(self, date, price, prediction):
        if prediction == 1:  # Price predicted to rise
            if self.cash >= price and self.entry_price is None:
                shares_to_buy = int(self.cash / price)
                if shares_to_buy > 0:
                    self.entry_price = price
                    return self.record_trade(date, 'BUY', price, shares_to_buy)

        if self.shares > 0 and self.entry_price is not None:
            profit = (price - self.entry_price) / self.entry_price
            if profit >= self.profit_target:
                trade = self.record_trade(date, 'SELL', price, self.shares)
                self.entry_price = None
                return trade

        return self.record_trade(date, 'HOLD', price, 0)

# Buy on "rise" prediction, sell on "fall" prediction
class BuyAndSellStrategy(TradingStrategy):
    def __init__(self, initial_capital=10000.0):
        super().__init__(initial_capital)

    def decide(self, date, price, prediction):
        if prediction == 1:  # Price predicted to rise
            if self.cash >= price:
                shares_to_buy = int(self.cash / price)
                if shares_to_buy > 0:
                    return self.record_trade(date, 'BUY', price, shares_to_buy)
        elif prediction == 0:  # Price predicted to fall
            if self.shares > 0:
                return self.record_trade(date, 'SELL', price, self.shares)

        return self.record_trade(date, 'HOLD', price, 0)

def backtest_strategy(strategy, data, predictions, start_date, end_date=None):
    """
    Backtest a trading strategy using historical data and predictions.
    """
    # Reset the strategy's state
    strategy.reset()

    # Ensure data and predictions have the same length
    if len(data) != len(predictions):
        raise ValueError(f"Data length ({len(data)}) does not match predictions length ({len(predictions)})")

    # Reset the index to avoid any indexing issues
    data = data.reset_index(drop=True)

    # Run the strategy day by day
    for i in range(len(data)):
        row = data.iloc[i]
        strategy.decide(row['Date'], row['Close'], predictions[i])

    # Convert trade history to a DataFrame
    history_df = pd.DataFrame([vars(t) for t in strategy.trade_history])

    # Calculate performance metrics
    initial_value = strategy.initial_capital
    final_value = strategy.trade_history[-1].portfolio_value if strategy.trade_history else initial_value
    total_return = (final_value - initial_value) / initial_value if initial_value else 0

    # Calculate daily returns
    if not history_df.empty:
        history_df['daily_return'] = history_df['portfolio_value'].pct_change()
    else:
        history_df['daily_return'] = 0

    # Compute metrics
    metrics = {
        'Initial Portfolio Value': initial_value,
        'Final Portfolio Value': final_value,
        'Total Return': total_return,
        'Total Return %': total_return * 100,
        'Number of Trades': len(history_df[history_df['action'].isin(['BUY', 'SELL'])]),
        'Average Position Size': history_df[history_df['action'] == 'BUY']['shares'].mean() if not history_df.empty else 0,
        'Max Drawdown %': (history_df['portfolio_value'].min() - initial_value) / initial_value * 100 if not history_df.empty else 0,
        'Sharpe Ratio': (np.sqrt(252) * history_df['daily_return'].mean() / history_df['daily_return'].std()) if not history_df.empty and history_df['daily_return'].std() != 0 else 0
    }

    return history_df, metrics

def plot_backtest_results(history_df: pd.DataFrame, ticker: str):
    '''
    Function to plot the backtest strategy data obtained
    '''
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=('Portfolio Value', 'Trading Activity'),
        row_heights=[0.7, 0.3]
    )

    # Portfolio Value Line
    fig.add_trace(
        go.Scatter(
            x=history_df['date'],
            y=history_df['portfolio_value'],
            name='Portfolio Value',
            line=dict(color='blue')
        ),
        row=1, col=1
    )

    # Buy Points
    buys = history_df[history_df['action'] == 'BUY']
    fig.add_trace(
        go.Scatter(
            x=buys['date'],
            y=buys['portfolio_value'],
            name='Buy',
            mode='markers',
            marker=dict(color='green', size=10, symbol='triangle-up')
        ),
        row=1, col=1
    )

    # Sell Points
    sells = history_df[history_df['action'] == 'SELL']
    fig.add_trace(
        go.Scatter(
            x=sells['date'],
            y=sells['portfolio_value'],
            name='Sell',
            mode='markers',
            marker=dict(color='red', size=10, symbol='triangle-down')
        ),
        row=1, col=1
    )

    # Trading Activity (Cash and Shares)
    fig.add_trace(
        go.Scatter(
            x=history_df['date'],
            y=history_df['total_cash'],
            name='Cash',
            line=dict(color='lightblue')
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=history_df['date'],
            y=history_df['total_shares'],
            name='Shares',
            line=dict(color='lightgreen'),
            yaxis='y3'
        ),
        row=2, col=1
    )

    # Update layout
    fig.update_layout(
        title=f'Backtest Results for {ticker}',
        height=800,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
    fig.update_yaxes(title_text="Cash ($)", row=2, col=1)

    return fig
