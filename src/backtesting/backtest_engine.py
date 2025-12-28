"""
Backtesting engine for strategy evaluation
"""

import numpy as np
import pandas as pd
from datetime import datetime
from src.utils.logger import logger
from src.risk_management.risk_manager import RiskManager


class Trade:
    """Represents a single trade"""
    
    def __init__(self, symbol: str, entry_date: datetime, entry_price: float,
                 quantity: int, side: str):
        self.symbol = symbol
        self.entry_date = entry_date
        self.entry_price = entry_price
        self.quantity = quantity
        self.side = side
        self.exit_date = None
        self.exit_price = None
        self.pnl = None
        self.pnl_pct = None
        self.duration = None
    
    def close(self, exit_date: datetime, exit_price: float):
        """Close the trade"""
        self.exit_date = exit_date
        self.exit_price = exit_price
        
        if self.side == 'buy':
            self.pnl = (exit_price - self.entry_price) * self.quantity
            self.pnl_pct = (exit_price - self.entry_price) / self.entry_price
        else:  # sell
            self.pnl = (self.entry_price - exit_price) * self.quantity
            self.pnl_pct = (self.entry_price - exit_price) / self.entry_price
        
        self.duration = (exit_date - self.entry_date).days
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'symbol': self.symbol,
            'entry_date': self.entry_date,
            'entry_price': self.entry_price,
            'quantity': self.quantity,
            'side': self.side,
            'exit_date': self.exit_date,
            'exit_price': self.exit_price,
            'pnl': self.pnl,
            'pnl_pct': self.pnl_pct,
            'duration': self.duration
        }


class BacktestEngine:
    """Backtesting engine for evaluating trading strategies"""
    
    def __init__(self, initial_capital: float = 100000, commission: float = 0.001,
                 slippage: float = 0.0005):
        """
        Args:
            initial_capital: Starting capital
            commission: Commission per trade as % of trade value
            slippage: Slippage as % of price
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        
        self.capital = initial_capital
        self.portfolio_value = initial_capital
        self.trades = []
        self.active_trades = {}
        self.portfolio_history = []
        self.equity_curve = []
        
        self.risk_manager = RiskManager(initial_capital)
    
    def run_backtest(self, data: pd.DataFrame, signals: pd.Series,
                    symbol: str = "STOCK") -> dict:
        """
        Run backtest on historical data with signals
        
        Args:
            data: OHLCV data
            signals: Trading signals
            symbol: Stock symbol
        
        Returns:
            Backtest results dictionary
        """
        logger.info(f"Starting backtest for {symbol}")
        
        position = None
        
        for i in range(len(data)):
            date = data.index[i]
            close_price = data['close'].iloc[i]
            
            # Get signal
            signal = signals.iloc[i] if i < len(signals) else 0
            
            # Process signal
            if signal == 1 and position is None:  # BUY signal
                position = Trade(
                    symbol=symbol,
                    entry_date=date,
                    entry_price=close_price * (1 + self.slippage),
                    quantity=1,
                    side='buy'
                )
                self.active_trades[symbol] = position
                logger.debug(f"BUY signal at {date}: {close_price}")
            
            elif signal == -1 and position is not None:  # SELL signal
                position.close(date, close_price * (1 - self.slippage))
                self.trades.append(position)
                self.active_trades.pop(symbol, None)
                logger.debug(f"SELL signal at {date}: {close_price}")
            
            # Update portfolio value
            self.portfolio_value = self.calculate_portfolio_value(close_price if position else None)
            self.equity_curve.append(self.portfolio_value)
            self.portfolio_history.append({
                'date': date,
                'portfolio_value': self.portfolio_value,
                'capital': self.capital,
                'position': 'open' if position else 'closed'
            })
        
        # Close any remaining position
        if position is not None:
            position.close(data.index[-1], data['close'].iloc[-1])
            self.trades.append(position)
        
        return self.calculate_metrics()
    
    def calculate_portfolio_value(self, current_price: float = None) -> float:
        """Calculate current portfolio value"""
        value = self.capital
        
        if current_price and self.active_trades:
            for trade in self.active_trades.values():
                if trade.side == 'buy':
                    value += (current_price - trade.entry_price) * trade.quantity
        
        return value
    
    def calculate_metrics(self) -> dict:
        """Calculate performance metrics"""
        if not self.trades:
            logger.warning("No trades to analyze")
            return {}
        
        trades_df = pd.DataFrame([t.to_dict() for t in self.trades])
        equity_series = pd.Series(self.equity_curve)
        
        # Basic metrics
        total_return = (self.portfolio_value - self.initial_capital) / self.initial_capital
        num_trades = len(self.trades)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] < 0])
        win_rate = winning_trades / num_trades if num_trades > 0 else 0
        
        # P&L metrics
        gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
        net_profit = trades_df['pnl'].sum()
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
        
        # Risk metrics
        returns = equity_series.pct_change()
        sharpe_ratio = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        
        max_equity = equity_series.max()
        max_drawdown = ((max_equity - equity_series.min()) / max_equity) if max_equity > 0 else 0
        
        # Calmar ratio
        annual_return = (self.portfolio_value / self.initial_capital) ** (252 / len(self.equity_curve)) - 1
        calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0
        
        metrics = {
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'num_trades': num_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'win_rate_pct': win_rate * 100,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'net_profit': net_profit,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'calmar_ratio': calmar_ratio,
            'final_portfolio_value': self.portfolio_value,
        }
        
        logger.info(f"Backtest Results:")
        logger.info(f"  Total Return: {metrics['total_return_pct']:.2f}%")
        logger.info(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        logger.info(f"  Win Rate: {metrics['win_rate_pct']:.2f}%")
        logger.info(f"  Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
        
        return metrics
    
    def export_trades(self, filepath: str):
        """Export trades to CSV"""
        trades_df = pd.DataFrame([t.to_dict() for t in self.trades])
        trades_df.to_csv(filepath, index=False)
        logger.info(f"Trades exported to {filepath}")
    
    def export_equity_curve(self, filepath: str):
        """Export equity curve to CSV"""
        equity_df = pd.DataFrame(self.portfolio_history)
        equity_df.to_csv(filepath, index=False)
        logger.info(f"Equity curve exported to {filepath}")


class WalkForwardAnalysis:
    """Walk-forward analysis for out-of-sample testing"""
    
    def __init__(self, train_period: int = 252, test_period: int = 63):
        """
        Args:
            train_period: Training period in days
            test_period: Testing period in days
        """
        self.train_period = train_period
        self.test_period = test_period
    
    def run_walk_forward(self, data: pd.DataFrame, strategy_func) -> dict:
        """
        Run walk-forward analysis
        
        Args:
            data: Historical data
            strategy_func: Function that takes data and returns signals
        
        Returns:
            Walk-forward results
        """
        results = {
            'windows': [],
            'in_sample_metrics': [],
            'out_sample_metrics': []
        }
        
        start = 0
        window_num = 0
        
        while start + self.train_period + self.test_period <= len(data):
            window_num += 1
            
            # Split into train and test
            train_end = start + self.train_period
            test_end = train_end + self.test_period
            
            train_data = data.iloc[start:train_end]
            test_data = data.iloc[train_end:test_end]
            
            logger.info(f"Window {window_num}: Train {train_data.index[0]} to {train_data.index[-1]}, "
                       f"Test {test_data.index[0]} to {test_data.index[-1]}")
            
            # Generate signals
            train_signals = strategy_func(train_data)
            test_signals = strategy_func(test_data)
            
            # Backtest
            backtester = BacktestEngine()
            
            # In-sample results
            in_sample_metrics = backtester.run_backtest(train_data, train_signals)
            
            # Out-of-sample results
            backtester = BacktestEngine()
            out_sample_metrics = backtester.run_backtest(test_data, test_signals)
            
            results['windows'].append({
                'train_start': train_data.index[0],
                'train_end': train_data.index[-1],
                'test_start': test_data.index[0],
                'test_end': test_data.index[-1]
            })
            results['in_sample_metrics'].append(in_sample_metrics)
            results['out_sample_metrics'].append(out_sample_metrics)
            
            # Move forward
            start += self.test_period
        
        return results


if __name__ == "__main__":
    # Example usage
    from src.data_layer.data_manager import DataManager
    from src.strategy_layer.quant_strategies import MeanReversionStrategy
    
    dm = DataManager()
    data = dm.fetch_historical_data("AAPL", "2023-01-01", "2024-01-01")
    data = dm.add_technical_indicators(data)
    
    strategy = MeanReversionStrategy()
    signals = strategy.generate_signals(data)
    
    backtester = BacktestEngine()
    metrics = backtester.run_backtest(data, signals, symbol="AAPL")
    print(f"Metrics: {metrics}")
