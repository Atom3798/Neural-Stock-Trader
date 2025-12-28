"""
Quantitative trading algorithms and strategies
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from src.utils.logger import logger
from src.utils.constants import SignalType


class TradingStrategy(ABC):
    """Abstract base class for trading strategies"""
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals"""
        pass
    
    @abstractmethod
    def get_parameters(self) -> dict:
        """Get strategy parameters"""
        pass
    
    @abstractmethod
    def set_parameters(self, params: dict):
        """Set strategy parameters"""
        pass


class MeanReversionStrategy(TradingStrategy):
    """Mean reversion trading strategy"""
    
    def __init__(self, window: int = 20, threshold: float = 2.0):
        """
        Args:
            window: Rolling window for mean and std calculation
            threshold: Number of standard deviations for signal generation
        """
        self.window = window
        self.threshold = threshold
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate mean reversion signals"""
        # Calculate rolling mean and std
        rolling_mean = data['close'].rolling(window=self.window).mean()
        rolling_std = data['close'].rolling(window=self.window).std()
        
        signals = pd.Series(0, index=data.index)
        
        # Signal generation
        upper_band = rolling_mean + (self.threshold * rolling_std)
        lower_band = rolling_mean - (self.threshold * rolling_std)
        
        # Buy when price is below lower band (mean reversion up)
        signals[data['close'] < lower_band] = SignalType.BUY.value
        
        # Sell when price is above upper band (mean reversion down)
        signals[data['close'] > upper_band] = SignalType.SELL.value
        
        return signals
    
    def get_parameters(self) -> dict:
        return {'window': self.window, 'threshold': self.threshold}
    
    def set_parameters(self, params: dict):
        if 'window' in params:
            self.window = params['window']
        if 'threshold' in params:
            self.threshold = params['threshold']


class MomentumStrategy(TradingStrategy):
    """Momentum and trend-following strategy"""
    
    def __init__(self, fast_period: int = 12, slow_period: int = 26, threshold: float = 0.02):
        """
        Args:
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            threshold: Signal strength threshold
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.threshold = threshold
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate momentum signals"""
        # Calculate EMAs
        ema_fast = data['close'].ewm(span=self.fast_period).mean()
        ema_slow = data['close'].ewm(span=self.slow_period).mean()
        
        signals = pd.Series(0, index=data.index)
        
        # Calculate momentum
        momentum = (ema_fast - ema_slow) / ema_slow
        
        # Strong signals
        signals[momentum > self.threshold] = SignalType.STRONG_BUY.value
        signals[momentum < -self.threshold] = SignalType.STRONG_SELL.value
        
        # Weak signals
        signals[(momentum > 0) & (momentum <= self.threshold)] = SignalType.BUY.value
        signals[(momentum < 0) & (momentum >= -self.threshold)] = SignalType.SELL.value
        
        return signals
    
    def get_parameters(self) -> dict:
        return {
            'fast_period': self.fast_period,
            'slow_period': self.slow_period,
            'threshold': self.threshold
        }
    
    def set_parameters(self, params: dict):
        if 'fast_period' in params:
            self.fast_period = params['fast_period']
        if 'slow_period' in params:
            self.slow_period = params['slow_period']
        if 'threshold' in params:
            self.threshold = params['threshold']


class StatisticalArbitrageStrategy(TradingStrategy):
    """Statistical arbitrage using pairs trading"""
    
    def __init__(self, lookback: int = 60, entry_threshold: float = 2.0,
                 exit_threshold: float = 0.5):
        """
        Args:
            lookback: Historical window for correlation calculation
            entry_threshold: Z-score threshold for entry
            exit_threshold: Z-score threshold for exit
        """
        self.lookback = lookback
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate statistical arbitrage signals"""
        signals = pd.Series(0, index=data.index)
        
        # Calculate rolling mean and std
        rolling_mean = data['close'].rolling(window=self.lookback).mean()
        rolling_std = data['close'].rolling(window=self.lookback).std()
        
        # Z-score
        z_score = (data['close'] - rolling_mean) / rolling_std
        
        # Entry signals
        signals[z_score < -self.entry_threshold] = SignalType.STRONG_BUY.value
        signals[z_score > self.entry_threshold] = SignalType.STRONG_SELL.value
        
        # Exit signals
        signals[(z_score >= -self.exit_threshold) & (z_score < 0)] = SignalType.BUY.value
        signals[(z_score <= self.exit_threshold) & (z_score > 0)] = SignalType.SELL.value
        
        return signals
    
    def get_parameters(self) -> dict:
        return {
            'lookback': self.lookback,
            'entry_threshold': self.entry_threshold,
            'exit_threshold': self.exit_threshold
        }
    
    def set_parameters(self, params: dict):
        if 'lookback' in params:
            self.lookback = params['lookback']
        if 'entry_threshold' in params:
            self.entry_threshold = params['entry_threshold']
        if 'exit_threshold' in params:
            self.exit_threshold = params['exit_threshold']


class MarketMakingStrategy(TradingStrategy):
    """Market making strategy with optimal bid-ask spreads"""
    
    def __init__(self, spread_percentage: float = 0.001, inventory_limit: int = 100):
        """
        Args:
            spread_percentage: Bid-ask spread as percentage of price
            inventory_limit: Maximum inventory position
        """
        self.spread_percentage = spread_percentage
        self.inventory_limit = inventory_limit
        self.current_inventory = 0
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate market making signals"""
        signals = pd.Series(0, index=data.index)
        
        # Calculate mid price
        mid_price = (data['high'] + data['low']) / 2
        spread = mid_price * self.spread_percentage
        
        # Place buy orders below mid price
        signals[self.current_inventory < self.inventory_limit] = SignalType.BUY.value
        
        # Place sell orders above mid price
        signals[self.current_inventory > -self.inventory_limit] = SignalType.SELL.value
        
        return signals
    
    def update_inventory(self, trade_side: str, quantity: int):
        """Update inventory after trade execution"""
        if trade_side == 'buy':
            self.current_inventory += quantity
        elif trade_side == 'sell':
            self.current_inventory -= quantity
    
    def get_parameters(self) -> dict:
        return {
            'spread_percentage': self.spread_percentage,
            'inventory_limit': self.inventory_limit,
            'current_inventory': self.current_inventory
        }
    
    def set_parameters(self, params: dict):
        if 'spread_percentage' in params:
            self.spread_percentage = params['spread_percentage']
        if 'inventory_limit' in params:
            self.inventory_limit = params['inventory_limit']


class PortfolioOptimizationStrategy(TradingStrategy):
    """Portfolio optimization using Markowitz and modern portfolio theory"""
    
    def __init__(self, lookback: int = 252, target_volatility: float = 0.15):
        """
        Args:
            lookback: Historical period for covariance estimation
            target_volatility: Target portfolio volatility
        """
        self.lookback = lookback
        self.target_volatility = target_volatility
    
    def calculate_efficient_frontier(self, returns: pd.DataFrame) -> dict:
        """Calculate efficient frontier"""
        # Calculate mean returns and covariance
        mean_returns = returns.mean()
        cov_matrix = returns.cov()
        
        # Number of assets
        n_assets = len(mean_returns)
        
        # Optimal weights using inverse volatility weighting
        inv_volatilities = 1.0 / np.sqrt(np.diag(cov_matrix))
        weights = inv_volatilities / inv_volatilities.sum()
        
        # Portfolio metrics
        portfolio_return = np.dot(weights, mean_returns)
        portfolio_volatility = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
        
        return {
            'weights': weights,
            'return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe_ratio': portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
        }
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate portfolio optimization signals"""
        signals = pd.Series(0, index=data.index)
        
        # Simplified: Use relative strength to allocate weights
        if 'rsi' in data.columns:
            signals[data['rsi'] < 30] = SignalType.STRONG_BUY.value
            signals[data['rsi'] > 70] = SignalType.STRONG_SELL.value
        
        return signals
    
    def get_parameters(self) -> dict:
        return {
            'lookback': self.lookback,
            'target_volatility': self.target_volatility
        }
    
    def set_parameters(self, params: dict):
        if 'lookback' in params:
            self.lookback = params['lookback']
        if 'target_volatility' in params:
            self.target_volatility = params['target_volatility']


class StrategyEnsemble:
    """Combine multiple strategies with voting mechanism"""
    
    def __init__(self, strategies: list, weights: list = None):
        """
        Args:
            strategies: List of trading strategies
            weights: Weights for each strategy (default: equal)
        """
        self.strategies = strategies
        
        if weights is None:
            weights = [1.0 / len(strategies)] * len(strategies)
        self.weights = np.array(weights) / np.sum(weights)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate ensemble signals"""
        all_signals = []
        
        for strategy in self.strategies:
            signals = strategy.generate_signals(data)
            all_signals.append(signals)
        
        # Weighted average of signals
        ensemble_signals = pd.Series(0.0, index=data.index)
        for signal, weight in zip(all_signals, self.weights):
            ensemble_signals += signal * weight
        
        # Threshold the ensemble signal
        final_signals = pd.Series(0, index=data.index)
        final_signals[ensemble_signals > 0.5] = SignalType.BUY.value
        final_signals[ensemble_signals < -0.5] = SignalType.SELL.value
        
        return final_signals
    
    def get_strategy_signals(self, data: pd.DataFrame) -> dict:
        """Get individual strategy signals for analysis"""
        signals = {}
        for i, strategy in enumerate(self.strategies):
            signals[f'strategy_{i}'] = strategy.generate_signals(data)
        return signals


if __name__ == "__main__":
    # Example usage
    from src.data_layer.data_manager import DataManager
    
    dm = DataManager()
    data = dm.fetch_historical_data("AAPL", "2023-01-01", "2024-01-01")
    data = dm.add_technical_indicators(data)
    
    strategies = [
        MeanReversionStrategy(window=20, threshold=2.0),
        MomentumStrategy(fast_period=12, slow_period=26),
        StatisticalArbitrageStrategy(lookback=60)
    ]
    
    ensemble = StrategyEnsemble(strategies)
    signals = ensemble.generate_signals(data)
    print(f"Signal distribution:\n{signals.value_counts()}")
