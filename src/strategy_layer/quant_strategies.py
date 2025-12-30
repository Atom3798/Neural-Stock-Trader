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


class VolumeWeightedStrategy(TradingStrategy):
    """Volume-weighted momentum strategy"""
    
    def __init__(self, window: int = 20, volume_threshold: float = 1.5):
        """
        Args:
            window: Rolling window for price momentum
            volume_threshold: Volume multiplier threshold for confirmation
        """
        self.window = window
        self.volume_threshold = volume_threshold
        
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate volume-weighted signals"""
        # Calculate momentum
        momentum = data['close'].pct_change(self.window)
        
        # Calculate average volume
        avg_volume = data['volume'].rolling(window=self.window).mean()
        volume_ratio = data['volume'] / avg_volume
        
        signals = pd.Series(0, index=data.index)
        
        # Buy signal: positive momentum + high volume
        buy_signal = (momentum > 0) & (volume_ratio > self.volume_threshold)
        signals[buy_signal] = SignalType.BUY.value
        
        # Sell signal: negative momentum + high volume
        sell_signal = (momentum < 0) & (volume_ratio > self.volume_threshold)
        signals[sell_signal] = SignalType.SELL.value
        
        return signals
    
    def get_parameters(self) -> dict:
        return {'window': self.window, 'volume_threshold': self.volume_threshold}
    
    def set_parameters(self, params: dict):
        self.window = params.get('window', self.window)
        self.volume_threshold = params.get('volume_threshold', self.volume_threshold)


class VolatilityAdaptiveStrategy(TradingStrategy):
    """Volatility-adaptive trading strategy that adjusts based on market regime"""
    
    def __init__(self, window: int = 20, volatility_percentile: float = 0.7):
        """
        Args:
            window: Rolling window for volatility calculation
            volatility_percentile: Percentile for volatility regime detection
        """
        self.window = window
        self.volatility_percentile = volatility_percentile
        
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate volatility-adapted signals"""
        # Calculate returns and volatility
        returns = data['close'].pct_change()
        volatility = returns.rolling(window=self.window).std()
        
        # Calculate volatility regime
        volatility_threshold = volatility.quantile(self.volatility_percentile)
        high_volatility = volatility > volatility_threshold
        
        # Calculate RSI for overbought/oversold
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        signals = pd.Series(0, index=data.index)
        
        # In high volatility: stricter RSI thresholds
        if high_volatility.iloc[-1]:
            buy_threshold, sell_threshold = 30, 70
        else:
            buy_threshold, sell_threshold = 35, 65
        
        signals[(rsi < buy_threshold) & (high_volatility)] = SignalType.BUY.value
        signals[(rsi > sell_threshold) & (high_volatility)] = SignalType.SELL.value
        
        return signals
    
    def get_parameters(self) -> dict:
        return {'window': self.window, 'volatility_percentile': self.volatility_percentile}
    
    def set_parameters(self, params: dict):
        self.window = params.get('window', self.window)
        self.volatility_percentile = params.get('volatility_percentile', self.volatility_percentile)


class PairsTradeStrategy(TradingStrategy):
    """Pairs trading strategy using correlation and mean reversion"""
    
    def __init__(self, window: int = 60, zscore_threshold: float = 2.0):
        """
        Args:
            window: Lookback window for correlation
            zscore_threshold: Z-score threshold for trading
        """
        self.window = window
        self.zscore_threshold = zscore_threshold
        
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate pairs trading signals"""
        signals = pd.Series(0, index=data.index)
        
        if len(data) < self.window:
            return signals
        
        # Calculate rolling correlation with a reference index (using highest correlated nearby price)
        returns = data['close'].pct_change()
        
        # Simple implementation: trade based on deviations from SMA
        sma = data['close'].rolling(window=self.window).mean()
        std = data['close'].rolling(window=self.window).std()
        
        zscore = (data['close'] - sma) / (std + 1e-10)
        
        # Buy when price is below mean (mean reversion)
        signals[zscore < -self.zscore_threshold] = SignalType.BUY.value
        
        # Sell when price is above mean
        signals[zscore > self.zscore_threshold] = SignalType.SELL.value
        
        return signals
    
    def get_parameters(self) -> dict:
        return {'window': self.window, 'zscore_threshold': self.zscore_threshold}
    
    def set_parameters(self, params: dict):
        self.window = params.get('window', self.window)
        self.zscore_threshold = params.get('zscore_threshold', self.zscore_threshold)


class MultiTimeframeStrategy(TradingStrategy):
    """Multi-timeframe strategy combining short and long-term trends"""
    
    def __init__(self, short_window: int = 10, long_window: int = 50):
        """
        Args:
            short_window: Short-term moving average window
            long_window: Long-term moving average window
        """
        self.short_window = short_window
        self.long_window = long_window
        
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate multi-timeframe signals"""
        # Short-term trend
        sma_short = data['close'].rolling(window=self.short_window).mean()
        
        # Long-term trend
        sma_long = data['close'].rolling(window=self.long_window).mean()
        
        # Medium-term momentum
        momentum = data['close'].pct_change(20)
        
        signals = pd.Series(0, index=data.index)
        
        # Buy: price above long-term trend, short > long, positive momentum
        buy_condition = (data['close'] > sma_long) & (sma_short > sma_long) & (momentum > 0.02)
        signals[buy_condition] = SignalType.BUY.value
        
        # Sell: price below long-term trend, short < long, negative momentum
        sell_condition = (data['close'] < sma_long) & (sma_short < sma_long) & (momentum < -0.02)
        signals[sell_condition] = SignalType.SELL.value
        
        return signals
    
    def get_parameters(self) -> dict:
        return {'short_window': self.short_window, 'long_window': self.long_window}
    
    def set_parameters(self, params: dict):
        self.short_window = params.get('short_window', self.short_window)
        self.long_window = params.get('long_window', self.long_window)


class MACDDivergenceStrategy(TradingStrategy):
    """MACD with divergence detection strategy"""
    
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        """
        Args:
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line EMA period
        """
        self.fast = fast
        self.slow = slow
        self.signal = signal
        
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate MACD divergence signals"""
        # Calculate MACD
        ema_fast = data['close'].ewm(span=self.fast).mean()
        ema_slow = data['close'].ewm(span=self.slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=self.signal).mean()
        histogram = macd - signal_line
        
        signals = pd.Series(0, index=data.index)
        
        # Buy signal: MACD crosses above signal line
        buy_signal = (histogram > 0) & (histogram.shift(1) <= 0)
        signals[buy_signal] = SignalType.BUY.value
        
        # Sell signal: MACD crosses below signal line
        sell_signal = (histogram < 0) & (histogram.shift(1) >= 0)
        signals[sell_signal] = SignalType.SELL.value
        
        return signals
    
    def get_parameters(self) -> dict:
        return {'fast': self.fast, 'slow': self.slow, 'signal': self.signal}
    
    def set_parameters(self, params: dict):
        self.fast = params.get('fast', self.fast)
        self.slow = params.get('slow', self.slow)
        self.signal = params.get('signal', self.signal)


class RSIWithConfirmationStrategy(TradingStrategy):
    """RSI strategy with momentum confirmation"""
    
    def __init__(self, rsi_period: int = 14, overbought: float = 70, oversold: float = 30):
        """
        Args:
            rsi_period: Period for RSI calculation
            overbought: RSI overbought threshold
            oversold: RSI oversold threshold
        """
        self.rsi_period = rsi_period
        self.overbought = overbought
        self.oversold = oversold
        
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate RSI signals with momentum confirmation"""
        # Calculate RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        # Momentum confirmation
        momentum = data['close'].pct_change(5)
        
        signals = pd.Series(0, index=data.index)
        
        # Buy: RSI oversold + positive momentum
        buy_condition = (rsi < self.oversold) & (momentum > 0)
        signals[buy_condition] = SignalType.BUY.value
        
        # Sell: RSI overbought + negative momentum
        sell_condition = (rsi > self.overbought) & (momentum < 0)
        signals[sell_condition] = SignalType.SELL.value
        
        return signals
    
    def get_parameters(self) -> dict:
        return {'rsi_period': self.rsi_period, 'overbought': self.overbought, 'oversold': self.oversold}
    
    def set_parameters(self, params: dict):
        self.rsi_period = params.get('rsi_period', self.rsi_period)
        self.overbought = params.get('overbought', self.overbought)
        self.oversold = params.get('oversold', self.oversold)


class BollingerBandStrategy(TradingStrategy):
    """Bollinger Bands squeeze and breakout strategy"""
    
    def __init__(self, window: int = 20, num_std: float = 2.0, squeeze_threshold: float = 0.3):
        """
        Args:
            window: Moving average window
            num_std: Number of standard deviations
            squeeze_threshold: Threshold for band squeeze detection
        """
        self.window = window
        self.num_std = num_std
        self.squeeze_threshold = squeeze_threshold
        
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate Bollinger Bands signals"""
        sma = data['close'].rolling(window=self.window).mean()
        std = data['close'].rolling(window=self.window).std()
        
        upper_band = sma + (std * self.num_std)
        lower_band = sma - (std * self.num_std)
        
        # Calculate band width
        band_width = (upper_band - lower_band) / sma
        band_width_sma = band_width.rolling(window=20).mean()
        
        signals = pd.Series(0, index=data.index)
        
        # Buy: Price breaks above upper band during squeeze
        buy_condition = (data['close'] > upper_band) & (band_width < band_width_sma * (1 + self.squeeze_threshold))
        signals[buy_condition] = SignalType.BUY.value
        
        # Sell: Price breaks below lower band during squeeze
        sell_condition = (data['close'] < lower_band) & (band_width < band_width_sma * (1 + self.squeeze_threshold))
        signals[sell_condition] = SignalType.SELL.value
        
        return signals
    
    def get_parameters(self) -> dict:
        return {'window': self.window, 'num_std': self.num_std, 'squeeze_threshold': self.squeeze_threshold}
    
    def set_parameters(self, params: dict):
        self.window = params.get('window', self.window)
        self.num_std = params.get('num_std', self.num_std)
        self.squeeze_threshold = params.get('squeeze_threshold', self.squeeze_threshold)


class TrendFollowingStrategy(TradingStrategy):
    """Advanced trend following with ADX confirmation"""
    
    def __init__(self, trend_window: int = 20, adx_threshold: float = 25):
        """
        Args:
            trend_window: Window for trend detection
            adx_threshold: Minimum ADX for strong trend
        """
        self.trend_window = trend_window
        self.adx_threshold = adx_threshold
        
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trend-following signals"""
        # Calculate ADX (simplified)
        high_low = data['high'] - data['low']
        high_close = abs(data['high'] - data['close'].shift(1))
        low_close = abs(data['low'] - data['close'].shift(1))
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=14).mean()
        
        # Directional indicators (simplified)
        up_move = data['high'].diff()
        down_move = -data['low'].diff()
        
        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)
        
        plus_di = 100 * (plus_dm.rolling(window=14).mean() / (atr + 1e-10))
        minus_di = 100 * (minus_dm.rolling(window=14).mean() / (atr + 1e-10))
        
        di_diff = abs(plus_di - minus_di)
        adx = di_diff.rolling(window=14).mean()
        
        signals = pd.Series(0, index=data.index)
        
        # Buy: Uptrend with strong ADX
        buy_condition = (plus_di > minus_di) & (adx > self.adx_threshold)
        signals[buy_condition] = SignalType.BUY.value
        
        # Sell: Downtrend with strong ADX
        sell_condition = (minus_di > plus_di) & (adx > self.adx_threshold)
        signals[sell_condition] = SignalType.SELL.value
        
        return signals
    
    def get_parameters(self) -> dict:
        return {'trend_window': self.trend_window, 'adx_threshold': self.adx_threshold}
    
    def set_parameters(self, params: dict):
        self.trend_window = params.get('trend_window', self.trend_window)
        self.adx_threshold = params.get('adx_threshold', self.adx_threshold)


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
    # Example usage demonstrating all strategies
    from src.data_layer.data_manager import DataManager
    
    dm = DataManager()
    data = dm.fetch_historical_data("AAPL", "2023-01-01", "2024-01-01")
    data = dm.add_technical_indicators(data)
    
    # Create comprehensive strategy ensemble with all algorithms
    strategies = [
        MeanReversionStrategy(window=20, threshold=2.0),
        MomentumStrategy(fast_period=12, slow_period=26),
        StatisticalArbitrageStrategy(lookback=60),
        VolumeWeightedStrategy(window=20, volume_threshold=1.5),
        VolatilityAdaptiveStrategy(window=20, volatility_percentile=0.7),
        PairsTradeStrategy(window=60, zscore_threshold=2.0),
        MultiTimeframeStrategy(short_window=10, long_window=50),
        MACDDivergenceStrategy(fast=12, slow=26, signal=9),
        RSIWithConfirmationStrategy(rsi_period=14, overbought=70, oversold=30),
        BollingerBandStrategy(window=20, num_std=2.0, squeeze_threshold=0.3),
        TrendFollowingStrategy(trend_window=20, adx_threshold=25)
    ]
    
    # Equal weights for all strategies
    ensemble = StrategyEnsemble(strategies)
    
    print("\n" + "="*80)
    print("TRADING SIGNAL ANALYSIS - 11 ADVANCED STRATEGIES")
    print("="*80)
    
    # Get ensemble signals
    signals = ensemble.generate_signals(data)
    print(f"\nEnsemble Signal Distribution:")
    print(signals.value_counts())
    
    # Get individual strategy signals
    strategy_signals = ensemble.get_strategy_signals(data)
    print(f"\nIndividual Strategy Performance:")
    for strategy_name, signals in strategy_signals.items():
        buy_count = (signals == 1).sum()
        sell_count = (signals == -1).sum()
        print(f"  {strategy_name}: {buy_count} BUY signals, {sell_count} SELL signals")
