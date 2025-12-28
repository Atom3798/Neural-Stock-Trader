"""
Risk management and position sizing
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from src.utils.logger import logger
from src.utils.constants import PositionSizing


class RiskManager:
    """Comprehensive risk management system"""
    
    def __init__(self, initial_capital: float, max_position_size: float = 0.1,
                 max_daily_loss: float = 0.05, max_drawdown: float = 0.20):
        """
        Args:
            initial_capital: Starting portfolio value
            max_position_size: Max position size as % of portfolio
            max_daily_loss: Max daily loss as % of portfolio
            max_drawdown: Max portfolio drawdown
        """
        self.initial_capital = initial_capital
        self.max_position_size = max_position_size
        self.max_daily_loss = max_daily_loss
        self.max_drawdown = max_drawdown
        
        self.current_capital = initial_capital
        self.peak_capital = initial_capital
        self.daily_pnl = 0
        self.positions = {}
    
    def calculate_max_position_size(self, current_price: float, stop_loss_distance: float) -> float:
        """Calculate maximum position size based on risk"""
        risk_per_unit = stop_loss_distance
        max_risk_amount = self.current_capital * self.max_position_size
        position_size = max_risk_amount / risk_per_unit if risk_loss_distance > 0 else 0
        return int(position_size)
    
    def check_position_limits(self, symbol: str, quantity: float, side: str) -> bool:
        """Check if position respects limits"""
        # Get current position
        current_position = self.positions.get(symbol, 0)
        new_position = current_position + quantity if side == 'buy' else current_position - quantity
        
        # Check max position size
        position_pct = abs(new_position) / self.current_capital
        if position_pct > self.max_position_size:
            logger.warning(f"Position {symbol} would exceed max size: {position_pct:.2%}")
            return False
        
        return True
    
    def check_daily_loss_limit(self) -> bool:
        """Check if daily loss limit has been exceeded"""
        if self.daily_pnl < -(self.current_capital * self.max_daily_loss):
            logger.warning(f"Daily loss limit exceeded: {self.daily_pnl:.2f}")
            return False
        return True
    
    def check_drawdown_limit(self) -> bool:
        """Check if maximum drawdown has been exceeded"""
        drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
        if drawdown > self.max_drawdown:
            logger.warning(f"Drawdown limit exceeded: {drawdown:.2%}")
            return False
        return True
    
    def update_capital(self, pnl: float, is_daily_reset: bool = False):
        """Update capital after trade"""
        self.current_capital += pnl
        self.daily_pnl += pnl
        
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital
        
        if is_daily_reset:
            self.daily_pnl = 0
    
    def get_risk_metrics(self) -> dict:
        """Get current risk metrics"""
        drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
        
        return {
            'current_capital': self.current_capital,
            'peak_capital': self.peak_capital,
            'daily_pnl': self.daily_pnl,
            'drawdown': drawdown,
            'drawdown_pct': drawdown * 100,
            'daily_loss_limit_remaining': (self.current_capital * self.max_daily_loss) - abs(self.daily_pnl),
        }


class PositionSizer(ABC):
    """Abstract base class for position sizing"""
    
    @abstractmethod
    def calculate_size(self, capital: float, volatility: float, **kwargs) -> float:
        """Calculate position size"""
        pass


class FixedPositionSizer(PositionSizer):
    """Fixed position size"""
    
    def __init__(self, fixed_size: float):
        self.fixed_size = fixed_size
    
    def calculate_size(self, capital: float, volatility: float, **kwargs) -> float:
        return self.fixed_size


class ProportionalPositionSizer(PositionSizer):
    """Position size proportional to portfolio"""
    
    def __init__(self, position_pct: float = 0.05):
        self.position_pct = position_pct
    
    def calculate_size(self, capital: float, volatility: float, **kwargs) -> float:
        return capital * self.position_pct


class KellyCriterionSizer(PositionSizer):
    """Position sizing using Kelly Criterion"""
    
    def __init__(self, kelly_fraction: float = 0.25):
        """
        Args:
            kelly_fraction: Fraction of full Kelly to use (usually 0.25)
        """
        self.kelly_fraction = kelly_fraction
    
    def calculate_size(self, capital: float, win_rate: float, avg_win: float,
                      avg_loss: float, **kwargs) -> float:
        """
        Calculate position size using Kelly Criterion
        
        Kelly Formula: f* = (bp - q) / b
        where:
            b = odds (avg_win / avg_loss)
            p = probability of win (win_rate)
            q = probability of loss (1 - win_rate)
        """
        if avg_loss == 0:
            return 0
        
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - win_rate
        
        # Full Kelly formula
        kelly_fraction = (b * p - q) / b
        kelly_fraction = max(0, min(kelly_fraction, 1))  # Clamp between 0 and 1
        
        # Use fractional Kelly (usually 25%)
        position_size = capital * kelly_fraction * self.kelly_fraction
        
        return position_size


class RiskParitySizer(PositionSizer):
    """Risk parity position sizing"""
    
    def __init__(self, target_risk: float = 0.02):
        """
        Args:
            target_risk: Target risk per position as % of capital
        """
        self.target_risk = target_risk
    
    def calculate_size(self, capital: float, volatility: float, **kwargs) -> float:
        """
        Calculate position size to achieve target risk
        
        Position Size = (Capital * Target Risk) / Volatility
        """
        if volatility == 0:
            return 0
        
        risk_amount = capital * self.target_risk
        position_size = risk_amount / volatility
        
        return position_size


class StopLoss:
    """Stop loss management"""
    
    def __init__(self, stop_type: str = "hard", stop_pct: float = 0.02):
        """
        Args:
            stop_type: 'hard' (fixed), 'trailing', or 'dynamic'
            stop_pct: Stop loss percentage
        """
        self.stop_type = stop_type
        self.stop_pct = stop_pct
        self.entry_price = None
        self.peak_price = None
        self.stop_price = None
    
    def set_entry(self, entry_price: float):
        """Set entry price"""
        self.entry_price = entry_price
        self.peak_price = entry_price
        self.stop_price = entry_price * (1 - self.stop_pct)
    
    def update(self, current_price: float):
        """Update stop loss based on type"""
        if self.stop_type == "trailing":
            # Move stop price up with price but never down
            if current_price > self.peak_price:
                self.peak_price = current_price
                self.stop_price = current_price * (1 - self.stop_pct)
        elif self.stop_type == "dynamic":
            # Dynamic stop based on volatility (would need volatility input)
            pass
    
    def is_triggered(self, current_price: float) -> bool:
        """Check if stop loss is triggered"""
        return current_price <= self.stop_price if self.stop_price else False
    
    def get_stop_price(self) -> float:
        """Get current stop price"""
        return self.stop_price


class TakeProfit:
    """Take profit management"""
    
    def __init__(self, take_profit_pct: float = 0.05):
        """
        Args:
            take_profit_pct: Take profit percentage
        """
        self.take_profit_pct = take_profit_pct
        self.entry_price = None
        self.target_price = None
    
    def set_entry(self, entry_price: float):
        """Set entry price and calculate target"""
        self.entry_price = entry_price
        self.target_price = entry_price * (1 + self.take_profit_pct)
    
    def is_triggered(self, current_price: float) -> bool:
        """Check if take profit is triggered"""
        return current_price >= self.target_price if self.target_price else False
    
    def get_target_price(self) -> float:
        """Get target price"""
        return self.target_price


class CorrelationAnalyzer:
    """Analyze correlation between positions"""
    
    def __init__(self, max_correlation: float = 0.7):
        """
        Args:
            max_correlation: Maximum allowed correlation between positions
        """
        self.max_correlation = max_correlation
    
    def check_position_correlation(self, returns_df: pd.DataFrame) -> dict:
        """Check correlation matrix for positions"""
        corr_matrix = returns_df.corr()
        
        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > self.max_correlation:
                    high_corr_pairs.append({
                        'asset1': corr_matrix.columns[i],
                        'asset2': corr_matrix.columns[j],
                        'correlation': corr_matrix.iloc[i, j]
                    })
        
        return {
            'correlation_matrix': corr_matrix,
            'high_correlation_pairs': high_corr_pairs,
            'portfolio_correlation': corr_matrix.values.mean()
        }


class VaR:
    """Value at Risk calculation"""
    
    @staticmethod
    def calculate_var(returns: pd.Series, confidence_level: float = 0.95) -> float:
        """
        Calculate Value at Risk
        
        Args:
            returns: Series of returns
            confidence_level: Confidence level (0.95 for 95%)
        
        Returns:
            VaR value
        """
        return returns.quantile(1 - confidence_level)
    
    @staticmethod
    def calculate_cvar(returns: pd.Series, confidence_level: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (Expected Shortfall)
        
        Args:
            returns: Series of returns
            confidence_level: Confidence level
        
        Returns:
            CVaR value
        """
        var = returns.quantile(1 - confidence_level)
        return returns[returns <= var].mean()


class CircuitBreaker:
    """Market circuit breaker for abnormal conditions"""
    
    def __init__(self, daily_loss_trigger: float = 0.03,
                 intraday_loss_trigger: float = 0.02):
        """
        Args:
            daily_loss_trigger: Daily loss trigger threshold
            intraday_loss_trigger: Intraday loss trigger threshold
        """
        self.daily_loss_trigger = daily_loss_trigger
        self.intraday_loss_trigger = intraday_loss_trigger
        self.daily_loss = 0
        self.intraday_loss = 0
        self.circuit_open = False
    
    def update_losses(self, pnl: float, is_daily_reset: bool = False):
        """Update losses"""
        self.intraday_loss += pnl
        self.daily_loss += pnl
        
        if is_daily_reset:
            self.daily_loss = 0
            self.intraday_loss = 0
            self.circuit_open = False
    
    def check_circuit(self, starting_capital: float) -> bool:
        """Check if circuit should be open (halt trading)"""
        if abs(self.daily_loss) / starting_capital > self.daily_loss_trigger:
            logger.warning("Daily loss circuit breaker triggered")
            self.circuit_open = True
            return True
        
        if abs(self.intraday_loss) / starting_capital > self.intraday_loss_trigger:
            logger.warning("Intraday loss circuit breaker triggered")
            self.circuit_open = True
            return True
        
        return False
    
    def is_open(self) -> bool:
        """Check if circuit is open"""
        return self.circuit_open


if __name__ == "__main__":
    # Example usage
    rm = RiskManager(initial_capital=100000)
    print(f"Risk metrics: {rm.get_risk_metrics()}")
    
    # Test position sizing
    kelly_sizer = KellyCriterionSizer(kelly_fraction=0.25)
    size = kelly_sizer.calculate_size(capital=100000, win_rate=0.55, avg_win=100, avg_loss=90)
    print(f"Kelly position size: {size}")
