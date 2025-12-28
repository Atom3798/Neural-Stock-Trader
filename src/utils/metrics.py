"""
Metrics calculation and reporting utilities
"""

import numpy as np
import pandas as pd
from datetime import datetime
from src.utils.logger import logger


class MetricsCalculator:
    """Calculate trading and risk metrics"""
    
    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02,
                              periods_per_year: int = 252) -> float:
        """
        Calculate Sharpe Ratio
        
        Sharpe = (Return - Risk-Free Rate) / Volatility
        """
        if returns.std() == 0:
            return 0
        
        excess_return = returns.mean() - (risk_free_rate / periods_per_year)
        annual_volatility = returns.std() * np.sqrt(periods_per_year)
        
        return excess_return / annual_volatility if annual_volatility > 0 else 0
    
    @staticmethod
    def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02,
                               periods_per_year: int = 252) -> float:
        """
        Calculate Sortino Ratio (penalizes only downside volatility)
        """
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            return float('inf')
        
        downside_volatility = downside_returns.std() * np.sqrt(periods_per_year)
        excess_return = returns.mean() * periods_per_year - risk_free_rate
        
        return excess_return / downside_volatility if downside_volatility > 0 else 0
    
    @staticmethod
    def calculate_calmar_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
        """
        Calculate Calmar Ratio = Annual Return / Max Drawdown
        """
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        annual_return = returns.mean() * periods_per_year
        
        return annual_return / abs(max_drawdown) if max_drawdown < 0 else 0
    
    @staticmethod
    def calculate_max_drawdown(prices: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + prices.pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    @staticmethod
    def calculate_recovery_time(prices: pd.Series) -> int:
        """Calculate time to recover from maximum drawdown"""
        cumulative = (1 + prices.pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        
        max_dd_idx = ((cumulative - running_max) / running_max).idxmin()
        recovery_idx = None
        
        for idx in prices.index[prices.index.get_loc(max_dd_idx):]:
            if cumulative[idx] >= running_max[max_dd_idx]:
                recovery_idx = idx
                break
        
        if recovery_idx is None:
            return -1  # Not recovered
        
        return (recovery_idx - max_dd_idx).days
    
    @staticmethod
    def calculate_information_ratio(returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """
        Calculate Information Ratio
        IR = (Portfolio Return - Benchmark Return) / Tracking Error
        """
        excess_returns = returns - benchmark_returns
        tracking_error = excess_returns.std()
        
        return excess_returns.mean() / tracking_error if tracking_error > 0 else 0
    
    @staticmethod
    def calculate_omega_ratio(returns: pd.Series, threshold: float = 0.0) -> float:
        """
        Calculate Omega Ratio
        Probability-weighted ratio of gains vs losses
        """
        excess_returns = returns - threshold
        
        gains = excess_returns[excess_returns > 0].sum()
        losses = abs(excess_returns[excess_returns < 0].sum())
        
        return gains / losses if losses > 0 else float('inf')
    
    @staticmethod
    def calculate_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
        """Calculate annualized volatility"""
        return returns.std() * np.sqrt(periods_per_year)
    
    @staticmethod
    def calculate_var(returns: pd.Series, confidence_level: float = 0.95) -> float:
        """Calculate Value at Risk"""
        return returns.quantile(1 - confidence_level)
    
    @staticmethod
    def calculate_cvar(returns: pd.Series, confidence_level: float = 0.95) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        var = returns.quantile(1 - confidence_level)
        return returns[returns <= var].mean()


class PerformanceReporter:
    """Generate performance reports"""
    
    def __init__(self, trades_df: pd.DataFrame, equity_curve: pd.Series,
                 benchmark: pd.Series = None):
        self.trades_df = trades_df
        self.equity_curve = equity_curve
        self.benchmark = benchmark
    
    def generate_report(self) -> dict:
        """Generate comprehensive performance report"""
        
        returns = self.equity_curve.pct_change()
        
        report = {
            'summary': self._calculate_summary_metrics(),
            'risk_metrics': self._calculate_risk_metrics(returns),
            'trade_metrics': self._calculate_trade_metrics(),
            'performance': self._calculate_performance_metrics(returns),
        }
        
        return report
    
    def _calculate_summary_metrics(self) -> dict:
        """Calculate summary metrics"""
        return {
            'total_trades': len(self.trades_df),
            'winning_trades': len(self.trades_df[self.trades_df['pnl'] > 0]),
            'losing_trades': len(self.trades_df[self.trades_df['pnl'] < 0]),
            'total_pnl': self.trades_df['pnl'].sum(),
            'final_equity': self.equity_curve.iloc[-1],
        }
    
    def _calculate_risk_metrics(self, returns: pd.Series) -> dict:
        """Calculate risk metrics"""
        calc = MetricsCalculator()
        
        return {
            'sharpe_ratio': calc.calculate_sharpe_ratio(returns),
            'sortino_ratio': calc.calculate_sortino_ratio(returns),
            'calmar_ratio': calc.calculate_calmar_ratio(returns),
            'max_drawdown': calc.calculate_max_drawdown(self.equity_curve),
            'volatility': calc.calculate_volatility(returns),
            'var_95': calc.calculate_var(returns),
            'cvar_95': calc.calculate_cvar(returns),
        }
    
    def _calculate_trade_metrics(self) -> dict:
        """Calculate trade-specific metrics"""
        if len(self.trades_df) == 0:
            return {}
        
        winning = self.trades_df[self.trades_df['pnl'] > 0]
        losing = self.trades_df[self.trades_df['pnl'] < 0]
        
        avg_win = winning['pnl'].mean() if len(winning) > 0 else 0
        avg_loss = losing['pnl'].mean() if len(losing) > 0 else 0
        
        return {
            'win_rate': len(winning) / len(self.trades_df),
            'loss_rate': len(losing) / len(self.trades_df),
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': abs(winning['pnl'].sum() / losing['pnl'].sum()) if len(losing) > 0 else 0,
            'avg_trade_duration': self.trades_df['duration'].mean(),
        }
    
    def _calculate_performance_metrics(self, returns: pd.Series) -> dict:
        """Calculate return metrics"""
        total_return = (self.equity_curve.iloc[-1] / self.equity_curve.iloc[0]) - 1
        annual_return = (1 + total_return) ** (252 / len(self.equity_curve)) - 1
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'monthly_return': (1 + total_return) ** (12 / (len(self.equity_curve) / 252)) - 1,
            'positive_months': sum(returns.resample('M').sum() > 0),
        }
    
    def print_report(self):
        """Print formatted performance report"""
        report = self.generate_report()
        
        print("\n" + "="*60)
        print("PERFORMANCE REPORT")
        print("="*60)
        
        print("\nSUMMARY")
        for key, value in report['summary'].items():
            print(f"  {key}: {value}")
        
        print("\nRISK METRICS")
        for key, value in report['risk_metrics'].items():
            print(f"  {key}: {value:.4f}")
        
        print("\nTRADE METRICS")
        for key, value in report['trade_metrics'].items():
            print(f"  {key}: {value:.4f}")
        
        print("\nPERFORMANCE")
        for key, value in report['performance'].items():
            print(f"  {key}: {value:.4f}")
        
        print("="*60 + "\n")


if __name__ == "__main__":
    # Example usage
    returns = pd.Series(np.random.randn(252) * 0.01)
    
    calc = MetricsCalculator()
    sharpe = calc.calculate_sharpe_ratio(returns)
    sortino = calc.calculate_sortino_ratio(returns)
    
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Sortino Ratio: {sortino:.2f}")
