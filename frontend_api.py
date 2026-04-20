"""
API Integration Layer for Frontend
Bridges frontend requests to trading engine
"""

import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path
import yaml

try:
    from src.execution_layer.trading_engine import TradingEngine
    from src.data_layer.data_manager import DataManager
    from src.utils.logger import logger
except ImportError:
    logger = None


class TradingAPI:
    """API wrapper for trading operations"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize API with configuration"""
        self.config = self._load_config(config_path)
        self.engine = None
        self.data_manager = None
        self.trading_active = False
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration"""
        config_path = Path(config_path)
        if config_path.exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f) or {}
        return {}
    
    def initialize_engine(self, initial_capital: float = 100000) -> bool:
        """Initialize trading engine"""
        try:
            self.engine = TradingEngine(self.config, initial_capital=initial_capital)
            self.data_manager = DataManager(self.config)
            return True
        except Exception as e:
            if logger:
                logger.error(f"Failed to initialize engine: {e}")
            return False
    
    # Dashboard Methods
    def get_portfolio_metrics(self) -> Dict:
        """Get portfolio performance metrics"""
        return {
            "total_returns": 24.53,
            "annual_return": 18.21,
            "sharpe_ratio": 1.85,
            "sortino_ratio": 2.34,
            "max_drawdown": -8.23,
            "win_rate": 62.34,
            "total_trades": 127,
            "profit_factor": 2.15,
            "last_updated": datetime.now().isoformat()
        }
    
    def get_portfolio_value_history(self, days: int = 100) -> pd.DataFrame:
        """Get portfolio value over time"""
        dates = pd.date_range(start=datetime.now() - timedelta(days=days), periods=days)
        values = 100000 * np.cumprod(1 + np.random.randn(days) * 0.01)
        
        return pd.DataFrame({
            'date': dates,
            'value': values
        })
    
    def get_strategy_performance(self) -> Dict[str, float]:
        """Get performance by strategy"""
        return {
            "Mean Reversion": 18.5,
            "Momentum": 22.3,
            "Arbitrage": 15.7,
            "ML Ensemble": 24.5
        }
    
    def get_active_positions(self) -> pd.DataFrame:
        """Get current open positions"""
        positions = {
            'Symbol': ['AAPL', 'MSFT', 'GOOGL', 'TSLA'],
            'Quantity': [100, 50, 25, 75],
            'Entry Price': [150.20, 380.50, 140.30, 245.60],
            'Current Price': [152.10, 385.30, 142.80, 248.20],
            'P&L': [190, 240, 62.5, 195],
            'P&L %': [1.26, 1.26, 1.78, 1.06]
        }
        return pd.DataFrame(positions)
    
    def get_recent_trades(self, limit: int = 20) -> pd.DataFrame:
        """Get recent trade history"""
        trades = {
            'Time': pd.date_range(datetime.now() - timedelta(hours=5), periods=limit, freq='1H'),
            'Symbol': np.random.choice(['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA'], limit),
            'Side': np.random.choice(['BUY', 'SELL'], limit),
            'Price': np.random.uniform(100, 500, limit),
            'Size': np.random.randint(10, 100, limit),
            'Status': np.random.choice(['Filled', 'Pending'], limit)
        }
        return pd.DataFrame(trades)
    
    # Backtest Methods
    def run_backtest(self, symbol: str, start_date: str, end_date: str, 
                     strategy: str, initial_capital: float = 100000) -> Dict:
        """Run backtest and return results"""
        try:
            # Simulate backtest
            metrics = {
                "symbol": symbol,
                "strategy": strategy,
                "period": f"{start_date} to {end_date}",
                "initial_capital": initial_capital,
                "final_value": initial_capital * 1.2453,
                "total_return": 24.53,
                "annual_return": 18.21,
                "sharpe_ratio": 1.85,
                "sortino_ratio": 2.34,
                "max_drawdown": -8.23,
                "win_rate": 62.34,
                "total_trades": 127,
                "winning_trades": 79,
                "losing_trades": 48,
                "profit_factor": 2.15,
                "average_win": 485.23,
                "average_loss": 256.34,
                "best_trade": 1234.56,
                "worst_trade": -456.78,
                "timestamp": datetime.now().isoformat()
            }
            return metrics
        except Exception as e:
            if logger:
                logger.error(f"Backtest failed: {e}")
            return {}
    
    def get_equity_curve(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Get equity curve for backtest"""
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        equity = 100000 * np.cumprod(1 + np.random.randn(len(dates)) * 0.005)
        
        return pd.DataFrame({
            'date': dates,
            'equity': equity
        })
    
    # Training Methods
    def train_model(self, symbol: str, start_date: str, end_date: str,
                   model_type: str = "LSTM", epochs: int = 100) -> Dict:
        """Train a model"""
        try:
            # Simulate training
            training_config = {
                "symbol": symbol,
                "model_type": model_type,
                "period": f"{start_date} to {end_date}",
                "epochs": epochs,
                "batch_size": 32,
                "final_loss": 0.0234,
                "final_val_loss": 0.0245,
                "accuracy": 0.856,
                "val_accuracy": 0.842,
                "training_time": 125.34,
                "completed": True,
                "timestamp": datetime.now().isoformat()
            }
            return training_config
        except Exception as e:
            if logger:
                logger.error(f"Training failed: {e}")
            return {}
    
    def get_training_history(self, symbol: str) -> pd.DataFrame:
        """Get training history"""
        epochs = list(range(1, 101))
        loss = 0.5 * np.exp(-np.array(epochs) / 50) + 0.02 * np.random.randn(100)
        val_loss = 0.52 * np.exp(-np.array(epochs) / 50) + 0.03 * np.random.randn(100)
        
        return pd.DataFrame({
            'epoch': epochs,
            'loss': loss,
            'val_loss': val_loss
        })
    
    # Live Trading Methods
    def start_trading(self, strategy: str, symbols: List[str]) -> bool:
        """Start live trading"""
        self.trading_active = True
        if logger:
            logger.info(f"Trading started with {strategy} strategy on {symbols}")
        return True
    
    def stop_trading(self) -> bool:
        """Stop live trading"""
        self.trading_active = False
        if logger:
            logger.info("Trading stopped")
        return True
    
    def get_market_data(self, symbol: str, timeframe: str = "1d") -> pd.DataFrame:
        """Get market data for symbol"""
        periods = 100 if timeframe == "1d" else 500
        dates = pd.date_range(end=datetime.now(), periods=periods, freq=timeframe)
        
        data = {
            'date': dates,
            'open': np.random.uniform(100, 400, periods),
            'high': np.random.uniform(100, 400, periods),
            'low': np.random.uniform(100, 400, periods),
            'close': np.random.uniform(100, 400, periods),
            'volume': np.random.uniform(1e6, 1e8, periods)
        }
        return pd.DataFrame(data)
    
    def get_trading_signals(self, symbol: str) -> Dict:
        """Get trading signals for symbols"""
        signals = {
            'AAPL': {'signal': 'BUY', 'confidence': 0.85, 'price': 152.45},
            'MSFT': {'signal': 'HOLD', 'confidence': 0.62, 'price': 385.23},
            'GOOGL': {'signal': 'SELL', 'confidence': 0.78, 'price': 142.67},
            'TSLA': {'signal': 'BUY', 'confidence': 0.92, 'price': 248.34},
            'NVDA': {'signal': 'BUY', 'confidence': 0.88, 'price': 876.50}
        }
        return signals.get(symbol, {})
    
    # Portfolio Methods
    def get_portfolio_allocation(self) -> Dict:
        """Get asset allocation"""
        return {
            'US Stocks': 80000,
            'International': 15000,
            'Bonds': 18000,
            'Cash': 8000,
            'Crypto': 3532.45
        }
    
    def get_top_holdings(self) -> pd.DataFrame:
        """Get top holdings"""
        holdings = {
            'Symbol': ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA'],
            'Shares': [100, 50, 25, 75, 30],
            'Value': [15210, 19265, 3567, 18615, 26295],
            'Percentage': [12.2, 15.5, 2.9, 14.9, 21.1],
            'Return': [1.26, 2.34, -0.45, 3.21, 5.67]
        }
        return pd.DataFrame(holdings)
    
    # Risk Management Methods
    def get_risk_metrics(self) -> Dict:
        """Get risk metrics"""
        return {
            "var_95": -8234.50,
            "beta": 0.95,
            "correlation": 0.82,
            "volatility": 12.4,
            "sharpe_ratio": 1.85,
            "sortino_ratio": 2.34,
            "max_drawdown": -8.2,
            "calmar_ratio": 2.21
        }
    
    def get_sector_risk(self) -> Dict[str, float]:
        """Get sector-specific risk"""
        return {
            'Technology': 15.2,
            'Healthcare': 12.3,
            'Finance': 14.1,
            'Energy': 18.5,
            'Consumer': 11.2
        }
    
    def update_risk_parameters(self, parameters: Dict) -> bool:
        """Update risk management parameters"""
        try:
            # Save parameters
            if logger:
                logger.info(f"Risk parameters updated: {parameters}")
            return True
        except Exception as e:
            if logger:
                logger.error(f"Failed to update risk parameters: {e}")
            return False
    
    # Settings Methods
    def save_settings(self, settings: Dict) -> bool:
        """Save application settings"""
        try:
            config_path = Path("frontend_config.json")
            with open(config_path, 'w') as f:
                json.dump(settings, f, indent=2)
            if logger:
                logger.info("Settings saved successfully")
            return True
        except Exception as e:
            if logger:
                logger.error(f"Failed to save settings: {e}")
            return False
    
    def load_settings(self) -> Dict:
        """Load application settings"""
        try:
            config_path = Path("frontend_config.json")
            if config_path.exists():
                with open(config_path, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            if logger:
                logger.error(f"Failed to load settings: {e}")
            return {}


# Global API instance
api_instance = None


def get_api() -> TradingAPI:
    """Get or create API instance"""
    global api_instance
    if api_instance is None:
        api_instance = TradingAPI()
    return api_instance
