"""
Main trading orchestrator and execution engine
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List
from src.utils.logger import logger
from src.data_layer.data_manager import DataManager
from src.data_layer.feature_engineer import FeatureEngineer
from src.model_layer.neural_networks import LSTMModel, GRUModel, EnsembleModel
from src.strategy_layer.quant_strategies import (
    MeanReversionStrategy, MomentumStrategy, 
    StatisticalArbitrageStrategy, StrategyEnsemble
)
from src.risk_management.risk_manager import RiskManager, KellyCriterionSizer
from src.backtesting.backtest_engine import BacktestEngine
import torch


class TradingPosition:
    """Represents an open trading position"""
    
    def __init__(self, symbol: str, quantity: float, entry_price: float,
                 entry_date: datetime, side: str = 'long'):
        self.symbol = symbol
        self.quantity = quantity
        self.entry_price = entry_price
        self.entry_date = entry_date
        self.side = side
        self.current_price = entry_price
        self.pnl = 0
        self.pnl_pct = 0
    
    def update_price(self, price: float):
        """Update current price"""
        self.current_price = price
        
        if self.side == 'long':
            self.pnl = (price - self.entry_price) * self.quantity
            self.pnl_pct = (price - self.entry_price) / self.entry_price
        else:  # short
            self.pnl = (self.entry_price - price) * self.quantity
            self.pnl_pct = (self.entry_price - price) / self.entry_price
    
    def to_dict(self) -> dict:
        return {
            'symbol': self.symbol,
            'quantity': self.quantity,
            'entry_price': self.entry_price,
            'current_price': self.current_price,
            'entry_date': self.entry_date,
            'side': self.side,
            'pnl': self.pnl,
            'pnl_pct': self.pnl_pct
        }


class TradingEngine:
    """Main trading engine orchestrating all components"""
    
    def __init__(self, config: dict, initial_capital: float = 100000):
        """
        Args:
            config: Configuration dictionary
            initial_capital: Starting capital
        """
        self.config = config
        self.initial_capital = initial_capital
        
        # Initialize components
        self.data_manager = DataManager()
        self.feature_engineer = FeatureEngineer()
        self.risk_manager = RiskManager(
            initial_capital=initial_capital,
            max_position_size=config.get('max_position_size', 0.1),
            max_daily_loss=config.get('max_daily_loss', 0.05),
            max_drawdown=config.get('max_drawdown', 0.20)
        )
        
        # Models and strategies
        self.models = {}
        self.strategies = {}
        self.positions: Dict[str, TradingPosition] = {}
        
        self.trade_history = []
        self.portfolio_history = []
        
        logger.info("Trading engine initialized")
    
    def initialize_models(self, device: str = 'cpu'):
        """Initialize neural network models"""
        config = self.config
        
        # LSTM model
        lstm = LSTMModel(
            input_size=config.get('input_size', 50),
            hidden_size=config.get('hidden_size', 128),
            num_layers=config.get('num_layers', 2),
            device=device
        )
        self.models['lstm'] = lstm
        
        # GRU model
        gru = GRUModel(
            input_size=config.get('input_size', 50),
            hidden_size=config.get('hidden_size', 64),
            num_layers=config.get('num_layers', 2),
            device=device
        )
        self.models['gru'] = gru
        
        logger.info("Models initialized")
    
    def initialize_strategies(self):
        """Initialize trading strategies"""
        # Quantitative strategies
        self.strategies['mean_reversion'] = MeanReversionStrategy(
            window=20, threshold=2.0
        )
        self.strategies['momentum'] = MomentumStrategy(
            fast_period=12, slow_period=26
        )
        self.strategies['statistical_arb'] = StatisticalArbitrageStrategy(
            lookback=60
        )
        
        # Strategy ensemble
        strategy_list = list(self.strategies.values())
        self.strategies['ensemble'] = StrategyEnsemble(strategy_list)
        
        logger.info("Strategies initialized")
    
    def prepare_training_data(self, symbol: str, start_date: str, end_date: str):
        """Prepare data for training models"""
        logger.info(f"Preparing training data for {symbol}")
        
        # Fetch data
        data = self.data_manager.fetch_historical_data(symbol, start_date, end_date)
        data = self.data_manager.add_technical_indicators(data)
        data = self.data_manager.clean_data(data)
        
        # Create features
        features = self.feature_engineer.create_features(data)
        
        # Convert to numpy array
        feature_cols = [col for col in features.columns if col not in ['open', 'high', 'low']]
        X = features[feature_cols].values
        
        # Scale features
        X_scaled, _, scaler = self.feature_engineer.scale_features(X)
        
        # Create sequences
        X_seq, y_seq = self.feature_engineer.create_sequences(X_scaled)
        
        # Split data
        split_data = self.feature_engineer.split_data(X_seq, y_seq)
        
        logger.info(f"Training data prepared: X shape {X_seq.shape}")
        
        return split_data, scaler, features.index
    
    def train_models(self, symbol: str, start_date: str, end_date: str, epochs: int = 100):
        """Train neural network models"""
        logger.info(f"Training models for {symbol}")
        
        split_data, scaler, dates = self.prepare_training_data(symbol, start_date, end_date)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Train LSTM
        lstm_model = self.models['lstm']
        logger.info("Training LSTM model...")
        lstm_model.train(
            split_data['X_train'], split_data['y_train'],
            split_data['X_val'], split_data['y_val'],
            epochs=epochs, batch_size=32
        )
        
        # Train GRU
        gru_model = self.models['gru']
        logger.info("Training GRU model...")
        gru_model.train(
            split_data['X_train'], split_data['y_train'],
            split_data['X_val'], split_data['y_val'],
            epochs=epochs, batch_size=32
        )
        
        logger.info("Model training completed")
        return self.models
    
    def generate_signals(self, data: pd.DataFrame, use_ensemble: bool = True) -> pd.Series:
        """Generate trading signals from strategies"""
        if use_ensemble:
            signals = self.strategies['ensemble'].generate_signals(data)
        else:
            # Use primary strategy
            signals = self.strategies['momentum'].generate_signals(data)
        
        return signals
    
    def execute_trade(self, symbol: str, signal: int, price: float, 
                     date: datetime, quantity: float = 1):
        """Execute a trade"""
        
        # Check risk limits
        if not self.risk_manager.check_daily_loss_limit():
            logger.warning("Daily loss limit exceeded, skipping trade")
            return False
        
        if not self.risk_manager.check_drawdown_limit():
            logger.warning("Drawdown limit exceeded, skipping trade")
            return False
        
        if signal == 1:  # BUY
            # Check if position already exists
            if symbol in self.positions:
                logger.info(f"Position already exists for {symbol}, skipping buy")
                return False
            
            # Check position limits
            if not self.risk_manager.check_position_limits(symbol, quantity, 'buy'):
                logger.warning(f"Position limit exceeded for {symbol}")
                return False
            
            # Create position
            position = TradingPosition(
                symbol=symbol,
                quantity=quantity,
                entry_price=price,
                entry_date=date,
                side='long'
            )
            self.positions[symbol] = position
            self.trade_history.append({
                'date': date,
                'symbol': symbol,
                'side': 'BUY',
                'quantity': quantity,
                'price': price
            })
            
            logger.info(f"BUY {quantity} {symbol} at {price} on {date}")
            return True
        
        elif signal == -1:  # SELL
            # Check if position exists
            if symbol not in self.positions:
                logger.info(f"No open position for {symbol}, skipping sell")
                return False
            
            # Close position
            position = self.positions[symbol]
            pnl = (price - position.entry_price) * quantity if position.side == 'long' else \
                  (position.entry_price - price) * quantity
            
            self.trade_history.append({
                'date': date,
                'symbol': symbol,
                'side': 'SELL',
                'quantity': quantity,
                'price': price,
                'pnl': pnl
            })
            
            # Update capital
            self.risk_manager.update_capital(pnl)
            
            del self.positions[symbol]
            logger.info(f"SELL {quantity} {symbol} at {price} on {date}, PnL: {pnl:.2f}")
            return True
        
        return False
    
    def update_portfolio(self, current_prices: Dict[str, float]):
        """Update portfolio with current prices"""
        total_equity = self.risk_manager.current_capital
        
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                position.update_price(current_prices[symbol])
                total_equity += position.pnl
        
        self.portfolio_history.append({
            'timestamp': datetime.now(),
            'total_equity': total_equity,
            'cash': self.risk_manager.current_capital,
            'positions': len(self.positions),
            'open_pnl': sum(p.pnl for p in self.positions.values())
        })
    
    def backtest_strategy(self, symbol: str, start_date: str, end_date: str) -> dict:
        """Backtest the trading strategy"""
        logger.info(f"Starting backtest for {symbol}")
        
        # Fetch and prepare data
        data = self.data_manager.fetch_historical_data(symbol, start_date, end_date)
        data = self.data_manager.add_technical_indicators(data)
        data = self.data_manager.clean_data(data)
        
        # Generate signals
        signals = self.generate_signals(data)
        
        # Run backtest
        backtester = BacktestEngine(
            initial_capital=self.initial_capital,
            commission=0.001,
            slippage=0.0005
        )
        
        metrics = backtester.run_backtest(data, signals, symbol=symbol)
        
        logger.info(f"Backtest completed for {symbol}")
        return metrics
    
    def get_portfolio_summary(self) -> dict:
        """Get current portfolio summary"""
        total_equity = self.risk_manager.current_capital
        
        for position in self.positions.values():
            total_equity += position.pnl
        
        positions_summary = [p.to_dict() for p in self.positions.values()]
        
        return {
            'total_equity': total_equity,
            'cash': self.risk_manager.current_capital,
            'positions': positions_summary,
            'num_open_positions': len(self.positions),
            'open_pnl': sum(p.pnl for p in self.positions.values()),
            'risk_metrics': self.risk_manager.get_risk_metrics()
        }
    
    def save_models(self, model_dir: str):
        """Save trained models"""
        for name, model in self.models.items():
            path = f"{model_dir}/{name}_model.pt"
            model.save(path)
        logger.info(f"Models saved to {model_dir}")
    
    def load_models(self, model_dir: str):
        """Load trained models"""
        for name, model in self.models.items():
            path = f"{model_dir}/{name}_model.pt"
            model.load(path)
        logger.info(f"Models loaded from {model_dir}")


if __name__ == "__main__":
    # Example configuration
    config = {
        'input_size': 50,
        'hidden_size': 128,
        'num_layers': 2,
        'max_position_size': 0.1,
        'max_daily_loss': 0.05,
        'max_drawdown': 0.20
    }
    
    # Initialize engine
    engine = TradingEngine(config, initial_capital=100000)
    engine.initialize_models()
    engine.initialize_strategies()
    
    # Run backtest
    metrics = engine.backtest_strategy("AAPL", "2023-01-01", "2024-01-01")
    print(f"Backtest Results: {metrics}")
