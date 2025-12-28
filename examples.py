"""
Example usage and testing script for NeuralStockTrader
"""

import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.execution_layer.trading_engine import TradingEngine
from src.data_layer.data_manager import DataManager
from src.data_layer.feature_engineer import FeatureEngineer
from src.strategy_layer.quant_strategies import (
    MeanReversionStrategy, MomentumStrategy, StrategyEnsemble
)
from src.backtesting.backtest_engine import BacktestEngine
from src.utils.metrics import MetricsCalculator, PerformanceReporter
from src.utils.logger import logger


def example_1_basic_backtest():
    """Example 1: Run a basic backtest"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Backtest")
    print("="*60)
    
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
    engine.initialize_strategies()
    
    # Run backtest
    metrics = engine.backtest_strategy("AAPL", "2023-01-01", "2024-01-01")
    
    # Print results
    print(f"\nBacktest Results for AAPL:")
    print(f"  Total Return: {metrics['total_return_pct']:.2f}%")
    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"  Win Rate: {metrics['win_rate_pct']:.2f}%")
    print(f"  Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
    print(f"  Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"  Number of Trades: {metrics['num_trades']}")


def example_2_strategy_comparison():
    """Example 2: Compare multiple strategies"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Strategy Comparison")
    print("="*60)
    
    # Fetch data
    dm = DataManager()
    data = dm.fetch_historical_data("MSFT", "2023-01-01", "2024-01-01")
    data = dm.add_technical_indicators(data)
    data = dm.clean_data(data)
    
    # Define strategies
    strategies = {
        'Mean Reversion': MeanReversionStrategy(window=20, threshold=2.0),
        'Momentum': MomentumStrategy(fast_period=12, slow_period=26),
    }
    
    # Test each strategy
    results = {}
    for name, strategy in strategies.items():
        signals = strategy.generate_signals(data)
        
        backtester = BacktestEngine(initial_capital=100000)
        metrics = backtester.run_backtest(data, signals, symbol="MSFT")
        
        results[name] = metrics
    
    # Compare results
    print(f"\nStrategy Performance Comparison:")
    print(f"{'Strategy':<20} {'Return':<15} {'Sharpe':<15} {'Win Rate':<15}")
    print("-" * 65)
    
    for name, metrics in results.items():
        print(f"{name:<20} {metrics['total_return_pct']:>6.2f}% {metrics['sharpe_ratio']:>13.2f} {metrics['win_rate_pct']:>13.2f}%")


def example_3_data_preparation():
    """Example 3: Data preparation and feature engineering"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Data Preparation & Feature Engineering")
    print("="*60)
    
    # Fetch data
    dm = DataManager()
    data = dm.fetch_historical_data("GOOGL", "2023-01-01", "2024-01-01")
    
    print(f"Raw data shape: {data.shape}")
    print(f"Columns: {data.columns.tolist()}")
    
    # Add technical indicators
    data = dm.add_technical_indicators(data)
    print(f"\nAfter adding indicators: {data.shape}")
    
    # Create features
    fe = FeatureEngineer()
    features = fe.create_features(data)
    print(f"After feature engineering: {features.shape}")
    print(f"Feature columns: {list(features.columns)[:10]}... (showing first 10)")
    
    # Create sequences
    X_data = features[features.columns[:-5]].values  # Exclude some columns
    X_seq, y_seq = fe.create_sequences(X_data, lookback=60)
    print(f"\nSequence shapes:")
    print(f"  X_seq shape: {X_seq.shape}")
    print(f"  y_seq shape: {y_seq.shape}")
    
    # Scale features
    X_scaled, _, scaler = fe.scale_features(X_seq)
    print(f"Scaled data range: [{X_scaled.min():.2f}, {X_scaled.max():.2f}]")


def example_4_risk_management():
    """Example 4: Risk management and position sizing"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Risk Management")
    print("="*60)
    
    from src.risk_management.risk_manager import (
        RiskManager, KellyCriterionSizer, RiskParitySizer, StopLoss, TakeProfit
    )
    
    # Initialize risk manager
    rm = RiskManager(initial_capital=100000)
    
    print(f"\nInitial Risk Metrics:")
    metrics = rm.get_risk_metrics()
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Test position sizing
    print(f"\nPosition Sizing Methods:")
    
    kelly_sizer = KellyCriterionSizer(kelly_fraction=0.25)
    kelly_size = kelly_sizer.calculate_size(
        capital=100000, win_rate=0.55, avg_win=100, avg_loss=90
    )
    print(f"  Kelly Criterion: ${kelly_size:.2f}")
    
    rp_sizer = RiskParitySizer(target_risk=0.02)
    rp_size = rp_sizer.calculate_size(capital=100000, volatility=0.015)
    print(f"  Risk Parity: ${rp_size:.2f}")
    
    # Test stop loss and take profit
    print(f"\nStop Loss & Take Profit:")
    
    sl = StopLoss(stop_type="hard", stop_pct=0.02)
    sl.set_entry(100)
    print(f"  Entry: $100, Stop Loss: ${sl.get_stop_price():.2f}")
    
    tp = TakeProfit(take_profit_pct=0.05)
    tp.set_entry(100)
    print(f"  Entry: $100, Take Profit Target: ${tp.get_target_price():.2f}")


def example_5_metrics_calculation():
    """Example 5: Calculate performance metrics"""
    print("\n" + "="*60)
    print("EXAMPLE 5: Performance Metrics")
    print("="*60)
    
    # Generate sample returns
    np.random.seed(42)
    daily_returns = np.random.randn(252) * 0.01 + 0.0005  # 5% annual return
    returns = pd.Series(daily_returns)
    
    # Calculate metrics
    calc = MetricsCalculator()
    
    print(f"\nCalculated Metrics:")
    print(f"  Sharpe Ratio: {calc.calculate_sharpe_ratio(returns):.2f}")
    print(f"  Sortino Ratio: {calc.calculate_sortino_ratio(returns):.2f}")
    print(f"  Calmar Ratio: {calc.calculate_calmar_ratio(returns):.2f}")
    print(f"  Volatility (Annual): {calc.calculate_volatility(returns):.2%}")
    print(f"  VaR (95%): {calc.calculate_var(returns):.2%}")
    print(f"  CVaR (95%): {calc.calculate_cvar(returns):.2%}")


def example_6_neural_network_training():
    """Example 6: Train neural network models"""
    print("\n" + "="*60)
    print("EXAMPLE 6: Neural Network Training")
    print("="*60)
    
    from src.model_layer.neural_networks import LSTMModel
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = LSTMModel(input_size=50, hidden_size=128, num_layers=2, device=str(device))
    
    # Generate sample data
    print("\nGenerating sample training data...")
    X_train = np.random.randn(500, 60, 50).astype(np.float32)
    y_train = np.random.randn(500).astype(np.float32)
    X_val = np.random.randn(100, 60, 50).astype(np.float32)
    y_val = np.random.randn(100).astype(np.float32)
    
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    
    # Train model
    print("\nTraining LSTM model (10 epochs for demo)...")
    train_losses, val_losses = model.train(
        X_train, y_train, X_val, y_val,
        epochs=10, batch_size=32
    )
    
    print(f"\nTraining completed!")
    print(f"  Final training loss: {train_losses[-1]:.4f}")
    print(f"  Final validation loss: {val_losses[-1]:.4f}")
    
    # Make predictions
    predictions = model.predict(X_val[:5])
    print(f"\nSample predictions shape: {predictions.shape}")


def run_all_examples():
    """Run all examples"""
    print("\n" + "="*60)
    print("NeuralStockTrader - Example Scripts")
    print("="*60)
    
    try:
        example_1_basic_backtest()
    except Exception as e:
        logger.error(f"Example 1 failed: {str(e)}")
    
    try:
        example_2_strategy_comparison()
    except Exception as e:
        logger.error(f"Example 2 failed: {str(e)}")
    
    try:
        example_3_data_preparation()
    except Exception as e:
        logger.error(f"Example 3 failed: {str(e)}")
    
    try:
        example_4_risk_management()
    except Exception as e:
        logger.error(f"Example 4 failed: {str(e)}")
    
    try:
        example_5_metrics_calculation()
    except Exception as e:
        logger.error(f"Example 5 failed: {str(e)}")
    
    try:
        example_6_neural_network_training()
    except Exception as e:
        logger.error(f"Example 6 failed: {str(e)}")
    
    print("\n" + "="*60)
    print("All examples completed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    # Run individual examples or all
    import sys
    
    if len(sys.argv) > 1:
        example_num = sys.argv[1]
        if example_num == "1":
            example_1_basic_backtest()
        elif example_num == "2":
            example_2_strategy_comparison()
        elif example_num == "3":
            example_3_data_preparation()
        elif example_num == "4":
            example_4_risk_management()
        elif example_num == "5":
            example_5_metrics_calculation()
        elif example_num == "6":
            example_6_neural_network_training()
        else:
            print(f"Unknown example: {example_num}")
    else:
        # Run example 1 by default
        example_1_basic_backtest()
