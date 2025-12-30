#!/usr/bin/env python3
"""
Comprehensive model training script for the Neural Stock Trader system.
Trains LSTM, GRU, and Ensemble models on multiple symbols with cross-validation.
"""

import os
import sys
import warnings
import torch
import numpy as np
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_layer.data_manager import YFinanceDataSource, DataManager
from data_layer.feature_engineer import FeatureEngineer
from model_layer.neural_networks import LSTMModel, GRUModel, EnsembleModel
from utils.config_manager import ConfigManager
from utils.logger import TradingLogger
from utils.metrics import PerformanceReporter

warnings.filterwarnings('ignore')

def train_models():
    """Train all models on multiple stock symbols."""
    
    # Initialize
    logger = TradingLogger()
    config = ConfigManager()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Configuration
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*2)  # 2 years of data
    
    print("\n" + "="*80)
    print("NEURAL STOCK TRADER - COMPREHENSIVE MODEL TRAINING")
    print("="*80 + "\n")
    
    logger.info(f"Training period: {start_date.date()} to {end_date.date()}")
    logger.info(f"Symbols: {', '.join(symbols)}")
    logger.info(f"Device: {device}")
    
    # Data fetching and preparation
    print("\n--- Phase 1: Data Collection ---\n")
    data_manager = DataManager(YFinanceDataSource())
    feature_engineer = FeatureEngineer()
    
    all_data = {}
    for symbol in symbols:
        logger.info(f"Fetching data for {symbol}...")
        try:
            df = data_manager.fetch_historical_data(symbol, start_date, end_date)
            df = data_manager.add_technical_indicators(df)
            all_data[symbol] = df
            logger.info(f"  ✓ Fetched {len(df)} days for {symbol}")
        except Exception as e:
            logger.warning(f"  ✗ Failed to fetch {symbol}: {str(e)}")
            continue
    
    if not all_data:
        logger.error("Failed to fetch data for any symbol!")
        return
    
    # Feature engineering and model training
    print("\n--- Phase 2: Feature Engineering ---\n")
    trained_models = {}
    
    for symbol, df in all_data.items():
        logger.info(f"Engineering features for {symbol}...")
        
        try:
            # Create features
            df_features = feature_engineer.create_features(df)
            
            # Create sequences
            X, y = feature_engineer.create_sequences(df_features, lookback=60)
            
            if len(X) < 100:
                logger.warning(f"  ✗ Insufficient data for {symbol} after sequence creation")
                continue
            
            # Scale features
            X_scaled, scaler = feature_engineer.scale_features(X, method='standard')
            
            # Split data
            split_idx = int(0.8 * len(X_scaled))
            X_train, y_train = X_scaled[:split_idx], y[:split_idx]
            X_val, y_val = X_scaled[split_idx:], y[split_idx:]
            
            logger.info(f"  ✓ Created {len(X)} sequences")
            logger.info(f"    Training set: {len(X_train)} samples")
            logger.info(f"    Validation set: {len(X_val)} samples")
            
            # Train LSTM
            print(f"\n  Training LSTM for {symbol}...")
            lstm_model = LSTMModel(
                input_size=X_train.shape[2],
                hidden_size=config.get('neural_network.lstm.hidden_size', 128),
                num_layers=config.get('neural_network.lstm.num_layers', 2),
                output_size=1,
                dropout=config.get('neural_network.lstm.dropout', 0.2),
                device=device
            )
            
            lstm_model.to(device)
            lstm_model.train_model(
                X_train, y_train, X_val, y_val,
                epochs=config.get('neural_network.training.epochs', 50),
                batch_size=config.get('neural_network.training.batch_size', 32),
                learning_rate=config.get('neural_network.training.learning_rate', 0.001)
            )
            
            logger.info(f"  ✓ LSTM trained for {symbol}")
            
            # Train GRU
            print(f"  Training GRU for {symbol}...")
            gru_model = GRUModel(
                input_size=X_train.shape[2],
                hidden_size=config.get('neural_network.gru.hidden_size', 64),
                num_layers=config.get('neural_network.gru.num_layers', 2),
                output_size=1,
                dropout=config.get('neural_network.gru.dropout', 0.2),
                device=device
            )
            
            gru_model.to(device)
            gru_model.train_model(
                X_train, y_train, X_val, y_val,
                epochs=config.get('neural_network.training.epochs', 50),
                batch_size=config.get('neural_network.training.batch_size', 32),
                learning_rate=config.get('neural_network.training.learning_rate', 0.001)
            )
            
            logger.info(f"  ✓ GRU trained for {symbol}")
            
            # Create Ensemble
            print(f"  Creating Ensemble for {symbol}...")
            ensemble_model = EnsembleModel(
                models=[lstm_model, gru_model],
                weights=[0.5, 0.5],
                device=device
            )
            
            logger.info(f"  ✓ Ensemble created for {symbol}")
            
            # Store models
            trained_models[symbol] = {
                'lstm': lstm_model,
                'gru': gru_model,
                'ensemble': ensemble_model,
                'scaler': scaler,
                'input_size': X_train.shape[2]
            }
            
        except Exception as e:
            logger.error(f"  ✗ Error training models for {symbol}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save trained models
    print("\n--- Phase 3: Saving Models ---\n")
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    for symbol, models in trained_models.items():
        try:
            lstm_path = os.path.join(models_dir, f'lstm_{symbol}.pt')
            gru_path = os.path.join(models_dir, f'gru_{symbol}.pt')
            ensemble_path = os.path.join(models_dir, f'ensemble_{symbol}.pt')
            
            torch.save(models['lstm'].state_dict(), lstm_path)
            torch.save(models['gru'].state_dict(), gru_path)
            torch.save(models['ensemble'].state_dict(), ensemble_path)
            
            logger.info(f"✓ Saved LSTM model for {symbol}")
            logger.info(f"✓ Saved GRU model for {symbol}")
            logger.info(f"✓ Saved Ensemble model for {symbol}")
        except Exception as e:
            logger.error(f"Failed to save models for {symbol}: {str(e)}")
    
    # Generate summary
    print("\n--- Training Summary ---\n")
    logger.info(f"Successfully trained models for {len(trained_models)} symbols:")
    for symbol in trained_models.keys():
        logger.info(f"  ✓ {symbol}: LSTM, GRU, Ensemble")
    
    logger.info(f"Models saved to: {models_dir}")
    logger.info(f"Training completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"\nModels saved to: {models_dir}")
    print(f"Trained symbols: {', '.join(trained_models.keys())}")
    print(f"\nNext steps:")
    print("  1. Use trained models in backtesting: python main.py --mode backtest --symbol AAPL")
    print("  2. Run examples: python examples.py")
    print("  3. Analyze results: Check logs/ directory\n")
    
    return trained_models

if __name__ == "__main__":
    try:
        trained_models = train_models()
    except Exception as e:
        logger = TradingLogger()
        logger.error(f"Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
