#!/usr/bin/env python3
"""
Minimal training script to test and train the neural network models.
"""

import os
import sys
import warnings
import torch
import numpy as np
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("\n" + "="*80)
print("NEURAL STOCK TRADER - MODEL TRAINING")
print("="*80 + "\n")

try:
    print("Loading modules...")
    from utils.logger import TradingLogger
    from utils.config_manager import ConfigManager
    
    logger = TradingLogger()
    config = ConfigManager()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("[OK] Modules loaded successfully")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] PyTorch version: {torch.__version__}")
    print(f"[INFO] NumPy version: {np.__version__}")
    
    # Create models directory
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    print("\nLoading data...")
    from data_layer.data_manager import YFinanceDataSource, DataManager
    
    data_manager = DataManager(YFinanceDataSource())
    
    # Train on a single symbol first
    symbol = 'AAPL'
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # 1 year
    
    print(f"\nFetching {symbol} data from {start_date.date()} to {end_date.date()}...")
    df = data_manager.fetch_historical_data(symbol, start_date, end_date)
    
    if df is None or len(df) == 0:
        print(f"[ERROR] Failed to fetch data for {symbol}")
        sys.exit(1)
    
    print(f"[OK] Fetched {len(df)} days of {symbol} data")
    
    # Add technical indicators
    print("\nAdding technical indicators...")
    df = data_manager.add_technical_indicators(df)
    print(f"[OK] Added technical indicators")
    
    # Feature engineering
    print("\nPreparing features...")
    
    # Use all numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df_clean = df[numeric_cols].dropna()
    
    print(f"[OK] Using {len(numeric_cols)} numeric features")
    print(f"[OK] Clean data: {len(df_clean)} rows")
    
    # Convert to numpy array
    data_array = df_clean.values
    
    # Create sequences manually with smaller lookback if needed
    lookback = min(30, len(data_array) // 4)  # Use 30 or less based on available data
    X, y = [], []
    close_idx = numeric_cols.index('close') if 'close' in numeric_cols else 3
    
    print(f"[INFO] Using lookback={lookback}")
    
    for i in range(len(data_array) - lookback):
        X.append(data_array[i:i + lookback])
        y.append(data_array[i + lookback, close_idx])
    
    X = np.array(X)
    y = np.array(y)
    print(f"[OK] Created {len(X)} training sequences")
    
    if len(X) < 30:
        print(f"[ERROR] Not enough data. Need at least 30 sequences, got {len(X)}")
        sys.exit(1)
    
    # Scale data
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_reshaped = X.reshape(-1, X.shape[-1])
    X_scaled = scaler.fit_transform(X_reshaped).reshape(X.shape)
    
    # Split train/val
    split_idx = int(0.8 * len(X_scaled))
    X_train = X_scaled[:split_idx]
    y_train = y[:split_idx]
    X_val = X_scaled[split_idx:]
    y_val = y[split_idx:]
    
    print(f"[OK] Training samples: {len(X_train)}")
    print(f"[OK] Validation samples: {len(X_val)}")
    print(f"[OK] Features per sample: {X_train.shape[2]}")
    
    # Train LSTM
    print("\n" + "-"*80)
    print("Training LSTM Model")
    print("-"*80)
    
    from model_layer.neural_networks import LSTMModel
    
    lstm = LSTMModel(
        input_size=X_train.shape[2],
        hidden_size=128,
        num_layers=2,
        output_size=1,
        dropout=0.2,
        device=device
    )
    lstm.to(device)
    
    print("\nTraining LSTM...")
    lstm.train_model(
        X_train, y_train, X_val, y_val,
        epochs=30,
        batch_size=32,
        learning_rate=0.001
    )
    print("[OK] LSTM training completed")
    
    # Train GRU
    print("\n" + "-"*80)
    print("Training GRU Model")
    print("-"*80)
    
    from model_layer.neural_networks import GRUModel
    
    gru = GRUModel(
        input_size=X_train.shape[2],
        hidden_size=64,
        num_layers=2,
        output_size=1,
        dropout=0.2,
        device=device
    )
    gru.to(device)
    
    print("\nTraining GRU...")
    gru.train_model(
        X_train, y_train, X_val, y_val,
        epochs=30,
        batch_size=32,
        learning_rate=0.001
    )
    print("[OK] GRU training completed")
    
    # Create Ensemble
    print("\n" + "-"*80)
    print("Creating Ensemble Model")
    print("-"*80)
    
    from model_layer.neural_networks import EnsembleModel
    
    ensemble = EnsembleModel(
        models=[lstm, gru],
        weights=[0.5, 0.5],
        device=device
    )
    print("[OK] Ensemble model created")
    
    # Save models
    print("\n" + "-"*80)
    print("Saving Models")
    print("-"*80)
    
    lstm_path = os.path.join(models_dir, f'lstm_{symbol}.pt')
    gru_path = os.path.join(models_dir, f'gru_{symbol}.pt')
    ensemble_path = os.path.join(models_dir, f'ensemble_{symbol}.pt')
    
    torch.save(lstm.state_dict(), lstm_path)
    torch.save(gru.state_dict(), gru_path)
    torch.save(ensemble.state_dict(), ensemble_path)
    
    print(f"[OK] LSTM saved to {lstm_path}")
    print(f"[OK] GRU saved to {gru_path}")
    print(f"[OK] Ensemble saved to {ensemble_path}")
    
    # Test predictions
    print("\n" + "-"*80)
    print("Testing Predictions")
    print("-"*80)
    
    lstm.eval()
    gru.eval()
    ensemble.eval()
    
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_val[:5], dtype=torch.float32).to(device)
        
        lstm_pred = lstm(X_test_tensor)
        gru_pred = gru(X_test_tensor)
        ensemble_pred = ensemble(X_test_tensor)
        
        print(f"\nFirst 5 predictions:")
        print(f"LSTM:     {lstm_pred.cpu().numpy().flatten()}")
        print(f"GRU:      {gru_pred.cpu().numpy().flatten()}")
        print(f"Ensemble: {ensemble_pred.cpu().numpy().flatten()}")
        print(f"Actual:   {y_val[:5].flatten()}")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"\n[OK] Models trained and saved for {symbol}")
    print(f"[OK] Location: {models_dir}")
    print(f"\nNext steps:")
    print(f"  1. Run backtest: python main.py --mode backtest --symbol {symbol}")
    print(f"  2. Run examples: python examples.py")
    print(f"  3. Check logs: logs/")
    print("\n")
    
except Exception as e:
    print(f"\n[ERROR] Error during training: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
