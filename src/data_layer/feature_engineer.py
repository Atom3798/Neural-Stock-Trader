"""
Feature engineering pipeline for preparing data for neural networks
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from src.utils.logger import logger
from src.utils.constants import DEFAULT_LOOKBACK


class FeatureEngineer:
    """Feature extraction and transformation"""
    
    def __init__(self, lookback: int = DEFAULT_LOOKBACK):
        self.lookback = lookback
        self.scalers = {}
    
    def create_features(self, data: pd.DataFrame, technical_only=False) -> pd.DataFrame:
        """Create comprehensive feature set"""
        features = data.copy()
        
        # Price-based features
        features['price_range'] = (features['high'] - features['low']) / features['close']
        features['body_size'] = abs(features['close'] - features['open']) / features['close']
        features['upper_wick'] = (features['high'] - np.maximum(features['open'], features['close'])) / features['close']
        features['lower_wick'] = (np.minimum(features['open'], features['close']) - features['low']) / features['close']
        
        # Volume features
        features['volume_ma_ratio'] = features['volume'] / features['volume'].rolling(20).mean()
        features['volume_volatility'] = features['volume'].rolling(20).std() / features['volume'].rolling(20).mean()
        
        # Volatility features
        features['volatility_10'] = features['returns'].rolling(10).std()
        features['volatility_20'] = features['returns'].rolling(20).std()
        features['volatility_50'] = features['returns'].rolling(50).std()
        
        # Trend features
        features['trend_sma_ratio'] = features['close'] / features['sma_20']
        features['trend_momentum'] = features['close'] - features['close'].shift(10)
        
        # High-low features
        features['hl_ratio'] = features['high'] / features['low']
        features['close_high_ratio'] = features['close'] / features['high']
        features['close_low_ratio'] = features['close'] / features['low']
        
        # Cross-moving average signals
        if 'ema_12' in features.columns and 'ema_26' in features.columns:
            features['ema_crossover'] = (features['ema_12'] > features['ema_26']).astype(int)
        
        if 'sma_20' in features.columns and 'sma_50' in features.columns:
            features['sma_crossover'] = (features['sma_20'] > features['sma_50']).astype(int)
        
        # RSI features
        if 'rsi' in features.columns:
            features['rsi_overbought'] = (features['rsi'] > 70).astype(int)
            features['rsi_oversold'] = (features['rsi'] < 30).astype(int)
        
        # Bollinger Bands features
        if all(x in features.columns for x in ['bb_upper', 'bb_lower', 'bb_middle']):
            features['bb_position'] = (features['close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
            features['bb_squeeze'] = (features['bb_upper'] - features['bb_lower']) / features['bb_middle']
        
        # MACD features
        if all(x in features.columns for x in ['macd', 'macd_signal']):
            features['macd_signal_crossover'] = (features['macd'] > features['macd_signal']).astype(int)
        
        # Drop NaN values
        features = features.dropna()
        
        logger.info(f"Created {len(features.columns)} features")
        return features
    
    def create_sequences(self, data: np.ndarray, lookback: int = None) -> tuple:
        """Create sequences for LSTM/GRU networks"""
        lookback = lookback or self.lookback
        X, y = [], []
        
        for i in range(len(data) - lookback):
            X.append(data[i:i + lookback])
            # Target: predict next close price (or log return)
            y.append(data[i + lookback, 3])  # 3 is close price column
        
        return np.array(X), np.array(y)
    
    def create_advanced_sequences(self, data: np.ndarray, lookback: int = None,
                                  forecast_horizon: int = 5) -> tuple:
        """Create sequences with multiple forecast horizons"""
        lookback = lookback or self.lookback
        X, y = [], []
        
        for i in range(len(data) - lookback - forecast_horizon + 1):
            X.append(data[i:i + lookback])
            # Multi-step forecast
            y.append(data[i + lookback:i + lookback + forecast_horizon, 3])
        
        return np.array(X), np.array(y)
    
    def scale_features(self, train_data: np.ndarray, test_data: np.ndarray = None,
                      method: str = 'standard') -> tuple:
        """Scale features using training data"""
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        # Reshape if needed
        original_shape = train_data.shape
        if len(train_data.shape) > 2:
            train_data_reshaped = train_data.reshape(-1, train_data.shape[-1])
        else:
            train_data_reshaped = train_data
        
        # Fit on training data
        scaler.fit(train_data_reshaped)
        train_scaled = scaler.transform(train_data_reshaped)
        
        # Reshape back
        if len(original_shape) > 2:
            train_scaled = train_scaled.reshape(original_shape)
        
        test_scaled = None
        if test_data is not None:
            original_test_shape = test_data.shape
            if len(test_data.shape) > 2:
                test_data_reshaped = test_data.reshape(-1, test_data.shape[-1])
            else:
                test_data_reshaped = test_data
            
            test_scaled = scaler.transform(test_data_reshaped)
            
            if len(original_test_shape) > 2:
                test_scaled = test_scaled.reshape(original_test_shape)
        
        self.scalers['default'] = scaler
        logger.info(f"Features scaled using {method} scaler")
        
        return train_scaled, test_scaled, scaler
    
    def inverse_scale(self, scaled_data: np.ndarray, scaler_name: str = 'default') -> np.ndarray:
        """Inverse scale the data"""
        if scaler_name not in self.scalers:
            logger.error(f"Scaler {scaler_name} not found")
            return scaled_data
        
        scaler = self.scalers[scaler_name]
        return scaler.inverse_transform(scaled_data)
    
    def apply_pca(self, train_data: np.ndarray, test_data: np.ndarray = None,
                  n_components: int = None) -> tuple:
        """Apply PCA for dimensionality reduction"""
        if n_components is None:
            n_components = int(train_data.shape[1] * 0.9)
        
        pca = PCA(n_components=n_components)
        
        # Reshape if needed
        original_shape = train_data.shape
        if len(train_data.shape) > 2:
            train_data_reshaped = train_data.reshape(-1, train_data.shape[-1])
        else:
            train_data_reshaped = train_data
        
        train_transformed = pca.fit_transform(train_data_reshaped)
        
        # Reshape back
        if len(original_shape) > 2:
            new_shape = (original_shape[0], original_shape[1], n_components)
            train_transformed = train_transformed.reshape(new_shape)
        
        test_transformed = None
        if test_data is not None:
            original_test_shape = test_data.shape
            if len(test_data.shape) > 2:
                test_data_reshaped = test_data.reshape(-1, test_data.shape[-1])
            else:
                test_data_reshaped = test_data
            
            test_transformed = pca.transform(test_data_reshaped)
            
            if len(original_test_shape) > 2:
                new_shape = (original_test_shape[0], original_test_shape[1], n_components)
                test_transformed = test_transformed.reshape(new_shape)
        
        logger.info(f"Applied PCA: {n_components} components, "
                   f"explained variance: {pca.explained_variance_ratio_.sum():.4f}")
        
        return train_transformed, test_transformed, pca
    
    def split_data(self, X: np.ndarray, y: np.ndarray, 
                   train_ratio: float = 0.7, val_ratio: float = 0.15) -> dict:
        """Split data into train, validation, and test sets"""
        n = len(X)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        split_data = {
            'X_train': X[:train_end],
            'y_train': y[:train_end],
            'X_val': X[train_end:val_end],
            'y_val': y[train_end:val_end],
            'X_test': X[val_end:],
            'y_test': y[val_end:],
        }
        
        logger.info(f"Data split - Train: {train_end}, Val: {val_end-train_end}, Test: {n-val_end}")
        return split_data


if __name__ == "__main__":
    # Example usage
    from src.data_layer.data_manager import DataManager
    
    dm = DataManager()
    data = dm.fetch_historical_data("AAPL", "2023-01-01", "2024-01-01")
    data = dm.add_technical_indicators(data)
    
    fe = FeatureEngineer()
    features = fe.create_features(data)
    print(f"Created features: {features.columns.tolist()}")
