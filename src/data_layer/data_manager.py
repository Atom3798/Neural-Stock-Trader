"""
Data manager for fetching and managing market data
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from src.utils.logger import logger


class DataSource(ABC):
    """Abstract base class for data sources"""
    
    @abstractmethod
    def fetch_data(self, symbol: str, start_date: str, end_date: str, 
                   timeframe: str = "1d") -> pd.DataFrame:
        """Fetch OHLCV data"""
        pass
    
    @abstractmethod
    def fetch_realtime_data(self, symbol: str) -> dict:
        """Fetch real-time data for a symbol"""
        pass


class YFinanceDataSource(DataSource):
    """Yahoo Finance data source"""
    
    def __init__(self):
        try:
            import yfinance as yf
            self.yf = yf
        except ImportError:
            logger.error("yfinance not installed. Install with: pip install yfinance")
            raise
    
    def fetch_data(self, symbol: str, start_date: str, end_date: str, 
                   timeframe: str = "1d") -> pd.DataFrame:
        """Fetch historical data from Yahoo Finance"""
        try:
            data = self.yf.download(symbol, start=start_date, end=end_date, 
                                   interval=timeframe, progress=False)
            data.columns = ['open', 'high', 'low', 'close', 'volume']
            data.index.name = 'date'
            logger.info(f"Fetched {len(data)} records for {symbol} from {start_date} to {end_date}")
            return data
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            raise
    
    def fetch_realtime_data(self, symbol: str) -> dict:
        """Fetch real-time data"""
        try:
            ticker = self.yf.Ticker(symbol)
            info = ticker.info
            return {
                'symbol': symbol,
                'price': info.get('currentPrice', 0),
                'bid': info.get('bid', 0),
                'ask': info.get('ask', 0),
                'volume': info.get('volume', 0),
                'timestamp': datetime.now(),
            }
        except Exception as e:
            logger.error(f"Error fetching real-time data for {symbol}: {str(e)}")
            return {}


class DataManager:
    """Manages market data fetching and preprocessing"""
    
    def __init__(self, data_source: DataSource = None):
        """Initialize with a data source"""
        self.data_source = data_source or YFinanceDataSource()
        self.data_cache = {}
    
    def fetch_historical_data(self, symbol: str, start_date: str, end_date: str,
                             timeframe: str = "1d") -> pd.DataFrame:
        """Fetch historical OHLCV data"""
        cache_key = f"{symbol}_{start_date}_{end_date}_{timeframe}"
        
        if cache_key in self.data_cache:
            logger.debug(f"Retrieved {symbol} from cache")
            return self.data_cache[cache_key]
        
        data = self.data_source.fetch_data(symbol, start_date, end_date, timeframe)
        self.data_cache[cache_key] = data
        return data
    
    def fetch_multiple_symbols(self, symbols: list, start_date: str, end_date: str,
                              timeframe: str = "1d") -> dict:
        """Fetch data for multiple symbols"""
        data = {}
        for symbol in symbols:
            try:
                data[symbol] = self.fetch_historical_data(symbol, start_date, end_date, timeframe)
            except Exception as e:
                logger.error(f"Failed to fetch data for {symbol}: {str(e)}")
        return data
    
    def get_realtime_data(self, symbol: str) -> dict:
        """Get real-time data for a symbol"""
        return self.data_source.fetch_realtime_data(symbol)
    
    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add common technical indicators to the data"""
        try:
            # Simple Moving Averages
            data['sma_20'] = data['close'].rolling(window=20).mean()
            data['sma_50'] = data['close'].rolling(window=50).mean()
            data['sma_200'] = data['close'].rolling(window=200).mean()
            
            # Exponential Moving Averages
            data['ema_12'] = data['close'].ewm(span=12).mean()
            data['ema_26'] = data['close'].ewm(span=26).mean()
            
            # RSI (Relative Strength Index)
            data['rsi'] = self._calculate_rsi(data['close'])
            
            # MACD (Moving Average Convergence Divergence)
            data['macd'], data['macd_signal'], data['macd_hist'] = self._calculate_macd(data['close'])
            
            # Bollinger Bands
            data['bb_middle'], data['bb_upper'], data['bb_lower'] = self._calculate_bollinger_bands(data['close'])
            
            # ATR (Average True Range)
            data['atr'] = self._calculate_atr(data)
            
            # Volume-based indicators
            data['obv'] = self._calculate_obv(data)
            data['vpt'] = self._calculate_vpt(data)
            
            # Log returns
            data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
            data['returns'] = data['close'].pct_change()
            
            # Volatility
            data['volatility'] = data['returns'].rolling(window=20).std()
            
            logger.info(f"Added technical indicators to data")
            return data
        except Exception as e:
            logger.error(f"Error adding technical indicators: {str(e)}")
            raise
    
    @staticmethod
    def _calculate_rsi(prices, period=14):
        """Calculate Relative Strength Index"""
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        rs = up / down if down != 0 else 0
        rsi = np.zeros_like(prices)
        rsi[:period] = 100. - 100. / (1. + rs)
        
        for i in range(period, len(prices)):
            delta = deltas[i-1]
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta
            
            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period
            rs = up / down if down != 0 else 0
            rsi[i] = 100. - 100. / (1. + rs)
        
        return rsi
    
    @staticmethod
    def _calculate_macd(prices, fast=12, slow=26, signal=9):
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_hist = macd - macd_signal
        return macd, macd_signal, macd_hist
    
    @staticmethod
    def _calculate_bollinger_bands(prices, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return sma, upper, lower
    
    @staticmethod
    def _calculate_atr(data, period=14):
        """Calculate Average True Range"""
        high = data['high']
        low = data['low']
        close = data['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr
    
    @staticmethod
    def _calculate_obv(data):
        """Calculate On Balance Volume"""
        obv = [0]
        for i in range(1, len(data)):
            if data['close'].iloc[i] > data['close'].iloc[i-1]:
                obv.append(obv[-1] + data['volume'].iloc[i])
            elif data['close'].iloc[i] < data['close'].iloc[i-1]:
                obv.append(obv[-1] - data['volume'].iloc[i])
            else:
                obv.append(obv[-1])
        return obv
    
    @staticmethod
    def _calculate_vpt(data):
        """Calculate Volume Price Trend"""
        roc = data['close'].pct_change()
        vpt = (roc * data['volume']).cumsum()
        return vpt
    
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare data"""
        # Remove duplicates
        data = data[~data.index.duplicated(keep='first')]
        
        # Handle missing values
        data = data.dropna()
        
        # Ensure proper data types
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        
        logger.info(f"Data cleaned: {len(data)} valid records")
        return data


if __name__ == "__main__":
    # Example usage
    dm = DataManager()
    data = dm.fetch_historical_data("AAPL", "2023-01-01", "2024-01-01")
    data = dm.add_technical_indicators(data)
    print(data.head())
