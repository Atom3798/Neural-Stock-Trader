"""
Constants and enumerations for the NeuralStockTrader system
"""

from enum import Enum

# Market Constants
class TimeFrame(Enum):
    """Trading timeframes"""
    ONE_MINUTE = "1m"
    FIVE_MINUTES = "5m"
    FIFTEEN_MINUTES = "15m"
    ONE_HOUR = "1h"
    FOUR_HOURS = "4h"
    ONE_DAY = "1d"
    ONE_WEEK = "1w"
    ONE_MONTH = "1M"


class OrderType(Enum):
    """Types of orders"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"
    VWAP = "vwap"


class OrderSide(Enum):
    """Order side: Buy or Sell"""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order status"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    ACCEPTED = "accepted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class PositionStatus(Enum):
    """Position status"""
    OPEN = "open"
    CLOSED = "closed"
    PARTIALLY_CLOSED = "partially_closed"


class SignalType(Enum):
    """Trading signals"""
    STRONG_BUY = 2
    BUY = 1
    HOLD = 0
    SELL = -1
    STRONG_SELL = -2


# Neural Network Architecture
class NNArchitecture(Enum):
    """Neural network architectures"""
    LSTM = "lstm"
    GRU = "gru"
    TRANSFORMER = "transformer"
    CNN = "cnn"
    ENSEMBLE = "ensemble"


class RLAlgorithm(Enum):
    """Reinforcement learning algorithms"""
    DQN = "dqn"
    DOUBLE_DQN = "double_dqn"
    PPO = "ppo"
    A3C = "a3c"
    TRPO = "trpo"


# Market Models
class MarketModel(Enum):
    """Game theory market models"""
    PERFECT_COMPETITION = "perfect_competition"
    MONOPOLY = "monopoly"
    OLIGOPOLY = "oligopoly"
    MONOPOLISTIC_COMPETITION = "monopolistic_competition"


# Quantitative Methods
class PositionSizing(Enum):
    """Position sizing methods"""
    FIXED = "fixed"
    PROPORTIONAL = "proportional"
    KELLY = "kelly"
    RISK_PARITY = "risk_parity"


class OptimizationMethod(Enum):
    """Portfolio optimization methods"""
    MARKOWITZ = "markowitz"
    BLACK_LITTERMAN = "black_litterman"
    RISK_PARITY = "risk_parity"
    MIN_VARIANCE = "min_variance"


# Technical Indicators
TECHNICAL_INDICATORS = [
    "RSI",
    "MACD",
    "BOLLINGER_BANDS",
    "ATR",
    "ADX",
    "STOCHASTIC",
    "VPT",
    "OBV",
    "EMA",
    "SMA",
    "CCI",
    "MOMENTUM",
    "ROC",
    "TRIX",
]

# Risk Metrics
RISK_METRICS = [
    "SHARPE_RATIO",
    "SORTINO_RATIO",
    "CALMAR_RATIO",
    "MAX_DRAWDOWN",
    "INFORMATION_RATIO",
    "TREYNOR_RATIO",
    "OMEGA_RATIO",
]

# Market Regimes
class MarketRegime(Enum):
    """Market regimes for adaptive strategy selection"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    MEAN_REVERTING = "mean_reverting"
    VOLATILE = "volatile"
    CHOPPY = "choppy"
    LOW_VOLATILITY = "low_volatility"
    HIGH_VOLATILITY = "high_volatility"


# Default Values
DEFAULT_LOOKBACK = 60  # 60 periods for feature extraction
DEFAULT_TRAIN_SPLIT = 0.8
DEFAULT_VALIDATION_SPLIT = 0.1
DEFAULT_TEST_SPLIT = 0.1

MIN_PORTFOLIO_VALUE = 100  # $100 minimum
MAX_POSITION_SIZE = 0.1  # 10% of portfolio per position
DEFAULT_COMMISSION = 0.001  # 0.1%
DEFAULT_SLIPPAGE = 0.0005  # 0.05%

# Thresholds
MIN_SHARPE_RATIO = 0.5
MIN_WIN_RATE = 0.35
MAX_DRAWDOWN_LIMIT = 0.20  # 20% max drawdown
MIN_DATA_POINTS = 100  # Minimum data points for training

# File Paths
DATA_DIR = "data"
MODEL_DIR = "models"
RESULTS_DIR = "results"
LOGS_DIR = "logs"
CONFIG_DIR = "config"
