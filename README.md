# Advanced Neural Network Stock Trading System

A comprehensive, production-ready stock trading system combining neural networks, game theory, and quantitative finance algorithms with reinforcement learning and adaptive strategy selection.

##  Project Overview

This system is designed to:
- **Learn & Improve**: Deep learning models that continuously adapt to market conditions
- **Integrate Game Theory**: Model market participant behavior and strategic interactions
- **Apply Quantitative Algorithms**: Implement proven trading strategies (mean reversion, momentum, arbitrage)
- **Manage Risk**: Sophisticated position sizing, portfolio optimization, and drawdown controls
- **Validate Performance**: Comprehensive backtesting with realistic transaction costs and slippage

##  Architecture

```
NeuralStockTrader/
├── src/
│   ├── data_layer/              # Data management & feature engineering
│   │   ├── data_manager.py      # OHLCV data fetching, technical indicators
│   │   └── feature_engineer.py  # Feature extraction, scaling, sequencing
│   │
│   ├── model_layer/             # Neural network models
│   │   └── neural_networks.py   # LSTM, GRU, Transformer, Ensemble models
│   │
│   ├── strategy_layer/          # Trading strategies
│   │   └── quant_strategies.py  # Mean reversion, momentum, arbitrage, etc.
│   │
│   ├── execution_layer/         # Trade execution
│   │   └── trading_engine.py    # Main orchestrator & position management
│   │
│   ├── risk_management/         # Risk controls
│   │   └── risk_manager.py      # Position sizing, stop-loss, portfolio optimization
│   │
│   ├── backtesting/             # Backtesting engine
│   │   └── backtest_engine.py   # Historical performance evaluation
│   │
│   └── utils/                   # Utilities
│       ├── logger.py            # Centralized logging
│       └── constants.py         # Enums and constants
│
├── config/
│   └── config.yaml              # System configuration
│
├── data/                        # Historical data storage
├── models/                      # Trained model storage
├── logs/                        # Trading logs
├── results/                     # Backtest results
├── notebooks/                   # Jupyter notebooks for analysis
├── tests/                       # Unit tests
├── main.py                      # Entry point
└── requirements.txt             # Dependencies
```

##  Getting Started

### Installation

```bash
# Clone the repository
cd NeuralStockTrader

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

#### Backtest a Strategy

```bash
# Simple backtest
python main.py --mode backtest --symbol AAPL --start-date 2023-01-01 --end-date 2024-01-01

# Backtest with model training
python main.py --mode backtest --symbol AAPL --train
```

#### Python API

```python
from src.execution_layer.trading_engine import TradingEngine
from src.data_layer.data_manager import DataManager

# Initialize
engine = TradingEngine(config, initial_capital=100000)
engine.initialize_models()
engine.initialize_strategies()

# Train models
engine.train_models("AAPL", "2023-01-01", "2024-01-01", epochs=100)

# Backtest
metrics = engine.backtest_strategy("AAPL", "2023-01-01", "2024-01-01")

# View results
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
print(f"Win Rate: {metrics['win_rate_pct']:.2f}%")
```

##  Core Components

### 1. Data Layer

**Data Manager** (`data_layer/data_manager.py`)
- Fetch OHLCV data from multiple sources (yfinance, Alpaca, Polygon)
- Calculate technical indicators (RSI, MACD, Bollinger Bands, ATR, etc.)
- Handle data caching and real-time updates

**Feature Engineer** (`data_layer/feature_engineer.py`)
- Create 50+ engineered features from raw price data
- Sequence creation for RNN models (60-period lookback window)
- Feature scaling (standard, MinMax) and dimensionality reduction (PCA)
- Train/validation/test split with temporal awareness

### 2. Model Layer

**Neural Network Models** (`model_layer/neural_networks.py`)

- **LSTM Model**: 
  - 2-layer LSTM with 128 hidden units
  - Dropout for regularization
  - Early stopping with validation monitoring

- **GRU Model**:
  - Lighter GRU alternative for faster training
  - 2-layer GRU with 64 hidden units

- **Ensemble Model**:
  - Combines multiple models with weighted averaging
  - Reduces overfitting through model diversity

Training features:
- Adam optimizer with learning rate scheduling
- Gradient clipping for stability
- MSE/MAE loss functions
- GPU support (CUDA)

### 3. Strategy Layer

**Quantitative Trading Strategies** (`strategy_layer/quant_strategies.py`)

1. **Mean Reversion Strategy**
   - Trades when price deviates 2σ from 20-period mean
   - Profitable in range-bound markets

2. **Momentum Strategy**
   - 12/26 EMA crossovers with dynamic thresholds
   - Catches trends early in volatile markets

3. **Statistical Arbitrage**
   - Z-score based entry/exit signals
   - Exploits temporary price divergences

4. **Market Making** (Advanced)
   - Optimal bid-ask spread calculation
   - Inventory management

5. **Portfolio Optimization**
   - Markowitz efficient frontier
   - Risk parity weighting

6. **Strategy Ensemble**
   - Combines all strategies with weighted voting
   - Reduces single-strategy risk

### 4. Risk Management

**Risk Manager** (`risk_management/risk_manager.py`)

Position Sizing Methods:
- **Kelly Criterion**: f* = (bp - q) / b with 0.25 fraction
- **Risk Parity**: Target risk % per position
- **Fixed Size**: Manual position sizing
- **Proportional**: % of portfolio

Risk Controls:
- Max position size: 10% of portfolio
- Max daily loss: 5% of portfolio
- Max drawdown: 20% limit
- Circuit breakers for abnormal losses
- Correlation analysis to prevent concentration

Stop Loss & Take Profit:
- Hard stops at fixed levels
- Trailing stops that follow price
- Dynamic adjustments based on volatility

Value at Risk (VaR):
- Historical VaR at 95% confidence
- Conditional VaR (Expected Shortfall)

### 5. Execution Engine

**Trading Engine** (`execution_layer/trading_engine.py`)

Features:
- Unified interface for all trading modes
- Position tracking and P&L calculation
- Portfolio rebalancing
- Trade execution with slippage & commissions
- Real-time portfolio monitoring

### 6. Backtesting Engine

**Backtest Engine** (`backtesting/backtest_engine.py`)

Capabilities:
- Realistic transaction costs (commissions, slippage)
- Trade logging with full P&L
- Equity curve tracking
- Walk-forward analysis for out-of-sample testing

Performance Metrics:
- **Return Metrics**: Total return, annualized return
- **Risk Metrics**: Sharpe ratio, Sortino ratio, Calmar ratio
- **Drawdown Metrics**: Max drawdown, recovery time
- **Trade Metrics**: Win rate, profit factor, average win/loss

##  Configuration

Edit `config/config.yaml` to customize:

```yaml
data:
  symbols: ["AAPL", "MSFT", "GOOGL", "TSLA"]
  timeframe: "1h"
  history_days: 365

neural_network:
  architecture: "lstm"  # lstm, gru, transformer, ensemble
  lstm:
    hidden_size: 128
    num_layers: 2
  training:
    epochs: 100
    batch_size: 32
    learning_rate: 0.001

risk_management:
  max_position_size: 0.1
  max_daily_loss: 0.05
  max_drawdown: 0.20
  position_sizing: "kelly"

backtesting:
  start_date: "2022-01-01"
  end_date: "2024-01-01"
  initial_capital: 100000
  commission: 0.001
  slippage: 0.0005
```

##  Workflow

### Typical Trading Workflow

```
1. Data Preparation
   ├── Fetch OHLCV data
   ├── Calculate technical indicators
   └── Engineer features

2. Model Training (Optional)
   ├── Split data into train/val/test
   ├── Normalize features
   ├── Train LSTM/GRU models
   └── Validate and save models

3. Signal Generation
   ├── Run all strategies
   ├── Ensemble voting
   └── Generate final signals (BUY/SELL/HOLD)

4. Risk Assessment
   ├── Check position limits
   ├── Calculate position size
   ├── Check daily loss limits
   └── Validate correlation limits

5. Trade Execution
   ├── Execute orders
   ├── Track positions
   ├── Update portfolio

6. Performance Monitoring
   ├── Track P&L
   ├── Monitor drawdown
   ├── Log all trades
   └── Generate reports
```

## 📈 Performance Metrics

### Key Ratios

**Sharpe Ratio** = (Return - Risk-Free Rate) / Volatility
- Target: > 1.0 is good, > 2.0 is excellent

**Sortino Ratio** = (Return - Risk-Free Rate) / Downside Volatility
- Penalizes only downside volatility

**Calmar Ratio** = Annual Return / Max Drawdown
- Target: > 1.0

**Profit Factor** = Gross Profit / Gross Loss
- Target: > 1.5 is profitable

**Win Rate** = Winning Trades / Total Trades
- Combined with profit factor matters more than win rate alone

##  Next Steps & Enhancement Ideas

### Phase 1 (Current)
-  Core architecture with LSTM/GRU models
-  Quantitative strategy ensemble
-  Risk management framework
-  Backtesting engine

### Phase 2
- [ ] Reinforcement Learning (DQN, PPO)
- [ ] Game theory multi-agent modeling
- [ ] Advanced sentiment analysis integration
- [ ] Options strategies

### Phase 3
- [ ] Live paper trading connection
- [ ] Real-time market data streaming
- [ ] Multi-asset optimization
- [ ] Advanced meta-learning

### Phase 4
- [ ] Live trading deployment
- [ ] Transaction cost prediction
- [ ] Optimal order execution
- [ ] Market impact modeling

##  Important Notes

1. **Backtesting Disclaimer**: Past performance does not guarantee future results. Models trained on historical data may not generalize to new market regimes.

2. **Risk Management**: Always use appropriate position sizing and risk controls. Never risk more than you can afford to lose.

3. **Market Regime Shifts**: The system should be monitored regularly and adjusted for changing market conditions.

4. **Transaction Costs**: Always include realistic commissions and slippage in backtests.

5. **Walk-Forward Analysis**: Use out-of-sample testing to validate strategy robustness.

##  References

### Game Theory
- Nash Equilibrium for market modeling
- Auction theory for order placement
- Signaling games for information asymmetry

### Quantitative Finance
- Modern Portfolio Theory (Markowitz)
- Black-Litterman model
- Kelly Criterion for position sizing
- Value at Risk (VaR) and CVaR

### Machine Learning
- LSTM & GRU for sequence modeling
- Ensemble methods for robust predictions
- Reinforcement learning for adaptive strategies

### Trading
- Technical analysis indicators
- Momentum and mean reversion
- Statistical arbitrage
- Market microstructure

##  License

This project is for educational and research purposes.

##  Contributing

Contributions welcome! Areas for improvement:
- Additional neural network architectures
- More quantitative strategies
- Reinforcement learning integration
- Real broker API connections
- Performance optimizations

##  Support

For issues, questions, or suggestions, please create an issue in the repository.

---

**Last Updated**: December 2024  
**Version**: 1.0.0  
**Status**: Production-Ready (Backtesting Only)
