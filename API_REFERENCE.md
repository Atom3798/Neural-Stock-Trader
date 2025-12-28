# NeuralStockTrader - API Reference

## Core Classes & Methods

### TradingEngine

Main orchestrator class for trading operations.

```python
from src.execution_layer.trading_engine import TradingEngine

# Initialize
engine = TradingEngine(config, initial_capital=100000)

# Methods
engine.initialize_models(device='cpu')          # Setup neural networks
engine.initialize_strategies()                  # Setup trading strategies
engine.train_models(symbol, start_date, end_date, epochs=100)  # Train models
engine.backtest_strategy(symbol, start_date, end_date)         # Run backtest
engine.generate_signals(data, use_ensemble=True)               # Generate signals
engine.execute_trade(symbol, signal, price, date)              # Execute trade
engine.update_portfolio(current_prices)                        # Update positions
engine.get_portfolio_summary()                                 # Get summary
engine.save_models(model_dir)                                  # Save trained models
```

### DataManager

Handles market data fetching and preprocessing.

```python
from src.data_layer.data_manager import DataManager

dm = DataManager()

# Fetch data
data = dm.fetch_historical_data(symbol, start_date, end_date, timeframe='1d')

# Multiple symbols
data_dict = dm.fetch_multiple_symbols(symbols, start_date, end_date)

# Real-time data
rt_data = dm.get_realtime_data(symbol)

# Add indicators
data = dm.add_technical_indicators(data)

# Clean data
data = dm.clean_data(data)
```

### FeatureEngineer

Feature creation and data preparation.

```python
from src.data_layer.feature_engineer import FeatureEngineer

fe = FeatureEngineer(lookback=60)

# Create features
features = fe.create_features(data)

# Create sequences
X, y = fe.create_sequences(data_array)

# Advanced sequences (multi-step)
X, y = fe.create_advanced_sequences(data_array, forecast_horizon=5)

# Scale features
X_train_scaled, X_test_scaled, scaler = fe.scale_features(X_train, X_test)

# PCA
X_train_pca, X_test_pca, pca = fe.apply_pca(X_train, X_test, n_components=30)

# Split data
split_data = fe.split_data(X, y, train_ratio=0.7, val_ratio=0.15)
```

### Neural Network Models

```python
from src.model_layer.neural_networks import LSTMModel, GRUModel, EnsembleModel
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# LSTM Model
lstm = LSTMModel(input_size=50, hidden_size=128, num_layers=2, device=device)
lstm.train(X_train, y_train, X_val, y_val, epochs=100, batch_size=32)
predictions = lstm.predict(X_test)

# GRU Model
gru = GRUModel(input_size=50, hidden_size=64, num_layers=2, device=device)
gru.train(X_train, y_train, X_val, y_val)
predictions = gru.predict(X_test)

# Ensemble
ensemble = EnsembleModel([lstm, gru], weights=[0.6, 0.4])
predictions = ensemble.predict(X_test)

# Save/Load
lstm.save('models/lstm.pt')
lstm.load('models/lstm.pt')
```

### Trading Strategies

```python
from src.strategy_layer.quant_strategies import (
    MeanReversionStrategy, MomentumStrategy, 
    StatisticalArbitrageStrategy, StrategyEnsemble
)

# Mean Reversion
mr_strategy = MeanReversionStrategy(window=20, threshold=2.0)
signals = mr_strategy.generate_signals(data)

# Momentum
mom_strategy = MomentumStrategy(fast_period=12, slow_period=26)
signals = mom_strategy.generate_signals(data)

# Statistical Arbitrage
sa_strategy = StatisticalArbitrageStrategy(lookback=60)
signals = sa_strategy.generate_signals(data)

# Get/Set parameters
params = mom_strategy.get_parameters()
mom_strategy.set_parameters({'fast_period': 10, 'slow_period': 30})

# Ensemble
strategies = [mr_strategy, mom_strategy, sa_strategy]
ensemble = StrategyEnsemble(strategies, weights=[0.4, 0.35, 0.25])
signals = ensemble.generate_signals(data)
```

### Risk Management

```python
from src.risk_management.risk_manager import (
    RiskManager, KellyCriterionSizer, RiskParitySizer,
    StopLoss, TakeProfit, CorrelationAnalyzer, VaR, CircuitBreaker
)

# Risk Manager
rm = RiskManager(initial_capital=100000, max_position_size=0.1)
rm.update_capital(pnl=500)
metrics = rm.get_risk_metrics()
rm.check_position_limits(symbol, quantity, side='buy')
rm.check_daily_loss_limit()
rm.check_drawdown_limit()

# Position Sizing
kelly_sizer = KellyCriterionSizer(kelly_fraction=0.25)
size = kelly_sizer.calculate_size(capital=100000, win_rate=0.55, 
                                  avg_win=100, avg_loss=90)

risk_parity = RiskParitySizer(target_risk=0.02)
size = risk_parity.calculate_size(capital=100000, volatility=0.015)

# Stop Loss
sl = StopLoss(stop_type='trailing', stop_pct=0.02)
sl.set_entry(100)
sl.update(101)
if sl.is_triggered(99):
    # Execute stop loss

# Take Profit
tp = TakeProfit(take_profit_pct=0.05)
tp.set_entry(100)
if tp.is_triggered(105):
    # Execute take profit

# VaR
var_95 = VaR.calculate_var(returns, confidence_level=0.95)
cvar_95 = VaR.calculate_cvar(returns, confidence_level=0.95)

# Circuit Breaker
cb = CircuitBreaker(daily_loss_trigger=0.03)
cb.update_losses(pnl=-500)
if cb.check_circuit(starting_capital=100000):
    # Halt trading
```

### Backtesting

```python
from src.backtesting.backtest_engine import BacktestEngine, WalkForwardAnalysis

# Basic backtest
bt = BacktestEngine(initial_capital=100000, commission=0.001, slippage=0.0005)
metrics = bt.run_backtest(data, signals, symbol='AAPL')

# Export results
bt.export_trades('results/trades.csv')
bt.export_equity_curve('results/equity.csv')

# Walk-forward analysis
wf = WalkForwardAnalysis(train_period=252, test_period=63)
results = wf.run_walk_forward(data, strategy_function)
```

### Configuration

```python
from src.utils.config_manager import ConfigManager

# Load config
config = ConfigManager('config/config.yaml')

# Get values
initial_capital = config.get('backtesting.initial_capital')
symbols = config.get('data.symbols')

# Set values
config.set('risk_management.max_position_size', 0.15)

# Validate
if config.validate():
    print("Configuration valid")

# Save
config.save_config()

# Export
config_dict = config.to_dict()
```

### Metrics Calculation

```python
from src.utils.metrics import MetricsCalculator, PerformanceReporter

# Calculate metrics
calc = MetricsCalculator()

sharpe = calc.calculate_sharpe_ratio(returns)
sortino = calc.calculate_sortino_ratio(returns)
calmar = calc.calculate_calmar_ratio(returns)
volatility = calc.calculate_volatility(returns)
var = calc.calculate_var(returns, confidence_level=0.95)

# Performance report
reporter = PerformanceReporter(trades_df, equity_curve)
report = reporter.generate_report()
reporter.print_report()
```

### Logging

```python
from src.utils.logger import logger

# Use throughout code
logger.debug("Debug message")
logger.info("Info message")
logger.warning("Warning message")
logger.error("Error message")
logger.critical("Critical message")
```

---

## Configuration Options

### Data Configuration
```yaml
data:
  symbols: ["AAPL", "MSFT"]
  timeframe: "1h"
  history_days: 365
```

### Neural Network Configuration
```yaml
neural_network:
  architecture: "lstm"
  lstm:
    input_size: 50
    hidden_size: 128
    num_layers: 2
  training:
    epochs: 100
    batch_size: 32
    learning_rate: 0.001
```

### Risk Management Configuration
```yaml
risk_management:
  max_position_size: 0.1
  max_daily_loss: 0.05
  max_drawdown: 0.20
  position_sizing: "kelly"
```

### Backtesting Configuration
```yaml
backtesting:
  initial_capital: 100000
  commission: 0.001
  slippage: 0.0005
```

---

## Common Patterns

### Pattern 1: Basic Backtest
```python
from src.execution_layer.trading_engine import TradingEngine

config = {'input_size': 50, 'hidden_size': 128}
engine = TradingEngine(config, initial_capital=100000)
engine.initialize_strategies()

metrics = engine.backtest_strategy("AAPL", "2023-01-01", "2024-01-01")
```

### Pattern 2: Custom Strategy
```python
from src.strategy_layer.quant_strategies import TradingStrategy

class MyStrategy(TradingStrategy):
    def generate_signals(self, data):
        # Your signal logic
        signals = pd.Series(0, index=data.index)
        # ... populate signals ...
        return signals
    
    def get_parameters(self):
        return {'param1': self.param1}
    
    def set_parameters(self, params):
        if 'param1' in params:
            self.param1 = params['param1']
```

### Pattern 3: Model Training
```python
from src.model_layer.neural_networks import LSTMModel
from src.data_layer.feature_engineer import FeatureEngineer

# Prepare data
fe = FeatureEngineer()
split_data, scaler, _ = engine.prepare_training_data("AAPL", start, end)

# Train model
model = LSTMModel(input_size=50)
model.train(split_data['X_train'], split_data['y_train'],
           split_data['X_val'], split_data['y_val'])

# Save
model.save('models/lstm.pt')
```

### Pattern 4: Risk Management
```python
from src.risk_management.risk_manager import RiskManager, KellyCriterionSizer

rm = RiskManager(initial_capital=100000)
kelly = KellyCriterionSizer()

position_size = kelly.calculate_size(100000, 0.55, 100, 90)

if rm.check_position_limits(symbol, position_size, 'buy'):
    execute_trade(symbol, position_size)
```

---

## Error Handling

```python
try:
    engine = TradingEngine(config)
    metrics = engine.backtest_strategy("AAPL", "2023-01-01", "2024-01-01")
except ValueError as e:
    logger.error(f"Invalid configuration: {e}")
except RuntimeError as e:
    logger.error(f"Execution error: {e}")
except Exception as e:
    logger.error(f"Unexpected error: {e}")
```

---

## Performance Tips

1. **Data Fetching**: Cache data between runs
2. **Feature Engineering**: Pre-compute features offline
3. **Model Training**: Use GPU with `device='cuda'`
4. **Backtesting**: Use simpler models for fast iteration
5. **Walk-Forward**: Parallelize window testing

---

## Version

Current Version: **1.0.0**  
Last Updated: December 2024
