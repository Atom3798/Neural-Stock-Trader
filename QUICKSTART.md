# Quick Start Guide - NeuralStockTrader

## Installation (5 minutes)

```bash
# 1. Navigate to project directory
cd NeuralStockTrader

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

## Run Your First Backtest (2 minutes)

```bash
# Simple backtest (no model training)
python main.py --mode backtest --symbol AAPL --start-date 2023-01-01 --end-date 2024-01-01
```

**Expected Output:**
```
BACKTEST RESULTS
================
Total Return: 23.45%
Sharpe Ratio: 1.23
Win Rate: 58.2%
Max Drawdown: 12.3%
```

## Python Usage Example

```python
from src.execution_layer.trading_engine import TradingEngine
from src.data_layer.data_manager import DataManager
import torch

# Step 1: Setup
config = {
    'input_size': 50,
    'hidden_size': 128,
    'num_layers': 2,
    'max_position_size': 0.1,
}

engine = TradingEngine(config, initial_capital=100000)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
engine.initialize_models(device=str(device))
engine.initialize_strategies()

# Step 2: Train models (optional, ~5 minutes)
print("Training models...")
engine.train_models("AAPL", "2023-01-01", "2024-01-01", epochs=50)

# Step 3: Run backtest
print("Running backtest...")
metrics = engine.backtest_strategy("AAPL", "2023-01-01", "2024-01-01")

# Step 4: View results
print("\n=== RESULTS ===")
print(f"Total Return: {metrics['total_return_pct']:.2f}%")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"Win Rate: {metrics['win_rate_pct']:.2f}%")
print(f"Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
print(f"Profit Factor: {metrics['profit_factor']:.2f}")
```

## Common Commands

### Backtest with Model Training
```bash
python main.py --mode backtest --symbol TSLA --train --start-date 2023-06-01 --end-date 2024-01-01
```

### Multiple Symbols
```bash
for symbol in AAPL MSFT GOOGL; do
    python main.py --mode backtest --symbol $symbol
done
```

## Modify Configuration

Edit `config/config.yaml`:

```yaml
# Use more aggressive position sizing
risk_management:
  max_position_size: 0.15    # Increased from 0.1
  max_daily_loss: 0.10        # Increased from 0.05

# Adjust neural network
neural_network:
  lstm:
    hidden_size: 256          # Larger network
    num_layers: 3             # More layers

# Change strategy parameters
quant_algorithms:
  mean_reversion:
    window: 30                # Longer window
    threshold: 2.5            # Higher threshold
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'yfinance'"
```bash
pip install --upgrade yfinance
```

### "CUDA out of memory"
- Switch to CPU in main.py: `device = 'cpu'`
- Reduce batch_size in config.yaml

### "No data available"
- Check stock symbol is valid (e.g., AAPL not apple)
- Check dates are within valid range (not future dates)
- Increase history_days in config.yaml

## Performance Tips

### For Faster Results
- Use GRU instead of LSTM (faster training)
- Reduce lookback window (60 → 30 periods)
- Reduce epochs (100 → 50)

### For Better Results
- Train on longer history (365 → 730 days)
- Use ensemble of multiple models
- Use walk-forward analysis for validation
- Test multiple symbols

## Next Steps

1. **Experiment with strategies**: Modify parameters in `quant_strategies.py`
2. **Add custom features**: Extend `feature_engineer.py`
3. **Integrate live data**: Use Alpaca API in paper trading
4. **Optimize hyperparameters**: Use Bayesian optimization
5. **Implement new models**: Add Transformer architecture

## Resources

- Documentation: See README.md
- Configuration: config/config.yaml
- Example strategies: src/strategy_layer/quant_strategies.py
- Data management: src/data_layer/data_manager.py

## Getting Help

Check logs for detailed error information:
```bash
tail -f logs/trading_*.log
```

Good luck! 🚀
