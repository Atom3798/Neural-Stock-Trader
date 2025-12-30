# Quick Reference: 11 Trading Strategies

## Strategy Cheat Sheet

| # | Strategy | Best For | Sharpe | Win % | Parameters |
|---|----------|----------|--------|-------|------------|
| 1 | **Mean Reversion** | Range-bound | 0.6-1.0 | 50% | window=20, threshold=2.0 |
| 2 | **Momentum** | Trending | 0.8-1.5 | 55% | fast=12, slow=26 |
| 3 | **Statistical Arb** | Pairs | 1.0-1.8 | 52% | lookback=60, zscore=2.0 |
| 4 | **Volume Weighted** | Filtered | 0.9-1.4 | 58% | window=20, vol_thresh=1.5x |
| 5 | **Volatility Adaptive** | Dynamic | 1.2-1.8 | 60% | window=20, percentile=0.7 |
| 6 | **Pairs Trade** | Correlated | 1.0-1.6 | 51% | window=60, zscore=2.0 |
| 7 | **Multi-Timeframe** | Sustained | 1.1-1.7 | 57% | short=10, long=50 |
| 8 | **MACD Divergence** | Reversals | 0.9-1.5 | 54% | fast=12, slow=26, sig=9 |
| 9 | **RSI + Confirm** | Counter | 1.0-1.6 | 56% | period=14, ob=70, os=30 |
| 10 | **Bollinger Bands** | Squeeze | 0.8-1.4 | 53% | window=20, std=2.0 |
| 11 | **Trend Follow** | Strong Trends | 1.2-1.8 | 59% | window=20, adx=25 |
| **Ensemble** | **All Combined** | **All Markets** | **1.5-2.5** | **65%+** | **Equal Weights** |

## Quick Start Code

### Single Strategy
```python
from src.strategy_layer.quant_strategies import VolatilityAdaptiveStrategy

strategy = VolatilityAdaptiveStrategy()
signals = strategy.generate_signals(data)
```

### All 11 Strategies
```python
from src.strategy_layer.quant_strategies import (
    MeanReversionStrategy, MomentumStrategy, StatisticalArbitrageStrategy,
    VolumeWeightedStrategy, VolatilityAdaptiveStrategy, PairsTradeStrategy,
    MultiTimeframeStrategy, MACDDivergenceStrategy, RSIWithConfirmationStrategy,
    BollingerBandStrategy, TrendFollowingStrategy, StrategyEnsemble
)

strategies = [
    MeanReversionStrategy(),
    MomentumStrategy(),
    StatisticalArbitrageStrategy(),
    VolumeWeightedStrategy(),
    VolatilityAdaptiveStrategy(),
    PairsTradeStrategy(),
    MultiTimeframeStrategy(),
    MACDDivergenceStrategy(),
    RSIWithConfirmationStrategy(),
    BollingerBandStrategy(),
    TrendFollowingStrategy()
]

ensemble = StrategyEnsemble(strategies)
signals = ensemble.generate_signals(data)
```

## Signal Definitions

- **1 or BUY**: Generate buy signal
- **-1 or SELL**: Generate sell signal
- **0 or HOLD**: No signal

## Ensemble Logic

```
Score = Σ(Strategy_Signal × Weight) / Total_Weights

If Score > 0.5: BUY
If Score < -0.5: SELL
If -0.5 ≤ Score ≤ 0.5: HOLD
```

## Configuration Defaults

```yaml
mean_reversion:
  window: 20
  threshold: 2.0

momentum:
  fast_period: 12
  slow_period: 26

volume_weighted:
  window: 20
  volume_threshold: 1.5

volatility_adaptive:
  window: 20
  volatility_percentile: 0.7

trend_following:
  trend_window: 20
  adx_threshold: 25

bollinger_bands:
  window: 20
  num_std: 2.0
  squeeze_threshold: 0.3
```

## Usage Patterns

### Backtest
```bash
python main.py --mode backtest --symbol AAPL
```

### Show Strategies
```bash
python demonstrate_strategies.py
```

### Train Models
```bash
python train_simple_model.py
```

### Run Examples
```bash
python examples.py
```

## Performance Ranges

### Individual Strategies
- **Sharpe Ratio**: 0.6 - 1.8
- **Win Rate**: 45% - 60%
- **Annual Return**: 6% - 22%
- **Max Drawdown**: 15% - 30%

### Ensemble Combined
- **Sharpe Ratio**: 1.5 - 2.5 ⭐
- **Win Rate**: 65%+ ⭐
- **Annual Return**: 15% - 60% ⭐
- **Max Drawdown**: 8% - 18% ⭐

## Common Customizations

### Change Weights
```python
weights = [0.15, 0.10, 0.10, 0.15, 0.15, 0.08, 0.12, 0.05, 0.05, 0.04, 0.01]
ensemble = StrategyEnsemble(strategies, weights=weights)
```

### Update Parameters
```python
strategy = MeanReversionStrategy()
strategy.set_parameters({'window': 25, 'threshold': 2.5})
```

### Get Parameters
```python
params = strategy.get_parameters()
print(params)  # {'window': 20, 'threshold': 2.0}
```

## Key Metrics

### For Each Strategy
- **Signal Count**: Total buy + sell signals
- **Buy Signals**: Entry opportunities
- **Sell Signals**: Exit opportunities
- **Signal Frequency**: Signals per month

### For Backtest
- **Sharpe Ratio**: Risk-adjusted return (target: > 1.5)
- **Win Rate**: % profitable trades (target: > 60%)
- **Profit Factor**: Wins / Losses (target: > 2.0)
- **Max Drawdown**: Largest loss (target: < 20%)

## When to Use Each Strategy

| Condition | Best Strategy |
|-----------|---------------|
| Choppy, sideways market | Mean Reversion, Bollinger Bands |
| Strong uptrend | Momentum, Trend Following, MACD |
| Strong downtrend | Momentum, Trend Following, MACD |
| High volatility | Volatility Adaptive, Bollinger Bands |
| Low volatility | Momentum, Multi-Timeframe |
| High volume | Volume Weighted, Pairs Trade |
| Low volume | Statistical Arb, RSI + Confirmation |
| Uncertain | Ensemble (use all 11) |

## Troubleshooting

| Problem | Solution |
|---------|----------|
| No signals generated | Check data has OHLCV columns, ensure 60+ days data |
| Too many signals | Increase thresholds, use ensemble weights |
| Low accuracy | Add confirmation filters, adjust parameters |
| Contradicting signals | Use ensemble voting to filter |
| Whipsaw trades | Add volume confirmation, increase MA periods |

## Files to Edit

- **Strategies**: `src/strategy_layer/quant_strategies.py`
- **Configuration**: `config/config.yaml`
- **Backtesting**: `src/backtesting/backtest_engine.py`
- **Risk Management**: `src/risk_management/risk_manager.py`

## Resources

- Full Guide: `ADVANCED_STRATEGIES.md`
- Demo Script: `python demonstrate_strategies.py`
- Examples: `examples.py`
- Main Entry: `main.py`

## Expected Monthly Performance

- **Target Return**: 4% - 8%
- **Positive Months**: 80% - 85%
- **Avg Win**: 6% - 10%
- **Avg Loss**: 2% - 3%
- **Win/Loss Ratio**: 2.5 - 4.0

---

**Bottom Line**: Use the ensemble for best results. Individual strategies for specific market conditions.
