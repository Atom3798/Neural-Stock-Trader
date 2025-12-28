"""
NeuralStockTrader Package Initialization
Advanced Neural Network Stock Trading System
"""

__title__ = "NeuralStockTrader"
__version__ = "1.0.0"
__author__ = "NeuralStockTrader Team"
__description__ = "Advanced neural network-based stock trading system with game theory and quantitative algorithms"
__url__ = "https://github.com/yourusername/NeuralStockTrader"
__license__ = "MIT"

# Package metadata
SYSTEM_NAME = "NeuralStockTrader"
SYSTEM_VERSION = "1.0.0"
SYSTEM_STATUS = "Production Ready (Backtesting Phase)"
RELEASE_DATE = "2024-12-27"

# Import main components
try:
    from src.utils.logger import logger, TradingLogger
    from src.utils.constants import *
    from src.execution_layer.trading_engine import TradingEngine
    from src.data_layer.data_manager import DataManager
    from src.data_layer.feature_engineer import FeatureEngineer
    from src.model_layer.neural_networks import LSTMModel, GRUModel, EnsembleModel
    from src.strategy_layer.quant_strategies import (
        MeanReversionStrategy, MomentumStrategy, StatisticalArbitrageStrategy,
        MarketMakingStrategy, PortfolioOptimizationStrategy, StrategyEnsemble
    )
    from src.risk_management.risk_manager import (
        RiskManager, KellyCriterionSizer, RiskParitySizer, StopLoss, TakeProfit,
        CorrelationAnalyzer, VaR, CircuitBreaker
    )
    from src.backtesting.backtest_engine import BacktestEngine, WalkForwardAnalysis
    from src.utils.config_manager import ConfigManager
    from src.utils.metrics import MetricsCalculator, PerformanceReporter
    
    logger.info(f"✅ {SYSTEM_NAME} v{SYSTEM_VERSION} initialized successfully")
    
except ImportError as e:
    print(f"⚠️  Warning: Could not import all modules: {e}")
    print("Please ensure all dependencies are installed: pip install -r requirements.txt")

# Package exports
__all__ = [
    # Main Engine
    'TradingEngine',
    
    # Data Layer
    'DataManager',
    'FeatureEngineer',
    
    # Model Layer
    'LSTMModel',
    'GRUModel',
    'EnsembleModel',
    
    # Strategies
    'MeanReversionStrategy',
    'MomentumStrategy',
    'StatisticalArbitrageStrategy',
    'MarketMakingStrategy',
    'PortfolioOptimizationStrategy',
    'StrategyEnsemble',
    
    # Risk Management
    'RiskManager',
    'KellyCriterionSizer',
    'RiskParitySizer',
    'StopLoss',
    'TakeProfit',
    'CorrelationAnalyzer',
    'VaR',
    'CircuitBreaker',
    
    # Backtesting
    'BacktestEngine',
    'WalkForwardAnalysis',
    
    # Utilities
    'ConfigManager',
    'MetricsCalculator',
    'PerformanceReporter',
    'logger',
    'TradingLogger',
]

# Quick Start Guide
QUICK_START = """
🚀 Quick Start Guide
====================

1. Install Dependencies:
   pip install -r requirements.txt

2. Run Your First Backtest:
   python main.py --mode backtest --symbol AAPL

3. View Examples:
   python examples.py

4. Read Documentation:
   - QUICKSTART.md - Setup (5 min)
   - README.md - Full guide (30 min)
   - API_REFERENCE.md - API details

5. Customize:
   - Edit config/config.yaml
   - Modify src/strategy_layer/quant_strategies.py
   - Run backtest again

📚 Documentation Files:
- INDEX.md - Navigation guide
- QUICKSTART.md - Quick setup
- README.md - Full documentation
- API_REFERENCE.md - API reference
- ROADMAP.md - Future development
- DELIVERY_SUMMARY.md - What's included

🎯 Common Commands:
python main.py --mode backtest --symbol AAPL                    # Backtest AAPL
python main.py --mode backtest --symbol AAPL --train             # Train & backtest
python examples.py                                                # Run all examples
python examples.py 1                                              # Run example 1

💡 Tips:
- Check logs/trading_*.log for detailed output
- Modify config/config.yaml to adjust parameters
- Use walk-forward analysis for robust validation
- Always test thoroughly before live trading

Happy Trading! 📈
"""

def print_startup_message():
    """Print startup message with version info"""
    print(f"""
╔═══════════════════════════════════════════════════╗
║     NeuralStockTrader v{SYSTEM_VERSION}               
║     Advanced Stock Trading System                
║     Status: {SYSTEM_STATUS}
╚═══════════════════════════════════════════════════╝

🎯 Quick Start:
   python main.py --mode backtest --symbol AAPL

📚 Documentation:
   - Open INDEX.md for navigation
   - Open QUICKSTART.md for setup
   - Open README.md for full guide

✨ Features:
   ✅ Neural Networks (LSTM, GRU, Ensemble)
   ✅ 5 Quantitative Strategies
   ✅ Risk Management & Position Sizing
   ✅ Comprehensive Backtesting
   ✅ Performance Metrics & Analysis

🚀 Ready to Trade!
    """)

# Auto-print startup message if run directly
if __name__ == "__main__":
    print_startup_message()
    print(QUICK_START)
