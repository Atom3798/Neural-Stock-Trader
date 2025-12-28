"""Package initialization for NeuralStockTrader"""

from src.utils.logger import logger, TradingLogger
from src.utils.constants import *

__version__ = "1.0.0"
__author__ = "NeuralStockTrader Team"

logger.info(f"NeuralStockTrader v{__version__} initialized")
