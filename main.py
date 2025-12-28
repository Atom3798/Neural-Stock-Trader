"""
Main entry point for NeuralStockTrader
"""

import yaml
import torch
import argparse
from datetime import datetime, timedelta
from src.execution_layer.trading_engine import TradingEngine
from src.utils.logger import logger


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Config file not found: {config_path}")
        raise


def main():
    """Main trading application"""
    
    parser = argparse.ArgumentParser(description='NeuralStockTrader - Advanced Stock Trading System')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--mode', type=str, choices=['backtest', 'paper', 'live'],
                       default='backtest', help='Trading mode')
    parser.add_argument('--symbol', type=str, default='AAPL',
                       help='Stock symbol to trade')
    parser.add_argument('--start-date', type=str,
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str,
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--train', action='store_true',
                       help='Train models before trading')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize trading engine
    logger.info("="*60)
    logger.info("NeuralStockTrader - Advanced Stock Trading System")
    logger.info("="*60)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Symbol: {args.symbol}")
    
    trading_engine = TradingEngine(config, initial_capital=config['backtesting']['initial_capital'])
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Initialize models and strategies
    trading_engine.initialize_models(device=str(device))
    trading_engine.initialize_strategies()
    
    # Determine date range
    if not args.end_date:
        args.end_date = datetime.now().strftime("%Y-%m-%d")
    if not args.start_date:
        end = datetime.strptime(args.end_date, "%Y-%m-%d")
        start = end - timedelta(days=365)
        args.start_date = start.strftime("%Y-%m-%d")
    
    logger.info(f"Date range: {args.start_date} to {args.end_date}")
    
    try:
        if args.mode == 'backtest':
            logger.info("Starting backtest...")
            
            if args.train:
                logger.info("Training models...")
                trading_engine.train_models(
                    args.symbol,
                    args.start_date,
                    args.end_date,
                    epochs=config['neural_network']['training']['epochs']
                )
            
            # Run backtest
            metrics = trading_engine.backtest_strategy(
                args.symbol,
                args.start_date,
                args.end_date
            )
            
            # Print results
            logger.info("="*60)
            logger.info("BACKTEST RESULTS")
            logger.info("="*60)
            
            for key, value in metrics.items():
                if isinstance(value, float):
                    logger.info(f"{key}: {value:.4f}")
                else:
                    logger.info(f"{key}: {value}")
            
            logger.info("="*60)
        
        elif args.mode == 'paper':
            logger.info("Paper trading mode not yet implemented")
        
        elif args.mode == 'live':
            logger.warning("LIVE TRADING MODE - Exercise caution!")
            logger.info("Live trading mode not yet implemented")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
