"""
Configuration management utilities
"""

import yaml
import json
from pathlib import Path
from src.utils.logger import logger


class ConfigManager:
    """Manage configuration loading and validation"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self.config = self.load_config()
    
    def load_config(self) -> dict:
        """Load configuration from YAML file"""
        try:
            path = Path(self.config_path)
            if not path.exists():
                logger.warning(f"Config file not found: {self.config_path}")
                return {}
            
            with open(path, 'r') as f:
                config = yaml.safe_load(f)
            
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            return {}
    
    def get(self, key: str, default=None):
        """Get configuration value by dot notation"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        
        return value
    
    def set(self, key: str, value):
        """Set configuration value by dot notation"""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
        logger.info(f"Configuration updated: {key} = {value}")
    
    def save_config(self, path: str = None):
        """Save configuration to file"""
        path = path or self.config_path
        try:
            with open(path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            logger.info(f"Configuration saved to {path}")
        except Exception as e:
            logger.error(f"Error saving config: {str(e)}")
    
    def to_dict(self) -> dict:
        """Export configuration as dictionary"""
        return self.config.copy()
    
    def validate(self) -> bool:
        """Validate configuration"""
        required_keys = [
            'system.name',
            'data.symbols',
            'neural_network.architecture',
            'risk_management.max_position_size',
            'backtesting.initial_capital'
        ]
        
        valid = True
        for key in required_keys:
            if self.get(key) is None:
                logger.warning(f"Missing required config: {key}")
                valid = False
        
        return valid


if __name__ == "__main__":
    config_manager = ConfigManager()
    
    # Test loading
    print(f"Initial capital: {config_manager.get('backtesting.initial_capital')}")
    
    # Test setting
    config_manager.set('risk_management.max_position_size', 0.15)
    
    # Test validation
    is_valid = config_manager.validate()
    print(f"Config valid: {is_valid}")
