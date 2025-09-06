"""
Configuration management system for Semantic-STGCNN

This module provides a flexible configuration system that supports loading
from YAML files, command-line arguments, and programmatic configuration.
"""

import yaml
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)


class Config:
    """
    Configuration class for Semantic-STGCNN.
    
    This class provides a flexible way to manage configuration parameters
    for training, evaluation, and inference of the Semantic-STGCNN model.
    """
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize configuration.
        
        Args:
            config_dict: Optional dictionary with configuration parameters
        """
        self._config = config_dict or {}
        self._frozen = False
    
    def __getattr__(self, name: str) -> Any:
        """Get configuration parameter."""
        if name.startswith('_'):
            return super().__getattribute__(name)
        
        if name in self._config:
            value = self._config[name]
            if isinstance(value, dict):
                return Config(value)
            return value
        
        raise AttributeError(f"Configuration parameter '{name}' not found")
    
    def __setattr__(self, name: str, value: Any) -> None:
        """Set configuration parameter."""
        if name.startswith('_'):
            super().__setattr__(name, value)
            return
        
        if hasattr(self, '_frozen') and self._frozen:
            raise AttributeError("Configuration is frozen and cannot be modified")
        
        self._config[name] = value
    
    def __contains__(self, name: str) -> bool:
        """Check if configuration parameter exists."""
        return name in self._config
    
    def get(self, name: str, default: Any = None) -> Any:
        """Get configuration parameter with default value."""
        try:
            return getattr(self, name)
        except AttributeError:
            return default
    
    def update(self, other: Union[Dict[str, Any], 'Config']) -> None:
        """Update configuration with another config or dictionary."""
        if isinstance(other, Config):
            other = other._config
        
        self._config.update(other)
    
    def freeze(self) -> None:
        """Freeze configuration to prevent modifications."""
        self._frozen = True
    
    def unfreeze(self) -> None:
        """Unfreeze configuration to allow modifications."""
        self._frozen = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self._config.copy()
    
    def to_yaml(self) -> str:
        """Convert configuration to YAML string."""
        return yaml.dump(self._config, default_flow_style=False, indent=2)
    
    def to_json(self) -> str:
        """Convert configuration to JSON string."""
        return json.dumps(self._config, indent=2)
    
    def save(self, path: Union[str, Path]) -> None:
        """Save configuration to file."""
        path = Path(path)
        
        if path.suffix.lower() == '.yaml' or path.suffix.lower() == '.yml':
            with open(path, 'w') as f:
                yaml.dump(self._config, f, default_flow_style=False, indent=2)
        elif path.suffix.lower() == '.json':
            with open(path, 'w') as f:
                json.dump(self._config, f, indent=2)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        
        logger.info(f"Configuration saved to {path}")
    
    def __repr__(self) -> str:
        """String representation of configuration."""
        return f"Config({self._config})"


def load_config(path: Union[str, Path]) -> Config:
    """
    Load configuration from file.
    
    Args:
        path: Path to configuration file (YAML or JSON)
        
    Returns:
        Config object
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    
    if path.suffix.lower() in ['.yaml', '.yml']:
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
    elif path.suffix.lower() == '.json':
        with open(path, 'r') as f:
            config_dict = json.load(f)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")
    
    logger.info(f"Configuration loaded from {path}")
    return Config(config_dict)


def save_config(config: Config, path: Union[str, Path]) -> None:
    """
    Save configuration to file.
    
    Args:
        config: Configuration object
        path: Path to save configuration file
    """
    config.save(path)


def merge_configs(*configs: Config) -> Config:
    """
    Merge multiple configurations.
    
    Args:
        *configs: Configuration objects to merge
        
    Returns:
        Merged configuration
    """
    merged_dict = {}
    
    for config in configs:
        merged_dict.update(config.to_dict())
    
    return Config(merged_dict)


def create_config_from_args(args: argparse.Namespace) -> Config:
    """
    Create configuration from command-line arguments.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Configuration object
    """
    config_dict = vars(args)
    return Config(config_dict)


def add_config_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add common configuration arguments to argument parser.
    
    Args:
        parser: Argument parser
        
    Returns:
        Updated argument parser
    """
    # Model parameters
    model_group = parser.add_argument_group('Model Parameters')
    model_group.add_argument('--n_stgcnn', type=int, default=2,
                           help='Number of ST-GCN layers')
    model_group.add_argument('--n_txpcnn', type=int, default=5,
                           help='Number of temporal prediction CNN layers')
    model_group.add_argument('--input_feat', type=int, default=21,
                           help='Number of input features per node')
    model_group.add_argument('--output_feat', type=int, default=2,
                           help='Number of output features per node')
    model_group.add_argument('--seq_len', type=int, default=8,
                           help='Length of input sequence')
    model_group.add_argument('--pred_seq_len', type=int, default=12,
                           help='Length of prediction sequence')
    model_group.add_argument('--kernel_size', type=int, default=3,
                           help='Temporal kernel size')
    
    # Training parameters
    train_group = parser.add_argument_group('Training Parameters')
    train_group.add_argument('--batch_size', type=int, default=32,
                           help='Batch size for training')
    train_group.add_argument('--learning_rate', type=float, default=0.001,
                           help='Learning rate')
    train_group.add_argument('--num_epochs', type=int, default=100,
                           help='Number of training epochs')
    train_group.add_argument('--weight_decay', type=float, default=1e-4,
                           help='Weight decay for regularization')
    train_group.add_argument('--grad_clip', type=float, default=1.0,
                           help='Gradient clipping threshold')
    
    # Dataset parameters
    data_group = parser.add_argument_group('Dataset Parameters')
    data_group.add_argument('--data_dir', type=str, required=True,
                          help='Directory containing dataset files')
    data_group.add_argument('--obs_len', type=int, default=8,
                          help='Number of observed time steps')
    data_group.add_argument('--pred_len', type=int, default=12,
                          help='Number of predicted time steps')
    data_group.add_argument('--skip', type=int, default=1,
                          help='Number of frames to skip')
    data_group.add_argument('--threshold', type=float, default=0.002,
                          help='Threshold for non-linear trajectory detection')
    data_group.add_argument('--min_ped', type=int, default=1,
                          help='Minimum number of pedestrians per sequence')
    data_group.add_argument('--delim', type=str, default='tab',
                          help='Delimiter in dataset files')
    
    # General parameters
    general_group = parser.add_argument_group('General Parameters')
    general_group.add_argument('--config', type=str,
                             help='Path to configuration file')
    general_group.add_argument('--output_dir', type=str, default='./outputs',
                             help='Output directory for results')
    general_group.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                             help='Directory for saving checkpoints')
    general_group.add_argument('--log_level', type=str, default='INFO',
                             choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                             help='Logging level')
    general_group.add_argument('--seed', type=int, default=42,
                             help='Random seed for reproducibility')
    general_group.add_argument('--device', type=str, default='auto',
                             help='Device to use (cpu, cuda, auto)')
    general_group.add_argument('--num_workers', type=int, default=4,
                             help='Number of data loader workers')
    
    return parser


def validate_config(config: Config) -> None:
    """
    Validate configuration parameters.
    
    Args:
        config: Configuration to validate
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Validate model parameters
    if config.get('n_stgcnn', 1) < 1:
        raise ValueError("n_stgcnn must be at least 1")
    
    if config.get('n_txpcnn', 1) < 1:
        raise ValueError("n_txpcnn must be at least 1")
    
    if config.get('input_feat', 2) < 2:
        raise ValueError("input_feat must be at least 2")
    
    if config.get('output_feat', 2) < 2:
        raise ValueError("output_feat must be at least 2")
    
    if config.get('seq_len', 8) < 1:
        raise ValueError("seq_len must be at least 1")
    
    if config.get('pred_seq_len', 12) < 1:
        raise ValueError("pred_seq_len must be at least 1")
    
    # Validate training parameters
    if config.get('batch_size', 32) < 1:
        raise ValueError("batch_size must be at least 1")
    
    if config.get('learning_rate', 0.001) <= 0:
        raise ValueError("learning_rate must be positive")
    
    if config.get('num_epochs', 100) < 1:
        raise ValueError("num_epochs must be at least 1")
    
    logger.info("Configuration validation passed")
