"""
Configuration management for Semantic-STGCNN

This module provides configuration management utilities for the Semantic-STGCNN
trajectory prediction framework.
"""

from .config import Config, load_config, save_config
from .defaults import get_default_config

__all__ = ['Config', 'load_config', 'save_config', 'get_default_config']
