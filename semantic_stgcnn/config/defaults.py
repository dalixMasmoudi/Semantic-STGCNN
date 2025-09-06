"""
Default configuration parameters for Semantic-STGCNN

This module defines the default configuration parameters used throughout
the Semantic-STGCNN framework.
"""

from .config import Config


def get_default_config() -> Config:
    """
    Get default configuration for Semantic-STGCNN.
    
    Returns:
        Config object with default parameters
    """
    config_dict = {
        # Model architecture parameters
        'model': {
            'name': 'semantic_stgcnn',
            'n_stgcnn': 2,              # Number of ST-GCN layers
            'n_txpcnn': 5,              # Number of temporal prediction CNN layers
            'input_feat': 21,           # Number of input features (with semantic features)
            'output_feat': 2,           # Number of output features (x, y coordinates)
            'seq_len': 8,               # Length of input sequence
            'pred_seq_len': 12,         # Length of prediction sequence
            'kernel_size': 3,           # Temporal kernel size
            'use_mdn': False,           # Whether to use Mixture Density Networks
            'dropout': 0.1,             # Dropout rate
            'residual': True,           # Whether to use residual connections
        },
        
        # Training parameters
        'training': {
            'batch_size': 32,           # Batch size for training
            'learning_rate': 0.001,     # Initial learning rate
            'num_epochs': 100,          # Number of training epochs
            'weight_decay': 1e-4,       # Weight decay for regularization
            'grad_clip': 1.0,           # Gradient clipping threshold
            'early_stopping': {
                'patience': 10,         # Early stopping patience
                'min_delta': 1e-4,      # Minimum improvement threshold
            },
            'scheduler': {
                'type': 'StepLR',       # Learning rate scheduler type
                'step_size': 30,        # Step size for StepLR
                'gamma': 0.1,           # Decay factor for StepLR
            },
            'optimizer': {
                'type': 'Adam',         # Optimizer type
                'betas': [0.9, 0.999],  # Adam beta parameters
                'eps': 1e-8,            # Adam epsilon parameter
            },
        },
        
        # Dataset parameters
        'dataset': {
            'name': 'sdd',              # Dataset name
            'data_dir': './DATASET/SDD_WITH_FEATURES_SPLITTED',
            'obs_len': 8,               # Number of observed time steps
            'pred_len': 12,             # Number of predicted time steps
            'skip': 1,                  # Number of frames to skip
            'threshold': 0.002,         # Threshold for non-linear trajectory detection
            'min_ped': 1,               # Minimum number of pedestrians per sequence
            'delim': 'tab',             # Delimiter in dataset files
            'norm_lap_matr': True,      # Whether to use normalized Laplacian matrix
            'augmentation': {
                'enabled': False,       # Whether to use data augmentation
                'rotation': [-0.1, 0.1], # Rotation angles for augmentation
                'translation': [(-0.5, 0.5), (-0.5, 0.5)], # Translation ranges
                'noise': 0.01,          # Noise standard deviation
            },
        },
        
        # Loss function parameters
        'loss': {
            'type': 'bivariate',        # Loss function type
            'smoothness_weight': 0.1,   # Weight for smoothness penalty
            'distance_weight': 0.1,     # Weight for distance penalty
            'linearity_weight': 0.05,   # Weight for linearity penalty
        },
        
        # Evaluation parameters
        'evaluation': {
            'metrics': ['ade', 'fde'],  # Evaluation metrics to compute
            'save_predictions': True,   # Whether to save predictions
            'visualize': True,          # Whether to create visualizations
        },
        
        # Data loading parameters
        'dataloader': {
            'num_workers': 4,           # Number of data loader workers
            'pin_memory': True,         # Whether to pin memory
            'shuffle_train': True,      # Whether to shuffle training data
            'shuffle_val': False,       # Whether to shuffle validation data
            'drop_last': True,          # Whether to drop last incomplete batch
        },
        
        # Logging and checkpointing
        'logging': {
            'level': 'INFO',            # Logging level
            'log_dir': './logs',        # Directory for log files
            'log_interval': 10,         # Logging interval (batches)
            'save_interval': 1000,      # Checkpoint saving interval (batches)
            'tensorboard': True,        # Whether to use TensorBoard logging
        },
        
        # Output and checkpointing
        'output': {
            'output_dir': './outputs',  # Output directory for results
            'checkpoint_dir': './checkpoints', # Directory for saving checkpoints
            'save_best_only': True,     # Whether to save only the best model
            'save_last': True,          # Whether to save the last model
        },
        
        # Hardware and performance
        'hardware': {
            'device': 'auto',           # Device to use (cpu, cuda, auto)
            'mixed_precision': False,   # Whether to use mixed precision training
            'compile_model': False,     # Whether to compile model (PyTorch 2.0+)
        },
        
        # Reproducibility
        'reproducibility': {
            'seed': 42,                 # Random seed
            'deterministic': True,      # Whether to use deterministic algorithms
            'benchmark': False,         # Whether to use cudnn benchmark
        },
        
        # Semantic features
        'semantic': {
            'enabled': True,            # Whether to use semantic features
            'feature_dim': 17,          # Dimension of semantic features
            'semantic_map_dir': './DATASET/SDD_semantic_maps_CORRECTED',
            'normalize_features': True, # Whether to normalize semantic features
        },
        
        # Graph construction
        'graph': {
            'adjacency_type': 'inverse_distance', # Type of adjacency matrix
            'normalize_adjacency': True, # Whether to normalize adjacency matrix
            'self_loops': False,        # Whether to include self loops
            'directed': False,          # Whether graph is directed
        },
        
        # Visualization parameters
        'visualization': {
            'plot_trajectories': True,  # Whether to plot trajectories
            'plot_predictions': True,   # Whether to plot predictions
            'plot_attention': False,    # Whether to plot attention weights
            'save_plots': True,         # Whether to save plots
            'plot_format': 'png',       # Format for saved plots
            'dpi': 300,                 # DPI for saved plots
        },
    }
    
    return Config(config_dict)


def get_training_config() -> Config:
    """Get configuration optimized for training."""
    config = get_default_config()
    
    # Training-specific overrides
    config.training.batch_size = 64
    config.training.num_epochs = 200
    config.dataloader.num_workers = 8
    config.logging.log_interval = 50
    config.hardware.mixed_precision = True
    
    return config


def get_evaluation_config() -> Config:
    """Get configuration optimized for evaluation."""
    config = get_default_config()
    
    # Evaluation-specific overrides
    config.training.batch_size = 1
    config.dataloader.shuffle_train = False
    config.dataloader.shuffle_val = False
    config.evaluation.save_predictions = True
    config.evaluation.visualize = True
    
    return config


def get_inference_config() -> Config:
    """Get configuration optimized for inference."""
    config = get_default_config()
    
    # Inference-specific overrides
    config.training.batch_size = 1
    config.dataloader.num_workers = 1
    config.hardware.mixed_precision = False
    config.logging.level = 'WARNING'
    
    return config


def get_debug_config() -> Config:
    """Get configuration for debugging."""
    config = get_default_config()
    
    # Debug-specific overrides
    config.training.batch_size = 2
    config.training.num_epochs = 2
    config.dataloader.num_workers = 0
    config.logging.level = 'DEBUG'
    config.logging.log_interval = 1
    config.reproducibility.deterministic = True
    
    return config
