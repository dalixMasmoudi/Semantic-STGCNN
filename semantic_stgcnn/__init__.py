"""
Semantic-STGCNN: A Semantic Spatio-Temporal Graph Convolutional Neural Network 
for Human Trajectory Prediction

This package implements a novel approach to human trajectory prediction by incorporating
semantic information into spatio-temporal graph convolutional neural networks.

Author: Dali Masmoudi
Institution: [Your Institution]
Year: 2024

References:
    - Social STGCNN: A Social Spatio-Temporal Graph Convolutional Neural Network 
      for Human Trajectory Prediction (Abduallah et al., 2020)
    - Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition
      (Yan et al., 2018)
"""

__version__ = "1.0.0"
__author__ = "Dali Masmoudi"
__email__ = "your.email@domain.com"

from .models.semantic_stgcnn import SemanticSTGCNN
from .models.layers import ConvTemporalGraphical, SpatialTemporalGCN
from .utils.dataset import TrajectoryDataset
from .utils.metrics import ade, fde, bivariate_loss

__all__ = [
    'SemanticSTGCNN',
    'ConvTemporalGraphical', 
    'SpatialTemporalGCN',
    'TrajectoryDataset',
    'ade',
    'fde', 
    'bivariate_loss'
]
