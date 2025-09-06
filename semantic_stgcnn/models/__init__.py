"""
Models module for Semantic-STGCNN

This module contains the core neural network architectures for semantic-enhanced
spatio-temporal graph convolutional networks for human trajectory prediction.
"""

from .semantic_stgcnn import SemanticSTGCNN
from .layers import ConvTemporalGraphical, SpatialTemporalGCN

__all__ = ['SemanticSTGCNN', 'ConvTemporalGraphical', 'SpatialTemporalGCN']
