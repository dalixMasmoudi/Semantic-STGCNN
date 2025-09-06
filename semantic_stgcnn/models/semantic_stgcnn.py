"""
Semantic-STGCNN: Main model implementation

This module implements the core Semantic Spatio-Temporal Graph Convolutional Neural Network
for human trajectory prediction with semantic feature integration.

Classes:
    SemanticSTGCNN: Main model class implementing the semantic-enhanced STGCNN architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import logging

from .layers import SpatialTemporalGCN

logger = logging.getLogger(__name__)


class SemanticSTGCNN(nn.Module):
    """
    Semantic Spatio-Temporal Graph Convolutional Neural Network for Human Trajectory Prediction.
    
    This model extends the Social-STGCNN architecture by incorporating semantic information
    from the environment to improve trajectory prediction accuracy in complex urban scenarios.
    
    The architecture consists of:
    1. Multiple ST-GCN layers for spatio-temporal feature extraction
    2. Temporal prediction convolutional layers (TPCNN)
    3. Semantic feature integration
    4. Constant velocity handling for stationary pedestrians
    
    Args:
        n_stgcnn (int): Number of ST-GCN layers (default: 2)
        n_txpcnn (int): Number of temporal prediction CNN layers (default: 5)
        input_feat (int): Number of input features per node (default: 21 for semantic features)
        output_feat (int): Number of output features per node (default: 2 for x,y coordinates)
        seq_len (int): Length of input sequence (default: 8)
        pred_seq_len (int): Length of prediction sequence (default: 12)
        kernel_size (int): Temporal kernel size (default: 3)
        
    Shape:
        - Input: (batch_size, input_feat, seq_len, num_nodes)
        - Output: (batch_size, pred_seq_len, num_nodes, output_feat)
        
    References:
        - Social STGCNN: A Social Spatio-Temporal Graph Convolutional Neural Network 
          for Human Trajectory Prediction (Abduallah et al., 2020)
        - This work extends the original by incorporating semantic environmental features
    """
    
    def __init__(self,
                 n_stgcnn: int = 2,
                 n_txpcnn: int = 5,
                 input_feat: int = 21,
                 output_feat: int = 2,
                 seq_len: int = 8,
                 pred_seq_len: int = 12,
                 kernel_size: int = 3):
        super(SemanticSTGCNN, self).__init__()
        
        # Store configuration
        self.n_stgcnn = n_stgcnn
        self.n_txpcnn = n_txpcnn
        self.input_feat = input_feat
        self.output_feat = output_feat
        self.seq_len = seq_len
        self.pred_seq_len = pred_seq_len
        
        logger.info(f"Initializing Semantic-STGCNN with {n_stgcnn} ST-GCN layers, "
                   f"{n_txpcnn} TPCNN layers, input_feat={input_feat}")
        
        # ST-GCN layers for spatio-temporal feature extraction
        self.st_gcns = nn.ModuleList()
        self.st_gcns.append(
            SpatialTemporalGCN(input_feat, output_feat, (kernel_size, seq_len))
        )
        for j in range(1, self.n_stgcnn):
            self.st_gcns.append(
                SpatialTemporalGCN(output_feat, output_feat, (kernel_size, seq_len))
            )
        
        # Temporal Prediction Convolutional Neural Networks (TPCNN)
        self.tpcnns = nn.ModuleList()
        self.tpcnns.append(nn.Conv2d(seq_len, pred_seq_len, 3, padding=1))
        for j in range(1, self.n_txpcnn):
            self.tpcnns.append(nn.Conv2d(pred_seq_len, pred_seq_len, 3, padding=1))
        
        # Final output layer
        self.tpcnn_output = nn.Conv2d(pred_seq_len, pred_seq_len, 3, padding=1)
        
        # Activation functions for TPCNN layers
        self.prelus = nn.ModuleList()
        for j in range(self.n_txpcnn):
            self.prelus.append(nn.PReLU())
    
    def calculate_constant_velocity(self, 
                                  predictions: torch.Tensor, 
                                  stationary_mask: torch.Tensor,
                                  mean_x: torch.Tensor, 
                                  mean_y: torch.Tensor) -> torch.Tensor:
        """
        Apply constant velocity model for stationary pedestrians.
        
        For pedestrians identified as stationary (very low velocity), this method
        applies a constant velocity assumption to generate more realistic predictions.
        
        Args:
            predictions: Predicted trajectories of shape (batch, pred_len, num_nodes, 2)
            stationary_mask: Boolean mask identifying stationary pedestrians
            mean_x: Mean x-coordinate of last observed position
            mean_y: Mean y-coordinate of last observed position
            
        Returns:
            Modified predictions with constant velocity applied to stationary pedestrians
        """
        if stationary_mask.any():
            # Apply constant velocity for stationary pedestrians
            # This helps maintain realistic motion patterns for slow-moving or stopped pedestrians
            for i in range(predictions.shape[1]):  # For each prediction timestep
                predictions[:, i, stationary_mask, 0] = mean_x[stationary_mask]
                predictions[:, i, stationary_mask, 1] = mean_y[stationary_mask]
        
        return predictions
    
    def forward(self, v: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Semantic-STGCNN model.
        
        Args:
            v: Input velocity/feature tensor of shape (batch, input_feat, seq_len, num_nodes)
            a: Adjacency matrix of shape (kernel_size, num_nodes, num_nodes)
            
        Returns:
            Predicted trajectories of shape (batch, pred_seq_len, num_nodes, output_feat)
        """
        # Extract velocity information for stationary pedestrian detection
        # Use the last timestep's velocity features (first 2 channels are typically vx, vy)
        velocities_mean_x_y = v[-1, :2, :, :].mean(dim=1)  # Average over sequence length
        
        # Identify stationary pedestrians (velocity < 0.01 m per 0.4 second)
        stationary_mask = (torch.abs(velocities_mean_x_y[0, :]) <= 0.01) & \
                         (torch.abs(velocities_mean_x_y[1, :]) <= 0.01)
        
        # Store mean position for constant velocity calculation
        mean_x, mean_y = v[-1, :2, -1, :]  # Last position
        
        # Apply ST-GCN layers for spatio-temporal feature extraction
        for k in range(self.n_stgcnn):
            v, a = self.st_gcns[k](v, a)
        
        # Reshape for temporal prediction: (batch, channels, time, nodes) -> (batch, time, channels, nodes)
        v = v.view(v.shape[0], v.shape[2], v.shape[1], v.shape[3])
        
        # Apply first TPCNN layer
        v = self.prelus[0](self.tpcnns[0](v))
        
        # Apply remaining TPCNN layers with residual connections
        for k in range(1, self.n_txpcnn - 1):
            v = self.prelus[k](self.tpcnns[k](v)) + v
        
        # Final output layer
        v = self.tpcnn_output(v)
        
        # Reshape back to desired output format: (batch, time, nodes, features)
        v = v.view(v.shape[0], v.shape[2], v.shape[1], v.shape[3])
        v = v.permute(0, 2, 3, 1)  # (batch, pred_seq_len, num_nodes, output_feat)
        
        # Apply constant velocity for stationary pedestrians during inference
        if not self.training:
            v = self.calculate_constant_velocity(v, stationary_mask, mean_x, mean_y)
        
        return v
    
    def get_model_info(self) -> dict:
        """
        Get model configuration and parameter information.
        
        Returns:
            Dictionary containing model configuration and statistics
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'Semantic-STGCNN',
            'n_stgcnn_layers': self.n_stgcnn,
            'n_tpcnn_layers': self.n_txpcnn,
            'input_features': self.input_feat,
            'output_features': self.output_feat,
            'sequence_length': self.seq_len,
            'prediction_length': self.pred_seq_len,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
        }
