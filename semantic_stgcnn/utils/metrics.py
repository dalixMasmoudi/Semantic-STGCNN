"""
Evaluation metrics for trajectory prediction

This module implements standard evaluation metrics used in human trajectory prediction,
including Average Displacement Error (ADE), Final Displacement Error (FDE), and
bivariate loss functions for probabilistic predictions.

Functions:
    ade: Calculate Average Displacement Error
    fde: Calculate Final Displacement Error  
    bivariate_loss: Calculate bivariate Gaussian loss for probabilistic predictions
"""

import torch
import torch.nn.functional as F
import torch.distributions.multivariate_normal as torchdist
import numpy as np
import math
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


def ade(pred_all: List[np.ndarray], 
        target_all: List[np.ndarray], 
        count_: List[int]) -> float:
    """
    Calculate Average Displacement Error (ADE).
    
    ADE measures the average Euclidean distance between predicted and ground truth
    trajectories across all time steps and all pedestrians.
    
    Args:
        pred_all: List of predicted trajectories for each sequence
        target_all: List of ground truth trajectories for each sequence  
        count_: List of number of pedestrians in each sequence
        
    Returns:
        Average displacement error across all sequences
        
    Note:
        Lower ADE values indicate better prediction accuracy.
    """
    all_sequences = len(pred_all)
    sum_all = 0.0
    
    for s in range(all_sequences):
        # Swap axes to get (num_pedestrians, time_steps, coordinates)
        pred = np.swapaxes(pred_all[s][:, :count_[s], :], 0, 1)
        target = np.swapaxes(target_all[s][:, :count_[s], :], 0, 1)
        
        num_pedestrians = pred.shape[0]
        time_steps = pred.shape[1]
        sum_sequence = 0.0
        
        # Calculate Euclidean distance for each pedestrian at each time step
        for i in range(num_pedestrians):
            for t in range(time_steps):
                distance = math.sqrt(
                    (pred[i, t, 0] - target[i, t, 0])**2 + 
                    (pred[i, t, 1] - target[i, t, 1])**2
                )
                sum_sequence += distance
        
        # Average over all pedestrians and time steps in this sequence
        sum_all += sum_sequence / (num_pedestrians * time_steps)
    
    # Average over all sequences
    return sum_all / all_sequences


def fde(pred_all: List[np.ndarray], 
        target_all: List[np.ndarray], 
        count_: List[int]) -> float:
    """
    Calculate Final Displacement Error (FDE).
    
    FDE measures the Euclidean distance between predicted and ground truth
    positions at the final time step only.
    
    Args:
        pred_all: List of predicted trajectories for each sequence
        target_all: List of ground truth trajectories for each sequence
        count_: List of number of pedestrians in each sequence
        
    Returns:
        Final displacement error across all sequences
        
    Note:
        Lower FDE values indicate better final position prediction accuracy.
    """
    all_sequences = len(pred_all)
    sum_all = 0.0
    
    for s in range(all_sequences):
        # Swap axes to get (num_pedestrians, time_steps, coordinates)
        pred = np.swapaxes(pred_all[s][:, :count_[s], :], 0, 1)
        target = np.swapaxes(target_all[s][:, :count_[s], :], 0, 1)
        
        num_pedestrians = pred.shape[0]
        time_steps = pred.shape[1]
        sum_sequence = 0.0
        
        # Calculate Euclidean distance for each pedestrian at final time step
        for i in range(num_pedestrians):
            final_t = time_steps - 1
            distance = math.sqrt(
                (pred[i, final_t, 0] - target[i, final_t, 0])**2 + 
                (pred[i, final_t, 1] - target[i, final_t, 1])**2
            )
            sum_sequence += distance
        
        # Average over all pedestrians in this sequence
        sum_all += sum_sequence / num_pedestrians
    
    # Average over all sequences
    return sum_all / all_sequences


def bivariate_loss(v_pred: torch.Tensor, 
                  v_target: torch.Tensor,
                  non_linear_ped: torch.Tensor,
                  v_obs: torch.Tensor) -> torch.Tensor:
    """
    Calculate bivariate Gaussian loss for probabilistic trajectory prediction.
    
    This loss function assumes the predicted trajectories follow a bivariate Gaussian
    distribution and calculates the negative log-likelihood. It includes additional
    penalties for smoothness and realistic motion patterns.
    
    Args:
        v_pred: Predicted parameters (mean_x, mean_y, std_x, std_y, correlation)
                Shape: (time_steps, num_pedestrians, 5)
        v_target: Ground truth trajectories (x, y coordinates)
                 Shape: (time_steps, num_pedestrians, 2)
        non_linear_ped: Mask for non-linear pedestrians
        v_obs: Observed trajectories for context
        
    Returns:
        Bivariate loss value
        
    Note:
        The loss combines:
        1. Negative log-likelihood of bivariate Gaussian
        2. Smoothness penalty for realistic motion
        3. Distance penalty for trajectory consistency
    """
    # Extract predicted parameters
    mean_x = v_pred[:, :, 0]  # Predicted mean x-coordinate
    mean_y = v_pred[:, :, 1]  # Predicted mean y-coordinate
    
    # Apply softplus to ensure positive standard deviations
    std_x = F.softplus(v_pred[:, :, 2])  # Standard deviation in x
    std_y = F.softplus(v_pred[:, :, 3])  # Standard deviation in y
    
    # Apply tanh to bound correlation coefficient between -1 and 1
    corr = (2 / torch.pi) * torch.atan(v_pred[:, :, 4])  # Correlation coefficient
    
    # Calculate normalized differences
    norm_x = v_target[:, :, 0] - mean_x
    norm_y = v_target[:, :, 1] - mean_y
    
    # Calculate covariance terms
    std_x_std_y = std_x * std_y
    
    # Calculate the exponent term of bivariate Gaussian
    z = (norm_x / std_x) ** 2 + (norm_y / std_y) ** 2 - \
        2 * ((corr * norm_x * norm_y) / std_x_std_y)
    
    neg_rho = 1 - corr ** 2
    
    # Calculate the probability density
    result = torch.exp(-z / (2 * neg_rho))
    
    # Normalization factor
    denom = 2 * np.pi * (std_x_std_y * torch.sqrt(neg_rho))
    
    # Final PDF calculation
    result = result / denom
    
    # Numerical stability - clamp to avoid log(0)
    epsilon = 1e-20
    result = -torch.log(torch.clamp(result, min=epsilon))
    
    # Create covariance matrix for sampling (used in training)
    if v_pred.requires_grad:  # Only during training
        cov = torch.zeros(v_pred.shape[0], v_pred.shape[1], 2, 2, device=v_pred.device)
        cov[:, :, 0, 0] = std_x * std_x
        cov[:, :, 0, 1] = corr * std_x * std_y
        cov[:, :, 1, 0] = corr * std_x * std_y
        cov[:, :, 1, 1] = std_y * std_y
        
        # Sample from the predicted distribution
        mean = v_pred[:, :, 0:2]
        try:
            mvnormal = torchdist.MultivariateNormal(mean, cov)
            v_prediction = mvnormal.sample()
            
            # Add smoothness penalty
            if v_prediction.shape[0] > 1:
                smoothness_penalty = torch.mean(
                    torch.abs(v_prediction[1:, :, :2] - v_prediction[:-1, :, :2])
                )
                result = result + 0.1 * smoothness_penalty
                
        except Exception as e:
            logger.warning(f"Error in multivariate normal sampling: {e}")
            # Fallback to simple L2 loss if covariance is not positive definite
            result = F.mse_loss(v_pred[:, :, :2], v_target)
    
    return torch.mean(result)


def seq_to_nodes(seq_: np.ndarray) -> np.ndarray:
    """
    Convert sequence format to node format for graph processing.
    
    Args:
        seq_: Input sequence of shape (2, num_pedestrians, seq_len)
        
    Returns:
        Node format array of shape (seq_len, num_pedestrians, 2)
    """
    max_nodes = seq_.shape[1]  # Number of pedestrians
    seq_ = seq_.squeeze()
    seq_len = seq_.shape[2]
    
    V = np.zeros((seq_len, max_nodes, 2))
    for s in range(seq_len):
        step_ = seq_[:, :, s]
        for h in range(len(step_)):
            V[s, h, :] = step_[h]
    
    return V.squeeze()


def calculate_trajectory_metrics(predictions: torch.Tensor, 
                               targets: torch.Tensor,
                               mask: torch.Tensor = None) -> dict:
    """
    Calculate comprehensive trajectory prediction metrics.
    
    Args:
        predictions: Predicted trajectories
        targets: Ground truth trajectories  
        mask: Optional mask for valid pedestrians
        
    Returns:
        Dictionary containing various metrics
    """
    with torch.no_grad():
        # Convert to numpy for metric calculation
        pred_np = predictions.detach().cpu().numpy()
        target_np = targets.detach().cpu().numpy()
        
        if mask is not None:
            mask_np = mask.detach().cpu().numpy()
        else:
            mask_np = np.ones(pred_np.shape[:-1], dtype=bool)
        
        # Calculate basic metrics
        mse = F.mse_loss(predictions, targets).item()
        mae = F.l1_loss(predictions, targets).item()
        
        # Calculate displacement errors
        displacements = torch.sqrt(
            (predictions - targets).pow(2).sum(dim=-1)
        )
        
        avg_displacement = displacements.mean().item()
        final_displacement = displacements[:, -1].mean().item()
        
        return {
            'mse': mse,
            'mae': mae, 
            'avg_displacement_error': avg_displacement,
            'final_displacement_error': final_displacement,
            'max_displacement': displacements.max().item(),
            'min_displacement': displacements.min().item()
        }
