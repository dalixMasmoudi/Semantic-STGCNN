"""
Graph utility functions for Semantic-STGCNN

This module provides utility functions for graph construction and manipulation
used in the trajectory prediction framework.
"""

import math
import numpy as np
import torch
from typing import Tuple


def anorm(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """
    Calculate normalized inverse distance between two points.
    
    Args:
        p1: First point (x, y)
        p2: Second point (x, y)
        
    Returns:
        Normalized inverse distance (1/distance, or 0 if points are identical)
    """
    norm = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    if norm == 0:
        return 0
    return 1 / norm


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


def nodes_to_seq(nodes: np.ndarray) -> np.ndarray:
    """
    Convert node format back to sequence format.
    
    Args:
        nodes: Node format array of shape (seq_len, num_pedestrians, 2)
        
    Returns:
        Sequence format array of shape (2, num_pedestrians, seq_len)
    """
    seq_len, max_nodes, _ = nodes.shape
    seq = np.zeros((2, max_nodes, seq_len))
    
    for s in range(seq_len):
        for h in range(max_nodes):
            seq[:, h, s] = nodes[s, h, :]
    
    return seq


def calculate_adjacency_matrix(positions: np.ndarray, 
                             method: str = 'inverse_distance',
                             threshold: float = None) -> np.ndarray:
    """
    Calculate adjacency matrix based on pedestrian positions.
    
    Args:
        positions: Array of shape (num_pedestrians, 2) with x,y coordinates
        method: Method for calculating adjacency ('inverse_distance', 'threshold', 'knn')
        threshold: Distance threshold for 'threshold' method
        
    Returns:
        Adjacency matrix of shape (num_pedestrians, num_pedestrians)
    """
    num_peds = positions.shape[0]
    adj_matrix = np.zeros((num_peds, num_peds))
    
    for i in range(num_peds):
        for j in range(num_peds):
            if i != j:
                if method == 'inverse_distance':
                    adj_matrix[i, j] = anorm(positions[i], positions[j])
                elif method == 'threshold':
                    distance = math.sqrt(
                        (positions[i, 0] - positions[j, 0])**2 + 
                        (positions[i, 1] - positions[j, 1])**2
                    )
                    adj_matrix[i, j] = 1.0 if distance < threshold else 0.0
    
    return adj_matrix


def normalize_adjacency_matrix(adj_matrix: np.ndarray, 
                             method: str = 'symmetric') -> np.ndarray:
    """
    Normalize adjacency matrix.
    
    Args:
        adj_matrix: Input adjacency matrix
        method: Normalization method ('symmetric', 'row', 'column')
        
    Returns:
        Normalized adjacency matrix
    """
    if method == 'symmetric':
        # D^(-1/2) * A * D^(-1/2)
        degree = np.sum(adj_matrix, axis=1)
        degree_inv_sqrt = np.power(degree, -0.5)
        degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0.0
        degree_matrix_inv_sqrt = np.diag(degree_inv_sqrt)
        return degree_matrix_inv_sqrt @ adj_matrix @ degree_matrix_inv_sqrt
    
    elif method == 'row':
        # Row normalization: each row sums to 1
        row_sums = np.sum(adj_matrix, axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        return adj_matrix / row_sums
    
    elif method == 'column':
        # Column normalization: each column sums to 1
        col_sums = np.sum(adj_matrix, axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1  # Avoid division by zero
        return adj_matrix / col_sums
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def create_temporal_graph(trajectories: np.ndarray, 
                         temporal_window: int = 3) -> np.ndarray:
    """
    Create temporal graph connections across time steps.
    
    Args:
        trajectories: Array of shape (seq_len, num_pedestrians, 2)
        temporal_window: Number of time steps to connect
        
    Returns:
        Temporal adjacency matrix
    """
    seq_len, num_peds, _ = trajectories.shape
    
    # Create temporal connections
    temporal_adj = np.zeros((seq_len * num_peds, seq_len * num_peds))
    
    for t in range(seq_len):
        for p in range(num_peds):
            current_idx = t * num_peds + p
            
            # Connect to same pedestrian in adjacent time steps
            for dt in range(1, min(temporal_window + 1, seq_len - t)):
                future_idx = (t + dt) * num_peds + p
                temporal_adj[current_idx, future_idx] = 1.0
                temporal_adj[future_idx, current_idx] = 1.0  # Symmetric
    
    return temporal_adj


def graph_laplacian(adj_matrix: np.ndarray, 
                   normalized: bool = True) -> np.ndarray:
    """
    Compute graph Laplacian matrix.
    
    Args:
        adj_matrix: Adjacency matrix
        normalized: Whether to compute normalized Laplacian
        
    Returns:
        Laplacian matrix
    """
    degree = np.sum(adj_matrix, axis=1)
    degree_matrix = np.diag(degree)
    
    if normalized:
        # Normalized Laplacian: I - D^(-1/2) * A * D^(-1/2)
        degree_inv_sqrt = np.power(degree, -0.5)
        degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0.0
        degree_matrix_inv_sqrt = np.diag(degree_inv_sqrt)
        normalized_adj = degree_matrix_inv_sqrt @ adj_matrix @ degree_matrix_inv_sqrt
        return np.eye(adj_matrix.shape[0]) - normalized_adj
    else:
        # Unnormalized Laplacian: D - A
        return degree_matrix - adj_matrix
