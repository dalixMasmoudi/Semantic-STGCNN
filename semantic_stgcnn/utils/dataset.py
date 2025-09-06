"""
Dataset utilities for Semantic-STGCNN

This module provides dataset loading and preprocessing utilities for trajectory prediction,
including graph construction and semantic feature integration.

Classes:
    TrajectoryDataset: PyTorch dataset for loading trajectory data with semantic features
    
Functions:
    seq_to_graph: Convert trajectory sequences to graph representation
    read_file: Read trajectory data from files
"""

import os
import math
import torch
import numpy as np
import networkx as nx
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


def read_file(path: str, delim: str = '\t') -> np.ndarray:
    """
    Read trajectory data from file.
    
    Args:
        path: Path to the trajectory file
        delim: Delimiter used in the file (default: tab)
        
    Returns:
        Numpy array with trajectory data
        
    Expected file format:
        frame_id ped_id x y [additional_features...]
    """
    try:
        data = []
        if delim == 'tab':
            delim = '\t'
        elif delim == 'space':
            delim = ' '
            
        with open(path, 'r') as f:
            for line in f:
                line = line.strip().split(delim)
                line = [float(i) for i in line]
                data.append(line)
        
        return np.asarray(data)
    except Exception as e:
        logger.error(f"Error reading file {path}: {e}")
        raise


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


def seq_to_graph(seq_: np.ndarray, 
                seq_rel: np.ndarray, 
                node_features: np.ndarray,
                norm_lap_matr: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert trajectory sequences to graph representation with adjacency matrix.
    
    This function creates a graph where nodes represent pedestrians and edges
    represent spatial relationships. The adjacency matrix is computed based on
    pedestrian proximity and can be normalized as a Laplacian matrix.
    
    Args:
        seq_: Absolute trajectory sequences of shape (2, num_peds, seq_len)
        seq_rel: Relative trajectory sequences of shape (2, num_peds, seq_len)  
        node_features: Additional node features of shape (num_features, num_peds, seq_len)
        norm_lap_matr: Whether to use normalized Laplacian matrix (default: True)
        
    Returns:
        Tuple of (node_features_tensor, adjacency_matrix_tensor)
        - node_features_tensor: Shape (seq_len, num_features, num_peds)
        - adjacency_matrix_tensor: Shape (seq_len, num_peds, num_peds)
    """
    seq_ = seq_.squeeze()
    seq_rel = seq_rel.squeeze()
    seq_len = seq_.shape[2]
    max_nodes = seq_.shape[0]
    
    # Combine absolute positions, relative positions, and additional features
    V = np.zeros((seq_len, node_features.shape[0], max_nodes))
    A = np.zeros((seq_len, max_nodes, max_nodes))
    
    for s in range(seq_len):
        step_ = seq_[:, :, s]  # Absolute positions at time s
        step_rel = seq_rel[:, :, s]  # Relative positions at time s
        step_features = node_features[:, :, s]  # Additional features at time s
        
        # Combine all features for this timestep
        V[s, :, :] = step_features
        
        # Create adjacency matrix based on spatial proximity
        for h in range(len(step_)):
            V[s, :2, h] = step_[h]  # Store absolute positions in first 2 channels
            V[s, 2:4, h] = step_rel[h]  # Store relative positions in next 2 channels
            
        # Build graph adjacency matrix
        for i in range(max_nodes):
            for j in range(max_nodes):
                if i != j:
                    # Calculate edge weight based on inverse distance
                    A[s, i, j] = anorm((V[s, 0, i], V[s, 1, i]), 
                                     (V[s, 0, j], V[s, 1, j]))
        
        # Normalize adjacency matrix if requested
        if norm_lap_matr:
            G = nx.from_numpy_array(A[s, :, :])
            A[s, :, :] = nx.normalized_laplacian_matrix(G).toarray()
    
    return torch.from_numpy(V).type(torch.float), torch.from_numpy(A).type(torch.float)


def poly_fit(traj: np.ndarray, traj_len: int, threshold: float) -> int:
    """
    Determine if a trajectory is linear or non-linear using polynomial fitting.
    
    Args:
        traj: Trajectory array of shape (2, traj_len)
        traj_len: Length of the trajectory
        threshold: Threshold for determining non-linearity
        
    Returns:
        1 if trajectory is non-linear, 0 if linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0


class TrajectoryDataset(Dataset):
    """
    PyTorch Dataset for loading trajectory data with semantic features.
    
    This dataset loader handles trajectory data in the format used by Stanford Drone Dataset
    and similar trajectory datasets, with support for semantic feature integration.
    
    Args:
        data_dir: Directory containing dataset files
        obs_len: Number of time-steps in input trajectories (default: 8)
        pred_len: Number of time-steps in output trajectories (default: 12)
        skip: Number of frames to skip while making the dataset (default: 1)
        threshold: Minimum error to be considered for non-linear trajectory (default: 0.002)
        min_ped: Minimum number of pedestrians in a sequence (default: 1)
        delim: Delimiter in the dataset files (default: tab)
        norm_lap_matr: Whether to use normalized Laplacian matrix (default: True)
        
    File format expected:
        frame_id ped_id x y [semantic_features...]
    """
    
    def __init__(self,
                 data_dir: str,
                 obs_len: int = 8,
                 pred_len: int = 12,
                 skip: int = 1,
                 threshold: float = 0.002,
                 min_ped: int = 1,
                 delim: str = '\t',
                 norm_lap_matr: bool = True):
        super(TrajectoryDataset, self).__init__()
        
        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim
        self.norm_lap_matr = norm_lap_matr
        self.threshold = threshold
        self.min_ped = min_ped
        
        logger.info(f"Loading trajectory dataset from {data_dir}")
        logger.info(f"Observation length: {obs_len}, Prediction length: {pred_len}")
        
        # Initialize data containers
        all_files = os.listdir(self.data_dir)
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        
        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []
        loss_mask_list = []
        non_linear_ped = []
        nodes_features = []
        
        # Process each file
        for path in all_files:
            logger.debug(f"Processing file: {path}")
            data = read_file(path, delim)
            
            # Extract basic trajectory information
            frames = np.unique(data[:, 0]).tolist()
            frame_data = []
            
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])
            
            num_sequences = int(math.ceil((len(frames) - self.seq_len + 1) / skip))
            
            # Process sequences
            for idx in range(0, num_sequences * self.skip + 1, skip):
                curr_seq_data = np.concatenate(
                    frame_data[idx:idx + self.seq_len], axis=0
                )
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                
                # Filter sequences with minimum number of pedestrians
                if len(peds_in_curr_seq) < self.min_ped:
                    continue
                
                curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_loss_mask = np.zeros((len(peds_in_curr_seq), self.seq_len))
                num_peds_considered = 0
                _non_linear_ped = []
                
                # Extract semantic features (assuming they start from column 4)
                num_features = data.shape[1] - 4  # Subtract frame_id, ped_id, x, y
                curr_node_features = np.zeros((len(peds_in_curr_seq), num_features + 4, self.seq_len))
                
                for _, ped_id in enumerate(peds_in_curr_seq):
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                    
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                    
                    if pad_end - pad_front != self.seq_len:
                        continue
                    
                    # Store trajectory data
                    curr_seq[num_peds_considered, :, pad_front:pad_end] = \
                        curr_ped_seq[:, 2:4].T
                    
                    # Store semantic features
                    if num_features > 0:
                        curr_node_features[num_peds_considered, 4:, pad_front:pad_end] = \
                            curr_ped_seq[:, 4:].T
                    
                    # Calculate relative positions
                    rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                    rel_curr_ped_seq[1:, :] = \
                        curr_ped_seq[1:, :] - curr_ped_seq[:-1, :]
                    _idx = num_peds_considered
                    curr_seq_rel[_idx, :, pad_front:pad_end] = \
                        rel_curr_ped_seq[:, 2:4].T
                    
                    # Loss mask
                    curr_loss_mask[num_peds_considered, pad_front:pad_end] = 1
                    
                    # Check for non-linear trajectory
                    _non_linear_ped.append(
                        poly_fit(curr_seq[num_peds_considered, :, :], self.pred_len, self.threshold)
                    )
                    
                    num_peds_considered += 1
                
                if num_peds_considered > self.min_ped:
                    non_linear_ped.append(_non_linear_ped)
                    num_peds_in_seq.append(num_peds_considered)
                    loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                    seq_list.append(curr_seq[:num_peds_considered])
                    seq_list_rel.append(curr_seq_rel[:num_peds_considered])
                    nodes_features.append(curr_node_features[:num_peds_considered])
        
        self.num_seq = len(seq_list)
        logger.info(f"Loaded {self.num_seq} sequences")
        
        # Convert to tensors
        seq_list = np.concatenate(seq_list, axis=0)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        non_linear_ped = np.concatenate(non_linear_ped)
        nodes_features = np.concatenate(nodes_features, axis=0)
        
        # Split into observation and prediction
        self.obs_traj = torch.from_numpy(
            seq_list[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj = torch.from_numpy(
            seq_list[:, :, self.obs_len:]).type(torch.float)
        self.obs_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, self.obs_len:]).type(torch.float)
        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
        self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float)
        self.node_features_obs = nodes_features[:, :, :self.obs_len]
        self.node_features_traj = nodes_features[:, :, self.obs_len:]
        
        # Create sequence start/end indices
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [
            (start, end) for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]
        
        logger.info(f"Dataset initialization complete. Total pedestrians: {len(self.obs_traj)}")
    
    def __len__(self) -> int:
        return self.num_seq
    
    def __getitem__(self, index: int):
        """Get a single sequence from the dataset."""
        start, end = self.seq_start_end[index]
        
        out = [
            self.obs_traj[start:end, :],
            self.pred_traj[start:end, :],
            self.obs_traj_rel[start:end, :],
            self.pred_traj_rel[start:end, :],
            self.non_linear_ped[start:end],
            self.loss_mask[start:end, :],
        ]
        
        # Convert to graph representation
        seq = np.concatenate([self.obs_traj[start:end, :], self.pred_traj[start:end, :]], axis=2)
        seq_rel = np.concatenate([self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :]], axis=2)
        node_features = np.concatenate([self.node_features_obs[start:end, :], self.node_features_traj[start:end, :]], axis=2)
        
        V_obs, A_obs = seq_to_graph(seq, seq_rel, node_features, self.norm_lap_matr)
        
        out.append(V_obs)
        out.append(A_obs)
        
        return tuple(out)
