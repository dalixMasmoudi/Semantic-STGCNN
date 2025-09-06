"""
Data preprocessing utilities for Semantic-STGCNN

This module provides functions for data preprocessing, feature extraction,
and data augmentation for trajectory prediction tasks.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import Tuple, List, Optional, Dict
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


def normalize_trajectories(trajectories: np.ndarray, 
                         method: str = 'minmax',
                         feature_range: Tuple[float, float] = (0, 1)) -> Tuple[np.ndarray, object]:
    """
    Normalize trajectory coordinates.
    
    Args:
        trajectories: Array of shape (..., 2) with x,y coordinates
        method: Normalization method ('minmax', 'standard', 'robust')
        feature_range: Range for MinMax scaling
        
    Returns:
        Tuple of (normalized_trajectories, scaler_object)
    """
    original_shape = trajectories.shape
    traj_reshaped = trajectories.reshape(-1, 2)
    
    if method == 'minmax':
        scaler = MinMaxScaler(feature_range=feature_range)
    elif method == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    normalized = scaler.fit_transform(traj_reshaped)
    normalized = normalized.reshape(original_shape)
    
    return normalized, scaler


def denormalize_trajectories(normalized_trajectories: np.ndarray, 
                           scaler: object) -> np.ndarray:
    """
    Denormalize trajectory coordinates.
    
    Args:
        normalized_trajectories: Normalized trajectory array
        scaler: Fitted scaler object
        
    Returns:
        Denormalized trajectories
    """
    original_shape = normalized_trajectories.shape
    normalized_reshaped = normalized_trajectories.reshape(-1, 2)
    
    denormalized = scaler.inverse_transform(normalized_reshaped)
    denormalized = denormalized.reshape(original_shape)
    
    return denormalized


def extract_velocity_features(trajectories: np.ndarray) -> np.ndarray:
    """
    Extract velocity features from trajectory coordinates.
    
    Args:
        trajectories: Array of shape (..., seq_len, 2) with x,y coordinates
        
    Returns:
        Velocity features of shape (..., seq_len-1, 2)
    """
    velocities = np.diff(trajectories, axis=-2)
    return velocities


def extract_acceleration_features(trajectories: np.ndarray) -> np.ndarray:
    """
    Extract acceleration features from trajectory coordinates.
    
    Args:
        trajectories: Array of shape (..., seq_len, 2) with x,y coordinates
        
    Returns:
        Acceleration features of shape (..., seq_len-2, 2)
    """
    velocities = extract_velocity_features(trajectories)
    accelerations = np.diff(velocities, axis=-2)
    return accelerations


def extract_distance_features(trajectories: np.ndarray) -> np.ndarray:
    """
    Extract distance-based features from trajectories.
    
    Args:
        trajectories: Array of shape (num_peds, seq_len, 2)
        
    Returns:
        Distance features including pairwise distances
    """
    num_peds, seq_len, _ = trajectories.shape
    
    # Calculate pairwise distances at each time step
    distance_features = np.zeros((num_peds, seq_len, num_peds))
    
    for t in range(seq_len):
        positions = trajectories[:, t, :]  # (num_peds, 2)
        
        for i in range(num_peds):
            for j in range(num_peds):
                if i != j:
                    dist = np.linalg.norm(positions[i] - positions[j])
                    distance_features[i, t, j] = dist
    
    return distance_features


def extract_semantic_features(trajectories: np.ndarray, 
                            semantic_map: np.ndarray,
                            map_bounds: Tuple[float, float, float, float]) -> np.ndarray:
    """
    Extract semantic features from trajectories based on semantic map.
    
    Args:
        trajectories: Array of shape (num_peds, seq_len, 2)
        semantic_map: Semantic map array
        map_bounds: (min_x, max_x, min_y, max_y) bounds of the semantic map
        
    Returns:
        Semantic features for each trajectory point
    """
    num_peds, seq_len, _ = trajectories.shape
    map_height, map_width = semantic_map.shape
    
    min_x, max_x, min_y, max_y = map_bounds
    
    semantic_features = np.zeros((num_peds, seq_len, 1))
    
    for p in range(num_peds):
        for t in range(seq_len):
            x, y = trajectories[p, t, :]
            
            # Convert world coordinates to map coordinates
            map_x = int((x - min_x) / (max_x - min_x) * map_width)
            map_y = int((y - min_y) / (max_y - min_y) * map_height)
            
            # Clamp to map bounds
            map_x = np.clip(map_x, 0, map_width - 1)
            map_y = np.clip(map_y, 0, map_height - 1)
            
            # Extract semantic value
            semantic_features[p, t, 0] = semantic_map[map_y, map_x]
    
    return semantic_features


def augment_trajectories(trajectories: np.ndarray, 
                        augmentation_params: Dict) -> List[np.ndarray]:
    """
    Apply data augmentation to trajectories.
    
    Args:
        trajectories: Input trajectories
        augmentation_params: Dictionary with augmentation parameters
        
    Returns:
        List of augmented trajectory arrays
    """
    augmented_trajs = [trajectories]  # Include original
    
    # Rotation augmentation
    if 'rotation' in augmentation_params:
        angles = augmentation_params['rotation']
        for angle in angles:
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            
            rotated = trajectories @ rotation_matrix.T
            augmented_trajs.append(rotated)
    
    # Translation augmentation
    if 'translation' in augmentation_params:
        translations = augmentation_params['translation']
        for tx, ty in translations:
            translated = trajectories + np.array([tx, ty])
            augmented_trajs.append(translated)
    
    # Scaling augmentation
    if 'scaling' in augmentation_params:
        scales = augmentation_params['scaling']
        for scale in scales:
            scaled = trajectories * scale
            augmented_trajs.append(scaled)
    
    # Noise augmentation
    if 'noise' in augmentation_params:
        noise_std = augmentation_params['noise']
        noise = np.random.normal(0, noise_std, trajectories.shape)
        noisy = trajectories + noise
        augmented_trajs.append(noisy)
    
    return augmented_trajs


def filter_trajectories(trajectories: np.ndarray, 
                       min_length: int = 5,
                       max_velocity: float = 10.0,
                       min_displacement: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter trajectories based on quality criteria.
    
    Args:
        trajectories: Input trajectories of shape (num_trajs, seq_len, 2)
        min_length: Minimum trajectory length
        max_velocity: Maximum allowed velocity
        min_displacement: Minimum total displacement
        
    Returns:
        Tuple of (filtered_trajectories, valid_indices)
    """
    valid_indices = []
    
    for i, traj in enumerate(trajectories):
        # Check minimum length
        if len(traj) < min_length:
            continue
        
        # Check velocity constraints
        velocities = np.diff(traj, axis=0)
        speeds = np.linalg.norm(velocities, axis=1)
        if np.any(speeds > max_velocity):
            continue
        
        # Check minimum displacement
        total_displacement = np.linalg.norm(traj[-1] - traj[0])
        if total_displacement < min_displacement:
            continue
        
        valid_indices.append(i)
    
    filtered_trajectories = trajectories[valid_indices]
    return filtered_trajectories, np.array(valid_indices)


def create_sliding_windows(trajectories: np.ndarray, 
                          window_size: int,
                          stride: int = 1) -> np.ndarray:
    """
    Create sliding windows from trajectory sequences.
    
    Args:
        trajectories: Input trajectories of shape (num_trajs, seq_len, features)
        window_size: Size of each window
        stride: Stride between windows
        
    Returns:
        Windowed trajectories of shape (num_windows, window_size, features)
    """
    num_trajs, seq_len, features = trajectories.shape
    windows = []
    
    for traj in trajectories:
        for start in range(0, seq_len - window_size + 1, stride):
            window = traj[start:start + window_size]
            windows.append(window)
    
    return np.array(windows)
