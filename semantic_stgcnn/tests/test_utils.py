"""
Unit tests for Semantic-STGCNN utilities.
"""

import pytest
import numpy as np
import torch
import tempfile
import os
from pathlib import Path

from semantic_stgcnn.utils import (
    ade, fde, bivariate_loss, calculate_trajectory_metrics,
    anorm, seq_to_nodes, read_file, poly_fit
)


class TestMetrics:
    """Test suite for evaluation metrics."""
    
    def test_ade_calculation(self):
        """Test Average Displacement Error calculation."""
        # Create simple test data
        pred_all = [np.array([[[1.0, 1.0], [2.0, 2.0]], [[1.1, 1.1], [2.1, 2.1]]])]
        target_all = [np.array([[[1.0, 1.0], [2.0, 2.0]], [[1.0, 1.0], [2.0, 2.0]]])]
        count_ = [2]  # 2 pedestrians
        
        ade_value = ade(pred_all, target_all, count_)
        
        # Should be small since predictions are close to targets
        assert isinstance(ade_value, float)
        assert ade_value >= 0
        assert ade_value < 1.0  # Should be small for this simple case
    
    def test_fde_calculation(self):
        """Test Final Displacement Error calculation."""
        # Create simple test data
        pred_all = [np.array([[[1.0, 1.0], [2.0, 2.0]], [[1.1, 1.1], [2.1, 2.1]]])]
        target_all = [np.array([[[1.0, 1.0], [2.0, 2.0]], [[1.0, 1.0], [2.0, 2.0]]])]
        count_ = [2]  # 2 pedestrians
        
        fde_value = fde(pred_all, target_all, count_)
        
        # Should be small since final predictions are close to targets
        assert isinstance(fde_value, float)
        assert fde_value >= 0
        assert fde_value < 1.0  # Should be small for this simple case
    
    def test_bivariate_loss(self):
        """Test bivariate loss calculation."""
        # Create sample tensors
        batch_size, seq_len, num_peds = 2, 12, 4
        
        # Predicted parameters (mean_x, mean_y, std_x, std_y, corr)
        v_pred = torch.randn(seq_len, num_peds, 5)
        
        # Ground truth trajectories
        v_target = torch.randn(seq_len, num_peds, 2)
        
        # Non-linear pedestrian mask
        non_linear_ped = torch.randint(0, 2, (num_peds,)).float()
        
        # Observed trajectories
        v_obs = torch.randn(8, num_peds, 2)
        
        loss = bivariate_loss(v_pred, v_target, non_linear_ped, v_obs)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar loss
        assert loss.item() >= 0  # Loss should be non-negative
    
    def test_calculate_trajectory_metrics(self):
        """Test comprehensive trajectory metrics calculation."""
        predictions = torch.randn(4, 12, 3, 2)  # batch, seq_len, num_peds, coords
        targets = torch.randn(4, 12, 3, 2)
        
        metrics = calculate_trajectory_metrics(predictions, targets)
        
        assert isinstance(metrics, dict)
        assert 'mse' in metrics
        assert 'mae' in metrics
        assert 'avg_displacement_error' in metrics
        assert 'final_displacement_error' in metrics
        
        # All metrics should be non-negative
        for key, value in metrics.items():
            assert value >= 0


class TestGraphUtils:
    """Test suite for graph utility functions."""
    
    def test_anorm_calculation(self):
        """Test normalized inverse distance calculation."""
        p1 = (0.0, 0.0)
        p2 = (3.0, 4.0)  # Distance = 5.0
        
        result = anorm(p1, p2)
        expected = 1.0 / 5.0
        
        assert abs(result - expected) < 1e-6
    
    def test_anorm_same_point(self):
        """Test anorm with identical points."""
        p1 = (1.0, 1.0)
        p2 = (1.0, 1.0)
        
        result = anorm(p1, p2)
        assert result == 0.0
    
    def test_seq_to_nodes(self):
        """Test sequence to nodes conversion."""
        # Create sample sequence data
        seq = np.random.randn(2, 3, 5)  # 2 coords, 3 pedestrians, 5 time steps
        
        nodes = seq_to_nodes(seq)
        
        # Should convert to (time_steps, pedestrians, coords)
        assert nodes.shape == (5, 3, 2)
        
        # Check that data is preserved
        for t in range(5):
            for p in range(3):
                assert np.allclose(nodes[t, p, :], seq[:, p, t])


class TestPreprocessing:
    """Test suite for preprocessing utilities."""
    
    def test_read_file(self):
        """Test file reading functionality."""
        # Create temporary test file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("0\t1\t100.0\t200.0\n")
            f.write("1\t1\t101.0\t201.0\n")
            f.write("0\t2\t150.0\t250.0\n")
            temp_path = f.name
        
        try:
            data = read_file(temp_path, delim='\t')
            
            assert isinstance(data, np.ndarray)
            assert data.shape == (3, 4)  # 3 rows, 4 columns
            assert data[0, 0] == 0  # frame_id
            assert data[0, 1] == 1  # ped_id
            assert data[0, 2] == 100.0  # x
            assert data[0, 3] == 200.0  # y
            
        finally:
            os.unlink(temp_path)
    
    def test_read_file_nonexistent(self):
        """Test error handling for non-existent file."""
        with pytest.raises(Exception):  # Should raise some exception
            read_file("nonexistent_file.txt")
    
    def test_poly_fit_linear(self):
        """Test polynomial fitting for linear trajectory."""
        # Create linear trajectory
        traj = np.array([
            [0, 1, 2, 3, 4],  # x coordinates
            [0, 1, 2, 3, 4]   # y coordinates
        ])
        
        result = poly_fit(traj, traj_len=5, threshold=0.1)
        
        # Should be classified as linear (0)
        assert result == 0.0
    
    def test_poly_fit_nonlinear(self):
        """Test polynomial fitting for non-linear trajectory."""
        # Create curved trajectory
        t = np.linspace(0, 4, 5)
        traj = np.array([
            t,           # x coordinates
            t**2         # y coordinates (parabolic)
        ])
        
        result = poly_fit(traj, traj_len=5, threshold=0.1)
        
        # Should be classified as non-linear (1)
        assert result == 1.0


class TestDatasetUtils:
    """Test suite for dataset utilities."""
    
    def test_trajectory_shapes(self):
        """Test that trajectory data maintains correct shapes."""
        # Simulate trajectory data
        num_peds, seq_len = 4, 8
        
        # Absolute trajectories
        abs_traj = np.random.randn(num_peds, 2, seq_len)
        
        # Relative trajectories
        rel_traj = np.diff(abs_traj, axis=2)
        rel_traj = np.concatenate([np.zeros((num_peds, 2, 1)), rel_traj], axis=2)
        
        assert abs_traj.shape == (num_peds, 2, seq_len)
        assert rel_traj.shape == (num_peds, 2, seq_len)
    
    def test_feature_dimensions(self):
        """Test semantic feature dimensions."""
        num_peds, seq_len = 3, 8
        semantic_features = 17  # Number of semantic features
        
        # Total features: x, y, vx, vy + semantic features
        total_features = 4 + semantic_features
        
        node_features = np.random.randn(num_peds, total_features, seq_len)
        
        assert node_features.shape == (num_peds, total_features, seq_len)
        
        # Extract different feature types
        positions = node_features[:, :2, :]  # x, y
        velocities = node_features[:, 2:4, :]  # vx, vy
        semantic = node_features[:, 4:, :]  # semantic features
        
        assert positions.shape == (num_peds, 2, seq_len)
        assert velocities.shape == (num_peds, 2, seq_len)
        assert semantic.shape == (num_peds, semantic_features, seq_len)


class TestIntegration:
    """Integration tests for utility functions."""
    
    def test_metrics_pipeline(self):
        """Test complete metrics calculation pipeline."""
        # Generate synthetic data
        batch_size, pred_len, num_peds = 4, 12, 3
        
        predictions = torch.randn(batch_size, pred_len, num_peds, 2)
        targets = predictions + 0.1 * torch.randn_like(predictions)  # Add small noise
        
        # Calculate metrics
        metrics = calculate_trajectory_metrics(predictions, targets)
        
        # All metrics should be reasonable
        assert 0 <= metrics['mse'] <= 1.0  # Should be small due to small noise
        assert 0 <= metrics['mae'] <= 1.0
        assert metrics['avg_displacement_error'] >= 0
        assert metrics['final_displacement_error'] >= 0
    
    def test_preprocessing_pipeline(self):
        """Test complete preprocessing pipeline."""
        # Create synthetic trajectory data
        frames = [0, 1, 2, 3, 4]
        ped_ids = [1, 2]
        
        data = []
        for frame in frames:
            for ped_id in ped_ids:
                x = frame * 10 + ped_id  # Simple trajectory
                y = frame * 5 + ped_id * 2
                # Add semantic features
                semantic_features = [0.1, 0.2, 0.3, 0.4, 0.5]
                row = [frame, ped_id, x, y] + semantic_features
                data.append(row)
        
        data = np.array(data)
        
        # Test data structure
        assert data.shape[1] == 9  # frame, ped_id, x, y + 5 semantic features
        
        # Test trajectory extraction
        ped1_data = data[data[:, 1] == 1]
        ped2_data = data[data[:, 1] == 2]
        
        assert len(ped1_data) == len(frames)
        assert len(ped2_data) == len(frames)
        
        # Test trajectory linearity
        ped1_traj = ped1_data[:, 2:4].T  # x, y coordinates
        linearity = poly_fit(ped1_traj, len(frames), threshold=0.1)
        
        # Should be linear since we created linear trajectories
        assert linearity == 0.0
