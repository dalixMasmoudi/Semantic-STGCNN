"""
Unit tests for Semantic-STGCNN models.
"""

import pytest
import torch
import torch.nn as nn
from semantic_stgcnn.models import SemanticSTGCNN, ConvTemporalGraphical, SpatialTemporalGCN


class TestConvTemporalGraphical:
    """Test suite for ConvTemporalGraphical layer."""
    
    def test_initialization(self):
        """Test layer initialization with default parameters."""
        layer = ConvTemporalGraphical(in_channels=2, out_channels=5, kernel_size=8)
        assert isinstance(layer, nn.Module)
        assert layer.kernel_size == 8
    
    def test_forward_pass(self):
        """Test forward pass with sample input."""
        layer = ConvTemporalGraphical(in_channels=2, out_channels=5, kernel_size=8)
        
        # Create sample input
        batch_size, seq_len, num_nodes = 1, 8, 4
        x = torch.randn(batch_size, 2, seq_len, num_nodes)
        A = torch.randn(8, num_nodes, num_nodes)  # kernel_size x num_nodes x num_nodes
        
        # Forward pass
        output, A_out = layer(x, A)
        
        # Check output shapes
        assert output.shape == (batch_size, 5, seq_len, num_nodes)
        assert A_out.shape == A.shape
    
    def test_invalid_adjacency_size(self):
        """Test error handling for invalid adjacency matrix size."""
        layer = ConvTemporalGraphical(in_channels=2, out_channels=5, kernel_size=8)
        
        x = torch.randn(1, 2, 8, 4)
        A = torch.randn(4, 4, 4)  # Wrong size (should be 8 x 4 x 4)
        
        with pytest.raises(AssertionError):
            layer(x, A)


class TestSpatialTemporalGCN:
    """Test suite for SpatialTemporalGCN layer."""
    
    def test_initialization(self):
        """Test layer initialization with default parameters."""
        layer = SpatialTemporalGCN(
            in_channels=2, 
            out_channels=5, 
            kernel_size=(3, 8)
        )
        assert isinstance(layer, nn.Module)
    
    def test_forward_pass(self):
        """Test forward pass with sample input."""
        layer = SpatialTemporalGCN(
            in_channels=2, 
            out_channels=5, 
            kernel_size=(3, 8)
        )
        
        # Create sample input
        batch_size, seq_len, num_nodes = 1, 8, 4
        x = torch.randn(batch_size, 2, seq_len, num_nodes)
        A = torch.randn(8, num_nodes, num_nodes)
        
        # Forward pass
        output, A_out = layer(x, A)
        
        # Check output shapes
        assert output.shape == (batch_size, 5, seq_len, num_nodes)
        assert A_out.shape == A.shape
    
    def test_residual_connection(self):
        """Test residual connection functionality."""
        # Same input/output channels for identity residual
        layer = SpatialTemporalGCN(
            in_channels=5, 
            out_channels=5, 
            kernel_size=(3, 8),
            residual=True
        )
        
        x = torch.randn(1, 5, 8, 4)
        A = torch.randn(8, 4, 4)
        
        output, _ = layer(x, A)
        assert output.shape == x.shape
    
    def test_no_residual(self):
        """Test layer without residual connection."""
        layer = SpatialTemporalGCN(
            in_channels=2, 
            out_channels=5, 
            kernel_size=(3, 8),
            residual=False
        )
        
        x = torch.randn(1, 2, 8, 4)
        A = torch.randn(8, 4, 4)
        
        output, _ = layer(x, A)
        assert output.shape == (1, 5, 8, 4)
    
    def test_invalid_kernel_size(self):
        """Test error handling for invalid kernel size."""
        with pytest.raises(AssertionError):
            SpatialTemporalGCN(
                in_channels=2, 
                out_channels=5, 
                kernel_size=(2, 8)  # Even temporal kernel size
            )


class TestSemanticSTGCNN:
    """Test suite for SemanticSTGCNN model."""
    
    def test_initialization_default(self):
        """Test model initialization with default parameters."""
        model = SemanticSTGCNN()
        assert isinstance(model, nn.Module)
        assert model.n_stgcnn == 2
        assert model.n_txpcnn == 5
        assert model.input_feat == 21
        assert model.output_feat == 2
    
    def test_initialization_custom(self):
        """Test model initialization with custom parameters."""
        model = SemanticSTGCNN(
            n_stgcnn=3,
            n_txpcnn=4,
            input_feat=15,
            output_feat=3,
            seq_len=10,
            pred_seq_len=8,
            kernel_size=5
        )
        assert model.n_stgcnn == 3
        assert model.n_txpcnn == 4
        assert model.input_feat == 15
        assert model.output_feat == 3
    
    def test_forward_pass(self):
        """Test model forward pass with sample input."""
        model = SemanticSTGCNN()
        
        # Create sample input
        batch_size, seq_len, num_nodes = 2, 8, 4
        input_feat = 21
        pred_seq_len = 12
        
        V = torch.randn(batch_size, input_feat, seq_len, num_nodes)
        A = torch.randn(seq_len, num_nodes, num_nodes)
        
        # Forward pass
        output = model(V, A)
        
        # Check output shape
        expected_shape = (batch_size, pred_seq_len, num_nodes, 2)
        assert output.shape == expected_shape
    
    def test_training_mode(self):
        """Test model behavior in training mode."""
        model = SemanticSTGCNN()
        model.train()
        
        V = torch.randn(1, 21, 8, 4)
        A = torch.randn(8, 4, 4)
        
        output = model(V, A)
        assert output.shape == (1, 12, 4, 2)
    
    def test_eval_mode(self):
        """Test model behavior in evaluation mode."""
        model = SemanticSTGCNN()
        model.eval()
        
        V = torch.randn(1, 21, 8, 4)
        A = torch.randn(8, 4, 4)
        
        with torch.no_grad():
            output = model(V, A)
        
        assert output.shape == (1, 12, 4, 2)
    
    def test_get_model_info(self):
        """Test model information retrieval."""
        model = SemanticSTGCNN()
        info = model.get_model_info()
        
        assert isinstance(info, dict)
        assert 'model_name' in info
        assert 'total_parameters' in info
        assert 'trainable_parameters' in info
        assert info['model_name'] == 'Semantic-STGCNN'
        assert info['total_parameters'] > 0
    
    def test_parameter_count(self):
        """Test that model has reasonable number of parameters."""
        model = SemanticSTGCNN()
        total_params = sum(p.numel() for p in model.parameters())
        
        # Should have reasonable number of parameters (not too few, not too many)
        assert 1_000_000 < total_params < 10_000_000
    
    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        model = SemanticSTGCNN()
        
        V = torch.randn(1, 21, 8, 4, requires_grad=True)
        A = torch.randn(8, 4, 4)
        
        output = model(V, A)
        loss = output.sum()
        loss.backward()
        
        # Check that input gradients exist
        assert V.grad is not None
        assert V.grad.shape == V.shape
        
        # Check that model parameters have gradients
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestModelIntegration:
    """Integration tests for model components."""
    
    def test_model_components_compatibility(self):
        """Test that all model components work together."""
        # Test individual components
        gcn_layer = ConvTemporalGraphical(21, 2, 8)
        stgcn_layer = SpatialTemporalGCN(21, 2, (3, 8))
        full_model = SemanticSTGCNN()
        
        # All should be nn.Module instances
        assert all(isinstance(layer, nn.Module) for layer in [gcn_layer, stgcn_layer, full_model])
    
    def test_different_input_sizes(self):
        """Test model with different input sizes."""
        model = SemanticSTGCNN()
        
        # Test different batch sizes
        for batch_size in [1, 4, 8]:
            V = torch.randn(batch_size, 21, 8, 4)
            A = torch.randn(8, 4, 4)
            
            output = model(V, A)
            assert output.shape == (batch_size, 12, 4, 2)
        
        # Test different number of nodes
        for num_nodes in [2, 6, 10]:
            V = torch.randn(2, 21, 8, num_nodes)
            A = torch.randn(8, num_nodes, num_nodes)
            
            output = model(V, A)
            assert output.shape == (2, 12, num_nodes, 2)
    
    def test_model_device_compatibility(self):
        """Test model works on different devices."""
        model = SemanticSTGCNN()
        
        # Test CPU
        V_cpu = torch.randn(1, 21, 8, 4)
        A_cpu = torch.randn(8, 4, 4)
        
        output_cpu = model(V_cpu, A_cpu)
        assert output_cpu.device.type == 'cpu'
        
        # Test CUDA if available
        if torch.cuda.is_available():
            model_cuda = model.cuda()
            V_cuda = V_cpu.cuda()
            A_cuda = A_cpu.cuda()
            
            output_cuda = model_cuda(V_cuda, A_cuda)
            assert output_cuda.device.type == 'cuda'
