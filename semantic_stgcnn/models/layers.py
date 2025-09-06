"""
Core neural network layers for Semantic-STGCNN

This module implements the fundamental building blocks of the Semantic-STGCNN architecture,
including graph convolutional layers and spatio-temporal graph convolution modules.

Classes:
    ConvTemporalGraphical: Basic graph convolution module
    SpatialTemporalGCN: Spatio-temporal graph convolutional layer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ConvTemporalGraphical(nn.Module):
    """
    The basic module for applying a graph convolution.
    
    This layer applies graph convolution on input sequences by combining spatial
    graph structure with temporal convolution operations.
    
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel (default: 1)
        t_stride (int): Stride of the temporal convolution (default: 1)
        t_padding (int): Temporal zero-padding added to both sides (default: 0)
        t_dilation (int): Spacing between temporal kernel elements (default: 1)
        bias (bool): If True, adds a learnable bias to the output (default: True)
        
    Shape:
        - Input[0]: Input graph sequence in (N, in_channels, T_in, V) format
        - Input[1]: Input graph adjacency matrix in (K, V, V) format
        - Output[0]: Output graph sequence in (N, out_channels, T_out, V) format
        - Output[1]: Graph adjacency matrix for output data in (K, V, V) format
        
    where:
        N is batch size
        K is the spatial kernel size
        T_in/T_out is length of input/output sequence
        V is the number of graph nodes
        
    References:
        Adapted from: https://github.com/yysijie/st-gcn/blob/master/net/st_gcn.py
    """
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 t_kernel_size: int = 1,
                 t_stride: int = 1,
                 t_padding: int = 0,
                 t_dilation: int = 1,
                 bias: bool = True):
        super(ConvTemporalGraphical, self).__init__()
        
        self.kernel_size = kernel_size
        
        # Temporal convolution layer
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias
        )

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the graph convolution layer.
        
        Args:
            x: Input tensor of shape (N, in_channels, T, V)
            A: Adjacency matrix of shape (K, V, V)
            
        Returns:
            Tuple of (output_tensor, adjacency_matrix)
        """
        assert A.size(0) == self.kernel_size, f"Adjacency matrix size {A.size(0)} != kernel size {self.kernel_size}"
        
        # Apply temporal convolution
        # Shape: (N, in_channels, T, V) -> (N, out_channels*K, T, V)
        x = self.conv(x)
        
        # Reshape for graph convolution
        # Shape: (N, out_channels*K, T, V) -> (N, out_channels, K, T, V)
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc//self.kernel_size, t, v)
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (N, out_channels, K, T, V)
        
        # Apply graph convolution using Einstein summation
        # Combines spatial features with graph structure
        x = torch.einsum('nckti,kij->nctj', (x, A))
        
        return x.contiguous(), A


class SpatialTemporalGCN(nn.Module):
    """
    Applies a spatial temporal graph convolution over an input graph sequence.
    
    This module combines spatial graph convolution with temporal convolution,
    incorporating residual connections and optional Mixture Density Network (MDN) support.
    
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of (temporal_kernel, spatial_kernel)
        use_mdn (bool): Whether to use Mixture Density Network activation (default: False)
        stride (int): Stride of the temporal convolution (default: 1)
        dropout (float): Dropout rate of the final output (default: 0)
        residual (bool): If True, applies a residual mechanism (default: True)
        
    Shape:
        - Input[0]: Input graph sequence in (N, in_channels, T_in, V) format
        - Input[1]: Input graph adjacency matrix in (K, V, V) format
        - Output[0]: Output graph sequence in (N, out_channels, T_out, V) format
        - Output[1]: Graph adjacency matrix for output data in (K, V, V) format
        
    Note:
        The temporal kernel size should be odd to ensure symmetric padding.
    """
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, int],
                 use_mdn: bool = False,
                 stride: int = 1,
                 dropout: float = 0,
                 residual: bool = True):
        super(SpatialTemporalGCN, self).__init__()
        
        assert len(kernel_size) == 2, "kernel_size must be a tuple of (temporal, spatial)"
        assert kernel_size[0] % 2 == 1, "Temporal kernel size must be odd for symmetric padding"
        
        # Calculate padding for temporal dimension
        padding = ((kernel_size[0] - 1) // 2, 0)
        self.use_mdn = use_mdn
        
        # Graph convolution layer
        self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size[1])
        
        # Temporal convolution network
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),  # Apply only on temporal dimension
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )
        
        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)
                ),
                nn.BatchNorm2d(out_channels),
            )
        
        self.prelu = nn.PReLU()

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the spatio-temporal GCN layer.
        
        Args:
            x: Input tensor of shape (N, in_channels, T, V)
            A: Adjacency matrix of shape (K, V, V)
            
        Returns:
            Tuple of (output_tensor, adjacency_matrix)
        """
        # Store residual connection
        res = self.residual(x)
        
        # Apply graph convolution
        x, A = self.gcn(x, A)
        
        # Apply temporal convolution with residual connection
        x = self.tcn(x) + res
        
        # Apply activation (skip for MDN to preserve raw outputs)
        if not self.use_mdn:
            x = self.prelu(x)
        
        return x, A
