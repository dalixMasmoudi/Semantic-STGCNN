import os
import math
import sys

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as Func
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

import torch.optim as optim




class ConvTemporalGraphical(nn.Module):
    #Source : https://github.com/yysijie/st-gcn/blob/master/net/st_gcn.py
    r"""The basic module for applying a graph convolution.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes. 
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,                      #SET KERNEL TO >1 TO CAPTURE TIME DEPENDENCY )maybe keep it because prediction is relat. short term so no need for big kernel)
                                                                    # #In the provided code, where the kernel size is (1, 1), it means that the convolution operation is performed with a 1x1 kernel.
                                                                    # This configuration applies a point-wise convolution, meaning that no spatial filtering
                                                                      # or temporal dependencies are considered by the convolutional operation. It performs a linear transformation on each element of the input independently.
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super(ConvTemporalGraphical,self).__init__()
        self.kernel_size = kernel_size
        '''
        In the context of the social_stgcnn class, the input tensor x represents the pedestrian graph. It has a shape of (8, 4, 2), where 8 corresponds to the number of time steps, 4 corresponds to the number of pedestrians, and 2 represents the number of features (speed along x and y).

Passing the input tensor x through the conv layer, which is an instance of nn.Conv2d, allows for applying convolutional operations on the input. However, since the input tensor x has a shape (8, 4, 2), it needs to be reshaped to match the expected input shape of the nn.Conv2d layer.

The nn.Conv2d layer in PyTorch expects the input tensor to have a shape (batch_size, channels, height, width). In this case, the input tensor x needs to be reshaped to (1, 2, 8, 4) before passing it through the conv layer.

Reshaping the tensor to (1, 2, 8, 4) indicates a batch size of 1, 2 input channels (representing the features), 8 height (corresponding to the number of time steps), and 4 width (representing the number of pedestrians).

By applying the convolution operation, the nn.Conv2d layer can learn spatial patterns or relationships within the pedestrian graph data, considering both the temporal dimension (time steps) and the spatial dimension (pedestrians). This enables the model to capture local and global dependencies in the data, which can be useful for tasks such as trajectory prediction or understanding pedestrian interactions.
'''
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)

    def forward(self, x, A):
        assert A.size(0) == self.kernel_size
        x = self.conv(x)      #            #(1*input_channels* seq length * ped number=->(1,out_channels_seq_length*ped number) ex : 1 2 8 4 > 1 5 8 4
                                        # goal is to increase the number of channels from in_channels to outut channels

                                        #linear transformation that projects the input tensor from a 2-dimensional feature space to a 5-dimensional feature space.
                                        # Each output channel of the convolution represents a learned combination of the input channels.
        x = torch.einsum('nctv,tvw->nctw', (x, A))



        '''nctv: This represents the input tensor x. It has four dimensions: n, c, t, and v. Here,

            n denotes the batch size,
            c denotes the number of channels (features),
            t denotes the number of time steps,
            v denotes the number of graph nodes (vertices).
            tvw: This represents the adjacency matrix tensor A. It has three dimensions: t, v, and w. Here,
            
            t denotes the number of time steps,
            v denotes the number of graph nodes (vertices),
            w denotes the number of graph nodes (vertices) as well.'''


        '''The einsum operation performs element-wise multiplication between the corresponding elements of the input tensors along the shared dimensions (t and v),
         effectively incorporating the graph structure into the convolutional operation.
         The resulting tensor has the same shape as the input tensor x, with an additional dimension w representing the graph structure.'''
        '''This operation allows the model to combine the information from the input tensor x with the graph structure encoded in the adjacency matrix A. 
        It effectively applies the graph convolutional operation on the input, taking into account the dependencies and relationships between the graph nodes.

In the context of pedestrian trajectory prediction, the einsum operation helps to capture the influence and interactions between pedestrians 
based on the graph structure defined by the adjacency matrix. By incorporating the graph information, the model can leverage the spatial and temporal 
dependencies of pedestrians' movements, improving the accuracy of the predictions'''



 #       simple example x = torch.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]])
 #                       A = torch.tensor([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]])

  #                      result = torch.einsum('nctv,tvw->nctw', (x, A))

   #               result =    tensor([[[[0.5, 0.6],
    #                                  [1.1, 1.3]],
     #
      #                               [[2.5, 2.6],
       #                               [3.5, 3.6]]]])
        return x.contiguous(), A   #elements of a tensor are stored in a continuous block of memory without any gaps or irregularities.
    

class st_gcn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel               (3,8 ) 3 for temporal kernel and 8/12 seqlength for spatial/graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,   #first diemsnion (temporal) should be odd to have  ceentral element when convolving, second dimension is for spatial
                 use_mdn = False,
                 stride=1,
                 dropout=0,
                 residual=True):
        super(st_gcn,self).__init__()
        
#         print("outstg",out_channels)

        assert len(kernel_size) == 2    # (temporal and graph convolving kernel)
        assert kernel_size[0] % 2 == 1 #This line checks if the first dimension of the kernel size is odd. This condition ensures the symmetric padding can be correctly applied.
        padding = ((kernel_size[0] - 1) // 2, 0) #The padding is calculated based on the kernel size. It uses asymmetric padding along the temporal dimension and no padding along the graph dimension.
        self.use_mdn = use_mdn #Mixture Density Network (MDN) as the activation function in the model.

                                #MDN is a probabilistic model that is commonly used in tasks involving generating or predicting probability distributions

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1]) ###apply graphical convolution thats why we take kernel_size[1]
        

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),    #only aPPLY  on temporal dimension
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

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
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.prelu = nn.PReLU()

    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.gcn(x, A)

        x = self.tcn(x) + res
        
        if not self.use_mdn:
            x = self.prelu(x)

        return x, A

class social_stgcnn(nn.Module):
    def __init__(self, n_stgcnn=2, n_txpcnn=5, input_feat=21, output_feat=2,
                 seq_len=8, pred_seq_len=12, kernel_size=3):


        super(social_stgcnn,self).__init__()
        self.n_stgcnn= n_stgcnn
        self.n_txpcnn = n_txpcnn #Temporal Convolutional Predictive Coding Neural Network
                
        self.st_gcns = nn.ModuleList()
        self.st_gcns.append(st_gcn(input_feat,output_feat,(kernel_size,seq_len)))
        for j in range(1,self.n_stgcnn):
            self.st_gcns.append(st_gcn(output_feat,output_feat,(kernel_size,seq_len)))    #layers are being added sequentially thats why output_feat,output_feat
        self.relu = nn.ReLU()
        self.tpcnns = nn.ModuleList()
        self.tpcnns.append(nn.Conv2d(seq_len,pred_seq_len,3,padding=1))
        for j in range(1,self.n_txpcnn):
            self.tpcnns.append(nn.Conv2d(pred_seq_len,pred_seq_len,3,padding=1))
        self.tpcnn_ouput = nn.Conv2d(pred_seq_len,pred_seq_len,3,padding=1)
            
            
        self.prelus = nn.ModuleList()
        for j in range(self.n_txpcnn):
            self.prelus.append(nn.PReLU())



        
    def calculate_paramter_constant_velocity(self,v,mask,mean_x,mean_y):
        epsilon = torch.tensor(1e-6)
        v[0, :, mask, 2]=  torch.log(torch.exp(epsilon) - 1)
        v[0, :, mask, 3] = torch.log(torch.exp(epsilon) - 1)
        v[0, :, mask, 4] =  0.0



        v[-1, :, mask, 0] = mean_x[mask]
        v[-1, :, mask, 1] =mean_y[mask]
       # print(v[0,2,:,mask])





        return v



        
    def forward(self,v,a):
        velocities_mean_x_y = v[-1, :2, :, :].mean(dim=1)
        mask = (torch.abs(velocities_mean_x_y[0, :]) <=0.01) & (torch.abs(velocities_mean_x_y[1, :]) <=0.01) #stationry if vecloitry is mean velocity is smaller than 0.01 m in 0.4 second
       # mask = condition.view(1, -1)
        mean_x, mean_y=v[-1,:2,-1,:]
        for k in range(self.n_stgcnn):
            v,a = self.st_gcns[k](v,a)







        v = v.view(v.shape[0],v.shape[2],v.shape[1],v.shape[3])
        
        v = self.prelus[0](self.tpcnns[0](v))

        for k in range(1,self.n_txpcnn-1):
            v =  self.prelus[k](self.tpcnns[k](v)) + v
            
        v = self.tpcnn_ouput(v)
        v = v.view(v.shape[0], v.shape[2], v.shape[1], v.shape[3])
        v =v.permute(0, 2, 3, 1)
        if not self.training:
            v = self.calculate_paramter_constant_velocity(v, mask,mean_x,mean_y)


        
        return v,a
        

