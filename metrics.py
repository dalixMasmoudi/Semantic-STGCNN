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

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from numpy import linalg as LA
import networkx as nx
import glob
import torch
from scipy.stats import linregress

import torch.nn as nn
import numpy as np
import torch.nn.functional as Func
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
torch.manual_seed(99)
import torch.optim as optim

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from numpy import linalg as LA
import networkx as nx

from utils_dali_torch import *
from metrics import *
import pickle
import argparse
from torch import autograd
import torch.optim.lr_scheduler as lr_scheduler
from model_dali import *
import torch.distributions.multivariate_normal as torchdist

def ade(predAll,targetAll,count_):
    All = len(predAll)
    sum_all = 0 
    for s in range(All):
        pred = np.swapaxes(predAll[s][:,:count_[s],:],0,1)
        target = np.swapaxes(targetAll[s][:,:count_[s],:],0,1)
        
        N = pred.shape[0]
        T = pred.shape[1]
        sum_ = 0 
        for i in range(N):
            for t in range(T):
                sum_+=math.sqrt((pred[i,t,0] - target[i,t,0])**2+(pred[i,t,1] - target[i,t,1])**2)
        sum_all += sum_/(N*T)
        
    return sum_all/All


def fde(predAll,targetAll,count_):
    All = len(predAll)
    sum_all = 0 
    for s in range(All):
        pred = np.swapaxes(predAll[s][:,:count_[s],:],0,1)
        target = np.swapaxes(targetAll[s][:,:count_[s],:],0,1)
        N = pred.shape[0]
        T = pred.shape[1]
        sum_ = 0 
        for i in range(N):
            for t in range(T-1,T):
                sum_+=math.sqrt((pred[i,t,0] - target[i,t,0])**2+(pred[i,t,1] - target[i,t,1])**2)
        sum_all += sum_/(N)

    return sum_all/All


def seq_to_nodes(seq_):
    max_nodes = seq_.shape[1] #number of pedestrians in the graph
    seq_ = seq_.squeeze()
    seq_len = seq_.shape[2]
    
    V = np.zeros((seq_len,max_nodes,2))
    for s in range(seq_len):
        step_ = seq_[:,:,s]
        for h in range(len(step_)): 
            V[s,h,:] = step_[h]
            
    return V.squeeze()

def nodes_rel_to_nodes_abs(nodes, init_node):
    nodes_ = np.zeros_like(nodes)
    for s in range(nodes.shape[0]):
        for ped in range(nodes.shape[1]):
            nodes_[s, ped, :2] = np.sum(nodes[:s+1, ped, :2], axis=0) + init_node[ped, :2]
            nodes_[s, ped, 2:] = nodes[s, ped, 2:]

    return nodes_.squeeze()

def closer_to_zero(current,new_v):
    dec =  min([(abs(current),current),(abs(new_v),new_v)])[1]
    if dec != current:
        return True
    else: 
        return False

'''def bivariate_loss(V_pred, V_trgt):
    # mux, muy, sx, sy, corr
    # assert V_pred.shape == V_trgt.shape
    normx = V_trgt[:, :, 0] - V_pred[:, :, 0]
    normy = V_trgt[:, :, 1] - V_pred[:, :, 1]

    sx = torch.exp(V_pred[:, :, 2])  # sx
    sy = torch.exp(V_pred[:, :, 3])  # sy
    corr = torch.tanh(V_pred[:, :, 4])  # corr

    sxsy = sx * sy

    z = (normx / sx) ** 2 + (normy / sy) ** 2 - 2 * ((corr * normx * normy) / sxsy)
    negRho = 1 - corr ** 2

    # Numerator
    result = torch.exp(-z / (2 * negRho))
    # Normalization factor
    denom = 2 * np.pi * (sxsy * torch.sqrt(negRho))

    # Final PDF calculation
    result = result / denom

    # Numerical stability
    epsilon = 1e-20

    result = -torch.log(torch.clamp(result, min=epsilon))
    result = torch.mean(result)

    return result'''


'''def bivariate_loss(V_pred, V_trgt):
    normx = V_trgt[:, :, 0] - V_pred[:, :, 0]
    normy = V_trgt[:, :, 1] - V_pred[:, :, 1]

    sx = torch.exp(V_pred[:, :, 2])  # sx
    sy = torch.exp(V_pred[:, :, 3])  # sy
    corr = torch.tanh(V_pred[:, :, 4])  # corr

    sxsy = sx * sy

    z = (normx / sx) ** 2 + (normy / sy) ** 2 - 2 * ((corr * normx * normy) / sxsy)
    negRho = 1 - corr ** 2

    # Numerator
    result = torch.exp(-z / (2 * negRho))
    # Normalization factor
    denom = 2 * np.pi * (sxsy * torch.sqrt(negRho))

    # Final PDF calculation
    result = result / denom

    # Numerical stability
    epsilon = 1e-20

    result = -torch.log(torch.clamp(result, min=epsilon))

    # Regularization term
    displacement_penalty = torch.abs(normx) + torch.abs(normy)
    displacement_penalty = displacement_penalty * torch.exp(-0.9 * displacement_penalty)



    # Combine the loss, regularization term, and smoothness penalty
    result = torch.mean(result) + torch.mean(displacement_penalty)

    return result'''
import torch
import numpy as np

import torch
import numpy as np

import torch
import numpy as np






'''def bivariate_loss(V_pred, V_trgt):
    normx = V_trgt[:, :, 0] - V_pred[:, :, 0]
    normy = V_trgt[:, :, 1] - V_pred[:, :, 1]

    sx = torch.exp(V_pred[:, :, 2])  # sx
    sy = torch.exp(V_pred[:, :, 3])  # sy
    corr = torch.tanh(V_pred[:, :, 4])  # corr

    sxsy = sx * sy

    z = (normx / sx) ** 2 + (normy / sy) ** 2 - 2 * ((corr * normx * normy) / sxsy)
    negRho = 1 - corr ** 2

    # Numerator
    result = torch.exp(-z / (2 * negRho))
    # Normalization factor
    denom = 2 * np.pi * (sxsy * torch.sqrt(negRho))

    # Final PDF calculation
    result = result / denom

    # Numerical stability
    epsilon = 1e-20

    result = -torch.log(torch.clamp(result, min=epsilon))

    # Linearity penalty along x-axis
    x_deviation_penalty = torch.abs(normx)

    # Apply linearity penalty
    result = result + x_deviation_penalty

    # Regularization term
    displacement_penalty = torch.sqrt(normx ** 2 + normy ** 2)

    # Combine the loss, linearity penalty, and regularization term
    result = torch.mean(result) + torch.mean(displacement_penalty)


    return result'''

import torch.nn.functional as F

def bivariate_loss(V_pred, V_trgt,non_linear_ped,V_obs):

    normx = V_trgt[:, :, 0] - V_pred[:, :, 0]
    normy = V_trgt[:, :, 1] - V_pred[:, :, 1]

    sx = torch.nn.functional.softplus(V_pred[:, :, 2]) # sx with maximum value

    sy = torch.nn.functional.softplus(V_pred[:, :, 3])
    corr = (2 / torch.pi) * torch.atan(V_pred[:, :, 4])


  #  print("Correlations between x and y coordinates for each pedestrian:")
   # print(correlations)
    sxsy = sx * sy


    z = (normx/sx)**2 + (normy/sy)**2 - 2*((corr*normx*normy)/sxsy)
    negRho = 1 - corr**2

    # Numerator
    result = torch.exp(-z/(2*negRho))
    # Normalization factor
    denom = 2 * np.pi * (sxsy * torch.sqrt(negRho))

    # Final PDF calculation
    result = result / denom

    # Numerical stability
    epsilon = 1e-20

    result = -torch.log(torch.clamp(result, min=epsilon))
    result = torch.mean(result)


   # distances = torch.cdist(V_pred[:, :, :2], V_trgt[:, :, :2], p=2)
    #l1_loss = nn.L1Loss()(V_pred[:,:,:2], V_trgt)

    # Combine the bivariate loss and L1 loss

    # Penalize the distance by adding to the loss
    # You can adjust this weight as needed
   # result +=  torch.mean(distances)+l1_loss

    return result


'''    linearity_penalty = torch.tensor(0.0, dtype=torch.float32).to(V_pred.device)
    for i, nl_ped in enumerate((non_linear_ped)[0]):
        if nl_ped == 1:
            # Penalize when the pedestrian trajectory is linear
            linearity_penalty += torch.mean(torch.abs(V_trgt[:, i, 0] - V_trgt[0, i, 0]))
            linearity_penalty += torch.mean(torch.abs(V_trgt[:, i, 1] - V_trgt[0, i, 1]))

    # Combine the loss, regularization terms, and linearity penalty
    result = torch.mean(result) + 5*linearity_penalty'''

'''   cov = torch.zeros(V_pred.shape[0], V_pred.shape[1], 2, 2).cuda()
    result = -torch.log(torch.clamp(result, min=epsilon))

    cov = torch.zeros(V_pred.shape[0], V_pred.shape[1], 2, 2).cuda()
    cov[:,:,0,0]= sx*sx
    cov[:,:,0,1]= corr*sx*sy
    cov[:,:,1,0]= corr*sx*sy
    cov[:,:,1,1]= sy*sy
    mean = V_pred[:,:,0:2]          #use V_pred paramters as inputs for the bivariate function to generate paths.
    #    print(mean,cov)
    mvnormal = torchdist.MultivariateNormal(mean,cov)
    V_prediction=mvnormal.sample()

    smoothness_penalty = torch.mean(torch.abs(V_prediction[1:, :, :2]- V_prediction[:-1, :, :2]))
    cov[:,:,0,0]= sx*sx
    cov[:,:,0,1]= corr*sx*sy
    cov[:,:,1,0]= corr*sx*sy
    cov[:,:,1,1]= sy*sy
    mean = V_pred[:,:,0:2]          #use V_pred paramters as inputs for the bivariate function to generate paths.
    #    print(mean,cov)
    mvnormal = torchdist.MultivariateNormal(mean,cov)
    V_prediction=mvnormal.sample()

    avg_distance =torch.min(torch.sqrt(torch.pow(V_trgt[1:, :, 0]-V_trgt[:-1, :, 0], 2)+torch.pow(V_trgt[1:, :, 1]-V_trgt[:-1, :, 1], 2)))#average stride for all pedestrians

    pred_distance = torch.mean(torch.sqrt(torch.pow(V_prediction[1:, :, 0]-V_prediction[:-1, :, 0], 2)+torch.pow(V_prediction[1:, :, 1]-V_prediction[:-1, :, 1], 2)))


  #  pred_distance =  torch.mean(torch.sqrt(torch.sum(torch.pow(V_pred[1:, 0, :2] -V_pred[:-1, 0, :2], 2),dim=1)))

    # Define a penalty based on the magnitude of the predicted step
    penalty = (torch.relu(pred_distance-avg_distance ))'''

'''def bivariate_loss(V_pred, V_trgt):
    # Calculate L2 distance between predicted and ground truth trajectories
    loss = torch.mean(torch.square(V_pred[:, :, :2] - V_trgt[:, :, :2]))
#
    return loss'''
#The result obtained from the bivariate_loss function measures the dissimilarity between the predicted and target distributions.
'''    #corr = (2 / torch.pi) * torch.atan(V_pred[:, :, 4]) # corr
    x_coords = V_trgt[:, :, 0]
    y_coords = V_trgt[:, :, 1]

    # Calculate the correlation between x and y coordinates for each pedestrian
    correlations = torch.zeros(V_trgt.size(1)).cuda()  # Initialize a tensor to store the correlations

    for i in range(V_trgt.size(1)):
        x_pedestrian_i = x_coords[:, i]
        y_pedestrian_i = y_coords[:, i]
        if torch.all(x_pedestrian_i == x_pedestrian_i[0]) or torch.all(y_pedestrian_i == y_pedestrian_i[0]):
            correlation = torch.tensor(1e-3)  # Set correlation to 0
        else:

            mean_x = torch.mean(x_pedestrian_i)
            mean_y = torch.mean(y_pedestrian_i)
            diff_x = x_pedestrian_i- mean_x
            diff_y = y_pedestrian_i - mean_y

            cov_xy = torch.sum(diff_x * diff_y)
            var_x = torch.sum(diff_x ** 2)
            var_y = torch.sum(diff_y ** 2)

            # Add epsilon to prevent division by zero
            epsilon = 1e-4
            correlation = cov_xy / (torch.sqrt(var_x + epsilon) * torch.sqrt(var_y + epsilon))
        correlations[i] = correlation

    corr = correlations.unsqueeze(0).repeat(V_trgt.size(0), 1).cuda()'''

'''    sx1 = torch.nn.functional.softplus(V_pred[:, :, 2]) # sx with maximum value
    sx = V_obs[0, :, :, 0].std(dim=0, keepdim=True)
    sy1 = torch.nn.functional.softplus(V_pred[:, :, 3])
    sy = V_obs[0, :, :, 1].std(dim=0, keepdim=True)
    epsilon = 1e-6
    sx = torch.where(sx > epsilon, sx, torch.full_like(sx, epsilon))
    sy = torch.where(sy > epsilon, sy, torch.full_like(sy, epsilon))
    corr = (2 / torch.pi) * torch.atan(V_pred[:, :, 4])'''
'''i tried different functoins for tanh tokeep range between -1 and 1 
Sigmoid (torch.sigmoid): The sigmoid function maps values to the range (0, 1). It is defined as sigmoid(x) = 1 / (1 + exp(-x)). You can scale the output of the sigmoid function to the range (-1, 1) by using the formula 2 * sigmoid(x) - 1.

Softsign (torch.nn.functional.softsign): The softsign function maps values to the range (-1, 1). It is defined as softsign(x) = x / (1 + |x|).

ArcTan (torch.atan): The arctangent function atan(x) maps values to the range (-π/2, π/2). You can scale the output of the arctangent function to the range (-1, 1) by using the formula 2/pi * atan(x).'''

'''
The negative log-likelihood is used as a loss function because it is a common choice for modeling the discrepancy between predicted probabilities and actual outcomes. It measures how well the predicted probabilities match the observed data.

In the context of the bivariate loss function, the computed PDF represents the probability density of the predicted values given the ground truth values. The goal is to minimize the difference between the predicted and ground truth values. By taking the negative logarithm of the PDF values, we transform the probabilities into a logarithmic scale and negate them.

The negative log-likelihood loss has several desirable properties:

It penalizes larger discrepancies between predicted and ground truth values more heavily. When the predicted values deviate significantly from the ground truth, the loss value will be larger.
It is differentiable, which allows for efficient gradient-based optimization methods to minimize the loss during training.
By minimizing the negative log-likelihood, we maximize the likelihood of the observed data given the predicted probabilities. In other words, we aim to find the parameters (in this case, the predicted values) that maximize the likelihood of the observed outcomes.
Overall, using the negative log-likelihood as a loss function is a common approach for probabilistic modeling tasks where the goal is to estimate the parameters that best fit the observed data.'''
