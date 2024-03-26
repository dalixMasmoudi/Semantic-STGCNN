import os
import math
import sys
import matplotlib.pyplot as plt
import torch
from scipy.stats import linregress
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
from tqdm import tqdm
import time
from sklearn.preprocessing import MinMaxScaler


def anorm(p1,p2): 
    NORM = math.sqrt((p1[0]-p2[0])**2+ (p1[1]-p2[1])**2)
    if NORM ==0:
        return 0
    return 1/(NORM)


def seq_to_graph_traj(seq_, seq_rel, norm_lap_matr=True):
    seq_ = seq_.squeeze()
    seq_rel = seq_rel.squeeze()
    seq_len = seq_.shape[2]
    max_nodes = seq_.shape[0]

    V = np.zeros((seq_len, max_nodes, 2))
    A = np.zeros((seq_len, max_nodes, max_nodes))
    for s in range(seq_len):
        step_ = seq_[:, :, s]
        step_rel = seq_rel[:, :, s]
        for h in range(len(step_)):
            V[s, h, :] = step_rel[h]
            A[s, h, h] = 1
            for k in range(h + 1, len(step_)):
                l2_norm = anorm(step_rel[h],
                                step_rel[k])  # If your nodes represent spatial locations or coordinates, computing
                # the L2 distance between nodes can capture the notion of proximity or similarity between them.
                # Nodes that are close to each other in terms of L2 distance will have a higher adjacency value, indicating a stronger connection or relationship.
                A[s, h, k] = l2_norm
                A[s, k, h] = l2_norm
        if norm_lap_matr:
            G = nx.from_numpy_array(A[s, :, :])
            A[s, :, :] = nx.normalized_laplacian_matrix(G).toarray()
    return torch.from_numpy(V).type(torch.float), \
        torch.from_numpy(A).type(torch.float)

def seq_to_graph_obs(seq_,seq_rel,node_features,norm_lap_matr = True):

    seq_ = seq_.squeeze()
    seq_rel = seq_rel.squeeze()
    seq_len = seq_.shape[2]
    max_nodes = seq_.shape[0]


    V = np.zeros((seq_len,max_nodes,21))
    A = np.zeros((seq_len,max_nodes,max_nodes))
    for s in range(seq_len):
        step_ = seq_[:,:,s]
        step_rel = seq_rel[:,:,s]
        node_features_step = node_features[:, :, s]
        for h in range(len(step_)):
            V[s, h, 0:2] = step_rel[h]
            V[s, h, 2:] = node_features_step[h]
            A[s,h,h] = 1
            for k in range(h+1,len(step_)):
                l2_norm = anorm(step_rel[h],step_rel[k]) #If your nodes represent spatial locations or coordinates, computing
                                                            # the L2 distance between nodes can capture the notion of proximity or similarity between them.
                                                             # Nodes that are close to each other in terms of L2 distance will have a higher adjacency value, indicating a stronger connection or relationship.
                A[s,h,k] = l2_norm
                A[s,k,h] = l2_norm
        if norm_lap_matr:
            G = nx.from_numpy_array(A[s,:,:])
            A[s,:,:] = nx.normalized_laplacian_matrix(G).toarray()

    return torch.from_numpy(V).type(torch.float),\
           torch.from_numpy(A).type(torch.float)


def poly_fit(traj, traj_len, threshold):
#1/True is linear
        x = traj[0, :traj_len]
        y = traj[1,:traj_len]

        # Check if x is constant and y is moving
        if np.all(x == x[0]) and not np.all(y == y[0]):
            return True

        # Check if y is constant and x is moving
        if np.all(y == y[0]) and not np.all(x == x[0]):
            return True
        x_threshold = np.abs(x - x[0]) <= 1
        y_threshold = np.abs(y - y[0]) <= 1
        # Return True if both x and y meet the threshold condition
        return np.all(x_threshold) or np.all(y_threshold)
        # Check if both x and y show a linear relationship
        x_slope, x_intercept, _, _, _ = linregress(range(len(x)), x)
        y_slope, y_intercept, _, _, _ = linregress(range(len(y)), y)

        if np.isclose(x_slope, 0) and np.isclose(y_slope, 0):
            return True

        return False

def read_file(_path, delim='\t'):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)


class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(
        self, data_dir, obs_len=8, pred_len=12, skip=1, threshold=0.002,
        min_ped=1, delim='\t',norm_lap_matr = True):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files
        """
        super(TrajectoryDataset, self).__init__()

        self.max_peds_in_frame = 0
        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim
        self.norm_lap_matr = norm_lap_matr

        all_files = os.listdir(self.data_dir)
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []
        loss_mask_list = []
        non_linear_ped = []
        nodes_features = []
        for path in all_files:
            data = read_file(path, delim)



            # Assuming 'data' is your dataset
            # Create an instance of MinMaxScaler
         #   scaler = MinMaxScaler()

            # Fit the scaler on your data
          #  scaler.fit(data[:, 5:] )

            # Transform the data using the fitted scaler
           # normalized_data = scaler.transform(data[:, 5:] )
            '''            min_vals = np.min(data[:, 5:],
                              axis=0)  # Compute minimum values for each column (excluding the first two columns)
            max_vals = np.max(data[:, 5:],
                              axis=0)  # Compute maximum values for each column (excluding the first two columns)

            # Broadcast min_vals and max_vals to have the same shape as 'data'
            min_vals_broadcast = np.expand_dims(min_vals, axis=0)
            max_vals_broadcast = np.expand_dims(max_vals, axis=0)

            # Add a small epsilon value to the denominator to avoid division by zero
            epsilon = 1e-8
            normalized_data = (data[:, 5:] - min_vals_broadcast) / (max_vals_broadcast - min_vals_broadcast + epsilon)

            # Assign the normalized values back to the original data array
            data[:, 5:] = normalized_data'''



          #  data[:, 2:4] = data[:, 2:4] / 100
        #    data[:, 2:] = data[:, 2:] /100
            frames = np.unique(data[:, 0]).tolist()
            frame_data = []
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])
            num_sequences = int(
                math.ceil((len(frames) - self.seq_len + 1) / skip))

            for idx in range(0, num_sequences * self.skip + 1, skip):
                curr_seq_data = np.concatenate(
                    frame_data[idx:idx + self.seq_len], axis=0)
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                self.max_peds_in_frame = max(self.max_peds_in_frame,len(peds_in_curr_seq))
                curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2,
                                         self.seq_len))
                curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_loss_mask = np.zeros((len(peds_in_curr_seq),
                                           self.seq_len))
                num_peds_considered = 0
                _non_linear_ped = []
                curr_node_features = np.zeros((len(peds_in_curr_seq), 19, self.seq_len))
                for _, ped_id in enumerate(peds_in_curr_seq):
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] ==
                                                 ped_id, :]
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                    if pad_end - pad_front != self.seq_len:
                        continue
                    curr_ped_node_features =np.transpose(curr_ped_seq[:, 4:])
                    curr_ped_seq = np.transpose(curr_ped_seq[:, 2:4])
                    curr_ped_seq = curr_ped_seq
                    # Make coordinates relative
                    rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                    rel_curr_ped_seq[:, 1:] = \
                        curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
                    _idx = num_peds_considered
                    if curr_ped_seq.shape[1] != 20: ##################### super wichtig
                        continue
                    curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                    curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
                    curr_node_features[_idx, :, pad_front:pad_end] = curr_ped_node_features
                    # Linear vs Non-Linear Trajectory
                    _non_linear_ped.append(
                        poly_fit(curr_ped_seq, pred_len, threshold))
                    curr_loss_mask[_idx, pad_front:pad_end] = 1
                    num_peds_considered += 1

                if num_peds_considered > min_ped:
                    non_linear_ped += _non_linear_ped
                    num_peds_in_seq.append(num_peds_considered)
                    loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                    seq_list.append(curr_seq[:num_peds_considered])
                    seq_list_rel.append(curr_seq_rel[:num_peds_considered])
                    nodes_features.append(curr_node_features[:num_peds_considered])

        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        non_linear_ped = np.asarray(non_linear_ped)
        nodes_features = np.concatenate(nodes_features, axis=0)

        # Convert numpy -> Torch Tensor
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
        self.node_features_obs=nodes_features [:, :, :self.obs_len]
        self.node_features_traj = nodes_features[:, :, self.obs_len:]
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [
            (start, end)
            for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]
        #Convert to Graphs
        self.v_obs = []
        self.A_obs = []
        self.v_pred = []
        self.A_pred = []
        print("Processing Data .....")
        pbar = tqdm(total=len(self.seq_start_end))
        for ss in range(len(self.seq_start_end)):#we take each sequence seperately.
            pbar.update(1)

            start, end = self.seq_start_end[ss]
          #  print(self.obs_traj[start:end,:])
            v_,a_ = seq_to_graph_obs(self.obs_traj[start:end,:],self.obs_traj_rel[start:end, :],self.node_features_obs[start:end, :],self.norm_lap_matr)
       #     visualize_graph(v_.numpy(), a_.numpy())
            self.v_obs.append(v_.clone()) #v_ is only the spatial graph. New graph for every time step
            self.A_obs.append(a_.clone())
            v_,a_=seq_to_graph_traj(self.pred_traj[start:end,:],self.pred_traj_rel[start:end, :],self.norm_lap_matr)
            self.v_pred.append(v_.clone())
            self.A_pred.append(a_.clone())
        #    self.A_pred.append(a_.clone())
        pbar.close()

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]

        out = [
            self.obs_traj[start:end, :], self.pred_traj[start:end, :],
            self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
            self.non_linear_ped[start:end], self.loss_mask[start:end, :],
            self.v_obs[index], self.A_obs[index],
            self.v_pred[index], self.A_pred[index]

        ]
        return out
