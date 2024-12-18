import sys
sys.path.append('/ssd1/ken/DrivAerNet-paddle-convert-main/utils')
import paddle_aux
import paddle
import os
"""
@author: Mohamed Elrefaie, mohamed.elrefaie@mit.edu mohamed.elrefaie@tum.de

This module is part of the research presented in the paper:
"DrivAerNet++: A Large-Scale Multimodal Car Dataset with Computational Fluid Dynamics Simulations and Deep Learning Benchmarks".

This module is used to define both point-cloud based and graph-based models, including RegDGCNN, PointNet, and several Graph Neural Network (GNN) models
for the task of surrogate modeling of the aerodynamic drag.
"""
import copy
import math
import numpy as np
import trimesh
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool, JumpingKnowledge
from torch_geometric.nn import BatchNorm


def knn(x, k):
    """
    Computes the k-nearest neighbors for each point in x.

    Args:
        x (torch.Tensor): The input tensor of shape (batch_size, num_dims, num_points).
        k (int): The number of nearest neighbors to find.

    Returns:
        torch.Tensor: Indices of the k-nearest neighbors for each point, shape (batch_size, num_points, k).
    """
    inner = -2 * paddle.matmul(x=x.transpose(perm=paddle_aux.
        transpose_aux_func(x.ndim, 2, 1)), y=x)
    xx = paddle.sum(x=x ** 2, axis=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(perm=paddle_aux.
        transpose_aux_func(xx.ndim, 2, 1))
    idx = pairwise_distance.topk(k=k, axis=-1)[1]
    return idx


def get_graph_feature(x, k=20, idx=None):
    """
    Constructs local graph features for each point by finding its k-nearest neighbors and
    concatenating the relative position vectors.

    Args:
        x (torch.Tensor): The input tensor of shape (batch_size, num_dims, num_points).
        k (int): The number of neighbors to consider for graph construction.
        idx (torch.Tensor, optional): Precomputed k-nearest neighbor indices.

    Returns:
        torch.Tensor: The constructed graph features of shape (batch_size, 2*num_dims, num_points, k).
    """
    batch_size = x.shape[0]
    num_points = x.shape[2]
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)
    device = x.place
    idx_base = paddle.arange(start=0, end=batch_size).view(-1, 1, 1
        ) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = tuple(x.shape)
    x = x.transpose(perm=paddle_aux.transpose_aux_func(x.ndim, 2, 1)
        ).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).tile(repeat_times=[1, 1,
        k, 1])
    feature = paddle.concat(x=(feature - x, x), axis=3).transpose(perm=[0, 
        3, 1, 2]).contiguous()
    return feature


class RegDGCNN(paddle.nn.Layer):
    """
    Deep Graph Convolutional Neural Network for Regression Tasks (RegDGCNN) for processing 3D point cloud data.

    This network architecture extracts hierarchical features from point clouds using graph-based convolutions,
    enabling effective learning of spatial structures.
    """

    def __init__(self, args, output_channels=1):
        """
        Initializes the RegDGCNN model with specified configurations.

        Args:
            args (dict): Configuration parameters including 'k' for the number of neighbors, 'emb_dims' for embedding
            dimensions, and 'dropout' rate.
            output_channels (int): Number of output channels (e.g., for drag prediction, this is 1).
        """
        super(RegDGCNN, self).__init__()
        self.args = args
        self.k = args['k']
        self.bn1 = paddle.nn.BatchNorm2D(num_features=256)
        self.bn2 = paddle.nn.BatchNorm2D(num_features=512)
        self.bn3 = paddle.nn.BatchNorm2D(num_features=512)
        self.bn4 = paddle.nn.BatchNorm2D(num_features=1024)
        self.bn5 = paddle.nn.BatchNorm1D(num_features=args['emb_dims'])
        self.conv1 = paddle.nn.Sequential(paddle.nn.Conv2D(in_channels=6,
            out_channels=256, kernel_size=1, bias_attr=False), self.bn1,
            paddle.nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = paddle.nn.Sequential(paddle.nn.Conv2D(in_channels=256 *
            2, out_channels=512, kernel_size=1, bias_attr=False), self.bn2,
            paddle.nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = paddle.nn.Sequential(paddle.nn.Conv2D(in_channels=512 *
            2, out_channels=512, kernel_size=1, bias_attr=False), self.bn3,
            paddle.nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = paddle.nn.Sequential(paddle.nn.Conv2D(in_channels=512 *
            2, out_channels=1024, kernel_size=1, bias_attr=False), self.bn4,
            paddle.nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = paddle.nn.Sequential(paddle.nn.Conv1D(in_channels=2304,
            out_channels=args['emb_dims'], kernel_size=1, bias_attr=False),
            self.bn5, paddle.nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = paddle.nn.Linear(in_features=args['emb_dims'] * 2,
            out_features=128, bias_attr=False)
        self.bn6 = paddle.nn.BatchNorm1D(num_features=128)
        self.dp1 = paddle.nn.Dropout(p=args['dropout'])
        self.linear2 = paddle.nn.Linear(in_features=128, out_features=64)
        self.bn7 = paddle.nn.BatchNorm1D(num_features=64)
        self.dp2 = paddle.nn.Dropout(p=args['dropout'])
        self.linear3 = paddle.nn.Linear(in_features=64, out_features=32)
        self.bn8 = paddle.nn.BatchNorm1D(num_features=32)
        self.dp3 = paddle.nn.Dropout(p=args['dropout'])
        self.linear4 = paddle.nn.Linear(in_features=32, out_features=16)
        self.bn9 = paddle.nn.BatchNorm1D(num_features=16)
        self.dp4 = paddle.nn.Dropout(p=args['dropout'])
        self.linear5 = paddle.nn.Linear(in_features=16, out_features=
            output_channels)

    def forward(self, x):
        """
        Forward pass of the model to process input data and predict outputs.

        Args:
            x (torch.Tensor): Input tensor representing a batch of point clouds.

        Returns:
            torch.Tensor: Model predictions for the input batch.
        """
        batch_size = x.shape[0]
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]
        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]
        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]
        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]
        x = paddle.concat(x=(x1, x2, x3, x4), axis=1)
        x = self.conv5(x)
        x1 = paddle.nn.functional.adaptive_max_pool1d(x=x, output_size=1).view(
            batch_size, -1)
        x2 = paddle.nn.functional.adaptive_avg_pool1d(x=x, output_size=1).view(
            batch_size, -1)
        x = paddle.concat(x=(x1, x2), axis=1)
        x = paddle.nn.functional.leaky_relu(x=self.bn6(self.linear1(x)),
            negative_slope=0.2)
        x = self.dp1(x)
        x = paddle.nn.functional.leaky_relu(x=self.bn7(self.linear2(x)),
            negative_slope=0.2)
        x = self.dp2(x)
        x = paddle.nn.functional.leaky_relu(x=self.bn8(self.linear3(x)),
            negative_slope=0.2)
        x = self.dp3(x)
        x = paddle.nn.functional.leaky_relu(x=self.bn9(self.linear4(x)),
            negative_slope=0.2)
        x = self.dp4(x)
        x = self.linear5(x)
        return x


class RegPointNet(paddle.nn.Layer):
    """
    PointNet-based regression model for 3D point cloud data.

    Args:
        args (dict): Configuration parameters including 'emb_dims' for embedding dimensions and 'dropout' rate.

    Methods:
        forward(x): Forward pass through the network.
    """

    def __init__(self, args):
        """
        Initialize the RegPointNet model for regression tasks with enhanced complexity,
        including additional layers and residual connections.

        Parameters:
            emb_dims (int): Dimensionality of the embedding space.
            dropout (float): Dropout probability.
        """
        super(RegPointNet, self).__init__()
        self.args = args
        self.conv1 = paddle.nn.Conv1D(in_channels=3, out_channels=512,
            kernel_size=1, bias_attr=False)
        self.conv2 = paddle.nn.Conv1D(in_channels=512, out_channels=1024,
            kernel_size=1, bias_attr=False)
        self.conv3 = paddle.nn.Conv1D(in_channels=1024, out_channels=1024,
            kernel_size=1, bias_attr=False)
        self.conv4 = paddle.nn.Conv1D(in_channels=1024, out_channels=1024,
            kernel_size=1, bias_attr=False)
        self.conv5 = paddle.nn.Conv1D(in_channels=1024, out_channels=1024,
            kernel_size=1, bias_attr=False)
        self.conv6 = paddle.nn.Conv1D(in_channels=1024, out_channels=args[
            'emb_dims'], kernel_size=1, bias_attr=False)
        self.bn1 = paddle.nn.BatchNorm1D(num_features=512)
        self.bn2 = paddle.nn.BatchNorm1D(num_features=1024)
        self.bn3 = paddle.nn.BatchNorm1D(num_features=1024)
        self.bn4 = paddle.nn.BatchNorm1D(num_features=1024)
        self.bn5 = paddle.nn.BatchNorm1D(num_features=1024)
        self.bn6 = paddle.nn.BatchNorm1D(num_features=args['emb_dims'])
        self.dropout_conv = paddle.nn.Dropout(p=args['dropout'])
        self.dropout_linear = paddle.nn.Dropout(p=args['dropout'])
        self.conv_shortcut = paddle.nn.Conv1D(in_channels=3, out_channels=
            args['emb_dims'], kernel_size=1, bias_attr=False)
        self.bn_shortcut = paddle.nn.BatchNorm1D(num_features=args['emb_dims'])
        self.linear1 = paddle.nn.Linear(in_features=args['emb_dims'],
            out_features=512, bias_attr=False)
        self.bn7 = paddle.nn.BatchNorm1D(num_features=512)
        self.linear2 = paddle.nn.Linear(in_features=512, out_features=256,
            bias_attr=False)
        self.bn8 = paddle.nn.BatchNorm1D(num_features=256)
        self.linear3 = paddle.nn.Linear(in_features=256, out_features=128)
        self.bn9 = paddle.nn.BatchNorm1D(num_features=128)
        self.linear4 = paddle.nn.Linear(in_features=128, out_features=64)
        self.bn10 = paddle.nn.BatchNorm1D(num_features=64)
        self.final_linear = paddle.nn.Linear(in_features=64, out_features=1)

    def forward(self, x):
        """
        Forward pass of the network.

        Parameters:
            x (Tensor): Input tensor of shape (batch_size, 3, num_points).

        Returns:
            Tensor: Output tensor of the predicted scalar value.
        """
        shortcut = self.bn_shortcut(self.conv_shortcut(x))
        x = paddle.nn.functional.relu(x=self.bn1(self.conv1(x)))
        x = self.dropout_conv(x)
        x = paddle.nn.functional.relu(x=self.bn2(self.conv2(x)))
        x = self.dropout_conv(x)
        x = paddle.nn.functional.relu(x=self.bn3(self.conv3(x)))
        x = self.dropout_conv(x)
        x = paddle.nn.functional.relu(x=self.bn4(self.conv4(x)))
        x = self.dropout_conv(x)
        x = paddle.nn.functional.relu(x=self.bn5(self.conv5(x)))
        x = self.dropout_conv(x)
        x = paddle.nn.functional.relu(x=self.bn6(self.conv6(x)))
        x = x + shortcut
        x = paddle.nn.functional.adaptive_max_pool1d(x=x, output_size=1
            ).squeeze(axis=-1)
        x = paddle.nn.functional.relu(x=self.bn7(self.linear1(x)))
        x = paddle.nn.functional.relu(x=self.bn8(self.linear2(x)))
        x = paddle.nn.functional.relu(x=self.bn9(self.linear3(x)))
        x = paddle.nn.functional.relu(x=self.bn10(self.linear4(x)))
        features = x
        x = self.final_linear(x)
        return x


class DragGNN(paddle.nn.Layer):
    """
    Graph Neural Network for predicting drag coefficients using GCNConv layers.

    Args:
        None

    Methods:
        forward(data): Forward pass through the network.
    """

    def __init__(self):
        super(DragGNN, self).__init__()
        self.conv1 = GCNConv(3, 512)
        self.conv2 = GCNConv(512, 1024)
        self.conv3 = GCNConv(1024, 512)
        self.fc1 = paddle.nn.Linear(in_features=512, out_features=128)
        self.fc2 = paddle.nn.Linear(in_features=128, out_features=1)

    def forward(self, data: Data) ->paddle.Tensor:
        """
        Forward pass through the network.

        Args:
            data (Data): Input graph data containing node features, edge indices, and batch indices.

        Returns:
            torch.Tensor: Output predictions for drag coefficients.
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = paddle.nn.functional.relu(x=self.conv1(x, edge_index))
        x = paddle.nn.functional.relu(x=self.conv2(x, edge_index))
        x = paddle.nn.functional.relu(x=self.conv3(x, edge_index))
        x = global_mean_pool(x, batch)
        x = paddle.nn.functional.relu(x=self.fc1(x))
        x = self.fc2(x)
        return x


class DragGNN_XL(paddle.nn.Layer):
    """
    Extended Graph Neural Network for predicting drag coefficients using GCNConv layers and BatchNorm layers.

    Args:
        None

    Methods:
        forward(data): Forward pass through the network.
    """

    def __init__(self):
        super(DragGNN_XL, self).__init__()
        self.conv1 = GCNConv(3, 64)
        self.conv2 = GCNConv(64, 128)
        self.conv3 = GCNConv(128, 128)
        self.conv4 = GCNConv(128, 256)
        self.bn1 = BatchNorm(64)
        self.bn2 = BatchNorm(128)
        self.bn3 = BatchNorm(128)
        self.bn4 = BatchNorm(256)
        self.dropout = paddle.nn.Dropout(p=0.4)
        self.fc = paddle.nn.Sequential(paddle.nn.Linear(in_features=256,
            out_features=128), paddle.nn.ReLU(), paddle.nn.Dropout(p=0.4),
            paddle.nn.Linear(in_features=128, out_features=64), paddle.nn.
            ReLU(), paddle.nn.Linear(in_features=64, out_features=1))

    def forward(self, data: Data) ->paddle.Tensor:
        """
        Forward pass through the network.

        Args:
            data (Data): Input graph data containing node features, edge indices, and batch indices.

        Returns:
            torch.Tensor: Output predictions for drag coefficients.
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = paddle.nn.functional.relu(x=self.bn1(self.conv1(x, edge_index)))
        x = self.dropout(x)
        x = paddle.nn.functional.relu(x=self.bn2(self.conv2(x, edge_index)))
        x = self.dropout(x)
        x = paddle.nn.functional.relu(x=self.bn3(self.conv3(x, edge_index)))
        x = self.dropout(x)
        x = paddle.nn.functional.relu(x=self.bn4(self.conv4(x, edge_index)))
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return x


class EnhancedDragGNN(paddle.nn.Layer):
    """
    Enhanced Graph Neural Network for predicting drag coefficients using both GCNConv and GATConv layers,
    with Jumping Knowledge for combining features from different layers.

    Args:
        None

    Methods:
        forward(data): Forward pass through the network.
    """

    def __init__(self):
        super(EnhancedDragGNN, self).__init__()
        self.gcn1 = GCNConv(3, 64)
        self.gat1 = GATConv(64, 64, heads=4, concat=True)
        self.bn1 = paddle.nn.BatchNorm1D(num_features=128)
        self.gcn2 = GCNConv(256, 128)
        self.gat2 = GATConv(128, 128, heads=2, concat=True)
        self.bn2 = paddle.nn.BatchNorm1D(num_features=256)
        self.gcn3 = GCNConv(256, 256)
        self.jk = JumpingKnowledge(mode='cat')
        self.fc1 = paddle.nn.Sequential(paddle.nn.Linear(in_features=256 * 
            3, out_features=128), paddle.nn.ReLU(), paddle.nn.BatchNorm1D(
            num_features=128))
        self.fc2 = paddle.nn.Linear(in_features=128, out_features=1)

    def forward(self, data: Data) ->paddle.Tensor:
        """
        Forward pass through the network.

        Args:
            data (Data): Input graph data containing node features, edge indices, and batch indices.

        Returns:
            torch.Tensor: Output predictions for drag coefficients.
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x1 = paddle.nn.functional.relu(x=self.gcn1(x, edge_index))
        x1 = paddle.nn.functional.dropout(x=x1, p=0.2, training=self.training)
        x1 = self.gat1(x1, edge_index)
        x2 = paddle.nn.functional.relu(x=self.bn1(self.gcn2(x1, edge_index)))
        x2 = paddle.nn.functional.dropout(x=x2, p=0.2, training=self.training)
        x2 = self.gat2(x2, edge_index)
        x3 = paddle.nn.functional.relu(x=self.bn2(self.gcn3(x2, edge_index)))
        x = self.jk([x1, x2, x3])
        x = global_mean_pool(x, batch)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
