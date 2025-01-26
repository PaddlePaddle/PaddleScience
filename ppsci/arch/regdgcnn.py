# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Created on Mon May 29 22:18:28 2023

@author: Mohamed Elrefaie, mohamed.elrefaie@mit.edu, mohamed.elrefaie@tum.de

This module is part of the research presented in the paper
"DrivAerNet: A Parametric Car Dataset for Data-driven Aerodynamic Design and Graph-Based Drag Prediction".
It extends the work by introducing a Deep Graph Convolutional Neural Network (RegDGCNN) model for Regression Tasks,
specifically designed for processing 3D point cloud data of car models from the DrivAerNet dataset.

The RegDGCNN model utilizes a series of graph-based convolutional layers to effectively capture the complex geometric
and topological structure of 3D car models, facilitating advanced aerodynamic analyses and predictions.
The model architecture incorporates several techniques, including dynamic graph construction,
EdgeConv operations, and global feature aggregation, to robustly learn from graphs and point cloud data.

Parts of this code are modified from the original version authored by Yue Wang
"""

from __future__ import annotations

from typing import Dict
from typing import Tuple

import numpy as np
import paddle


def transpose_aux_func(dims, dim0, dim1):
    perm = list(range(dims))
    perm[dim0], perm[dim1] = perm[dim1], perm[dim0]
    return perm


class DataAugmentation:
    """
    Class encapsulating various data augmentation techniques for point clouds.
    """

    @staticmethod
    def translate_pointcloud(
        pointcloud: paddle.Tensor,
        translation_range: Tuple[float, float] = (2.0 / 3.0, 3.0 / 2.0),
    ) -> paddle.Tensor:
        """
        Translates the pointcloud by a random factor within a given range.

        Args:
            pointcloud: The input point cloud as a paddle.Tensor.
            translation_range: A tuple specifying the range for translation factors.

        Returns:
            Translated point cloud as a paddle.Tensor.
        """
        xyz1 = np.random.uniform(
            low=translation_range[0], high=translation_range[1], size=[3]
        )
        xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
        translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype(
            "float32"
        )
        return paddle.to_tensor(data=translated_pointcloud, dtype="float32")

    @staticmethod
    def jitter_pointcloud(
        pointcloud: paddle.Tensor, sigma: float = 0.01, clip: float = 0.02
    ) -> paddle.Tensor:
        """
        Adds Gaussian noise to the pointcloud.

        Args:
            pointcloud: The input point cloud as a paddle.Tensor.
            sigma: Standard deviation of the Gaussian noise.
            clip: Maximum absolute value for noise.

        Returns:
            Jittered point cloud as a paddle.Tensor.
        """
        N, C = tuple(pointcloud.shape)
        jittered_pointcloud = pointcloud + paddle.clip(
            x=sigma * paddle.randn(shape=[N, C]), min=-clip, max=clip
        )
        return jittered_pointcloud

    @staticmethod
    def drop_points(pointcloud: paddle.Tensor, drop_rate: float = 0.1) -> paddle.Tensor:
        """
        Randomly removes points from the point cloud based on the drop rate.

        Args:
            pointcloud: The input point cloud as a paddle.Tensor.
            drop_rate: The percentage of points to be randomly dropped.

        Returns:
            The point cloud with points dropped as a paddle.Tensor.
        """
        num_drop = int(drop_rate * pointcloud.shape[0])
        drop_indices = np.random.choice(pointcloud.shape[0], num_drop, replace=False)
        keep_indices = np.setdiff1d(np.arange(pointcloud.shape[0]), drop_indices)
        dropped_pointcloud = pointcloud[keep_indices, :]
        return dropped_pointcloud


def knn(x, k):
    """
    Computes the k-nearest neighbors for each point in x.

    Args:
        x (paddle.Tensor): The input tensor of shape (batch_size, num_dims, num_points).
        k (int): The number of nearest neighbors to find.

    Returns:
        paddle.Tensor: Indices of the k-nearest neighbors for each point, shape (batch_size, num_points, k).
    """
    inner = -2 * paddle.matmul(
        x=x.transpose(perm=transpose_aux_func(x.ndim, 2, 1)), y=x
    )
    xx = paddle.sum(x=x**2, axis=1, keepdim=True)
    pairwise_distance = (
        -xx - inner - xx.transpose(perm=transpose_aux_func(xx.ndim, 2, 1))
    )
    idx = pairwise_distance.topk(k=k, axis=-1)[1]
    return idx


def get_graph_feature(x, k=20, idx=None):
    """
    Constructs local graph features for each point by finding its k-nearest neighbors and
    concatenating the relative position vectors.

    Args:
        x (paddle.Tensor): The input tensor of shape (batch_size, num_dims, num_points).
        k (int): The number of neighbors to consider for graph construction.
        idx (paddle.Tensor, optional): Precomputed k-nearest neighbor indices.

    Returns:
        paddle.Tensor: The constructed graph features of shape (batch_size, 2*num_dims, num_points, k).
    """
    batch_size = x.shape[0]
    num_points = x.shape[2]
    x = x.reshape([batch_size, -1, num_points])
    if idx is None:
        idx = knn(x, k=k)
    idx_base = paddle.arange(start=0, end=batch_size).reshape([-1, 1, 1]) * num_points
    idx = idx + idx_base
    idx = idx.reshape([-1])
    _, num_dims, _ = tuple(x.shape)
    x = x.transpose(perm=transpose_aux_func(x.ndim, 2, 1)).contiguous()
    feature = x.reshape([batch_size * num_points, -1])[idx, :]
    feature = feature.reshape([batch_size, num_points, k, num_dims])
    x = x.reshape([batch_size, num_points, 1, num_dims]).tile(repeat_times=[1, 1, k, 1])
    feature = (
        paddle.concat(x=(feature - x, x), axis=3)
        .transpose(perm=[0, 3, 1, 2])
        .contiguous()
    )
    del x, idx, idx_base
    paddle.device.cuda.empty_cache()
    return feature


class RegDGCNN(paddle.nn.Layer):
    """Deep Graph Convolutional Neural Network for Regression Tasks (RegDGCNN) designed to process 3D point cloud data.

    This network architecture extracts hierarchical features from point clouds using graph-based convolutions,
    enabling effective learning of spatial structures.

    Args:
        input_keys (Tuple[str, ...]): Keys for input data fields.
        label_keys (Tuple[str, ...]): Keys for label data fields.
        weight_keys (Tuple[str, ...]): Keys for weight data fields.
        args (dict): Configuration parameters including:
            - 'k' (int): Number of neighbors for graph convolution.
            - 'emb_dims' (int): Embedding dimensions for feature aggregation.
            - 'dropout' (float): Dropout rate for regularization.
        output_channels (int, optional): Number of output channels. Defaults to 1.
    """

    def __init__(
        self,
        input_keys: Tuple[str, ...],
        label_keys: Tuple[str, ...],
        weight_keys: Tuple[str, ...],
        args: dict,
        output_channels=1,
    ):

        super(RegDGCNN, self).__init__()
        self.input_keys = input_keys
        self.label_keys = label_keys
        self.weight_keys = weight_keys
        self.args = args
        self.k = args["k"]
        self.bn1 = paddle.nn.BatchNorm2D(num_features=256)
        self.bn2 = paddle.nn.BatchNorm2D(num_features=512)
        self.bn3 = paddle.nn.BatchNorm2D(num_features=512)
        self.bn4 = paddle.nn.BatchNorm2D(num_features=1024)
        self.bn5 = paddle.nn.BatchNorm1D(num_features=args["emb_dims"])
        self.conv1 = paddle.nn.Sequential(
            paddle.nn.Conv2D(
                in_channels=6, out_channels=256, kernel_size=1, bias_attr=False
            ),
            self.bn1,
            paddle.nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv2 = paddle.nn.Sequential(
            paddle.nn.Conv2D(
                in_channels=256 * 2, out_channels=512, kernel_size=1, bias_attr=False
            ),
            self.bn2,
            paddle.nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv3 = paddle.nn.Sequential(
            paddle.nn.Conv2D(
                in_channels=512 * 2, out_channels=512, kernel_size=1, bias_attr=False
            ),
            self.bn3,
            paddle.nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv4 = paddle.nn.Sequential(
            paddle.nn.Conv2D(
                in_channels=512 * 2, out_channels=1024, kernel_size=1, bias_attr=False
            ),
            self.bn4,
            paddle.nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv5 = paddle.nn.Sequential(
            paddle.nn.Conv1D(
                in_channels=2304,
                out_channels=args["emb_dims"],
                kernel_size=1,
                bias_attr=False,
            ),
            self.bn5,
            paddle.nn.LeakyReLU(negative_slope=0.2),
        )
        self.linear1 = paddle.nn.Linear(
            in_features=args["emb_dims"] * 2, out_features=128, bias_attr=False
        )
        self.bn6 = paddle.nn.BatchNorm1D(num_features=128)
        self.dp1 = paddle.nn.Dropout(p=args["dropout"])
        self.linear2 = paddle.nn.Linear(in_features=128, out_features=64)
        self.bn7 = paddle.nn.BatchNorm1D(num_features=64)
        self.dp2 = paddle.nn.Dropout(p=args["dropout"])
        self.linear3 = paddle.nn.Linear(in_features=64, out_features=32)
        self.bn8 = paddle.nn.BatchNorm1D(num_features=32)
        self.dp3 = paddle.nn.Dropout(p=args["dropout"])
        self.linear4 = paddle.nn.Linear(in_features=32, out_features=16)
        self.bn9 = paddle.nn.BatchNorm1D(num_features=16)
        self.dp4 = paddle.nn.Dropout(p=args["dropout"])
        self.linear5 = paddle.nn.Linear(in_features=16, out_features=output_channels)

    def forward(self, x: paddle.Tensor) -> Dict[str, paddle.Tensor]:
        """
        Forward pass of the model to process input data and predict outputs.

        Args:
            x (paddle.Tensor): Input tensor representing a batch of point clouds.

        Returns:
            Dict[str, paddle.Tensor]: Model predictions for the input batch.

        """

        x = x[self.input_keys[0]]
        batch_size = x.shape[0]
        # Initialize an empty list to store the processed samples
        processed_samples = []
        # Apply data augmentation and normalization for each sample in the batch
        augmentation = DataAugmentation()
        for i in range(batch_size):
            sample = x[i].numpy()  # Convert to numpy array for data augmentation
            sample = augmentation.translate_pointcloud(sample)
            sample = augmentation.jitter_pointcloud(sample)
            processed_samples.append(sample)

        # Stack the processed samples back into a batch tensor
        x_processed = paddle.to_tensor(np.stack(processed_samples, axis=0))

        # Ensure the processed tensor has the same shape as the original input
        if x_processed.shape != x.shape:
            raise ValueError(
                f"Processed tensor shape {x_processed.shape} does not match original input shape {x.shape}"
            )
        x = x_processed.transpose(perm=[0, 2, 1])

        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(axis=-1, keepdim=False)
        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(axis=-1, keepdim=False)
        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(axis=-1, keepdim=False)
        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(axis=-1, keepdim=False)
        x = paddle.concat(x=(x1, x2, x3, x4), axis=1)
        x = self.conv5(x)
        x1 = paddle.nn.functional.adaptive_max_pool1d(x=x, output_size=1).reshape(
            [batch_size, -1]
        )
        x2 = paddle.nn.functional.adaptive_avg_pool1d(x=x, output_size=1).reshape(
            [batch_size, -1]
        )
        x = paddle.concat(x=(x1, x2), axis=1)
        x = paddle.nn.functional.leaky_relu(
            x=self.bn6(self.linear1(x)), negative_slope=0.2
        )
        x = self.dp1(x)
        x = paddle.nn.functional.leaky_relu(
            x=self.bn7(self.linear2(x)), negative_slope=0.2
        )
        x = self.dp2(x)
        x = paddle.nn.functional.leaky_relu(
            x=self.bn8(self.linear3(x)), negative_slope=0.2
        )
        x = self.dp3(x)
        x = paddle.nn.functional.leaky_relu(
            x=self.bn9(self.linear4(x)), negative_slope=0.2
        )
        x = self.dp4(x)
        x = self.linear5(x)
        return {self.label_keys[0]: x}
