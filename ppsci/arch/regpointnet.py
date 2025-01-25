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

# Copyright 2024 Mohamed Elrefaie
"""
@author: Mohamed Elrefaie, mohamed.elrefaie@mit.edu mohamed.elrefaie@tum.de

This module is part of the research presented in the paper:
"DrivAerNet++: A Large-Scale Multimodal Car Dataset with Computational Fluid Dynamics Simulations and Deep Learning Benchmarks".

This module is used to define point-cloud models, includingPointNet
for the task of surrogate modeling of the aerodynamic drag.
"""

from __future__ import annotations

from typing import Dict
from typing import Tuple

import numpy as np
import paddle


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


class RegPointNet(paddle.nn.Layer):
    """
    PointNet-based regression model for 3D point cloud data.

    This network architecture is designed to process 3D point cloud data using a series of convolutional layers,
    followed by fully connected layers, enabling effective learning of spatial structures and features.

    Args:
        input_keys (Tuple[str, ...]): Keys for input data fields.
        label_keys (Tuple[str, ...]): Keys for label data fields.
        weight_keys (Tuple[str, ...]): Keys for weight data fields.
        args (dict): Configuration parameters including:
            - 'emb_dims' (int): Dimensionality of the embedding space.
            - 'dropout' (float): Dropout probability.

    Methods:
        forward(x): Forward pass through the network.
    """

    def __init__(
        self,
        input_keys: Tuple[str, ...],
        label_keys: Tuple[str, ...],
        weight_keys: Tuple[str, ...],
        args,
    ):
        super().__init__()
        self.input_keys = input_keys
        self.label_keys = label_keys
        self.weight_keys = weight_keys
        self.args = args
        self.conv1 = paddle.nn.Conv1D(
            in_channels=3, out_channels=512, kernel_size=1, bias_attr=False
        )
        self.conv2 = paddle.nn.Conv1D(
            in_channels=512, out_channels=1024, kernel_size=1, bias_attr=False
        )
        self.conv3 = paddle.nn.Conv1D(
            in_channels=1024, out_channels=1024, kernel_size=1, bias_attr=False
        )
        self.conv4 = paddle.nn.Conv1D(
            in_channels=1024, out_channels=1024, kernel_size=1, bias_attr=False
        )
        self.conv5 = paddle.nn.Conv1D(
            in_channels=1024, out_channels=1024, kernel_size=1, bias_attr=False
        )
        self.conv6 = paddle.nn.Conv1D(
            in_channels=1024,
            out_channels=args["emb_dims"],
            kernel_size=1,
            bias_attr=False,
        )
        self.bn1 = paddle.nn.BatchNorm1D(num_features=512)
        self.bn2 = paddle.nn.BatchNorm1D(num_features=1024)
        self.bn3 = paddle.nn.BatchNorm1D(num_features=1024)
        self.bn4 = paddle.nn.BatchNorm1D(num_features=1024)
        self.bn5 = paddle.nn.BatchNorm1D(num_features=1024)
        self.bn6 = paddle.nn.BatchNorm1D(num_features=args["emb_dims"])
        self.dropout_conv = paddle.nn.Dropout(p=args["dropout"])
        self.dropout_linear = paddle.nn.Dropout(p=args["dropout"])
        self.conv_shortcut = paddle.nn.Conv1D(
            in_channels=3, out_channels=args["emb_dims"], kernel_size=1, bias_attr=False
        )
        self.bn_shortcut = paddle.nn.BatchNorm1D(num_features=args["emb_dims"])
        self.linear1 = paddle.nn.Linear(
            in_features=args["emb_dims"], out_features=512, bias_attr=False
        )
        self.bn7 = paddle.nn.BatchNorm1D(num_features=512)
        self.linear2 = paddle.nn.Linear(
            in_features=512, out_features=256, bias_attr=False
        )
        self.bn8 = paddle.nn.BatchNorm1D(num_features=256)
        self.linear3 = paddle.nn.Linear(in_features=256, out_features=128)
        self.bn9 = paddle.nn.BatchNorm1D(num_features=128)
        self.linear4 = paddle.nn.Linear(in_features=128, out_features=64)
        self.bn10 = paddle.nn.BatchNorm1D(num_features=64)
        self.final_linear = paddle.nn.Linear(in_features=64, out_features=1)

    def min_max_normalize(self, data: paddle.Tensor) -> paddle.Tensor:
        """
        Normalizes the data to the range [0, 1] based on min and max values.
        """
        min_vals = data.min(axis=0, keepdim=True)
        max_vals = data.max(axis=0, keepdim=True)
        normalized_data = (data - min_vals) / (max_vals - min_vals)
        return normalized_data

    def forward(self, x) -> Dict[str, paddle.Tensor]:
        """
        Forward pass of the network.

        Args:
            x (paddle.Tensor): Input tensor of shape (batch_size, 3, num_points).

        Returns:
            Dict[str, paddle.Tensor]: A dictionary where the key is the first element of `self.label_keys`
                                       and the value is the output tensor of the predicted scalar value.
        """

        x = x[self.input_keys[0]]
        bs = x.shape[0]

        # Ensure x is in the shape (bs, C, L)
        if len(x.shape) != 3:
            raise ValueError(
                f"Input tensor should have shape (batch_size, C, L), but got {x.shape}"
            )

        # Initialize an empty list to store the processed samples
        processed_samples = []

        # Apply data augmentation and normalization for each sample in the batch
        augmentation = DataAugmentation()
        for i in range(bs):
            sample = x[i].numpy()  # Convert to numpy array for data augmentation
            sample = augmentation.translate_pointcloud(sample)
            sample = augmentation.jitter_pointcloud(sample)
            sample = self.min_max_normalize(sample)
            processed_samples.append(sample)

        # Stack the processed samples back into a batch tensor
        x_processed = paddle.to_tensor(np.stack(processed_samples, axis=0))

        # Ensure the processed tensor has the same shape as the original input
        if x_processed.shape != x.shape:
            raise ValueError(
                f"Processed tensor shape {x_processed.shape} does not match original input shape {x.shape}"
            )

        x_processed = x_processed.transpose(perm=[0, 2, 1])

        shortcut = self.bn_shortcut(self.conv_shortcut(x_processed))
        x = paddle.nn.functional.relu(x=self.bn1(self.conv1(x_processed)))
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
        x = paddle.nn.functional.adaptive_max_pool1d(x=x, output_size=1).squeeze(
            axis=-1
        )
        x = paddle.nn.functional.relu(x=self.bn7(self.linear1(x)))
        x = paddle.nn.functional.relu(x=self.bn8(self.linear2(x)))
        x = paddle.nn.functional.relu(x=self.bn9(self.linear3(x)))
        x = paddle.nn.functional.relu(x=self.bn10(self.linear4(x)))
        x = self.final_linear(x)
        return {self.label_keys[0]: x}
