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
Created on Tue Dec 19 20:54:56 2023

@author: Mohamed Elrefaie, mohamed.elrefaie@mit.edu mohamed.elrefaie@tum.de

This module is part of the research presented in the paper":
"DrivAerNet: A Parametric Car Dataset for Data-driven Aerodynamic Design and Graph-Based Drag Prediction".
"""
from __future__ import annotations

import logging
import os
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Tuple

import numpy as np
import paddle
import pandas as pd


class DrivAerNetDataset(paddle.io.Dataset):
    """
    Paddle Dataset class for the DrivAerNet dataset, handling loading, transforming, and augmenting 3D car models.

    This dataset is specifically designed for aerodynamic tasks, including training machine learning models
    to predict aerodynamic coefficients such as drag coefficient (Cd) from 3D car models.

    Args:
        input_keys (Tuple[str, ...]): Tuple specifying the keys for input features.
            These keys correspond to the attributes of the dataset used as input to the model.
            For example, "vertices" represents the 3D point cloud vertices of car models.
        label_keys (Tuple[str, ...]): Tuple specifying the keys for ground-truth labels.
            These keys correspond to the target values, such as aerodynamic coefficients like Cd.
            Example: ("cd_value",)
        weight_keys (Tuple[str, ...]): Tuple specifying the keys for optional sample weights.
            These keys represent weighting factors that may be used to adjust loss computation
            during model training. Useful for handling sample imbalance.
            Example: ("weight_keys",)
        subset_dir (str): Path to the directory containing subset information.
            This directory typically contains files that divide the dataset into training,
            validation, and test subsets using a list of model IDs.
        ids_file (str): Path to the text file containing model IDs for the current subset.
            Each line in the file corresponds to a unique model ID that defines which
            models belong to the subset (e.g., training set or test set).
        root_dir (str): Directory containing the STL files of 3D car models.
            Each STL file is expected to represent a single car model and is named according
            to the corresponding model ID. This is the primary data source.
        csv_file (str): Path to the CSV file containing metadata for car models.
            This file typically includes aerodynamic properties (e.g., drag coefficient)
            and other descriptive attributes mapped to each model ID.
        num_points (int): Fixed number of points to sample from each 3D model.
            If a 3D model has more points than `num_points`, it will be randomly subsampled.
            If it has fewer points, it will be zero-padded to reach the desired number.
        transform (Optional[Callable]): An optional callable for applying data transformations.
            This can include augmentations such as scaling, rotation, jittering, or other preprocessing
            steps applied to the 3D point clouds before they are passed to the model.
        pointcloud_exist (bool): Whether the point clouds are pre-processed and saved as `.pt` files.
            If `True`, the dataset will directly load the pre-saved point clouds instead of generating them from STL files.
        train_fractions (float): Fraction of the training data to use. Useful for experiments where only a portion of the data is needed.
        mode (str): Mode of operation, either "train", "eval", or "test". Determines how the dataset behaves.

    Examples:
        >>> import ppsci
        >>> dataset = ppsci.data.dataset.DrivAerNetDataset(
        ...     input_keys=("vertices",),
        ...     label_keys=("cd_value",),
        ...     weight_keys=("weight_keys",),
        ...     subset_dir="/path/to/subset_dir",
        ...     ids_file="train_ids.txt",
        ...     root_dir="/path/to/DrivAerNetDataset",
        ...     csv_file="/path/to/aero_metadata.csv",
        ...     num_points=1024,
        ...     transform=None,
        ... )  # doctest: +SKIP
    """

    def __init__(
        self,
        input_keys: Tuple[str, ...],
        label_keys: Tuple[str, ...],
        weight_keys: Tuple[str, ...],
        subset_dir: str,
        ids_file: str,
        root_dir: str,
        csv_file: str,
        num_points: int,
        transform: Optional[Callable] = None,
        pointcloud_exist: bool = True,
        train_fractions=1.0,
        mode="eval",
    ):

        super().__init__()
        self.root_dir = root_dir
        try:
            self.data_frame = pd.read_csv(csv_file)
        except Exception as e:
            logging.error(f"Failed to load CSV file: {csv_file}. Error: {e}")
            raise
        self.input_keys = input_keys
        self.label_keys = label_keys
        self.weight_keys = weight_keys
        self.subset_dir = subset_dir
        self.ids_file = ids_file
        self.transform = transform
        self.num_points = num_points
        self.pointcloud_exist = pointcloud_exist
        self.mode = mode
        self.train_fractions = train_fractions
        self.cache = {}

        try:
            with open(os.path.join(self.subset_dir, self.ids_file), "r") as file:
                subset_ids = file.read().split()
            self.subset_indices = self.data_frame[
                self.data_frame["Design"].isin(subset_ids)
            ].index.tolist()
            self.data_frame = self.data_frame.loc[self.subset_indices].reset_index(
                drop=True
            )
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Error loading subset file {self.ids_file}: {e}")

        if self.mode == "train":
            self.data_frame = self.data_frame.sample(frac=self.train_fractions)
        else:
            self.data_frame = self.data_frame

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.data_frame)

    def min_max_normalize(self, data: paddle.Tensor) -> paddle.Tensor:
        """
        Normalizes the data to the range [0, 1] based on min and max values.

        Args:
            data: Input data as a paddle.Tensor.

        Returns:
            Normalized data as a paddle.Tensor.
        """
        min_vals, _ = data.min(axis=0, keepdim=True)
        max_vals, _ = data.max(axis=0, keepdim=True)
        normalized_data = (data - min_vals) / (max_vals - min_vals)
        return normalized_data

    def _sample_or_pad_vertices(
        self, vertices: paddle.Tensor, num_points: int
    ) -> paddle.Tensor:
        """
        Subsamples or pads the vertices of the model to a fixed number of points.

        Args:
            vertices: The vertices of the 3D model as a paddle.Tensor.
            num_points: The desired number of points for the model.

        Returns:
            The vertices standardized to the specified number of points.
        """
        num_vertices = vertices.shape[0]
        if num_vertices > num_points:
            indices = np.random.choice(num_vertices, num_points, replace=False)
            vertices = vertices[indices]
        elif num_vertices < num_points:
            padding = paddle.zeros(
                shape=(num_points - num_vertices, 3), dtype="float32"
            )
            vertices = paddle.concat(x=(vertices, padding), axis=0)
        return vertices

    def _load_point_cloud(self, design_id: str) -> Optional[paddle.Tensor]:
        load_path = os.path.join(self.root_dir, f"{design_id}.paddle_tensor")
        if os.path.exists(load_path) and os.path.getsize(load_path) > 0:
            try:
                vertices = paddle.load(path=str(load_path))
                num_vertices = vertices.shape[0]

                if num_vertices > self.num_points:
                    indices = np.random.choice(
                        num_vertices, self.num_points, replace=False
                    )
                    vertices = vertices.numpy()[indices]
                    vertices = paddle.to_tensor(vertices)

                return vertices
            except (EOFError, RuntimeError, ValueError) as e:
                raise Exception(
                    f"Error loading point cloud from {load_path}: {e}"
                ) from e

    def __getitem__(
        self, idx: int, apply_augmentations: bool = True
    ) -> Tuple[
        Dict[str, paddle.Tensor],
        Dict[str, paddle.Tensor],
        Dict[str, paddle.Tensor],
    ]:
        """
        Retrieves a sample and its corresponding label from the dataset, with an option to apply augmentations.

        Args:
            idx (int): Index of the sample to retrieve.
            apply_augmentations (bool, optional): Whether to apply data augmentations. Defaults to True.

        Returns:
            Tuple[paddle.Tensor, paddle.Tensor]: The sample (point cloud) and its label (Cd value).
        """
        if paddle.is_tensor(x=idx):
            idx = idx.tolist()

        if idx in self.cache:
            return self.cache[idx]

        row = self.data_frame.iloc[idx]
        design_id = row["Design"]
        cd_value = row["Average Cd"]
        if self.pointcloud_exist:
            try:
                vertices = self._load_point_cloud(design_id)
                if vertices is None:
                    raise ValueError(
                        f"Point cloud for design {design_id} is not found or corrupted."
                    )
            except Exception as e:
                raise ValueError(
                    f"Failed to load point cloud for design {design_id}: {e}"
                )

        if self.transform:
            vertices = self.transform(vertices)
        cd_value = np.array(float(cd_value), dtype=np.float32).reshape([-1])
        self.cache[idx] = (
            {self.input_keys[0]: vertices},
            {self.label_keys[0]: cd_value},
            {self.weight_keys[0]: np.array(1, dtype=np.float32)},
        )

        return (
            {self.input_keys[0]: vertices},
            {self.label_keys[0]: cd_value},
            {self.weight_keys[0]: np.array(1, dtype=np.float32)},
        )
