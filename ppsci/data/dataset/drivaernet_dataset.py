import logging
import os
from typing import Callable
from typing import Optional
from typing import Tuple

import numpy as np
import paddle
import pandas as pd
import pyvista as pv
import seaborn as sns
import trimesh

"""
Created on Tue Dec 19 20:54:56 2023

@author: Mohamed Elrefaie, mohamed.elrefaie@mit.edu mohamed.elrefaie@tum.de

This module is part of the research presented in the paper":
"DrivAerNet: A Parametric Car Dataset for Data-driven Aerodynamic Design and Graph-Based Drag Prediction".

The module defines a Paddle Dataset for loading and transforming 3D car models from the DrivAerNet dataset
stored as STL files.
It includes functionality to subsample or pad the vertices of the models to a fixed number of points as well as
visualization methods for the DrivAerNet dataset.
"""


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


    """
    Paddle Dataset class for the DrivAerNet dataset, handling loading, transforming, and augmenting 3D car models.

    This dataset is specifically designed for aerodynamic tasks, including training machine learning models
    to predict aerodynamic coefficients such as drag coefficient (Cd) from 3D car models.

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
    ):
        """
        Initializes the DrivAerNetDataset instance.

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
        """

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
        self.augmentation = DataAugmentation()

        try:
            with open(os.path.join(self.subset_dir, self.ids_file), "r") as file:
                subset_ids = file.read().split()
            self.data_frame[self.data_frame["Design"].isin(subset_ids)].index.tolist()
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Error loading subset file {self.ids_file}: {e}")

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

    def __getitem__(
        self, idx: int, apply_augmentations: bool = True
    ) -> tuple[
        dict[str, paddle.Tensor], dict[str, paddle.Tensor], dict[str, paddle.Tensor]
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
        row = self.data_frame.iloc[idx]
        design_id = row["Design"]
        cd_value = row["Average Cd"]
        geometry_path = os.path.join(self.root_dir, f"{design_id}.stl")
        try:
            mesh = trimesh.load(geometry_path, force="mesh")
        except Exception as e:
            logging.error(f"Failed to load STL file: {geometry_path}. Error: {e}")
            raise
        vertices = paddle.to_tensor(data=mesh.vertices, dtype="float32")
        vertices = self._sample_or_pad_vertices(vertices, self.num_points)
        if apply_augmentations:
            vertices = self.augmentation.translate_pointcloud(vertices.numpy())
            vertices = self.augmentation.jitter_pointcloud(vertices)
        if self.transform:
            vertices = self.transform(vertices)
        cd_value = paddle.to_tensor(data=float(cd_value), dtype="float32").reshape([-1])

        return (
            {self.input_keys[0]: vertices},
            {self.label_keys[0]: cd_value},
            {self.weight_keys[0]: paddle.to_tensor(1)},
        )

        # return vertices, cd_value

    def visualize_mesh(self, idx):
        """
        Visualize the STL mesh for a specific design from the dataset.

        Args:
            idx (int): Index of the design to visualize in the dataset.

        This function loads the mesh from the STL file corresponding to the design ID at the given index,
        wraps it using PyVista for visualization, and then sets up a PyVista plotter to display the mesh.
        """
        row = self.data_frame.iloc[idx]
        design_id = row["Design"]
        geometry_path = os.path.join(self.root_dir, f"{design_id}.stl")
        try:
            mesh = trimesh.load(geometry_path, force="mesh")
        except Exception as e:
            logging.error(f"Failed to load STL file: {geometry_path}. Error: {e}")
            raise
        pv_mesh = pv.wrap(mesh)
        plotter = pv.Plotter()
        plotter.add_mesh(pv_mesh, color="lightgrey", show_edges=True)
        plotter.add_axes()
        camera_position = [
            (-11.073024242161921, -5.621499358347753, 5.862225824910342),
            (1.458462064391673, 0.002314306982062475, 0.6792134746589196),
            (0.34000174095454166, 0.10379556639001211, 0.9346792479485448),
        ]
        plotter.camera_position = camera_position
        plotter.show()

    def visualize_mesh_withNode(self, idx):
        """
        Visualizes the mesh for a specific design from the dataset with nodes highlighted.

        Args:
            idx (int): Index of the design to visualize in the dataset.

        This function loads the mesh from the STL file and highlights the nodes (vertices) of the mesh using spheres.
        It uses seaborn to obtain visually distinct colors for the mesh and nodes.
        """
        row = self.data_frame.iloc[idx]
        design_id = row["Design"]
        geometry_path = os.path.join(self.root_dir, f"{design_id}.stl")
        try:
            mesh = trimesh.load(geometry_path, force="mesh")
            pv_mesh = pv.wrap(mesh)
        except Exception as e:
            logging.error(f"Failed to load STL file: {geometry_path}. Error: {e}")
            raise
        plotter = pv.Plotter()
        sns_blue = sns.color_palette("colorblind")[0]
        plotter.add_mesh(
            pv_mesh, color="lightgrey", show_edges=True, edge_color="black"
        )
        nodes = pv_mesh.points
        plotter.add_points(
            nodes, color=sns_blue, point_size=10, render_points_as_spheres=True
        )
        plotter.add_axes()
        plotter.show()

    def visualize_point_cloud(self, idx):
        """
        Visualizes the point cloud for a specific design from the dataset.

        Args:
            idx (int): Index of the design to visualize in the dataset.

        This function retrieves the vertices for the specified design, converts them into a point cloud,
        and uses the z-coordinate for color mapping. PyVista's Eye-Dome Lighting is enabled for improved depth perception.
        """
        vertices, _ = self.__getitem__(idx)
        vertices = vertices.numpy()
        point_cloud = pv.PolyData(vertices)
        colors = vertices[:, 2]
        point_cloud["colors"] = colors
        plotter = pv.Plotter()
        plotter.add_points(
            point_cloud,
            scalars="colors",
            cmap="Blues",
            point_size=3,
            render_points_as_spheres=True,
        )
        plotter.enable_eye_dome_lighting()
        plotter.add_axes()
        camera_position = [
            (-11.073024242161921, -5.621499358347753, 5.862225824910342),
            (1.458462064391673, 0.002314306982062475, 0.6792134746589196),
            (0.34000174095454166, 0.10379556639001211, 0.9346792479485448),
        ]
        plotter.camera_position = camera_position
        plotter.show()

    def visualize_augmentations(self, idx):
        """
        Visualizes various augmentations applied to the point cloud of a specific design in the dataset.

        Args:
            idx (int): Index of the sample in the dataset to be visualized.

        This function retrieves the original point cloud for the specified design and then applies a series of augmentations,
        including translation, jittering, and point dropping. Each version of the point cloud (original and augmented) is then
        visualized in a 2x2 grid using PyVista to illustrate the effects of these augmentations.
        """
        vertices, _ = self.__getitem__(idx, apply_augmentations=False)
        original_pc = pv.PolyData(vertices.numpy())
        translated_pc = self.augmentation.translate_pointcloud(vertices.numpy())
        jittered_pc = self.augmentation.jitter_pointcloud(translated_pc)
        dropped_pc = self.augmentation.drop_points(jittered_pc)
        plotter = pv.Plotter(shape=(2, 2))
        plotter.subplot(0, 0)
        plotter.add_text("Original Point Cloud", font_size=10)
        plotter.add_mesh(original_pc, color="black", point_size=3)
        plotter.subplot(0, 1)
        plotter.add_text("Translated Point Cloud", font_size=10)
        plotter.add_mesh(
            pv.PolyData(translated_pc.numpy()), color="lightblue", point_size=3
        )
        plotter.subplot(1, 0)
        plotter.add_text("Jittered Point Cloud", font_size=10)
        plotter.add_mesh(
            pv.PolyData(jittered_pc.numpy()), color="lightgreen", point_size=3
        )
        plotter.subplot(1, 1)
        plotter.add_text("Dropped Point Cloud", font_size=10)
        plotter.add_mesh(pv.PolyData(dropped_pc.numpy()), color="salmon", point_size=3)
        plotter.show()
